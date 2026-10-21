"""The state of a rented day: which phase is done, what it wrote, how long it took.

``<workdir>/state/day.json`` is one small JSON file, replaced atomically after every change.  A rerun reads it and skips a
phase only when the phase finished AND every output it recorded is still what it was (``fingerprint``): a half-copied
checkpoint, a part somebody edited or a folder that was emptied makes the phase run again.  A state that belongs to another
profile or another configuration is refused and left as it is: it is never deleted to make room for a new day.

The recorded seconds are measurements; they are what turns the estimator's UNVERIFIED numbers into numbers one can trust.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

PHASES = ("preflight", "fetch", "targets", "traces", "targets_traces", "train", "soup", "calibrate", "g4", "public_round", "publish")
VERSION = 1
FILE_NAME = "day.json"
#: A file up to this size is hashed whole; a bigger one (a 12 GB shard) by its size, its first and its last ``EDGE`` bytes.
FULL_LIMIT = 64 << 20
EDGE = 1 << 20


class StateError(RuntimeError):
    """The state file cannot be used, with the reason and what to do."""


def _file_digest(path: Path, full_limit: int, edge: int) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(f"{size}|".encode("ascii"))
    with open(path, "rb") as handle:
        if size <= full_limit:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        else:
            digest.update(handle.read(edge))
            handle.seek(max(0, size - edge))
            digest.update(handle.read(edge))
    return digest.hexdigest()


def fingerprint(path: "str | os.PathLike[str]", *, full_limit: int = FULL_LIMIT, edge: int = EDGE) -> str:
    """A digest of a file or of the tree under a folder: names, sizes and content (big files: size, head and tail only).

    The middle of a file bigger than ``full_limit`` is not read; that is the price of checking a 50 GB checkpoint in a second.
    Unfinished ``*.tmp`` files are not part of the tree."""
    root = Path(path)
    if root.is_file():
        return _file_digest(root, full_limit, edge)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} does not exist")
    entries = []
    for file in sorted((f for f in root.rglob("*") if f.is_file() and not f.name.endswith(".tmp")), key=lambda f: f.relative_to(root).as_posix()):
        entries.append([file.relative_to(root).as_posix(), _file_digest(file, full_limit, edge)])
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest()


class DayState:
    def __init__(self, directory: "str | os.PathLike[str]", *, identity: Mapping[str, str], clock: Callable[[], float] = time.time) -> None:
        self.directory = Path(directory)
        self.path = self.directory / FILE_NAME
        self.identity = dict(identity)
        self.clock = clock
        self._phases: Dict[str, Dict[str, Any]] = {}
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                stored, phases = data["identity"], data["phases"]
                if data.get("version") != VERSION or not isinstance(phases, dict):
                    raise ValueError("unknown layout")
            except (ValueError, KeyError, TypeError) as exc:
                raise StateError(f"{self.path} is not a state file of this tool ({exc}); it was left as it is: move it away or use another --workdir") from exc
            if stored != self.identity:
                raise StateError(f"{self.path} belongs to another profile or configuration (it was written for {stored}, this run is {self.identity}); "
                                 "it was left as it is: finish that day with the same inputs, or use another --workdir")
            self._phases = phases

    # ------------------------------------------------------------------ reading

    def _check(self, phase: str) -> str:
        if phase not in PHASES:
            raise KeyError(f"unknown phase {phase!r}; the phases are {', '.join(PHASES)}")
        return phase

    def record(self, phase: str) -> Dict[str, Any]:
        return json.loads(json.dumps(self._phases.get(self._check(phase), {"status": "pending"})))

    def status(self, phase: str) -> str:
        return self._phases.get(self._check(phase), {}).get("status", "pending")

    def seconds(self, phase: str) -> Optional[float]:
        return self._phases.get(self._check(phase), {}).get("seconds")

    def why_not_reusable(self, phase: str) -> Optional[str]:
        """``None`` when the phase can be skipped, else the reason it must run again."""
        status = self.status(phase)
        if status == "skipped":
            return None
        if status != "done":
            return f"{phase} is {status}"
        for name, entry in self._phases[phase].get("outputs", {}).items():
            where = Path(entry["path"])
            try:
                now = fingerprint(where)
            except FileNotFoundError:
                return f"output {name} ({where}) of {phase} is gone"
            if now != entry["fingerprint"]:
                return f"output {name} ({where}) of {phase} changed after the phase finished"
        return None

    def reusable(self, phase: str) -> bool:
        return self.why_not_reusable(phase) is None

    # ------------------------------------------------------------------ writing

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        pending = self.path.with_name(self.path.name + ".tmp")
        pending.write_text(json.dumps({"version": VERSION, "identity": self.identity, "phases": self._phases}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(pending, self.path)

    def begin(self, phase: str) -> None:
        self._phases[self._check(phase)] = {"status": "running", "started": self.clock()}
        self._save()

    def done(self, phase: str, *, outputs: Mapping[str, "str | os.PathLike[str]"], detail: Optional[Mapping[str, Any]] = None) -> None:
        record = self._phases.get(self._check(phase))
        if not record or record.get("status") != "running":
            raise StateError(f"{phase} was not begun")
        finished = self.clock()
        record.update(status="done", finished=finished, seconds=finished - record["started"], detail=dict(detail or {}),
                      outputs={name: {"path": str(Path(path)), "fingerprint": fingerprint(path)} for name, path in outputs.items()})
        self._save()

    def _end(self, phase: str, status: str, reason: str) -> None:
        finished = self.clock()
        record = self._phases.setdefault(self._check(phase), {"started": finished})
        record.update(status=status, reason=reason, finished=finished, seconds=finished - record["started"])
        self._save()

    def fail(self, phase: str, reason: str) -> None:
        self._end(phase, "failed", reason)

    def stop(self, phase: str, reason: str) -> None:
        """A gate said no: the day stops here, and the next run asks the gate again."""
        self._end(phase, "stopped", reason)

    def skip(self, phase: str, reason: str) -> None:
        self._phases[self._check(phase)] = {"status": "skipped", "reason": reason}
        self._save()
