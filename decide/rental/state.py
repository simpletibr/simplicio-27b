"""The state of a rented day: which phase is done, what it wrote, how long it took.

``<workdir>/state/day.json`` is one small JSON file, replaced atomically after every change.  A rerun reads it and skips a
phase only when the phase finished AND every output it recorded is still what it was (``fingerprint``): a half-copied
checkpoint, a part somebody edited or a folder that was emptied makes the phase run again.  A state that belongs to another
profile or another configuration is refused and left as it is: it is never deleted to make room for a new day.

The recorded seconds are measurements; they are what turns the estimator's UNVERIFIED numbers into numbers one can trust.

The state file is also INPUT from outside: it comes back from the work repo on a new machine, and anybody can edit it.  So what it
names is checked before it is used (``parse_state``): every output must sit strictly inside the work folder, spelled without ``..``,
backslash or drive letter, and not leave it through a link; an output name is one plain path part.  A phase also remembers the
fingerprints of the outputs it was built on (``built_on``), so that a phase whose input changed is not mistaken for a finished one.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

PHASES = ("preflight", "fetch", "targets", "traces", "targets_traces", "train", "soup", "calibrate", "g4", "public_round", "publish")
VERSION = 1
FILE_NAME = "day.json"
#: A file up to this size is hashed whole; a bigger one (a 12 GB shard) by its size, its first and its last ``EDGE`` bytes.
FULL_LIMIT = 64 << 20
EDGE = 1 << 20


#: Folders of the work folder that are not outputs of a phase.
RESERVED = ("state", ".restore")
#: An output name is one plain path part: it becomes a folder name in the work repo.
NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class StateError(RuntimeError):
    """The state file cannot be used, with the reason and what to do."""


def output_path_problem(root: "str | os.PathLike[str]", raw: Any) -> Optional[str]:
    """Why ``raw`` (an output path read from a state file) may not be used, or ``None``: it must be a plain path strictly inside ``root``."""
    if not isinstance(raw, str) or not raw:
        return "is not a path"
    if "\\" in raw or "\x00" in raw:
        return "has a backslash or a NUL character"
    if re.match(r"^[A-Za-z]:", raw):
        return "starts with a drive letter"
    if ".." in Path(raw).parts:
        return "climbs with '..'"
    base = Path(os.path.abspath(root))
    target = Path(os.path.abspath(raw))
    try:
        relative = target.relative_to(base)
    except ValueError:
        return "is outside the work folder"
    if not relative.parts:
        return "is the work folder itself"
    if relative.parts[0] in RESERVED:
        return f"is inside {relative.parts[0]!r}, which is not an output folder"
    try:
        Path(os.path.realpath(target)).relative_to(Path(os.path.realpath(base)))
    except ValueError:
        return "leaves the work folder through a link"
    return None


def parse_state(text: str, *, identity: Mapping[str, str], root: "str | os.PathLike[str] | None", label: str = "the state file") -> Dict[str, Dict[str, Any]]:
    """The phases of a state file, or ``StateError``.  Nothing is written or moved here: this is the check that comes first."""
    try:
        data = json.loads(text)
        stored, phases = data["identity"], data["phases"]
        if data.get("version") != VERSION or not isinstance(phases, dict) or not isinstance(stored, dict):
            raise ValueError("unknown layout")
    except (ValueError, KeyError, TypeError) as exc:
        raise StateError(f"{label} is not a state file of this tool ({exc}); it was left as it is: move it away or use another --workdir") from exc
    if stored != dict(identity):
        raise StateError(f"{label} belongs to another profile or configuration (it was written for {stored}, this run is {dict(identity)}); "
                         "it was left as it is: finish that day with the same inputs, or use another --workdir")
    for phase, record in phases.items():
        if phase not in PHASES or not isinstance(record, dict):
            raise StateError(f"{label} names {phase!r}, which is not a phase of the day; it was left as it is")
        outputs = record.get("outputs", {})
        if not isinstance(outputs, dict):
            raise StateError(f"{label}: the outputs of {phase} are not a table; it was left as it is")
        for name, entry in outputs.items():
            if not isinstance(name, str) or not NAME_OK.match(name) or not isinstance(entry, dict):
                raise StateError(f"{label}: {phase} has an output called {name!r}, which is not a plain name; it was left as it is")
            problem = output_path_problem(root, entry.get("path")) if root is not None else None
            if problem:
                raise StateError(f"{label}: output {name!r} of {phase} ({entry.get('path')!r}) {problem}; it was left as it is, nothing was written from it")
    return phases


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


def digest_tree(path: "str | os.PathLike[str]") -> str:
    """Like ``fingerprint`` but reads EVERY byte: what G4 measured and what is published are compared with this, not with the quick one.

    It costs one pass over the checkpoint (a minute or more for 50 GB, not measured here, UNVERIFIED), which is small next to the hours it guards."""
    return fingerprint(path, full_limit=sys.maxsize)


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
    def __init__(self, directory: "str | os.PathLike[str]", *, identity: Mapping[str, str], clock: Callable[[], float] = time.time,
                 root: "str | os.PathLike[str] | None" = None) -> None:
        self.directory = Path(directory)
        self.path = self.directory / FILE_NAME
        self.identity = dict(identity)
        self.clock = clock
        self._phases: Dict[str, Dict[str, Any]] = {}
        if self.path.is_file():
            try:
                text = self.path.read_text(encoding="utf-8")
            except (OSError, ValueError) as exc:                      # a file that is not text is not a state file either
                raise StateError(f"{self.path} is not a state file of this tool ({exc}); it was left as it is: move it away or use another --workdir") from exc
            self._phases = parse_state(text, identity=self.identity, root=root, label=str(self.path))

    # ------------------------------------------------------------------ reading

    def _check(self, phase: str) -> str:
        if phase not in PHASES:
            raise KeyError(f"unknown phase {phase!r}; the phases are {', '.join(PHASES)}")
        return phase

    def record(self, phase: str) -> Dict[str, Any]:
        return json.loads(json.dumps(self._phases.get(self._check(phase), {"status": "pending"})))

    def status(self, phase: str) -> str:
        return self._phases.get(self._check(phase), {}).get("status", "pending")

    def built_on(self, phase: str) -> Dict[str, Any]:
        """The fingerprints of the outputs the phase was built on, as they were when it finished (``{}``: none recorded)."""
        return json.loads(json.dumps(self._phases.get(self._check(phase), {}).get("built_on", {})))

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

    def done(self, phase: str, *, outputs: Mapping[str, "str | os.PathLike[str]"], detail: Optional[Mapping[str, Any]] = None,
             built_on: Optional[Mapping[str, Any]] = None) -> None:
        record = self._phases.get(self._check(phase))
        if not record or record.get("status") != "running":
            raise StateError(f"{phase} was not begun")
        finished = self.clock()
        record.update(status="done", finished=finished, seconds=finished - record["started"], detail=dict(detail or {}), built_on=json.loads(json.dumps(built_on or {})),
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
