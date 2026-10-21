"""Idempotent stage and row checkpoints that survive a dead session.

A run is a list of named stages (fidelity, merge:<candidate>, eval:<candidate>, calibrate, materialize, ...).
`State` records which stages finished and which rows of a long loop are done in `<workdir>/state.json`, written
atomically, and mirrors that file (plus append-only result logs) to a `Store`: the private Hugging Face dataset
named in the config when HF_TOKEN exists, nothing otherwise.  Opening a state merges the local file with the
store's copy, so a new session, or a second platform working on the same run, continues instead of starting over.

Rules the code keeps:

* Finishing a stage always pushes; other saves push at most every `min_push_seconds`.
* A failed push never stops the run; it is kept in `push_error` and retried on the next push.
* Tokens are never written: error strings go through `redact`.
* Merging is a union (a stage done anywhere is done), so two sessions cannot undo each other.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

SCHEMA_VERSION = 1
STATE_NAME = "state.json"
_RANK = {"done": 3, "failed": 2, "running": 1}
_TOKEN = re.compile(r"hf_[A-Za-z0-9_\-]{8,}")
_BEARER = re.compile(r"(Bearer\s+)\S+", re.IGNORECASE)


class StateError(RuntimeError):
    pass


def redact(text: Any) -> str:
    """Remove Hugging Face tokens and bearer credentials from a message before it is stored or printed."""
    return _BEARER.sub(r"\1[redacted]", _TOKEN.sub("hf_[redacted]", str(text)))


# --------------------------------------------------------------------------- stores

class Store:
    """A remote folder of small files."""

    def read(self, name: str) -> Optional[bytes]:
        raise NotImplementedError

    def write(self, name: str, data: bytes) -> None:
        raise NotImplementedError

    def list(self, prefix: str = "") -> List[str]:
        raise NotImplementedError


def _safe_name(name: str) -> str:
    parts = name.split("/")
    if not name or name.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe store path: {name!r}")
    return name


class MemoryStore(Store):
    def __init__(self) -> None:
        self.files: Dict[str, bytes] = {}
        self.writes = 0

    def read(self, name: str) -> Optional[bytes]:
        return self.files.get(_safe_name(name))

    def write(self, name: str, data: bytes) -> None:
        self.files[_safe_name(name)] = bytes(data)
        self.writes += 1

    def list(self, prefix: str = "") -> List[str]:
        return sorted(k for k in self.files if k.startswith(prefix))


class DirStore(Store):
    """A local directory used as the remote (a mounted Drive, or a test)."""

    def __init__(self, root: os.PathLike) -> None:
        self.root = Path(root)

    def read(self, name: str) -> Optional[bytes]:
        path = self.root / _safe_name(name)
        return path.read_bytes() if path.is_file() else None

    def write(self, name: str, data: bytes) -> None:
        path = self.root / _safe_name(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def list(self, prefix: str = "") -> List[str]:
        if not self.root.is_dir():
            return []
        found = (p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
        return sorted(n for n in found if n.startswith(prefix))


class HfDatasetStore(Store):
    """A private Hugging Face dataset repo.  Needs `huggingface_hub`; UNVERIFIED against the live Hub."""

    def __init__(self, repo_id: str, token: str, *, endpoint: Optional[str] = None, private: bool = True,
                 create: bool = True) -> None:
        if not token:
            raise StateError("HF_TOKEN is required to use the Hugging Face state dataset")
        self.repo_id = repo_id
        self.endpoint = endpoint
        self.private = private
        self.create = create
        self._token = token
        self._api: Any = None
        self._created = False

    def __repr__(self) -> str:
        return f"HfDatasetStore({self.repo_id!r})"

    def _hub(self) -> Any:
        import huggingface_hub

        return huggingface_hub

    def _client(self) -> Any:
        if self._api is None:
            self._api = self._hub().HfApi(token=self._token, endpoint=self.endpoint)
        return self._api

    def _ensure_repo(self) -> None:
        if self.create and not self._created:
            self._client().create_repo(self.repo_id, repo_type="dataset", private=self.private, exist_ok=True)
            self._created = True

    def read(self, name: str) -> Optional[bytes]:
        hub = self._hub()
        missing = tuple(getattr(hub.errors, n) for n in ("EntryNotFoundError", "RepositoryNotFoundError", "RevisionNotFoundError")
                        if hasattr(hub.errors, n))
        try:
            path = hub.hf_hub_download(self.repo_id, _safe_name(name), repo_type="dataset", token=self._token,
                                       endpoint=self.endpoint, force_download=True)
        except missing:
            return None
        return Path(path).read_bytes()

    def write(self, name: str, data: bytes) -> None:
        self._ensure_repo()
        self._client().upload_file(path_or_fileobj=bytes(data), path_in_repo=_safe_name(name), repo_id=self.repo_id,
                                   repo_type="dataset", commit_message=f"sd: {name}")

    def list(self, prefix: str = "") -> List[str]:
        hub = self._hub()
        try:
            names = self._client().list_repo_files(self.repo_id, repo_type="dataset")
        except hub.errors.RepositoryNotFoundError:
            return []
        return sorted(n for n in names if n.startswith(prefix))


def store_from_env(repo_id: str, environ: Optional[Mapping[str, str]] = None, **kw: Any) -> Optional[Store]:
    """The state dataset store when an HF_TOKEN is available (env, Kaggle Secrets, Colab userdata), else None."""
    from decide.sd import env

    token = env.get_secret("HF_TOKEN", environ=environ)
    return HfDatasetStore(repo_id, token, **kw) if token else None


# --------------------------------------------------------------------------- merging

def fresh_data(now: float) -> Dict[str, Any]:
    return {"version": SCHEMA_VERSION, "seq": 0, "created": now, "updated": now,
            "stages": {}, "rows": {}, "meta": {}, "files": {}, "notes": []}


def _valid(data: Any) -> bool:
    return (isinstance(data, dict) and isinstance(data.get("stages"), dict) and isinstance(data.get("rows"), dict)
            and isinstance(data.get("version"), int))


def _better_stage(a: Mapping[str, Any], b: Mapping[str, Any]) -> Mapping[str, Any]:
    ka = (_RANK.get(a.get("status"), 0), a.get("finished") or a.get("started") or 0.0, a.get("attempts", 0))
    kb = (_RANK.get(b.get("status"), 0), b.get("finished") or b.get("started") or 0.0, b.get("attempts", 0))
    return a if ka >= kb else b


def merge_states(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """Union of two states; `a` wins ties.  A finished stage beats a running or failed one."""
    if max(a.get("version", 0), b.get("version", 0)) > SCHEMA_VERSION:
        raise StateError("the state was written by a newer version of the kit")
    out = fresh_data(min(a.get("created", 0.0), b.get("created", 0.0)))
    out["seq"] = max(a.get("seq", 0), b.get("seq", 0))
    out["updated"] = max(a.get("updated", 0.0), b.get("updated", 0.0))
    for name in sorted(set(a.get("stages", {})) | set(b.get("stages", {}))):
        sa, sb = a.get("stages", {}).get(name), b.get("stages", {}).get(name)
        out["stages"][name] = dict(sa if sb is None else sb if sa is None else _better_stage(sa, sb))
    for stage in sorted(set(a.get("rows", {})) | set(b.get("rows", {}))):
        out["rows"][stage] = {**b.get("rows", {}).get(stage, {}), **a.get("rows", {}).get(stage, {})}
    out["meta"] = {**b.get("meta", {}), **a.get("meta", {})}
    out["files"] = {**b.get("files", {}), **a.get("files", {})}
    for name, rec in b.get("files", {}).items():
        mine = a.get("files", {}).get(name)
        if mine is not None and rec.get("size", 0) > mine.get("size", 0):
            out["files"][name] = dict(rec)
    out["notes"] = list(dict.fromkeys([*a.get("notes", []), *b.get("notes", [])]))
    return out


# --------------------------------------------------------------------------- state

class State:
    def __init__(self, workdir: os.PathLike, store: Optional[Store] = None, *, data: Optional[Dict[str, Any]] = None,
                 min_push_seconds: float = 600.0, autosave_rows: int = 25,
                 clock: Callable[[], float] = time.monotonic, now: Callable[[], float] = time.time) -> None:
        self.workdir = Path(workdir)
        self.path = self.workdir / STATE_NAME
        self.store = store
        self.min_push_seconds = float(min_push_seconds)
        self.autosave_rows = int(autosave_rows)
        self._clock, self._now = clock, now
        self.data: Dict[str, Any] = data if data is not None else fresh_data(now())
        self.push_error: Optional[str] = None
        self._last_push: Optional[float] = None
        self._dirty_rows = 0
        self._file_push: Dict[str, float] = {}

    # ---- opening
    @classmethod
    def open(cls, workdir: os.PathLike, store: Optional[Store] = None, **kw: Any) -> "State":
        st = cls(workdir, store, **kw)
        st.workdir.mkdir(parents=True, exist_ok=True)
        local = st._read_local()
        remote = st._read_remote()
        if local is not None and remote is not None:
            st.data = merge_states(local, remote)
        elif local is not None or remote is not None:
            st.data = dict(local if local is not None else remote)    # type: ignore[arg-type]
        if st.data.get("version", SCHEMA_VERSION) > SCHEMA_VERSION:
            raise StateError("the state was written by a newer version of the kit")
        if local is None or (remote is not None and st.data != local):
            st._write_local()
        return st

    def _note(self, text: str) -> None:
        self.data.setdefault("notes", []).append(redact(text))

    def _read_local(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not _valid(data):
                raise ValueError("not a state document")
            return data
        except (OSError, ValueError) as exc:
            aside = self.path.with_name(STATE_NAME + ".corrupt")
            os.replace(self.path, aside)
            self._note(f"local state was corrupt ({type(exc).__name__}) and moved to {aside.name}")
            return None

    def _read_remote(self) -> Optional[Dict[str, Any]]:
        if self.store is None:
            return None
        try:
            raw = self.store.read(STATE_NAME)
            if raw is None:
                return None
            data = json.loads(raw.decode("utf-8"))
            if not _valid(data):
                raise ValueError("not a state document")
            return data
        except Exception as exc:
            self._note(f"remote state unreadable: {type(exc).__name__}: {exc}")
            return None

    # ---- persistence
    def _dump(self) -> str:
        return json.dumps(self.data, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False)

    def _write_local(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.data["seq"] = int(self.data.get("seq", 0)) + 1
        self.data["updated"] = self._now()
        tmp = self.path.with_name(STATE_NAME + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(self._dump())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)
        self._dirty_rows = 0

    def save(self, force_push: bool = False) -> None:
        self._write_local()
        due = self._last_push is None or self._clock() - self._last_push >= self.min_push_seconds
        if self.store is not None and (force_push or due):
            self.push()

    def flush(self) -> None:
        if self._dirty_rows:
            self.save()

    def push(self, files_now: bool = False) -> bool:
        """Upload changed tracked files, then state.json.  Returns False (and records why) when the store failed."""
        if self.store is None:
            return False
        self._write_local()
        try:
            for remote, rec in sorted(self.data["files"].items()):
                self._push_file(remote, rec, files_now)
            self.store.write(STATE_NAME, self._dump().encode("utf-8"))
        except Exception as exc:
            self.push_error = redact(f"{type(exc).__name__}: {exc}")
            return False
        self.push_error = None
        self._last_push = self._clock()
        return True

    # ---- stages
    def stage_status(self, name: str) -> Optional[str]:
        rec = self.data["stages"].get(name)
        return None if rec is None else rec["status"]

    def is_done(self, name: str) -> bool:
        return self.stage_status(name) == "done"

    def result(self, name: str) -> Any:
        rec = self.data["stages"].get(name)
        return None if rec is None else rec.get("result")

    def begin(self, name: str) -> None:
        rec = self.data["stages"].get(name, {})
        self.data["stages"][name] = {"status": "running", "attempts": rec.get("attempts", 0) + 1, "started": self._now()}
        self.save()

    def finish(self, name: str, result: Any = None) -> None:
        try:
            json.dumps(result, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise StateError(f"result of stage {name!r} is not plain JSON: {exc}") from exc
        rec = self.data["stages"].setdefault(name, {"attempts": 1, "started": self._now()})
        rec.pop("error", None)
        rec.update(status="done", finished=self._now(), result=result)
        self.save(force_push=True)

    def fail(self, name: str, error: Any) -> None:
        rec = self.data["stages"].setdefault(name, {"attempts": 1, "started": self._now()})
        rec.update(status="failed", finished=self._now(), error=redact(error))
        self.save(force_push=True)

    def run_stage(self, name: str, fn: Callable[..., Any], *args: Any, force: bool = False, **kw: Any) -> Any:
        """Run `fn` once: a finished stage returns its stored result; a failed or dead one runs again."""
        if self.is_done(name) and not force:
            return self.result(name)
        self.begin(name)
        try:
            value = fn(*args, **kw)
        except Exception as exc:
            self.fail(name, f"{type(exc).__name__}: {exc}")
            raise
        self.finish(name, value)
        return value

    # ---- rows
    def mark_row(self, stage: str, key: str, value: Any = True) -> None:
        self.data["rows"].setdefault(stage, {})[key] = value
        self._dirty_rows += 1
        if self._dirty_rows >= self.autosave_rows:
            self.save()

    def rows(self, stage: str) -> Dict[str, Any]:
        return dict(self.data["rows"].get(stage, {}))

    def row_done(self, stage: str, key: str) -> bool:
        return key in self.data["rows"].get(stage, {})

    # ---- meta
    def set_meta(self, key: str, value: Any) -> None:
        self.data["meta"][key] = value
        self.save()

    def meta(self, key: str, default: Any = None) -> Any:
        return self.data["meta"].get(key, default)

    # ---- mirrored files (append-only logs such as results.jsonl)
    def track_file(self, local: os.PathLike, remote: str) -> None:
        rel = os.path.relpath(Path(local).resolve(), self.workdir.resolve())
        if rel.startswith(".."):
            raise StateError(f"{local} is outside the work directory {self.workdir}")
        self.data["files"].setdefault(_safe_name(remote), {"local": Path(rel).as_posix(), "size": 0})

    def _push_file(self, remote: str, rec: Dict[str, Any], files_now: bool) -> None:
        path = self.workdir / rec["local"]
        if not path.is_file():
            return
        size = path.stat().st_size
        if size == rec.get("size", 0) and rec.get("pushed"):
            return
        last = self._file_push.get(remote)
        if not files_now and last is not None and self._clock() - last < self.min_push_seconds:
            return
        assert self.store is not None
        self.store.write(remote, path.read_bytes())
        rec.update(size=size, pushed=True)
        self._file_push[remote] = self._clock()

    def restore_files(self) -> List[str]:
        """Download tracked files that are missing, or shorter than the pushed copy, from the store."""
        restored: List[str] = []
        if self.store is None:
            return restored
        for remote, rec in sorted(self.data["files"].items()):
            path = self.workdir / rec["local"]
            if path.is_file() and path.stat().st_size >= rec.get("size", 0):
                continue
            data = self.store.read(remote)
            if data is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            restored.append(remote)
        return restored


def open_state(cfg: Mapping[str, Any], workdir: Optional[os.PathLike] = None,
               environ: Optional[Mapping[str, str]] = None, **kw: Any) -> State:
    """The state for this platform: workdir from the platform, store from HF_TOKEN and cfg.hub.state_dataset."""
    from decide.sd import env

    platform = env.detect_platform(environ)
    path = Path(workdir) if workdir else env.default_workdir(platform, environ)
    store = store_from_env(cfg["hub"]["state_dataset"], environ, endpoint=cfg["hub"].get("endpoint"))
    return State.open(path, store, **kw)
