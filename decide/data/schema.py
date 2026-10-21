"""Formats of the Simplicio-Decide-27B data pipeline (see SCHEMA.md).

The request schema is the one of pplx-decider v1.2 (Apache-2.0): a state and typed questions.  Gold labels are
option *keys* (as ``options()`` of the pplx protocol returns them), never positional codes.  Only the standard
library is needed for validation; pyarrow is imported lazily by the parquet helpers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
MAX_OPTIONS = 255
QUESTION_TYPES = ("choice", "noul", "score")
CORPUS_COLUMNS = ("example_id", "source", "license", "split", "lang", "request", "gold")
TEACHER_COLUMNS = ("example_id", "question_id", "teacher", "codes", "probs", "temperature")
SPLITS = ("train", "dev")
MANIFEST_NAME = "manifest.json"
PROB_TOLERANCE = 1e-3
BATCH_ROWS = 20_000
_LANG = re.compile(r"^(mul|[a-z]{2,3}(-[A-Za-z0-9]{2,8})*)$")


class SchemaError(ValueError):
    """A row or request does not follow the schema; the message lists every problem found."""


class ManifestError(SchemaError):
    """manifest.json is missing, unreadable, or does not match the files next to it."""


# --------------------------------------------------------------------------- hashing

def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_digest(request: Any) -> str:
    return digest(canonical_json(request))


def sha256_file(path: os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- requests

def option_keys(question: Mapping[str, Any]) -> List[str]:
    """Option keys in prompt order, exactly as ``pplx_decider.protocol.options`` returns them."""
    kind = question.get("type")
    if kind == "choice":
        return [str(k) for k in question["criteria"]]
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["false", "true"]


def option_texts(question: Mapping[str, Any]) -> List[Any]:
    """Raw option descriptions in prompt order (``None`` descriptions fall back to the key / the pplx defaults)."""
    kind = question.get("type")
    if kind == "choice":
        return [key if value is None else value for key, value in question["criteria"].items()]
    if kind == "score":
        return list(question["criteria"])
    criteria = question.get("criteria") or {}
    return [criteria.get("false") or "No / false", criteria.get("true") or "Yes / true"]


def _is_content(value: Any) -> bool:
    """``Content`` of pplx-decider's types.py: text, or any JSON object or list (the encoder serializes it)."""
    return isinstance(value, (str, Mapping, list, tuple))


def validate_request(request: Any) -> List[str]:
    """Problems with a request (empty list = valid).

    ``Encoder.validate`` of pplx-decider v1.2 plus the types of its ``types.py``, which the encoder does not check but
    relies on: ``state`` and every description are ``Content`` (text, object or list), a noul's ``criteria`` is
    ``{"true"|"false": Content | None}`` or absent, a choice option may have no description (``None``)."""
    if not isinstance(request, Mapping):
        return ["request must be an object"]
    problems: List[str] = []
    if "state" not in request:
        problems.append("missing 'state'")
    elif not _is_content(request["state"]):
        problems.append("'state' must be text, an object or a list")
    if "images" in request:
        problems.append("images are not supported in the corpus")
    questions = request.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        return problems + ["'questions' must be a non-empty object"]
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid:
            problems.append("question ids must be non-empty strings")
            continue
        kind = question.get("type") if isinstance(question, Mapping) else None
        if kind not in QUESTION_TYPES:
            problems.append(f"{qid}: type must be one of {QUESTION_TYPES}")
            continue
        instructions = question.get("instructions")
        if instructions is not None and not _is_content(instructions):
            problems.append(f"{qid}: instructions must be text, an object or a list")
            continue
        criteria = question.get("criteria")
        if kind == "choice" and not isinstance(criteria, Mapping):
            problems.append(f"{qid}: choice criteria must be an object (option key -> description)")
            continue
        if kind == "score" and not isinstance(criteria, list):
            problems.append(f"{qid}: score criteria must be a list of level descriptions")
            continue
        if kind == "noul" and criteria is not None and not (isinstance(criteria, Mapping) and set(criteria) <= {"true", "false"}):
            problems.append(f"{qid}: noul criteria must be absent or an object with the keys 'true' and 'false'")
            continue
        described = list(criteria.values()) if isinstance(criteria, Mapping) else list(criteria or [])
        if any(not (_is_content(d) or (d is None and kind != "score")) for d in described):
            problems.append(f"{qid}: option descriptions must be text, an object or a list")
            continue
        count = len(option_keys(question))
        if not 1 <= count <= MAX_OPTIONS:
            problems.append(f"{qid}: {count} options; 1 to {MAX_OPTIONS} are supported")
        if kind == "score" and count < 2:
            problems.append(f"{qid}: a score needs at least two ordered levels")
    try:
        canonical_json(request)
    except (TypeError, ValueError) as exc:
        problems.append(f"request is not JSON serializable: {exc}")
    return problems


def validate_gold(request: Mapping[str, Any], gold: Any) -> List[str]:
    if not isinstance(gold, Mapping):
        return ["gold must be an object (question id -> option key or null)"]
    questions = request.get("questions", {})
    problems: List[str] = []
    for qid, key in gold.items():
        if qid not in questions:
            problems.append(f"gold has an unknown question id {qid!r}")
        elif key is None:
            continue
        elif not isinstance(key, str) or key not in option_keys(questions[qid]):
            problems.append(f"gold for {qid!r} is not one of its option keys: {key!r}")
    return problems


# --------------------------------------------------------------------------- corpus rows

def _nonempty_str(row: Mapping[str, Any], name: str, problems: List[str]) -> None:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{name} must be a non-empty string")


def validate_corpus_row(row: Mapping[str, Any]) -> List[str]:
    missing = [c for c in CORPUS_COLUMNS if c not in row]
    if missing:
        return [f"missing columns: {missing}"]
    problems: List[str] = []
    for name in ("example_id", "source", "license"):
        _nonempty_str(row, name, problems)
    if row["split"] not in SPLITS:
        problems.append(f"split must be one of {SPLITS}")
    if not isinstance(row["lang"], str) or not _LANG.match(row["lang"]):
        problems.append("lang must be an ISO 639 code (en, pt, pt-BR, ...) or 'mul'")
    try:
        request = json.loads(row["request"])
        gold = json.loads(row["gold"])
    except (TypeError, ValueError) as exc:
        return problems + [f"request and gold must be JSON strings: {exc}"]
    problems += [f"request: {p}" for p in validate_request(request)]
    if not problems:
        problems += [f"gold: {p}" for p in validate_gold(request, gold)]
    return problems


def corpus_row(*, example_id: str, source: str, license: str, split: str, lang: str,
               request: Mapping[str, Any], gold: Mapping[str, Any]) -> Dict[str, str]:
    """Build a corpus row (request and gold as canonical JSON) and validate it."""
    row = {"example_id": example_id, "source": source, "license": license, "split": split, "lang": lang,
           "request": canonical_json(request), "gold": canonical_json(gold)}
    problems = validate_corpus_row(row)
    if problems:
        raise SchemaError(f"{example_id}: " + "; ".join(problems))
    return row


def parse_row(row: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return json.loads(row["request"]), json.loads(row["gold"])


# --------------------------------------------------------------------------- teacher rows

def validate_teacher_row(row: Mapping[str, Any]) -> List[str]:
    missing = [c for c in TEACHER_COLUMNS if c not in row]
    if missing:
        return [f"missing columns: {missing}"]
    problems: List[str] = []
    for name in ("example_id", "question_id", "teacher"):
        _nonempty_str(row, name, problems)
    codes, probs = row["codes"], row["probs"]
    if not isinstance(codes, (list, tuple)) or not codes or not all(isinstance(c, str) and c for c in codes):
        problems.append("codes must be a non-empty list of non-empty strings")
    elif len(set(codes)) != len(codes):
        problems.append("codes must be unique")
    if not isinstance(probs, (list, tuple)) or not probs:
        problems.append("probs must be a non-empty list")
    else:
        values = [p for p in probs if isinstance(p, (int, float)) and not isinstance(p, bool)]
        if len(values) != len(probs) or any(not math.isfinite(p) or p < 0 for p in values):
            problems.append("probs must be finite and non-negative")
        elif abs(sum(values) - 1.0) > PROB_TOLERANCE:
            problems.append(f"probs must sum to 1 (got {sum(values):.6f})")
        if isinstance(codes, (list, tuple)) and len(codes) != len(probs):
            problems.append("codes and probs must have the same length")
    t = row["temperature"]
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0:
        problems.append("temperature must be a positive finite number")
    return problems


# --------------------------------------------------------------------------- parquet

def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - message only
        raise ImportError("pyarrow is required for parquet files (install the pinned packages of decide/data/requirements.txt)") from exc
    return pa, pq


def corpus_arrow_schema():
    pa, _ = _arrow()
    return pa.schema([pa.field(c, pa.string()) for c in CORPUS_COLUMNS])


def teacher_arrow_schema():
    pa, _ = _arrow()
    return pa.schema([pa.field("example_id", pa.string()), pa.field("question_id", pa.string()),
                      pa.field("teacher", pa.string()), pa.field("codes", pa.list_(pa.string())),
                      pa.field("probs", pa.list_(pa.float32())), pa.field("temperature", pa.float64())])


def _write_part(path: Path, rows: Sequence[Mapping[str, Any]], schema, columns: Sequence[str]) -> Dict[str, Any]:
    pa, pq = _arrow()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([{c: r[c] for c in columns} for r in rows], schema=schema)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return {"name": path.name, "rows": len(rows), "sha256": sha256_file(path)}


def write_corpus_part(path: os.PathLike, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Atomically write one parquet part; returns {name, rows, sha256} for the manifest."""
    return _write_part(Path(path), rows, corpus_arrow_schema(), CORPUS_COLUMNS)


def write_teacher_part(path: os.PathLike, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return _write_part(Path(path), rows, teacher_arrow_schema(), TEACHER_COLUMNS)


def _listed_parts(directory: Path, verify: bool) -> Optional[List[Path]]:
    """The parts a ``manifest.json`` lists, checked against their sha256; ``None`` when there is no usable list."""
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ManifestError(f"{manifest_path} is not valid JSON: {exc}") from exc
    entries = None
    if isinstance(manifest, Mapping):
        if isinstance(manifest.get("parts"), list):
            entries = manifest["parts"]
        elif isinstance(manifest.get("file"), Mapping):      # the proxy manifest names its single file
            entries = [manifest["file"]]
    if entries is None:
        return None
    files: List[Path] = []
    for entry in entries:
        name = entry.get("name") if isinstance(entry, Mapping) else None
        relative = Path(name) if isinstance(name, str) and name else None
        if relative is None or relative.is_absolute() or ".." in relative.parts:
            raise ManifestError(f"{manifest_path}: part name {name!r} must be a relative path inside the directory")
        file = directory / relative
        if not file.is_file():
            raise ManifestError(f"{manifest_path} lists {name} but the file is missing (an interrupted copy?)")
        expected = entry.get("sha256")
        if verify:
            if not isinstance(expected, str) or not expected:
                raise ManifestError(f"{manifest_path}: part {name} has no sha256, so it cannot be checked")
            if sha256_file(file) != expected:
                raise ManifestError(f"{file} does not match the sha256 in {manifest_path}: the part changed after the manifest was written")
        files.append(file)
    return files


def corpus_files(path: os.PathLike, *, require_manifest: bool = False, verify: bool = True) -> List[Path]:
    """The parquet files of a dataset.

    * a file: itself;
    * a directory with a ``manifest.json`` listing its ``parts`` (or the single ``file`` of a proxy): exactly those
      parts, in the manifest order, each checked against its sha256 (``verify``).  Any other ``*.parquet`` next to
      them (an old build, shards a smaller cap left behind) is ignored: what the manifest does not list is not data;
    * a directory without a usable list: every ``*.parquet`` below it, sorted, unless ``require_manifest`` (then
      ``ManifestError``: an unfinished or hand-made directory is not a build output).
    """
    p = Path(path)
    if not p.is_dir():
        return [p]
    listed = _listed_parts(p, verify)
    if listed is not None:
        return listed
    if require_manifest:
        raise ManifestError(f"{p} has no {MANIFEST_NAME} listing its parts: it is not the output of a finished build "
                            "(rebuild it, or pass the parquet file itself)")
    return sorted(f for f in p.rglob("*.parquet") if f.is_file())


def seal_directory(directory: os.PathLike, *, kind: str = "corpus", extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Write ``manifest.json`` for a folder of parquet parts that no builder sealed (labels copied from Modal, a corpus
    assembled by hand): every ``*.parquet`` below it, sorted by relative path, with its rows and sha256.

    Builders that know their parts (``build_corpus``, ``decontaminate``, ``build_targets``) write their own manifest;
    this is for the rest.  Sealing again replaces the manifest, so run it after the folder changed.  Interrupted
    ``*.tmp`` files are not parts.  ``extra`` adds fields (``sources``, ...) but cannot replace the listing."""
    root = Path(directory)
    reserved = {"parts", "rows", "kind", "schema_version"}
    clash = sorted(reserved & set(extra or {}))
    if clash:
        raise ValueError(f"extra cannot set {', '.join(clash)}: the listing is computed from the parts")
    _, pq = _arrow()
    files = sorted((f for f in root.rglob("*.parquet") if f.is_file()), key=lambda f: f.relative_to(root).as_posix())
    if not files:
        raise SchemaError(f"{root}: no *.parquet parts to seal")
    parts = [{"name": f.relative_to(root).as_posix(), "rows": pq.ParquetFile(f).metadata.num_rows, "sha256": sha256_file(f)} for f in files]
    manifest: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "kind": kind, "rows": sum(p["rows"] for p in parts), "parts": parts, **dict(extra or {})}
    write_manifest(root / MANIFEST_NAME, manifest)
    return manifest


def _read_rows(path: os.PathLike, columns: Sequence[str], *, require_manifest: bool = False) -> Iterator[Dict[str, Any]]:
    _, pq = _arrow()
    for file in corpus_files(path, require_manifest=require_manifest):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=BATCH_ROWS, columns=list(columns)):
            yield from batch.to_pylist()


def read_corpus(path: os.PathLike) -> Iterator[Dict[str, Any]]:
    """The rows of a corpus: a parquet file the caller names, or a directory read through its ``manifest.json``.

    A directory is never globbed: without a manifest that lists its parts (a build that was interrupted, a folder
    assembled by hand) it is refused with ``ManifestError``, because a glob would also read the parts nobody approved."""
    return _read_rows(path, CORPUS_COLUMNS, require_manifest=True)


def assert_unique_example_ids(files: Iterable[os.PathLike], *, limit: int = 5) -> int:
    """Number of ids in ``files``; ``SchemaError`` when one appears twice (SCHEMA.md section 3: ``example_id`` is unique)."""
    _, pq = _arrow()
    first_seen: Dict[str, str] = {}
    repeated: List[str] = []
    repeats = 0
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=BATCH_ROWS, columns=["example_id"]):
            for example_id in batch.column(0).to_pylist():
                if example_id in first_seen:
                    repeats += 1
                    if len(repeated) < limit:
                        repeated.append(f"{example_id!r} ({first_seen[example_id]} and {Path(file).name})")
                else:
                    first_seen[example_id] = Path(file).name
    if repeats:
        raise SchemaError(f"{repeats} example_id values repeat (SCHEMA.md section 3 requires unique ids), e.g. " + "; ".join(repeated))
    return len(first_seen)


def read_teacher(path: os.PathLike) -> Iterator[Dict[str, Any]]:
    return _read_rows(path, TEACHER_COLUMNS)


def assert_manifest_owner(directory: os.PathLike, tool: str) -> Optional[Dict[str, Any]]:
    """The ``manifest.json`` that ``tool`` wrote in ``directory`` earlier, ``None`` if there is none.

    A manifest says what a folder is.  One that ``tool`` did not write (another tool's, a hand-made one, an unreadable
    one) is not ``tool``'s to delete or replace: ``ManifestError``, and nothing is touched."""
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ManifestError(f"{path} is not valid JSON ({exc}), so {tool} cannot tell that it wrote it and will not delete or replace it; "
                            "use another --out") from exc
    owner = manifest.get("tool") if isinstance(manifest, Mapping) else None
    if owner != tool:
        raise ManifestError(f"{path} was not written by {tool} (its `tool` is {owner!r}, kind {manifest.get('kind') if isinstance(manifest, Mapping) else None!r}): "
                            f"{tool} will not delete or replace the manifest of another folder; use another --out")
    return dict(manifest)


def claim_manifest(directory: os.PathLike, tool: str) -> Optional[Dict[str, Any]]:
    """Start of a run of ``tool`` in ``directory``: remove the manifest it wrote earlier and return it.

    A folder without a manifest is an unfinished build that readers refuse, so an interrupted run never leaves the
    manifest of the run before it describing files it already replaced.  The returned manifest tells the run what the
    last finished run wrote (and so what it may delete).  A manifest ``tool`` did not write is refused (see
    :func:`assert_manifest_owner`)."""
    previous = assert_manifest_owner(directory, tool)
    if previous is not None:
        (Path(directory) / MANIFEST_NAME).unlink()
    return previous


def write_manifest(path: os.PathLike, manifest: Mapping[str, Any]) -> None:
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def iter_chunks(items: Iterable[Any], size: int) -> Iterator[List[Any]]:
    chunk: List[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
