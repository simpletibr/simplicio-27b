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
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

SCHEMA_VERSION = 1
MAX_OPTIONS = 255
QUESTION_TYPES = ("choice", "noul", "score")
CORPUS_COLUMNS = ("example_id", "source", "license", "split", "lang", "request", "gold")
TEACHER_COLUMNS = ("example_id", "question_id", "teacher", "codes", "probs", "temperature")
SPLITS = ("train", "dev")
PROB_TOLERANCE = 1e-3
BATCH_ROWS = 20_000
_LANG = re.compile(r"^(mul|[a-z]{2,3}(-[A-Za-z0-9]{2,8})*)$")


class SchemaError(ValueError):
    """A row or request does not follow the schema; the message lists every problem found."""


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


def validate_request(request: Any) -> List[str]:
    """Problems with a request (empty list = valid).  Mirrors ``Encoder.validate`` of pplx-decider v1.2."""
    if not isinstance(request, Mapping):
        return ["request must be an object"]
    problems: List[str] = []
    if "state" not in request:
        problems.append("missing 'state'")
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
        if kind == "choice" and not isinstance(question.get("criteria"), Mapping):
            problems.append(f"{qid}: choice criteria must be an object (option key -> description)")
            continue
        if kind == "score" and not isinstance(question.get("criteria"), list):
            problems.append(f"{qid}: score criteria must be a list of level descriptions")
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
        raise ImportError("pyarrow is required for parquet files: pip install pyarrow") from exc
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


def corpus_files(path: os.PathLike) -> List[Path]:
    """The parquet files of a dataset: the file itself, or every ``*.parquet`` under a directory, sorted."""
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.rglob("*.parquet") if f.is_file())
    return [p]


def _read_rows(path: os.PathLike, columns: Sequence[str]) -> Iterator[Dict[str, Any]]:
    _, pq = _arrow()
    for file in corpus_files(path):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=BATCH_ROWS, columns=list(columns)):
            yield from batch.to_pylist()


def read_corpus(path: os.PathLike) -> Iterator[Dict[str, Any]]:
    return _read_rows(path, CORPUS_COLUMNS)


def read_teacher(path: os.PathLike) -> Iterator[Dict[str, Any]]:
    return _read_rows(path, TEACHER_COLUMNS)


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
