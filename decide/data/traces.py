"""Thinker traces: the optional reasoning that the plan-v2 decider reads next to the state (SCHEMA.md, section 10).

One row per (example, thinker): the thinking text of Qwen3.8-27B (or any other thinker), how many tokens it took,
whether it finished inside the budget and the digest of the request it was made for (``request_sha256``, the
``schema.request_digest`` of the corpus request).  The rental day writes these parts; ``decide.train.build_targets``
joins them to the corpus by ``example_id`` and refuses a trace whose digest is not the one of the corpus request, so a
reasoning is never attached to a request it was not written for.  Only the standard library is needed to validate;
pyarrow is imported lazily.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

from . import schema as S

TRACE_COLUMNS = ("example_id", "thinker", "reasoning", "tokens", "finish_reason", "request_sha256")
_DIGEST = re.compile(r"[0-9a-f]{64}")
#: ``stop``: the thinker closed its thinking inside the budget.  ``length``: the budget cut it.
FINISH_REASONS = ("stop", "length")


def _text(row: Mapping[str, Any], name: str, problems: List[str]) -> None:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{name} must be a non-empty string")


def validate_trace_row(row: Mapping[str, Any]) -> List[str]:
    """Problems with one trace row (empty list = valid)."""
    missing = [c for c in TRACE_COLUMNS if c not in row]
    if missing:
        return [f"missing columns: {missing}"]
    problems: List[str] = []
    for name in ("example_id", "thinker", "reasoning"):
        _text(row, name, problems)
    tokens = row["tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0 or tokens > 2**31 - 1:
        problems.append("tokens must be an integer >= 0")
    if row["finish_reason"] not in FINISH_REASONS:
        problems.append(f"finish_reason must be one of {FINISH_REASONS}")
    digest = row["request_sha256"]
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        problems.append("request_sha256 must be 64 lowercase hex digits (schema.request_digest of the request the trace was made for)")
    return problems


def trace_row(*, example_id: str, thinker: str, reasoning: str, tokens: int, finish_reason: str, request_sha256: str) -> Dict[str, Any]:
    """Build a trace row and validate it."""
    row = {"example_id": example_id, "thinker": thinker, "reasoning": reasoning, "tokens": tokens, "finish_reason": finish_reason,
           "request_sha256": request_sha256}
    problems = validate_trace_row(row)
    if problems:
        raise S.SchemaError(f"{example_id}: " + "; ".join(problems))
    return row


def trace_arrow_schema():
    pa, _ = S._arrow()
    return pa.schema([pa.field("example_id", pa.string()), pa.field("thinker", pa.string()), pa.field("reasoning", pa.string()),
                      pa.field("tokens", pa.int32()), pa.field("finish_reason", pa.string()), pa.field("request_sha256", pa.string())])


def write_trace_part(path: "os.PathLike[str] | str", rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Atomically write one parquet part of traces; returns ``{name, rows, sha256}`` for the manifest."""
    for row in rows:
        problems = validate_trace_row(row)
        if problems:
            raise S.SchemaError(f"{row.get('example_id')}: " + "; ".join(problems))
    return S._write_part(Path(path), rows, trace_arrow_schema(), TRACE_COLUMNS)


def read_traces(path: "os.PathLike[str] | str") -> Iterator[Dict[str, Any]]:
    """Every trace row of a parquet file or of the ``*.parquet`` parts under a folder."""
    return S._read_rows(path, TRACE_COLUMNS)
