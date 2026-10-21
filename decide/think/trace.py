"""The `Raciocínio:` block: the thinker's reasoning, appended after the state of a request.

The hybrid of ADR 0004 ("pensar sob demanda") has the thinker reason about a question and the decider
read state + reasoning to return its probabilities.  This module is the one place that writes and reads
the block, so the training side (decide/data) and the inference side (the gate) cannot drift apart.

The format is decided in section 10 of decide/data/SCHEMA.md (issue #69): an optional top-level `reasoning` string of the
request, shown as a part headed `Raciocínio:` between the state and the first question, and a folded form for an engine that
only knows `state`, `describe(state).rstrip() + "\n\nRaciocínio:\n" + reasoning` (`fold_reasoning` of decide/train/protocol.py).
This module is the thinker side of that contract: it turns a row of results.jsonl into the text of `reasoning` and writes and
reads the folded form.  The two sides must give the same folded state, and the golden cases both reproduce are in
tests/fold_reasoning_cases.json (a test of this repository reads them, and fails if one diverges).  This module repeats the rule
instead of importing decide/train, which is not standard library only and belongs to another squad; the golden file is the lock.
Where it and SCHEMA.md differ, SCHEMA.md wins and this module changes.  What this module adds: a row cut by the thinker's budget
is refused unless `allow_truncated` is set (SCHEMA.md keeps a `length` trace, so set it when writing traces for the corpus).
A state that is not a string (a mapping, a list) is written as JSON, like `describe` there.

Folded layout:

    <state, trailing whitespace removed>
    <one empty line>
    Raciocínio:
    <the trace, verbatim>

* The marker is a whole line (Unicode NFC, surrounding spaces ignored).  The block is the last thing in
  the state and runs to its end, so the LAST marker line starts it.
* The trace never holds a marker line: `render_trace` and `attach_trace` write `Raciocínio :` instead.
* A state that already has a block is refused, never overwritten: strip it first with `split_trace`.
* Long traces keep their END (that is where the conclusion is) behind a one-line notice of how many
  characters were dropped.  The budget is in characters because this module is standard library only
  (roughly 3 to 4 characters per token; the real token budget is a decision of the experiment).  A budget
  under MIN_MAX_CHARS is refused (`ValueError`): it cannot hold the notice, and what is left of a clip that
  cannot is not defined by SCHEMA.md.  A cut that leaves a label line at the start of the tail is neutralized
  like any other label line, so the folded state has exactly one.

The sample of the GT and the public suite are evaluation data: traces made from them never go into a
training corpus (ADR 0004, item 6).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple

MARKER = "Raciocínio:"
DEFAULT_MAX_CHARS = 8000
MIN_MAX_CHARS = 64            # the notice of a clip is about 35 characters: below this the clip falls back to a raw tail that the two ways to write the block treat differently
MODES = ("reasoning", "final", "both")
GT_MANIFEST = Path(__file__).with_name("gt-sample-manifest.json")
_GT_IDS: Dict[Path, FrozenSet[str]] = {}


class TraceError(ValueError):
    """The state already has a block, or the text cannot be turned into one."""


class EvaluationDataError(TraceError):
    """The request is one of the gate's sample: its traces are evaluation data and never go into a corpus (ADR 0004, item 6)."""


def gt_sample_ids(manifest: Optional[Path] = None) -> FrozenSet[str]:
    """The run ids of the sample in a manifest (default: the committed gt-sample-manifest.json)."""
    path = Path(manifest) if manifest else GT_MANIFEST
    if path not in _GT_IDS:
        data = json.loads(path.read_text(encoding="utf-8"))
        _GT_IDS[path] = frozenset(rid for b in (data.get("benchmarks") or {}).values() for rid in b.get("run_ids", []))
    return _GT_IDS[path]


def _refuse_eval_data(source_id: Optional[str]) -> None:
    if source_id is not None and source_id in gt_sample_ids():
        raise EvaluationDataError(f"{source_id} is a request of the gate's sample: a trace made on it is evaluation data and must never enter "
                                  "a training corpus (ADR 0004, item 6)")


def describe_state(state: Any) -> str:
    """The text of a state: a string as it is, anything else as JSON (`describe` of decide/train/protocol.py)."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _is_marker(line: str) -> bool:
    return unicodedata.normalize("NFC", line).strip() == MARKER


def has_trace(state: Any) -> bool:
    return any(_is_marker(line) for line in describe_state(state).split("\n"))


def _clean(text: Optional[str]) -> str:
    """Normalize newlines, trim line ends, keep at most one blank line in a row, neutralize marker lines."""
    if not text:
        return ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    for line in lines:
        line = line.rstrip(" \t")
        if _is_marker(line):
            line = line[: len(line) - len(line.lstrip())] + "Raciocínio :"
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n")


def clip(text: str, max_chars: int) -> str:
    """At most `max_chars` characters of `text`: the end of it, on a line boundary, after a notice of what was dropped."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    reserve = len("[...  caracteres omitidos ...]") + len(str(len(text))) + 1          # notice + newline, worst case
    budget = max_chars - reserve
    if budget <= 0:
        return text[-max_chars:]
    tail = text[-budget:]
    if text[-budget - 1] != "\n" and "\n" in tail:
        rest = tail.split("\n", 1)[1]
        if rest:
            tail = rest
    return f"[... {len(text) - len(tail)} caracteres omitidos ...]\n{tail}"


def _check_budget(max_chars: int) -> None:
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < MIN_MAX_CHARS:
        raise ValueError(f"max_chars must be an integer of at least {MIN_MAX_CHARS} (room for the notice of a clip), got {max_chars!r}")


def render_trace(reasoning: Optional[str] = None, final: Optional[str] = None, *, mode: str = "reasoning",
                 max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The text that goes under the marker.  `mode` picks the thinker's reasoning, its final answer text, or both."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    _check_budget(max_chars)
    parts = []
    if mode in ("reasoning", "both"):
        parts.append(_clean(reasoning))
    if mode in ("final", "both"):
        parts.append(_clean(final))
    return _clean(clip("\n\n".join(p for p in parts if p), max_chars))             # a cut can leave a label line at the start of the tail


def attach_trace(state: Any, trace: Optional[str], *, source_id: Optional[str] = None) -> str:
    """`state` followed by the block.  An empty trace adds nothing.  Refuses a state that already has a block, and, when the
    request is named by `source_id`, one of the gate's sample (evaluation data)."""
    body = _clean(trace)
    if not body.strip():
        return state
    _refuse_eval_data(source_id)
    if has_trace(state):
        raise TraceError("the state already has a Raciocínio: block; split_trace() it first")
    head = describe_state(state).rstrip()
    return (head + "\n\n" if head else "") + MARKER + "\n" + body


def fold_reasoning(state: Any, reasoning: Any, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The folded state of SCHEMA.md section 10: `fold_reasoning` of decide/train/protocol.py, for a state and a reasoning text.

    The state loses the whitespace at its end, then come one blank line (none for an empty state), the label line and the reasoning
    after `_clean`, clipped to `max_chars`.  A state that already has a label line and a `reasoning` that is not a non-empty string
    are refused, never repaired.  (SCHEMA.md also refuses a reasoning that is empty after cleaning and clipping; with a budget of at
    least MIN_MAX_CHARS and a reasoning that is not blank that cannot happen.)
    """
    _check_budget(max_chars)
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise TraceError("'reasoning' must be a non-empty string")
    if has_trace(state):
        raise TraceError("the state already holds a Raciocínio: block, so it cannot also take a reasoning")
    body = _clean(clip(_clean(reasoning), max_chars))
    head = describe_state(state).rstrip()
    return (head + "\n\n" if head else "") + MARKER + "\n" + body


def split_trace(state: str) -> Tuple[str, Optional[str]]:
    """(state without the block, trace) or (state, None).  Inverse of `attach_trace` for a clean trace."""
    lines = state.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if _is_marker(lines[i]):
            return "\n".join(lines[:i]).rstrip(), "\n".join(lines[i + 1:])
    return state, None


def from_record(rec: Dict[str, Any], *, mode: str = "reasoning", max_chars: int = DEFAULT_MAX_CHARS,
                allow_truncated: bool = False) -> Optional[str]:
    """The trace of one results row of thinker.py, or None when the row cannot give one.

    Not usable: a row that is not `ok`, one whose budget ran out (no conclusion) unless `allow_truncated`,
    and one without the requested text (a run without --keep-reasoning keeps only the final answer).  A row of the gate's
    sample (its id is in gt-sample-manifest.json) is refused with `EvaluationDataError`: it is evaluation data.
    """
    _refuse_eval_data(rec.get("id"))
    if rec.get("status") != "ok" or (rec.get("truncated") and not allow_truncated):
        return None
    return render_trace(rec.get("reasoning"), rec.get("final"), mode=mode, max_chars=max_chars) or None
