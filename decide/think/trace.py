"""The `Raciocínio:` block: the thinker's reasoning, appended after the state of a request.

The hybrid of ADR 0004 ("pensar sob demanda") has the thinker reason about a question and the decider
read state + reasoning to return its probabilities.  This module is the one place that writes and reads
the block, so the training side (decide/data) and the inference side (the gate) cannot drift apart.

The format is decided in section 10 of decide/data/SCHEMA.md (issue #69; it was on the branch decide/69-treino,
not on main, when this was checked): an optional top-level `reasoning` string of the request, shown as a part headed
`Raciocínio:` between the state and the first question, and a folded form for an engine that only knows `state`,
`describe(state) + "\n\nRaciocínio:\n" + reasoning` (`fold_reasoning` of decide/train/protocol.py there).  This module
is the thinker side of that contract: it turns a row of results.jsonl into the text of `reasoning` and writes and
reads the folded form.  Where it and SCHEMA.md differ, SCHEMA.md wins and this module changes.  What differs, found
by READING fold_reasoning (not by running it): the trailing whitespace of the state is removed here, marker lines
inside the trace are neutralized, and a row cut by the thinker's budget is refused unless `allow_truncated` is set
(SCHEMA.md keeps a `length` trace, so set it when writing traces for the corpus).

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
  (roughly 3 to 4 characters per token; the real token budget is a decision of the experiment).

The sample of the GT and the public suite are evaluation data: traces made from them never go into a
training corpus (ADR 0004, item 6).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional, Tuple

MARKER = "Raciocínio:"
DEFAULT_MAX_CHARS = 8000
MODES = ("reasoning", "final", "both")


class TraceError(ValueError):
    """The state already has a block, or the text cannot be turned into one."""


def _is_marker(line: str) -> bool:
    return unicodedata.normalize("NFC", line).strip() == MARKER


def has_trace(state: str) -> bool:
    return any(_is_marker(line) for line in state.split("\n"))


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


def render_trace(reasoning: Optional[str] = None, final: Optional[str] = None, *, mode: str = "reasoning",
                 max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The text that goes under the marker.  `mode` picks the thinker's reasoning, its final answer text, or both."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    parts = []
    if mode in ("reasoning", "both"):
        parts.append(_clean(reasoning))
    if mode in ("final", "both"):
        parts.append(_clean(final))
    return clip("\n\n".join(p for p in parts if p), max_chars)


def attach_trace(state: str, trace: Optional[str]) -> str:
    """`state` followed by the block.  An empty trace adds nothing.  Refuses a state that already has a block."""
    body = _clean(trace)
    if not body.strip():
        return state
    if has_trace(state):
        raise TraceError("the state already has a Raciocínio: block; split_trace() it first")
    head = state.rstrip()
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
    and one without the requested text (a run with --no-reasoning keeps only the final answer).
    """
    if rec.get("status") != "ok" or (rec.get("truncated") and not allow_truncated):
        return None
    return render_trace(rec.get("reasoning"), rec.get("final"), mode=mode, max_chars=max_chars) or None
