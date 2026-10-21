"""Text normalization, n-gram hashing and canary detection shared by the decontamination and the builders."""
from __future__ import annotations

import functools
import hashlib
import re
import unicodedata
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .schema import option_texts

_WORD = re.compile(r"\w+", re.UNICODE)
MASK = (1 << 64) - 1
_MULT = 0x9E3779B97F4A7C15
_CANARY_GUID = re.compile(r"canary\s+guid\s*[:=]?\s*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_CANARY_PHRASE = re.compile(r"should\s+never\s+appear\s+in\s+training\s+corpora", re.I)


def words(text: str) -> List[str]:
    """Case-folded NFKC words (``\\w+``): punctuation and spacing never matter."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def has_letters(tokens: Sequence[str]) -> bool:
    """True when some token holds a letter: "42" or "3.5" are numbers, not statements."""
    return any(ch.isalpha() for token in tokens for ch in token)


def contains_canary(text: str) -> bool:
    """True for the BIG-bench canary (``canary GUID <uuid>``) or its "should never appear in training corpora" line.

    Whitespace of any kind (spaces, line breaks) may separate the words, and full-width letters count (NFKC)."""
    text = unicodedata.normalize("NFKC", text)
    return bool(_CANARY_GUID.search(text) or _CANARY_PHRASE.search(text))


def _strings(value: Any) -> List[str]:
    """Every string inside a JSON value, object keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [s for k, v in value.items() for s in ([k] if isinstance(k, str) else []) + _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def request_has_canary(request: Any) -> bool:
    """The canary anywhere in a request, also split over line breaks or over several texts (state, instruction, options).

    The search runs over the texts joined by line breaks, not over the JSON: there a line break is a backslash and an n.
    The texts are joined without the object keys between them (a canary split over two texts is still contiguous);
    the keys, which can hold a text of their own, are searched in a second pass."""
    return contains_canary("\n".join(leaf_texts(request))) or contains_canary("\n".join(_strings(request)))


#: Lines that only carry format, never content.  A line that matches one of them is not a "short statement" for the
#: decontamination rule ``short_line``: the SEARCH/REPLACE markers of a code-edit prompt say nothing about the item, and 213
#: of the 320 Simplicio-Loop traces were rejected for sharing one with the reference.  The list is explicit and versioned
#: (SCHEMA.md section 9 lists every entry): change it only together with ``MARKER_LINES_VERSION``, which is part of the
#: decontamination fingerprint, so a resumed run never keeps rows screened with an older list.
MARKER_LINES_VERSION = 1
#: (regex that must match the whole stripped line, case-insensitive; an example; why the line is format and not content)
MARKER_LINES: Tuple[Tuple[str, str, str], ...] = (
    (r"<{3,}\s*SEARCH", "<<<<<<< SEARCH", "opens a SEARCH/REPLACE edit block"),
    (r"={3,}", "=======", "separates the two halves of a SEARCH/REPLACE edit block (no letters, so it never counted; listed for completeness)"),
    (r">{3,}\s*REPLACE", ">>>>>>> REPLACE", "closes a SEARCH/REPLACE edit block (the Simplicio-Loop traces write four chevrons)"),
    (r"(?:`{3,}|~{3,})[\w+#.-]*", "```python", "code fence, with or without a language tag"),
)
_MARKER_RES = tuple(re.compile(pattern, re.IGNORECASE) for pattern, _, _ in MARKER_LINES)


def is_marker_line(line: str) -> bool:
    """True when the whole stripped line is one of the format-only markers of ``MARKER_LINES``."""
    stripped = line.strip()
    return any(r.fullmatch(stripped) for r in _MARKER_RES)


@functools.lru_cache(maxsize=1 << 20)
def _token_hash(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def hash64(text: str, salt: str = "") -> int:
    return int.from_bytes(hashlib.blake2b((salt + "\x00" + text).encode("utf-8"), digest_size=8).digest(), "big")


def ngram_hashes(tokens: Sequence[str], n: int) -> List[int]:
    """64-bit hash of every window of ``n`` consecutive tokens (a polynomial rolling hash over token hashes)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    size = len(tokens)
    if size < n:
        return []
    th = [_token_hash(t) for t in tokens]
    top = pow(_MULT, n - 1, 1 << 64)
    h = 0
    for k in range(n):
        h = (h * _MULT + th[k]) & MASK
    out = [h]
    for i in range(1, size - n + 1):
        h = ((h - th[i - 1] * top) * _MULT + th[i + n - 1]) & MASK
        out.append(h)
    return out


def leaf_texts(value: Any) -> List[str]:
    """Every string (and number) inside a JSON value, in order; object keys are not text."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bool):
        return [str(value).lower()]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, Mapping):
        return [t for v in value.values() for t in leaf_texts(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in leaf_texts(v)]
    return [str(value)]


def question_parts(request: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Per question: the texts a reader sees (state, instructions, option descriptions), in order."""
    state = leaf_texts(request.get("state"))
    parts: Dict[str, List[str]] = {}
    for qid, question in request.get("questions", {}).items():
        parts[qid] = state + leaf_texts(question.get("instructions")) + [
            t for d in option_texts(question) for t in leaf_texts(d)]
    return parts


def question_texts(request: Mapping[str, Any]) -> Dict[str, str]:
    """Word-normalized text of each question (state + instructions + options): the unit of the exact-match check.

    Case, punctuation and spacing do not matter, so a reformatted copy of a reference question still matches."""
    return {qid: " ".join(words("\n".join(p))) for qid, p in question_parts(request).items()}


def _distinct_lines(chunks: Sequence[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for chunk in chunks:
        for line in chunk.split("\n"):
            line = line.strip()
            if line:
                seen.setdefault(line, None)
    return list(seen)


def state_lines(request: Mapping[str, Any]) -> List[str]:
    """Distinct non-empty lines of the state."""
    return _distinct_lines(leaf_texts(request.get("state")))


def instruction_lines(request: Mapping[str, Any]) -> List[str]:
    """Distinct non-empty lines of the instructions of every question."""
    return _distinct_lines([t for q in request.get("questions", {}).values() for t in leaf_texts(q.get("instructions"))])


def option_lines(request: Mapping[str, Any]) -> List[str]:
    """Distinct non-empty lines of the option descriptions of every question."""
    return _distinct_lines([t for q in request.get("questions", {}).values() for d in option_texts(q) for t in leaf_texts(d)])


def request_lines(request: Mapping[str, Any]) -> List[str]:
    """Distinct non-empty lines of every text of the request (state once, then each question), in order."""
    chunks = list(leaf_texts(request.get("state")))
    for question in request.get("questions", {}).values():
        chunks += leaf_texts(question.get("instructions"))
        chunks += [t for d in option_texts(question) for t in leaf_texts(d)]
    return _distinct_lines(chunks)
