"""Text normalization, n-gram hashing and canary detection shared by the decontamination and the builders."""
from __future__ import annotations

import functools
import hashlib
import re
import unicodedata
from typing import Any, Dict, List, Mapping, Sequence

from .schema import option_texts

_WORD = re.compile(r"\w+", re.UNICODE)
_SPACE = re.compile(r"\s+")
MASK = (1 << 64) - 1
_MULT = 0x9E3779B97F4A7C15
_CANARY_GUID = re.compile(r"canary\s+guid\s*[:=]?\s*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_CANARY_PHRASE = re.compile(r"should\s+never\s+appear\s+in\s+training\s+corpora", re.I)


def normalize(text: str) -> str:
    """NFKC, case-folded, whitespace collapsed."""
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def words(text: str) -> List[str]:
    """Case-folded NFKC words (``\\w+``): punctuation and spacing never matter."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def contains_canary(text: str) -> bool:
    """True for the BIG-bench canary (``canary GUID <uuid>``) or its "should never appear in training corpora" line."""
    return bool(_CANARY_GUID.search(text) or _CANARY_PHRASE.search(text))


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


def request_lines(request: Mapping[str, Any]) -> List[str]:
    """Distinct non-empty lines of every text of the request (state once, then each question), in order."""
    seen: Dict[str, None] = {}
    state = leaf_texts(request.get("state"))
    chunks = list(state)
    for question in request.get("questions", {}).values():
        chunks += leaf_texts(question.get("instructions"))
        chunks += [t for d in option_texts(question) for t in leaf_texts(d)]
    for chunk in chunks:
        for line in chunk.split("\n"):
            line = line.strip()
            if line:
                seen.setdefault(line, None)
    return list(seen)
