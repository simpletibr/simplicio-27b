"""Option-order test-time augmentation (TTA): present the options in another order, map the answer back, average.

A decision model reads the options as a list, so its answer carries a small bias toward some positions.
Asking the same question with the options in a second order and averaging the two distributions (after
mapping each back to the original order) cancels part of that bias.  This module is pure Python: it owns
the permutations and their exact inverse, nothing else.

Conventions (a permutation ``perm`` of ``range(n)``):

* ``presented[i] = original[perm[i]]``                       (what the model sees, ``apply``)
* ``original[perm[i]] = presented[i]``                       (what we recover, ``unpermute``)

Only ``choice`` questions with at least two options are permuted.  ``noul`` questions have fixed
false/true options and ``score`` questions are ordered scales: both keep a single view.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

MODES = ("reverse", "rotate")


def identity(n: int) -> List[int]:
    return list(range(n))


def rotation(n: int, shift: int) -> List[int]:
    """presented[i] = original[(i + shift) % n]."""
    if n == 0:
        return []
    return [(i + shift) % n for i in range(n)]


def permutations(n: int, views: int, mode: str = "reverse") -> List[List[int]]:
    """Distinct option orders for a question of `n` options, the identity first.

    mode "reverse": identity, then the reversed order (the second view d3 itself averages), then evenly
    spaced cyclic shifts.  mode "rotate": evenly spaced cyclic shifts only.  Duplicate orders are dropped,
    so a two-option question never runs more than two views.
    """
    if mode not in MODES:
        raise ValueError(f"tta mode must be one of {MODES}")
    if views < 1:
        raise ValueError("views must be >= 1")
    if n < 2 or views == 1:
        return [identity(n)]
    candidates: List[List[int]] = [identity(n)]
    if mode == "reverse":
        candidates.append(list(reversed(range(n))))
        extra = range(2, views)
    else:
        extra = range(1, views)
    for v in extra:
        candidates.append(rotation(n, (v * n) // views))
    out: List[List[int]] = []
    for perm in candidates:
        if perm not in out:
            out.append(perm)
    return out[:views]


def check(perm: Sequence[int]) -> None:
    if sorted(perm) != list(range(len(perm))):
        raise ValueError(f"not a permutation: {list(perm)}")


def apply(seq: Sequence[Any], perm: Sequence[int]) -> List[Any]:
    """The sequence as the model sees it."""
    check(perm)
    if len(seq) != len(perm):
        raise ValueError("sequence and permutation differ in length")
    return [seq[p] for p in perm]


def invert(perm: Sequence[int]) -> List[int]:
    check(perm)
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def unpermute(presented: Sequence[Any], perm: Sequence[int]) -> List[Any]:
    """Map values given per presented position back to the original option order."""
    check(perm)
    if len(presented) != len(perm):
        raise ValueError("values and permutation differ in length")
    original: List[Any] = [None] * len(perm)
    for i, p in enumerate(perm):
        original[p] = presented[i]
    return original


def average(rows: Sequence[Sequence[float]]) -> List[float]:
    """Element-wise mean of equally long probability rows (already in the original option order)."""
    if not rows:
        raise ValueError("nothing to average")
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ValueError("rows differ in length")
    return [sum(r[i] for r in rows) / len(rows) for i in range(n)]


def permutable(question: Mapping[str, Any]) -> bool:
    return question.get("type") == "choice" and isinstance(question.get("criteria"), Mapping) and len(question["criteria"]) >= 2


def plan(request: Mapping[str, Any], views: int, mode: str = "reverse") -> List[Dict[str, List[int]]]:
    """One {question_id: permutation} map per view, view 0 being the original order.

    Every question takes part in as many views as there are views overall: a question with fewer distinct
    orders than that (two options, or not permutable) repeats its last order, which keeps the views aligned
    so a request can be run once per view and the answers mapped back question by question.
    """
    per_question: Dict[str, List[List[int]]] = {}
    for qid, q in request["questions"].items():
        n = len(q["criteria"]) if permutable(q) else 0
        per_question[qid] = permutations(n, views, mode) if n else [[]]
    count = max((len(p) for p in per_question.values()), default=1)
    return [{qid: perms[min(v, len(perms) - 1)] for qid, perms in per_question.items()} for v in range(count)]


def permute_request(request: Mapping[str, Any], perms: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    """A copy of the request whose choice questions list their options in the permuted order."""
    questions: Dict[str, Any] = {}
    for qid, q in request["questions"].items():
        perm = perms.get(qid)
        if perm and permutable(q):
            keys = list(q["criteria"])
            questions[qid] = {**q, "criteria": {keys[p]: q["criteria"][keys[p]] for p in perm}}
        else:
            questions[qid] = q
    return {**request, "questions": questions}


def restore_question(values_by_presented: Sequence[float], perm: Sequence[int]) -> List[float]:
    """Probabilities (or logits) given in presented order -> original option order."""
    return unpermute(values_by_presented, perm) if perm else list(values_by_presented)
