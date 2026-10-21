"""Per-example option permutation with an exact inverse.

Choice options are shown in a different order for every (example, question), chosen by a hash so that it is
reproducible anywhere (no dependence on Python's ``hash``, on the RNG state or on the order examples are visited):
resuming a run, or running on another machine, shows the same order.  The algorithm is documented and pinned by
golden values in tests/test_contract_69.py:

    key = "decide-permute-v1|{seed}|{epoch}|{example_id}|{question_id}|{n}"
    draws = SHA-256(key || counter) read as 64-bit big-endian words
    Fisher-Yates from the end: j = draw % (i + 1); swap(perm[i], perm[j])

``perm[j]`` is the original index shown at new position ``j``.  Only ``choice`` questions are permuted: ``noul``
is always [false, true] and ``score`` levels are ordered.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple, TypeVar

T = TypeVar("T")


def _draws(seed: int, example_id: str, question_id: str, n: int, epoch: int) -> Iterator[int]:
    key = f"decide-permute-v1|{seed}|{epoch}|{example_id}|{question_id}|{n}".encode("utf-8")
    counter = 0
    while True:
        block = hashlib.sha256(key + counter.to_bytes(4, "big")).digest()
        for offset in range(0, 32, 8):
            yield int.from_bytes(block[offset:offset + 8], "big")
        counter += 1


def permutation(n: int, *, seed: int, example_id: str, question_id: str, epoch: int = 0) -> List[int]:
    perm = list(range(n))
    draws = _draws(seed, example_id, question_id, n, epoch)
    for i in range(n - 1, 0, -1):
        j = next(draws) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def invert(perm: Sequence[int]) -> List[int]:
    inverse = [0] * len(perm)
    for new_position, original in enumerate(perm):
        inverse[original] = new_position
    return inverse


def apply_order(values: Sequence[T], perm: Sequence[int]) -> List[T]:
    """``new[j] = values[perm[j]]``."""
    return [values[p] for p in perm]


def restore_order(values: Sequence[T], perm: Sequence[int]) -> List[T]:
    """Exact inverse of :func:`apply_order`: ``restore_order(apply_order(x, p), p) == x``."""
    original: List[Any] = [None] * len(perm)
    for new_position, source in enumerate(perm):
        original[source] = values[new_position]
    return original


def permute_request(request: Mapping[str, Any], *, seed: int, example_id: str, epoch: int = 0,
                    enabled: bool = True) -> Tuple[Dict[str, Any], Dict[str, List[int]]]:
    """A copy of the request with the options of every choice question reordered, and the permutations used."""
    new = copy.deepcopy(dict(request))
    perms: Dict[str, List[int]] = {}
    if not enabled:
        return new, perms
    for qid, question in new["questions"].items():
        if question.get("type") != "choice":
            continue
        items = list(question["criteria"].items())
        perm = permutation(len(items), seed=seed, example_id=example_id, question_id=qid, epoch=epoch)
        question["criteria"] = dict(apply_order(items, perm))
        perms[qid] = perm
    return new, perms


def permute_targets(targets: Mapping[str, Any], perms: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    """Targets (original option order) -> targets in the permuted order.  Questions without a permutation pass through."""
    return {qid: (apply_order(values, perms[qid]) if qid in perms and values is not None else values) for qid, values in targets.items()}


def restore_targets(values: Mapping[str, Any], perms: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    """Probabilities in the permuted order -> original option order (exact inverse of :func:`permute_targets`)."""
    return {qid: (restore_order(vals, perms[qid]) if qid in perms and vals is not None else vals) for qid, vals in values.items()}
