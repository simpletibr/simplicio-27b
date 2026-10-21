"""Step bookkeeping: learning-rate schedule, token-budget micro-batches and the data order of each epoch.

``learning_rate_factor`` is autojev's (Apache-2.0, see NOTICE): linear warmup, then cosine decay to a 0.1 floor.
The epoch order uses the same SHA-256 Fisher-Yates stream as ``permute`` so that a resumed run, or another machine
or Python version, visits the examples in exactly the same order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .permute import permutation


def learning_rate_factor(step: int, total_steps: int, warmup_ratio: float) -> float:
    """Multiplier of the base learning rate at 1-based ``step``."""
    warmup = max(1, int(warmup_ratio * total_steps))
    if step <= warmup:
        return step / warmup
    return 0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))


def microbatches(lengths: Sequence[int], *, max_rows: int, token_budget: int) -> List[List[int]]:
    """Group consecutive rows (callers sort by length) so that ``max(length) * rows <= token_budget``.

    A row longer than the budget still goes alone.  Returns lists of indices into ``lengths``.
    """
    groups: List[List[int]] = []
    pending: List[int] = []
    longest = 0
    for index, length in enumerate(lengths):
        if pending and (len(pending) == max_rows or max(longest, length) * (len(pending) + 1) > token_budget):
            groups.append(pending)
            pending, longest = [], 0
        pending.append(index)
        longest = max(longest, length)
    if pending:
        groups.append(pending)
    return groups


def epoch_order(rows: int, *, seed: int, epoch: int) -> List[int]:
    return permutation(rows, seed=seed, example_id="epoch-order", question_id="-", epoch=epoch)


@dataclass(frozen=True)
class StepPlan:
    rows: int
    effective_batch: int
    epochs: int

    @property
    def per_epoch(self) -> int:
        return math.ceil(self.rows / self.effective_batch)

    @property
    def total_steps(self) -> int:
        return self.per_epoch * self.epochs

    def batch_indices(self, step: int, *, seed: int, order: Optional[Sequence[int]] = None) -> List[int]:
        """Row indices of the 0-based ``step``; the last batch of an epoch is the remainder.

        ``order`` is the ``epoch_order`` of the step's epoch, when the caller already has it (computing it costs one
        hash per row, so a training loop keeps it for the whole epoch).
        """
        if not 0 <= step < self.total_steps:
            raise IndexError(f"step {step} outside 0..{self.total_steps - 1}")
        epoch, within = divmod(step, self.per_epoch)
        if order is None:
            order = epoch_order(self.rows, seed=seed, epoch=epoch)
        elif len(order) != self.rows:
            raise ValueError("order must list every row once")
        return list(order[within * self.effective_batch:(within + 1) * self.effective_batch])


def plan_steps(*, rows: int, effective_batch: int, epochs: int) -> StepPlan:
    if rows < 1 or effective_batch < 1 or epochs < 1:
        raise ValueError("rows, effective_batch and epochs must be positive")
    return StepPlan(rows, effective_batch, epochs)
