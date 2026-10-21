"""Soft cross-entropy (plus an optional Brier term) over the answer options, and the option mask.

Adapted from ``autojev/loss.py`` of perplexity-ai/pplx-decider-v1.1-27b (Apache License 2.0; see LICENSE, NOTICE).
Changes: per-question ``weights`` (a question that has no target gets weight 0, so it adds neither loss nor
gradient); ``mask_logits`` lives here, next to the loss it protects, instead of inside the model; a batch whose
weights are all zero returns a zero that still back-propagates (DDP needs every rank to call ``backward``).
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Union

import torch
import torch.nn.functional as F

#: Logit given to options a question does not have.  Finite on purpose: ``-inf`` would make ``0 * -inf`` a NaN.
MASK_VALUE = -1e9


class DecisionLoss(NamedTuple):
    total: torch.Tensor
    cross_entropy: torch.Tensor
    brier: torch.Tensor


def mask_logits(logits: torch.Tensor, counts: Union[torch.Tensor, Sequence[int]]) -> torch.Tensor:
    """Options at or beyond a question's option count get ``MASK_VALUE``: no probability and no gradient."""
    counts = torch.as_tensor(counts, device=logits.device)
    beyond = torch.arange(logits.shape[-1], device=logits.device)[None] >= counts[:, None]
    return logits.masked_fill(beyond, MASK_VALUE)


def decision_loss(logits: torch.Tensor, target: torch.Tensor, brier_weight: float = 0.0,
                  weights: Optional[torch.Tensor] = None) -> DecisionLoss:
    """Mean over questions of ``-sum(target * log_softmax(logits))`` (and of the Brier score).

    ``weights`` (one per row) turns the mean into a weighted mean; rows with weight 0 are ignored.
    """
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    cross_entropy_rows = -(target * log_probabilities).sum(-1)
    brier_rows = (log_probabilities.exp() - target).square().sum(-1)
    if weights is None:
        cross_entropy, brier = cross_entropy_rows.mean(), brier_rows.mean()
    else:
        weights = weights.to(device=cross_entropy_rows.device, dtype=cross_entropy_rows.dtype)
        mass = weights.sum()
        if float(mass) <= 0.0:
            zero = logits.float().sum() * 0.0
            return DecisionLoss(zero, zero, zero)
        cross_entropy = (cross_entropy_rows * weights).sum() / mass
        brier = (brier_rows * weights).sum() / mass
    total = cross_entropy if brier_weight == 0 else cross_entropy + brier_weight * brier
    return DecisionLoss(total, cross_entropy, brier)
