"""From an :class:`~decide.train.data.Example` to a model-ready item: option permutation, targets that follow it,
the reasoning policy and the length limit.  Pure Python (no torch), so the whole recipe is testable without a GPU.

For every (example, epoch) the options of each ``choice`` question are shown in a hash-chosen order
(``permute.permutation``) and the target distribution is moved with them, so the label never depends on the position
of the right option.  ``noul`` (false, true) and ``score`` (ordered levels) are never permuted.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .data import Example
from .permute import permute_request, permute_targets
from .protocol import Encoded, RequestEncoder, has_reasoning_block

REASONING_MODES = ("off", "mixed", "always")


class TooLong(ValueError):
    """The request does not fit the token limit, with or without its reasoning."""

    def __init__(self, length: int, limit: int) -> None:
        super().__init__(f"{length} tokens exceed the limit of {limit}")
        self.length, self.limit = length, limit


@dataclass(frozen=True)
class ReasoningPolicy:
    """Whether an example is shown with its thinker's reasoning.

    ``off``: never.  ``always``: whenever the example has one.  ``mixed``: for a ``fraction`` of the examples that have
    one, chosen by a hash of (seed, epoch, example id) so the choice is reproducible and changes every epoch; the
    rest train without it, which keeps the decider usable when nobody thinks.
    """

    mode: str = "off"
    fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in REASONING_MODES:
            raise ValueError(f"reasoning mode must be one of {REASONING_MODES}, got {self.mode!r}")
        if isinstance(self.fraction, bool) or not isinstance(self.fraction, (int, float)) or not math.isfinite(self.fraction) or not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"reasoning fraction must be in [0, 1], got {self.fraction!r}")

    def uses(self, example_id: str, has_reasoning: bool, *, seed: int, epoch: int) -> bool:
        if not has_reasoning or self.mode == "off":
            return False
        if self.mode == "always" or self.fraction >= 1.0:
            return True
        if self.fraction <= 0.0:
            return False
        digest = hashlib.sha256(f"decide-reasoning-v1|{seed}|{epoch}|{example_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / 2.0 ** 64 < self.fraction


@dataclass(frozen=True)
class Item:
    example_id: str
    encoded: Encoded
    #: One entry per question, in ``encoded.keys`` order and in the order the options were shown; None = no target.
    targets: Tuple[Optional[Tuple[float, ...]], ...]
    #: Question id -> permutation used (``perm[j]`` = original index shown at position j); choice questions only.
    perms: Dict[str, List[int]] = field(default_factory=dict)
    reasoning: bool = False
    #: True when the reasoning was left out because the request did not fit the limit with it.
    fell_back: bool = False

    @property
    def length(self) -> int:
        return self.encoded.length


def make_item(example: Example, encoder: RequestEncoder, *, seed: int, epoch: int = 0, alpha: float = 0.5, permute: bool = True,
              reasoning: Optional[ReasoningPolicy] = None, max_length: Optional[int] = None) -> Item:
    """Build the item of one example for one epoch.  Raises :class:`TooLong` when it cannot fit ``max_length``."""
    policy = reasoning or ReasoningPolicy()
    # a state that already holds a `Raciocínio:` block cannot take a second one (fold_reasoning refuses it): that example trains without
    show = policy.uses(example.example_id, example.reasoning is not None and not has_reasoning_block(example.request["state"]), seed=seed, epoch=epoch)
    targets = {question_id: example.target(question_id, alpha=alpha) for question_id in example.request["questions"]}
    fell_back = False
    while True:
        request = dict(example.request)
        if show:
            request["reasoning"] = example.reasoning
        shown, perms = permute_request(request, seed=seed, example_id=example.example_id, epoch=epoch, enabled=permute)
        encoded = encoder.encode(shown)
        if max_length is None or encoded.length <= max_length:
            break
        if not show:
            raise TooLong(encoded.length, max_length)
        show, fell_back = False, True
    moved = permute_targets(targets, perms)
    ordered = tuple(None if moved[key] is None else tuple(moved[key]) for key in encoded.keys)
    return Item(example_id=example.example_id, encoded=encoded, targets=ordered, perms=perms, reasoning=show, fell_back=fell_back)
