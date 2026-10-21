"""The 255 single-token answer codes of the pplx-decider checkpoints (v1.1, v1.2 and flash share them).

A..Z, then AA, AB, ... in order, skipping the two-letter codes that Qwen's tokenizer does not encode as one token.
The checkpoint's ``decision_config.json`` is the authority; ``JointModel`` checks that it carries exactly this list.
"""
from __future__ import annotations

import itertools
import string
from typing import Tuple

MAX_OPTIONS = 255
#: Two-letter codes that are not a single token in the Qwen tokenizer (measured on the v1.2 tokenizer.json).
SKIPPED_CODES = frozenset(
    "BQ BZ CJ CQ CZ DQ DZ EJ EY FJ FQ FV FZ GJ GK GQ GZ HJ IY JF JG JH JL JN JQ".split()
)


def _build() -> Tuple[str, ...]:
    candidates = list(string.ascii_uppercase) + ["".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)]
    return tuple(code for code in candidates if code not in SKIPPED_CODES)[:MAX_OPTIONS]


ANSWER_CODES: Tuple[str, ...] = _build()
assert len(ANSWER_CODES) == MAX_OPTIONS and ANSWER_CODES[-1] == "JT"
