"""Reading the targets dataset (decide/data/SCHEMA.md, section 11): the trainer's only data input.

An :class:`Example` is one corpus request with, per question, the gold option index and the combined teacher
distribution; :meth:`Example.target` mixes them (``alpha * gold + (1 - alpha) * combined``).  The thinker's
reasoning, when there is one, is kept apart from the request so the trainer can decide, example by example, whether
to show it (see ``batching.ReasoningPolicy``).  No torch needed; pyarrow only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from decide.data import schema as S

from .build_targets import TARGET_COLUMNS
from .targets import mix_gold


@dataclass(frozen=True)
class Example:
    example_id: str
    source: str
    split: str
    lang: str
    request: Dict[str, Any]
    gold_index: Dict[str, Optional[int]]
    combined: Dict[str, Optional[List[float]]]
    teachers: Tuple[str, ...]
    reasoning: Optional[str]

    def options(self, question_id: str) -> int:
        """Number of options of a question (``KeyError`` for an unknown question id)."""
        return len(S.option_keys(self.request["questions"][question_id]))

    def target(self, question_id: str, *, alpha: float) -> Optional[List[float]]:
        """The training distribution over the question's options in their original order, or None when the question
        has neither a gold label nor a teacher label."""
        return mix_gold(self.combined.get(question_id), self.gold_index.get(question_id), self.options(question_id), alpha)


def example_from_row(row: Mapping[str, Any]) -> Example:
    request, gold, combined = json.loads(row["request"]), json.loads(row["gold"]), json.loads(row["combined"])
    gold_index: Dict[str, Optional[int]] = {}
    distributions: Dict[str, Optional[List[float]]] = {}
    for question_id, question in request["questions"].items():
        keys = S.option_keys(question)
        key = gold.get(question_id)
        if key is not None and key not in keys:
            raise ValueError(f"{row['example_id']}/{question_id}: gold {key!r} is not one of the options")
        gold_index[question_id] = None if key is None else keys.index(key)
        values = combined.get(question_id)
        if values is not None and len(values) != len(keys):
            raise ValueError(f"{row['example_id']}/{question_id}: {len(values)} probabilities for {len(keys)} options")
        distributions[question_id] = None if values is None else [float(v) for v in values]
    return Example(example_id=row["example_id"], source=row["source"], split=row["split"], lang=row["lang"], request=request,
                   gold_index=gold_index, combined=distributions, teachers=tuple(json.loads(row["teachers"])),
                   reasoning=row["reasoning"] or None)


def load_examples(path: "str | Path", *, split: Optional[str] = None) -> List[Example]:
    """Every example of a targets dataset (a folder of parts or one parquet file), optionally of one split."""
    return [example_from_row(row) for row in S._read_rows(path, TARGET_COLUMNS) if split is None or row["split"] == split]
