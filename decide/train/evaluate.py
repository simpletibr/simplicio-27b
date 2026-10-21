"""Evaluation of a joint model on a split of the targets dataset: raw logits, calibration temperatures, metrics.

Only questions with a gold label count (the teacher distributions are soft; accuracy, NLL and ECE need a hard label).
Examples are shown in their original option order (no permutation).  The temperature is fitted per question type
(``calibrate.fit_temperatures``).  To report an honest calibrated number, half of the examples (chosen by a hash of the
example id) fit a temperature and the other half is scored with it (``heldout``); the temperature that goes into the
checkpoint is fitted on all of them (``fitted.temperature``).
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from . import calibrate
from .batching import Item, ReasoningPolicy, TooLong, make_item
from .data import Example
from .model import JointModel, collate
from .protocol import RequestEncoder
from .schedule import microbatches


def fold_of(example_id: str) -> int:
    """0 or 1, by a hash of the id: the same example is always on the same side."""
    return hashlib.sha256(("decide-fold-v1|" + example_id).encode("utf-8")).digest()[0] & 1


@torch.inference_mode()
def question_logits(model: JointModel, items: Sequence[Item], pad_id: int, *, token_budget: int = 16384, max_rows: int = 32,
                    device: "str | torch.device" = "cpu") -> List[List[List[float]]]:
    """Raw logits (valid options only) of every question of every item, in the order of ``items``."""
    was_training = model.training
    model.eval()
    try:
        order = sorted(range(len(items)), key=lambda i: items[i].length)
        results: List[Optional[List[List[float]]]] = [None] * len(items)
        for group in microbatches([items[i].length for i in order], max_rows=max_rows, token_budget=token_budget):
            chosen = [order[g] for g in group]
            batch = collate([items[i].encoded for i in chosen], pad_id, device)
            logits = model(batch).cpu()
            cursor = 0
            for i in chosen:
                counts = items[i].encoded.counts
                results[i] = [logits[cursor + q, :count].tolist() for q, count in enumerate(counts)]
                cursor += len(counts)
    finally:
        model.train(was_training)
    return [r if r is not None else [] for r in results]


def evaluate_examples(model: JointModel, encoder: RequestEncoder, examples: Sequence[Example], *, reasoning: bool = False, seed: int = 0,
                      max_length: Optional[int] = None, token_budget: int = 16384, max_rows: int = 32,
                      device: "str | torch.device" = "cpu", minimum: int = 50) -> Dict[str, Any]:
    """Accuracy, NLL, Brier, ECE and temperatures on the gold-labelled questions of ``examples``.

    ``reasoning=True`` scores only the examples that carry a reasoning and shows it; ``False`` never shows it.
    """
    policy = ReasoningPolicy("always" if reasoning else "off")
    chosen: List[Example] = []
    items: List[Item] = []
    too_long = 0
    for example in examples:
        if reasoning and example.reasoning is None:
            continue
        if all(index is None for index in example.gold_index.values()):
            continue
        try:
            item = make_item(example, encoder, seed=seed, epoch=0, permute=False, reasoning=policy, max_length=max_length)
        except TooLong:
            too_long += 1
            continue
        chosen.append(example)
        items.append(item)
    report: Dict[str, Any] = {"examples": len(items), "skipped_too_long": too_long, "reasoning": reasoning}
    if not items:
        return report
    logits = question_logits(model, items, encoder.pad_id, token_budget=token_budget, max_rows=max_rows, device=device)
    rows: List[Tuple[str, Sequence[float], int, str]] = []
    for example, item, per_question in zip(chosen, items, logits):
        for key, values in zip(item.encoded.keys, per_question):
            label = example.gold_index.get(key)
            if label is not None:
                rows.append((example.request["questions"][key]["type"], values, label, example.example_id))
    if not rows:
        return report
    values = [r[1] for r in rows]
    labels = [r[2] for r in rows]
    fit = calibrate.fit_temperatures([(r[0], r[1], r[2]) for r in rows], minimum=minimum)
    report.update({"questions": len(rows), "raw": calibrate.metrics(values, labels, 1.0), "fitted": calibrate.metrics(values, labels, fit["all"]),
                   "temperature": fit})
    fit_side = [r for r in rows if fold_of(r[3]) == 0]
    score_side = [r for r in rows if fold_of(r[3]) == 1]
    if fit_side and score_side:
        held = calibrate.fit_temperature([r[1] for r in fit_side], [r[2] for r in fit_side])
        report["heldout"] = calibrate.metrics([r[1] for r in score_side], [r[2] for r in score_side], held)
    return report
