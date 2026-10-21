"""Temperature fitting and calibration metrics on raw logits.

Adapted from ``autojev/evaluate.py`` of perplexity-ai/pplx-decider-v1.1-27b (Apache License 2.0; see LICENSE,
NOTICE).  Changes: only ``fit_temperature`` and the metrics are kept, and they work on plain lists of logits and
integer labels instead of ``Example``/``Prediction`` records (no httpx, no Jev endpoint, no bootstrap);
``fit_temperatures`` fits one temperature per question type (the plan calibrates choice, noul and score apart).

Convention: probabilities are ``softmax(logits / T)``.  ``T > 1`` softens, ``T < 1`` sharpens.  The fit minimises the
negative log-likelihood of the gold labels with a golden-section search over ``log T`` in ``[ln 0.05, ln 20]``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ECE_BINS = 15
T_MIN, T_MAX = 0.05, 20.0


def _padded(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    if not logits or len(logits) != len(labels):
        raise ValueError("Temperature fitting needs nonempty matching logits and hard labels")
    width = max(map(len, logits))
    values = np.full((len(logits), width), -np.inf, dtype=np.float64)
    for index, (row, label) in enumerate(zip(logits, labels)):
        if not len(row) or not 0 <= label < len(row) or not np.isfinite(np.asarray(row, dtype=np.float64)).all():
            raise ValueError("Temperature fitting requires finite, unpadded valid logits")
        values[index, :len(row)] = row
    return values, np.asarray(labels, dtype=np.int64)


def fit_temperature(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """The ``T`` that minimises the NLL of ``labels`` under ``softmax(logits / T)``."""
    values, target = _padded(logits, labels)
    values = values - values.max(axis=1, keepdims=True)
    target_logits = values[np.arange(len(values)), target]

    def loss(log_temperature: float) -> float:
        inverse = math.exp(-log_temperature)
        return float(np.mean(np.log(np.exp(values * inverse).sum(axis=1)) - target_logits * inverse))

    low, high = math.log(T_MIN), math.log(T_MAX)
    ratio = (math.sqrt(5) - 1) / 2
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    left_loss, right_loss = loss(left), loss(right)
    for _ in range(80):
        if left_loss <= right_loss:
            high, right, right_loss = right, left, left_loss
            left = high - ratio * (high - low)
            left_loss = loss(left)
        else:
            low, left, left_loss = left, right, right_loss
            right = low + ratio * (high - low)
            right_loss = loss(right)
    candidates = [math.log(T_MIN), (low + high) / 2, math.log(T_MAX), 0.0]
    return math.exp(min(candidates, key=loss))


def fit_temperatures(items: Iterable[Tuple[str, Sequence[float], int]], *, minimum: int = 50) -> Dict[str, Any]:
    """One temperature per group (``all`` plus each question type seen).

    ``items`` are ``(group, logits, label_index)``.  A group with fewer than ``minimum`` rows is not fitted on its
    own: it gets the ``all`` temperature, and ``fallback`` lists it so the report says so.
    """
    rows = list(items)
    if not rows:
        raise ValueError("Temperature fitting needs at least one labelled question")
    overall = fit_temperature([logits for _, logits, _ in rows], [label for _, _, label in rows])
    groups: Dict[str, List[Tuple[Sequence[float], int]]] = {}
    for group, logits, label in rows:
        groups.setdefault(group, []).append((logits, label))
    result: Dict[str, Any] = {"all": overall, "counts": {"all": len(rows)}, "fallback": []}
    for group, members in sorted(groups.items()):
        result["counts"][group] = len(members)
        if len(members) < minimum:
            result[group] = overall
            result["fallback"].append(group)
        else:
            result[group] = fit_temperature([logits for logits, _ in members], [label for _, label in members])
    return result


def metrics(logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float = 1.0) -> Dict[str, Any]:
    """Accuracy, NLL, Brier and ECE (15 equal-width confidence bins) of ``softmax(logits / temperature)``."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    values, target = _padded(logits, labels)
    scaled = (values - values.max(axis=1, keepdims=True)) / temperature
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    count = len(values)
    rows = np.arange(count)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities[rows, predicted]
    correct = (predicted == target).astype(np.float64)
    onehot = np.zeros_like(probabilities)
    onehot[rows, target] = 1.0
    nll = -np.log(np.maximum(probabilities[rows, target], 1e-12))
    bins = np.minimum(ECE_BINS - 1, (confidence * ECE_BINS).astype(np.int64))
    ece = 0.0
    for index in range(ECE_BINS):
        selected = bins == index
        if selected.any():
            ece += float(selected.sum()) / count * abs(float(confidence[selected].mean()) - float(correct[selected].mean()))
    return {
        "count": count,
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(np.square(probabilities - onehot).sum(axis=1).mean()),
        "ece": ece,
        "temperature": float(temperature),
    }
