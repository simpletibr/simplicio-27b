"""Calibration of the decision head: temperature per question type, NLL / Brier / ECE, holdout protocol.

The model returns one logit per option.  Dividing the logits of ONE view by a temperature T before the softmax changes how
confident the probabilities are but not which option wins.  With option-order TTA the probabilities are the mean of one softmax
per view, and T decides how much each view weighs against the others, so T can change which option wins: with the views
[[6, 0, 5], [0, 2, 0]] option 1 wins at T=0.5 (0.482 against 0.449) and option 0 wins at T=4 (0.387 against 0.332).  So with TTA a
temperature moves argmax-based scores as well as NLL, Brier and ECE: it is a model parameter, to be fitted on data that is not
evaluation data (ADR 0004, item 6; `api.fit_temperature` enforces it) and to be reported with the score.

Teacher labels follow from the same fact.  A teacher's probabilities are made from the per-view raw logits that the engine keeps in
`raw["logits"]` (`reprobabilities(raw["logits"][qid], T)`: one softmax per view at the teacher's T, then the mean), not by
re-temperaturing the probabilities in the response, which are already a mean at one T and cannot be turned into the mean at another.

Everything here works on :class:`Item` records (raw logits per view, original option order, gold index) and uses
only numpy, so it runs anywhere.  The caller decides where the logits and gold labels come from.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

EPS = 1e-12
GROUPINGS = ("question_type", "type_k")
DEFAULT_GRID = (0.5, 4.0)
COARSE_POINTS = 33
GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0

TemperatureLike = Union[float, Callable[["Item"], float]]


@dataclass(frozen=True)
class Item:
    """One labelled question: `logits[view][option]` are raw (before any temperature), in the original option order."""

    run_id: str
    question_id: str
    qtype: str
    logits: Sequence[Sequence[float]]
    gold: Optional[int]


# --------------------------------------------------------------------------- probabilities

def _softmax(z: np.ndarray, t: float) -> np.ndarray:
    """Softmax of z / t along the last axis, stable for large logits."""
    if not t > 0.0 or not math.isfinite(t):
        raise ValueError(f"temperature must be a positive finite number, got {t!r}")
    x = np.asarray(z, dtype=np.float64) / t
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def softmax(z: Sequence[float], t: float = 1.0) -> np.ndarray:
    return _softmax(np.asarray(z, dtype=np.float64), t)


def _views(item: Item, views: Optional[Sequence[int]] = None) -> np.ndarray:
    z = np.asarray(item.logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 1:
        raise ValueError(f"logits of {item.run_id}/{item.question_id} must be views x options, got shape {z.shape}")
    return z if views is None else z[list(views)]


def item_probs(item: Item, t: float, views: Optional[Sequence[int]] = None) -> np.ndarray:
    """Mean over views of softmax(logits / t): the distribution TTA reports."""
    return _softmax(_views(item, views), t).mean(axis=0)


def reprobabilities(logits_by_view: Sequence[Sequence[float]], t: float) -> List[float]:
    """The same mean-of-softmaxes from bare logits (what the engine stores), as plain floats."""
    return [float(x) for x in _softmax(np.asarray(logits_by_view, dtype=np.float64), t).mean(axis=0)]


# --------------------------------------------------------------------------- metrics

def _stack_bucket(members: Sequence[Tuple[np.ndarray, int]]) -> Tuple[np.ndarray, np.ndarray]:
    return np.stack([m[0] for m in members]), np.array([m[1] for m in members], dtype=np.int64)


def _buckets(items: Iterable[Item], temp_of: Callable[[Item], float], views: Optional[Sequence[int]]):
    """Labelled items grouped by (views, options, temperature) so each group is one vectorised softmax."""
    out: Dict[Tuple[Tuple[int, int], float], List[Tuple[np.ndarray, int]]] = {}
    for it in items:
        if it.gold is None:
            continue
        z = _views(it, views)
        if not 0 <= int(it.gold) < z.shape[1]:
            raise ValueError(f"gold {it.gold} is outside the {z.shape[1]} options of {it.run_id}/{it.question_id}")
        out.setdefault((z.shape, float(temp_of(it))), []).append((z, int(it.gold)))
    return out


def _stats(z: np.ndarray, gold: np.ndarray, t: float):
    """(nll, brier, confidence, correct) arrays for one stacked bucket (N x V x K logits) at temperature t."""
    p = _softmax(z, t).mean(axis=1)                                # N x K
    rows = np.arange(len(gold))
    onehot = np.zeros_like(p)
    onehot[rows, gold] = 1.0
    return (-np.log(np.maximum(p[rows, gold], EPS)), ((p - onehot) ** 2).sum(axis=1), p.max(axis=1),
            (p.argmax(axis=1) == gold).astype(np.float64))


def _per_item(items: Iterable[Item], temp_of: Callable[[Item], float], views: Optional[Sequence[int]] = None):
    """Arrays (nll, brier, confidence, correct) over the labelled items."""
    parts = [_stats(*_stack_bucket(members), t) for (_shape, t), members in _buckets(items, temp_of, views).items()]
    if not parts:
        empty = np.zeros(0)
        return empty, empty, empty, empty
    return tuple(np.concatenate([part[i] for part in parts]) for i in range(4))


def _as_temp_fn(t: TemperatureLike) -> Callable[[Item], float]:
    return t if callable(t) else (lambda _item, _t=float(t): _t)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Top-label expected calibration error with equal-width confidence bins (confidence 1.0 is in the last bin)."""
    n = len(conf)
    if n == 0:
        return float("nan")
    idx = np.minimum((conf * bins).astype(np.int64), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() / n * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(total)


def metrics(items: Iterable[Item], t: TemperatureLike, views: Optional[Sequence[int]] = None, bins: int = 10) -> Dict[str, Any]:
    """n, nll, brier (summed over options), accuracy and ece of the labelled items at temperature `t`.

    `t` is a number or a function item -> number (a different temperature per group).  Unlabelled items are ignored.
    """
    nll, brier, conf, correct = _per_item(items, _as_temp_fn(t), views)
    if len(nll) == 0:
        return {"n": 0, "nll": None, "brier": None, "accuracy": None, "ece": None}
    return {"n": int(len(nll)), "nll": float(nll.mean()), "brier": float(brier.mean()),
            "accuracy": float(correct.mean()), "ece": ece(conf, correct, bins)}


def nll(items: Iterable[Item], t: TemperatureLike, views: Optional[Sequence[int]] = None) -> float:
    values = _per_item(items, _as_temp_fn(t), views)[0]
    if len(values) == 0:
        raise ValueError("no labelled items")
    return float(values.mean())


# --------------------------------------------------------------------------- fitting

def fit_temperature_items(items: Iterable[Item], grid: Tuple[float, float] = DEFAULT_GRID,
                          views: Optional[Sequence[int]] = None) -> float:
    """The temperature in `grid` that minimises the mean NLL of the labelled items.

    A log-spaced scan finds the basin (the TTA mixture is not always convex in T), then a golden-section search
    in log T refines inside the two neighbouring scan points.  The result never leaves the grid.
    """
    lo, hi = float(grid[0]), float(grid[1])
    if not 0.0 < lo < hi:
        raise ValueError(f"grid must satisfy 0 < low < high, got {grid!r}")
    labelled = [it for it in items if it.gold is not None]
    if not labelled:
        raise ValueError("no labelled items to fit a temperature on")
    stacks = [_stack_bucket(members) for members in _buckets(labelled, lambda _it: 1.0, views).values()]
    total = sum(len(g) for _z, g in stacks)

    def loss(t: float) -> float:
        return float(sum(_stats(z, g, t)[0].sum() for z, g in stacks) / total)

    scan = np.geomspace(lo, hi, COARSE_POINTS)
    values = [loss(float(t)) for t in scan]
    k = int(np.argmin(values))
    a, b = math.log(scan[max(k - 1, 0)]), math.log(scan[min(k + 1, len(scan) - 1)])
    c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    fc, fd = loss(math.exp(c)), loss(math.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = loss(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = loss(math.exp(d))
        if b - a < 1e-7:
            break
    best = math.exp((a + b) / 2.0)
    return float(min(max(best, lo), hi))


def k_bucket(k: int) -> str:
    return "1" if k < 2 else "2" if k == 2 else "3-4" if k <= 4 else "5-8" if k <= 8 else "9+"


def group_of(item: Item, by: str) -> str:
    if by == "question_type":
        return item.qtype
    if by == "type_k":
        return f"{item.qtype}:{k_bucket(len(item.logits[0]))}"
    raise ValueError(f"by must be one of {GROUPINGS}, got {by!r}")


def split_holdout(items: Sequence[Item], holdout: float, seed: int = 0) -> Tuple[List[Item], List[Item]]:
    """(fit, holdout) split by request: all questions of one request land on the same side.

    Requests are ranked by sha256(seed:run_id) and the first round(holdout * requests) of them form the holdout,
    so the split is deterministic, independent of the order of `items`, and has the asked-for size.
    """
    if not 0.0 <= holdout < 1.0:
        raise ValueError("holdout must be in [0, 1)")
    ids = sorted({it.run_id for it in items})
    ranked = sorted(ids, key=lambda rid: hashlib.sha256(f"{seed}:{rid}".encode("utf-8")).hexdigest())
    held = set(ranked[: int(round(holdout * len(ranked)))])
    fit = [it for it in items if it.run_id not in held]
    hold = [it for it in items if it.run_id in held]
    return fit, hold


def fit_groups(items: Sequence[Item], by: str = "question_type", holdout: float = 0.5, seed: int = 0,
               baseline: float = 1.0, grid: Tuple[float, float] = DEFAULT_GRID, min_items: int = 30,
               views: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Fit one temperature per group on the fit side of a by-request split and report both sides.

    Groups with fewer than `min_items` labelled fit items use the temperature fitted on all of them (listed in
    `fallback`).  `holdout.before` is the holdout at `baseline` (the temperature the model ships with),
    `holdout.after` is the holdout at the fitted per-group temperatures.
    """
    items = list(items)
    fit, hold = split_holdout(items, holdout, seed)
    fit_labelled = [it for it in fit if it.gold is not None]
    if not fit_labelled:
        raise ValueError("no labelled items to fit on")
    default = fit_temperature_items(fit_labelled, grid, views)
    groups = sorted({group_of(it, by) for it in items if it.gold is not None})
    counts = {g: sum(1 for it in fit_labelled if group_of(it, by) == g) for g in groups}
    temps: Dict[str, float] = {}
    fallback: List[str] = []
    for g in groups:
        if counts[g] >= min_items:
            temps[g] = fit_temperature_items([it for it in fit_labelled if group_of(it, by) == g], grid, views)
        else:
            temps[g] = default
            fallback.append(g)

    def per_group(it: Item) -> float:
        return temps.get(group_of(it, by), default)

    out: Dict[str, Any] = {
        "by": by, "baseline": float(baseline), "default": default, "temperatures": temps, "fallback": fallback,
        "fit_counts": counts, "grid": [float(grid[0]), float(grid[1])], "seed": seed, "holdout_fraction": holdout,
        "n_fit": len(fit_labelled), "n_holdout": sum(1 for it in hold if it.gold is not None),
        "fit": {"before": metrics(fit_labelled, baseline, views), "after": metrics(fit_labelled, per_group, views)},
        "holdout": {"before": metrics(hold, baseline, views), "after": metrics(hold, per_group, views)},
    }
    return out
