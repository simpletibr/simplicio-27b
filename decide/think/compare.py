"""Compare the thinker with the decision models on the same rows and give the GT verdict (issue #75).

The question of the gate: does Qwen3.8-27B with thinking beat the best one-pass Jev decision model by
at least +10 accuracy points, with the lower end of the paired-bootstrap 95% interval above 0?

Inputs: the private pack written by `sample.py` (per row: the gold and every decision model's
per-option probabilities) and the results file written by `thinker.py`.  Both sides are scored on the
SAME rows, the way the kit scores them:

* a decision model answers with the argmax of its published probabilities (first maximum in option
  order, the rule the Fase 0A tables use);
* a row is a hit when every question of the row is right (SATA-Bench is case-exact in the kit) and,
  for ChessBench, when the move is in the accepted set of ties;
* an unparseable thinker reply, or a terminal failure, is a miss and is counted; a row whose last
  record is a transient `error` is missing, not wrong, and is listed;
* chance-corrected skill = clip((accuracy - chance) / (1 - chance), 0, 1) with the kit's chance of
  the benchmark.

The verdict uses the macro average over the benchmarks (each benchmark weighs the same, as in the
index) of the thinker minus the reference, the decision model with the best macro accuracy on these
rows.  The interval is a percentile bootstrap over rows, resampled inside each benchmark, with the
thinker and the reference resampled on the same rows (paired).  Picking the best of 8 models on the
same rows favors the reference, which is the cautious side for a gate; `ci_reselected` repeats the
interval picking the best model again inside every resample, for the reader who wants it stricter.

CLI:  python -m decide.think.compare --pack pack.jsonl.gz --thinker results.jsonl --out DIR
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import common as C

THINKER = "thinker"
TERMINAL_STATUSES = C.TERMINAL_STATUSES      # anything else (`error`) is transient: the row is retried on resume
REPORT_FORMAT = "think-gt-report/1"


class PackError(ValueError):
    """The pack or the results do not hold what the comparison needs."""


# ---------------------------------------------------------------------------- picks and hits

def argmax_key(probs: Dict[str, float], order: Sequence[str]) -> str:
    """The option with the highest probability; the first one in `order` wins a tie."""
    best, best_value = None, -math.inf
    for k in order:
        v = probs[k]
        if v > best_value:
            best, best_value = k, v
    return best  # type: ignore[return-value]


def decision_picks(row: dict, model: str) -> Dict[str, str]:
    """qid -> the key a decision model picks on a pack row: the argmax of its published probabilities."""
    dec = row["decisions"][model]
    picks = {}
    for qid, q in row["questions"].items():
        probs = (dec.get(qid) or {}).get("probs")
        order = [str(k) for k in q["criteria"]]
        if not isinstance(probs, dict) or any(k not in probs for k in order):
            raise PackError(f"{row['id']}: {model} has no probability for every option of {qid}")
        picks[qid] = argmax_key(probs, order)
    return picks


def thinker_picks(rec: dict, row: dict) -> Dict[str, Optional[str]]:
    answers = rec.get("answers") or {}
    return {qid: answers.get(qid) for qid in row["questions"]}


def row_hit(gold: Dict[str, List[str]], picks: Dict[str, Optional[str]]) -> bool:
    """True when every question is answered with one of its accepted keys."""
    return bool(gold) and all(picks.get(qid) is not None and picks[qid] in accepted for qid, accepted in gold.items())


def row_chance(row: dict) -> float:
    """Probability that a uniform random guess hits the row: the product over its questions."""
    p = 1.0
    for qid, q in row["questions"].items():
        p *= len(row["gold"][qid]) / len(q["criteria"])
    return p


def row_confidence(row: dict, model: str) -> float:
    """How sure a decision model is of a row: the highest probability of its least sure question."""
    return min(max(row["decisions"][model][qid]["probs"].values()) for qid in row["questions"])


# ----------------------------------------------------------------------------------- loading

def load_pack(path: Any) -> List[dict]:
    rows = list(C.iter_jsonl(path, complete_lines_only=False))
    if not rows:
        raise PackError(f"{path} holds no rows")
    return rows


def load_thinker(path: Any) -> Tuple[Optional[dict], List[dict]]:
    """(header, row records in file order).  A last line cut by a crash is ignored."""
    header, records = None, []
    for rec in C.iter_jsonl(path):
        kind = rec.get("kind", "row" if "id" in rec else None)
        if kind == "header":
            header = header or rec
        elif kind == "row":
            records.append(rec)
    return header, records


# ------------------------------------------------------------------------------------ scoring

@dataclass
class Scored:
    systems: List[str]                                   # [THINKER, *models]
    models: List[str]
    benchmarks: List[int]                                # ascending, only those with at least one scored row
    ids: Dict[int, List[str]]
    hits: Dict[str, Dict[int, List[int]]]                # system -> benchmark -> 0/1 per row, same order as `ids`
    chance: Dict[int, float]                             # mean chance of the sampled rows, per benchmark
    records: Dict[int, List[dict]]                       # thinker records aligned with `ids`
    conf: Dict[str, Dict[int, List[float]]] = field(default_factory=dict)      # model -> benchmark -> row_confidence, aligned with `ids`
    pack_rows: int = 0
    missing: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    foreign: List[str] = field(default_factory=list)
    choice_mismatch: Dict[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.missing and not self.errors

    def n(self, bench: int) -> int:
        return len(self.ids[bench])

    def rows_scored(self) -> int:
        return sum(len(v) for v in self.ids.values())

    def correct(self, system: str, bench: int) -> int:
        return sum(self.hits[system][bench])

    def acc(self, system: str, bench: int) -> float:
        return self.correct(system, bench) / self.n(bench)

    def macro_acc(self, system: str) -> float:
        return statistics.fmean(self.acc(system, b) for b in self.benchmarks)

    def pooled_acc(self, system: str) -> float:
        return sum(self.correct(system, b) for b in self.benchmarks) / self.rows_scored()

    def chance_of(self, bench: int) -> float:
        """The kit's chance of the benchmark (the sample's own mean chance for an id the kit does not list)."""
        return C.chance_of(bench) if bench in C.BENCHMARKS else self.chance[bench]

    def skill(self, system: str, bench: int) -> float:
        return C.skill(self.acc(system, bench), self.chance_of(bench))

    def macro_skill(self, system: str) -> float:
        return statistics.fmean(self.skill(system, b) for b in self.benchmarks)


def score_pack(pack_rows: Iterable[dict], records: Iterable[dict], *, models: Optional[Sequence[str]] = None,
               benchmarks: Optional[Iterable[int]] = None) -> Scored:
    """Score the thinker and every decision model on the rows the thinker has finished."""
    rows = list(pack_rows)
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise PackError("the pack has duplicate ids")
    latest: Dict[str, dict] = {}
    foreign: List[str] = []
    for rec in records:
        rid = rec.get("id")
        if rid in by_id:
            latest[rid] = rec                                # the last record of an id wins (a retry replaces an error)
        elif rid is not None and rid not in foreign:
            foreign.append(rid)
    present = set.intersection(*(set(r.get("decisions", {})) for r in rows)) if rows else set()
    if models is None:
        model_list = sorted(present)
    else:
        absent = [m for m in models if m not in present]
        if absent:
            raise ValueError(f"decision models not in every pack row: {absent}; the pack has {sorted(present)}")
        model_list = sorted(set(models))
    keep = None if benchmarks is None else {int(b) for b in benchmarks}
    systems = [THINKER, *model_list]
    ids: Dict[int, List[str]] = {}
    hits: Dict[str, Dict[int, List[int]]] = {s: {} for s in systems}
    chance_rows: Dict[int, List[float]] = {}
    conf: Dict[str, Dict[int, List[float]]] = {m: {} for m in model_list}
    recs: Dict[int, List[dict]] = {}
    missing, errors = [], []
    mismatch = {m: 0 for m in model_list}
    in_scope = 0
    for r in rows:
        b = int(r["benchmark"])
        if keep is not None and b not in keep:
            continue
        in_scope += 1
        rec = latest.get(r["id"])
        if rec is None:
            missing.append(r["id"])
            continue
        status = rec.get("status", "ok")
        if status not in TERMINAL_STATUSES:
            errors.append(r["id"])
            continue
        ids.setdefault(b, []).append(r["id"])
        recs.setdefault(b, []).append(rec)
        chance_rows.setdefault(b, []).append(row_chance(r))
        hits[THINKER].setdefault(b, []).append(int(row_hit(r["gold"], thinker_picks(rec, r) if status == "ok" else {})))
        for m in model_list:
            picks = decision_picks(r, m)
            hits[m].setdefault(b, []).append(int(row_hit(r["gold"], picks)))
            conf[m].setdefault(b, []).append(row_confidence(r, m))
            for qid, dec in r["decisions"][m].items():
                if dec.get("choice") is not None and str(dec["choice"]) != picks.get(qid):
                    mismatch[m] += 1
                    break
    benches = sorted(ids)
    return Scored(systems=systems, models=model_list, benchmarks=benches, ids={b: ids[b] for b in benches},
                  hits={s: {b: hits[s][b] for b in benches} for s in systems},
                  chance={b: statistics.fmean(chance_rows[b]) for b in benches}, records={b: recs[b] for b in benches},
                  conf={m: {b: conf[m][b] for b in benches} for m in model_list}, pack_rows=in_scope, missing=missing, errors=errors, foreign=foreign, choice_mismatch=mismatch)


def pick_reference(s: Scored) -> Optional[str]:
    """The decision model with the best macro accuracy on these rows; a tie goes to the first name."""
    if not s.models or not s.benchmarks:
        return None
    return min(s.models, key=lambda m: (-round(s.macro_acc(m), 12), m))


GATE_THRESHOLDS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01)


def gate_curve(s: Scored, reference: str, thresholds: Sequence[float] = GATE_THRESHOLDS) -> List[Dict[str, float]]:
    """Offline "think on demand": send a row to the thinker when the reference model's confidence is below tau.

    tau 0 is the decision model alone, tau above 1 the thinker alone.  Exploratory: tau is read on the same
    rows it is scored on, so the middle of the curve is optimistic.  It says how much a confidence gate
    could add at best with these two systems, not what a deployed gate will do.
    """
    out = []
    ref_macro = s.macro_acc(reference)
    for tau in thresholds:
        routed = total = 0
        accs = []
        for b in s.benchmarks:
            conf = s.conf[reference][b]
            hybrid = [t if c < tau else m for t, m, c in zip(s.hits[THINKER][b], s.hits[reference][b], conf)]
            routed += sum(c < tau for c in conf)
            total += len(conf)
            accs.append(sum(hybrid) / len(hybrid))
        macro = statistics.fmean(accs)
        out.append({"tau": tau, "routed": routed / total, "macro_acc": macro, "delta_vs_reference_pts": 100.0 * (macro - ref_macro)})
    return out


# ---------------------------------------------------------------------------------- bootstrap

def percentile_ci(values: Sequence[float], level: float = C.CI_LEVEL) -> Tuple[float, float]:
    """Percentile interval: the values at rank floor(a*B) and ceil((1-a)*B)-1 of the sorted replicates, a=(1-level)/2."""
    xs = sorted(values)
    if not xs:
        raise ValueError("no replicates")
    b = len(xs)
    a = (1.0 - level) / 2.0
    lo = min(b - 1, max(0, math.floor(round(a * b, 9))))
    hi = min(b - 1, max(0, math.ceil(round((1.0 - a) * b, 9)) - 1))
    return xs[lo], xs[hi]


@dataclass
class Replicates:
    B: int
    seed: int
    systems: List[str]
    benchmarks: List[int]
    n: Dict[int, int]
    counts: Dict[str, Dict[int, "array[int]"]]           # system -> benchmark -> rows right in each replicate

    def bench(self, system: str, bench: int) -> List[float]:
        n = self.n[bench]
        return [c / n for c in self.counts[system][bench]]

    def macro(self, system: str) -> List[float]:
        cols = [(self.counts[system][b], self.n[b]) for b in self.benchmarks]
        k = len(cols)
        return [sum(col[r] / n for col, n in cols) / k for r in range(self.B)]

    def delta_pts(self, a: str, b: str, bench: Optional[int] = None) -> List[float]:
        """Replicate by replicate, a minus b in accuracy points (macro, or one benchmark)."""
        xs, ys = (self.macro(a), self.macro(b)) if bench is None else (self.bench(a, bench), self.bench(b, bench))
        return [100.0 * (x - y) for x, y in zip(xs, ys)]


def bootstrap(hits: Dict[str, Dict[int, List[int]]], *, B: int = C.BOOTSTRAP_B, seed: int = C.SEED) -> Replicates:
    """Paired, stratified bootstrap over rows.

    Draw order (the contract): replicate by replicate; inside a replicate, benchmarks in ascending id;
    inside a benchmark, n draws of int(random() * n) from random.Random(seed).  Every system reads the
    same drawn rows, so the comparison is paired.  The 0/1 outcomes of all systems are packed into one
    integer per row, so one C-speed sum counts every system's hits in the resample.
    """
    systems = list(hits)
    if not systems:
        raise ValueError("no systems")
    benches = sorted(hits[systems[0]])
    if not benches:
        raise ValueError("no rows")
    n = {b: len(hits[systems[0]][b]) for b in benches}
    for s in systems:
        if {b: len(v) for b, v in hits[s].items()} != n:
            raise ValueError(f"system {s!r} was not scored on the same rows")
    width = max(n.values()).bit_length() + 1              # headroom: a field never carries into the next one
    mask = (1 << width) - 1
    packed = {b: [sum(hits[s][b][i] << (width * k) for k, s in enumerate(systems)) for i in range(n[b])] for b in benches}
    counts = {s: {b: array("I", [0]) * B for b in benches} for s in systems}
    draw = random.Random(seed).random
    for r in range(B):
        for b in benches:
            m = n[b]
            table = packed[b]
            total = sum(map(table.__getitem__, [int(draw() * m) for _ in range(m)]))
            for k, s in enumerate(systems):
                counts[s][b][r] = (total >> (width * k)) & mask
    return Replicates(B=B, seed=seed, systems=systems, benchmarks=benches, n=n, counts=counts)


# ------------------------------------------------------------------------------------ verdict

def decide(delta_pts: float, ci_lo: float, ci_hi: float, *, min_delta: float = C.GT_MIN_DELTA_PTS) -> Dict[str, Any]:
    """PASS when delta >= min_delta and the CI floor is above 0.  Rounded to 9 places: float noise decides nothing."""
    reasons = []
    if round(delta_pts, 9) < min_delta:
        reasons.append("delta_below_min")
    if not round(ci_lo, 9) > 0:
        reasons.append("ci_low_not_above_zero")
    status = "FAIL" if reasons else "PASS"
    return {"status": status, "reasons": reasons, "can_still_pass": status == "FAIL" and round(ci_hi, 9) >= min_delta}


def _pts(x: float) -> str:
    return f"{x:+.1f}"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = sorted(values)
    if not xs:
        return {"total": 0, "mean": None, "median": None, "p80": None, "max": None}
    return {"total": sum(xs), "mean": sum(xs) / len(xs), "median": statistics.median(xs),
            "p80": xs[min(len(xs) - 1, math.ceil(len(xs) * 0.8) - 1)], "max": xs[-1]}


def looks_garbled(text: Optional[str], *, window: int = 2000, min_chars: int = 200) -> bool:
    """Noise, not language: in the first `window` characters, one character is over 40% of them or letters and digits are under 40%.

    On a T4 the model runs in fp16, which can overflow and make it emit a run of one token.  Such a row is
    not a wrong answer of the model, and a FAIL made of them says nothing about it.  Too short to judge is False.
    """
    if not text:
        return False
    chunk = "".join(text[:window].split())
    if len(chunk) < min_chars:
        return False
    most = max(chunk.count(ch) for ch in set(chunk))
    readable = sum(ch.isalnum() for ch in chunk)
    return most / len(chunk) > 0.4 or readable / len(chunk) < 0.4


def _is_unparseable(rec: dict) -> bool:
    if rec.get("status", "ok") != "ok":
        return False
    answers = rec.get("answers") or {}
    return not answers or any(v is None for v in answers.values())


def thinker_stats(s: Scored, benchmarks: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    recs = [r for b in (benchmarks or s.benchmarks) for r in s.records[b]]
    tokens = [r["tokens"]["completion"] for r in recs if isinstance(r.get("tokens"), dict) and r["tokens"].get("completion") is not None]
    seconds = [r["seconds"] for r in recs if isinstance(r.get("seconds"), (int, float))]
    return {
        "rows": len(recs),
        "failed": sum(r.get("status", "ok") == "failed" for r in recs),
        "unparseable": sum(_is_unparseable(r) for r in recs),
        "truncated": sum(bool(r.get("truncated")) for r in recs),
        "garbled": sum(looks_garbled(r.get("reasoning")) or looks_garbled(r.get("final")) for r in recs),
        "completion_tokens": _summary(tokens),
        "seconds": _summary(seconds),
    }


def latency_slack(stats: Dict[str, Any], rule_ms: float = C.LATENCY_RULE_MS) -> Optional[Dict[str, Any]]:
    """How many requests may think before the mean latency passes the kit's rule.

    A hybrid answers a fraction f of the requests with the thinker (mean time t) and the rest with the
    decider.  Ignoring the decider's own time, mean latency = f * t, so f <= rule / t.  The time comes
    from the backend that ran the thinker, not from the board's RTX PRO 6000 (one request at a time), so
    the number is an upper bound and it is UNVERIFIED for the board.
    """
    seconds = (stats or {}).get("seconds") or {}
    mean = seconds.get("mean")
    if not stats.get("rows") or mean is None:
        return None
    frac = 1.0 if mean <= 0 else min(1.0, rule_ms / (1000.0 * mean))
    return {"mean_seconds": mean, "rule_ms": float(rule_ms), "max_thinking_fraction": frac,
            "unit": "mean latency per request, decider time ignored"}


def _discordant(s: Scored, reference: str) -> Dict[str, int]:
    out = {"thinker_only": 0, "reference_only": 0, "both": 0, "neither": 0}
    for b in s.benchmarks:
        for t, m in zip(s.hits[THINKER][b], s.hits[reference][b]):
            out["both" if t and m else "thinker_only" if t else "reference_only" if m else "neither"] += 1
    return out


def build_report(pack_rows: Iterable[dict], records: Iterable[dict], *, header: Optional[dict] = None,
                 B: int = C.BOOTSTRAP_B, seed: int = C.SEED, min_delta: float = C.GT_MIN_DELTA_PTS,
                 models: Optional[Sequence[str]] = None, pack_sha256: Optional[str] = None) -> Dict[str, Any]:
    s = score_pack(pack_rows, records, models=models)
    ref = pick_reference(s)
    report: Dict[str, Any] = {
        "format": REPORT_FORMAT,
        "config": {"bootstrap_b": B, "seed": seed, "ci_level": C.CI_LEVEL, "min_delta_pts": min_delta,
                   "unit": "macro average over benchmarks; paired bootstrap over rows resampled inside each benchmark"},
        "pack": {"rows": s.pack_rows, "content_sha256": pack_sha256},
        "thinker": {"config": (header or {}).get("config"), "config_fingerprint": (header or {}).get("config_fingerprint")},
        "coverage": {"rows_in_pack": s.pack_rows, "rows_scored": s.rows_scored(), "complete": s.complete, "missing": len(s.missing),
                     "errors": len(s.errors), "foreign_ids": len(s.foreign), "missing_ids": s.missing[:50], "error_ids": s.errors[:50]},
        "reference": ref,
        "models": s.models,
        "choice_mismatch": s.choice_mismatch,
    }
    if not s.benchmarks or ref is None:
        reason = "no_scored_rows" if not s.benchmarks else "no_decision_models"
        report["verdict"] = {"status": "INCONCLUSIVE", "reasons": [reason], "can_still_pass": False, "delta_pts": None, "ci": None,
                             "ci_reselected": None, "reference": ref, "min_delta_pts": min_delta}
        report["benchmarks"], report["systems"], report["deltas"], report["thinker_stats"] = {}, {}, {}, thinker_stats(s)
        report["latency"], report["notice"] = None, C.EVAL_ONLY_NOTICE
        return report
    rep = bootstrap(s.hits, B=B, seed=seed)
    report["benchmarks"] = {}
    for b in s.benchmarks:
        lo, hi = percentile_ci(rep.delta_pts(THINKER, ref, bench=b))
        report["benchmarks"][str(b)] = {
            "name": C.benchmark_name(b), "n": s.n(b), "chance": s.chance_of(b), "chance_sample": s.chance[b],
            "systems": {x: {"correct": s.correct(x, b), "acc": s.acc(x, b), "skill": s.skill(x, b)} for x in s.systems},
            "vs_reference": {"delta_pts": 100 * (s.acc(THINKER, b) - s.acc(ref, b)), "ci": [lo, hi]},
            "thinker": thinker_stats(s, [b]),
        }
    report["systems"] = {x: {"macro_acc": s.macro_acc(x), "pooled_acc": s.pooled_acc(x), "macro_skill": s.macro_skill(x)} for x in s.systems}
    report["deltas"] = {}
    for m in s.models:
        lo, hi = percentile_ci(rep.delta_pts(THINKER, m))
        report["deltas"][m] = {"delta_pts": 100 * (s.macro_acc(THINKER) - s.macro_acc(m)), "ci": [lo, hi]}
    delta = report["deltas"][ref]["delta_pts"]
    lo, hi = report["deltas"][ref]["ci"]
    macros = {m: rep.macro(m) for m in s.models}
    thinker_series = rep.macro(THINKER)
    reselected = [100.0 * (thinker_series[r] - max(macros[m][r] for m in s.models)) for r in range(B)]
    report["verdict"] = {**decide(delta, lo, hi, min_delta=min_delta), "delta_pts": delta, "ci": [lo, hi],
                         "ci_reselected": list(percentile_ci(reselected)), "reference": ref, "min_delta_pts": min_delta,
                         "thinker_macro_acc": s.macro_acc(THINKER), "reference_macro_acc": s.macro_acc(ref)}
    report["discordant"] = _discordant(s, ref)
    report["thinker_stats"] = thinker_stats(s)
    report["latency"] = latency_slack(report["thinker_stats"])
    report["gate_curve"] = gate_curve(s, ref)
    if report["latency"] is not None:
        for g in report["gate_curve"]:
            g["within_latency_rule"] = g["routed"] <= report["latency"]["max_thinking_fraction"] + 1e-12
    report["notice"] = C.EVAL_ONLY_NOTICE
    return report


def verdict_line(report: Dict[str, Any]) -> str:
    v, cov = report["verdict"], report["coverage"]
    scope = f"{cov['rows_scored']} of {cov['rows_in_pack']} rows"
    if v["status"] == "INCONCLUSIVE":
        why = {"no_scored_rows": "no thinker row scored yet", "no_decision_models": "the pack has no decision model"}.get(v["reasons"][0], v["reasons"][0])
        return f"GT VERDICT: INCONCLUSIVE | {why} ({scope})"
    level = int(round(100 * report["config"]["ci_level"]))
    lo, hi = v["ci"]
    head = f"GT VERDICT: {v['status']}" + (f" ({', '.join(v['reasons'])})" if v["reasons"] else "")
    parts = [head,
             f"thinker {_pct(v['thinker_macro_acc'])} vs {v['reference']} {_pct(v['reference_macro_acc'])} (best decision model, macro over {len(report['benchmarks'])} benchmarks)",
             f"delta {_pts(v['delta_pts'])} pts, {level}% CI [{_pts(lo)}, {_pts(hi)}]",
             f"rule: delta >= {_pts(v['min_delta_pts'])} and CI low > 0"]
    if v["status"] == "FAIL" and v["can_still_pass"]:
        parts.append(f"the CI still holds {_pts(v['min_delta_pts'])}: more rows could pass")
    parts.append(f"{scope}" + ("" if cov["complete"] else f", PARTIAL: {cov['missing']} not run, {cov['errors']} with a transient error"))
    return " | ".join(parts)


# ----------------------------------------------------------------------------------- markdown

def _thinker_lines(cfg: Optional[Dict[str, Any]]) -> List[str]:
    """Which thinker made the numbers, and where the run is not the plain gate (quantized, thinking off, other model)."""
    if not cfg:
        return ["Thinker: configuracao desconhecida (os resultados nao trazem cabecalho).", ""]
    model, rev = cfg.get("model"), cfg.get("revision")
    thinking = (cfg.get("chat_template_kwargs") or {}).get("enable_thinking", True) is not False
    quant = cfg.get("quantization")
    name = f"{model}@{rev[:7]}" if rev else f"{model} (sem revisao)"
    line = (f"Thinker: {name}, backend {cfg.get('backend')}, quantizacao {quant or 'nao informada'}, "
            f"max_tokens {cfg.get('max_tokens')}, thinking {'ligado' if thinking else 'desligado'}"
            + (f", esforco {cfg['reasoning_effort']}" if cfg.get("reasoning_effort") else "") + ".")
    out = [line, ""]
    pinned = set(C.PINNED_REVISIONS.items())
    if (model, rev) not in pinned:
        out += [f"AVISO: modelo ou revisao diferente do par fixado ({C.QWEN_MODEL}@{C.QWEN_REVISION[:7]} ou {C.QWEN_FP8_MODEL}@{C.QWEN_FP8_REVISION[:7]}): "
                "o resultado vale para o que rodou, nao para o gate como registrado.", ""]
    if quant not in (None, "none"):
        out += [f"AVISO: o thinker rodou quantizado ({quant}): um PASS vale (quantizar nao costuma melhorar o modelo, UNVERIFIED), "
                "mas um FAIL nao prova que o modelo em bf16 ou FP8 falharia; confirme pela API ou pelo Modal.", ""]
    if not thinking:
        out += ["AVISO: thinking desligado: isto nao e o gate GT, que mede o modelo pensando.", ""]
    return out


def render_markdown(report: Dict[str, Any]) -> str:
    out = ["# Gate GT (F2b): pensar vs decidir em um passe", "", verdict_line(report), ""]
    v, cfg, cov = report["verdict"], report["config"], report["coverage"]
    if v["status"] == "INCONCLUSIVE":
        out.append("Nenhuma linha do thinker foi pontuada ainda.")
        return "\n".join(out) + "\n"
    systems = [THINKER, *report["models"]]
    ref = report["reference"]
    out += _thinker_lines((report.get("thinker") or {}).get("config"))
    out += [f"Referencia: **{ref}**, o melhor modelo de decisao em acuracia macro nestas linhas. Bootstrap pareado e estratificado por benchmark, "
            f"B={cfg['bootstrap_b']}, seed={cfg['seed']}, IC {int(round(100 * cfg['ci_level']))}%. "
            f"IC com o melhor modelo reescolhido a cada reamostragem: [{_pts(v['ci_reselected'][0])}, {_pts(v['ci_reselected'][1])}].", "",
            "## Acuracia por benchmark (%)", "",
            "| benchmark | n | acaso | " + " | ".join(systems) + f" | thinker - {ref} (IC) |",
            "|---|---:|---:|" + "---:|" * len(systems) + "---:|"]
    for b, row in report["benchmarks"].items():
        vs = row["vs_reference"]
        out.append(f"| {row['name']} | {row['n']} | {100 * row['chance']:.1f} | "
                   + " | ".join(f"{100 * row['systems'][x]['acc']:.1f}" for x in systems)
                   + f" | {_pts(vs['delta_pts'])} [{_pts(vs['ci'][0])}, {_pts(vs['ci'][1])}] |")
    sysrows = report["systems"]
    out.append(f"| **macro** | {cov['rows_scored']} | | " + " | ".join(f"**{100 * sysrows[x]['macro_acc']:.1f}**" for x in systems)
               + f" | **{_pts(v['delta_pts'])}** [{_pts(v['ci'][0])}, {_pts(v['ci'][1])}] |")
    out.append("| pooled | " + f"{cov['rows_scored']} | | " + " | ".join(f"{100 * sysrows[x]['pooled_acc']:.1f}" for x in systems) + " | |")
    out += ["", "Cada benchmark tem poucas linhas: o IC por benchmark e largo e nao decide nada sozinho; o veredito usa a media macro.", "",
            "## Habilidade corrigida pelo acaso (skill, como o kit)", "",
            "| benchmark | " + " | ".join(systems) + " |", "|---|" + "---:|" * len(systems)]
    for b, row in report["benchmarks"].items():
        out.append(f"| {row['name']} | " + " | ".join(f"{100 * row['systems'][x]['skill']:.1f}" for x in systems) + " |")
    out.append("| **macro** | " + " | ".join(f"**{100 * sysrows[x]['macro_skill']:.1f}**" for x in systems) + " |")
    out += ["", "## Thinker contra cada modelo de decisao (macro, pontos)", "", "| modelo | acuracia macro | delta | IC |", "|---|---:|---:|---:|"]
    for m in report["models"]:
        d = report["deltas"][m]
        out.append(f"| {m}{' (referencia)' if m == ref else ''} | {100 * sysrows[m]['macro_acc']:.1f} | {_pts(d['delta_pts'])} | [{_pts(d['ci'][0])}, {_pts(d['ci'][1])}] |")
    dc = report["discordant"]
    out += ["", f"Contra {ref}, linha a linha: so o thinker acerta {dc['thinker_only']}, so {ref} acerta {dc['reference_only']}, "
            f"ambos {dc['both']}, nenhum {dc['neither']}. Um roteador perfeito entre os dois acertaria "
            f"{100 * (dc['thinker_only'] + dc['reference_only'] + dc['both']) / max(1, sum(dc.values())):.1f}% das linhas (teto de um gate, nao uma medida).", ""]
    lat = report.get("latency")
    out += [f"## Gate offline (exploratorio): mandar ao thinker so as linhas em que {ref} tem confianca abaixo de tau", "",
            "| tau | linhas enviadas ao thinker | acuracia macro | delta vs " + ref + " |" + (" cabe na regra de latencia |" if lat else ""),
            "|---:|---:|---:|---:|" + ("---:|" if lat else "")]
    for g in report["gate_curve"]:
        cell = (" sim |" if g.get("within_latency_rule") else " nao |") if lat else ""
        out.append(f"| {g['tau']:.2f} | {100 * g['routed']:.0f}% | {100 * g['macro_acc']:.1f} | {_pts(g['delta_vs_reference_pts'])} |{cell}")
    out += ["", "tau 0 = so o modelo de decisao; tau acima de 1 = so o thinker. O tau e lido nas mesmas linhas em que e medido, entao o meio da curva e otimista: "
            "e o teto do que um gate por confianca poderia dar com estes dois sistemas, nao uma previsao.", ""]
    st = report["thinker_stats"]
    ct, sc = st["completion_tokens"], st["seconds"]
    out += ["## Custo e falhas do thinker", "",
            f"- linhas: {st['rows']}; sem resposta legivel (contadas como erro): {st['unparseable']}; orcamento de tokens esgotado: {st['truncated']}; falhas terminais: {st['failed']}",
            f"- tokens gerados: total {ct['total']}, mediana {ct['median']}, p80 {ct['p80']}, maximo {ct['max']}",
            f"- segundos por linha: total {sc['total']}, mediana {sc['median']}, p80 {sc['p80']}, maximo {sc['max']}"]
    if st["rows"] and st["truncated"] / st["rows"] > 0.10:
        out.append("- AVISO: mais de 10% das linhas bateram no orcamento de tokens; um FAIL assim pode vir do orcamento, nao do modelo. Reexecute as truncadas com `--max-tokens` maior.")
    thinking_on = ((report.get("thinker") or {}).get("config") or {}).get("chat_template_kwargs", {}).get("enable_thinking", True) is not False
    if thinking_on and st["rows"] and isinstance(ct.get("median"), (int, float)) and ct["median"] < C.MIN_THINKING_TOKENS:
        out.append(f"- AVISO: a mediana e de {ct['median']:g} tokens por linha: o modelo quase nao pensou (o endpoint ignorou o pedido de thinking?). "
                   "Isto nao e o gate GT; ligue o thinking no endpoint e rode de novo com outro `--out` (README, \"Backends\").")
    if st.get("garbled"):
        out.append(f"- AVISO: {st['garbled']} linhas com saida ilegivel (ruido, nao linguagem). Na T4 a causa provavel e o fp16 estourando (UNVERIFIED); "
                   "essas linhas nao dizem nada sobre o modelo, e um FAIL feito delas nao e um veredito (README, \"Se a saida for ruido\").")
    if not cov["complete"]:
        out.append(f"- AVISO: execucao parcial, {cov['missing']} linhas sem resultado e {cov['errors']} com erro transitorio; todos os sistemas foram comparados so nas linhas feitas.")
    if any(report["choice_mismatch"].values()):
        out.append(f"- AVISO: a escolha publicada difere do argmax das probabilidades em {report['choice_mismatch']} linhas (usamos o argmax).")
    if lat:
        rule = f"{lat['rule_ms']:,.0f}".replace(",", ".")                     # 1.000, the way the docs write it
        out += ["", f"## Folga de latencia (regra do kit: media <= {rule} ms; UNVERIFIED para o board)", "",
                f"- tempo medio por linha do thinker neste backend: {lat['mean_seconds']:.1f} s",
                f"- se o thinker atender mais que {100 * lat['max_thinking_fraction']:.1f}% das requisicoes, a latencia media passa de {rule} ms",
                "- e um limite superior: ignora o tempo do decisor, e o board mede numa RTX PRO 6000, uma requisicao por vez; o tempo daqui vem do backend que rodou o thinker",
                "- a coluna \"cabe na regra de latencia\" do gate acima marca os tau cuja fracao enviada ao thinker fica dentro desse limite"]
    out += ["", "Uso: os pedidos da suite publica desta amostra e os tracos do thinker sobre eles sao dado de avaliacao; nunca entram em treino (ADR 0004, item 6)."]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m decide.think.compare", description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True, help="pack.jsonl.gz written by sample.py (private)")
    ap.add_argument("--thinker", required=True, help="results.jsonl written by thinker.py")
    ap.add_argument("--out", help="directory for report.json and report.md")
    ap.add_argument("--bootstrap", type=int, default=C.BOOTSTRAP_B, help="bootstrap replicates (default %(default)s)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--min-delta", type=float, default=C.GT_MIN_DELTA_PTS, help="points the thinker must win by (default %(default)s)")
    ap.add_argument("--models", help="comma-separated decision models to compare against (default: all in the pack)")
    ap.add_argument("--ignore-pack-mismatch", action="store_true", help="accept results made from another pack (ids must still match)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = load_pack(args.pack)
        header, records = load_thinker(args.thinker)
        from .sample import pack_content_sha256
        pack_sha = pack_content_sha256(args.pack)
        if header and header.get("pack_content_sha256") and header["pack_content_sha256"] != pack_sha and not args.ignore_pack_mismatch:
            raise PackError("the results were made from another pack (content sha256 differs); rebuild the pack with the same manifest or pass --ignore-pack-mismatch")
        report = build_report(rows, records, header=header, B=args.bootstrap, seed=args.seed, min_delta=args.min_delta,
                              models=[m for m in (args.models or "").split(",") if m] or None, pack_sha256=pack_sha)
    except (PackError, ValueError, OSError) as e:
        print(f"[compare] {e}", file=sys.stderr)
        return 2
    md = render_markdown(report)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        C.write_json(out / "report.json", report)
        (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    print(verdict_line(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
