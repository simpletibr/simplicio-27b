"""The gates the rented day answers to: label coverage (F3), the GT evidence for the thinker (F2b) and G4 (F6).

* **Label coverage.**  At least ``label_coverage`` of the corpus has a teacher label, or the day does not start paying for GPUs.
* **GT evidence.**  The thinker-trace step runs only with a ``report.json`` of ``decide.think.compare`` (``think-gt-report/1``,
  Squad C, PR #78) that says PASS, is complete, and was made with the same model (the config's, or its FP8 build) with thinking on.
  Fields read, all from that file: ``format``, ``verdict.status``, ``verdict.delta_pts``, ``verdict.ci``, ``coverage.complete``,
  ``thinker.config.{model,revision,chat_template_kwargs.enable_thinking}``.  UNVERIFIED: written from the code of that pull
  request, not from a report the GPU produced.
* **G4.**  The new checkpoint must score at least ``margin_points`` more than the baseline on the home private proxy, with the
  scorer of ``decide.data.proxy_private`` (100 x the mean chance-corrected skill per source), before the public round.  There is
  no parameter that skips it.  The verdict also carries a paired bootstrap 95% interval of the difference; an interval that
  includes zero is reported (``ci_includes_zero``) but is not part of the plan's gate, so it does not change PASS to FAIL.
  The proxy is the lab-free stand-in of the board (G1 measures how well it ranks models), not the board.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from decide.data import proxy_private as PP
from decide.data import schema as S

GT_REPORT_FORMAT = "think-gt-report/1"
DEFAULT_SEED = 20261021
SCOPE = ("home private proxy (decide.data.proxy_private): a stand-in for the board, not the board; "
         "the public round and the lab's agreement (G5) are what the board sees")


# --------------------------------------------------------------------------- label coverage (F3)

def label_coverage(report: Mapping[str, Any], *, minimum: float) -> Dict[str, Any]:
    """Share of the corpus rows that kept a teacher label: ``report`` has ``rows_in`` (corpus rows) and ``rows_out`` (rows with a teacher label;
    see :func:`teacher_coverage`, ``build_targets`` alone also keeps the rows that only have gold)."""
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not 0 < minimum <= 1:
        raise ValueError(f"minimum must be a share in (0, 1], got {minimum!r}")
    try:
        rows_in, rows_out = report["rows_in"], report["rows_out"]
    except (KeyError, TypeError) as exc:
        raise ValueError("the targets report needs rows_in and rows_out") from exc
    if isinstance(rows_in, bool) or isinstance(rows_out, bool) or not isinstance(rows_in, int) or not isinstance(rows_out, int) or rows_in < 0 or not 0 <= rows_out <= max(rows_in, 0):
        raise ValueError(f"rows_in and rows_out must be whole numbers with 0 <= rows_out <= rows_in, got {rows_in!r} and {rows_out!r}")
    if rows_in == 0:
        return {"coverage": 0.0, "minimum": minimum, "passed": False, "rows_in": 0, "rows_out": 0, "reason": "no rows in the corpus, so nothing was labeled"}
    coverage = rows_out / rows_in
    passed = round(coverage, 12) >= minimum
    return {"coverage": coverage, "minimum": minimum, "passed": passed, "rows_in": rows_in, "rows_out": rows_out,
            "reason": f"{rows_out} of {rows_in} rows have a label ({coverage:.4f} {'>=' if passed else '<'} {minimum})"}


def teacher_coverage(targets_path: "str | os.PathLike[str]", report: Mapping[str, Any]) -> Dict[str, Any]:
    """``rows_in`` / ``rows_out`` for :func:`label_coverage`: the corpus rows, and those that kept a label from at least one teacher.

    ``build_targets`` keeps a row that has only a gold label (``rows_out`` of its report), so its own count cannot tell whether the
    teachers labeled the corpus.  The ``teachers`` column of the targets lists the teachers each row used."""
    _, pq = S._arrow()
    labeled = 0
    for file in S.corpus_files(targets_path, require_manifest=True):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=S.BATCH_ROWS, columns=["teachers"]):
            labeled += sum(1 for value in batch.column(0).to_pylist() if json.loads(value))
    return {"rows_in": report["rows_in"], "rows_out": labeled, "dropped_unlabeled": report.get("dropped_unlabeled", 0)}


# --------------------------------------------------------------------------- the GT evidence (F2b)

def gt_evidence(report_path: "str | os.PathLike[str] | None", *, thinker: Mapping[str, Any]) -> Dict[str, Any]:
    """Whether the thinker-trace step may run, and with which model and revision.  Never raises on a bad report: it says no, and why."""
    mode = thinker.get("enabled")
    if mode not in ("auto", "off") or isinstance(mode, bool):
        raise ValueError(f"thinker.enabled must be 'auto' or 'off', got {mode!r}")
    out: Dict[str, Any] = {"enabled": False, "reasons": [], "model": None, "revision": None, "delta_pts": None, "ci": None, "report_sha256": None}
    if mode == "off":
        out["reasons"].append("thinker.enabled is off")
        return out
    if report_path is None:
        out["reasons"].append("no GT report (pass --gt-report with the report.json of decide.think.compare)")
        return out
    path = Path(report_path)
    try:
        raw = path.read_bytes()
        report = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        out["reasons"].append(f"GT report unreadable: {exc}")
        return out
    out["report_sha256"] = hashlib.sha256(raw).hexdigest()
    if not isinstance(report, dict):
        out["reasons"].append("GT report unreadable: not a JSON object")
        return out
    reasons: List[str] = out["reasons"]
    if report.get("format") != GT_REPORT_FORMAT:
        reasons.append(f"format is {report.get('format')!r}, this tool reads {GT_REPORT_FORMAT!r}")
    verdict = report.get("verdict") if isinstance(report.get("verdict"), dict) else {}
    if verdict.get("status") != "PASS":
        reasons.append(f"verdict is {verdict.get('status')!r}, not 'PASS'")
    out["delta_pts"], out["ci"] = verdict.get("delta_pts"), verdict.get("ci")
    coverage = report.get("coverage") if isinstance(report.get("coverage"), dict) else {}
    if coverage.get("complete") is not True:
        reasons.append("coverage is not complete: some rows of the sample were not scored")
    config = ((report.get("thinker") or {}).get("config") or {}) if isinstance(report.get("thinker"), dict) else {}
    model, revision = config.get("model"), config.get("revision")
    wanted = thinker.get("model")
    if model not in (wanted, f"{wanted}-FP8"):
        reasons.append(f"model is {model!r}, the config asks for {wanted!r} (or its -FP8 build)")
    if not isinstance(revision, str) or not revision:
        reasons.append("revision is missing in the GT report, so the traces cannot be pinned to the model the gate measured")
    elif thinker.get("revision") and revision != thinker["revision"]:
        reasons.append(f"revision {revision} differs from the config's {thinker['revision']}")
    if (config.get("chat_template_kwargs") or {}).get("enable_thinking") is False:
        reasons.append("thinking was off in the GT run: that is not the gate")
    out["model"], out["revision"] = model, revision
    out["enabled"] = not reasons
    return out


# --------------------------------------------------------------------------- G4

Request = Tuple[str, Mapping[str, Any], Mapping[str, Any], str]


def proxy_requests(proxy_path: "str | os.PathLike[str]") -> List[Request]:
    """``(example_id, request, gold, source)`` of every row of the proxy, as ``proxy_private.score_rows`` takes them."""
    out: List[Request] = []
    for row in S.read_corpus(proxy_path):
        request, gold = S.parse_row(row)
        request = {key: value for key, value in request.items() if key != "reasoning"}      # G4 measures the decider alone
        out.append((row["example_id"], request, gold, row["source"]))
    return out


def proxy_to_requests(proxy_path: "str | os.PathLike[str]", out_path: "str | os.PathLike[str]") -> int:
    """Write the proxy as the JSONL the kit's ``run_requests`` reads (``example_id``, ``request``, ``gold``); returns the row count."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + ".tmp")
    count = 0
    with open(pending, "w", encoding="utf-8") as handle:
        for example_id, request, gold, source in proxy_requests(proxy_path):
            handle.write(json.dumps({"example_id": example_id, "request": request, "gold": gold, "source": source}, ensure_ascii=False) + "\n")
            count += 1
    os.replace(pending, target)
    return count


def _units(requests: Sequence[Request], results: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[Tuple[str, str], Tuple[int, float]]]:
    """source -> {(example_id, question_id): (right, chance)} with the rules of ``proxy_private.score_rows``."""
    table: Dict[Tuple[str, str], List[float]] = {}
    for row in results:
        rid, qid = PP._row_id(row), row.get("question_id")
        probs = row.get("probs", row.get("probabilities"))
        if rid is not None and qid is not None and probs is not None:
            table[(rid, str(qid))] = [float(p) for p in probs]
    units: Dict[str, Dict[Tuple[str, str], Tuple[int, float]]] = {}
    for eid, request, gold, source in requests:
        for qid, key in gold.items():
            if key is None:
                continue
            keys = S.option_keys(request["questions"][qid])
            probs = table.get((eid, qid))
            if probs is None or len(probs) != len(keys) or sum(probs) <= 0 or any(p < 0 or not math.isfinite(p) for p in probs):
                continue
            right = 1 if max(range(len(probs)), key=probs.__getitem__) == keys.index(key) else 0
            units.setdefault(source, {})[(eid, qid)] = (right, 1.0 / len(keys))
    return units


def _macro_skill(sources: Mapping[str, Tuple[float, float, int]]) -> float:
    skills = []
    for right, chance, n in (sources[name] for name in sorted(sources)):
        accuracy, mean_chance = right / n, chance / n
        skills.append((accuracy - mean_chance) / (1 - mean_chance) if mean_chance < 1 else 0.0)
    return sum(skills) / len(skills)


def _score_view(scored: Mapping[str, Any], scale: float) -> Dict[str, Any]:
    return {"proxy_points": None if scored["proxy"] is None else scale * scored["proxy"], "n": scored["n"], "missing": scored["missing"], "bad": scored["bad"],
            "per_source": scored["per_source"], "nll": scored["nll"], "brier": scored["brier"]}


def _percentile(sorted_values: Sequence[float], share: float) -> float:
    position = share * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def g4_verdict(requests: Iterable[Request], new_results: Iterable[Mapping[str, Any]], baseline_results: Iterable[Mapping[str, Any]], *,
               margin_points: float, scale: float = 100, resamples: int = 1000, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """PASS when the new checkpoint beats the baseline by ``margin_points`` or more on the proxy; INVALID when either side is incomplete."""
    if isinstance(margin_points, bool) or not isinstance(margin_points, (int, float)) or margin_points < 0:
        raise ValueError(f"margin_points must be a number >= 0, got {margin_points!r}")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError(f"resamples must be a positive whole number, got {resamples!r}")
    requests = list(requests)
    new_results, baseline_results = list(new_results), list(baseline_results)
    new, base = PP.score_rows(requests, new_results), PP.score_rows(requests, baseline_results)
    verdict: Dict[str, Any] = {"margin_points": margin_points, "scale": scale, "resamples": resamples, "seed": seed, "scope": SCOPE, "status": "INVALID",
                               "reasons": [], "notes": [], "new": _score_view(new, scale), "baseline": _score_view(base, scale), "delta_points": None,
                               "ci95": None, "ci_above_zero": None, "questions": new["n"]}
    reasons: List[str] = verdict["reasons"]
    for label, scored in (("new", new), ("baseline", base)):
        if scored["missing"]:
            reasons.append(f"{label}_missing_rows")
        if scored["bad"]:
            reasons.append(f"{label}_bad_rows")
    if new["proxy"] is None or base["proxy"] is None or not new["n"] or not base["n"]:
        reasons.append("no_scored_questions")
    if reasons:
        return verdict
    delta = scale * (new["proxy"] - base["proxy"])
    verdict["delta_points"] = delta

    new_units, base_units = _units(requests, new_results), _units(requests, baseline_results)
    sources = sorted(new_units)
    keys = {name: sorted(new_units[name]) for name in sources}
    if sources != sorted(base_units) or any(keys[name] != sorted(base_units[name]) for name in sources):
        verdict["reasons"].append("different_questions")
        return verdict
    arrays = {name: ([new_units[name][k][0] for k in keys[name]], [base_units[name][k][0] for k in keys[name]], [new_units[name][k][1] for k in keys[name]]) for name in sources}
    whole_new = {name: (sum(arrays[name][0]), sum(arrays[name][2]), len(keys[name])) for name in sources}
    if abs(scale * _macro_skill(whole_new) - verdict["new"]["proxy_points"]) > 1e-9:
        raise AssertionError("the bootstrap units disagree with proxy_private.score_rows: the scorer changed")
    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(resamples):
        new_side: Dict[str, Tuple[float, float, int]] = {}
        base_side: Dict[str, Tuple[float, float, int]] = {}
        for name in sources:
            right_new, right_base, chance = arrays[name]
            n = len(chance)
            picks = [rng.randrange(n) for _ in range(n)]
            chance_sum = sum(chance[i] for i in picks)
            new_side[name] = (sum(right_new[i] for i in picks), chance_sum, n)
            base_side[name] = (sum(right_base[i] for i in picks), chance_sum, n)
        deltas.append(scale * (_macro_skill(new_side) - _macro_skill(base_side)))
    deltas.sort()
    low, high = _percentile(deltas, 0.025), _percentile(deltas, 0.975)
    verdict["ci95"] = [low, high]
    verdict["ci_above_zero"] = low > 0
    if not verdict["ci_above_zero"]:
        verdict["notes"].append("ci_includes_zero")
    if round(delta, 9) >= margin_points:
        verdict["status"] = "PASS"
    else:
        verdict["status"] = "FAIL"
        verdict["reasons"].append("delta_below_margin")
    return verdict
