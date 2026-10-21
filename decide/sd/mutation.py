"""Mutation check for the numbers this kit stands on: inject one deliberate one-line bug, expect a test to fail.

A green test suite says the tests pass, not that they would notice a wrong merge.  This module puts a plausible one-line bug into the
merge math, the option-order inverse mapping, the temperature fit, the engine wiring, the loader and the Hub reader, runs the tests that are
meant to guard that code, and reports KILLED (some test failed) or SURVIVED (every test passed: a hole in the tests).

    python -m decide.sd.mutation                      # every mutation
    python -m decide.sd.mutation --only tta           # one group (merge, tta, calibrate, engine, loader, config, api, remote, budget, state)
    python -m decide.sd.mutation --id k-bucket-boundary --id ta-sign-flip      # chosen mutations (to re-check a survivor)
    python -m decide.sd.mutation --list
    python -m decide.sd.mutation --json evidence.json

The bug goes into a COPY of the repository (decide/, tests/ and notebooks/decide/requirements.txt in a temporary folder), never into the
working tree, so a crash or Ctrl-C cannot leave a broken file behind.  "Revert" is therefore: the copy is thrown away, and the sha256 of
every file that was mutated is compared before and after to prove the working tree is byte-identical.

Three things make a verdict mean something:

* the unmutated copy must pass the same tests first, with NO skipped test (an interpreter without torch would "kill" nothing);
* the mutated source must still compile (a syntax error would "kill" everything for the wrong reason);
* the text that is replaced must occur exactly once in its file and be one line, so the table cannot rot when the code moves.

Exit code 0 only if every mutation was killed and the working tree is untouched; 1 if one survived; 2 if the check itself cannot judge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
COPIED = ("decide", "tests", "notebooks/decide/requirements.txt")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
GROUPS = ("merge", "tta", "calibrate", "engine", "loader", "config", "api", "remote", "budget", "state")


@dataclass(frozen=True)
class Mutation:
    group: str
    id: str
    file: str                       # relative to the repository root
    old: str                        # one line, occurs exactly once in `file`
    new: str                        # its replacement: also one line, and different
    tests: Tuple[str, ...]          # unittest ids, run from the tests/ folder
    why: str                        # the mistake this stands for


MERGE_TESTS = ("test_contract_65.MergeMathContract",)
TTA_TESTS = ("test_contract_65.TtaContract", "test_flow_65.EngineFlow")
CAL_TESTS = ("test_contract_65.CalibrateContract",)
ORIGIN_TESTS = ("test_contract_65_api.FitTemperatureOriginContract", "test_contract_65_evaluate.OriginContract")
ENGINE_TESTS = ("test_flow_65.EngineFlow", "test_flow_65_api.EngineExtrasFlow")
LOADER_TESTS = ("test_flow_65.LoaderFlow",)
STATE_TESTS = ("test_contract_65.StateContract",)
BUDGET_TESTS = ("test_contract_65.BudgetContract", "test_contract_65_evaluate", "test_flow_65_evaluate.EvaluateFlow")
CONFIG_TESTS = ("test_contract_65.ConfigContract",)
API_TESTS = ("test_contract_65_api.NormalizeSpecContract", "test_contract_65_api.SpecForContract")
REMOTE_TESTS = ("test_flow_65.RemoteFlow",)

M = "decide/sd/merge.py"
T = "decide/sd/tta.py"
C = "decide/sd/calibrate.py"
E = "decide/sd/engine.py"
BU = "decide/sd/budget.py"
S = "decide/sd/state.py"
EV = "decide/sd/evaluate.py"
L = "decide/sd/loader.py"
CF = "decide/sd/config.py"
AP = "decide/sd/api.py"
R = "decide/sd/remote.py"

MUTATIONS: Tuple[Mutation, ...] = (
    # ---- merge math: task arithmetic, TIES, LoRA delta, how they are combined
    Mutation("merge", "ta-sign-flip", M, "out.add_(m.float() - b, alpha=float(w))", "out.add_(b - m.float(), alpha=float(w))", MERGE_TESTS,
             "the donor moves the anchor the wrong way"),
    Mutation("merge", "ta-ignore-weight", M, "out.add_(m.float() - b, alpha=float(w))", "out.add_(m.float() - b, alpha=1.0)", MERGE_TESTS,
             "the merge weight lambda is ignored"),
    Mutation("merge", "ties-sign-always-plus", M, "elected = torch.where((wv * stack).sum(0) >= 0, 1.0, -1.0)",
             "elected = torch.where((wv * stack).sum(0) >= 0, 1.0, 1.0)", MERGE_TESTS, "the sign election always picks +"),
    Mutation("merge", "ties-mean-not-disjoint", M, "den = (wv * agree).sum(0)", "den = wv.sum(0) * torch.ones_like(num)", MERGE_TESTS,
             "the disjoint mean divides by all weights, not by the donors that agree"),
    Mutation("merge", "ties-scale-dropped", M, "return b + total * merged", "return b + merged", MERGE_TESTS,
             "the TIES result is not scaled by the sum of the weights"),
    Mutation("merge", "trim-keeps-too-many", M, "k = min(n, max(1, math.ceil(round(density * n, 9))))",
             "k = min(n, max(1, math.ceil(round(density * n, 9)) + 1))", MERGE_TESTS, "trim keeps one entry more than density says"),
    Mutation("merge", "trim-keeps-smallest", M, "keep = a > thr", "keep = a < thr", MERGE_TESTS, "trim keeps the smallest magnitudes"),
    Mutation("merge", "lora-scale-wrong", M, "return alpha / math.sqrt(r) if rslora else alpha / r",
             "return alpha / math.sqrt(r) if rslora else alpha", MERGE_TESTS, "alpha is not divided by the rank"),
    Mutation("merge", "lora-weight-ignored", M, "return (weight * lora_scale(alpha, r, rslora)) * (B.float() @ A.float())",
             "return (lora_scale(alpha, r, rslora)) * (B.float() @ A.float())", MERGE_TESTS, "the small LoRA weight is not applied"),
    Mutation("merge", "lora-before-donors", M, "out = apply_lora(out, A, B, alpha, r, w, rs)", "out = apply_lora(anchor.float(), A, B, alpha, r, w, rs)",
             MERGE_TESTS, "the LoRA is added to the bare anchor and the donors' merge is thrown away"),
    Mutation("merge", "base-reference-drops-anchor", M, "models, weights = [anchor, *donors], [1.0, *donor_weights]",
             "models, weights = [anchor, *donors], [0.0, *donor_weights]", MERGE_TESTS, "with a base model the anchor no longer enters with weight 1"),
    Mutation("merge", "head-averaged", M, "return refit_head if refit_head is not None else anchor_head",
             "return (refit_head + anchor_head) / 2 if refit_head is not None else anchor_head", MERGE_TESTS,
             "heads are averaged (fase0b: never)"),

    # ---- option-order TTA: the permutations and, above all, their exact inverse
    Mutation("tta", "unpermute-uses-forward-map", T, "original[p] = presented[i]", "original[i] = presented[p]", TTA_TESTS,
             "the answer is mapped back with the permutation, not its inverse (invisible for a reversal, which is its own inverse)"),
    Mutation("tta", "invert-uses-forward-map", T, "inv[p] = i", "inv[i] = p", TTA_TESTS, "invert() returns the permutation itself"),
    Mutation("tta", "rotation-direction", T, "return [(i + shift) % n for i in range(n)]", "return [(i - shift) % n for i in range(n)]", TTA_TESTS,
             "cyclic shifts run the other way, against the documented convention"),
    Mutation("tta", "average-wrong-count", T, "return [sum(r[i] for r in rows) / len(rows) for i in range(n)]",
             "return [sum(r[i] for r in rows) / max(1, len(rows) - 1) for i in range(n)]", TTA_TESTS, "the mean divides by one view too few"),
    Mutation("tta", "permuted-text-stays-put", T, 'questions[qid] = {**q, "criteria": {keys[p]: q["criteria"][keys[p]] for p in perm}}',
             'questions[qid] = {**q, "criteria": {keys[i]: q["criteria"][keys[p]] for i, p in enumerate(perm)}}', TTA_TESTS,
             "the option labels keep their order while their descriptions move"),
    Mutation("tta", "no-dedupe-of-orders", T, "if perm not in out:", "if True:", TTA_TESTS, "a two-option question runs the same order twice"),
    Mutation("tta", "plan-repeats-first-order", T, "perms[min(v, len(perms) - 1)]", "perms[0]", TTA_TESTS,
             "a question with fewer distinct orders repeats the identity instead of its last order"),
    Mutation("tta", "restore-uses-forward-map", T, "return unpermute(values_by_presented, perm) if perm else list(values_by_presented)",
             "return apply(values_by_presented, perm) if perm else list(values_by_presented)", TTA_TESTS,
             "the engine's mapping back applies the forward permutation"),

    # ---- temperature: the softmax, the NLL, the search, the split
    Mutation("calibrate", "temperature-multiplies", C, "x = np.asarray(z, dtype=np.float64) / t", "x = np.asarray(z, dtype=np.float64) * t", CAL_TESTS,
             "T multiplies the logits instead of dividing them"),
    Mutation("calibrate", "fit-picks-worst-scan-point", C, "k = int(np.argmin(values))", "k = int(np.argmax(values))", CAL_TESTS,
             "the coarse scan keeps the worst temperature"),
    Mutation("calibrate", "golden-section-wrong-branch", C, "if fc < fd:", "if fc > fd:", CAL_TESTS, "the refinement walks uphill"),
    Mutation("calibrate", "nll-ignores-gold", C, "-np.log(np.maximum(p[rows, gold], EPS))", "-np.log(np.maximum(p.max(axis=1), EPS))", CAL_TESTS,
             "NLL scores the most likely option instead of the gold one"),
    Mutation("calibrate", "brier-mean-not-sum", C, "((p - onehot) ** 2).sum(axis=1)", "((p - onehot) ** 2).mean(axis=1)", CAL_TESTS,
             "Brier averages over options instead of summing"),
    Mutation("calibrate", "ece-without-abs", C, "abs(float(correct[mask].mean()) - float(conf[mask].mean()))",
             "(float(correct[mask].mean()) - float(conf[mask].mean()))", CAL_TESTS, "ECE lets over- and under-confidence cancel"),
    Mutation("calibrate", "mean-of-logits-not-softmaxes", C,
             "return [float(x) for x in _softmax(np.asarray(logits_by_view, dtype=np.float64), t).mean(axis=0)]",
             "return [float(x) for x in _softmax(np.asarray(logits_by_view, dtype=np.float64).mean(axis=0, keepdims=True), t).mean(axis=0)]",
             CAL_TESTS, "views are averaged as logits, not as probabilities"),
    Mutation("calibrate", "holdout-leaks-into-fit", C, "fit = [it for it in items if it.run_id not in held]", "fit = list(items)", CAL_TESTS,
             "the temperature is fitted on the rows it is judged on"),
    Mutation("calibrate", "holdout-size-off-by-one", C, "held = set(ranked[: int(round(holdout * len(ranked)))])",
             "held = set(ranked[: int(round(holdout * len(ranked))) + 1])", CAL_TESTS, "the holdout has one request more than asked"),
    Mutation("calibrate", "small-groups-never-fall-back", C, "if counts[g] >= min_items:", "if counts[g] >= 0:", CAL_TESTS,
             "a group with three rows gets its own temperature"),
    Mutation("calibrate", "k-bucket-boundary", C, '"3-4" if k <= 4', '"3-4" if k < 4', CAL_TESTS, "four options fall in the wrong bucket"),
    Mutation("calibrate", "fit-on-the-public-suite-allowed", AP, "if evaluation and allow_evaluation_data is not True:",
             "if False and allow_evaluation_data is not True:", ORIGIN_TESTS, "a temperature is fitted on public-suite rows without anyone asking for it"),
    Mutation("calibrate", "any-truthy-value-lifts-the-guard", AP, "if evaluation and allow_evaluation_data is not True:",
             "if evaluation and not allow_evaluation_data:", ORIGIN_TESTS, "allow_evaluation_data='False' lets evaluation rows through"),
    Mutation("calibrate", "result-forgets-the-evaluation-rows", AP, 'out["evaluation_rows"] = sum(evaluation.values())', 'out["evaluation_rows"] = 0',
             ORIGIN_TESTS, "a fit on evaluation data does not say so"),
    Mutation("calibrate", "gt-sample-is-not-evaluation-data", EV, 'EVALUATION_ORIGINS = (ORIGIN_SUITE, "gt-sample")', "EVALUATION_ORIGINS = (ORIGIN_SUITE,)",
             ORIGIN_TESTS, "the GT sample can be fitted on"),
    Mutation("calibrate", "suite-row-declares-another-origin", EV, 'meta = {**marker, "origin": ORIGIN_SUITE}', 'meta = {"origin": ORIGIN_SUITE, **marker}',
             ORIGIN_TESTS, "a suite row whose marker says another origin is no longer a suite row"),
    Mutation("calibrate", "plain-rows-have-no-origin", EV, 'meta = {"run_id": run_id, "origin": ORIGIN_REQUESTS, **{k: row[k]', 'meta = {"run_id": run_id, **{k: row[k]',
             ORIGIN_TESTS, "a request row carries no origin into the results"),
    Mutation("calibrate", "origins-missing-from-environment", EV, '"started_utc": _utc(), "origins": origins}', '"started_utc": _utc()}', ORIGIN_TESTS,
             "environment.json does not say what the run was made of"),
    Mutation("calibrate", "origins-missing-from-state", EV, 'state.set_meta(f"{prefix}/origins", origins)', 'state.set_meta(f"{prefix}/origins", {})', ORIGIN_TESTS,
             "the state store does not say what the run was made of"),
    Mutation("calibrate", "table-drops-the-origin", EV, 'base = {k: row.get(k) for k in ("run_id", "catalog_id", "group_id", "track", "origin", "status")}',
             'base = {k: row.get(k) for k in ("run_id", "catalog_id", "group_id", "track", "status")}', ORIGIN_TESTS, "the probability table forgets where its rows came from"),
    Mutation("calibrate", "no-origin-counts-as-requests", EV, "str(row.get(\"origin\") or ORIGIN_UNKNOWN)", "str(row.get(\"origin\") or ORIGIN_REQUESTS)", ORIGIN_TESTS,
             "a table that does not say where its rows came from is taken for clean ones"),

    # ---- the engine: where the inverse mapping and the temperature are applied
    Mutation("engine", "engine-skips-restoration", E, "rows = [tta.restore_question(view_logits[v][key], plans[v][key]) for v in range(len(plans))]",
             "rows = [list(view_logits[v][key]) for v in range(len(plans))]", ENGINE_TESTS, "the views are not mapped back to the original order"),
    Mutation("engine", "engine-ignores-calibration", E, "t = self.temperature_for(question)", "t = self.temperature", ENGINE_TESTS,
             "the calibrated temperatures never reach the answer"),
    Mutation("engine", "engine-general-key-first", E, 'for key in (f"{kind}:{calibrate.k_bucket(n)}", kind, "default"):',
             'for key in (kind, f"{kind}:{calibrate.k_bucket(n)}", "default"):', ENGINE_TESTS,
             "the temperature of the whole type wins over the one of its option count"),
    Mutation("engine", "oom-condemns-the-whole-batch", E, "return [out for job in jobs for out in self._run_jobs([job])]",
             "return [exc for _job in jobs]", ("test_flow_65.EngineFlow", "test_flow_65_evaluate.EvaluateFlow"),
             "one request that runs out of memory alone makes every request of its batch unsupported for good"),

    # ---- the weights must be the ones that were measured: pinned commits, and a file that is the size it says
    Mutation("config", "config-accepts-a-moving-branch", CF, "return isinstance(value, str) and REVISION.fullmatch(value) is not None",
             'return isinstance(value, str) and (value == "main" or REVISION.fullmatch(value) is not None)', CONFIG_TESTS,
             "'main' passes as a revision, so a push to the Hub changes the weights under a running sweep"),
    Mutation("config", "config-skips-the-rehearsal-pins", CF, 'for key, entry in (models.get("rehearsal") or {}).items():', "for key, entry in {}.items():",
             CONFIG_TESTS, "a model reached through the rehearsal section escapes the pin check"),
    Mutation("config", "config-skips-the-published-datasets", CF, "for label, entry in dataset_entries(raw):", "for label, entry in []:", CONFIG_TESTS,
             "the rows the fidelity check compares with can change under it"),
    Mutation("api", "spec-accepts-an-unpinned-anchor", AP, 'return {"anchor": anchor, "anchor_revision": _revision(spec.get("anchor_revision"), f"spec.anchor_revision ({anchor})"), "donors": donors,',
             'return {"anchor": anchor, "anchor_revision": spec.get("anchor_revision"), "donors": donors,', API_TESTS, "a hand-written spec reads the anchor from a moving branch"),
    Mutation("api", "spec-accepts-an-unpinned-donor", AP, '"revision": _revision(raw.get("revision"), f"{what} ({repo}): revision")}', '"revision": raw.get("revision")}',
             API_TESTS, "a donor or the LoRA is read from a moving branch"),
    Mutation("remote", "file-size-unchecked", R, "if total is not None and total != self.implied_size:", "if total is None and total != self.implied_size:", REMOTE_TESTS,
             "a truncated or padded shard is read as if it were sound"),
    Mutation("remote", "manifest-bytes-unchecked", R, "if sf.implied_size != entry[\"bytes\"]:", "if False and sf.implied_size != entry[\"bytes\"]:", REMOTE_TESTS,
             "the size the publisher listed is not compared with the file"),
    Mutation("remote", "manifest-sha256-unchecked", R, "if SHA256.fullmatch(tag) and tag != entry[\"sha256\"]:", "if SHA256.fullmatch(tag) and tag != tag:", REMOTE_TESTS,
             "the sha256 the publisher listed is not compared with the one the Hub holds"),

    # ---- a donor must change the anchor where it says it does, and the Simplicio delta must enter once
    Mutation("loader", "donor-coverage-unchecked", L, "if missing and not allow_partial:", "if False and missing and not allow_partial:", LOADER_TESTS,
             "a donor whose names match nothing loads without a word and the 'merge' is the anchor bit for bit"),
    Mutation("loader", "partial-donors-allowed-by-default", L, "allow_partial: bool = False", "allow_partial: bool = True", LOADER_TESTS,
             "the explicit switch for a partial donor is on unless somebody turns it off"),
    Mutation("loader", "base-naming-donor-accepted", L, "    if odd:", "    if False and odd:", LOADER_TESTS,
             "a merged base-model checkpoint (lm_head, mtp, model. prefix) is accepted as a donor"),
    Mutation("loader", "lora-repo-also-a-donor", L, 'if lora_spec and d["repo"].lower() == lora_spec["repo"].lower():',
             'if False and lora_spec and d["repo"].lower() == lora_spec["repo"].lower():', LOADER_TESTS, "the Simplicio delta is counted as a donor and as the LoRA"),
    Mutation("loader", "anchor-as-its-own-donor", L, 'if d["repo"].lower() == spec["anchor"].lower():', 'if False and d["repo"].lower() == spec["anchor"].lower():',
             LOADER_TESTS, "a donor equal to the anchor passes: its task vector is zero"),
    Mutation("api", "spec-lora-repo-also-a-donor", AP, 'if lora is not None and part["repo"].lower() == lora["repo"].lower():',
             'if False and lora is not None and part["repo"].lower() == lora["repo"].lower():', API_TESTS, "a spec with Simplicio-27B as donor and as LoRA is accepted"),
    Mutation("api", "spec-anchor-as-its-own-donor", AP, 'if part["repo"].lower() == anchor.lower():', 'if False and part["repo"].lower() == anchor.lower():', API_TESTS,
             "a spec whose donor is the anchor is accepted"),
    Mutation("config", "config-lora-repo-also-a-donor", CF, 'need(not (d.get("enabled", True) and lora_repo and str(d.get("repo", "")).lower() == lora_repo),',
             'need(not (False and d.get("enabled", True) and lora_repo and str(d.get("repo", "")).lower() == lora_repo),', CONFIG_TESTS,
             "the config accepts Simplicio-27B as an enabled donor with the LoRA on"),
    Mutation("config", "config-anchor-as-its-own-donor", CF, 'need(not (d.get("enabled", True) and str(d.get("repo", "")).lower() == anchor_repo),',
             'need(not (False and d.get("enabled", True) and str(d.get("repo", "")).lower() == anchor_repo),', CONFIG_TESTS, "the config accepts the anchor as a donor"),

    # ---- the time budget: a session that is over must not start another step
    Mutation("budget", "spent-budget-still-affords-a-free-step", BU, "return left > 0.0 and left >= float(next_cost)",
             "return left >= float(next_cost)", BUDGET_TESTS, "a chunk of unknown cost runs although the session is inside its reserve"),
    Mutation("budget", "first-chunk-not-priced", EV, "cost = sum(i.tokens() for i in piece) / tokens_per_second if tokens_per_second else 0.0",
             "cost = 0.0", BUDGET_TESTS, "the plan's measured throughput is ignored and the first chunk is always free"),

    # ---- the config says what will run, or it says no
    Mutation("config", "config-ignores-unknown-keys", CF, "for message in _unknown_keys(schema, raw[section], section):", "for message in []:", CONFIG_TESTS,
             "tta.rotatons, merge.max_candidate and the like run with the default and say nothing"),
    Mutation("config", "config-accepts-unknown-sections", CF, "if key not in SCHEMA and key not in set(foreign):", "if False and key not in SCHEMA and key not in set(foreign):",
             CONFIG_TESTS, "a section 'mergee' is read by nobody and accepted"),
    Mutation("config", "overlay-may-bring-any-section", CF, 'foreign.append(extra.name.split(".")[1])', "foreign.extend(yaml.safe_load(extra.read_text()) or {})",
             CONFIG_TESTS, "an overlay file can smuggle in a misspelled section of ours"),
    Mutation("config", "lora-weight-grid-allows-zero", CF, "all(_number(x) and x > 0 for x in weights)", "all(_number(x) and x >= 0 for x in weights)", CONFIG_TESTS,
             "a LoRA weight of zero runs the plain anchor under the name of a LoRA candidate"),
    Mutation("config", "session-longer-than-twelve-hours", CF, '0 < b.get("session_hours", 0) <= SESSION_HOURS_MAX,', '0 < b.get("session_hours", 0),', CONFIG_TESTS,
             "a plan for a 14 h Kaggle session passes"),
    Mutation("config", "calibration-grid-unordered", CF, "0 < window[0] < window[1]", "0 < window[0]", CONFIG_TESTS, "a search window [4, 0.5] only fails at the fit"),
    Mutation("config", "text-passes-as-a-number", CF, 'return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and x not in (float("inf"), float("-inf"))',
             "return True", CONFIG_TESTS, "a weight written as '0.5' ends in a TypeError instead of a config error"),
    Mutation("config", "plain-http-endpoint-accepted", CF, "need(problem is None,", "need(True or problem is None,", CONFIG_TESTS,
             "an http:// endpoint on another machine passes the config"),
    Mutation("remote", "token-over-plain-http", R, "            if problem:", "            if False and problem:", REMOTE_TESTS,
             "the reader builds a client that would send the token unencrypted to another machine"),

    # ---- a safetensors header is a claim about bytes; it is checked before any tensor is read
    Mutation("remote", "header-negative-offset-accepted", R, "or not 0 <= offsets[0] <= offsets[1]:", "or not offsets[0] <= offsets[1]:", REMOTE_TESTS,
             "a negative offset reads the header's own bytes as a tensor"),
    Mutation("remote", "header-size-not-checked-against-shape", R, "if offsets[1] - offsets[0] != numel * DTYPE_SIZES[dtype]:",
             "if False and offsets[1] - offsets[0] != numel * DTYPE_SIZES[dtype]:", REMOTE_TESTS, "a tensor whose byte span is not shape x dtype is reshaped from the wrong bytes"),
    Mutation("remote", "header-gap-or-overlap-accepted", R, "if a != expected:", "if a < expected:", REMOTE_TESTS, "a hole between two tensors passes"),
    Mutation("remote", "header-longer-than-the-file-accepted", R, "if total is not None and 8 + n > total:", "if False and total is not None and 8 + n > total:", REMOTE_TESTS,
             "a header length beyond the end of the file is read as an empty model"),
    Mutation("remote", "header-cut-short-accepted", R, "if len(first) < 8 + n:", "if False and len(first) < 8 + n:", REMOTE_TESTS,
             "a header that stops early is parsed as far as it goes"),
    Mutation("remote", "shard-name-from-the-index-unchecked", R, "if not safe_path(shard_name):", "if False and not safe_path(shard_name):", REMOTE_TESTS,
             "a shard name with .. or ? from a remote index goes into the URL"),
    Mutation("remote", "url-built-from-any-path", R, "if not safe_path(filename):", "if False and not safe_path(filename):", REMOTE_TESTS,
             "any file name, however hostile, becomes part of a URL"),
    Mutation("remote", "dotdot-segments-allowed", R, 'return isinstance(name, str) and SAFE_PATH.fullmatch(name) is not None and all(part not in (".", "..") for part in name.split("/"))',
             "return isinstance(name, str) and SAFE_PATH.fullmatch(name) is not None", REMOTE_TESTS, "'..' passes as a path segment"),

    # ---- the three the review's own mutation run found alive: a token on a redirect, a state file torn in half, a head from a donor
    Mutation("remote", "token-follows-a-cross-origin-redirect", R, "if self._origin(url) != first:", "if False and self._origin(url) != first:", REMOTE_TESTS,
             "the CDN a download is redirected to is shown the Hugging Face token"),
    Mutation("remote", "redirect-to-plain-http-allowed", R, 'if scheme not in ("http", "https") or (was == "https" and scheme == "http"):',
             'if scheme not in ("http", "https"):', REMOTE_TESTS, "a redirect may lower https to http"),
    Mutation("state", "state-written-in-place", S, 'tmp = self.path.with_name(STATE_NAME + ".tmp")', "tmp = self.path", STATE_TESTS,
             "a disk that fills halfway through leaves a torn state.json instead of the previous one"),
    Mutation("state", "absolute-local-path-allowed", S, "if pure.is_absolute():", "if False:", STATE_TESTS,
             "a state.json can point a mirrored file at an absolute path, even one inside the work directory"),
    Mutation("state", "dotdot-local-path-allowed", S, 'if ".." in pure.parts or pure.parts in ((), (".",)):', 'if pure.parts in ((), (".",)):', STATE_TESTS,
             "a state.json can point a mirrored file at 'runs/../x'"),
    Mutation("state", "drive-letter-path-allowed", S, '"\\0" in local or _DRIVE.match(local):', '"\\0" in local or False:', STATE_TESTS,
             "a state.json can name 'C:/...' as a local path"),
    Mutation("state", "link-out-of-the-work-directory-allowed", S, "if resolved != root and root not in resolved.parents:", "if False:", STATE_TESTS,
             "a link inside the work directory carries a restore or a push outside it"),
    Mutation("state", "link-to-the-work-directory-itself-allowed", S, "elif resolved == root:", "elif False:", STATE_TESTS,
             "a mirrored file can be the work directory itself"),
    Mutation("state", "store-state-not-cleaned", S, 'self._clean_files(data, "the store")', "pass", STATE_TESTS,
             "the paths of a state.json read from the store are used as they are"),
    Mutation("state", "local-state-not-cleaned", S, "self._clean_files(data)", "pass", STATE_TESTS,
             "the paths of a state.json read from the disk are used as they are"),
    Mutation("state", "mirror-name-not-checked", S, "_safe_name(str(remote))", "str(remote)", STATE_TESTS,
             "a state.json can name a store path that climbs"),
    Mutation("state", "files-that-are-not-a-table-kept", S, "if not isinstance(files, dict):", "if False:", STATE_TESTS,
             "a state.json whose files table is a list breaks the merge"),
    Mutation("state", "push-reads-any-local-path", S, "path = self._local_path(remote, rec)", 'path = self.workdir / rec["local"]', STATE_TESTS,
             "a push uploads a file that is not in the work directory"),
    Mutation("state", "restore-writes-any-local-path", S, "path = self._local_path(remote, entry)", 'path = self.workdir / entry["local"]', STATE_TESTS,
             "a restore overwrites a file that is not in the work directory"),
    Mutation("state", "track-file-refuses-a-dotted-folder", S, 'if rel == ".." or rel.startswith(".." + os.sep) or os.path.isabs(rel):',
             'if rel.startswith(".."):', STATE_TESTS, "a folder called '..runs' is taken for a way out of the work directory"),
    Mutation("loader", "donor-readout-lent", L, 'head = anchor.shard("readout.safetensors").read_tensor("weight")',
             'head = (donors[0] if donors else anchor).shard("readout.safetensors").read_tensor("weight")', LOADER_TESTS,
             "the readout head comes from a donor that ships one (heads are never merged or lent)"),

    # ---- a question that carries more than the prompt can hold is refused, not answered without the rest
    Mutation("engine", "reasoning-silently-dropped", E, 'if request.get("reasoning") is not None:', 'if False and request.get("reasoning") is not None:', ENGINE_TESTS,
             "a request with a reasoning text is answered as if the text were not there, and scored as if it had been read"),

    # ---- the loader: which tensors are merged with which weights
    Mutation("loader", "donor-weights-misaligned", L, "[weights[i] for i in plan.donors]", "[weights[i] for i in reversed(plan.donors)]",
             ("test_flow_65.LoaderFlow",), "donor i is merged with the weight of donor j"),
)


# --------------------------------------------------------------------------- the table itself

def groups() -> List[str]:
    seen: List[str] = []
    for m in MUTATIONS:
        if m.group not in seen:
            seen.append(m.group)
    return seen


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(source: str, mutation: Mutation) -> str:
    """`source` with the one replaced line; raises ValueError unless the text to replace occurs exactly once."""
    count = source.count(mutation.old)
    if count != 1:
        raise ValueError(f"{mutation.id}: the text to replace occurs {count} times in {mutation.file}, not once")
    return source.replace(mutation.old, mutation.new)


def table_problems(root: Path, mutations: Sequence[Mutation] = MUTATIONS) -> List[str]:
    """Everything wrong with the table, so it fails loudly when the code it points at moves: unknown group, duplicate id, missing file,
    a pattern that is not exactly once, a no-op, a multi-line edit, a mutated source that does not compile."""
    problems: List[str] = []
    ids = [m.id for m in mutations]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"{dup}: the id is used twice")
    for m in mutations:
        if m.group not in GROUPS:
            problems.append(f"{m.id}: unknown group {m.group!r}")
        if not m.tests:
            problems.append(f"{m.id}: no test guards it")
        if "\n" in m.old or "\n" in m.new:
            problems.append(f"{m.id}: a mutation is one line")
        if m.old == m.new:
            problems.append(f"{m.id}: the replacement is the same text")
        path = root / m.file
        if not path.is_file():
            problems.append(f"{m.id}: {m.file} does not exist")
            continue
        try:
            mutated = apply(path.read_text(encoding="utf-8"), m)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        try:
            compile(mutated, m.file, "exec")
        except SyntaxError as exc:
            problems.append(f"{m.id}: the mutated {m.file} does not compile ({exc.msg}, line {exc.lineno})")
    return problems


def one_line_changed(before: str, after: str) -> bool:
    """True when the two sources have the same number of lines and differ in exactly one of them."""
    a, b = before.splitlines(), after.splitlines()
    return len(a) == len(b) and sum(1 for x, y in zip(a, b) if x != y) == 1


# --------------------------------------------------------------------------- running tests

@dataclass
class Outcome:
    returncode: int
    ran: int
    skipped: int
    failed: List[str]               # ids of tests that failed or errored
    tail: str                       # the last line unittest printed
    seconds: float

    @property
    def green(self) -> bool:
        return self.returncode == 0


_RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
_SKIPPED = re.compile(r"skipped=(\d+)")
_FAILED = re.compile(r"^(?:FAIL|ERROR): (\S+)", re.M)


def parse_unittest(output: str, returncode: int, seconds: float = 0.0) -> Outcome:
    """What `python -m unittest` printed (to stderr), reduced to what a verdict needs."""
    ran = _RAN.search(output)
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    tail = lines[-1] if lines else ""
    skipped = _SKIPPED.search(tail)
    return Outcome(returncode, int(ran.group(1)) if ran else 0, int(skipped.group(1)) if skipped else 0, _FAILED.findall(output), tail, seconds)


def run_tests(root: Path, tests: Sequence[str], python: str, timeout: float) -> Outcome:
    """Run unittest ids from `<root>/tests`; a run that times out counts as red with the reason in `tail`."""
    started = time.time()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(name, None)
    try:
        proc = subprocess.run([python, "-m", "unittest", *tests], cwd=root / "tests", env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Outcome(1, 0, 0, [], f"timed out after {timeout:.0f} s", time.time() - started)
    return parse_unittest(proc.stderr + "\n" + proc.stdout, proc.returncode, time.time() - started)


def copy_repo(root: Path, dest: Path) -> None:
    """Only what the guarded tests read: the kit, the tests and the requirements file the pipeline checks the libraries against."""
    for rel in COPIED:
        source, target = root / rel, dest / rel
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=IGNORE)
        else:
            shutil.copy2(source, target)


# --------------------------------------------------------------------------- the check

@dataclass
class Result:
    mutation: Mutation
    status: str                     # KILLED | SURVIVED
    detail: str
    killed_by: List[str]
    seconds: float


class CannotJudge(RuntimeError):
    """The check itself is broken (a stale table, a red or skipping baseline, a file that did not restore); no verdict is given."""


def run(mutations: Sequence[Mutation], root: Path = ROOT, python: str = sys.executable, timeout: float = 900.0,
        say: Callable[[str], None] = print, keep: bool = False,
        runner: Callable[[Path, Sequence[str], str, float], Outcome] = run_tests) -> Dict[str, object]:
    """Mutate a copy of `root` one mutation at a time and report which ones the tests noticed.  Raises `CannotJudge` instead of guessing."""
    problems = table_problems(root, mutations)
    if problems:
        raise CannotJudge("the mutation table does not fit the code:\n  " + "\n  ".join(problems))
    targets = sorted({m.file for m in mutations})
    before = {rel: sha256(root / rel) for rel in targets}
    work = Path(tempfile.mkdtemp(prefix="sd-mutation-"))
    results: List[Result] = []
    baselines: Dict[Tuple[str, ...], Outcome] = {}
    try:
        copy_repo(root, work)
        pristine = {rel: (work / rel).read_text(encoding="utf-8") for rel in targets}
        for m in mutations:
            if m.tests not in baselines:
                base = runner(work, m.tests, python, timeout)
                if not base.green or base.ran == 0:
                    raise CannotJudge(f"the tests for {m.id} are not green before the mutation ({base.tail or 'no output'}): {', '.join(m.tests)}")
                if base.skipped:
                    raise CannotJudge(f"{base.skipped} of {base.ran} tests for {m.id} are skipped in this interpreter ({python}), so no verdict "
                                      "would mean anything: run the check where torch, numpy and the other test dependencies are installed")
                baselines[m.tests] = base
            target = work / m.file
            mutated = apply(pristine[m.file], m)
            if not one_line_changed(pristine[m.file], mutated):
                raise CannotJudge(f"{m.id}: the edit changed more or less than one line")
            target.write_text(mutated, encoding="utf-8")
            try:
                outcome = runner(work, m.tests, python, timeout)
            finally:
                target.write_text(pristine[m.file], encoding="utf-8")
            if sha256(target) != hashlib.sha256(pristine[m.file].encode("utf-8")).hexdigest():
                raise CannotJudge(f"{m.file} was not restored in the working copy")
            killed = not outcome.green
            detail = outcome.tail if killed else f"every one of {outcome.ran} tests passed with the bug in place"
            results.append(Result(m, "KILLED" if killed else "SURVIVED", detail, outcome.failed[:3], outcome.seconds))
            say(f"  {m.group:11s} {m.id:34s} {results[-1].status:8s} {detail}")
    finally:
        if keep:
            say(f"  (kept the working copy: {work})")
        else:
            shutil.rmtree(work, ignore_errors=True)
    after = {rel: sha256(root / rel) for rel in targets}
    if after != before:
        changed = sorted(rel for rel in targets if after[rel] != before[rel])
        raise CannotJudge(f"the working tree changed while the check ran: {', '.join(changed)}")
    survivors = [r.mutation.id for r in results if r.status == "SURVIVED"]
    return {"mutations": len(results), "killed": len(results) - len(survivors), "survivors": survivors, "tree_untouched": True,
            "files": {rel: before[rel] for rel in targets}, "results": results,
            "baselines": {", ".join(k): {"ran": v.ran, "seconds": round(v.seconds, 1)} for k, v in baselines.items()}}


def evidence(report: Dict[str, object], python: str) -> Dict[str, object]:
    """The report as plain JSON, for a pull request."""
    results: Iterable[Result] = report["results"]       # type: ignore[assignment]
    return {
        "python": python, "mutations": report["mutations"], "killed": report["killed"], "survivors": report["survivors"],
        "tree_untouched": report["tree_untouched"], "sha256_of_mutated_files": report["files"], "baselines": report["baselines"],
        "results": [{"group": r.mutation.group, "id": r.mutation.id, "file": r.mutation.file, "status": r.status, "killed_by": r.killed_by,
                     "why": r.mutation.why, "seconds": round(r.seconds, 1)} for r in results],
    }


def main(argv: Optional[Sequence[str]] = None, *, say: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(prog="python -m decide.sd.mutation", description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=GROUPS, help="one group of mutations")
    parser.add_argument("--id", action="append", default=[], metavar="ID", help="one mutation by id (repeatable); see --list")
    parser.add_argument("--list", action="store_true", help="list the mutations and stop")
    parser.add_argument("--json", metavar="FILE", help="write the evidence as JSON")
    parser.add_argument("--python", default=sys.executable, help="interpreter that runs the tests (default: this one)")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds allowed to one test run")
    parser.add_argument("--keep", action="store_true", help="keep the temporary working copy")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.id) - {m.id for m in MUTATIONS})
    if unknown:
        parser.error(f"unknown mutation id(s): {', '.join(unknown)} (see --list)")
    chosen = [m for m in MUTATIONS if (not args.only or m.group == args.only) and (not args.id or m.id in args.id)]
    if args.list:
        for m in chosen:
            say(f"{m.group:11s} {m.id:34s} {m.file}  [{', '.join(m.tests)}]\n{'':12s}{m.why}")
        return 0
    say(f"{len(chosen)} mutation(s), tests run with {args.python}")
    try:
        report = run(chosen, ROOT, args.python, args.timeout, say, args.keep)
    except CannotJudge as exc:
        say(f"cannot judge: {exc}")
        return 2
    say(f"\n{report['killed']}/{report['mutations']} killed; the working tree is byte-identical (sha256 of the {len(report['files'])} mutated files "  # type: ignore[arg-type]
        "compared before and after)")
    if report["survivors"]:
        say("SURVIVED (a hole in the tests): " + ", ".join(report["survivors"]))     # type: ignore[arg-type]
    if args.json:
        Path(args.json).write_text(json.dumps(evidence(report, args.python), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        say(f"evidence written to {args.json}")
    return 1 if report["survivors"] else 0


if __name__ == "__main__":
    sys.exit(main())
