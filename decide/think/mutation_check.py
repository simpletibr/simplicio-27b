"""Mutation check: plant a one-line bug in a COPY of the kit, run the tests, expect them to fail.

    python -m decide.think.mutation_check                 # every mutant
    python -m decide.think.mutation_check --group answers # one group (answers, bootstrap, verdict, scoring, resume, sample, trace, serve, report, provenance, security, pin, hosted, notebook, readme)
    python -m decide.think.mutation_check --id A1         # one mutant
    python -m decide.think.mutation_check --list

For each mutant: the file must hold `old` exactly once, the unmutated copy must pass, then the tests run
against the mutated copy (contract first, flow only if the contract did not notice).  A mutant that no test
notices SURVIVES, the run exits 1, and that is a missing test.  The working tree is never touched: the
mutants live in a temporary copy of decide/think, the notebooks, the test files, the two documents the README links to and
scripts/check_vllm_args.py.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from . import common as C

ROOT = Path(__file__).resolve().parents[2]
COPY = ("decide/think", "decide/PLAN.md", "docs/adr/0004-pensar-sob-demanda.md", "notebooks/decide", "notebooks/Simplicio_27B_Training_Colab.ipynb",
        "tests/test_contract_75.py", "tests/test_flow_75.py", "tests/think_nb_support.py", "scripts/check_vllm_args.py")
SUITES = (("contract", "tests.test_contract_75"), ("flow", "tests.test_flow_75"))


class Mutant(NamedTuple):
    id: str
    group: str
    file: str
    old: str
    new: str
    why: str


T = "decide/think/"
MUTANTS: Tuple[Mutant, ...] = (
    # --- reading the thinker's reply
    Mutant("A1", "answers", T + "answers.py", "for m in reversed(list(pattern.finditer(text))):", "for m in list(pattern.finditer(text)):",
           "the FIRST ANSWER line wins instead of the last"),
    Mutant("A2", "answers", T + "answers.py", "i = text.rfind(THINK_CLOSE)", "i = text.find(THINK_CLOSE)",
           "the text is split at the first </think>, not the last"),
    Mutant("A3", "answers", T + "answers.py", "return reasoning.strip(), text[i + len(THINK_CLOSE):].strip()", "return reasoning.strip(), text.strip()",
           "the reasoning is read for the answer too"),
    Mutant("A4", "answers", T + "answers.py", "if m and (m.group(1).isupper() or m.group(2) or not s[m.end():].strip()):", "if m:",
           "the article 'a' counts as option A"),
    Mutant("A5", "answers", T + "answers.py", "return text[j + len(THINK_OPEN):].strip(), None", 'return "", text[j + len(THINK_OPEN):].strip()',
           "a reply cut while thinking is read as a final answer"),
    Mutant("A6", "answers", T + "answers.py", "if m.start() > best_pos:", "if m.start() < best_pos:",
           "the last bare label is no longer the one that counts"),
    Mutant("A7", "answers", T + "answers.py", "single = len(options) == 1", "single = True",
           "multi-question replies are read as single-question ones"),
    # --- paired stratified bootstrap
    Mutant("B1", "bootstrap", T + "compare.py", "lo = min(b - 1, max(0, math.floor(round(a * b, 9))))", "lo = min(b - 1, max(0, math.ceil(round(a * b, 9))))",
           "lower CI rank rounds up"),
    Mutant("B2", "bootstrap", T + "compare.py", "hi = min(b - 1, max(0, math.ceil(round((1.0 - a) * b, 9)) - 1))", "hi = min(b - 1, max(0, math.ceil(round((1.0 - a) * b, 9))))",
           "upper CI rank is off by one"),
    Mutant("B3", "bootstrap", T + "compare.py", "hits[s][b][i] << (width * k)", "hits[s][b][(i + k) % n[b]] << (width * k)",
           "systems are no longer scored on the same resampled rows (pairing broken)"),
    Mutant("B4", "bootstrap", T + "compare.py", "[int(draw() * m) for _ in range(m)]", "[int(draw() * m) for _ in range(m + 1)]",
           "a resample draws one row too many"),
    Mutant("B5", "bootstrap", T + "compare.py", "draw = random.Random(seed).random", "draw = random.Random(0).random",
           "the seed is ignored"),
    Mutant("B6", "bootstrap", T + "compare.py", "return [sum(col[r] / n for col, n in cols) / k for r in range(self.B)]", "return [sum(col[r] / n for col, n in cols) / (k + 1) for r in range(self.B)]",
           "the macro average divides by one benchmark too many"),
    Mutant("B7", "bootstrap", T + "compare.py", "total = sum(map(table.__getitem__, [int(draw() * m) for _ in range(m)]))",
           "total = sum(map(table.__getitem__, [int(draw() * m) for _ in range(m)])) if b != benches[0] else sum(table)",
           "the first benchmark is not resampled (stratification broken)"),
    # --- the GT verdict
    Mutant("D1", "verdict", T + "compare.py", "if round(delta_pts, 9) < min_delta:", "if round(delta_pts, 9) <= min_delta:",
           "exactly +10.0 fails instead of passing"),
    Mutant("D2", "verdict", T + "compare.py", "if not round(ci_lo, 9) > 0:", "if not round(ci_lo, 9) >= 0:",
           "a CI floor of exactly 0 passes"),
    Mutant("D3", "verdict", T + "compare.py", "round(ci_hi, 9) >= min_delta", "round(ci_hi, 9) > min_delta",
           "can_still_pass is strict"),
    Mutant("D4", "verdict", T + "compare.py", "if round(delta_pts, 9) < min_delta:", "if delta_pts < min_delta:",
           "float noise decides the verdict"),
    Mutant("D5", "verdict", T + "compare.py", "return min(s.models, key=lambda m: (-round(s.macro_acc(m), 12), m))", "return max(s.models, key=lambda m: (-round(s.macro_acc(m), 12), m))",
           "the reference is the WORST decision model"),
    # --- scoring
    Mutant("S1", "scoring", T + "compare.py", "if v > best_value:", "if v >= best_value:",
           "a tie in the probabilities goes to the last option, not the first"),
    Mutant("S2", "scoring", T + "compare.py", "return bool(gold) and all(picks.get(qid) is not None and picks[qid] in accepted for qid, accepted in gold.items())",
           "return bool(gold) and any(picks.get(qid) is not None and picks[qid] in accepted for qid, accepted in gold.items())",
           "SATA rows need only one question right"),
    Mutant("S3", "scoring", T + "compare.py", "p *= len(row[\"gold\"][qid]) / len(q[\"criteria\"])", "p += len(row[\"gold\"][qid]) / len(q[\"criteria\"])",
           "row chance adds instead of multiplying"),
    Mutant("S4", "scoring", T + "common.py", "return min(1.0, max(0.0, (accuracy - chance) / (1.0 - chance)))", "return min(1.0, max(0.0, (accuracy - chance) / (1.0 + chance)))",
           "skill is not the kit's formula"),
    Mutant("S5", "scoring", T + "compare.py", "return statistics.fmean(self.acc(system, b) for b in self.benchmarks)", "return self.pooled_acc(system)",
           "the macro average is the pooled one"),
    # --- resume
    Mutant("R1", "resume", T + "common.py", 'TERMINAL_STATUSES = ("ok", "failed")', 'TERMINAL_STATUSES = ("ok", "failed", "error")',
           "a transient error row is never retried"),
    Mutant("R2", "resume", T + "thinker.py", 'if rec.get("truncated") and int(rec.get("max_tokens") or 0) < max_tokens:', 'if rec.get("truncated") and int(rec.get("max_tokens") or 0) > max_tokens:',
           "rows cut by a smaller budget are not redone when the budget grows"),
    Mutant("R3", "resume", T + "thinker.py", 'if int(config.get("max_tokens") or 0) < seen:', 'if int(config.get("max_tokens") or 0) > seen:',
           "the budget may go down, and a raise is refused"),
    Mutant("R4", "resume", T + "thinker.py", 'FINGERPRINT_EXCLUDE = ("max_tokens", "seed", "endpoint")', 'FINGERPRINT_EXCLUDE = ("max_tokens", "seed", "endpoint", "model")',
           "another model can resume somebody else's file"),
    # --- sample
    Mutant("M1", "sample", T + "sample.py", 'return C.sha256_text("|".join([str(seed), kind, *map(str, parts)]))', 'return C.sha256_text("|".join([kind, *map(str, parts)]))',
           "the seed does not reach the draw"),
    # --- the Raciocinio: block
    Mutant("T1", "trace", T + "trace.py", "tail = text[-budget:]", "tail = text[:budget]",
           "a long trace keeps its beginning instead of its end"),
    Mutant("T2", "trace", T + "trace.py", 'return unicodedata.normalize("NFC", line).strip() == MARKER', "return line.strip() == MARKER",
           "a decomposed-Unicode marker is missed"),
    Mutant("T3", "trace", T + "trace.py", "for i in range(len(lines) - 1, -1, -1):", "for i in range(len(lines)):",
           "the FIRST marker line starts the block"),
    Mutant("T4", "trace", T + "trace.py", "    if has_trace(state):\n        raise TraceError", "    if False:\n        raise TraceError",
           "a state with a block gets a second one"),
    # --- vllm serve
    Mutant("V1", "serve", T + "serve.py", "if host not in LOOPBACK:", "if False:",
           "the server can be bound to a public address"),
    Mutant("V2", "serve", T + "serve.py", 'json.dumps({"enable_thinking": bool(thinking)})', 'json.dumps({"enable_thinking": not thinking})',
           "thinking is served the wrong way round"),
    Mutant("V3", "serve", T + "serve.py", "    if revision:\n        argv += [\"--revision\", revision]", "    if False:\n        argv += [\"--revision\", revision]",
           "the pinned revision is not passed to vllm"),
    # --- report
    Mutant("L1", "report", T + "compare.py", 'mean = seconds.get("mean")', 'mean = seconds.get("median")',
           "the latency rule is read on the median, not the mean"),
    Mutant("L2", "report", T + "compare.py", "frac = 1.0 if mean <= 0 else min(1.0, rule_ms / (1000.0 * mean))", "frac = 1.0 if mean <= 0 else rule_ms / (1000.0 * mean)",
           "the fraction of requests that may think can pass 100%"),
    Mutant("L3", "report", T + "compare.py", 'if quant not in (None, "none"):', "if False:",
           "a 4-bit run gets no one-sided warning"),
    Mutant("L4", "report", T + "common.py", "return most / len(chunk) > 0.4 or readable / len(chunk) < 0.4", "return most / len(chunk) > 0.4",
           "a row of symbol noise with no repeated character reads as language"),
    Mutant("L5", "report", T + "common.py", 'chunk = "".join(text[:window].split())', 'chunk = "".join(text.split())',
           "the garbled-output check reads the whole reply, not its first 2000 characters"),
    # --- serve + think + compare on one box, and the Modal app around it
    Mutant("H1", "hosted", T + "hosted.py", "return code or report_code", "return report_code",
           "a thinker that stopped after consecutive errors exits 0"),
    Mutant("H2", "hosted", T + "hosted.py", "if code in (0, 3) and results.is_file():", "if results.is_file():",
           "the report is rewritten after the thinker refused the protocol"),
    Mutant("H3", "hosted", T + "hosted.py", "    finally:\n        server.stop()", "    finally:\n        pass",
           "the server keeps the GPU after an error"),
    Mutant("H4", "hosted", T + "hosted.py", "        if not server.alive():\n            return False", "        if False:\n            return False",
           "the wait goes on for 30 minutes after the server process died"),
    Mutant("H5", "hosted", T + "hosted.py", "return max(32768, max_tokens + CONTEXT_MARGIN)", "return 32768",
           "a bigger thinking budget does not fit the server's context"),
    Mutant("H6", "hosted", T + "hosted.py", "    if manifest:\n        problems = sample.verify(manifest, pack)", "    if False:\n        problems = sample.verify(manifest, pack)",
           "a pack that is not the sample starts a 29 GiB download"),
    Mutant("O1", "hosted", T + "modal_app.py", 'os.environ.get("SIMPLICIO_THINK_GPU", "H100")', 'os.environ.get("SIMPLICIO_THINK_GPU", "A10G")',
           "the Modal function asks for a GPU that cannot hold the FP8 checkpoint"),
    Mutant("O2", "hosted", T + "modal_app.py", "time_budget_s=float(TIMEOUT_S - MARGIN_S)", "time_budget_s=float(TIMEOUT_S)",
           "the thinker starts rows until the container is killed"),
    Mutant("O3", "hosted", T + "modal_app.py", "        thread.join(timeout=5)\n        commit_now()", "        thread.join(timeout=5)\n        pass",
           "the results are not committed to the Volume when the gate ends"),
    # --- a key that must not reach the results header, and an endpoint that does not think
    Mutant("X1", "report", T + "thinker.py", "    refuse_secrets(obj, what)\n    refuse_protocol_keys(obj, what)", "    refuse_protocol_keys(obj, what)",
           "a key typed into --extra-body is copied to the results header and the report"),
    Mutant("X2", "report", T + "thinker.py", "if SECRET_KEY.match(str(k)):", "if False:",
           "a field named api_key goes through"),
    Mutant("X3", "report", T + "thinker.py", "elif isinstance(obj, str) and SECRET_VALUE.match(obj):", "elif False:",
           "a value shaped like a key goes through under an innocent field name"),
    Mutant("W1", "report", T + "compare.py", 'ct["median"] < C.MIN_THINKING_TOKENS', 'ct["median"] <= C.MIN_THINKING_TOKENS',
           "a median of exactly 100 tokens is called not thinking"),
    Mutant("W2", "report", T + "compare.py", 'if thinking_on and st["rows"] and isinstance(', 'if st["rows"] and isinstance(',
           "a thinking-off control is told it did not think"),
    Mutant("W3", "report", T + "thinker.py", "stats.completion_tokens / stats.ok < C.MIN_THINKING_TOKENS", "stats.completion_tokens / stats.ok <= C.MIN_THINKING_TOKENS",
           "an average of exactly 100 tokens is called not thinking"),
    Mutant("W4", "report", T + "thinker.py", "if thinking and stats.ok and", "if stats.ok and",
           "the thinking-off control gets the warning in the log"),
    # --- what the results header says about how the model ran
    Mutant("P1", "provenance", T + "thinker.py", "nf4_config_kwargs(torch_dtype))", "nf4_config_kwargs(torch.float16))",
           "the NF4 matmuls ignore --dtype: a bf16 rerun still computes in fp16"),
    Mutant("P2", "provenance", T + "thinker.py", '"dtype": args.dtype if args.backend == "transformers" else None,', '"dtype": None,',
           "the dtype is not recorded in the results header, so a rerun in another dtype resumes the same file"),
    # --- the notebooks: a flag that is renamed, a protocol that is changed, a pin that drifts
    Mutant("N1", "notebook", "notebooks/decide/think_gt_kaggle.ipynb", "--quant nf4", "--quant none",
           "the Kaggle notebook stops quantizing (a 27B model on 2 x 15 GB T4)"),
    Mutant("N2", "notebook", T + "thinker.py", 'ap.add_argument("--time-budget-s", type=float,', 'ap.add_argument("--time-budget", type=float,',
           "the Kaggle command line is refused by the thinker"),
    Mutant("N3", "notebook", "notebooks/decide/think_gt_api.ipynb", "del key", "pass",
           "the API key stays in a variable of the notebook"),
    Mutant("N4", "notebook", "notebooks/decide/think_gt_kaggle.ipynb", 'if LIMIT else \\"\\"', 'if True else \\"\\"',
           "the full run still passes --limit"),
    Mutant("N5", "notebook", "notebooks/decide/think_gt_api.ipynb", 'SIMPLICIO_SHA = \\"', 'SIMPLICIO_SHA = \\"0',
           "one notebook pins something that is not a full commit"),
    Mutant("N6", "notebook", T + "compare.py", 'ap.add_argument("--thinker", required=True', 'ap.add_argument("--thinker-results", required=True',
           "the report command of the notebooks is refused by compare"),
    Mutant("N7", "notebook", T + "common.py", '"GIT_DIR", "GIT_WORK_TREE", "GIT_IMPLICIT_WORK_TREE",', '"GIT_WORK_TREE", "GIT_IMPLICIT_WORK_TREE",',
           "the git commands of the flow tests inherit GIT_DIR from a pre-push hook and commit to the repository being pushed"),
    # --- the README: a pin that drifts, a heading the code points at, a command that stops parsing
    Mutant("E1", "readme", T + "README.md", "`Qwen/Qwen3.8-27B-FP8` em `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`", "`Qwen/Qwen3.8-27B-FP8` em `017b9c7af6b5689d5dd426a76e0bc077eb5ca20b`",
           "the README pins another FP8 revision than the code"),
    Mutant("E2", "readme", T + "README.md", "8791ecba4ed4e6eabd7b6708baacb0e2791adc326e0b9483ce6b44c7b505b7db  d3-turbo/results.jsonl.gz",
           "9791ecba4ed4e6eabd7b6708baacb0e2791adc326e0b9483ce6b44c7b505b7db  d3-turbo/results.jsonl.gz",
           "the sha256 block of the inputs disagrees with the manifest"),
    Mutant("E3", "readme", T + "README.md", "## Tempo e custo", "## Tempo",
           "a heading that modal_app.py points at was renamed"),
    Mutant("E4", "readme", T + "README.md", "--concurrency 8 --max-tokens 16384 --pack", "--concurrency 8 --max-tokenz 16384 --pack",
           "a pasted thinker command has a flag the thinker does not know"),
    Mutant("E5", "readme", T + ".gitignore", "*.jsonl.gz\n", "",
           "the private pack is no longer ignored by git"),
    Mutant("E6", "readme", T + "README.md", "(../../docs/adr/0004-pensar-sob-demanda.md)", "(../../docs/adr/0004-pensar.md)",
           "a relative link points at nothing"),
    Mutant("E7", "readme", T + "README.md", "**4** o servidor não subiu", "**5** o servidor não subiu",
           "the README documents another exit code than hosted.py"),
    Mutant("E8", "readme", T + "README.md", "python -m decide.think.hosted --pack pack.jsonl.gz --work work --limit 3", "python -m decide.think.hosted --pack pack.jsonl.gz --workdir work --limit 3",
           "the pasted hosted probe has a flag hosted.py does not know"),
    # --- review of #78, blocker 1: PASS and FAIL only for a run that is the registered gate
    Mutant("D6", "verdict", T + "compare.py", "    elif incomplete:\n", "    elif False:\n",
           "a run with rows missing is told PASS or FAIL instead of PARTIAL"),
    Mutant("D7", "verdict", T + "compare.py", "    elif exploratory:\n", "    elif False:\n",
           "a run outside the registered protocol is told PASS or FAIL instead of EXPLORATORY"),
    Mutant("D8", "verdict", T + "compare.py", "    if inconclusive:\n        verdict.update(", "    if False:\n        verdict.update(",
           "a thinker that was not seen thinking is told PASS or FAIL"),
    Mutant("D9", "verdict", T + "compare.py", "    if inconclusive:\n        verdict.update(", "    if inconclusive and not incomplete:\n        verdict.update(",
           "PARTIAL outranks INCONCLUSIVE"),
    Mutant("D10", "verdict", T + "compare.py", "    elif incomplete:\n", "    elif incomplete and not exploratory:\n",
           "EXPLORATORY outranks PARTIAL"),
    Mutant("D11", "verdict", T + "compare.py", 'OFFICIAL = ("PASS", "FAIL")', 'OFFICIAL = ("PASS", "FAIL", "PARTIAL")',
           "a PARTIAL run is printed like a verdict"),
    Mutant("D12", "verdict", T + "compare.py", '    if B != C.BOOTSTRAP_B:\n        out.append("bootstrap_b")', '    if False:\n        out.append("bootstrap_b")',
           "a run with another number of resamples is not marked"),
    Mutant("D13", "verdict", T + "compare.py", '    if seed != C.SEED:\n        out.append("seed")', '    if False:\n        out.append("seed")',
           "a run with another bootstrap seed is not marked"),
    Mutant("D14", "verdict", T + "compare.py", 'if round(min_delta, 9) != C.GT_MIN_DELTA_PTS:', 'if False:',
           "a run against another target is not marked"),
    Mutant("D15", "verdict", T + "compare.py", 'if models is not None and set(models) != set(present):', 'if False:',
           "a comparison with a subset of the decision models is not marked"),
    Mutant("D16", "verdict", T + "compare.py", "    out += [x for x in pack_issues if x not in out]\n", "",
           "a pack that is not the manifest's is not marked"),
    Mutant("D17", "verdict", T + "compare.py", "    out += thinker_deviations(cfg)\n", "",
           "a thinker that is not the registered one is not marked"),
    Mutant("D18", "verdict", T + "compare.py", 'st["usage_missing"] > MAX_NO_USAGE_SHARE * st["ok"]:', 'st["usage_missing"] >= MAX_NO_USAGE_SHARE * st["ok"]:',
           "exactly half of the replies without usage is called too many"),
    Mutant("D19", "verdict", T + "compare.py", 'if st["ok"] and st["usage_missing"] > MAX_NO_USAGE_SHARE * st["ok"]:', 'if False:',
           "replies without a token count do not mark the run"),
    Mutant("D20", "verdict", T + "compare.py", 'out.append("thinking_off")', 'pass',
           "a thinking-off control is not marked"),
    Mutant("D21", "verdict", T + "compare.py", 'if (model, rev) not in set(C.PINNED_REVISIONS.items()):', 'if False:',
           "another model or revision is not marked"),
    Mutant("D22", "verdict", T + "compare.py", '"revision_not_declared" if model in C.PINNED_REVISIONS and not rev else "model_not_pinned"', '"model_not_pinned"',
           "a pinned model with no declared revision gets the wrong reason"),
    Mutant("D23", "verdict", T + "compare.py", '    if cfg.get("standard_only"):\n        out.append("standard_only")', '    if False:\n        out.append("standard_only")',
           "--standard-only is not marked in the report and status"),
    Mutant("D24", "verdict", T + "compare.py", 'if thinking and cfg.get("sampling") is not None and dict(cfg["sampling"]) != dict(C.THINKING_SAMPLING):', 'if False:',
           "another sampling is not marked"),
    Mutant("D25", "verdict", T + "compare.py", 'if cfg.get("reasoning_effort") not in (None, "xhigh"):', 'if False:',
           "another reasoning effort is not marked"),
    Mutant("D26", "verdict", T + "compare.py", "    if protocol_keys_in(cfg):\n", "    if False:\n",
           "protocol fields set by an older results file's extra body are not marked"),
    Mutant("D27", "verdict", T + "compare.py", '    if not cfg:\n        return ["thinker_config_unknown"]', '    if False:\n        return ["thinker_config_unknown"]',
           "a results file without a header is taken as the registered thinker"),
    Mutant("D28", "verdict", T + "compare.py", 'if st["ok"] and st["thought"] < MIN_THOUGHT_SHARE * st["ok"]:', 'if st["ok"] and st["thought"] <= MIN_THOUGHT_SHARE * st["ok"]:',
           "exactly half of the rows showing thinking is called not detected"),
    Mutant("D29", "verdict", T + "compare.py", 'MIN_THOUGHT_SHARE = 0.5 ', 'MIN_THOUGHT_SHARE = 0.0 ',
           "a thinker never seen thinking still passes"),
    Mutant("D30", "verdict", T + "compare.py", '    if st["rows"] and not st["ok"]:\n        return ["no_replies"]', '    if False:\n        return ["no_replies"]',
           "a run in which nobody answered is scored like a thinker that was wrong"),
    Mutant("D31", "verdict", T + "compare.py", '    if not thinking_requested(cfg):\n        return []', '    if False:\n        return []',
           "a thinking-off control is called INCONCLUSIVE"),
    Mutant("D32", "verdict", T + "compare.py", '.get("enable_thinking", True) is not False\n\n\ndef thinker_deviations', '.get("enable_thinking", False) is not False\n\n\ndef thinker_deviations',
           "a results file without a header is read as a thinking-off control"),
    Mutant("D33", "verdict", T + "compare.py", '        issues.append("pack_mismatch_ignored")', '        pass',
           "--ignore-pack-mismatch leaves no trace in the status"),
    Mutant("D34", "verdict", T + "compare.py", '        issues.append("manifest_not_checked")', '        pass',
           "a missing manifest leaves the verdict unmarked"),
    Mutant("D35", "verdict", T + "compare.py", '        issues.append("pack_not_manifest")', '        pass',
           "a pack that is not the manifest's leaves the verdict unmarked"),
    Mutant("D36", "verdict", T + "compare.py", "pack_sha256=pack_sha, pack_issues=issues)", "pack_sha256=pack_sha)",
           "compare.main forgets what it found about the pack"),
    Mutant("D37", "verdict", T + "compare.py", "        if status not in TERMINAL_STATUSES:\n            errors.append(r[\"id\"])\n            continue", "        if False:\n            errors.append(r[\"id\"])\n            continue",
           "a row with a transient error is scored as wrong instead of waiting for its next try"),
    # --- review of #78, blocker 1 and the optionals: rows that cannot be answered, what is stored, where it came from
    Mutant("R5", "resume", T + "thinker.py", 'if streaks[r["id"]] >= max_attempts:', 'if streaks[r["id"]] > max_attempts:',
           "a row gets one more try than --max-attempts"),
    Mutant("R6", "resume", T + "thinker.py", 'streaks[r["id"]] = streaks.get(r["id"], 0) + 1', 'streaks[r["id"]] = 1',
           "a row that keeps failing is never given up"),
    Mutant("R7", "resume", T + "thinker.py", 'out[r["id"]] = out.get(r["id"], 0) + 1 if r.get("status") == "error" else 0', 'out[r["id"]] = out.get(r["id"], 0) + 1 if r.get("status") == "error" else out.get(r["id"], 0)',
           "the count of errors in a row survives a success"),
    Mutant("R8", "resume", T + "thinker.py", 'or (retry_failed and rec.get("status") == "failed")', 'or (False and rec.get("status") == "failed")',
           "--retry-failed does not run the failed rows again"),
    Mutant("R9", "resume", T + "thinker.py", 'str(e)[:120]}", transient=False)]', 'str(e)[:120]}", transient=True)]',
           "a row that does not fit the GPU with one sequence is retried for ever"),
    Mutant("R10", "resume", T + "common.py", "DEFAULT_MAX_ATTEMPTS = 3", "DEFAULT_MAX_ATTEMPTS = 30",
           "a row gets thirty tries before it counts as an error"),
    Mutant("R11", "resume", T + "thinker.py", '    if keep_reasoning:\n        rec["reasoning"] = p.reasoning', '    if True:\n        rec["reasoning"] = p.reasoning',
           "the results store the reasoning on the evaluation sample"),
    Mutant("R12", "resume", T + "thinker.py", "keep_reasoning=args.keep_reasoning and not args.no_reasoning", "keep_reasoning=not args.no_reasoning",
           "the reasoning is stored unless the user asks not to"),
    Mutant("R13", "resume", T + "thinker.py", '    if endpoint:\n        base["endpoint"] = endpoint', '    if False:\n        base["endpoint"] = endpoint',
           "a row does not say which endpoint answered it"),
    Mutant("R14", "resume", T + "thinker.py", 'endpoint = store.config.get("endpoint")', "endpoint = None",
           "the runner never learns the endpoint"),
    # --- review of #78, blocker 2: a thinker that was not seen thinking never passes
    Mutant("W5", "report", T + "thinker.py", "return tokens >= C.MIN_THINKING_TOKENS or (not p.no_close and bool(p.reasoning.strip()))", "return tokens > C.MIN_THINKING_TOKENS or (not p.no_close and bool(p.reasoning.strip()))",
           "a reply of exactly 100 tokens is not thinking"),
    Mutant("W6", "report", T + "thinker.py", "return tokens >= C.MIN_THINKING_TOKENS or (not p.no_close and bool(p.reasoning.strip()))", "return tokens >= C.MIN_THINKING_TOKENS or bool(p.reasoning.strip())",
           "the whole reply of an endpoint that did not think is counted as its thinking"),
    Mutant("W7", "report", T + "thinker.py", "return tokens >= C.MIN_THINKING_TOKENS or (not p.no_close and bool(p.reasoning.strip()))", "return tokens >= C.MIN_THINKING_TOKENS",
           "reasoning returned apart does not show thinking"),
    Mutant("W8", "report", T + "thinker.py", "    if thinking and stats.no_thinking:", "    if stats.no_thinking:",
           "the thinking-off control is told that its replies show no thinking"),
    Mutant("W9", "report", T + "thinker.py", "    if thinking and stats.no_close:", "    if stats.no_close:",
           "the thinking-off control is told that its replies did not close the thinking"),
    Mutant("W10", "report", T + "compare.py", '    if st.get("no_close"):', '    if False:',
           "the report does not count the replies that stopped without closing the thinking"),
    Mutant("W11", "report", T + "compare.py", '    if st.get("ok") and st["thought"] < st["ok"]:', '    if st.get("ok") and st["thought"] < st["ok"] - 1:',
           "one reply without thinking goes unmentioned"),
    Mutant("W12", "report", T + "compare.py", '    if st.get("usage_missing"):', '    if False:',
           "replies without a token count go unmentioned"),
    Mutant("W13", "report", T + "compare.py", '    if st.get("read_by_fallback"):', '    if False:',
           "the report does not count the replies read without an ANSWER line"),
    Mutant("W14", "report", T + "compare.py", '    if st.get("gave_up"):', '    if False:',
           "the report does not count the rows given up"),
    Mutant("W15", "report", T + "compare.py", '    if len(st.get("endpoints") or []) > 1:', '    if len(st.get("endpoints") or []) > 2:',
           "two endpoints in one file go unmentioned"),
    Mutant("W16", "report", T + "compare.py", '    if (((report.get("thinker") or {}).get("config") or {}).get("standard_only")):', '    if False:',
           "--standard-only goes unmentioned in the report"),
    Mutant("W17", "report", T + "compare.py", 'for how in (rec.get("read_by") or {}).values())', 'for how in ())',
           "no reply counts as read by a fallback"),
    Mutant("W18", "report", T + "compare.py", 'any(how not in (None, "marker") for how in', 'any(how not in (None, "marker", "bare") for how in',
           "a reply read from a loose label is not counted as a fallback"),
    Mutant("W19", "report", T + "thinker.py", '(f" ({gave_up} given up after repeated errors)" if gave_up else "")', '""',
           "the log does not say how many rows were given up"),
    Mutant("W20", "report", T + "common.py", '"frequency_penalty", "repetition_penalty", "seed", "stream", "stream_options", "chat_template_kwargs", "reasoning_effort",', '"frequency_penalty", "repetition_penalty", "seed", "stream", "stream_options", "reasoning_effort",',
           "--extra-body may switch thinking off with chat_template_kwargs"),
    Mutant("W21", "report", T + "common.py", '"max_completion_tokens", "temperature", "top_p",', '"max_completion_tokens", "top_p",',
           "--extra-body may change the temperature"),
    # --- review of #78, blocker 3: the key goes over https or to this computer, and nowhere else
    Mutant("K1", "security", T + "thinker.py", "opener: Callable[..., Any] = open_no_redirect,", "opener: Callable[..., Any] = urllib.request.urlopen,",
           "the backend follows redirects again, key included"),
    Mutant("K2", "security", T + "thinker.py", "        return None                                           # urllib then raises", "        return super().redirect_request(req, fp, code, msg, headers, newurl)   # urllib then raises",
           "the no-redirect handler follows the redirect"),
    Mutant("K3", "security", T + "thinker.py", "_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)", "_NO_REDIRECT_OPENER = urllib.request.build_opener()",
           "the no-redirect opener is the default one"),
    Mutant("K4", "security", T + "thinker.py", 'if api_key and parts.scheme.lower() != "https" and not is_loopback_host(parts.hostname):', 'if False:',
           "the key is sent in clear over http:// to another host"),
    Mutant("K5", "security", T + "thinker.py", 'if api_key and parts.scheme.lower() != "https" and not is_loopback_host(parts.hostname):', 'if api_key and parts.scheme.lower() != "https":',
           "the key is refused over http:// to this computer (a local vLLM or an SSH tunnel)"),
    Mutant("K6", "security", T + "thinker.py", "return ipaddress.ip_address(host).is_loopback", "return ipaddress.ip_address(host).is_private",
           "a private-network address (10.x, 192.168.x) counts as this computer"),
    Mutant("K7", "security", T + "thinker.py", '    if host == "localhost":\n        return True', '    if host == "localhostx":\n        return True',
           "localhost does not count as this computer"),
    Mutant("K8", "security", T + "thinker.py", 'if parts.scheme.lower() not in ("http", "https") or not parts.hostname:', 'if not parts.hostname:',
           "a base URL with another scheme is accepted"),
    Mutant("K9", "security", T + "thinker.py", "if str(k).lower() in C.PROTOCOL_KEYS)\n        if bad:", "if str(k) in C.PROTOCOL_KEYS)\n        if bad:",
           "the backend refuses protocol fields only when they are spelled in lower case"),
    Mutant("K10", "security", T + "thinker.py", "    refuse_protocol_keys(obj, what)\n    return obj", "    return obj",
           "--extra-body may set the fields of the protocol"),
    Mutant("K11", "security", T + "thinker.py", "        bad = sorted(k for k in (extra_body or {}) if str(k).lower() in C.PROTOCOL_KEYS)\n        if bad:", "        bad = []\n        if bad:",
           "the backend accepts an extra body that sets protocol fields when built from code"),
    Mutant("K12", "security", T + "thinker.py", '                if 300 <= e.code < 400:\n                    raise BackendError(f"HTTP {e.code}: {_redirect_hint(e)}") from e', '                if False:\n                    raise BackendError(f"HTTP {e.code}: {_redirect_hint(e)}") from e',
           "a redirect on a row is retried as a transient error"),
    Mutant("K13", "security", T + "thinker.py", '            if 300 <= e.code < 400:\n                raise BackendError(f"preflight failed: HTTP {e.code}: {_redirect_hint(e)}") from e', '            if False:\n                raise BackendError(f"preflight failed: HTTP {e.code}: {_redirect_hint(e)}") from e',
           "a redirect on the preflight has no hint"),
    Mutant("K14", "security", T + "thinker.py", 'where = sanitize_url((e.headers.get("Location") if e.headers else None) or "") or "an unknown address"', 'where = (e.headers.get("Location") if e.headers else None) or "an unknown address"',
           "the redirect message shows the query of the address it points to"),
    # --- review of #78, blocker 4: the pin must be in the history of the checkout, or the test fails and says how to fetch it
    Mutant("G1", "pin", "tests/think_nb_support.py", 'if git("merge-base", "--is-ancestor", sha, "HEAD").returncode != 0:', "if False:",
           "a pin that is not in the history of HEAD passes"),
    Mutant("G2", "pin", "tests/think_nb_support.py", 'if git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:', "if False:",
           "a pin that is not in the clone passes"),
    Mutant("G3", "pin", "tests/think_nb_support.py", '" (it is a shallow clone)" if shallow else ""', '""',
           "the message does not say that the clone is shallow"),
    Mutant("G4", "pin", "tests/think_nb_support.py", "Get it with `git fetch --unshallow`, or", "Get it with `git pull`, or",
           "the message does not say how to fetch the pinned commit"),
    Mutant("G5", "pin", "tests/think_nb_support.py", 'return "not_git", "not a git checkout"', 'return "ok", ""',
           "a source archive is told its pin is fine"),
    Mutant("G6", "pin", T + "mutation_check.py", "return {**C.without_local_git_vars(dict(os.environ)), \"THINK_MUTATION_CHECK\": \"1\"}", "return {**dict(os.environ), \"THINK_MUTATION_CHECK\": \"1\"}",
           "the tests of a mutant inherit GIT_DIR from a pre-push hook"),
    Mutant("G7", "pin", T + "common.py", 'if k not in LOCAL_GIT_VARS and not k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))}', 'if k not in LOCAL_GIT_VARS}',
           "the injected git config of a hook is kept"),
    # --- review of #78, optionals: evaluation data, the reader, the sample, the image
    Mutant("T5", "trace", T + "trace.py", '    _refuse_eval_data(rec.get("id"))\n', "",
           "a row of the gate's sample becomes a training trace"),
    Mutant("T6", "trace", T + "trace.py", "    _refuse_eval_data(source_id)\n", "",
           "a trace is attached to a request of the gate's sample"),
    Mutant("T7", "trace", T + "trace.py", "if source_id is not None and source_id in gt_sample_ids():", "if source_id is not None and source_id not in gt_sample_ids():",
           "only the ids that are NOT of the gate's sample are refused"),
    Mutant("A8", "answers", T + "answers.py", "    m = found[-1]\n", "    m = found[0]\n",
           "the FIRST ANSWER line decides when its value is on the next line or invalid"),
    Mutant("A9", "answers", T + "answers.py", "    if not value.strip():\n        value = next(", "    if False:\n        value = next(",
           "an ANSWER line with the value on the next line is unanswered"),
    Mutant("A10", "answers", T + "answers.py", '    if present:\n        return hit, "marker"', '    if present and hit:\n        return hit, "marker"',
           "a last ANSWER line that is not an option falls back to an earlier guess"),
    Mutant("A11", "answers", T + "answers.py", '    if present:\n        return hit, "marker"', '    if present:\n        return hit, "bare"',
           "an ANSWER line is counted as a fallback reading"),
    Mutant("A12", "answers", T + "thinker.py", "elif thinking and not truncated and THINK_CLOSE not in content and not gen.reasoning_tokens and content.strip():", "elif False and not truncated and THINK_CLOSE not in content and not gen.reasoning_tokens and content.strip():",
           "a reply that stopped without closing the thinking is read as an answer"),
    Mutant("A13", "answers", T + "thinker.py", "elif thinking and not truncated and THINK_CLOSE not in content and not gen.reasoning_tokens and content.strip():", "elif thinking and not truncated and THINK_CLOSE not in content and content.strip():",
           "a reply that counted reasoning tokens but hid the text is unreadable"),
    Mutant("A14", "answers", T + "thinker.py", "elif thinking and not truncated and THINK_CLOSE not in content and not gen.reasoning_tokens and content.strip():", "elif not truncated and THINK_CLOSE not in content and not gen.reasoning_tokens and content.strip():",
           "the thinking-off control has its replies called unclosed thinking"),
    Mutant("M2", "sample", T + "sample.py", 'if int((row.get("_evaluation") or {}).get("catalog_id", -1)) not in ACCEPTED_BENCHMARKS:', "if False:",
           "any benchmark's accepted answers count as gold"),
    Mutant("M3", "sample", T + "sample.py", "ACCEPTED_BENCHMARKS = (31, 50)", "ACCEPTED_BENCHMARKS = (25, 31, 50)",
           "a third benchmark reads the accepted answers"),
    Mutant("O4", "hosted", T + "modal_app.py", '"**/*.jsonl", ', "",
           "the Modal image carries a results file left in the folder"),
    Mutant("O5", "hosted", T + "modal_app.py", '"**/pack*", ', "",
           "the Modal image carries the private pack left in the folder"),
    Mutant("O6", "hosted", T + "modal_app.py", '"**/work", "**/report"]', '"**/work"]',
           "the Modal image carries a report folder"),
    Mutant("H7", "hosted", T + "hosted.py", '*(["--manifest", manifest] if manifest else [])', "*[]",
           "hosted compares against whatever manifest the working directory has"),
    # --- review of #78, optionals: a value with a space or an & in a notebook command
    Mutant("N8", "notebook", "notebooks/decide/think_gt_kaggle.ipynb", "--pack {q(PACK)} --manifest decide/think/gt-sample-manifest.json --out {q(WORK)}/results.jsonl --time-budget-s", "--pack {PACK} --manifest decide/think/gt-sample-manifest.json --out {q(WORK)}/results.jsonl --time-budget-s",
           "a path with a space breaks the Kaggle thinker command"),
    Mutant("N9", "notebook", "notebooks/decide/think_gt_api.ipynb", "--base-url {q(BASE_URL)}", "--base-url {BASE_URL}",
           "a base URL with an & splits the command in two"),
    Mutant("N10", "notebook", "notebooks/decide/think_gt_api.ipynb", "--model {q(MODEL)}", "--model {MODEL}",
           "a model id with a space splits the command"),
    Mutant("N11", "notebook", "notebooks/decide/think_gt_api.ipynb", "{q(EXTRA_BODY)}", "{EXTRA_BODY}",
           "the extra body is not quoted for the shell"),
    Mutant("N12", "notebook", "notebooks/decide/think_gt_kaggle.ipynb", "from shlex import quote as q", "q = str  #",
           "the quoting function of the Kaggle notebook quotes nothing"),
    Mutant("N13", "notebook", "notebooks/decide/think_gt_kaggle.ipynb", "`/kaggle/working` é a saída da versão salva", "`/kaggle/working` é uma pasta",
           "the Kaggle notebook no longer warns that its output is public with the notebook"),
    # --- review of #78, README
    Mutant("E9", "readme", T + "README.md", "`extra_body_overrides_protocol`, `usage_missing` |", "`extra_body_overrides_protocol` |",
           "a reason the report prints is not in the README"),
    Mutant("E10", "readme", T + "README.md", "`test_the_pinned_commit_holds_the_code_these_tests_cover`", "`test_the_pinned_commit_is_part_of_this_checkout`",
           "the README names a test that does not exist"),
    Mutant("E11", "readme", T + "README.md", "mammoth-1 (#1 do board", "mammoth-2 (#1 do board",
           "the README no longer says that the board's #1 is outside the reference"),
    Mutant("E12", "readme", T + "README.md", "**Ela só viaja por `https://` ou para o próprio computador**", "**Ela viaja por onde for**",
           "the README says nothing about where the key may go"),
    Mutant("E13", "readme", T + "README.md", "`/kaggle/working` é a saída da versão salva.", "`/kaggle/working` é uma pasta.",
           "the README no longer warns that the saved output of a public notebook is public"),
)


def prepare(dst: Path) -> None:
    for rel in COPY:
        src = ROOT / rel
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, target)
    (dst / "tests" / "__init__.py").touch()


def test_env() -> Dict[str, str]:
    """The environment of the tests of a mutant: ours, minus the variables that name a git repository.  Run from a pre-push hook in a
    linked worktree this process has GIT_DIR set, and a test that runs git in a temporary directory would act on the real repository."""
    return {**C.without_local_git_vars(dict(os.environ)), "THINK_MUTATION_CHECK": "1"}


def run_tests(cwd: Path, suites: Sequence[Tuple[str, str]], timeout: int = 600) -> Tuple[bool, str, str]:
    """(all passed, name of the suite that failed, first failure line)."""
    for name, module in suites:
        r = subprocess.run([sys.executable, "-m", "unittest", "-f", module], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=test_env())
        if r.returncode != 0:
            text = r.stderr + r.stdout
            hit = re.search(r"^(?:FAIL|ERROR): (\S+) \((\S+)\)", text, re.M)
            first = f"{hit.group(2)}.{hit.group(1)}" if hit else text.strip().splitlines()[-1][:120]
            return False, name, first
    return True, "", ""


def check(mutants: Sequence[Mutant], log=print) -> Dict[str, str]:
    results: Dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        prepare(work)
        ok, name, first = run_tests(work, SUITES)
        if not ok:
            raise SystemExit(f"baseline is not green ({name}: {first}); fix the tests before mutating")
        log(f"baseline green ({' + '.join(m for _, m in SUITES)})")
        for m in mutants:
            path = work / m.file
            original = path.read_text(encoding="utf-8")
            if original.count(m.old) != 1:
                raise SystemExit(f"{m.id}: `old` occurs {original.count(m.old)} times in {m.file}, it must occur exactly once")
            t0 = time.time()
            try:
                path.write_text(original.replace(m.old, m.new), encoding="utf-8")
                for p in path.parent.glob("__pycache__"):
                    shutil.rmtree(p, ignore_errors=True)
                ok, suite, first = run_tests(work, SUITES)
            finally:
                path.write_text(original, encoding="utf-8")
            status = "SURVIVED" if ok else "killed"
            results[m.id] = status
            log(f"{m.id:3s} {status:8s} {m.file.split('/')[-1]:12s} {m.why}" + ("" if ok else f"  <- {suite}: {first}") + f"  [{time.time() - t0:.1f}s]")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m decide.think.mutation_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--group", action="append", help="only this group (repeatable)")
    ap.add_argument("--id", action="append", help="only this mutant (repeatable)")
    ap.add_argument("--list", action="store_true", help="list the mutants and exit")
    args = ap.parse_args(argv)
    chosen: List[Mutant] = [m for m in MUTANTS if (not args.group or m.group in args.group) and (not args.id or m.id in args.id)]
    if not chosen:
        print("no mutant matches", file=sys.stderr)
        return 2
    if args.list:
        for m in chosen:
            print(f"{m.id:3s} {m.group:10s} {m.file}: {m.why}")
        return 0
    res = check(chosen)
    survived = [k for k, v in res.items() if v == "SURVIVED"]
    print(f"{len(res) - len(survived)} of {len(res)} mutants killed" + (f"; SURVIVED: {', '.join(survived)}" if survived else ""))
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
