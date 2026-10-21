"""Mutation check: plant a one-line bug in a COPY of the kit, run the tests, expect them to fail.

    python -m decide.think.mutation_check                 # every mutant
    python -m decide.think.mutation_check --group answers # one group (answers, bootstrap, verdict, scoring, resume, sample, trace, serve, report, provenance, hosted, notebook, readme)
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
    Mutant("L4", "report", T + "compare.py", "return most / len(chunk) > 0.4 or readable / len(chunk) < 0.4", "return most / len(chunk) > 0.4",
           "a row of symbol noise with no repeated character reads as language"),
    Mutant("L5", "report", T + "compare.py", 'chunk = "".join(text[:window].split())', 'chunk = "".join(text.split())',
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
    Mutant("X1", "report", T + "thinker.py", "    refuse_secrets(obj, what)\n    return obj", "    return obj",
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
    Mutant("N7", "notebook", "tests/test_flow_75.py", '"GIT_DIR", "GIT_WORK_TREE", "GIT_IMPLICIT_WORK_TREE",', '"GIT_WORK_TREE", "GIT_IMPLICIT_WORK_TREE",',
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


def run_tests(cwd: Path, suites: Sequence[Tuple[str, str]], timeout: int = 600) -> Tuple[bool, str, str]:
    """(all passed, name of the suite that failed, first failure line)."""
    for name, module in suites:
        r = subprocess.run([sys.executable, "-m", "unittest", module], cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "THINK_MUTATION_CHECK": "1"})
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
