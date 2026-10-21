"""Mutation check of ``decide.rental`` (issue #71, F6): does the test suite notice a deliberate one-line bug?

    python -m decide.rental.mutation_check --list                   # the mutants, one per line
    python -m decide.rental.mutation_check                          # all of them, on a private copy of the working tree
    python -m decide.rental.mutation_check --ids G05 R03            # some of them
    python -m decide.rental.mutation_check --json result.json       # also write the result

The same runner as ``decide.train.mutation_check`` (F4): a mutant is one exact text of one source file replaced by another, put in a
private copy of the repository, and the tests that should notice it (``tests/test_contract_71.py`` and ``tests/test_flow_71.py``) run
in their own ``unittest`` processes, stopping at the first failing test.  KILLED when a test fails, SURVIVED when all pass.  The working
tree is never modified.  Exit status 0 when every mutant that is not marked equivalent was killed.

``tests/test_mutation_anchors_71.py`` runs in ``make check`` and fails when a mutant no longer applies (its ``old`` text moved or
changed).  Running the mutants takes a long time (each one runs the contract tests and, when those pass, the flow tests, which train a
tiny model); the measured time is in the pull request of F6.

The mutants sit where a wrong line would cost money or leak something: the gates (label coverage before any GPU hour, the GT report
before the thinker's traces, G4 before the public round and the upload), what the children and the Hub can see (the token), what a
rerun trusts (the state and the fingerprints of the outputs), the order and the arguments of the commands of the day, and the checks that
the data was screened (decontaminated, sealed, tied to the request it answers).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from decide.train.mutation_check import Mutant as _Mutant
from decide.train.mutation_check import Result, check_anchors, run_mutants, summarize

ROOT = Path(__file__).resolve().parents[2]
#: The test files of F6, cheapest first.
DEFAULT_TESTS: Tuple[str, ...] = ("test_contract_71.py", "test_flow_71.py")


def Mutant(id: str, file: str, old: str, new: str, why: str = "", equivalent: str = "") -> _Mutant:      # noqa: N802 - reads like the class of F4
    """A mutant of F6: the same record as in F4, with the F6 test files as the default."""
    return _Mutant(id, file, old, new, why, DEFAULT_TESTS, equivalent)


# ---------------------------------------------------------------------------------------------------------------- the mutants

GATES = 'decide/rental/gates.py'
STATE = 'decide/rental/state.py'
PLAN = 'decide/rental/plan.py'
RUN = 'decide/rental/run.py'
HUB = 'decide/rental/hub.py'
INPUTS = 'decide/rental/inputs.py'
KIT = 'decide/rental/kit.py'
PROXY_EVAL = 'decide/rental/proxy_eval.py'
TRACES = 'decide/rental/traces.py'

MUTANTS: Tuple[Mutant, ...] = (
    Mutant('G01', GATES, 'passed = round(coverage, 12) >= minimum', 'passed = round(coverage, 12) > minimum', 'a corpus with exactly the minimum share of labeled rows is refused'),
    Mutant('G02', GATES, '"minimum": minimum, "passed": False, "rows_in": 0', '"minimum": minimum, "passed": True, "rows_in": 0', 'an empty corpus counts as fully labeled'),
    Mutant('G03', GATES, 'if json.loads(value))', 'if True)', 'rows that only have a gold answer count as labeled by a teacher'),
    Mutant('G04', GATES, 'if verdict.get("status") != "PASS":', 'if verdict.get("status") == "FAIL":', 'a GT report whose verdict is PARTIAL or INCONCLUSIVE turns the thinker on'),
    Mutant('G05', GATES, 'if coverage.get("complete") is not True:', 'if coverage.get("complete") is False:', 'a GT report whose coverage is not complete turns the thinker on'),
    Mutant('G06', GATES, 'if model not in (wanted, f"{wanted}-FP8"):', 'if False:', 'traces from a model other than the one the GT gate measured'),
    Mutant('G07', GATES, 'elif thinker.get("revision") and revision != thinker["revision"]:', 'elif False:', 'traces from a revision other than the one pinned in the config'),
    Mutant('G08', GATES, 'if (config.get("chat_template_kwargs") or {}).get("enable_thinking") is False:', 'if False:', 'a GT run with thinking off counts as the gate'),
    Mutant('G09', GATES, 'if round(delta, 9) >= margin_points:', 'if round(delta, 9) > margin_points:', 'a G4 gain of exactly the margin fails'),
    Mutant('G10', GATES, '        if scored["missing"]:\n            reasons.append(f"{label}_missing_rows")', '        if False:\n            reasons.append(f"{label}_missing_rows")', 'G4 scores a run that is missing rows'),
    Mutant('G11', GATES, '        if scored["bad"]:\n            reasons.append(f"{label}_bad_rows")', '        if False:\n            reasons.append(f"{label}_bad_rows")', 'G4 scores a run with malformed probabilities'),
    Mutant('G12', GATES, 'if sources != sorted(base_units) or any(keys[name] != sorted(base_units[name]) for name in sources):', 'if False:', 'G4 compares two runs that answered different questions',
           equivalent='both runs were scored by score_rows on the same requests and the verdict returns before this line unless every gold question of both has a well-formed answer; _units applies the same rules as score_rows, so the two sets of questions are the same by then.  The line guards a future change of _units or score_rows'),
    Mutant('G13', GATES, 'verdict["ci_above_zero"] = low > 0', 'verdict["ci_above_zero"] = high > 0', 'the interval flag looks at the wrong end of the interval'),
    Mutant('G14', GATES, 'base_side[name] = (sum(right_base[i] for i in picks), chance_sum, n)', 'base_side[name] = (sum(right_base), chance_sum, n)', 'the bootstrap draws the two sides independently, so the interval is far too wide'),
    Mutant('G15', GATES, 'request = {key: value for key, value in request.items() if key != "reasoning"}', 'request = dict(request)', "the thinker's notes reach the decider in G4"),
    Mutant('G16', GATES, 'if report.get("format") != GT_REPORT_FORMAT:', 'if False:', 'a GT report in another format is read as if it were this one'),
    Mutant('G17', GATES, 'GT_REPORT_FORMAT = "think-gt-report/2"', 'GT_REPORT_FORMAT = "think-gt-report/1"', 'the day reads a format name that decide.think.compare no longer writes'),
    Mutant('S01', STATE, 'if now != entry["fingerprint"]:', 'if False:', 'a phase whose output changed on disk is reused'),
    Mutant('S02', STATE, 'if stored != self.identity:', 'if False:', 'the state of another profile or config is accepted'),
    Mutant('S03', STATE, 'if status == "skipped":\n            return None', 'if False:\n            return None', 'the state does not treat a skipped phase as settled, so every rerun asks its question again'),
    Mutant('S04', STATE, 'digest.update(f"{size}|".encode("ascii"))', 'pass', 'two outputs of the same name and different size look the same'),
    Mutant('S05', STATE, 'if f.is_file() and not f.name.endswith(".tmp")', 'if f.is_file()', 'half-written files change the fingerprint of an output'),
    Mutant('S06', STATE, 'if not record or record.get("status") != "running":', 'if False:', 'a phase is finished that never began, or finished twice'),
    Mutant('P01', PLAN, 'if profile.gpus > 1:', 'if profile.gpus >= 1:', 'a single GPU is started through torchrun'),
    Mutant('P02', PLAN, '"--seed", str(seeds(profile)[index - 1])]', '"--seed", str(seeds(profile)[0])]', 'every seed of the average trains with the first seed'),
    Mutant('P03', PLAN, '    if resume:\n        argv.append("--resume")', '    if False:\n        argv.append("--resume")', 'a resumed run starts from the beginning'),
    Mutant('P04', PLAN, 'return paths.seed_dir(1) / "final" if seed_count == 1 else paths.soup', 'return paths.seed_dir(1) / "final" if seed_count > 1 else paths.soup', 'the checkpoint that gets calibrated is not the soup'),
    Mutant('P05', PLAN, 'elif item.suffix == WEIGHT_SUFFIX:', 'elif False:', 'the 50 GB of weights are copied, not linked, when the checkpoint is staged'),
    Mutant('P06', PLAN, 'if (dst / STAGED_MARK).is_file():\n            shutil.rmtree(dst)', 'if dst.is_dir():\n            shutil.rmtree(dst)', 'staging again deletes a folder the day did not make'),
    Mutant('P07', PLAN, '"--targets", str(targets), "--write",', '"--targets", str(targets),', 'the calibration is fitted but not written into the checkpoint'),
    Mutant('P08', PLAN, 'argv += ["--traces", str(paths.traces / thinker), "--thinker", thinker]', 'argv += ["--traces", str(paths.traces / thinker)]', "the thinker's name is not passed to build_targets"),
    Mutant('P09', PLAN, 'argv += ["--traces", str(paths.traces / thinker), "--thinker", thinker]', 'argv += ["--traces", str(paths.traces), "--thinker", thinker]', 'build_targets is given the folder of all thinkers, not the one it can read through a manifest'),
    Mutant('R01', RUN, 'if not coverage["passed"]:', 'if False:', 'the day trains on a corpus with too few labels'),
    Mutant('R02', RUN, 'if verdict["status"] != "PASS":', 'if False:', 'the day goes on after G4 failed'),
    Mutant('R03', RUN, 'if self.state.status("g4") != "done":\n            raise DayError("the public round needs G4 passed first")', 'if False:\n            raise DayError("the public round needs G4 passed first")', 'the public round runs without G4 having passed'),
    Mutant('R04', RUN, 'if self.state.status("g4") != "done":\n            raise DayError("nothing is published before G4 passes")', 'if False:\n            raise DayError("nothing is published before G4 passes")', 'weights are published without G4 having passed'),
    Mutant('R05', RUN, 'if info.get("status") != "complete":\n                raise DayError(f"the kit ended the {model.name} run', 'if False:\n                raise DayError(f"the kit ended the {model.name} run', 'G4 scores a run the kit did not finish'),
    Mutant('R06', RUN, 'if info.get("status") != "complete":\n            raise DayError(f"the kit ended the public round', 'if False:\n            raise DayError(f"the kit ended the public round', 'an unfinished public round is recorded as done'),
    Mutant('R07', RUN, '        env.pop(name, None)', '        pass', 'the Hub token reaches the training child process'),
    Mutant('R08', RUN, 'env["HF_HUB_OFFLINE"] = "1"', 'env["HF_HUB_OFFLINE"] = "0"', 'a child process may go online for a model or a dataset'),
    Mutant('R09', RUN, 'except Exception as exc:                  # noqa: BLE001 - a backup that fails', 'except KeyError as exc:                  # noqa: BLE001 - a backup that fails', 'a backup that fails stops the day'),
    Mutant('R10', RUN, 'if phase == "public_round" and status == "skipped" and self._public_requests() is not None:', 'if False:', 'the public requests file given on a second run is ignored: the round stays skipped'),
    Mutant('R11', RUN, 'if "public round" not in (state.record("publish").get("detail") or {}).get("uploaded", []):', 'if False:', 'the results of a public round that finished after the upload are never sent'),
    Mutant('R12', RUN, '            if late:\n                raise DayError(', '            if False:\n                raise DayError(', 'a passing GT report given after the model trained without traces is silently ignored'),
    Mutant('R13', RUN, 'if phase == "traces" and status == "skipped" and self._evidence()["enabled"]:', 'if False:', 'a passing GT report given on a second run is ignored: the traces stay skipped'),
    Mutant('R14', RUN, 'if phase == "targets_traces" and status == "skipped" and state.status("traces") == "done":', 'if False:', 'the targets with traces are not made after the traces were'),
    Mutant('H01', HUB, 'if not self.is_private(repo, kind=kind):\n            raise HubError(f"repo {repo} is public; nothing was uploaded")', 'if False:\n            raise HubError(f"repo {repo} is public; nothing was uploaded")', 'an upload goes to a repo that turned public'),
    Mutant('H02', HUB, 'if not self._api.repo_info(repo, repo_type=_kind(kind)).private:', 'if False:', 'the Hugging Face client uploads to a public repo'),
    Mutant('H03', HUB, 'if _SECRET.search(file.name):', 'if False:', 'files named like secrets are uploaded'),
    Mutant('H04', HUB, ') + (f" ({hint})" if hint else "")) from None', ') + (f" ({hint})" if hint else ""))', 'the original error, with request details, is printed next to the Hub error'),
    Mutant('H05', HUB, 'first = self._scrub(str(exc).strip().splitlines()[0]) if str(exc).strip() else ""', 'first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""', 'the token shows in the message of a Hub error'),
    Mutant('I01', RUN, 'facts["corpus"]["screened"] = IN.check_corpus_screened(paths.corpus)', 'facts["corpus"]["screened"] = {}', 'a corpus that did not come out of decontaminate is trained on'),
    Mutant('I02', RUN, 'name=key, manifest_required=True)', 'name=key, manifest_required=False)', 'a folder without a manifest.json is read by file name'),
    Mutant('I03', INPUTS, 'BT.corpus_provenance(path, allow_undecontaminated=False)', 'BT.corpus_provenance(path, allow_undecontaminated=True)', 'a corpus without decontamination.json is accepted'),
    Mutant('K01', KIT, 'if config.get("temperature_is_placeholder"):', 'if False:', 'a checkpoint with the placeholder temperature is loaded'),
    Mutant('K02', KIT, 'if self.repo is not None and not (isinstance(self.revision, str) and _REVISION.fullmatch(self.revision)):', 'if False:', 'a model from the Hub is loaded without a pinned revision'),
    Mutant('K03', KIT, 'if (self.path is None) == (self.repo is None):', 'if False:', 'a model given both as a folder and as a repo'),
    Mutant('E01', PROXY_EVAL, 'if scored["missing"] or scored["bad"] or scored["proxy"] is None:', 'if False:', 'a run with missing or malformed rows is scored as if it were whole'),
    Mutant('E02', PROXY_EVAL, 'return EXIT_OK if verdict["status"] == "PASS" else EXIT_STOPPED', 'return EXIT_OK', 'proxy_eval compare exits 0 when the gate fails'),
    Mutant('E03', PROXY_EVAL, 'if status != "complete":', 'if False:', 'proxy_eval scores a run the kit did not finish'),
    Mutant('E04', PROXY_EVAL, '([0-9a-f]{40})', '([0-9a-f]{4,40})', 'a commit written with fewer than 40 hex digits is accepted as a pin'),
    Mutant('T01', TRACES, '        if _is_marker(line):', '        if False:', 'a line of the reasoning that looks like the end of the block is not neutralized'),
    Mutant('T02', TRACES, 'if finish not in ("stop", "length"):', 'if finish not in ("stop",):', 'a thinker answer cut by the budget is thrown away'),
    Mutant('T03', TRACES, 'elif finish == "length" or content.lstrip().startswith("<think>"):', 'elif False:', 'an unclosed thinking block is kept as the answer'),
    Mutant('T04', TRACES, 'request_sha256=S.request_digest(S.parse_row(row)[0])', 'request_sha256=""', 'traces cannot be tied to the request they answer'),
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m decide.rental.mutation_check", description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="print the mutants and exit")
    parser.add_argument("--anchors", action="store_true", help="only check that every mutant applies to the working tree")
    parser.add_argument("--ids", nargs="+", help="run only these mutants")
    parser.add_argument("--repo", default=str(ROOT), help="the repository to copy (default: this one)")
    parser.add_argument("--json", help="also write the result to this file")
    args = parser.parse_args(argv)
    chosen = [m for m in MUTANTS if not args.ids or m.id in args.ids]
    unknown = sorted(set(args.ids or ()) - {m.id for m in MUTANTS})
    if unknown:
        parser.error(f"unknown mutant ids: {', '.join(unknown)}")
    if args.list:
        for m in chosen:
            print(f"{m.id:5} {m.file:34} {m.why}" + ("  [equivalent]" if m.equivalent else ""))
        return 0
    problems = check_anchors(chosen, Path(args.repo))
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    if args.anchors:
        print(f"{len(chosen)} mutants apply")
        return 0

    def show(result: Result) -> None:
        print(f"{result.id:5} {result.verdict:8} {result.seconds:6.0f}s  {result.by}" + (f"  (marked equivalent: {result.equivalent})" if result.equivalent else ""), flush=True)

    results = run_mutants(chosen, Path(args.repo), progress=show)
    totals = summarize(results)
    print(json.dumps(totals))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": totals, "results": [r.__dict__ for r in results]}, indent=1) + "\n", encoding="utf-8")
    return 0 if totals["survived"] == 0 and totals["not_applied"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
