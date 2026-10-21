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
before the thinker's traces, G4 before the public round and the upload, and the scorer they call, ``decide/data/proxy_private.py``), what the children and the Hub can see (the token and the other credentials of the machine), what a
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
PREFLIGHT = 'decide/rental/preflight.py'
KIT = 'decide/rental/kit.py'
PROXY_EVAL = 'decide/rental/proxy_eval.py'
TRACES = 'decide/rental/traces.py'
PRIVATE = 'decide/data/proxy_private.py'

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
    Mutant('S02', STATE, 'if stored != dict(identity):', 'if False:', 'the state of another profile or config is accepted'),
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
    Mutant('R07', RUN, '        del env[name]', '        pass', 'the Hub token reaches the training child process'),
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
    Mutant('K04', KIT, 'spec = {"anchor": f"dir:{path.resolve()}", "name": model.name}', 'spec = {"anchor": str(path.resolve()), "name": model.name}', 'a checkpoint folder is named to the kit as a bare path, which the kit takes for a Hub repo that needs a commit'),
    Mutant('K05', KIT, 'spec = {"anchor": f"dir:{path.resolve()}", "name": model.name}', 'spec = {"anchor": f"dir:{path.resolve()}", "anchor_revision": "0" * 40, "name": model.name}', 'a revision is sent with a checkpoint folder, which the kit refuses'),
    Mutant('K06', KIT, 'spec = {"anchor": f"dir:{path.resolve()}", "name": model.name}', 'spec = {"anchor": f"dir:{path}", "name": model.name}', 'a relative checkpoint path is handed to the kit, which reads it from whatever the working folder is'),
    Mutant('K07', KIT, '        if self.factories:\n', '        if False:\n', 'the test hooks of the kit are not passed through, so the real-kit tests would try to fetch a processor'),
    Mutant('K08', KIT, 'budget=budget, resume=True))', 'budget=budget, resume=False))', 'a rerun after a crash starts the model over instead of resuming'),
    Mutant('E01', PROXY_EVAL, 'if scored["missing"] or scored["bad"] or scored["proxy"] is None:', 'if False:', 'a run with missing or malformed rows is scored as if it were whole'),
    Mutant('E02', PROXY_EVAL, 'return EXIT_OK if verdict["status"] == "PASS" else EXIT_STOPPED', 'return EXIT_OK', 'proxy_eval compare exits 0 when the gate fails'),
    Mutant('E03', PROXY_EVAL, 'if status != "complete":', 'if False:', 'proxy_eval scores a run the kit did not finish'),
    Mutant('E04', PROXY_EVAL, '([0-9a-f]{40})', '([0-9a-f]{4,40})', 'a commit written with fewer than 40 hex digits is accepted as a pin'),
    Mutant('E05', PROXY_EVAL, 'seed=args.seed, min_rows=args.min_rows)', 'seed=args.seed, min_rows=0)', 'proxy_eval compare (G2) passes a gain measured on a handful of rows'),
    Mutant('E06', PROXY_EVAL, 'type=_min_rows, default=G.DEFAULT_MIN_ROWS,', 'type=_min_rows, default=0,', 'proxy_eval compare has no floor unless someone types one'),
    Mutant('E07', PROXY_EVAL, '    if value < 1:\n        raise argparse.ArgumentTypeError(f"{text!r}: the floor', '    if value < 0:\n        raise argparse.ArgumentTypeError(f"{text!r}: the floor', '--min-rows 0 is accepted and turns the floor off'),
    Mutant('T01', TRACES, '        if _is_marker(line):', '        if False:', 'a line of the reasoning that looks like the end of the block is not neutralized'),
    Mutant('T02', TRACES, 'if finish not in ("stop", "length"):', 'if finish not in ("stop",):', 'a thinker answer cut by the budget is thrown away'),
    Mutant('T03', TRACES, 'elif finish == "length" or content.lstrip().startswith("<think>"):', 'elif False:', 'an unclosed thinking block is kept as the answer'),
    Mutant('T04', TRACES, 'request_sha256=S.request_digest(S.parse_row(row)[0])', 'request_sha256=""', 'traces cannot be tied to the request they answer'),
    Mutant('G18', GATES, 'if min_rows and sum(1 for _, _, gold, _ in requests if any(key is not None for key in gold.values())) < min_rows:', 'if False:', 'G4 passes on a handful of rows'),
    Mutant('G19', GATES, 'if expected_rows is not None and len(requests) != expected_rows:', 'if False:', 'G4 scores a part of the proxy that its manifest declares whole'),
    Mutant('G20', GATES, 'if min_rows and sum(1 for _, _, gold, _ in requests if any(key is not None for key in gold.values())) < min_rows:', 'if min_rows and len(requests) < min_rows:', 'rows without a gold answer count toward the minimum that G4 asks for'),
    Mutant('G21', GATES, 'DEFAULT_MIN_ROWS = 100\n', 'DEFAULT_MIN_ROWS = 50\n', 'the default floor of G2 drifts from the one of G4'),
    Mutant('D01', PRIVATE, 'if key in table and table[key] != values:', 'if False:', 'two different answers to one question are not noticed: the last one is scored, in silence'),
    Mutant('D02', PRIVATE, 'table[key] != values', 'table[key] == values', 'the same row twice (a run appended to itself) makes the run invalid, and two different ones do not'),
    Mutant('D03', PRIVATE, '            if (eid, qid) in conflicting:\n                bad += 1', '            if False:\n                bad += 1', 'score_rows scores the last of two different answers'),
    Mutant('D04', PRIVATE, '                bad += 1\n                clashes += 1', '                missing += 1\n                clashes += 1', 'a question answered twice with different numbers is reported as having no row'),
    Mutant('D05', PRIVATE, '                clashes += 1', '                clashes += 0', 'the verdict does not say how many questions were answered twice with different numbers'),
    Mutant('D06', GATES, '(eid, qid) in conflicting or probs is None', 'probs is None', 'the bootstrap scores the last of two different answers while score_rows refuses them (the two disagree)'),
    Mutant('D07', GATES, '"conflicting": scored["conflicting"],', '"conflicting": 0,', 'the verdict hides the questions answered twice with different numbers'),
    Mutant('D08', PROXY_EVAL, 'answered twice with different numbers)" if scored["conflicting"] else "")', 'answered twice with different numbers)" if False else "")', 'the score command says only "invalid" for a question answered twice with different numbers'),
    Mutant('V01', PRIVATE, 'return (1.0 / len(best) if gold in best else 0.0), len(best) > 1', 'return (1.0 if gold in best else 0.0), len(best) > 1', 'a tie for the top counts as a whole right answer when the gold option is one of the tied ones'),
    Mutant('V02', PRIVATE, 'return (1.0 / len(best) if gold in best else 0.0), len(best) > 1', 'return (1.0 / len(best) if gold == best[0] else 0.0), len(best) > 1', 'the first option of a tie is the one that counts (a uniform model is right whenever the gold option is the first)'),
    Mutant('V03', PRIVATE, 'return (1.0 / len(best) if gold in best else 0.0), len(best) > 1', 'return (1.0 / len(best)), len(best) > 1', 'a tie among options that do not include the gold one still earns credit'),
    Mutant('V04', PRIVATE, 'return (1.0 / len(best) if gold in best else 0.0), len(best) > 1', 'return (1.0 / len(best) if gold in best else 0.0), False', 'the score never says that answers had a shared top'),
    Mutant('V05', PRIVATE, '            ties += tied\n', '            ties += 0\n', 'the verdict does not count the answers with a shared top'),
    Mutant('V06', GATES, 'right, _ = PP.top_credit(probs, keys.index(key))', 'right = 1 if max(range(len(probs)), key=probs.__getitem__) == keys.index(key) else 0', 'the bootstrap uses the first-index rule while the scorer counts a tie as 1/m (the two disagree)'),
    Mutant('V07', GATES, '"ties": scored["ties"],', '"ties": 0,', 'the verdict view hides the answers with a shared top'),
    Mutant('F01', PREFLIGHT, 'if not (_number(margin) and margin > 0):', 'if False:', 'a G4 margin of 0 is accepted: a checkpoint that is no better passes'),
    Mutant('F02', PREFLIGHT, 'MIN_LABEL_COVERAGE = 0.9', 'MIN_LABEL_COVERAGE = 0.0', 'a label coverage of 1 percent is accepted as the F3 gate'),
    Mutant('F03', PREFLIGHT, 'if _same_repo(hub.get(first), hub.get(second)):', 'if False:', 'the backup of the day, with the private proxy, may go to the repo of the weights'),
    Mutant('F05', PREFLIGHT, 'first.lower() == second.lower()', 'first == second', 'ME/Weights and me/weights are taken for two repos, and the backup with the proxy goes to the repo of the weights'),
    Mutant('F04', PREFLIGHT, 'minimum >= 1):', 'minimum >= 0):', 'a floor of 0 questions is accepted for G4'),
    Mutant('H06', HUB, r'|(^|[_-])api[_-]?keys?([._-]|$)', '', 'apikey.txt is uploaded'),
    Mutant('H07', HUB, r'|(^|[_-])o?auth([._-]|$)', '', 'auth.json is uploaded'),
    Mutant('H08', HUB, r'|^\.(netrc|npmrc|pypirc|pgpass)$', '', '.netrc is uploaded'),
    Mutant('H09', HUB, r'|(^|[_-])keys?([._-]|$)', '', 'openai_key.txt and gcp-key.json are uploaded'),
    Mutant('H10', HUB, r'|(^|[_-])bearer([._-]|$)', '', 'bearer.txt is uploaded'),
    Mutant('H11', HUB, r'|service[_-]?account', '', 'service-account.json is uploaded'),
    Mutant('I04', INPUTS, 'if not (isinstance(leak, dict) and leak.get("checked") is True and leak.get("leaks") == 0):', 'if False:', 'a proxy whose second pass never ran against the clean corpus is scored in G4'),
    Mutant('I05', INPUTS, '    if shared:\n        problems.append(', '    if False:\n        problems.append(', 'a proxy that shares a source with the training corpus is scored in G4'),
    Mutant('I06', INPUTS, 'if not isinstance(rows, int) or not any(ref.get("requests") == rows for ref in references):', 'if False:', 'a corpus that decontaminate screened without the proxy is trained on'),
    Mutant('I07', RUN, 'if rows["with_gold"] < cfg["gates"]["g4_min_rows"]:', 'if False:', 'a proxy below the floor is found only after the training, at G4'),
    Mutant('I09', RUN, '\n                               expected_rows=IN.proxy_rows(self.paths.proxy)["declared"])', ')', 'G4 is not given the rows the manifest of the proxy declares'),
    Mutant('I10', RUN, 'if rows["declared"] != rows["read"]:', 'if False:', 'a proxy whose parts hold other rows than its manifest declares is scored'),
    Mutant('I08', RUN, 'seed=gates["bootstrap"]["seed"], min_rows=gates["g4_min_rows"],', 'seed=gates["bootstrap"]["seed"],', 'G4 is not given the floor of rows'),
    Mutant('S07', STATE, '    if ".." in Path(raw).parts:', '    if False:', 'a path of the state that climbs with ".." is accepted'),
    Mutant('S08', STATE, 'if "\\\\" in raw or "\\x00" in raw:', 'if False:', 'a path of the state with a backslash or a NUL character is accepted'),
    Mutant('S09', STATE, 'if re.match(r"^[A-Za-z]:", raw):', 'if False:', 'a path of the state that starts with a drive letter is accepted'),
    Mutant('S10', STATE, 'Path(os.path.realpath(target)).relative_to(Path(os.path.realpath(base)))', 'Path(os.path.realpath(target)).relative_to(Path(os.path.realpath(target)))', 'a path inside the work folder that leaves it through a symlink is accepted'),
    Mutant('S11', STATE, '    if not relative.parts:', '    if False:', 'the work folder itself is accepted as the place of an output'),
    Mutant('S12', STATE, 'if relative.parts[0] in RESERVED:', 'if False:', 'an output of the state may sit in the folder of the state itself or of the restore'),
    Mutant('S13', STATE, 'NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")', 'NAME_OK = re.compile(r"^.*$")', 'the name of an output may climb out of its folder in the work repo'),
    Mutant('S14', STATE, 'return fingerprint(path, full_limit=sys.maxsize)', 'return fingerprint(path)', 'the digest that ties G4 to the checkpoint does not read the middle of the big files'),
    Mutant('S15', STATE, 'built_on=json.loads(json.dumps(built_on or {})),', 'built_on={},', 'a phase does not remember what it was built on'),
    Mutant('R15', RUN, '            if problem:\n                raise DayError(f"{phase} cannot run now', '            if False:\n                raise DayError(f"{phase} cannot run now', '--only walks around every gate: a phase runs before what it needs'),
    Mutant('R16', RUN, '"train": ("fetch:start", "targets", "traces", "targets_traces"),', '"train": ("fetch:start", "traces", "targets_traces"),', 'the training does not need the targets that passed the label gate'),
    Mutant('R17', RUN, '"g4": ("fetch:proxy,start", "calibrate:final"),', '"g4": ("fetch:proxy,start",),', 'G4 does not need the calibrated checkpoint'),
    Mutant('R18', RUN, '"publish": ("g4", "public_round"),', '"publish": ("public_round",),', 'the upload does not need G4'),
    Mutant('R19', RUN, '"targets": ("fetch:corpus,labels",),', '"targets": (),', 'the targets are built before the data was fetched and sealed'),
    Mutant('R20', RUN, '        if status == "skipped" and dep in MAY_BE_SKIPPED:\n            pass', '        if status == "skipped":\n            pass', 'a phase that cannot be skipped counts as settled when it was'),
    Mutant('R21', RUN, '            if recorded != now:', '            if False:', 'a phase whose inputs changed since it finished is reused'),
    Mutant('R22', RUN, '        if detail.get("final_digest") != now:', '        if False:', 'the public round and the upload run on a checkpoint that is not the one G4 measured'),
    Mutant('R23', RUN, '        if detail.get("start_fingerprint") != self._start_fingerprint():', '        if False:', 'G4 stands after the start checkpoint its baseline was measured on changed'),
    Mutant('R24', RUN, 'self._digests[key] = ST.digest_tree(self.paths.final)', 'self._digests[key] = ST.fingerprint(self.paths.final)', 'G4 is tied to the edges of the big files, not to every byte'),
    Mutant('R25', RUN, '        if out.exists() and known.get(key) != identity:', '        if False:', 'the kit resumes a run from the results of another checkpoint'),
    Mutant('R26', RUN, 'identities = {"baseline": measured["baseline"], "new": measured["final_digest"]}', 'identities = {"baseline": measured["final_digest"], "new": measured["final_digest"]}', 'the baseline results are set aside whenever the new checkpoint changes (hours of kit time lost)'),
    Mutant('R27', RUN, 'verdict["measured"] = measured', 'verdict["measured"] = {}', 'the verdict does not say which bytes it measured'),
    Mutant('R28', RUN, '        problem = self._measured_problem()\n        if problem:\n            raise DayError(f"nothing is published that G4 did not measure', '        problem = self._measured_problem()\n        if False:\n            raise DayError(f"nothing is published that G4 did not measure', 'the upload itself does not look at the bytes (only the check before it does)'),
    Mutant('R29', RUN, '        problem = self._measured_problem()\n        if problem:\n            raise DayError(f"the public round needs G4 passed on this very checkpoint', '        problem = self._measured_problem()\n        if False:\n            raise DayError(f"the public round needs G4 passed on this very checkpoint', 'the public round itself does not look at the bytes (only the check before it does)'),
    Mutant('R30', RUN, 'fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)', 'fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)', 'two orchestrators work in the same folder'),
    Mutant('R31', RUN, '            with contextlib.suppress(OSError):\n                lock.unlink()', '            with contextlib.suppress(OSError):\n                pass', 'the lock file is left behind'),
    Mutant('R32', RUN, 'earlier = ST.parse_state(text, identity=self.identity, root=paths.workdir,', 'earlier = ST.parse_state(text, identity=self.identity, root=None,', 'the restore writes where the state tells it to, even outside the work folder'),
    Mutant('R33', RUN, '        shutil.rmtree(scratch, ignore_errors=True)                # nothing left over from a stopped restore is trusted', '        pass', 'files left in the scratch folder by an earlier stopped restore are moved into the day'),
    Mutant('R34', RUN, 'clock=self.clock, root=self.paths.workdir)', 'clock=self.clock)', 'a state file already on the disk is not checked for paths outside the work folder'),
    Mutant('R35', RUN, 'wanted = names.split(",") if names else sorted(outputs)', 'wanted = sorted(outputs)', 'a phase is built again when an output it does not read changed (or not when one it reads did)'),
    Mutant('R36', RUN, '        if why:\n            return f"{why}, so {dep} cannot be trusted"', '        if False:\n            return f"{why}, so {dep} cannot be trusted"', 'a phase runs on an output that changed on the disk after its gate'),
    Mutant('R37', RUN, '        stale = self._stale(dep)\n        if stale:\n            return f"{dep} has to be decided again first', '        stale = self._stale(dep)\n        if False:\n            return f"{dep} has to be decided again first', 'a phase runs before a phase that is due (traces for a GT report that passes now)'),
    Mutant('R38', RUN, '        self._digests = {}                        # a phase may have changed the bytes (calibrate does)', '        pass', 'the digest of the checkpoint is remembered across a phase that changes it'),
    Mutant('R39', RUN, '                same_file = os.fstat(handle).st_ino == os.stat(lock).st_ino', '                same_file = True', 'a lock file removed by the previous holder while we opened it is used',
           equivalent='the race needs two processes to open the lock inside the microseconds between the holder unlinking it and releasing it; a test cannot place a process there without hooks in os.open.  The check is the standard flock-then-compare-inodes guard'),
    Mutant('R40', RUN, '    for name in dropped_names(cfg, env):\n        del env[name]', '    for name in []:\n        del env[name]', 'the children get the Hub token, the thinker key and every credential of the machine'),
    Mutant('R41', RUN, 'if name in configured or (name not in KEEP_NAMES and CREDENTIAL_NAME.search(name)))', 'if name in configured)', 'only the two configured names are left out: OPENAI_API_KEY, AWS_* and the rest reach the children'),
    Mutant('R42', RUN, 'if name in configured or (name not in KEEP_NAMES and CREDENTIAL_NAME.search(name)))', 'if (name not in KEEP_NAMES and CREDENTIAL_NAME.search(name)))', 'a configured name that does not look like a credential (token_env: HUB_LOGIN) reaches the children'),
    Mutant('R43', RUN, '(name not in KEEP_NAMES and CREDENTIAL_NAME.search(name))', '(CREDENTIAL_NAME.search(name))', 'HF_HUB_DISABLE_IMPLICIT_TOKEN is dropped, so the child may read a token file again'),
    Mutant('R44', RUN, '^(?:aws|azure)_|', '', 'AWS_PROFILE and AZURE_TENANT_ID reach the children'),
    Mutant('R45', RUN, '^mysql_pwd$", re.IGNORECASE)', '^mysql_pwd$")', 'db_password and other lower-case names reach the children'),
    Mutant('R46', RUN, 'bearer)(?:_|$)"', 'bearer)"', 'a name that only starts with a credential word (TOKENIZERS_PARALLELISM) is dropped'),
    Mutant('R47', RUN, 'r"(?:^|_)(?:api_?keys?', 'r"(?:api_?keys?', 'a name that only ends in a credential word (MONKEY_PATCH) is dropped'),
    Mutant('R48', RUN, '            self._env_noted = True\n', '            pass\n', 'the day log repeats the note before every step'),
    Mutant('R49', RUN, "({', '.join(left_out)})", "()", 'the day log says that variables were left out but not which'),
    Mutant('R50', RUN, '|^git_config_|', '|^git_config_key_|', 'GIT_CONFIG_COUNT and GIT_CONFIG_VALUE_n reach the children (their git fails, the value can be an Authorization header)'),
    Mutant('R51', RUN, '|^pgpass|^mysql_pwd$"', '|^mysql_pwd$"', 'PGPASS and PGPASSFILE reach the children'),
    Mutant('R52', RUN, '|^mysql_pwd$"', '"', 'MYSQL_PWD reaches the children'),
    Mutant('R53', RUN, 'r"|(?:token|secret|passw(?:or)?d)s?(?:_?\\d+)?$"', 'r""', 'a name with the credential word glued at the end (MYTOKEN, ROOTPASSWORD) reaches the children'),
    Mutant('R54', RUN, 's?(?:_?\\d+)?$"', 's?$"', 'TOKEN2 and SLACKTOKEN_2 reach the children'),
    Mutant('R55', RUN, 'passw(?:or)?d)s?(?:_?', 'passw(?:or)?d)(?:_?', 'MYTOKENS reaches the children'),
    Mutant('R56', RUN, '|^mysql_pwd$"', '|pwd$"', 'PWD and OLDPWD, the working directory, are dropped'),
    Mutant('R57', RUN, '|^git_config_|', '|^git_config|', 'GIT_CONFIGURED_BY, which is no credential, is dropped'),
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
