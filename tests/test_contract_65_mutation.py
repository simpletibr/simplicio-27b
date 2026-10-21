"""Contract tests for decide/sd/mutation.py (issue #65, F0): the check that a deliberate one-line bug makes a test fail.

The check itself must not lie.  These tests do not run the real mutations (that takes minutes and lives in the pull request as
evidence: `python -m decide.sd.mutation`); they hold the machinery: the table fits the code, a survivor is reported as a survivor, a red or
skipping baseline is "cannot judge" instead of a verdict, the working tree is never touched, and the unittest output is read correctly.
The end-to-end tests use a tiny synthetic repository in a temporary folder, with the real runner and a real subprocess.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from decide_support import ROOT

from decide.sd import mutation as mu

REAL = {m.id: m for m in mu.MUTATIONS}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def make_repo(root: Path, guarded: bool = True, skipping: bool = False) -> None:
    """decide/sd/calc.py plus a test that guards `add` and `scale`; with guarded=False nothing tests `scale`."""
    write(root / "decide" / "__init__.py", "")
    write(root / "decide" / "sd" / "__init__.py", "")
    write(root / "decide" / "sd" / "calc.py", """
        def add(a, b):
            return a + b


        def scale(x, k):
            return x * k
        """)
    lines = [
        "import sys",
        "import unittest",
        "from pathlib import Path",
        "",
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))",
        "from decide.sd import calc",
        "",
        "",
        "class CalcTests(unittest.TestCase):",
    ]
    if skipping:
        lines.append("    @unittest.skip('no gpu here')")
    lines += ["    def test_values(self):", "        self.assertEqual(calc.add(2, 3), 5)"]
    if guarded:
        lines.append("        self.assertEqual(calc.scale(3, 4), 12)")
    write(root / "tests" / "test_calc.py", "\n".join(lines) + "\n")
    write(root / "notebooks" / "decide" / "requirements.txt", "# none\n")


ADD_BUG = mu.Mutation("merge", "add-becomes-sub", "decide/sd/calc.py", "return a + b", "return a - b", ("test_calc.CalcTests",), "wrong operator")
SCALE_BUG = mu.Mutation("merge", "scale-off-by-one", "decide/sd/calc.py", "return x * k", "return x * (k + 1)", ("test_calc.CalcTests",), "off by one")


class TheTableFitsTheCode(unittest.TestCase):
    def test_every_mutation_points_at_exactly_one_line_that_exists_once_and_still_compiles(self):
        self.assertEqual(mu.table_problems(ROOT), [])

    def test_ids_are_unique_and_every_group_is_known_and_has_mutations(self):
        ids = [m.id for m in mu.MUTATIONS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(mu.groups()), set(mu.GROUPS))
        for m in mu.MUTATIONS:
            self.assertTrue(m.why)
            self.assertTrue(m.tests)

    def test_the_three_things_the_squad_must_prove_are_covered_by_enough_mutations(self):
        by_group = {g: [m for m in mu.MUTATIONS if m.group == g] for g in mu.GROUPS}
        self.assertGreaterEqual(len(by_group["merge"]), 7)             # task arithmetic, TIES, trim, LoRA scale, LoRA weight, base, heads
        self.assertGreaterEqual(len(by_group["tta"]), 6)               # the inverse mapping above all
        self.assertGreaterEqual(len(by_group["calibrate"]), 6)         # the temperature fit and its split
        ids = set(REAL)
        for needed in ("unpermute-uses-forward-map", "invert-uses-forward-map", "restore-uses-forward-map", "engine-skips-restoration",
                       "ta-sign-flip", "lora-scale-wrong", "temperature-multiplies", "golden-section-wrong-branch", "holdout-leaks-into-fit"):
            self.assertIn(needed, ids)

    def test_the_mutated_files_are_the_modules_the_kit_stands_on(self):
        files = {m.file for m in mu.MUTATIONS}
        self.assertEqual(files, {"decide/sd/merge.py", "decide/sd/tta.py", "decide/sd/calibrate.py", "decide/sd/engine.py", "decide/sd/loader.py",
                                 "decide/sd/config.py", "decide/sd/api.py", "decide/sd/remote.py", "decide/sd/budget.py",
                                 "decide/sd/evaluate.py", "decide/sd/state.py"})

    def test_the_tests_named_in_the_table_exist(self):
        for m in mu.MUTATIONS:
            for test_id in m.tests:
                module, _, cls = test_id.partition(".")
                source = (ROOT / "tests" / f"{module}.py").read_text(encoding="utf-8")
                with self.subTest(f"{m.id}: {test_id}"):
                    self.assertIsNotNone(re.search(rf"^class {cls}\b", source, re.M), test_id)

    def test_a_stale_table_is_reported_with_the_reason(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        make_repo(root)
        cases = {
            "missing text": mu.Mutation("merge", "a", "decide/sd/calc.py", "return a * b", "return a", ("t",), "x"),
            "twice": mu.Mutation("merge", "b", "decide/sd/calc.py", "return", "return 0 +", ("t",), "x"),
            "no-op": mu.Mutation("merge", "c", "decide/sd/calc.py", "return a + b", "return a + b", ("t",), "x"),
            "two lines": mu.Mutation("merge", "d", "decide/sd/calc.py", "return a + b", "x = 1\n    return a + b", ("t",), "x"),
            "missing file": mu.Mutation("merge", "e", "decide/sd/nope.py", "x", "y", ("t",), "x"),
            "syntax error": mu.Mutation("merge", "f", "decide/sd/calc.py", "return a + b", "return a +", ("t",), "x"),
            "unknown group": mu.Mutation("other", "g", "decide/sd/calc.py", "return a + b", "return a", ("t",), "x"),
            "no tests": mu.Mutation("merge", "h", "decide/sd/calc.py", "return a + b", "return a", (), "x"),
        }
        for name, mutation in cases.items():
            with self.subTest(name):
                self.assertEqual(len(mu.table_problems(root, [mutation])), 1, mu.table_problems(root, [mutation]))
        self.assertIn("used twice", "\n".join(mu.table_problems(root, [ADD_BUG, ADD_BUG])))


class Helpers(unittest.TestCase):
    def test_apply_replaces_exactly_one_occurrence_or_refuses(self):
        self.assertEqual(mu.apply("a = 1\nb = 2\n", mu.Mutation("merge", "x", "f", "b = 2", "b = 3", ("t",), "w")), "a = 1\nb = 3\n")
        for old in ("a", "zzz"):
            with self.assertRaises(ValueError):
                mu.apply("a = 1\nb = 2 # a\n", mu.Mutation("merge", "x", "f", old, "q", ("t",), "w"))

    def test_one_line_changed_counts_changed_lines_not_characters(self):
        self.assertTrue(mu.one_line_changed("a\nb\nc\n", "a\nB\nc\n"))
        self.assertFalse(mu.one_line_changed("a\nb\nc\n", "a\nb\nc\n"))
        self.assertFalse(mu.one_line_changed("a\nb\nc\n", "A\nB\nc\n"))
        self.assertFalse(mu.one_line_changed("a\nb\n", "a\nb\nc\n"))

    def test_unittest_output_is_read_into_numbers_and_failed_ids(self):
        green = "..ss\n----------------------------------------------------------------------\nRan 4 tests in 0.1s\n\nOK (skipped=2)\n"
        out = mu.parse_unittest(green, 0, 1.5)
        self.assertEqual((out.ran, out.skipped, out.failed, out.tail, out.green), (4, 2, [], "OK (skipped=2)", True))
        red = ("FAIL: test_a (mod.Cls.test_a)\n---\nTraceback\nERROR: test_b (mod.Cls.test_b)\n---\n\n"
               "Ran 9 tests in 0.2s\n\nFAILED (failures=1, errors=1)\n")
        out = mu.parse_unittest(red, 1)
        self.assertEqual((out.ran, out.skipped, out.failed, out.green), (9, 0, ["test_a", "test_b"], False))
        self.assertEqual(mu.parse_unittest("", 1).ran, 0)

    def test_the_copy_holds_the_kit_the_tests_and_the_requirements_and_nothing_else(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src, dst = Path(tmp.name) / "src", Path(tmp.name) / "dst"
        make_repo(src)
        write(src / "README.md", "untouched\n")
        write(src / "decide" / "sd" / "__pycache__" / "x.pyc", "junk")
        mu.copy_repo(src, dst)
        self.assertTrue((dst / "decide" / "sd" / "calc.py").is_file())
        self.assertTrue((dst / "tests" / "test_calc.py").is_file())
        self.assertTrue((dst / "notebooks" / "decide" / "requirements.txt").is_file())
        self.assertFalse((dst / "README.md").exists())
        self.assertFalse((dst / "decide" / "sd" / "__pycache__").exists())


class EndToEnd(unittest.TestCase):
    """The real runner, a real subprocess, a synthetic repository."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "repo"
        self.lines = []

    def go(self, mutations, **kw):
        return mu.run(mutations, self.root, sys.executable, 120.0, self.lines.append, **kw)

    def digest(self):
        return {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.root.rglob("*")) if p.is_file()}

    def test_a_bug_a_test_notices_is_killed_and_one_nobody_tests_survives(self):
        make_repo(self.root, guarded=False)
        before = self.digest()
        report = self.go([ADD_BUG, SCALE_BUG])
        status = {r.mutation.id: r.status for r in report["results"]}
        self.assertEqual(status, {"add-becomes-sub": "KILLED", "scale-off-by-one": "SURVIVED"})
        self.assertEqual((report["killed"], report["survivors"]), (1, ["scale-off-by-one"]))
        self.assertIn("every one of 1 tests passed with the bug in place", [r for r in report["results"] if r.status == "SURVIVED"][0].detail)
        self.assertEqual(self.digest(), before)                                     # the working tree is byte-identical, pycache included

    def test_once_the_missing_test_is_added_the_same_bug_is_killed(self):
        make_repo(self.root, guarded=True)
        report = self.go([ADD_BUG, SCALE_BUG])
        self.assertEqual(report["survivors"], [])
        self.assertEqual(report["killed"], 2)
        self.assertEqual([r.killed_by for r in report["results"]], [["test_values"], ["test_values"]])

    def test_a_red_baseline_is_cannot_judge_not_a_wall_of_kills(self):
        make_repo(self.root)
        write(self.root / "decide" / "sd" / "calc.py", "def add(a, b):\n    return a + b + 1\n\n\ndef scale(x, k):\n    return x * k\n")
        with self.assertRaises(mu.CannotJudge) as ctx:
            self.go([SCALE_BUG])
        self.assertIn("not green before the mutation", str(ctx.exception))

    def test_a_skipping_baseline_is_cannot_judge_because_a_skipped_test_kills_nothing(self):
        make_repo(self.root, skipping=True)
        with self.assertRaises(mu.CannotJudge) as ctx:
            self.go([ADD_BUG])
        self.assertIn("skipped in this interpreter", str(ctx.exception))

    def test_a_stale_table_stops_before_any_test_runs(self):
        make_repo(self.root)
        stale = mu.Mutation("merge", "stale", "decide/sd/calc.py", "return a / b", "return a", ("test_calc.CalcTests",), "x")
        with self.assertRaises(mu.CannotJudge) as ctx:
            self.go([stale])
        self.assertIn("does not fit the code", str(ctx.exception))
        self.assertEqual(self.lines, [])

    def test_a_working_tree_that_changes_during_the_check_is_reported(self):
        make_repo(self.root)
        target = self.root / "decide" / "sd" / "calc.py"

        def meddling(work, tests, python, timeout):
            target.write_text(target.read_text(encoding="utf-8") + "# edited while the check ran\n", encoding="utf-8")
            return mu.run_tests(work, tests, python, timeout)

        with self.assertRaises(mu.CannotJudge) as ctx:
            self.go([ADD_BUG], runner=meddling)
        self.assertIn("working tree changed", str(ctx.exception))

    def test_a_test_run_that_hangs_counts_as_red_with_the_reason(self):
        make_repo(self.root)
        write(self.root / "tests" / "test_calc.py", """
            import time
            import unittest


            class CalcTests(unittest.TestCase):
                def test_values(self):
                    time.sleep(30)
            """)
        outcome = mu.run_tests(self.root, ("test_calc.CalcTests",), sys.executable, 1.0)
        self.assertFalse(outcome.green)
        self.assertIn("timed out", outcome.tail)

    def test_the_command_line_exit_codes_and_the_evidence_file(self):
        make_repo(self.root, guarded=False)
        out = Path(self._tmp.name) / "evidence.json"
        report = self.go([ADD_BUG, SCALE_BUG])
        data = mu.evidence(report, sys.executable)
        self.assertEqual((data["mutations"], data["killed"], data["survivors"], data["tree_untouched"]), (2, 1, ["scale-off-by-one"], True))
        self.assertEqual([r["status"] for r in data["results"]], ["KILLED", "SURVIVED"])
        self.assertIn("decide/sd/calc.py", data["sha256_of_mutated_files"])
        out.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["survivors"], ["scale-off-by-one"])

    def test_the_list_option_prints_every_mutation_with_its_reason_and_runs_nothing(self):
        lines = []
        self.assertEqual(mu.main(["--list"], say=lines.append), 0)
        text = "\n".join(lines)
        self.assertEqual(sum(1 for m in mu.MUTATIONS if m.id in text), len(mu.MUTATIONS))
        self.assertIn(REAL["unpermute-uses-forward-map"].why, text)
        lines.clear()
        self.assertEqual(mu.main(["--list", "--id", "k-bucket-boundary", "--id", "ta-sign-flip"], say=lines.append), 0)
        shown = [ln for ln in lines if re.match(r"\w+\s+\S+\s+decide/", ln)]
        self.assertEqual(sorted(ln.split()[1] for ln in shown), ["k-bucket-boundary", "ta-sign-flip"])
        with self.assertRaises(SystemExit):
            mu.main(["--list", "--id", "no-such-mutation"], say=lines.append)
        lines.clear()
        self.assertEqual(mu.main(["--list", "--only", "tta"], say=lines.append), 0)
        self.assertNotIn("ta-sign-flip", "\n".join(lines))
        self.assertIn("unpermute-uses-forward-map", "\n".join(lines))


class NothingBecomesPartOfTheTree(unittest.TestCase):
    def test_the_module_never_writes_into_the_repository_it_checks(self):
        source = (ROOT / "decide" / "sd" / "mutation.py").read_text(encoding="utf-8")
        receivers = sorted(set(re.findall(r"([\w.()]+)\.write_text\(", source)))
        # the file being mutated, which is always inside the temporary copy, and the evidence file whose path the caller names
        self.assertEqual(receivers, ["Path(args.json)", "target"])
        self.assertIn("work = Path(tempfile.mkdtemp(", source)
        self.assertIn("target = work / m.file", source)
        self.assertIn("shutil.rmtree", source)

    def test_it_carries_no_token_and_does_not_pass_one_to_the_tests(self):
        source = (ROOT / "decide" / "sd" / "mutation.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{10,}", source))
        self.assertIn('env.pop(name, None)', source)


if __name__ == "__main__":
    unittest.main()
