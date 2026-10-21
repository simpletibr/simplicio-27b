"""The mutation list of F4 (``decide/train/mutation_check.py``) still applies to the code, and the runner tells a killed mutant from a survivor.

The mutants only mean something while their ``old`` text is in the file exactly once: the day a refactor moves a line, this test fails
and the list is updated (a mutant that no longer applies would be silently counted as killed by a hasty reading of the log).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from decide_b_support import ROOT

sys.path.insert(0, str(ROOT))

from decide.train import mutation_check as MC  # noqa: E402


class MutantsApply(unittest.TestCase):
    def test_every_mutant_applies_to_the_working_tree(self) -> None:
        self.assertEqual(MC.check_anchors(MC.MUTANTS), [])

    def test_ids_are_unique_and_every_mutant_says_what_it_breaks(self) -> None:
        ids = [m.id for m in MC.MUTANTS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 140)
        for m in MC.MUTANTS:
            self.assertTrue(m.why.strip(), m.id)
            self.assertTrue(m.tests, m.id)

    def test_only_the_two_unreachable_mutants_are_marked_equivalent(self) -> None:
        """A19 (the manifest is demanded twice, and the first time comes first) and G05 (the boundary of the warmup gives 1.0 on both sides)."""
        self.assertEqual([m.id for m in MC.MUTANTS if m.equivalent], ["A19", "G05"])
        for m in MC.MUTANTS:
            if m.equivalent:
                self.assertGreater(len(m.equivalent), 60, f"{m.id}: say why no test can tell it from the original")

    def test_the_check_notices_an_anchor_that_moved_or_is_ambiguous(self) -> None:
        missing = MC.Mutant("Z1", "decide/train/loss.py", "text that is not in the file", "x", "why")
        twice = MC.Mutant("Z2", "decide/train/loss.py", "import", "x", "why")
        same = MC.Mutant("Z3", "decide/train/loss.py", "torch", "torch", "why")
        nofile = MC.Mutant("Z4", "decide/train/nothing.py", "a", "b", "why")
        notest = MC.Mutant("Z5", "decide/train/loss.py", "torch.nn", "torch.no", "why", tests=("test_nothing_69.py",))
        problems = MC.check_anchors([missing, twice, same, nofile, notest])
        self.assertEqual(len(problems), 5, problems)
        self.assertIn("occurs 0 times", problems[0])
        self.assertIn("not once", problems[1])
        self.assertIn("same text", problems[2])
        self.assertIn("does not exist", problems[3])
        self.assertIn("test_nothing_69.py", problems[4])


class RunnerTellsKilledFromSurvived(unittest.TestCase):
    """On a two-file repository: one mutant a test notices, one that changes nothing the test looks at, one that does not apply."""

    def test_verdicts_and_the_original_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "pkg").mkdir()
            (repo / "tests").mkdir()
            (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            source = "def value():\n    return 1\n\n\ndef unused():\n    return 2\n"
            (repo / "pkg" / "mod.py").write_text(source, encoding="utf-8")
            (repo / "tests" / "test_tiny.py").write_text("import unittest\nfrom pkg import mod\n\n\nclass T(unittest.TestCase):\n    def test_value(self):\n"
                                                         "        self.assertEqual(mod.value(), 1)\n", encoding="utf-8")
            tests = ("test_tiny.py",)
            mutants = [MC.Mutant("K", "pkg/mod.py", "return 1", "return 3", "the value", tests=tests),
                       MC.Mutant("S", "pkg/mod.py", "return 2", "return 4", "nobody looks", tests=tests),
                       MC.Mutant("E", "pkg/mod.py", "return 2", "return 4", "nobody looks, and says so", tests=tests, equivalent="unused"),
                       MC.Mutant("N", "pkg/mod.py", "return 99", "return 100", "not there", tests=tests)]
            results = {r.id: r for r in MC.run_mutants(mutants, repo)}
            self.assertEqual({k: v.verdict for k, v in results.items()}, {"K": "KILLED", "S": "SURVIVED", "E": "SURVIVED", "N": "NOT APPLIED"})
            self.assertEqual(results["K"].by, "T.test_value", "the class and the test, each once")
            self.assertEqual((repo / "pkg" / "mod.py").read_text(encoding="utf-8"), source, "the repository itself is never mutated")
            totals = MC.summarize(list(results.values()))
            self.assertEqual(totals, {"mutants": 4, "killed": 1, "survived_equivalent": 1, "survived": 1, "not_applied": 1})


class FirstFailureNamesTheTest(unittest.TestCase):
    def test_python_3_11_and_later_print_the_test_name_inside_the_parentheses_too(self) -> None:
        output = "FAIL: test_a (tests.test_mod.Klass.test_a)\n----\nERROR: test_b (tests.test_mod.Klass.test_b)\n"
        self.assertEqual(MC.first_failure(output), "Klass.test_a")

    def test_the_older_format_prints_the_class_only(self) -> None:
        self.assertEqual(MC.first_failure("ERROR: test_a (tests.test_mod.Klass)\n"), "Klass.test_a")

    def test_a_module_level_failure_and_no_failure(self) -> None:
        self.assertEqual(MC.first_failure("ERROR: setUpClass (tests.test_mod.Klass)\n"), "Klass.setUpClass")
        self.assertEqual(MC.first_failure("ERROR: test_mod (unittest.loader._FailedTest.test_mod)\n"), "_FailedTest.test_mod")
        self.assertEqual(MC.first_failure("Ran 3 tests\n\nOK\n"), "")


if __name__ == "__main__":
    unittest.main()
