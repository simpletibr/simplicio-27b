"""The mutation list of F6 (``decide/rental/mutation_check.py``) still applies to the code.

The mutants only mean something while their ``old`` text is in the file exactly once: the day a refactor moves a line, this test fails and
the list is updated (a mutant that no longer applies would be counted as "not killed" at best, and read as killed at worst).  The runner
itself is the one of F4 and is tested by ``tests/test_mutation_anchors_69.py``.
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest

from decide_b_support import ROOT

sys.path.insert(0, str(ROOT))

from decide.rental import mutation_check as MC  # noqa: E402


class MutantsApply(unittest.TestCase):
    def test_every_mutant_applies_to_the_working_tree(self) -> None:
        self.assertEqual(MC.check_anchors(MC.MUTANTS), [])

    def test_ids_are_unique_and_every_mutant_says_what_it_breaks(self) -> None:
        ids = [m.id for m in MC.MUTANTS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 60)
        for m in MC.MUTANTS:
            self.assertTrue(m.why.strip(), m.id)
            self.assertTrue(m.file.startswith("decide/rental/"), f"{m.id}: the mutants of F6 are in decide/rental")
            self.assertEqual(m.tests, ("test_contract_71.py", "test_flow_71.py"), m.id)

    def test_the_mutants_cover_every_module_of_the_day_that_decides_something(self) -> None:
        files = {m.file for m in MC.MUTANTS}
        for name in ("gates", "state", "plan", "run", "hub", "inputs", "kit", "proxy_eval", "traces"):
            self.assertIn(f"decide/rental/{name}.py", files)

    def test_only_the_unreachable_mutants_are_marked_equivalent_and_say_why(self) -> None:
        """G12: the two runs of G4 have been scored by the same rules on the same requests before the line is reached.  R39: the inode check of the
        lock guards a race between two processes that a test cannot place."""
        self.assertEqual([m.id for m in MC.MUTANTS if m.equivalent], ["G12", "R39"])
        for m in MC.MUTANTS:
            if m.equivalent:
                self.assertGreater(len(m.equivalent), 60, f"{m.id}: say why no test can tell it from the original")

    def test_the_command_line_lists_checks_and_refuses_an_unknown_id(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(MC.main(["--list"]), 0)
            self.assertEqual(MC.main(["--anchors"]), 0)
        listing = out.getvalue()
        for m in MC.MUTANTS:
            self.assertIn(m.id, listing)
        self.assertIn(f"{len(MC.MUTANTS)} mutants apply", listing)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            MC.main(["--ids", "ZZ99"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
