"""Flow tests for the executable eval tasks (#26).

End to end: the harness loads the 40 tasks converted from data/unseen_eval_120.json, runs the
reference answer (`oracle.txt`) through the same SEARCH/REPLACE and pytest path a model answer
takes, and the statement a model sees never carries the reference answer.

The `--oracle` run needs the packages of benchmarks/harness/requirements.txt in the Python that
runs this file (the harness runs pytest with its own interpreter); without them that test is
skipped and says what to install.
"""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "benchmarks/harness/tasks"
sys.path.insert(0, str(ROOT / "benchmarks/harness"))
import decontam_tasks  # noqa: E402
import harness  # noqa: E402

MISSING = [m for m in ("pytest", "pydantic", "fastapi", "pandas", "sqlalchemy") if importlib.util.find_spec(m) is None]
NEEDS_DEPS = unittest.skipIf(
    MISSING,
    f"missing {', '.join(MISSING)}: python3 -m pip install -r benchmarks/harness/requirements.txt",
)


def eval_tasks():
    """The tasks converted from the eval file: the ones that record their `source_ids`."""
    return [t for t in harness.load_tasks(TASKS) if "source_ids" in t]


def added_lines(task):
    """Lines of the reference answer that are new, i.e. not part of what it searches for."""
    lines = set()
    oracle = (Path(task["dir"]) / "oracle.txt").read_text(encoding="utf-8")
    for search, replace in harness.parse_blocks(oracle):
        old = {line.strip() for line in search.splitlines()}
        lines |= {line.strip() for line in replace.splitlines() if line.strip() and line.strip() not in old}
    return lines


class OracleFlow(unittest.TestCase):
    @NEEDS_DEPS
    def test_oracle_passes_on_every_task_and_tests_fail_without_it(self):
        # run_oracle prints BAD when the tests pass on the untouched file, or when the
        # reference answer does not apply or does not pass them.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = harness.main(["--tasks", str(TASKS), "--oracle"])
        lines = out.getvalue().splitlines()
        self.assertEqual([line for line in lines if not line.startswith("OK ")], [], out.getvalue())
        self.assertEqual(code, 0)
        ok = {line[3:] for line in lines}
        self.assertEqual(len(eval_tasks()), 40)
        self.assertLessEqual({t["id"] for t in eval_tasks()}, ok)

    def test_an_empty_answer_fails_every_task_without_running_tests(self):
        for task in eval_tasks():
            with self.subTest(task=task["id"]):
                res = harness.evaluate(task, "no blocks here")
                self.assertFalse(res["passed"])
                self.assertEqual(res["apply_error"], "no_blocks")


class PromptHasNoAnswer(unittest.TestCase):
    def test_reference_answer_is_not_in_the_statement_or_the_code(self):
        for task in eval_tasks():
            with self.subTest(task=task["id"]):
                raw = (Path(task["dir"]) / "task.json").read_text(encoding="utf-8")
                prompt = harness.build_messages(task, "simplicio", False)[-1]["content"]
                oracle = (Path(task["dir"]) / "oracle.txt").read_text(encoding="utf-8")
                for search, replace in harness.parse_blocks(oracle):
                    self.assertNotIn(replace, prompt)
                    self.assertNotIn(replace, raw)
                for line in sorted(added_lines(task)):
                    self.assertNotIn(line, prompt, f"line of the reference answer is in the prompt: {line!r}")
                    self.assertNotIn(line, json.loads(raw)["instruction"])
                self.assertNotIn("TRAP", prompt)
                self.assertNotIn("<patch>", prompt)

    def test_old_prompt_is_kept_only_in_source_prompt(self):
        for task in eval_tasks():
            with self.subTest(task=task["id"]):
                prompt = harness.build_messages(task, "simplicio", False)[-1]["content"]
                self.assertTrue(prompt.endswith("Tarefa: " + task["instruction"]))
                self.assertTrue(prompt.startswith("Contexto: \nArquivo: " + task["edit_file"] + "\n"))
                if task["instruction"] != task["source_prompt"]:
                    self.assertNotIn(task["source_prompt"], prompt)


class DecontaminationFlow(unittest.TestCase):
    def test_committed_report_matches_a_fresh_run_and_has_no_overlap(self):
        committed = json.loads((TASKS / decontam_tasks.REPORT).read_text(encoding="utf-8"))
        fresh = decontam_tasks.build_report()
        self.assertEqual(committed, fresh)
        self.assertEqual([t["id"] for t in fresh["tasks"] if t["overlaps"]], [])
        self.assertLessEqual({t["id"] for t in eval_tasks()}, {t["id"] for t in fresh["tasks"]})

    def test_overlap_is_detected_when_a_task_copies_the_training_data(self):
        with decontam_tasks.corpus_files(decontam_tasks.DATA)[0].open(encoding="utf-8") as fh:
            first = json.loads(fh.readline())
        passage = next(s for s in decontam_tasks.strings(first) if len(decontam_tasks.tokens(s)) >= 20)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "copied"
            (d / "files").mkdir(parents=True)
            (d / "task.json").write_text(json.dumps({"id": "copied", "instruction": passage, "edit_file": "m.py"}))
            (d / "files" / "m.py").write_text("X = 1\n")
            (d / "oracle.txt").write_text("")
            report = decontam_tasks.build_report(tmp)
        self.assertEqual([t["id"] for t in report["tasks"]], ["copied"])
        self.assertGreater(report["tasks"][0]["overlaps"], 0)
        self.assertTrue(report["tasks"][0]["examples"])


if __name__ == "__main__":
    unittest.main()
