"""Contract tests for the eval harness (#25).

Interfaces other pieces consume: the verdict read from pytest's JUnit report, the `report`
field of each JSONL row, the exit codes of the CLI, and the error rows written when the
endpoint answers with something that is not a chat completion.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tests/fixtures/harness_tasks"
sys.path.insert(0, str(ROOT / "benchmarks/harness"))
import harness  # noqa: E402

PASSED = '<testcase classname="t" name="a" time="0.01"/>'
FAILED = '<testcase classname="t" name="b"><failure message="x">boom</failure></testcase>'
ERRORED = '<testcase classname="t" name="c"><error message="x">boom</error></testcase>'
SKIPPED = '<testcase classname="t" name="d"><skipped type="pytest.skip" message="x"/></testcase>'


def junit(*cases, attrs=""):
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites>'
        f'<testsuite name="pytest" {attrs}>{"".join(cases)}</testsuite></testsuites>'
    )


class RawEndpoint:
    """Answers every POST with the same bytes and counts the requests."""

    def __init__(self, payload):
        self.hits = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.hits += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, {"SIMPLETI_API_KEY": "k"}):
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = harness.main(argv)
    return code, out.getvalue(), err.getvalue()


class ReportVerdict(unittest.TestCase):
    def read(self, content):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "junit.xml"
            if content is not None:
                path.write_text(content, encoding="utf-8")
            return harness.read_report(path)

    def assertVerdict(self, report, ok, reason):
        self.assertIs(report["ok"], ok, report)
        self.assertEqual(report["reason"], reason, report)

    def test_report_fields(self):
        self.assertEqual(set(self.read(junit(PASSED))), {"ok", "reason", "tests", "failures", "errors", "skipped"})

    def test_clean_report_is_ok(self):
        report = self.read(junit(PASSED, PASSED))
        self.assertVerdict(report, True, None)
        self.assertEqual((report["tests"], report["failures"], report["errors"], report["skipped"]), (2, 0, 0, 0))

    def test_testsuite_as_root_is_accepted(self):
        xml = f'<testsuite name="pytest" tests="1">{PASSED}</testsuite>'
        self.assertVerdict(self.read(xml), True, None)

    def test_missing_report_is_fail(self):
        self.assertVerdict(self.read(None), False, "report_missing")

    def test_unreadable_reports_are_fail(self):
        for content in ("", "not xml at all", "<testsuites><testsuite>", junit(PASSED, attrs='failures="x"')):
            with self.subTest(content=content[:30]):
                self.assertVerdict(self.read(content), False, "report_unreadable")

    def test_report_that_is_a_directory_is_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertVerdict(harness.read_report(tmp), False, "report_unreadable")

    def test_oversized_report_is_fail(self):
        with mock.patch.object(harness, "MAX_REPORT_BYTES", 10):
            self.assertVerdict(self.read(junit(PASSED)), False, "report_too_large")

    def test_zero_cases_is_fail(self):
        report = self.read(junit(attrs='tests="0"'))
        self.assertVerdict(report, False, "no_tests_executed")
        self.assertEqual(report["tests"], 0)

    def test_failure_or_error_is_fail(self):
        for bad in (FAILED, ERRORED):
            with self.subTest(case=bad[:50]):
                self.assertVerdict(self.read(junit(PASSED, bad)), False, "tests_failed")

    def test_counts_in_suite_attributes_are_not_ignored(self):
        self.assertVerdict(self.read(junit(PASSED, attrs='failures="1"')), False, "tests_failed")
        self.assertVerdict(self.read(junit(PASSED, attrs='errors="1"')), False, "tests_failed")

    def test_only_skipped_is_fail(self):
        self.assertVerdict(self.read(junit(SKIPPED)), False, "no_tests_executed")

    def test_partly_skipped_is_fail(self):
        report = self.read(junit(PASSED, SKIPPED))
        self.assertVerdict(report, False, "tests_skipped")
        self.assertEqual(report["skipped"], 1)


class BootContract(unittest.TestCase):
    def test_boot_writes_the_report_where_the_harness_says(self):
        self.assertIn('"--junitxml=" + sys.argv[1]', harness.BOOT)

    def test_run_tests_returns_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "tests").mkdir()
            (work / "tests/test_x.py").write_text("def test_x():\n    assert True\n")
            res = harness.run_tests(work, timeout=30)
        if res["returncode"] != 0 and "No module named 'pytest'" in res["stdout_tail"]:
            self.skipTest("pytest is not installed: python3 -m pip install -r benchmarks/harness/requirements.txt")
        self.assertEqual(set(res), {"returncode", "timed_out", "stdout_tail", "report"})
        self.assertTrue(res["report"]["ok"], res)
        self.assertEqual(res["report"]["tests"], 1)

    def test_evaluate_row_has_a_report_field_even_when_nothing_ran(self):
        task = next(t for t in harness.load_tasks(TASKS) if t["id"] == "ex_safe_divide")
        res = harness.evaluate(task, "no blocks here")
        self.assertIn("report", res)
        self.assertIsNone(res["report"])
        self.assertFalse(res["passed"])


class NoTasks(unittest.TestCase):
    """n = 0 used to raise ValueError from wilson_interval after the loop."""

    def run_with_empty_tasks(self, extra):
        fake = RawEndpoint(b"{}")
        self.addCleanup(fake.close)
        with tempfile.TemporaryDirectory() as tmp:
            tasks, out_dir = Path(tmp) / "tasks", Path(tmp) / "results"
            tasks.mkdir()
            (tasks / "loose.txt").write_text("files without task.json are ignored")
            argv = ["--tasks", str(tasks), "--base-url", fake.url, "--model", "m", "--out-dir", str(out_dir)]
            code, out, err = run_cli(argv + extra)
            created = out_dir.exists()
        return code, out, err, created, fake.hits

    def test_empty_tasks_dir_is_an_explicit_error_not_a_traceback(self):
        code, out, err, created, hits = self.run_with_empty_tasks([])
        self.assertEqual(code, 2)
        self.assertIn("no tasks found", err)
        self.assertEqual(out, "")
        self.assertFalse(created, "nothing may be written when there is nothing to run")
        self.assertEqual(hits, 0)

    def test_empty_tasks_dir_is_an_error_in_oracle_mode_too(self):
        code, out, err, created, hits = self.run_with_empty_tasks(["--oracle"])
        self.assertEqual(code, 2)
        self.assertIn("no tasks found", err)
        self.assertEqual(out, "")


class MalformedEndpointAnswers(unittest.TestCase):
    """AttributeError and TypeError from an odd answer are a FAIL of the task, not a crash."""

    CASES = {
        "AttributeError": b'{"choices": [{"message": null, "finish_reason": "stop"}]}',
        "TypeError": b"[]",
        "TypeError (choices null)": b'{"choices": null}',
        "TypeError (content not text)": b'{"choices": [{"message": {"content": 123}, "finish_reason": "stop"}]}',
    }

    def run_against(self, payload):
        fake = RawEndpoint(payload)
        self.addCleanup(fake.close)
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = run_cli([
                "--tasks", str(TASKS), "--base-url", fake.url, "--model", "odd/m",
                "--out-dir", tmp, "--run-name", "odd",
            ])
            rows = [json.loads(line) for line in (Path(tmp) / "odd.jsonl").read_text().splitlines()]
            summary = json.loads((Path(tmp) / "odd.summary.json").read_text())
        return code, out, err, rows, summary, fake.hits

    def test_each_odd_answer_becomes_a_failed_task_and_the_run_goes_on(self):
        for label, payload in self.CASES.items():
            with self.subTest(case=label):
                code, out, err, rows, summary, hits = self.run_against(payload)
                self.assertEqual(code, 1)
                self.assertEqual(hits, 3, "every task is still tried after the first one fails")
                self.assertEqual([r["task"] for r in rows], ["ex_dedupe_order", "ex_parse_port", "ex_safe_divide"])
                for row in rows:
                    self.assertTrue(row["error"].startswith(label.split(" ")[0] + ":"), row["error"])
                    self.assertIs(row["passed"], False)
                    self.assertIs(row["applied"], False)
                    self.assertIsNone(row["raw_output"])
                self.assertEqual(out.count(": FAIL"), 3)
                self.assertNotIn("Traceback", err)
                self.assertEqual((summary["n_tasks"], summary["passed"], summary["errors"]), (3, 0, 3))


if __name__ == "__main__":
    unittest.main()
