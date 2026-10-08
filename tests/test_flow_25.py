"""Flow tests for the eval harness verdict (#25): a patch must not pass on the exit code alone.

The model's code is imported in the same process as pytest, so a module that runs
`import os; os._exit(0)` ends the process with exit code 0 before a single test runs.
The verdict has to come from the JUnit report pytest writes at the end of its session.
"""
import contextlib
import importlib.util
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

EXIT = "import os; os._exit(0)"
NEEDS_PYTEST = unittest.skipUnless(
    importlib.util.find_spec("pytest"),
    "pytest is not installed: python3 -m pip install -r benchmarks/harness/requirements.txt",
)


def task_by_id(task_id):
    return next(t for t in harness.load_tasks(TASKS) if t["id"] == task_id)


def exit_patch(source):
    """A SEARCH/REPLACE answer that puts os._exit(0) on top of the module."""
    first = source.splitlines()[0]
    return f"<patch>\n<<<< SEARCH\n{first}\n====\n{EXIT}\n{first}\n>>>> REPLACE\n</patch>\n"


class EvilEndpoint:
    """OpenAI-compatible fake that answers every task with the os._exit(0) patch."""

    def __init__(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
                last = body["messages"][-1]["content"]
                code = last.split("Codigo Atual:\n", 1)[1].rsplit("\nTarefa:", 1)[0]
                payload = json.dumps({
                    "choices": [{"message": {"role": "assistant", "content": exit_patch(code)},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }).encode()
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


@NEEDS_PYTEST
class OsExitVerdict(unittest.TestCase):
    def test_os_exit_zero_patch_is_fail(self):
        task = task_by_id("ex_safe_divide")
        res = harness.evaluate(task, exit_patch(harness._source(task)))
        self.assertTrue(res["applied"])
        # The process really exits with 0: this is what the old verdict (returncode == 0) accepted.
        self.assertEqual(res["returncode"], 0)
        self.assertFalse(res["timed_out"])
        self.assertFalse(res["passed"])
        self.assertFalse(res["report"]["ok"])
        self.assertEqual(res["report"]["reason"], "report_missing")
        self.assertEqual(res["report"]["tests"], 0)

    def test_os_exit_zero_is_fail_for_every_example_task(self):
        for task in harness.load_tasks(TASKS):
            with self.subTest(task=task["id"]):
                res = harness.evaluate(task, exit_patch(harness._source(task)))
                self.assertEqual(res["returncode"], 0)
                self.assertFalse(res["passed"])

    def test_os_exit_zero_through_the_cli_is_fail(self):
        fake = EvilEndpoint()
        self.addCleanup(fake.close)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"SIMPLETI_API_KEY": "k"}):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = harness.main([
                    "--tasks", str(TASKS), "--base-url", fake.url, "--model", "evil/os-exit",
                    "--out-dir", tmp, "--run-name", "evil",
                ])
            rows = [json.loads(line) for line in (Path(tmp) / "evil.jsonl").read_text().splitlines()]
            summary = json.loads((Path(tmp) / "evil.summary.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual([r["task"] for r in rows], ["ex_dedupe_order", "ex_parse_port", "ex_safe_divide"])
        self.assertEqual([r["passed"] for r in rows], [False, False, False])
        self.assertEqual([r["returncode"] for r in rows], [0, 0, 0])
        self.assertEqual({r["report"]["reason"] for r in rows}, {"report_missing"})
        self.assertEqual((summary["n_tasks"], summary["applied"], summary["passed"], summary["errors"]), (3, 3, 0, 0))
        self.assertEqual(out.getvalue().count(": FAIL"), 3)
        self.assertNotIn(": PASS", out.getvalue())

    def test_correct_patch_still_passes_with_a_report(self):
        task = task_by_id("ex_safe_divide")
        res = harness.evaluate(task, (task["dir"] / "oracle.txt").read_text())
        self.assertTrue(res["passed"])
        self.assertEqual(res["returncode"], 0)
        self.assertTrue(res["report"]["ok"])
        self.assertGreaterEqual(res["report"]["tests"], 1)
        self.assertEqual((res["report"]["failures"], res["report"]["errors"], res["report"]["skipped"]), (0, 0, 0))

    def test_unpatched_source_still_fails_on_the_tests(self):
        task = task_by_id("ex_safe_divide")
        res = harness.check_source(task, harness._source(task))
        self.assertFalse(res["passed"])
        self.assertEqual(res["report"]["reason"], "tests_failed")
        self.assertGreaterEqual(res["report"]["failures"], 1)

    def test_oracle_cli_still_ok(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = harness.main(["--tasks", str(TASKS), "--oracle"])
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue().splitlines(),
                         ["OK ex_dedupe_order", "OK ex_parse_port", "OK ex_safe_divide"])


if __name__ == "__main__":
    unittest.main()
