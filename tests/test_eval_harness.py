import importlib.util
import json
import os
import shutil
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


class FakeEndpoint:
    def __init__(self, replies):
        self.requests = []
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode())
                requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                last = body["messages"][-1]["content"]
                content = ""
                for edit_file, reply in replies.items():
                    if f"Arquivo: {edit_file}\n" in last:
                        content = reply
                payload = json.dumps({
                    "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
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


def make_task(root, test_body, source="X = 1\n"):
    root = Path(root)
    (root / "t/files").mkdir(parents=True)
    (root / "t/tests").mkdir(parents=True)
    (root / "t/files/m.py").write_text(source)
    (root / "t/tests/test_m.py").write_text(test_body)
    return {"id": "t", "edit_file": "m.py", "instruction": "i", "context": "c", "dir": root / "t"}


class PureTests(unittest.TestCase):
    def test_parse_four_and_seven_char_markers(self):
        text = (
            "<think>x</think><patch>\n<<<< SEARCH\na = 1\n====\na = 2\n>>>> REPLACE\nfoo\n"
            "<<<<<<< SEARCH\nb = 1\n=======\nb = 3\n>>>>>>> REPLACE\n</patch>"
        )
        self.assertEqual(harness.parse_blocks(text), [("a = 1", "a = 2"), ("b = 1", "b = 3")])

    def test_unclosed_block_is_ignored(self):
        self.assertEqual(harness.parse_blocks("<<<< SEARCH\na\n====\nb\n"), [])

    def test_apply_errors(self):
        for blocks, msg in (([], "no_blocks"), ([("", "x")], "empty_search"), ([("zzz", "x")], "search_not_found")):
            with self.assertRaises(harness.PatchError) as cm:
                harness.apply_blocks("a = 1\n", blocks)
            self.assertIn(msg, str(cm.exception))

    def test_apply_replaces_first_occurrence_in_order(self):
        self.assertEqual(harness.apply_blocks("x\nx\ny\n", [("x", "z"), ("y", "w")]), "z\nx\nw\n")

    def test_known_values(self):
        lo, hi = harness.wilson_interval(56, 120)
        self.assertAlmostEqual(lo, 0.37983, places=5)
        self.assertAlmostEqual(hi, 0.55557, places=5)
        self.assertEqual(harness.wilson_interval(0, 10)[0], 0.0)
        self.assertEqual(harness.wilson_interval(10, 10)[1], 1.0)
        with self.assertRaises(ValueError):
            harness.wilson_interval(0, 0)

    def test_duplicate_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a", "b"):
                shutil.copytree(TASKS / "ex_safe_divide", Path(tmp) / name)
                path = Path(tmp) / name / "task.json"
                data = json.loads(path.read_text())
                data["id"] = name
                path.write_text(json.dumps(data))
            with self.assertRaises(ValueError) as cm:
                harness.load_tasks(tmp)
            self.assertIn("duplicate", str(cm.exception))

    def test_build_messages(self):
        task = next(t for t in harness.load_tasks(TASKS) if t["id"] == "ex_safe_divide")
        msgs = harness.build_messages(task, "simplicio", False)
        self.assertEqual(msgs[0], {"role": "system", "content": harness.SYSTEM_PROMPT})
        self.assertIn("Arquivo: calc.py\nCodigo Atual:\ndef safe_divide(a, b):", msgs[1]["content"])
        msgs = harness.build_messages(task, "none", True)
        self.assertEqual([m["role"] for m in msgs], ["user"])
        self.assertTrue(msgs[0]["content"].endswith(harness.FORMAT_HINT))

    def test_sandbox_env_has_no_secrets(self):
        with mock.patch.dict(os.environ, {"SIMPLETI_API_KEY": "secret-for-test"}):
            env = harness.sandbox_env(Path("/tmp"))
        self.assertNotIn("secret-for-test", env.values())
        self.assertEqual(env["HTTPS_PROXY"], "http://127.0.0.1:9")
        self.assertEqual(env["TASK_SRC"], str(Path("/tmp")))


@unittest.skipUnless(
    importlib.util.find_spec("pytest"),
    "pytest is not installed: python3 -m pip install -r benchmarks/harness/requirements.txt",
)
class SandboxTests(unittest.TestCase):
    def test_network_is_blocked(self):
        body = (
            "import os, socket\nimport pytest\n"
            "def test_net():\n"
            "    with pytest.raises(OSError):\n"
            "        socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
            "def test_src():\n"
            "    assert open(os.environ['TASK_SRC'] + '/m.py').read() == 'X = 1\\n'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(tmp, body)
            self.assertTrue(harness.check_source(task, "X = 1\n")["passed"])

    def test_timeout_kills_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(tmp, "import time\ndef test_sleep():\n    time.sleep(30)\n")
            res = harness.check_source(task, "X = 1\n", timeout=2)
        self.assertTrue(res["timed_out"])
        self.assertFalse(res["passed"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "RLIMIT_AS is only applied on Linux")
    def test_memory_limit_on_linux(self):
        body = "import pytest\ndef test_mem():\n    with pytest.raises(MemoryError):\n        bytearray(3 << 30)\n"
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(tmp, body)
            self.assertTrue(harness.check_source(task, "X = 1\n")["passed"])

    def test_oracle_mode_passes_example_tasks(self):
        self.assertEqual(harness.main(["--tasks", str(TASKS), "--oracle"]), 0)

    def test_run_against_fake_endpoint(self):
        replies = {}
        for t in harness.load_tasks(TASKS):
            replies[t["edit_file"]] = (t["dir"] / "oracle.txt").read_text()
        replies["listutil.py"] = "<<<< SEARCH\nnot in the file\n====\nx\n>>>> REPLACE\n"
        fake = FakeEndpoint(replies)
        try:
            with tempfile.TemporaryDirectory() as out, mock.patch.dict(os.environ, {"SIMPLETI_API_KEY": "test-key"}):
                rc = harness.main([
                    "--tasks", str(TASKS), "--base-url", fake.url, "--model", "fake/m",
                    "--chat-template-kwargs", '{"enable_thinking": false}',
                    "--extra-body", '{"reasoning_effort": "none"}',
                    "--out-dir", out, "--run-name", "r1",
                ])
                rows = [json.loads(line) for line in (Path(out) / "r1.jsonl").read_text().splitlines()]
                summary = json.loads((Path(out) / "r1.summary.json").read_text())
        finally:
            fake.close()
        self.assertEqual(rc, 0)
        self.assertEqual([r["task"] for r in rows], ["ex_dedupe_order", "ex_parse_port", "ex_safe_divide"])
        self.assertEqual([r["passed"] for r in rows], [False, True, True])
        self.assertEqual(rows[0]["apply_error"], "search_not_found in block 0")
        self.assertEqual(
            (summary["n_tasks"], summary["passed"], summary["applied"], summary["errors"]), (3, 2, 2, 0)
        )
        self.assertEqual(summary["extra_body"], {"reasoning_effort": "none"})
        req = fake.requests[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["auth"], "Bearer test-key")
        self.assertEqual(req["body"]["temperature"], 0)
        self.assertEqual(req["body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(req["body"]["reasoning_effort"], "none")
        self.assertEqual(req["body"]["messages"][0]["content"], harness.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
