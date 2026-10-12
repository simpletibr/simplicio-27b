from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = ROOT / "scripts" / "mcp_server.py"


class McpServerTests(unittest.TestCase):
    def run_rpc(self, requests: list[dict]) -> list[dict]:
        proc = subprocess.Popen(
            [sys.executable, str(SERVER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        input_data = "\n".join(json.dumps(r) for r in requests) + "\n"
        stdout, _ = proc.communicate(input=input_data, timeout=5)
        responses = []
        for line in stdout.splitlines():
            line = line.strip()
            if line:
                responses.append(json.loads(line))
        return responses

    def test_initialize(self) -> None:
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resps = self.run_rpc([req])
        self.assertEqual(len(resps), 1)
        r = resps[0]
        self.assertEqual(r["id"], 1)
        self.assertEqual(r["result"]["serverInfo"]["name"], "simplicio-27b")

    def test_tools_list(self) -> None:
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        resps = self.run_rpc([req])
        self.assertEqual(len(resps), 1)
        tools = resps[0]["result"]["tools"]
        names = [t["name"] for t in tools]
        self.assertIn("apply_surgical_diff", names)
        self.assertIn("validate_patch", names)
        self.assertIn("format_prompt", names)

    def test_format_prompt(self) -> None:
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "format_prompt",
                "arguments": {"context": "def foo(): pass", "task": "add return True"}
            }
        }
        resps = self.run_rpc([req])
        text = resps[0]["result"]["content"][0]["text"]
        data = json.loads(text)
        self.assertIn("Simplicio 27B", data["system"])
        self.assertIn("add return True", data["user"])

    def test_validate_patch_4_and_7_chars(self) -> None:
        patch_4 = "<<<< SEARCH\na = 1\n====\na = 2\n>>>> REPLACE"
        patch_7 = "<<<<<<< SEARCH\nb = 1\n=======\nb = 2\n>>>>>>> REPLACE"
        for patch in (patch_4, patch_7):
            with self.subTest(patch=patch):
                req = {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "validate_patch", "arguments": {"patch": patch}}
                }
                resps = self.run_rpc([req])
                text = resps[0]["result"]["content"][0]["text"]
                self.assertTrue(text.startswith("Válido."))

    def test_apply_diff_4_and_7_chars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "test.py"
            f.write_text("def run():\n    x = 10\n    return x\n", encoding="utf-8")

            # 4 characters
            patch_4 = "<<<< SEARCH\n    x = 10\n====\n    x = 20\n>>>> REPLACE"
            req_4 = {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "apply_surgical_diff", "arguments": {"path": str(f), "patch": patch_4}}
            }
            resps = self.run_rpc([req_4])
            self.assertIn("Sucesso", resps[0]["result"]["content"][0]["text"])
            self.assertIn("x = 20", f.read_text(encoding="utf-8"))

            # 7 characters
            patch_7 = "<<<<<<< SEARCH\n    return x\n=======\n    return x * 2\n>>>>>>> REPLACE"
            req_7 = {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "apply_surgical_diff", "arguments": {"path": str(f), "patch": patch_7}}
            }
            resps = self.run_rpc([req_7])
            self.assertIn("Sucesso", resps[0]["result"]["content"][0]["text"])
            self.assertIn("return x * 2", f.read_text(encoding="utf-8"))

    def test_apply_diff_search_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "test.py"
            f.write_text("x = 1\n", encoding="utf-8")
            patch = "<<<< SEARCH\nnot in file\n====\ny = 2\n>>>> REPLACE"
            req = {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "apply_surgical_diff", "arguments": {"path": str(f), "patch": patch}}
            }
            resps = self.run_rpc([req])
            self.assertIn("Falha", resps[0]["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
