"""Issue #5: probe before set_upstream; clear on failure path."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

import serve_colab  # noqa: E402


class ProbeTests(unittest.TestCase):
    def test_probe_accepts_stop_and_usage(self) -> None:
        payload = {
            "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
        }
        with patch.object(serve_colab, "_http_json", return_value=(200, payload)):
            out = serve_colab.probe_completion("http://127.0.0.1:8001", "simplicio-27b")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")

    def test_probe_rejects_unknown_finish(self) -> None:
        payload = {
            "choices": [{"message": {"content": ""}, "finish_reason": None}],
            "usage": {},
        }
        with patch.object(serve_colab, "_http_json", return_value=(200, payload)):
            with self.assertRaises(RuntimeError):
                serve_colab.probe_completion("http://127.0.0.1:8001", "simplicio-27b")

    def test_probe_rejects_empty_content(self) -> None:
        payload = {
            "choices": [{"message": {"content": "  "}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
        }
        with patch.object(serve_colab, "_http_json", return_value=(200, payload)):
            with self.assertRaises(RuntimeError):
                serve_colab.probe_completion("http://127.0.0.1:8001", "simplicio-27b")

    def test_probe_all_sends_both_ids(self) -> None:
        sent: list[str] = []
        payload = {
            "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
        }

        def fake_http(url, body=None, timeout=30):
            sent.append(body["model"])
            return 200, payload

        with patch.object(serve_colab, "_http_json", side_effect=fake_http):
            serve_colab.probe_all("http://127.0.0.1:8001")
        self.assertEqual(sent, ["simplicio-27b", "simpleti/simplicio-27b"])

    def test_set_upstream_not_called_when_probe_fails(self) -> None:
        calls: list[tuple] = []

        def fake_http(url, body=None, timeout=30):
            calls.append((url, body))
            return 200, {"choices": [{"finish_reason": None}], "usage": {}}

        with patch.object(serve_colab, "_http_json", side_effect=fake_http):
            with self.assertRaises(RuntimeError):
                serve_colab.probe_completion("http://gate", "simplicio-27b")
        self.assertTrue(all("set_upstream" not in (url or "") for url, _ in calls))


class LaunchTests(unittest.TestCase):
    def test_serve_cmd_runs_the_script(self) -> None:
        self.assertEqual(
            serve_colab.serve_cmd(),
            ["bash", str(serve_colab.DEPLOY / "serve_vllm.sh"), "wesleysimplicio/Simplicio-27B", str(serve_colab.VLLM_PORT)],
        )

    def test_no_second_flag_list(self) -> None:
        text = Path(serve_colab.__file__).read_text(encoding="utf-8")
        for word in ("vllm_cmd", "--served-model-name", "bitsandbytes", "LORA_MODULE", "--stop"):
            self.assertNotIn(word, text)

    def test_early_exit_shows_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "vllm.log"
            log.write_text("\n".join(f"line {i}" for i in range(100)) + "\n", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(2)"])
            proc.wait()
            with self.assertRaises(SystemExit) as ctx:
                serve_colab.wait_vllm(proc, log, timeout_s=30)
        message = str(ctx.exception)
        self.assertIn("line 60", message)
        self.assertIn("line 99", message)
        self.assertNotIn("line 59", message)


class ClearUpstreamTests(unittest.TestCase):
    def test_clear_tries_null_url(self) -> None:
        seen: list[dict] = []

        def fake_http(url, body=None, timeout=30):
            seen.append(body)
            if body == {"clear": True}:
                return 200, {"ok": True, "upstream": None}
            return 400, {"error": "no"}

        with patch.object(serve_colab, "_http_json", side_effect=fake_http):
            serve_colab.clear_upstream("k")
        self.assertEqual(seen[0], {"clear": True})


if __name__ == "__main__":
    unittest.main()
