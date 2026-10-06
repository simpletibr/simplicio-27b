"""Issue #5: probe before set_upstream; clear on failure path."""

from __future__ import annotations

import sys
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
            out = serve_colab.probe_completion("http://127.0.0.1:8001")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")

    def test_probe_rejects_unknown_finish(self) -> None:
        payload = {
            "choices": [{"message": {"content": ""}, "finish_reason": None}],
            "usage": {},
        }
        with patch.object(serve_colab, "_http_json", return_value=(200, payload)):
            with self.assertRaises(RuntimeError):
                serve_colab.probe_completion("http://127.0.0.1:8001")

    def test_set_upstream_not_called_when_probe_fails(self) -> None:
        calls: list[tuple] = []

        def fake_http(url, body=None, timeout=30):
            calls.append((url, body))
            return 200, {"choices": [{"finish_reason": None}], "usage": {}}

        with patch.object(serve_colab, "_http_json", side_effect=fake_http):
            with self.assertRaises(RuntimeError):
                serve_colab.probe_completion("http://gate")
        self.assertTrue(all("set_upstream" not in (url or "") for url, _ in calls))


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
