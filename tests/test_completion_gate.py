"""Issue #5: completions without finish_reason/usage are not HTTP 200."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

from completion_gate import enforce  # noqa: E402


def _ok_json(**overrides):
    payload = {
        "id": "chatcmpl-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }
    payload.update(overrides)
    return json.dumps(payload).encode()


class JsonGateTests(unittest.TestCase):
    def test_valid_stop(self) -> None:
        status, body, _ = enforce(200, _ok_json(), "application/json")
        self.assertEqual(status, 200)
        self.assertIn(b"pong", body)

    def test_valid_length(self) -> None:
        raw = _ok_json()
        payload = json.loads(raw)
        payload["choices"][0]["finish_reason"] = "length"
        status, _, _ = enforce(200, json.dumps(payload).encode(), "application/json")
        self.assertEqual(status, 200)

    def test_missing_finish_is_502(self) -> None:
        payload = json.loads(_ok_json())
        payload["choices"][0]["finish_reason"] = None
        status, body, _ = enforce(200, json.dumps(payload).encode(), "application/json")
        self.assertEqual(status, 502)
        self.assertIn(b"BadCompletionError", body)

    def test_unknown_finish_is_502(self) -> None:
        payload = json.loads(_ok_json())
        payload["choices"][0]["finish_reason"] = "unknown"
        status, body, _ = enforce(200, json.dumps(payload).encode(), "application/json")
        self.assertEqual(status, 502)
        self.assertIn(b"finish_reason", body)

    def test_missing_usage_is_502(self) -> None:
        payload = json.loads(_ok_json())
        del payload["usage"]
        status, _, _ = enforce(200, json.dumps(payload).encode(), "application/json")
        self.assertEqual(status, 502)

    def test_upstream_error_passes_through(self) -> None:
        raw = b'{"error":{"message":"This model\'s maximum context length is 40960 tokens"}}'
        status, body, _ = enforce(400, raw, "application/json")
        self.assertEqual(status, 400)
        self.assertEqual(body, raw)


class SseGateTests(unittest.TestCase):
    def test_valid_stream(self) -> None:
        raw = (
            b'data: {"choices":[{"delta":{"content":"pong"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":1,"total_tokens":9}}\n\n'
            b"data: [DONE]\n"
        )
        status, body, _ = enforce(200, raw, "text/event-stream")
        self.assertEqual(status, 200)
        self.assertEqual(body, raw)

    def test_stream_without_finish_is_502(self) -> None:
        raw = (
            b'data: {"choices":[{"delta":{"content":"pong"},"finish_reason":null}]}\n\n'
            b"data: [DONE]\n"
        )
        status, body, _ = enforce(200, raw, "text/event-stream")
        self.assertEqual(status, 502)
        self.assertIn(b"BadCompletionError", body)

    def test_empty_stream_is_502(self) -> None:
        status, _, _ = enforce(200, b"", "text/event-stream")
        self.assertEqual(status, 502)


class PhpContractTests(unittest.TestCase):
    def test_clear_contract(self) -> None:
        php = (Path(__file__).resolve().parents[1] / "deploy" / "set_upstream.php").read_text()
        self.assertIn("clear", php)
        self.assertIn("upstream_url", php)
        self.assertIn("Unauthorized", php)


if __name__ == "__main__":
    unittest.main()
