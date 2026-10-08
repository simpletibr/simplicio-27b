"""Issue #5: completions without finish_reason/usage are not HTTP 200."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

from completion_gate import StreamCheck, enforce, sse_error_event  # noqa: E402


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


class StreamCheckTests(unittest.TestCase):
    def _check(self, *lines: bytes) -> StreamCheck:
        check = StreamCheck()
        for line in lines:
            check.feed(line)
        return check

    def test_valid_stream(self) -> None:
        check = self._check(
            b'data: {"choices":[{"delta":{"content":"pong"},"finish_reason":null}]}\n', b"\n",
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n', b"\n",
            b'data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":1,"total_tokens":9}}\n', b"\n",
            b"data: [DONE]\n", b"\n",
        )
        self.assertIsNone(check.problem())
        self.assertEqual(check.finish_reason, "stop")
        self.assertEqual(check.usage["total_tokens"], 9)
        self.assertTrue(check.done)

    def test_valid_stream_without_usage_has_no_problem(self) -> None:
        check = self._check(b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n', b"\n")
        self.assertIsNone(check.problem())
        self.assertIsNone(check.usage)

    def test_stream_without_finish_is_a_problem(self) -> None:
        check = self._check(b'data: {"choices":[{"delta":{"content":"pong"},"finish_reason":null}]}\n')
        self.assertIn("finish_reason", check.problem())

    def test_unknown_finish_is_a_problem(self) -> None:
        check = self._check(b'data: {"choices":[{"delta":{},"finish_reason":"unknown"}]}\n')
        self.assertIn("finish_reason", check.problem())

    def test_empty_stream_is_a_problem(self) -> None:
        self.assertEqual(self._check().problem(), "empty stream")

    def test_keep_alive_comment_is_ignored(self) -> None:
        check = self._check(b": keep-alive\n", b"\n")
        self.assertEqual(check.events, 0)

    def test_upstream_error_event_needs_no_second_error(self) -> None:
        check = self._check(
            b'data: {"error":{"message":"CUDA out of memory","type":"InternalServerError","code":500}}\n'
        )
        self.assertEqual(check.upstream_error, "CUDA out of memory")
        self.assertIsNone(check.problem())

    def test_error_event_shape(self) -> None:
        event = sse_error_event("x")
        self.assertTrue(event.startswith(b"data: "))
        self.assertTrue(event.endswith(b"\n\n"))
        self.assertEqual(json.loads(event[6:])["error"]["type"], "BadCompletionError")


class PhpContractTests(unittest.TestCase):
    def test_clear_contract(self) -> None:
        php = (Path(__file__).resolve().parents[1] / "gateway" / "set_upstream.php").read_text()
        self.assertIn("clear", php)
        self.assertIn("upstream_url", php)
        self.assertIn("Unauthorized", php)


if __name__ == "__main__":
    unittest.main()
