"""Contract test for issue #16: what the completion gate promises to its clients.

Same setup as test_flow_16: the real gate (serve_colab._GateHandler) in front of a fake upstream, both on
ephemeral 127.0.0.1 ports. No external network, no vLLM, no credentials.

Contract:
  * non-stream 200 without a valid finish_reason (or without usage) -> HTTP 502, JSON BadCompletionError;
  * a stream cannot change the 200 it already sent, so a stream that ends without a valid finish_reason
    gets ONE terminal SSE error event (the same error object as the JSON 502 body) and then `[DONE]`;
  * a valid stream is never turned into an error, with or without a usage event;
  * every stream answers with Content-Type text/event-stream.
"""

from __future__ import annotations

import http.client
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

import completion_gate  # noqa: E402
import serve_colab  # noqa: E402


def _event(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _chat(finish, content="pong") -> bytes:
    return _event({"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}]})


CONTENT = _chat(None)
STOP = _chat("stop", "")
UNKNOWN = _chat("unknown", "")
USAGE = _event({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}})
UPSTREAM_ERROR = _event({"error": {"message": "CUDA out of memory", "type": "InternalServerError", "code": 500}})
KEEP_ALIVE = b": keep-alive\n\n"
DONE = b"data: [DONE]\n\n"

# Streaming scenarios: the chunks the fake upstream sends, and whether it ends the body cleanly.
STREAMS: dict[str, tuple[list[bytes], bool]] = {
    "stream-valid": ([CONTENT, KEEP_ALIVE, STOP, USAGE, DONE], True),
    "stream-no-usage": ([CONTENT, STOP, DONE], True),
    "stream-no-finish": ([CONTENT, DONE], True),
    "stream-unknown-finish": ([CONTENT, UNKNOWN, DONE], True),
    "stream-empty": ([DONE], True),
    "stream-drop": ([CONTENT], False),
    "stream-upstream-error": ([CONTENT, UPSTREAM_ERROR, DONE], True),
}


def _json_body(finish="stop", usage=True, choices=True) -> dict:
    body: dict = {"id": "x", "object": "chat.completion"}
    if choices:
        body["choices"] = [
            {"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": finish}
        ]
    else:
        body["choices"] = []
    if usage:
        body["usage"] = {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
    return body


JSONS: dict[str, tuple[int, dict]] = {
    "json-ok": (200, _json_body()),
    "json-null-finish": (200, _json_body(finish=None)),
    "json-unknown-finish": (200, _json_body(finish="unknown")),
    "json-no-usage": (200, _json_body(usage=False)),
    "json-no-choices": (200, _json_body(choices=False)),
    "json-400": (400, {"error": {"message": "maximum context length is 40960 tokens", "code": 400}}),
}


class FakeUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _reply_json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        # Not a chat completion: the gate must forward it untouched even though it has no choices.
        self._reply_json(200, {"object": "list", "data": [{"id": "simplicio-27b"}]})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        scenario = json.loads(self.rfile.read(length))["model"]
        if scenario in JSONS:
            self._reply_json(*JSONS[scenario])
            return
        chunks, clean = STREAMS[scenario]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for data in chunks:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
        if clean:
            self.wfile.write(b"0\r\n\r\n")
        else:
            self.close_connection = True  # drop: no terminating chunk


def payloads(body: bytes) -> list[str]:
    """The `data:` payloads of an SSE body, in order (keep-alive comments are skipped)."""
    out = []
    for block in body.decode("utf-8").split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                out.append(line[5:].strip())
    return out


class GateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
        threading.Thread(target=cls.upstream.serve_forever, daemon=True).start()
        cls.old_port = serve_colab.VLLM_PORT
        serve_colab.VLLM_PORT = cls.upstream.server_address[1]
        cls.gate = ThreadingHTTPServer(("127.0.0.1", 0), serve_colab._GateHandler)
        threading.Thread(target=cls.gate.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        for server in (cls.gate, cls.upstream):
            server.shutdown()
            server.server_close()
        serve_colab.VLLM_PORT = cls.old_port

    def _request(self, method: str, path: str, scenario: str = "", stream: bool = False):
        conn = http.client.HTTPConnection("127.0.0.1", self.gate.server_address[1], timeout=10)
        self.addCleanup(conn.close)
        body = None
        if method == "POST":
            body = json.dumps({"model": scenario, "stream": stream, "messages": []})
        conn.request(method, path, body=body,
                     headers={"Content-Type": "application/json",
                              serve_colab.UPSTREAM_HEADER: serve_colab.GATE_TOKEN})
        resp = conn.getresponse()
        return resp, resp.read()

    def _chat(self, scenario: str, stream: bool):
        return self._request("POST", "/v1/chat/completions", scenario, stream)

    def _stream(self, scenario: str):
        resp, body = self._chat(scenario, stream=True)
        self.assertEqual(resp.status, 200, "a stream cannot change the status it already sent")
        return resp, payloads(body)

    def _assert_terminal_error(self, items: list[str], needle: str) -> None:
        """[..., <one BadCompletionError event>, [DONE]] and no other error event."""
        self.assertEqual(items[-1], "[DONE]")
        self.assertEqual(items.count("[DONE]"), 1)
        error = json.loads(items[-2])["error"]
        self.assertEqual(error["type"], "BadCompletionError")
        self.assertEqual(error["code"], 502)
        self.assertIn(needle, error["message"])
        self.assertEqual(sum(1 for item in items if "BadCompletionError" in item), 1)

    # (a) non-stream: no valid finish_reason / usage -> 502

    def test_non_stream_without_valid_finish_reason_is_502(self) -> None:
        for scenario in ("json-null-finish", "json-unknown-finish", "json-no-choices"):
            with self.subTest(scenario):
                resp, body = self._chat(scenario, stream=False)
                self.assertEqual(resp.status, 502)
                self.assertEqual(resp.getheader("Content-Type"), "application/json; charset=utf-8")
                error = json.loads(body)["error"]
                self.assertEqual(error["type"], "BadCompletionError")
                self.assertEqual(error["code"], 502)

    def test_non_stream_502_names_the_finish_reason_problem(self) -> None:
        _, body = self._chat("json-unknown-finish", stream=False)
        self.assertIn("finish_reason", json.loads(body)["error"]["message"])

    def test_non_stream_without_usage_is_502(self) -> None:
        resp, body = self._chat("json-no-usage", stream=False)
        self.assertEqual(resp.status, 502)
        self.assertIn("usage", json.loads(body)["error"]["message"])

    def test_non_stream_valid_and_upstream_4xx_pass_unchanged(self) -> None:
        resp, body = self._chat("json-ok", stream=False)
        self.assertEqual((resp.status, json.loads(body)), (200, JSONS["json-ok"][1]))
        resp, body = self._chat("json-400", stream=False)
        self.assertEqual((resp.status, json.loads(body)), (400, JSONS["json-400"][1]))

    def test_routes_other_than_chat_completions_are_not_gated(self) -> None:
        resp, body = self._request("GET", "/v1/models")
        self.assertEqual(resp.status, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], "simplicio-27b")

    # (b) stream: a bad end gets a terminal error event, not a silent 200

    def test_stream_without_finish_reason_gets_terminal_error_event(self) -> None:
        _, items = self._stream("stream-no-finish")
        self.assertIn('"content": "pong"', items[0])  # what was valid was forwarded as it came
        self._assert_terminal_error(items, "finish_reason")

    def test_stream_with_unknown_finish_reason_gets_terminal_error_event(self) -> None:
        _, items = self._stream("stream-unknown-finish")
        self._assert_terminal_error(items, "'unknown'")

    def test_empty_stream_gets_terminal_error_event(self) -> None:
        _, items = self._stream("stream-empty")
        self.assertEqual(len(items), 2)  # error event + [DONE]
        self._assert_terminal_error(items, "empty stream")

    def test_upstream_dropping_mid_stream_gets_terminal_error_event(self) -> None:
        _, items = self._stream("stream-drop")
        self._assert_terminal_error(items, "finish_reason")

    def test_valid_stream_is_never_an_error_with_or_without_usage(self) -> None:
        for scenario, has_usage in (("stream-valid", True), ("stream-no-usage", False)):
            with self.subTest(scenario):
                resp, items = self._stream(scenario)
                self.assertEqual(items[-1], "[DONE]")
                self.assertEqual(items.count("[DONE]"), 1)
                self.assertFalse(any("BadCompletionError" in item for item in items))
                self.assertEqual(any('"usage"' in item for item in items), has_usage)
                self.assertIn('"finish_reason": "stop"', "".join(items))

    def test_upstream_error_event_is_forwarded_once_without_a_second_error(self) -> None:
        _, items = self._stream("stream-upstream-error")
        self.assertEqual(sum(1 for item in items if "CUDA out of memory" in item), 1)
        self.assertFalse(any("BadCompletionError" in item for item in items))
        self.assertEqual(items.count("[DONE]"), 1)

    def test_terminal_error_is_documented(self) -> None:
        for module in (completion_gate, serve_colab):
            with self.subTest(module.__name__):
                doc = " ".join((module.__doc__ or "").split())
                self.assertIn("SSE error event", doc)
                self.assertIn("finish_reason", doc)
        # the event carries the same error object as the JSON 502 body
        event = completion_gate.sse_error_event("x")
        self.assertEqual(
            json.loads(event[len(b"data: "):]), json.loads(completion_gate._error_body("x"))
        )

    # (c) every stream is text/event-stream

    def test_stream_content_type_is_event_stream(self) -> None:
        for scenario in STREAMS:
            with self.subTest(scenario):
                resp, _ = self._stream(scenario)
                self.assertTrue(resp.getheader("Content-Type", "").startswith("text/event-stream"))
                self.assertEqual(resp.getheader("Cache-Control"), "no-cache")


if __name__ == "__main__":
    unittest.main()
