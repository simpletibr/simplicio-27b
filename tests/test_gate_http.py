"""Issue #16: the gate forwards SSE as it arrives and never turns a valid stream into a 502.

Starts the real gate (serve_colab._GateHandler) on an ephemeral 127.0.0.1 port, in front of a fake
upstream on another ephemeral port. No external network, no vLLM, no credentials.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

import serve_colab  # noqa: E402


def _event(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


CONTENT = _event({"choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": None}]})
STOP = _event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
USAGE = _event({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}})
UPSTREAM_ERROR = _event(
    {"error": {"message": "CUDA out of memory", "type": "InternalServerError", "param": None, "code": 500}}
)
KEEP_ALIVE = b": keep-alive\n\n"
DONE = b"data: [DONE]\n\n"

JSON_OK = {
    "id": "x",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}
JSON_NO_FINISH = {
    **JSON_OK,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": None}],
}


class FakeUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    sent: list[int] = []
    aborted = threading.Event()

    def log_message(self, *args) -> None:
        pass

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        scenario = json.loads(self.rfile.read(length))["model"]
        if scenario.startswith("json-"):
            payload = {"json-ok": JSON_OK, "json-no-finish": JSON_NO_FINISH}[scenario]
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if scenario == "slow":
            self._chunk(CONTENT)
            time.sleep(1.0)
            for data in (KEEP_ALIVE, STOP, USAGE, DONE):
                self._chunk(data)
        elif scenario == "no-usage":
            for data in (CONTENT, STOP, DONE):
                self._chunk(data)
        elif scenario == "no-finish":
            for data in (CONTENT, DONE):
                self._chunk(data)
        elif scenario == "drop":
            self._chunk(CONTENT)
            self.close_connection = True
            return  # no terminating chunk: the connection just ends
        elif scenario == "upstream-error":
            for data in (CONTENT, UPSTREAM_ERROR, DONE):
                self._chunk(data)
        elif scenario == "long":
            try:
                for _ in range(50):
                    self._chunk(CONTENT)
                    type(self).sent.append(1)
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                type(self).aborted.set()
                return
        else:
            raise AssertionError(f"unknown scenario {scenario!r}")
        self.wfile.write(b"0\r\n\r\n")


class GateHttpTests(unittest.TestCase):
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

    def _open(self, scenario: str, stream: bool = True):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.gate.server_address[1]}/v1/chat/completions",
            data=json.dumps({"model": scenario, "stream": stream, "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=10)

    def test_first_chunk_arrives_before_upstream_finishes(self) -> None:
        start = time.monotonic()
        resp = self._open("slow")
        first = resp.readline()
        first_at = time.monotonic() - start
        rest = resp.read()
        total = time.monotonic() - start
        self.assertTrue(first.startswith(b"data: "), first)
        self.assertLess(first_at, 0.5, "the first line only arrived with the rest (buffering)")
        self.assertGreaterEqual(total, 1.0)
        self.assertIn(KEEP_ALIVE, rest)
        self.assertTrue(rest.endswith(DONE))
        self.assertNotIn(b"BadCompletionError", first + rest)

    def test_valid_stream_without_usage_is_200_without_error(self) -> None:
        resp = self._open("no-usage")
        body = resp.read()
        self.assertEqual(resp.status, 200)
        self.assertNotIn(b"error", body)
        self.assertEqual(body.count(DONE), 1)
        self.assertTrue(body.endswith(DONE))

    def test_stream_without_finish_gets_terminal_error(self) -> None:
        resp = self._open("no-finish")
        body = resp.read()
        self.assertEqual(resp.status, 200)
        self.assertTrue(body.startswith(CONTENT))
        self.assertIn(b"BadCompletionError", body)
        self.assertLess(body.index(b"BadCompletionError"), body.index(DONE))
        self.assertEqual(body.count(DONE), 1)
        self.assertTrue(body.endswith(DONE))

    def test_upstream_drop_gets_terminal_error(self) -> None:
        resp = self._open("drop")
        body = resp.read()
        self.assertEqual(resp.status, 200)
        self.assertIn(b"BadCompletionError", body)
        self.assertTrue(body.endswith(DONE))

    def test_upstream_error_event_is_forwarded_once(self) -> None:
        resp = self._open("upstream-error")
        body = resp.read()
        self.assertIn(b"CUDA out of memory", body)
        self.assertNotIn(b"BadCompletionError", body)
        self.assertEqual(body.count(DONE), 1)

    def test_client_abort_closes_upstream(self) -> None:
        FakeUpstream.sent.clear()
        FakeUpstream.aborted.clear()
        resp = self._open("long")
        resp.readline()
        resp.close()
        self.assertTrue(FakeUpstream.aborted.wait(3.0), "upstream kept generating after the client left")
        self.assertLess(len(FakeUpstream.sent), 50)

    def test_json_unchanged(self) -> None:
        resp = self._open("json-ok", stream=False)
        self.assertEqual(resp.status, 200)
        self.assertEqual(json.loads(resp.read()), JSON_OK)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._open("json-no-finish", stream=False)
        with ctx.exception as err:
            self.assertEqual(err.code, 502)
            self.assertIn(b"BadCompletionError", err.read())


if __name__ == "__main__":
    unittest.main()
