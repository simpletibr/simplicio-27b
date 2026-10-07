"""Flow test for issue #16: client -> completion gate -> fake vLLM, SSE in real time.

Starts the real gate (serve_colab._GateHandler) on an ephemeral 127.0.0.1 port, in front of a fake
upstream on another ephemeral port. No external network, no vLLM, no credentials.

The streaming checks do not depend on the wall clock. The fake upstream sends one SSE event and then
blocks until the test says it already received that event through the gate. A gate that buffers the
whole body (the old `resp.read()`) never lets the client see the first event, so the upstream's wait
times out and the recorded log shows it.
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

import serve_colab  # noqa: E402


def _event(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


CONTENT_A = _event({"choices": [{"index": 0, "delta": {"content": "po"}, "finish_reason": None}]})
CONTENT_B = _event({"choices": [{"index": 0, "delta": {"content": "ng"}, "finish_reason": None}]})
STOP = _event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
USAGE = _event({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}})
DONE = b"data: [DONE]\n\n"
JSON_OK = {
    "id": "x",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}
WAIT = 3.0  # how long the fake upstream waits for the client before giving up (failure path only)


class FakeUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    ack = threading.Semaphore(0)  # released by the test: "I received event N through the gate"
    release = threading.Event()  # set by the test after it read the first line
    finished = threading.Event()  # set when the upstream sent its last byte
    aborted = threading.Event()  # set when a write to the gate failed (gate closed the upstream)
    log: list[str] = []
    sent = 0

    @classmethod
    def reset(cls) -> None:
        cls.ack = threading.Semaphore(0)
        cls.release = threading.Event()
        cls.finished = threading.Event()
        cls.aborted = threading.Event()
        cls.log = []
        cls.sent = 0

    def log_message(self, *args) -> None:
        pass

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def _sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        scenario = json.loads(self.rfile.read(length))["model"]
        cls = type(self)
        if scenario == "json-ok":
            raw = json.dumps(JSON_OK).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif scenario == "lockstep":
            self._sse_headers()
            for i, data in enumerate((CONTENT_A, CONTENT_B, STOP, USAGE)):
                self._chunk(data)
                cls.log.append(f"sent:{i}")
                got = cls.ack.acquire(timeout=WAIT)
                cls.log.append(f"{'acked' if got else 'timeout'}:{i}")
            self._chunk(DONE)
            self.wfile.write(b"0\r\n\r\n")
            cls.finished.set()
        elif scenario == "pause":
            self._sse_headers()
            self._chunk(CONTENT_A)
            cls.release.wait(WAIT)
            for data in (STOP, USAGE, DONE):
                self._chunk(data)
            self.wfile.write(b"0\r\n\r\n")
            cls.finished.set()
        elif scenario == "long":
            self._sse_headers()
            try:
                for _ in range(50):
                    self._chunk(CONTENT_A)
                    cls.sent += 1
                    threading.Event().wait(0.1)
            except (BrokenPipeError, ConnectionResetError):
                cls.aborted.set()
                return
            self.wfile.write(b"0\r\n\r\n")
        else:
            raise AssertionError(f"unknown scenario {scenario!r}")


class GateFlowTests(unittest.TestCase):
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

    def setUp(self) -> None:
        FakeUpstream.reset()

    def _post(self, scenario: str, stream: bool = True) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection("127.0.0.1", self.gate.server_address[1], timeout=10)
        self.addCleanup(conn.close)
        body = json.dumps({"model": scenario, "stream": stream, "messages": []})
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        return conn.getresponse()

    def test_valid_sse_arrives_in_chunks_with_http_200(self) -> None:
        resp = self._post("lockstep")
        self.assertEqual(resp.status, 200)
        self.assertTrue(resp.getheader("Content-Type", "").startswith("text/event-stream"))
        received = []
        for i in range(4):  # one SSE event = a `data:` line plus a blank line
            line, blank = resp.readline(), resp.readline()
            self.assertTrue(line.startswith(b"data: "), (i, line))
            self.assertEqual(blank, b"\n", (i, blank))
            received.append(line + blank)
            FakeUpstream.ack.release()  # only now may the upstream send event i+1
        rest = resp.read()
        # Interleaved log: the upstream got each ack before it produced the next event. A gate that
        # buffered the body would have left the upstream waiting (`timeout:`) for an ack.
        self.assertEqual(
            FakeUpstream.log,
            ["sent:0", "acked:0", "sent:1", "acked:1", "sent:2", "acked:2", "sent:3", "acked:3"],
        )
        self.assertEqual(received, [CONTENT_A, CONTENT_B, STOP, USAGE])
        self.assertEqual(rest, DONE)  # the gate writes its own [DONE], once, at the end
        self.assertNotIn(b"BadCompletionError", b"".join(received) + rest)

    def test_first_data_line_arrives_before_upstream_finishes(self) -> None:
        resp = self._post("pause")
        first = resp.readline()
        self.assertTrue(first.startswith(b"data: "), first)
        self.assertFalse(
            FakeUpstream.finished.is_set(),
            "the first data: line only reached the client after the upstream finished (buffering)",
        )
        FakeUpstream.release.set()
        rest = resp.read()
        self.assertTrue(FakeUpstream.finished.wait(WAIT))
        self.assertEqual(resp.status, 200)
        self.assertTrue(rest.endswith(DONE))
        self.assertNotIn(b"BadCompletionError", rest)

    def test_valid_non_stream_json_still_passes(self) -> None:
        resp = self._post("json-ok", stream=False)
        body = resp.read()
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "application/json")
        self.assertEqual(resp.getheader("Content-Length"), str(len(body)))
        self.assertEqual(json.loads(body), JSON_OK)

    def test_client_disconnect_closes_upstream(self) -> None:
        resp = self._post("long")
        self.assertTrue(resp.readline().startswith(b"data: "))
        resp.close()
        self.assertTrue(FakeUpstream.aborted.wait(5.0), "upstream kept generating after the client left")
        self.assertLess(FakeUpstream.sent, 50)


if __name__ == "__main__":
    unittest.main()
