"""Issue #21: scripts/live_acceptance.py contra um gateway falso em 127.0.0.1 (porta efêmera).

Sem rede externa, sem produção, sem chave real. O FakeGateway também serve aos testes
test_flow_21 e test_contract_21.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import live_acceptance  # noqa: E402

IDS = [
    "i2-id-simplicio-27b",
    "i2-id-simpleti-simplicio-27b",
    "i2-tool-auto",
    "i2-tool-none-oi",
    "i3-pong",
    "i3-agente-oi-system",
    "i3-agente-oi-tools",
    "i3-segundo-turno",
    "i4-prompt-31692",
    "i4-acima-do-teto",
    "i5-stream-include-usage",
    "i5-stream-sem-options",
    "i5-todo-200-valido",
]
CHECK_KEYS = {"id", "issue", "criterion", "expected", "actual", "passed", "problems", "exchanges"}
XML_TOOL = (
    "<tool_call>\n<function=webfetch>\n<parameter=url>\nhttps://example.com\n</parameter>\n</function>\n</tool_call>"
)


class FakeGateway(BaseHTTPRequestHandler):
    """Gateway OpenAI-compatível mínimo. `mode` escolhe o defeito: good, down, no_finish, xml_content,
    forbidden (403 do Cloudflare), redirect (302 para `redirect_to`) ou echo_key (devolve a chave no content,
    para provar o scrub)."""

    mode = "good"
    token = "test-key"
    redirect_to = ""
    user_agents: list[str] = []
    requests: list[dict] = []

    @classmethod
    def reset(cls, token: str = "test-key") -> None:
        cls.mode = "good"
        cls.token = token
        cls.user_agents.clear()
        cls.requests.clear()

    def log_message(self, *args) -> None:
        pass

    def _send(self, status: int, data: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, obj: dict, headers: dict[str, str] | None = None) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json", headers)

    def do_POST(self) -> None:  # noqa: N802
        cls = FakeGateway
        agent = self.headers.get("User-Agent", "")
        cls.user_agents.append(agent)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                             "body": body})
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "not found"}})
        if cls.mode == "redirect":
            return self._send(302, b"", "text/plain", {"Location": cls.redirect_to})
        if not agent.startswith("simplicio-live-acceptance/") or cls.mode == "forbidden":
            return self._send(403, b"error code: 1010", "text/plain")  # Cloudflare contra o UA do urllib
        if self.headers.get("Authorization") != f"Bearer {cls.token}":
            return self._json(401, {"error": {"message": "missing_api_key"}})
        if cls.mode == "down":
            return self._json(503, {"error": {"message": "upstream down"}}, {"Retry-After": "30"})
        if body.get("model") not in (live_acceptance.MODEL, live_acceptance.ALIAS):
            return self._json(404, {"error": {"message": "model not found"}})
        messages = body.get("messages") or []
        prompt = sum(len(m.get("content") or "") for m in messages) // 4 + 8
        if prompt + body.get("max_tokens", 0) > live_acceptance.load_context()["MAX_MODEL_LEN"]:
            return self._json(400, {"error": {"message": "This model's maximum context length is 40960 tokens."}})

        last = messages[-1] if messages else {}
        last_content = last.get("content") or ""
        content: str | None
        tool_calls = None
        finish: str | None = "stop"
        if last.get("role") == "tool":
            content = "O título é Example Domain."
        elif body.get("tool_choice") == "auto" and "webfetch" in last_content:
            if cls.mode == "xml_content":
                content = XML_TOOL
            else:
                content = None
                finish = "tool_calls"
                tool_calls = [{"id": "call_1", "type": "function", "function": {
                    "name": "webfetch", "arguments": "{\"url\": \"https://example.com\"}"}}]
        elif "pong" in last_content:
            content = "pong"
        else:
            content = "Olá! Como posso ajudar?"
        if cls.mode == "echo_key" and content:
            content += f" (chave {cls.token})"
        usage = {"prompt_tokens": prompt, "completion_tokens": 5, "total_tokens": prompt + 5}
        if cls.mode == "no_finish":
            finish = None

        if body.get("stream"):
            events: list = [
                {"choices": [{"index": 0, "delta": {"content": content or ""}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            ]
            if cls.mode != "no_finish":
                events.append({"choices": [], "usage": usage})
            data = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
            return self._send(200, data.encode("utf-8"), "text/event-stream")

        message: dict = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        reply: dict = {"id": "chatcmpl-fake", "object": "chat.completion", "model": body["model"],
                       "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
        if cls.mode != "no_finish":
            reply["usage"] = usage
        self._json(200, reply)


class LiveAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeGateway.reset()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGateway)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        FakeGateway.reset()

    def run_script(self, *extra: str) -> tuple[int, dict | None, str | None]:
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"SIMPLETI_API_KEY": "test-key"}, clear=True), \
                contextlib.redirect_stdout(io.StringIO()):
            code = live_acceptance.main(["--base-url", self.base_url, "--out-dir", tmp, *extra])
            files = sorted(Path(tmp).glob("acceptance-*.json"))
            text = files[0].read_text(encoding="utf-8") if files else None
        return code, (json.loads(text) if text else None), text

    @staticmethod
    def check(receipt: dict, check_id: str) -> dict:
        return next(c for c in receipt["checks"] if c["id"] == check_id)

    def test_all_checks_pass_against_fake(self) -> None:
        code, receipt, _ = self.run_script()
        self.assertEqual(code, 0)
        assert receipt is not None
        self.assertTrue(receipt["passed"])
        self.assertEqual([c["id"] for c in receipt["checks"]], IDS)
        for chk in receipt["checks"]:
            self.assertEqual(set(chk), CHECK_KEYS, chk["id"])
            self.assertTrue(chk["passed"], (chk["id"], chk["problems"]))

    def test_key_never_written(self) -> None:
        _, _, text = self.run_script()
        assert text is not None
        self.assertNotIn("test-key", text)

    def test_user_agent_is_custom(self) -> None:
        self.run_script()
        self.assertTrue(FakeGateway.user_agents)
        for agent in FakeGateway.user_agents:
            self.assertTrue(agent.startswith("simplicio-live-acceptance/"), agent)

    def test_long_prompts(self) -> None:
        _, receipt, text = self.run_script()
        assert receipt is not None and text is not None
        self.assertGreaterEqual(self.check(receipt, "i4-prompt-31692")["actual"]["usage"]["prompt_tokens"], 31692)
        self.assertEqual(self.check(receipt, "i4-acima-do-teto")["actual"]["status"], 400)
        self.assertNotIn(json.dumps(live_acceptance.FILLER * 10, ensure_ascii=False)[1:-1], text)
        self.assertIn("caracteres omitidos", text)

    def test_missing_finish_fails(self) -> None:
        FakeGateway.mode = "no_finish"
        code, receipt, _ = self.run_script()
        self.assertEqual(code, 1)
        assert receipt is not None
        self.assertFalse(self.check(receipt, "i5-todo-200-valido")["passed"])

    def test_tool_xml_in_content_fails(self) -> None:
        FakeGateway.mode = "xml_content"
        code, receipt, _ = self.run_script()
        self.assertEqual(code, 1)
        assert receipt is not None
        self.assertFalse(self.check(receipt, "i2-tool-auto")["passed"])

    def test_missing_key_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), \
                contextlib.redirect_stderr(io.StringIO()):
            code = live_acceptance.main(["--base-url", self.base_url, "--out-dir", tmp])
            self.assertEqual(list(Path(tmp).glob("acceptance-*.json")), [])
        self.assertEqual(code, 2)

    def test_down_phase(self) -> None:
        FakeGateway.mode = "down"
        code, receipt, _ = self.run_script("--phase", "down")
        self.assertEqual(code, 0)
        assert receipt is not None
        self.assertEqual(self.check(receipt, "i5-upstream-fora")["actual"]["retry_after"], "30")

    def test_down_phase_fails_when_up(self) -> None:
        code, _, _ = self.run_script("--phase", "down")
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
