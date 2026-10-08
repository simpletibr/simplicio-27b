"""Flow test for issue #18: o caminho gateway -> gate -> vLLM, ponta a ponta, sem rede externa e sem produção.

Sobe o gate de verdade (serve_colab._GateHandler) numa porta efêmera de 127.0.0.1, na frente de um vLLM
falso em outra porta efêmera. O vLLM falso imita o vLLM com VLLM_API_KEY: exige `Authorization: Bearer
<VLLM_TOKEN>` em /v1/* e deixa /metrics, /health, /docs, /tokenize e /invocations abertos (é o que o vLLM
0.31.0 faz, e é por isso que o gate tem allowlist). Os tokens vêm de serve_colab (secrets, gerados a cada
execução); a chave do usuário e a chave admin são geradas aqui. Nada disso vai para o repo.

Caminhos cobertos:
  * sem o token de upstream (ou com token errado/malformado): 401, o vLLM não é tocado;
  * com o token certo: 200 e o corpo do vLLM chega intacto (JSON e SSE);
  * rota ou método fora da allowlist, mesmo com o token certo: 404 (ou recusado), o vLLM não é tocado,
    inclusive as rotas que o vLLM deixa abertas;
  * o vLLM recebe `Authorization: Bearer <VLLM_TOKEN>` (o do gate, não o do usuário) e nunca o header do
    token de upstream; o cliente não vê nenhum dos tokens na resposta, nem em erro 401/404/502;
  * a cadeia do dono: serve_colab.set_upstream() -> set_upstream.php (php -S) -> arquivo de estado 0600 ->
    upstream_token.php lê o token -> o "proxy" manda o header -> o gate deixa passar; {"clear": true} apaga.
A parte PHP precisa de `php` no PATH, o mesmo requisito do `make check`; sem ele a classe é pulada com o motivo.
"""

from __future__ import annotations

import contextlib
import hmac
import http.client
import io
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

import serve_colab  # noqa: E402

PHP = shutil.which("php")
USER_KEY = "user-" + secrets.token_urlsafe(24)
OPEN_MARK = "OPEN-ROUTE-REACHED-VLLM"
MODELS = {"object": "list", "data": [{"id": "simplicio-27b", "object": "model"}]}
OK_CHAT = {
    "id": "chatcmpl-flow18",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}
SSE = [
    b'data: {"choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": null}]}\n\n',
    b'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n',
    b'data: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}\n\n',
    b"data: [DONE]\n\n",
]


class FakeVllm(BaseHTTPRequestHandler):
    """vLLM com VLLM_API_KEY: /v1/* exige a chave; as outras rotas ficam abertas."""

    key = ""
    seen: list[dict] = []

    def log_message(self, *args) -> None:
        pass

    def _send(self, status: int, ctype: str, raw: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _handle(self, method: str) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).seen.append(
            {"method": method, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": raw}
        )
        route = self.path.split("?", 1)[0]
        if route.startswith("/v1") and self.headers.get("Authorization") != f"Bearer {type(self).key}":
            self._send(401, "application/json", b'{"error": "Unauthorized"}')
        elif (method, route) == ("GET", "/v1/models"):
            self._send(200, "application/json", json.dumps(MODELS).encode())
        elif (method, route) == ("POST", "/v1/chat/completions"):
            if json.loads(raw or b"{}").get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for line in SSE:
                    self.wfile.write(line)
                    self.wfile.flush()
                self.close_connection = True
            else:
                self._send(200, "application/json", json.dumps(OK_CHAT).encode())
        else:
            self._send(200, "text/plain", OPEN_MARK.encode())

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")


def serve(handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def request(
    port: int, method: str, path: str, headers: dict[str, str] | None = None, body: dict | None = None
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, raw
    finally:
        conn.close()


class GateFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        FakeVllm.key = serve_colab.VLLM_TOKEN
        cls.vllm = serve(FakeVllm)
        cls.old_port = serve_colab.VLLM_PORT
        serve_colab.VLLM_PORT = cls.vllm.server_address[1]
        cls.gate = serve(serve_colab._GateHandler)
        cls.port = cls.gate.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        for server in (cls.gate, cls.vllm):
            server.shutdown()
            server.server_close()
        serve_colab.VLLM_PORT = cls.old_port

    def setUp(self) -> None:
        FakeVllm.seen.clear()

    def gate_call(self, method: str, path: str, token: str | None = None, body: dict | None = None, **headers):
        sent = dict(headers)
        if token is not None:
            sent[serve_colab.UPSTREAM_HEADER] = token
        return request(self.port, method, path, sent, body)

    def assert_no_secret(self, status_headers: dict[str, str], raw: bytes) -> None:
        blob = json.dumps(status_headers).encode() + raw
        for secret in (serve_colab.VLLM_TOKEN, serve_colab.GATE_TOKEN, USER_KEY):
            self.assertNotIn(secret.encode(), blob)

    def test_fake_vllm_really_requires_its_key(self) -> None:
        """Sem isto os testes de repasse seriam vazios: o vLLM falso recusa o que não traz a chave dele."""
        port = self.vllm.server_address[1]
        self.assertEqual(request(port, "GET", "/v1/models")[0], 401)
        self.assertEqual(request(port, "GET", "/v1/models", {"Authorization": f"Bearer {USER_KEY}"})[0], 401)
        self.assertEqual(request(port, "GET", "/metrics")[2].decode(), OPEN_MARK)
        FakeVllm.seen.clear()

    # (a) sem o token: 401
    def test_without_token_is_401_and_vllm_is_untouched(self) -> None:
        for method, path in (("GET", "/v1/models"), ("POST", "/v1/chat/completions"), ("GET", "/metrics")):
            with self.subTest(method=method, path=path):
                status, headers, raw = self.gate_call(
                    method, path, body={"model": "simplicio-27b", "messages": []} if method == "POST" else None
                )
                self.assertEqual(status, 401)
                self.assertTrue(headers["content-type"].startswith("application/json"), headers)
                self.assertEqual(
                    json.loads(raw),
                    {"error": {"message": "missing or invalid upstream token", "type": "GateError"}},
                )
                self.assert_no_secret(headers, raw)
        self.assertEqual(FakeVllm.seen, [])

    def test_malformed_or_misplaced_token_is_401(self) -> None:
        good = serve_colab.GATE_TOKEN
        for label, token, extra in (
            ("vazio", "", {}),
            ("prefixo", good[:-1], {}),
            ("com sufixo", good + "x", {}),
            ("minúsculas", good.swapcase(), {}),
            ("não ASCII", "é" * 43, {}),
            ("token do vLLM", serve_colab.VLLM_TOKEN, {}),
            ("só no Authorization", None, {"Authorization": f"Bearer {good}"}),
        ):
            with self.subTest(label):
                status, headers, raw = self.gate_call(
                    "POST", "/v1/chat/completions", token, {"model": "simplicio-27b", "messages": []}, **extra
                )
                self.assertEqual(status, 401)
                self.assert_no_secret(headers, raw)
        self.assertEqual(FakeVllm.seen, [])

    def test_garbage_content_length_is_still_a_401(self) -> None:
        """Antes da autenticação, um Content-Length inválido (inclusive "²", que str.isdigit() aceita) não derruba o handler."""
        for value in ("abc", "-5", "²", "1.5"):
            with self.subTest(value=value):
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
                try:
                    conn.putrequest("POST", "/v1/chat/completions")
                    conn.putheader("Content-Length", value)
                    conn.endheaders()
                    resp = conn.getresponse()
                    resp.read()
                    self.assertEqual(resp.status, 401)
                finally:
                    conn.close()
        self.assertEqual(FakeVllm.seen, [])

    def test_token_is_compared_with_hmac_compare_digest(self) -> None:
        with patch.object(hmac, "compare_digest", wraps=hmac.compare_digest) as spy:
            self.gate_call("GET", "/v1/models", serve_colab.GATE_TOKEN)
        spy.assert_called_once_with(serve_colab.GATE_TOKEN.encode(), serve_colab.GATE_TOKEN.encode())

    # (b) com o token certo: 200 e repasse
    def test_right_token_forwards_chat_and_models(self) -> None:
        sent = {"model": "simplicio-27b", "max_tokens": 8, "messages": [{"role": "user", "content": "ping"}]}
        status, _, raw = self.gate_call("POST", "/v1/chat/completions", serve_colab.GATE_TOKEN, sent)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), OK_CHAT)
        status, _, raw = self.gate_call("GET", "/v1/models", serve_colab.GATE_TOKEN)
        self.assertEqual((status, json.loads(raw)), (200, MODELS))
        self.assertEqual([(s["method"], s["path"]) for s in FakeVllm.seen],
                         [("POST", "/v1/chat/completions"), ("GET", "/v1/models")])
        self.assertEqual(json.loads(FakeVllm.seen[0]["body"]), sent)

    def test_query_string_is_forwarded_but_the_route_is_exact(self) -> None:
        status, _, _ = self.gate_call("GET", "/v1/models?limit=1", serve_colab.GATE_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(FakeVllm.seen[-1]["path"], "/v1/models?limit=1")

    def test_stream_is_gated_and_forwarded(self) -> None:
        body = {"model": "simplicio-27b", "stream": True, "messages": []}
        status, _, raw = self.gate_call("POST", "/v1/chat/completions", None, body)
        self.assertEqual((status, FakeVllm.seen), (401, []))
        self.assertNotIn(b"data:", raw)
        status, headers, raw = self.gate_call("POST", "/v1/chat/completions", serve_colab.GATE_TOKEN, body)
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["content-type"])
        self.assertTrue(raw.startswith(SSE[0]), raw)
        self.assertTrue(raw.endswith(b"data: [DONE]\n\n"), raw)
        self.assertEqual(FakeVllm.seen[0]["headers"]["authorization"], f"Bearer {serve_colab.VLLM_TOKEN}")
        self.assert_no_secret(headers, raw)

    # (c) rota fora da allowlist: 404
    def test_routes_outside_the_allowlist_are_404_even_with_the_right_token(self) -> None:
        body = {"model": "simplicio-27b", "messages": []}
        for method, path in (
            ("GET", "/metrics"),
            ("GET", "/health"),
            ("GET", "/docs"),
            ("GET", "/openapi.json"),
            ("POST", "/tokenize"),
            ("POST", "/invocations"),
            ("POST", "/v1/completions"),
            ("POST", "/v1/embeddings"),
            ("GET", "/v1/chat/completions"),
            ("POST", "/v1/models"),
            ("GET", "/v1/models/"),
            ("POST", "/v1/chat/completions/"),
            ("GET", "/V1/models"),
            ("GET", "/v1/models/../../metrics"),
            ("GET", "/v1/models/%2e%2e/metrics"),
            ("GET", "//metrics"),
            ("GET", "/v1/models#/../metrics"),
            ("GET", "http://127.0.0.1/metrics"),
        ):
            with self.subTest(method=method, path=path):
                status, headers, raw = self.gate_call(method, path, serve_colab.GATE_TOKEN, body if method == "POST" else None)
                self.assertEqual(status, 404)
                self.assertNotIn(OPEN_MARK.encode(), raw)
                self.assertEqual(
                    json.loads(raw), {"error": {"message": "not found", "type": "GateError"}}
                )
                self.assert_no_secret(headers, raw)
        self.assertEqual(FakeVllm.seen, [])

    def test_other_methods_never_reach_vllm(self) -> None:
        for method in ("PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
            with self.subTest(method=method):
                status, _, _ = self.gate_call(method, "/v1/models", serve_colab.GATE_TOKEN)
                self.assertGreaterEqual(status, 400)
        self.assertEqual(FakeVllm.seen, [])

    # (d) o vLLM recebe o token dele; o cliente não vê token nenhum
    def test_vllm_gets_its_own_token_and_the_client_sees_none(self) -> None:
        status, headers, raw = self.gate_call(
            "POST",
            "/v1/chat/completions",
            serve_colab.GATE_TOKEN,
            {"model": "simplicio-27b", "messages": []},
            Authorization=f"Bearer {USER_KEY}",
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(FakeVllm.seen), 1)
        upstream = FakeVllm.seen[0]["headers"]
        self.assertEqual(upstream["authorization"], f"Bearer {serve_colab.VLLM_TOKEN}")
        self.assertNotIn("x-simpleti-upstream-token", upstream)
        self.assertNotIn(USER_KEY, json.dumps(upstream))
        self.assertNotIn(serve_colab.GATE_TOKEN, json.dumps(upstream))
        self.assert_no_secret(headers, raw)

    def test_upstream_failure_does_not_leak_a_token(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            closed = sock.getsockname()[1]
        with patch.object(serve_colab, "VLLM_PORT", closed):
            status, headers, raw = self.gate_call(
                "POST", "/v1/chat/completions", serve_colab.GATE_TOKEN, {"model": "simplicio-27b", "messages": []},
                Authorization=f"Bearer {USER_KEY}",
            )
        self.assertEqual(status, 502)
        self.assert_no_secret(headers, raw)


@unittest.skipUnless(PHP, "php não instalado: a cadeia de registro precisa de php -S")
class RegistrationChainFlowTests(unittest.TestCase):
    URL = "https://flow-test-18.trycloudflare.com"

    @classmethod
    def setUpClass(cls) -> None:
        cls.admin = secrets.token_hex(32)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.state = Path(cls.tmp.name) / "state" / "upstream.json"
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        php_port = sock.getsockname()[1]
        sock.close()
        cls.php = subprocess.Popen(
            [PHP, "-S", f"127.0.0.1:{php_port}", str(ROOT / "gateway" / "set_upstream.php")],
            env={**os.environ, "SIMPLETI_ADMIN_KEY": cls.admin, "SIMPLETI_UPSTREAM_FILE": str(cls.state)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cls.endpoint = f"http://127.0.0.1:{php_port}/set_upstream.php"
        deadline = time.monotonic() + 10
        while True:
            try:
                socket.create_connection(("127.0.0.1", php_port), timeout=1).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    cls.php.kill()
                    cls.tmp.cleanup()
                    raise RuntimeError("php -S não subiu")
                time.sleep(0.1)
        FakeVllm.key = serve_colab.VLLM_TOKEN
        cls.vllm = serve(FakeVllm)
        cls.old_port = serve_colab.VLLM_PORT
        serve_colab.VLLM_PORT = cls.vllm.server_address[1]
        cls.gate = serve(serve_colab._GateHandler)

    @classmethod
    def tearDownClass(cls) -> None:
        for server in (cls.gate, cls.vllm):
            server.shutdown()
            server.server_close()
        serve_colab.VLLM_PORT = cls.old_port
        cls.php.terminate()
        cls.php.wait(timeout=5)
        cls.tmp.cleanup()

    def setUp(self) -> None:
        FakeVllm.seen.clear()

    def php_eval(self, code: str, *args: str) -> str:
        return subprocess.run([PHP, "-r", code, *args], capture_output=True, text=True, check=True).stdout

    def stored_token(self):
        out = self.php_eval(
            "require $argv[1]; echo json_encode(simpleti_upstream_token($argv[2]));",
            str(ROOT / "gateway" / "upstream_token.php"),
            str(self.state),
        )
        return json.loads(out)

    def test_register_open_gate_clear(self) -> None:
        with patch.object(serve_colab, "GATEWAY_SET", self.endpoint):
            reply = serve_colab.set_upstream(self.URL, self.admin)
            self.assertEqual(reply, {"ok": True, "upstream": self.URL})
            self.assertNotIn(serve_colab.GATE_TOKEN, json.dumps(reply))
            self.assertEqual(self.state.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(self.state.read_text())["upstream_token"], serve_colab.GATE_TOKEN)

            # O "proxy" lê o token pelo helper PHP e o manda no header; o gate deixa passar.
            token = self.stored_token()
            self.assertEqual(token, serve_colab.GATE_TOKEN)
            status, _, raw = request(
                self.gate.server_address[1], "GET", "/v1/models", {serve_colab.UPSTREAM_HEADER: token}
            )
            self.assertEqual((status, json.loads(raw)), (200, MODELS))
            self.assertEqual(FakeVllm.seen[0]["headers"]["authorization"], f"Bearer {serve_colab.VLLM_TOKEN}")

            # GET /v1/status continua público e não carrega nem o token nem a URL do túnel.
            status_out = self.php_eval(
                "putenv('SIMPLETI_UPSTREAM_FILE=' . $argv[1]); require $argv[2];",
                str(self.state),
                str(ROOT / "gateway" / "status.php"),
            )
            self.assertIs(json.loads(status_out)["upstream_registered"], True)
            self.assertNotIn(serve_colab.GATE_TOKEN, status_out)
            self.assertNotIn(self.URL, status_out)

            with contextlib.redirect_stdout(io.StringIO()):
                serve_colab.clear_upstream(self.admin)
        self.assertFalse(self.state.exists())
        self.assertIsNone(self.stored_token())

    def test_session_without_token_cannot_register(self) -> None:
        status, payload = serve_colab._http_json(
            self.endpoint, {"upstream_url": self.URL}, headers=serve_colab.admin_headers(self.admin)
        )
        self.assertEqual((status, payload), (400, {"error": "invalid upstream_token"}))
        self.assertFalse(self.state.exists())

    def test_wrong_admin_key_stores_nothing(self) -> None:
        with patch.object(serve_colab, "GATEWAY_SET", self.endpoint):
            with self.assertRaises(RuntimeError) as ctx:
                serve_colab.set_upstream(self.URL, secrets.token_hex(32))
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn(serve_colab.GATE_TOKEN, str(ctx.exception))
        self.assertFalse(self.state.exists())


if __name__ == "__main__":
    unittest.main()
