"""Flow test for issue #19: o caminho heartbeat -> set_upstream.php -> lease -> 503 ou repasse, ponta a ponta.

Sobe o código PHP de verdade (gateway/set_upstream.php, gateway/upstream_lease.php, gateway/upstream_token.php,
gateway/status.php) num `php -S` de 127.0.0.1 numa porta efêmera, com um "upstream" falso em outra porta
efêmera. A rota de completion de produção (api/chat.php) não está neste repo; o ROUTER abaixo é o dublê dela:
chama o mesmo guard (`simpleti_require_live_upstream()`) e depois repassa ao upstream falso com o token por
execução. A chave admin e o token do gate são gerados aqui; nada vai para o repo, e não há rede externa nem produção.
O prazo do lease é curto (SIMPLETI_LEASE_TTL_S=2) só nestes testes.

Caminhos cobertos:
  * (a) sem heartbeat a rota de completion responde 503 com o corpo fixo e Retry-After: 30, e o upstream não é tocado;
  * (b) com heartbeat autenticado (serve_colab.set_upstream, o cliente que o heartbeat usa) a rota repassa ao upstream
    falso com o token por execução;
  * (c) sem novo heartbeat o lease vence e a rota volta a 503; um novo heartbeat reabre a rota;
    serve_colab.heartbeat_loop de verdade mantém a rota aberta além do prazo e, parado, deixa o lease vencer;
  * (d) heartbeat sem token, ou com token errado, é 401 e não renova o lease;
  * sem SIMPLETI_UPSTREAM_FILE (ou com caminho relativo) a rota fica fechada, mesmo com um arquivo de estado
    no diretório temporário: não existe caminho padrão.
A parte PHP precisa de `php` no PATH, o mesmo requisito do `make check`; sem ele a classe é pulada com o motivo.
"""

from __future__ import annotations

import contextlib
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
GATEWAY = ROOT / "gateway"
sys.path.insert(0, str(ROOT / "deploy"))

import serve_colab  # noqa: E402

PHP = shutil.which("php")
URL = "https://flow-test-19.trycloudflare.com"
TTL_S = 2
FIXED_503 = {"error": {"type": "upstream_unavailable", "message": "Simplicio 27B upstream is offline. Retry later."}}
UPSTREAM_REPLY = {"id": "chatcmpl-flow19", "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop"}]}
CHAT = {"model": "simplicio-27b", "max_tokens": 8, "messages": [{"role": "user", "content": "ping"}]}

# Dublê de api/chat.php: o mesmo guard de produção, depois repasse ao upstream falso (FAKE_UPSTREAM_PORT).
ROUTER = r"""<?php
$gw = getenv('GATEWAY_DIR');
$path = parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH);
if ($path === '/api/set_upstream.php') {
    require $gw . '/set_upstream.php';
    return;
}
if ($path === '/v1/status') {
    require $gw . '/status.php';
    return;
}
if ($path === '/v1/chat/completions' && ($_SERVER['REQUEST_METHOD'] ?? '') === 'POST') {
    require_once $gw . '/upstream_lease.php';
    require_once $gw . '/upstream_token.php';
    $live = simpleti_require_live_upstream();
    $token = simpleti_upstream_token(simpleti_state_file());
    $ctx = stream_context_create(['http' => [
        'method' => 'POST',
        'ignore_errors' => true,
        'timeout' => 10,
        'header' => "Content-Type: application/json\r\nX-Simpleti-Upstream-Token: $token\r\nX-Registered-Upstream: $live\r\n",
        'content' => file_get_contents('php://input'),
    ]]);
    $out = file_get_contents('http://127.0.0.1:' . getenv('FAKE_UPSTREAM_PORT') . '/v1/chat/completions', false, $ctx);
    preg_match('#HTTP/\S+ (\d{3})#', $http_response_header[0] ?? '', $m);
    http_response_code((int) ($m[1] ?? 502));
    header('Content-Type: application/json');
    echo $out === false ? '{}' : $out;
    return;
}
http_response_code(404);
"""


class FakeUpstream(BaseHTTPRequestHandler):
    """O Colab visto pelo gateway: só atende quem manda o token por execução."""

    token = ""
    seen: list[dict] = []

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": raw})
        ok = self.headers.get("X-Simpleti-Upstream-Token") == type(self).token and self.path == "/v1/chat/completions"
        body = json.dumps(UPSTREAM_REPLY if ok else {"error": "bad token"}).encode()
        self.send_response(200 if ok else 401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_php(root: Path, env: dict[str, str]) -> tuple[subprocess.Popen, int]:
    router = root / "router.php"
    router.write_text(ROUTER, encoding="utf-8")
    port = free_port()
    proc = subprocess.Popen(
        [PHP, "-S", f"127.0.0.1:{port}", str(router)],
        env={**{k: v for k, v in os.environ.items() if not k.startswith("SIMPLETI_")}, "GATEWAY_DIR": str(GATEWAY), **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10
    while True:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return proc, port
        except OSError:
            if time.monotonic() > deadline:
                proc.kill()
                raise RuntimeError("php -S não subiu")
            time.sleep(0.1)


def call(port: int, method: str, path: str, body: dict | None = None, headers: dict[str, str] | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, raw
    finally:
        conn.close()


@unittest.skipUnless(PHP, "php não instalado: o fluxo do lease precisa de php -S")
class LeaseFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.admin = secrets.token_hex(32)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.state = cls.root / "state" / "upstream.json"
        FakeUpstream.token = serve_colab.GATE_TOKEN
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
        threading.Thread(target=cls.upstream.serve_forever, daemon=True).start()
        cls.php, cls.port = start_php(
            cls.root,
            {
                "SIMPLETI_ADMIN_KEY": cls.admin,
                "SIMPLETI_UPSTREAM_FILE": str(cls.state),
                "SIMPLETI_LEASE_TTL_S": str(TTL_S),
                "FAKE_UPSTREAM_PORT": str(cls.upstream.server_address[1]),
            },
        )
        cls.endpoint = f"http://127.0.0.1:{cls.port}/api/set_upstream.php"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.php.terminate()
        cls.php.wait(timeout=5)
        cls.upstream.shutdown()
        cls.upstream.server_close()
        cls.tmp.cleanup()

    def setUp(self) -> None:
        FakeUpstream.seen.clear()
        if self.state.exists():
            self.state.unlink()

    def heartbeat(self) -> dict:
        """Um beat do serve_colab: o mesmo set_upstream(public_url, key) que o lambda de main() chama."""
        with patch.object(serve_colab, "GATEWAY_SET", self.endpoint):
            return serve_colab.set_upstream(URL, self.admin)

    def completion(self):
        return call(self.port, "POST", "/v1/chat/completions", CHAT)

    def assert_fixed_503(self, status: int, headers: dict[str, str], raw: bytes) -> None:
        self.assertEqual(status, 503)
        self.assertEqual(headers["retry-after"], "30")
        self.assertEqual(headers["content-type"], "application/json; charset=utf-8")
        self.assertEqual(json.loads(raw), FIXED_503)
        for secret in (serve_colab.GATE_TOKEN, self.admin, URL):
            self.assertNotIn(secret.encode(), raw)
            self.assertNotIn(secret, json.dumps(headers))

    def wait_for_503(self, limit_s: float = 10.0) -> float:
        start = time.monotonic()
        while time.monotonic() - start < limit_s:
            status, headers, raw = self.completion()
            if status == 503:
                self.assert_fixed_503(status, headers, raw)
                return time.monotonic() - start
            self.assertEqual(status, 200)
            time.sleep(0.2)
        self.fail(f"o lease de {TTL_S}s não venceu em {limit_s}s")

    # (a) sem heartbeat: 503
    def test_without_heartbeat_the_completion_route_is_503_with_the_fixed_body(self) -> None:
        self.assert_fixed_503(*self.completion())
        self.assertEqual(FakeUpstream.seen, [])

    def test_state_without_updated_at_is_503(self) -> None:
        """Um estado do formato antigo (campo "updated") nunca é um lease vivo."""
        self.state.parent.mkdir(parents=True, exist_ok=True)
        self.state.write_text(
            json.dumps({"upstream_url": URL, "upstream_token": serve_colab.GATE_TOKEN, "updated": int(time.time())})
        )
        self.assert_fixed_503(*self.completion())
        self.assertEqual(FakeUpstream.seen, [])

    # (b) com heartbeat autenticado: repasse
    def test_authenticated_heartbeat_opens_the_route_to_the_fake_upstream(self) -> None:
        self.assert_fixed_503(*self.completion())
        before = int(time.time())
        self.assertEqual(self.heartbeat(), {"ok": True, "upstream": URL})
        saved = json.loads(self.state.read_text())
        self.assertNotIn("updated", saved)
        self.assertIsInstance(saved["updated_at"], int)
        self.assertLessEqual(before, saved["updated_at"])
        self.assertLessEqual(saved["updated_at"], int(time.time()))
        self.assertEqual(self.state.stat().st_mode & 0o777, 0o600)

        status, headers, raw = self.completion()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), UPSTREAM_REPLY)
        self.assertNotIn("retry-after", headers)
        self.assertEqual(len(FakeUpstream.seen), 1)
        seen = FakeUpstream.seen[0]
        self.assertEqual(json.loads(seen["body"]), CHAT)
        self.assertEqual(seen["headers"]["x-simpleti-upstream-token"], serve_colab.GATE_TOKEN)
        self.assertEqual(seen["headers"]["x-registered-upstream"], URL)
        self.assertNotIn(serve_colab.GATE_TOKEN.encode(), raw)

    def test_status_stays_public_and_token_free_while_the_lease_is_live(self) -> None:
        self.heartbeat()
        status, _, raw = call(self.port, "GET", "/v1/status")
        body = json.loads(raw)
        self.assertEqual((status, body["upstream_registered"]), (200, True))
        self.assertRegex(body["updated_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        for secret in (serve_colab.GATE_TOKEN, self.admin, URL):
            self.assertNotIn(secret.encode(), raw)

    # (c) o lease vence e volta a 503; o heartbeat renova
    def test_lease_expires_without_a_new_heartbeat_and_a_new_one_reopens_it(self) -> None:
        self.heartbeat()
        self.assertEqual(self.completion()[0], 200)
        elapsed = self.wait_for_503()
        self.assertLess(elapsed, TTL_S + 3)
        reached = len(FakeUpstream.seen)
        self.assertGreaterEqual(reached, 1)
        self.assert_fixed_503(*self.completion())
        self.assertEqual(len(FakeUpstream.seen), reached)  # vencido: o upstream não é mais tocado
        self.heartbeat()
        self.assertEqual(self.completion()[0], 200)
        self.assertEqual(len(FakeUpstream.seen), reached + 1)

    def test_real_heartbeat_loop_outlives_the_ttl_and_stops_cleanly(self) -> None:
        stop = threading.Event()
        beats: list[object] = []

        def register() -> object:
            beats.append(self.heartbeat())
            return beats[-1]

        loop = threading.Thread(
            target=serve_colab.heartbeat_loop,
            args=(register, {"vllm": lambda: True}, stop, 0.3),
            daemon=True,
        )
        with patch.object(serve_colab, "vllm_healthy", return_value=True):
            loop.start()
            try:
                deadline = time.monotonic() + TTL_S + 1.2  # sem beats o lease já teria vencido (ver o teste acima)
                while time.monotonic() < deadline:
                    if self.state.exists():
                        self.assertEqual(self.completion()[0], 200)
                    time.sleep(0.25)
                self.assertGreaterEqual(len(beats), 5)
            finally:
                stop.set()
                loop.join(timeout=10)
        self.assertFalse(loop.is_alive())
        count = len(beats)
        self.wait_for_503()
        self.assertEqual(len(beats), count)

    def test_clear_closes_the_route_at_once(self) -> None:
        self.heartbeat()
        self.assertEqual(self.completion()[0], 200)
        with patch.object(serve_colab, "GATEWAY_SET", self.endpoint), contextlib.redirect_stdout(io.StringIO()):
            serve_colab.clear_upstream(self.admin)
        self.assertFalse(self.state.exists())
        self.assert_fixed_503(*self.completion())

    # (d) heartbeat sem token: 401
    def test_heartbeat_without_token_is_401_and_stores_nothing(self) -> None:
        body = {"upstream_url": URL, "upstream_token": serve_colab.GATE_TOKEN}
        status, headers, raw = call(self.port, "POST", "/api/set_upstream.php", body)
        self.assertEqual((status, headers["www-authenticate"]), (401, "Bearer"))
        self.assertEqual(json.loads(raw), {"error": "Unauthorized"})
        self.assertFalse(self.state.exists())
        self.assert_fixed_503(*self.completion())

    def test_heartbeat_with_wrong_or_misplaced_token_is_401_and_does_not_renew(self) -> None:
        self.heartbeat()
        first = json.loads(self.state.read_text())
        time.sleep(1.1)  # updated_at tem resolução de 1 s: um beat aceito mudaria o valor
        body = {"upstream_url": "https://other-19.trycloudflare.com", "upstream_token": "B" * 43}
        for label, headers, query in (
            ("chave errada", {"Authorization": "Bearer " + secrets.token_hex(32)}, ""),
            ("sem Bearer", {"Authorization": self.admin}, ""),
            ("Basic", {"Authorization": "Basic " + self.admin}, ""),
            ("chave só na query", {}, "?key=" + self.admin),
            ("token do gate no lugar da chave", {"Authorization": "Bearer " + serve_colab.GATE_TOKEN}, ""),
            ("token do gate no header do gate", {serve_colab.UPSTREAM_HEADER: serve_colab.GATE_TOKEN}, ""),
        ):
            with self.subTest(label):
                status, _, _ = call(self.port, "POST", "/api/set_upstream.php" + query, body, headers)
                self.assertEqual(status, 401)
        self.assertEqual(json.loads(self.state.read_text()), first)

    # sem caminho padrão: fecha
    def test_without_the_state_path_the_route_is_closed_even_with_a_state_file_in_the_temp_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planted = {"upstream_url": URL, "upstream_token": serve_colab.GATE_TOKEN, "updated_at": int(time.time())}
            for name in ("simpleti-upstream.json", "upstream.json", "state.json"):
                (root / name).write_text(json.dumps(planted))
            cases = {
                "sem a variável": {},
                "vazia": {"SIMPLETI_UPSTREAM_FILE": ""},
                "relativa": {"SIMPLETI_UPSTREAM_FILE": "upstream.json"},
            }
            for label, extra in cases.items():
                with self.subTest(label):
                    proc, port = start_php(
                        root,
                        {
                            **extra,
                            "TMPDIR": str(root),
                            "SIMPLETI_LEASE_TTL_S": "3600",
                            "FAKE_UPSTREAM_PORT": str(self.upstream.server_address[1]),
                        },
                    )
                    try:
                        status, headers, raw = call(port, "POST", "/v1/chat/completions", CHAT)
                        self.assert_fixed_503(status, headers, raw)
                    finally:
                        proc.terminate()
                        proc.wait(timeout=5)
        self.assertEqual(FakeUpstream.seen, [])


if __name__ == "__main__":
    unittest.main()
