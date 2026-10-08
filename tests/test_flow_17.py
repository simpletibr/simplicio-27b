"""Flow test for issue #17: GET /v1/status ponta a ponta, sem rede externa e sem produção.

Sobe `php -S` numa porta efêmera de 127.0.0.1 com cópias de gateway/status.php e
gateway/set_upstream.php dentro de um diretório temporário (o `DEPLOY_SHA` que o dono grava no deploy
fica ao lado da cópia de status.php, como no servidor). Um `router.php` do teste faz o papel da regra
`/v1/status -> api/status.php` do servidor (a regra de verdade é do .htaccess ou da Cloudflare e não
está no repo). A chave admin é gerada em memória a cada execução e nunca é escrita em arquivo.

Caminhos cobertos:
  * GET /v1/status sem estado: HTTP 200, JSON com as três chaves fixas e valores vazios;
  * POST em set_upstream.php (Bearer) e depois GET /v1/status: upstream_registered=true e updated_at em
    ISO 8601 UTC, sem a URL do túnel, sem o host e sem a chave no corpo; depois {"clear": true} volta a false;
  * DEPLOY_SHA válido aparece em deploy_sha; inválido vira null; arquivo de estado corrompido não vaza erro;
  * método diferente de GET é recusado com 405, Allow: GET, e não altera o estado;
  * ?key=<chave> na consulta não é ecoado.
Precisa de `php` no PATH, o mesmo requisito do `make check`; sem ele a classe é pulada com o motivo.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"upstream_registered", "updated_at", "deploy_sha"}
TUNNEL = "https://flow-test-17.trycloudflare.com"
ROUTER = """<?php
$path = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);
if ($path === '/v1/status') {
    require __DIR__ . '/api/status.php';
    return true;
}
if ($path === '/api/set_upstream.php') {
    require __DIR__ . '/api/set_upstream.php';
    return true;
}
http_response_code(404);
echo '{"error":"not found"}';
"""


@unittest.skipUnless(shutil.which("php"), "php não instalado: o teste de fluxo precisa de php -S")
class StatusFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.admin_key = secrets.token_hex(32)
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.api = base / "api"
        cls.api.mkdir()
        cls.state = base / "state" / "upstream.json"
        for name in ("status.php", "set_upstream.php"):
            shutil.copy(ROOT / "gateway" / name, cls.api / name)
        (base / "router.php").write_text(ROUTER, encoding="utf-8")
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        cls.port = sock.getsockname()[1]
        sock.close()
        cls.proc = subprocess.Popen(
            ["php", "-S", f"127.0.0.1:{cls.port}", str(base / "router.php")],
            env={
                **os.environ,
                "SIMPLETI_ADMIN_KEY": cls.admin_key,
                "SIMPLETI_UPSTREAM_FILE": str(cls.state),
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 10
        while True:
            try:
                socket.create_connection(("127.0.0.1", cls.port), timeout=1).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    cls.proc.kill()
                    cls.tmp.cleanup()
                    raise RuntimeError("php -S não subiu")
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proc.terminate()
        cls.proc.wait(timeout=5)
        cls.tmp.cleanup()

    def setUp(self) -> None:
        self.state.unlink(missing_ok=True)
        (self.api / "DEPLOY_SHA").unlink(missing_ok=True)

    def call(self, method: str, path: str, body=None, auth: str | None = None):
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read().decode()
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read().decode()

    def register(self, url: str = TUNNEL) -> None:
        status, _, text = self.call(
            "POST",
            "/api/set_upstream.php",
            {"upstream_url": url, "upstream_token": "A" * 43},
            auth=f"Bearer {self.admin_key}",
        )
        self.assertEqual(status, 200, text)

    def assert_clean(self, text: str) -> None:
        for leak in (self.admin_key, "trycloudflare", "flow-test-17", "upstream_url", str(self.state), "Fatal", "Warning"):
            self.assertNotIn(leak, text)

    def test_get_without_state(self) -> None:
        status, headers, text = self.call("GET", "/v1/status")
        self.assertEqual(status, 200, text)
        self.assertTrue(headers["content-type"].startswith("application/json"), headers)
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertEqual(
            json.loads(text), {"upstream_registered": False, "updated_at": None, "deploy_sha": None}
        )
        self.assert_clean(text)

    def test_register_then_status_then_clear(self) -> None:
        self.register()
        status, _, text = self.call("GET", "/v1/status")
        self.assertEqual(status, 200, text)
        body = json.loads(text)
        self.assertEqual(set(body), KEYS)
        self.assertIs(body["upstream_registered"], True)
        self.assertRegex(body["updated_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        stamp = datetime.strptime(body["updated_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        self.assertLess(abs((datetime.now(timezone.utc) - stamp).total_seconds()), 120)
        self.assertIsNone(body["deploy_sha"])
        self.assert_clean(text)

        status, _, text = self.call(
            "POST", "/api/set_upstream.php", {"clear": True}, auth=f"Bearer {self.admin_key}"
        )
        self.assertEqual(status, 200, text)
        status, _, text = self.call("GET", "/v1/status")
        self.assertEqual(status, 200, text)
        self.assertEqual(
            json.loads(text), {"upstream_registered": False, "updated_at": None, "deploy_sha": None}
        )

    def test_deploy_sha_valid_and_invalid(self) -> None:
        sha = "0123456789abcdef" * 2 + "01234567"
        (self.api / "DEPLOY_SHA").write_text(sha + "\n", encoding="utf-8")
        _, _, text = self.call("GET", "/v1/status")
        self.assertEqual(json.loads(text)["deploy_sha"], sha)
        (self.api / "DEPLOY_SHA").write_text("not-a-sha\n", encoding="utf-8")
        _, _, text = self.call("GET", "/v1/status")
        self.assertIsNone(json.loads(text)["deploy_sha"])

    def test_corrupt_state_file_does_not_leak(self) -> None:
        self.state.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.state.write_text("{not json", encoding="utf-8")
        status, _, text = self.call("GET", "/v1/status")
        self.assertEqual(status, 200, text)
        self.assertIs(json.loads(text)["upstream_registered"], False)
        self.assert_clean(text)

    def test_non_get_is_refused_and_changes_nothing(self) -> None:
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            with self.subTest(method=method):
                status, headers, text = self.call(
                    method, "/v1/status", {"upstream_url": TUNNEL}, auth=f"Bearer {self.admin_key}"
                )
                self.assertEqual(status, 405, text)
                self.assertEqual(headers["allow"], "GET")
                self.assertEqual(json.loads(text), {"error": "method not allowed"})
                self.assert_clean(text)
        self.assertFalse(self.state.exists())
        _, _, text = self.call("GET", "/v1/status")
        self.assertIs(json.loads(text)["upstream_registered"], False)

    def test_query_key_is_not_echoed(self) -> None:
        self.register()
        status, _, text = self.call("GET", f"/v1/status?key={self.admin_key}")
        self.assertEqual(status, 200, text)
        self.assertEqual(set(json.loads(text)), KEYS)
        self.assert_clean(text)


if __name__ == "__main__":
    unittest.main()
