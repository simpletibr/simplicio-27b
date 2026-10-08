"""Contract test for issue #19: as interfaces do lease que o proxy, o serve_colab.py e o /v1/status consomem.

  * o 503 do lease tem o corpo e os headers que a issue lista: Content-Type `application/json; charset=utf-8`,
    `Retry-After: 30` e o corpo `{"error": {"type": "upstream_unavailable", "message": "Simplicio 27B upstream is
    offline. Retry later."}}`, para cada jeito de o upstream estar fora (sem arquivo, vencido, formato antigo, JSON
    quebrado, sem a variável de caminho);
  * a fronteira do prazo: vivo até TTL s depois do beat, vencido em TTL + 1; TTL padrão 90, configurável por
    SIMPLETI_LEASE_TTL_S (inteiro positivo; qualquer outro valor cai no padrão); o padrão é pelo menos 3 beats
    do HEARTBEAT_S do serve_colab e o Retry-After é um beat;
  * o lease não toca no segredo: upstream_lease.php nunca lê `upstream_token`, nada do que ele responde (503) ou do
    que o heartbeat imprime carrega o token do gate, a chave admin ou a URL do túnel, e o arquivo de estado continua 0600;
  * o caminho do estado vem só de SIMPLETI_UPSTREAM_FILE, absoluto, sem padrão e nunca no diretório temporário
    (mesma regra da #18); o campo é `updated_at`, e o nome antigo não existe mais em gateway/ nem em tests/test_gateway.py;
  * o heartbeat reusa a autenticação existente: o mesmo set_upstream() com `Authorization: Bearer <chave admin>`.
Sem rede externa e sem produção. As partes PHP precisam de `php` no PATH, o mesmo requisito do `make check`.
"""

from __future__ import annotations

import ast
import contextlib
import http.client
import io
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GATEWAY = ROOT / "gateway"
LEASE = GATEWAY / "upstream_lease.php"
SERVE_PY = ROOT / "deploy" / "serve_colab.py"
sys.path.insert(0, str(ROOT / "deploy"))

import serve_colab  # noqa: E402

PHP = shutil.which("php")
URL = "https://contract-19.trycloudflare.com"
TOKEN = "T" * 43
BODY_503 = '{"error":{"type":"upstream_unavailable","message":"Simplicio 27B upstream is offline. Retry later."}}'
GUARD = "<?php\nrequire getenv('LEASE_LIB');\necho 'proxied to ' . simpleti_require_live_upstream();\n"


def php_eval(code: str, *args: str, env: dict[str, str | None] | None = None) -> str:
    full = {**os.environ}
    for key, value in (env or {}).items():
        if value is None:
            full.pop(key, None)
        else:
            full[key] = value
    return subprocess.run(
        [PHP, "-r", code, *args], capture_output=True, text=True, check=True, timeout=30, env=full
    ).stdout


def live_upstream(state: object, now: int, ttl: str | None = None):
    out = php_eval(
        "require $argv[1]; echo json_encode(simpleti_live_upstream(json_decode($argv[2], true), (int) $argv[3]));",
        str(LEASE), json.dumps(state), str(now),
        env={"SIMPLETI_LEASE_TTL_S": ttl},
    )
    return json.loads(out)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@unittest.skipUnless(PHP, "php não instalado")
class Http503ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.state = root / "state" / "upstream.json"
        (root / "router.php").write_text(GUARD, encoding="utf-8")
        cls.servers: dict[str, tuple[subprocess.Popen, int]] = {}
        base = {k: v for k, v in os.environ.items() if not k.startswith("SIMPLETI_")}
        for name, extra in (("configured", {"SIMPLETI_UPSTREAM_FILE": str(cls.state)}), ("unconfigured", {})):
            port = free_port()
            proc = subprocess.Popen(
                [PHP, "-S", f"127.0.0.1:{port}", str(root / "router.php")],
                env={**base, "LEASE_LIB": str(LEASE), "TMPDIR": str(root), **extra},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            cls.servers[name] = (proc, port)
        deadline = time.monotonic() + 10
        for _, port in cls.servers.values():
            while True:
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=1).close()
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        cls.tearDownClass()
                        raise RuntimeError("php -S não subiu")
                    time.sleep(0.1)

    @classmethod
    def tearDownClass(cls) -> None:
        for proc, _ in cls.servers.values():
            proc.terminate()
            proc.wait(timeout=5)
        cls.tmp.cleanup()

    def setUp(self) -> None:
        if self.state.exists():
            self.state.unlink()

    def get(self, which: str = "configured"):
        conn = http.client.HTTPConnection("127.0.0.1", self.servers[which][1], timeout=15)
        try:
            conn.request("GET", "/v1/chat/completions")
            resp = conn.getresponse()
            return resp.status, resp.getheaders(), resp.read().decode()
        finally:
            conn.close()

    def write_state(self, state: object | str) -> None:
        self.state.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.state.write_text(state if isinstance(state, str) else json.dumps(state), encoding="utf-8")

    def assert_issue_503(self, status: int, headers: list[tuple[str, str]], text: str) -> None:
        self.assertEqual(status, 503)
        names = [k.lower() for k, _ in headers]
        self.assertEqual(names.count("retry-after"), 1)
        self.assertEqual(names.count("content-type"), 1)
        sent = {k.lower(): v for k, v in headers}
        self.assertEqual(sent["retry-after"], "30")
        self.assertEqual(sent["content-type"], "application/json; charset=utf-8")
        self.assertEqual(text, BODY_503)
        for secret in (TOKEN, URL):
            self.assertNotIn(secret, text)
            self.assertNotIn(secret, json.dumps(headers))

    def test_every_way_of_being_down_gives_the_same_503(self) -> None:
        now = int(time.time())
        stale = {"upstream_url": URL, "upstream_token": TOKEN, "updated_at": now - 3600}
        cases = {
            "sem arquivo": None,
            "vencido": stale,
            "no futuro (relógio recuado)": {"upstream_url": URL, "upstream_token": TOKEN, "updated_at": now + 3600},
            "muito no futuro": {"upstream_url": URL, "upstream_token": TOKEN, "updated_at": now + 10 * 365 * 86400},
            "formato antigo (updated)": {"upstream_url": URL, "upstream_token": TOKEN, "updated": now},
            "updated_at em texto": {"upstream_url": URL, "upstream_token": TOKEN, "updated_at": str(now)},
            "updated_at nulo": {"upstream_url": URL, "upstream_token": TOKEN, "updated_at": None},
            "sem URL": {"upstream_token": TOKEN, "updated_at": now},
            "URL vazia": {"upstream_url": "", "upstream_token": TOKEN, "updated_at": now},
            "JSON quebrado": "{not json",
            "JSON que não é objeto": "[1, 2, 3]",
            "arquivo vazio": "",
        }
        for label, state in cases.items():
            with self.subTest(label):
                if state is not None:
                    self.write_state(state)
                self.assert_issue_503(*self.get())

    def test_live_lease_is_not_a_503(self) -> None:
        self.write_state({"upstream_url": URL, "upstream_token": TOKEN, "updated_at": int(time.time())})
        status, headers, text = self.get()
        self.assertEqual((status, text), (200, f"proxied to {URL}"))
        self.assertNotIn("retry-after", [k.lower() for k, _ in headers])

    def test_no_state_path_variable_is_503_even_with_a_live_state_in_the_temp_dir(self) -> None:
        # TMPDIR do servidor "unconfigured" é a raiz do teste: um padrão sys_get_temp_dir()/... acharia este arquivo.
        root = self.state.parent.parent
        live = {"upstream_url": URL, "upstream_token": TOKEN, "updated_at": int(time.time())}
        for name in ("simpleti-upstream.json", "upstream.json"):
            (root / name).write_text(json.dumps(live), encoding="utf-8")
        self.assert_issue_503(*self.get("unconfigured"))

    def test_lease_response_never_carries_the_state(self) -> None:
        for state in ({"upstream_url": URL, "upstream_token": TOKEN, "updated_at": 1}, "garbage " + TOKEN):
            self.write_state(state)
            status, headers, text = self.get()
            self.assert_issue_503(status, headers, text)


@unittest.skipUnless(PHP, "php não instalado")
class DeadlineContractTests(unittest.TestCase):
    STATE = {"upstream_url": URL, "updated_at": 1000}

    def test_boundary_with_the_default_ttl(self) -> None:
        self.assertEqual(live_upstream(self.STATE, 1000), URL)
        self.assertEqual(live_upstream(self.STATE, 1090), URL)
        self.assertIsNone(live_upstream(self.STATE, 1091))

    def test_ttl_comes_from_the_environment(self) -> None:
        self.assertEqual(live_upstream(self.STATE, 1005, "5"), URL)
        self.assertIsNone(live_upstream(self.STATE, 1006, "5"))
        self.assertEqual(live_upstream(self.STATE, 1300, "300"), URL)
        self.assertIsNone(live_upstream(self.STATE, 1301, "300"))

    def test_ttl_is_capped_at_3600(self) -> None:
        state = {"upstream_url": URL, "updated_at": 1000}
        for value in ("3600", "3601", "7200", "99999", "999999"):
            with self.subTest(value=value):
                self.assertEqual(live_upstream(state, 1000 + 3600, value), URL)
                self.assertIsNone(live_upstream(state, 1000 + 3601, value))
        self.assertEqual(live_upstream(state, 1000 + 3599, "3599"), URL)
        self.assertIsNone(live_upstream(state, 1000 + 3600, "3599"))
        out = php_eval("require $argv[1]; echo simpleti_lease_ttl();", str(LEASE), env={"SIMPLETI_LEASE_TTL_S": "999999"})
        self.assertEqual(out, "3600")
        self.assertEqual(re.search(r"const SIMPLETI_LEASE_TTL_MAX_S = (\d+);", LEASE.read_text(encoding="utf-8")).group(1), "3600")

    def test_updated_at_in_the_future_is_expired_beyond_a_5_second_tolerance(self) -> None:
        for ttl in (None, "5", "3600"):
            with self.subTest(ttl=ttl):
                self.assertEqual(live_upstream({"upstream_url": URL, "updated_at": 1005}, 1000, ttl), URL)
                self.assertIsNone(live_upstream({"upstream_url": URL, "updated_at": 1006}, 1000, ttl))
                self.assertIsNone(live_upstream({"upstream_url": URL, "updated_at": 1000 + 86400}, 1000, ttl))
                self.assertIsNone(live_upstream({"upstream_url": URL, "updated_at": 2**62}, 1000, ttl))
        # a tolerância não encurta o prazo normal: o carimbo "agora" continua vivo até o fim do TTL
        self.assertEqual(live_upstream({"upstream_url": URL, "updated_at": 1000}, 1090), URL)
        self.assertIsNone(live_upstream({"upstream_url": URL, "updated_at": 1000}, 1091))

    def test_invalid_ttl_falls_back_to_the_default(self) -> None:
        for value in ("", "0", "-1", "abc", "1.5", " 5", "5 ", "5\n", "0x10", "1000000", "+5", "٥"):
            with self.subTest(value=value):
                self.assertEqual(live_upstream(self.STATE, 1090, value), URL)
                self.assertIsNone(live_upstream(self.STATE, 1091, value))

    def test_invalid_states_are_not_live(self) -> None:
        for state in (None, {}, {"upstream_url": URL}, {"upstream_url": URL, "updated": 1000},
                      {"upstream_url": URL, "updated_at": "1000"}, {"upstream_url": URL, "updated_at": 1000.5},
                      {"upstream_url": URL, "updated_at": True}, {"upstream_url": "", "updated_at": 1000},
                      {"upstream_url": 5, "updated_at": 1000}, {"upstream_url": None, "updated_at": 1000}):
            with self.subTest(state=state):
                self.assertIsNone(live_upstream(state, 1000))

    def test_default_ttl_covers_three_beats_and_retry_after_is_one_beat(self) -> None:
        text = LEASE.read_text(encoding="utf-8")
        ttl = int(re.search(r"const SIMPLETI_LEASE_TTL_DEFAULT_S = (\d+);", text).group(1))
        retry = int(re.search(r"const SIMPLETI_RETRY_AFTER_S = (\d+);", text).group(1))
        self.assertEqual((ttl, retry, serve_colab.HEARTBEAT_S), (90, 30, 30))
        self.assertGreaterEqual(ttl, 3 * serve_colab.HEARTBEAT_S)
        self.assertEqual(retry, serve_colab.HEARTBEAT_S)

    def test_state_file_helper_needs_an_absolute_path(self) -> None:
        for value, expected in (("/var/lib/simpleti/up.json", "/var/lib/simpleti/up.json"), ("up.json", None), ("", None), (None, None)):
            with self.subTest(value=value):
                out = php_eval("require $argv[1]; echo json_encode(simpleti_state_file());", str(LEASE),
                               env={"SIMPLETI_UPSTREAM_FILE": value})
                self.assertEqual(json.loads(out), expected)


class LeaseSourceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lease = LEASE.read_text(encoding="utf-8")
        self.set_upstream = (GATEWAY / "set_upstream.php").read_text(encoding="utf-8")
        self.status = (GATEWAY / "status.php").read_text(encoding="utf-8")

    def test_state_path_has_no_default_and_is_never_under_tmp(self) -> None:
        self.assertNotIn("sys_get_temp_dir", self.lease)
        self.assertNotIn("/tmp", self.lease)
        self.assertEqual(self.lease.count("getenv('SIMPLETI_UPSTREAM_FILE')"), 1)
        file_lines = [
            next(line for line in text.splitlines() if line.strip().startswith("$file = "))
            for text in (self.lease, self.set_upstream, self.status)
        ]
        self.assertEqual({line.strip() for line in file_lines}, {"$file = getenv('SIMPLETI_UPSTREAM_FILE') ?: '';"})
        self.assertIn("strncmp($file, '/', 1) === 0", self.lease)  # o mesmo teste de caminho absoluto de set_upstream.php

    def test_the_lease_never_reads_or_prints_the_upstream_token(self) -> None:
        self.assertNotIn("upstream_token", self.lease)
        self.assertNotIn("X-Simpleti-Upstream-Token", self.lease)
        self.assertNotIn("SIMPLETI_ADMIN_KEY", self.lease)
        self.assertNotIn("file_put_contents", self.lease)  # o lease só lê; quem escreve é set_upstream.php

    def test_only_updated_at_exists(self) -> None:
        for path in sorted(GATEWAY.glob("*.php")) + [ROOT / "tests" / "test_gateway.py"]:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("'updated'", text, path.name)
            self.assertNotIn('"updated"', text, path.name)
        self.assertIn("'updated_at' => time()", self.set_upstream)
        self.assertEqual(self.status.count("$state['updated_at']"), 2)
        self.assertIn("is_int($at)", self.lease)

    def test_existing_state_dir_must_already_be_private(self) -> None:
        check = "(fileperms($dir) & 0077) !== 0"
        self.assertIn(check, self.set_upstream)
        self.assertIn("respond(500, ['error' => 'upstream dir too open']);", self.set_upstream)
        self.assertIn("error_log(", self.set_upstream)
        self.assertNotIn("chmod($dir", self.set_upstream)  # recusa e registra; nunca corrige o modo sozinho
        order = [self.set_upstream.index(x) for x in ("mkdir($dir, 0700, true)", check, "upstream dir not writable", "$tmp = tempnam(")]
        self.assertEqual(order, sorted(order))
        # limpar o upstream continua possível com o diretório aberto: só gravar o token é recusado
        self.assertLess(self.set_upstream.index("respond(200, ['ok' => true, 'upstream' => null]);"), self.set_upstream.index(check))

    def test_existing_auth_and_token_checks_are_still_there(self) -> None:
        for needle in (
            "hash_equals($expected, $key)",
            "strlen($expected) < 32",
            "respond(401, ['error' => 'Unauthorized'])",
            "respond(400, ['error' => 'invalid upstream_token'])",
            "chmod($tmp, 0600)",
        ):
            self.assertIn(needle, self.set_upstream)
        self.assertLess(self.set_upstream.index("invalid upstream_token"), self.set_upstream.index("$tmp = tempnam("))

    @unittest.skipUnless(PHP, "php não instalado")
    def test_php_lint(self) -> None:
        for path in sorted(GATEWAY.glob("*.php")):
            proc = subprocess.run([PHP, "-l", str(path)], capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("No syntax errors detected", proc.stdout)


class HeartbeatClientContractTests(unittest.TestCase):
    def test_heartbeat_reuses_set_upstream_with_the_admin_bearer(self) -> None:
        admin = "admin-" + secrets.token_hex(16)
        sent: list[dict] = []

        def fake_http(url, body=None, timeout=30, headers=None):
            sent.append({"url": url, "body": body, "headers": headers})
            return 200, {"ok": True, "upstream": URL}

        class Stop:
            def __init__(self) -> None:
                self.n = 0

            def wait(self, timeout: float) -> bool:
                self.n += 1
                return self.n > 2

        with patch.object(serve_colab, "_http_json", side_effect=fake_http), \
                patch.object(serve_colab, "vllm_healthy", return_value=True):
            serve_colab.heartbeat_loop(lambda: serve_colab.set_upstream(URL, admin), {"vllm": lambda: True}, Stop())
        self.assertEqual(len(sent), 2)
        for item in sent:
            self.assertEqual(item["url"], serve_colab.GATEWAY_SET)
            self.assertEqual(item["body"], {"upstream_url": URL, "upstream_token": serve_colab.GATE_TOKEN})
            self.assertEqual(item["headers"], {"Authorization": f"Bearer {admin}"})

    def test_heartbeat_output_never_carries_a_secret(self) -> None:
        admin = "admin-" + secrets.token_hex(16)
        err = io.StringIO()

        def fake_http(url, body=None, timeout=30, headers=None):
            return 500, {"error": "cannot write upstream file"}

        class Stop:
            n = 0

            def wait(self, timeout: float) -> bool:
                type(self).n += 1
                return type(self).n > 2

        with patch.object(serve_colab, "_http_json", side_effect=fake_http), \
                patch.object(serve_colab, "vllm_healthy", side_effect=[False, True]), \
                contextlib.redirect_stderr(err):
            serve_colab.heartbeat_loop(lambda: serve_colab.set_upstream(URL, admin), {"vllm": lambda: True}, Stop())
        out = err.getvalue()
        self.assertIn("heartbeat: set_upstream falhou: set_upstream HTTP 500", out)
        self.assertIn("lease não renovado", out)
        for secret in (admin, serve_colab.GATE_TOKEN, serve_colab.VLLM_TOKEN):
            self.assertNotIn(secret, out)

    def test_heartbeat_interval_is_30_seconds(self) -> None:
        self.assertEqual(serve_colab.HEARTBEAT_S, 30)
        self.assertEqual(serve_colab.heartbeat_loop.__defaults__, (serve_colab.HEARTBEAT_S,))


class ServeColabWiringContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SERVE_PY.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)
        self.main = next(n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef) and n.name == "main")

    def test_every_exit_signal_runs_shutdown(self) -> None:
        handled = {
            ast.unparse(call.args[0])
            for call in ast.walk(self.main)
            if isinstance(call, ast.Call) and ast.unparse(call.func) == "signal.signal"
            and ast.unparse(call.args[1]) == "lambda *_: (shutdown(), sys.exit(0))"
        }
        self.assertEqual(handled, {"signal.SIGTERM", "signal.SIGINT", "signal.SIGHUP"})
        self.assertEqual(self.source.count("signal.SIGHUP"), 1)

    def test_cloudflared_log_is_a_file_not_a_pipe(self) -> None:
        self.assertNotIn("stderr=subprocess.PIPE", self.source)
        self.assertNotIn("stdout=subprocess.PIPE", self.source)

    def test_shutdown_stops_the_beat_before_clearing_the_upstream(self) -> None:
        shutdown = next(n for n in ast.walk(self.main) if isinstance(n, ast.FunctionDef) and n.name == "shutdown")
        text = ast.unparse(shutdown)
        self.assertLess(text.index("stop_beat.set()"), text.index("beat.join("))
        self.assertLess(text.index("beat.join("), text.index("clear_upstream(key)"))

    def test_main_starts_the_beat_on_the_registered_tunnel_and_watches_every_child(self) -> None:
        text = ast.unparse(self.main)
        self.assertIn("probe_public(probe_all, public_url)", text)
        self.assertIn("lambda: set_upstream(public_url, key)", text)
        self.assertIn("heartbeat_loop", text)
        for child in ("'vllm'", "'gate'", "'cloudflared'"):
            self.assertIn(child, text)
        self.assertLess(text.index("registered = True"), text.index("beat.start()"))

    def test_start_gate_returns_the_server_and_its_thread(self) -> None:
        with patch.object(serve_colab, "GATE_PORT", 0), contextlib.redirect_stdout(io.StringIO()):
            httpd, thread = serve_colab.start_gate()
        try:
            self.assertEqual(httpd.server_address[0], "127.0.0.1")
            self.assertTrue(thread.is_alive())
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
