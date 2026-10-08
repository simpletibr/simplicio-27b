"""Issue #18: o gate, o vLLM e o set_upstream.php só aceitam tráfego do gateway.

Só biblioteca padrão. Sem rede externa e sem produção: o gate de verdade (serve_colab._GateHandler) sobe
numa porta efêmera de 127.0.0.1, na frente de um vLLM falso em outra porta. Os tokens são gerados a cada
execução (secrets) e nunca são escritos no repo.

Nota: o proxy PHP de produção não está versionado em gateway/ (só set_upstream.php, status.php e o helper
upstream_token.php), então não há teste do proxy aqui. A edição dele é item do dono (ver o PR).
"""

from __future__ import annotations

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
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

import serve_colab  # noqa: E402

PHP = shutil.which("php")
TOKEN = secrets.token_urlsafe(32)
ADMIN = secrets.token_hex(32)
OK_CHAT = {
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}
MODELS = {"object": "list", "data": [{"id": "simplicio-27b", "object": "model"}]}


class FakeVllm(BaseHTTPRequestHandler):
    seen: list[tuple[str, str, dict[str, str]]] = []

    def log_message(self, *args) -> None:
        pass

    def _reply(self, method: str, payload: dict) -> None:
        type(self).seen.append((method, self.path, {k.lower(): v for k, v in self.headers.items()}))
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        self._reply("GET", MODELS)

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._reply("POST", OK_CHAT)


class GateAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vllm = ThreadingHTTPServer(("127.0.0.1", 0), FakeVllm)
        threading.Thread(target=cls.vllm.serve_forever, daemon=True).start()
        cls.old_port = serve_colab.VLLM_PORT
        serve_colab.VLLM_PORT = cls.vllm.server_address[1]
        cls.gate = ThreadingHTTPServer(("127.0.0.1", 0), serve_colab._GateHandler)
        threading.Thread(target=cls.gate.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        for server in (cls.gate, cls.vllm):
            server.shutdown()
            server.server_close()
        serve_colab.VLLM_PORT = cls.old_port

    def setUp(self) -> None:
        FakeVllm.seen.clear()

    def call(self, method: str, path: str, token: str | None, auth: str | None = None):
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers[serve_colab.UPSTREAM_HEADER] = token
        if auth is not None:
            headers["Authorization"] = auth
        data = json.dumps({"model": "simplicio-27b", "messages": []}).encode() if method == "POST" else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.gate.server_address[1]}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, json.loads(exc.read())

    def test_without_token_is_401(self) -> None:
        for method, path in (("GET", "/v1/models"), ("POST", "/v1/chat/completions"), ("GET", "/metrics")):
            with self.subTest(method=method, path=path):
                status, _ = self.call(method, path, None)
                self.assertEqual(status, 401)
        self.assertEqual(FakeVllm.seen, [])

    def test_wrong_token_is_401(self) -> None:
        status, _ = self.call("POST", "/v1/chat/completions", "x" * 43)
        self.assertEqual(status, 401)
        self.assertEqual(FakeVllm.seen, [])

    def test_other_routes_are_404(self) -> None:
        for method, path in (
            ("GET", "/metrics"),
            ("GET", "/docs"),
            ("GET", "/health"),
            ("POST", "/tokenize"),
            ("POST", "/v1/completions"),
            ("POST", "/invocations"),
            ("GET", "/v1/chat/completions"),
            ("POST", "/v1/models"),
            ("GET", "/v1/models/../../metrics"),
        ):
            with self.subTest(method=method, path=path):
                status, _ = self.call(method, path, serve_colab.GATE_TOKEN)
                self.assertEqual(status, 404)
        self.assertEqual(FakeVllm.seen, [])

    def test_allowed_routes_reach_vllm_with_vllm_token_only(self) -> None:
        status, body = self.call("GET", "/v1/models", serve_colab.GATE_TOKEN, auth="Bearer user-key")
        self.assertEqual((status, body), (200, MODELS))
        status, body = self.call("POST", "/v1/chat/completions", serve_colab.GATE_TOKEN, auth="Bearer user-key")
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(len(FakeVllm.seen), 2)
        for _method, _path, headers in FakeVllm.seen:
            self.assertEqual(headers["authorization"], f"Bearer {serve_colab.VLLM_TOKEN}")
            self.assertNotIn("x-simpleti-upstream-token", headers)


class ServeColabTokenTests(unittest.TestCase):
    def test_tokens_are_long_and_distinct(self) -> None:
        self.assertGreaterEqual(len(serve_colab.VLLM_TOKEN), 43)
        self.assertGreaterEqual(len(serve_colab.GATE_TOKEN), 43)
        self.assertNotEqual(serve_colab.VLLM_TOKEN, serve_colab.GATE_TOKEN)

    def test_serve_env_binds_loopback_with_vllm_key(self) -> None:
        self.assertEqual(serve_colab.serve_env()["HOST"], "127.0.0.1")
        self.assertEqual(serve_colab.serve_env()["VLLM_API_KEY"], serve_colab.VLLM_TOKEN)

    def test_serve_env_has_no_gateway_credentials(self) -> None:
        admin = secrets.token_hex(32)
        planted = {
            "SIMPLETI_ADMIN_KEY": admin,
            "SIMPLETI_API_KEY": "k-" + secrets.token_hex(16),
            "GITHUB_TOKEN": "g-" + secrets.token_hex(16),
            "CUDA_VISIBLE_DEVICES": "0",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "INFO",
            "GPU_MEM": "0.9",
            "LANG": "C.UTF-8",
        }
        with patch.dict(os.environ, planted):
            env = serve_colab.serve_env()
        for name in ("SIMPLETI_ADMIN_KEY", "SIMPLETI_API_KEY", "GITHUB_TOKEN"):
            self.assertNotIn(name, env)
        for secret in (admin, planted["SIMPLETI_API_KEY"], planted["GITHUB_TOKEN"], serve_colab.GATE_TOKEN):
            self.assertNotIn(secret, env.values())
        for name in ("CUDA_VISIBLE_DEVICES", "NCCL_DEBUG", "VLLM_LOGGING_LEVEL", "GPU_MEM", "LANG", "PATH"):
            self.assertIn(name, env)
        self.assertEqual(env["HOST"], "127.0.0.1")
        self.assertEqual(env["VLLM_API_KEY"], serve_colab.VLLM_TOKEN)

    def test_tunnel_env_drops_every_simpleti_variable(self) -> None:
        with patch.dict(os.environ, {"SIMPLETI_ADMIN_KEY": "a" * 64, "SIMPLETI_SET_UPSTREAM_URL": "u", "KEEP_ME": "1"}):
            env = serve_colab.tunnel_env()
        self.assertEqual([n for n in env if n.startswith("SIMPLETI_")], [])
        self.assertEqual(env["KEEP_ME"], "1")
        self.assertIn("PATH", env)

    def test_no_wildcard_bind(self) -> None:
        self.assertNotIn("0.0.0.0", Path(serve_colab.__file__).read_text(encoding="utf-8"))

    def test_set_upstream_sends_gate_token(self) -> None:
        seen: dict = {}

        def fake_http(url, body=None, timeout=30, headers=None):
            seen["body"] = body
            return 200, {"ok": True}

        with patch.object(serve_colab, "_http_json", side_effect=fake_http):
            serve_colab.set_upstream("https://abc.trycloudflare.com", "admin-test")
        self.assertEqual(seen["body"]["upstream_token"], serve_colab.GATE_TOKEN)

    def test_probe_sends_gate_token(self) -> None:
        seen: dict = {}

        def fake_http(url, body=None, timeout=30, headers=None):
            seen["headers"] = headers
            return 200, OK_CHAT

        with patch.object(serve_colab, "_http_json", side_effect=fake_http):
            serve_colab.probe_completion("http://127.0.0.1:8001", "simplicio-27b")
        self.assertEqual(seen["headers"][serve_colab.UPSTREAM_HEADER], serve_colab.GATE_TOKEN)

    def test_tokens_never_printed(self) -> None:
        for number, line in enumerate(Path(serve_colab.__file__).read_text(encoding="utf-8").splitlines(), 1):
            if "print(" in line or "log_message(" in line:
                self.assertNotIn("_TOKEN", line, f"serve_colab.py:{number}")

    def test_serve_script_defaults_to_loopback(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "HOST"}
        env["VLLM_BIN"] = "echo"
        out = subprocess.run(
            ["bash", str(ROOT / "deploy" / "serve_vllm.sh")], env=env, capture_output=True, text=True, check=True
        )
        self.assertIn("--host 127.0.0.1 ", out.stdout.splitlines()[-1])


@unittest.skipIf(PHP is None, "php não está no PATH")
class SetUpstreamTokenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "upstream.json"
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.proc = subprocess.Popen(
            [PHP, "-S", f"127.0.0.1:{self.port}", str(ROOT / "gateway" / "set_upstream.php")],
            env={**os.environ, "SIMPLETI_ADMIN_KEY": ADMIN, "SIMPLETI_UPSTREAM_FILE": str(self.state)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 10
        while True:
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=1).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    self.proc.kill()
                    self.tmp.cleanup()
                    raise RuntimeError("php -S não subiu")
                time.sleep(0.1)

    def tearDown(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=5)
        self.tmp.cleanup()

    def post(self, body: dict) -> tuple[int, str]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/set_upstream.php",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {ADMIN}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, exc.read().decode()

    def test_missing_token_is_400(self) -> None:
        status, _ = self.post({"upstream_url": "https://abc.trycloudflare.com"})
        self.assertEqual(status, 400)
        self.assertFalse(self.state.exists())

    def test_short_token_is_400(self) -> None:
        status, _ = self.post({"upstream_url": "https://abc.trycloudflare.com", "upstream_token": "short"})
        self.assertEqual(status, 400)
        self.assertFalse(self.state.exists())

    def test_token_is_stored_private_and_not_echoed(self) -> None:
        status, text = self.post({"upstream_url": "https://abc.trycloudflare.com", "upstream_token": TOKEN})
        self.assertEqual(status, 200)
        self.assertNotIn(TOKEN, text)
        self.assertEqual(json.loads(self.state.read_text())["upstream_token"], TOKEN)
        self.assertEqual(self.state.stat().st_mode & 0o777, 0o600)

    def test_clear_removes_token(self) -> None:
        self.post({"upstream_url": "https://abc.trycloudflare.com", "upstream_token": TOKEN})
        self.assertTrue(self.state.exists())
        status, _ = self.post({"clear": True})
        self.assertEqual(status, 200)
        self.assertFalse(self.state.exists())


@unittest.skipIf(PHP is None, "php não está no PATH")
class StatePathTests(unittest.TestCase):
    """O estado agora guarda um segredo: sem caminho padrão em /tmp, e o diretório que o script cria é 0700."""

    URL = "https://abc.trycloudflare.com"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.systmp = self.base / "systmp"  # é o que sys_get_temp_dir() devolve para o PHP (TMPDIR)
        self.cwd = self.base / "cwd"
        self.systmp.mkdir()
        self.cwd.mkdir()
        self.proc = None

    def tearDown(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        self.tmp.cleanup()

    def env(self, **extra: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in {"SIMPLETI_UPSTREAM_FILE", "TMPDIR"}}
        return {**env, "SIMPLETI_ADMIN_KEY": ADMIN, "TMPDIR": str(self.systmp), **extra}

    def start(self, **extra: str) -> int:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        self.proc = subprocess.Popen(
            [PHP, "-S", f"127.0.0.1:{port}", str(ROOT / "gateway" / "set_upstream.php")],
            env=self.env(**extra),
            cwd=self.cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 10
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                return port
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("php -S não subiu")
                time.sleep(0.1)

    def post(self, port: int, body: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/set_upstream.php",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {ADMIN}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, json.loads(exc.read())

    def assert_nothing_written_outside_the_state_path(self) -> None:
        self.assertEqual(list(self.systmp.iterdir()), [], "o PHP escreveu no diretório temporário do sistema")
        self.assertEqual(list(self.cwd.iterdir()), [], "o PHP escreveu no diretório de trabalho")

    def test_sources_have_no_default_path_and_no_tmp(self) -> None:
        for name in ("set_upstream.php", "status.php"):
            text = (ROOT / "gateway" / name).read_text(encoding="utf-8")
            self.assertNotIn("sys_get_temp_dir", text, name)
            self.assertNotIn("simpleti-upstream.json", text, name)

    def test_without_the_path_variable_it_fails_closed(self) -> None:
        port = self.start()
        valid = {"upstream_url": self.URL, "upstream_token": TOKEN}
        for label, body in (("registrar", valid), ("limpar", {"clear": True})):
            with self.subTest(label):
                status, payload = self.post(port, body)
                self.assertEqual((status, payload), (500, {"error": "upstream file not configured"}))
        self.assert_nothing_written_outside_the_state_path()

    def test_relative_path_fails_closed(self) -> None:
        port = self.start(SIMPLETI_UPSTREAM_FILE="state/upstream.json")
        status, payload = self.post(port, {"upstream_url": self.URL, "upstream_token": TOKEN})
        self.assertEqual((status, payload), (500, {"error": "upstream file not configured"}))
        self.assert_nothing_written_outside_the_state_path()

    def test_directories_it_creates_are_0700_and_the_file_0600(self) -> None:
        state = self.base / "new" / "state" / "upstream.json"
        port = self.start(SIMPLETI_UPSTREAM_FILE=str(state))
        status, payload = self.post(port, {"upstream_url": self.URL, "upstream_token": TOKEN})
        self.assertEqual((status, payload), (200, {"ok": True, "upstream": self.URL}))
        self.assertEqual(state.parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual(state.parent.parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual(state.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(state.read_text())["upstream_token"], TOKEN)
        self.assertEqual(sorted(p.name for p in state.parent.iterdir()), ["upstream.json"])
        self.assert_nothing_written_outside_the_state_path()

    def test_unusable_directory_does_not_fall_back_to_the_system_tmp(self) -> None:
        blocker = self.base / "blocker"
        blocker.write_text("um arquivo, não um diretório", encoding="utf-8")
        port = self.start(SIMPLETI_UPSTREAM_FILE=str(blocker / "sub" / "upstream.json"))
        status, payload = self.post(port, {"upstream_url": self.URL, "upstream_token": TOKEN})
        self.assertEqual((status, payload), (500, {"error": "cannot create upstream dir"}))
        self.assert_nothing_written_outside_the_state_path()

    def test_status_does_not_read_the_old_default_path(self) -> None:
        old = self.systmp / "simpleti-upstream.json"
        old.write_text(json.dumps({"upstream_url": self.URL, "updated": 1}), encoding="utf-8")

        def registered(**extra: str) -> bool:
            out = subprocess.run(
                [PHP, str(ROOT / "gateway" / "status.php")],
                env=self.env(**extra), capture_output=True, text=True, check=True, cwd=self.cwd,
            ).stdout
            return json.loads(out)["upstream_registered"]

        self.assertIs(registered(), False)
        self.assertIs(registered(SIMPLETI_UPSTREAM_FILE=str(old)), True)  # controle: o arquivo é lido quando configurado


@unittest.skipIf(PHP is None, "php não está no PATH")
class UpstreamTokenHelperTests(unittest.TestCase):
    def token_for(self, content: str | None):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "upstream.json"
            if content is not None:
                state.write_text(content, encoding="utf-8")
            out = subprocess.run(
                [
                    PHP,
                    "-r",
                    "require $argv[1]; echo json_encode(simpleti_upstream_token($argv[2]));",
                    str(ROOT / "gateway" / "upstream_token.php"),
                    str(state),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        return json.loads(out)

    def test_valid_token(self) -> None:
        self.assertEqual(self.token_for(json.dumps({"upstream_token": TOKEN})), TOKEN)

    def test_invalid_state_is_null(self) -> None:
        for label, content in (
            ("sem arquivo", None),
            ("não é json", "not json"),
            ("sem campo", "{}"),
            ("curto", json.dumps({"upstream_token": "short"})),
            ("com quebra de linha", json.dumps({"upstream_token": TOKEN + "\n"})),
            ("não string", json.dumps({"upstream_token": 123})),
        ):
            with self.subTest(label):
                self.assertIsNone(self.token_for(content))


if __name__ == "__main__":
    unittest.main()
