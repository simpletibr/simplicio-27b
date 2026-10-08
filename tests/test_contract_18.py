"""Contract test for issue #18: o que o gate, o vLLM e o gateway prometem uns aos outros.

Fixa, sem rede e sem produção:
  * o vLLM sobe em 127.0.0.1 (serve_vllm.sh sem HOST, serve_env() mesmo com HOST=0.0.0.0 no ambiente,
    start_gate() em 127.0.0.1) e a chave dele chega por VLLM_API_KEY no ambiente, nunca no argv;
  * a allowlist do gate tem exatamente GET /v1/models e POST /v1/chat/completions, o header do token é
    X-Simpleti-Upstream-Token e as respostas de recusa são 401 e 404 com o JSON {"error": {"message", "type"}};
  * os tokens (VLLM_TOKEN, GATE_TOKEN) só aparecem nas seis funções esperadas de serve_colab.py e nunca em
    print/log/raise; o formato deles casa com a regex de set_upstream.php e upstream_token.php;
  * set_upstream.php exige `upstream_token`, grava o estado com modo 0600 e não o devolve; upstream_token.php
    não tem os textos que #19 e #20 usam para achar o proxy;
  * o vLLM herda só uma allowlist de variáveis de ambiente (nunca SIMPLETI_ADMIN_KEY nem o token do gate) e o
    cloudflared herda o ambiente sem nenhuma SIMPLETI_*; provado com processos de verdade que despejam o
    nome e o hash (nunca o valor) de cada variável que receberam;
  * pedido com Transfer-Encoding (chunked) é recusado com 411 `length required` (depois do 401 e do 404, antes
    de tocar o vLLM); set_upstream.php e status.php não têm caminho padrão em /tmp (SIMPLETI_UPSTREAM_FILE
    obrigatório, absoluto) e o diretório que o script cria é 0700;
  * nenhum token literal de 16+ caracteres no repo (git ls-files, mais os novos ainda não adicionados). O
    scanner é provado com um valor fictício gerado aqui (secrets) e não olha este próprio arquivo, que
    carrega as regexes.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
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
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

import serve_colab  # noqa: E402

SERVE_SH = ROOT / "deploy" / "serve_vllm.sh"
SERVE_PY = ROOT / "deploy" / "serve_colab.py"
GATEWAY = ROOT / "gateway"
SELF = Path(__file__).resolve().relative_to(ROOT).as_posix()
PHP = shutil.which("php")

FAKE_VLLM = """#!{python}
import hashlib, json, os, sys
with open(os.environ["CONTRACT_OUT"], "w") as out:
    json.dump({{
        "argv": sys.argv[1:],
        "host_env": os.environ.get("HOST"),
        "env_names": sorted(os.environ),
        "env_sha": sorted(hashlib.sha256(v.encode()).hexdigest() for v in os.environ.values()),
        "key_ok": hashlib.sha256(os.environ.get("VLLM_API_KEY", "").encode()).hexdigest() == os.environ["CONTRACT_KEY_SHA"],
    }}, out)
"""

# Textos que só aparecem com um valor literal de 16+ caracteres colado: o placeholder `<random token>`, uma
# variável (`$upstreamToken`, `VLLM_TOKEN`) ou uma expressão (`"A" * 43`) não casam.
LITERAL_TOKEN_PATTERNS = {
    "header de upstream com valor literal": re.compile(
        r"""x-simpleti-upstream-token['"]?\s*[:=]\s*['"]?[A-Za-z0-9_-]{16,}""", re.I
    ),
    "upstream_token com string literal": re.compile(
        r"""upstream_token['"]?\s*(?::|=>|=)\s*['"][A-Za-z0-9_-]{16,}['"]""", re.I
    ),
    "VLLM_API_KEY com valor literal": re.compile(r"""VLLM_API_KEY['"]?\s*[:=]\s*['"]?[A-Za-z0-9_-]{16,}"""),
    "--api-key com valor literal": re.compile(r"""--api-key[ =]['"]?[A-Za-z0-9_-]{16,}"""),
    "VLLM/GATE/UPSTREAM_TOKEN = 'literal'": re.compile(
        r"""\b(?:VLLM|GATE|UPSTREAM)_TOKEN\s*=\s*['"][A-Za-z0-9_-]{16,}['"]"""
    ),
}
PHP_TOKEN_REGEX = re.compile(r"preg_match\('/(\^\[A-Za-z0-9_-\]\{32,128\}\$)/D'")


def scan_text(text: str) -> list[tuple[int, str]]:
    """(linha, rótulo) de cada achado; nunca devolve o valor achado."""
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in LITERAL_TOKEN_PATTERNS.items():
            if pattern.search(line):
                found.append((number, label))
    return found


def repo_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [ROOT / p for p in out.split("\0") if p]


def read_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw:
        return None
    return raw.decode("utf-8", errors="replace")


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def raw_http(port: int, head: str, body: bytes = b"") -> tuple[int, dict[str, str], bytes]:
    """Manda o pedido byte a byte como escrito (http.client não deixa montar Transfer-Encoding à mão)."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(head.replace("\n", "\r\n").encode() + b"\r\n\r\n" + body)
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    raw_head, _, raw_body = data.partition(b"\r\n\r\n")
    lines = raw_head.decode("latin-1").split("\r\n")
    headers = {k.lower(): v.strip() for k, v in (line.split(":", 1) for line in lines[1:])}
    return int(lines[0].split()[1]), headers, raw_body


def fake_vllm_run(env_extra: dict[str, str], cmd: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "vllm"
        out = Path(tmp) / "out.json"
        fake.write_text(FAKE_VLLM.format(python=sys.executable), encoding="utf-8")
        fake.chmod(0o755)
        env = {
            **env_extra,
            "VLLM_BIN": str(fake),
            "CONTRACT_OUT": str(out),
            "CONTRACT_KEY_SHA": hashlib.sha256(serve_colab.VLLM_TOKEN.encode()).hexdigest(),
        }
        subprocess.run(cmd, env=env, check=True, capture_output=True, timeout=30)
        return json.loads(out.read_text(encoding="utf-8"))


class BindContractTests(unittest.TestCase):
    def test_serve_script_defaults_to_loopback(self) -> None:
        base = {k: v for k, v in os.environ.items() if k not in {"HOST", "VLLM_API_KEY"}}
        argv = fake_vllm_run(base, ["bash", str(SERVE_SH)])["argv"]
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")

    def test_serve_script_documents_the_default_and_the_key(self) -> None:
        env_line = next(line for line in SERVE_SH.read_text(encoding="utf-8").splitlines() if line.startswith("# Env:"))
        self.assertIn("HOST (default 127.0.0.1)", env_line)
        self.assertIn("VLLM_API_KEY (vLLM then requires it on /v1/*)", env_line)

    def test_colab_launch_is_loopback_with_key_in_env_not_argv(self) -> None:
        with patch.dict(os.environ, {"HOST": "0.0.0.0"}):
            env = serve_colab.serve_env()
        self.assertEqual(env["HOST"], "127.0.0.1")
        self.assertEqual(env["VLLM_API_KEY"], serve_colab.VLLM_TOKEN)
        ran = fake_vllm_run(env, serve_colab.serve_cmd())
        self.assertEqual(ran["argv"][ran["argv"].index("--host") + 1], "127.0.0.1")
        self.assertEqual(ran["host_env"], "127.0.0.1")
        self.assertTrue(ran["key_ok"], "o vLLM não recebeu VLLM_API_KEY=<VLLM_TOKEN> pelo ambiente")
        for text in (json.dumps(ran["argv"]), " ".join(serve_colab.serve_cmd())):
            self.assertNotIn(serve_colab.VLLM_TOKEN, text)
            self.assertNotIn("--api-key", text)
        self.assertNotIn("--api-key", SERVE_SH.read_text(encoding="utf-8"))

    def test_main_launches_vllm_with_serve_env(self) -> None:
        tree = ast.parse(SERVE_PY.read_text(encoding="utf-8"))
        launches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "subprocess.Popen"
            and node.args
            and ast.unparse(node.args[0]) == "serve_cmd()"
        ]
        self.assertEqual(len(launches), 1)
        self.assertEqual({k.arg: ast.unparse(k.value) for k in launches[0].keywords}["env"], "serve_env()")

    def test_vllm_environment_is_an_allowlist(self) -> None:
        admin = "admin-" + secrets.token_hex(16)
        canary = "canary-" + secrets.token_hex(16)
        planted = {
            "SIMPLETI_ADMIN_KEY": admin,
            "SIMPLETI_API_KEY": canary,
            "GITHUB_TOKEN": canary,
            "AWS_SECRET_ACCESS_KEY": canary,
            "ANTHROPIC_API_KEY": canary,
            "SOME_UNLISTED_SECRET": canary,
            "VLLM_API_KEY": "outer-" + secrets.token_hex(16),  # a do ambiente de fora não pode vencer a do gate
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HOME": "/hf",
        }
        with patch.dict(os.environ, planted):
            env = serve_colab.serve_env()
            ran = fake_vllm_run(env, serve_colab.serve_cmd())
        names, hashes = set(ran["env_names"]), set(ran["env_sha"])
        for name in ("SIMPLETI_ADMIN_KEY", "SIMPLETI_API_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY",
                     "ANTHROPIC_API_KEY", "SOME_UNLISTED_SECRET"):
            self.assertNotIn(name, names)
            self.assertNotIn(name, env)
        for secret in (admin, canary, serve_colab.GATE_TOKEN, planted["VLLM_API_KEY"]):
            self.assertNotIn(sha(secret), hashes)
            self.assertNotIn(secret, env.values())
        self.assertIn(sha(serve_colab.VLLM_TOKEN), hashes)  # a chave do vLLM chegou: o teste não é vazio
        for name in ("PATH", "HOST", "VLLM_API_KEY", "CUDA_VISIBLE_DEVICES", "HF_HOME"):
            self.assertIn(name, names)
        self.assertTrue(ran["key_ok"])
        self.assertEqual(ran["host_env"], "127.0.0.1")

    def test_vllm_allowlist_covers_what_serve_script_and_vllm_read(self) -> None:
        with patch.dict(os.environ, {"GPU_MEM": "0.5", "VLLM_BIN": "x", "NCCL_DEBUG": "INFO", "LANG": "C",
                                      "LC_ALL": "C", "HTTPS_PROXY": "p", "SSL_CERT_FILE": "c"}):
            env = serve_colab.serve_env()
        for name in ("GPU_MEM", "VLLM_BIN", "NCCL_DEBUG", "LANG", "LC_ALL", "HTTPS_PROXY", "SSL_CERT_FILE", "PATH"):
            self.assertIn(name, env)

    def test_cloudflared_inherits_no_simpleti_variable(self) -> None:
        admin = "admin-" + secrets.token_hex(16)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "env.json"
            fake = Path(tmp) / "cloudflared"
            fake.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                "with open(os.environ['CONTRACT_OUT'], 'w') as out:\n"
                "    json.dump(sorted(os.environ), out)\n"
                "sys.stderr.write('INF https://abc-def.trycloudflare.com\\n')\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            planted = {"SIMPLETI_ADMIN_KEY": admin, "SIMPLETI_SET_UPSTREAM_URL": "https://x.invalid/",
                       "CLOUDFLARED_BIN": str(fake), "CONTRACT_OUT": str(out)}
            with patch.dict(os.environ, planted), contextlib.redirect_stdout(io.StringIO()):
                proc, url = serve_colab.start_cloudflared(8001)
            proc.wait(timeout=10)
            proc.stderr.close()
            names = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(url, "https://abc-def.trycloudflare.com")
        self.assertEqual([n for n in names if n.startswith("SIMPLETI_")], [])
        self.assertIn("PATH", names)
        self.assertIn("CONTRACT_OUT", names)

    def test_every_child_process_gets_an_explicit_environment(self) -> None:
        tree = ast.parse(SERVE_PY.read_text(encoding="utf-8"))
        envs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "subprocess.Popen":
                first = ast.unparse(node.args[0])
                who = "vllm" if first == "serve_cmd()" else "cloudflared" if "tunnel" in first else first
                envs[who] = {k.arg: ast.unparse(k.value) for k in node.keywords}.get("env")
        self.assertEqual(envs, {"vllm": "serve_env()", "cloudflared": "tunnel_env()"})

    def test_gate_binds_loopback_only(self) -> None:
        with patch.object(serve_colab, "GATE_PORT", 0), contextlib.redirect_stdout(io.StringIO()):
            httpd = serve_colab.start_gate()
        try:
            self.assertEqual(httpd.server_address[0], "127.0.0.1")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_no_wildcard_bind_and_host_only_in_serve_env(self) -> None:
        for path in repo_files():
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(("deploy/", "notebooks/")) or rel == "README.md":
                text = read_text(path)
                if text is not None:
                    self.assertNotIn("0.0.0.0", text, rel)
        host_lines = [
            line.strip() for line in SERVE_PY.read_text(encoding="utf-8").splitlines() if re.search(r"\bHOST\b", line)
        ]
        self.assertEqual(len(host_lines), 1, host_lines)
        self.assertIn('"HOST": LOCAL', host_lines[0])


class GateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = ThreadingHTTPServer(("127.0.0.1", 0), serve_colab._GateHandler)
        threading.Thread(target=cls.gate.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.gate.shutdown()
        cls.gate.server_close()

    def call(self, method: str, path: str, token: str | None):
        conn = http.client.HTTPConnection("127.0.0.1", self.gate.server_address[1], timeout=10)
        try:
            headers = {} if token is None else {serve_colab.UPSTREAM_HEADER: token}
            conn.request(method, path, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            return resp, raw
        finally:
            conn.close()

    def test_allowlist_is_exactly_the_two_routes_of_the_issue(self) -> None:
        self.assertEqual(
            serve_colab.GATE_ROUTES, {("GET", "/v1/models"), ("POST", "/v1/chat/completions")}
        )
        self.assertEqual(serve_colab.LOCAL, "127.0.0.1")

    def test_header_name(self) -> None:
        self.assertEqual(serve_colab.UPSTREAM_HEADER, "X-Simpleti-Upstream-Token")

    def test_refusal_codes_and_shape(self) -> None:
        for label, method, path, token, status, message in (
            ("sem token", "GET", "/v1/models", None, 401, "missing or invalid upstream token"),
            ("token errado", "GET", "/v1/models", "x" * 43, 401, "missing or invalid upstream token"),
            ("sem token em rota fora", "GET", "/metrics", None, 401, "missing or invalid upstream token"),
            ("rota fora", "GET", "/metrics", serve_colab.GATE_TOKEN, 404, "not found"),
            ("método fora", "POST", "/v1/models", serve_colab.GATE_TOKEN, 404, "not found"),
        ):
            with self.subTest(label):
                resp, raw = self.call(method, path, token)
                self.assertEqual(resp.status, status)
                self.assertEqual(resp.getheader("Content-Type"), "application/json")
                self.assertEqual(int(resp.getheader("Content-Length")), len(raw))
                self.assertEqual(json.loads(raw), {"error": {"message": message, "type": "GateError"}})
                for secret in (serve_colab.GATE_TOKEN, serve_colab.VLLM_TOKEN):
                    self.assertNotIn(secret.encode(), raw)

    def test_transfer_encoding_is_411_after_401_and_404(self) -> None:
        port = self.gate.server_address[1]
        good = f"{serve_colab.UPSTREAM_HEADER}: {serve_colab.GATE_TOKEN}"
        base = "POST /v1/chat/completions HTTP/1.1\nHost: gate"
        with socket.socket() as probe:  # nada pode chegar ao vLLM: porta fechada, caso chegue vira 502, não 411
            probe.bind(("127.0.0.1", 0))
            closed = probe.getsockname()[1]
        with patch.object(serve_colab, "VLLM_PORT", closed):
            for label, head, body, status, message in (
                ("chunked", f"{base}\n{good}\nTransfer-Encoding: chunked", b"", 411, "length required"),
                ("minúsculas", f"{base}\n{good}\ntransfer-encoding: chunked", b"", 411, "length required"),
                ("lista", f"{base}\n{good}\nTransfer-Encoding: gzip, chunked", b"", 411, "length required"),
                ("identity", f"{base}\n{good}\nTransfer-Encoding: identity", b"", 411, "length required"),
                ("com Content-Length", f"{base}\n{good}\nContent-Length: 2\nTransfer-Encoding: chunked", b"{}",
                 411, "length required"),
                ("GET", f"GET /v1/models HTTP/1.1\nHost: gate\n{good}\nTransfer-Encoding: chunked", b"",
                 411, "length required"),
                ("sem token", f"{base}\nTransfer-Encoding: chunked", b"", 401, "missing or invalid upstream token"),
                ("rota fora", f"POST /metrics HTTP/1.1\nHost: gate\n{good}\nTransfer-Encoding: chunked", b"",
                 404, "not found"),
            ):
                with self.subTest(label):
                    code, headers, raw = raw_http(port, head, body)
                    self.assertEqual(code, status)
                    self.assertEqual(headers["content-type"], "application/json")
                    self.assertEqual(int(headers["content-length"]), len(raw))
                    self.assertEqual(json.loads(raw), {"error": {"message": message, "type": "GateError"}})

    def test_comparison_is_constant_time_and_route_is_not_substring(self) -> None:
        source = SERVE_PY.read_text(encoding="utf-8")
        self.assertIn("hmac.compare_digest(", source)
        for forbidden in ("== GATE_TOKEN", "GATE_TOKEN ==", "!= GATE_TOKEN", "GATE_TOKEN !="):
            self.assertNotIn(forbidden, source)
        self.assertIn('self.path.split("?", 1)[0]) not in GATE_ROUTES', source)
        self.assertNotIn("startswith", source[source.index("def _allowed"):source.index("def do_GET")])

    def test_client_credentials_are_not_forwarded(self) -> None:
        source = SERVE_PY.read_text(encoding="utf-8")
        self.assertIn('"authorization"', source)
        self.assertIn("UPSTREAM_HEADER.lower()", source)
        self.assertIn('headers["Authorization"] = f"Bearer {VLLM_TOKEN}"', source)


class TokenHandlingContractTests(unittest.TestCase):
    EXPECTED_USES = {
        "serve_env": {"VLLM_TOKEN"},
        "wait_vllm": {"VLLM_TOKEN"},
        "probe_completion": {"GATE_TOKEN"},
        "set_upstream": {"GATE_TOKEN"},
        "_passthrough": {"VLLM_TOKEN"},
        "_allowed": {"GATE_TOKEN"},
    }

    def setUp(self) -> None:
        self.tree = ast.parse(SERVE_PY.read_text(encoding="utf-8"))

    def uses(self) -> dict[str, set[str]]:
        found: dict[str, set[str]] = {}

        class Visitor(ast.NodeVisitor):
            stack: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
                if node.id in {"VLLM_TOKEN", "GATE_TOKEN"}:
                    found.setdefault(self.stack[-1] if self.stack else "<module>", set()).add(node.id)

        Visitor().visit(self.tree)
        return found

    def test_tokens_flow_only_where_expected(self) -> None:
        found = self.uses()
        self.assertEqual(found.pop("<module>"), {"VLLM_TOKEN", "GATE_TOKEN"})  # só as duas atribuições
        self.assertEqual(found, self.EXPECTED_USES)

    def test_tokens_are_not_printed_logged_or_raised(self) -> None:
        for node in ast.walk(self.tree):
            sink = None
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name == "print" or name.endswith((".log_message", ".write", ".warning", ".error", ".info", ".debug")):
                    sink = node
            elif isinstance(node, ast.Raise):
                sink = node
            if sink is not None:
                for sub in ast.walk(sink):
                    if isinstance(sub, ast.Name):
                        self.assertFalse(sub.id.endswith("_TOKEN"), f"serve_colab.py:{sub.lineno}")

    def test_issue_grep_commands_print_nothing(self) -> None:
        pattern = re.compile(r"(print|log_message)\(.*_TOKEN")
        for path in (ROOT / "deploy").rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh"}:
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    self.assertIsNone(pattern.search(line), f"{path.relative_to(ROOT)}:{number}")

    def test_token_format_matches_what_php_accepts(self) -> None:
        regexes = set()
        for name in ("set_upstream.php", "upstream_token.php"):
            found = PHP_TOKEN_REGEX.findall((GATEWAY / name).read_text(encoding="utf-8"))
            self.assertTrue(found, name)
            regexes.update(found)
        self.assertEqual(len(regexes), 1, "set_upstream.php e upstream_token.php validam formatos diferentes")
        # PHP `$` + flag /D == fim exato da string; em Python isso é `\Z` (o `$` aceitaria um "\n" no fim).
        accepted = re.compile(regexes.pop()[:-1] + r"\Z")
        samples = [serve_colab.VLLM_TOKEN, serve_colab.GATE_TOKEN] + [secrets.token_urlsafe(32) for _ in range(200)]
        for sample in samples:
            self.assertIsNotNone(accepted.match(sample))
        for bad in ("", "short", "a" * 31, "a" * 129, ("a" * 43) + "\n", ("a" * 42) + "="):
            self.assertIsNone(accepted.match(bad), repr(bad[:8]))

    def test_registration_body_has_exactly_url_and_token(self) -> None:
        seen: dict = {}

        def fake_http(url, body=None, timeout=30, headers=None):
            seen.update(body=body, headers=headers)
            return 200, {"ok": True}

        with patch.object(serve_colab, "_http_json", side_effect=fake_http):
            serve_colab.set_upstream("https://abc.trycloudflare.com", "admin-test")
        self.assertEqual(
            seen["body"], {"upstream_url": "https://abc.trycloudflare.com", "upstream_token": serve_colab.GATE_TOKEN}
        )
        self.assertEqual(seen["headers"], {"Authorization": "Bearer admin-test"})

    def test_clear_does_not_send_the_token(self) -> None:
        bodies: list = []

        def fake_http(url, body=None, timeout=30, headers=None):
            bodies.append(body)
            return 200, {"ok": True, "upstream": None}

        with patch.object(serve_colab, "_http_json", side_effect=fake_http), contextlib.redirect_stdout(io.StringIO()):
            serve_colab.clear_upstream("admin-test")
        self.assertEqual(bodies, [{"clear": True}])


class GatewayPhpContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.set_upstream = (GATEWAY / "set_upstream.php").read_text(encoding="utf-8")
        self.helper = (GATEWAY / "upstream_token.php").read_text(encoding="utf-8")
        self.status = (GATEWAY / "status.php").read_text(encoding="utf-8")

    def test_set_upstream_requires_the_token_and_keeps_the_state_private(self) -> None:
        self.assertIn("respond(400, ['error' => 'invalid upstream_token']);", self.set_upstream)
        self.assertIn("'upstream_token' => $token", self.set_upstream)
        self.assertIn("chmod($tmp, 0600)", self.set_upstream)
        self.assertNotIn("0644", self.set_upstream)
        self.assertIn("respond(200, ['ok' => true, 'upstream' => $url]);", self.set_upstream)
        self.assertIn("respond(200, ['ok' => true, 'upstream' => null]);", self.set_upstream)
        for code in ("405", "500", "401", "400", "200"):
            self.assertIn(f"respond({code},", self.set_upstream)

    def test_token_is_validated_before_anything_is_written(self) -> None:
        self.assertLess(self.set_upstream.index("invalid upstream_token"), self.set_upstream.index("$tmp = tempnam("))

    def test_helper_has_nothing_the_proxy_discovery_greps_for(self) -> None:
        for text in ("X-Simpleti-Upstream-Token", "SIMPLETI_UPSTREAM_FILE", "upstream_url", "simpleti-upstream"):
            self.assertNotIn(text, self.helper)
        self.assertIn("function simpleti_upstream_token(string $file): ?string", self.helper)

    def test_state_path_has_no_default_and_is_never_under_tmp(self) -> None:
        file_lines = []
        for name, text in (("set_upstream.php", self.set_upstream), ("status.php", self.status)):
            self.assertNotIn("sys_get_temp_dir", text, name)
            self.assertNotIn("/tmp", text.replace("system temp dir", ""), name)
            file_lines.append(next(line for line in text.splitlines() if line.startswith("$file = ")))
        self.assertEqual(file_lines, ["$file = getenv('SIMPLETI_UPSTREAM_FILE') ?: '';"] * 2)
        self.assertIn("respond(500, ['error' => 'upstream file not configured']);", self.set_upstream)
        self.assertIn("strncmp($file, '/', 1) !== 0", self.set_upstream)

    def test_created_state_directory_is_private(self) -> None:
        self.assertIn("mkdir($dir, 0700, true)", self.set_upstream)
        self.assertNotIn("0750", self.set_upstream)
        self.assertIn("respond(500, ['error' => 'cannot create upstream dir']);", self.set_upstream)
        self.assertIn("respond(500, ['error' => 'upstream dir not writable']);", self.set_upstream)
        # o diretório é validado antes de tempnam(), que sem isso cairia no diretório temporário do sistema
        self.assertLess(self.set_upstream.index("upstream dir not writable"), self.set_upstream.index("$tmp = tempnam("))

    def test_status_never_reads_the_token(self) -> None:
        self.assertNotIn("upstream_token", self.status)

    @unittest.skipUnless(PHP, "php não instalado")
    def test_php_lint(self) -> None:
        for name in ("set_upstream.php", "upstream_token.php", "status.php"):
            subprocess.run([PHP, "-l", str(GATEWAY / name)], check=True, capture_output=True)


class NoLiteralTokenContractTests(unittest.TestCase):
    def test_scanner_flags_a_generated_token_and_ignores_placeholders(self) -> None:
        fake = secrets.token_urlsafe(32)  # fictício, gerado agora, nunca escrito em arquivo
        for text in (
            f"X-Simpleti-Upstream-Token: {fake}",
            f"'X-Simpleti-Upstream-Token: {fake}'",
            f'{{"upstream_token": "{fake}"}}',
            f"'upstream_token' => '{fake}',",
            f"VLLM_API_KEY={fake} ./deploy/serve_vllm.sh",
            f'env = {{"VLLM_API_KEY": "{fake}"}}',
            f"vllm serve --api-key {fake}",
            f'GATE_TOKEN = "{fake}"',
        ):
            with self.subTest(text=text.replace(fake, "<gerado>")):
                self.assertTrue(scan_text(text))
        for text in (
            "VLLM_API_KEY=<random token> ./deploy/serve_vllm.sh",
            "$headers[] = 'X-Simpleti-Upstream-Token: ' . $upstreamToken;",
            '{"upstream_url": url, "upstream_token": GATE_TOKEN}',
            'body = {"upstream_token": "A" * 43}',
            'TOKEN = "A" * 43',
            "VLLM_TOKEN = secrets.token_urlsafe(32)",
            "GATE_TOKEN = secrets.token_urlsafe(32)  # only the gateway sends it to the gate",
            'assertEqual(env["VLLM_API_KEY"], serve_colab.VLLM_TOKEN)',
            'headers={"Authorization": f"Bearer {VLLM_TOKEN}"}',
        ):
            with self.subTest(text=text):
                self.assertEqual(scan_text(text), [])

    def test_no_literal_token_in_any_versioned_file(self) -> None:
        offenders = []
        scanned = 0
        for path in repo_files():
            rel = path.relative_to(ROOT).as_posix()
            if rel == SELF:
                continue
            text = read_text(path)
            if text is None:
                continue
            scanned += 1
            offenders.extend(f"{rel}:{number}: {label}" for number, label in scan_text(text))
        self.assertGreater(scanned, 100)  # o scan olhou o repo de verdade
        self.assertEqual(offenders, [])

    def test_no_high_entropy_string_literal_in_the_files_of_this_issue(self) -> None:
        looks_like_token = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_-]{32,}$")
        for rel in (
            "deploy/serve_colab.py",
            "tests/test_tunnel_auth.py",
            "tests/test_flow_18.py",
            "tests/test_contract_18.py",
        ):
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertIsNone(looks_like_token.match(node.value), f"{rel}:{node.lineno}")
        for rel in ("gateway/set_upstream.php", "gateway/upstream_token.php", "deploy/serve_vllm.sh"):
            for number, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
                for quoted in re.findall(r"""['"]([A-Za-z0-9_-]{32,})['"]""", line):
                    self.assertFalse(re.search(r"\d", quoted) and re.search(r"[A-Za-z]", quoted), f"{rel}:{number}")

    def test_docs_tell_the_owner_how_to_put_a_key_in_front(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        guide = (ROOT / "deploy" / "DISTRIBUTION_GUIDE.md").read_text(encoding="utf-8")
        for text in (readme, guide):
            self.assertIn("127.0.0.1", text)
            self.assertIn("GET /v1/models", text)
            self.assertIn("POST /v1/chat/completions", text)
            self.assertIn("VLLM_API_KEY=<random token>", text)
        self.assertIn("/metrics", guide)


if __name__ == "__main__":
    unittest.main()
