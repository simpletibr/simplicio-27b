"""Flow test for issue #21: scripts/live_acceptance.py ponta a ponta contra um gateway falso.

Sobe o FakeGateway (tests/test_live_acceptance.py) numa porta efêmera de 127.0.0.1 e roda o script como
processo de verdade (python -I, ambiente mínimo), então confere código de saída, stdout, stderr e recibo.
Sem rede externa, sem produção, sem chave real: o token abaixo é inventado.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_live_acceptance import IDS, FakeGateway  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "live_acceptance.py"
TOKEN = "sk-flow21-4b7e9c0d1f3a5c6e-NOT-A-REAL-KEY"


class Sink(BaseHTTPRequestHandler):
    """Outro host, alvo de um redirect: guarda tudo o que chega."""

    seen: list[dict] = []

    def log_message(self, *args) -> None:
        pass

    def _record(self) -> None:
        Sink.seen.append({"method": self.command, "headers": dict(self.headers.items())})
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_POST = _record


class FlowTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeGateway.reset(TOKEN)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGateway)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.port = self.server.server_address[1]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        FakeGateway.reset()

    def cli(self, *extra: str, port: int | None = None) -> tuple[subprocess.CompletedProcess, str | None]:
        """Roda o script; devolve o processo e o texto do recibo gravado em --out-dir (ou None)."""
        url = f"http://127.0.0.1:{port or self.port}/v1"
        env = {"PATH": os.environ.get("PATH", ""), "SIMPLETI_API_KEY": TOKEN}
        with tempfile.TemporaryDirectory() as tmp:
            where = [] if "--out" in extra else ["--out-dir", tmp]
            proc = subprocess.run([sys.executable, "-I", str(SCRIPT), "--base-url", url, "--timeout", "30",
                                   *where, *extra], env=env, capture_output=True, text=True, timeout=180)
            files = sorted(Path(tmp).glob("acceptance-*.json"))
            text = files[0].read_text(encoding="utf-8") if files else None
        return proc, text

    def assertNoToken(self, proc: subprocess.CompletedProcess, text: str | None) -> None:
        self.assertNotIn(TOKEN, proc.stdout)
        self.assertNotIn(TOKEN, proc.stderr)
        self.assertNotIn(TOKEN, text or "")

    def test_all_pass_exits_zero(self) -> None:
        proc, text = self.cli()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        assert text is not None
        receipt = json.loads(text)
        self.assertTrue(receipt["passed"])
        self.assertEqual([c["id"] for c in receipt["checks"]], IDS)
        self.assertTrue(all(c["passed"] and not c["problems"] for c in receipt["checks"]))
        lines = proc.stdout.splitlines()
        self.assertEqual(lines[:-1], [f"PASS {i}" for i in IDS])
        self.assertTrue(lines[-1].startswith("recibo: "), lines[-1])
        self.assertEqual(proc.stderr, "")

    def test_missing_finish_reason_fails_with_reason(self) -> None:
        FakeGateway.mode = "no_finish"
        proc, text = self.cli()
        self.assertNotEqual(proc.returncode, 0)
        assert text is not None
        receipt = json.loads(text)
        self.assertFalse(receipt["passed"])
        todo = next(c for c in receipt["checks"] if c["id"] == "i5-todo-200-valido")
        self.assertFalse(todo["passed"])
        self.assertTrue(any("finish_reason" in p for p in todo["problems"]), todo["problems"])
        first = next(c for c in receipt["checks"] if c["id"] == "i2-id-simplicio-27b")
        self.assertFalse(first["passed"])
        self.assertTrue(any("finish_reason" in p for p in first["problems"]), first["problems"])
        self.assertIn("FAIL i2-id-simplicio-27b: finish_reason", proc.stdout)
        self.assertEqual(receipt["summary"]["failed"], len(receipt["summary"]["failed_ids"]))

    def test_token_never_leaks_even_when_server_echoes_it(self) -> None:
        FakeGateway.mode = "echo_key"  # o gateway devolve a chave dentro do content
        proc, text = self.cli()
        self.assertNoToken(proc, text)
        assert text is not None
        self.assertIn("<chave omitida>", text)  # a chave chegou ao recibo e foi removida
        self.assertTrue(any(r["headers"].get("authorization") == f"Bearer {TOKEN}" for r in FakeGateway.requests))

    def test_token_never_leaks_on_failure_paths(self) -> None:
        for mode in ("forbidden", "down", "no_finish", "xml_content"):
            with self.subTest(mode=mode):
                FakeGateway.mode = mode
                proc, text = self.cli()
                self.assertNotEqual(proc.returncode, 0)
                self.assertNoToken(proc, text)

    def test_wrong_key_fails_with_401_never_pass(self) -> None:
        FakeGateway.token = "outra-chave"
        proc, text = self.cli()
        self.assertEqual(proc.returncode, 1)
        assert text is not None
        receipt = json.loads(text)
        self.assertEqual(receipt["summary"]["passed"], 0)
        self.assertIn("status 401", proc.stdout)
        self.assertNoToken(proc, text)

    def test_forbidden_403_is_fail_with_reason(self) -> None:
        FakeGateway.mode = "forbidden"
        proc, text = self.cli()
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("PASS ", proc.stdout)
        self.assertIn("status 403", proc.stdout)
        assert text is not None
        self.assertFalse(json.loads(text)["passed"])

    def test_redirect_is_not_followed_and_key_is_not_forwarded(self) -> None:
        Sink.seen.clear()
        other = ThreadingHTTPServer(("127.0.0.1", 0), Sink)
        threading.Thread(target=other.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(other.server_close)
        self.addCleanup(other.shutdown)
        FakeGateway.mode = "redirect"
        FakeGateway.redirect_to = f"http://127.0.0.1:{other.server_address[1]}/capture"
        proc, text = self.cli()
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(Sink.seen, [])  # o Authorization nunca chegou ao outro host
        self.assertIn("status 302", proc.stdout)
        self.assertNotIn("PASS ", proc.stdout)
        self.assertNoToken(proc, text)

    def test_unreachable_server_is_fail_not_crash(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        proc, text = self.cli(port=port)  # nada escuta nessa porta
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stdout + proc.stderr)
        self.assertNotIn("PASS ", proc.stdout)
        assert text is not None
        receipt = json.loads(text)
        self.assertEqual(receipt["summary"]["failed"], 13)
        self.assertTrue(all(c["problems"] for c in receipt["checks"]))
        self.assertTrue(any("sem resposta" in p for c in receipt["checks"] for p in c["problems"]))

    def test_down_phase_passes_when_upstream_is_down(self) -> None:
        FakeGateway.mode = "down"
        proc, text = self.cli("--phase", "down")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("PASS i5-upstream-fora", proc.stdout)
        assert text is not None
        receipt = json.loads(text)
        self.assertEqual(receipt["phase"], "down")
        self.assertEqual([c["id"] for c in receipt["checks"]], ["i5-upstream-fora"])
        self.assertEqual(receipt["checks"][0]["actual"]["retry_after"], "30")

    def test_out_dash_prints_pure_json_on_stdout(self) -> None:
        proc, text = self.cli("--out", "-")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIsNone(text)  # nada gravado em --out-dir
        receipt = json.loads(proc.stdout)
        self.assertTrue(receipt["passed"])
        self.assertEqual(proc.stderr.splitlines(), [f"PASS {i}" for i in IDS])
        self.assertNoToken(proc, None)

    def test_out_file_is_written_exactly_there(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sub" / "recibo.json"
            proc, _ = self.cli("--out", str(target))
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue(json.loads(target.read_text(encoding="utf-8"))["passed"])
            self.assertIn(f"recibo: {target}", proc.stdout)


if __name__ == "__main__":
    unittest.main()
