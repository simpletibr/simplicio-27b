"""Contract test for issue #21: o que scripts/live_acceptance.py promete a quem lê o recibo e ao gateway.

  * o recibo tem chaves fixas e um campo de versão (schema), timestamps UTC e URL-base sem credenciais;
  * cada checagem traz id, issue, criterion, expected, passed, problems, actual e exchanges;
  * o script não fixa a URL de produção (a URL-base vem de --base-url ou SIMPLETI_BASE_URL);
  * o script usa só a biblioteca padrão e não importa nada de deploy/;
  * códigos de saída: 0 tudo passou, 1 alguma falhou, 2 falta configuração;
  * os requests seguem a tabela da issue: POST /v1/chat/completions, Bearer, JSON, User-Agent próprio, temperature 0.

Roda main() em-processo contra o FakeGateway (127.0.0.1, porta efêmera). Sem rede externa, sem produção.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import runpy
import sys
import tempfile
import threading
import unittest
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import live_acceptance  # noqa: E402
from test_live_acceptance import IDS, FakeGateway  # noqa: E402

SCRIPT = ROOT / "scripts" / "live_acceptance.py"
SCRIPT_TEXT = SCRIPT.read_text(encoding="utf-8")
KEY = "contract-key-21"
TOP_KEYS = ["schema", "base_url", "phase", "git_commit", "started_at", "finished_at", "context", "passed",
            "summary", "checks"]
SUMMARY_KEYS = ["total", "passed", "failed", "failed_ids"]
CHECK_KEYS = ["id", "issue", "criterion", "expected", "passed", "problems", "actual", "exchanges"]
EXCHANGE_KEYS = ["request", "status", "latency_ms", "content_type", "retry_after", "body"]
ACTUAL_KEYS = ["status", "finish_reason", "content", "tool_calls", "usage", "latency_ms"]
ISSUE_OF = {i: int(i[1]) for i in IDS}
UTC_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
EXPECTED_MAX_TOKENS = [32, 32, 256, 256, 64, 512, 512, 512, 1, 1, 16, 16, 64, 64]


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeGateway.reset(KEY)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGateway)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        FakeGateway.reset()

    def run_script(self, *args: str, key: str | None = KEY, env: dict[str, str] | None = None,
                   url: str | None = None) -> dict:
        environ = {"PATH": os.environ.get("PATH", ""), **(env or {})}
        if key is not None:
            environ["SIMPLETI_API_KEY"] = key
        argv = list(args) if "--base-url" in args or url == "" else ["--base-url", url or self.base_url, *args]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, environ, clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            out_dir = Path(tmp) / "novo"  # não existe: o script que cria (ou não)
            code = live_acceptance.main([*argv, "--out-dir", str(out_dir)])
            files = sorted(out_dir.glob("acceptance-*.json")) if out_dir.exists() else []
            text = files[0].read_text(encoding="utf-8") if files else None
            return {"code": code, "out": out.getvalue(), "err": err.getvalue(), "text": text,
                    "receipt": json.loads(text) if text else None, "dir_created": out_dir.exists(),
                    "file": files[0].name if files else None}

    # ---- (a) schema do recibo ----

    def test_receipt_has_fixed_keys_and_schema_version(self) -> None:
        run = self.run_script(env={})
        self.assertEqual(run["code"], 0, run["out"] + run["err"])
        receipt = run["receipt"]
        self.assertEqual(list(receipt), TOP_KEYS)
        self.assertEqual(receipt["schema"], "simplicio.live-acceptance/v1")
        self.assertEqual(receipt["schema"], live_acceptance.SCHEMA)
        self.assertEqual(receipt["phase"], "up")
        self.assertIs(receipt["passed"], True)
        self.assertEqual(list(receipt["summary"]), SUMMARY_KEYS)
        self.assertEqual(receipt["summary"], {"total": 13, "passed": 13, "failed": 0, "failed_ids": []})
        self.assertEqual(sorted(receipt["context"]), ["MAX_MODEL_LEN", "MEASURED_AGENT_PROMPT", "OUTPUT_BUDGET"])
        commit = receipt["git_commit"]
        self.assertTrue(commit is None or re.fullmatch(r"[0-9a-f]{40}", commit), commit)

    def test_timestamps_are_utc_and_file_name_matches(self) -> None:
        run = self.run_script()
        receipt = run["receipt"]
        for name in ("started_at", "finished_at"):
            self.assertRegex(receipt[name], UTC_STAMP)
        started = datetime.strptime(receipt["started_at"], "%Y-%m-%dT%H:%M:%SZ")
        finished = datetime.strptime(receipt["finished_at"], "%Y-%m-%dT%H:%M:%SZ")
        self.assertLessEqual(started, finished)
        self.assertEqual(run["file"], f"acceptance-{started.strftime('%Y%m%dT%H%M%SZ')}.json")

    def test_base_url_in_receipt_has_no_credentials(self) -> None:
        run = self.run_script(url=self.base_url + "/")
        self.assertEqual(run["receipt"]["base_url"], self.base_url)  # sem a barra final
        for bad in ("@", "?", "#"):
            self.assertNotIn(bad, run["receipt"]["base_url"])

    def test_check_and_exchange_shapes(self) -> None:
        receipt = self.run_script()["receipt"]
        self.assertEqual([c["id"] for c in receipt["checks"]], IDS)
        for chk in receipt["checks"]:
            self.assertEqual(list(chk), CHECK_KEYS, chk["id"])
            self.assertEqual(chk["issue"], ISSUE_OF[chk["id"]], chk["id"])
            self.assertTrue(chk["criterion"].strip() and chk["expected"].strip(), chk["id"])
            self.assertIsInstance(chk["passed"], bool)
            self.assertIsInstance(chk["problems"], list)
            if chk["id"] != "i5-todo-200-valido":
                self.assertTrue(chk["exchanges"], chk["id"])
                self.assertEqual(list(chk["actual"]), ACTUAL_KEYS, chk["id"])
            for ex in chk["exchanges"]:
                self.assertEqual(list(ex), EXCHANGE_KEYS, chk["id"])
                self.assertEqual(ex["status"], 200 if chk["id"] != "i4-acima-do-teto" else 400)
                self.assertIsInstance(ex["latency_ms"], int)
        self.assertEqual(self.check(receipt, "i5-todo-200-valido")["actual"], {"exchanges_200": 13})

    def test_criteria_are_the_aceite_bullets(self) -> None:
        by_id = {c["id"]: c["criterion"] for c in self.run_script()["receipt"]["checks"]}
        self.assertTrue(by_id["i2-id-simplicio-27b"].startswith("`model` `simplicio-27b` e `simpleti/simplicio-27b`"))
        self.assertEqual(by_id["i2-id-simplicio-27b"], by_id["i2-id-simpleti-simplicio-27b"])
        self.assertIn('`tool_choice: "auto"`', by_id["i2-tool-auto"])
        self.assertIn('`tool_choice: "none"`', by_id["i2-tool-none-oi"])
        self.assertTrue(by_id["i3-pong"].startswith("“Reply with exactly the single word pong”"))
        self.assertEqual(by_id["i3-agente-oi-system"], by_id["i3-agente-oi-tools"])
        self.assertIn("`<system>Tool result`", by_id["i3-segundo-turno"])
        self.assertIn("31692 tokens", by_id["i4-prompt-31692"])
        self.assertIn("acima do teto", by_id["i4-acima-do-teto"])
        for check_id in IDS[-3:]:
            self.assertTrue(by_id[check_id].startswith("Uma completion 200 sempre traz `finish_reason`"), check_id)

    def test_down_phase_receipt_has_one_check(self) -> None:
        FakeGateway.mode = "down"
        run = self.run_script("--phase", "down")
        self.assertEqual(run["code"], 0)
        receipt = run["receipt"]
        self.assertEqual(list(receipt), TOP_KEYS)
        self.assertEqual(receipt["phase"], "down")
        chk = receipt["checks"][0]
        self.assertEqual((chk["id"], chk["issue"], chk["passed"]), ("i5-upstream-fora", 5, True))
        self.assertEqual(list(chk["actual"]), ["status", "retry_after", "body"])
        self.assertTrue(chk["criterion"].startswith("Matar o processo do notebook"))

    @staticmethod
    def check(receipt: dict, check_id: str) -> dict:
        return next(c for c in receipt["checks"] if c["id"] == check_id)

    # ---- (b) sem URL de produção fixa; (c) só stdlib ----

    def test_script_has_no_production_url(self) -> None:
        self.assertEqual(SCRIPT_TEXT.lower().count("simpleti.com.br"), 0)  # grep -c simpleti.com.br == 0
        hosts = re.findall(r"https?://[^\s\"'`)>\\]+", SCRIPT_TEXT)
        self.assertTrue(hosts)  # o prompt da checagem de webfetch usa https://example.com
        for found in hosts:
            self.assertTrue(found.startswith("https://example.com"), found)
        self.assertFalse(hasattr(live_acceptance, "DEFAULT_BASE_URL"))

    def test_script_uses_only_the_standard_library(self) -> None:
        names: set[str] = set()
        for node in ast.walk(ast.parse(SCRIPT_TEXT)):
            if isinstance(node, ast.Import):
                names |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "import relativo")
                names.add((node.module or "").split(".")[0])
        self.assertTrue(names)
        self.assertEqual(names - set(sys.stdlib_module_names), set(), "módulo fora da biblioteca padrão")
        for banned in ("importlib", "__import__", "sys.path"):
            self.assertNotIn(banned, SCRIPT_TEXT)
        # mesmo grep do critério de pronto da issue: nada de deploy/ nem de sys.path em scripts/
        grep = re.compile(r"^(from|import) .*(completion_gate|serve_colab)|sys\.path", re.MULTILINE)
        for path in (ROOT / "scripts").glob("*.py"):
            self.assertIsNone(grep.search(path.read_text(encoding="utf-8")), path.name)
        self.assertEqual(sorted((ROOT / "scripts").glob("requirements*")), [])

    # ---- códigos de saída e configuração ----

    def test_exit_codes(self) -> None:
        self.assertEqual(self.run_script()["code"], 0)
        FakeGateway.mode = "no_finish"
        self.assertEqual(self.run_script()["code"], 1)
        FakeGateway.mode = "good"
        self.assertEqual(self.run_script("--phase", "down")["code"], 1)  # upstream no ar: fase down falha

    def test_missing_key_exits_2_and_creates_nothing(self) -> None:
        run = self.run_script(key=None)
        self.assertEqual(run["code"], 2)
        self.assertIn("SIMPLETI_API_KEY ausente", run["err"])
        self.assertFalse(run["dir_created"])
        self.assertEqual(FakeGateway.requests, [])
        self.assertEqual(self.run_script(key="   ")["code"], 2)  # chave só com espaços também é ausente

    def test_malformed_key_exits_2_without_echo(self) -> None:
        for bad in ("abc\ndef-chave-torta", "abc def-chave-torta", "chave-tórta-é", "abc\tdef-chave-torta"):
            with self.subTest(key=bad):
                run = self.run_script(key=bad)
                self.assertEqual(run["code"], 2)
                self.assertNotIn("chave-t", run["out"] + run["err"])
                self.assertFalse(run["dir_created"])
        self.assertEqual(FakeGateway.requests, [])

    def test_missing_base_url_exits_2_without_any_default(self) -> None:
        run = self.run_script("--phase", "up", url="")
        self.assertEqual(run["code"], 2)
        self.assertIn("SIMPLETI_BASE_URL", run["err"])
        self.assertFalse(run["dir_created"])
        self.assertEqual(FakeGateway.requests, [])

    def test_base_url_from_environment(self) -> None:
        run = self.run_script("--phase", "up", url="", env={"SIMPLETI_BASE_URL": self.base_url})
        self.assertEqual(run["code"], 0, run["out"] + run["err"])
        self.assertEqual(run["receipt"]["base_url"], self.base_url)

    def test_base_url_with_credentials_or_query_is_rejected_without_echo(self) -> None:
        port = self.server.server_address[1]
        for bad in (f"http://usuario:segredo-url@127.0.0.1:{port}/v1", f"http://127.0.0.1:{port}/v1?key=segredo-url",
                    f"http://127.0.0.1:{port}/v1#segredo-url", "ftp://127.0.0.1/v1", "127.0.0.1:8000/v1"):
            with self.subTest(url=bad):
                run = self.run_script(url=bad)
                self.assertEqual(run["code"], 2)
                self.assertNotIn("segredo-url", run["out"] + run["err"])
                self.assertFalse(run["dir_created"])
        self.assertEqual(FakeGateway.requests, [])

    def test_stdout_protocol(self) -> None:
        run = self.run_script()
        lines = run["out"].splitlines()
        self.assertEqual(lines[:-1], [f"PASS {i}" for i in IDS])
        self.assertRegex(lines[-1], r"^recibo: .+acceptance-\d{8}T\d{6}Z\.json$")
        FakeGateway.mode = "no_finish"
        failed = self.run_script()["out"].splitlines()
        for line in failed[:-1]:
            self.assertRegex(line, r"^FAIL i\d-[a-z0-9-]+: .+$")

    # ---- o que o script manda ao gateway ----

    def test_requests_follow_the_issue_table(self) -> None:
        self.assertEqual(self.run_script()["code"], 0)
        requests = FakeGateway.requests
        self.assertEqual([r["body"]["max_tokens"] for r in requests], EXPECTED_MAX_TOKENS)
        for req in requests:
            self.assertEqual(req["path"], "/v1/chat/completions")
            self.assertEqual(req["headers"]["authorization"], f"Bearer {KEY}")
            self.assertEqual(req["headers"]["content-type"], "application/json")
            self.assertTrue(req["headers"]["user-agent"].startswith("simplicio-live-acceptance/"))
            self.assertEqual(req["body"]["temperature"], 0)
            self.assertIn(req["body"]["model"], ("simplicio-27b", "simpleti/simplicio-27b"))
        self.assertEqual([r["body"]["model"] for r in requests].count("simpleti/simplicio-27b"), 1)
        tooled = [r["body"] for r in requests if "tools" in r["body"]]
        self.assertEqual([b["tool_choice"] for b in tooled], ["auto", "none", "auto", "auto"])
        for body in tooled:
            for tool in body["tools"]:
                self.assertEqual(tool["type"], "function")
                self.assertEqual(tool["function"]["parameters"]["type"], "object")
        streams = [r["body"] for r in requests if r["body"].get("stream")]
        self.assertEqual(len(streams), 2)
        self.assertEqual(streams[0]["stream_options"], {"include_usage": True})
        self.assertNotIn("stream_options", streams[1])
        second = next(r["body"] for r in requests if r["body"]["messages"][-1]["role"] == "tool")
        arguments = second["messages"][2]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(arguments, str)  # string JSON: o vLLM converte em dict antes do template
        self.assertEqual(json.loads(arguments), {"url": "https://example.com"})

    def test_script_runs_as_main_module(self) -> None:
        with patch.object(sys, "argv", [str(SCRIPT)]), patch.dict(os.environ, {}, clear=True), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as raised:
                runpy.run_path(str(SCRIPT), run_name="__main__")
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("SIMPLETI_API_KEY ausente", err.getvalue())


if __name__ == "__main__":
    unittest.main()
