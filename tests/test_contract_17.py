"""Contract test for issue #17: o que gateway/ promete a quem o consome (cliente, proxy, dono no deploy).

Fixa, sem rede e sem produção:
  * o schema de GET /v1/status: exatamente `upstream_registered` (bool), `updated_at` (ISO 8601 UTC ou
    null) e `deploy_sha` (40 hex ou null), em cada estado possível; o código recusa não-GET com 405 e
    `Allow: GET`, responde `application/json; charset=utf-8` e `Cache-Control: no-store`;
  * status.php lê o mesmo arquivo de estado que set_upstream.php grava (mesma linha `$file = ` e os
    campos `upstream_url` e `updated_at`) e nunca devolve `upstream_url`;
  * o schema de GET /v1/models em gateway/models.json: campos, ids na ordem, valores de deploy/;
  * gateway/ não tem segredo, `Bearer` literal com token, `hf_`, `sk-`, nem host de túnel real, nem
    arquivo de dados (`.env`, simpleti-upstream.json); `.env.example`, se existir, não traz valores;
  * `php -l` passa em todo .php de gateway/ (pulado com motivo se `php` não estiver instalado);
  * o set_upstream.php que ficava em deploy/ saiu do repo e nada mais aponta para o caminho antigo;
  * gateway/DEPLOY_SHA é ignorado pelo git (o dono o gera no deploy).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATEWAY = ROOT / "gateway"
STATUS_KEYS = {"upstream_registered", "updated_at", "deploy_sha"}
ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SHA = "0123456789abcdef" * 2 + "01234567"

SECRET_PATTERNS = {
    "chave literal (password/secret/token/api_key = 'valor')": re.compile(
        r"""(pass(word|wd)?|secret|token|api_?key)['"]?\s*(=>|=|,)\s*['"][^'"]{8,}['"]""", re.I
    ),
    "Bearer literal com token": re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    "token do Hugging Face (hf_)": re.compile(r"\bhf_[A-Za-z0-9]{16,}"),
    "chave sk-": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    "chave privada": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "host de túnel real": re.compile(r"https?://[a-z0-9-]+\.trycloudflare\.com", re.I),
    "URL com credencial": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
}
DATA_FILE_NAMES = {".env", "simpleti-upstream.json", "upstream.json"}
# Único falso positivo conhecido: o `"unit": "token", "cost_usd"` do catálogo (a unidade de preço, não um segredo).
BENIGN_MATCHES = frozenset({'token", "cost_usd"'})


def gateway_files() -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(GATEWAY):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        found.extend(Path(dirpath) / n for n in filenames if not n.endswith(".pyc"))
    return sorted(found)


def run_status(state=None, sha=None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        shutil.copy(GATEWAY / "status.php", tmp_dir / "status.php")
        if state is not None:
            (tmp_dir / "upstream.json").write_text(
                state if isinstance(state, str) else json.dumps(state), encoding="utf-8"
            )
        if sha is not None:
            (tmp_dir / "DEPLOY_SHA").write_text(sha + "\n", encoding="utf-8")
        proc = subprocess.run(
            ["php", str(tmp_dir / "status.php")],
            env=dict(os.environ, SIMPLETI_UPSTREAM_FILE=str(tmp_dir / "upstream.json")),
            capture_output=True, text=True, timeout=30, check=True,
        )
    return json.loads(proc.stdout)


@unittest.skipUnless(shutil.which("php"), "php não instalado: sem ele não há como executar status.php")
class StatusSchemaTests(unittest.TestCase):
    def assert_schema(self, body: dict) -> None:
        self.assertEqual(set(body), STATUS_KEYS)
        self.assertIsInstance(body["upstream_registered"], bool)
        if body["updated_at"] is not None:
            self.assertRegex(body["updated_at"], ISO_UTC)
        if body["deploy_sha"] is not None:
            self.assertRegex(body["deploy_sha"], r"^[0-9a-f]{40}$")

    def test_schema_in_every_state(self) -> None:
        states = {
            "sem estado": {},
            "registrado": {"state": {"upstream_url": "https://x-y.trycloudflare.com", "updated_at": 1790000000}},
            "registrado, updated_at ausente": {"state": {"upstream_url": "https://x-y.trycloudflare.com"}},
            "url vazia": {"state": {"upstream_url": "", "updated_at": 1}},
            "url não string": {"state": {"upstream_url": 7, "updated_at": 1}},
            "estado corrompido": {"state": "{not json"},
            "sha válido": {"sha": SHA},
            "sha inválido": {"sha": "XYZ"},
            "sha maiúsculo": {"sha": SHA.upper()},
        }
        for name, kwargs in states.items():
            with self.subTest(state=name):
                self.assert_schema(run_status(**kwargs))

    def test_semantics(self) -> None:
        self.assertEqual(
            run_status({"upstream_url": "https://x-y.trycloudflare.com", "updated_at": 1790000000}, SHA),
            {"upstream_registered": True, "updated_at": "2026-09-21T14:13:20Z", "deploy_sha": SHA},
        )
        # sem `updated_at` inteiro, registrado mas sem carimbo; URL vazia ou não string não conta como registro
        self.assertEqual(
            run_status({"upstream_url": "https://x-y.trycloudflare.com"}),
            {"upstream_registered": True, "updated_at": None, "deploy_sha": None},
        )
        for bad in ("", 7):
            with self.subTest(upstream_url=bad):
                self.assertIs(run_status({"upstream_url": bad, "updated_at": 1})["upstream_registered"], False)
        self.assertIsNone(run_status(sha=SHA.upper())["deploy_sha"])

    def test_never_returns_the_tunnel(self) -> None:
        url = "https://very-secret-name.trycloudflare.com"
        text = json.dumps(run_status({"upstream_url": url, "updated_at": 1}))
        for leak in ("very-secret-name", "trycloudflare", "upstream_url"):
            self.assertNotIn(leak, text)


class StatusSourceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.status = (GATEWAY / "status.php").read_text(encoding="utf-8")
        self.set_upstream = (GATEWAY / "set_upstream.php").read_text(encoding="utf-8")

    def test_headers_and_method_contract(self) -> None:
        self.assertIn("header('Content-Type: application/json; charset=utf-8');", self.status)
        self.assertIn("header('Cache-Control: no-store');", self.status)
        self.assertIn("http_response_code(405);", self.status)
        self.assertIn("header('Allow: GET');", self.status)
        self.assertIn("!== 'GET'", self.status)

    def test_output_has_only_the_three_fixed_keys(self) -> None:
        block = self.status[self.status.rindex("echo json_encode(["):]
        keys = set(re.findall(r"'([a-z_]+)'\s*=>", block))
        self.assertEqual(keys, STATUS_KEYS)

    def test_reads_the_state_set_upstream_writes(self) -> None:
        def file_line(src: str) -> str:
            return next(line.strip() for line in src.splitlines() if line.startswith("$file = "))

        self.assertEqual(file_line(self.status), file_line(self.set_upstream))
        self.assertIn("SIMPLETI_UPSTREAM_FILE", file_line(self.status))
        for field in ("upstream_url", "updated_at"):
            self.assertIn(f"'{field}'", self.status)
            self.assertIn(f"'{field}'", self.set_upstream)

    def test_set_upstream_keeps_the_bearer_contract(self) -> None:
        self.assertIn("Authorization", self.set_upstream)
        self.assertIn("'Bearer '", self.set_upstream)
        self.assertNotIn("$_GET", self.set_upstream)
        for code in ("405", "500", "401", "400", "200"):
            self.assertIn(f"respond({code},", self.set_upstream)


class ModelsCatalogContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads((GATEWAY / "models.json").read_text(encoding="utf-8"))["data"]

    def test_fields_and_values(self) -> None:
        instruct = json.loads((ROOT / "deploy" / "hf_instruct.json").read_text(encoding="utf-8"))
        ctx = dict(
            line.split("=", 1)
            for line in (ROOT / "deploy" / "context.env").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        )
        self.assertEqual([m["id"] for m in self.data], ["simpleti/simplicio-27b", "simplicio-27b"])
        for item in self.data:
            with self.subTest(id=item["id"]):
                for field in (
                    "id", "name", "hugging_face_id", "created", "description",
                    "context_length", "architecture", "input_modalities", "output_modalities",
                ):
                    self.assertIn(field, item)
                self.assertEqual(item["created"], 1791136800)
                self.assertEqual(item["hugging_face_id"], "wesleysimplicio/Simplicio-27B")
                self.assertEqual(item["context_length"], int(ctx["MAX_MODEL_LEN"]))
                self.assertEqual(item["architecture"]["instruct_type"], instruct["instruct_type"])
                supported = item["input_modalities"][0]["supported_inputs"]["max_context_length"]
                self.assertEqual(supported, {"value": int(ctx["MAX_MODEL_LEN"]), "unit": "token"})
                out = item["output_modalities"][0]["supported_outputs"]["max_output_tokens"]
                self.assertEqual(out, {"value": int(ctx["OUTPUT_BUDGET"]), "unit": "token"})
        self.assertNotEqual(self.data[0]["id"], self.data[1]["id"])
        self.assertEqual({k: v for k, v in self.data[0].items() if k != "id"},
                         {k: v for k, v in self.data[1].items() if k != "id"})

    def test_score_claim_carries_the_readme_qualifier(self) -> None:
        # Mesmo qualificador do README: as 56/120 execuções são um teste por casamento de texto.
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("passed a text-matching check", readme)
        for item in self.data:
            with self.subTest(id=item["id"]):
                text = item["description"]
                self.assertEqual(text.count("56 of 120 runs"), 1)
                self.assertIn("passed a text-matching check", text)
                self.assertNotIn("passed 56 of 120", text)

    def test_price_does_not_contradict_the_readme(self) -> None:
        # O preço por token é decisão do dono (opção 1 ou 2 da #14). Enquanto o README disser que não há
        # preço público, nem pricing.json nem models.json podem trazer valor.
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        price = json.loads((GATEWAY / "pricing.json").read_text(encoding="utf-8"))
        if "no public per-token price" in readme:
            self.assertEqual(price, {})
            for item in self.data:
                self.assertNotIn("pricing", item)
                for side in ("input_modalities", "output_modalities"):
                    self.assertNotIn("pricing", item[side][0])
        else:
            self.assertEqual(set(price) - {"prompt", "completion", "cached_prompt"}, set())

    def test_pricing_shape_when_present(self) -> None:
        for item in self.data:
            if "pricing" not in item:
                continue
            with self.subTest(id=item["id"]):
                self.assertEqual(set(item["pricing"]), {"prompt", "completion", "request", "image"})
                kinds = [p["type"] for p in item["input_modalities"][0]["pricing"]]
                self.assertEqual(kinds, ["prompt", "cached_prompt"])
                self.assertEqual([p["type"] for p in item["output_modalities"][0]["pricing"]], ["completion"])


class NoSecretsTests(unittest.TestCase):
    def test_gateway_dir_is_not_empty(self) -> None:
        names = {p.name for p in gateway_files()}
        self.assertTrue({"status.php", "set_upstream.php", "build_catalog.py", "models.json"} <= names, names)

    def test_no_secret_patterns(self) -> None:
        hits = []
        for path in gateway_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in SECRET_PATTERNS.items():
                for match in pattern.finditer(text):
                    if re.sub(r"\s+", " ", match.group(0)) in BENIGN_MATCHES:
                        continue
                    line = text.count("\n", 0, match.start()) + 1
                    hits.append(f"{path.relative_to(ROOT)}:{line}: {label}")
        self.assertEqual(hits, [])

    def test_patterns_catch_what_they_should(self) -> None:
        # Controle negativo: sem isto, um padrão quebrado deixaria o teste acima verde sem proteger nada.
        samples = {
            "chave literal (password/secret/token/api_key = 'valor')": "$token = 'abcdefgh12345';",
            "Bearer literal com token": "Authorization: Bearer abcdefghijklmnop1234",
            "token do Hugging Face (hf_)": "HF=hf_" + "a" * 20,
            "chave sk-": "k = sk-" + "b" * 20,
            "chave privada": "-----BEGIN RSA PRIVATE KEY-----",
            "host de túnel real": "https://abc-def.trycloudflare.com",
            "URL com credencial": "https://user:pw@example.com/x",
        }
        self.assertEqual(set(samples), set(SECRET_PATTERNS))
        for label, text in samples.items():
            with self.subTest(label=label):
                self.assertIsNotNone(SECRET_PATTERNS[label].search(text))

    def test_no_data_files_and_env_example_has_no_values(self) -> None:
        for path in gateway_files():
            self.assertNotIn(path.name, DATA_FILE_NAMES, f"dado versionado em gateway/: {path}")
            if path.name == ".env.example":
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        name, sep, value = line.partition("=")
                        self.assertEqual((bool(sep), value.strip()), (True, ""), f".env.example com valor: {name}")

    def test_deploy_sha_is_ignored_not_tracked(self) -> None:
        self.assertEqual(
            subprocess.run(["git", "check-ignore", "-q", "gateway/DEPLOY_SHA"], cwd=ROOT).returncode, 0
        )
        tracked = subprocess.run(
            ["git", "ls-files", "gateway/DEPLOY_SHA"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
        self.assertEqual(tracked, "")


@unittest.skipUnless(shutil.which("php"), "php não instalado: php -l não foi executado")
class PhpLintTests(unittest.TestCase):
    def test_php_lint_every_gateway_file(self) -> None:
        php_files = [p for p in gateway_files() if p.suffix == ".php"]
        self.assertTrue(php_files)
        for path in php_files:
            with self.subTest(file=path.name):
                proc = subprocess.run(["php", "-l", str(path)], capture_output=True, text=True, timeout=30)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertIn("No syntax errors detected", proc.stdout)


class MovedFromDeployTests(unittest.TestCase):
    def test_old_path_is_gone_and_unreferenced(self) -> None:
        self.assertFalse(ROOT.joinpath("deploy", "set_upstream.php").exists())
        self.assertTrue((GATEWAY / "set_upstream.php").is_file())
        old = "deploy" + "/set_upstream"
        old_join = '"deploy" ' + '/ "set_upstream.php"'
        live = "set_upstream" + ".live"
        proc = subprocess.run(
            ["git", "grep", "-n", "-e", old, "-e", live, "-e", old_join],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual((proc.returncode, proc.stdout), (1, ""))

    def test_public_url_did_not_change(self) -> None:
        source = (ROOT / "deploy" / "serve_colab.py").read_text(encoding="utf-8")
        self.assertIn("https://simpleti.com.br/api/set_upstream.php", source)


if __name__ == "__main__":
    unittest.main()
