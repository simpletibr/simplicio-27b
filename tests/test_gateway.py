"""Issue #17: o catálogo de gateway/models.json sai do repo e gateway/status.php responde o estado público."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gateway"))
sys.path.insert(0, str(ROOT / "deploy"))

from build_catalog import IDS, OUT, PRICING, render  # noqa: E402
from serve_colab import load_context  # noqa: E402


class CatalogTests(unittest.TestCase):
    def test_committed_file_is_fresh(self) -> None:
        self.assertEqual(OUT.read_text(encoding="utf-8"), render())

    def test_values_come_from_repo(self) -> None:
        instruct = json.loads((ROOT / "deploy" / "hf_instruct.json").read_text(encoding="utf-8"))
        ctx = load_context()
        data = json.loads(render())["data"]
        self.assertEqual([m["id"] for m in data], ["simpleti/simplicio-27b", "simplicio-27b"])
        self.assertEqual(sorted(IDS), sorted(instruct["served_model_names"]))
        for item in data:
            with self.subTest(id=item["id"]):
                self.assertEqual(item["context_length"], ctx["MAX_MODEL_LEN"])
                self.assertEqual(item["architecture"]["instruct_type"], instruct["instruct_type"])
                out = item["output_modalities"][0]["supported_outputs"]["max_output_tokens"]["value"]
                self.assertEqual(out, ctx["OUTPUT_BUDGET"])

    def test_pricing_follows_file(self) -> None:
        price = json.loads(PRICING.read_text(encoding="utf-8"))
        for item in json.loads(render())["data"]:
            with self.subTest(id=item["id"]):
                self.assertEqual("pricing" in item, bool(price))
                if price:
                    self.assertEqual(item["pricing"]["prompt"], price["prompt"])
                    self.assertEqual(
                        item["output_modalities"][0]["pricing"][0]["cost_usd"], price["completion"]
                    )

    def test_no_withdrawn_claims(self) -> None:
        text = render()
        for claim in ("96.5", "480", "Top 12", "32768", "hallucination"):
            with self.subTest(claim=claim):
                self.assertNotIn(claim, text)
        self.assertEqual(text.count("56 of 120 runs"), 2)


@unittest.skipIf(shutil.which("php") is None, "php não instalado")
class StatusTests(unittest.TestCase):
    def run_status(self, state=None, sha=None) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            shutil.copy(ROOT / "gateway" / "status.php", tmp_dir / "status.php")
            if state is not None:
                (tmp_dir / "upstream.json").write_text(json.dumps(state), encoding="utf-8")
            if sha is not None:
                (tmp_dir / "DEPLOY_SHA").write_text(sha + "\n", encoding="utf-8")
            proc = subprocess.run(
                ["php", str(tmp_dir / "status.php")],
                env=dict(os.environ, SIMPLETI_UPSTREAM_FILE=str(tmp_dir / "upstream.json")),
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        self.assertNotIn("trycloudflare", proc.stdout)
        return json.loads(proc.stdout)

    def test_no_state(self) -> None:
        self.assertEqual(
            self.run_status(),
            {"upstream_registered": False, "updated_at": None, "deploy_sha": None},
        )

    def test_registered_with_sha(self) -> None:
        self.assertEqual(
            self.run_status({"upstream_url": "https://abc.trycloudflare.com", "updated_at": 0}, "a" * 40),
            {"upstream_registered": True, "updated_at": "1970-01-01T00:00:00Z", "deploy_sha": "a" * 40},
        )

    def test_bad_sha_is_null(self) -> None:
        self.assertIsNone(self.run_status(sha="not-a-sha")["deploy_sha"])


if __name__ == "__main__":
    unittest.main()
