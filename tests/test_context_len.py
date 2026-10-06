"""Issue #4: one context length everywhere."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

from serve_colab import load_context, vllm_cmd  # noqa: E402

SKIP_PARTS = {".git", "__pycache__", ".simplicio", "node_modules", "tests"}
LEN_RE = re.compile(r"(?:max-model-len|num_ctx)\s*[\"'= ]*(\d+)(?!\d)")


class ContextLenTests(unittest.TestCase):
    def test_budget_fits_measured_prompt(self) -> None:
        ctx = load_context()
        self.assertGreaterEqual(
            ctx["MAX_MODEL_LEN"],
            ctx["MEASURED_AGENT_PROMPT"] + ctx["OUTPUT_BUDGET"],
        )
        self.assertGreater(ctx["MEASURED_AGENT_PROMPT"], 30000)

    def test_literals_match_constant(self) -> None:
        expected = str(load_context()["MAX_MODEL_LEN"])
        found: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            if path.suffix in {".png", ".svg", ".gguf", ".jsonl"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for match in LEN_RE.finditer(text):
                found.append(f"{path.relative_to(ROOT)}:{match.group(1)}")
                self.assertEqual(match.group(1), expected, f"{path} {match.group(0)}")
        self.assertTrue(any(":40960" in row or f":{expected}" in row for row in found), found)

    def test_vllm_cmd_uses_constant(self) -> None:
        ctx = load_context()
        cmd = vllm_cmd(ctx)
        idx = cmd.index("--max-model-len")
        self.assertEqual(cmd[idx + 1], str(ctx["MAX_MODEL_LEN"]))
        self.assertNotIn("16384", cmd)
        self.assertIn("simpleti/simplicio-27b", cmd)

    def test_modelfiles_agree(self) -> None:
        ctx = load_context()
        for rel in ("Modelfile", "deploy/Modelfile"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn(f"PARAMETER num_ctx {ctx['MAX_MODEL_LEN']}", text)

    def test_no_colab_16384_max_model_len(self) -> None:
        py = (ROOT / "deploy" / "serve_colab.py").read_text(encoding="utf-8")
        self.assertNotIn("16384", py)


if __name__ == "__main__":
    unittest.main()
