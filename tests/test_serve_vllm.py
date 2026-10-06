"""Guardrails for deploy/serve_vllm.sh, chat template, and README."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVE = (ROOT / "deploy" / "serve_vllm.sh").read_text(encoding="utf-8")
TEMPLATE = (ROOT / "deploy" / "chat_template_chatml.jinja").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
COLAB = (ROOT / "Simplicio_27B_Serve_Colab.ipynb").read_text(encoding="utf-8")
INSTRUCT = (ROOT / "deploy" / "hf_instruct.json").read_text(encoding="utf-8")
CONTEXT = (ROOT / "deploy" / "context.env").read_text(encoding="utf-8")


def max_model_len() -> str:
    for line in CONTEXT.splitlines():
        if line.startswith("MAX_MODEL_LEN="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("MAX_MODEL_LEN missing")


class ServeScriptTests(unittest.TestCase):
    def test_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(ROOT / "deploy" / "serve_vllm.sh")], check=True)

    def test_required_flags(self) -> None:
        for token in (
            "--enable-auto-tool-choice",
            "--tool-call-parser simplicio",
            "--tool-parser-plugin",
            "--reasoning-parser qwen3",
            "--chat-template",
            "--max-model-len \"${MAX_MODEL_LEN}\"",
            "simplicio-27b",
            "simpleti/simplicio-27b",
            "<|im_end|>",
            "</deliver>",
            "</tool>",
            "context.env",
        ):
            self.assertIn(token, SERVE, token)

    def test_old_preset_is_gone(self) -> None:
        self.assertNotIn("--chat-template-preset", SERVE)


class ChatTemplateTests(unittest.TestCase):
    def test_chatml_and_think_prefill(self) -> None:
        self.assertIn("<|im_start|>assistant", TEMPLATE)
        self.assertIn("<|im_end|>", TEMPLATE)
        self.assertIn("<think>", TEMPLATE)
        self.assertIn("<tool>", TEMPLATE)
        self.assertIn("at most one tool per turn", TEMPLATE)

    def test_hf_instruct_type(self) -> None:
        self.assertIn('"instruct_type": "chatml"', INSTRUCT)
        self.assertIn("qwen3", INSTRUCT)
        self.assertIn("simplicio", INSTRUCT)
        self.assertIn(max_model_len(), INSTRUCT)


class ReadmeTests(unittest.TestCase):
    def test_readme_serve_is_the_script(self) -> None:
        self.assertIn("./deploy/serve_vllm.sh", README)
        self.assertIsNone(__import__("re").search(r"--max-model-len 4096(?!\d)", README))
        self.assertIn(f"--max-model-len {max_model_len()}", README)

    def test_readme_documents_tool_flags(self) -> None:
        self.assertIn("--enable-auto-tool-choice", README)
        self.assertIn("--reasoning-parser qwen3", README)
        self.assertIn("simpleti/simplicio-27b", README)


class ColabTests(unittest.TestCase):
    def test_colab_runs_serve_colab(self) -> None:
        self.assertIn("deploy/serve_colab.py", COLAB)
        self.assertIn("context.env", COLAB)
        self.assertIn("completion_gate.py", COLAB)
        self.assertNotIn("16384", COLAB)


if __name__ == "__main__":
    unittest.main()
