"""Ollama: one Modelfile with the Qwen renderer/parser, 40960 context and the training system prompt."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYWORDS = ("FROM ", "RENDERER ", "PARSER ", "REQUIRES ", "PARAMETER ", "TEMPLATE ", "ADAPTER ", "MESSAGE ")
DIRECTIVES = [
    "FROM ./Qwen3.8-27B.Q4_K_M.gguf", "FROM ./Qwen3.8-27B.BF16-mmproj.gguf",
    "RENDERER qwen3.5", "PARSER qwen3.5", "REQUIRES 0.30.0", "PARAMETER num_ctx 40960",
    "PARAMETER temperature 0.7", "PARAMETER top_p 0.8", "PARAMETER top_k 20", "PARAMETER min_p 0",
    "PARAMETER presence_penalty 1.5", "PARAMETER repeat_penalty 1", 'PARAMETER stop "<|im_end|>"',
]


def training_system_prompt() -> str:
    nb = json.loads((ROOT / "Simplicio_27B_Training_Colab.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if src.startswith("system_prompt = ("):
            return ast.literal_eval(ast.parse(src).body[0].value)
    raise AssertionError("system_prompt cell not found")


class ModelfileTests(unittest.TestCase):
    text = (ROOT / "Modelfile").read_text(encoding="utf-8")

    def test_only_one_modelfile(self) -> None:
        self.assertFalse((ROOT / "deploy" / "Modelfile").exists())
        self.assertFalse((ROOT / "install.sh").exists())

    def test_directives(self) -> None:
        lines = [x for x in self.text.splitlines() if x.startswith(KEYWORDS)]
        self.assertEqual(lines, DIRECTIVES)

    def test_system_is_training_prompt(self) -> None:
        self.assertIn(f'SYSTEM """{training_system_prompt()}"""\n', self.text)

    def test_readme_ollama_section(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for needle in ("--think=false", '"think": false', '"reasoning_effort": "none"', "RENDERER qwen3.5"):
            self.assertIn(needle, readme)
        for needle in ("install.sh", "32,768", "does not declare tools"):
            self.assertNotIn(needle, readme)


if __name__ == "__main__":
    unittest.main()
