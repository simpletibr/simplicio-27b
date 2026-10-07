"""Guardrails: the README states only what was measured.

An issue that later replaces one of these numbers or sentences must update this file in the same PR.
"""
import re
import unittest
from pathlib import Path

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")


class ReadmeClaims(unittest.TestCase):
    def test_withdrawn_phrases_absent(self):
        for phrase in ("of every layer", "120/120", "Unit test passes",
                       "follows the format more reliably", "passes 46.7% of"):
            self.assertNotIn(phrase, README)

    def test_required_statements_present(self):
        for phrase in ("40 unique Python tasks", "never executes the patched code",
                       "16 full-attention layers", "max_new_tokens=480", "temperature 0",
                       "Q4_K_M GGUF", "image input has not been evaluated", "38.0%", "32.2%"):
            self.assertIn(phrase, README)

    def test_no_32768_context(self):
        self.assertIsNone(re.search(r"32[,.]?768", README))


if __name__ == "__main__":
    unittest.main()
