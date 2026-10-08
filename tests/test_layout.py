"""Repository layout: notebooks live in notebooks/ and GitHub links to main resolve."""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_LINK = re.compile(
    r"https://github\.com/simpletibr/simplicio-27b/(?:blob|tree)/main/([^)\s#?\"'<>`\]]+)"
)


def tracked(*pathspec: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *pathspec],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [p for p in out.split("\0") if p]


class LayoutTests(unittest.TestCase):
    def test_no_notebook_at_root(self) -> None:
        self.assertEqual(tracked(":(glob)*.ipynb"), [])

    def test_main_links_resolve(self) -> None:
        missing = [
            f"{md}: {target}"
            for md in tracked("*.md")
            for target in MAIN_LINK.findall((ROOT / md).read_text(encoding="utf-8"))
            if not (ROOT / target).exists()
        ]
        self.assertEqual(missing, [])
