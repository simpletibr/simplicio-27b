"""Contract test for issue #30 (repository hygiene).

Pins what other parts of the project and the README rely on, with static checks over the tracked
files (no network, no GPU):
  * no tracked text file cites a Colab notebook by its old root name, and every notebook the
    README names exists where it names it;
  * every path in the first column of the README's Repository table exists;
  * the README Development section says where notebooks live and which test enforces it;
  * the .gitignore keeps the three rules that predate #30 and holds the rules #30 adds;
  * the repository root holds only the files in ALLOWED_ROOT_FILES: no .ipynb and no
    pyproject.toml (#30 forbids both). A new root file has to be added to that list on purpose.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Tracked files that may sit at the repository root. generate_dataset.py stays until #28 replaces it.
ALLOWED_ROOT_FILES = frozenset({
    ".gitignore", "LICENSE", "Makefile", "Modelfile", "README.md", "generate_dataset.py",
})
FORBIDDEN_ROOT_FILES = ("pyproject.toml",)

# A notebook that now lives in notebooks/, named with no `/` in front of it (neither `notebooks/`
# nor a commit permalink `/<sha>/`): the old root layout. Names of notebooks that were deleted
# (the 2026 Benchmarks one, cited by README through a b8567df permalink) are not in notebooks/,
# so they are not flagged.
NOTEBOOK_CITATION = re.compile(r"notebooks/(Simplicio_27B_[A-Za-z0-9_]+_Colab\.ipynb)")

GITIGNORE_BEFORE_30 = ("__pycache__/", "*.py[cod]", ".simplicio/")
GITIGNORE_ADDED_BY_30 = (
    "*.gguf", "*.safetensors", "vllm.log", "cloudflared.log", "/cloudflared",
    "benchmarks/harness/results/", ".pytest_cache/", ".ruff_cache/", ".ipynb_checkpoints/",
    ".env", ".DS_Store", ".venv/", "venv/", "scratchpad/", ".claude/worktrees/",
)
README_LAYOUT_SENTENCE = (
    "Colab notebooks live in `notebooks/`; `tests/test_layout.py` fails on a notebook at the "
    "repository root or on a `blob/main` link to a missing file."
)


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout
    return [p for p in out.split("\0") if p]


def text_files() -> list[str]:
    """Tracked files that are readable text, excluding notebooks (their cells are content, not links)."""
    found = []
    for rel in tracked():
        path = ROOT / rel
        if rel.endswith(".ipynb") or not path.is_file():
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found.append(rel)
    return found


def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


class NotebookPathContractTest(unittest.TestCase):
    def test_no_file_cites_a_notebook_by_its_old_root_name(self) -> None:
        names = sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb"))
        self.assertTrue(names)
        old = re.compile(r"(^|[^/])(" + "|".join(map(re.escape, names)) + ")")
        hits = []
        for rel in text_files():
            for n, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
                if old.search(line):
                    hits.append(f"{rel}:{n}")
        self.assertEqual(hits, [])

    def test_every_notebook_the_files_cite_exists_under_notebooks(self) -> None:
        cited = {m for rel in text_files()
                 for m in NOTEBOOK_CITATION.findall((ROOT / rel).read_text(encoding="utf-8"))}
        self.assertTrue(cited, "the README is expected to cite at least one notebook")
        self.assertEqual(sorted(n for n in cited if not (ROOT / "notebooks" / n).is_file()), [])

    def test_readme_cites_every_notebook_in_the_repository(self) -> None:
        on_disk = sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb"))
        self.assertTrue(on_disk)
        self.assertEqual([n for n in on_disk if f"notebooks/{n}" not in readme()], [])

    def test_repository_table_paths_exist(self) -> None:
        table = readme().split("## Repository", 1)[1].split("\n## ", 1)[0]
        paths = [p for row in table.splitlines() if row.startswith("| `")
                 for p in re.findall(r"`([^`]+)`", row.split("|")[1])]
        self.assertGreaterEqual(len(paths), 5)
        self.assertEqual([p for p in paths if not (ROOT / p).exists()], [])

    def test_readme_development_section_names_the_layout_rule(self) -> None:
        dev = readme().split("## Development", 1)[1].split("\n## ", 1)[0]
        self.assertIn(README_LAYOUT_SENTENCE, dev)
        self.assertTrue((ROOT / "tests" / "test_layout.py").is_file())


class GitignoreContractTest(unittest.TestCase):
    def lines(self) -> list[str]:
        return [ln.strip() for ln in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]

    def test_original_rules_come_first(self) -> None:
        self.assertEqual(tuple(self.lines()[:3]), GITIGNORE_BEFORE_30)

    def test_rules_added_by_issue_30_are_present(self) -> None:
        self.assertEqual([r for r in GITIGNORE_ADDED_BY_30 if r not in self.lines()], [])

    def test_no_rule_hides_a_negation_or_the_whole_tree(self) -> None:
        # a broad rule would silently drop new files: no `*`, no `/`, no `!` re-includes
        bad = [ln for ln in self.lines() if ln in ("*", "/", "**", "*/") or ln.startswith("!")]
        self.assertEqual(bad, [])


class RootLayoutContractTest(unittest.TestCase):
    def root_files(self) -> list[str]:
        return [p for p in tracked() if "/" not in p]

    def test_root_holds_only_the_allowed_files(self) -> None:
        self.assertEqual(sorted(set(self.root_files()) - ALLOWED_ROOT_FILES), [])

    def test_allowed_files_are_all_still_there(self) -> None:
        self.assertEqual(sorted(ALLOWED_ROOT_FILES - set(self.root_files())), [])

    def test_no_notebook_and_no_pyproject_at_root(self) -> None:
        root = self.root_files()
        self.assertEqual([p for p in root if p.endswith(".ipynb")], [])
        self.assertEqual([p for p in root if p in FORBIDDEN_ROOT_FILES], [])
        self.assertTrue(any(p.startswith("notebooks/") for p in tracked()))


if __name__ == "__main__":
    unittest.main()
