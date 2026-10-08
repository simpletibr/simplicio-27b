"""Flow test for issue #30 (repository hygiene): .gitignore rules and the notebooks/ move.

Two end-to-end paths, with no network and no GPU:
  * the repository's own .gitignore is copied into a throwaway git repo and `git check-ignore` is
    asked about the paths it must ignore (weights, serving logs, caches, venvs, agent state) and the
    ones it must not (deploy/serve_vllm.sh, gateway/.env.example, a nested deploy/cloudflared);
    git's global and system excludes are switched off so only the file under test decides;
  * after the notebooks were moved, the gate still passes: ruff, shellcheck, php -l and the vLLM
    argv step run through `make`, and the test modules that open a notebook run through unittest.
    The `test` step of `make check` is this very run, so it is not started again (it would recurse).
Needs git, make, ruff, shellcheck and php on the PATH, the same requirement as `make check` itself.
"""

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
SKIP_ENV = "SIMPLICIO_SKIP_VLLM_ARGS"

IGNORED = (
    "pkg/__pycache__/mod.cpython-313.pyc",
    "mod.pyc",
    ".simplicio/state.json",
    "Qwen3.8-27B.Q4_K_M.gguf",
    "model-00001-of-00018.safetensors",
    "vllm.log",
    "cloudflared.log",
    "cloudflared",
    "benchmarks/harness/results/r.jsonl",
    ".pytest_cache/v/cache/lastfailed",
    ".ruff_cache/0.1/x",
    "notebooks/.ipynb_checkpoints/a-checkpoint.ipynb",
    ".env",
    ".DS_Store",
    ".venv/bin/python",
    "venv/bin/python",
    "scratchpad/notes.txt",
    ".claude/worktrees/w1/README.md",
)
NOT_IGNORED = (
    "deploy/serve_vllm.sh",
    "gateway/.env.example",
    "deploy/cloudflared",  # `/cloudflared` is anchored to the root
    "notebooks/Simplicio_27B_Serve_Colab.ipynb",
    "Modelfile",
    "README.md",
    "tests/test_flow_30.py",
)
# Test modules that open a notebook by path, plus the layout test itself.
MOVED_PATH_TESTS = (
    "tests.test_serve_vllm",
    "tests.test_modelfile",
    "tests.test_flow_22",
    "tests.test_flow_31",
    "tests.test_contract_31",
    "tests.test_layout",
)


def clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in ("MAKEFLAGS", "MAKELEVEL", "MFLAGS", "PYTHONPATH", SKIP_ENV)}
    env.update(extra)
    return env


def git_env() -> dict[str, str]:
    return clean_env(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
                     GIT_CONFIG_NOSYSTEM="1")


class GitignoreFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("git") is None:
            raise AssertionError("git is not on the PATH")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True, env=git_env())
        shutil.copy(ROOT / ".gitignore", self.repo / ".gitignore")

    def check_ignore(self, *paths: str) -> list[str]:
        r = subprocess.run(["git", "check-ignore", "--no-index", "--", *paths], cwd=self.repo,
                           capture_output=True, text=True, env=git_env())
        self.assertIn(r.returncode, (0, 1), r.stderr)
        return r.stdout.splitlines()

    def test_every_expected_path_is_ignored(self) -> None:
        self.assertEqual(sorted(self.check_ignore(*IGNORED)), sorted(IGNORED))

    def test_serving_files_are_not_ignored(self) -> None:
        self.assertEqual(self.check_ignore(*NOT_IGNORED), [])

    def test_issue_acceptance_commands(self) -> None:
        eight = ("Qwen3.8-27B.Q4_K_M.gguf", "model-00001-of-00018.safetensors", "vllm.log",
                 "cloudflared.log", "cloudflared", ".pytest_cache/x",
                 "benchmarks/harness/results/r.jsonl", ".env")
        self.assertEqual(len(self.check_ignore(*eight)), 8)
        r = subprocess.run(["git", "check-ignore", "--no-index", "deploy/serve_vllm.sh",
                            "gateway/.env.example"], cwd=self.repo, capture_output=True,
                           text=True, env=git_env())
        self.assertEqual((r.returncode, r.stdout), (1, ""))

    def test_no_tracked_file_falls_under_the_rules(self) -> None:
        r = subprocess.run(["git", "ls-files", "-ci", "--exclude-standard"], cwd=ROOT,
                           capture_output=True, text=True, env=git_env())
        self.assertEqual((r.returncode, r.stdout.split()), (0, []), r.stderr)


class GateAfterMoveFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [t for t in ("git", "make", "ruff", "shellcheck", "php") if shutil.which(t) is None]
        if missing:
            raise AssertionError(f"not on PATH (needed by make check): {', '.join(missing)}")

    def test_notebooks_are_valid_and_only_under_notebooks(self) -> None:
        out = subprocess.run(["git", "ls-files", "-z", "--", "*.ipynb"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout
        paths = [p for p in out.split("\0") if p]
        self.assertTrue(paths)
        for rel in paths:
            with self.subTest(rel):
                self.assertTrue(rel.startswith("notebooks/") and rel.count("/") == 1, rel)
                self.assertIn("cells", json.loads((ROOT / rel).read_text(encoding="utf-8")))

    def test_make_gate_steps_pass(self) -> None:
        r = subprocess.run(["make", "lint", "shellcheck", "phplint", "vllm-args"], cwd=ROOT,
                           capture_output=True, text=True, timeout=300,
                           env=clean_env(**{SKIP_ENV: "1"}))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("All checks passed!", r.stdout)
        self.assertRegex(r.stdout, r"(?m)^(SKIP|OK): ")

    def test_modules_that_open_a_notebook_pass(self) -> None:
        r = subprocess.run([sys.executable, "-m", "unittest", *MOVED_PATH_TESTS], cwd=ROOT,
                           capture_output=True, text=True, timeout=300, env=clean_env())
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("OK", r.stderr.splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
