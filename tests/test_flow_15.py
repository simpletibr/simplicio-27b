"""Flow test for the local gate (issue #15): git push -> .githooks/pre-push -> make check.

Everything happens in a throwaway git repo under a temp dir, pushing to a local bare repo.
No network, no credentials, no real vllm. The repo gets copies of the files the gate uses, so the
real Makefile, hook and check_vllm_args.py run end to end. Needs make, ruff, shellcheck and git on the
PATH, the same requirement as `make check` itself.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_ENV = "SIMPLICIO_SKIP_VLLM_ARGS"
COPIED = (
    "Makefile",
    ".githooks/pre-push",
    "scripts/check_vllm_args.py",
    "deploy/serve_vllm.sh",
    "deploy/context.env",
)
TRIVIAL_TEST = (
    "import unittest\n\n\n"
    "class Trivial(unittest.TestCase):\n"
    "    def test_ok(self):\n"
    "        self.assertTrue(True)\n\n\n"
    'if __name__ == "__main__":\n'
    "    unittest.main()\n"
)


class PrePushFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [t for t in ("git", "make", "ruff", "shellcheck") if shutil.which(t) is None]
        if missing:
            raise AssertionError(f"not on PATH (needed by make check): {', '.join(missing)}")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.repo, self.remote, self.bin = base / "repo", base / "remote.git", base / "bin"
        self.repo.mkdir()
        self.bin.mkdir()
        self.env = {
            k: v for k, v in os.environ.items()
            if k not in ("MAKEFLAGS", "MAKELEVEL", "MFLAGS", SKIP_ENV)
        }
        self.env.update(
            GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
            GIT_AUTHOR_NAME="gate", GIT_AUTHOR_EMAIL="gate@example.invalid",
            GIT_COMMITTER_NAME="gate", GIT_COMMITTER_EMAIL="gate@example.invalid",
        )
        self.git("init", "-q", "-b", "main", cwd=self.repo)
        self.git("init", "-q", "--bare", str(self.remote), cwd=base)
        for rel in COPIED:
            dest = self.repo / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)  # copy2 keeps the executable bit of the hook
        (self.repo / "tests").mkdir()
        (self.repo / "tests" / "test_trivial.py").write_text(TRIVIAL_TEST, encoding="utf-8")
        self.git("config", "core.hooksPath", ".githooks", cwd=self.repo)
        self.git("config", "commit.gpgsign", "false", cwd=self.repo)
        self.git("add", "-A", cwd=self.repo)
        self.git("commit", "-q", "-m", "Gate fixture.", cwd=self.repo)

    def git(self, *args, cwd=None, env=None):
        return subprocess.run(
            ["git", *args], cwd=cwd or self.repo, env=env or self.env,
            capture_output=True, text=True, timeout=300,
        )

    def push(self, skip=True, no_vllm_python=False):
        env = dict(self.env)
        if skip:
            env[SKIP_ENV] = "1"
        if no_vllm_python:
            # `python3 -S` has no site-packages, so a vllm installed on this machine stays invisible.
            wrapper = self.bin / "python3"
            wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n', encoding="utf-8")
            wrapper.chmod(0o755)
            env["PATH"] = str(self.bin) + os.pathsep + env["PATH"]
        return self.git("push", str(self.remote), "HEAD:refs/heads/main", env=env)

    def remote_head(self):
        r = self.git("--git-dir", str(self.remote), "rev-parse", "--verify", "-q", "refs/heads/main")
        return r.stdout.strip() if r.returncode == 0 else None

    def commit_file(self, name, content):
        (self.repo / name).write_text(content, encoding="utf-8")
        self.git("add", name)
        self.git("commit", "-q", "-m", f"Add {name}.")
        return self.git("rev-parse", "HEAD").stdout.strip()

    def test_clean_push_passes_the_hook(self):
        self.commit_file("clean.py", "def ok():\n    return 1\n")
        r = self.push()
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("SKIP: vllm not importable; argv extracted, not validated", out)
        self.assertEqual(self.remote_head(), self.git("rev-parse", "HEAD").stdout.strip())

    def test_lint_error_blocks_the_push(self):
        bad = self.commit_file("bad.py", "x = (\n")
        r = self.push()
        out = r.stdout + r.stderr
        self.assertNotEqual(r.returncode, 0, out)
        self.assertIn("lint] Error 1", out)
        self.assertIn("failed to push", out)
        self.assertNotEqual(self.remote_head(), bad)
        self.assertIsNone(self.remote_head(), "blocked push must not create the remote branch")

    def test_unused_import_blocks_the_push(self):
        self.commit_file("unused.py", "import os\n")
        r = self.push()
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIsNone(self.remote_head())

    def test_without_vllm_and_without_env_var_the_push_is_blocked(self):
        self.commit_file("clean.py", "def ok():\n    return 1\n")
        r = self.push(skip=False, no_vllm_python=True)
        out = r.stdout + r.stderr
        self.assertNotEqual(r.returncode, 0, out)
        self.assertIn("vllm not importable: instale vllm==0.31.0", out)
        self.assertIn(f"{SKIP_ENV}=1", out)
        self.assertIsNone(self.remote_head())

    def test_without_vllm_the_env_var_lets_a_clean_push_through(self):
        self.commit_file("clean.py", "def ok():\n    return 1\n")
        r = self.push(skip=True, no_vllm_python=True)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("SKIP: vllm not importable", out)
        self.assertIsNotNone(self.remote_head())


if __name__ == "__main__":
    unittest.main()
