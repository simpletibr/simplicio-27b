"""Contract test for the local gate (issue #15).

Pins the interfaces other parts of the project rely on: the five steps of `make check` and their order,
the executable pre-push hook, the argv that deploy/serve_vllm.sh hands to `vllm serve`, the rejection of
a flag vLLM does not have, and the files and variable the README Development section names.
No network and no real vllm: a stub vllm package stands in for the parser.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "check_vllm_args.py"
SERVE = ROOT / "deploy" / "serve_vllm.sh"
SKIP_ENV = "SIMPLICIO_SKIP_VLLM_ARGS"
ORDER = ("unittest", "ruff", "shellcheck", "php -l", "check_vllm_args.py")


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("MAKEFLAGS", "MAKELEVEL", "MFLAGS", "PYTHONPATH", SKIP_ENV)}
    env.update(extra)
    return env


def extract_argv(script):
    """The exact `vllm serve` argv a bash script produces, using the gate's own extractor."""
    r = subprocess.run(
        [sys.executable, "-S", "-c",
         "import json, sys\n"
         f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
         "import check_vllm_args\n"
         "from pathlib import Path\n"
         "print(json.dumps(check_vllm_args.extract_argv(Path(sys.argv[1]))))\n",
         str(script)],
        capture_output=True, text=True, timeout=60, env=clean_env(),
    )
    if r.returncode != 0:
        raise AssertionError(r.stderr)
    return json.loads(r.stdout)


def write_stub_vllm(dest, flags):
    """A vllm package whose parser knows exactly `flags` (every flag takes any number of values)."""
    plain = ("--tool-call-parser", "--reasoning-parser")
    cli_args = (
        "def make_arg_parser(p):\n"
        "    p.add_argument('model_tag', nargs='?')\n"
        + "".join(f"    p.add_argument({f!r})\n" for f in plain)
        + "".join(f"    p.add_argument({f!r}, nargs='*')\n" for f in flags if f not in plain)
        + "    return p\n"
        "def validate_parsed_serve_args(args):\n"
        "    pass\n"
    )
    files = {
        "vllm/__init__.py": "__version__ = '0.31.0'\n",
        "vllm/entrypoints/__init__.py": "",
        "vllm/entrypoints/launchers/__init__.py": "",
        "vllm/entrypoints/launchers/cli_args.py": cli_args,
        "vllm/utils/__init__.py": "",
        "vllm/utils/argparse_utils.py": "from argparse import ArgumentParser as FlexibleArgumentParser\n",
        "vllm/tool_parsers/__init__.py": (
            "class ToolParserManager:\n    list_registered = staticmethod(lambda: ['qwen3_coder'])\n"),
        "vllm/reasoning/__init__.py": (
            "class ReasoningParserManager:\n    list_registered = staticmethod(lambda: ['qwen3'])\n"),
    }
    for rel, content in files.items():
        path = Path(dest) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


class MakeCheckContract(unittest.TestCase):
    def test_make_n_check_lists_the_five_steps_in_order(self):
        r = subprocess.run(["make", "-n", "check"], cwd=ROOT, capture_output=True, text=True,
                           timeout=60, env=clean_env())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("missing separator", r.stdout + r.stderr)
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        self.assertEqual(len(lines), len(ORDER), r.stdout)
        for line, needle in zip(lines, ORDER):
            self.assertIn(needle, line)

    def test_makefile_does_not_export_the_skip_variable(self):
        self.assertNotIn(SKIP_ENV, (ROOT / "Makefile").read_text(encoding="utf-8"))


class PrePushHookContract(unittest.TestCase):
    def test_hook_is_executable_in_the_worktree_and_in_the_index(self):
        hook = ROOT / ".githooks" / "pre-push"
        self.assertTrue(os.access(hook, os.X_OK), "pre-push is not executable")
        r = subprocess.run(["git", "ls-files", "-s", ".githooks/pre-push"], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("100755"), r.stdout)

    def test_hook_runs_make_check_from_the_repo_root(self):
        text = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("#!/bin/sh\n"))
        self.assertIn("git rev-parse --show-toplevel", text)
        self.assertRegex(text, r"(?m)^make check$")


class VllmArgvContract(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def run_check(self, script, flags):
        write_stub_vllm(self.tmp / "stub", flags)
        env = clean_env(PYTHONPATH=str(self.tmp / "stub"))
        return subprocess.run([sys.executable, "-S", str(CHECK), "--script", str(script)],
                              capture_output=True, text=True, timeout=60, env=env)

    def test_serve_script_argv_shape(self):
        argv = extract_argv(SERVE)
        self.assertEqual(argv[0], "serve")
        self.assertEqual(argv[argv.index("--tool-call-parser") + 1], "qwen3_coder")
        self.assertEqual(argv[argv.index("--reasoning-parser") + 1], "qwen3")
        self.assertNotIn("--stop", argv)

    def test_real_argv_is_accepted_by_a_parser_that_knows_its_flags(self):
        flags = [a for a in extract_argv(SERVE) if re.fullmatch(r"--[a-z0-9-]+", a)]
        r = self.run_check(SERVE, flags)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("OK:", r.stdout)

    def test_argv_with_a_nonexistent_flag_is_rejected(self):
        flags = [a for a in extract_argv(SERVE) if re.fullmatch(r"--[a-z0-9-]+", a)]
        broken = self.tmp / "deploy"
        broken.mkdir()
        (broken / "context.env").write_bytes((ROOT / "deploy" / "context.env").read_bytes())
        text = SERVE.read_text(encoding="utf-8")
        tail = "--enable-force-include-usage"
        self.assertEqual(text.count(tail), 1)
        (broken / "serve_vllm.sh").write_text(
            text.replace(tail, tail + " \\\n    --stop '</deliver>'"), encoding="utf-8")
        r = self.run_check(broken / "serve_vllm.sh", flags)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("unrecognized arguments: --stop", r.stderr)
        self.assertIn("FAIL", r.stdout)

    def test_no_vllm_is_an_error_unless_the_variable_is_set(self):
        r = subprocess.run([sys.executable, "-S", str(CHECK)], capture_output=True, text=True,
                           timeout=60, env=clean_env())
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("vllm not importable", r.stderr)
        r = subprocess.run([sys.executable, "-S", str(CHECK)], capture_output=True, text=True,
                           timeout=60, env=clean_env(**{SKIP_ENV: "1"}))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.strip(), "SKIP: vllm not importable; argv extracted, not validated")


class ReadmeContract(unittest.TestCase):
    def setUp(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        match = re.search(r"(?ms)^## Development\n(.*?)(?=^## |\Z)", text)
        self.assertIsNotNone(match, "README has no Development section")
        self.dev = match.group(1)

    def test_development_section_names_tools_files_and_variable(self):
        for tool in ("python3", "ruff", "shellcheck", "php", "make"):
            self.assertIn(f"`{tool}`", self.dev)
        for needle in ("make check", "core.hooksPath .githooks", "deploy/serve_vllm.sh",
                       f"{SKIP_ENV}=1", "vllm==0.31.0"):
            self.assertIn(needle, self.dev)

    def test_readme_does_not_say_the_gate_passes_without_vllm_by_default(self):
        self.assertNotIn("otherwise that step prints `SKIP`", self.dev)
        self.assertIn("fails with exit code 1", self.dev)

    def test_files_named_by_the_readme_exist(self):
        for rel in ("Makefile", ".githooks/pre-push", "scripts/check_vllm_args.py",
                    "deploy/serve_vllm.sh"):
            self.assertTrue((ROOT / rel).is_file(), rel)


if __name__ == "__main__":
    unittest.main()
