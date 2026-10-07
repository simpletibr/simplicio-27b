"""Tests for scripts/check_vllm_args.py with a fake vllm package (no GPU, no real vllm)."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "check_vllm_args.py"
SKIP_ENV = "SIMPLICIO_SKIP_VLLM_ARGS"
STUB = {
    "vllm/__init__.py": "__version__ = '0.31.0'\n",
    "vllm/entrypoints/__init__.py": "",
    "vllm/entrypoints/launchers/__init__.py": "",
    "vllm/entrypoints/launchers/cli_args.py": (
        "def make_arg_parser(p):\n"
        "    p.add_argument('model_tag', nargs='?')\n"
        "    p.add_argument('--tool-call-parser')\n"
        "    p.add_argument('--reasoning-parser')\n"
        "    return p\n"
        "def validate_parsed_serve_args(args):\n"
        "    pass\n"
    ),
    "vllm/utils/__init__.py": "",
    "vllm/utils/argparse_utils.py": "from argparse import ArgumentParser as FlexibleArgumentParser\n",
    "vllm/tool_parsers/__init__.py": (
        "class ToolParserManager:\n    list_registered = staticmethod(lambda: ['qwen3_coder'])\n"
    ),
    "vllm/reasoning/__init__.py": (
        "class ReasoningParserManager:\n    list_registered = staticmethod(lambda: ['qwen3'])\n"
    ),
}


class CheckVllmArgsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        for rel, content in STUB.items():
            path = self.tmp / "stub" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def run_check(self, serve_line=None, stub=False, skip=False):
        if serve_line is None:
            script = ROOT / "deploy" / "serve_vllm.sh"
        else:
            script = self.tmp / "serve.sh"
            script.write_text(f"#!/usr/bin/env bash\n{serve_line}\n", encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(self.tmp / "stub") if stub else "")
        env.pop(SKIP_ENV, None)  # an exported value in the caller's shell must not leak in
        if skip:
            env[SKIP_ENV] = "1"
        return subprocess.run(
            [sys.executable, "-S", str(CHECK), "--script", str(script)],
            env=env, capture_output=True, text=True, timeout=60,
        )

    def test_real_script_fails_without_vllm(self):
        r = self.run_check()
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn(
            "vllm not importable: instale vllm==0.31.0 ou exporte SIMPLICIO_SKIP_VLLM_ARGS=1 "
            "para pular esta checagem", r.stderr)
        self.assertNotIn("SKIP", r.stdout)

    def test_real_script_skips_with_env_var_without_vllm(self):
        r = self.run_check(skip=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("SKIP: vllm not importable; argv extracted, not validated", r.stdout)

    def test_script_without_vllm_serve_fails(self):
        r = self.run_check("echo no-serve")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("did not call", r.stderr)

    def test_known_flags_pass(self):
        r = self.run_check(
            "exec vllm serve m --tool-call-parser qwen3_coder --reasoning-parser qwen3", stub=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("OK:", r.stdout)

    def test_unknown_flag_fails(self):
        r = self.run_check("exec vllm serve m --stop '</deliver>'", stub=True)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("unrecognized arguments: --stop", r.stderr)

    def test_unregistered_tool_parser_fails(self):
        r = self.run_check("exec vllm serve m --tool-call-parser simplicio", stub=True)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("is not registered", r.stdout)


if __name__ == "__main__":
    unittest.main()
