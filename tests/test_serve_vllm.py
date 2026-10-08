"""Guardrails for deploy/serve_vllm.sh, the only vLLM serve command."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "deploy" / "serve_vllm.sh"
SERVE_TEXT = SERVE.read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
GUIDE = (ROOT / "deploy" / "DISTRIBUTION_GUIDE.md").read_text(encoding="utf-8")
COLAB = (ROOT / "Simplicio_27B_Serve_Colab.ipynb").read_text(encoding="utf-8")
INSTRUCT = json.loads((ROOT / "deploy" / "hf_instruct.json").read_text(encoding="utf-8"))
CONTEXT = (ROOT / "deploy" / "context.env").read_text(encoding="utf-8")
DELETED = (
    "deploy/simplicio_tool_parser.py",
    "deploy/simplicio_tool_xml.py",
    "deploy/chat_template_chatml.jinja",
    "tests/test_simplicio_tool_xml.py",
)


def max_model_len() -> str:
    for line in CONTEXT.splitlines():
        if line.startswith("MAX_MODEL_LEN="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("MAX_MODEL_LEN missing")


def serve_argv(*args: str) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "vllm"
        fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
        fake.chmod(0o755)
        env = {k: v for k, v in os.environ.items() if k not in {"HOST", "GPU_MEM"}}
        env["VLLM_BIN"] = str(fake)
        out = subprocess.run(["bash", str(SERVE), *args], env=env,
                             capture_output=True, text=True, check=True)
    lines = out.stdout.splitlines()
    return lines[lines.index("serve"):]


class ServeScriptTests(unittest.TestCase):
    def test_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SERVE)], check=True)

    def test_exact_argv(self) -> None:
        self.assertEqual(
            serve_argv(),
            ["serve", "wesleysimplicio/Simplicio-27B", "--host", "0.0.0.0",
             "--port", "8000", "--max-model-len", max_model_len(),
             "--gpu-memory-utilization", "0.92",
             "--served-model-name", "simplicio-27b", "simpleti/simplicio-27b",
             "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
             "--reasoning-parser", "qwen3",
             "--default-chat-template-kwargs", '{"enable_thinking": false}',
             "--sse-keep-alive-interval", "15",
             "--enable-force-include-usage"],
        )

    def test_positional_model_and_port(self) -> None:
        argv = serve_argv("org/other", "9000")
        self.assertEqual(argv[1], "org/other")
        self.assertEqual(argv[argv.index("--port") + 1], "9000")

    def test_forbidden_flags_absent(self) -> None:
        argv = serve_argv()
        for flag in ("--stop", "--chat-template", "--tool-parser-plugin",
                     "--enable-lora", "--lora-modules", "--quantization",
                     "--load-format"):
            self.assertNotIn(flag, argv)
        for word in ("--stop", "bitsandbytes", "simplicio_tool", "chat_template_chatml"):
            self.assertNotIn(word, SERVE_TEXT)
        self.assertIn('exec "${VLLM_BIN}" serve', SERVE_TEXT)

    def test_hf_instruct_matches_script(self) -> None:
        self.assertEqual(
            INSTRUCT,
            {"instruct_type": "qwen3", "chat_template": "model",
             "default_chat_template_kwargs": {"enable_thinking": False},
             "reasoning_parser": "qwen3", "tool_call_parser": "qwen3_coder",
             "served_model_names": ["simplicio-27b", "simpleti/simplicio-27b"],
             "max_model_len": int(max_model_len())},
        )

    def test_custom_parser_files_deleted(self) -> None:
        for rel in DELETED:
            self.assertFalse((ROOT / rel).exists(), rel)

    @unittest.skipUnless(importlib.util.find_spec("vllm"), "vllm não instalado")
    def test_vllm_accepts_argv(self) -> None:
        from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        args = make_arg_parser(FlexibleArgumentParser()).parse_args(serve_argv()[1:])
        validate_parsed_serve_args(args)
        self.assertEqual(args.tool_call_parser, "qwen3_coder")
        self.assertEqual(args.default_chat_template_kwargs, {"enable_thinking": False})
        self.assertTrue(args.enable_force_include_usage)


class DocsTests(unittest.TestCase):
    def test_readme_serve_section(self) -> None:
        for s in ("./deploy/serve_vllm.sh", f"--max-model-len {max_model_len()}",
                  "vllm==0.31.0", "--tool-call-parser qwen3_coder",
                  "--reasoning-parser qwen3", "enable_thinking", "simpleti/simplicio-27b"):
            self.assertIn(s, README)
        for s in ("--tool-call-parser simplicio", "chat_template_chatml",
                  "simplicio_tool_parser", "4-bit base with the LoRA"):
            self.assertNotIn(s, README)

    def test_distribution_guide(self) -> None:
        for s in ("--tool-call-parser qwen3_coder", "vllm==0.31.0"):
            self.assertIn(s, GUIDE)
        self.assertNotIn("--tool-call-parser simplicio", GUIDE)

    def test_colab_pins_vllm_and_runs_script(self) -> None:
        for s in ("vllm==0.31.0", "serve_vllm.sh", "deploy/serve_colab.py",
                  "context.env", "completion_gate.py"):
            self.assertIn(s, COLAB)
        for s in ("bitsandbytes", "simplicio_tool", "chat_template_chatml", "16384", "A100"):
            self.assertNotIn(s, COLAB)


if __name__ == "__main__":
    unittest.main()
