"""Validate the argv of deploy/serve_vllm.sh with vLLM 0.31.0's own CLI parser.

Without an importable vllm it fails with exit code 1. Set SIMPLICIO_SKIP_VLLM_ARGS=1 to skip
only this check (it then prints SKIP and exits 0).
Usage: python3 scripts/check_vllm_args.py [--script PATH]
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED = "0.31.0"
SKIP_ENV = "SIMPLICIO_SKIP_VLLM_ARGS"
FAKE_VLLM = """#!{python}
import json, os, sys
if sys.argv[1:2] == ["serve"]:
    open(os.environ["CHECK_VLLM_ARGV_OUT"], "w").write(json.dumps(sys.argv[1:]))
"""


def extract_argv(script: Path) -> list[str]:
    """Run the script with a fake `vllm` (PATH and VLLM_BIN); return the exact `vllm serve` argv."""
    with tempfile.TemporaryDirectory() as tmp:
        fake, out = Path(tmp) / "vllm", Path(tmp) / "argv.json"
        fake.write_text(FAKE_VLLM.format(python=sys.executable), encoding="utf-8")
        fake.chmod(0o755)
        env = dict(os.environ, PATH=tmp + os.pathsep + os.environ.get("PATH", ""),
                   VLLM_BIN=str(fake), CHECK_VLLM_ARGV_OUT=str(out))
        subprocess.run(["bash", str(script)], env=env, check=True, timeout=30,
                       stdout=subprocess.DEVNULL)
        if not out.exists():
            sys.exit(f"FAIL: {script} did not call `vllm serve`")
        return json.loads(out.read_text(encoding="utf-8"))


def validate(argv: list[str]) -> int:
    import vllm
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    if vllm.__version__.split("+")[0] != PINNED:
        print(f"FAIL: vllm {vllm.__version__} installed, pin is {PINNED}")
        return 1
    parser = make_arg_parser(FlexibleArgumentParser(prog="vllm serve"))
    try:
        args = parser.parse_args(argv[1:])  # argv[0] == "serve"; unknown flag -> SystemExit
    except SystemExit:
        print("FAIL: vLLM's parser rejected the argv (error above)")
        return 1
    try:
        validate_parsed_serve_args(args)
    except Exception as exc:  # vLLM raises TypeError/ValueError here
        print(f"FAIL: validate_parsed_serve_args: {exc}")
        return 1
    if args.tool_call_parser and args.tool_call_parser not in ToolParserManager.list_registered():
        print(f"FAIL: --tool-call-parser {args.tool_call_parser} is not registered")
        return 1
    if args.reasoning_parser and args.reasoning_parser not in ReasoningParserManager.list_registered():
        print(f"FAIL: --reasoning-parser {args.reasoning_parser} is not registered")
        return 1
    print(f"OK: vllm {PINNED} accepts the {len(argv) - 1} serve arguments")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--script", type=Path, default=ROOT / "deploy" / "serve_vllm.sh")
    argv = extract_argv(ap.parse_args().script)
    if importlib.util.find_spec("vllm") is None:
        if os.environ.get(SKIP_ENV) == "1":
            print("SKIP: vllm not importable; argv extracted, not validated")
            return 0
        print(f"vllm not importable: instale vllm=={PINNED} ou exporte {SKIP_ENV}=1 "
              "para pular esta checagem", file=sys.stderr)
        return 1
    return validate(argv)


if __name__ == "__main__":
    sys.exit(main())
