"""The `vllm serve` command for the thinker, for a GPU you rent or own (A100 80 GB, H100, RTX PRO 6000) and for Modal.

    python -m decide.think.serve --print           # the command, to copy
    python -m decide.think.serve --run             # run it (replaces this process)
    python -m decide.think.serve --wait-url http://127.0.0.1:8000/v1 --timeout 1800   # block until it answers

NOT for the free Kaggle T4 (compute capability 7.5).  Reading the source of the pinned vLLM (0.31.0)
without running it: it registers no bitsandbytes quantization, it takes no bf16 below capability 8.0,
and the FP8 checkpoint is 28.7 GiB against 2 x 15 GiB of T4.  The Kaggle notebook therefore uses
transformers with NF4.  That reading is UNVERIFIED on a real T4 (README, "Backends").

Defaults: the FP8 checkpoint (pinned revision), thinking on, vLLM's `qwen3` reasoning parser (so the
reasoning arrives apart in `reasoning_content`/`reasoning` and `content` is the final answer), text only
(`--language-model-only`), loopback only.  The server is never exposed by this module: on Modal the runner
sits in the same container, elsewhere use the platform's tunnel.  A key goes in the environment
(`VLLM_API_KEY`), never on the command line, where `ps` would show it.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, List, Optional, Sequence

from . import common as C

VLLM_VERSION = "0.31.0"                      # must equal the pin of the repository (scripts/check_vllm_args.py)
DEFAULT_PORT = 8000
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def vllm_command(model: str = C.QWEN_FP8_MODEL, revision: Optional[str] = C.QWEN_FP8_REVISION, *, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, served_name: Optional[str] = None, max_model_len: int = 32768,
                 gpu_memory_utilization: float = 0.90, max_num_seqs: int = 32, tensor_parallel_size: int = 1,
                 dtype: str = "auto", thinking: bool = True, language_model_only: bool = True,
                 enforce_eager: bool = False, download_dir: Optional[str] = None, extra: Sequence[str] = ()) -> List[str]:
    """argv of `vllm serve` (the first item is the binary)."""
    if host not in LOOPBACK:
        raise ValueError(f"host {host!r} is not a loopback address; do not expose the server, use the platform's tunnel")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if min(max_model_len, max_num_seqs, tensor_parallel_size, port) < 1:
        raise ValueError("max_model_len, max_num_seqs, tensor_parallel_size and port must be positive")
    argv = ["vllm", "serve", model]
    if revision:
        argv += ["--revision", revision]
    argv += ["--host", host, "--port", str(port), "--served-model-name", served_name or model,
             "--max-model-len", str(max_model_len), "--gpu-memory-utilization", f"{gpu_memory_utilization:g}",
             "--max-num-seqs", str(max_num_seqs), "--dtype", dtype,
             "--reasoning-parser", "qwen3",
             "--default-chat-template-kwargs", json.dumps({"enable_thinking": bool(thinking)})]
    if tensor_parallel_size > 1:
        argv += ["--tensor-parallel-size", str(tensor_parallel_size)]
    if language_model_only:
        argv.append("--language-model-only")
    if enforce_eager:
        argv.append("--enforce-eager")
    if download_dir:
        argv += ["--download-dir", download_dir]
    return argv + list(extra)


def base_url(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}/v1"


def wait_ready(url: str, timeout_s: float, *, api_key: Optional[str] = None, interval_s: float = 5.0,
               opener: Callable = urllib.request.urlopen, sleep: Callable[[float], None] = time.sleep,
               clock: Callable[[], float] = time.monotonic) -> bool:
    """Poll GET {url}/models until the server answers (any HTTP answer except a 5xx counts: 401 means up, key needed)."""
    deadline = clock() + timeout_s
    req = urllib.request.Request(url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    while True:
        try:
            with opener(req, timeout=10):
                return True
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        if clock() >= deadline:
            return False
        sleep(interval_s)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m decide.think.serve", description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print", action="store_true", help="print the command")
    mode.add_argument("--run", action="store_true", help="run it (replaces this process; needs vllm==%s installed)" % VLLM_VERSION)
    mode.add_argument("--wait-url", metavar="URL", help="block until URL (.../v1) answers; exit 1 on timeout")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds for --wait-url (default %(default)s)")
    ap.add_argument("--model", default=C.QWEN_FP8_MODEL)
    ap.add_argument("--revision", default=None, help="default: the pinned revision when --model is the pinned FP8 or bf16 repo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--served-model-name", help="default: the model id, which is what thinker.py asks for with --model")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--no-thinking", action="store_true", help="serve with thinking off by default (not the GT protocol)")
    ap.add_argument("--with-vision", action="store_true", help="do not pass --language-model-only")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--download-dir")
    ap.add_argument("extra", nargs="*", help="more vllm flags, after --")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.wait_url:
        ok = wait_ready(args.wait_url, args.timeout, api_key=os.environ.get("VLLM_API_KEY"))
        print(f"[serve] {args.wait_url} {'answers' if ok else 'did not answer in time'}")
        return 0 if ok else 1
    try:
        cmd = vllm_command(args.model, args.revision or C.PINNED_REVISIONS.get(args.model), host=args.host, port=args.port,
                           served_name=args.served_model_name, max_model_len=args.max_model_len,
                           gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
                           tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, thinking=not args.no_thinking,
                           language_model_only=not args.with_vision, enforce_eager=args.enforce_eager,
                           download_dir=args.download_dir, extra=args.extra)
    except ValueError as e:
        print(f"[serve] {e}", file=sys.stderr)
        return 2
    if args.print:
        print(shlex.join(cmd))
        return 0
    print(f"[serve] {shlex.join(cmd)}", flush=True)
    os.execvp(cmd[0], cmd)
    return 0                                   # unreachable: execvp replaces the process


if __name__ == "__main__":
    sys.exit(main())
