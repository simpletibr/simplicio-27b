"""The whole gate on one box with a big GPU: serve the thinker with vLLM, run the sample through it, compare.

    python -m decide.think.hosted --pack pack.jsonl.gz --work work --limit 3     # a probe of 3 rows: tokens/s, cost, does it run
    python -m decide.think.hosted --pack pack.jsonl.gz --work work                # every pending row, then the report

For a rented or own GPU with 40 GB or more (A100 80 GB, H100, RTX PRO 6000 96 GB) that has `vllm==0.31.0`, and for the
Modal app of this folder (`modal_app.py`), which calls `run_gate` in its container.  NOT for the free Kaggle T4 (README,
"Backends").

It does what `serve.py`, `thinker.py` and `compare.py` do one by one, in this order, and stops the server at the end or
on any error: the pack against the manifest (before 29 GiB are downloaded), `vllm serve` (FP8 checkpoint, pinned
revision, loopback only), wait until it answers (and stop waiting if the server process dies), the thinker (it resumes
from `work/results.jsonl`), the report in `work/report/`.  Run it again with the same arguments after a crash and it
continues where it stopped.

Exit codes: 0 done; 2 the thinker or the pack refused (bad manifest, another protocol in the results file); 3 the thinker
stopped after consecutive errors (the report is still written); 4 the server did not come up.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from . import common as C
from . import compare, sample, serve, thinker

DEFAULT_MANIFEST = Path(__file__).with_name("gt-sample-manifest.json")
EXIT_NO_SERVER = 4
CONTEXT_MARGIN = 8192                 # tokens for the prompt (the longest in the sample is about 5 thousand) on top of the budget


class Server:
    """A running `vllm serve` (a stand-in with no process for an endpoint that is already up)."""

    def __init__(self, proc: Any, url: str, log_path: Optional[Path] = None, log_file: Any = None):
        self.proc, self.url, self.log_path, self.log_file = proc, url, log_path, log_file

    def alive(self) -> bool:
        return self.proc is None or self.proc.poll() is None

    def tail(self, lines: int = 20) -> str:
        if not self.log_path or not Path(self.log_path).is_file():
            return ""
        text = Path(self.log_path).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(text[-lines:])

    def stop(self, timeout_s: float = 120.0) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None


def context_length(max_tokens: int) -> int:
    """Context the server needs: the prompt and the whole thinking budget, never less than the 32k default of serve.py."""
    return max(32768, max_tokens + CONTEXT_MARGIN)


def start_vllm(work: Path, model: str, revision: Optional[str], *, port: int = serve.DEFAULT_PORT, max_tokens: int = 16384,
               concurrency: int = 16, gpu_memory_utilization: float = 0.90, tensor_parallel_size: int = 1) -> Server:
    """`vllm serve` in the background, its output in `work/vllm.log` (appended: a restart keeps the earlier lines)."""
    cmd = serve.vllm_command(model, revision, port=port, max_model_len=context_length(max_tokens), max_num_seqs=max(32, concurrency),
                             gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size)
    log_path = work / "vllm.log"
    log_file = open(log_path, "ab")
    try:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        log_file.close()
        raise FileNotFoundError(f"`{cmd[0]}` is not on the PATH: run `pip install vllm==0.31.0` on a GPU box (README, Backends)") from None
    except OSError:
        log_file.close()
        raise
    return Server(proc, serve.base_url(port=port), log_path, log_file)


def wait_until_up(server: Server, timeout_s: float, *, slice_s: float = 15.0, ready: Callable[..., bool] = serve.wait_ready,
                  clock: Callable[[], float] = time.monotonic) -> bool:
    """True when the server answers; False when its process died or `timeout_s` passed."""
    deadline = clock() + timeout_s
    while True:
        if not server.alive():
            return False
        left = deadline - clock()
        if left <= 0:
            return False
        if ready(server.url, min(slice_s, left)):
            return True


def run_gate(*, pack: str, work: str, manifest: Optional[str] = str(DEFAULT_MANIFEST), model: str = C.QWEN_FP8_MODEL,
             revision: Optional[str] = None, limit: Optional[int] = None, max_tokens: int = thinker.DEFAULT_MAX_TOKENS,
             concurrency: int = 16, time_budget_s: Optional[float] = None, ready_timeout_s: float = 1800.0,
             port: int = serve.DEFAULT_PORT, gpu_memory_utilization: float = 0.90, tensor_parallel_size: int = 1,
             thinker_extra: Sequence[str] = (), compare_extra: Sequence[str] = (),
             start_server: Optional[Callable[[Path], Server]] = None, wait: Optional[Callable[..., bool]] = None,
             log: Callable[[str], None] = print) -> int:
    """Serve, think, compare.  `start_server` and `wait` are the seams for a test or for another way to host the model."""
    folder = Path(work)
    folder.mkdir(parents=True, exist_ok=True)
    wait = wait or wait_until_up
    if revision is None:
        revision = C.PINNED_REVISIONS.get(model)
    elif revision == "none":
        revision = None
    if manifest:
        problems = sample.verify(manifest, pack)
        if problems:
            for p in problems:
                log(f"[hosted] manifest: {p}")
            log("[hosted] the pack is not the sample of the manifest: nothing was started")
            return 2
    start = start_server or (lambda w: start_vllm(w, model, revision, port=port, max_tokens=max_tokens, concurrency=concurrency,
                                                  gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size))
    log(f"[hosted] starting the server for {model}" + (f"@{revision[:7]}" if revision else "") + f" (waits up to {ready_timeout_s:.0f} s; the first start downloads the weights)")
    server = start(folder)
    try:
        if not wait(server, ready_timeout_s):
            log("[hosted] the server did not come up" + (" (its process ended)" if not server.alive() else " in time"))
            tail = server.tail()
            if tail:
                log("[hosted] last lines of vllm.log:\n" + tail)
            return EXIT_NO_SERVER
        log(f"[hosted] server up at {server.url}")
        results = folder / "results.jsonl"
        argv: List[str] = ["--backend", "openai", "--base-url", server.url, "--model", model, "--pack", pack, "--out", str(results),
                           "--concurrency", str(concurrency), "--max-tokens", str(max_tokens)]
        if revision:
            argv += ["--revision", revision]
        if manifest:
            argv += ["--manifest", manifest]
        if limit:
            argv += ["--limit", str(limit)]
        if time_budget_s:
            argv += ["--time-budget-s", str(time_budget_s)]
        code = thinker.main(argv + list(thinker_extra))
        if code in (0, 3) and results.is_file():
            report_code = compare.main(["--pack", pack, "--thinker", str(results), "--out", str(folder / "report"), *compare_extra])
            return code or report_code
        return code
    finally:
        server.stop()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m decide.think.hosted", description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True, help="pack.jsonl.gz written by sample.py (private)")
    ap.add_argument("--work", required=True, help="folder for results.jsonl, report/ and vllm.log; run again with the same one to resume")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="the sample manifest to check the pack against (default: the committed one)")
    ap.add_argument("--model", default=C.QWEN_FP8_MODEL, help="default: the FP8 checkpoint, which fits one 80 GB GPU")
    ap.add_argument("--revision", help="default: the pinned revision of --model; 'none' reads the branch main")
    ap.add_argument("--limit", type=int, help="run at most this many pending rows (use 3 first)")
    ap.add_argument("--max-tokens", type=int, default=thinker.DEFAULT_MAX_TOKENS, help="thinking budget per question (default %(default)s)")
    ap.add_argument("--concurrency", type=int, default=16, help="requests in flight (default %(default)s)")
    ap.add_argument("--time-budget-s", type=float, help="stop starting new rows after this many seconds")
    ap.add_argument("--ready-timeout", type=float, default=1800.0, help="seconds to wait for the server (default %(default)s)")
    ap.add_argument("--port", type=int, default=serve.DEFAULT_PORT)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_gate(pack=args.pack, work=args.work, manifest=args.manifest, model=args.model, revision=args.revision, limit=args.limit,
                        max_tokens=args.max_tokens, concurrency=args.concurrency, time_budget_s=args.time_budget_s,
                        ready_timeout_s=args.ready_timeout, port=args.port, gpu_memory_utilization=args.gpu_memory_utilization,
                        tensor_parallel_size=args.tensor_parallel_size)
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"[hosted] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
