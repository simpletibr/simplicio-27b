"""Modal app: the F2b gate on a rented GPU, one command.  THIS IS THE ONLY MODULE OF decide/think THAT IMPORTS `modal`.

What is NOT verified: that this app builds its image, finds the GPU and runs on a Modal account.  It was written against
the client source of modal 1.6.2 (installed in a scratch environment: the decorators accept these arguments and the
function registers) and was never run on Modal.  Treat the first launch, with `--limit 3`, as a test.  Everything the
function does is `hosted.run_gate`, which is tested without Modal.

Run it in MODULE mode from the repository root (script mode cannot find the `decide` package in the container):

    pip install modal==1.6.2 && modal setup
    modal run -m decide.think.modal_app --pack pack.jsonl.gz --limit 3        # a probe: 3 rows, tokens/s, what it costs
    modal run -m decide.think.modal_app --pack pack.jsonl.gz                   # every pending row; run it again to resume
    modal volume get simplicio-think-gt-state gt/results.jsonl .               # the results (private: they hold the questions)

The pack is checked against the committed manifest on your machine before anything is rented.  The FP8 checkpoint
(28.7 GiB, pinned revision) is downloaded once into a Volume and reused.  The results are committed to a second Volume
every minute, so a container that Modal replaces resumes where it stopped.  Pick the GPU with the environment
(`SIMPLICIO_THINK_GPU=A100-80GB modal run ...`); the default is H100.  What it costs is on the Modal pricing page for
that GPU, times the hours the probe says (README, "Tempo e custo"); it is not estimated here (UNVERIFIED).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import modal

if not __package__:                                       # `modal run decide/think/modal_app.py`: the container could not import `decide`
    raise RuntimeError("run this app in module mode from the repository root:  modal run -m decide.think.modal_app --help")

ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "simplicio-think-gt"
STATE_VOLUME = "simplicio-think-gt-state"
CACHE_VOLUME = "simplicio-think-hf-cache"
GPU = os.environ.get("SIMPLICIO_THINK_GPU", "H100")
VLLM_PIN = "vllm==0.31.0"                                 # serve.VLLM_VERSION; a test compares them
TIMEOUT_S = 12 * 3600
MARGIN_S = 3600                                           # the thinker stops starting rows this long before the container is killed
HF_CACHE, STATE = "/root/.cache/huggingface", "/state"
PACK_PATH = "/tmp/pack.jsonl.gz"                         # not in a Volume: the pack holds benchmark text
# The pack and the results hold benchmark text and the model's reasoning about it (decide/think/.gitignore keeps them out of git): the image
# must not carry them either if someone keeps them in this folder.  gt-sample-manifest.json, ids and hashes only, is what the container needs.
IGNORE = ["**/__pycache__", "**/*.pyc", "**/.DS_Store", "**/README.md", "**/*.jsonl", "**/*.jsonl.gz", "**/pack*", "**/work", "**/report"]

image = (modal.Image.debian_slim(python_version="3.12").pip_install(VLLM_PIN)
         .env({"PYTHONPATH": "/root", "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
         .add_local_dir(ROOT / "decide" / "think", "/root/decide/think", ignore=IGNORE))
state_volume = modal.Volume.from_name(STATE_VOLUME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
app = modal.App(APP_NAME, include_source=False)           # the code reaches the container through `image`


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={HF_CACHE: cache_volume, STATE: state_volume},
              retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=10.0), max_containers=1)
def gate(pack: bytes, run: str = "gt", limit: Optional[int] = None, max_tokens: int = 16384, concurrency: int = 16) -> dict:
    """Serve the thinker, run the pending rows, write the report.  Returns the small things; the results stay in the Volume."""
    from decide.think import hosted

    work = Path(STATE) / run
    work.mkdir(parents=True, exist_ok=True)
    pack_path = Path(PACK_PATH)
    pack_path.write_bytes(pack)
    stop, one_commit_at_a_time = threading.Event(), threading.Lock()

    def commit_now() -> None:
        with one_commit_at_a_time:
            for volume in (state_volume, cache_volume):
                try:
                    volume.commit()
                except Exception as exc:                  # noqa: BLE001 - a failed commit must not kill the run; the next one retries
                    print(f"[modal] volume commit failed: {type(exc).__name__}: {exc}", flush=True)

    def ticker() -> None:
        while not stop.wait(60):
            commit_now()
    thread = threading.Thread(target=ticker, daemon=True)
    thread.start()
    try:
        code = hosted.run_gate(pack=str(pack_path), work=str(work), limit=limit, max_tokens=max_tokens, concurrency=concurrency,
                               time_budget_s=float(TIMEOUT_S - MARGIN_S))
    finally:
        stop.set()
        thread.join(timeout=5)
        commit_now()
    report = work / "report"
    read = lambda p: p.read_text(encoding="utf-8") if p.is_file() else ""  # noqa: E731
    md = read(report / "report.md")
    return {"exit": code, "report_md": md, "report_json": read(report / "report.json"),
            "verdict": next((line for line in reversed(md.splitlines()) if line.startswith("GT VERDICT:")), ""),
            "results_in_volume": f"{run}/results.jsonl"}


@app.local_entrypoint()
def main(pack: str, out: str = "think-gt-out", run: str = "gt", limit: int = 0, max_tokens: int = 16384, concurrency: int = 16) -> None:
    from decide.think import hosted, sample

    try:
        problems = sample.verify(str(hosted.DEFAULT_MANIFEST), pack)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read the pack, nothing was rented: {exc}") from None
    if problems:
        raise SystemExit("the pack is not the sample of the manifest; nothing was rented:\n" + "\n".join(problems))
    result = gate.remote(Path(pack).read_bytes(), run, limit or None, max_tokens, concurrency)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    if result["report_md"]:
        (folder / "report.md").write_text(result["report_md"], encoding="utf-8")
        (folder / "report.json").write_text(result["report_json"], encoding="utf-8")
    print(result["verdict"] or f"no report (exit {result['exit']}): see the Modal logs")
    print(f"results (private): modal volume get {STATE_VOLUME} {result['results_in_volume']} {folder}/")
    raise SystemExit(result["exit"])
