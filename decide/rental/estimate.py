"""Hours and dollars of a rented day, and the memory warnings that matter before paying for it.

    python -m decide.rental.estimate --profile econ_b200
    python -m decide.rental.estimate --profile full_8xh100 --gpu H200 --json
    python -m decide.rental.estimate --profile econ_b200 --tokens-per-example 371 --traces --thinker-tps 1500

The formulas are the plan's (PLAN.md section 6): training costs ``6 * P * tokens`` FLOPs, inference ``2 * P * tokens``, a GPU
delivers ``MFU`` (0.40) of its peak dense BF16 throughput, and a rented machine bills every GPU for every second.  **Every
number here is an estimate, marked UNVERIFIED**: the peak throughput and the prices are tables (``hardware.py``), the
tokens per example of a profile is an assumption, the fixed times (set-up, disk and network) are guesses, and the thinker's
speed is not estimated at all unless you pass a measured ``--thinker-tps``.  The run itself writes the measured seconds of
every step (``state/*.json``) and ``day_report.json``; those are the numbers to quote.

Memory (also UNVERIFIED, plain arithmetic): DDP full fine-tuning keeps the bf16 weights *and* the bf16 gradients on every
GPU, 104.3 GB for this model, which does not fit an 80 GB H100.  The optimizer (``decide.train.optim.CPUOffloadAdamW``) lives
on the host: 16 bytes per trainable parameter, 417 GB of RAM for a full fine-tune.  This module needs neither torch nor pyarrow.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import hardware as HW
from .profile import Profile, ProfileError, load_profile

STATUS = "UNVERIFIED"
HOST_MEMORY_WARNING_GB = 100


def _positive(value: Any, name: str, *, integer: bool = False, upper: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be at most {upper}, got {value!r}")
    return value


def _phase(name: str, basis: str, *, flops: Optional[float] = None, seconds: Optional[float], price: float, gpus: int) -> Dict[str, Any]:
    return {"name": name, "flops": flops, "seconds": seconds, "hours": None if seconds is None else seconds / 3600,
            "usd": None if seconds is None else seconds * gpus * price, "basis": basis, "status": STATUS}


def memory_report(profile: Profile, gpu: HW.Gpu, mode: str, lora_rank: int) -> Dict[str, Any]:
    trainable = HW.trainable_parameters(mode, lora_rank)
    extra = trainable - (HW.PARAMS if mode == "full" else 0)      # adapters and the head are separate tensors; the backbone is not
    needed = 2 * HW.PARAMS + (2 * HW.PARAMS if mode == "full" else 0) + 4 * extra
    return {"mode": mode, "trainable_parameters": trainable, "gpu_gb_needed": needed / 1e9, "gpu_gb_available": gpu.memory_gb,
            "fits": needed / 1e9 <= gpu.memory_gb, "host_gb_optimizer": HW.OPTIMIZER_BYTES_PER_PARAM * trainable / 1e9,
            "basis": "bf16 weights (+ bf16 gradients for full fine-tuning, + bf16 adapters and their gradients for LoRA); activations, "
                     "the CUDA context and fragmentation are not counted"}


def estimate(profile: Profile, *, gpu: Optional[str] = None, gpus: Optional[int] = None, examples: Optional[int] = None,
             tokens_per_example: Optional[int] = None, seeds: Optional[int] = None, mfu: Optional[float] = None,
             usd_per_gpu_hour: Optional[float] = None, traces: bool = False, thinker_tokens_per_second: Optional[float] = None,
             public_round: bool = True) -> Dict[str, Any]:
    """The estimate of a profile; keyword arguments override the profile's assumptions and are listed in ``assumptions.overridden``."""
    overridden: List[str] = []

    def pick(name: str, given: Any, default: Any) -> Any:
        if given is None:
            return default
        overridden.append(name)
        return given

    card = HW.gpu(pick("gpu", gpu, profile.gpu))
    count = int(_positive(pick("gpus", gpus, profile.gpus), "gpus", integer=True))
    n_examples = int(_positive(pick("examples", examples, profile.examples), "examples", integer=True))
    per_example = int(_positive(pick("tokens_per_example", tokens_per_example, profile.tokens_per_example), "tokens_per_example", integer=True))
    n_seeds = int(_positive(pick("seeds", seeds, profile.seeds), "seeds", integer=True))
    utilization = float(_positive(pick("mfu", mfu, HW.MFU), "mfu", upper=1.0))
    if usd_per_gpu_hour is not None:
        price = float(_positive(usd_per_gpu_hour, "usd_per_gpu_hour")) / 3600
        overridden.append("usd_per_gpu_hour")
    else:
        price = card.usd_per_second
    if thinker_tokens_per_second is not None:
        _positive(thinker_tokens_per_second, "thinker_tokens_per_second")
    mode, rank, epochs = profile.train["mode"], int(profile.train.get("lora_rank", 0) or 0), int(profile.train.get("epochs", 1))

    one_gpu = card.bf16_tflops * 1e12 * utilization
    machine = count * one_gpu
    tokens_trained = n_examples * epochs * n_seeds * per_example
    load_seconds = HW.WEIGHT_BYTES / (HW.ASSUMED_IO_GB_PER_S * 1e9)
    phases: List[Dict[str, Any]] = []

    phases.append(_phase("setup", f"guess: {HW.ASSUMED_SETUP_SECONDS} s to install, download the checkpoint and the data, and load the model",
                         seconds=HW.ASSUMED_SETUP_SECONDS, price=price, gpus=count))
    if traces:
        generated = int(profile.thinker["examples"]) * int(profile.thinker["mean_tokens"])
        if thinker_tokens_per_second is None:
            phases.append(_phase("traces", f"not estimated: needs a measured generation throughput (pass --thinker-tps; the run records it); "
                                           f"{generated} thinking tokens at {profile.thinker['mean_tokens']} per example is the assumed volume",
                                 seconds=None, price=price, gpus=count))
        else:
            phases.append(_phase("traces", f"{generated} thinking tokens / ({thinker_tokens_per_second} tokens/s per GPU x {count} GPUs), "
                                           f"the throughput you passed", seconds=generated / thinker_tokens_per_second / count, price=price, gpus=count))
    train_flops = 6 * HW.PARAMS * tokens_trained
    phases.append(_phase("train", f"6 * P * tokens: {tokens_trained} tokens ({n_examples} examples x {epochs} epoch(s) x {n_seeds} seed(s) x "
                                  f"{per_example} tokens) at {count} x {card.name} x {utilization:.2f} of peak",
                         flops=train_flops, seconds=train_flops / machine, price=price, gpus=count))
    soup_seconds = 0.0 if n_seeds < 2 else (n_seeds + 1) * HW.WEIGHT_BYTES / (HW.ASSUMED_IO_GB_PER_S * 1e9)
    phases.append(_phase("soup", "one seed: no average" if n_seeds < 2 else f"reads {n_seeds} checkpoints and writes one at {HW.ASSUMED_IO_GB_PER_S} GB/s (guess)",
                         seconds=soup_seconds, price=price, gpus=count))
    dev_tokens = n_examples * HW.DEV_FRACTION * per_example
    calibrate_flops = 2 * HW.PARAMS * dev_tokens
    phases.append(_phase("calibrate", f"2 * P * tokens over the dev split ({HW.DEV_FRACTION:.0%} of the examples) on one GPU, plus one model load",
                         flops=calibrate_flops, seconds=calibrate_flops / one_gpu + load_seconds, price=price, gpus=count))
    g4_tokens = 2 * HW.PROXY_ROWS * per_example
    g4_flops = 2 * HW.PARAMS * g4_tokens
    phases.append(_phase("g4", f"2 * P * tokens: the candidate and the baseline over {HW.PROXY_ROWS} proxy rows on one GPU, plus two model loads",
                         flops=g4_flops, seconds=g4_flops / one_gpu + 2 * load_seconds, price=price, gpus=count))
    day_names = {"setup", "traces", "train", "soup", "calibrate", "g4"}
    if public_round:
        public_flops = 2 * HW.PARAMS * HW.PUBLIC_ROUND_INPUT_TOKENS
        phases.append(_phase("public_round", f"2 * P * {HW.PUBLIC_ROUND_INPUT_TOKENS} billed input tokens, one kit worker per GPU "
                                             f"({count} x {card.name}), plus one model load",
                             flops=public_flops, seconds=public_flops / machine + load_seconds, price=price, gpus=count))
        phases.append(_phase("publish", f"uploads {HW.WEIGHT_BYTES / 1e9:.1f} GB at {HW.ASSUMED_IO_GB_PER_S} GB/s (guess)",
                             seconds=HW.WEIGHT_BYTES / (HW.ASSUMED_IO_GB_PER_S * 1e9), price=price, gpus=count))

    def total(names: Sequence[str]) -> Dict[str, Any]:
        known = [p for p in phases if p["name"] in names and p["seconds"] is not None]
        missing = [p["name"] for p in phases if p["name"] in names and p["seconds"] is None]
        seconds = sum(p["seconds"] for p in known)
        return {"seconds": seconds, "hours": seconds / 3600, "usd": sum(p["usd"] for p in known), "missing": missing}

    day = total(day_names)
    totals: Dict[str, Any] = {"day_seconds": day["seconds"], "day_hours": day["hours"], "day_usd": day["usd"], "not_estimated": day["missing"]}
    if public_round:
        everything = total(day_names | {"public_round", "publish"})
        totals.update({"with_public_round_seconds": everything["seconds"], "with_public_round_hours": everything["hours"],
                       "with_public_round_usd": everything["usd"]})

    memory = memory_report(profile, card, mode, rank)
    warnings: List[Dict[str, str]] = []
    if mode == "full" and count > 1 and not memory["fits"]:
        warnings.append({"code": "ddp_does_not_fit", "text": (
            f"DDP full fine-tuning keeps the bf16 weights and the bf16 gradients on every GPU: {memory['gpu_gb_needed']:.1f} GB, and a "
            f"each {card.name} has {card.memory_gb:g} GB. It does not fit. Use GPUs with more memory (H200 141 GB, B200 180 GB) or FSDP, which "
            f"decide/train does not have. Weights-plus-gradients arithmetic, UNVERIFIED.")})
    elif mode == "full" and not memory["fits"]:
        warnings.append({"code": "ddp_does_not_fit", "text": (
            f"Full fine-tuning keeps bf16 weights and bf16 gradients on the GPU: {memory['gpu_gb_needed']:.1f} GB, and the {card.name} has "
            f"{card.memory_gb:g} GB. It does not fit. Use GPUs with more memory (H200 141 GB, B200 180 GB) or FSDP, which decide/train does "
            f"not have. UNVERIFIED.")})
    if memory["fits"]:
        warnings.append({"code": "activations_not_counted", "text": (
            f"UNVERIFIED: {memory['gpu_gb_needed']:.1f} GB of {card.memory_gb:g} GB are weights, gradients and adapters; activations, the CUDA "
            f"context and fragmentation come out of the remaining {card.memory_gb - memory['gpu_gb_needed']:.1f} GB. A first short run measures it "
            f"(events.jsonl has gpu_peak_gb).")})
    if memory["host_gb_optimizer"] >= HOST_MEMORY_WARNING_GB:
        warnings.append({"code": "host_memory", "text": (
            f"The optimizer lives on the host: {memory['host_gb_optimizer']:.0f} GB of RAM on rank 0 (fp32 master copy, fp32 gradient copy "
            f"and two Adam moments, 16 bytes per trainable parameter). UNVERIFIED.")})
    if profile.host_ram_gb < memory["host_gb_optimizer"]:
        warnings.append({"code": "profile_host_ram_too_small", "text": (
            f"The profile asks for {profile.host_ram_gb:g} GB of host RAM and the optimizer needs {memory['host_gb_optimizer']:.0f} GB. UNVERIFIED.")})
    if traces and thinker_tokens_per_second is None:
        warnings.append({"code": "traces_not_estimated", "text": "The thinker-trace step has no time estimate without a measured --thinker-tps; "
                                                                   "the totals leave it out."})

    return {
        "status": STATUS,
        "profile": profile.name,
        "title": profile.title,
        "assumptions": {
            "params": HW.PARAMS, "gpu": card.name, "gpus": count, "mfu": utilization, "effective_tflops": count * card.bf16_tflops * utilization,
            "usd_per_gpu_second": price, "examples": n_examples, "epochs": epochs, "seeds": n_seeds, "tokens_per_example": per_example,
            "tokens_trained": tokens_trained, "public_round_tokens": HW.PUBLIC_ROUND_INPUT_TOKENS, "io_gb_per_s": HW.ASSUMED_IO_GB_PER_S,
            "price_source": HW.PRICE_SOURCE, "spec_source": HW.SPEC_SOURCE, "overridden": overridden,
        },
        "phases": phases,
        "totals": totals,
        "plan_section_6": {"hours": profile.plan.get("hours"), "usd": profile.plan.get("usd")},
        "memory": memory,
        "warnings": warnings,
    }


def render_text(report: Dict[str, Any]) -> str:
    a, lines = report["assumptions"], []
    lines.append(f"{report['profile']}: {report['title']} -- every number below is an estimate, UNVERIFIED")
    lines.append(f"  assumption: {a['gpus']} x {a['gpu']}, {a['effective_tflops']:.0f} TFLOPS effective (MFU {a['mfu']:.2f} of peak dense BF16), "
                 f"{a['usd_per_gpu_second'] * 3600:.2f} USD per GPU-hour -- UNVERIFIED")
    lines.append(f"  assumption: {a['examples']} examples x {a['epochs']} epoch(s) x {a['seeds']} seed(s) x {a['tokens_per_example']} tokens per example "
                 f"= {a['tokens_trained']} tokens -- UNVERIFIED")
    if a["overridden"]:
        lines.append(f"  overridden on the command line: {', '.join(a['overridden'])} -- UNVERIFIED")
    for phase in report["phases"]:
        if phase["seconds"] is None:
            lines.append(f"phase {phase['name']}: no estimate ({phase['basis']}) -- UNVERIFIED")
        else:
            lines.append(f"phase {phase['name']}: {phase['hours']:.2f} h, {phase['usd']:.2f} USD -- UNVERIFIED -- {phase['basis']}")
    t = report["totals"]
    lines.append(f"day (set-up to the G4 verdict): {t['day_hours']:.2f} h, {t['day_usd']:.2f} USD -- UNVERIFIED")
    if "with_public_round_hours" in t:
        lines.append(f"day with the public round and the upload: {t['with_public_round_hours']:.2f} h, {t['with_public_round_usd']:.2f} USD -- UNVERIFIED")
    if t["not_estimated"]:
        lines.append(f"left out of the totals (no estimate): {', '.join(t['not_estimated'])} -- UNVERIFIED")
    plan = report["plan_section_6"]
    lines.append(f"PLAN.md section 6 says {plan['hours']} h and {plan['usd']} USD for this profile -- UNVERIFIED in the plan too")
    m = report["memory"]
    lines.append(f"memory per GPU: {m['gpu_gb_needed']:.1f} GB needed of {m['gpu_gb_available']:g} GB -- UNVERIFIED; "
                 f"host RAM for the optimizer: {m['host_gb_optimizer']:.0f} GB -- UNVERIFIED")
    for warning in report["warnings"]:
        lines.append(f"warning {warning['code']}: {warning['text']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m decide.rental.estimate", description=__doc__.split("\n\n")[0])
    add = parser.add_argument
    add("--profile", required=True, help="a shipped profile name (econ_b200, full_8xh100) or a YAML path")
    add("--gpu", help="replace the profile's GPU (B200, H200, H100, A100-80GB)")
    add("--gpus", type=int)
    add("--examples", type=int)
    add("--tokens-per-example", type=int, help="the profile's value is an assumption; pass the mean you measured")
    add("--seeds", type=int)
    add("--mfu", type=float, help=f"share of peak throughput (default {HW.MFU})")
    add("--price-per-gpu-hour", type=float, help="USD per GPU-hour of your provider (default: the Modal list price of the GPU)")
    add("--traces", action="store_true", help="include the thinker-trace step (needs --thinker-tps for a time)")
    add("--thinker-tps", type=float, help="measured generation throughput of the thinker, tokens per second per GPU")
    add("--no-public-round", action="store_true", help="leave the public round and the upload out")
    add("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = load_profile(args.profile)
        report = estimate(profile, gpu=args.gpu, gpus=args.gpus, examples=args.examples, tokens_per_example=args.tokens_per_example, seeds=args.seeds,
                          mfu=args.mfu, usd_per_gpu_hour=args.price_per_gpu_hour, traces=args.traces, thinker_tokens_per_second=args.thinker_tps,
                          public_round=not args.no_public_round)
    except (ProfileError, KeyError, ValueError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) and error.args else error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
