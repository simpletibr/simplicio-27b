"""Fine-tune the joint decider on a targets dataset: LoRA or full weights, one GPU or ``torchrun`` DDP on one node.

    python -m decide.train.train --targets targets --checkpoint pplx-decider-v1.2 --out runs/seed1
    torchrun --nproc_per_node 8 -m decide.train.train --mode full --targets targets --checkpoint ... --out runs/full1

Adapted from ``autojev/train.py`` of pplx-decider-v1.1-27b (Apache License 2.0; see LICENSE, NOTICE).  Changes: one
joint sequence per request with a readout at every question marker; the targets come from ``build_targets``
(gold mixed with a configurable teacher combination); options are shuffled by example and epoch; an optional
thinker reasoning is shown to a fraction of the examples; LoRA as well as full fine-tuning; the loss is the mean over
questions with a target; the dev split calibrates the temperature that goes into the checkpoint.

The data order, the option order and the reasoning draws depend only on ``--seed``, the epoch and the example id, so a
resumed run (``--resume``) continues exactly where it stopped.  The optimizer keeps fp32 master weights on the CPU
(``optim.CPUOffloadAdamW``) and only rank 0 steps it; the other ranks receive the new trainable weights.

Output folder: ``run.json`` (arguments and fingerprint), ``events.jsonl`` (progress), ``resume.pt`` (resume state),
``final/`` (checkpoint with LoRA merged and the fitted temperature) and ``summary.json``.  ``final/`` is written as ``final.tmp``
and renamed when whole, so a kill during the save never leaves a half checkpoint under the name ``final`` (a ``--resume`` removes a
stale ``final.tmp``, trains the steps after its last resume state again and saves).  The checkpoint carries the LICENSE and the NOTICE of
the one the run started from, a ``Modifications:`` line, and ``training`` in its ``decision_config.json``: the method, the steps, the
precision, the corpus provenance of the targets (``corpus``, from ``build_targets``) and the number of requests the length limit dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from decide.data import schema as S

from . import events
from .batching import Item, ReasoningPolicy, TooLong, make_item
from .codes import ANSWER_CODES
from .data import Example, load_examples
from .distributed import DistributedTraining
from .evaluate import brief, evaluate_examples, subset
from .loss import decision_loss
from .model import JointModel, LoraSpec, check_codes, collate, load_tokenizer, read_config, target_tensors
from .optim import CPUOffloadAdamW
from .protocol import RequestEncoder
from .schedule import epoch_order, learning_rate_factor, microbatches, plan_steps, step_seed
from .temperature import CalibrationError, calibration_fields

DEFAULT_LR = {"lora": 2e-5, "full": 2e-6}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DEFAULT_SEED = 20261021


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m decide.train.train", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add = parser.add_argument
    add("--targets", required=True, help="targets dataset made by decide.train.build_targets (SCHEMA.md, section 11)")
    add("--checkpoint", required=True, help="joint checkpoint to start from (pplx-decider-v1.2 layout)")
    add("--out", required=True, help="run folder (new, or the one to --resume)")
    add("--mode", choices=("lora", "full"), default="lora")
    add("--lora-rank", type=int, default=128)
    add("--lora-alpha", type=float, help="default: the rank")
    add("--lora-dropout", type=float, default=0.0)
    add("--epochs", type=int, default=1)
    add("--effective-batch", type=int, default=256, help="requests per optimizer step")
    add("--batch-rows", type=int, default=32, help="most requests in one micro-batch")
    add("--token-budget", type=int, default=8192, help="padded tokens per micro-batch (longest * rows)")
    add("--max-length", type=int, default=8192, help="requests longer than this (after dropping the reasoning) are skipped")
    add("--lr", type=float, help=f"default: {DEFAULT_LR}")
    add("--warmup-ratio", type=float, default=0.15)
    add("--weight-decay", type=float, default=0.01)
    add("--grad-clip", type=float, default=1.0)
    add("--brier-weight", type=float, default=0.0)
    add("--alpha", type=float, default=0.5, help="weight of the gold label in the target (1 = gold only, 0 = teachers only)")
    add("--no-permute", action="store_true", help="do not shuffle the options of choice questions")
    add("--seed", type=int, default=DEFAULT_SEED)
    add("--reasoning", choices=("off", "mixed", "always"), default="off",
        help="show the thinker's reasoning when the example has one (folded into the state by protocol.fold_reasoning, SCHEMA.md section 10; "
             "the thinker side must reproduce tests/fold_reasoning_cases.json before any training that uses it)")
    add("--reasoning-fraction", type=float, default=0.5, help="with --reasoning mixed: the share of examples that show it")
    add("--max-train", type=int, default=0, help="use only the first N training examples (pilot runs)")
    add("--max-steps", type=int, default=0, help="stop after N optimizer steps (the schedule still spans the whole run)")
    add("--eval-every", type=int, default=0, help="evaluate on dev every N steps (0: only at the end)")
    add("--max-dev", type=int, default=0, help="evaluate on at most N dev examples (0: all)")
    add("--resume-every", type=int, default=100)
    add("--resume", action="store_true", help="continue the run in --out from its resume.pt")
    add("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None, help="default: on when CUDA is available")
    add("--device", default="auto")
    add("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto",
        help="precision of the model: auto = bf16 on a GPU, fp32 on a CPU.  A T4 has no native bf16 (fp16 or fp32 there; fp16 has no loss scaling "
             "here, so a nonfinite loss or gradient stops the run); part of the fingerprint of a run, so --resume cannot change it")
    add("--cpu-threads", type=int, default=8)
    add("--dry-run", action="store_true", help="read the data, print the plan and stop (no model is loaded)")
    return parser


def validate(args: argparse.Namespace) -> None:
    for name in ("epochs", "effective_batch", "batch_rows", "token_budget", "max_length", "lora_rank", "resume_every", "cpu_threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_train", "max_steps", "eval_every", "max_dev"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if not 0.0 < args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be strictly between 0 and 1")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    for name in ("weight_decay", "brier_weight", "lora_dropout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and not negative")
    if not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("--lr must be positive")
    ReasoningPolicy(args.reasoning, args.reasoning_fraction)       # validates both


def digest_of(path: Path) -> str:
    """Fingerprint of a targets dataset: the name and sha256 of each part its manifest lists (``ManifestError`` if a part does not
    match, so a resumed run never continues on data that changed under it).  A single parquet file counts as one part."""
    entries = sorted((file.name, S.sha256_file(file)) for file in S.corpus_files(path, require_manifest=True))
    return hashlib.sha256(json.dumps(entries).encode("utf-8")).hexdigest()


#: Small files of a checkpoint that decide what the model is and how a request is read; the weights are added by name.
DIGEST_FILES = ("decision_config.json", "joint_config.json", "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "model.safetensors.index.json")


def checkpoint_digest(checkpoint: Path) -> str:
    """Fingerprint of the checkpoint a run starts from: the small files above and every ``*.safetensors`` file (the backbone shards
    and the readout), each by name and sha256 of its content.  Documentation (README.md) is not part of it.  The 27B checkpoint is
    about 54 GB, so this reads it once at the start of a run and of a resume (time UNVERIFIED on the rented disk)."""
    names = [name for name in DIGEST_FILES if (checkpoint / name).is_file()] + sorted(file.name for file in checkpoint.glob("*.safetensors"))
    return hashlib.sha256(json.dumps([(name, S.sha256_file(checkpoint / name)) for name in names]).encode("utf-8")).hexdigest()


def targets_provenance(targets: Path) -> Optional[Dict[str, Any]]:
    """The ``corpus`` block of the manifest of the targets (where the rows came from, ``build_targets``), or None if there is none."""
    try:
        manifest = json.loads((targets / S.MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    corpus = manifest.get("corpus") if isinstance(manifest, dict) else None
    return corpus if isinstance(corpus, dict) else None


def resolve_dtype(requested: str, cuda: bool) -> str:
    return ("bf16" if cuda else "fp32") if requested == "auto" else requested


def pick_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def save_resume(path: Path, step: int, optimizer: CPUOffloadAdamW, fingerprint: Dict[str, Any], dropped: Dict[str, Any]) -> None:
    pending = path.with_name(path.name + ".pending")
    torch.save({"format": 1, "step": step, "fingerprint": fingerprint, "optimizer": optimizer.state_dict(), "dropped": dropped}, pending)
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)


def main(argv: Optional[Sequence[str]] = None, *, tokenizer: Any = None) -> int:
    """Run the training.  ``tokenizer`` replaces the checkpoint's own (tests pass a small stand-in)."""
    args = build_parser().parse_args(argv)
    validate(args)
    out, targets, checkpoint = Path(args.out), Path(args.targets), Path(args.checkpoint)
    train = load_examples(targets, split="train")
    if args.max_train:
        train = train[:args.max_train]
    dev = subset(load_examples(targets, split="dev"), args.max_dev)
    if not train:
        raise ValueError(f"{targets}: no training examples (split 'train')")
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.mode]
    plan = plan_steps(rows=len(train), effective_batch=args.effective_batch, epochs=args.epochs)
    stop = min(plan.total_steps, args.max_steps) if args.max_steps else plan.total_steps
    policy = ReasoningPolicy(args.reasoning, args.reasoning_fraction)
    summary_plan = {"train_examples": len(train), "dev_examples": len(dev), "steps": plan.total_steps, "steps_per_epoch": plan.per_epoch,
                    "stop_after": stop, "effective_batch": args.effective_batch, "epochs": args.epochs, "lr": lr, "mode": args.mode,
                    "train_with_reasoning": sum(e.reasoning is not None for e in train), "dev_with_reasoning": sum(e.reasoning is not None for e in dev)}
    if args.dry_run:
        print(json.dumps(summary_plan, sort_keys=True))
        return 0

    torch.set_num_threads(args.cpu_threads)
    distributed = DistributedTraining.initialize()
    try:
        with events.default_log(out / "events.jsonl"):          # this run's own log, however many runs this process makes
            return _train(args, distributed, out, targets, checkpoint, train, dev, plan, stop, policy, lr, summary_plan, tokenizer)
    finally:
        distributed.close()


def _train(args: argparse.Namespace, distributed: DistributedTraining, out: Path, targets: Path, checkpoint: Path, train: List[Example],
           dev: List[Example], plan, stop: int, policy: ReasoningPolicy, lr: float, summary_plan: Dict[str, Any], tokenizer: Any) -> int:
    device = pick_device(args.device) if distributed.world_size == 1 else str(distributed.device)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.backends.cuda.enable_cudnn_sdp(False)
    gradient_checkpointing = cuda if args.gradient_checkpointing is None else args.gradient_checkpointing
    run_file = out / "run.json"
    if run_file.exists() and not args.resume:
        raise ValueError(f"{out} already holds a run; pass --resume to continue it or choose a new --out")
    if args.resume and not (out / "resume.pt").is_file():
        raise ValueError(f"--resume: {out / 'resume.pt'} does not exist")
    if args.resume and (out / "final").exists():
        raise ValueError(f"--resume: {out / 'final'} already exists, so the run is finished; there is nothing to continue")
    if distributed.primary:
        out.mkdir(parents=True, exist_ok=True)
    distributed.barrier()

    config = read_config(checkpoint)
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(checkpoint)
    check_codes(tokenizer, config)
    encoder = RequestEncoder(tokenizer, ANSWER_CODES)
    dtype_name = resolve_dtype(args.dtype, cuda)
    provenance = targets_provenance(targets)
    fingerprint = {
        "targets": digest_of(targets), "checkpoint": checkpoint_digest(checkpoint), "world_size": distributed.world_size, "dtype": dtype_name,
        **{key: getattr(args, key) for key in ("mode", "lora_rank", "lora_alpha", "lora_dropout", "epochs", "effective_batch", "batch_rows", "token_budget",
                                                "max_length", "warmup_ratio", "weight_decay", "grad_clip", "brier_weight", "alpha", "no_permute", "seed",
                                                "reasoning", "reasoning_fraction", "max_train")},
        "lr": lr,
    }

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    model = JointModel.load(checkpoint, mode=args.mode, device=device, lora=LoraSpec(args.lora_rank, args.lora_alpha if args.lora_alpha is not None else float(args.lora_rank),
                                                                                  args.lora_dropout),
                            gradient_checkpointing=gradient_checkpointing, dtype=DTYPES[dtype_name])
    optimizer = CPUOffloadAdamW(model.trainable(), lr=lr, weight_decay=args.weight_decay) if distributed.primary else None
    trainable = [parameter for _, parameter in model.trainable()]
    step = 0
    dropped_requests, dropped_examples = 0, set()          # the requests the length limit left out, by request and by distinct example
    if args.resume:
        state = torch.load(out / "resume.pt", map_location="cpu", weights_only=True) if distributed.primary else None
        saved = distributed.broadcast({key: state[key] for key in ("format", "step", "fingerprint")} if state is not None else None)
        if saved["format"] != 1 or saved["fingerprint"] != fingerprint:
            differing = sorted(key for key in set(fingerprint) | set(saved["fingerprint"]) if fingerprint.get(key) != saved["fingerprint"].get(key))
            raise ValueError(f"--resume: the saved run differs in {', '.join(differing)}")
        step = int(saved["step"])
        if state is not None and isinstance(state.get("dropped"), dict):
            dropped_requests, dropped_examples = int(state["dropped"]["requests"]), set(state["dropped"]["examples"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])      # also copies the saved weights into the trainable parameters
        distributed.broadcast_parameters(model)                # the other ranks get them from rank 0
        del state
    elif distributed.primary:
        run_file.write_text(json.dumps({"args": vars(args), "fingerprint": fingerprint, "plan": summary_plan}, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    distributed.barrier()
    training_model = distributed.wrap(model)
    if distributed.primary:
        events.record("training_started", plan=summary_plan, step=step, trainable_parameters=sum(p.numel() for p in trainable), world_size=distributed.world_size,
                      dtype=dtype_name, corpus=provenance)

    order_cache: Dict[int, List[int]] = {}

    def evaluate(reason: str) -> Dict[str, Any]:
        report = {"plain": evaluate_examples(model, encoder, dev, reasoning=False, seed=args.seed, max_length=args.max_length,
                                             token_budget=args.token_budget * 2, max_rows=args.batch_rows, device=device)}
        if policy.mode != "off" and any(e.reasoning is not None for e in dev):
            report["reasoning"] = evaluate_examples(model, encoder, dev, reasoning=True, seed=args.seed, max_length=args.max_length,
                                                    token_budget=args.token_budget * 2, max_rows=args.batch_rows, device=device)
        events.record("evaluation", reason=reason, step=step, **{k: brief(v) for k, v in report.items()})
        return report

    while step < stop:
        began = time.monotonic()
        epoch = step // plan.per_epoch
        if epoch not in order_cache:
            order_cache.clear()
            order_cache[epoch] = epoch_order(plan.rows, seed=args.seed, epoch=epoch)
        indices = plan.batch_indices(step, seed=args.seed, order=order_cache[epoch])
        torch.manual_seed(step_seed(args.seed, step, distributed.rank))      # dropout masks: a function of (seed, step, rank), so a resumed run matches
        items: List[Item] = []
        too_long = 0
        for index in indices:
            try:
                items.append(make_item(train[index], encoder, seed=args.seed, epoch=epoch, alpha=args.alpha, permute=not args.no_permute,
                                       reasoning=policy, max_length=args.max_length))
            except TooLong:
                too_long += 1
                dropped_examples.add(train[index].example_id)
        dropped_requests += too_long
        items.sort(key=lambda item: item.length)
        masses = {id(item): sum(target is not None for target in item.targets) for item in items}
        total_mass = sum(masses.values())
        step += 1
        if total_mass == 0:
            if distributed.primary:
                events.record("training_step", step=step, skipped="no question with a target", dropped_too_long=too_long)
            continue

        model.train()
        model.zero_grad(set_to_none=True)
        if optimizer is not None:
            optimizer.zero_grad()
        sums = {"loss": 0.0, "cross_entropy": 0.0, "brier": 0.0}
        tokens = 0
        split = lambda rows: [[rows[i] for i in group] for group in microbatches([row.length for row in rows], max_rows=args.batch_rows, token_budget=args.token_budget)]  # noqa: E731
        for batch_items, count in distributed.batches(items, split):
            batch = collate([item.encoded for item in batch_items], encoder.pad_id, device)
            target, weights = target_tensors([item.targets for item in batch_items], device)
            logits = training_model(batch)
            components = decision_loss(logits, target, args.brier_weight, weights)
            if not torch.isfinite(components.total):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            mass = sum(masses[id(item)] for item in batch_items) if count else 0
            (components.total * (mass * distributed.world_size / total_mass)).backward()
            sums["loss"] += float(components.total.detach()) * mass
            sums["cross_entropy"] += float(components.cross_entropy.detach()) * mass
            sums["brier"] += float(components.brier.detach()) * mass
            tokens += batch.tokens if count else 0
            del logits, components, target, weights, batch
        loss_sum, tokens = distributed.totals(sums["loss"], tokens)
        cross_entropy_sum, _ = distributed.totals(sums["cross_entropy"], 0)
        brier_sum, _ = distributed.totals(sums["brier"], 0)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"Nonfinite gradient at step {step}")
        factor = learning_rate_factor(step, plan.total_steps, args.warmup_ratio)
        if optimizer is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr * factor
            optimizer.step()
        distributed.broadcast_parameters(model)
        if cuda:
            torch.cuda.synchronize()
        if distributed.primary:
            events.record("training_step", step=step, epoch=epoch, loss=loss_sum / total_mass, cross_entropy=cross_entropy_sum / total_mass,
                          brier=brier_sum / total_mass, learning_rate=lr * factor, questions=total_mass, examples=len(items), input_tokens=tokens,
                          gradient_norm=gradient_norm, step_seconds=time.monotonic() - began, elapsed_seconds=time.monotonic() - started,
                          dropped_too_long=too_long, with_reasoning=sum(item.reasoning for item in items), reasoning_fell_back=sum(item.fell_back for item in items),
                          gpu_peak_gb=(torch.cuda.max_memory_allocated() / 1e9) if cuda else None)
        if args.eval_every and step % args.eval_every == 0 and step < stop and dev:
            distributed.on_primary(lambda: evaluate("periodic"))
        if optimizer is not None and (step % args.resume_every == 0 or step == stop) and step < plan.total_steps:
            save_resume(out / "resume.pt", step, optimizer, fingerprint, {"requests": dropped_requests, "examples": sorted(dropped_examples)})

    summary: Dict[str, Any] = {"steps": step, "planned_steps": plan.total_steps, "complete": step == plan.total_steps, "plan": summary_plan,
                               "dropped_too_long": dropped_requests, "dropped_too_long_examples": len(dropped_examples), "dtype": dtype_name,
                               "elapsed_seconds": time.monotonic() - started}

    def finish() -> None:
        if not summary["complete"]:
            summary["note"] = "stopped early (--max-steps); continue with --resume"
        else:
            model.merge()
            report = evaluate("final") if dev else {}
            extra: Dict[str, Any] = {"training": {"method": args.mode, "steps": step, "examples": len(train), "alpha": args.alpha, "seed": args.seed,
                                                  "reasoning": args.reasoning, "dtype": dtype_name, "corpus": provenance,
                                                  "dropped_too_long": dropped_requests, "run_fingerprint": fingerprint}}
            temperature = None
            try:       # the same fit, and the same keys, as ``python -m decide.train.temperature --write``
                fields = calibration_fields(report.get("plain", {}), report.get("reasoning"))
                temperature = fields.pop("temperature")
                extra.update(fields)
            except CalibrationError:
                pass   # no dev question with a gold label: the checkpoint keeps the temperature it started with
            shutil.rmtree(out / "final.tmp", ignore_errors=True)        # what a killed save left
            model.save(out / "final.tmp", temperature=temperature, extra=extra,
                       modification=f"fine-tuned with decide.train.train ({args.mode}, {step} steps, seed {args.seed}, {dtype_name}); backbone and readout "
                                    "weights and decision_config.json (temperature, calibration, training record) changed.")
            os.replace(out / "final.tmp", out / "final")                # whole or not at all
            saved = out / "final"
            summary.update({"checkpoint": str(saved), "temperature": temperature, "evaluation": {k: brief(v) for k, v in report.items()}})
            events.record("checkpoint_saved", path=str(saved), temperature=temperature)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        events.record("training_finished", **{k: v for k, v in summary.items() if k != "plan"})

    distributed.on_primary(finish)
    return 0


if __name__ == "__main__":
    sys.exit(main())
