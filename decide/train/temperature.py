"""Fit the temperature of a joint checkpoint on the dev split of a targets dataset, and store it in ``decision_config.json``.

    python -m decide.train.temperature --checkpoint soup --targets targets             # report only
    python -m decide.train.temperature --checkpoint soup --targets targets --write     # and update soup/decision_config.json

A seed soup (``decide.train.soup``) averages the temperatures of its inputs, which is only a placeholder: the averaged
weights have their own logit scale.  Run this on the soup before using it.  The fit is the one ``train`` does at the end of a
run: ``softmax(logits / T)`` with the ``T`` that minimises the negative log-likelihood of the gold labels, one ``T`` for all
questions (the one the checkpoint carries) and one per question type (``choice``, ``noul``, ``score``; a type with fewer than 50
questions takes the overall ``T`` and is listed in ``temperature_fallback``).  When the dev examples carry thinker reasoning, a
second ``T`` is fitted with the reasoning shown (``temperature_with_reasoning``).  Only questions with a gold label count.

``--write`` changes the calibration keys of ``decision_config.json`` and nothing else (atomic replace of that one small file);
the weights are never touched.  Without ``--write`` the checkpoint is not modified.

The model is loaded in the precision it is served in (bf16 on a GPU, fp32 on a CPU).  The parts that decide what goes into the
file (:func:`calibration_fields`, :func:`write_calibration`) need neither torch nor a model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

TYPES = ("choice", "noul", "score")
#: Keys of ``decision_config.json`` that describe a fit; a new fit replaces all of them, so the file never mixes two fits.
CALIBRATION_KEYS = ("temperature_by_type", "temperature_fallback", "temperature_with_reasoning", "temperature_is_placeholder", "calibration")
DEFAULT_SEED = 20261021


class CalibrationError(ValueError):
    """There is nothing to fit the temperature on."""


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return float(value)


def calibration_fields(plain: Mapping[str, Any], reasoning: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The ``decision_config.json`` entries for a fit, from the reports of ``evaluate.evaluate_examples``.

    ``plain`` is the report without reasoning (its ``all`` temperature is the one the checkpoint carries); ``reasoning`` the
    report with the reasoning shown, if there was one."""
    fit = plain.get("temperature") if plain else None
    if not fit or fit.get("all") is None or not plain.get("questions"):
        raise CalibrationError("no dev question has a gold label, so there is nothing to fit the temperature on")
    counts = fit.get("counts", {})
    fields: Dict[str, Any] = {
        "temperature": float(fit["all"]),
        "temperature_by_type": {kind: float(fit[kind]) for kind in TYPES if kind in fit},
        "temperature_fallback": list(fit.get("fallback", [])),
        "temperature_is_placeholder": False,
        "calibration": {"split": "dev", "examples": plain["examples"], "questions": plain["questions"], "skipped_too_long": plain.get("skipped_too_long", 0),
                        "questions_by_type": {kind: counts[kind] for kind in TYPES if kind in counts}},
    }
    thinking = (reasoning or {}).get("temperature") or {}
    if thinking.get("all") is not None:
        fields["temperature_with_reasoning"] = float(thinking["all"])
        fields["calibration"]["with_reasoning"] = {"examples": reasoning["examples"], "questions": reasoning["questions"]}
    return fields


def write_calibration(checkpoint: "str | os.PathLike[str]", fields: Mapping[str, Any]) -> Dict[str, Any]:
    """Store ``fields`` (from :func:`calibration_fields`) in the ``decision_config.json`` of ``checkpoint`` and return the new config.

    Everything is validated before the file is touched; the file is replaced atomically.  The calibration keys of an earlier fit
    are removed first, and a ``soup`` block, if there is one, stops being flagged as a placeholder."""
    path = Path(checkpoint) / "decision_config.json"
    temperature = _positive(fields.get("temperature"), "temperature")
    by_type = {str(kind): _positive(value, f"temperature_by_type[{kind}]") for kind, value in (fields.get("temperature_by_type") or {}).items()}
    thinking = fields.get("temperature_with_reasoning")
    if thinking is not None:
        thinking = _positive(thinking, "temperature_with_reasoning")
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in CALIBRATION_KEYS:
        config.pop(key, None)
    config["temperature"] = temperature
    config["temperature_by_type"] = by_type
    config["temperature_fallback"] = list(fields.get("temperature_fallback") or [])
    config["temperature_is_placeholder"] = False
    if thinking is not None:
        config["temperature_with_reasoning"] = thinking
    if fields.get("calibration") is not None:
        config["calibration"] = dict(fields["calibration"])
    if isinstance(config.get("soup"), dict) and "temperature_is_placeholder" in config["soup"]:
        config["soup"]["temperature_is_placeholder"] = False
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, path)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m decide.train.temperature", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add = parser.add_argument
    add("--checkpoint", required=True, help="joint checkpoint folder (the soup, or a run's final/)")
    add("--targets", required=True, help="targets dataset made by decide.train.build_targets (SCHEMA.md, section 11)")
    add("--split", default="dev", help="split to fit on (default: dev)")
    add("--max-examples", type=int, default=0, help="fit on at most N examples, chosen by a hash of the id (0: all)")
    add("--reasoning", choices=("auto", "off"), default="auto", help="auto: also fit with the reasoning shown when the examples carry it")
    add("--max-length", type=int, default=8192, help="requests longer than this are skipped")
    add("--token-budget", type=int, default=16384, help="padded tokens per batch")
    add("--batch-rows", type=int, default=32, help="most requests in one batch")
    add("--seed", type=int, default=DEFAULT_SEED)
    add("--device", default="auto")
    add("--cpu-threads", type=int, default=8)
    add("--write", action="store_true", help="store the fit in the checkpoint's decision_config.json (default: only report)")
    add("--report", help="also write the JSON report to this file")
    return parser


def main(argv: Optional[Sequence[str]] = None, *, tokenizer: Any = None) -> int:
    """Run the fit.  ``tokenizer`` replaces the checkpoint's own (tests pass a small stand-in)."""
    args = build_parser().parse_args(argv)
    for name in ("max_length", "token_budget", "batch_rows", "cpu_threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_examples < 0:
        raise ValueError("--max-examples cannot be negative")

    import torch   # imported here so that ``--help`` and the pure functions above work without it

    from .codes import ANSWER_CODES
    from .data import load_examples
    from .evaluate import brief, evaluate_examples, subset
    from .model import JointModel, check_codes, load_tokenizer, read_config
    from .protocol import RequestEncoder

    checkpoint, targets = Path(args.checkpoint), Path(args.targets)
    examples = subset(load_examples(targets, split=args.split), args.max_examples)
    if not examples:
        raise ValueError(f"{targets}: no examples in split {args.split!r}")
    torch.set_num_threads(args.cpu_threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    config = read_config(checkpoint)
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(checkpoint)
    check_codes(tokenizer, config)
    encoder = RequestEncoder(tokenizer, ANSWER_CODES)
    model = JointModel.load(checkpoint, mode="none", device=device)
    common: Dict[str, Any] = dict(seed=args.seed, max_length=args.max_length, token_budget=args.token_budget, max_rows=args.batch_rows, device=device)
    plain = evaluate_examples(model, encoder, examples, reasoning=False, **common)
    thinking = None
    if args.reasoning == "auto" and any(example.reasoning is not None for example in examples):
        thinking = evaluate_examples(model, encoder, examples, reasoning=True, **common)
    fields = calibration_fields(plain, thinking)
    report: Dict[str, Any] = {"checkpoint": str(checkpoint), "targets": str(targets), "split": args.split, "previous_temperature": config["temperature"],
                              "plain": brief(plain), "fields": fields, "written": False}
    if thinking is not None:
        report["reasoning"] = brief(thinking)
    if args.write:
        write_calibration(checkpoint, fields)
        report["written"] = True
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
