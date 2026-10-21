"""A kit-compatible decision engine on top of the vendored pplx-decider reference path.

`AssembledDecider` is the upstream `Decider` with its constructor replaced: it takes a backbone and a readout that
`loader.py` built (merged, quantised, placed on devices) instead of reading a checkpoint directory.  Everything
that computes logits (`reference_logits`, batching by length, left padding, the noncausal-attention hook) is the
vendored upstream code, unchanged.

`DecisionEngine` is what the scoring kit calls: `engine(state, questions) -> (response, raw)`.  With option-order
TTA it runs the request once per view, maps every view's logits back to the original option order with the exact
inverse permutation (`tta.py`), and reports the mean over views of softmax(logits / temperature).  The raw logits of
every view are kept in `raw` so calibration and teacher labels can be computed later without running the model.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from decide.sd import calibrate, tta
from decide.sd.vendor.pplx_decider.model import DEFAULT_MAX_LENGTH, Decider
from decide.sd.vendor.pplx_decider.protocol import (
    LEGACY_MAX_PIXELS,
    MAX_OPTIONS,
    Encoded,
    Encoder,
    LengthError,
    RequestError,
    answer,
)

from decide.sd.kitapi import Engine, Unsupported

SUPPORTED_TYPES = ("choice", "noul")                      # what the Decision Index kit scores
ALL_TYPES = ("choice", "noul", "score")                   # what the pplx protocol can answer (teacher labels use all three)
OOM_ERRORS: Tuple[type, ...] = tuple(e for e in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)) if e)


class AssembledDecider(Decider):
    """`Decider` around a backbone and readout that are already in memory."""

    def __init__(self, backbone: torch.nn.Module, readout: torch.nn.Module, processor: Any, decision_config: Mapping[str, Any],
                 *, device: str = "cpu", max_length: Optional[int] = DEFAULT_MAX_LENGTH, batch_tokens: int = 4096,
                 encoder: Any = None) -> None:
        self.checkpoint = Path("<assembled>")
        self.device = torch.device(device)
        self.temperature = float(decision_config["temperature"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Saved temperature must be positive and finite")
        self.codes = list(decision_config["codes"])
        self.max_length = max_length
        self.batch_tokens = int(batch_tokens)
        self.max_pixels = int(decision_config.get("max_pixels") or LEGACY_MAX_PIXELS)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
        if len(self.codes) != MAX_OPTIONS or any(
            tokenizer.encode(code, add_special_tokens=False) != [token_id]
            for code, token_id in zip(self.codes, decision_config["token_ids"], strict=True)
        ):
            raise ValueError("Checkpoint answer codes differ from its tokenizer")
        self.processor = processor
        self.encoder = encoder if encoder is not None else Encoder(processor, self.codes)     # pplx joint prompt unless told otherwise
        self.backbone = backbone
        self.readout = readout
        self.fast = None
        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)      # the evaluated SDPA kernel choice

    # ---- planned execution (the upstream `_run` without re-planning)
    def run_plans(self, plans: Sequence[Sequence[Encoded]]) -> List[Dict[str, List[float]]]:
        """Logits (valid options only, before temperature) per question for every planned request."""
        flat = [(index, chunk) for index, plan in enumerate(plans) for chunk in plan]
        results: Dict[Tuple[int, str], List[float]] = {}
        stack = [group for group in self._batches(flat)]
        while stack:
            group = stack.pop(0)
            try:
                rows = self.reference_logits([c for _, c in group])
            except OOM_ERRORS as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(group) == 1:
                    raise Unsupported(f"out of memory at {group[0][1].length} tokens: {exc}") from exc
                half = len(group) // 2
                stack[0:0] = [group[:half], group[half:]]
                self.batch_tokens = max(1, min(self.batch_tokens, half * max(c.length for _, c in group)))
                continue
            for (index, chunk), values in zip(group, rows, strict=True):
                for key, row in zip(chunk.keys, values, strict=True):
                    results[index, key] = row
        return [{key: results[index, key] for plan_chunk in plan for key in plan_chunk.keys} for index, plan in enumerate(plans)]

    def run_batch(self, plans: Sequence[Sequence[Encoded]], requests: Sequence[Mapping[str, Any]]) -> List[Dict[str, List[float]]]:
        """Logits per question for every planned request: by the packed fast path when one is attached, else the reference path.

        `plans[i]` is the encoding of `requests[i]`; the fast path plans again by itself, so it is given the requests.
        """
        if self.fast is None:
            return self.run_plans(plans)
        rows = list(self.fast.run(list(requests)))
        if len(rows) != len(plans):
            raise RuntimeError(f"the fast path answered {len(rows)} requests out of {len(plans)}")
        return [{key: list(values) for key, values in logits.items()} for _request, logits, _chunks in rows]


def _check_request(request: Mapping[str, Any], types: Sequence[str] = SUPPORTED_TYPES) -> None:
    for key, question in request["questions"].items():
        if not isinstance(question, Mapping) or question.get("type") not in types:
            raise Unsupported(f"question {key!r}: this engine serves only {tuple(types)} questions (the kit scores {SUPPORTED_TYPES})")
    if request.get("images"):
        raise Unsupported("this engine loads the text tower only; image requests are not supported")


class DecisionEngine(Engine):
    """`engine(state, questions) -> (response, raw)` for the Decision Index kit."""

    name = "simplicio-decide"

    def __init__(self, decider: AssembledDecider, *, tta_rotations: int = 1, tta_mode: str = "reverse",
                 temperature: Optional[float] = None, temperatures: Optional[Mapping[str, float]] = None,
                 provenance: Optional[Mapping[str, Any]] = None, name: Optional[str] = None, check_finite: bool = True,
                 question_types: Sequence[str] = SUPPORTED_TYPES, **options: Any) -> None:
        super().__init__(**options)
        bad = [t for t in question_types if t not in ALL_TYPES]
        if bad:
            raise ValueError(f"question_types may only hold {ALL_TYPES}, got {bad}")
        self.question_types = tuple(question_types)
        if tta_mode not in tta.MODES:
            raise ValueError(f"tta_mode must be one of {tta.MODES}")
        self.decider = decider
        self.views = max(1, int(tta_rotations))
        self.tta_mode = tta_mode
        self.temperature = float(temperature) if temperature is not None else decider.temperature
        self.temperatures = dict(temperatures or {})
        self.check_finite = check_finite
        self.name = name or self.name
        self.provenance = dict(provenance or {})
        self.provenance.setdefault("temperature", self.temperature)
        self.provenance.setdefault("tta", {"views": self.views, "mode": tta_mode})

    # ---- helpers
    def temperature_for(self, question: Mapping[str, Any]) -> float:
        """The divisor for one question: `type:k-bucket`, else `type`, else `default`, else the model's own temperature."""
        kind = question["type"]
        n = 2 if kind == "noul" else len(question.get("criteria") or ())
        for key in (f"{kind}:{calibrate.k_bucket(n)}", kind, "default"):
            if key in self.temperatures:
                return float(self.temperatures[key])
        return self.temperature

    def _views(self, request: Mapping[str, Any]) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, Any]]]:
        plans = tta.plan(request, self.views, self.tta_mode)
        return plans, [tta.permute_request(request, p) for p in plans]

    def _finish(self, request: Mapping[str, Any], plans: Sequence[Mapping[str, Sequence[int]]],
                view_logits: Sequence[Mapping[str, Sequence[float]]], tokens: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        logits: Dict[str, List[List[float]]] = {}
        for key in request["questions"]:
            rows = [tta.restore_question(view_logits[v][key], plans[v][key]) for v in range(len(plans))]
            if self.check_finite and not all(math.isfinite(x) for row in rows for x in row):
                raise FloatingPointError(f"non-finite logits for question {key!r} (fp16 overflow? try compute_dtype bfloat16 or float32)")
            logits[key] = rows
        answers: Dict[str, Any] = {}
        applied: Dict[str, float] = {}
        for key, question in request["questions"].items():
            t = self.temperature_for(question)
            applied[key] = t
            answers[key] = answer(question, calibrate.reprobabilities(logits[key], t))
        response = {"model": self.name, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0, "sequences": len(plans)}}
        codes = {key: list(self.decider.codes[:len(rows[0])]) for key, rows in logits.items()}       # answer codes of the valid options
        raw = {"logits": logits, "codes": codes, "temperature": self.decider.temperature, "applied_temperatures": applied,
               "views": len(plans), "tokens": tokens}
        return response, raw

    # ---- kit interface
    def __call__(self, state: Any, questions: Mapping[str, Any], **inputs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        out = self.batch([{"state": state, "questions": questions, **inputs}])[0]
        if isinstance(out, BaseException):
            raise out
        return out

    def batch(self, requests: Sequence[Mapping[str, Any]]) -> List[Any]:
        """One (response, raw) per request, or the exception that request raised (`Unsupported`, `FloatingPointError`).

        All views of all requests are planned first; a request that cannot be encoded does not stop the others.
        """
        results: List[Any] = [None] * len(requests)
        jobs: List[Tuple[int, Mapping[str, Any], List[Dict[str, List[int]]], List[List[Encoded]], List[Dict[str, Any]]]] = []
        for i, request in enumerate(requests):
            try:
                _check_request(request, self.question_types)
                plans, view_requests = self._views(request)
                encoded = [self.decider.encoder.plan(r, self.decider.max_length) for r in view_requests]
            except (LengthError, RequestError) as exc:
                results[i] = Unsupported(str(exc))
                continue
            except Unsupported as exc:
                results[i] = exc
                continue
            jobs.append((i, request, plans, encoded, view_requests))
        if jobs:
            flat = [chunks for _i, _r, _p, encoded, _v in jobs for chunks in encoded]
            flat_requests = [view for _i, _r, _p, _e, views in jobs for view in views]
            try:
                logits = self.decider.run_batch(flat, flat_requests)
            except Unsupported as exc:
                for i, *_ in jobs:
                    results[i] = exc
                return results
            cursor = 0
            for i, request, plans, encoded, _views in jobs:
                mine = logits[cursor:cursor + len(encoded)]
                cursor += len(encoded)
                tokens = sum(c.length for chunks in encoded for c in chunks)
                try:
                    results[i] = self._finish(request, plans, mine, tokens)
                except FloatingPointError as exc:
                    results[i] = exc
        return results

    # ---- bookkeeping the kit's runner reads
    def synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def runtime(self) -> Dict[str, Any]:
        import transformers

        info: Dict[str, Any] = {"torch": torch.__version__, "transformers": transformers.__version__,
                                "device": str(self.decider.device), "views": self.views}
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return info

    def close(self) -> None:
        self.decider.backbone = None                         # type: ignore[assignment]
        self.decider.readout = None                          # type: ignore[assignment]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

