"""The d3 protocol (vllm-sr/d3, d3-mini): one question per sequence, the d3 prompt, last-token readout, float32 head, T = 1.

This path exists for the fidelity check (F1).  The per-row results published for d3 and d3-mini were produced with this
protocol, so comparing our NF4 or fp16 run against them has to feed the same prompts through the same readout.  The
model code is not d3's: it is the backbone the loader assembled (noncausal full attention in the full-attention layers,
causal DeltaNet) and the vendored `Decider` reference path, which d3's own `PromptEncoder` reproduces layer for layer.

What differs from the pplx joint protocol and is handled here:

* each question is its own sequence (`d3_format.render`), so `Encoded` chunks hold one key and the readout position is the
  last token;
* the readout is applied in float32 to the float32 hidden state (`CastReadout`), as `readout_dtype` says;
* the checkpoint's own `decision_config.json` carries the 255 answer codes (measured: the same list and token ids as
  pplx-decider-v1.2's), T = 1 and no input limit.

d3-turbo (`prefix_bidir` attention) is a different attention rule and is not served here.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import nn

from decide.sd.vendor.d3 import d3_format
from decide.sd.vendor.pplx_decider.protocol import Encoded, LengthError, RequestError

FORMAT_ID = d3_format.FORMAT_ID
MAX_OPTIONS = d3_format.MAX_OPTIONS
KINDS = ("choice", "noul")


class CastReadout(nn.Module):
    """`x @ W.T` with the head kept (and computed) in float32, whatever dtype the hidden state has."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.detach().to(torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32) @ self.weight.t()


class TokenizerOnly:
    """What `AssembledDecider` needs from a processor when only text is served."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer


def _json_ok(value: Any) -> bool:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def rendered_question(key: str, question: Any) -> Dict[str, Any]:
    """The question in the `d3_format` contract (what d3's `normalize_question` produces), or `RequestError`."""
    if not isinstance(question, Mapping):
        raise RequestError(f"Question {key!r}: a question is an object with a type")
    kind = question.get("type")
    instructions = question.get("instructions")
    if instructions is not None and not _json_ok(instructions):
        raise RequestError(f"Question {key!r}: instructions are not representable as JSON")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= MAX_OPTIONS:
            raise RequestError(f"Question {key!r}: choice criteria must map 1 to {MAX_OPTIONS} option keys to descriptions")
        if any(not isinstance(k, str) or not k for k in criteria):
            raise RequestError(f"Question {key!r}: choice option keys must be nonempty strings")
        if not _json_ok(dict(criteria)):
            raise RequestError(f"Question {key!r}: criteria are not representable as JSON")
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}
    if kind == "noul":
        if criteria is not None:
            if not isinstance(criteria, Mapping) or set(criteria) - {"false", "true"}:
                raise RequestError(f"Question {key!r}: noul criteria may only describe 'false' and 'true'")
            if not _json_ok(dict(criteria)):
                raise RequestError(f"Question {key!r}: criteria are not representable as JSON")
        out: Dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria is not None:
            out["criteria"] = dict(criteria)
        return out
    raise RequestError(f"Question {key!r}: type must be one of {KINDS} for the d3 protocol, got {kind!r}")


class D3Encoder:
    """Builds and tokenizes one d3 prompt per question.  Same surface as the pplx `Encoder` that the engine calls."""

    def __init__(self, tokenizer: Any, codes: Sequence[str]) -> None:
        if len(codes) != MAX_OPTIONS:
            raise ValueError(f"the d3 protocol has {MAX_OPTIONS} answer codes, got {len(codes)}")
        self.tokenizer = tokenizer
        self.codes = tuple(codes)
        pad = tokenizer.pad_token_id
        if pad is None:
            raise ValueError("the tokenizer has no pad token")
        self.pad_id: int = int(pad)
        probe = "State:\nA"
        if tokenizer(probe)["input_ids"] != tokenizer(probe, add_special_tokens=False)["input_ids"]:
            raise ValueError("the tokenizer adds special tokens; d3 prompts are tokenized exactly as rendered")

    def validate(self, request: Mapping[str, Any]) -> None:
        if not isinstance(request, Mapping) or "state" not in request or "questions" not in request:
            raise RequestError("A request needs 'state' and 'questions'")
        questions = request["questions"]
        if not isinstance(questions, Mapping) or not questions:
            raise RequestError("A request needs at least one question")
        if not _json_ok(request["state"]):
            raise RequestError("The state is not representable as JSON")
        for key, question in questions.items():
            if not isinstance(key, str) or not key:
                raise RequestError("Question IDs must be nonempty strings")
            rendered_question(key, question)

    def text(self, state: Any, question: Mapping[str, Any]) -> str:
        return d3_format.render(self.tokenizer, state, question, self.codes)

    def plan(self, request: Mapping[str, Any], max_length: Optional[int]) -> List[Encoded]:
        """One chunk per question, in request order.  A prompt over `max_length` fails the whole request (nothing is truncated)."""
        self.validate(request)
        texts = []
        for key, question in request["questions"].items():
            rendered = rendered_question(key, question)
            texts.append((key, rendered, self.text(request["state"], rendered)))
        ids = self.tokenizer([t for _k, _r, t in texts], add_special_tokens=False)["input_ids"]
        chunks: List[Encoded] = []
        for (key, rendered, _text), sequence in zip(texts, ids):
            if max_length is not None and len(sequence) > max_length:
                raise LengthError(f"Question {key!r} needs {len(sequence)} tokens, over the {max_length}-token limit; nothing is truncated")
            tensor = torch.tensor(sequence, dtype=torch.long)
            chunks.append(Encoded(keys=(key,), counts=(len(d3_format.options(rendered)[0]),), input_ids=tensor,
                                  markers=torch.tensor([len(sequence) - 1], dtype=torch.long)))
        return chunks


def check_decision_config(config: Mapping[str, Any]) -> None:
    """The d3 checkpoint features this path implements; anything else is refused by name."""
    if config.get("format_id") != FORMAT_ID or config.get("prompt", "d3") != "d3":
        raise ValueError(f"expected a {FORMAT_ID} checkpoint with the d3 prompt, got format_id={config.get('format_id')!r}")
    if config.get("attention_mode") != "noncausal_full_attention":
        raise ValueError(f"attention_mode {config.get('attention_mode')!r} is not served (only noncausal_full_attention; d3-turbo's prefix_bidir is a different rule)")
    if config.get("pooling", "last") != "last":
        raise ValueError("expected last-token pooling")
    if config.get("readout_dtype", "float32") != "float32":
        raise ValueError("this path computes the d3 head in float32")
    if len(config.get("codes", [])) != MAX_OPTIONS or len(config.get("token_ids", [])) != MAX_OPTIONS:
        raise ValueError(f"the checkpoint must carry {MAX_OPTIONS} answer codes and token ids")
