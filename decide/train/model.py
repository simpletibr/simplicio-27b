"""The joint decider: a Qwen3.5 backbone and a linear readout applied at the question markers.

Adapted from ``pplx_decider/model.py`` of perplexity-ai/pplx-decider-v1.2 and ``autojev/model.py`` of
pplx-decider-v1.1-27b (Apache License 2.0; see LICENSE, NOTICE).  Changes: the checkpoint is always a v1.2 joint
checkpoint (``joint_config.json`` must declare the version-3 protocol); the batch is built from already tokenized
``Encoded`` requests; LoRA (peft) is an alternative to full fine-tuning; the output keeps the layout of the input
checkpoint so the result loads in the pplx ``Decider`` and in the Squad A engines, with the ``LICENSE`` and the ``NOTICE`` of
the input checkpoint and a ``Modifications:`` line (``decide.train.notices``); no image path.

Which layers LoRA adapts (``LORA_TARGETS``; a choice of this repository, not of the plan in decide/PLAN.md, which only fixes the
rank): the attention projections ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj`` of the full-attention layers, the MLP
``gate_proj``, ``up_proj``, ``down_proj`` and, because most layers of Qwen3.5 are DeltaNet (linear attention) layers, their
``in_proj_qkv``, ``in_proj_z`` and ``out_proj``.  The tiny gates ``in_proj_a`` and ``in_proj_b`` of the DeltaNet layers, the
embeddings and the vision tower stay frozen.  Leaving the DeltaNet projections out would adapt only the few full-attention layers.
Whether that is the best set is UNVERIFIED (no ablation was run).

One forward pass reads one left-padded batch of joint sequences and returns the masked logits of every question
(``MAX_OPTIONS`` columns; options a question does not have get ``loss.MASK_VALUE``).
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file, save_file
from transformers.masking_utils import create_recurrent_attention_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

from .codes import ANSWER_CODES, MAX_OPTIONS
from .loss import mask_logits
from .notices import carry_notices
from .protocol import PROTOCOL, Encoded
from .soup import NOT_COPIED

#: Linear layers of the language model that LoRA adapts (the tiny ``in_proj_a`` / ``in_proj_b`` gates are left alone).
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj")
MODES = ("none", "lora", "full")


def enable_noncausal_full_attention(text_model: torch.nn.Module) -> None:
    """Full-attention layers attend to the whole (non-padding) sequence; linear-attention layers stay causal."""

    def mask_inputs(module: torch.nn.Module, args: tuple, kwargs: dict) -> tuple:
        if args:
            raise ValueError("The noncausal mask hook requires keyword inputs")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("Noncausal classification does not support a KV cache")
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs["input_ids"])
        padding = kwargs.get("attention_mask")
        if padding is None:
            padding = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        if isinstance(padding, dict):  # already converted
            return args, kwargs
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(config=module.config, inputs_embeds=embeddings, attention_mask=padding),
        }
        return args, kwargs

    text_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)


@dataclass(frozen=True)
class LoraSpec:
    rank: int = 128
    alpha: float = 128.0
    dropout: float = 0.0
    targets: Tuple[str, ...] = LORA_TARGETS

    def regex(self) -> str:
        """peft ``target_modules`` regular expression: the listed linear layers of the language model only."""
        return r".*language_model\.layers\.\d+\..*\.(" + "|".join(self.targets) + ")"


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor        # [B, T] left padded
    attention_mask: torch.Tensor   # [B, T]
    rows: torch.Tensor             # [Q] sequence of each question
    cols: torch.Tensor             # [Q] position of its readout marker
    counts: torch.Tensor           # [Q] number of options of each question
    tokens: int                    # non-padding tokens


def collate(encoded: Sequence[Encoded], pad_id: int, device: "torch.device | str" = "cpu") -> Batch:
    """Left-pad the sequences (the layout of the reference path) and locate the readout markers."""
    width = max(item.length for item in encoded)
    ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.long)
    rows: List[int] = []
    cols: List[int] = []
    counts: List[int] = []
    for row, item in enumerate(encoded):
        offset = width - item.length
        ids[row, offset:] = torch.tensor(item.input_ids, dtype=torch.long)
        mask[row, offset:] = 1
        rows += [row] * len(item.markers)
        cols += [marker + offset for marker in item.markers]
        counts += list(item.counts)
    tokens = int(mask.sum())
    return Batch(ids.to(device), mask.to(device), torch.tensor(rows, device=device), torch.tensor(cols, device=device),
                 torch.tensor(counts, device=device), tokens)


def target_tensors(targets: Sequence[Sequence[Optional[Sequence[float]]]], device: "torch.device | str" = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """``[Q, MAX_OPTIONS]`` target distributions and ``[Q]`` weights (0 for a question without a target) of a batch."""
    flat = [question for per_item in targets for question in per_item]
    values = torch.zeros((len(flat), MAX_OPTIONS), dtype=torch.float32)
    weights = torch.zeros(len(flat), dtype=torch.float32)
    for index, target in enumerate(flat):
        if target is not None:
            values[index, :len(target)] = torch.tensor(list(target), dtype=torch.float32)
            weights[index] = 1.0
    return values.to(device), weights.to(device)


def load_tokenizer(checkpoint: "str | Path") -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    tokenizer.padding_side = "left"
    return tokenizer


def read_config(checkpoint: "str | Path") -> Dict[str, Any]:
    """``decision_config.json`` of a joint checkpoint, after the checks that make it trainable here."""
    root = Path(checkpoint)
    protocol_path = root / "joint_config.json"
    if not protocol_path.is_file() or json.loads(protocol_path.read_text(encoding="utf-8")) != PROTOCOL:
        raise ValueError(f"{root}: not a joint inline-readout checkpoint (joint_config.json must declare protocol version 3)")
    config = json.loads((root / "decision_config.json").read_text(encoding="utf-8"))
    if config.get("attention_mode") != "noncausal_full_attention" or config.get("pooling", "last") != "last":
        raise ValueError(f"{root}: expected a noncausal checkpoint with per-position readouts")
    if list(config.get("codes", [])) != list(ANSWER_CODES):
        raise ValueError(f"{root}: the answer codes differ from decide.train.codes.ANSWER_CODES")
    temperature = config.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"{root}: the saved temperature must be positive and finite")
    return config


def check_codes(tokenizer: Any, config: Dict[str, Any]) -> None:
    """Every answer code must still be the single token the checkpoint was trained with."""
    for code, token_id in zip(config["codes"], config["token_ids"]):
        if list(tokenizer.encode(code, add_special_tokens=False)) != [token_id]:
            raise ValueError(f"answer code {code!r} is not the single token {token_id} in this tokenizer")


class JointModel(torch.nn.Module):
    """Backbone + readout.  ``forward(batch)`` returns the masked float32 logits of every question in the batch."""

    def __init__(self, backbone: torch.nn.Module, readout: torch.nn.Linear, config: Dict[str, Any], source: Path, mode: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.readout = readout
        self.config_data = config
        self.source = source
        self.mode = mode
        self.temperature = float(config["temperature"])

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, checkpoint: "str | Path", *, mode: str = "none", device: "str | torch.device" = "cpu", dtype: Optional[torch.dtype] = None,
             lora: Optional[LoraSpec] = None, gradient_checkpointing: bool = False) -> "JointModel":
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        root = Path(checkpoint)
        config = read_config(root)
        device = torch.device(device)
        dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
        backbone = Qwen3_5Model.from_pretrained(str(root), dtype=dtype, attn_implementation="sdpa")
        hidden = backbone.config.text_config.hidden_size
        readout = torch.nn.Linear(hidden, MAX_OPTIONS, bias=False, dtype=dtype)
        readout.load_state_dict(load_file(str(root / "readout.safetensors")))
        enable_noncausal_full_attention(backbone.language_model)
        backbone.requires_grad_(False)
        readout.requires_grad_(False)
        if mode == "full":
            backbone.language_model.requires_grad_(True)       # the vision tower is unused: no gradient, no optimizer state
            readout.requires_grad_(True)
        if gradient_checkpointing and mode != "none":
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if mode == "lora":
            from peft import LoraConfig, get_peft_model

            spec = lora or LoraSpec()
            backbone = get_peft_model(backbone, LoraConfig(r=spec.rank, lora_alpha=spec.alpha, lora_dropout=spec.dropout,
                                                           target_modules=spec.regex(), bias="none"))
            readout.requires_grad_(True)
        model = cls(backbone, readout, config, root, mode)
        model.to(device)
        model.train(mode != "none")
        if mode == "none":
            model.eval()
        return model

    # ------------------------------------------------------------------ forward
    def forward(self, batch: Batch) -> torch.Tensor:
        hidden = self.backbone(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False).last_hidden_state
        return mask_logits(self.readout(hidden[batch.rows, batch.cols]).float(), batch.counts)

    # ------------------------------------------------------------------ training helpers
    def trainable(self) -> List[Tuple[str, torch.nn.Parameter]]:
        return [(name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad]

    def merge(self) -> None:
        """Fold the LoRA adapters into the backbone weights (once, at the end of training)."""
        if self.mode == "lora":
            self.backbone = self.backbone.merge_and_unload()
            self.mode = "merged"

    # ------------------------------------------------------------------ saving
    def save(self, directory: "str | Path", *, temperature: Optional[float] = None, extra: Optional[Dict[str, Any]] = None,
             modification: str = "weights, readout and decision_config.json changed by decide.train.model") -> Path:
        """Write a checkpoint with the layout of the one this model was loaded from (never into a non-empty folder), with the
        ``LICENSE`` and the ``NOTICE`` of that one and a ``Modifications:`` line saying ``modification``."""
        if self.mode == "lora":
            raise RuntimeError("merge() the LoRA adapters before saving")
        destination = Path(directory)
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Refusing to overwrite checkpoint contents: {destination}")
        scale = self.temperature if temperature is None else float(temperature)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("temperature must be positive and finite")
        destination.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(destination), max_shard_size="5GB")
        save_file({"weight": self.readout.weight.detach().cpu().contiguous()}, str(destination / "readout.safetensors"))
        skip = NOT_COPIED | {"config.json", "model.safetensors.index.json"}
        for item in sorted(self.source.iterdir()):
            if item.is_file() and item.suffix != ".safetensors" and item.name not in skip and not (destination / item.name).exists():
                shutil.copyfile(item, destination / item.name)
        config = dict(self.config_data)
        config.update(extra or {})
        config["temperature"] = scale
        (destination / "decision_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        carry_notices(self.source, destination, modification)
        self.temperature = scale
        return destination
