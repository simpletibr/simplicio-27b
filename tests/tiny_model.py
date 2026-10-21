"""A tiny Qwen3.5 decision checkpoint for tests: random weights, the real tensor names and file layout.

Imports torch and transformers at call time only, so importing this module never fails in a bare environment;
callers are decorated with `needs_transformers`.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

HIDDEN = 64
LAYERS = 4
VOCAB = 320
TEMPERATURE = 1.3823494677922652
VENDOR_CONFIG = Path(__file__).resolve().parents[1] / "decide" / "sd" / "vendor" / "pplx_decider" / "decision_config.json"
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|fim_suffix|>", "<|pad|>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for c in m['content'] %}{% if c['type'] == 'text' %}{{ c['text'] }}{% endif %}{% endfor %}"
    "{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def config_dict() -> dict:
    return dict(
        architectures=["Qwen3_5Model"], model_type="qwen3_5", image_token_id=290, video_token_id=291,
        vision_start_token_id=292, vision_end_token_id=293, tie_word_embeddings=False,
        text_config=dict(
            model_type="qwen3_5_text", vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=LAYERS,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_conv_kernel_dim=4, linear_key_head_dim=16,
            linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention"] * (LAYERS - 1) + ["full_attention"], max_position_embeddings=2048,
            rope_parameters={"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_theta": 10000,
                             "rope_type": "default", "partial_rotary_factor": 0.25},
            partial_rotary_factor=0.25, attn_output_gate=True, full_attention_interval=LAYERS,
            mamba_ssm_dtype="float32", rms_norm_eps=1e-6, tie_word_embeddings=False),
        vision_config=dict(model_type="qwen3_5_vision", depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=HIDDEN, num_position_embeddings=64, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, deepstack_visual_indexes=[]),
    )


def decision_config(token_ids: List[int]) -> dict:
    real = json.loads(VENDOR_CONFIG.read_text())
    return {**real, "temperature": TEMPERATURE, "token_ids": token_ids}


class TinyProcessor:
    """Just what `Encoder` touches: tokenizer, chat template, and a call that returns input ids."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.image_processor = SimpleNamespace(size={})

    def apply_chat_template(self, messages, **kw):
        return self.tokenizer.apply_chat_template(messages, **kw)

    def __call__(self, text, images=None, return_tensors="pt"):
        if images:
            raise NotImplementedError("the tiny processor has no image support")
        enc = self.tokenizer(text[0], add_special_tokens=False, return_tensors=return_tensors)
        return {"input_ids": enc["input_ids"]}


def tokenizer_and_codes():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    codes = json.loads(VENDOR_CONFIG.read_text())["codes"]
    words = ["<unk>", *SPECIALS, *codes, "State", ":", "Question", "Options", "system", "user", "assistant", "yes", "no",
             "image", "0", "1", "2", "3", "4", "5"]
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(SPECIALS)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<|pad|>", eos_token="<|endoftext|>",
                                   additional_special_tokens=SPECIALS[1:4])
    fast.chat_template = CHAT_TEMPLATE
    fast.padding_side = "left"
    return fast, codes, [vocab[c] for c in codes]


def processor_factory(repo_id: str = "", revision: Optional[str] = None):
    tokenizer, _codes, _ids = tokenizer_and_codes()
    return TinyProcessor(tokenizer)


def backbone_state(seed: int) -> Dict[str, "object"]:
    """A random backbone state dict with the decision-backbone names (no 'model.' prefix), float32."""
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    torch.manual_seed(seed)
    model = Qwen3_5Model(Qwen3_5Config.from_dict(config_dict()))
    state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    for name, t in state.items():                       # make the numbers less symmetric than the init
        if t.ndim >= 2:
            t.mul_(1.0 + 0.01 * (seed % 7))
    return state


def head_weight(seed: int):
    import torch

    g = torch.Generator().manual_seed(1000 + seed)
    return torch.randn(255, HIDDEN, generator=g) * 0.2


def safetensors_blob(tensors: Dict[str, "object"]) -> bytes:
    from safetensors.torch import save

    return save({k: v.contiguous() for k, v in tensors.items()})


def publish_backbone(hub, repo: str, state: Dict[str, "object"], *, shards: int = 2, dtype=None, head: Optional["object"] = None,
                     decision: bool = False, fp32_suffixes: Tuple[str, ...] = ()) -> Tuple[Dict[str, "object"], List[str]]:
    """Put a backbone on the fake hub as `shards` safetensors files plus an index; returns (published tensors, names).

    Floating tensors are stored as `dtype` (bf16, like the real checkpoints) except names ending in `fp32_suffixes`.
    """
    import torch

    dtype = dtype or torch.bfloat16
    published = {}
    for k, v in state.items():
        keep = (not v.is_floating_point()) or k.endswith(fp32_suffixes)
        published[k] = v if keep else v.to(dtype)
    names = sorted(published)
    per = -(-len(names) // shards)
    weight_map = {}
    for i in range(shards):
        chunk = names[i * per:(i + 1) * per]
        if not chunk:
            continue
        fname = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        hub.add(repo, fname, safetensors_blob({n: published[n] for n in chunk}))
        weight_map.update({n: fname for n in chunk})
    hub.add(repo, "model.safetensors.index.json", json.dumps({"metadata": {}, "weight_map": weight_map}).encode())
    hub.add(repo, "config.json", json.dumps(config_dict()).encode())
    if decision:
        _tok, _codes, token_ids = tokenizer_and_codes()
        hub.add(repo, "decision_config.json", json.dumps(decision_config(token_ids)).encode())
        hub.add(repo, "joint_config.json", (Path(VENDOR_CONFIG).parent / "joint_config.json").read_bytes())
        hub.add(repo, "readout.safetensors", safetensors_blob({"weight": (head if head is not None else head_weight(0)).to(dtype)}))
    return published, names


def publish_lora(hub, repo: str, anchor_state: Dict[str, "object"], *, rank: int = 4, alpha: float = 4.0, seed: int = 0,
                 modules: Optional[List[str]] = None, style: str = "peft-conditional") -> Dict[str, Tuple["object", "object"]]:
    """A LoRA adapter over the MLP and attention projections; returns {anchor tensor name: (A, B)} (float32)."""
    import torch

    g = torch.Generator().manual_seed(500 + seed)
    targets = modules or [n for n in sorted(anchor_state)
                          if n.startswith("language_model.layers.") and n.endswith((".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight", ".self_attn.q_proj.weight"))]
    tensors, pairs = {}, {}
    for name in targets:
        out_f, in_f = anchor_state[name].shape
        a = torch.randn(rank, in_f, generator=g) * 0.1
        b = torch.randn(out_f, rank, generator=g) * 0.1
        module = "base_model.model.model." + name[:-len(".weight")]
        tensors[module + ".lora_A.weight"] = a
        tensors[module + ".lora_B.weight"] = b
        pairs[name] = (a, b)
    hub.add(repo, "adapter_model.safetensors", safetensors_blob(tensors))
    hub.add(repo, "adapter_config.json", json.dumps({"peft_type": "LORA", "r": rank, "lora_alpha": alpha, "use_rslora": False,
                                                     "use_dora": False, "bias": "none"}).encode())
    return pairs


REQUEST = {
    "state": "alpha beta gamma",
    "questions": {
        "route": {"type": "choice", "instructions": "Pick one", "criteria": {"book": "yes", "ask": "no", "refuse": "image", "defer": "0"}},
        "urgent": {"type": "noul", "instructions": "Is it urgent"},
    },
}


def second_request() -> dict:
    return {"state": "delta " * 40, "questions": {"pick": {"type": "choice", "criteria": {"x": "1", "y": "2", "z": "3"}}}}


def dense_decider(state, head, *, batch_tokens: int = 4096, max_length: Optional[int] = 100_000, dtype=None):
    """An AssembledDecider over a dense (unquantised) tiny backbone, built independently of loader.py."""
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    from decide.sd.engine import AssembledDecider
    from decide.sd.vendor.pplx_decider.model import enable_noncausal_full_attention

    dtype = dtype or torch.float32
    model = Qwen3_5Model._from_config(Qwen3_5Config.from_dict(config_dict()), attn_implementation="sdpa", dtype=dtype)
    model.load_state_dict({k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()}, strict=True)
    enable_noncausal_full_attention(model.language_model)
    model.requires_grad_(False).eval()
    readout = torch.nn.Linear(HIDDEN, 255, bias=False, dtype=dtype)
    readout.weight.data = head.to(dtype).clone()
    readout.requires_grad_(False).eval()
    _tok, _codes, token_ids = tokenizer_and_codes()
    return AssembledDecider(model, readout, processor_factory(), decision_config(token_ids), device="cpu",
                            max_length=max_length, batch_tokens=batch_tokens)
