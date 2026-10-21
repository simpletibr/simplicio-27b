"""Merge-on-load: stream the checkpoints, merge tensor by tensor, quantise to NF4, assemble the decider.

Nothing here writes the merged model to disk.  For every tensor of the anchor (pplx-decider-v1.2) the loader reads
the same tensor from each donor and, when asked, the Simplicio LoRA factors; merges them in float32 (`merge.merge_one`);
and puts the result straight onto its device: matmul weights become `Nf4Linear` buffers, everything else is cast to
the compute dtype.  Peak memory is the model itself plus a few tensors in flight, so a 27B backbone loads on a 16 GB
GPU pair or, with `offload_layers`, on a single one.

Heads are never merged: the readout comes from the anchor (or from a refit file).  Vision tensors are skipped unless
asked for.  See fase0b for why only backbones are merged.
"""
from __future__ import annotations

import json
import re
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from decide.sd import d3 as d3_protocol
from decide.sd import merge as M
from decide.sd import nf4
from decide.sd.engine import AssembledDecider
from decide.sd.remote import HubRepo, RemoteError
from decide.sd.vendor.pplx_decider.model import enable_noncausal_full_attention
from decide.sd.vendor.pplx_decider.protocol import PROTOCOL

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
PROTOCOLS = ("pplx_joint", "d3")
LAYER = re.compile(r"^language_model\.layers\.(\d+)\.")


class LoaderError(RuntimeError):
    pass


@dataclass
class LoadOptions:
    quant: str = "nf4"                                     # nf4 | none
    compute_dtype: str = "float16"
    absmax_dtype: str = "float16"                          # NF4 block statistics (float32 = bitsandbytes bytes)
    devices: Sequence[str] = ("cpu",)                      # layers are split over these, in proportion to `capacities`
    capacities: Optional[Sequence[float]] = None           # usable GiB per device (defaults to equal)
    embeddings_device: Optional[str] = None                # defaults to devices[0]; "cpu" keeps the 2.4 GiB table off the GPU
    offload_layers: int = 0                                # last N layers live on the CPU and visit the last device per pass
    merge_device: Optional[str] = None
    load_vision: bool = False
    skip_merge: Sequence[str] = ("embed_tokens", "visual.")
    nf4_skip: Sequence[str] = ("in_proj_a", "in_proj_b")   # tiny gating projections stay dense (see config.yaml)
    workers: int = 3
    window_mb: int = 256
    batch_tokens: int = 4096
    max_length: Optional[int] = None
    head_path: Optional[str] = None                        # safetensors with a refit readout ("weight": [255, hidden])

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], **overrides: Any) -> "LoadOptions":
        e, m = cfg["eval"], cfg["merge"]
        base = dict(quant="none" if e["quant"] in ("bf16", "fp16", "none") else e["quant"],
                    compute_dtype={"bf16": "bfloat16", "fp16": "float16"}.get(e["quant"], e["compute_dtype"]),
                    skip_merge=tuple(m.get("skip_patterns", ["embed_tokens", "visual."])),
                    batch_tokens=int(e.get("batch_tokens", 4096)), window_mb=int(m.get("chunk_mb", 256)),
                    nf4_skip=tuple(e.get("nf4_skip", ["in_proj_a", "in_proj_b"])),
                    absmax_dtype=e.get("absmax_dtype", "float16"))
        base.update(overrides)
        return cls(**base)


# --------------------------------------------------------------------------- planning

def split_layers(n_layers: int, capacities: Sequence[float]) -> List[int]:
    """Device index of each of `n_layers` consecutive layers, proportional to `capacities` (largest remainder)."""
    caps = [max(0.0, float(c)) for c in capacities]
    if not caps or sum(caps) <= 0:
        raise LoaderError("no device capacity to place the layers on")
    exact = [n_layers * c / sum(caps) for c in caps]
    counts = [int(x) for x in exact]
    for i in sorted(range(len(caps)), key=lambda j: exact[j] - counts[j], reverse=True)[: n_layers - sum(counts)]:
        counts[i] += 1
    out: List[int] = []
    for device, count in enumerate(counts):
        out += [device] * count
    return out


@dataclass
class TensorPlan:
    name: str
    donors: List[int] = field(default_factory=list)       # indices of donors that hold this tensor
    lora: Optional[str] = None                             # LoRA module key
    merged: bool = False


def plan_tensors(anchor_names: Sequence[str], donor_names: Sequence[Sequence[str]], lora_modules: Mapping[str, Any],
                 skip_merge: Sequence[str], load_vision: bool) -> List[TensorPlan]:
    mapper = M.NameMapper(anchor_names)
    lora_target = {}
    for module in lora_modules:
        target = mapper.to_anchor(module)
        if target is not None:
            lora_target[target] = module
    donor_sets = [set(names) for names in donor_names]
    plans: List[TensorPlan] = []
    for name in anchor_names:
        if name.startswith("visual.") and not load_vision:
            continue
        if any(p in name for p in skip_merge):
            plans.append(TensorPlan(name))
            continue
        donors = [i for i, names in enumerate(donor_sets) if name in names]
        lora = lora_target.get(name)
        plans.append(TensorPlan(name, donors, lora, merged=bool(donors or lora)))
    return plans


class LoraAdapter:
    """The Simplicio adapter: A/B factors per module, read once (about 640 MB in float32 for r=32)."""

    def __init__(self, repo: HubRepo, weight: float) -> None:
        self.repo, self.weight = repo, float(weight)
        cfg = repo.read_json("adapter_config.json")
        if cfg.get("use_dora"):
            raise LoaderError("DoRA adapters are not supported")
        self.r = int(cfg["r"])
        self.alpha = float(cfg["lora_alpha"])
        self.rslora = bool(cfg.get("use_rslora", False))
        self.file = repo.shard("adapter_model.safetensors")
        self.pairs = M.pair_lora_keys(self.file.names())

    def modules(self) -> List[str]:
        return list(self.pairs)

    def factors(self, module: str) -> Tuple[torch.Tensor, torch.Tensor]:
        a_key, b_key = self.pairs[module]
        return self.file.read_tensor(a_key).float(), self.file.read_tensor(b_key).float()


# --------------------------------------------------------------------------- model construction

class NoVision(nn.Module):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the vision tower was not loaded (LoadOptions.load_vision=False)")


class DeviceEmbedding(nn.Module):
    """Embedding table that may live on another device than the layers (e.g. the CPU)."""

    def __init__(self, table: nn.Embedding, out_device: str) -> None:
        super().__init__()
        self.table, self.out_device = table, torch.device(out_device)

    @property
    def weight(self) -> torch.Tensor:
        return self.table.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.table(ids.to(self.table.weight.device)).to(self.out_device)


def _move(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    return obj


def _mover(device: torch.device):
    def pre(module: nn.Module, args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        return _move(args, device), _move(kwargs, device)
    return pre


def _offload_hooks(layer: nn.Module, run_device: torch.device) -> None:
    """Keep a decoder layer on the CPU and visit `run_device` for the duration of its forward pass."""
    layer.register_forward_pre_hook(lambda m, a, kw: (m.to(run_device), (_move(a, run_device), _move(kw, run_device)))[1], with_kwargs=True)
    layer.register_forward_hook(lambda m, a, kw, out: (m.to("cpu"), out)[1], with_kwargs=True)


def _empty_model(config: Mapping[str, Any]):
    from accelerate import init_empty_weights
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    cfg = Qwen3_5Config.from_dict(dict(config))
    with init_empty_weights():
        return Qwen3_5Model._from_config(cfg, attn_implementation="sdpa", dtype=torch.float32)


def _submodule(model: nn.Module, path: str) -> nn.Module:
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _set_tensor(model: nn.Module, name: str, tensor: torch.Tensor) -> None:
    path, _, attr = name.rpartition(".")
    module = _submodule(model, path) if path else model
    if attr in module._parameters:
        module._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
    elif attr in module._buffers:
        module._buffers[attr] = tensor
    else:
        raise LoaderError(f"{name}: the model has no such parameter")


def _linears(model: nn.Module, skip: Sequence[str]) -> Dict[str, nn.Linear]:
    out = {}
    for name, module in model.language_model.named_modules():
        if isinstance(module, nn.Linear) and not any(s in name for s in skip):
            out["language_model." + name] = module
    return out


def _swap(model: nn.Module, path: str, new: nn.Module) -> None:
    parent_path, _, attr = path.rpartition(".")
    setattr(_submodule(model, parent_path), attr, new)


# --------------------------------------------------------------------------- the loader

def _prefetch(names: Sequence[str], fetch: Callable[[str], Any], workers: int) -> Any:
    """Yield (name, fetch(name)) in order while up to 2*workers fetches run ahead in threads."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        queue: Deque[Tuple[str, Future]] = deque()
        it = iter(names)
        try:
            for _ in range(2 * max(1, workers)):
                nxt = next(it, None)
                if nxt is None:
                    break
                queue.append((nxt, pool.submit(fetch, nxt)))
            while queue:
                name, future = queue.popleft()
                value = future.result()
                nxt = next(it, None)
                if nxt is not None:
                    queue.append((nxt, pool.submit(fetch, nxt)))
                yield name, value
        finally:
            for _name, future in queue:
                future.cancel()


def load_decider(spec: Mapping[str, Any], options: Optional[LoadOptions] = None, *,
                 repo_factory: Optional[Callable[[str, Optional[str]], HubRepo]] = None,
                 processor_factory: Optional[Callable[[str, Optional[str]], Any]] = None,
                 tokenizer_factory: Optional[Callable[[str, Optional[str]], Any]] = None,
                 progress: Optional[Callable[[int, int, str], None]] = None, token: Optional[str] = None) -> AssembledDecider:
    """Build an `AssembledDecider` for `spec` = {anchor, donors[{repo, weight}], lora{repo, weight}|None, method, density, ...}.

    `spec["protocol"]` is `pplx_joint` (default: the joint prompt of pplx-decider, the protocol of the model we ship) or
    `d3` (one question per sequence, last-token readout; used to check fidelity against the published d3 results).
    """
    opt = options or LoadOptions()
    if opt.quant not in ("nf4", "none"):
        raise LoaderError("LoadOptions.quant must be 'nf4' or 'none'")
    cdtype = DTYPES[opt.compute_dtype]
    absmax_dtype = DTYPES[opt.absmax_dtype]
    make_repo = repo_factory or (lambda repo, rev: HubRepo(repo, rev, token=token))
    anchor = make_repo(spec["anchor"], spec.get("anchor_revision"))
    donors = [make_repo(d["repo"], d.get("revision")) for d in spec.get("donors", [])]
    weights = [float(d["weight"]) for d in spec.get("donors", [])]
    lora_spec = spec.get("lora")
    lora = LoraAdapter(make_repo(lora_spec["repo"], lora_spec.get("revision")), lora_spec["weight"]) if lora_spec and float(lora_spec["weight"]) else None
    method = spec.get("method") or ("task_arithmetic" if donors else "anchor")
    if not donors:
        method = "anchor"
    density = float(spec.get("density") or 0.5)
    base_repo = make_repo(spec["base"]["repo"], spec["base"].get("revision")) if spec.get("reference") == "base" and spec.get("base") else None

    # -- small files of the anchor: protocol, calibration, head
    protocol = spec.get("protocol", "pplx_joint")
    if protocol not in PROTOCOLS:
        raise LoaderError(f"spec.protocol must be one of {PROTOCOLS}, got {protocol!r}")
    decision_config = anchor.read_json("decision_config.json")
    if protocol == "pplx_joint":
        try:
            joint = anchor.read_json("joint_config.json")
        except RemoteError as exc:
            raise LoaderError(f"{spec['anchor']} has no joint_config.json ({exc}); it is not a pplx joint checkpoint. "
                              "A d3 checkpoint needs spec protocol 'd3'.") from exc
        if joint != PROTOCOL:
            raise LoaderError("the anchor does not declare the joint inline-readout protocol (version 3)")
        if decision_config.get("attention_mode") != "noncausal_full_attention" or decision_config.get("pooling", "last") != "last":
            raise LoaderError("expected a noncausal checkpoint with per-position readouts")
    else:
        try:
            d3_protocol.check_decision_config(decision_config)
        except ValueError as exc:
            raise LoaderError(str(exc)) from exc
    config = anchor.read_json("config.json")

    # -- what to read
    anchor_names = anchor.file_order(anchor.names())
    plans = plan_tensors(anchor_names, [d.names() for d in donors], lora.pairs if lora else {}, opt.skip_merge, opt.load_vision)
    unmatched = [m for m in (lora.modules() if lora else []) if M.NameMapper(anchor_names).to_anchor(m) is None]
    if unmatched:
        raise LoaderError(f"{len(unmatched)} LoRA modules match no anchor tensor, first: {unmatched[0]}")

    # -- devices
    devices = [torch.device(d) for d in opt.devices]
    n_layers = int(config["text_config"]["num_hidden_layers"])
    caps = list(opt.capacities) if opt.capacities else [1.0] * len(devices)
    if opt.embeddings_device in (None, str(devices[0])) and len(devices) > 1 and not opt.capacities:
        caps[0] = 0.85
    layer_device = [devices[i] for i in split_layers(n_layers, caps)]
    offloaded = set(range(n_layers - opt.offload_layers, n_layers)) if opt.offload_layers else set()
    embed_device = torch.device(opt.embeddings_device) if opt.embeddings_device else devices[0]
    first, last = devices[0], devices[-1]
    merge_device = torch.device(opt.merge_device) if opt.merge_device else (first if first.type == "cuda" else torch.device("cpu"))

    def device_of(name: str) -> torch.device:
        m = LAYER.match(name)
        if m:
            i = int(m.group(1))
            return torch.device("cpu") if i in offloaded else layer_device[i]
        if "embed_tokens" in name:
            return embed_device
        if name.startswith("language_model.norm"):
            return last
        return first

    # -- skeleton, with NF4 linears installed where the weights will land
    model = _empty_model(config)
    nf4_modules: Dict[str, nf4.Nf4Linear] = {}
    if opt.quant == "nf4":
        for path, lin in _linears(model, opt.nf4_skip).items():
            dev = device_of(path + ".weight")
            layer = nf4.Nf4Linear(lin.in_features, lin.out_features, None, compute_dtype=cdtype, absmax_dtype=absmax_dtype, device=dev)
            _swap(model, path, layer)
            nf4_modules[path + ".weight"] = layer
    if not opt.load_vision:
        model.visual = NoVision()

    # -- stream, merge, place
    by_name = {p.name: p for p in plans}
    names = [p.name for p in plans]

    def fetch(name: str) -> Dict[str, Any]:
        plan = by_name[name]
        got: Dict[str, Any] = {"anchor": anchor.read_tensor(name)}
        got["donors"] = [donors[i].read_tensor(name) for i in plan.donors]
        if plan.lora is not None and lora is not None:
            got["lora"] = lora.factors(plan.lora)
        if plan.merged and base_repo is not None:
            got["base"] = base_repo.read_tensor(M.NameMapper(anchor_names).to_base(name))
        return got

    for done, (name, got) in enumerate(_prefetch(names, fetch, opt.workers), start=1):
        plan = by_name[name]
        a: torch.Tensor = got["anchor"]
        if plan.merged:
            for d in got["donors"]:
                if d.shape != a.shape:
                    raise LoaderError(f"{name}: donor shape {tuple(d.shape)} differs from the anchor's {tuple(a.shape)}")
            lora_arg = None
            if "lora" in got and lora is not None:
                lora_arg = (got["lora"][0].to(merge_device), got["lora"][1].to(merge_device), lora.alpha, lora.r, lora.weight, lora.rslora)
            tensor = M.merge_one(
                a.to(merge_device), [d.to(merge_device) for d in got["donors"]], [weights[i] for i in plan.donors], method,
                base=got["base"].to(merge_device) if "base" in got else None, density=density, lora=lora_arg)
        else:
            tensor = a
        target = device_of(name)
        if name in nf4_modules:
            nf4_modules[name].set_weight(tensor.to(target))
        else:
            keep32 = a.dtype == torch.float32 and a.ndim == 1
            _set_tensor(model, name, tensor.to(torch.float32 if keep32 else cdtype).to(target))
        if progress is not None:
            progress(done, len(names), name)

    # -- everything must have landed
    missing = [n for n, p in model.named_parameters() if p.is_meta and (opt.load_vision or not n.startswith("visual."))]
    if missing:
        raise LoaderError(f"{len(missing)} parameters were not in the checkpoint, first: {missing[0]}")

    _place(model, layer_device, offloaded, embed_device, first, last)
    enable_noncausal_full_attention(model.language_model)
    model.requires_grad_(False).eval()

    head_file = opt.head_path
    if head_file:
        from safetensors.torch import load_file

        head = load_file(head_file)["weight"]
    else:
        head = anchor.shard("readout.safetensors").read_tensor("weight")
    if protocol == "d3":
        readout = d3_protocol.CastReadout(head).eval().to(first)
        if tokenizer_factory is not None:
            tokenizer = tokenizer_factory(spec["anchor"], spec.get("anchor_revision"))
        elif processor_factory is not None:
            tokenizer = processor_factory(spec["anchor"], spec.get("anchor_revision")).tokenizer
        else:
            tokenizer = _auto_tokenizer(token)(spec["anchor"], spec.get("anchor_revision"))
        tokenizer.padding_side = "left"
        return AssembledDecider(model, readout, d3_protocol.TokenizerOnly(tokenizer), decision_config, device=str(first),
                                max_length=opt.max_length or None, batch_tokens=opt.batch_tokens,
                                encoder=d3_protocol.D3Encoder(tokenizer, decision_config["codes"]))
    readout = nn.Linear(head.shape[1], head.shape[0], bias=False, dtype=cdtype)
    readout.weight.data = head.to(cdtype)
    readout.requires_grad_(False).eval().to(first)

    processor = (processor_factory or _auto_processor(token))(spec["anchor"], spec.get("anchor_revision"))
    return AssembledDecider(model, readout, processor, decision_config, device=str(first),
                            max_length=opt.max_length if opt.max_length else 262_144, batch_tokens=opt.batch_tokens)


def _auto_tokenizer(token: Optional[str]) -> Callable[[str, Optional[str]], Any]:
    def make(repo: str, revision: Optional[str]) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(repo, revision=revision, token=token)
    return make


def _auto_processor(token: Optional[str]) -> Callable[[str, Optional[str]], Any]:
    def make(repo: str, revision: Optional[str]) -> Any:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(repo, revision=revision, token=token)
    return make


def _place(model: nn.Module, layer_device: Sequence[torch.device], offloaded: set, embed_device: torch.device,
           first: torch.device, last: torch.device) -> None:
    """Hooks that carry activations from one device to the next, and bring the output back to the first device."""
    lm = model.language_model
    lm.rotary_emb.to(first)
    lm.embed_tokens = DeviceEmbedding(lm.embed_tokens, str(first))
    for i, layer in enumerate(lm.layers):
        if i in offloaded:
            _offload_hooks(layer, last)
        elif layer_device[i] != first or len({str(d) for d in layer_device}) > 1:
            layer.register_forward_pre_hook(_mover(layer_device[i]), with_kwargs=True)
    lm.norm.register_forward_pre_hook(_mover(last), with_kwargs=True)

    def bring_back(module: nn.Module, args: tuple, kwargs: dict, output: Any) -> Any:
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = output.last_hidden_state.to(first)
        return output

    lm.register_forward_hook(bring_back, with_kwargs=True)


def describe(spec: Mapping[str, Any], options: LoadOptions) -> Dict[str, Any]:
    """The recipe as plain JSON, for provenance in results and model cards."""
    return json.loads(json.dumps({"spec": spec, "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in options.__dict__.items()}}, default=str))
