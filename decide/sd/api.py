"""The interface the rest of the project calls: load a model, run requests, score them, fit a temperature.

    engine = load_engine(spec_for(cfg, "ta_l0.25"), quant="nf4", tta_rotations=2)
    info   = run_requests(engine, "requests.jsonl", "work/run", budget=budget)       # per-row probabilities, resumable
    scores = score("work/run", subset_manifest="work/sample")                        # the kit's own scorer, on a sample
    fit    = fit_temperature("work/run/results.parquet")                             # per question type, holdout by request

A `spec` names a model by recipe, never by file: the anchor checkpoint, the donors whose backbones are merged into it,
an optional LoRA adapter, the merge method.  Nothing is written to disk; the loader merges while it streams the weights
(`loader.py`).  Heads are never merged.

Importing this module does not import torch: the pure helpers (`normalize_spec`, `spec_for`, `parse_device`,
`quant_settings`) work anywhere, and `load_engine` imports the heavy parts when it is called.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from decide.sd import calibrate, env, evaluate
from decide.sd import config as C

__all__ = ["ApiError", "normalize_spec", "spec_for", "spec_names", "parse_device", "quant_settings", "load_options", "load_engine",
           "run_requests", "score", "fit_temperature"]

SPEC_KEYS = ("anchor", "anchor_revision", "donors", "lora", "method", "density", "protocol", "reference", "base", "name")
DONOR_KEYS = ("repo", "weight", "revision", "name")
QUANT_NAMES = ("nf4", "fp16", "bf16", "none")
DEVICE = re.compile(r"cuda:\d+")
SAMPLE_PLACES = ("subset", "sample")
BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


class ApiError(RuntimeError):
    """A request to this interface that cannot be served, with the reason in plain words."""


# --------------------------------------------------------------------------- the recipe of a model

def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(f"{what} must be a number, got {value!r}")
    if value != value or value in (float("inf"), float("-inf")):
        raise ApiError(f"{what} must be finite, got {value!r}")
    return float(value)


def _repo(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiError(f"{what} must be a Hugging Face repo id such as 'org/name'")
    return value.strip()


def _revision(value: Any, what: str) -> str:
    if not C.is_revision(value):
        raise ApiError(f"{what} must be a full 40-hex commit hash, got {value!r}: a branch moves, so the weights could change between two reads")
    return value


def _checked_part(raw: Any, what: str, forbid: bool) -> Optional[Dict[str, Any]]:
    """One donor or the LoRA: {repo, weight, revision}; None when its weight is zero (nothing to read)."""
    if not isinstance(raw, Mapping):
        raise ApiError(f"{what} must be an object {{repo, weight}}")
    extra = sorted(set(raw) - set(DONOR_KEYS))
    if extra:
        raise ApiError(f"{what}: unknown key(s) {extra}")
    repo = _repo(raw.get("repo"), f"{what}.repo")
    if "weight" not in raw:
        raise ApiError(f"{what} ({repo}): weight is required")
    weight = _number(raw["weight"], f"{what}.weight")
    if weight < 0:
        raise ApiError(f"{what} ({repo}): weight must be >= 0")
    tag = repo.lower()
    if forbid and any(f in tag for f in C.FORBIDDEN_DONORS):
        raise ApiError(f"donor {repo} is not allowed (license or incompatible readout; see fase0b)")
    if weight == 0:
        return None
    return {"repo": repo, "weight": weight, "revision": _revision(raw.get("revision"), f"{what} ({repo}): revision")}


def normalize_spec(spec: Any, cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The spec with every key present and every value checked; the input is left alone.

    Zero-weight donors and a zero-weight LoRA are dropped (they would be downloaded for nothing).  With no donor the
    method is `anchor`.  `reference: base` takes `base` from the spec or from `cfg.models.base`.
    """
    if not isinstance(spec, Mapping):
        raise ApiError("spec must be an object with at least 'anchor'")
    extra = sorted(set(spec) - set(SPEC_KEYS))
    if extra:
        raise ApiError(f"unknown spec key(s) {extra}; allowed: {list(SPEC_KEYS)}")
    anchor = _repo(spec.get("anchor"), "spec.anchor")
    donors_in = spec.get("donors") or []
    if not isinstance(donors_in, (list, tuple)):
        raise ApiError("spec.donors must be a list of {repo, weight}")
    donors: List[Dict[str, Any]] = []
    seen = set()
    for raw in donors_in:
        part = _checked_part(raw, "donor", True)
        repo = _repo(raw.get("repo"), "donor.repo")
        if repo in seen:
            raise ApiError(f"donor {repo} is listed twice")
        seen.add(repo)
        if part is not None:
            donors.append(part)
    lora = None
    if spec.get("lora"):
        lora = _checked_part(spec["lora"], "lora", False)
    method = spec.get("method")
    if method is not None and method not in (*C.METHODS, "anchor"):
        raise ApiError(f"spec.method must be one of {[*C.METHODS, 'anchor']}, got {method!r}")
    if not donors:
        method = "anchor"
    elif method in (None, "anchor"):
        if method == "anchor":
            raise ApiError("spec.method 'anchor' cannot merge donors; use task_arithmetic or ties")
        method = "task_arithmetic"
    density = None
    if method == "ties":
        density = 0.5 if spec.get("density") is None else _number(spec["density"], "spec.density")
        if not 0.0 < density <= 1.0:
            raise ApiError(f"spec.density must be in (0, 1], got {density}")
    protocol = spec.get("protocol") or "pplx_joint"
    if protocol not in C.PROTOCOLS:
        raise ApiError(f"spec.protocol must be one of {list(C.PROTOCOLS)}, got {protocol!r}")
    reference = spec.get("reference") or "anchor"
    if reference not in C.REFERENCES:
        raise ApiError(f"spec.reference must be one of {list(C.REFERENCES)}, got {reference!r}")
    base = None
    if reference == "base":
        given = spec.get("base")
        if given:
            base_repo = _repo(given.get("repo") if isinstance(given, Mapping) else None, "spec.base.repo")
            base = {"repo": base_repo, "revision": _revision(given.get("revision"), f"spec.base ({base_repo}): revision")}
        elif cfg is not None and (cfg.get("models") or {}).get("base"):
            entry = cfg["models"]["base"]
            base = {"repo": entry["repo"], "revision": _revision(entry.get("revision"), f"models.base ({entry['repo']}): revision")}
        else:
            raise ApiError("spec.reference is 'base' but there is no base: pass spec.base or a config with models.base")
    name = spec.get("name")
    if name is not None and not isinstance(name, str):
        raise ApiError("spec.name must be text")
    return {"anchor": anchor, "anchor_revision": _revision(spec.get("anchor_revision"), f"spec.anchor_revision ({anchor})"), "donors": donors, "lora": lora, "method": method,
            "density": density, "protocol": protocol, "reference": reference, "base": base, "name": name}


def _fixed_names(cfg: Mapping[str, Any]) -> List[str]:
    models = cfg["models"]
    names = ["d3"] if models.get("d3") else []
    return [f"rehearsal:{key}" for key in (models.get("rehearsal") or {})] + names


def spec_names(cfg: Mapping[str, Any]) -> List[str]:
    """Every name `spec_for` accepts for this config: the candidates, then the rehearsal and fidelity models."""
    try:
        candidates = [c.name for c in C.enumerate_candidates(cfg)]
    except C.ConfigError as exc:
        raise ApiError(str(exc)) from exc
    return candidates + _fixed_names(cfg)


def _lone(cfg: Mapping[str, Any], entry: Mapping[str, Any], name: str) -> Dict[str, Any]:
    return normalize_spec({"anchor": entry["repo"], "anchor_revision": entry.get("revision"),
                           "protocol": entry.get("protocol", "pplx_joint"), "name": name}, cfg)


def spec_for(cfg: Mapping[str, Any], name: str = "anchor") -> Dict[str, Any]:
    """The spec of a named model of the config.

    `anchor`; a merge candidate (`ta_l0.25`, `ties_l0.5_d0.5`, ...); the LoRA on top of either (`anchor+lora0.1`,
    `ta_l0.25+lora0.3`); `rehearsal:pplx_flash` / `rehearsal:d3_mini` (the 4B models); `d3` (the 27B d3 with its own
    protocol, used for the fidelity check).
    """
    models = cfg["models"]
    if name.startswith("rehearsal:"):
        entry = (models.get("rehearsal") or {}).get(name.split(":", 1)[1])
        if entry:
            return _lone(cfg, entry, name)
    elif name == "d3" and models.get("d3"):
        return _lone(cfg, models["d3"], name)
    try:
        candidates = {c.name: c for c in C.enumerate_candidates(cfg)}
    except C.ConfigError as exc:
        raise ApiError(str(exc)) from exc
    candidate = candidates.get(name)
    if candidate is None and "+lora" in name:
        lora_cfg = models.get("lora") or {}
        if not lora_cfg.get("repo") or lora_cfg.get("enabled", True) is False:
            raise ApiError(f"{name}: the LoRA is disabled in the config (models.lora)")
        stem, _, weight = name.rpartition("+lora")
        try:
            number = float(weight)
        except ValueError:
            number = None
        base = candidates.get(stem)
        if base is not None and number is not None and number > 0 and not base.lora_weight:
            candidate = C.with_lora(base, number)
    if candidate is None:
        raise ApiError(f"unknown model name {name!r}; known: {', '.join(spec_names(cfg))} (a LoRA weight can follow any candidate: 'ta_l0.25+lora0.3')")
    anchor = models["anchor"]
    by_name = {d["name"]: d for d in models.get("donors", [])}
    donors = [{"repo": by_name[n]["repo"], "weight": w, "revision": by_name[n].get("revision")} for n, w in candidate.donors]
    lora = None
    if candidate.lora_weight > 0:
        lora = {"repo": models["lora"]["repo"], "weight": candidate.lora_weight, "revision": models["lora"].get("revision")}
    return normalize_spec({"anchor": anchor["repo"], "anchor_revision": anchor.get("revision"), "donors": donors, "lora": lora,
                           "method": None if candidate.method == "anchor" else candidate.method,
                           "density": candidate.density if candidate.method == "ties" else None,
                           "protocol": anchor.get("protocol", "pplx_joint"), "reference": candidate.reference, "name": name}, cfg)


# --------------------------------------------------------------------------- devices and precision

def parse_device(device: Any) -> Union[str, Tuple[str, ...]]:
    """`auto`, or a tuple of devices from 'cpu', 'cuda', 'cuda:N' (text separated by commas, or a list)."""
    if isinstance(device, str):
        text = device.strip()
        if text == "auto":
            return "auto"
        parts = [p.strip() for p in text.split(",")]
    elif isinstance(device, (list, tuple)) and device:
        parts = [str(p).strip() for p in device]
    else:
        raise ApiError(f"device must be 'auto', 'cpu', 'cuda', 'cuda:N' or a list of them, got {device!r}")
    out: List[str] = []
    for part in parts:
        if part == "cpu":
            out.append("cpu")
        elif part == "cuda":
            out.append("cuda:0")
        elif DEVICE.fullmatch(part):
            out.append(part)
        else:
            raise ApiError(f"device {part!r}: use 'auto', 'cpu', 'cuda', 'cuda:N' or a comma-separated list")
    return tuple(out)


def quant_settings(quant: str, default_compute: str) -> Tuple[str, str]:
    """(loader quant, compute dtype) for a quant name: nf4 | fp16 | bf16 | none (unquantised, in the config's compute dtype)."""
    if quant == "nf4":
        return "nf4", default_compute
    if quant == "fp16":
        return "none", "float16"
    if quant == "bf16":
        return "none", "bfloat16"
    if quant == "none":
        return "none", default_compute
    raise ApiError(f"quant must be one of {list(QUANT_NAMES)}, got {quant!r}")


def load_options(cfg: Mapping[str, Any], *, quant: Optional[str] = None, device: Any = "auto", hardware: Optional[env.Hardware] = None,
                 allow_cpu_offload: bool = False, model_gb: Optional[float] = None, **overrides: Any) -> Any:
    """`LoadOptions` for this machine.  `device='auto'` asks `env` how to lay the model out on the GPUs that are present.

    A dense (unquantised) model on GPUs needs `model_gb`, its size in memory.  Any `LoadOptions` field can be overridden.
    """
    from decide.sd.loader import LoadOptions

    e = cfg["eval"]
    loader_quant, compute = quant_settings(quant or e["quant"], e["compute_dtype"])
    unknown = sorted(set(overrides) - set(LoadOptions.__dataclass_fields__))
    if unknown:
        raise ApiError(f"unknown load option(s) {unknown}; see LoadOptions")
    fields: Dict[str, Any] = {"quant": loader_quant, "compute_dtype": compute}
    devices = parse_device(device)
    if devices == "auto":
        hw = hardware if hardware is not None else env.detect_hardware()
        if hw.gpu_count == 0:
            devices = ("cpu",)
        else:
            if loader_quant == "nf4":
                plan = env.plan_devices(hw.gpu_mem_gb, "nf4", allow_cpu_offload, e.get("max_gpu_gb"))
            elif model_gb is None:
                raise ApiError("laying a dense model out on GPUs needs model_gb (its size in memory); load_engine measures it")
            else:
                plan = env.plan_dense_devices(hw.gpu_mem_gb, model_gb, e.get("max_gpu_gb"))
            if not plan.ok:
                raise ApiError(" ".join(plan.notes))
            order = sorted(plan.max_memory_gb)
            devices = tuple(f"cuda:{i}" for i in order)
            fields["capacities"] = tuple(plan.max_memory_gb[i] for i in order)
            if plan.embeddings_device == "cpu":
                fields["embeddings_device"] = "cpu"
            fields["offload_layers"] = plan.cpu_offload_layers
    fields["devices"] = devices
    if all(d == "cpu" for d in devices) and fields["compute_dtype"] == "float16":
        fields["compute_dtype"] = "float32"                      # half precision is not a CPU dtype worth the trouble
    return LoadOptions.from_config(cfg, **{**fields, **overrides})


# --------------------------------------------------------------------------- loading

def attach_fast(decider: Any, *, fp8: bool = False) -> None:
    """Attach the packed fast path of pplx-decider (CUDA, flash-linear-attention).  UNVERIFIED on a GPU here."""
    try:
        from decide.sd.vendor.pplx_decider.fast import FastPath
        from decide.sd.vendor.pplx_decider.model import FastOptions

        decider.fast = FastPath(decider, FastOptions(fp8=fp8))
    except (ValueError, ImportError, RuntimeError) as exc:
        raise ApiError(f"the fast path could not start: {exc}") from exc


def load_engine(spec: Mapping[str, Any], *, quant: Optional[str] = None, device: Any = "auto", tta_rotations: int = 1,
                tta_mode: Optional[str] = None, temperature: Optional[float] = None, temperatures: Optional[Mapping[str, float]] = None,
                cfg: Optional[Mapping[str, Any]] = None, hardware: Optional[env.Hardware] = None, allow_cpu_offload: bool = False,
                name: Optional[str] = None, token: Optional[str] = None, progress: Optional[Callable[[int, int, str], None]] = None,
                question_types: Optional[Sequence[str]] = None, fast: bool = False, fp8: bool = False,
                factories: Optional[Mapping[str, Callable[..., Any]]] = None, **overrides: Any) -> Any:
    """A kit-compatible engine (`engine(state, questions) -> (response, raw)`, plus `batch`) for the model `spec` describes.

    `quant` is nf4 | fp16 | bf16 | none (default: the config's `eval.quant`); `tta_rotations` is the number of option orders
    averaged per choice question; `temperatures` maps a question type (or `type:k-bucket`, or `default`) to the divisor
    used instead of the model's own.  `fast`/`fp8` attach pplx-decider's packed path: `fp8` changes the probabilities
    (the evaluated model is no longer the published one) and is recorded as such in the engine's provenance.
    `factories` (repo, processor, tokenizer) are test hooks.  Other keyword arguments override `LoadOptions` fields.
    """
    from decide.sd import engine as engine_mod
    from decide.sd import loader
    from decide.sd.remote import HubRepo

    cfg = cfg if cfg is not None else C.load_config()
    spec = normalize_spec(spec, cfg)
    if fp8 and not fast:
        raise ApiError("fp8 is a mode of the fast path; pass fast=True as well (and read the warning: it changes the model)")
    if fast and spec["protocol"] != "pplx_joint":
        raise ApiError("the fast path serves the pplx joint protocol only")
    factories = dict(factories or {})
    secret = token or env.get_secret("HF_TOKEN")
    endpoint = cfg["hub"].get("endpoint", "https://huggingface.co")
    make_repo = factories.get("repo") or (lambda repo, revision=None: HubRepo(repo, revision, endpoint=endpoint, token=secret))
    quant_name = quant or cfg["eval"]["quant"]
    loader_quant, compute = quant_settings(quant_name, cfg["eval"]["compute_dtype"])
    model_gb = None
    if loader_quant == "none" and parse_device(device) == "auto" and (hardware or env.detect_hardware()).gpu_count:
        stored = loader.stored_bytes(make_repo(spec["anchor"], spec["anchor_revision"]))
        model_gb = stored * BYTES.get(compute, 2) / 2 / 1024 ** 3
    options = load_options(cfg, quant=quant_name, device=device, hardware=hardware, allow_cpu_offload=allow_cpu_offload,
                           model_gb=model_gb, **overrides)
    if fast and all(d == "cpu" for d in options.devices):
        raise ApiError("the fast path needs a CUDA device")
    decider = loader.load_decider(spec, options, repo_factory=make_repo, processor_factory=factories.get("processor"),
                                  tokenizer_factory=factories.get("tokenizer"), progress=progress, token=secret)
    if fast:
        attach_fast(decider, fp8=fp8)
    provenance = loader.describe(spec, options)
    provenance["fast"] = {"enabled": bool(fast), "fp8": bool(fp8), "changes_the_model": bool(fp8)}
    types = tuple(question_types) if question_types else engine_mod.SUPPORTED_TYPES
    return engine_mod.DecisionEngine(decider, tta_rotations=tta_rotations, tta_mode=tta_mode or cfg["tta"].get("mode", "reverse"),
                                     temperature=temperature, temperatures=temperatures, provenance=provenance,
                                     name=name or spec.get("name") or "simplicio-decide", question_types=types)


# --------------------------------------------------------------------------- running, scoring, calibrating

def run_requests(engine: Any, requests_path: Union[str, Path], out_path: Union[str, Path], *, budget: Any = None, resume: bool = True,
                 **options: Any) -> Dict[str, Any]:
    """Answer every request of a file that has no final result yet; per-row results, raw logits and a probability table follow.

    Returns at least `{"rows": new rows, "tokens": input tokens of the new rows, "seconds": wall time, "status": ...}`.
    `out_path` is a directory (or a `.parquet` path, whose folder then holds the run).  See `evaluate.run_requests`.
    """
    return evaluate.run_requests(engine, requests_path, out_path, budget=budget, resume=resume, **options)


def _results_file(results_path: Union[str, Path]) -> Path:
    p = Path(results_path)
    p = p / evaluate.RESULTS_NAME if p.is_dir() else p
    if not p.exists():
        raise ApiError(f"{p}: no results file here")
    return p


def _find_sample(results: Path) -> Path:
    for base in (results.parent, results.parent.parent):
        for place in (base, *(base / name for name in SAMPLE_PLACES)):
            if (place / evaluate.sample_mod.MANIFEST_NAME).exists():
                return place
    raise ApiError(f"{results}: cannot tell which sample these results answer; pass subset_manifest=<sample directory or its manifest>")


def score(results_path: Union[str, Path], *, subset_manifest: Union[str, Path, None] = None, edition: Optional[str] = None,
          engine_name: Optional[str] = None) -> Dict[str, Any]:
    """The Decision Index of a run over its sample, by the kit's own scorer: `{"public", "areas", "per_benchmark", ...}`.

    `subset_manifest` is the sample directory (or its manifest file); without it the sample next to the run is used.
    The number is a sample index: a partial suite, not the board's score.
    """
    results = _results_file(results_path)
    sample = Path(subset_manifest) if subset_manifest else _find_sample(results)
    try:
        return evaluate.score_sample(results, sample, edition=edition, engine=engine_name or "simplicio-decide")
    except evaluate.EvaluateError as exc:
        raise ApiError(str(exc)) from exc


def fit_temperature(results_path: Union[str, Path], *, by: str = "question_type", holdout: float = 0.5, seed: Optional[int] = None,
                    grid: Optional[Sequence[float]] = None, min_items: int = 30, baseline: Optional[float] = None,
                    cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """One temperature per question group, fitted by NLL on half of the requests and judged on the other half.

    Reads the probability table of a run (the `.parquet` file, or its folder): it needs the raw logits and the gold labels.
    `temperatures` in the result can be passed to `load_engine(temperatures=...)`; `holdout` says whether it helped.
    """
    if by not in calibrate.GROUPINGS:
        raise ApiError(f"by must be one of {list(calibrate.GROUPINGS)}, got {by!r}")
    table = Path(results_path)
    table = table / evaluate.TABLE_NAME if table.is_dir() else table
    if not table.exists():
        raise ApiError(f"{table}: no probability table here; run_requests writes it when a run completes")
    cfg = cfg if cfg is not None else C.load_config()
    try:
        items = evaluate.table_items(table)
        own = evaluate.table_temperature(table)
    except evaluate.EvaluateError as exc:
        raise ApiError(str(exc)) from exc
    if not items:
        raise ApiError(f"{table}: no question has a gold label, so there is nothing to fit on")
    window = tuple(float(x) for x in (grid if grid is not None else cfg["calibration"].get("grid", calibrate.DEFAULT_GRID)))
    out = calibrate.fit_groups(items, by=by, holdout=holdout, seed=int(cfg["eval"].get("seed", 0)) if seed is None else int(seed),
                               baseline=float(baseline if baseline is not None else (own or 1.0)), grid=window, min_items=min_items)
    out["source"] = str(table)
    out["n_items"] = len(items)
    return out
