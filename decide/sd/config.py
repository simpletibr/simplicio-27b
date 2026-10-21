"""decide/config.yaml loading (with decide/config.*.yaml overlays), validation and candidate enumeration."""
from __future__ import annotations

import copy
import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

from decide.sd.remote import endpoint_problem, is_local

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
OVERLAY_GLOB = "config.*.yaml"          # decide/config.data.yaml, decide/config.rental.yaml, ... (Squad B)
METHODS = ("task_arithmetic", "ties")
REFERENCES = ("anchor", "base")
QUANTS = ("nf4", "fp16", "bf16", "none")
TTA_MODES = ("reverse", "rotate")
PROTOCOLS = ("pplx_joint", "d3")        # pplx_joint: the joint prompt we ship; d3: one question per sequence (fidelity only)
# fase0b: Lychee is CC BY-NC (cannot ship), amos has no compatible readout head; neither may be a donor.
FORBIDDEN_DONORS = ("lychee", "amos")
SESSION_HOURS_MAX = 12.0               # a Kaggle session ends at 12 h; a longer plan could never finish

# Every key the kit reads, by section.  A dict is a mapping with these keys; None is a value; {"*": ...} is a mapping of any names;
# [x] is a list of x.  A key that is not here is an error (a typo would otherwise run with the default and say nothing).
_MODEL = {"repo": None, "revision": None, "protocol": None}
SCHEMA: Dict[str, Any] = {
    "hub": {"user": None, "state_dataset": None, "winner_repo": None, "endpoint": None},
    "models": {"base": {"repo": None, "revision": None}, "anchor": {"name": None, **_MODEL},
               "donors": [{"name": None, "repo": None, "weight": None, "enabled": None, "revision": None}],
               "lora": {"repo": None, "revision": None, "weight": None, "weight_grid": None, "enabled": None},
               "rehearsal": {"*": _MODEL}, "d3": _MODEL},
    "merge": {"method": None, "reference": None, "lambda_grid": None, "max_candidates": None, "ties": {"density": None},
              "head": None, "skip_patterns": None, "chunk_mb": None},
    "eval": {"per_benchmark": None, "sweep_per_benchmark": None, "seed": None, "edition": None, "max_proxy_tokens": None, "quant": None,
             "compute_dtype": None, "max_gpu_gb": None, "batch_tokens": None, "min_per_benchmark": None, "nf4_skip": None, "absmax_dtype": None},
    "budget": {"session_hours": None, "reserve_minutes": None, "weekly_gpu_hours": None},
    "calibration": {"by": None, "holdout": None, "grid": None},
    "tta": {"rotations": None, "mode": None},
    "fidelity": {"min_rows": None, "seed": None, "published": {"*": {"dataset": None, "revision": None, "run": None}}},
    "gates": {"g0_agree_4b": None, "g0_agree_27b_nf4": None, "g2_min_delta": None, "g3_min_delta": None, "calib_min_delta": None},
    "modal": {"gpu": None, "volume": None, "secret": None, "shard_requests": None, "timeout_minutes": None, "billed_tokens_jev_full": None,
              "pricing_as_of": None},
    "kit": {"repo": None, "commit": None},
}
GROUPINGS = ("question_type", "type_k")      # what calibrate.fit_groups(by=...) can fit (a test keeps the two lists equal)
REVISION = re.compile(r"[0-9a-f]{40}")      # a Hub revision is a full commit hash: a branch ("main") moves under a sweep that takes days


class Cfg(dict):
    """dict with attribute access (recursively).  Missing keys raise AttributeError."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(x: Any) -> Any:
    if isinstance(x, Mapping):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _merge(dst: dict, src: Mapping) -> dict:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def overlay_paths(base: Path) -> list:
    """Overlay files next to `base`, in alphabetical order (the base file itself never matches the glob)."""
    return sorted(base.parent.glob(OVERLAY_GLOB))


def load_config(path: Optional[os.PathLike] = None, overrides: Optional[Mapping] = None,
                overlays: Optional[bool] = None) -> Cfg:
    """Load the kit config.

    With no `path` (and no SD_CONFIG) the default decide/config.yaml is read and every decide/config.*.yaml
    next to it is deep-merged on top, in alphabetical order.  An explicit `path` is read alone unless
    `overlays=True`.  `overrides` win over everything.
    """
    explicit = path or os.environ.get("SD_CONFIG")
    p = Path(explicit or DEFAULT_PATH)
    raw = yaml.safe_load(p.read_text()) or {}
    foreign: List[str] = []
    if (explicit is None) if overlays is None else overlays:
        for extra in overlay_paths(p):
            _merge(raw, yaml.safe_load(extra.read_text()) or {})
            foreign.append(extra.name.split(".")[1])                 # decide/config.<name>.yaml owns the section <name>
    if overrides:
        _merge(raw, overrides)
    validate(raw, foreign=foreign)
    return _wrap(raw)


class ConfigError(ValueError):
    pass


def is_revision(value: Any) -> bool:
    """True for a full lowercase 40-hex commit hash, the only kind of revision this kit reads."""
    return isinstance(value, str) and REVISION.fullmatch(value) is not None


def model_entries(models: Mapping) -> List[Tuple[str, Mapping]]:
    """Every model repo the `models` section names, as `(label, entry)`: base, anchor, lora, d3, each donor, each rehearsal model."""
    out: List[Tuple[str, Mapping]] = []
    for key in ("base", "anchor", "lora", "d3"):
        if isinstance(models.get(key), Mapping):
            out.append((f"models.{key}", models[key]))
    for i, donor in enumerate(models.get("donors") or []):
        if isinstance(donor, Mapping):
            out.append((f"models.donors[{donor.get('name', i)}]", donor))
    for key, entry in (models.get("rehearsal") or {}).items():
        if isinstance(entry, Mapping):
            out.append((f"models.rehearsal.{key}", entry))
    return out


def dataset_entries(raw: Mapping) -> List[Tuple[str, Mapping]]:
    """The published result datasets the fidelity check compares against, as `(label, entry)`."""
    published = (raw.get("fidelity") or {}).get("published") or {}
    return [(f"fidelity.published.{key}", entry) for key, entry in published.items() if isinstance(entry, Mapping)]


def _unknown_keys(schema: Any, value: Any, path: str) -> Iterable[str]:
    """Messages for every key of `value` that `schema` does not list, and for every value of the wrong shape."""
    if schema is None:
        return
    if isinstance(schema, list):
        if not isinstance(value, list):
            yield f"{path} must be a list"
            return
        for i, item in enumerate(value):
            yield from _unknown_keys(schema[0], item, f"{path}[{i}]")
        return
    if not isinstance(value, Mapping):
        yield f"{path} must be a mapping"
        return
    if set(schema) == {"*"}:
        for key, item in value.items():
            yield from _unknown_keys(schema["*"], item, f"{path}.{key}")
        return
    for key in value:
        if key not in schema:
            near = difflib.get_close_matches(str(key), [str(k) for k in schema], n=1, cutoff=0.6)
            yield f"unknown key {key!r} in {path}" + (f" (did you mean {near[0]!r}?)" if near else f"; allowed: {sorted(schema)}")
    for key, sub in schema.items():
        if key in value:
            yield from _unknown_keys(sub, value[key], f"{path}.{key}")


def _number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and x not in (float("inf"), float("-inf"))


def _whole(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate(raw: Mapping, foreign: Iterable[str] = ()) -> None:
    """Raise `ConfigError` for anything this kit would otherwise run differently from what was written.

    `foreign` are the sections other squads bring in overlay files (`decide/config.<name>.yaml` brings `<name>`): they are theirs to check.
    """
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ConfigError(msg)

    for key in raw:
        if key not in SCHEMA and key not in set(foreign):
            near = difflib.get_close_matches(str(key), [*SCHEMA, *foreign], n=1, cutoff=0.6)
            raise ConfigError(f"unknown section {key!r}" + (f" (did you mean {near[0]!r}?)" if near else f"; sections are {sorted(SCHEMA)}")
                              + "; another squad's section belongs in decide/config.<section>.yaml")
    for section in ("hub", "models", "merge", "eval", "budget", "calibration", "tta"):
        need(section in raw, f"missing section: {section}")
    for section, schema in SCHEMA.items():
        if section in raw:
            for message in _unknown_keys(schema, raw[section], section):
                raise ConfigError(message)
    hub = raw["hub"]
    problem = endpoint_problem(hub.get("endpoint", "https://huggingface.co"))
    need(problem is None, f"hub.endpoint {problem}")
    m = raw["merge"]
    need(m.get("method") in METHODS, f"merge.method must be one of {METHODS}")
    need(m.get("reference", "anchor") in REFERENCES, f"merge.reference must be one of {REFERENCES}")
    grid = m.get("lambda_grid", [])
    need(isinstance(grid, list) and len(grid) >= 1 and all(_number(x) and x > 0 for x in grid), "merge.lambda_grid: a list of positive numbers")
    d = (m.get("ties") or {}).get("density", 0.5)
    need(_number(d) and 0 < d <= 1, "merge.ties.density must be a number in (0, 1]")
    donors = raw["models"].get("donors", [])
    names = [x["name"] for x in donors]
    need(len(names) == len(set(names)), "duplicate donor names")
    need(all(_number(x.get("weight", 0)) and x.get("weight", 0) >= 0 for x in donors), "donor weights must be numbers >= 0")
    lora_repo = str((raw["models"].get("lora") or {}).get("repo", "")).lower() if (raw["models"].get("lora") or {}).get("enabled", True) else ""
    anchor_repo = str(raw["models"].get("anchor", {}).get("repo", "")).lower()
    for d in donors:
        need(not (d.get("enabled", True) and lora_repo and str(d.get("repo", "")).lower() == lora_repo),
             f"donor {d.get('name')!r} is both a donor and the LoRA ({lora_repo}): its delta would be counted twice")
        need(not (d.get("enabled", True) and str(d.get("repo", "")).lower() == anchor_repo),
             f"donor {d.get('name')!r} is the anchor ({anchor_repo}): its task vector is zero, so the 'merge' would be the anchor")
        tag = f"{d.get('name', '')} {d.get('repo', '')}".lower()
        need(not (d.get("enabled", True) and any(f in tag for f in FORBIDDEN_DONORS)),
             f"donor {d.get('name')!r} is not allowed (license or incompatible readout; see fase0b)")
    need("repo" in raw["models"].get("anchor", {}), "models.anchor.repo is required")
    for label, entry in model_entries(raw["models"]):
        if is_local(entry.get("repo")):                                             # dir:PATH, a checkpoint folder on this machine: not on the Hub, no commit
            need(entry.get("revision") is None, f"{label} is a folder on this machine and has no revision, got {entry.get('revision')!r}")
            continue
        need(is_revision(entry.get("revision")),
             f"{label}.revision must be a full 40-hex commit hash, got {entry.get('revision')!r}: a branch moves between sessions, "
             "so no number of a sweep would belong to one set of weights")
    for label, entry in dataset_entries(raw):
        need(is_revision(entry.get("revision")), f"{label}.revision must be a full 40-hex commit hash, got {entry.get('revision')!r}: "
                                                 "the rows it is compared with must not change under the check")
    lora = raw["models"].get("lora") or {}
    need(_number(lora.get("weight", 0)) and lora.get("weight", 0) >= 0, "lora.weight must be a number >= 0")
    weights = lora.get("weight_grid", [])
    need(isinstance(weights, list) and all(_number(x) and x > 0 for x in weights),
         "lora.weight_grid: a list of positive numbers (a zero or negative weight would run the plain anchor under the name of a LoRA)")
    e = raw["eval"]
    need(e.get("quant") in QUANTS, f"eval.quant must be one of {QUANTS}")
    need(e.get("compute_dtype") in ("float16", "bfloat16", "float32"), "eval.compute_dtype invalid")
    need(_whole(e.get("per_benchmark", 0)) and e.get("per_benchmark", 0) >= 1, "eval.per_benchmark must be a whole number >= 1")
    need(_whole(e.get("sweep_per_benchmark", e.get("per_benchmark", 0))) and e.get("sweep_per_benchmark", e.get("per_benchmark", 0)) >= 1,
         "eval.sweep_per_benchmark must be a whole number >= 1")
    need(_whole(m.get("max_candidates", 6)) and m.get("max_candidates", 6) >= 1, "merge.max_candidates must be a whole number >= 1")
    b = raw["budget"]
    need(_number(b.get("session_hours", 0)) and 0 < b.get("session_hours", 0) <= SESSION_HOURS_MAX,
         f"budget.session_hours must be a number in (0, {SESSION_HOURS_MAX:g}]: a Kaggle session ends at 12 h")
    need(_number(b.get("reserve_minutes", 0)) and b.get("reserve_minutes", 0) >= 0, "budget.reserve_minutes must be a number >= 0")
    need(b["reserve_minutes"] < b["session_hours"] * 60, "budget.reserve_minutes >= session")
    c = raw["calibration"]
    need(_number(c.get("holdout", 0.5)) and 0 < c.get("holdout", 0.5) < 1, "calibration.holdout must be a number in (0, 1)")
    window = c.get("grid", [0.5, 4.0])
    need(isinstance(window, list) and len(window) == 2 and all(_number(x) for x in window) and 0 < window[0] < window[1],
         "calibration.grid must be [low, high] with 0 < low < high")
    need(_whole(raw["tta"].get("rotations", 0)) and raw["tta"].get("rotations", 0) >= 0, "tta.rotations must be a whole number >= 0")
    need(raw["tta"].get("mode", "reverse") in TTA_MODES, f"tta.mode must be one of {TTA_MODES}")
    need(raw["merge"].get("head", "anchor") in ("anchor", "refit"), "merge.head must be anchor|refit")
    # the numbers the rest of the kit takes as they come: each has a range, so a typo fails here and not a night later
    positive = lambda x: _whole(x) and x >= 1                                      # noqa: E731
    need(positive(e.get("batch_tokens", 4096)), "eval.batch_tokens must be a whole number >= 1")
    need(e.get("max_proxy_tokens") is None or positive(e["max_proxy_tokens"]), "eval.max_proxy_tokens must be a whole number >= 1 (or null for no limit)")
    need(positive(e.get("min_per_benchmark", 1)), "eval.min_per_benchmark must be a whole number >= 1")
    need(e.get("max_gpu_gb") is None or (_number(e["max_gpu_gb"]) and e["max_gpu_gb"] > 0), "eval.max_gpu_gb must be a number > 0 (or null)")
    need(_whole(e.get("seed", 0)), "eval.seed must be a whole number")
    need(positive(m.get("chunk_mb", 256)), "merge.chunk_mb must be a whole number >= 1")
    need(_number(b.get("weekly_gpu_hours", 0)) and b.get("weekly_gpu_hours", 0) >= 0, "budget.weekly_gpu_hours must be a number >= 0")
    need(c.get("by", "question_type") in GROUPINGS, f"calibration.by must be one of {GROUPINGS}")
    fid, gates = raw.get("fidelity") or {}, raw.get("gates") or {}
    need(positive(fid.get("min_rows", 1)), "fidelity.min_rows must be a whole number >= 1")
    need(_whole(fid.get("seed", 0)), "fidelity.seed must be a whole number")
    for key in ("g0_agree_4b", "g0_agree_27b_nf4"):
        need(_number(gates.get(key, 0)) and 0 <= gates.get(key, 0) <= 1, f"gates.{key} must be a number in [0, 1]")
    for key in ("g2_min_delta", "g3_min_delta", "calib_min_delta"):
        need(_number(gates.get(key, 0)), f"gates.{key} must be a number")


@dataclass(frozen=True)
class Candidate:
    """One model variant to evaluate.  `name` is stable and used as the key in state and files."""

    name: str
    method: str                    # anchor | task_arithmetic | ties
    lam: float = 0.0
    density: float = 1.0
    donors: tuple = ()             # ((donor_name, weight), ...) weights already include lambda
    lora_weight: float = 0.0
    reference: str = "anchor"

    @property
    def is_anchor(self) -> bool:
        return self.method == "anchor" and self.lora_weight == 0.0


def _fmt(x: float) -> str:
    return f"{x:g}".replace("-", "m")


def enumerate_candidates(cfg: Mapping, include_lora: bool = True) -> list:
    """Deterministic candidate list: anchor, merge grid (cfg.merge.method), then +LoRA variants.

    Donor weights are normalised to sum 1 over the enabled donors, then scaled by lambda.
    """
    donors = [d for d in cfg["models"]["donors"] if d.get("enabled", True) and d.get("weight", 0) > 0]
    total = sum(d["weight"] for d in donors)
    method = cfg["merge"]["method"]
    ref = cfg["merge"].get("reference", "anchor")
    density = float(cfg["merge"].get("ties", {}).get("density", 0.5))
    out = [Candidate("anchor", "anchor", reference=ref)]
    for lam in cfg["merge"]["lambda_grid"]:
        if not donors or lam <= 0:
            continue
        dw = tuple((d["name"], round(lam * d["weight"] / total, 10)) for d in donors)
        name = f"ta_l{_fmt(lam)}" if method == "task_arithmetic" else f"ties_l{_fmt(lam)}_d{_fmt(density)}"
        out.append(Candidate(name, method, lam, density if method == "ties" else 1.0, dw, 0.0, ref))
    lora = cfg["models"].get("lora") or {}
    if include_lora and lora.get("enabled", True):
        weights = lora.get("weight_grid") or [lora.get("weight", 0.1)]
        for w in weights:
            out.append(Candidate(f"anchor+lora{_fmt(w)}", "anchor", 0.0, 1.0, (), float(w), ref))
    names = [c.name for c in out]
    if len(names) != len(set(names)):
        raise ConfigError("candidate names collide: " + ", ".join(names))
    cap = int(cfg["merge"].get("max_candidates", 6))
    if len(out) > cap:
        raise ConfigError(f"{len(out)} candidates exceed merge.max_candidates={cap}; shrink merge.lambda_grid or raise the cap on purpose")
    return out


def sweep_per_benchmark(cfg: Mapping) -> int:
    """Groups per benchmark for the merge sweep: never more than the final sample has."""
    e = cfg["eval"]
    return min(int(e.get("sweep_per_benchmark", e["per_benchmark"])), int(e["per_benchmark"]))


def with_lora(candidate: Candidate, weight: float) -> Candidate:
    """Same merge, plus the Simplicio LoRA delta at `weight`."""
    return Candidate(f"{candidate.name}+lora{_fmt(weight)}", candidate.method, candidate.lam, candidate.density,
                     candidate.donors, float(weight), candidate.reference)
