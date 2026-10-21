"""decide/config.yaml loading (with decide/config.*.yaml overlays), validation and candidate enumeration."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
OVERLAY_GLOB = "config.*.yaml"          # decide/config.data.yaml, decide/config.rental.yaml, ... (Squad B)
METHODS = ("task_arithmetic", "ties")
REFERENCES = ("anchor", "base")
QUANTS = ("nf4", "fp16", "bf16", "none")
TTA_MODES = ("reverse", "rotate")
# fase0b: Lychee is CC BY-NC (cannot ship), amos has no compatible readout head; neither may be a donor.
FORBIDDEN_DONORS = ("lychee", "amos")


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
    if (explicit is None) if overlays is None else overlays:
        for extra in overlay_paths(p):
            _merge(raw, yaml.safe_load(extra.read_text()) or {})
    if overrides:
        _merge(raw, overrides)
    validate(raw)
    return _wrap(raw)


class ConfigError(ValueError):
    pass


def validate(raw: Mapping) -> None:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ConfigError(msg)

    for section in ("hub", "models", "merge", "eval", "budget", "calibration", "tta"):
        need(section in raw, f"missing section: {section}")
    m = raw["merge"]
    need(m.get("method") in METHODS, f"merge.method must be one of {METHODS}")
    need(m.get("reference", "anchor") in REFERENCES, f"merge.reference must be one of {REFERENCES}")
    grid = m.get("lambda_grid", [])
    need(len(grid) >= 1 and all(isinstance(x, (int, float)) and x >= 0 for x in grid), "merge.lambda_grid: non-negative numbers")
    d = (m.get("ties") or {}).get("density", 0.5)
    need(0 < d <= 1, "merge.ties.density must be in (0, 1]")
    donors = raw["models"].get("donors", [])
    names = [x["name"] for x in donors]
    need(len(names) == len(set(names)), "duplicate donor names")
    need(all(x.get("weight", 0) >= 0 for x in donors), "donor weights must be >= 0")
    for d in donors:
        tag = f"{d.get('name', '')} {d.get('repo', '')}".lower()
        need(not (d.get("enabled", True) and any(f in tag for f in FORBIDDEN_DONORS)),
             f"donor {d.get('name')!r} is not allowed (license or incompatible readout; see fase0b)")
    need("repo" in raw["models"].get("anchor", {}), "models.anchor.repo is required")
    lora = raw["models"].get("lora") or {}
    need(lora.get("weight", 0) >= 0, "lora.weight must be >= 0")
    e = raw["eval"]
    need(e.get("quant") in QUANTS, f"eval.quant must be one of {QUANTS}")
    need(e.get("compute_dtype") in ("float16", "bfloat16", "float32"), "eval.compute_dtype invalid")
    need(int(e.get("per_benchmark", 0)) >= 1, "eval.per_benchmark must be >= 1")
    need(int(e.get("sweep_per_benchmark", e.get("per_benchmark", 0))) >= 1, "eval.sweep_per_benchmark must be >= 1")
    need(int(m.get("max_candidates", 6)) >= 1, "merge.max_candidates must be >= 1")
    b = raw["budget"]
    need(b.get("session_hours", 0) > 0 and b.get("reserve_minutes", 0) >= 0, "budget values invalid")
    need(b["reserve_minutes"] < b["session_hours"] * 60, "budget.reserve_minutes >= session")
    c = raw["calibration"]
    need(0 < c.get("holdout", 0.5) < 1, "calibration.holdout must be in (0, 1)")
    need(int(raw["tta"].get("rotations", 0)) >= 0, "tta.rotations must be >= 0")
    need(raw["tta"].get("mode", "reverse") in TTA_MODES, f"tta.mode must be one of {TTA_MODES}")
    need(raw["merge"].get("head", "anchor") in ("anchor", "refit"), "merge.head must be anchor|refit")


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
