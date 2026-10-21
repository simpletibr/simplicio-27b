"""Pure merge math on torch tensors (no I/O, no model classes).

* task arithmetic:  ref + sum_i w_i * (model_i - ref)
* TIES:             trim each task vector to its top-`density` entries by magnitude, elect a sign per
                    entry from the weighted sum, keep only entries that agree with it, take the weighted
                    mean of the agreeing entries (the "disjoint mean") and scale by sum(w).
* LoRA delta:       weight * (alpha/r) * (B @ A)   (alpha/sqrt(r) for rsLoRA)
* names:            LoRA keys and base-model keys carry a "model." prefix that the decision backbones drop.

All arithmetic is float32 whatever the storage dtype (checkpoints are bf16).  Heads are never averaged:
the donors read out at a different position with a different head (fase0b), so `choose_head` returns
the anchor's head unless a refit head is supplied.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

METHODS = ("task_arithmetic", "ties")


def _check(ref: torch.Tensor, models: Sequence[torch.Tensor], weights: Sequence[float]) -> None:
    if len(models) != len(weights):
        raise ValueError(f"{len(models)} models but {len(weights)} weights")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be >= 0")
    for m in models:
        if m.shape != ref.shape:
            raise ValueError(f"shape mismatch: {tuple(m.shape)} vs {tuple(ref.shape)}")


def task_arithmetic(base: torch.Tensor, models: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    """base + sum_i w_i * (model_i - base), float32."""
    _check(base, models, weights)
    b = base.float()
    out = b.clone()
    for m, w in zip(models, weights):
        if w != 0:
            out.add_(m.float() - b, alpha=float(w))
    return out


def trim_topk(t: torch.Tensor, density: float) -> torch.Tensor:
    """Keep exactly k = ceil(density * numel) entries of largest |value| (ties: lowest flat index), zero the rest."""
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    n = t.numel()
    if n == 0:
        return t.clone()
    k = min(n, max(1, math.ceil(round(density * n, 9))))
    if k >= n:
        return t.clone()
    flat = t.reshape(-1)
    a = flat.abs()
    thr = torch.kthvalue(a, n - k + 1).values            # k-th largest magnitude
    keep = a > thr
    need = k - int(keep.sum())
    if need > 0:
        eq = (a == thr).nonzero().flatten()
        keep[eq[:need]] = True
    return (flat * keep).reshape(t.shape)


def ties(base: torch.Tensor, models: Sequence[torch.Tensor], weights: Sequence[float], density: float = 0.5) -> torch.Tensor:
    """TIES merge of `models` around `base`; the result is scaled by sum(weights) (see module doc)."""
    _check(base, models, weights)
    b = base.float()
    total = float(sum(weights))
    if not models or total == 0:
        return b.clone()
    stack = torch.stack([trim_topk(m.float() - b, density) for m in models])
    wv = torch.tensor(list(weights), dtype=torch.float32).view(-1, *([1] * b.dim()))
    elected = torch.where((wv * stack).sum(0) >= 0, 1.0, -1.0)
    agree = (torch.sign(stack) == elected) & (stack != 0)
    num = (wv * stack * agree).sum(0)
    den = (wv * agree).sum(0)
    merged = torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num))
    return b + total * merged


# --------------------------------------------------------------------------- LoRA

def lora_scale(alpha: float, r: int, rslora: bool = False) -> float:
    return alpha / math.sqrt(r) if rslora else alpha / r


def lora_delta(A: torch.Tensor, B: torch.Tensor, alpha: float, r: int, weight: float = 1.0, rslora: bool = False) -> torch.Tensor:
    """weight * (alpha/r) * (B @ A).  A: [r, in], B: [out, r]  ->  [out, in] float32."""
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("lora A and B must be 2-D")
    if A.shape[0] != r or B.shape[1] != r:
        raise ValueError(f"lora rank mismatch: r={r}, A{tuple(A.shape)}, B{tuple(B.shape)}")
    return (weight * lora_scale(alpha, r, rslora)) * (B.float() @ A.float())


def apply_lora(W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, alpha: float, r: int, weight: float = 1.0, rslora: bool = False) -> torch.Tensor:
    out = W.float()
    if weight == 0:
        return out
    delta = lora_delta(A, B, alpha, r, weight, rslora)
    if delta.shape != out.shape:
        raise ValueError(f"lora delta {tuple(delta.shape)} does not fit weight {tuple(out.shape)}")
    return out + delta


_LORA_KEY = re.compile(r"^(?:base_model\.model\.)?(?P<mod>.+?)\.lora_(?P<ab>[AB])(?:\.[A-Za-z0-9_\-]+)?\.weight$")


def parse_lora_key(key: str) -> Optional[Tuple[str, str]]:
    """'base_model.model.<module>.lora_A[.default].weight' -> (<module>, 'A'); None if not a LoRA key."""
    m = _LORA_KEY.match(key)
    return (m.group("mod"), m.group("ab")) if m else None


def pair_lora_keys(keys: Iterable[str]) -> Dict[str, Tuple[str, str]]:
    """{module: (A_key, B_key)}; raises if a module has only one half."""
    halves: Dict[str, Dict[str, str]] = {}
    for k in keys:
        p = parse_lora_key(k)
        if p:
            halves.setdefault(p[0], {})[p[1]] = k
    out = {}
    for mod, h in halves.items():
        for ab in ("A", "B"):
            if ab not in h:
                raise ValueError(f"module {mod}: missing lora_{ab}")
        out[mod] = (h["A"], h["B"])
    return out


class NameMapper:
    """Maps between base-model names ('model.language_model...') and decision-backbone names ('language_model...')."""

    TOP = ("language_model.", "visual.")

    def __init__(self, anchor_names: Iterable[str]) -> None:
        self.names = set(anchor_names)

    def to_anchor(self, module_or_tensor: str) -> Optional[str]:
        """Resolve a LoRA module path (or tensor name) to an anchor tensor name, or None."""
        base = module_or_tensor if module_or_tensor.endswith(".weight") else module_or_tensor + ".weight"
        for cand in (base, base[len("model."):] if base.startswith("model.") else None, "model." + base):
            if cand and cand in self.names:
                return cand
        return None

    def to_base(self, anchor_name: str) -> str:
        return "model." + anchor_name if anchor_name.startswith(self.TOP) else anchor_name


# --------------------------------------------------------------------------- one tensor

def merge_one(
    anchor: torch.Tensor,
    donors: Sequence[torch.Tensor],
    donor_weights: Sequence[float],
    method: str,
    *,
    base: Optional[torch.Tensor] = None,
    density: float = 0.5,
    lora: Optional[Tuple] = None,
) -> torch.Tensor:
    """Merged float32 tensor for one parameter.

    reference = anchor when `base` is None (donors move the anchor), else `base` (anchor enters with weight 1).
    `lora` = (A, B, alpha, r, weight[, rslora]) adds the Simplicio delta on top.
    """
    if method not in METHODS + ("anchor",):
        raise ValueError(f"unknown merge method {method!r}")
    if len(donors) != len(donor_weights):
        raise ValueError("donors and donor_weights differ in length")
    active = method != "anchor" and any(w != 0 for w in donor_weights)
    if not active:
        out = anchor.float()
    elif base is None:
        out = task_arithmetic(anchor, donors, donor_weights) if method == "task_arithmetic" else ties(anchor, donors, donor_weights, density)
    else:
        models, weights = [anchor, *donors], [1.0, *donor_weights]
        out = task_arithmetic(base, models, weights) if method == "task_arithmetic" else ties(base, models, weights, density)
    if lora is not None:
        A, B, alpha, r, w = lora[:5]
        rs = bool(lora[5]) if len(lora) > 5 else False
        out = apply_lora(out, A, B, alpha, r, w, rs)
    return out


def choose_head(anchor_head: torch.Tensor, refit_head: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The anchor's readout head, or a head refit after the merge.  Heads are never averaged (fase0b)."""
    return refit_head if refit_head is not None else anchor_head
