"""Pure-PyTorch NF4 (4-bit NormalFloat) weights: quantise once, dequantise inside the forward pass.

This is the bitsandbytes NF4 format without double quantisation: the weight is flattened, cut into blocks of
64, each block is scaled by its largest magnitude and every value is replaced by the index of the nearest of the
16 NF4 code points; two indices share a byte, the first in the high nibble.  The packed bytes and the dequantised
values are bit-identical to `bitsandbytes.functional.quantize_4bit(..., compress_statistics=False)` (the test
suite compares them when bitsandbytes is installed).

Why not just bitsandbytes?  It is the faster path on CUDA and `loader.py` uses it when asked, but this module has
no dependency, works on CPU and on any GPU, and lets the loader quantise tensor by tensor while the merge streams
in.  Block statistics may be stored in float16 (`absmax_dtype`) to save about 0.6 GiB over the 24e9 weights of the
27B text tower; float32 keeps the bnb bytes exactly.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

BLOCK = 64
NF4_CODE = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141029358, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)
CHUNK_ELEMENTS = 1 << 24


def code_table(device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(NF4_CODE, dtype=dtype, device=device)


def quantize_nf4(weight: torch.Tensor, block: int = BLOCK, absmax_dtype: torch.dtype = torch.float32
                 ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """(packed uint8 [n/2], absmax [n/block], shape) of `weight`, on the weight's device."""
    if block % 2:
        raise ValueError("block size must be even")
    shape = tuple(weight.shape)
    flat = weight.detach().reshape(-1)
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.numel() // block
    code = code_table(flat.device)
    mid = (code[1:] + code[:-1]) / 2
    packed = torch.empty(flat.numel() // 2, dtype=torch.uint8, device=flat.device)
    absmax = torch.empty(blocks, dtype=torch.float32, device=flat.device)
    step = max(1, CHUNK_ELEMENTS // block)
    for b0 in range(0, blocks, step):
        b1 = min(blocks, b0 + step)
        x = flat[b0 * block:b1 * block].float().reshape(b1 - b0, block)
        scale = x.abs().amax(dim=1)
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        idx = torch.bucketize(x / safe[:, None], mid).to(torch.uint8)
        packed[b0 * block // 2:b1 * block // 2] = ((idx[:, 0::2] << 4) | idx[:, 1::2]).reshape(-1)
        absmax[b0:b1] = scale
    return packed, absmax.to(absmax_dtype), shape


def byte_table(device=None, dtype=torch.float16) -> torch.Tensor:
    """256 x 2 table: byte -> (code of the high nibble, code of the low nibble)."""
    code = code_table(device, torch.float32).to(dtype)
    both = torch.arange(256, device=device)
    return torch.stack([code[both >> 4], code[both & 0x0F]], dim=1)


def dequantize_nf4(packed: torch.Tensor, absmax: torch.Tensor, shape: Tuple[int, ...], block: int = BLOCK,
                   dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """The weight back as `dtype` (the compute dtype)."""
    n = math.prod(shape)
    values = byte_table(packed.device, dtype).index_select(0, packed.int()).reshape(-1, block)
    out = (values * absmax.to(dtype)[:, None]).reshape(-1)[:n]
    return out.reshape(shape)


class Nf4Linear(nn.Module):
    """`nn.Linear` whose weight lives as NF4 and is expanded to `compute_dtype` for each forward pass."""

    def __init__(self, in_features: int, out_features: int, bias: Optional[torch.Tensor] = None,
                 compute_dtype: torch.dtype = torch.float16, block: int = BLOCK, absmax_dtype: torch.dtype = torch.float32,
                 device=None) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.compute_dtype, self.block, self.absmax_dtype = compute_dtype, block, absmax_dtype
        n = in_features * out_features
        blocks = -(-n // block)
        self.register_buffer("packed", torch.zeros(blocks * block // 2, dtype=torch.uint8, device=device))
        self.register_buffer("absmax", torch.zeros(blocks, dtype=absmax_dtype, device=device))
        self.bias = None if bias is None else nn.Parameter(bias.detach().to(compute_dtype), requires_grad=False)

    @classmethod
    def from_weight(cls, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, **kw) -> "Nf4Linear":
        out_features, in_features = weight.shape
        module = cls(in_features, out_features, bias, device=weight.device, **kw)
        module.set_weight(weight)
        return module

    def set_weight(self, weight: torch.Tensor) -> None:
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(f"weight {tuple(weight.shape)} does not fit ({self.out_features}, {self.in_features})")
        packed, absmax, _ = quantize_nf4(weight.to(self.packed.device), self.block, self.absmax_dtype)
        self.packed.copy_(packed)
        self.absmax.copy_(absmax)

    def dequantized(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return dequantize_nf4(self.packed, self.absmax, (self.out_features, self.in_features), self.block, dtype or self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized(self.compute_dtype)
        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        return F.linear(x.to(self.compute_dtype), weight, bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, block={self.block}, compute={self.compute_dtype}"
