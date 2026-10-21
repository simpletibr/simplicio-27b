"""Tables the estimator, the preflight and the Modal variant share: GPUs, prices and the size of the model.

Each number has its source beside it.  Prices and peak throughput change; nothing here is a measurement of this
project, and every figure built from them is UNVERIFIED until the day is run and timed (``state/*.json`` records the
measured seconds of each step).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Gpu:
    name: str
    memory_gb: float          # HBM capacity in decimal GB, as the vendor lists it
    bf16_tflops: float        # peak dense tensor throughput (no sparsity)
    usd_per_second: float     # list price of one GPU for one second


PRICE_SOURCE = "Modal pricing page, 2026-10-21 (PLAN.md section 6): list price per GPU-second"
SPEC_SOURCE = ("peak dense BF16 TFLOPS (no sparsity) and HBM capacity of H100 SXM, H200 SXM and B200 SXM from "
               "https://inferencex.semianalysis.com/gpu-specs read on 2026-10-21; A100 80GB: NVIDIA datasheet, 312 dense")

GPUS: Dict[str, Gpu] = {
    "B200": Gpu("B200", 180, 2250, 0.001736),
    "H200": Gpu("H200", 141, 989, 0.001261),
    "H100": Gpu("H100", 80, 989, 0.001097),
    "A100-80GB": Gpu("A100-80GB", 80, 312, 0.000694),
}

#: ``metadata.total_parameters`` and ``total_size`` (bf16) of model.safetensors.index.json of perplexity-ai/pplx-decider-v1.2.
PARAMS = 26_085_330_160
WEIGHT_BYTES = 52_170_660_320
#: Sum of ``in_features + out_features`` over the Linear layers LoRA adapts (``decide.train.model.LORA_TARGETS``), counted on the meta
#: device from the config.json of pplx-decider-v1.2 (transformers 5.17.0): a rank-r adapter has ``r * LORA_UNITS_PER_RANK`` parameters.
#: ``DECIDE_PPLX_CONFIG_DIR`` makes test_contract_71.ArchitectureParity recount it.
LORA_UNITS_PER_RANK = 6_799_360
READOUT_PARAMS = 255 * 5120           # readout head: 255 answer codes x hidden size; trained in both modes
#: Input tokens the Jev board bills for the full public round (PLAN.md section 2).
PUBLIC_ROUND_INPUT_TOKENS = 291_826_668
#: Share of peak throughput a training or inference run is assumed to reach (PLAN.md section 6).
MFU = 0.40
#: ``decide.train.optim.CPUOffloadAdamW`` keeps, on the host, an fp32 master copy, an fp32 gradient copy and the two Adam moments.
OPTIMIZER_BYTES_PER_PARAM = 16
#: Guesses, all UNVERIFIED: disk and network speed, fixed set-up time (install, download, model load), rows of the home proxy,
#: share of the corpus that is the dev split.
ASSUMED_IO_GB_PER_S = 0.5
ASSUMED_SETUP_SECONDS = 1800
PROXY_ROWS = 1500
DEV_FRACTION = 0.01


def gpu(name: str) -> Gpu:
    try:
        return GPUS[name]
    except KeyError:
        raise KeyError(f"unknown GPU {name!r}; known: {', '.join(GPUS)}") from None


def trainable_parameters(mode: str, lora_rank: int) -> int:
    """Parameters the optimizer updates: the adapters (or the whole backbone) plus the readout head."""
    if mode == "lora":
        return LORA_UNITS_PER_RANK * int(lora_rank) + READOUT_PARAMS
    if mode == "full":
        return PARAMS + READOUT_PARAMS
    raise ValueError(f"mode must be 'lora' or 'full', got {mode!r}")
