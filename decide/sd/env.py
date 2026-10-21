"""Platform detection, working paths, secrets and device-capacity advice.

One place for everything that differs between Kaggle, Colab, Modal and a plain Linux box:
the working directory, where HF_TOKEN comes from, and how the 27B model can be laid out on
the GPUs that are present.  Pure functions take their inputs as arguments so they can be
tested without any of those platforms.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

PLATFORMS = ("kaggle", "colab", "modal", "hfjob", "generic")

# Approximate sizes of Qwen3.8-27B decision backbones, text tower only, in GiB (the unit torch and
# nvidia-smi report).  Derived from the pplx-decider-v1.2 safetensors index: 26.085e9 params in
# total, of which embed_tokens = 248320*5120 = 1.27e9 and the vision tower ~0.46e9.  These are
# sizing ESTIMATES (UNVERIFIED on a real GPU), not measurements.
NF4_TEXT_GB = 11.6      # 24.1e9 quantised matmul params * 0.5 B + double-quant statistics
EMBED_FP16_GB = 2.37    # 1.27e9 params * 2 B
BF16_TEXT_GB = 47.7     # text tower in bf16 (embeddings included)
VISION_GB = 0.9
ACTIVATION_HEADROOM_GB = 1.5
NF4_LAYER_GB = NF4_TEXT_GB / 64


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detect_platform(
    environ: Optional[Mapping[str, str]] = None,
    exists: Callable[[str], bool] = os.path.exists,
    module_available: Callable[[str], bool] = _module_available,
) -> str:
    """Return one of PLATFORMS.  Environment variables win over directories."""
    env = os.environ if environ is None else environ
    if env.get("SD_PLATFORM") in PLATFORMS:
        return env["SD_PLATFORM"]
    if env.get("MODAL_TASK_ID") or env.get("MODAL_IS_REMOTE") == "1":
        return "modal"
    if env.get("HF_JOB_ID"):
        return "hfjob"
    if env.get("KAGGLE_KERNEL_RUN_TYPE") or env.get("KAGGLE_URL_BASE") or exists("/kaggle/working"):
        return "kaggle"
    if env.get("COLAB_RELEASE_TAG") or env.get("COLAB_GPU") is not None or module_available("google.colab"):
        return "colab"
    return "generic"


def default_workdir(platform: str, environ: Optional[Mapping[str, str]] = None, exists: Callable[[str], bool] = os.path.exists) -> Path:
    env = os.environ if environ is None else environ
    if env.get("SD_WORKDIR"):
        return Path(env["SD_WORKDIR"])
    if platform == "kaggle":
        return Path("/kaggle/working")
    if platform == "colab":
        return Path("/content/sd_work")
    if platform == "modal":
        return Path("/cache/sd_work") if exists("/cache") else Path("/tmp/sd_work")
    return Path.cwd() / "work"


def get_secret(
    name: str,
    environ: Optional[Mapping[str, str]] = None,
    kaggle_client: Optional[Callable[[str], str]] = None,
    colab_userdata: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    """Look a secret up: process env first, then Kaggle Secrets, then Colab userdata.

    Never raises and never logs the value.  The two provider callables exist for tests.
    """
    env = os.environ if environ is None else environ
    value = env.get(name)
    if value:
        return value
    if kaggle_client is None and _module_available("kaggle_secrets"):
        def kaggle_client(n: str) -> str:  # pragma: no cover - needs Kaggle
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret(n)
    if kaggle_client is not None:
        try:
            value = kaggle_client(name)
            if value:
                return value
        except Exception:
            pass
    if colab_userdata is None and _module_available("google.colab"):
        def colab_userdata(n: str) -> str:  # pragma: no cover - needs Colab
            from google.colab import userdata

            return userdata.get(n)
    if colab_userdata is not None:
        try:
            value = colab_userdata(name)
            if value:
                return value
        except Exception:
            pass
    return None


def ensure_hf_token(environ: Optional[dict] = None, **providers) -> Optional[str]:
    """Export HF_TOKEN into the environment if some provider has it; return it (or None)."""
    env = os.environ if environ is None else environ
    token = get_secret("HF_TOKEN", environ=env, **providers)
    if token:
        env["HF_TOKEN"] = token
    return token


def require_hf_token(**providers) -> str:
    token = ensure_hf_token(**providers)
    if not token:
        raise RuntimeError(
            "HF_TOKEN not found. Kaggle: Add-ons > Secrets > HF_TOKEN (attach it to this notebook). "
            "Colab: the key icon > HF_TOKEN. Elsewhere: export HF_TOKEN=..."
        )
    return token


# --------------------------------------------------------------------------- hardware

@dataclass(frozen=True)
class Hardware:
    platform: str
    workdir: str
    gpu_names: tuple = ()
    gpu_mem_gb: tuple = ()
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    cuda_capability: tuple = ()

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_mem_gb)

    def summary(self) -> str:
        gpus = ", ".join(f"{n} ({m:.1f} GB)" for n, m in zip(self.gpu_names, self.gpu_mem_gb)) or "none"
        return f"platform={self.platform} workdir={self.workdir} gpus=[{gpus}] ram={self.ram_gb:.1f} GB disk_free={self.disk_free_gb:.1f} GB"


def parse_nvidia_smi(text: str) -> tuple:
    """Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`."""
    names, mem = [], []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.rsplit(",", 1)]
        if len(parts) != 2 or not re.fullmatch(r"\d+(\.\d+)?", parts[1]):
            continue
        names.append(parts[0])
        mem.append(round(float(parts[1]) / 1024.0, 2))  # MiB -> GiB (reported as "GB" below)
    return tuple(names), tuple(mem)


def read_meminfo_gb(path: str = "/proc/meminfo") -> float:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        pass
    return 0.0


def detect_hardware(workdir: Optional[os.PathLike] = None) -> Hardware:
    platform = detect_platform()
    wd = Path(workdir) if workdir else default_workdir(platform)
    names: tuple = ()
    mem: tuple = ()
    cap: tuple = ()
    try:  # torch is optional here
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = tuple(torch.cuda.get_device_name(i) for i in range(n))
            mem = tuple(round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2) for i in range(n))
            cap = tuple(torch.cuda.get_device_capability(i) for i in range(n))
    except Exception:
        pass
    if not names and shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20,
            ).stdout
            names, mem = parse_nvidia_smi(out)
        except Exception:
            pass
    probe = wd if wd.exists() else Path("/")
    try:
        free = round(shutil.disk_usage(probe).free / 1024**3, 1)
    except OSError:
        free = 0.0
    return Hardware(platform, str(wd), names, mem, read_meminfo_gb(), free, cap)


# --------------------------------------------------------------------------- capacity plan

@dataclass
class DevicePlan:
    strategy: str                 # split2 | single | single_tight | bf16_single | insufficient
    embeddings_device: str = "cuda:0"
    max_memory_gb: dict = field(default_factory=dict)
    cpu_offload_layers: int = 0
    notes: list = field(default_factory=list)
    ok: bool = True


def plan_devices(
    gpu_mem_gb: Sequence[float],
    quant: str = "nf4",
    allow_cpu_offload: bool = False,
    max_gpu_gb: Optional[float] = None,
) -> DevicePlan:
    """Decide how to lay a 27B decision backbone out on the GPUs that are present.

    Numbers are sizing estimates (see module constants), not measurements.
    """
    mem = [float(m) for m in gpu_mem_gb]
    if max_gpu_gb is not None:
        mem = [min(m, float(max_gpu_gb) + 1.0) for m in mem]
    usable = [max(0.0, m - 0.9) for m in mem]  # CUDA context + allocator slack
    if quant == "bf16":
        need = BF16_TEXT_GB + ACTIVATION_HEADROOM_GB
        if usable and max(usable) >= need:
            return DevicePlan("bf16_single", "cuda:0", {0: round(max(usable), 1)}, notes=["bf16 text tower fits on one GPU"])
        if sum(usable) >= need and len(usable) > 1:
            return DevicePlan("split2", "cuda:0", {i: round(u, 1) for i, u in enumerate(usable)}, notes=["bf16 split across GPUs (pipeline, one active at a time)"])
        return DevicePlan("insufficient", "cpu", {}, notes=[f"bf16 needs about {need:.0f} GB of GPU memory"], ok=False)
    need_full = NF4_TEXT_GB + EMBED_FP16_GB + ACTIVATION_HEADROOM_GB
    if len(usable) >= 2 and sum(usable[:2]) >= need_full and min(usable[:2]) >= need_full / 2 - 1.0:
        return DevicePlan("split2", "cuda:0", {0: round(usable[0], 1), 1: round(usable[1], 1)}, notes=["NF4: layers split across 2 GPUs, one active at a time"])
    if usable and usable[0] >= need_full:
        return DevicePlan("single", "cuda:0", {0: round(usable[0], 1)}, notes=["NF4 fits on one GPU"])
    tight = NF4_TEXT_GB + ACTIVATION_HEADROOM_GB
    if usable and usable[0] >= tight:
        notes = ["single 16 GB-class GPU (Colab T4): embeddings stay on CPU; marginal, keep max tokens small"]
        return DevicePlan("single_tight", "cpu", {0: round(usable[0], 1)}, notes=notes)
    if usable and allow_cpu_offload and usable[0] >= 8.0:
        k = int(max(1, -(-(tight - usable[0]) // NF4_LAYER_GB)) + 1)  # ceil, plus one layer of slack
        return DevicePlan(
            "single_tight", "cpu", {0: round(usable[0], 1)}, cpu_offload_layers=k,
            notes=[f"CPU offload of {k} decoder layers (moved to the GPU on demand): slow and UNVERIFIED"],
        )
    return DevicePlan("insufficient", "cpu", {}, notes=[f"NF4 needs about {tight:.0f} GB of GPU memory; found {sum(usable):.1f} GB"], ok=False)
