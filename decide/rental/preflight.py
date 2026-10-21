"""What is checked before the first dollar of GPU time is spent: the machine is the one the profile counted on, and nothing else is wrong.

Problems stop the day.  Warnings are printed.  The estimator's memory warnings that mean "this will not run" (the DDP full fine-tune
that needs 104 GB on an 80 GB GPU, a host with less RAM than the optimizer needs) are problems unless the owner names them in
``--ignore-warning``; the others (activations are not counted, the optimizer lives on the host, no thinker-token estimate) are
only printed.  Nothing here runs a model.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from . import estimate as E
from .profile import Profile

BLOCKING = ("ddp_does_not_fit", "profile_host_ram_too_small")
KNOWN = BLOCKING + ("activations_not_counted", "host_memory", "traces_not_estimated")
RAM_SLACK = 0.95          # a host sold as 512 GB reports a little less
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Probes:
    """How the machine is looked at (replaced in tests)."""

    gpus: Callable[[], List[Tuple[str, float]]]
    free_gb: Callable[[Path], float]
    ram_gb: Callable[[], float]


def _gpus() -> List[Tuple[str, float]]:
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    if done.returncode == 0:
        for line in done.stdout.splitlines():
            name, _, memory = line.rpartition(",")
            if name.strip():
                found.append((name.strip(), round(float(memory) * 1024 * 1024 / 1e9, 1)))
    return found


def _free_gb(path: Path) -> float:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def _ram_gb() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


def system_probes() -> Probes:
    return Probes(gpus=_gpus, free_gb=_free_gb, ram_gb=_ram_gb)


def preflight(profile: Profile, cfg: Mapping[str, Any], workdir: "str | os.PathLike[str]", *, probes: Probes, allow_cpu: bool = False,
              ignore_warnings: Sequence[str] = (), public_requests: "str | os.PathLike[str] | None" = None) -> Dict[str, Any]:
    unknown = sorted(set(ignore_warnings) - set(KNOWN))
    if unknown:
        raise ValueError(f"unknown warning name(s) {unknown}; the estimator's are {list(KNOWN)}")
    problems: List[str] = []
    warnings: List[str] = []

    found = probes.gpus()
    family = profile.gpu.split("-")[0]
    matching = [gpu for gpu in found if family in gpu[0]]
    if len(matching) < profile.gpus:
        text = f"the profile needs {profile.gpus} x {profile.gpu}; this machine has {len(matching)} of them ({len(found)} GPU(s) in all: {[name for name, _ in found]})"
        if allow_cpu:
            warnings.append(f"{text}; continuing on the CPU because of --allow-cpu: this is a rehearsal, not a training day")
        else:
            problems.append(text)

    disk = probes.free_gb(Path(workdir))
    if disk < profile.disk_gb:
        problems.append(f"disk: {disk:.0f} GB free under {workdir}, the profile needs {profile.disk_gb:g} GB")
    ram = probes.ram_gb()
    if ram < profile.host_ram_gb * RAM_SLACK:
        problems.append(f"RAM: {ram:.0f} GB, the profile needs {profile.host_ram_gb:g} GB")

    ignored = set(ignore_warnings)
    for warning in E.estimate(profile)["warnings"]:
        line = f"{warning['code']}: {warning['text']}"
        if warning["code"] in BLOCKING and warning["code"] not in ignored:
            problems.append(f"{line} (to run anyway, say so by name: --ignore-warning {warning['code']})")
        elif warning["code"] in BLOCKING:
            warnings.append(f"ignored on request, {line}")
        else:
            warnings.append(line)

    if cfg["hub"].get("private") is not True:
        problems.append("hub.private must be true: the day only writes to private repos; making a repo public is the owner's decision, outside this tool")
    for where, revision in (("start.revision", cfg["start"].get("revision")), ("gates.baseline.revision", cfg["gates"]["baseline"].get("revision"))):
        if not (isinstance(revision, str) and _COMMIT.fullmatch(revision)):
            problems.append(f"{where} must be a 40-character commit (a branch name moves under the day), got {revision!r}")
    if public_requests is not None and not Path(public_requests).is_file():
        problems.append(f"public requests file {public_requests} does not exist")

    return {"ok": not problems, "problems": problems, "warnings": warnings,
            "facts": {"gpus": [[name, memory] for name, memory in found], "free_gb": disk, "ram_gb": ram}}
