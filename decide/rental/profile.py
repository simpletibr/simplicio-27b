"""The profiles of the rented day (PLAN.md section 6): which machine, how much data, and the options of the trainer.

A profile is a YAML file in ``decide/rental/profiles`` (or any path).  It never holds a path, a seed, a device or a secret:
the orchestrator owns those (``--targets``, ``--checkpoint``, ``--out``, ``--seed`` of ``decide.train.train``).  The numbers
of the plan (hours, dollars) are kept as the text the plan has, only to be printed next to the estimate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence

from . import hardware as HW

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
VERSION = 1
TOP_LEVEL = ("version", "name", "title", "plan", "hardware", "workload", "train", "train_with_traces", "thinker")
#: Options of ``python -m decide.train.train`` a profile may set (checked against the real parser by test_contract_71).
TRAIN_OPTIONS = ("mode", "lora_rank", "lora_alpha", "lora_dropout", "epochs", "effective_batch", "batch_rows", "token_budget", "max_length",
                 "lr", "warmup_ratio", "weight_decay", "grad_clip", "brier_weight", "alpha", "no_permute", "reasoning", "reasoning_fraction",
                 "max_train", "max_steps", "eval_every", "max_dev", "resume_every", "gradient_checkpointing", "cpu_threads")
#: Options that are on/off switches: ``True`` is ``--flag``; ``gradient_checkpointing`` also has ``--no-...``.
SWITCHES = ("no_permute", "gradient_checkpointing")


class ProfileError(ValueError):
    """A profile that cannot be used, with the reason."""


@dataclass(frozen=True)
class Profile:
    name: str
    title: str
    gpu: str
    gpus: int
    host_ram_gb: float
    disk_gb: float
    examples: int
    tokens_per_example: int
    seeds: int
    train: Mapping[str, Any]
    train_with_traces: Mapping[str, Any]
    thinker: Mapping[str, Any]
    plan: Mapping[str, Any]
    raw: Mapping[str, Any]
    digest: str

    def train_argv(self, *, with_traces: bool = False) -> List[str]:
        """The ``decide.train.train`` flags this profile sets (not the paths, the seed or the device)."""
        options = dict(self.train)
        if with_traces:
            options.update(self.train_with_traces)
        return cli_args(options)


def cli_args(options: Mapping[str, Any]) -> List[str]:
    """``{"lora_rank": 128, "no_permute": True}`` -> ``["--lora-rank", "128", "--no-permute"]``; ``None`` values are left out."""
    argv: List[str] = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if key in SWITCHES:
            if value is True:
                argv.append(flag)
            elif value is False and key == "gradient_checkpointing":
                argv.append("--no-gradient-checkpointing")
            continue
        argv += [flag, str(value)]
    return argv


def _number(value: Any, where: str, *, minimum: float = 0, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{where} must be a number, got {value!r}")
    if integer and int(value) != value:
        raise ProfileError(f"{where} must be a whole number, got {value!r}")
    if not value > minimum:
        raise ProfileError(f"{where} must be greater than {minimum}, got {value!r}")
    return value


def _mapping(raw: Mapping[str, Any], key: str, allowed: Sequence[str], *, required: Sequence[str] = ()) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProfileError(f"{key} must be a mapping")
    extra = sorted(set(value) - set(allowed))
    if extra:
        raise ProfileError(f"{key}: unknown key(s) {extra}; allowed: {list(allowed)}")
    missing = [name for name in required if name not in value]
    if missing:
        raise ProfileError(f"{key}: missing {missing}")
    return value


def profile_from_dict(data: Mapping[str, Any]) -> Profile:
    if not isinstance(data, Mapping):
        raise ProfileError("a profile is a mapping")
    extra = sorted(set(data) - set(TOP_LEVEL))
    if extra:
        raise ProfileError(f"unknown top-level key(s) {extra}; allowed: {list(TOP_LEVEL)}")
    if data.get("version") != VERSION:
        raise ProfileError(f"version must be {VERSION}, got {data.get('version')!r}")
    name, title = data.get("name"), data.get("title")
    if not isinstance(name, str) or not name or not isinstance(title, str) or not title:
        raise ProfileError("name and title are required text")
    hardware = _mapping(data, "hardware", ("gpu", "gpus", "host_ram_gb", "disk_gb"), required=("gpu", "gpus", "host_ram_gb", "disk_gb"))
    workload = _mapping(data, "workload", ("examples", "tokens_per_example", "seeds"), required=("examples", "tokens_per_example", "seeds"))
    train = _mapping(data, "train", TRAIN_OPTIONS, required=("mode",))
    if train["mode"] not in ("lora", "full"):
        raise ProfileError(f"train.mode must be 'lora' or 'full', got {train['mode']!r}")
    with_traces = _mapping(data, "train_with_traces", TRAIN_OPTIONS)
    thinker = _mapping(data, "thinker", ("examples", "mean_tokens"), required=("examples", "mean_tokens"))
    plan = _mapping(data, "plan", ("hours", "usd", "status"), required=("hours", "usd"))
    try:
        HW.gpu(hardware["gpu"])
    except KeyError as exc:
        raise ProfileError(str(exc.args[0])) from None
    _number(hardware["gpus"], "hardware.gpus", integer=True)
    _number(hardware["host_ram_gb"], "hardware.host_ram_gb")
    _number(hardware["disk_gb"], "hardware.disk_gb")
    _number(workload["examples"], "workload.examples", integer=True)
    _number(workload["tokens_per_example"], "workload.tokens_per_example", integer=True)
    _number(workload["seeds"], "workload.seeds", integer=True)
    _number(thinker["examples"], "thinker.examples", integer=True)
    _number(thinker["mean_tokens"], "thinker.mean_tokens", integer=True)
    if train["mode"] == "lora":
        _number(train.get("lora_rank", 0), "train.lora_rank", integer=True)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return Profile(name=name, title=title, gpu=hardware["gpu"], gpus=int(hardware["gpus"]), host_ram_gb=float(hardware["host_ram_gb"]),
                   disk_gb=float(hardware["disk_gb"]), examples=int(workload["examples"]), tokens_per_example=int(workload["tokens_per_example"]),
                   seeds=int(workload["seeds"]), train=dict(train), train_with_traces=dict(with_traces), thinker=dict(thinker), plan=dict(plan),
                   raw=json.loads(canonical), digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest())


def profile_names() -> List[str]:
    return sorted(path.stem for path in PROFILES_DIR.glob("*.yaml"))


def load_profile(name_or_path: "str | Path") -> Profile:
    """A shipped profile by name (``econ_b200``) or any YAML file by path."""
    try:
        import yaml
    except ImportError as exc:      # pragma: no cover - message only
        raise ImportError("PyYAML is required to read a profile (install the pinned packages of decide/train/requirements.txt)") from exc
    text = str(name_or_path)
    path = PROFILES_DIR / f"{text}.yaml"
    if not path.is_file():
        path = Path(text)
        if not path.is_file():
            raise ProfileError(f"no profile {text!r}; shipped profiles: {', '.join(profile_names())}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileError(f"{path}: not valid YAML ({exc})") from exc
    return profile_from_dict(data)
