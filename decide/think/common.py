"""Constants and small helpers shared by the GT kit (issue #75, F2b).  Standard library only.

The GT gate asks one question: on the same public knowledge/reasoning rows, does Qwen3.8-27B with
thinking beat the best one-pass decision model by at least 10 accuracy points, with the lower end of
the paired-bootstrap 95% interval above 0?  Everything in this package either prepares those rows
(`sample`), runs the thinker (`thinker`), or compares the systems (`compare`).

The benchmark questions are evaluation data under their own terms (GPQA asks not to post them in
plain text, BBH carries the BIG-bench canary, HLE is gated).  Nothing in this package writes them
to a tracked file: the sample pack stays private, the committed manifest holds ids and hashes only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

SEED = 20261021
PER_BENCHMARK = 40
BOOTSTRAP_B = 10_000
GT_MIN_DELTA_PTS = 10.0
# The kit refuses a model whose median, mean or p80 latency passes this (README of the kit, edition 0.4; ADR 0004, fact 4).
LATENCY_RULE_MS = 1000.0
CI_LEVEL = 0.95

# The model under test, pinned by commit: `main` of the Hub repo on 2026-10-21 (last modified 2026-08-14).
QWEN_MODEL = "Qwen/Qwen3.8-27B"
QWEN_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN_FP8_MODEL = "Qwen/Qwen3.8-27B-FP8"
QWEN_FP8_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
PINNED_REVISIONS = {QWEN_MODEL: QWEN_REVISION, QWEN_FP8_MODEL: QWEN_FP8_REVISION}      # the two checkpoints the gate is registered for
# Sampling for thinking mode, from the model card.
THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}

# A result row is final when its status is one of these; anything else (`error`) is retried on resume.
TERMINAL_STATUSES = ("ok", "failed")

# Written into the manifest, the results header and the report (ADR 0004, item 6).
EVAL_ONLY_NOTICE = (
    "evaluation data only: the public-suite requests of this sample and the thinker's traces on them "
    "must never enter a training corpus (ADR 0004, item 6)"
)

PACK_FORMAT = "think-gt-pack/1"
MANIFEST_FORMAT = "think-gt-manifest/1"
RESULT_FORMAT = "think-gt-result/1"
PROMPT_VERSION = "think-gt-prompt/1"

# Jev Decision Index 0.3 catalog id -> (name, chance).  The chance is the benchmark-level value of
# "chance" in decision_index/data/index-0.3.json (kit commit 25c6ec7), the one the kit uses for the
# chance-corrected skill: skill = clip((accuracy - chance) / (1 - chance), 0, 1).
BENCHMARKS: Dict[int, Tuple[str, float]] = {
    25: ("GPQA Diamond", 0.25),
    28: ("WinoGrande", 0.5),
    30: ("GSM8K", 0.175),
    31: ("ChessBench", 0.0819),
    32: ("MuSR", 0.371),
    33: ("SATA-Bench", 0.0131),
    43: ("CRUXEval", 0.3697),
    44: ("CLadder", 0.5),
    45: ("HLE", 0.1641),
    57: ("MMLU-Pro", 0.1109),
    58: ("BBH", 0.3101),
}
DEFAULT_BENCHMARKS: Tuple[int, ...] = tuple(sorted(BENCHMARKS))

# The 8 decision models of Fase 0A, as named in the committed manifest.
DECISION_MODELS: Tuple[str, ...] = (
    "startlux-27b", "d3-turbo", "drex-max", "ajev-lm6q", "d3", "amos", "linne-oak", "lychee",
)


def benchmark_name(catalog_id: int) -> str:
    return BENCHMARKS.get(int(catalog_id), (f"benchmark-{catalog_id}", 0.0))[0]


def chance_of(catalog_id: int) -> float:
    return BENCHMARKS[int(catalog_id)][1]


def skill(accuracy: float, chance: float) -> float:
    """Chance-corrected skill as the kit does it, clipped to [0, 1]."""
    if chance >= 1.0:
        return accuracy
    return min(1.0, max(0.0, (accuracy - chance) / (1.0 - chance)))


# ---------------------------------------------------------------------------------------- hashing

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_digests(path: os.PathLike) -> Dict[str, str]:
    """sha256 of the bytes as stored and, for .gz, of the content: the first is what the Hub lists, the second what a recompression keeps."""
    out = {"sha256": sha256_file(path)}
    if str(path).endswith(".gz"):
        h = hashlib.sha256()
        with gzip.open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        out["uncompressed_sha256"] = h.hexdigest()
    return out


def fingerprint(obj: Any) -> str:
    """Short stable id of a JSON-able config."""
    return sha256_text(canonical_json(obj))[:16]


# ------------------------------------------------------------------------------------------ files

def open_text(path: os.PathLike, mode: str = "rt"):
    """Open a text file, gunzipping when the name ends in .gz."""
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def iter_jsonl(path: os.PathLike, *, complete_lines_only: bool = True) -> Iterator[dict]:
    """Yield the JSON objects of a JSONL file.

    A last line without its newline is a write that a crash cut short; it is skipped by default, the
    way the kit's own reader does it.  Blank lines are ignored.
    """
    with open_text(path) as f:
        for line in f:
            if complete_lines_only and not line.endswith("\n"):
                break
            if line.strip():
                yield json.loads(line)


_APPEND_LOCK = threading.Lock()


def append_jsonl(path: os.PathLike, record: dict) -> None:
    """Append one record as a single line and force it to disk.  Safe to call from several threads."""
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _APPEND_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def repair_jsonl(path: os.PathLike) -> int:
    """Drop a trailing partial line (no newline) in place; return the number of bytes removed."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    data = p.read_bytes()
    if data.endswith(b"\n"):
        return 0
    cut = data.rfind(b"\n") + 1
    with open(p, "r+b") as f:
        f.truncate(cut)
    return len(data) - cut


def write_jsonl_gz(path: os.PathLike, records: Iterable[dict]) -> int:
    """Write records to a gzip JSONL file atomically (temp file, then rename)."""
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    os.replace(tmp, p)
    return n


def write_json(path: os.PathLike, obj: Any) -> str:
    """Write pretty, key-sorted JSON atomically; return its sha256 (of the exact bytes written)."""
    p = Path(path)
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)
    return sha256_text(text)


def env_or(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value else default
