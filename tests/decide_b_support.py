"""Shared helpers for the decide/data, decide/train and decide/rental tests (Squad B).

Not a test module (unittest discovery only loads test*.py).  The standard library is enough for most tests;
pyarrow, numpy, torch and transformers are only touched by the helpers that say so, and the tests that need them
skip cleanly when they are missing.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def have(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAVE_PYARROW = have("pyarrow")
HAVE_YAML = have("yaml")
HAVE_NUMPY = have("numpy")
HAVE_TORCH = have("torch")
HAVE_TRANSFORMERS = HAVE_TORCH and have("transformers")
HAVE_PEFT = HAVE_TORCH and have("peft")

needs_pyarrow = unittest.skipUnless(HAVE_PYARROW, "pyarrow is not installed")
needs_yaml = unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
needs_numpy = unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
needs_torch = unittest.skipUnless(HAVE_TORCH, "torch is not installed")
needs_transformers = unittest.skipUnless(HAVE_TRANSFORMERS, "torch + transformers are not installed")
needs_peft = unittest.skipUnless(HAVE_PEFT, "peft is not installed")

_VOCAB = (
    "river stone market lantern copper violin harbor winter orchid signal paper engine garden bridge "
    "falcon thunder marble canvas pepper silver window carpet meadow rocket saddle ladder hammer whisper "
    "cabinet compass glacier harvest isolate journey kettle lizard mirror nugget oyster pillow quartz "
    "ribbon sketch timber umbrella velvet walnut yonder zephyr anchor basalt cinder dagger ember fossil"
).split()


def lorem(seed: int, count: int) -> str:
    """Deterministic pseudo-text of `count` words; different seeds share (almost) no 13-grams."""
    rng = random.Random(seed)
    return " ".join(rng.choice(_VOCAB) + (str(rng.randrange(1000)) if rng.random() < 0.5 else "") for _ in range(count))


def choice_request(
    instructions: str = "Which city is the capital of France?",
    options: Sequence[str] = ("Paris", "Rome", "Berlin", "Madrid"),
    state: Any = None,
    keys: Optional[Sequence[str]] = None,
    qid: str = "q1",
) -> Dict[str, Any]:
    keys = list(keys) if keys is not None else [chr(65 + i) for i in range(len(options))]
    return {
        "state": {} if state is None else state,
        "questions": {qid: {"type": "choice", "instructions": instructions, "criteria": dict(zip(keys, options))}},
    }


def noul_request(instructions: str = "Is the sky blue?", state: Any = None) -> Dict[str, Any]:
    return {"state": {} if state is None else state,
            "questions": {"q1": {"type": "noul", "instructions": instructions, "criteria": {"true": "yes", "false": "no"}}}}


def score_request(levels: Sequence[str] = ("bad", "ok", "good"), state: Any = "a movie review") -> Dict[str, Any]:
    return {"state": state, "questions": {"q1": {"type": "score", "instructions": "Rate it.", "criteria": list(levels)}}}


def multi_request() -> Dict[str, Any]:
    req = choice_request()
    req["questions"]["q2"] = {"type": "noul", "instructions": "Is it in Europe?"}
    req["questions"]["q3"] = {"type": "score", "instructions": "How big?", "criteria": ["small", "medium", "large"]}
    return req


def kit_row(row_id: str, request: Dict[str, Any], expected: Optional[Dict[str, str]] = None, dataset: str = "Toy") -> Dict[str, Any]:
    """A row in the decision-index suite format (state, questions, expected, _evaluation)."""
    return {
        "id": row_id, "family": dataset, "split": "test", "state": request["state"], "questions": request["questions"],
        "expected": expected or {}, "metadata": {}, "scoring": {}, "_evaluation": {"dataset": dataset, "run_id": row_id},
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as fh:  # type: ignore[call-overload]
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[call-overload]
        return [json.loads(line) for line in fh if line.strip()]


def board_index(scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """A minimal board index.json: engine -> {v, same_skills, new_domains, pub}."""
    return {"models": [{"engine": engine, "name": engine, "meta": {"kind": "full fine-tune"}, "v03": dict(v, engine=engine)}
                       for engine, v in scores.items()]}
