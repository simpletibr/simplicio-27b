"""Simplicio-Loop traces -> decision examples (PT-BR).

The traces are the owner's own synthetic data (license ``owner``).  Two tasks are derived from each trace:

* ``next_point``: which description is the right one for point N of the 50-point loop, given the task, the code
  context and the points so far?  Distractors are the same point of *other* traces (hardest by word overlap).
* ``patch_fits``: does this SEARCH/REPLACE patch belong to this task?  Negatives are patches of other traces.

Nothing here executes the traces; they are text.  Rows keep the extra key ``task`` (the builder drops it).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import schema as S
from . import text as T

_TAG = re.compile(r"^</?(simplicio_loop|orient|plan|patch|validate|deliver)>$")
_OPEN = re.compile(r"^<(orient|plan|patch|validate|deliver)>$")
_POINT = re.compile(r"^\[(?:Ponto|Point) (\d{1,2}): ([^\]]+)\](?: (.*))?$")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_trajectory(text: str) -> List[Dict[str, Any]]:
    """Points of a loop envelope: ``{n, name, text, phase}``; patch blocks stay inside the text of their point."""
    points: List[Dict[str, Any]] = []
    phase: Optional[str] = None
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        opened = _OPEN.match(stripped)
        if opened:
            phase = opened.group(1)
            continue
        if _TAG.match(stripped):
            continue
        m = _POINT.match(stripped)
        if m:
            points.append({"n": int(m.group(1)), "name": m.group(2), "text": (m.group(3) or "").strip(), "phase": phase})
        elif points:
            points[-1]["text"] = (points[-1]["text"] + "\n" + line).strip()
    return points


def _label(point: Dict[str, Any]) -> str:
    return f"[Ponto {point['n']}: {point['name']}] {point['text']}".strip()


def _overlap(a: str, b: str) -> float:
    wa, wb = set(T.words(a)), set(T.words(b))
    return len(wa & wb) / max(1, len(wa | wb))


def _letters(options: Sequence[str]) -> Dict[str, str]:
    return dict(zip(LETTERS, options))


def _row(task: str, example_id: str, request: Dict[str, Any], gold: Dict[str, Any], id_prefix: str) -> Dict[str, Any]:
    return {"example_id": example_id, "source": "loop", "license": "owner", "lang": "pt", "request": request, "gold": gold, "task": task}


def convert_records(records: Sequence[Dict[str, Any]], *, seed: int = 0, per_record: int = 2, id_prefix: str = "loop",
                    distractors: int = 3) -> List[Dict[str, Any]]:
    """Decision rows for every record (``instruction``, ``context``, ``simplicio_trajectory``); deterministic in ``seed``."""
    parsed = [parse_trajectory(r.get("simplicio_trajectory", "")) for r in records]
    by_n: Dict[int, List[Tuple[int, str]]] = {}
    for i, points in enumerate(parsed):
        for p in points:
            by_n.setdefault(p["n"], []).append((i, p["text"]))
    rows: List[Dict[str, Any]] = []
    for i, (record, points) in enumerate(zip(records, parsed)):
        rng = random.Random(f"{seed}:{id_prefix}:{i}")
        rid = str(record.get("id", i))
        state_base = {"instruction": record.get("instruction", ""), "context": record.get("context", "")}
        eligible = []
        for k, p in enumerate(points):
            alts = sorted({t for j, t in by_n[p["n"]] if j != i and t and t != p["text"]})
            if p["text"] and len(alts) >= distractors:
                eligible.append((k, alts))
        rng.shuffle(eligible)
        for k, alts in eligible[:per_record]:
            p = points[k]
            pool = rng.sample(alts, min(len(alts), 30))
            hard = sorted(pool, key=lambda t: -_overlap(t, p["text"]))[:distractors]
            options = hard + [p["text"]]
            rng.shuffle(options)
            state = dict(state_base, trajetoria_ate_aqui="\n".join(_label(q) for q in points[:k]))
            question = {"type": "choice", "instructions": f"Qual descricao corresponde ao ponto {p['n']} ({p['name']}) do loop de 50 pontos para esta tarefa?",
                        "criteria": _letters(options)}
            request = {"state": state, "questions": {"q1": question}}
            gold = {"q1": LETTERS[options.index(p["text"])]}
            rows.append(_row("next_point", f"{id_prefix}:{rid}:next_point:{k}", request, gold, id_prefix))
        patch = next((p["text"] for p in points if p["phase"] == "patch" and "<<<< SEARCH" in p["text"]), None)
        others = sorted({q["text"] for j, ps in enumerate(parsed) if j != i for q in ps
                         if q["phase"] == "patch" and "<<<< SEARCH" in q["text"] and q["text"] != patch})
        if patch and others and per_record > 0:
            for rep in range(min(per_record, 2)):
                positive = rng.random() < 0.5
                shown = patch if positive else rng.choice(others)
                request = {"state": dict(state_base, patch=shown),
                           "questions": {"q1": {"type": "noul", "instructions": "Este patch SEARCH/REPLACE resolve a tarefa descrita?",
                                                "criteria": {"true": "Sim, o patch resolve a tarefa.", "false": "Nao, o patch e de outra tarefa."}}}}
                rows.append(_row("patch_fits", f"{id_prefix}:{rid}:patch_fits:{rep}", request, {"q1": "true" if positive else "false"}, id_prefix))
    for row in rows:
        problems = S.validate_request(row["request"]) or S.validate_gold(row["request"], row["gold"])
        if problems:
            raise S.SchemaError(f"{row['example_id']}: {problems}")
    return rows


def convert_file(path: Path, *, seed: int = 0, per_record: int = 2, id_prefix: str = "loop") -> List[Dict[str, Any]]:
    records = []
    with open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return convert_records(records, seed=seed, per_record=per_record, id_prefix=id_prefix)
