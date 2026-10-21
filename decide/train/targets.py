"""Training targets: a configurable combiner of teacher distributions, and the mix with the gold label.

The target is NOT hard-wired to "mean of the teachers" (phase 0A: equal-weight averaging did not beat the best single
member).  A combiner JSON, produced upstream (see decide/data/SCHEMA.md, section 5), says how to combine:

    single           one teacher's distribution
    mean             weighted arithmetic mean of the probabilities (weights normalized)
    calibrated_mean  each teacher is first brought to its own ``temperature`` (p' ~ p^(T_stored / T)), then the mean
    logpool          weighted log-linear pool, p ~ prod p_i^w_i (weights used as given, result renormalized)

with overrides per question type (``choice``, ``noul``, ``score``).  Without a combiner the labels may hold exactly
one teacher; with two or more the builder refuses and lists their names.  The final target is
``alpha * gold + (1 - alpha) * combined`` (the combined distribution alone when there is no gold label, the gold
alone when there is no teacher label).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

from .codes import ANSWER_CODES

VERSION = 1
METHODS = ("single", "mean", "calibrated_mean", "logpool")
QUESTION_TYPES = ("choice", "noul", "score")
MISSING_POLICIES = ("error", "drop", "renormalize")
LOG_FLOOR = 1e-12
_TOP_KEYS = {"version", "method", "teacher", "teachers", "by_question_type", "meta", "comment"}
_TYPE_KEYS = {"method", "teacher", "teachers"}
_TEACHER_KEYS = {"weight", "temperature"}


class TargetError(ValueError):
    """The labels, the combiner or the mixing parameters are not usable."""


class MissingTeacher(TargetError):
    """A teacher the combiner needs has no label for this question."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = sorted(names)
        super().__init__(f"no label from teacher(s) {', '.join(self.names)}")


@dataclass(frozen=True)
class TeacherLabel:
    probs: Tuple[float, ...]
    temperature: float


def _positive(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise TargetError(f"{what} must be a positive finite number, got {value!r}")
    return float(value)


def normalize(values: Sequence[float]) -> list:
    total = float(sum(values))
    if not math.isfinite(total) or total <= 0:
        raise TargetError("a distribution needs positive finite mass")
    return [float(v) / total for v in values]


def recalibrate(probs: Sequence[float], stored_temperature: float, new_temperature: float) -> list:
    """Bring a distribution produced at ``stored_temperature`` to ``new_temperature``: p' ~ p^(T / T')."""
    exponent = _positive(stored_temperature, "temperature") / _positive(new_temperature, "temperature")
    return normalize([p ** exponent if p > 0 else 0.0 for p in probs])


@dataclass(frozen=True)
class _Spec:
    method: str
    teacher: Optional[str]
    teachers: Dict[str, Tuple[float, Optional[float]]]      # name -> (weight, temperature)

    def used(self) -> Set[str]:
        """The teachers this spec actually reads."""
        if self.method == "single":
            return {self.teacher} if self.teacher else set()
        return set(self.teachers)


def _teachers_block(raw: Any, where: str, inherited: Optional[Mapping[str, Tuple[float, Optional[float]]]] = None) -> Dict[str, Tuple[float, Optional[float]]]:
    if not isinstance(raw, Mapping):
        raise TargetError(f"{where}: 'teachers' must be an object (teacher name -> {{weight, temperature}})")
    out: Dict[str, Tuple[float, Optional[float]]] = {}
    for name, entry in raw.items():
        if not isinstance(name, str) or not name:
            raise TargetError(f"{where}: teacher names must be non-empty strings")
        if not isinstance(entry, Mapping):
            raise TargetError(f"{where}: teacher {name!r} must be an object")
        unknown = set(entry) - _TEACHER_KEYS
        if unknown:
            raise TargetError(f"{where}: teacher {name!r} has unknown field(s) {sorted(unknown)}")
        base_weight, base_temperature = (inherited or {}).get(name, (1.0, None))
        weight = _positive(entry["weight"], f"{where}: weight of {name!r}") if "weight" in entry else base_weight
        temperature = _positive(entry["temperature"], f"{where}: temperature of {name!r}") if "temperature" in entry else base_temperature
        out[name] = (weight, temperature)
    return out


def _check_spec(spec: _Spec, where: str) -> _Spec:
    if spec.method not in METHODS:
        raise TargetError(f"{where}: method must be one of {METHODS}, got {spec.method!r}")
    if spec.method == "single":
        if not isinstance(spec.teacher, str) or not spec.teacher:
            raise TargetError(f"{where}: method 'single' needs a 'teacher'")
    else:
        if not spec.teachers:
            raise TargetError(f"{where}: method {spec.method!r} needs a non-empty 'teachers' object")
        if spec.method == "calibrated_mean":
            lacking = sorted(name for name, (_, t) in spec.teachers.items() if t is None)
            if lacking:
                raise TargetError(f"{where}: calibrated_mean needs a 'temperature' for {', '.join(lacking)}")
    return spec


class Combiner:
    """Turns the labels of several teachers for one question into one distribution."""

    def __init__(self, top: _Spec, by_type: Mapping[str, _Spec], sha256: str) -> None:
        self._top, self._by_type, self.sha256 = top, dict(by_type), sha256

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, obj: Any, *, sha256: Optional[str] = None) -> "Combiner":
        if not isinstance(obj, Mapping):
            raise TargetError("the combiner must be a JSON object")
        unknown = set(obj) - _TOP_KEYS
        if unknown:
            raise TargetError(f"unknown combiner field(s): {sorted(unknown)}")
        if obj.get("version") != VERSION or isinstance(obj.get("version"), bool):
            raise TargetError(f"combiner version must be {VERSION}, got {obj.get('version')!r}")
        method = obj.get("method")
        top_teachers = _teachers_block(obj["teachers"], "top level") if "teachers" in obj else {}
        top = _check_spec(_Spec(method, obj.get("teacher"), top_teachers), "top level")
        resolved: Dict[str, _Spec] = {}
        for qtype, override in (obj.get("by_question_type") or {}).items():
            if qtype not in QUESTION_TYPES:
                raise TargetError(f"by_question_type: unknown question type {qtype!r} (use one of {QUESTION_TYPES})")
            if not isinstance(override, Mapping):
                raise TargetError(f"by_question_type.{qtype} must be an object")
            extra = set(override) - _TYPE_KEYS
            if extra:
                raise TargetError(f"by_question_type.{qtype}: unknown field(s) {sorted(extra)}")
            teachers = (_teachers_block(override["teachers"], f"by_question_type.{qtype}", inherited=top.teachers)
                        if "teachers" in override else top.teachers)
            resolved[qtype] = _check_spec(_Spec(override.get("method", top.method), override.get("teacher", top.teacher), dict(teachers)),
                                          f"by_question_type.{qtype}")
        digest = sha256 or hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        return cls(top, resolved, digest)

    @classmethod
    def load(cls, path: "str | Path") -> "Combiner":
        raw = Path(path).read_bytes()
        try:
            obj = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise TargetError(f"{path}: not valid JSON ({exc})") from exc
        return cls.from_dict(obj, sha256=hashlib.sha256(raw).hexdigest())

    @classmethod
    def single(cls, teacher: str) -> "Combiner":
        return cls.from_dict({"version": VERSION, "method": "single", "teacher": teacher})

    # ------------------------------------------------------------------ queries
    def teachers(self) -> Set[str]:
        """Every teacher name the combiner can use, for any question type."""
        names = set(self._top.used())
        for spec in self._by_type.values():
            names |= spec.used()
        return names

    def spec(self, question_type: str) -> _Spec:
        if question_type not in QUESTION_TYPES:
            raise TargetError(f"unknown question type {question_type!r}")
        return self._by_type.get(question_type, self._top)

    # ------------------------------------------------------------------ combination
    def combine(self, question_type: str, labels: Mapping[str, TeacherLabel], *, missing: str = "error") -> Optional[list]:
        """The combined distribution, or None when ``missing='drop'`` and a needed teacher has no label."""
        if missing not in MISSING_POLICIES:
            raise TargetError(f"missing must be one of {MISSING_POLICIES}")
        spec = self.spec(question_type)
        needed = [spec.teacher] if spec.method == "single" else list(spec.teachers)
        absent = [name for name in needed if name not in labels]
        if absent:
            if missing == "error" or (missing == "renormalize" and len(absent) == len(needed)):
                raise MissingTeacher(absent)
            if missing == "drop":
                return None
        present = [name for name in needed if name in labels]
        widths = {len(labels[name].probs) for name in present}
        if len(widths) != 1:
            raise TargetError(f"teachers disagree on the number of options: {sorted(widths)}")
        if spec.method == "single":
            return normalize(labels[present[0]].probs)
        weights = {name: spec.teachers[name][0] for name in present}
        if spec.method == "logpool":
            logits = [sum(weights[name] * math.log(max(labels[name].probs[i], LOG_FLOOR)) for name in present) for i in range(widths.pop())]
            top = max(logits)
            return normalize([math.exp(z - top) for z in logits])
        total = sum(weights.values())
        dists = {}
        for name in present:
            label = labels[name]
            target_t = spec.teachers[name][1]
            dists[name] = recalibrate(label.probs, label.temperature, target_t) if spec.method == "calibrated_mean" and target_t is not None else label.probs
        return normalize([sum(weights[name] * dists[name][i] for name in present) / total for i in range(len(dists[present[0]]))])


def resolve_combiner(path: "str | Path | None", available: Set[str]) -> Combiner:
    """The combiner to use given the teachers found in the labels."""
    if path is None:
        if len(available) == 1:
            return Combiner.single(next(iter(available)))
        if not available:
            raise TargetError("no teacher labels were found")
        raise TargetError("no combiner given but the labels hold several teachers: "
                          f"{', '.join(sorted(available))}; pass --combiner with a JSON that says how to combine them "
                          "(or point --labels at a single teacher)")
    combiner = Combiner.load(path)
    unknown = combiner.teachers() - set(available)
    if unknown:
        raise TargetError(f"the combiner uses teacher(s) {', '.join(sorted(unknown))} with no labels (found: {', '.join(sorted(available)) or 'none'})")
    return combiner


def mix_gold(combined: Optional[Sequence[float]], gold_index: Optional[int], n: int, alpha: float) -> Optional[list]:
    """``alpha * onehot(gold) + (1 - alpha) * combined``; either side alone when the other is absent."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise TargetError(f"alpha must be in [0, 1], got {alpha!r}")
    if combined is not None and len(combined) != n:
        raise TargetError(f"combined distribution has {len(combined)} entries, the question has {n} options")
    if gold_index is not None and not 0 <= gold_index < n:
        raise TargetError(f"gold index {gold_index} outside 0..{n - 1}")
    if gold_index is None and combined is None:
        return None
    if combined is None:
        return [1.0 if i == gold_index else 0.0 for i in range(n)]
    if gold_index is None:
        return list(combined)
    return [alpha * (1.0 if i == gold_index else 0.0) + (1.0 - alpha) * combined[i] for i in range(n)]


def label_from_row(row: Mapping[str, Any], n: int) -> TeacherLabel:
    """Validate one teacher-label row (SCHEMA.md section 4) against the question's option count."""
    from decide.data import schema as S

    problems = S.validate_teacher_row(row)
    where = f"{row.get('example_id')}/{row.get('question_id')} ({row.get('teacher')})"
    if problems:
        raise TargetError(f"{where}: " + "; ".join(problems))
    codes = list(row["codes"])
    if codes != list(ANSWER_CODES[:n]):
        raise TargetError(f"{where}: codes {codes[:4]}... are not the first {n} answer codes {list(ANSWER_CODES[:n])[:4]}... "
                          "(labels must be in the original option order)")
    return TeacherLabel(tuple(float(p) for p in row["probs"]), float(row["temperature"]))
