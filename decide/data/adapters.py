"""Adapters: one dataset row -> ``(request, gold, native_id)`` in the request schema of SCHEMA.md, or ``None``.

An adapter is a pure function of the row, its ``args`` (from sources.yaml) and an :class:`AdapterContext`;
anything random derives from ``(seed, source, row index)`` so a rebuild gives the same corpus.  Rows that do not
fit (empty text, label out of range, duplicate options, ...) return ``None`` and are counted by the builder.
Instructions are written for this project; they do not copy the wording of the public suite.
"""
from __future__ import annotations

import bisect
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import schema as S

Converted = Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class AdapterContext:
    source: str
    index: int
    seed: int = 0
    features: Mapping[str, Any] = field(default_factory=dict)

    def rng(self, *salt: Any) -> random.Random:
        material = f"{self.seed}:{self.source}:{self.index}:" + ":".join(str(s) for s in salt)
        return random.Random(int(S.digest(material)[:16], 16))


# --------------------------------------------------------------------------- helpers

def _get(row: Mapping[str, Any], path: Any) -> Any:
    if not isinstance(path, str):
        return None
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ("" if value is None else str(value).strip())


def _keys(count: int) -> List[str]:
    return list(_LETTERS[:count]) if count <= len(_LETTERS) else [f"option_{i}" for i in range(count)]


def _native(row: Mapping[str, Any], args: Mapping[str, Any]) -> Optional[str]:
    value = _get(row, args.get("id"))
    return None if value is None or value == "" else str(value)


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _class_names(feature: Any) -> Optional[List[str]]:
    """ClassLabel names from a Hugging Face ``features`` entry (plain, sequence or list forms)."""
    if isinstance(feature, Mapping):
        if "names" in feature:
            return [str(n) for n in feature["names"]]
        for inner in ("feature", "inner"):
            if inner in feature:
                return _class_names(feature[inner])
    if isinstance(feature, list) and feature:
        return _class_names(feature[0])
    return None


def _single_choice_request(instructions: str, texts: Sequence[str], state: Any = None) -> Tuple[Dict[str, Any], List[str]]:
    keys = _keys(len(texts))
    return {"state": {} if state in (None, "") else state,
            "questions": {"q1": {"type": "choice", "instructions": instructions, "criteria": dict(zip(keys, texts))}}}, keys


# --------------------------------------------------------------------------- adapters

def mcq(row: Mapping[str, Any], args: Mapping[str, Any], ctx: AdapterContext) -> Optional[Converted]:
    """Multiple choice. args: question, choices (field | [fields]), answer, answer_kind index|letter|text,
    answer_offset (subtracted from index answers), context (optional field -> state), id."""
    question = _text(_get(row, args.get("question")))
    spec = args.get("choices")
    labels: Optional[List[str]] = None
    if isinstance(spec, (list, tuple)):
        raw: Any = [_get(row, f) for f in spec]
    else:
        raw = _get(row, spec)
        if isinstance(raw, Mapping) and "text" in raw:
            labels = [str(x).strip() for x in (raw.get("label") or [])] or None
            raw = raw["text"]
    if not question or not isinstance(raw, (list, tuple)):
        return None
    texts = [_text(t) for t in raw]
    if not 2 <= len(texts) <= S.MAX_OPTIONS or any(not t for t in texts) or len(set(texts)) != len(texts):
        return None
    answer = _get(row, args.get("answer"))
    kind = args.get("answer_kind", "index")
    idx: Optional[int] = None
    if kind == "index":
        n = _int(answer)
        idx = None if n is None else n - int(args.get("answer_offset", 0))
    elif kind == "letter":
        wanted = _text(answer).upper()
        pool = [x.upper() for x in labels] if labels else list(_LETTERS)
        idx = pool.index(wanted) if wanted in pool else None
    elif kind == "text":
        idx = texts.index(_text(answer)) if _text(answer) in texts else None
    if idx is None or not 0 <= idx < len(texts):
        return None
    context = _text(_get(row, args.get("context"))) if args.get("context") else ""
    request, keys = _single_choice_request(question, texts, context)
    return request, {"q1": keys[idx]}, _native(row, args)


def classification(row: Mapping[str, Any], args: Mapping[str, Any], ctx: AdapterContext) -> Optional[Converted]:
    """Text classification as one choice question; options are the class names (keys option_0, option_1, ...).
    args: text, label (int | name | one-element list), label_names (list | "auto"), instruction with {text}, id."""
    text = _text(_get(row, args.get("text")))
    label = _get(row, args.get("label"))
    if isinstance(label, (list, tuple)):
        if len(label) != 1:
            return None
        label = label[0]
    names = args.get("label_names")
    if names == "auto":
        names = _class_names(ctx.features.get(args.get("label")))
    if not text or label is None or not isinstance(names, (list, tuple)) or not 2 <= len(names) <= S.MAX_OPTIONS:
        return None
    names = [str(n) for n in names]
    idx = names.index(label) if isinstance(label, str) and label in names else _int(label)
    if idx is None or not 0 <= idx < len(names):
        return None
    instruction = str(args.get("instruction", "{text}")).replace("{text}", text)
    keys = [f"option_{i}" for i in range(len(names))]
    request = {"state": {}, "questions": {"q1": {"type": "choice", "instructions": instruction, "criteria": dict(zip(keys, names))}}}
    return request, {"q1": keys[idx]}, _native(row, args)


def nli(row: Mapping[str, Any], args: Mapping[str, Any], ctx: AdapterContext) -> Optional[Converted]:
    """Premise / hypothesis -> entailment, neutral or contradiction (keys A, B, C).  args: premise, hypothesis,
    label (name or index), labels, id."""
    premise, hypothesis = _text(_get(row, args.get("premise"))), _text(_get(row, args.get("hypothesis")))
    labels = [str(x) for x in args.get("labels", ["entailment", "neutral", "contradiction"])]
    label = _get(row, args.get("label"))
    idx = labels.index(label) if isinstance(label, str) and label in labels else _int(label)
    if not premise or not hypothesis or idx is None or not 0 <= idx < len(labels):
        return None
    instruction = f"How does the hypothesis relate to the premise?\nPremise: {premise}\nHypothesis: {hypothesis}"
    request, keys = _single_choice_request(instruction, labels)
    return request, {"q1": keys[idx]}, _native(row, args)


def binary(row: Mapping[str, Any], args: Mapping[str, Any], ctx: AdapterContext) -> Optional[Converted]:
    """A yes/no (noul) question about a text.  args: text (-> state), label, positive (value) or threshold, question, id."""
    text = _text(_get(row, args.get("text")))
    label = _get(row, args.get("label"))
    question = _text(args.get("question"))
    if not text or label is None or not question:
        return None
    if "positive" in args:
        truth = label == args["positive"] or str(label) == str(args["positive"])
    else:
        try:
            truth = float(label) >= float(args.get("threshold", 0.5))
        except (TypeError, ValueError):
            return None
    request = {"state": text, "questions": {"q1": {"type": "noul", "instructions": question,
                                                   "criteria": {"true": str(args.get("true_text", "Yes")), "false": str(args.get("false_text", "No"))}}}}
    return request, {"q1": "true" if truth else "false"}, _native(row, args)


_FINAL = re.compile(r"####\s*(-?[\d,]*\.?\d+)")


def _final_number(answer: Any) -> Optional[str]:
    m = _FINAL.search(answer) if isinstance(answer, str) else None
    if not m:
        return None
    number = float(m.group(1).replace(",", ""))
    return str(int(number)) if number == int(number) else repr(number)


def prepare_gsm8k(rows: Iterable[Mapping[str, Any]], args: Mapping[str, Any]) -> Dict[str, Any]:
    """Pre-pass: the pool of final answers that other problems borrow their distractors from."""
    pool = {n for n in (_final_number(_get(r, args.get("answer"))) for r in rows) if n is not None}
    return {"pool": sorted(pool, key=float)}


def gsm8k(row: Mapping[str, Any], args: Mapping[str, Any], ctx: AdapterContext) -> Optional[Converted]:
    """Math word problem as a numeric multiple choice.  Distractors are the final answers of *other* problems of
    similar size (never computed from the gold), the gold position is random.  args: question, answer, options, pool."""
    question, gold = _text(_get(row, args.get("question"))), _final_number(_get(row, args.get("answer")))
    count = int(args.get("options", 4))
    pool = args.get("pool") or []
    if not question or gold is None or count < 2:
        return None
    pairs = args.get("_pairs")
    if pairs is None:
        pairs = sorted({(float(x), x) for x in pool})
        try:
            args["_pairs"] = pairs  # type: ignore[index]  # cache on the (mutable) args dict
        except TypeError:
            pass
    values = [v for v, _ in pairs]
    pos = bisect.bisect_left(values, float(gold))
    for width in (15, 60, len(pairs)):
        window = [x for v, x in pairs[max(0, pos - width): pos + width + 1] if v != float(gold)]
        if len(window) >= count - 1:
            break
    if len(window) < count - 1:
        return None
    rng = ctx.rng("gsm8k")
    options = rng.sample(window, count - 1) + [gold]
    rng.shuffle(options)
    request, keys = _single_choice_request(question, options)
    return request, {"q1": keys[options.index(gold)]}, _native(row, args)


Adapter = Callable[[Mapping[str, Any], Mapping[str, Any], AdapterContext], Optional[Converted]]
ADAPTERS: Dict[str, Adapter] = {"mcq": mcq, "classification": classification, "nli": nli, "binary": binary, "gsm8k": gsm8k}
PREPARERS: Dict[str, Callable[[Iterable[Mapping[str, Any]], Mapping[str, Any]], Dict[str, Any]]] = {"gsm8k": prepare_gsm8k}
