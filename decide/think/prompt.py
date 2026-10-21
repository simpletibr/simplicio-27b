"""What the thinker is asked: state, questions and labeled options, and the reply format.

The Jev request schema is `{"state": ..., "questions": {qid: {"type": "choice", "instructions": ...,
"criteria": {key: description}}}}` (see decide/data/SCHEMA.md).  A decision model reads it in one
pass; the thinker gets the same content as plain text and may reason before replying.

Options get positional letters A, B, C, ... when there are at most 26 of them, because the keys of
the suite are not made to be typed back (`option_3`, `(B)`).  A yes/no or true/false pair keeps its
own words, and a long list (every legal chess move) keeps its keys.  The reader in `answers.py` maps
whatever the model writes back to the original key, which is what the gold and the decision models
use.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_WORD_PAIRS = ({"yes", "no"}, {"true", "false"})


class UnsupportedQuestion(ValueError):
    """The request holds a question type the thinker does not answer (only `choice` is supported)."""


@dataclass(frozen=True)
class Option:
    label: str   # what the model is asked to write
    key: str     # the key of the suite / of the decision models
    text: str    # the description shown next to the label


@dataclass(frozen=True)
class Prompt:
    text: str
    options: Dict[str, List[Option]]


def option_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict) and set(value) == {"uci", "san"}:
        return str(value["san"]).strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    return str(value)


def option_list(criteria: Any) -> List[Option]:
    if not isinstance(criteria, dict) or not criteria:
        raise UnsupportedQuestion("a choice question needs a non-empty criteria object")
    keys = [str(k) for k in criteria]
    use_letters = len(keys) <= len(LETTERS) and {k.casefold() for k in keys} not in _WORD_PAIRS
    labels = list(LETTERS[: len(keys)]) if use_letters else keys
    return [Option(label=lab, key=key, text=option_text(val)) for lab, key, val in zip(labels, keys, criteria.values())]


def render_state(state: Any) -> str:
    if state is None or state == "" or state == {} or state == []:
        return ""
    if isinstance(state, str):
        return state.strip()
    if isinstance(state, dict):
        lines = []
        for k, v in state.items():
            if isinstance(v, str):
                lines.append(f"{k}:\n{v.strip()}" if "\n" in v.strip() else f"{k}: {v.strip()}")
            else:
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        return "\n".join(lines)
    return json.dumps(state, ensure_ascii=False, indent=2)


def _option_line(o: Option) -> str:
    text = o.text.rstrip().replace("\n", "\n   ")
    if text.casefold() == o.label.casefold():
        return f"- {o.label}"
    return f"{o.label}. {text}"


INTRO = ("Answer the test item below. You may think for as long as you need before you reply; "
         "only the final reply lines are read.")


def build_prompt(request: Dict[str, Any]) -> Prompt:
    """Turn a request into the thinker prompt and the option tables the reader needs."""
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("request has no questions")
    sections = [INTRO]
    state = render_state(request.get("state"))
    if state:
        sections.append("[Context]\n" + state)
    options: Dict[str, List[Option]] = {}
    for qid, q in questions.items():
        if (q.get("type") or "choice") != "choice":
            raise UnsupportedQuestion(f"question {qid!r} is of type {q.get('type')!r}; only 'choice' is supported")
        opts = option_list(q.get("criteria"))
        options[qid] = opts
        block = [f"[Question {qid}]"]
        instructions = (q.get("instructions") or "").strip()
        if instructions:
            block.append(instructions)
        block.append("Options:")
        block.extend(_option_line(o) for o in opts)
        sections.append("\n".join(block))
    first_q = next(iter(options))
    example = options[first_q][0].label
    sections.append(
        "[Reply format]\n"
        "Think first. Any instruction above that says not to reason or not to explain applies to your visible reply only; "
        "reasoning inside your thinking block is always allowed.\n"
        "When you have decided, end your visible reply with exactly one line per question, in this form:\n"
        "ANSWER <question id>: <label>\n"
        "Use one of the labels shown next to the options. For example:\n"
        f"ANSWER {first_q}: {example}"
    )
    return Prompt(text="\n\n".join(sections), options=options)
