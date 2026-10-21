"""Read the thinker's reply: split off the reasoning, then find one option per question.

The prompt asks for `ANSWER <question id>: <label>` lines.  Models drift, so the reader accepts the
usual variants, but it never guesses from the reasoning: only the text after the last `</think>` is
read, and a reply that cannot be read is `None`, which the comparison counts as wrong.

Order of the attempts for one question:
  1. `ANSWER <qid>: <value>` (also `ANSWER[qid]`, `ANSWER qid = value`, with markdown around it; the
     value may be on the next line).  If the reply has such a line, the LAST one decides, and if its
     value is not an option the question is unanswered: a model that ends on an unreadable line has
     not answered, and an earlier line or a bare label in the reasoning would be a guess;
  2. only when there is no such line: a line that starts with the question id (`option_3: yes`,
     `- option_3 - yes`), when the request has more than one question (again the last one decides);
  3. a single-question request only: the last valid of `answer is <value>`, `answer: <value>`,
     `\\boxed{<value>}`, and as the last resort the last bare label in the reply.
`extract_detailed` says which of these found each answer (`marker`, `line`, `generic`, `boxed`,
`bare`); everything but `marker` is a fallback and the report counts the rows that needed one.
A value is a label (`B`, `(B)`, `B)`, `yes`, `e2e4`), a key (`option_2`) or the exact text of one
option (`Nf3`, `17`).  A lowercase single letter is a label only with a delimiter (`a)`), because
`a` is an article.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .prompt import Option

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_reasoning(text: Optional[str]) -> Tuple[str, Optional[str]]:
    """(reasoning, final).  `final` is None when the thinking block was never closed."""
    if text is None:
        return "", None
    i = text.rfind(THINK_CLOSE)
    if i >= 0:
        reasoning = text[:i]
        j = reasoning.find(THINK_OPEN)
        if j >= 0:
            reasoning = reasoning[j + len(THINK_OPEN):]
        return reasoning.strip(), text[i + len(THINK_CLOSE):].strip()
    j = text.find(THINK_OPEN)
    if j >= 0:
        return text[j + len(THINK_OPEN):].strip(), None
    return "", text.strip()


_LATEX = re.compile(r"\\(?:boxed|text|textbf|mathrm|mathbf)\s*\{([^{}]*)\}")
_FILLER = re.compile(r"^(?:option|choice|answer|letter|is|the|correct|it'?s|therefore|thus|so)\b[\s:,\-\u2013\u2014]*", re.I)
_LETTER = re.compile(r"^[\(\[]?([A-Za-z])(?:([\)\]\.:])|(?![A-Za-z0-9_]))")
_EDGE = "*_`~\"'\u201c\u201d\u2018\u2019 \t\r\n"


def _clean(value: str) -> str:
    s = _LATEX.sub(r"\1", value)
    s = s.strip().strip("$").strip()
    return s.strip(_EDGE)


def _norm(text: str) -> str:
    return " ".join(text.split())


def match_value(value: str, options: List[Option]) -> Optional[Option]:
    """Map the text after `ANSWER <qid>:` to one option, or None."""
    s = _clean(value)
    while s:
        m = _FILLER.match(s)
        if not m:
            break
        s = _clean(s[m.end():])
    if not s:
        return None
    letters = {o.label: o for o in options if len(o.label) == 1 and o.label.isalpha()}
    if letters:
        m = _LETTER.match(s)
        if m and (m.group(1).isupper() or m.group(2) or not s[m.end():].strip()):
            hit = letters.get(m.group(1).upper())
            if hit:
                return hit
    words = sorted((o for o in options if o.label not in letters), key=lambda o: -len(o.label))
    for o in words:
        if re.match(re.escape(o.label) + r"(?![A-Za-z0-9_])", s, re.I):
            return o
    first = s.split()[0].strip(",.;:)(") if s.split() else ""
    for o in options:
        if len(o.key) > 1 and first and first.casefold() == o.key.casefold():   # one-letter keys are the letter rule's job
            return o
    for pred in (lambda o: _norm(o.text) == _norm(s.rstrip(".")),
                 lambda o: _norm(o.text).casefold() == _norm(s.rstrip(".")).casefold()):
        hits = [o for o in options if pred(o)]
        if len(hits) == 1:
            return hits[0]
    return None


def _last_valid(pattern: "re.Pattern[str]", text: str, options: List[Option]) -> Optional[Option]:
    for m in reversed(list(pattern.finditer(text))):
        hit = match_value(m.group("val"), options)
        if hit:
            return hit
    return None


_GENERIC = re.compile(r"(?i)\banswer\b[^\n:=]{0,30}?(?:\bis\b|[:=\-\u2013\u2014])\s*(?P<val>[^\n]*)")
_BOXED = re.compile(r"\\boxed\s*\{(?P<val>[^{}]*)\}")


def _standalone(text: str, options: List[Option]) -> Optional[Option]:
    best, best_pos = None, -1
    for o in options:
        letter = len(o.label) == 1 and o.label.isalpha()
        pat = r"(?<![A-Za-z0-9_])" + re.escape(o.label) + r"(?![A-Za-z0-9_])"
        for m in re.finditer(pat, text, 0 if letter else re.I):
            if m.start() > best_pos:
                best, best_pos = o, m.start()
    return best


def _decisive(pattern: "re.Pattern[str]", text: str, options: List[Option]) -> Tuple[bool, Optional[Option]]:
    """(there is such a line, the option its LAST occurrence names).  An empty value reads the next non-empty line."""
    found = list(pattern.finditer(text))
    if not found:
        return False, None
    m = found[-1]
    value = m.group("val")
    if not value.strip():
        value = next((line for line in text[m.end():].split("\n") if line.strip()), "")
    return True, match_value(value, options)


def _find(final: str, qid: str, options: List[Option], single: bool) -> Tuple[Optional[Option], Optional[str]]:
    q = re.escape(qid)
    marked = re.compile(r"(?i)\banswer\s*[\[\(]?\s*" + q + r"(?![\w])\s*[\]\)]?\s*[:=\-\u2013\u2014]\s*(?P<val>[^\n]*)")
    present, hit = _decisive(marked, final, options)
    if present:
        return hit, "marker"
    if not single:
        line = re.compile(r"(?im)^[\s>*_`#\-\u2022\d.)]*" + q + r"(?![\w])[\s*_`]*[:=\-\u2013\u2014][ \t]*(?P<val>[^\n]*)$")
        present, hit = _decisive(line, final, options)
        return (hit, "line") if present else (None, None)
    hit = _last_valid(_GENERIC, final, options)
    if hit:
        return hit, "generic"
    hit = _last_valid(_BOXED, final, options)
    if hit:
        return hit, "boxed"
    hit = _standalone(final, options)
    return (hit, "bare") if hit else (None, None)


def extract_detailed(final: Optional[str], options: Dict[str, List[Option]]) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """qid -> (original option key or None, how it was found: marker, line, generic, boxed, bare, or None)."""
    if not final:
        return {qid: (None, None) for qid in options}
    single = len(options) == 1
    out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for qid, opts in options.items():
        hit, how = _find(final, qid, opts, single)
        out[qid] = (hit.key if hit else None, how)
    return out


def extract(final: Optional[str], options: Dict[str, List[Option]]) -> Dict[str, Optional[str]]:
    """qid -> original option key, or None when the reply holds no valid answer for it."""
    return {qid: key for qid, (key, _how) in extract_detailed(final, options).items()}


def extract_from_text(text: Optional[str], options: Dict[str, List[Option]]) -> Dict[str, Optional[str]]:
    """Split the reasoning off `text` and read the answers from what follows it."""
    _, final = split_reasoning(text)
    return extract(final, options)
