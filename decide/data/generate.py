"""Optional generator of new-domain examples (PT-BR / EN).  OFF by default.

The generator is provider-agnostic: a *provider* is anything with ``name`` and ``complete(prompt, seed=None) -> str``.
Ready-made ones: ``EchoProvider`` (tests), ``CommandProvider`` (any CLI that reads the prompt on stdin and writes
the reply on stdout) and ``OpenAICompatibleProvider`` (any ``/chat/completions`` endpoint; the key comes from an
environment variable, never from a file).  The terms of use of the provider you choose govern the generated text:
the corpus row records the provider in ``source`` and uses the license ``provider-terms``.

Generated items are never trusted: every reply is parsed as JSON, converted and validated like any other row, and
bad ones are counted, not raised.  Decontamination still runs on them.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from . import schema as S

DEFAULT_ENABLED = False
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class GeneratorDisabled(RuntimeError):
    """generate_items() was called while the generator is off (``data.generator.enabled`` / ``enabled=True``)."""


class Provider(Protocol):
    name: str

    def complete(self, prompt: str, seed: Optional[int] = None) -> str: ...


@dataclass(frozen=True)
class TaskSpec:
    domain: str
    lang: str          # "pt" or "en"
    qtype: str         # choice | noul | score
    description: str
    n_options: int = 4


class EchoProvider:
    name = "echo"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, prompt: str, seed: Optional[int] = None) -> str:
        return self.reply


class CommandProvider:
    """Runs ``argv`` (no shell) with the prompt on stdin; the reply is stdout."""

    def __init__(self, argv: Sequence[str], name: str = "command", timeout: float = 300.0) -> None:
        self.argv, self.name, self.timeout = list(argv), name, timeout

    def complete(self, prompt: str, seed: Optional[int] = None) -> str:
        done = subprocess.run(self.argv, input=prompt, capture_output=True, text=True, timeout=self.timeout, check=True)
        return done.stdout


LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


class OpenAICompatibleProvider:
    """Any ``/chat/completions`` endpoint.  The bearer key goes over https only (a local server on this machine may use http)."""

    def __init__(self, base_url: str, model: str, api_key_env: str = "GENERATOR_API_KEY", name: str = "openai-compatible",
                 timeout: float = 120.0) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS):
            raise ValueError(f"base_url must be https:// (http:// only for {', '.join(LOCAL_HOSTS)}): the API key would travel in clear text; got {base_url!r}")
        self.base_url, self.model, self.api_key_env, self.name, self.timeout = base_url.rstrip("/"), model, api_key_env, name, timeout

    def complete(self, prompt: str, seed: Optional[int] = None) -> str:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.9}
        if seed is not None:
            body["seed"] = seed
        key = os.environ.get(self.api_key_env, "")
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - URL is chosen by the operator
            return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]


_SHAPES = {
    "choice": '{"state": "...", "question": "...", "options": ["...", "..."], "answer": <index of the correct option>}',
    "noul": '{"state": "...", "question": "...", "answer": true or false}',
    "score": '{"state": "...", "question": "...", "levels": ["lowest", "...", "highest"], "answer": <index of the level>}',
}


def build_prompt(spec: TaskSpec, n: int, seed: int) -> str:
    language = {"pt": "Brazilian Portuguese", "en": "English"}.get(spec.lang, spec.lang)
    return (
        f"Write {n} different decision examples in {language} for the domain '{spec.domain}': {spec.description}.\n"
        f"Each example is a realistic situation (the state) and one {spec.qtype} question about it with exactly one defensible answer.\n"
        f"Invent the scenarios; do not copy any benchmark or textbook. Vary length, difficulty and wording (variation seed {seed}).\n"
        f"Reply with a JSON array only, no commentary. Each element has this shape:\n{_SHAPES[spec.qtype]}"
    )


def parse_reply(reply: str) -> Optional[List[Any]]:
    text = reply.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except ValueError:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except ValueError:
            return None
    return value if isinstance(value, list) else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def item_to_request(spec: TaskSpec, item: Any) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if not isinstance(item, dict):
        return None
    state, question = _text(item.get("state")), _text(item.get("question"))
    if not state or not question:
        return None
    answer = item.get("answer")
    if spec.qtype == "noul":
        if not isinstance(answer, bool):
            return None
        req = {"state": state, "questions": {"q1": {"type": "noul", "instructions": question}}}
        return req, {"q1": "true" if answer else "false"}
    key = "options" if spec.qtype == "choice" else "levels"
    values = item.get(key)
    if not isinstance(values, list) or not 2 <= len(values) <= 10:
        return None
    texts = [_text(v) for v in values]
    if any(not t for t in texts) or len(set(texts)) != len(texts):
        return None
    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(texts):
        return None
    if spec.qtype == "choice":
        req = {"state": state, "questions": {"q1": {"type": "choice", "instructions": question, "criteria": dict(zip(LETTERS, texts))}}}
        return req, {"q1": LETTERS[answer]}
    req = {"state": state, "questions": {"q1": {"type": "score", "instructions": question, "criteria": texts}}}
    return req, {"q1": str(answer)}


def generate_items(spec: TaskSpec, provider: Provider, *, n: int, seed: int, enabled: bool = DEFAULT_ENABLED,
                   batch_size: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Ask the provider for ``n`` items (in ``ceil(n / batch_size)`` calls); returns (rows, stats).

    Rows have example_id, source, license, lang, request, gold.  ``stats``: calls, accepted, rejected, duplicates.
    """
    if not enabled:
        raise GeneratorDisabled("the generator is off; set data.generator.enabled: true and pass enabled=True to use it")
    stats = {"calls": 0, "accepted": 0, "rejected": 0, "duplicates": 0}
    rows: List[Dict[str, Any]] = []
    seen = set()
    source = f"gen:{provider.name}:{spec.domain}"
    remaining, call = n, 0
    while remaining > 0:
        want = min(batch_size, remaining)
        remaining -= want
        stats["calls"] += 1
        items = parse_reply(provider.complete(build_prompt(spec, want, seed + call), seed=seed + call))
        call += 1
        if items is None:
            stats["rejected"] += 1
            continue
        for item in items[:want]:
            converted = item_to_request(spec, item)
            if converted is None or S.validate_request(converted[0]) or S.validate_gold(*converted):
                stats["rejected"] += 1
                continue
            request, gold = converted
            digest = S.request_digest(request)
            if digest in seen:
                stats["duplicates"] += 1
                continue
            seen.add(digest)
            stats["accepted"] += 1
            rows.append({"example_id": f"{source}:{digest[:16]}", "source": source, "license": "provider-terms",
                         "lang": spec.lang, "request": request, "gold": gold})
    return rows, stats


def provider_from_config(cfg: Mapping[str, Any]) -> "Provider":
    """The provider named by the ``data.generator`` section of config.data.yaml (``command`` or ``openai-compatible``)."""
    kind = cfg.get("provider")
    if kind == "command":
        argv = cfg.get("command")
        if not isinstance(argv, (list, tuple)) or not argv or not all(isinstance(a, str) and a for a in argv):
            raise ValueError("data.generator.command must be a non-empty list of strings: the program and its arguments (the prompt goes to stdin)")
        return CommandProvider(argv, name="command")
    if kind == "openai-compatible":
        if not cfg.get("base_url") or not cfg.get("model"):
            raise ValueError("data.generator needs base_url and model for the openai-compatible provider")
        return OpenAICompatibleProvider(str(cfg["base_url"]), str(cfg["model"]), api_key_env=str(cfg.get("api_key_env") or "GENERATOR_API_KEY"))
    raise ValueError(f"data.generator.provider must be 'command' or 'openai-compatible' (got {kind!r})")


def generate_rows(cfg: Mapping[str, Any], *, seed: int, provider: Optional["Provider"] = None) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    """Rows of every task of the ``data.generator`` section, grouped by source name (``gen:<provider>:<domain>``).

    The section must say ``enabled: true`` (``GeneratorDisabled`` otherwise).  The same request produced by two tasks is
    kept once.  ``stats``: calls, accepted (rows kept), rejected, duplicates."""
    if not cfg.get("enabled"):
        raise GeneratorDisabled("the generator is off: set data.generator.enabled: true in the config (and pass --generate)")
    tasks = cfg.get("tasks") or []
    if not tasks:
        raise ValueError("data.generator.tasks is empty: list the domains to generate ({domain, lang, qtype, description})")
    provider = provider or provider_from_config(cfg)
    totals = {"calls": 0, "accepted": 0, "rejected": 0, "duplicates": 0}
    seen: set = set()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for i, task in enumerate(tasks):
        missing = [k for k in ("domain", "lang", "qtype", "description") if not isinstance(task, Mapping) or not task.get(k)]
        if missing or task["qtype"] not in _SHAPES:
            raise ValueError(f"data.generator.tasks[{i}] needs domain, lang, qtype ({'|'.join(_SHAPES)}) and description (missing: {missing or 'a valid qtype'})")
        spec = TaskSpec(domain=str(task["domain"]), lang=str(task["lang"]), qtype=str(task["qtype"]), description=str(task["description"]),
                        n_options=int(task.get("n_options", 4)))
        rows, stats = generate_items(spec, provider, n=int(cfg.get("per_task", 200)), seed=seed + 1000 * i, enabled=True)
        totals["calls"] += stats["calls"]
        totals["rejected"] += stats["rejected"]
        totals["duplicates"] += stats["duplicates"]
        for row in rows:
            if row["example_id"] in seen:
                totals["duplicates"] += 1
                continue
            seen.add(row["example_id"])
            totals["accepted"] += 1
            out.setdefault(row["source"], []).append(row)
    return out, totals
