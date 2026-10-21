"""Run Qwen3.8-27B with thinking on the sample pack: one result line per row, resumable.

Two backends, one runner:

* `openai`: any OpenAI-compatible chat endpoint (a local `vllm serve`, a hosted API).  Standard
  library only.  The base URL and the key come from the environment (THINK_BASE_URL or
  OPENAI_BASE_URL; THINK_API_KEY or OPENAI_API_KEY); the key is never printed or written.
* `transformers`: the model in this process, NF4 weights with fp16 compute (a free Kaggle T4).  Batches
  of consecutive rows; a CUDA out-of-memory error halves the batch.

The prompt shows the state, the question and labeled options and asks for `ANSWER <id>: <label>`
(`prompt.py`).  The reply is read after the last `</think>`; a reply that cannot be read is `None`,
which `compare.py` counts as wrong.  A generation that hit the token budget while still thinking is
`truncated` and never read as an answer.

The results file is JSONL, append-only, one fsync per row.  The first line is a header with the
configuration and the pack hash.  Run it again with the same arguments and it continues where it
stopped: finished rows are skipped, `error` rows (a flaky endpoint, an out-of-memory) are retried, a
row cut by the budget is redone only if the budget went up.  A different model, prompt, sampling or
pack is refused, so one results file is one protocol.

    python -m decide.think.thinker --pack pack.jsonl.gz --out results.jsonl --backend openai
    python -m decide.think.thinker --pack pack.jsonl.gz --out results.jsonl --backend transformers
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import common as C
from .answers import THINK_CLOSE, extract, split_reasoning
from .prompt import Option, Prompt, UnsupportedQuestion, build_prompt

DEFAULT_MAX_TOKENS = 16_384
INSTRUCT_SAMPLING = {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
FINAL_TAIL_CHARS = 2000
FINGERPRINT_EXCLUDE = ("max_tokens", "seed", "endpoint")      # the budget is handled on its own; the others do not change the protocol
USER_AGENT = "simplicio-think-gt/1"
DTYPE_SHORT = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}


def nf4_config_kwargs(compute_dtype: Any) -> Dict[str, Any]:
    """Arguments of the BitsAndBytesConfig of the NF4 backend: 4-bit NormalFloat weights, double quantization, matmuls in `compute_dtype`."""
    return {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": compute_dtype, "bnb_4bit_use_double_quant": True}


class BackendError(RuntimeError):
    """The backend cannot work as configured (bad key, unknown model, rejected parameter): stop the run."""


class ResumeMismatch(RuntimeError):
    """The results file was made with another configuration or pack, or is not a results file."""


# ------------------------------------------------------------------------------ one generation

@dataclass(frozen=True)
class Item:
    id: str
    prompt: str
    seed: Optional[int] = None


@dataclass
class GenResult:
    reasoning: Optional[str] = None          # the thinking, when the server returns it apart from the reply
    content: Optional[str] = None            # the reply (with <think> tags when the server did not split them)
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    seconds: float = 0.0
    error: Optional[str] = None
    transient: bool = True                   # for an error: True = retry later (`error`), False = this row cannot be answered (`failed`)


@dataclass
class Parsed:
    reasoning: str
    final: Optional[str]
    answers: Dict[str, Optional[str]]
    truncated: bool


def interpret(gen: GenResult, options: Dict[str, List[Option]]) -> Parsed:
    """Split the thinking from the reply and read the answers.  Thinking is never read as an answer."""
    content = gen.content if isinstance(gen.content, str) else ""
    truncated = gen.finish_reason == "length"
    if gen.reasoning is not None:                           # the server already split the thinking off
        reasoning = gen.reasoning
        _, final = split_reasoning(content)                 # a stray </think> inside the reply is still honored
    else:
        reasoning, final = split_reasoning(content)
        if truncated and THINK_CLOSE not in content:        # the template opens <think> itself: a cut reply is all thinking
            reasoning, final = content, None
    final = final or None
    return Parsed(reasoning=reasoning, final=final, answers=extract(final, options), truncated=truncated)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_record(row: dict, prompt: Prompt, gen: GenResult, *, attempt: int, max_tokens: int, keep_reasoning: bool) -> dict:
    base = {"kind": "row", "id": row["id"], "benchmark": int(row["benchmark"]), "attempt": attempt, "max_tokens": max_tokens, "time": now_iso()}
    nothing = {qid: None for qid in prompt.options}
    if gen.error:
        return {**base, "status": "error" if gen.transient else "failed", "error": str(gen.error)[:500], "answers": nothing,
                "truncated": False, "finish_reason": gen.finish_reason, "seconds": round(gen.seconds, 3)}
    p = interpret(gen, prompt.options)
    rec = {**base, "status": "ok", "answers": p.answers, "finish_reason": gen.finish_reason, "truncated": p.truncated,
           "tokens": {"prompt": gen.prompt_tokens, "completion": gen.completion_tokens, "reasoning": gen.reasoning_tokens},
           "seconds": round(gen.seconds, 3), "final": p.final[-FINAL_TAIL_CHARS:] if p.final else None}
    if keep_reasoning:
        rec["reasoning"] = p.reasoning
    return rec


def row_seed(seed: Optional[int], row_id: str) -> Optional[int]:
    """A sampling seed per row, stable across resumes and reorderings."""
    if seed is None:
        return None
    return int(hashlib.sha256(f"{seed}|{row_id}".encode()).hexdigest()[:8], 16) % (2 ** 31)


def fingerprint_of(config: Dict[str, Any]) -> str:
    return C.fingerprint({k: v for k, v in config.items() if k not in FINGERPRINT_EXCLUDE})


def sanitize_url(url: Optional[str]) -> Optional[str]:
    """scheme://host[:port]/path without credentials, query or fragment."""
    if not url:
        return None
    u = urllib.parse.urlsplit(url)
    host = u.hostname or ""
    netloc = host + (f":{u.port}" if u.port else "")
    return urllib.parse.urlunsplit((u.scheme, netloc, u.path, "", ""))


# --------------------------------------------------------------------------------- the results file

class ResultStore:
    """Append-only JSONL: a header line, then one line per attempt of a row."""

    def __init__(self, path: Path, config: Dict[str, Any], header: dict, seen: int):
        self.path, self.config, self.header, self._seen = path, config, header, seen

    @staticmethod
    def read(path: Any) -> Tuple[Optional[dict], List[dict]]:
        header, rows = None, []
        for rec in C.iter_jsonl(path):
            if rec.get("kind") == "header":
                header = header or rec
            elif rec.get("kind") == "row":
                rows.append(rec)
        return header, rows

    @classmethod
    def open(cls, path: Any, config: Dict[str, Any], pack_sha256: str, rows_in_pack: int, overwrite: bool = False) -> "ResultStore":
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fp = fingerprint_of(config)
        header = {"kind": "header", "format": C.RESULT_FORMAT, "created": now_iso(), "config": config, "config_fingerprint": fp,
                  "pack_content_sha256": pack_sha256, "rows_in_pack": rows_in_pack, "prompt_version": C.PROMPT_VERSION,
                  "notice": C.EVAL_ONLY_NOTICE}
        if overwrite or not p.exists() or p.stat().st_size == 0:
            if p.exists():
                p.unlink()
            C.append_jsonl(p, header)
            return cls(p, config, header, int(config.get("max_tokens") or 0))
        C.repair_jsonl(p)
        old, rows = cls.read(p)
        if old is None or old.get("format") != C.RESULT_FORMAT:
            raise ResumeMismatch(f"{p} is not a {C.RESULT_FORMAT} results file; use another --out or --overwrite")
        problems = []
        if old.get("config_fingerprint") != fp:
            before, now = old.get("config") or {}, config
            keys = sorted(k for k in set(before) | set(now) if k not in FINGERPRINT_EXCLUDE and before.get(k) != now.get(k))
            problems.append("config differs in " + ", ".join(f"{k} ({str(before.get(k))[:60]} -> {str(now.get(k))[:60]})" for k in keys))
        if old.get("pack_content_sha256") != pack_sha256:
            problems.append("the pack differs (content sha256 is not the one of the first run)")
        seen = max([int((old.get("config") or {}).get("max_tokens") or 0)] + [int(r.get("max_tokens") or 0) for r in rows])
        if int(config.get("max_tokens") or 0) < seen:
            problems.append(f"max_tokens went down ({seen} -> {config.get('max_tokens')}): one file, one budget; it may only go up")
        if problems:
            raise ResumeMismatch(f"{p}: cannot resume: " + "; ".join(problems) + ". Use another --out, or --overwrite to start over.")
        return cls(p, config, old, max(seen, int(config.get("max_tokens") or 0)))

    def append(self, rec: dict) -> None:
        C.append_jsonl(self.path, rec)

    def records(self) -> List[dict]:
        return self.read(self.path)[1]

    def latest(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for r in self.records():
            out[r["id"]] = r
        return out

    def attempts(self) -> Dict[str, int]:
        return dict(Counter(r["id"] for r in self.records()))

    def max_tokens_seen(self) -> int:
        return self._seen

    def done_ids(self, max_tokens: int) -> Set[str]:
        """Rows that need no more work under this budget."""
        out = set()
        for rid, rec in self.latest().items():
            if rec.get("status") not in C.TERMINAL_STATUSES:
                continue
            if rec.get("truncated") and int(rec.get("max_tokens") or 0) < max_tokens:
                continue
            out.add(rid)
        return out


# ---------------------------------------------------------------------------------------- the runner

@dataclass
class RunStats:
    total: int = 0
    done: int = 0
    ok: int = 0
    failed: int = 0
    errors: int = 0
    unparseable: int = 0
    truncated: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    stopped: Optional[str] = None


def _fmt_duration(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def run(rows: Sequence[dict], backend: Any, store: ResultStore, *, max_tokens: int, limit: Optional[int] = None,
        time_budget_s: Optional[float] = None, benchmarks: Optional[Iterable[int]] = None, keep_reasoning: bool = True,
        log: Callable[[str], None] = print, clock: Callable[[], float] = time.monotonic, stop: Optional[Callable[[], bool]] = None,
        max_consecutive_errors: int = 8, seed: Optional[int] = None) -> RunStats:
    """Run the pending rows (pack order) through the backend; every finished row is on disk before the next starts."""
    done = store.done_ids(max_tokens)
    keep = None if benchmarks is None else {int(b) for b in benchmarks}
    pending = [r for r in rows if r["id"] not in done and (keep is None or int(r["benchmark"]) in keep)]
    if limit is not None:
        pending = pending[:limit]
    attempts = store.attempts()
    stats = RunStats(total=len(pending))
    prompts: Dict[str, Prompt] = {}
    runnable: List[dict] = []
    for r in pending:
        try:
            prompts[r["id"]] = build_prompt({"state": r.get("state"), "questions": r["questions"]})
            runnable.append(r)
        except (UnsupportedQuestion, ValueError) as e:
            store.append({"kind": "row", "id": r["id"], "benchmark": int(r["benchmark"]), "attempt": attempts.get(r["id"], 0) + 1, "max_tokens": max_tokens,
                          "time": now_iso(), "status": "failed", "error": f"cannot build a prompt: {e}"[:500], "answers": {}, "truncated": False,
                          "finish_reason": None, "seconds": 0.0})
            stats.done += 1
            stats.failed += 1
    size = max(1, int(getattr(backend, "batch_size", 1)))
    workers = max(1, int(getattr(backend, "workers", 1)))
    chunks = [runnable[i:i + size] for i in range(0, len(runnable), size)]
    start = clock()
    state = {"consecutive": 0}

    def why_stop() -> Optional[str]:
        if stop is not None and stop():
            return "stop requested"
        if time_budget_s is not None and clock() - start >= time_budget_s:
            return "time budget"
        if state["consecutive"] >= max_consecutive_errors:
            return f"{max_consecutive_errors} consecutive errors"
        return None

    def items_of(chunk: List[dict]) -> List[Item]:
        return [Item(r["id"], prompts[r["id"]].text, row_seed(seed, r["id"])) for r in chunk]

    def finish(chunk: List[dict], outcome: Any) -> None:
        if isinstance(outcome, BaseException):
            gens = [GenResult(error=f"{type(outcome).__name__}: {outcome}"[:300], transient=True)] * len(chunk)
        elif not isinstance(outcome, list) or len(outcome) != len(chunk):
            gens = [GenResult(error="backend returned the wrong number of results", transient=True)] * len(chunk)
        else:
            gens = outcome
        for r, gen in zip(chunk, gens):
            rec = make_record(r, prompts[r["id"]], gen, attempt=attempts.get(r["id"], 0) + 1, max_tokens=max_tokens, keep_reasoning=keep_reasoning)
            store.append(rec)
            stats.done += 1
            stats.seconds += gen.seconds or 0.0
            if rec["status"] == "error":
                stats.errors += 1
                state["consecutive"] += 1
            else:
                state["consecutive"] = 0
                if rec["status"] == "failed":
                    stats.failed += 1
                else:
                    stats.ok += 1
                    stats.completion_tokens += int(gen.completion_tokens or 0)
                    stats.truncated += bool(rec["truncated"])
                    stats.unparseable += any(v is None for v in rec["answers"].values())
            elapsed = clock() - start
            eta = f" ETA {_fmt_duration(elapsed / stats.done * (stats.total - stats.done))}" if elapsed > 0 and stats.total > stats.done else ""
            rate = f" {stats.completion_tokens / elapsed:.1f} tok/s" if elapsed > 0 and stats.completion_tokens else ""
            flags = "".join(f" {name}" for name, on in (("truncated", rec.get("truncated")),
                                                       ("unparseable", rec["status"] == "ok" and any(v is None for v in rec["answers"].values()))) if on)
            tok = (rec.get("tokens") or {}).get("completion")
            log(f"[think] {stats.done}/{stats.total} {rec['id']} {rec['status']}{flags}"
                + (f" {tok} tok" if tok is not None else "") + f" {rec.get('seconds', 0):.0f}s |{rate}{eta}")

    if workers == 1:
        for chunk in chunks:
            reason = why_stop()
            if reason:
                stats.stopped = reason
                break
            try:
                outcome: Any = backend.generate(items_of(chunk))
            except BackendError:
                raise
            except Exception as e:                            # noqa: BLE001 - any backend failure becomes `error` rows
                outcome = e
            finish(chunk, outcome)
        return stats

    ex = ThreadPoolExecutor(max_workers=workers)
    inflight: Dict[Any, List[dict]] = {}
    fatal: Optional[BackendError] = None
    source = iter(chunks)
    drained = False
    try:
        while True:
            while not drained and fatal is None and len(inflight) < workers:
                reason = why_stop()
                if reason:
                    stats.stopped = stats.stopped or reason
                    drained = True
                    break
                chunk = next(source, None)
                if chunk is None:
                    drained = True
                    break
                inflight[ex.submit(backend.generate, items_of(chunk))] = chunk
            if not inflight:
                break
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in finished:
                chunk = inflight.pop(fut)
                try:
                    outcome = fut.result()
                except BackendError as e:
                    fatal = fatal or e
                    continue
                except Exception as e:                        # noqa: BLE001
                    outcome = e
                finish(chunk, outcome)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    if fatal is not None:
        raise fatal
    return stats


# --------------------------------------------------------------------------------- OpenAI-compatible

_ROW_ERRORS = ("context length", "context_length", "maximum context", "too long", "max_model_len", "max_tokens is too large",
               "reduce the length", "exceeds the model", "input is too long", "prompt is too long")


def classify_http_error(code: int, body: str) -> str:
    """`retry` (transient), `row` (this prompt cannot be answered) or `fatal` (the setup is wrong: stop)."""
    if code in (408, 425, 429) or code >= 500:
        return "retry"
    if code in (400, 413, 422) and any(s in body.lower() for s in _ROW_ERRORS):
        return "row"
    return "fatal"


class _Transient(Exception):
    pass


def _text_of(content: Any) -> Optional[str]:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):                             # content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


class OpenAIBackend:
    """Chat Completions over HTTP with the standard library: streaming, retries, no key in any output."""
    name = "openai"
    batch_size = 1

    def __init__(self, *, base_url: str, model: str, api_key: Optional[str] = None, sampling: Optional[Dict[str, Any]] = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS, chat_template_kwargs: Optional[Dict[str, Any]] = None,
                 reasoning_effort: Optional[str] = None, extra_body: Optional[Dict[str, Any]] = None, standard_only: bool = False,
                 stream: bool = True, timeout: float = 3600.0, retries: int = 4, concurrency: int = 8, revision: Optional[str] = None,
                 sleep: Callable[[float], None] = time.sleep, opener: Callable[..., Any] = urllib.request.urlopen,
                 log: Callable[[str], None] = lambda s: None):
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.model, self.api_key = model, api_key
        self.sampling = dict(C.THINKING_SAMPLING if sampling is None else sampling)
        self.max_tokens = max_tokens
        self.chat_template_kwargs = {"enable_thinking": True} if chat_template_kwargs is None else dict(chat_template_kwargs)
        self.reasoning_effort, self.extra_body, self.standard_only = reasoning_effort, dict(extra_body or {}), standard_only
        self.stream, self.timeout, self.retries = stream, timeout, retries
        self.workers = max(1, int(concurrency))
        self.sleep, self.opener, self.log = sleep, opener, log

    def describe(self) -> Dict[str, Any]:
        return {"endpoint": sanitize_url(self.base_url), "stream": self.stream, "concurrency": self.workers, "timeout": self.timeout, "retries": self.retries}

    def headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "User-Agent": USER_AGENT, "Accept": "text/event-stream" if self.stream else "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def payload(self, item: Item, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": self.model, "messages": [{"role": "user", "content": item.prompt}],
                                "max_tokens": self.max_tokens if max_tokens is None else max_tokens}
        nonstandard = ("top_k", "min_p", "repetition_penalty")
        for k, v in self.sampling.items():
            if v is not None and not (self.standard_only and k in nonstandard):
                body[k] = v
        if self.chat_template_kwargs and not self.standard_only:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if item.seed is not None:
            body["seed"] = item.seed
        if self.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        body.update(self.extra_body)
        return body

    # -- one request
    def _request(self, body: Dict[str, Any]) -> GenResult:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers=self.headers(), method="POST")
        t0 = time.monotonic()
        with self.opener(req, timeout=self.timeout) as resp:
            if self.stream:
                return self._read_stream(resp, t0)
            return self._read_json(resp.read(), t0)

    @staticmethod
    def _usage(g: GenResult, usage: Any) -> None:
        if isinstance(usage, dict):
            g.prompt_tokens = usage.get("prompt_tokens")
            g.completion_tokens = usage.get("completion_tokens")
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict):
                g.reasoning_tokens = details.get("reasoning_tokens")

    def _read_json(self, raw: bytes, t0: float) -> GenResult:
        try:
            obj = json.loads(raw)
            choice = obj["choices"][0]
            msg = choice.get("message") or {}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            raise _Transient(f"unreadable reply: {e}") from e
        g = GenResult(reasoning=msg.get("reasoning_content") or msg.get("reasoning"), content=_text_of(msg.get("content")),
                      finish_reason=choice.get("finish_reason"), seconds=time.monotonic() - t0)
        self._usage(g, obj.get("usage"))
        return g

    def _read_stream(self, resp: Any, t0: float) -> GenResult:
        reasoning: List[str] = []
        content: List[str] = []
        finish, usage, saw_done = None, None, False
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(data)
            except ValueError as e:
                raise _Transient(f"unreadable stream chunk: {e}") from e
            if isinstance(chunk, dict) and chunk.get("error"):
                raise _Transient(f"stream error: {str(chunk['error'])[:200]}")
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                r = delta.get("reasoning_content") or delta.get("reasoning")
                if r:
                    reasoning.append(r)
                c = _text_of(delta.get("content"))
                if c:
                    content.append(c)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if chunk.get("usage"):
                usage = chunk["usage"]
        if finish is None and not saw_done:
            raise _Transient("the stream ended before the reply was complete")
        g = GenResult(reasoning="".join(reasoning) if reasoning else None, content="".join(content), finish_reason=finish, seconds=time.monotonic() - t0)
        self._usage(g, usage)
        return g

    def _one(self, item: Item) -> GenResult:
        body = self.payload(item)
        last = "no attempt"
        for attempt in range(self.retries + 1):
            delay = min(60.0, 2.0 ** (attempt + 1))
            try:
                return self._request(body)
            except urllib.error.HTTPError as e:
                try:
                    text = e.read().decode("utf-8", "replace")[:500]
                except Exception:                              # noqa: BLE001
                    text = ""
                kind = classify_http_error(e.code, text)
                last = f"HTTP {e.code}: {text[:300]}"
                if kind == "fatal":
                    raise BackendError(f"{last} (check the base URL, the key, the model name and the parameters)") from e
                if kind == "row":
                    return GenResult(error=last, transient=False)
                try:
                    delay = min(120.0, float(e.headers.get("Retry-After"))) if e.headers and e.headers.get("Retry-After") else delay
                except (TypeError, ValueError):
                    pass
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException, _Transient) as e:
                last = f"{type(e).__name__}: {getattr(e, 'reason', e)}"[:300]
            if attempt < self.retries:
                self.log(f"[think] {item.id}: {last}; retry {attempt + 1}/{self.retries} in {delay:.0f}s")
                self.sleep(delay)
        return GenResult(error=last, transient=True)

    def generate(self, items: Sequence[Item]) -> List[GenResult]:
        return [self._one(i) for i in items]

    def preflight(self) -> GenResult:
        """One tiny request: a wrong URL, key, model name or parameter fails here instead of on row 1."""
        small = OpenAIBackend.payload(self, Item("preflight", "Reply with the single letter A."), max_tokens=16)
        small.pop("seed", None)
        req = urllib.request.Request(self.url, data=json.dumps(small).encode("utf-8"), headers=self.headers(), method="POST")
        try:
            with self.opener(req, timeout=min(self.timeout, 600)) as resp:
                return self._read_stream(resp, time.monotonic()) if self.stream else self._read_json(resp.read(), time.monotonic())
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")[:300] if hasattr(e, "read") else ""
            raise BackendError(f"preflight failed: HTTP {e.code}: {text}") from e
        except (urllib.error.URLError, OSError, http.client.HTTPException, _Transient) as e:
            raise BackendError(f"preflight failed: {type(e).__name__}: {getattr(e, 'reason', e)}") from e


# ------------------------------------------------------------------------------------ transformers

class TransformersBackend:
    """The model in this process.  NF4 weights with fp16 compute fit a Kaggle T4 x2; nothing here was run on a T4."""
    name = "transformers"
    workers = 1

    def __init__(self, *, model: str, revision: Optional[str] = None, quant: str = "nf4", batch_size: int = 1,
                 max_tokens: int = DEFAULT_MAX_TOKENS, sampling: Optional[Dict[str, Any]] = None,
                 chat_template_kwargs: Optional[Dict[str, Any]] = None, reasoning_effort: Optional[str] = None,
                 dtype: str = "float16", device_map: str = "auto", gpu_mem_gib: Optional[float] = None, model_class: str = "auto",
                 attn_implementation: Optional[str] = None, trust_remote_code: bool = False,
                 log: Callable[[str], None] = lambda s: None):
        import torch                                           # imported here: the rest of the package needs no torch
        import transformers
        self.torch = torch
        self.model_id, self.revision, self.quant = model, revision, quant
        self.batch_size, self.max_tokens, self.log = max(1, batch_size), max_tokens, log
        self.sampling = dict(C.THINKING_SAMPLING if sampling is None else sampling)
        self.template_kwargs = {"enable_thinking": True} if chat_template_kwargs is None else dict(chat_template_kwargs)
        if reasoning_effort:
            self.template_kwargs["reasoning_effort"] = reasoning_effort
        rev = {"revision": revision} if revision else {}
        self.tok = transformers.AutoTokenizer.from_pretrained(model, padding_side="left", trust_remote_code=trust_remote_code, **rev)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        config = transformers.AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code, **rev)
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        qc = None
        if quant == "nf4":
            qc = transformers.BitsAndBytesConfig(**nf4_config_kwargs(torch_dtype))      # the NF4 matmuls run in the dtype you chose, not always in fp16
        elif quant != "none":
            raise ValueError(f"unknown quantization {quant!r} (nf4 or none)")
        kwargs: Dict[str, Any] = {"device_map": device_map, "trust_remote_code": trust_remote_code, **rev}
        if qc is not None:
            kwargs["quantization_config"] = qc
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        if gpu_mem_gib and torch.cuda.is_available():
            kwargs["max_memory"] = {i: f"{gpu_mem_gib}GiB" for i in range(torch.cuda.device_count())}
        multimodal = any(a.endswith("ForConditionalGeneration") for a in (getattr(config, "architectures", None) or [])) or hasattr(config, "vision_config")
        order = {"causal": ["AutoModelForCausalLM"], "image-text": ["AutoModelForImageTextToText"],
                 "auto": ["AutoModelForImageTextToText", "AutoModelForCausalLM"] if multimodal else ["AutoModelForCausalLM", "AutoModelForImageTextToText"]}[model_class]
        last: Optional[Exception] = None
        self.model = None
        for name in order:
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                try:
                    self.model = cls.from_pretrained(model, dtype=torch_dtype, **kwargs)
                except TypeError:                              # an older transformers calls it torch_dtype
                    self.model = cls.from_pretrained(model, torch_dtype=torch_dtype, **kwargs)
                break
            except ValueError as e:                            # "unrecognized configuration class": try the next loader
                last = e
        if self.model is None:
            raise BackendError(f"could not load {model}: {last}")
        self.model.eval()
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        self.eos_ids = set(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos if gen_eos is not None else self.tok.eos_token_id])
        self.eos_ids.discard(None)

    def describe(self) -> Dict[str, Any]:
        return {"batch_size": self.batch_size, "quant": self.quant}

    def render(self, prompt: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **self.template_kwargs)

    def generate(self, items: Sequence[Item]) -> List[GenResult]:
        torch = self.torch
        try:
            return self._generate(list(items))
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            if len(items) == 1:
                return [GenResult(error=f"CUDA out of memory with one sequence and a budget of {self.max_tokens} tokens: {str(e)[:120]}", transient=True)]
            self.log(f"[think] out of memory with a batch of {len(items)}; splitting it")
            half = len(items) // 2
            return self.generate(items[:half]) + self.generate(items[half:])

    def _generate(self, items: List[Item]) -> List[GenResult]:
        torch, s = self.torch, self.sampling
        texts = [self.render(i.prompt) for i in items]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        if items[0].seed is not None:
            torch.manual_seed(items[0].seed)
        kwargs: Dict[str, Any] = {"max_new_tokens": self.max_tokens, "pad_token_id": self.tok.pad_token_id}
        temperature = s.get("temperature")
        if temperature:
            kwargs.update(do_sample=True, temperature=temperature)
            if s.get("top_p") is not None:
                kwargs["top_p"] = s["top_p"]
            if s.get("top_k"):
                kwargs["top_k"] = s["top_k"]
            if s.get("min_p"):
                kwargs["min_p"] = s["min_p"]
        else:
            kwargs["do_sample"] = False
        if s.get("repetition_penalty") not in (None, 1.0):
            kwargs["repetition_penalty"] = s["repetition_penalty"]
        t0 = time.monotonic()
        with torch.inference_mode():
            out = self.model.generate(**enc, **kwargs)
        wall = time.monotonic() - t0
        width = enc["input_ids"].shape[1]
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        results = []
        for k in range(len(items)):
            ids = out[k, width:].tolist()
            cut = next((j for j, t in enumerate(ids) if t in self.eos_ids), None)
            finish = "stop" if cut is not None else "length"
            if cut is not None:
                ids = ids[:cut]
            results.append(GenResult(content=self.tok.decode(ids, skip_special_tokens=False), finish_reason=finish, prompt_tokens=int(prompt_lens[k]),
                                     completion_tokens=len(ids), seconds=wall / len(items)))
        return results


# ----------------------------------------------------------------------------------------------- CLI

def _env_first(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


# The extra body is copied into the header of the results file and into the report, which people paste into issues.
SECRET_KEY = re.compile(r"^(api[-_]?key|x-api-key|secret|client[-_]?secret|password|authorization|(access|auth|bearer|id|refresh)[-_]?token|token)$", re.I)
SECRET_VALUE = re.compile(r"^(sk-[A-Za-z0-9_-]{16,}|Bearer\s+\S{8,}|hf_[A-Za-z0-9]{20,})")


def refuse_secrets(obj: Any, what: str, path: str = "") -> None:
    """Stop a key that was typed into a JSON option: it would be written to the results header and to the report."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if SECRET_KEY.match(str(k)):
                raise SystemExit(f"{what}: `{path}{k}` looks like a secret. Put the key in the environment (THINK_API_KEY), "
                                 "not in an option: this JSON is written to the results file and to the report.")
            refuse_secrets(v, what, f"{path}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            refuse_secrets(v, what, f"{path.rstrip('.')}[{i}].")
    elif isinstance(obj, str) and SECRET_VALUE.match(obj):
        raise SystemExit(f"{what}: the value at `{path.rstrip('.')}` looks like a key or a token. Put it in the environment (THINK_API_KEY), "
                         "not in an option: this JSON is written to the results file and to the report.")


def _json_arg(text: Optional[str], what: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except ValueError as e:
        raise SystemExit(f"{what} must be a JSON object: {e}")
    if not isinstance(obj, dict):
        raise SystemExit(f"{what} must be a JSON object")
    refuse_secrets(obj, what)
    return obj


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m decide.think.thinker", description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True, help="pack.jsonl.gz from sample.py (private)")
    ap.add_argument("--out", required=True, help="results.jsonl; run again with the same arguments to resume")
    ap.add_argument("--manifest", help="manifest.json: check the pack against it before running")
    ap.add_argument("--backend", choices=("openai", "transformers"), default="openai")
    ap.add_argument("--model", default=C.QWEN_MODEL)
    ap.add_argument("--revision", help=f"Hub revision. transformers: default is the pinned one of {C.QWEN_MODEL} ({C.QWEN_REVISION[:12]}...) or of its -FP8 twin; "
                                       "'none' reads the branch main. openai: not sent, only recorded, so declare what the server runs")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="generation budget per question, thinking included (default %(default)s)")
    ap.add_argument("--time-budget-s", type=float, help="stop starting new rows after this many seconds (a Kaggle session is 9-12 h)")
    ap.add_argument("--limit", type=int, help="run at most this many pending rows (use 3 first to time the run)")
    ap.add_argument("--benchmarks", help="comma-separated catalog ids to run (default: all in the pack)")
    ap.add_argument("--no-reasoning", action="store_true", help="do not store the thinking text in the results")
    ap.add_argument("--seed", type=int, help="sampling seed (a stable seed per row is derived from it)")
    ap.add_argument("--overwrite", action="store_true", help="start the results file over instead of resuming")
    ap.add_argument("--thinking", choices=("on", "off"), default="on", help="off = same model without thinking (a control)")
    ap.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), help="model default is xhigh")
    for k, v in C.THINKING_SAMPLING.items():
        ap.add_argument("--" + k.replace("_", "-"), type=float, help=f"default {v} thinking / {INSTRUCT_SAMPLING[k]} non-thinking")
    ap.add_argument("--max-consecutive-errors", type=int, default=8, help="abort after this many rows in a row end in an error")
    g = ap.add_argument_group("openai backend")
    g.add_argument("--base-url", help="e.g. http://127.0.0.1:8000/v1 (env THINK_BASE_URL or OPENAI_BASE_URL)")
    g.add_argument("--api-key-env", help="name of the env var that holds the key (default THINK_API_KEY, then OPENAI_API_KEY)")
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--timeout", type=float, default=3600.0)
    g.add_argument("--retries", type=int, default=4)
    g.add_argument("--no-stream", action="store_true")
    g.add_argument("--standard-only", action="store_true", help="send only OpenAI-standard fields (no top_k, min_p, repetition_penalty, chat_template_kwargs)")
    g.add_argument("--extra-body", help="JSON object merged into every request (provider-specific fields)")
    g.add_argument("--no-preflight", action="store_true")
    t = ap.add_argument_group("transformers backend")
    t.add_argument("--quant", choices=("nf4", "none"), default="nf4")
    t.add_argument("--batch-size", type=int, default=1)
    t.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16",
                   help="dtype of the weights that stay in 16 bits and of the NF4 matmuls (default float16: a T4 has no native bf16; "
                        "bfloat16 is the model's own dtype and runs emulated on a T4, slower, not tried). It is part of the protocol: another dtype needs another --out")
    t.add_argument("--device-map", default="auto")
    t.add_argument("--gpu-mem-gib", type=float, help="cap per GPU in GiB, to leave room for the cache")
    t.add_argument("--model-class", choices=("auto", "causal", "image-text"), default="auto")
    t.add_argument("--attn-implementation")
    t.add_argument("--trust-remote-code", action="store_true")
    return ap


def make_config(args: argparse.Namespace) -> Dict[str, Any]:
    thinking = args.thinking == "on"
    sampling = dict(C.THINKING_SAMPLING if thinking else INSTRUCT_SAMPLING)
    for k in C.THINKING_SAMPLING:
        v = getattr(args, k)
        if v is not None:
            sampling[k] = v
    revision = args.revision
    if revision is None and args.backend == "transformers":
        revision = C.PINNED_REVISIONS.get(args.model)          # an endpoint serves what it serves: its revision is only what you declare
    if revision == "none":
        revision = None
    endpoint = None
    if args.backend == "openai":
        endpoint = sanitize_url(args.base_url or _env_first("THINK_BASE_URL", "OPENAI_BASE_URL"))
    return {"backend": args.backend, "model": args.model, "revision": revision, "prompt_version": C.PROMPT_VERSION,
            "sampling": sampling, "chat_template_kwargs": {"enable_thinking": thinking}, "reasoning_effort": args.reasoning_effort,
            "extra_body": _json_arg(args.extra_body, "--extra-body"), "standard_only": bool(args.standard_only),
            "quantization": (f"nf4-{DTYPE_SHORT[args.dtype]}" if args.quant == "nf4" else "none") if args.backend == "transformers" else None,
            "dtype": args.dtype if args.backend == "transformers" else None,
            "max_tokens": args.max_tokens, "seed": args.seed, "endpoint": endpoint}


def make_backend(args: argparse.Namespace, config: Dict[str, Any], log: Callable[[str], None]) -> Any:
    if args.backend == "openai":
        base = args.base_url or _env_first("THINK_BASE_URL", "OPENAI_BASE_URL")
        if not base:
            raise SystemExit("openai backend: pass --base-url or set THINK_BASE_URL / OPENAI_BASE_URL")
        key = os.environ.get(args.api_key_env) if args.api_key_env else _env_first("THINK_API_KEY", "OPENAI_API_KEY")
        return OpenAIBackend(base_url=base, model=args.model, api_key=key, sampling=config["sampling"], max_tokens=args.max_tokens,
                             chat_template_kwargs=config["chat_template_kwargs"], reasoning_effort=args.reasoning_effort,
                             extra_body=config["extra_body"], standard_only=args.standard_only, stream=not args.no_stream, timeout=args.timeout,
                             retries=args.retries, concurrency=args.concurrency, log=log)
    return TransformersBackend(model=args.model, revision=config["revision"], quant=args.quant, batch_size=args.batch_size, max_tokens=args.max_tokens,
                               sampling=config["sampling"], chat_template_kwargs=config["chat_template_kwargs"], reasoning_effort=args.reasoning_effort,
                               dtype=args.dtype, device_map=args.device_map, gpu_mem_gib=args.gpu_mem_gib, model_class=args.model_class,
                               attn_implementation=args.attn_implementation, trust_remote_code=args.trust_remote_code, log=log)


def summarize(stats: RunStats, store: ResultStore, *, thinking: bool = True) -> str:
    latest = store.latest()
    by = Counter(r.get("status") for r in latest.values())
    trunc = sum(bool(r.get("truncated")) for r in latest.values())
    text = (f"[think] this run: {stats.done}/{stats.total} rows ({stats.ok} ok, {stats.failed} failed, {stats.errors} error), "
            f"{stats.completion_tokens} tokens, stopped: {stats.stopped or 'all pending rows done'}\n"
            f"[think] file now: {len(latest)} rows ({by.get('ok', 0)} ok, {by.get('failed', 0)} failed, {by.get('error', 0)} to retry), {trunc} cut by the budget")
    if thinking and stats.ok and stats.completion_tokens / stats.ok < C.MIN_THINKING_TOKENS:
        text += (f"\n[think] WARNING: {stats.completion_tokens / stats.ok:.0f} tokens per reply on average: the model barely thought. "
                 "Check that the endpoint has thinking on (README, Backends); a run like this is not the GT protocol.")
    return text


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = lambda s: print(s, flush=True)  # noqa: E731
    from . import sample as S
    try:
        rows = S.read_pack(args.pack)
        if args.manifest:
            problems = S.verify(args.manifest, args.pack)
            if problems:
                for p in problems:
                    log(f"[think] manifest: {p}")
                return 2
        pack_sha = S.pack_content_sha256(args.pack)
        config = make_config(args)
        store = ResultStore.open(args.out, config, pack_sha, len(rows), overwrite=args.overwrite)
        backend = make_backend(args, config, log)
        if args.backend == "openai" and not args.no_preflight:
            pre = backend.preflight()
            log(f"[think] preflight ok ({backend.describe()['endpoint']}; thinking returned apart from the reply: {pre.reasoning is not None})")
    except (ResumeMismatch, BackendError, FileNotFoundError, ValueError) as e:
        log(f"[think] {e}")
        return 2
    stop_flag = threading.Event()

    def on_signal(signum: int, _frame: Any) -> None:
        if stop_flag.is_set():
            raise KeyboardInterrupt
        log("[think] stop requested: finishing the rows in flight (press again to quit now)")
        stop_flag.set()
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, on_signal)
        except ValueError:                                     # not the main thread: keep the default
            pass
    log(f"[think] {len(rows)} rows in the pack; {len(store.done_ids(args.max_tokens))} done; backend {args.backend}, model {args.model}, budget {args.max_tokens} tokens")
    try:
        stats = run(rows, backend, store, max_tokens=args.max_tokens, limit=args.limit, time_budget_s=args.time_budget_s,
                    benchmarks=[int(b) for b in args.benchmarks.split(",") if b] if args.benchmarks else None, keep_reasoning=not args.no_reasoning,
                    log=log, stop=stop_flag.is_set, max_consecutive_errors=args.max_consecutive_errors, seed=args.seed)
    except BackendError as e:
        log(f"[think] {e}")
        return 2
    finally:
        for sig, handler in previous.items():                  # leave the caller's Ctrl-C alone (a notebook kernel, a test runner)
            signal.signal(sig, handler)
    log(summarize(stats, store, thinking=args.thinking == "on"))
    return 3 if stats.stopped and "consecutive errors" in stats.stopped else 0


if __name__ == "__main__":
    sys.exit(main())
