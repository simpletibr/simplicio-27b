"""Recibo JSON dos critérios de aceite das issues #2 a #5, medidos contra um gateway ao vivo.

Verificador independente do servidor: só biblioteca padrão, nada importado de deploy/.
A chave de API entra só por SIMPLETI_API_KEY e nunca vai para o recibo, para o stdout nem para o stderr.
A URL-base entra por --base-url ou SIMPLETI_BASE_URL (sem credenciais, sem query).
Uma checagem que não consegue rodar (rede, 403, 404, exceção) é FAIL com o motivo, nunca PASS.

Uso: SIMPLETI_API_KEY=... python3 scripts/live_acceptance.py --base-url URL [--phase up|down]
Saída: 0 se todas as checagens passaram, 1 se alguma falhou, 2 se falta configuração
(chave, URL-base) ou o recibo não pôde ser gravado.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KEY_ENV = "SIMPLETI_API_KEY"
BASE_URL_ENV = "SIMPLETI_BASE_URL"
SCHEMA = "simplicio.live-acceptance/v1"
USER_AGENT = "simplicio-live-acceptance/1"
VALID_FINISH = ("stop", "length", "tool_calls")
MODEL = "simplicio-27b"
ALIAS = "simpleti/simplicio-27b"
OMIT_OVER = 2000  # prompts com mais caracteres que isto aparecem como "<N caracteres omitidos>"
MAX_TEXT = 20000  # corpo não-JSON guardado no recibo
MAX_ACTUAL_CONTENT = 500
MIN_EXCHANGES_200 = 10
FILLER = "O Simplicio 27B mede o contexto com este texto repetido.\n"
SYSTEM_AGENT = (
    "Você é um agente de código. Ferramentas disponíveis: bash (roda um comando no shell), "
    "read (lê um arquivo) e webfetch (baixa uma URL). Use uma ferramenta só quando o pedido precisar dela."
)
USER_WEBFETCH = "Use a ferramenta webfetch para abrir https://example.com e me diga o título da página."


def _tool(name: str, description: str, param: str) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {param: {"type": "string"}}, "required": [param]}}}


WEBFETCH = _tool("webfetch", "Fetch a URL and return its content as text.", "url")
TOOLS = [_tool("bash", "Run a shell command and return its output.", "command"),
         _tool("read", "Read a file from the workspace.", "path"), WEBFETCH]
SECOND_TURN = [
    {"role": "system", "content": SYSTEM_AGENT},
    {"role": "user", "content": USER_WEBFETCH},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "webfetch", "arguments": "{\"url\": \"https://example.com\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1",
     "content": "<html><head><title>Example Domain</title></head><body><h1>Example Domain</h1></body></html>"},
]

# Bullets de "## Aceite" das issues #2 a #5, copiados sem o prefixo "- ".
C2_IDS = "`model` `simplicio-27b` e `simpleti/simplicio-27b` respondem 200 em `/v1/chat/completions`."
C2_TOOL = (
    'Com a tool `webfetch` e `tool_choice: "auto"`, a resposta é 200 e traz `tool_calls`, '
    "não 400 e não um XML de tool dentro de `content`."
)
C2_NONE = (
    'Com `tool_choice: "none"` e o user `oi`, `finish_reason` é `stop` '
    "e a saudação está fora do canal de raciocínio."
)
C3_PONG = (
    "“Reply with exactly the single word pong” `max_tokens: 64` → `finish_reason: stop`, "
    "`content` é `pong`, raciocínio fora do `content`."
)
C3_AGENT = (
    "System prompt que só lista `bash` / `read` / `webfetch` e user `oi`, `max_tokens: 512` → "
    "ou uma saudação com `finish_reason: stop`, ou uma única tool call. "
    "O bloco `<tool>` não se repete e o completion não estoura 512."
)
C3_SECOND = (
    "O segundo turno de uma sessão que já executou um `webfetch` não reescreve o resultado da tool "
    "nem inventa `<system>Tool result`."
)
C4_PROMPT = (
    "`POST /v1/chat/completions` com um prompt de 31692 tokens de entrada, "
    "no processo lançado por `deploy/serve_colab.py`, responde 200."
)
C4_CEILING = "Um prompt acima do teto responde 4xx com o corpo de erro do vLLM, não 200 sem `usage`."
C5_FINISH = (
    "Uma completion 200 sempre traz `finish_reason` em `stop` | `length` | `tool_calls` e um objeto `usage`."
)
C5_DOWN = "Matar o processo do notebook tira o upstream. O request seguinte não é 200 com corpo vazio."

# id -> (issue, critério de aceite, "passa se")
META: dict[str, tuple[int, str, str]] = {
    "i2-id-simplicio-27b": (2, C2_IDS, "200 válido"),
    "i2-id-simpleti-simplicio-27b": (2, C2_IDS, "200 válido"),
    "i2-tool-auto": (2, C2_TOOL, (
        '200 válido; `finish_reason == "tool_calls"`; `tool_calls[0].function.name == "webfetch"`; '
        '`json.loads(arguments)["url"]` começa com `https://example.com` (não-JSON falha); '
        "content sem `<tool_call>` e sem `<function=`")),
    "i2-tool-none-oi": (2, C2_NONE, (
        "200 válido; `stop`; `content.strip()` não vazio; sem `<think>` e sem `</think>`")),
    "i3-pong": (3, C3_PONG, '200 válido; `stop`; `content.strip() == "pong"`'),
    "i3-agente-oi-system": (3, C3_AGENT, (
        '200 válido; `stop`; `usage.completion_tokens < 512`; `content.count("<tool") <= 1`')),
    "i3-agente-oi-tools": (3, C3_AGENT, (
        "200 válido; `completion_tokens < 512`; ou (`stop`, sem `tool_calls`, content não vazio e sem `<tool`) "
        "ou (exatamente 1 tool call)")),
    "i3-segundo-turno": (3, C3_SECOND, (
        "200 válido; `stop` ou `tool_calls`; `completion_tokens < 512`; no máximo 1 tool call; "
        "content sem `Tool result`, `<system`, `<tool_response>`, `<tool_call>`, `<function=`, `</think>`")),
    "i4-prompt-31692": (4, C4_PROMPT, (
        "200 válido; `MEASURED_AGENT_PROMPT <= usage.prompt_tokens <= MAX_MODEL_LEN - 16`")),
    "i4-acima-do-teto": (4, C4_CEILING, (
        "status 400–499 (menos 401, 403, 404 e 429, que não são erro de contexto) e corpo JSON com "
        "`error.message` string não vazia (formato `ErrorResponse` do vLLM 0.31.0); "
        "sem calibração, falha com `sem calibração` e sem request")),
    "i5-stream-include-usage": (5, C5_FINISH, "`stream_problems(ex) == []`"),
    "i5-stream-sem-options": (5, C5_FINISH, "`stream_problems(ex) == []`"),
    "i5-todo-200-valido": (5, C5_FINISH, (
        "toda troca com status 200 passa em `completion_problems` (JSON) ou `stream_problems` (SSE), "
        "e há >= 10 trocas 200")),
    "i5-upstream-fora": (5, C5_DOWN, "status 500–599 e `str(body).strip()` não vazio"),
}


@dataclass
class Exchange:
    request: dict
    status: int
    latency_ms: int
    content_type: str
    retry_after: str | None
    body: Any


@dataclass
class Check:
    id: str
    issue: int
    criterion: str
    expected: str
    passed: bool = False
    problems: list[str] = field(default_factory=list)
    actual: dict = field(default_factory=dict)
    exchanges: list[Exchange] = field(default_factory=list)


@dataclass
class Ctx:
    base_url: str
    key: str
    timeout: float
    context: dict[str, int]
    calibration: tuple[float, float] | None = None
    all_exchanges: list[Exchange] = field(default_factory=list)


def make_check(check_id: str) -> Check:
    issue, criterion, expected = META[check_id]
    return Check(id=check_id, issue=issue, criterion=criterion, expected=expected)


def load_context(path: Path = ROOT / "deploy" / "context.env") -> dict[str, int]:
    values: dict[str, int] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition("=")
        try:
            values[name.strip()] = int(value.strip())
        except ValueError:
            raise ValueError(f"{path}:{number}: esperado CHAVE=inteiro") from None
    return values


def sanitize(body: dict) -> dict:
    out = deepcopy(body)
    for message in out.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and len(content) > OMIT_OVER:
            message["content"] = f"<{len(content)} caracteres omitidos>"
    return out


def _read_body(raw: bytes, content_type: str) -> Any:
    text = raw.decode("utf-8", errors="replace")
    if "text/event-stream" in content_type.lower():
        chunks: list[Any] = []
        done = False
        for line in text.split("\n"):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                continue
            try:
                chunks.append(json.loads(data))
            except ValueError:
                chunks.append({"unparsed": data})
        return {"chunks": chunks, "done": done}
    try:
        return json.loads(text)
    except ValueError:
        return text[:MAX_TEXT]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Não segue redirect: o Authorization não pode ir para outro host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post(ctx: Ctx, body: dict) -> Exchange:
    url = ctx.base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {ctx.key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    opener = urllib.request.build_opener(_NoRedirect)
    status, content_type, retry_after = 0, "", None
    payload: Any
    started = time.monotonic()
    try:
        with opener.open(request, timeout=ctx.timeout) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type") or ""
            retry_after = resp.headers.get("Retry-After")
            raw = resp.read()
        payload = _read_body(raw, content_type)
    except urllib.error.HTTPError as exc:
        status = exc.code
        content_type = exc.headers.get("Content-Type") or ""
        retry_after = exc.headers.get("Retry-After")
        try:
            raw = exc.read()
        except (OSError, http.client.HTTPException):
            raw = b""
        payload = _read_body(raw, content_type)
    except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException, ValueError) as exc:
        status, content_type, retry_after = 0, "", None
        payload = str(exc)
    exchange = Exchange(
        request=sanitize(body),
        status=status,
        latency_ms=int((time.monotonic() - started) * 1000),
        content_type=content_type,
        retry_after=retry_after,
        body=payload,
    )
    ctx.all_exchanges.append(exchange)
    return exchange


def chat(model: str, messages: list, max_tokens: int, **extra: Any) -> dict:
    return {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": messages, **extra}


def filler(n: int, max_tokens: int) -> dict:
    return chat(MODEL, [{"role": "user", "content": FILLER * n + "Responda só: ok"}], max_tokens)


def repeats(ctx: Ctx, target: int) -> int:
    if ctx.calibration is None:
        raise RuntimeError("sem calibração")
    base, per = ctx.calibration
    return math.ceil((target - base) / per)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _clip(value: Any, limit: int = 200) -> str:
    return str(value)[:limit]


def _first_choice(obj: Any) -> dict | None:
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return None


def _status_problem(ex: Exchange) -> str:
    if ex.status == 0:
        return f"sem resposta: {_clip(ex.body)}"
    return f"status {ex.status} (esperado 200): {_clip(ex.body)}"


def completion_problems(ex: Exchange) -> list[str]:
    if ex.status != 200:
        return [_status_problem(ex)]
    if not isinstance(ex.body, dict):
        return [f"corpo 200 não é um objeto JSON: {_clip(ex.body)}"]
    problems: list[str] = []
    choice = _first_choice(ex.body)
    if choice is None:
        problems.append("choices[0] ausente")
    elif choice.get("finish_reason") not in VALID_FINISH:
        problems.append(
            f"finish_reason ausente ou inválido: {choice.get('finish_reason')!r} (esperado stop, length ou tool_calls)"
        )
    usage = ex.body.get("usage")
    if not isinstance(usage, dict):
        problems.append("usage ausente")
    else:
        for name in ("prompt_tokens", "completion_tokens"):
            if not _is_int(usage.get(name)):
                problems.append(f"usage.{name} ausente ou não inteiro")
    return problems


def _stream_text(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        delta = (_first_choice(chunk) or {}).get("delta")
        content = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(content, str):
            parts.append(content)
    return "".join(parts)


def stream_problems(ex: Exchange) -> list[str]:
    if ex.status != 200:
        return [_status_problem(ex)]
    problems: list[str] = []
    if "text/event-stream" not in ex.content_type.lower():
        problems.append(f"Content-Type {ex.content_type!r} não é text/event-stream")
    if not isinstance(ex.body, dict) or not isinstance(ex.body.get("chunks"), list):
        problems.append(f"corpo não é um stream SSE: {_clip(ex.body)}")
        return problems
    chunks = [c for c in ex.body["chunks"] if isinstance(c, dict)]
    if ex.body.get("done") is not True:
        problems.append("stream sem [DONE]")
    if not any((_first_choice(c) or {}).get("finish_reason") in VALID_FINISH for c in chunks):
        problems.append("nenhum chunk com finish_reason em stop, length ou tool_calls")
    last = chunks[-1] if chunks else {}
    usage = last.get("usage") if isinstance(last.get("usage"), dict) else {}
    tokens = usage.get("completion_tokens")
    if not (_is_int(tokens) and tokens >= 1):
        problems.append("último chunk sem usage.completion_tokens inteiro >= 1")
    text = _stream_text(chunks)
    if not text:
        problems.append("conteúdo do stream vazio")
    if "</think>" in text:
        problems.append("conteúdo do stream traz </think>")
    return problems


def summary(ex: Exchange) -> dict:
    body = ex.body if isinstance(ex.body, dict) else {}
    finish: Any = None
    content: Any = None
    tool_calls: int | None = None
    usage: Any = None
    if isinstance(body.get("chunks"), list):
        chunks = [c for c in body["chunks"] if isinstance(c, dict)]
        finish = next((f for f in ((_first_choice(c) or {}).get("finish_reason") for c in chunks) if f), None)
        content = _stream_text(chunks) or None
        usage = next((c["usage"] for c in reversed(chunks) if isinstance(c.get("usage"), dict)), None)
    else:
        choice = _first_choice(body) or {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        finish = choice.get("finish_reason")
        content = message.get("content")
        calls = message.get("tool_calls")
        tool_calls = len(calls) if isinstance(calls, list) else None
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
    return {
        "status": ex.status,
        "finish_reason": finish,
        "content": content[:MAX_ACTUAL_CONTENT] if isinstance(content, str) else None,
        "tool_calls": tool_calls,
        "usage": usage,
        "latency_ms": ex.latency_ms,
    }


# ---- helpers das checagens (só chamados depois de completion_problems == []) ----

def _message(ex: Exchange) -> dict:
    message = (_first_choice(ex.body) or {}).get("message")
    return message if isinstance(message, dict) else {}


def _content(ex: Exchange) -> str:
    content = _message(ex).get("content")
    return content if isinstance(content, str) else ""


def _finish(ex: Exchange) -> Any:
    return (_first_choice(ex.body) or {}).get("finish_reason")


def _tool_calls(ex: Exchange) -> list:
    calls = _message(ex).get("tool_calls")
    return calls if isinstance(calls, list) else []


def _completion_tokens(ex: Exchange) -> int:
    return ex.body["usage"]["completion_tokens"]


def _send(ctx: Ctx, chk: Check, body: dict) -> Exchange:
    ex = post(ctx, body)
    chk.exchanges.append(ex)
    return ex


def _done(chk: Check, ex: Exchange | None, problems: list[str]) -> Check:
    chk.problems = problems
    chk.actual = summary(ex) if ex is not None else {}
    chk.passed = not problems
    return chk


def _expect_stop(ex: Exchange, problems: list[str]) -> None:
    if _finish(ex) != "stop":
        problems.append(f"finish_reason {_finish(ex)!r}, esperado 'stop'")


def _forbid(text: str, markers: tuple[str, ...], problems: list[str]) -> None:
    for marker in markers:
        if marker in text:
            problems.append(f"content traz {marker!r}")


def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ---- checagens: uma por linha da tabela da issue #21 ----

def _check_id(ctx: Ctx, check_id: str, model: str) -> Check:
    chk = make_check(check_id)
    ex = _send(ctx, chk, chat(model, _user("Responda só: ok"), 32))
    return _done(chk, ex, completion_problems(ex))


def check_i2_id_simplicio_27b(ctx: Ctx) -> Check:
    return _check_id(ctx, "i2-id-simplicio-27b", MODEL)


def check_i2_id_simpleti_simplicio_27b(ctx: Ctx) -> Check:
    return _check_id(ctx, "i2-id-simpleti-simplicio-27b", ALIAS)


def check_i2_tool_auto(ctx: Ctx) -> Check:
    chk = make_check("i2-tool-auto")
    ex = _send(ctx, chk, chat(MODEL, _user(USER_WEBFETCH), 256, tools=[WEBFETCH], tool_choice="auto"))
    problems = completion_problems(ex)
    if not problems:
        if _finish(ex) != "tool_calls":
            problems.append(f"finish_reason {_finish(ex)!r}, esperado 'tool_calls'")
        calls = _tool_calls(ex)
        call = calls[0] if calls and isinstance(calls[0], dict) else None
        if call is None:
            problems.append("tool_calls ausente")
        else:
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            if function.get("name") != "webfetch":
                problems.append(f"tool_calls[0].function.name {function.get('name')!r}, esperado 'webfetch'")
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                problems.append("function.arguments não é uma string JSON")
            else:
                try:
                    parsed = json.loads(arguments)
                except ValueError:
                    problems.append("function.arguments não é JSON válido")
                else:
                    url = parsed.get("url") if isinstance(parsed, dict) else None
                    if not (isinstance(url, str) and url.startswith("https://example.com")):
                        problems.append(f"arguments.url inesperada: {url!r}")
        _forbid(_content(ex), ("<tool_call>", "<function="), problems)
    return _done(chk, ex, problems)


def check_i2_tool_none_oi(ctx: Ctx) -> Check:
    chk = make_check("i2-tool-none-oi")
    ex = _send(ctx, chk, chat(MODEL, _user("oi"), 256, tools=[WEBFETCH], tool_choice="none"))
    problems = completion_problems(ex)
    if not problems:
        _expect_stop(ex, problems)
        if not _content(ex).strip():
            problems.append("content vazio")
        _forbid(_content(ex), ("<think>", "</think>"), problems)
    return _done(chk, ex, problems)


def check_i3_pong(ctx: Ctx) -> Check:
    chk = make_check("i3-pong")
    ex = _send(ctx, chk, chat(MODEL, _user("Reply with exactly the single word pong"), 64))
    problems = completion_problems(ex)
    if not problems:
        _expect_stop(ex, problems)
        if _content(ex).strip() != "pong":
            problems.append(f"content {_clip(_content(ex).strip(), 80)!r}, esperado 'pong'")
    return _done(chk, ex, problems)


def _agent_messages() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_AGENT}, {"role": "user", "content": "oi"}]


def check_i3_agente_oi_system(ctx: Ctx) -> Check:
    chk = make_check("i3-agente-oi-system")
    ex = _send(ctx, chk, chat(MODEL, _agent_messages(), 512))
    problems = completion_problems(ex)
    if not problems:
        _expect_stop(ex, problems)
        if _completion_tokens(ex) >= 512:
            problems.append(f"completion_tokens {_completion_tokens(ex)} >= 512")
        if _content(ex).count("<tool") > 1:
            problems.append(f"content repete o bloco <tool ({_content(ex).count('<tool')}x)")
    return _done(chk, ex, problems)


def check_i3_agente_oi_tools(ctx: Ctx) -> Check:
    chk = make_check("i3-agente-oi-tools")
    ex = _send(ctx, chk, chat(MODEL, _agent_messages(), 512, tools=TOOLS, tool_choice="auto"))
    problems = completion_problems(ex)
    if not problems:
        if _completion_tokens(ex) >= 512:
            problems.append(f"completion_tokens {_completion_tokens(ex)} >= 512")
        content, calls = _content(ex), _tool_calls(ex)
        greeting = _finish(ex) == "stop" and not calls and bool(content.strip()) and "<tool" not in content
        if not greeting and len(calls) != 1:
            problems.append(
                f"nem saudação com stop nem exatamente 1 tool call (finish_reason {_finish(ex)!r}, "
                f"{len(calls)} tool calls, content {'vazio' if not content.strip() else 'com <tool' if '<tool' in content else 'ok'})"
            )
    return _done(chk, ex, problems)


def check_i3_segundo_turno(ctx: Ctx) -> Check:
    chk = make_check("i3-segundo-turno")
    ex = _send(ctx, chk, chat(MODEL, SECOND_TURN, 512, tools=TOOLS, tool_choice="auto"))
    problems = completion_problems(ex)
    if not problems:
        if _finish(ex) not in ("stop", "tool_calls"):
            problems.append(f"finish_reason {_finish(ex)!r}, esperado 'stop' ou 'tool_calls'")
        if _completion_tokens(ex) >= 512:
            problems.append(f"completion_tokens {_completion_tokens(ex)} >= 512")
        if len(_tool_calls(ex)) > 1:
            problems.append(f"{len(_tool_calls(ex))} tool calls, no máximo 1")
        _forbid(_content(ex), ("Tool result", "<system", "<tool_response>", "<tool_call>", "<function=", "</think>"),
                problems)
    return _done(chk, ex, problems)


def check_i4_prompt_31692(ctx: Ctx) -> Check:
    chk = make_check("i4-prompt-31692")
    ex1 = _send(ctx, chk, filler(100, 1))
    ex2 = _send(ctx, chk, filler(600, 1))
    p1 = ex1.body["usage"].get("prompt_tokens") if ex1.status == 200 and isinstance(ex1.body, dict) \
        and isinstance(ex1.body.get("usage"), dict) else None
    p2 = ex2.body["usage"].get("prompt_tokens") if ex2.status == 200 and isinstance(ex2.body, dict) \
        and isinstance(ex2.body.get("usage"), dict) else None
    if not (_is_int(p1) and _is_int(p2)) or p2 <= p1:
        ctx.calibration = None
        return _done(chk, ex2, [
            f"calibração falhou: status {ex1.status}/{ex2.status}, prompt_tokens {p1}/{p2} "
            f"({_clip(ex1.body if ex1.status != 200 else ex2.body, 120)})"
        ])
    per = (p2 - p1) / 500
    ctx.calibration = (p1 - 100 * per, per)
    ex = _send(ctx, chk, filler(repeats(ctx, ctx.context["MEASURED_AGENT_PROMPT"]) + 1, 16))
    problems = completion_problems(ex)
    if not problems:
        tokens = ex.body["usage"]["prompt_tokens"]
        low, high = ctx.context["MEASURED_AGENT_PROMPT"], ctx.context["MAX_MODEL_LEN"] - 16
        if not low <= tokens <= high:
            problems.append(f"usage.prompt_tokens {tokens} fora de [{low}, {high}]")
    return _done(chk, ex, problems)


def check_i4_acima_do_teto(ctx: Ctx) -> Check:
    chk = make_check("i4-acima-do-teto")
    if ctx.calibration is None:
        return _done(chk, None, ["sem calibração"])
    ex = _send(ctx, chk, filler(repeats(ctx, ctx.context["MAX_MODEL_LEN"] + 1024), 16))
    problems: list[str] = []
    if not (400 <= ex.status <= 499) or ex.status in (401, 403, 404, 429):
        problems.append(_status_problem(ex).replace("esperado 200", "esperado 4xx de contexto"))
    else:
        error = ex.body.get("error") if isinstance(ex.body, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if not (isinstance(message, str) and message.strip()):
            problems.append(f"corpo sem error.message do vLLM: {_clip(ex.body)}")
    return _done(chk, ex, problems)


def _check_stream(ctx: Ctx, check_id: str, **extra: Any) -> Check:
    chk = make_check(check_id)
    ex = _send(ctx, chk, chat(MODEL, _user("oi"), 64, stream=True, **extra))
    return _done(chk, ex, stream_problems(ex))


def check_i5_stream_include_usage(ctx: Ctx) -> Check:
    return _check_stream(ctx, "i5-stream-include-usage", stream_options={"include_usage": True})


def check_i5_stream_sem_options(ctx: Ctx) -> Check:
    return _check_stream(ctx, "i5-stream-sem-options")


def check_i5_todo_200_valido(ctx: Ctx) -> Check:
    chk = make_check("i5-todo-200-valido")
    problems: list[str] = []
    count = 0
    for index, ex in enumerate(ctx.all_exchanges, 1):
        if ex.status != 200:
            continue
        count += 1
        found = stream_problems(ex) if "text/event-stream" in ex.content_type.lower() else completion_problems(ex)
        if found:
            problems.append(f"troca {index} ({ex.request.get('model')}): {', '.join(found)}")
    if count < MIN_EXCHANGES_200:
        problems.append(f"só {count} trocas 200, esperado >= {MIN_EXCHANGES_200}")
    chk.problems, chk.actual, chk.passed = problems, {"exchanges_200": count}, not problems
    return chk


def check_i5_upstream_fora(ctx: Ctx) -> Check:
    chk = make_check("i5-upstream-fora")
    ex = _send(ctx, chk, chat(MODEL, _user("oi"), 16))
    problems: list[str] = []
    if not 500 <= ex.status <= 599:
        problems.append(_status_problem(ex).replace("esperado 200", "esperado 5xx"))
    elif not str(ex.body).strip():
        problems.append(f"status {ex.status} com corpo vazio")
    chk.problems, chk.passed = problems, not problems
    chk.actual = {"status": ex.status, "retry_after": ex.retry_after, "body": str(ex.body)[:500]}
    return chk


CHECKS_UP: list[Callable[[Ctx], Check]] = [
    check_i2_id_simplicio_27b,
    check_i2_id_simpleti_simplicio_27b,
    check_i2_tool_auto,
    check_i2_tool_none_oi,
    check_i3_pong,
    check_i3_agente_oi_system,
    check_i3_agente_oi_tools,
    check_i3_segundo_turno,
    check_i4_prompt_31692,
    check_i4_acima_do_teto,
    check_i5_stream_include_usage,
    check_i5_stream_sem_options,
    check_i5_todo_200_valido,
]
CHECKS_DOWN: list[Callable[[Ctx], Check]] = [check_i5_upstream_fora]


def run_check(fn: Callable[[Ctx], Check], ctx: Ctx) -> Check:
    """Roda uma checagem; qualquer exceção vira FAIL com o motivo, nunca PASS e nunca traceback."""
    try:
        return fn(ctx)
    except Exception as exc:
        chk = make_check(fn.__name__.removeprefix("check_").replace("_", "-"))
        chk.problems = [f"exceção ao executar a checagem: {type(exc).__name__}: {exc}"]
        return chk


# ---- URL-base, chave, recibo ----

def base_url_problem(url: str) -> str | None:
    """Motivo pelo qual a URL-base não serve, ou None. Nunca repete a URL na mensagem."""
    try:
        parts = urllib.parse.urlsplit(url)
        _ = parts.port
    except ValueError:
        return "URL-base inválida"
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "URL-base precisa começar com http:// ou https:// e ter host"
    if parts.username is not None or parts.password is not None:
        return "URL-base não pode conter credenciais; a chave vai só em SIMPLETI_API_KEY"
    if parts.query or parts.fragment:
        return "URL-base não pode ter query nem fragmento"
    return None


def clean_base_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def scrub(text: str, key: str) -> str:
    """Remove a chave (e suas formas escapadas em JSON) de qualquer texto que sai do processo."""
    for form in sorted({key, json.dumps(key, ensure_ascii=False)[1:-1], json.dumps(key)[1:-1]}, key=len, reverse=True):
        if form:
            text = text.replace(form, "<chave omitida>")
    return text


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _say(line: str, stream: Any) -> None:
    try:
        print(line, file=stream)
    except UnicodeEncodeError:
        print(line.encode("ascii", "backslashreplace").decode("ascii"), file=stream)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live_acceptance.py",
        description="Recibo JSON dos critérios de aceite das issues #2 a #5. "
                    f"A chave de API vem só de {KEY_ENV}.",
    )
    parser.add_argument("--base-url", default=None,
                        help=f"URL-base da API (termina em /v1), sem credenciais. Padrão: ${BASE_URL_ENV}")
    parser.add_argument("--phase", choices=("up", "down"), default="up",
                        help="up: upstream no ar (13 checagens); down: upstream desligado (1 checagem)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks" / "live",
                       help="diretório do recibo acceptance-<UTC>.json (padrão: benchmarks/live)")
    where.add_argument("--out", default=None,
                       help="arquivo exato do recibo; '-' imprime o JSON no stdout (PASS/FAIL vão ao stderr)")
    parser.add_argument("--timeout", type=float, default=300.0, help="timeout por request, em segundos")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        _say(f"{KEY_ENV} ausente", sys.stderr)
        return 2
    if not key.isascii() or not key.isprintable() or any(c.isspace() for c in key):
        # sem repetir a chave: espaço, quebra de linha ou controle quebrariam o header Authorization
        _say(f"{KEY_ENV} inválida: só caracteres ASCII visíveis, sem espaços", sys.stderr)
        return 2
    base_url = (args.base_url or os.environ.get(BASE_URL_ENV, "")).strip()
    if not base_url:
        _say(f"URL-base ausente: passe --base-url ou defina {BASE_URL_ENV}", sys.stderr)
        return 2
    problem = base_url_problem(base_url)
    if problem:
        _say(problem, sys.stderr)
        return 2
    if not args.timeout > 0:
        _say("--timeout precisa ser maior que zero", sys.stderr)
        return 2
    try:
        context = load_context()
    except (OSError, ValueError) as exc:
        _say(f"não consegui ler deploy/context.env: {exc}", sys.stderr)
        return 2
    missing = [name for name in ("MAX_MODEL_LEN", "MEASURED_AGENT_PROMPT") if name not in context]
    if missing:
        _say(f"deploy/context.env sem {', '.join(missing)}", sys.stderr)
        return 2

    ctx = Ctx(base_url=clean_base_url(base_url), key=key, timeout=args.timeout, context=context)
    log = sys.stderr if args.out == "-" else sys.stdout
    started = _utc_now()
    checks: list[Check] = []
    for fn in CHECKS_UP if args.phase == "up" else CHECKS_DOWN:
        chk = run_check(fn, ctx)
        checks.append(chk)
        _say(scrub(f"PASS {chk.id}" if chk.passed else f"FAIL {chk.id}: {'; '.join(chk.problems)}", key), log)
    finished = _utc_now()

    failed = [chk.id for chk in checks if not chk.passed]
    receipt = {
        "schema": SCHEMA,
        "base_url": ctx.base_url,
        "phase": args.phase,
        "git_commit": git_commit(),
        "started_at": _stamp(started),
        "finished_at": _stamp(finished),
        "context": context,
        "passed": not failed,
        "summary": {"total": len(checks), "passed": len(checks) - len(failed), "failed": len(failed),
                    "failed_ids": failed},
        "checks": [asdict(chk) for chk in checks],
    }
    text = scrub(json.dumps(receipt, ensure_ascii=False, indent=2), key) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        path = Path(args.out) if args.out else args.out_dir / f"acceptance-{started.strftime('%Y%m%dT%H%M%SZ')}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            _say(f"não consegui gravar o recibo: {exc}", sys.stderr)
            return 2
        _say(f"recibo: {path}", log)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
