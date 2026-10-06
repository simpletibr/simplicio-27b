"""Reject chat completions that would look like OpenCode finish=unknown.

A 200 from this gate always has finish_reason in {stop, length, tool_calls}
and a usage object. Anything else becomes HTTP 502. 4xx/5xx pass through.
"""

from __future__ import annotations

import json
from typing import Any

VALID_FINISH = frozenset({"stop", "length", "tool_calls"})
BAD_STATUS = 502
BAD_TYPE = "application/json; charset=utf-8"


def _error_body(message: str) -> bytes:
    return json.dumps(
        {
            "error": {
                "message": message,
                "type": "BadCompletionError",
                "code": BAD_STATUS,
            }
        },
        ensure_ascii=False,
    ).encode("utf-8")


def finish_reason_of(choice: dict[str, Any] | None) -> str | None:
    if not isinstance(choice, dict):
        return None
    reason = choice.get("finish_reason")
    if isinstance(reason, str) and reason:
        return reason
    return None


def usage_ok(payload: dict[str, Any]) -> bool:
    usage = payload.get("usage")
    if not isinstance(usage, dict) or not usage:
        return False
    return any(
        isinstance(usage.get(key), int) and usage[key] >= 0
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input", "output")
    )


def json_completion_ok(payload: dict[str, Any]) -> str | None:
    """Return None if valid, else an error message."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "completion missing choices"
    reason = finish_reason_of(choices[0])
    if reason not in VALID_FINISH:
        return f"completion finish_reason must be stop|length|tool_calls, got {reason!r}"
    if not usage_ok(payload):
        return "completion missing usage"
    return None


def parse_sse_payloads(raw: bytes) -> list[dict[str, Any]]:
    text = raw.decode("utf-8", errors="replace")
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def sse_completion_ok(raw: bytes) -> str | None:
    events = parse_sse_payloads(raw)
    if not events:
        return "empty stream"
    reason = None
    usage_payload: dict[str, Any] | None = None
    for event in events:
        if event.get("usage"):
            usage_payload = event
        choices = event.get("choices") or []
        if choices:
            found = finish_reason_of(choices[0])
            if found:
                reason = found
    if reason not in VALID_FINISH:
        return f"stream finish_reason must be stop|length|tool_calls, got {reason!r}"
    if usage_payload is None or not usage_ok(usage_payload):
        return "stream missing usage"
    return None


def is_sse(content_type: str, raw: bytes) -> bool:
    ctype = (content_type or "").lower()
    if "text/event-stream" in ctype:
        return True
    head = raw.lstrip()[:32]
    return head.startswith(b"data:")


def enforce(status: int, raw: bytes, content_type: str) -> tuple[int, bytes, str]:
    """Return (status, body, content_type)."""
    if status != 200:
        return status, raw, content_type or "application/json"
    if is_sse(content_type, raw):
        err = sse_completion_ok(raw)
        if err:
            return BAD_STATUS, _error_body(err), BAD_TYPE
        return status, raw, content_type or "text/event-stream"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return BAD_STATUS, _error_body("completion is not JSON"), BAD_TYPE
    if not isinstance(payload, dict):
        return BAD_STATUS, _error_body("completion is not an object"), BAD_TYPE
    err = json_completion_ok(payload)
    if err:
        return BAD_STATUS, _error_body(err), BAD_TYPE
    return status, raw, content_type or "application/json"
