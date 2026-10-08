"""Reject chat completions that would look like OpenCode finish=unknown.

A non-stream 200 from this gate always has finish_reason in {stop, length,
tool_calls} and a usage object; anything else becomes HTTP 502. Streams are
forwarded as they arrive (StreamCheck) and end with an SSE error event when
finish_reason is missing. 4xx/5xx pass through.
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


SSE_DONE = b"data: [DONE]\n\n"


def sse_error_event(message: str) -> bytes:
    """One SSE event carrying the same error object as the JSON 502 body."""
    return b"data: " + _error_body(message) + b"\n\n"


class StreamCheck:
    """Watch an SSE stream line by line while it is forwarded to the client."""

    def __init__(self) -> None:
        self.events = 0
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.upstream_error: str | None = None
        self.done = False

    def feed(self, line: bytes) -> None:
        text = line.decode("utf-8", errors="replace").strip()
        if not text.startswith("data:"):
            return
        data = text[5:].strip()
        if data == "[DONE]":
            self.done = True
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return
        if not isinstance(obj, dict):
            return
        self.events += 1
        err = obj.get("error")
        if isinstance(err, dict):
            self.upstream_error = str(err.get("message") or "upstream error")
        if usage_ok(obj):
            self.usage = obj["usage"]
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            found = finish_reason_of(choices[0])
            if found:
                self.finish_reason = found

    def problem(self) -> str | None:
        """Message for the terminal error event, or None when none is needed."""
        if self.upstream_error is not None:
            return None  # the upstream error event already reached the client
        if self.events == 0:
            return "empty stream"
        if self.finish_reason not in VALID_FINISH:
            return (
                "stream finish_reason must be stop|length|tool_calls, "
                f"got {self.finish_reason!r}"
            )
        return None


def enforce(status: int, raw: bytes, content_type: str) -> tuple[int, bytes, str]:
    """Check a whole non-stream response. Return (status, body, content_type)."""
    if status != 200:
        return status, raw, content_type or "application/json"
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
