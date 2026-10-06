"""Parse Simplicio 27B tool XML without importing vLLM.

The served model emits one or more of:

    <tool>
    <name>webfetch</name>
    <params>
    <url>https://example.com</url>
    </params>
    </tool>

and then often repeats the block until max_tokens. Hosts need the first
complete call as OpenAI ``tool_calls``, with everything after dropped.
"""

from __future__ import annotations

import json
import re
from typing import Any

TOOL_OPEN = "<tool>"
TOOL_CLOSE = "</tool>"
HERMES_OPEN = "<tool_call>"
HERMES_CLOSE = "</tool_call>"

_NAME = re.compile(r"<name>\s*([^<]+?)\s*</name>", re.IGNORECASE | re.DOTALL)
_PARAM_PAIR = re.compile(
    r"<([A-Za-z_][\w-]*)>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_HERMES = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def first_tool_span(text: str) -> tuple[int, int] | None:
    """Byte span of the first complete ``<tool>…</tool>`` or hermes block."""
    xml_start = text.find(TOOL_OPEN)
    xml_end = text.find(TOOL_CLOSE)
    if xml_start != -1 and xml_end != -1 and xml_end > xml_start:
        return xml_start, xml_end + len(TOOL_CLOSE)

    hermes_start = text.find(HERMES_OPEN)
    hermes_end = text.find(HERMES_CLOSE)
    if hermes_start != -1 and hermes_end != -1 and hermes_end > hermes_start:
        return hermes_start, hermes_end + len(HERMES_CLOSE)
    return None


def params_to_dict(inner: str) -> dict[str, Any]:
    blob = re.sub(r"</?params>", "", inner, flags=re.IGNORECASE).strip()
    if blob.startswith("{"):
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    pairs = _PARAM_PAIR.findall(blob)
    if pairs:
        return {key: value.strip() for key, value in pairs}
    if blob:
        return {"input": blob}
    return {}


def parse_tool_block(block: str) -> dict[str, Any] | None:
    hermes = _HERMES.search(block)
    if hermes:
        try:
            payload = json.loads(hermes.group(1))
        except json.JSONDecodeError:
            return None
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {"input": arguments}
        return {"name": name.strip(), "arguments": arguments}

    name_match = _NAME.search(block)
    if not name_match:
        return None
    name = name_match.group(1).strip()
    params_start = block.lower().find("<params>")
    if params_start == -1:
        inner = block[name_match.end():]
    else:
        inner = block[params_start:]
    return {"name": name, "arguments": params_to_dict(inner)}


def extract_first_tool(text: str) -> dict[str, Any]:
    """Return ``{content, tool}`` where ``tool`` is the first call or None."""
    span = first_tool_span(text)
    if span is None:
        return {"content": text, "tool": None}
    start, end = span
    content = text[:start].rstrip()
    tool = parse_tool_block(text[start:end])
    return {"content": content or None, "tool": tool}


def extract_first_tool_as_openai(text: str) -> dict[str, Any]:
    """Same as extract_first_tool, with arguments JSON-encoded for OpenAI."""
    extracted = extract_first_tool(text)
    tool = extracted["tool"]
    if tool is None:
        return extracted
    return {
        "content": extracted["content"],
        "tool": {
            "name": tool["name"],
            "arguments": json.dumps(tool["arguments"], ensure_ascii=False),
        },
    }
