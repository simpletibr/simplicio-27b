"""vLLM tool-parser plugin for Simplicio 27B.

Register with:

    --enable-auto-tool-choice \
    --tool-parser-plugin /abs/path/deploy/simplicio_tool_parser.py \
    --tool-call-parser simplicio

The extract logic lives in ``simplicio_tool_xml.py`` so tests run without vLLM.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

_DEPLOY = Path(__file__).resolve().parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from simplicio_tool_xml import TOOL_CLOSE, TOOL_OPEN, extract_first_tool_as_openai  # noqa: E402

try:
    from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (  # type: ignore
        ToolParser,
        ToolParserManager,
    )
    from vllm.entrypoints.openai.protocol import (  # type: ignore
        ChatCompletionRequest,
        DeltaFunctionCall,
        DeltaMessage,
        DeltaToolCall,
        ExtractedToolCallInformation,
        FunctionCall,
        ToolCall,
    )
    try:
        from vllm.entrypoints.chat_utils import make_tool_call_id  # type: ignore
    except ImportError:  # pragma: no cover
        def make_tool_call_id() -> str:
            return "chatcmpl-tool-simplicio"
except ImportError:  # vLLM 0.31+ layout
    from vllm.tool_parsers import ToolParser, ToolParserManager  # type: ignore
    from vllm.entrypoints.openai.protocol import (  # type: ignore
        ChatCompletionRequest,
        DeltaFunctionCall,
        DeltaMessage,
        DeltaToolCall,
        ExtractedToolCallInformation,
        FunctionCall,
        ToolCall,
    )

    def make_tool_call_id() -> str:  # type: ignore
        return "chatcmpl-tool-simplicio"


@ToolParserManager.register_module("simplicio")
class SimplicioToolParser(ToolParser):
    """First ``<tool>`` block becomes a single OpenAI tool call."""

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        extracted = extract_first_tool_as_openai(model_output)
        tool = extracted["tool"]
        if tool is None:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=extracted["content"] or model_output,
            )
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(
                    type="function",
                    id=make_tool_call_id(),
                    function=FunctionCall(
                        name=tool["name"],
                        arguments=tool["arguments"],
                    ),
                )
            ],
            content=extracted["content"],
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        if TOOL_OPEN not in current_text and "<tool_call>" not in current_text:
            return DeltaMessage(content=delta_text)

        prev_done = TOOL_CLOSE in previous_text or "</tool_call>" in previous_text
        now_done = TOOL_CLOSE in current_text or "</tool_call>" in current_text
        if prev_done:
            return None
        if not now_done:
            return None

        extracted = extract_first_tool_as_openai(current_text)
        tool = extracted["tool"]
        if tool is None:
            return None
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=tool["name"],
                        arguments=tool["arguments"],
                    ).model_dump(exclude_none=True)
                    if hasattr(DeltaFunctionCall, "model_dump")
                    else {"name": tool["name"], "arguments": tool["arguments"]},
                )
            ]
        )
