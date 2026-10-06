"""Unit tests for Simplicio 27B tool XML extraction. No vLLM required."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

from simplicio_tool_xml import extract_first_tool, extract_first_tool_as_openai, params_to_dict


LOOPING_XML = """<tool>
<name>bash</name>
<params>
<command>pwd && ls -la</command>
<params>
</params>
</tool>
</invoke>
<tool>
<name>read</name>
<params>
<path>/projetos/README.md</path>
<params>
</params>
</tool>
<tool>
<name>webfetch</name>
<params>
<url>https://api.github.com/repos/openclaw/openclaw</url>
<params>
</params>
</tool>
"""


class ParamsToDictTests(unittest.TestCase):
    def test_nested_tags(self) -> None:
        self.assertEqual(
            params_to_dict("<url>https://example.com</url>"),
            {"url": "https://example.com"},
        )

    def test_json_blob(self) -> None:
        self.assertEqual(
            params_to_dict('{"url": "https://opencode.ai/docs/cli", "format": "text"}'),
            {"url": "https://opencode.ai/docs/cli", "format": "text"},
        )


class ExtractFirstToolTests(unittest.TestCase):
    def test_plain_text_has_no_tool(self) -> None:
        extracted = extract_first_tool("Oi! Como posso te ajudar hoje?")
        self.assertIsNone(extracted["tool"])
        self.assertEqual(extracted["content"], "Oi! Como posso te ajudar hoje?")

    def test_looping_xml_keeps_only_the_first_call(self) -> None:
        extracted = extract_first_tool(LOOPING_XML)
        self.assertIsNotNone(extracted["tool"])
        assert extracted["tool"] is not None
        self.assertEqual(extracted["tool"]["name"], "bash")
        self.assertEqual(extracted["tool"]["arguments"]["command"], "pwd && ls -la")
        self.assertNotIn("webfetch", json.dumps(extracted["tool"]))
        self.assertNotIn("read", extracted["tool"]["name"])

    def test_webfetch_url(self) -> None:
        text = (
            "<answer>hi</answer>\n"
            "<tool>\n<name>webfetch</name>\n"
            "<params>\n<url>https://opencode.ai/docs/cli</url>\n"
            "<format>text</format>\n</params>\n</tool>"
        )
        extracted = extract_first_tool(text)
        self.assertEqual(extracted["content"], "<answer>hi</answer>")
        assert extracted["tool"] is not None
        self.assertEqual(extracted["tool"]["name"], "webfetch")
        self.assertEqual(
            extracted["tool"]["arguments"]["url"],
            "https://opencode.ai/docs/cli",
        )

    def test_openai_arguments_are_json(self) -> None:
        extracted = extract_first_tool_as_openai(LOOPING_XML)
        assert extracted["tool"] is not None
        args = json.loads(extracted["tool"]["arguments"])
        self.assertEqual(args["command"], "pwd && ls -la")

    def test_hermes_json_block(self) -> None:
        text = '<tool_call>{"name": "bash", "arguments": {"command": "pwd"}}</tool_call>'
        extracted = extract_first_tool(text)
        assert extracted["tool"] is not None
        self.assertEqual(extracted["tool"]["name"], "bash")
        self.assertEqual(extracted["tool"]["arguments"]["command"], "pwd")


if __name__ == "__main__":
    unittest.main()
