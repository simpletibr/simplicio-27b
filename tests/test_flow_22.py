"""Flow test for issue #22: the Modelfile is read the way `ollama create` reads it, offline.

  * A small parser follows the Ollama Modelfile grammar: comments, blank lines, one instruction per
    line (FROM, PARAMETER, RENDERER, PARSER, REQUIRES, TEMPLATE, SYSTEM, ...) and triple-quoted values
    that may span lines. Any other line is a parse error, as it is for Ollama.
  * From the parsed file it builds the `params` blob Ollama stores in the registry (repeated `stop`
    lines become a list) and the model `config` (renderer, parser, requires), then compares them with
    the values issue #22 expects `ollama show` and the registry to report.
No ollama binary, no network, no GGUF.
"""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

from serve_colab import load_context  # noqa: E402

INSTRUCTIONS = {
    "FROM", "PARAMETER", "TEMPLATE", "SYSTEM", "ADAPTER", "LICENSE", "MESSAGE",
    "RENDERER", "PARSER", "REQUIRES",
}
INT_PARAMS = {"num_ctx", "top_k", "num_predict", "seed", "repeat_last_n", "num_keep", "mirostat"}
FLOAT_PARAMS = {"temperature", "top_p", "min_p", "typical_p", "repeat_penalty", "presence_penalty",
                "frequency_penalty", "mirostat_tau", "mirostat_eta"}
# What issue #22 lists as the registry `params` blob (step 16 of "Critério de pronto").
EXPECTED_PARAMS = {
    "min_p": 0, "num_ctx": 40960, "presence_penalty": 1.5, "repeat_penalty": 1,
    "stop": ["<|im_end|>"], "temperature": 0.7, "top_k": 20, "top_p": 0.8,
}
EXPECTED_CONFIG = {"renderer": "qwen3.5", "parser": "qwen3.5", "requires": "0.30.0"}


class ModelfileSyntaxError(ValueError):
    pass


def parse_modelfile(text: str) -> list[tuple[str, str]]:
    """Return [(INSTRUCTION, raw value)] in file order. Raises ModelfileSyntaxError on anything else."""
    out: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(" ")
        name = name.upper()
        if name not in INSTRUCTIONS:
            raise ModelfileSyntaxError(f"line {i}: unknown instruction {line!r}")
        rest = rest.strip()
        if not rest:
            raise ModelfileSyntaxError(f"line {i}: {name} has no value")
        if rest.startswith('"""'):
            body = rest[3:]
            while '"""' not in body:
                if i >= len(lines):
                    raise ModelfileSyntaxError(f"line {i}: unterminated triple-quoted string")
                body += "\n" + lines[i]
                i += 1
            body, _, tail = body.partition('"""')
            if tail.strip():
                raise ModelfileSyntaxError(f"line {i}: text after closing quotes: {tail!r}")
            rest = body
        out.append((name, rest))
    return out


def unquote(value: str) -> str:
    return ast.literal_eval(value) if value[:1] in "\"'" else value


def number(key: str, raw: str) -> int | float:
    if key in INT_PARAMS:
        return int(raw)
    if key in FLOAT_PARAMS:
        return float(raw)
    raise ModelfileSyntaxError(f"PARAMETER {key}: not a known numeric parameter")


def params_blob(instructions: list[tuple[str, str]]) -> dict:
    """The `params` layer `ollama create` writes: numbers typed, `stop` collected into a list."""
    params: dict = {}
    for name, value in instructions:
        if name != "PARAMETER":
            continue
        key, _, raw = value.partition(" ")
        raw = raw.strip()
        if key == "stop":
            params.setdefault("stop", []).append(unquote(raw))
        else:
            params[key] = number(key, raw)
    return params


def config_blob(instructions: list[tuple[str, str]]) -> dict:
    cfg: dict = {"renderer": None, "parser": None, "requires": None}
    for name, value in instructions:
        if name.lower() in cfg:
            cfg[name.lower()] = value
    return cfg


def training_system_prompt() -> str:
    nb = json.loads((ROOT / "Simplicio_27B_Training_Colab.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if src.startswith("system_prompt = ("):
            return ast.literal_eval(ast.parse(src).body[0].value)
    raise AssertionError("system_prompt cell not found")


class ModelfileFlowTest(unittest.TestCase):
    text = (ROOT / "Modelfile").read_text(encoding="utf-8")

    def setUp(self) -> None:
        self.ins = parse_modelfile(self.text)

    def test_file_parses_with_only_known_instructions(self) -> None:
        self.assertEqual({n for n, _ in self.ins} - INSTRUCTIONS, set())
        self.assertTrue(self.text.endswith("\n"))

    def test_one_local_gguf_from_line_and_no_vision_projector(self) -> None:
        froms = [v for n, v in self.ins if n == "FROM"]
        self.assertEqual(froms, ["./Qwen3.8-27B.Q4_K_M.gguf"])
        self.assertNotIn("mmproj", " ".join(v for n, v in self.ins if n != "SYSTEM"))

    def test_qwen_renderer_and_parser_without_legacy_template(self) -> None:
        """With RENDERER/PARSER set, a legacy TEMPLATE would be ignored by Ollama 0.31.1 (issue #22)."""
        self.assertEqual(config_blob(self.ins), EXPECTED_CONFIG)
        self.assertNotIn("TEMPLATE", {n for n, _ in self.ins})

    def test_num_ctx_is_40960_and_matches_context_env(self) -> None:
        params = params_blob(self.ins)
        self.assertIsInstance(params["num_ctx"], int)
        self.assertEqual(params["num_ctx"], 40960)
        self.assertEqual(params["num_ctx"], load_context()["MAX_MODEL_LEN"])

    def test_params_blob_is_what_the_issue_expects_in_the_registry(self) -> None:
        self.assertEqual(params_blob(self.ins), EXPECTED_PARAMS)

    def test_stop_keeps_the_final_phase_and_drops_the_old_extras(self) -> None:
        stops = params_blob(self.ins)["stop"]
        self.assertEqual(stops, ["<|im_end|>"])
        self.assertNotIn("</deliver>", stops)
        data = ROOT / "data" / "simplicio_loop_50pts_train.jsonl"
        rows = [json.loads(x) for x in data.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row["simplicio_trajectory"].rstrip().endswith("</deliver>\n</simplicio_loop>"))

    def test_system_is_the_training_prompt_not_the_marketing_text(self) -> None:
        systems = [v for n, v in self.ins if n == "SYSTEM"]
        self.assertEqual(systems, [training_system_prompt()])
        self.assertNotIn("desperdicio zero de tokens", systems[0])

    def test_parser_rejects_what_ollama_would_reject(self) -> None:
        for bad in ("PARAMTER num_ctx 40960\n", "PARAMETER\n", 'SYSTEM """never closed\n', "just text\n"):
            with self.assertRaises(ModelfileSyntaxError, msg=bad):
                parse_modelfile(bad)

    def test_multiline_triple_quoted_value(self) -> None:
        got = parse_modelfile('FROM ./a.gguf\nSYSTEM """line one\nline two"""\nPARAMETER num_ctx 8\n')
        self.assertEqual(got, [("FROM", "./a.gguf"), ("SYSTEM", "line one\nline two"), ("PARAMETER", "num_ctx 8")])


if __name__ == "__main__":
    unittest.main()
