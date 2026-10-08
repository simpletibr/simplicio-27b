"""Contract test for issue #22: what the other files and tools expect from the Modelfile.

  * `MAX_MODEL_LEN` (deploy/context.env, also read by serve_vllm.sh and hf_instruct.json) is the
    Modelfile's `num_ctx`.
  * There is one Modelfile in the repository, tracked and on disk, and nothing but the tests names
    a second one. It has exactly one FROM (the text model); the vision projector is left out.
  * No PARAMETER key is given two different values (`stop` is the one key Ollama accumulates).
  * The README and the distribution guide quote what the Modelfile sets: file names, sampling values,
    context, renderer/parser and the minimum Ollama version.
No network, no ollama.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

from serve_colab import load_context  # noqa: E402

SKIP_DIRS = {".git", "__pycache__", ".ruff_cache", "node_modules", ".simplicio"}
# Ollama's documented PARAMETER names, plus the penalties Ollama also accepts. Guards typos.
KNOWN_PARAMS = {
    "num_ctx", "temperature", "top_k", "top_p", "min_p", "typical_p", "repeat_last_n", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "num_predict", "seed", "stop", "mirostat", "mirostat_eta",
    "mirostat_tau", "num_keep",
}
MULTI_VALUED = {"stop"}


def top_level_lines(text: str) -> list[str]:
    """Stripped lines that are not the body of a triple-quoted block (the opening line is kept)."""
    out: list[str] = []
    in_block = False
    for line in text.splitlines():
        if in_block:
            in_block = '"""' not in line
            continue
        stripped = line.strip()
        out.append(stripped)
        if stripped.count('"""') == 1:
            in_block = True
    return out


def parameter_pairs(text: str) -> list[tuple[str, str]]:
    """(key, value) of every `PARAMETER key value` line, outside triple-quoted blocks."""
    pairs: list[tuple[str, str]] = []
    for stripped in top_level_lines(text):
        if stripped.upper().startswith("PARAMETER"):
            parts = stripped.split(None, 2)
            if len(parts) == 3:
                pairs.append((parts[1], parts[2]))
    return pairs


def from_values(text: str) -> list[str]:
    """The value of every `FROM` line. Ollama documents one; a second one is the vision projector trap."""
    return [s.split(None, 1)[1] for s in top_level_lines(text) if s.upper().startswith("FROM ")]


def conflicting_keys(pairs: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Keys that appear with more than one value. Multi-valued keys may differ, but not repeat."""
    seen: dict[str, list[str]] = defaultdict(list)
    for key, value in pairs:
        seen[key].append(value)
    bad: dict[str, set[str]] = {}
    for key, values in seen.items():
        if key in MULTI_VALUED:
            if len(values) != len(set(values)):
                bad[key] = set(values)
        elif len(set(values)) > 1:
            bad[key] = set(values)
    return bad


def modelfile_paths() -> list[str]:
    found = []
    for path in ROOT.rglob("*"):
        if path.is_file() and not SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
            if path.name.lower().startswith("modelfile"):
                found.append(path.relative_to(ROOT).as_posix())
    return sorted(found)


class ContextContractTest(unittest.TestCase):
    text = (ROOT / "Modelfile").read_text(encoding="utf-8")

    def test_max_model_len_is_the_modelfile_num_ctx(self) -> None:
        ctx = load_context()["MAX_MODEL_LEN"]
        num_ctx = [v for k, v in parameter_pairs(self.text) if k == "num_ctx"]
        self.assertEqual(num_ctx, [str(ctx)])
        self.assertEqual(ctx, 40960)

    def test_raw_context_env_line_and_hf_instruct_agree(self) -> None:
        env = (ROOT / "deploy" / "context.env").read_text(encoding="utf-8")
        match = re.search(r"^MAX_MODEL_LEN=(\d+)$", env, re.MULTILINE)
        self.assertIsNotNone(match)
        hf = json.loads((ROOT / "deploy" / "hf_instruct.json").read_text(encoding="utf-8"))
        self.assertEqual(hf["max_model_len"], int(match.group(1)))
        self.assertIn(f"PARAMETER num_ctx {match.group(1)}\n", self.text)

    def test_the_output_budget_still_fits_in_num_ctx(self) -> None:
        ctx = load_context()
        self.assertGreaterEqual(ctx["MAX_MODEL_LEN"], ctx["MEASURED_AGENT_PROMPT"] + ctx["OUTPUT_BUDGET"])


class SingleModelfileContractTest(unittest.TestCase):
    def test_one_modelfile_on_disk(self) -> None:
        self.assertEqual(modelfile_paths(), ["Modelfile"])

    def test_one_modelfile_tracked_by_git(self) -> None:
        run = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, text=True, check=False)
        if run.returncode != 0:
            self.skipTest("not a git checkout")
        tracked = [p for p in run.stdout.split("\0") if p and Path(p).name.lower().startswith("modelfile")]
        self.assertEqual(tracked, ["Modelfile"])

    def test_exactly_one_from_and_it_is_the_text_model(self) -> None:
        """Ollama documents one FROM. A second one (the mmproj) can make the tag be built from the projector."""
        text = (ROOT / "Modelfile").read_text(encoding="utf-8")
        self.assertEqual(from_values(text), ["./Qwen3.8-27B.Q4_K_M.gguf"])

    def test_the_from_checker_fails_on_two_froms(self) -> None:
        two = "FROM ./Qwen3.8-27B.Q4_K_M.gguf\nFROM ./Qwen3.8-27B.BF16-mmproj.gguf\nPARAMETER num_ctx 40960\n"
        self.assertEqual(len(from_values(two)), 2)
        self.assertNotEqual(from_values(two), ["./Qwen3.8-27B.Q4_K_M.gguf"])
        self.assertEqual(from_values("# FROM ./x.gguf\nSYSTEM \"\"\"\nFROM ./y.gguf\n\"\"\"\nFROM ./z.gguf\n"), ["./z.gguf"])

    def test_nothing_but_tests_points_at_a_second_modelfile(self) -> None:
        run = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, text=True, check=False)
        if run.returncode != 0:
            self.skipTest("not a git checkout")
        offenders = []
        for rel in filter(None, run.stdout.split("\0")):
            if rel.startswith("tests/") or not (ROOT / rel).is_file():
                continue
            try:
                text = (ROOT / rel).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "deploy/Modelfile" in text:
                offenders.append(rel)
        self.assertEqual(offenders, [])


class ParameterContractTest(unittest.TestCase):
    text = (ROOT / "Modelfile").read_text(encoding="utf-8")

    def test_no_key_has_two_different_values(self) -> None:
        self.assertEqual(conflicting_keys(parameter_pairs(self.text)), {})

    def test_every_parameter_key_is_a_known_ollama_option(self) -> None:
        keys = {k for k, _ in parameter_pairs(self.text)}
        self.assertEqual(keys - KNOWN_PARAMS, set())
        self.assertTrue(keys)

    def test_the_checker_catches_a_conflict(self) -> None:
        bad = "PARAMETER temperature 0.2\nPARAMETER top_k 20\nPARAMETER temperature 0.7\n"
        self.assertEqual(conflicting_keys(parameter_pairs(bad)), {"temperature": {"0.2", "0.7"}})

    def test_the_checker_allows_distinct_stops_and_flags_a_repeated_one(self) -> None:
        ok = 'PARAMETER stop "<|im_end|>"\nPARAMETER stop "<|endoftext|>"\n'
        self.assertEqual(conflicting_keys(parameter_pairs(ok)), {})
        repeated = 'PARAMETER stop "<|im_end|>"\nPARAMETER stop "<|im_end|>"\n'
        self.assertEqual(list(conflicting_keys(parameter_pairs(repeated))), ["stop"])

    def test_parameters_inside_the_system_block_are_not_parameters(self) -> None:
        text = 'SYSTEM """\nPARAMETER temperature 9\n"""\nPARAMETER temperature 0.7\n'
        self.assertEqual(parameter_pairs(text), [("temperature", "0.7")])


class DocumentedContractTest(unittest.TestCase):
    modelfile = (ROOT / "Modelfile").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "deploy" / "DISTRIBUTION_GUIDE.md").read_text(encoding="utf-8")

    def test_readme_quotes_every_sampling_value_of_the_modelfile(self) -> None:
        for key, value in parameter_pairs(self.modelfile):
            if key in {"stop", "num_ctx"}:
                continue
            self.assertIn(f"`{key} {value}`", self.readme, f"README does not quote {key} {value}")

    def test_readme_and_guide_quote_the_context_and_renderer(self) -> None:
        self.assertIn("`num_ctx 40960`", self.readme)
        self.assertIn("40,960-token context (`deploy/context.env`)", self.guide)
        for doc in (self.readme, self.guide):
            self.assertIn("RENDERER qwen3.5", doc)
            self.assertIn("PARSER qwen3.5", doc)

    def test_minimum_ollama_version_is_the_same_everywhere(self) -> None:
        requires = re.search(r"^REQUIRES (\S+)$", self.modelfile, re.MULTILINE).group(1)
        self.assertIn(f"Ollama {requires} or newer", self.readme)
        self.assertIn(f">= {requires})", self.modelfile)

    def test_from_file_is_the_one_the_guide_lists(self) -> None:
        froms = from_values(self.modelfile)
        self.assertEqual(len(froms), 1)
        self.assertIn(f"`{froms[0].removeprefix('./')}`", self.guide)

    def test_vision_projector_left_out_on_purpose_is_said_everywhere(self) -> None:
        self.assertNotIn("mmproj", " ".join(from_values(self.modelfile)))
        for label, doc in (("Modelfile", self.modelfile), ("README", self.readme), ("guide", self.guide)):
            self.assertIn("out on purpose", doc, label)
            self.assertIn("text only", doc, label)
        self.assertNotIn("holds the Q4_K_M GGUF and the base model's vision projector", self.readme)
        self.assertNotIn("combines two files", self.guide)

    def test_readme_explains_the_stop_difference_with_the_56_120_run(self) -> None:
        self.assertIn("The tag stops only at `<|im_end|>`, while the 56/120 run also stopped at `</deliver>`", self.readme)
        self.assertIn('"stop": ["</deliver>"]', self.readme)

    def test_training_stop_and_template_are_gone_from_docs(self) -> None:
        for doc in (self.readme, self.guide):
            for stale in ("top_k` 40", "repeat_penalty` 1.1", "32,768", "32768", "does not declare tools"):
                self.assertNotIn(stale, doc)


if __name__ == "__main__":
    unittest.main()
