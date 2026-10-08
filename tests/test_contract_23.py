"""Contract test for issue #23: what the Hub, the README and the other files expect from scripts/hf_publish.py.

  * The allowlist is exactly what the issue lists (mirrored files, adapter files and `lora/`, what the Hub
    keeps). Against a Hub listing, files outside it are orphans: a warning, never an operation.
  * `--publish` has no delete or copy anywhere: no such operation type, no such Hub call, no such string.
  * The card is the README.md, unchanged, with the metadata the issue sets (pretty_name, license, base_model,
    base_model_relation, library_name, tags, homepage). Every result figure in it traces to
    benchmarks/live_colab_g4_bf16_n120.json; a figure the benchmark file does not back is refused.
  * No token or key literal in the script, the card or the notebook; the token enters only through HF_TOKEN;
    the script imports only the standard library at the top (huggingface_hub is loaded inside HubClient).
  * The layout the issue asks for: lora/ documented, no weights in git, the notebook reads the adapter from lora/.
No network.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hf_publish  # noqa: E402

SCRIPT = ROOT / "scripts" / "hf_publish.py"
README = (ROOT / "README.md").read_text(encoding="utf-8")
MODELFILE = (ROOT / "Modelfile").read_text(encoding="utf-8")
BENCH = json.loads((ROOT / "benchmarks" / "live_colab_g4_bf16_n120.json").read_text(encoding="utf-8"))
NOTEBOOK = ROOT / "notebooks/Simplicio_27B_Merge_Colab.ipynb"
# Straight from issue #23.
ISSUE_MIRRORED = ("README.md", "LICENSE", "Modelfile")
ISSUE_ADAPTER = ("adapter_config.json", "adapter_model.safetensors")
ISSUE_REMOTE_KEEP = (
    ".gitattributes", "config.json", "generation_config.json", "model.safetensors.index.json",
    "model-*.safetensors", "tokenizer*.json", "chat_template.jinja", "*processor_config.json",
    "*.gguf", "lora/*",
)
ISSUE_SAMPLING = {
    "temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
    "repeat_penalty": "repetition_penalty", "presence_penalty": "presence_penalty",
}
ISSUE_FRONT_MATTER = {
    "pretty_name": "Simplicio 27B", "license": "apache-2.0", "base_model": "Qwen/Qwen3.8-27B",
    "base_model_relation": "finetune", "library_name": "transformers",
    "homepage": "https://simpleti.com.br/simplicio-27b/",
}
ISSUE_TAGS = ["qwen", "unsloth", "lora", "code-generation", "simplicio-loop", "software-engineering",
              "surgical-diff", "agentic-coding"]
# Phrases the issue wants gone; built by concatenation so that this file does not contain them.
OLD_PHRASES = ("the base comes from " + "adapter_config.json", "unsloth resolve a base via " + "adapter_config")
LITERAL_SECRETS = (r"\bhf_[A-Za-z0-9]{20,}", r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}", r"\bsk-[A-Za-z0-9]{20,}",
                   r"AKIA[0-9A-Z]{16}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----")


def tracked_and_new(*pathspec: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *pathspec],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [p for p in out.split("\0") if p]


class AllowlistContract(unittest.TestCase):
    def test_allowlist_is_what_the_issue_lists(self):
        self.assertEqual(hf_publish.MIRRORED, ISSUE_MIRRORED)
        self.assertEqual(hf_publish.ADAPTER, ISSUE_ADAPTER)
        self.assertEqual(hf_publish.ADAPTER_DIR, "lora")
        self.assertEqual(hf_publish.REMOTE_KEEP, ISSUE_REMOTE_KEEP)
        self.assertEqual(hf_publish.SAMPLING, ISSUE_SAMPLING)
        self.assertEqual(hf_publish.REPO_ID, "wesleysimplicio/Simplicio-27B")

    def test_scan_accepts_exactly_the_allowlist(self):
        scan = hf_publish.scan_source(ROOT)
        self.assertEqual(set(scan.entries), {*ISSUE_MIRRORED, "assets/simplicio-logo.png"}
                         | {f"lora/{name}" for name in ISSUE_ADAPTER if (ROOT / "lora" / name).is_file()})
        self.assertTrue(scan.refused)  # deploy/context.env and gateway/upstream_token.php look like credentials
        self.assertEqual({path for path, _ in scan.refused} & set(scan.entries), set())

    def test_weight_formats_are_all_refused_outside_the_adapter(self):
        self.assertEqual(set(hf_publish.WEIGHT_SUFFIXES),
                         {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt", ".h5", ".onnx"})
        root = ROOT.resolve()
        allowed = {*hf_publish.MIRRORED, "lora/adapter_model.safetensors", "lora/adapter_config.json",
                   *(f"assets/x{suffix}" for suffix in hf_publish.WEIGHT_SUFFIXES)}
        verdicts = {rel: hf_publish.judge(rel, root / rel, root, allowed, set())[0]
                    for rel in allowed | {"lora/extra.bin"}}
        self.assertEqual({rel for rel, verdict in verdicts.items() if verdict == "ok"} & set(allowed),
                         {*hf_publish.MIRRORED, "lora/adapter_model.safetensors", "lora/adapter_config.json"})
        self.assertEqual(verdicts["lora/extra.bin"], "refuse")
        for suffix in hf_publish.WEIGHT_SUFFIXES:
            self.assertEqual(verdicts[f"assets/x{suffix}"], "refuse", suffix)

    def test_the_gguf_path_only_accepts_the_gguf_suffix(self):
        self.assertEqual(hf_publish.gguf_targets("FROM ./a.gguf\nFROM ./B.GGUF\n"), {"a.gguf", "B.GGUF"})
        self.assertEqual(hf_publish.gguf_targets(MODELFILE), {"Qwen3.8-27B.Q4_K_M.gguf"})
        for bad in ("x.safetensors", "x.bin", "x.pt", "x.ckpt", "x.h5", "x.onnx", "x", "x.gguf.bin",
                    "sub/x.gguf", "../x.gguf"):
            with self.subTest(bad), self.assertRaises(hf_publish.Refused):
                hf_publish.gguf_targets(f"FROM ./{bad}\n")
        self.assertIn("ggufs = gguf_targets(modelfile) if allow_gguf else set()",
                      SCRIPT.read_text(encoding="utf-8"))

    def test_assets_in_the_allowlist_are_the_ones_the_readme_cites(self):
        cited = set(hf_publish.ASSET_RE.findall(README))
        self.assertEqual({p for p in hf_publish.scan_source(ROOT).entries if p.startswith("assets/")},
                         {p for p in cited if (ROOT / p).is_file()})

    def test_orphans_on_the_hub_are_warnings_never_operations(self):
        needed = (".gitattributes", "config.json", "generation_config.json", "model.safetensors.index.json",
                  "model-00001-of-00018.safetensors", "tokenizer.json", "tokenizer_config.json",
                  "chat_template.jinja", "processor_config.json", "preprocessor_config.json",
                  "lora/adapter_config.json", "lora/adapter_model.safetensors", "Qwen3.8-27B.Q4_K_M.gguf",
                  "Qwen3.8-27B.BF16-mmproj.gguf")
        orphans = ("assets/leaderboard_top10.svg", "benchmarks/AUDIT_RESPONSE_AND_PROOF.md",
                   "deploy/serve_vllm.sh", "model.safetensors", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "added_tokens.json", "training_args.bin")
        remote = {p: hf_publish.RemoteFile(p, "a" * 40) for p in (*needed, *orphans)}
        scan = hf_publish.scan_source(ROOT)
        plan = hf_publish.build_plan(scan.entries, scan.modelfile, remote, b"{}")
        self.assertEqual({type(op) for op in plan.ops}, {hf_publish.Add})
        self.assertEqual({op.path for op in plan.ops},
                         {"README.md", "LICENSE", "Modelfile", "assets/simplicio-logo.png", "generation_config.json"})
        self.assertEqual(sorted(w.split(": ", 1)[1] for w in plan.warnings), sorted(orphans))
        self.assertTrue(all(w.startswith("órfão no Hub, não removido: ") for w in plan.warnings))

    def test_every_modelfile_from_must_exist_on_the_hub(self):
        self.assertEqual(hf_publish.FROM_RE.findall(MODELFILE), ["Qwen3.8-27B.Q4_K_M.gguf"])


class CardContract(unittest.TestCase):
    def test_front_matter_is_what_the_issue_sets(self):
        meta = hf_publish.front_matter(README)
        for key, value in ISSUE_FRONT_MATTER.items():
            self.assertEqual(meta.get(key), value, key)
        self.assertEqual(meta["tags"], ISSUE_TAGS)
        self.assertEqual(meta["pipeline_tag"], "text-generation")
        self.assertEqual(meta["language"], ["en", "pt"])
        self.assertEqual(hf_publish.CARD_META, {**ISSUE_FRONT_MATTER, "pipeline_tag": "text-generation"})
        self.assertEqual(list(hf_publish.CARD_TAGS), ISSUE_TAGS)

    def test_two_front_matter_lines_within_the_first_25(self):
        head = README.splitlines()[:25]
        self.assertEqual(sum(line in ("library_name: transformers", "base_model_relation: finetune")
                             for line in head), 2)

    def test_card_is_the_readme_unchanged(self):
        self.assertEqual(hf_publish.card_problems(README, BENCH), [])
        self.assertEqual(hf_publish.build_card(README, BENCH), README)

    def test_card_text_for_the_hub_layout(self):
        self.assertIn('subfolder="lora"', README)
        self.assertIn("AutoModelForImageTextToText", README)
        self.assertIn("the LoRA adapter in `lora/` (0.64 GB)", README)
        self.assertIn("`lora/adapter_config.json`", README)
        self.assertNotIn("### Python (Unsloth)", README)
        self.assertIn("| `scripts/hf_publish.py` |", README)

    def test_old_adapter_phrases_are_gone_everywhere(self):
        hits = [f"{path}: {phrase}" for path in tracked_and_new()
                if (ROOT / path).is_file() and not path.endswith((".png", ".jsonl"))
                for phrase in OLD_PHRASES
                if phrase in (ROOT / path).read_text(encoding="utf-8", errors="ignore")]
        self.assertEqual(hits, [])

    def test_result_figures_trace_to_the_benchmark_file(self):
        claims, numerators = hf_publish.benchmark_claims(BENCH)
        self.assertEqual(claims, ["56/120 runs (46.7%)", "104/120 (86.7%)", "69.9 / 103", "438 s"])
        for claim in claims:
            self.assertIn(claim, README)
        body = hf_publish.strip_section(README, hf_publish.WITHDRAWN_HEADING)
        cited = {int(n) for n in re.findall(r"\b(\d+)(?:/| of )120\b", body)}
        self.assertTrue(cited)
        self.assertLessEqual(cited, numerators)
        categories = BENCH["categories"]
        self.assertEqual(sum(c["pass"] for c in categories.values()), BENCH["functional_pass"])
        self.assertEqual(sum(c["diff"] for c in categories.values()), BENCH["diff_matched"])
        for category, expected in (("Surgical diff and AST precision", (17, 20)),
                                   ("Edge-case correctness", (6, 27)),
                                   ("Nonexistent and deprecated API traps", (27, 30)),
                                   ("Adversarial and out-of-distribution", (6, 27))):
            row = next(line for line in README.splitlines() if line.startswith(f"| {category} |"))
            cells = [c.strip() for c in row.strip("|").split("|")]
            self.assertEqual((int(cells[2]), int(cells[3])), expected, category)
        by_name = {k.split("_", 1)[0]: v for k, v in categories.items()}
        self.assertEqual([(by_name[str(i)]["pass"], by_name[str(i)]["diff"]) for i in range(1, 5)],
                         [(17, 20), (6, 27), (27, 30), (6, 27)])

    def test_a_figure_the_benchmark_does_not_back_is_refused(self):
        wrong_pass = README.replace("56/120 runs (46.7%)", "57/120 runs (47.5%)")
        wrong_ratio = README.replace("104/120 (86.7%)", "110/120 (91.7%)")
        extra = README + "\nIt passed 116 of 120 runs.\n"
        revived = README.replace("## Quick start", "The model scores 96.5%.\n\n## Quick start", 1)
        for label, text in (("pass count", wrong_pass), ("patch applied", wrong_ratio),
                            ("unbacked N of 120", extra), ("withdrawn figure", revived)):
            with self.subTest(label):
                self.assertTrue(hf_publish.card_problems(text, BENCH))
                with self.assertRaises(hf_publish.Refused):
                    hf_publish.build_card(text, BENCH)
        self.assertTrue(hf_publish.card_problems(README, {**BENCH, "functional_pass": 57}))
        self.assertTrue(hf_publish.card_problems(README, {k: v for k, v in BENCH.items() if k != "n"}))

    def test_wrong_metadata_is_refused(self):
        for label, text in (
                ("license", README.replace("license: apache-2.0", "license: mit", 1)),
                ("base_model", README.replace("base_model: Qwen/Qwen3.8-27B", "base_model: Qwen/Other", 1)),
                ("relation", README.replace("base_model_relation: finetune", "base_model_relation: adapter", 1)),
                ("tag", README.replace("- lora\n", "", 1)),
                ("pretty_name", README.replace("pretty_name: Simplicio 27B", "pretty_name: X", 1)),
                ("no front matter", README.split("---\n", 2)[2])):
            with self.subTest(label):
                self.assertTrue(hf_publish.card_problems(text, BENCH))

    def test_generation_config_numbers_come_from_the_modelfile(self):
        params = hf_publish.sampling_params(MODELFILE)
        in_modelfile = {float(v) for name, v in hf_publish.PARAMETER_RE.findall(MODELFILE)
                        if name in ISSUE_SAMPLING}
        self.assertEqual({float(v) for v in params.values()}, in_modelfile)
        self.assertEqual(params["temperature"], 0.7)


class NoDeleteContract(unittest.TestCase):
    """Nothing in scripts/hf_publish.py can delete, move or copy a file on the Hub."""

    SCRIPT_TREE = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    BANNED = re.compile(r"(?i)delete|remove|unlink|rmtree|rmdir|copy|move|rename|squash|CommitOperation(?!Add)")

    def test_no_identifier_or_string_names_a_delete_or_copy(self):
        names = set()
        for node in ast.walk(self.SCRIPT_TREE):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.alias):
                names.add(node.name)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node is not \
                    self.SCRIPT_TREE.body[0].value:  # the module docstring explains the rule in prose
                names.add(node.value)
        self.assertEqual(sorted(n for n in names if self.BANNED.search(n)), [])

    def test_the_only_operation_type_is_add(self):
        classes = {name for name, obj in vars(hf_publish).items()
                   if isinstance(obj, type) and obj.__module__ == "hf_publish"}
        self.assertEqual(classes, {"Entry", "RemoteFile", "Add", "Scan", "Plan", "Refused", "HubClient"})
        self.assertNotIn("Copy", classes | set(vars(hf_publish)))
        self.assertNotIn("Delete", classes | set(vars(hf_publish)))
        self.assertEqual(hf_publish.Plan().ops, [])

    def test_hub_client_makes_only_read_calls_and_one_commit_of_adds(self):
        client = next(n for n in self.SCRIPT_TREE.body if isinstance(n, ast.ClassDef) and n.name == "HubClient")
        calls = {}
        for node in ast.walk(client):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) \
                    and isinstance(node.value.value, ast.Name) and node.value.value.id == "self":
                calls.setdefault(node.value.attr, set()).add(node.attr)
        self.assertEqual(calls, {
            "_api": {"model_info", "list_repo_tree", "hf_hub_download", "create_commit"},
            "_hub": {"CommitOperationAdd"},
        })
        # HfApi(token=...) is the only other thing the client builds.
        built = {node.func.attr for node in ast.walk(client) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                 and node.func.value.id == "hub"}
        self.assertEqual(built, {"HfApi"})

    def test_commit_description_links_the_remote_head(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertRegex(source, r'description = f"\{GITHUB_COMMIT_URL\}\{upstream\}"')
        self.assertIn('git("rev-parse", "@{upstream}")', source)
        self.assertNotRegex(source, r"description = [^\n]*rev-parse")


class NoSecretsContract(unittest.TestCase):
    def test_no_literal_token_or_key_in_script_card_or_notebook(self):
        for path in (SCRIPT, ROOT / "README.md", ROOT / "lora" / "README.md", NOTEBOOK):
            text = path.read_text(encoding="utf-8")
            for pattern in LITERAL_SECRETS:
                self.assertIsNone(re.search(pattern, text), f"{path.name}: {pattern}")
            self.assertIsNone(hf_publish.SECRET_RE.search(text), path.name)

    def test_detector_catches_what_it_should(self):
        for sample in ("hf_" + "x" * 34, "Bearer " + "a" * 24, "sk-" + "b" * 24, "AKIA" + "C" * 16,
                       "-----BEGIN RSA PRIVATE KEY-----"):
            self.assertIsNotNone(hf_publish.SECRET_RE.search(sample), sample)
        self.assertIsNone(hf_publish.SECRET_RE.search("Authorization: Bearer <token>"))

    def test_token_only_comes_from_the_environment(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(hf_publish.TOKEN_ENV, "HF_TOKEN")
        self.assertNotIn("--token", source)
        self.assertNotIn("getpass", source)
        self.assertEqual(re.findall(r"env\.get\((\w+)", source), ["TOKEN_ENV"])

    def test_only_the_standard_library_is_imported_at_the_top(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        top = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
        top |= {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom) and node.module}
        self.assertLessEqual(top, set(sys.stdlib_module_names))
        inner = [node for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
                 for node in ast.walk(fn) if isinstance(node, ast.Import)]
        self.assertEqual([alias.name for node in inner for alias in node.names], ["huggingface_hub"])

    def test_publish_is_opt_in(self):
        parser_source = SCRIPT.read_text(encoding="utf-8")
        self.assertRegex(parser_source, r'add_argument\("--publish", "--apply", dest="publish", action="store_true"')
        self.assertEqual(hf_publish.COMMIT_MESSAGE, "Sync from GitHub: allowlist, adapter in lora/, card metadata")


class LayoutContract(unittest.TestCase):
    def test_lora_is_documented_and_holds_no_weights(self):
        self.assertEqual([p for p in tracked_and_new("lora") if p != "lora/README.md"], [])
        self.assertEqual(tracked_and_new("*.safetensors", "*.gguf"), [])
        self.assertIn("`lora/`", (ROOT / "lora" / "README.md").read_text(encoding="utf-8"))
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("*.safetensors", ignored)
        self.assertIn("*.gguf", ignored)

    def test_notebook_reads_the_adapter_from_lora(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        nb = json.loads(text)
        self.assertEqual(text, json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
        cell = "".join(nb["cells"][4]["source"])
        self.assertIn('allow_patterns=["lora/*"]', cell)
        self.assertIn("model_name=ADAPTER", cell)
        self.assertIn("lora/", "".join(nb["cells"][0]["source"]))

    def test_script_is_executable_and_documented_in_the_readme(self):
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)
        self.assertTrue(SCRIPT.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3\n"))
        self.assertIn("`lora/` |", README)


if __name__ == "__main__":
    unittest.main()
