"""Contract tests for the executable eval tasks (#26).

Interfaces other pieces consume: the layout `load_tasks` and the A/B run (#27) read, the link
between each task and `data/unseen_eval_120.json`, the guarantee that no statement hands the
answer to the model, and the fact that the old eval file was not touched. No network, no model.
"""
import ast
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "benchmarks/harness/tasks"
EVAL_FILE = ROOT / "data/unseen_eval_120.json"
sys.path.insert(0, str(ROOT / "benchmarks/harness"))
import harness  # noqa: E402

KEYS = {"id", "instruction", "context", "edit_file", "category", "source_ids", "source_prompt", "forbidden_symbols"}
# types.py shadows the stdlib module, so the harness (which puts the work dir on sys.path) cannot use that name.
RENAMED = {"cat1_ast_diff_06_diff_optional_type_union": "item_lookup.py"}
# Literals the old prompts handed over (answer, API or code to write). None may be back in an instruction.
LEAKS = {
    "cat1_ast_diff_01": ["Field(", "min_length=11"],
    "cat1_ast_diff_02": ["deprecated=True"],
    "cat1_ast_diff_03": ["frozen=True"],
    "cat1_ast_diff_04": ["await http_get"],
    "cat1_ast_diff_05": ["logger.info(", "%s"],
    "cat1_ast_diff_06": ["str | None"],
    "cat1_ast_diff_07": ["try/except FileNotFoundError"],
    "cat1_ast_diff_08": ["[x for x in nums if x % 2 == 0]"],
    "cat1_ast_diff_09": ["{k: v.upper() for k, v in items}"],
    "cat1_ast_diff_10": ["remove brackets", "generator expression"],
    "cat2_edge_functional_05": ["with open("],
    "cat2_edge_functional_06": ["container: list | None", "= None"],
    "cat2_edge_functional_07": ["deepcopy"],
    "cat2_edge_functional_08": ["\\Z"],
    "cat2_edge_functional_09": ["Decimal", "decimal"],
    "cat2_edge_functional_10": ["threading.Lock", "self._lock"],
    "cat3_ghost_api_trap_01": ["field_validator", "@validator"],
    "cat3_ghost_api_trap_02": ["pd.concat", "df.append"],
    "cat3_ghost_api_trap_03": ["select(", "session.query", "scalars"],
    "cat3_ghost_api_trap_04": [".flatten()", "chain"],
    "cat3_ghost_api_trap_05": ["json.load", "json.loads"],
    "cat3_ghost_api_trap_06": ["utcnow", "timezone.utc"],
    "cat3_ghost_api_trap_07": ["asyncio.run", "get_event_loop"],
    "cat3_ghost_api_trap_08": ["has_key", "k in d"],
    "cat3_ghost_api_trap_09": ["isdigit", "isnumeric", "is_digit"],
    "cat3_ghost_api_trap_10": ["math.round", "round(x)"],
    "cat4_adv_ood_01": ["subtotal * 1.1", "return 0"],
    "cat4_adv_ood_03": ["key=lambda", "x['id']"],
    "cat4_adv_ood_05": ["casefold"],
    "cat4_adv_ood_07": ["items = list(gen)", "list(gen)"],
    "cat4_adv_ood_09": [":="],
    "cat4_adv_ood_10": ["suppress(FileNotFoundError)", "contextlib"],
}
# The 8 tasks whose old statement already described behavior only, so the issue keeps it.
KEPT = {
    "cat2_edge_functional_01", "cat2_edge_functional_02", "cat2_edge_functional_03", "cat2_edge_functional_04",
    "cat4_adv_ood_02", "cat4_adv_ood_04", "cat4_adv_ood_06", "cat4_adv_ood_08",
}


def eval_dirs():
    return sorted(
        d for d in TASKS.iterdir()
        if d.is_dir() and (d / "task.json").is_file() and "source_ids" in json.loads((d / "task.json").read_text())
    )


def meta(d):
    return json.loads((d / "task.json").read_text(encoding="utf-8"))


def unique_rows():
    groups = {}
    for r in json.loads(EVAL_FILE.read_text(encoding="utf-8")):
        groups.setdefault((r["original_code"], r["task"], r["test_assertion"]), []).append(r)
    return list(groups.values())


class TasksContract(unittest.TestCase):
    def test_forty_unique_tasks_cover_the_120_rows(self):
        dirs = eval_dirs()
        self.assertEqual(len(dirs), 40)
        rows = unique_rows()
        self.assertEqual(len(rows), 40)
        self.assertEqual(sorted(d.name for d in dirs), sorted(rs[0]["id"] for rs in rows))
        covered = sorted(i for d in dirs for i in meta(d)["source_ids"])
        self.assertEqual(covered, sorted(r["id"] for rs in rows for r in rs))
        self.assertEqual(len(covered), 120)
        self.assertEqual(len(set(covered)), 120)

    def test_tasks_dir_holds_only_the_eval_tasks_and_the_examples_live_in_fixtures(self):
        self.assertEqual(len(list(TASKS.glob("*/task.json"))), 40)
        self.assertEqual(sorted(p.parent.name for p in TASKS.glob("*/task.json")), [d.name for d in eval_dirs()])
        self.assertEqual(
            sorted(p.parent.name for p in (ROOT / "tests/fixtures/harness_tasks").glob("*/task.json")),
            ["ex_dedupe_order", "ex_parse_port", "ex_safe_divide"],
        )

    def test_load_tasks_accepts_the_directory_and_finds_every_task(self):
        loaded = {t["id"] for t in harness.load_tasks(TASKS)}
        self.assertEqual({d.name for d in eval_dirs()}, loaded)

    def test_task_json_has_exactly_the_agreed_keys_and_links_to_the_eval_file(self):
        by_id = {rs[0]["id"]: rs for rs in unique_rows()}
        for d in eval_dirs():
            with self.subTest(task=d.name):
                m = meta(d)
                rs = by_id[d.name]
                self.assertEqual(set(m), KEYS)
                self.assertEqual(m["id"], d.name)
                self.assertEqual(m["context"], "")
                self.assertEqual(m["source_ids"], [r["id"] for r in rs])
                self.assertEqual(m["source_prompt"], rs[0]["task"])
                self.assertEqual(m["category"], rs[0]["category"])
                self.assertEqual(m["forbidden_symbols"], rs[0]["forbidden_symbols"])
                self.assertEqual(m["edit_file"], RENAMED.get(d.name, rs[0]["file"]))
                self.assertTrue(m["instruction"].strip())

    def test_each_task_has_files_pytest_tests_and_an_oracle(self):
        for d in eval_dirs():
            with self.subTest(task=d.name):
                m = meta(d)
                source = d / "files" / m["edit_file"]
                self.assertTrue(source.is_file())
                self.assertFalse((d / "files" / "tests").exists())
                tests = sorted((d / "tests").glob("test_*.py"))
                self.assertEqual([t.name for t in tests], ["test_task.py"])
                self.assertEqual(sorted(p.name for p in d.rglob("conftest.py")), [])
                self.assertEqual(sorted(p.name for p in d.rglob("__init__.py")), [])
                blocks = harness.parse_blocks((d / "oracle.txt").read_text(encoding="utf-8"))
                self.assertGreaterEqual(len(blocks), 1)
                self.assertNotEqual(harness.apply_blocks(source.read_text(encoding="utf-8"), blocks),
                                    source.read_text(encoding="utf-8"))

    def test_pytest_files_follow_the_header_and_never_load_at_module_level(self):
        for d in eval_dirs():
            with self.subTest(task=d.name):
                text = (d / "tests/test_task.py").read_text(encoding="utf-8")
                tree = ast.parse(text)
                edit_file = meta(d)["edit_file"]
                consts = {
                    n.targets[0].id: n.value.value for n in tree.body
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.targets[0], ast.Name)
                }
                self.assertEqual(consts.get("FILE"), edit_file)
                funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
                self.assertIn("load", funcs)
                self.assertTrue(any(f.startswith("test_") for f in funcs))
                self.assertIn('os.environ["TASK_SRC"]', text)
                # load() is called inside tests only: a module-level call would run the file at collection.
                top_calls = [
                    n for n in tree.body if isinstance(n, (ast.Expr, ast.Assign))
                    and any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == "load" for c in ast.walk(n))
                ]
                self.assertEqual(top_calls, [])
                self.assertEqual("import ast" in text.splitlines(), "def fn_ast" in text)

    def test_requirements_pin_the_packages_the_tasks_import(self):
        lines = (ROOT / "benchmarks/harness/requirements.txt").read_text().split()
        self.assertEqual(
            lines,
            ["pytest==9.1.1", "pydantic==2.13.5", "fastapi==0.142.2", "pandas==3.0.6", "SQLAlchemy==2.1.3"],
        )
        self.assertEqual((TASKS / "ruff.toml").read_text().strip(), 'extend-exclude = ["*/files"]')

    def test_decontamination_report_is_committed_with_the_corpus_hashes(self):
        report = json.loads((TASKS / "decontamination.json").read_text(encoding="utf-8"))
        self.assertEqual(report["n"], 13)
        self.assertEqual(
            {c["path"]: c["sha256"] for c in report["corpus"]},
            {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((ROOT / "data").glob("*.jsonl"))},
        )
        self.assertEqual([t["id"] for t in report["tasks"] if t["overlaps"]], [])
        self.assertLessEqual({d.name for d in eval_dirs()}, {t["id"] for t in report["tasks"]})


class NoAnswerInThePrompt(unittest.TestCase):
    def test_no_instruction_carries_the_answer_or_a_trap_hint(self):
        for d in eval_dirs():
            m = meta(d)
            with self.subTest(task=d.name):
                self.assertNotIn("TRAP", m["instruction"])
                for key, literals in LEAKS.items():
                    if d.name.startswith(key):
                        for literal in literals:
                            self.assertNotIn(literal, m["instruction"], f"{d.name} leaks {literal!r}")

    def test_32_statements_were_rewritten_and_8_were_kept(self):
        changed = {d.name for d in eval_dirs() if meta(d)["instruction"] != meta(d)["source_prompt"]}
        self.assertEqual(len(changed), 32)
        kept = {d.name for d in eval_dirs()} - changed
        self.assertEqual(len(kept), 8)
        self.assertTrue(all(any(n.startswith(k) for k in KEPT) for n in kept))
        self.assertEqual({meta(d)["context"] for d in eval_dirs()}, {""})
        for category in ("cat1", "cat2", "cat3", "cat4"):
            self.assertEqual(sum(d.name.startswith(category + "_") for d in eval_dirs()), 10)
        self.assertEqual(sum("TRAP" in meta(d)["instruction"] for d in eval_dirs()), 0)


class EvalFileUntouched(unittest.TestCase):
    """Update this class when a later issue deliberately changes or deletes the file."""

    def test_eval_file_still_has_120_rows_and_40_unique_tasks(self):
        rows = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 120)
        self.assertEqual(len(unique_rows()), 40)

    def test_eval_file_is_identical_to_origin_main(self):
        def git(*args):
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)

        if git("rev-parse", "--verify", "--quiet", "origin/main").returncode != 0:
            self.skipTest("origin/main is not available in this checkout")
        diff = git("diff", "--quiet", "origin/main", "--", "data/unseen_eval_120.json")
        self.assertEqual(diff.returncode, 0, "data/unseen_eval_120.json differs from origin/main")


if __name__ == "__main__":
    unittest.main()
