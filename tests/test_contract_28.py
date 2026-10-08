"""Contract tests for the dataset v2 (#28).

Interfaces other pieces consume: the JSONL rows that the training step (#29) renders with the
serving chat template, the decontamination guarantee against the eval (benchmarks/harness/tasks and
data/unseen_eval_120.json, checked with the criterion of benchmarks/harness/decontam_tasks.py), the
numbers in data/v2/STATS.md, and the fact that the old eval file was not touched. No network, no model.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
EVAL_FILE = ROOT / "data" / "unseen_eval_120.json"
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset_v2 as b  # noqa: E402
dc, FORMAT_HINT, SYSTEM_PROMPT = b.decontam_tasks, b.FORMAT_HINT, b.SYSTEM_PROMPT

KEYS = {"id", "repo", "license", "commit", "parent", "path", "changed_lines", "source", "scope",
        "instruction_source", "messages"}


def load(name: str) -> list[dict]:
    text = (V2 / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def rows() -> list[dict]:
    return load("train.jsonl") + load("val.jsonl")


def pinned_commit_available() -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{b.LOCAL.pin}^{{commit}}"],
                          capture_output=True).returncode == 0


def eval_statements() -> list[tuple[str, str]]:
    """(where, text) for every statement and reference answer of the eval, new and old."""
    out = []
    for d in dc.task_dirs(dc.TASKS):
        meta = json.loads((d / "task.json").read_text(encoding="utf-8"))
        out.append((f"{d.name}/instruction", meta["instruction"].strip()))
        out.append((f"{d.name}/oracle", (d / "oracle.txt").read_text(encoding="utf-8").strip()))
        for search, replace in b.parse_blocks((d / "oracle.txt").read_text(encoding="utf-8")):
            out += [(f"{d.name}/oracle-search", search.strip()), (f"{d.name}/oracle-replace", replace.strip())]
    for r in json.loads(EVAL_FILE.read_text(encoding="utf-8")):
        out.append((f"{r['id']}/task", r["task"].strip()))
    return [(where, text) for where, text in out if len(text) >= 40]


def leaks(row: dict) -> list[str]:
    """Where a statement or reference answer of the eval is inside a string of the row, verbatim."""
    texts = list(dc.strings(row))
    return [where for where, needle in eval_statements() if any(needle in text for text in texts)]


class FileFormat(unittest.TestCase):
    def test_layout_of_data_v2(self):
        self.assertEqual(sorted(p.name for p in V2.iterdir()), ["LICENSES", "STATS.md", "train.jsonl", "val.jsonl"])
        rows_ = rows()
        licenses = {p.name for p in (V2 / "LICENSES").iterdir()}
        self.assertLessEqual({r.replace("/", "__") + ".txt" for r in {x["repo"] for x in rows_}}, licenses)
        # The v1 corpus is not where data/v2 lives: decontamination.json of the eval tasks (#26) keeps its meaning.
        self.assertEqual(sorted(p.name for p in (ROOT / "data").glob("*.jsonl")),
                         ["simplicio_loop_50pts_train.jsonl", "simplicio_loop_50pts_val.jsonl"])

    def test_files_are_valid_jsonl_with_one_object_per_line_and_a_final_newline(self):
        for name in ("train.jsonl", "val.jsonl"):
            raw = (V2 / name).read_bytes()
            self.assertTrue(raw == b"" or raw.endswith(b"\n"), name)
            self.assertNotIn(b"\r", raw, name)
            lines = raw.decode("utf-8").split("\n")[:-1]
            self.assertTrue(all(line.strip() for line in lines), f"{name}: blank line")
            for i, line in enumerate(lines, 1):
                obj = json.loads(line)
                self.assertIsInstance(obj, dict, f"{name}:{i}")
                self.assertEqual(json.dumps(obj, ensure_ascii=False), line, f"{name}:{i}")
        self.assertGreaterEqual(len(load("train.jsonl")), 1)

    def test_every_row_has_the_serving_message_format(self):
        ids = [r["id"] for r in rows()]
        self.assertEqual(len(ids), len(set(ids)))
        for r in rows():
            with self.subTest(row=r["id"]):
                self.assertEqual(set(r), KEYS)
                self.assertEqual([m["role"] for m in r["messages"]], ["system", "user", "assistant"])
                self.assertTrue(all(set(m) == {"role", "content"} for m in r["messages"]))
                system, user, answer = (m["content"] for m in r["messages"])
                self.assertEqual(system, SYSTEM_PROMPT)
                self.assertTrue(user.startswith("Contexto: Projeto Python " + r["repo"] + ".\nArquivo: " + r["path"]
                                                + "\nCodigo Atual:\n"))
                self.assertIn("\nTarefa: ", user)
                self.assertNotIn(FORMAT_HINT.strip(), user)
                # The answer is only SEARCH/REPLACE blocks with the 4-character markers.
                self.assertTrue(answer.startswith("<<<< SEARCH\n") and answer.endswith("\n>>>> REPLACE"))
                self.assertGreaterEqual(len(b.parse_blocks(answer)), 1)
                self.assertNotIn("<think>", answer)
                for marker in b.BANNED:
                    self.assertNotIn(marker, answer)
                self.assertEqual(r["source"], "real_commit")
                self.assertIn(r["scope"], {"single_file_commit", "multi_file_commit"})
                self.assertIn(r["instruction_source"], {"commit_subject", "curated_from_commit"})
                self.assertIn(r["license"], {"Apache-2.0", "MIT", "BSD-3-Clause"})
                self.assertRegex(r["commit"], r"^[0-9a-f]{40}$")
                self.assertRegex(r["parent"], r"^[0-9a-f]{40}$")
                self.assertTrue(1 <= r["changed_lines"] <= b.MAX_CHANGED_LINES)

    def test_no_marker_word_of_an_invented_result_in_either_file(self):
        for name in ("train.jsonl", "val.jsonl"):
            text = (V2 / name).read_text(encoding="utf-8")
            self.assertEqual(text.count("COMMIT_READY") + text.count("VERIFIED_GREEN"), 0, name)

    def test_every_row_has_the_license_text_of_its_repo(self):
        for r in rows():
            path = V2 / "LICENSES" / (r["repo"].replace("/", "__") + ".txt")
            self.assertGreater(path.stat().st_size, 100, r["id"])
        ours = V2 / "LICENSES" / "simpletibr__simplicio-27b.txt"
        if ours.is_file():
            self.assertEqual(hashlib.sha256(ours.read_bytes()).hexdigest(),
                             hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest())


class Decontamination(unittest.TestCase):
    def test_no_row_shares_a_13_gram_with_any_eval_task_or_the_old_eval_rows(self):
        grams = b.eval_grams()
        self.assertEqual([r["id"] for r in rows() if b.contaminated(r, grams)], [])

    def test_the_repo_report_finds_no_overlap_between_data_v2_and_the_40_tasks(self):
        report = dc.build_report(dc.TASKS, V2, ROOT)
        self.assertEqual([c["path"] for c in report["corpus"]], ["data/v2/train.jsonl", "data/v2/val.jsonl"])
        self.assertEqual(len(report["tasks"]), 40)
        self.assertEqual([t["id"] for t in report["tasks"] if t["overlaps"]], [])

    def test_no_statement_and_no_reference_answer_of_the_eval_is_inside_a_row(self):
        self.assertGreaterEqual(len(eval_statements()), 120)
        for r in rows():
            self.assertEqual(leaks(r), [], r["id"])

    def test_the_checks_are_not_vacuous_a_copied_statement_or_oracle_is_flagged(self):
        grams = b.eval_grams()
        first = next(iter(dc.task_dirs(dc.TASKS)))
        meta = json.loads((first / "task.json").read_text(encoding="utf-8"))
        oracle = (first / "oracle.txt").read_text(encoding="utf-8")
        old_row = json.loads(EVAL_FILE.read_text(encoding="utf-8"))[0]
        copies = {"instruction": meta["instruction"], "oracle": oracle, "old statement": old_row["task"]}
        for label, text in copies.items():
            row = rows()[0] | {"messages": [{"role": "user", "content": "Tarefa: " + text}]}
            with self.subTest(copied=label):
                self.assertNotEqual(leaks(row), [])
        # The 13-gram test is the one the repo uses; it needs 13 tokens, so it sees the long copies only.
        for label in ("instruction", "oracle"):
            row = rows()[0] | {"messages": [{"role": "user", "content": "Tarefa: " + copies[label]}]}
            with self.subTest(ngram=label):
                self.assertTrue(b.contaminated(row, grams))
                with tempfile.TemporaryDirectory() as tmp:
                    (Path(tmp) / "x.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
                    report = dc.build_report(dc.TASKS, tmp, tmp)
                self.assertIn(first.name, [t["id"] for t in report["tasks"] if t["overlaps"]])
        old_all = rows()[0] | {"messages": [{"role": "user", "content": "\n".join(dc.strings(old_row))}]}
        self.assertTrue(b.contaminated(old_all, grams))
        self.assertFalse(b.contaminated(rows()[0], grams))


class StatsAndReproducibility(unittest.TestCase):
    def test_stats_md_agrees_with_the_files_and_lists_what_is_missing(self):
        stats = (V2 / "STATS.md").read_text(encoding="utf-8")
        train, val = load("train.jsonl"), load("val.jsonl")
        self.assertIn(f"- Linhas: train {len(train)}, val {len(val)}", stats)
        self.assertIn(f"- Origem: {len(train) + len(val)} de commit real, 0 sinteticos", stats)
        self.assertIn(f"faltam {max(0, b.MIN_TRAIN - len(train))}", stats)
        self.assertEqual(sum(r["source"] == "real_commit" for r in train + val), len(train) + len(val))
        if len(train) < b.MIN_TRAIN:
            self.assertIn("## Pendencias", stats)
            for s in b.EXTERNAL:
                self.assertIn(f"| {s.name} | {s.license} | `{s.pin}` |", stats)
        self.assertIn(f"| {b.LOCAL.name} | {b.LOCAL.license} | `{b.LOCAL.pin}` |", stats)

    @unittest.skipUnless(pinned_commit_available(), f"{b.LOCAL.pin} is not in this checkout (shallow clone?)")
    def test_rebuilding_from_the_pinned_history_gives_the_committed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(b.main(["--out", tmp, "--workdir", str(Path(tmp) / "w")]), 0)
            for name in ("train.jsonl", "val.jsonl", "STATS.md"):
                self.assertEqual((Path(tmp) / name).read_bytes(), (V2 / name).read_bytes(), name)

    @unittest.skipUnless(pinned_commit_available(), f"{b.LOCAL.pin} is not in this checkout (shallow clone?)")
    def test_every_row_is_a_commit_of_the_pinned_history_that_touches_its_file(self):
        for r in rows():
            with self.subTest(row=r["id"]):
                ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", r["commit"], b.LOCAL.pin])
                self.assertEqual(ancestor.returncode, 0)
                parent = subprocess.run(["git", "-C", str(ROOT), "rev-parse", r["commit"] + "^"], capture_output=True, text=True)
                self.assertEqual(parent.stdout.strip(), r["parent"])
                changed = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", r["parent"], r["commit"]],
                                         capture_output=True, text=True).stdout.split()
                self.assertIn(r["path"], changed)


class EvalFileUntouched(unittest.TestCase):
    """Update this class when a later issue deliberately changes or deletes the file."""

    def test_eval_file_still_has_120_rows_and_40_unique_tasks(self):
        data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 120)
        self.assertEqual(len({(r["original_code"], r["task"], r["test_assertion"]) for r in data}), 40)

    def test_eval_file_is_identical_to_origin_main(self):
        def git(*args):
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)

        if git("rev-parse", "--verify", "--quiet", "origin/main").returncode != 0:
            self.skipTest("origin/main is not available in this checkout")
        diff = git("diff", "--quiet", "origin/main", "--", "data/unseen_eval_120.json")
        self.assertEqual(diff.returncode, 0, "data/unseen_eval_120.json differs from origin/main")

    def test_builder_and_tests_do_not_write_to_the_eval_inputs(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [EVAL_FILE, *sorted(dc.TASKS.rglob("*"))] if p.is_file()}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            b.build([b.LOCAL], Path(tmp) / "o", Path(tmp) / "w", b.eval_grams(), limit=1)
        after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
