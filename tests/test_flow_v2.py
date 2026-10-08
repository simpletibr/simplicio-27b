"""Fluxo do dataset v2 (scripts/build_dataset_v2.py): o gerador roda em modo pequeno, todo exemplo
passa em loop_check --complete e o patch do envelope, aplicado ao arquivo original, faz o teste passar.

Sem rede, sem modelo, sem GPU. As partes que rodam o pytest de verdade (o mesmo harness do eval) so
rodam onde o pytest esta instalado (python3 -m pip install -r benchmarks/harness/requirements.txt),
como em tests/test_eval_harness.py; as demais (cada forma de bug, o envelope, as recusas) rodam em
qualquer Python.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_dataset_v2 as bd  # noqa: E402

# Os mesmos modulos que o gerador carregou (famílias, loop_check, harness): um unico conjunto de classes.
fam, loop_check, harness = bd.fam, bd.loop_check, bd.harness

needs_pytest = unittest.skipUnless(
    importlib.util.find_spec("pytest"),
    "pytest is not installed: python3 -m pip install -r benchmarks/harness/requirements.txt",
)


def run_main(*argv: str) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = bd.main(list(argv))
    return code, out.getvalue() + err.getvalue()


def with_block(trajectory: str, *, search: str | None = None, replace: str | None = None) -> str:
    """Troca so o SEARCH e/ou o REPLACE do bloco do envelope, sem mexer no resto do texto."""
    head, rest = trajectory.split("<<<< SEARCH\n", 1)
    old_search, rest = rest.split("\n====\n", 1)
    old_replace, tail = rest.split("\n>>>> REPLACE", 1)
    new_search = old_search if search is None else search
    new_replace = old_replace if replace is None else replace
    return f"{head}<<<< SEARCH\n{new_search}\n====\n{new_replace}\n>>>> REPLACE{tail}"


def read_rows(directory: Path) -> list[dict]:
    rows = []
    for name in ("train.jsonl", "val.jsonl"):
        rows += [json.loads(line) for line in (directory / name).read_text(encoding="utf-8").splitlines() if line]
    return rows


class EveryShapeTests(unittest.TestCase):
    """Sem pytest: cada uma das formas de bug gera um arquivo com defeito e um patch que o corrige."""

    def test_every_shape_fails_before_and_passes_after_the_patch(self) -> None:
        checked = 0
        for family, shapes in sorted(fam.SHAPES.items()):
            for shape_name, _ in shapes:
                for di in (0, 17, 49):
                    for k in range(fam.VARIANTS):
                        with self.subTest(family=family, shape=shape_name, domain=di, k=k):
                            case = bd.assemble(family, shape_name, di, k)
                            self.assertIsNotNone(case)
                            before = fam.observe(case.original, case.tests)
                            after = fam.observe(case.fixed, case.tests)
                            self.assertTrue([n for n, m in before.items() if m], "o bug deve derrubar algum teste")
                            self.assertEqual([n for n, m in after.items() if m], [], "o arquivo corrigido deve passar tudo")
                            blocks = harness.parse_blocks(bd.patch_block(case.search, case.replace))
                            self.assertEqual(len(blocks), 1)
                            self.assertEqual(harness.apply_blocks(case.original, blocks), case.fixed)
                            self.assertEqual(case.original.count(case.search), 1)
                            self.assertNotIn(case.search, case.fixed)
                            checked += 1
        self.assertGreaterEqual(checked, 68 * 3 * fam.VARIANTS)

    def test_the_families_the_task_asks_for_exist(self) -> None:
        wanted = {"off_by_one", "inverted_comparison", "mutable_default", "missing_strip", "operation_order",
                  "wrong_type", "dict_key"}
        self.assertLessEqual(wanted, set(fam.SHAPES))
        self.assertGreaterEqual(len(fam.SHAPES), 15)
        self.assertGreaterEqual(len(fam.candidates()), 2222, "a capacidade precisa cobrir 2000 train + val")

    def test_envelope_of_a_built_case_has_the_50_points(self) -> None:
        for family, shape_name, di, k in (("off_by_one", "page_slice", 3, 0), ("mutable_default", "list_default", 5, 1),
                                         ("return_flow", "missing_return", 7, 0), ("dict_key", "group_rows", 9, 1)):
            with self.subTest(shape=shape_name):
                case = bd.assemble(family, shape_name, di, k)
                before = fam.observe(case.original, case.tests)
                trajectory = bd.build_trajectory(case, before, case.fixed)
                self.assertEqual(loop_check.check(trajectory, complete=True), [])
                self.assertEqual(trajectory.count("[Ponto "), 50)
                self.assertTrue(trajectory.isascii())
                blocks = harness.parse_blocks(trajectory)
                self.assertEqual(blocks, [(case.search, case.replace)])
                self.assertIn("<<<< SEARCH\n", trajectory)  # marcadores de exatamente 4 caracteres
                self.assertNotIn("<<<<<", trajectory)

    def test_a_candidate_whose_bug_is_not_a_bug_is_refused(self) -> None:
        draft = fam.build_draft("off_by_one", "span_days", 0, 0)
        equivalent = dataclasses.replace(draft, bug="return 1 + last - first", fix="return last - first + 1")
        with mock.patch.object(bd.fam, "build_draft", return_value=equivalent):
            self.assertEqual(bd.process(("off_by_one", "span_days", 0, 0)), "bug_not_observed")

    def test_a_candidate_whose_fix_does_not_fix_is_refused(self) -> None:
        draft = fam.build_draft("off_by_one", "span_days", 0, 0)
        wrong = dataclasses.replace(draft, fix="return last - first + 2")
        with mock.patch.object(bd.fam, "build_draft", return_value=wrong):
            self.assertEqual(bd.process(("off_by_one", "span_days", 0, 0)), "fix_not_observed")

    def test_without_pytest_the_generator_stops_with_a_clear_message(self) -> None:
        with mock.patch.object(importlib.util, "find_spec", return_value=None), tempfile.TemporaryDirectory() as tmp:
            code, text = run_main("--out", tmp, "--limit", "2")
        self.assertEqual(code, 2)
        self.assertIn("pytest", text)

    def test_cli_help_runs(self) -> None:
        proc = subprocess.run([sys.executable, str(ROOT / "scripts/build_dataset_v2.py"), "--help"],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--verify", proc.stdout)


@needs_pytest
class SmallRunTests(unittest.TestCase):
    """Com pytest: o gerador de verdade, em modo pequeno, e uma reverificacao independente."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.code, cls.log = run_main("--out", str(cls.out), "--limit", "16", "--workers", "2")
        cls.rows = read_rows(cls.out)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_run_succeeds_and_writes_the_files(self) -> None:
        self.assertEqual(self.code, 0, self.log)
        self.assertEqual({p.name for p in self.out.iterdir()}, {"train.jsonl", "val.jsonl", "manifest.json"})
        self.assertEqual(len(self.rows), 16)
        self.assertEqual(len({r["id"] for r in self.rows}), 16)
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["rejected"], {})
        self.assertEqual(manifest["decontamination"]["eval_tasks"]["with_overlap"], 0)
        self.assertEqual(manifest["decontamination"]["unseen_eval_120"]["with_overlap"], 0)

    def test_records_keep_the_fields_of_the_current_dataset(self) -> None:
        v1 = json.loads((ROOT / "data/simplicio_loop_50pts_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        for row in self.rows:
            self.assertLessEqual(set(v1), set(row))
            for key in v1:
                self.assertIsInstance(row[key], str)
                self.assertTrue(row[key].strip())

    def test_every_example_passes_loop_check_complete(self) -> None:
        for row in self.rows:
            with self.subTest(id=row["id"]):
                self.assertEqual(loop_check.check(row["simplicio_trajectory"], complete=True), [])
        for name in ("train.jsonl", "val.jsonl"):
            if (self.out / name).read_text(encoding="utf-8").strip():
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/loop_check.py"), "--jsonl", str(self.out / name), "--complete"],
                    capture_output=True, text=True, timeout=120)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_the_patch_in_the_envelope_makes_the_test_pass(self) -> None:
        for row in self.rows:
            with self.subTest(id=row["id"]):
                prefix = f"Arquivo: {row['edit_file']}\nCodigo Atual:\n"
                self.assertTrue(row["context"].startswith(prefix))
                original = row["context"][len(prefix):]
                blocks = harness.parse_blocks(row["simplicio_trajectory"])
                self.assertEqual(len(blocks), 1)
                patched = harness.apply_blocks(original, blocks)
                self.assertNotEqual(patched, original)
                with tempfile.TemporaryDirectory() as tmp:
                    task_dir = Path(tmp) / "t"
                    (task_dir / "files").mkdir(parents=True)
                    (task_dir / "tests").mkdir()
                    (task_dir / "files" / row["edit_file"]).write_text(original, encoding="utf-8")
                    (task_dir / "tests" / Path(row["test_file"]).name).write_text(row["test_source"], encoding="utf-8")
                    task = {"dir": task_dir, "edit_file": row["edit_file"]}
                    before = harness.check_source(task, original)
                    after = harness.check_source(task, patched)
                self.assertFalse(before["passed"], "o teste deve falhar no arquivo original")
                self.assertGreaterEqual(before["report"]["failures"], 1)
                self.assertEqual(before["report"]["errors"], 0, "falha de teste, nao erro de importacao")
                self.assertTrue(after["passed"], after["stdout_tail"])
                self.assertEqual(after["report"]["failures"] + after["report"]["errors"], 0)

    def test_examples_come_from_many_shapes(self) -> None:
        # o modo pequeno percorre as formas em rodizio (ordem de familia, forma): 16 exemplos = 16 formas
        self.assertEqual(len({r["shape"] for r in self.rows}), 16)
        self.assertGreaterEqual(len({r["family"] for r in self.rows}), 4)

    def test_verify_mode_accepts_the_output(self) -> None:
        code, log = run_main("--verify", str(self.out), "--sample", "4")
        self.assertEqual(code, 0, log)
        self.assertIn("0 com problema", log)

    def test_verify_rejects_a_patch_that_does_not_fix_the_test(self) -> None:
        row = dict(self.rows[0])
        search, _ = harness.parse_blocks(row["simplicio_trajectory"])[0]
        row["simplicio_trajectory"] = with_block(row["simplicio_trajectory"], replace=search)
        # o envelope continua valido (loop_check passa), mas o patch virou um no-op: o teste continua falhando
        self.assertEqual(loop_check.check(row["simplicio_trajectory"], complete=True), [])
        problems = bd.verify_record(row)
        self.assertTrue(any(p.startswith("teste:") for p in problems), problems)

    def test_verify_rejects_a_search_that_is_not_in_the_file(self) -> None:
        row = dict(self.rows[0])
        search, _ = harness.parse_blocks(row["simplicio_trajectory"])[0]
        row["simplicio_trajectory"] = with_block(row["simplicio_trajectory"], search=search + " # ausente")
        problems = bd.verify_record(row)
        self.assertTrue(any("search_not_found" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
