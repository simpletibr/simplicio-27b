"""Contrato do dataset v2 (data/v2 e scripts/build_dataset_v2.py).

Fixa o que outros componentes consomem: os nomes e o formato dos arquivos de treino (os mesmos tres
campos do dataset atual), o envelope de 50 pontos que scripts/loop_check.py --complete aceita, o
formato SEARCH/REPLACE de 4 caracteres que benchmarks/harness/harness.py aplica, e as garantias de
dados: determinismo, split sem sobreposicao, decontaminacao zero contra o eval e o dataset atual
intacto. Sem rede, sem modelo, sem GPU. As checagens que rodam o pytest de verdade exigem pytest
(benchmarks/harness/requirements.txt); as demais rodam em qualquer Python.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_dataset_v2 as bd  # noqa: E402

# Os mesmos modulos que o gerador carregou: um unico conjunto de classes.
fam, loop_check, harness, decontam_tasks = bd.fam, bd.loop_check, bd.harness, bd.decontam

V2 = ROOT / "data/v2"
TRAIN_CAP = 2000
# O dataset atual e a referencia: estes sha256 sao os de origin/main; o v2 nunca os altera.
CURRENT_DATA = {
    "data/simplicio_loop_50pts_train.jsonl": "e0eb6d4304bdc24df22ac354707868a589fa754c52d37fb8236c762c3d934606",
    "data/simplicio_loop_50pts_val.jsonl": "267e93e0e1d84788054f955e218807b0082b0c29510990a4bac5ebf3d5a929d7",
    "data/unseen_eval_120.json": "b77450bfd88f619240637230d7f46d2f2c64c1f0eb93c78f6a02e593fd87cf2a",
}
FIELDS = ("instruction", "context", "simplicio_trajectory")
EXTRA_FIELDS = ("id", "family", "shape", "domain", "edit_file", "test_file", "test_source", "tests_total",
                "tests_failed_before", "original_sha256", "patched_sha256")

needs_pytest = unittest.skipUnless(
    importlib.util.find_spec("pytest"),
    "pytest is not installed: python3 -m pip install -r benchmarks/harness/requirements.txt",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (V2 / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def quiet_main(*argv: str) -> int:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return bd.main(list(argv))


class CommittedDataTests(unittest.TestCase):
    """Os arquivos que estao no repositorio."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.train = load("train.jsonl")
        cls.val = load("val.jsonl")
        cls.manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))

    def test_file_names_and_sizes(self) -> None:
        self.assertEqual(sorted(p.name for p in V2.iterdir()), ["manifest.json", "train.jsonl", "val.jsonl"])
        self.assertGreater(len(self.train), 1000)
        self.assertLessEqual(len(self.train), TRAIN_CAP)
        self.assertGreater(len(self.val), 100)
        # val e a fracao de grupos 1 em 10: entre 5% e 20% do total
        share = len(self.val) / (len(self.val) + len(self.train))
        self.assertTrue(0.05 <= share <= 0.20, share)

    def test_records_have_the_three_fields_of_the_current_dataset_plus_audit_fields(self) -> None:
        v1 = json.loads((ROOT / "data/simplicio_loop_50pts_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(tuple(v1), FIELDS)
        for rows in (self.train, self.val):
            for row in rows:
                self.assertEqual(tuple(row)[: len(FIELDS)], FIELDS)
                self.assertEqual(set(row), set(FIELDS) | set(EXTRA_FIELDS))
                for key in FIELDS + ("id", "family", "shape", "domain", "edit_file", "test_file", "test_source"):
                    self.assertIsInstance(row[key], str)
                    self.assertTrue(row[key].strip(), key)

    def test_every_envelope_passes_loop_check_complete(self) -> None:
        bad = []
        for rows in (self.train, self.val):
            for row in rows:
                problems = loop_check.check(row["simplicio_trajectory"], complete=True)
                if problems:
                    bad.append((row["id"], problems[:2]))
        self.assertEqual(bad, [])

    def test_loop_check_cli_accepts_both_files(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/loop_check.py"), "--jsonl", str(V2 / "train.jsonl"),
             "--jsonl", str(V2 / "val.jsonl"), "--complete"],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout[-800:] + proc.stderr[-400:])
        self.assertIn("0 falham", proc.stdout)

    def test_every_patch_is_a_single_four_character_block_that_applies_once(self) -> None:
        for rows in (self.train, self.val):
            for row in rows:
                trajectory = row["simplicio_trajectory"]
                self.assertEqual(trajectory.count("<<<< SEARCH\n"), 1, row["id"])
                self.assertEqual(trajectory.count("\n====\n"), 1, row["id"])
                self.assertEqual(trajectory.count("\n>>>> REPLACE\n"), 1, row["id"])
                self.assertNotRegex(trajectory, r"(?m)^(<{5,7} SEARCH|={5,7}|>{5,7} REPLACE)$")
                prefix = f"Arquivo: {row['edit_file']}\nCodigo Atual:\n"
                self.assertTrue(row["context"].startswith(prefix), row["id"])
                original = row["context"][len(prefix):]
                blocks = harness.parse_blocks(trajectory)
                self.assertEqual(len(blocks), 1, row["id"])
                search, replace = blocks[0]
                self.assertEqual(original.count(search), 1, f"{row['id']}: o SEARCH deve existir uma vez")
                patched = harness.apply_blocks(original, blocks)
                self.assertNotEqual(patched, original)
                self.assertNotIn(search, patched, f"{row['id']}: reaplicar o patch nao pode funcionar")
                ast.parse(original)
                ast.parse(patched)
                self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), row["original_sha256"], row["id"])
                self.assertEqual(hashlib.sha256(patched.encode()).hexdigest(), row["patched_sha256"], row["id"])

    def test_the_test_source_imports_the_module_it_checks(self) -> None:
        for rows in (self.train, self.val):
            for row in rows:
                module = Path(row["edit_file"]).stem
                self.assertRegex(row["test_source"], rf"(?m)^from {module} import ")
                self.assertEqual(row["test_file"], f"tests/test_{module}.py")
                self.assertGreaterEqual(row["tests_failed_before"], 1)
                self.assertGreater(row["tests_total"], row["tests_failed_before"])

    def test_train_and_val_do_not_overlap_and_have_no_duplicates(self) -> None:
        for key in ("id", "context", "simplicio_trajectory", "test_source", "original_sha256", "patched_sha256"):
            train, val = [r[key] for r in self.train], [r[key] for r in self.val]
            self.assertEqual(len(set(train)), len(train), f"{key} duplicado dentro de train")
            self.assertEqual(len(set(val)), len(val), f"{key} duplicado dentro de val")
            self.assertEqual(set(train) & set(val), set(), f"{key} em train e em val")
        lines_train = (V2 / "train.jsonl").read_text(encoding="utf-8").splitlines()
        lines_val = (V2 / "val.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(set(lines_train)), len(lines_train))
        self.assertEqual(len(set(lines_val)), len(lines_val))
        self.assertEqual(set(lines_train) & set(lines_val), set())
        prompts = [(r["instruction"], r["context"]) for r in self.train + self.val]
        self.assertEqual(len(set(prompts)), len(prompts))

    def test_split_is_the_documented_stable_hash_of_the_group(self) -> None:
        for name, rows in (("train", self.train), ("val", self.val)):
            for row in rows:
                group = f"{row['family']}/{row['shape']}/{row['domain']}"
                self.assertEqual(bd.split_of(group), name, row["id"])
        groups_train = {f"{r['family']}/{r['shape']}/{r['domain']}" for r in self.train}
        groups_val = {f"{r['family']}/{r['shape']}/{r['domain']}" for r in self.val}
        self.assertEqual(groups_train & groups_val, set(), "um grupo nunca esta nos dois lados")

    def test_manifest_matches_the_files(self) -> None:
        manifest = self.manifest
        self.assertIn("locais e sinteticas", manifest["sources"])
        for name, rows in (("train.jsonl", self.train), ("val.jsonl", self.val)):
            self.assertEqual(manifest["files"][name]["examples"], len(rows))
            self.assertEqual(manifest["files"][name]["sha256"], sha256_file(V2 / name))
        by_family: dict[str, dict[str, int]] = {}
        for split, rows in (("train", self.train), ("val", self.val)):
            for row in rows:
                entry = by_family.setdefault(row["family"], {"train": 0, "val": 0})
                entry[split] += 1
        self.assertEqual({k: {"train": v["train"], "val": v["val"]} for k, v in manifest["families"].items()}, by_family)
        self.assertEqual(manifest["train_cap"], TRAIN_CAP)
        self.assertEqual(manifest["candidates_enumerated"], len(fam.candidates()))

    def test_the_bug_families_are_varied(self) -> None:
        families = {r["family"] for r in self.train}
        self.assertLessEqual({"off_by_one", "inverted_comparison", "mutable_default", "missing_strip",
                              "operation_order", "wrong_type", "dict_key"}, families)
        self.assertGreaterEqual(len(families), 15)
        self.assertGreaterEqual(len({r["shape"] for r in self.train}), 60)
        self.assertGreaterEqual(len({r["domain"] for r in self.train}), 40)
        smallest = min(sum(1 for r in self.train if r["family"] == f) for f in families)
        self.assertGreaterEqual(smallest, 20, "nenhuma familia pode ficar com menos de 20 exemplos no train")

    def test_tasks_are_not_copies_of_the_eval(self) -> None:
        tasks = harness.load_tasks(ROOT / "benchmarks/harness/tasks")
        eval_code = {(Path(t["dir"]) / "files" / t["edit_file"]).read_text(encoding="utf-8") for t in tasks}
        eval_instructions = {t["instruction"] for t in tasks}
        unseen = json.loads((ROOT / "data/unseen_eval_120.json").read_text(encoding="utf-8"))
        unseen_code = {e["original_code"] for e in unseen}
        for row in self.train + self.val:
            code = row["context"].split("Codigo Atual:\n", 1)[1]
            self.assertNotIn(code, eval_code)
            self.assertNotIn(code, unseen_code)
            self.assertNotIn(row["instruction"], eval_instructions)


@unittest.skipUnless((V2 / "train.jsonl").is_file(), "data/v2 nao existe")
class DecontaminationTests(unittest.TestCase):
    """Nenhum exemplo do v2 compartilha um 13-grama com o eval (criterio de decontam_tasks.py)."""

    def test_committed_v2_has_zero_overlap_with_the_eval_tasks_and_unseen_eval(self) -> None:
        fresh = bd.decontamination(V2)  # decontam_tasks.build_report sobre data/v2 + o unseen_eval_120
        self.assertEqual(fresh["eval_tasks"]["checked"], 40)
        self.assertEqual(fresh["eval_tasks"]["with_overlap"], 0)
        self.assertEqual(fresh["eval_tasks"]["ids_with_overlap"], [])
        self.assertEqual(fresh["unseen_eval_120"]["checked"], 120)
        self.assertEqual(fresh["unseen_eval_120"]["with_overlap"], 0)
        manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["decontamination"], fresh)

    def test_the_detector_is_not_vacuous(self) -> None:
        """Um exemplo que copia uma tarefa do eval, ou uma entrada do unseen_eval_120, e apanhado."""
        task_dir = decontam_tasks.task_dirs(ROOT / "benchmarks/harness/tasks")[0]
        unseen = json.loads((ROOT / "data/unseen_eval_120.json").read_text(encoding="utf-8"))
        longest = max(range(len(unseen)), key=lambda i: len(decontam_tasks.tokens("\n".join(decontam_tasks.strings(unseen[i])))))
        with tempfile.TemporaryDirectory() as tmp:
            planted = Path(tmp) / "planted.jsonl"
            rows = [{"text": decontam_tasks.task_text(task_dir)},
                    {"text": "\n".join(decontam_tasks.strings(unseen[longest]))}]
            planted.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            result = bd.decontamination(Path(tmp))
        self.assertIn(task_dir.name, result["eval_tasks"]["ids_with_overlap"])
        self.assertIn(longest, result["unseen_eval_120"]["indexes_with_overlap"])  # e os irmaos da mesma tarefa

    def test_the_generator_index_flags_eval_text_and_not_clean_code(self) -> None:
        text = decontam_tasks.task_text(decontam_tasks.task_dirs(ROOT / "benchmarks/harness/tasks")[0])
        grams = bd.eval_grams()
        self.assertTrue(decontam_tasks.ngrams(text) & grams)
        clean = "def soma(a, b):\n    return a + b\n"
        self.assertFalse(decontam_tasks.ngrams(clean) & grams)


class CurrentDataUntouchedTests(unittest.TestCase):
    def test_current_datasets_and_unseen_eval_have_the_reference_hashes(self) -> None:
        for rel, digest in CURRENT_DATA.items():
            self.assertEqual(sha256_file(ROOT / rel), digest, f"{rel} foi alterado")

    def test_v2_lives_in_its_own_directory_and_writes_nothing_beside_the_current_data(self) -> None:
        for rel in CURRENT_DATA:
            self.assertTrue((ROOT / rel).is_file(), rel)
        for name in ("train.jsonl", "val.jsonl", "manifest.json"):
            self.assertFalse((ROOT / "data" / name).exists(), f"data/{name} nao deve existir: o v2 fica em data/v2")
        self.assertTrue(V2.is_dir())

    def test_generator_default_output_is_data_v2(self) -> None:
        self.assertEqual(bd.DEFAULT_OUT, V2)


class DeterminismTests(unittest.TestCase):
    """Sem random sem semente: a saida nao depende do ambiente."""

    def test_planning_does_not_depend_on_pythonhashseed(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import hashlib, build_dataset_v2 as bd\n"
            "h = hashlib.sha256()\n"
            "for c in bd.interleaved_candidates()[:400]:\n"
            "    case = bd.assemble(*c)\n"
            "    h.update(case.key.encode())\n"
            "print(h.hexdigest())\n" % str(ROOT / "scripts")
        )
        digests = set()
        for seed in ("1", "2", "random"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            digests.add(proc.stdout.strip())
        self.assertEqual(len(digests), 1, digests)
        h = hashlib.sha256()
        for c in bd.interleaved_candidates()[:400]:
            h.update(bd.assemble(*c).key.encode())
        self.assertEqual(digests, {h.hexdigest()})

    def test_no_module_uses_random_or_the_network(self) -> None:
        forbidden = {"random", "secrets", "socket", "urllib", "http", "requests", "httpx", "ssl", "ftplib", "smtplib"}
        for rel in ("scripts/build_dataset_v2.py", "scripts/dataset_v2_families.py"):
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertEqual(imported & forbidden, set(), rel)

    def test_ordering_and_split_are_stable_functions_of_their_inputs(self) -> None:
        self.assertEqual(bd.interleaved_candidates(), bd.interleaved_candidates())
        self.assertEqual(bd.split_of("off_by_one/page_slice/invoice"), bd.split_of("off_by_one/page_slice/invoice"))
        sample = {bd.split_of(f"g/{i}") for i in range(200)}
        self.assertEqual(sample, {"train", "val"})

    @unittest.skipUnless(importlib.util.find_spec("pytest"), "pytest is not installed")
    def test_two_runs_write_identical_files_whatever_the_worker_count(self) -> None:
        before = {rel: sha256_file(ROOT / rel) for rel in CURRENT_DATA}
        outputs = []
        for workers in ("1", "3"):
            with tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(quiet_main("--out", tmp, "--limit", "14", "--workers", workers), 0)
                outputs.append({p.name: p.read_bytes() for p in Path(tmp).iterdir()})
        self.assertEqual(set(outputs[0]), {"train.jsonl", "val.jsonl", "manifest.json"})
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual({rel: sha256_file(ROOT / rel) for rel in CURRENT_DATA}, before)

    @unittest.skipUnless(importlib.util.find_spec("pytest") and (V2 / "train.jsonl").is_file(), "pytest or data/v2 missing")
    def test_committed_prefix_is_reproduced_by_a_fresh_run(self) -> None:
        """O inicio do arquivo gerado agora (modo pequeno) existe, linha por linha, no data/v2 commitado
        (o limite 14 para antes do corte de 2000, entao os mesmos candidatos sao aceitos na mesma ordem)."""
        committed = set((V2 / "train.jsonl").read_text(encoding="utf-8").splitlines()) | set(
            (V2 / "val.jsonl").read_text(encoding="utf-8").splitlines())
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(quiet_main("--out", tmp, "--limit", "14", "--workers", "2"), 0)
            fresh = [line for name in ("train.jsonl", "val.jsonl")
                     for line in (Path(tmp) / name).read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(fresh), 14)
        self.assertEqual([line for line in fresh if line not in committed], [])


@needs_pytest
class CommittedDataRerunTests(unittest.TestCase):
    @unittest.skipUnless((V2 / "train.jsonl").is_file(), "data/v2 nao existe")
    def test_a_sample_of_the_committed_examples_still_fails_before_and_passes_after(self) -> None:
        rows = load("train.jsonl")[::180] + load("val.jsonl")[::60]
        self.assertGreaterEqual(len(rows), 10)
        for row in rows:
            with self.subTest(id=row["id"]):
                self.assertEqual(bd.verify_record(row), [])


if __name__ == "__main__":
    unittest.main()
