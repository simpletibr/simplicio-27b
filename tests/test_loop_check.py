"""scripts/loop_check.py: validador do envelope do simplicio-loop (50 pontos).

Sem rede, sem modelo, sem GPU. Os casos de falha partem de um envelope mínimo válido e
estragam uma coisa por vez; a CLI roda em subprocesso para checar o código de saída.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "loop_check.py"
sys.path.insert(0, str(ROOT / "scripts"))

import loop_check  # noqa: E402

PHASES = loop_check.PHASES
BLOCK = "<<<< SEARCH\nx = 1\n====\nx = 2\n>>>> REPLACE"
BODIES = {
    "orient": "[Ponto 1: Raiz] raiz confirmada",
    "plan": "[Ponto 11: Passos] passo 1",
    "patch": "[Ponto 21: Diff Cirurgico]\n" + BLOCK,
    "validate": "[Ponto 31: Validacao Estatica] ok",
    "deliver": "[Ponto 41: Poda de Tokens] ok",
}


def build(order=PHASES, bodies=None, wrap=True) -> str:
    merged = {**BODIES, **(bodies or {})}
    inner = "\n".join(f"<{p}>\n{merged[p]}\n</{p}>" for p in order)
    return f"<simplicio_loop>\n{inner}\n</simplicio_loop>\n" if wrap else inner + "\n"


def full_envelope() -> str:
    """Envelope com os 50 pontos, um por linha, e o bloco do patch entre o 21 e o 22."""
    bodies = {}
    for phase, (low, high) in loop_check.POINT_RANGES.items():
        lines = [f"[Ponto {n}: Nome {n}] texto" for n in range(low, high + 1)]
        if phase == "patch":
            lines.insert(1, BLOCK)
        bodies[phase] = "\n".join(lines)
    return build(bodies=bodies)


def rules(problems: list[str]) -> set[str]:
    return {loop_check.rule_of(p) for p in problems}


def run_cli(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(CHECK), *args],
        input=stdin, capture_output=True, text=True, encoding="utf-8", env=env, cwd=ROOT, timeout=60,
    )


class ValidEnvelopeTests(unittest.TestCase):
    def test_minimal_envelope_passes(self) -> None:
        self.assertEqual(loop_check.check(build()), [])

    def test_phases_without_numbered_points_pass(self) -> None:
        bodies = {p: "…" for p in PHASES if p != "patch"}
        self.assertEqual(loop_check.check(build(bodies=bodies)), [])

    def test_point_label_in_english_passes(self) -> None:
        text = build().replace("[Ponto", "[Point")
        self.assertEqual(loop_check.check(text), [])

    def test_point_names_are_not_checked(self) -> None:
        text = build(bodies={"orient": "[Ponto 1: Root]\n[Ponto 2: Identificacao de Raiz] x"})
        self.assertEqual(loop_check.check(text), [])

    def test_range_boundaries_are_accepted(self) -> None:
        for phase, (low, high) in loop_check.POINT_RANGES.items():
            body = f"[Ponto {low}: A] x\n[Ponto {high}: B] y"
            if phase == "patch":
                body = f"[Ponto {low}: A] x\n{BLOCK}\n[Ponto {high}: B] y"
            with self.subTest(phase=phase):
                self.assertEqual(loop_check.check(build(bodies={phase: body})), [])

    def test_several_blocks_pass(self) -> None:
        body = BODIES["patch"] + "\n" + BLOCK.replace("x = 1", "y = 1")
        self.assertEqual(loop_check.check(build(bodies={"patch": body})), [])

    def test_empty_replace_is_allowed(self) -> None:
        body = "<<<< SEARCH\nlinha velha\n====\n>>>> REPLACE"
        self.assertEqual(loop_check.check(build(bodies={"patch": body})), [])

    def test_marker_text_inside_block_is_content(self) -> None:
        body = "<<<< SEARCH\n</patch>\n[Ponto 99: Dentro do codigo]\n====\nnovo\n>>>> REPLACE"
        self.assertEqual(loop_check.check(build(bodies={"patch": body})), [])

    def test_readme_example_follows_its_own_envelope(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        found = re.search(r"## Output format.*?```text\n(.*?)```", readme, re.S)
        self.assertIsNotNone(found, "bloco de exemplo do README não encontrado")
        self.assertEqual(loop_check.check(found.group(1)), [])


class StructureTests(unittest.TestCase):
    def test_each_missing_phase_fails(self) -> None:
        for phase in PHASES:
            order = tuple(p for p in PHASES if p != phase)
            with self.subTest(missing=phase):
                problems = loop_check.check(build(order=order))
                self.assertIn(f"fase-ausente: falta <{phase}>", problems)

    def test_out_of_order_phases_fail(self) -> None:
        order = ("orient", "patch", "plan", "validate", "deliver")
        problems = loop_check.check(build(order=order))
        self.assertIn("fase-fora-de-ordem", rules(problems))

    def test_reversed_phases_fail(self) -> None:
        self.assertIn("fase-fora-de-ordem", rules(loop_check.check(build(order=PHASES[::-1]))))

    def test_repeated_phase_fails(self) -> None:
        text = build().replace("</deliver>", "</deliver>\n<deliver>\n[Ponto 42: X] y\n</deliver>")
        self.assertIn("fase-repetida", rules(loop_check.check(text)))

    def test_phase_not_closed_fails(self) -> None:
        text = build().replace("</plan>\n", "")
        self.assertIn("fase-nao-fechada", rules(loop_check.check(text)))

    def test_missing_envelope_tags_fail(self) -> None:
        self.assertIn("envelope", rules(loop_check.check(build(wrap=False))))
        self.assertIn("envelope", rules(loop_check.check(build().replace("</simplicio_loop>", ""))))

    def test_text_outside_envelope_fails(self) -> None:
        self.assertIn("fora-do-envelope", rules(loop_check.check("Claro, aqui vai:\n" + build())))
        self.assertIn("fora-do-envelope", rules(loop_check.check(build() + "Pronto.\n")))
        self.assertIn("fora-do-envelope", rules(loop_check.check("<think>\n\n</think>\n" + build())))

    def test_text_between_phases_fails(self) -> None:
        text = build().replace("</orient>\n", "</orient>\nconversa solta\n")
        self.assertIn("texto-fora-das-fases", rules(loop_check.check(text)))

    def test_empty_and_non_text_input_fail(self) -> None:
        self.assertEqual(rules(loop_check.check("")), {"entrada"})
        self.assertEqual(rules(loop_check.check("   \n")), {"entrada"})
        self.assertEqual(rules(loop_check.check(None)), {"entrada"})
        self.assertEqual(rules(loop_check.check(["<simplicio_loop>"])), {"entrada"})


class PointTests(unittest.TestCase):
    def test_point_outside_its_phase_fails(self) -> None:
        cases = {
            "orient": "[Ponto 11: Passos] x",
            "plan": "[Ponto 5: Custo] x",
            "validate": "[Ponto 41: Poda] x",
            "deliver": "[Ponto 50: Selo] x\n[Ponto 31: Estatica] y",
        }
        for phase, body in cases.items():
            with self.subTest(phase=phase):
                problems = loop_check.check(build(bodies={phase: body}))
                self.assertIn("ponto-fora-da-fase", rules(problems))

    def test_point_in_patch_outside_range_fails_even_after_block(self) -> None:
        body = BODIES["patch"] + "\n[Ponto 31: Estatica] x"
        self.assertIn("ponto-fora-da-fase", rules(loop_check.check(build(bodies={"patch": body}))))

    def test_point_between_phases_fails(self) -> None:
        text = build().replace("</orient>\n", "</orient>\n[Ponto 11: Passos] x\n")
        self.assertIn("ponto-sem-fase", rules(loop_check.check(text)))

    def test_malformed_point_fails(self) -> None:
        for bad in ("[Ponto 1 Raiz] x", "[Ponto X: Raiz] x", "[Ponto 1:] x", "[Ponto 1: Raiz x"):
            with self.subTest(bad=bad):
                problems = loop_check.check(build(bodies={"orient": bad}))
                self.assertIn("ponto-malformado", rules(problems))

    def test_default_mode_does_not_demand_all_50_points(self) -> None:
        self.assertEqual(loop_check.check(build()), [])
        self.assertEqual(loop_check.check(build(), complete=False), [])


class PatchTests(unittest.TestCase):
    def patch_problems(self, body: str) -> list[str]:
        return loop_check.check(build(bodies={"patch": body}))

    def test_patch_without_blocks_fails(self) -> None:
        self.assertIn("patch-sem-bloco", rules(self.patch_problems("[Ponto 21: Diff] sem bloco")))
        self.assertIn("patch-sem-bloco", rules(self.patch_problems("")))

    def test_seven_character_markers_fail(self) -> None:
        body = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"
        problems = self.patch_problems(body)
        self.assertIn("patch-marcador", rules(problems))
        self.assertIn("patch-sem-bloco", rules(problems))

    def test_three_character_markers_fail(self) -> None:
        body = "<<< SEARCH\nx = 1\n====\nx = 2\n>>> REPLACE"
        self.assertIn("patch-marcador", rules(self.patch_problems(body)))

    def test_indented_markers_are_not_blocks(self) -> None:
        body = "  <<<< SEARCH\nx = 1\n  ====\nx = 2\n  >>>> REPLACE"
        self.assertIn("patch-sem-bloco", rules(self.patch_problems(body)))

    def test_block_without_replace_fails(self) -> None:
        self.assertIn("patch-bloco", rules(self.patch_problems("<<<< SEARCH\nx = 1\n====\nx = 2")))

    def test_block_without_divider_fails(self) -> None:
        body = "<<<< SEARCH\nx = 1\n>>>> REPLACE"
        self.assertIn("patch-bloco", rules(self.patch_problems(body)))

    def test_empty_search_fails(self) -> None:
        body = "<<<< SEARCH\n====\nx = 2\n>>>> REPLACE"
        self.assertIn("patch-search-vazio", rules(self.patch_problems(body)))

    def test_divider_or_replace_outside_block_fails(self) -> None:
        self.assertIn("patch-bloco", rules(self.patch_problems(BLOCK + "\n====")))
        self.assertIn("patch-bloco", rules(self.patch_problems(BLOCK + "\n>>>> REPLACE")))


class CompleteModeTests(unittest.TestCase):
    def test_full_envelope_passes_both_modes(self) -> None:
        text = full_envelope()
        self.assertEqual(loop_check.check(text), [])
        self.assertEqual(loop_check.check(text, complete=True), [])

    def test_minimal_envelope_fails_complete_mode(self) -> None:
        problems = loop_check.check(build(), complete=True)
        self.assertIn("ponto-ausente", rules(problems))
        self.assertEqual(loop_check.check(build()), [])

    def test_missing_point_is_named(self) -> None:
        text = full_envelope().replace("[Ponto 7: Nome 7] texto\n", "")
        problems = loop_check.check(text, complete=True)
        self.assertEqual(problems, ["ponto-ausente: <orient> sem os pontos 7"])

    def test_repeated_point_fails(self) -> None:
        text = full_envelope().replace("[Ponto 34: Nome 34]", "[Ponto 33: Nome 33]")
        self.assertIn("ponto-repetido", rules(loop_check.check(text, complete=True)))

    def test_points_out_of_order_fail(self) -> None:
        text = full_envelope().replace("[Ponto 2: Nome 2] texto\n[Ponto 3: Nome 3] texto\n",
                                       "[Ponto 3: Nome 3] texto\n[Ponto 2: Nome 2] texto\n")
        self.assertEqual(rules(loop_check.check(text, complete=True)), {"ponto-fora-de-ordem"})


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, name: str, text: str) -> str:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_valid_file_exits_0(self) -> None:
        done = run_cli(self.write("ok.txt", build()))
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        self.assertTrue(done.stdout.startswith("OK"), done.stdout)

    def test_invalid_file_exits_1_and_names_the_problem(self) -> None:
        done = run_cli(self.write("ruim.txt", build(order=("orient", "plan", "patch", "validate"))))
        self.assertEqual(done.returncode, 1)
        self.assertIn("FALHA", done.stdout)
        self.assertIn("fase-ausente: falta <deliver>", done.stdout)

    def test_any_failing_file_makes_the_run_fail(self) -> None:
        good = self.write("ok.txt", build())
        bad = self.write("ruim.txt", build(bodies={"patch": "sem bloco"}))
        done = run_cli(good, bad)
        self.assertEqual(done.returncode, 1)
        self.assertIn(f"OK     {good}", done.stdout)
        self.assertIn(f"FALHA  {bad}", done.stdout)

    def test_unreadable_file_exits_1(self) -> None:
        done = run_cli(str(self.dir / "nao-existe.txt"))
        self.assertEqual(done.returncode, 1)
        self.assertIn("FALHA", done.stdout)

    def test_no_arguments_is_a_usage_error(self) -> None:
        done = run_cli()
        self.assertEqual(done.returncode, 2)
        self.assertIn("ARQUIVO", done.stderr)

    def test_stdin_dash(self) -> None:
        self.assertEqual(run_cli("-", stdin=build()).returncode, 0)
        self.assertEqual(run_cli("-", stdin="so texto").returncode, 1)

    def test_complete_flag(self) -> None:
        minimal = self.write("min.txt", build())
        full = self.write("full.txt", full_envelope())
        self.assertEqual(run_cli(minimal).returncode, 0)
        self.assertEqual(run_cli("--complete", minimal).returncode, 1)
        self.assertEqual(run_cli("--complete", full).returncode, 0)

    def jsonl(self, *outputs: str, field: str = "simplicio_trajectory", raw: tuple[str, ...] = ()) -> str:
        lines = [json.dumps({field: o}) for o in outputs] + list(raw)
        return self.write("d.jsonl", "\n".join(lines) + "\n")

    def test_jsonl_all_valid_exits_0_and_counts(self) -> None:
        done = run_cli("--jsonl", self.jsonl(build(), full_envelope()))
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertIn("2 linhas, 2 passam, 0 falham", done.stdout)

    def test_jsonl_counts_passes_failures_and_reasons(self) -> None:
        bad_plan = build(order=("orient", "patch", "plan", "validate", "deliver"))
        bad_patch = build(bodies={"patch": "sem bloco"})
        path = self.jsonl(build(), bad_plan, bad_patch, bad_plan, raw=("{não é json",))
        done = run_cli("--jsonl", path)
        self.assertEqual(done.returncode, 1)
        self.assertIn("5 linhas, 1 passam, 4 falham", done.stdout)
        self.assertIn("fase-fora-de-ordem: 2 de 5", done.stdout)
        self.assertIn("patch-sem-bloco: 1 de 5", done.stdout)
        self.assertIn("jsonl: 1 de 5", done.stdout)

    def test_jsonl_missing_field_and_custom_field(self) -> None:
        path = self.jsonl(build(), field="saida")
        missing = run_cli("--jsonl", path)
        self.assertEqual(missing.returncode, 1)
        self.assertIn("campo: 1 de 1", missing.stdout)
        custom = run_cli("--jsonl", path, "--field", "saida")
        self.assertEqual(custom.returncode, 0, custom.stdout)

    def test_jsonl_skips_blank_lines(self) -> None:
        path = self.write("d.jsonl", json.dumps({"simplicio_trajectory": build()}) + "\n\n")
        done = run_cli("--jsonl", path)
        self.assertIn("1 linhas, 1 passam, 0 falham", done.stdout)


class DatasetTests(unittest.TestCase):
    """Os arquivos de data/ (gerados por generate_dataset.py) seguem o envelope e os 50 pontos."""

    def test_dataset_examples_follow_the_loop(self) -> None:
        for name in ("simplicio_loop_50pts_train.jsonl", "simplicio_loop_50pts_val.jsonl"):
            lines = (ROOT / "data" / name).read_text(encoding="utf-8").splitlines()
            self.assertTrue(lines, name)
            for no, line in enumerate(lines, 1):
                with self.subTest(file=name, line=no):
                    text = json.loads(line)[loop_check.DEFAULT_FIELD]
                    self.assertEqual(loop_check.check(text, complete=True), [])


if __name__ == "__main__":
    unittest.main()
