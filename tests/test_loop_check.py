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


class EmptyPhaseTests(unittest.TestCase):
    def test_empty_phase_fails(self) -> None:
        for phase in ("orient", "plan", "validate", "deliver"):
            for body in ("", "   \n\n  "):
                with self.subTest(phase=phase, body=repr(body)):
                    problems = loop_check.check(build(bodies={phase: body}))
                    self.assertIn(f"fase-vazia: <{phase}> (linha ", "\n".join(problems))

    def test_envelope_with_only_a_patch_block_fails(self) -> None:
        empty = {p: "" for p in ("orient", "plan", "validate", "deliver")}
        problems = loop_check.check(build(bodies=empty))
        self.assertEqual(len([p for p in problems if p.startswith("fase-vazia")]), 4)

    def test_ellipsis_counts_as_content(self) -> None:
        bodies = {p: "…" for p in ("orient", "plan", "validate", "deliver")}
        self.assertEqual(loop_check.check(build(bodies=bodies)), [])

    def test_patch_phase_without_content_reports_the_missing_block(self) -> None:
        self.assertIn("patch-sem-bloco", rules(loop_check.check(build(bodies={"patch": ""}))))


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

    def test_point_label_case_and_list_prefix_are_checked(self) -> None:
        labels = ("[ponto 45: X] x", "[PONTO 45: X] x", "[point 45: X] x", "[Point 45: X] x",
                  "- [Ponto 45: X] x", "* [ponto 45: X] x", "+ [Point 45: X] x", "• [Ponto 45: X] x",
                  "1. [Ponto 45: X] x", "12) [ponto 45: X] x", "  - [Ponto 45: X] x")
        for label in labels:
            with self.subTest(label=label):
                problems = loop_check.check(build(bodies={"plan": label}))
                self.assertIn("ponto-fora-da-fase", rules(problems))

    def test_point_label_case_and_list_prefix_inside_range_pass(self) -> None:
        for label in ("[ponto 11: X] x", "- [Ponto 11: X] x", "1. [point 11: X] x", "* [PONTO 20: X] x"):
            with self.subTest(label=label):
                self.assertEqual(loop_check.check(build(bodies={"plan": label})), [])

    def test_malformed_point_with_prefix_or_lowercase_fails(self) -> None:
        for bad in ("- [Ponto X: Y] x", "[ponto 1 Raiz] x", "1. [point 1:] x"):
            with self.subTest(bad=bad):
                self.assertIn("ponto-malformado", rules(loop_check.check(build(bodies={"orient": bad}))))

    def test_prefixed_points_count_in_complete_mode(self) -> None:
        text = full_envelope().replace("[Ponto 7: Nome 7]", "- [ponto 7: Nome 7]")
        self.assertEqual(loop_check.check(text, complete=True), [])

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
        problems = self.patch_problems(body)
        self.assertIn("patch-sem-bloco", rules(problems))
        self.assertIn("patch-marcador", rules(problems))

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



class PatchMarkerHardeningTests(unittest.TestCase):
    """Marcadores exatos: 4 caracteres, sozinhos na linha, sem indentação, em qualquer ponto de <patch>."""

    def patch_problems(self, body: str) -> list[str]:
        return loop_check.check(build(bodies={"patch": body}))

    def test_wrong_length_replace_closing_an_open_block_fails(self) -> None:
        for n in (5, 6, 7):
            with self.subTest(length=n):
                body = f"<<<< SEARCH\nx = 1\n====\nx = 2\n{'>' * n} REPLACE"
                self.assertIn("patch-marcador", rules(self.patch_problems(body)))

    def test_seven_character_replace_cannot_hide_behind_a_later_valid_block(self) -> None:
        body = (
            "<<<< SEARCH\nx = 1\n====\nx = 2\n>>>>>>> REPLACE\n"
            "<<<< SEARCH\ny = 1\n====\ny = 2\n>>>> REPLACE"
        )
        self.assertIn("patch-marcador", rules(self.patch_problems(body)))

    def test_wrong_length_divider_inside_a_search_fails(self) -> None:
        for n in (5, 6, 7):
            with self.subTest(length=n):
                body = f"<<<< SEARCH\nx = 1\n{'=' * n}\nx = 2\n>>>> REPLACE"
                self.assertIn("patch-marcador", rules(self.patch_problems(body)))

    def test_wrong_length_divider_inside_a_replace_fails(self) -> None:
        body = "<<<< SEARCH\nx = 1\n====\nx = 2\n=======\nx = 3\n>>>> REPLACE"
        self.assertIn("patch-marcador", rules(self.patch_problems(body)))

    def test_wrong_length_search_inside_an_open_block_fails(self) -> None:
        for n in (5, 7):
            with self.subTest(length=n):
                body = f"<<<< SEARCH\nx = 1\n{'<' * n} SEARCH\n====\nx = 2\n>>>> REPLACE"
                self.assertIn("patch-marcador", rules(self.patch_problems(body)))

    def test_wrong_length_markers_outside_a_block_fail(self) -> None:
        for n in (5, 6, 7):
            with self.subTest(length=n):
                body = BLOCK + f"\n{'<' * n} SEARCH\ny = 1\n{'=' * n}\ny = 2\n{'>' * n} REPLACE"
                self.assertIn("patch-marcador", rules(self.patch_problems(body)))

    def test_indented_search_or_replace_inside_an_open_block_fails(self) -> None:
        inside_search = "<<<< SEARCH\nx = 1\n    >>>> REPLACE\n====\nx = 2\n>>>> REPLACE"
        inside_replace = "<<<< SEARCH\nx = 1\n====\nx = 2\n  <<<< SEARCH\n>>>> REPLACE"
        for body in (inside_search, inside_replace):
            with self.subTest(body=body):
                self.assertIn("patch-marcador", rules(self.patch_problems(body)))

    def test_indented_block_after_a_valid_one_fails(self) -> None:
        body = BLOCK + "\n  <<<< SEARCH\ny = 1\n  ====\ny = 2\n  >>>> REPLACE"
        self.assertIn("patch-marcador", rules(self.patch_problems(body)))

    def test_tab_indented_marker_fails(self) -> None:
        body = BLOCK + "\n\t<<<< SEARCH\ny = 1\n====\ny = 2\n>>>> REPLACE"
        self.assertIn("patch-marcador", rules(self.patch_problems(body)))

    def test_marker_with_extra_or_missing_space_fails(self) -> None:
        for search, replace in (("<<<<  SEARCH", ">>>> REPLACE"), ("<<<<SEARCH", ">>>> REPLACE"),
                                ("<<<< SEARCH", ">>>>  REPLACE"), ("<<<< SEARCH", ">>>>REPLACE")):
            with self.subTest(search=search, replace=replace):
                body = f"{search}\nx = 1\n====\nx = 2\n{replace}"
                self.assertIn("patch-marcador", rules(self.patch_problems(body)))

    def test_trailing_whitespace_on_markers_is_allowed(self) -> None:
        body = "<<<< SEARCH  \nx = 1\n====\t\nx = 2\n>>>> REPLACE \n"
        self.assertEqual(self.patch_problems(body), [])

    def test_indented_divider_inside_a_block_is_content(self) -> None:
        # Limitação documentada: pode ser o sublinhado de um título RST/Markdown no código editado.
        body = "<<<< SEARCH\n    ====\n====\n    ====\n>>>> REPLACE"
        self.assertEqual(self.patch_problems(body), [])

    def test_whitespace_only_search_is_empty(self) -> None:
        for search in ("\n", "   \n", "\n  \n\t\n"):
            with self.subTest(search=repr(search)):
                body = f"<<<< SEARCH\n{search}====\nx = 2\n>>>> REPLACE"
                self.assertIn("patch-search-vazio", rules(self.patch_problems(body)))

    def test_search_marker_outside_patch_fails(self) -> None:
        for phase in ("orient", "plan", "validate", "deliver"):
            with self.subTest(phase=phase):
                text = build(bodies={phase: BODIES[phase] + "\n<<<< SEARCH\nx\n====\ny\n>>>> REPLACE"})
                self.assertIn("patch-fora-do-patch", rules(loop_check.check(text)))

    def test_search_marker_between_phases_fails(self) -> None:
        text = build().replace("</orient>\n", "</orient>\n<<<< SEARCH\n")
        self.assertIn("patch-fora-do-patch", rules(loop_check.check(text)))

    def test_markers_inside_think_do_not_count_as_a_patch(self) -> None:
        in_patch = loop_check.check(build(bodies={"patch": "<think>\n" + BLOCK + "\n</think>"}))
        self.assertIn("patch-no-think", rules(in_patch))
        self.assertIn("patch-sem-bloco", rules(in_patch))
        in_plan = loop_check.check(build(bodies={"plan": "<think>\n<<<< SEARCH\n</think>\n[Ponto 11: P] x"}))
        self.assertIn("patch-no-think", rules(in_plan))

    def test_block_after_closed_think_still_counts(self) -> None:
        body = "<think>\npensando\n</think>\n" + BLOCK
        self.assertEqual(self.patch_problems(body), [])

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

    def test_jsonl_empty_file_exits_1_with_a_clear_message(self) -> None:
        for name, content in (("vazio.jsonl", ""), ("brancos.jsonl", "\n   \n\t\n\n")):
            with self.subTest(name=name):
                done = run_cli("--jsonl", self.write(name, content))
                self.assertEqual(done.returncode, 1, done.stdout)
                self.assertIn("nenhuma linha para validar", done.stdout)
                self.assertNotIn("0 falham", done.stdout)

    def test_jsonl_empty_file_makes_a_multi_file_run_fail(self) -> None:
        good = self.jsonl(build())
        empty = self.write("vazio.jsonl", "")
        done = run_cli("--jsonl", good, "--jsonl", empty)
        self.assertEqual(done.returncode, 1)
        self.assertIn("1 linhas, 1 passam, 0 falham", done.stdout)

    def test_empty_text_file_exits_1(self) -> None:
        for content in ("", "  \n\n"):
            with self.subTest(content=repr(content)):
                done = run_cli(self.write("vazio.txt", content))
                self.assertEqual(done.returncode, 1)
                self.assertIn("entrada: saída vazia", done.stdout)

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
