"""decide/RUNBOOK.md against the code on this checkout (issue #72, F7).

The RUNBOOK tells the owner which commands to run.  A command that names a module that is not in the tree, or an option the
module does not have, wastes a session (a rented GPU day, in the worst case).  These tests read every ``python -m decide...``
command and every repository path the RUNBOOK names and check them against the checkout:

* the module exists, and each ``--option`` of the command is in the ``--help`` of that module (and of its subcommand);
* the path exists;
* what is NOT on this checkout yet (``NOT_ON_MAIN``) is the only thing allowed to be missing, and the section "Avisos de hoje"
  of the RUNBOOK names it.  When the pull request that brings it lands, the entry here stops being true, the test fails, and the
  RUNBOOK has to be revised (the warnings taken out, the commands checked) in the same change.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "decide" / "RUNBOOK.md"

#: Paths (by prefix, from the repository root) that the RUNBOOK names and that the main branch does not have yet:
#: (who brings them, the name the section "Avisos de hoje" must use for them).
NOT_ON_MAIN: Dict[str, Tuple[str, str]] = {
    "decide/modal": ("Squad A, part 2 of the kit", "decide/modal"),
    "notebooks/decide/docs": ("Squad A, part 2 of the kit", "notebooks/decide/docs"),
}
WARNINGS = "Avisos de hoje"
#: Phrases of the warning the owner must see before any session: the official GT verdict comes from the Kaggle notebook only, until the hosted gate has
#: operational evidence.  (The warning about the core of the kit was taken out when its three blockers were fixed on main, #86; the one about the
#: false GT FAIL stays, reworded, because its limits were chosen and never measured on a real provider.)
WARNING_PHRASES = ("think_gt_kaggle.ipynb", "backend `transformers`")
#: What only the retired warning about the kit said.  Text of a warning that was taken out must not stay anywhere in the RUNBOOK.
RETIRED_PHRASES = ("Não rode sessões de GPU do kit", "bloqueantes abertos", "bloqueantes do kit", "sem GPU até as correções",
                   "as correções dele ainda não entraram", "O G2 espera isso", "PR de correção (avisos 1 e 2)",
                   # part 2: the fold is on main since #95 (opt-in, no command-line flag), and the 10% rule counts two kinds of row (decide/think/compare.py)
                   "o motor do kit ainda não chama", "o motor do kit chamar", "faltam duas coisas", "de 10% das que responderam",
                   "Esta seção ainda não foi revista")

HelpOf = Callable[[str, Tuple[str, ...]], str]


def section(text: str, title: str) -> str:
    """The text from the heading that contains ``title`` to the next ``## `` heading."""
    match = re.search(rf"^## [^\n]*{re.escape(title)}[^\n]*\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    return match.group(1) if match else ""


def missing_from_main(name: str, root: Path, not_on_main: Dict[str, Tuple[str, str]] = NOT_ON_MAIN) -> Tuple[bool, str]:
    """(is it in ``not_on_main``, which entry) for a path or a module path that is not in ``root``."""
    for prefix in not_on_main:
        if name == prefix or name.startswith(prefix + "/"):
            return True, prefix
    return False, ""


def module_exists(module: str, root: Path) -> bool:
    base = root.joinpath(*module.split("."))
    return base.with_suffix(".py").is_file() or (base / "__main__.py").is_file()


def present(path: str, root: Path) -> bool:
    """A file, or a folder that holds a file other than interpreter leftovers (an empty ``__pycache__`` is not code)."""
    target = root / path.rstrip("/")
    if target.is_file():
        return True
    return target.is_dir() and any(f.is_file() and "__pycache__" not in f.parts for f in target.rglob("*"))


def path_exists(path: str, root: Path) -> bool:
    return present(path, root) or present("decide/" + path, root)         # links in the RUNBOOK are relative to decide/ or to the root


def runbook_problems(text: str, root: Path, help_of: HelpOf, not_on_main: Dict[str, Tuple[str, str]] = NOT_ON_MAIN) -> List[str]:
    problems: List[str] = []
    warnings = section(text, WARNINGS)
    if not warnings:
        problems.append(f"the RUNBOOK has no section '{WARNINGS}'")
    for phrase in WARNING_PHRASES:
        if phrase not in warnings:
            problems.append(f"the section '{WARNINGS}' does not say {phrase!r}")
    for phrase in RETIRED_PHRASES:
        if phrase in text:
            problems.append(f"{phrase!r} is text of a warning that was taken out and is still in the RUNBOOK")
    listed = len(re.findall(r"^\d+\. \*\*", warnings, flags=re.M))
    for match in re.finditer(r"\bavisos? (\d+)(?: e (\d+))?", text, flags=re.I):
        for number in filter(None, match.groups()):
            if listed and int(number) > listed:
                problems.append(f"{match.group(0)!r} points at a warning that does not exist: the section '{WARNINGS}' lists {listed}")
    named: Dict[str, bool] = {}                                              # NOT_ON_MAIN entries the RUNBOOK uses
    joined = re.sub(r"\\\n\s*", " ", text)
    seen: set = set()

    for match in re.finditer(r"-m (decide[\w.]*)(?:::\w+)?([^`\n]*)", joined):
        module, args = match.group(1), match.group(2)
        is_python = joined[max(0, match.start() - 7):match.start()] == "python "
        if not module_exists(module, root):
            absent, prefix = missing_from_main(module.replace(".", "/"), root, not_on_main)
            if absent:
                named[prefix] = True
            else:
                problems.append(f"{module}: the module is not in the tree and is not declared in NOT_ON_MAIN")
            continue
        if not is_python:
            continue
        tokens = args.split()
        sub: Tuple[str, ...] = ()
        if tokens and not tokens[0].startswith("-"):
            choices = re.search(r"\{([\w,-]+)\}", help_of(module, ()))
            if choices and tokens[0] in choices.group(1).split(","):
                sub = (tokens[0],)
        key = (module, sub, tuple(re.findall(r"--[A-Za-z][A-Za-z0-9-]*", args)))
        if key in seen:
            continue
        seen.add(key)
        help_text = help_of(module, sub)
        for flag in key[2]:
            if not re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text):
                problems.append(f"{' '.join((module,) + sub)}: {flag} is not an option of the module (see its --help)")

    for token in sorted(set(re.findall(r"`([^`\s]+)`", text))):
        if not re.match(r"^(?:decide|notebooks|docs|tests|scripts|data)/[\w./*-]+$", token) or "*" in token:
            continue
        if not path_exists(token, root):
            absent, prefix = missing_from_main(token.rstrip("/"), root, not_on_main)
            if absent:
                named[prefix] = True
            else:
                problems.append(f"{token}: the path is not in the tree and is not declared in NOT_ON_MAIN")

    for prefix, (who, label) in not_on_main.items():
        if present(prefix, root) or module_exists(prefix.replace("/", "."), root):
            problems.append(f"{prefix} is now on this checkout ({who}): take it out of NOT_ON_MAIN and revise the RUNBOOK (warnings, commands)")
        elif named.get(prefix) and label not in warnings:
            problems.append(f"the RUNBOOK uses {prefix} ({who}), which is not on main, but the section '{WARNINGS}' does not name {label}")
    return problems


def real_help(module: str, sub: Tuple[str, ...]) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "GIT_COMMON_DIR")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    done = subprocess.run([sys.executable, "-m", module, *sub, "--help"], cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
    if done.returncode != 0:
        if "ModuleNotFoundError" in done.stderr and "No module named 'decide" not in done.stderr:
            raise unittest.SkipTest(done.stderr.strip().splitlines()[-1])
        raise AssertionError(f"python -m {module} {' '.join(sub)} --help exited {done.returncode}: {done.stderr.strip()[-300:]}")
    return done.stdout


class RunbookAgainstTheTreeTest(unittest.TestCase):
    def test_every_command_and_path_of_the_runbook_exists_or_is_declared_missing(self) -> None:
        cache: Dict[Tuple[str, Tuple[str, ...]], str] = {}

        def help_of(module: str, sub: Tuple[str, ...]) -> str:
            if (module, sub) not in cache:
                cache[(module, sub)] = real_help(module, sub)
            return cache[(module, sub)]

        self.assertEqual(runbook_problems(RUNBOOK.read_text(encoding="utf-8"), ROOT, help_of), [])


class RunbookFactsTest(unittest.TestCase):
    """What the RUNBOOK says about the thinker gate and the fold of the reasoning (part 2), against the constants it quotes.

    The numbers and names below are read from the code, so a change there fails this test and the RUNBOOK has to be revised in the same change."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = RUNBOOK.read_text(encoding="utf-8")

    def test_the_unreadable_share_is_stated_as_the_gate_applies_it(self) -> None:
        """INCONCLUSIVE when ``reasoning_hidden + no_close`` is over the share of the rows that answered (``decide/think/compare.py``): both kinds of row, not only the hidden ones."""
        from decide.think import common, compare

        self.assertIn(f"{compare.MAX_UNREADABLE_SHARE:.0%} das linhas que responderam", self.text)
        self.assertIn(f"{common.MIN_THINKING_TOKENS} tokens", self.text)
        self.assertIn("linhas que esconderam o raciocínio ou pararam sem fechar o pensamento", self.text)
        self.assertIn("reasoning_hidden + no_close", self.text)

    def test_the_session_of_the_day_says_which_gt_report_turns_the_traces_on(self) -> None:
        from decide.rental import gates
        from decide.think import compare

        self.assertEqual(gates.GT_REPORT_FORMAT, compare.REPORT_FORMAT, "the day reads the format the gate writes")
        day = section(self.text, "Sessão 8")
        for phrase in (compare.REPORT_FORMAT, "--gt-report", "`PASS`", "EXPLORATORY", "INCONCLUSIVE"):
            self.assertIn(phrase, day, f"the session of the rented day does not say {phrase!r}")

    def test_the_fold_is_described_as_the_kit_has_it(self) -> None:
        import inspect

        from decide.sd import evaluate
        from decide.train import protocol

        self.assertEqual(inspect.signature(evaluate.run_requests).parameters["reasoning"].default, "refuse")
        for mode in evaluate.REASONING_MODES:
            self.assertIn(f'"{mode}"', self.text)
        self.assertIn('reasoning="fold"', self.text)
        self.assertIn(f"{protocol.REASONING_MAX_CHARS} caracteres", self.text)
        self.assertIn(protocol.REASONING_LABEL, self.text)
        self.assertIn("decide.sd.evaluate.run_requests", self.text)
        self.assertIn("decide/train/protocol.py", self.text, "the fold imports the training protocol: the RUNBOOK says the whole repository is needed")
        self.assertIn("tests/fold_reasoning_cases.json", self.text)

    def test_there_is_no_command_line_flag_for_the_fold_and_the_runbook_does_not_invent_one(self) -> None:
        flags = [(path, match) for path in sorted((ROOT / "decide").rglob("*.py"))
                 for match in re.findall(r"""["']--[\w-]*fold[\w-]*["']""", path.read_text(encoding="utf-8"))]
        self.assertEqual(flags, [], "a command-line flag for the fold now exists: the RUNBOOK says there is none")
        self.assertEqual(re.findall(r"--[\w-]*fold[\w-]*", self.text), [])
        self.assertIn("Não existe flag de linha de comando para o fold", self.text)

    def test_the_budget_limit_of_the_fold_points_at_its_issue(self) -> None:
        self.assertRegex(self.text, r"issues?/97|#97")


class CheckerNoticesTest(unittest.TestCase):
    """The checker itself: a RUNBOOK that lies is caught (so the test above is not green by construction)."""

    #: what the checker is given as "not on main yet": fixed here, so these tests do not change when the real NOT_ON_MAIN does
    DECLARED = {"decide/later": ("someone", "decide/later"), "notebooks/decide/docs": ("someone", "notebooks/decide/docs"),
                "decide/later.yaml": ("someone", "decide/later.yaml")}
    WARN = ("## Avisos de hoje (leia)\n\n1. **Kaggle** " + " ".join(WARNING_PHRASES) + "\n2. **Tabela** decide/later decide/later.yaml "
            "notebooks/decide/docs\n\n## 0. Como ler\n")

    def tree(self, files: Sequence[str]) -> Path:
        root = Path(tempfile.mkdtemp(prefix="runbook-tree-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        for name in files:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text("", encoding="utf-8")
        return root

    @staticmethod
    def helps(module: str, sub: Tuple[str, ...]) -> str:
        return "usage: x {build,score} ...\n  --inputs INPUTS\n  --out OUT\n  --use-this\n"

    def check(self, body: str, files: Sequence[str] = ("decide/train/soup.py",), warn: str = None) -> List[str]:
        return runbook_problems((self.WARN if warn is None else warn) + body, self.tree(files), self.helps, self.DECLARED)

    def test_a_good_runbook_has_no_problems(self) -> None:
        self.assertEqual(self.check("```\npython -m decide.train.soup --inputs a b \\\n    --out c\n```\n"), [])

    def test_a_module_that_is_not_in_the_tree_is_a_problem(self) -> None:
        self.assertTrue(any("decide.train.nope" in p for p in self.check("`python -m decide.train.nope --out c`\n")))

    def test_an_option_the_module_does_not_have_is_a_problem(self) -> None:
        problems = self.check("`python -m decide.train.soup --inputs a --out c --nope`\n")
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("--nope", problems[0])

    def test_an_option_on_a_continued_line_is_checked_too(self) -> None:
        self.assertTrue(any("--nope" in p for p in self.check("```\npython -m decide.train.soup --inputs a \\\n    --nope\n```\n")))

    def test_a_missing_module_is_allowed_only_if_declared_and_named_in_the_warnings(self) -> None:
        body = "`python -m decide.later.run --profile x`\n"
        self.assertEqual(self.check(body), [])
        problems = self.check(body, warn="## Avisos de hoje\n\n" + " ".join(WARNING_PHRASES) + "\n\n## 0. x\n")
        self.assertTrue(any("decide/later" in p and "does not name" in p for p in problems), problems)

    def test_a_missing_path_is_a_problem_unless_declared(self) -> None:
        self.assertTrue(any("decide/nothing/file.py" in p for p in self.check("see `decide/nothing/file.py`\n")))
        self.assertEqual(self.check("see `notebooks/decide/docs/04_rodada_paga.md`\n"), [])

    def test_a_declared_path_that_now_exists_is_a_problem(self) -> None:
        problems = self.check("text\n", files=("decide/train/soup.py", "decide/later/run.py"))
        self.assertTrue(any("decide/later is now on this checkout" in p for p in problems), problems)

    def test_the_warnings_are_required(self) -> None:
        self.assertTrue(any("has no section" in p for p in self.check("text\n", warn="## 0. Como ler\n")))
        for phrase in WARNING_PHRASES:
            warn = self.WARN.replace(phrase, "")
            self.assertTrue(any(phrase in p for p in self.check("text\n", warn=warn)), phrase)

    def test_text_of_a_warning_that_was_taken_out_is_a_problem(self) -> None:
        for phrase in RETIRED_PHRASES:
            problems = self.check(f"text {phrase} text\n")
            self.assertTrue(any(phrase in p and "taken out" in p for p in problems), (phrase, problems))

    def test_a_reference_to_a_warning_that_does_not_exist_is_a_problem(self) -> None:
        for text in ("see (aviso 3)\n", "(avisos 1 e 3)\n", "Aviso 9\n"):
            problems = self.check(text)
            self.assertTrue(any("does not exist" in p for p in problems), (text, problems))
        self.assertEqual(self.check("(aviso 1), (aviso 2) and (avisos 1 e 2). O aviso do perfil completo e outra coisa.\n"), [])

    def test_a_subcommand_is_checked_with_its_own_help(self) -> None:
        seen: List[Tuple[str, Tuple[str, ...]]] = []

        def helps(module: str, sub: Tuple[str, ...]) -> str:
            seen.append((module, sub))
            return self.helps(module, sub)
        runbook_problems(self.WARN + "`python -m decide.train.soup build --out c`\n", self.tree(["decide/train/soup.py"]), helps, self.DECLARED)
        self.assertIn(("decide.train.soup", ("build",)), seen)


class ChildEnvironmentLimitsTest(unittest.TestCase):
    """The RUNBOOK says what the filter on the variables of the training children leaves through (``decide.rental.run.CREDENTIAL_NAME``)."""

    def test_the_runbook_says_what_is_dropped_and_what_still_passes(self) -> None:
        from decide.rental import run

        text = RUNBOOK.read_text(encoding="utf-8")
        line = next((l for l in text.splitlines() if l.startswith("- Processos filhos")), "")
        self.assertTrue(line, "the line about the child processes is gone")
        for phrase in ("GIT_CONFIG_", "PGPASSWORD", "MYSQL_PWD", "KUBECONFIG", "NETRC", "DOCKER_CONFIG", "no valor", "logs/day.log"):
            self.assertIn(phrase, line, f"the RUNBOOK does not say {phrase!r} about the environment of the children")
        for name in ("KUBECONFIG", "NETRC", "DOCKER_CONFIG"):
            self.assertIsNone(run.CREDENTIAL_NAME.search(name), f"{name} now is dropped: the RUNBOOK says it passes")


if __name__ == "__main__":
    unittest.main()
