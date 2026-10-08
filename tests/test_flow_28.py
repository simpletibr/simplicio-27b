"""Flow tests for the dataset v2 builder (#28).

End to end, with no network and no model: scripts/build_dataset_v2.py runs in small mode over the
real history of this repository, and over a throwaway git repository whose commits cover every
filter, and each row it writes is checked the way a trainer would consume it: SEARCH/REPLACE with
markers of 4 characters (`<<<<`, `====`, `>>>>`), a SEARCH that appears verbatim in the parent
version of the file, and blocks that turn that parent into the committed file byte for byte.

The history test is skipped, with a message, when the pinned commit is not in the checkout
(for example a shallow clone).
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset_v2 as b  # noqa: E402
SYSTEM_PROMPT, USER_TEMPLATE, apply_blocks, parse_blocks = b.SYSTEM_PROMPT, b.USER_TEMPLATE, b.apply_blocks, b.parse_blocks

ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
}
OPEN, DIVIDER, CLOSE = "<<<< SEARCH", "====", ">>>> REPLACE"


def run_git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], env=ENV, capture_output=True, check=True).stdout
    return out.decode("utf-8")


def pinned_commit_available() -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{b.LOCAL.pin}^{{commit}}"],
                          capture_output=True).returncode == 0


def quiet_build(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return b.build(*args, **kwargs)


class RowChecks(unittest.TestCase):
    """What every row must satisfy, whichever repository it came from."""

    def check_row(self, row: dict, repo_dir: Path) -> None:
        rid = row["id"]
        roles = [m["role"] for m in row["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"], rid)
        system, user, answer = (m["content"] for m in row["messages"])
        self.assertEqual(system, SYSTEM_PROMPT, rid)
        self.assertEqual(row["source"], "real_commit", rid)
        # Markers: exactly the 4-character form, nothing around the blocks (no format hint, no <think>).
        lines = answer.splitlines()
        self.assertEqual(lines[0], OPEN, rid)
        self.assertEqual(lines[-1], CLOSE, rid)
        self.assertEqual(lines.count(OPEN), lines.count(CLOSE), rid)
        self.assertEqual(lines.count(DIVIDER), lines.count(OPEN), rid)
        for line in lines:
            self.assertIsNone(re.match(r"^(<{5,}|>{5,}|={5,})", line), f"{rid}: marker longer than 4 characters")
        self.assertNotIn("<think>", answer, rid)
        self.assertNotIn("</think>", answer, rid)
        self.assertNotIn("Responda com um ou mais blocos", answer, rid)
        blocks = parse_blocks(answer)
        self.assertEqual(len(blocks), lines.count(OPEN), rid)
        # The file the model sees is the parent version, and every SEARCH is verbatim in it.
        old = run_git(repo_dir, "show", f"{row['parent']}:{row['path']}")
        new = run_git(repo_dir, "show", f"{row['commit']}:{row['path']}")
        expected_user = USER_TEMPLATE.format(
            context=f"Projeto Python {row['repo']}.", edit_file=row["path"], code=old,
            instruction=user.rsplit("\nTarefa: ", 1)[1])
        self.assertEqual(user, expected_user, rid)
        for search, _ in blocks:
            self.assertTrue(search, rid)
            self.assertIn(search, old, f"{rid}: SEARCH is not verbatim in the source file")
            self.assertIn(search, user.split("Codigo Atual:\n", 1)[1].rsplit("\nTarefa: ", 1)[0], rid)
        self.assertEqual(apply_blocks(old, blocks), new, rid)


class SmallModeOnTheRealHistory(RowChecks):
    @unittest.skipUnless(pinned_commit_available(), f"{b.LOCAL.pin} is not in this checkout (shallow clone?)")
    def test_small_mode_rows_are_search_replace_and_search_is_verbatim_in_the_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "v2"
            result = quiet_build([b.LOCAL], out, Path(tmp) / "work", b.eval_grams(), limit=2)
            rows = result["train"] + result["val"]
            self.assertGreaterEqual(len(rows), 1)
            self.assertLessEqual(len(rows), 2)
            for row in rows:
                with self.subTest(row=row["id"]):
                    self.check_row(row, ROOT)
                    self.assertEqual(row["repo"], b.LOCAL.name)
            written = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(written, result["train"])
            self.assertTrue((out / "STATS.md").is_file())
            self.assertTrue((out / "LICENSES" / "simpletibr__simplicio-27b.txt").is_file())

    @unittest.skipUnless(pinned_commit_available(), f"{b.LOCAL.pin} is not in this checkout (shallow clone?)")
    def test_cli_small_mode_exits_0_and_fails_only_when_the_milestone_is_asked_for(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--out", str(Path(tmp) / "o"), "--workdir", str(Path(tmp) / "w"), "--limit", "1"]
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(b.main(argv), 0)
                self.assertEqual(b.main([*argv, "--min-train", str(b.MIN_TRAIN)]), 1)
            self.assertIn("FAIL: train has", printed.getvalue())
            self.assertRegex(printed.getvalue(), r"(?m)^train \d+ val \d+$")


class FixtureRepository(RowChecks):
    """A throwaway repository: one commit per filter, so the whole pipeline runs without the network."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.repo = cls.tmp / "fixture"
        cls.repo.mkdir()
        run_git(cls.repo, "init", "-q", "-b", "main")

        def commit(files: dict[str, str | None], message: str) -> None:
            for name, text in files.items():
                path = cls.repo / name
                if text is None:
                    path.unlink()
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8", newline="")
            run_git(cls.repo, "add", "-A")
            run_git(cls.repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", message)

        ports = "def parse_port(value):\n    return int(value)\n\n\ndef is_valid(port):\n    return 0 < port < 65536\n"
        big = "".join(f"V{i} = {i}\n" for i in range(450))
        task = next(t for t in (b.decontam_tasks.task_dirs(b.decontam_tasks.TASKS))
                    if len(b.decontam_tasks.tokens(json.loads((t / "task.json").read_text())["instruction"])) >= 20)
        cls.eval_text = json.loads((task / "task.json").read_text(encoding="utf-8"))["instruction"]
        commit({"LICENSE": "MIT fixture license\n", "app/ports.py": ports, "app/loader.py": "def load():\n    return 1\n",
                "app/parser.py": "def parse():\n    return 1\n", "app/client.py": "URL = 'http://localhost'\n", "app/banner.py": "TEXT = ''\n", "app/big.py": big,
                "app/old.py": "DEFAULT = 1\n"}, "Initial import")
        commit({"app/ports.py": ports.replace("    return int(value)", "    if value is None:\n        return 0\n    return int(value)")},
               "fix: handled None in parse_port (#12)")
        commit({"app/ports.py": ports.replace("    return int(value)", "    if value is None:\n        return 0\n    return int(value)")
                .replace("def is_valid(port):\n", 'def is_valid(port):\n    """Tell whether port is in range."""\n')},
               "Add docstring to is_valid")
        commit({"app/parser.py": "def parse(:\n    return 1\n"}, "Fix the parser module")
        commit({"app/client.py": "URL = 'http://localhost'\nAPI_KEY = 'abcdef0123456789abcdef'\n"},
               "Use a configured API key in client")
        commit({"app/ports.py": ports.replace("    return int(value)", "    return int(str(value))"),
                "app/loader.py": "def load():\n    return 2\n"}, "Update ports and loader together")
        commit({"app/ports.py": ports.replace("    return int(value)", "    return int(str(value).strip())")}, "Bump version")
        commit({"app/banner.py": f"TEXT = {cls.eval_text!r}\n"}, "Update banner text for the page")
        commit({"app/big.py": big.replace("V3 = 3", "V3 = 10")}, "Change the value of V3 to ten")
        commit({"app/old.py": "DEFAULT = 2\n"}, "Change the old default to two")
        commit({"app/old.py": None}, "Remove the old module")
        cls.pin = run_git(cls.repo, "rev-parse", "HEAD").strip()
        commit({"app/ports.py": ports.replace("    return int(value)", "    return int(float(value))")},
               "Convert the port through float first")
        cls.late = run_git(cls.repo, "rev-parse", "HEAD").strip()
        cls.source = b.Source("fixture/app", "MIT", cls.pin, "LICENSE", url=str(cls.repo), require_at_pin=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def build(self, sources, name="out"):
        return quiet_build(sources, self.tmp / name, self.tmp / "work", b.eval_grams())

    def test_each_filter_drops_its_commit_and_the_kept_rows_are_valid(self):
        result = self.build([self.source])
        stats = result["per_source"]["fixture/app"]
        self.assertEqual(stats["reasons"], {
            "bad_message": 1,           # "Bump version" is too short to be an instruction
            "contaminated": 1,          # the banner holds the statement of an eval task
            "file_too_long": 1,
            "kept": 1,
            "multi_file_commit": 2,     # the subject describes two files, so it is not used for either
            "no_code_change": 1,        # a docstring was added
            "not_text_modification": 1, # the commit deletes the file
            "removed_from_pin": 1,      # the file is gone at the pin
            "secret": 1,
            "syntax_error": 1,
        })
        self.assertEqual(stats["contaminated"], 1)
        self.assertEqual(len(result["train"]), 1)
        row = result["train"][0]
        self.assertEqual((row["path"], row["scope"], row["instruction_source"]),
                         ("app/ports.py", "single_file_commit", "commit_subject"))
        self.assertTrue(row["messages"][1]["content"].endswith("\nTarefa: Handle None in parse_port."))
        self.check_row(row, self.repo)

    def test_pin_limits_what_is_read_and_the_license_is_copied_from_the_pin(self):
        result = self.build([self.source], "pin")
        self.assertNotIn(self.late[:12], json.dumps(result["train"]))
        self.assertEqual((self.tmp / "pin" / "LICENSES" / "fixture__app.txt").read_text(), "MIT fixture license\n")
        late = dataclasses.replace(self.source, pin=self.late)
        again = self.build([late], "late")
        self.assertIn(self.late[:12], json.dumps(again["train"]))

    def test_edits_of_files_that_were_removed_later_are_kept_only_when_asked_for(self):
        loose = dataclasses.replace(self.source, require_at_pin=False)
        result = self.build([loose], "loose")
        self.assertEqual(sorted(r["path"] for r in result["train"]), ["app/old.py", "app/ports.py"])
        for row in result["train"]:
            self.check_row(row, self.repo)

    def test_validation_repository_goes_to_val_and_the_rest_to_train(self):
        val = dataclasses.replace(self.source, name=b.VAL_REPO)
        result = self.build([self.source, val], "split")
        self.assertEqual({r["repo"] for r in result["train"]}, {"fixture/app"})
        self.assertEqual({r["repo"] for r in result["val"]}, {b.VAL_REPO})
        out = self.tmp / "split"
        self.assertEqual(len((out / "val.jsonl").read_text(encoding="utf-8").splitlines()), 1)
        self.assertIn("| fixture/app |", (out / "STATS.md").read_text(encoding="utf-8"))

    def test_curated_instruction_is_used_for_a_multi_file_commit_and_the_cap_and_limit_apply(self):
        sha = run_git(self.repo, "log", "--format=%H", "--grep=Update ports and loader together").strip()
        curated = dataclasses.replace(self.source, curated={sha: "Troque int(value) por int(str(value)) em parse_port."})
        result = self.build([curated], "curated")
        mine = [r for r in result["train"] if r["commit"] == sha]
        self.assertEqual(sorted(r["path"] for r in mine), ["app/loader.py", "app/ports.py"])
        self.assertEqual({r["scope"] for r in mine}, {"multi_file_commit"})
        self.assertEqual({r["instruction_source"] for r in mine}, {"curated_from_commit"})
        self.assertTrue(all(r["messages"][1]["content"].endswith("Troque int(value) por int(str(value)) em parse_port.")
                            for r in mine))
        for row in result["train"]:
            self.check_row(row, self.repo)
        capped = quiet_build([curated], self.tmp / "cap", self.tmp / "work", b.eval_grams(), cap=1)
        self.assertEqual(len(capped["train"]), 1)
        limited = quiet_build([curated], self.tmp / "lim", self.tmp / "work", b.eval_grams(), limit=1)
        self.assertEqual(len(limited["train"]), 1)

    def test_building_twice_gives_the_same_bytes(self):
        self.build([self.source], "a")
        self.build([self.source], "b")
        for name in ("train.jsonl", "val.jsonl", "STATS.md"):
            self.assertEqual((self.tmp / "a" / name).read_bytes(), (self.tmp / "b" / name).read_bytes(), name)


class Functions(unittest.TestCase):
    def test_blocks_round_trip(self):
        old = "def f(x):\n    return x\n\n\ndef g():\n    pass\n"
        new = "def f(x):\n    if x is None:\n        return 0\n    return x\n\n\ndef g():\n    pass\n"
        self.assertEqual(apply_blocks(old, parse_blocks(b.render(b.blocks_for(old, new)))), new)

    def test_instruction_from_commit(self):
        self.assertEqual(b.instruction_from_commit("fix: handled None in parse_port (#123)\n"), "Handle None in parse_port.")
        self.assertIsNone(b.instruction_from_commit("Bump version to 1.2.3"))
        self.assertIsNone(b.instruction_from_commit("Merge branch 'main' into dev"))
        self.assertIsNone(b.instruction_from_commit("security: read admin key from env instead of hardcoding"))

    def test_docstring_only_change_is_not_code(self):
        self.assertEqual(b.code_dump('def f():\n    """a"""\n    return 1\n'), b.code_dump('def f():\n    """b"""\n    return 1\n'))
        self.assertNotEqual(b.code_dump("def f():\n    return 1\n"), b.code_dump("def f():\n    return 2\n"))

    def test_secret_pattern_flags_literal_credentials_and_not_variables(self):
        self.assertTrue(b.SECRET_RE.search("url = 'https://h/api.php?key=abcd_efgh_ijkl_mnop'"))
        self.assertTrue(b.SECRET_RE.search('API_TOKEN = "abcdef0123456789abcdef"'))
        self.assertFalse(b.SECRET_RE.search("url = f'{BASE}?key={key}'"))
        self.assertFalse(b.SECRET_RE.search("key = os.environ['SIMPLETI_ADMIN_KEY']"))

    def test_external_list_is_the_ten_repositories_of_the_issue_with_pinned_commits(self):
        self.assertEqual(len(b.EXTERNAL), 10)
        self.assertEqual(len({s.name for s in b.EXTERNAL}), 10)
        self.assertIn(b.VAL_REPO, {s.name for s in b.EXTERNAL})
        for s in b.EXTERNAL:
            self.assertRegex(s.pin, r"^[0-9a-f]{40}$")
            self.assertIn(s.license, {"MIT", "BSD-3-Clause"})
            self.assertEqual(s.url, f"https://github.com/{s.name}.git")


if __name__ == "__main__":
    unittest.main()
