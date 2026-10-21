"""Contract added after the independent review of PR #79 (F4, issue #69): what the review found that no test pinned.

* the targets only come from a corpus that went through ``decide.data.decontaminate``, and say which one (provenance);
* the three properties that make the trained model the served model: non-causal full attention, the left-padding offset, the
  position of the readout (tiny Qwen3.5, CPU, a few seconds);
* the hold-out metric is scored on rows the temperature was not fitted on;
* ``--out`` guards (symlink, ``..``), files that are not the builder's, forged manifests, manifests required for labels and traces,
  the digest of the request in a trace, ``--missing error``, the fingerprint of the checkpoint;
* one ``fold_reasoning`` with a golden file (``tests/fold_reasoning_cases.json``) that the thinker side conforms to.

Everything runs on a CPU, offline.  The tests that need torch, transformers or pyarrow skip cleanly when they are missing.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from decide_b_support import (
    ROOT, FakeTokenizer, choice_request, decontaminated, make_tiny_checkpoint, needs_pyarrow, needs_torch, needs_transformers,
)

try:
    import torch
except ImportError:                                  # the tests that need it are skipped
    torch = None

CASES = ROOT / "tests" / "fold_reasoning_cases.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FoldReasoningContract(unittest.TestCase):
    """``decide.train.protocol.fold_reasoning`` is the one definition of the ``Raciocínio:`` block (SCHEMA.md, section 10)."""

    def setUp(self) -> None:
        from decide.train import protocol as PR
        from decide.train.codes import ANSWER_CODES

        self.PR, self.codes = PR, ANSWER_CODES
        self.cases = json.loads(CASES.read_text(encoding="utf-8"))

    def request(self, case, **extra):
        return {"state": case["state"], "reasoning": case["reasoning"], "questions": choice_request()["questions"], **extra}

    def test_the_golden_cases_are_the_ones_the_review_found(self) -> None:
        names = {case["name"] for case in self.cases}
        for wanted in ("state with trailing whitespace", "reasoning that holds the label line", "reasoning with CRLF and CR",
                       "reasoning with three or more blank lines", "a cut that leaves the label at the start of the tail",
                       "indent on the first line of the reasoning stays"):
            self.assertIn(wanted, names)

    def test_every_golden_case_folds_to_the_expected_state(self) -> None:
        for case in self.cases:
            kwargs = {} if case["max_chars"] is None else {"max_chars": case["max_chars"]}
            folded = self.PR.fold_reasoning(self.request(case), **kwargs)
            self.assertEqual(folded["state"], case["folded_state"], case["name"])
            self.assertNotIn("reasoning", folded, case["name"])
            self.assertEqual(folded["questions"], choice_request()["questions"])

    def test_the_prompt_of_a_request_with_reasoning_is_the_prompt_of_its_folded_state(self) -> None:
        for case in self.cases:
            if case["max_chars"] is not None:
                continue                                    # the clip is the caller's budget; the default budget is tested below
            request = self.request(case)
            folded = self.PR.fold_reasoning(request)
            self.assertEqual(self.PR.messages(request, self.codes), self.PR.messages(folded, self.codes), case["name"])

    def test_the_clip_keeps_the_end_and_is_idempotent(self) -> None:
        case = next(c for c in self.cases if c["max_chars"] is not None)
        once = self.PR.fold_reasoning(self.request(case), max_chars=case["max_chars"])["state"]
        block = once.split("Raciocínio:\n", 1)[1]
        self.assertLessEqual(len(block), case["max_chars"])
        self.assertTrue(block.startswith("[... "))
        self.assertTrue(block.endswith(case["reasoning"].rstrip().splitlines()[-1]))
        again = self.PR.fold_reasoning({**self.request(case), "reasoning": block}, max_chars=case["max_chars"])["state"]
        self.assertEqual(again, once, "a reasoning that was already clipped is not clipped again")

    def test_a_cut_that_leaves_the_label_at_the_start_of_the_tail_is_neutralized_too(self) -> None:
        # Squad C's review of the thinker side (PR 90): "x" + spaces + the label, cut after the spaces, left the label alone on a line of
        # the tail, so the folded state had two label lines.  SCHEMA.md section 10 promises one: the label rule applies after the cut too.
        text = "x" + " " * 600 + "Racioc\u00ednio:"
        for budget in (64, 100, 300):
            with self.subTest(max_chars=budget):
                state = self.PR.fold_reasoning({"state": "s", "reasoning": text, "questions": choice_request()["questions"]}, max_chars=budget)["state"]
                labels = [line for line in state.split("\n") if self.PR.has_reasoning_block(line)]
                self.assertEqual(len(labels), 1, state[-80:])
                self.assertTrue(state.endswith("Racioc\u00ednio :"), state[-40:])
                self.assertTrue(state.startswith("s\n\nRacioc\u00ednio:\n[... "))

    def test_the_folded_state_has_exactly_one_label_line_whatever_the_reasoning_and_the_cut(self) -> None:
        import random

        rng = random.Random(20261021)
        label = "Racioc\u00ednio:"
        pieces = ["a", "bb ", "  ", "\t", label, "\n", " " + label + " ", "Racioc\u00ednio :", "\r\n", "word " * 12]
        samples = 0
        for _ in range(300):
            filler = "".join(rng.choice(["ab ", "cd", "e "]) for _ in range(rng.randint(40, 120)))
            gap = rng.choice([" ", "\t", "  \t "]) * rng.randint(30, 90)
            tail = rng.choice(["", "\nlast line", "\n\n" + label, "\n" + label + "\nmore"])
            texts = [filler + gap + label + tail, "".join(rng.choice(pieces) for _ in range(rng.randint(3, 60)))]
            for text in texts:
                if not text.strip():
                    continue
                for budget in range(64, 150, 3):
                    state = self.PR.fold_reasoning({"state": "s", "reasoning": text, "questions": choice_request()["questions"]},
                                                   max_chars=budget)["state"]
                    self.assertEqual(sum(self.PR.has_reasoning_block(line) for line in state.split("\n")), 1, (text, budget, state))
                    samples += 1
        self.assertGreater(samples, 10000)

    def test_the_indent_of_the_first_line_of_the_reasoning_stays(self) -> None:
        # Review of #87, R3: the rule takes the line breaks at the ends of the reasoning, not the spaces and tabs that start its first line.
        def fold(reasoning):
            return self.PR.fold_reasoning({"state": "s", "reasoning": reasoning, "questions": choice_request()["questions"]})["state"]
        self.assertEqual(fold("  indented\nx"), "s\n\nRacioc\u00ednio:\n  indented\nx")
        self.assertEqual(fold("\n\n\tindented\n"), "s\n\nRacioc\u00ednio:\n\tindented")

    def test_the_default_budget_is_the_one_of_the_thinker_side(self) -> None:
        self.assertEqual(self.PR.REASONING_MAX_CHARS, 8000)
        long = "x" * 20000
        folded = self.PR.fold_reasoning({"state": "s", "reasoning": long, "questions": choice_request()["questions"]})
        self.assertLessEqual(len(folded["state"].split("Raciocínio:\n", 1)[1]), 8000)

    def test_a_state_that_already_has_a_block_is_refused(self) -> None:
        request = {"state": "s\n\nRaciocínio:\nold", "reasoning": "new", "questions": choice_request()["questions"]}
        with self.assertRaises(self.PR.RequestError):
            self.PR.fold_reasoning(request)
        with self.assertRaises(self.PR.RequestError):
            self.PR.messages(request, self.codes)
        with self.assertRaises(self.PR.RequestError):
            self.PR.validate(request, self.codes)

    def test_schema_md_defines_the_normalization(self) -> None:
        text = (ROOT / "decide" / "data" / "SCHEMA.md").read_text(encoding="utf-8")
        section = text.split("## 10.", 1)[1].split("## 11.", 1)[0]
        for word in ("rstrip", "CRLF", "três ou mais linhas em branco", "fold_reasoning", "8000", "fold_reasoning_cases.json", "depois do corte"):
            self.assertIn(word, section)


class ProtocolRefusesImages(unittest.TestCase):
    def test_a_request_with_images_is_refused_not_silently_read_as_text(self) -> None:
        from decide.train import protocol as PR
        from decide.train.codes import ANSWER_CODES

        request = {**choice_request(), "images": ["/nonexistent.png"]}
        with self.assertRaises(PR.RequestError) as caught:
            PR.validate(request, ANSWER_CODES)
        self.assertIn("images", str(caught.exception))
        PR.validate(choice_request(), ANSWER_CODES)


# ------------------------------------------------------------------------------------------------------------- what the targets come from

class TargetsFixture(unittest.TestCase):
    """A seven-question corpus (the first five labeled by one teacher) in a temporary folder."""

    def setUp(self) -> None:
        from decide.data import schema as S
        from decide.train import build_targets as BT

        self.S, self.BT = S, BT
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.requests = {}
        rows = []
        for i in range(7):
            request = choice_request(instructions=f"Question number {i}?", options=["a", "b", "c"], keys=["x", "y", "z"])
            self.requests[f"s:{i}"] = request
            gold = {"q1": None if i in (4, 6) else ["x", "y", "z"][i % 3]}
            rows.append(S.corpus_row(example_id=f"s:{i}", source="s", license="mit", split="dev" if i == 5 else "train", lang="en", request=request, gold=gold))
        S.write_corpus_part(self.root / "corpus" / "s-0000.parquet", rows)
        self.write_teacher("t1", range(5))

    def write_teacher(self, name: str, indexes) -> None:
        rows = [{"example_id": f"s:{i}", "question_id": "q1", "teacher": name, "codes": ["A", "B", "C"], "probs": [0.6, 0.3, 0.1], "temperature": 1.0} for i in indexes]
        self.S.write_teacher_part(self.root / "labels" / name / "part-0000.parquet", rows)
        self.S.seal_directory(self.root / "labels" / name, kind="labels")

    def build(self, out: str = "targets", *, corpus=None, **kw) -> dict:
        return self.BT.build(corpus or self.root / "corpus", [self.root / "labels" / "t1"], self.root / out, **kw)

    def stamp_corpus(self, tool: str) -> None:
        """Rewrite the manifest of the corpus as the manifest of ``tool`` (what build_corpus / decontaminate / a person would leave)."""
        manifest = self.S.seal_directory(self.root / "corpus")
        manifest["tool"] = tool
        self.S.write_manifest(self.root / "corpus" / "manifest.json", manifest)


@needs_pyarrow
class CorpusProvenance(TargetsFixture):
    """``build_targets`` only trains on a corpus that went through the decontamination, and records which one."""

    def refused(self, **kw) -> str:
        with self.assertRaises(self.BT.TargetError) as caught:
            self.build(**kw)
        self.assertFalse((self.root / "targets" / "manifest.json").exists(), "a refused run writes nothing the trainer could read")
        return str(caught.exception)

    def test_a_raw_corpus_is_refused_and_the_message_says_what_to_do(self) -> None:
        self.stamp_corpus("build_corpus")
        message = self.refused()
        self.assertIn("build_corpus", message)
        self.assertIn("decontaminate", message)
        self.assertIn("--allow-undecontaminated", message)

    def test_a_folder_sealed_by_hand_is_refused(self) -> None:
        self.S.seal_directory(self.root / "corpus")
        self.assertIn("seal_directory", self.refused())

    def test_a_loose_parquet_file_is_refused(self) -> None:
        self.assertIn("folder", self.refused(corpus=self.root / "corpus" / "s-0000.parquet"))

    def test_the_stamp_without_its_report_is_refused(self) -> None:
        decontaminated(self.root / "corpus")
        (self.root / "corpus" / "decontamination.json").unlink()
        self.assertIn("decontamination.json", self.refused())

    def test_a_manifest_of_another_tool_is_refused_even_next_to_a_good_report(self) -> None:
        decontaminated(self.root / "corpus")
        manifest = json.loads((self.root / "corpus" / "manifest.json").read_text(encoding="utf-8"))
        manifest["tool"] = "build_corpus"                      # the report is there and carries the stamp, the manifest is not decontaminate's
        self.S.write_manifest(self.root / "corpus" / "manifest.json", manifest)
        self.assertIn("build_corpus", self.refused())

    def test_a_report_that_decontaminate_did_not_write_is_refused(self) -> None:
        decontaminated(self.root / "corpus")
        self.S.write_manifest(self.root / "corpus" / "decontamination.json", {"tool": "somebody", "rows_out": 7})
        self.assertIn("decontamination.json", self.refused())

    def test_a_manifest_that_names_another_report_file_is_refused(self) -> None:
        decontaminated(self.root / "corpus")
        manifest = json.loads((self.root / "corpus" / "manifest.json").read_text(encoding="utf-8"))
        manifest["decontamination"] = "../elsewhere.json"
        self.S.write_manifest(self.root / "corpus" / "manifest.json", manifest)
        self.assertIn("decontamination", self.refused())

    def test_a_clean_corpus_is_accepted_and_the_targets_say_which_one(self) -> None:
        decontaminated(self.root / "corpus")
        report = self.build()
        expected = {"tool": "decontaminate", "manifest_sha256": sha256(self.root / "corpus" / "manifest.json"),
                    "decontamination_sha256": sha256(self.root / "corpus" / "decontamination.json"), "allow_undecontaminated": False}
        self.assertEqual(report["corpus"], expected)
        manifest = json.loads((self.root / "targets" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["corpus"], expected)
        on_disk = json.loads((self.root / "targets" / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["corpus"], expected)

    def test_the_real_decontaminate_output_is_accepted(self) -> None:
        from decide.data import decontaminate as D

        self.S.seal_directory(self.root / "corpus")                    # decontaminate reads its input through a manifest too
        index = D.ReferenceIndex()
        index.freeze()
        D.decontaminate_corpus(self.root / "corpus", self.root / "clean", index)
        report = self.build(corpus=self.root / "clean")
        self.assertEqual(report["corpus"]["tool"], "decontaminate")
        self.assertEqual(report["corpus"]["decontamination_sha256"], sha256(self.root / "clean" / "decontamination.json"))
        self.assertEqual(report["rows_in"], 7)

    def test_the_way_out_is_explicit_and_recorded_in_the_manifest_and_the_report(self) -> None:
        self.stamp_corpus("build_corpus")
        report = self.build(allow_undecontaminated=True)
        expected = {"tool": "build_corpus", "manifest_sha256": sha256(self.root / "corpus" / "manifest.json"), "decontamination_sha256": None,
                    "allow_undecontaminated": True}
        self.assertEqual(report["corpus"], expected)
        manifest = json.loads((self.root / "targets" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["corpus"], expected)
        loose = self.build("loose", corpus=self.root / "corpus" / "s-0000.parquet", allow_undecontaminated=True)
        self.assertEqual(loose["corpus"], {"tool": None, "manifest_sha256": None, "decontamination_sha256": None, "allow_undecontaminated": True})

    def test_the_command_line_has_the_flag_and_a_refusal_exits_with_the_message(self) -> None:
        self.stamp_corpus("build_corpus")
        argv = ["--corpus", str(self.root / "corpus"), "--labels", str(self.root / "labels" / "t1"), "--out", str(self.root / "cli")]
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(self.BT.TargetError):
                self.BT.main(argv)
            self.assertEqual(self.BT.main(argv + ["--allow-undecontaminated"]), 0)
        self.assertTrue(json.loads((self.root / "cli" / "report.json").read_text(encoding="utf-8"))["corpus"]["allow_undecontaminated"])

    def test_changing_the_way_out_rebuilds_the_parts(self) -> None:
        self.stamp_corpus("build_corpus")
        self.build(allow_undecontaminated=True)
        decontaminated(self.root / "corpus")
        again = self.build()
        self.assertEqual(again["parts_resumed"], 0, "the provenance is part of what a finished part was built from")


@needs_pyarrow
class TargetsHardening(TargetsFixture):
    def setUp(self) -> None:
        super().setUp()
        decontaminated(self.root / "corpus")

    def test_a_file_that_is_not_the_builders_is_never_overwritten_even_with_the_name_of_a_part(self) -> None:
        targets = self.root / "targets"
        targets.mkdir()
        (targets / "s-0000.parquet").write_bytes(b"the user's own file")
        with self.assertRaises(self.BT.TargetError) as caught:
            self.build()
        self.assertIn("s-0000.parquet", str(caught.exception))
        self.assertEqual((targets / "s-0000.parquet").read_bytes(), b"the user's own file")

    def test_a_part_the_builder_wrote_is_still_replaced(self) -> None:
        self.build()
        rebuilt = self.build(allow_undecontaminated=True)           # other provenance, same part name: the builder's own file, rewritten
        self.assertEqual(rebuilt["parts_resumed"], 0)
        self.assertEqual(self.build()["parts_resumed"], 0)
        self.assertEqual(self.build()["parts_resumed"], 1)

    def test_a_forged_manifest_cannot_make_the_builder_delete_outside_out(self) -> None:
        victim = self.root / "victim.parquet"
        victim.write_bytes(b"precious")
        targets = self.root / "targets"
        targets.mkdir()
        for name in ("../victim.parquet", str(victim), "sub/x.parquet", ""):
            self.S.write_manifest(targets / "manifest.json", {"tool": "build_targets", "kind": "targets", "parts": [{"name": name}]})
            with self.assertRaises(self.BT.TargetError, msg=name):
                self.build()
            self.assertEqual(victim.read_bytes(), b"precious", name)

    def test_labels_and_traces_are_read_through_a_manifest_like_the_corpus(self) -> None:
        loose = self.root / "labels" / "loose"
        self.S.write_teacher_part(loose / "part-0000.parquet", [{"example_id": "s:0", "question_id": "q1", "teacher": "loose", "codes": ["A", "B", "C"],
                                                                 "probs": [0.6, 0.3, 0.1], "temperature": 1.0}])
        with self.assertRaises(self.BT.TargetError) as caught:
            self.BT.build(self.root / "corpus", [loose], self.root / "targets")
        self.assertIn("manifest", str(caught.exception))
        self.write_trace("alpha", [("s:0", "think", 3, "stop")], seal=False)
        with self.assertRaises(self.BT.TargetError) as caught:
            self.build(traces_path=self.root / "traces")
        self.assertIn("manifest", str(caught.exception))

    def write_trace(self, thinker: str, items, *, seal: bool = True, digest=None) -> None:
        from decide.data import traces as TR

        rows = [TR.trace_row(example_id=e, thinker=thinker, reasoning=text, tokens=n, finish_reason=f,
                             request_sha256=(digest or {}).get(e) or self.S.request_digest(self.requests[e])) for e, text, n, f in items]
        TR.write_trace_part(self.root / "traces" / thinker / "part-0000.parquet", rows)
        if seal:
            self.S.seal_directory(self.root / "traces", kind="traces")

    def test_a_trace_made_for_another_request_is_refused_by_name(self) -> None:
        self.write_trace("alpha", [("s:0", "think", 3, "stop"), ("s:1", "think", 3, "stop")], digest={"s:1": "f" * 64})
        with self.assertRaises(self.BT.TargetError) as caught:
            self.build(traces_path=self.root / "traces")
        self.assertIn("s:1", str(caught.exception))
        self.assertIn("request_sha256", str(caught.exception))

    def test_a_trace_made_for_this_request_is_attached(self) -> None:
        self.write_trace("alpha", [("s:0", "think", 3, "stop")])
        report = self.build(traces_path=self.root / "traces")
        self.assertEqual(report["reasoning_rows"], 1)

    def test_only_the_trace_parts_the_manifest_lists_are_read(self) -> None:
        from decide.data import traces as TR

        self.write_trace("alpha", [("s:0", "think", 3, "stop")])
        TR.write_trace_part(self.root / "traces" / "alpha" / "part-0001.parquet",
                            [TR.trace_row(example_id="s:1", thinker="alpha", reasoning="stray", tokens=3, finish_reason="stop",
                                          request_sha256=self.S.request_digest(self.requests["s:1"]))])       # a part nobody sealed
        report = self.build(traces_path=self.root / "traces")
        self.assertEqual((report["reasoning_rows"], report["traces_unmatched"]), (1, 0))

    def test_a_row_whose_state_already_holds_the_label_line_does_not_take_the_trace(self) -> None:
        from decide.data import schema as S

        request = choice_request("Why?", ["a", "b", "c"], keys=["x", "y", "z"], state="s\n\nRaciocínio:\nalready here")
        self.requests["s:9"] = request
        S.write_corpus_part(self.root / "corpus9" / "s-0000.parquet", [S.corpus_row(example_id="s:9", source="s", license="mit", split="train", lang="en", request=request,
                                                                                     gold={"q1": "x"})])
        decontaminated(self.root / "corpus9")
        self.write_trace("alpha", [("s:9", "think", 3, "stop")])
        report = self.BT.build(self.root / "corpus9", [self.root / "labels" / "t1"], self.root / "t9", traces_path=self.root / "traces")
        self.assertEqual((report["reasoning_rows"], report["reasoning_state_has_block"]), (0, 1))

    def test_missing_error_also_refuses_when_every_teacher_is_missing(self) -> None:
        with self.assertRaises(self.BT.TargetError) as caught:
            self.build(missing="error")
        self.assertIn("s:5", str(caught.exception))

    def test_the_report_counts_the_questions_no_teacher_labeled(self) -> None:
        report = self.build()
        self.assertEqual((report["questions"], report["unlabeled_questions"]), (7, 2))
        self.write_teacher("t2", range(7))
        both = self.BT.build(self.root / "corpus", [self.root / "labels" / "t1", self.root / "labels" / "t2"], self.root / "t12",
                             combiner_path=self.combiner(), missing="renormalize")
        self.assertEqual(both["unlabeled_questions"], 0)

    def combiner(self) -> str:
        path = self.root / "combiner.json"
        path.write_text(json.dumps({"version": 1, "method": "mean", "teachers": {"t1": {}, "t2": {}}}), encoding="utf-8")
        return str(path)

    def test_out_is_judged_by_where_it_really_is(self) -> None:
        corpus, labels = self.root / "corpus", self.root / "labels" / "t1"
        link = self.root / "link-to-corpus"
        os.symlink(corpus, link, target_is_directory=True)
        inside_labels = self.root / "targets" / ".." / "labels" / "t1" / "x"
        for out in (link, inside_labels):
            with self.assertRaises(self.BT.TargetError, msg=str(out)):
                self.BT.build(corpus, [labels], out)
        self.assertFalse((labels / "x").exists())
        real = self.root / "real-labels"
        os.symlink(self.root / "labels", real, target_is_directory=True)
        with self.assertRaises(self.BT.TargetError, msg="the labels given through a symlink, --out inside the real folder"):
            self.BT.build(corpus, [real / "t1"], self.root / "labels" / "t1" / "inside")


# ------------------------------------------------------------------------------------------------------------- the trained model is the served model

@needs_transformers
class JointModelProperties(unittest.TestCase):
    """The three properties no loss curve can show: non-causal full attention, the left-padding offset and the readout position."""

    @classmethod
    def setUpClass(cls) -> None:
        from decide.train import model as MODEL
        from decide.train.codes import ANSWER_CODES
        from decide.train.protocol import RequestEncoder

        cls.MODEL, cls.tmp = MODEL, tempfile.TemporaryDirectory()
        cls.tokenizer = FakeTokenizer()
        cls.base = make_tiny_checkpoint(Path(cls.tmp.name) / "base", cls.tokenizer, seed=3)
        cls.encoder = RequestEncoder(cls.tokenizer, ANSWER_CODES)
        cls.model = MODEL.JointModel.load(cls.base, mode="none")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def two_questions(self, second: str, first: str = "Which one is the capital?"):
        request = choice_request(first, ["Paris", "Rome", "Berlin"], state="A short state about capitals.")
        request["questions"]["q2"] = {"type": "choice", "instructions": second, "criteria": {"A": "left", "B": "right"}}
        return self.encoder.encode(request)

    def logits(self, model, encoded):
        with torch.no_grad():
            return model(self.MODEL.collate(encoded if isinstance(encoded, list) else [encoded], self.encoder.pad_id))

    def test_a_question_reads_what_comes_after_its_marker_because_full_attention_is_not_causal(self) -> None:

        one, other = self.two_questions("Is it north or south?"), self.two_questions("Completely different words here: zephyr anchor basalt cinder.")
        self.assertEqual(one.markers[0], other.markers[0], "the first marker does not move: only text after it changes")
        moved = (self.logits(self.model, one)[0, :3] - self.logits(self.model, other)[0, :3]).abs().max().item()
        self.assertGreater(moved, 1e-4, "the first question's readout must see the second question")
        with mock.patch.object(self.MODEL, "enable_noncausal_full_attention", lambda text_model: None):
            causal = self.MODEL.JointModel.load(self.base, mode="none")
        still = (self.logits(causal, one)[0, :3] - self.logits(causal, other)[0, :3]).abs().max().item()
        self.assertLess(still, 1e-6, "control: without the hook the first question cannot see the second, so this test can tell the two apart")

    def test_an_item_alone_and_the_same_item_in_a_batch_with_a_longer_one_give_the_same_logits(self) -> None:
        short = self.two_questions("Is it north or south?")
        longer = self.encoder.encode(choice_request("Which one?", ["a", "b", "c"], state=" ".join(["river stone market lantern"] * 60)))
        self.assertGreater(longer.length, short.length + 100)
        alone = self.logits(self.model, short)
        batched = self.logits(self.model, [short, longer])
        self.assertEqual(alone.shape[0], len(short.markers))
        self.assertTrue(torch.allclose(alone, batched[:alone.shape[0]], atol=1e-5), (alone - batched[:alone.shape[0]]).abs().max().item())
        flipped = self.logits(self.model, [longer, short])
        self.assertTrue(torch.allclose(alone, flipped[len(longer.markers):], atol=1e-5), "the offset is right whichever row the short item is in")

    def test_the_logits_are_the_readout_of_the_hidden_state_at_the_markers(self) -> None:
        encoded = self.two_questions("Is it north or south?")
        batch = self.MODEL.collate([encoded], self.encoder.pad_id)
        with torch.no_grad():
            hidden = self.model.backbone(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False).last_hidden_state
            by_hand = self.model.readout(hidden[0, list(encoded.markers)]).float()
            got = self.model(batch)
        for question, count in enumerate(encoded.counts):
            self.assertTrue(torch.allclose(got[question, :count], by_hand[question, :count], atol=1e-6), question)
        shifted = self.model.readout(hidden[0, [m - 1 for m in encoded.markers]]).float()
        self.assertGreater((got[0, :3] - shifted[0, :3]).abs().max().item(), 1e-4, "control: the token before the marker reads something else")


@needs_transformers
class HeldoutIsNotInSample(unittest.TestCase):
    """The calibrated number the evaluation reports is scored on the half of the examples the temperature was NOT fitted on."""

    def test_the_fit_side_and_the_score_side_are_disjoint_and_both_are_used(self) -> None:
        from decide.train import calibrate
        from decide.train import evaluate as EV
        from decide.train.codes import ANSWER_CODES
        from decide.train.data import Example
        from decide.train.protocol import RequestEncoder

        encoder = RequestEncoder(FakeTokenizer(), ANSWER_CODES)
        examples = []
        for i in range(12):
            request = choice_request(f"Question {i}?", ["a", "b", "c"], keys=["x", "y", "z"])
            examples.append(Example(example_id=f"d:{i}", source="d", split="dev", lang="en", request=request, gold_index={"q1": i % 3},
                                    combined={"q1": None}, teachers=(), reasoning=None))

        def fake_logits(model, items, pad_id, **kwargs):
            return [[[float(number), 0.0, 0.0]] for number, _ in enumerate(items)]       # a distinct logit vector per question: the row is its number

        fits, scores = [], []
        real_fit, real_metrics = calibrate.fit_temperature, calibrate.metrics

        def spy_fit(values, labels, *args, **kwargs):
            fits.append({int(v[0]) for v in values})
            return real_fit(values, labels, *args, **kwargs)

        def spy_metrics(values, labels, temperature):
            scores.append(({int(v[0]) for v in values}, temperature))
            return real_metrics(values, labels, temperature)

        by_parity = lambda example_id: int(example_id.split(":")[1]) % 2  # noqa: E731
        with mock.patch.object(EV, "question_logits", fake_logits), mock.patch.object(EV, "fold_of", by_parity), \
                mock.patch.object(calibrate, "fit_temperature", spy_fit), mock.patch.object(calibrate, "metrics", spy_metrics):
            report = EV.evaluate_examples(None, encoder, examples, minimum=1)
        self.assertIn("heldout", report)
        fit_side = fits[-1]                                  # the last single fit is the hold-out one (fit_temperatures fits per type)
        score_side = scores[-1][0]
        self.assertTrue(fit_side and score_side)
        self.assertEqual(fit_side & score_side, set(), "the hold-out metric must not see the rows its temperature was fitted on")
        self.assertEqual(len(fit_side | score_side), 12)
        order = {example.example_id: n for n, example in enumerate(examples)}
        self.assertEqual({n for example_id, n in order.items() if by_parity(example_id) == 0}, fit_side)


    def test_the_reasoning_report_leaves_out_the_examples_whose_state_has_the_block(self) -> None:
        from decide.train import evaluate as EV
        from decide.train.codes import ANSWER_CODES
        from decide.train.data import Example
        from decide.train.protocol import RequestEncoder

        encoder = RequestEncoder(FakeTokenizer(), ANSWER_CODES)
        examples = []
        for i, state in enumerate(["plain state", "plain state two", "s\n\nRaciocínio:\nalready here", "plain state four"]):
            examples.append(Example(example_id=f"r:{i}", source="r", split="dev", lang="en", request=choice_request(f"Question {i}?", ["a", "b", "c"], keys=["x", "y", "z"], state=state),
                                    gold_index={"q1": i % 3}, combined={"q1": None}, teachers=(), reasoning="a thought"))
        examples.append(Example(example_id="r:9", source="r", split="dev", lang="en", request=choice_request("Question 9?", ["a", "b", "c"], keys=["x", "y", "z"]),
                                gold_index={"q1": 0}, combined={"q1": None}, teachers=(), reasoning=None))
        with mock.patch.object(EV, "question_logits", lambda model, items, pad_id, **kwargs: [[[0.0, 1.0, 2.0]] for _ in items]):
            shown = EV.evaluate_examples(None, encoder, examples, reasoning=True, minimum=1)
            plain = EV.evaluate_examples(None, encoder, examples, reasoning=False, minimum=1)
        self.assertEqual(shown["examples"], 3, "three have a reasoning and a state that can take it")
        self.assertEqual(plain["examples"], 5)


# ------------------------------------------------------------------------------------------------------------- the checkpoint a run started from

@needs_torch
class CheckpointFingerprint(unittest.TestCase):
    def folder(self, root: Path, name: str, *, shard: bytes = b"backbone weights", index: str = "one", head: bytes = b"same head") -> Path:
        folder = root / name
        folder.mkdir()
        (folder / "decision_config.json").write_text('{"temperature": 1.0}', encoding="utf-8")
        (folder / "readout.safetensors").write_bytes(head)
        (folder / "model.safetensors").write_bytes(shard)
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "model.safetensors"}, "tag": index}), encoding="utf-8")
        return folder

    def test_the_backbone_weights_and_their_index_are_part_of_the_fingerprint(self) -> None:
        from decide.train import train as TRAIN

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = TRAIN.checkpoint_digest(self.folder(root, "a"))
            self.assertEqual(TRAIN.checkpoint_digest(root / "a"), base)
            variants = {
                "another index": self.folder(root, "b", index="two"),
                "same size, other bytes": self.folder(root, "c", shard=b"backbone WEIGHTS"),
                "other size": self.folder(root, "d", shard=b"backbone weights, one more"),
                "other head": self.folder(root, "e", head=b"other head"),
            }
            for what, folder in variants.items():
                self.assertNotEqual(TRAIN.checkpoint_digest(folder), base, what)
            (root / "a" / "README.md").write_text("a model card edit", encoding="utf-8")
            self.assertEqual(TRAIN.checkpoint_digest(root / "a"), base, "documentation is not the checkpoint")


if __name__ == "__main__":
    unittest.main()
