"""Flows added after the independent review of PR #79 (F4, issue #69): what a run leaves behind, on a tiny random Qwen3.5 (CPU, offline).

* the checkpoint says which corpus it was trained on (and whether the decontamination check was bypassed);
* ``final/`` appears whole or not at all, and a resume cleans what a killed save left;
* ``--dtype`` reaches the model and the record of the run, and a resume cannot change it;
* the requests the length limit dropped are counted in ``summary.json`` and in the checkpoint;
* ``final/`` and the soup carry the LICENSE and the NOTICE of the checkpoint they come from, plus a ``Modifications:`` line;
* the soup only deletes a work folder it made itself;
* the package docs say what the code does.

The tests skip cleanly when torch, transformers or pyarrow is missing.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest import mock

from decide_b_support import (
    HAVE_PYARROW, HAVE_TRANSFORMERS, ROOT, FakeTokenizer, choice_request, make_targets_dataset, make_tiny_checkpoint,
)

_STATE: Dict[str, Any] = {}

RUN = ("--batch-rows", "8", "--token-budget", "8192", "--max-length", "1024", "--cpu-threads", "2", "--device", "cpu", "--no-permute")
SHORT = ("--epochs", "1", "--effective-batch", "4", *RUN)                 # two steps
THREE = ("--epochs", "3", "--effective-batch", "8", *RUN)                 # three steps over the eight training examples
FULL = ("--mode", "full", "--lr", "5e-3")
#: Twelve steps with a resume state every two: a run that dies after the last step restarts from step 10.
ARGS = ("--epochs", "6", "--effective-batch", "4", "--batch-rows", "4", "--token-budget", "4096", "--max-length", "1024",
        "--cpu-threads", "2", "--device", "cpu", "--resume-every", "2")


def setUpModule() -> None:
    if not (HAVE_TRANSFORMERS and HAVE_PYARROW):
        raise unittest.SkipTest("torch, transformers and pyarrow are needed")
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    make_tiny_checkpoint(root / "base", FakeTokenizer())
    make_targets_dataset(root / "data", train=8, dev=6)
    make_targets_dataset(root / "mixed", train=8, dev=6, long_every=4)         # examples 0, 4 (train) and 8, 12 (dev) have 144 tokens, the rest 104
    from safetensors import safe_open

    from decide.train import build_targets as BT
    from decide.train import model as MODEL
    from decide.train import soup as SOUP
    from decide.train import temperature as TEMPERATURE
    from decide.train import train as TRAIN

    _STATE.update(tmp=tmp, root=root, base=root / "base", data=root / "data", targets=root / "data" / "targets", mixed=root / "mixed" / "targets", runs={},
                  safe_open=safe_open,
                  BT=BT, MODEL=MODEL, SOUP=SOUP, TEMPERATURE=TEMPERATURE, TRAIN=TRAIN)


def tearDownModule() -> None:
    if "tmp" in _STATE:
        _STATE["tmp"].cleanup()
    _STATE.clear()


def argv_of(name: str, *extra: str, targets: "Path | None" = None) -> list:
    return ["--targets", str(targets or _STATE["targets"]), "--checkpoint", str(_STATE["base"]), "--out", str(_STATE["root"] / "runs" / name), *extra]


def train(name: str, *extra: str, targets: "Path | None" = None, cache: bool = True) -> Path:
    """Run ``decide.train.train`` in this process on the tiny checkpoint; the run folder is ``runs/<name>``."""
    runs = _STATE["runs"]
    if cache and name in runs:
        return runs[name]
    os.environ.pop("DECIDE_EVENTS", None)
    with contextlib.redirect_stdout(io.StringIO()):
        rc = _STATE["TRAIN"].main(argv_of(name, *extra, targets=targets), tokenizer=FakeTokenizer())
    assert rc == 0
    runs[name] = _STATE["root"] / "runs" / name
    return runs[name]


def resume(name: str, *extra: str, targets: "Path | None" = None) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return _STATE["TRAIN"].main(argv_of(name, *extra, "--resume", targets=targets), tokenizer=FakeTokenizer())


def events(out: Path) -> list:
    path = out / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


def config(directory: Path) -> Dict[str, Any]:
    return json.loads((directory / "decision_config.json").read_text(encoding="utf-8"))


def summary(out: Path) -> Dict[str, Any]:
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


class ProvenanceReachesTheCheckpoint(unittest.TestCase):
    """What the targets say about their corpus is copied into the run and into the checkpoint (``decision_config.json``)."""

    def test_the_checkpoint_and_the_event_log_say_which_corpus_the_run_trained_on(self) -> None:
        out = train("provenance", *SHORT, *FULL)
        wanted = json.loads((_STATE["targets"] / "manifest.json").read_text(encoding="utf-8"))["corpus"]
        self.assertEqual(wanted["tool"], "decontaminate")
        self.assertFalse(wanted["allow_undecontaminated"])
        self.assertEqual(config(out / "final")["training"]["corpus"], wanted)
        started = [e for e in events(out) if e["kind"] == "training_started"]
        self.assertEqual(started[0]["corpus"], wanted)

    def test_a_dataset_built_with_the_way_out_shows_it_in_the_checkpoint(self) -> None:
        other = _STATE["data"] / "targets_open"
        with contextlib.redirect_stdout(io.StringIO()):
            _STATE["BT"].main(["--corpus", str(_STATE["data"] / "corpus"), "--labels", str(_STATE["data"] / "labels"), "--out", str(other),
                               "--allow-undecontaminated"])
        out = train("provenance_open", *SHORT, *FULL, targets=other)
        self.assertTrue(config(out / "final")["training"]["corpus"]["allow_undecontaminated"])


class FinalIsAtomic(unittest.TestCase):
    """``final/`` is written under another name and renamed when whole: a kill during the save never leaves a half checkpoint."""

    def test_a_save_that_dies_leaves_no_final_and_the_resume_makes_it(self) -> None:
        MODEL = _STATE["MODEL"]
        real = MODEL.JointModel.save

        def dies(self, directory, **kwargs):
            real(self, directory, **kwargs)                       # everything is written ...
            raise OSError("the machine was preempted while the checkpoint was written")        # ... and the run dies before it is done

        with mock.patch.object(MODEL.JointModel, "save", dies):
            with self.assertRaises(OSError):
                train("cut", *ARGS, *FULL, cache=False)
        out = _STATE["root"] / "runs" / "cut"
        self.assertFalse((out / "final").exists(), "a checkpoint that was not finished is not called final")
        self.assertTrue((out / "resume.pt").is_file())
        self.assertFalse((out / "summary.json").exists())
        self.assertEqual(resume("cut", *ARGS, *FULL), 0)
        self.assertTrue((out / "final" / "decision_config.json").is_file())
        self.assertFalse((out / "final.tmp").exists())
        MODEL.JointModel.load(out / "final", mode="none")
        self.assertTrue(summary(out)["complete"])

    def test_a_stale_final_tmp_is_cleaned_by_the_resume(self) -> None:
        out = train("stale", *ARGS, *FULL, "--max-steps", "5", cache=False)
        (out / "final.tmp").mkdir()
        (out / "final.tmp" / "half-written.safetensors").write_bytes(b"cut in the middle")
        self.assertEqual(resume("stale", *ARGS, *FULL), 0)
        self.assertFalse((out / "final.tmp").exists())
        self.assertFalse((out / "final" / "half-written.safetensors").exists())
        self.assertTrue((out / "final" / "readout.safetensors").is_file())

    def test_resuming_a_run_that_already_has_its_final_is_refused_before_it_trains(self) -> None:
        out = train("has_final", *ARGS, *FULL, cache=False)
        self.assertTrue((out / "final").is_dir())
        before = len(events(out))
        with self.assertRaises(ValueError) as caught:
            resume("has_final", *ARGS, *FULL)
        self.assertIn("the run is finished", str(caught.exception), "a finished run has no resume.pt any more: the message must not say that is the problem")
        self.assertEqual(len(events(out)), before, "a refused resume trains nothing")


#: Twelve steps and the default --resume-every (100): the run never reaches a step that is a multiple of it.
NO_EVERY = ("--epochs", "6", "--effective-batch", "4", "--batch-rows", "4", "--token-budget", "4096", "--max-length", "1024",
            "--cpu-threads", "2", "--device", "cpu")


def training_steps(out: Path) -> list:
    return [e["step"] for e in events(out) if e["kind"] == "training_step"]


def model_tensors(final: Path) -> Dict[str, Any]:
    with _STATE["safe_open"](str(final / "model.safetensors"), "pt") as handle:
        return {name: handle.get_tensor(name).float() for name in handle.keys()}


class ResumeStateAtTheEnd(unittest.TestCase):
    """The last step also saves ``resume.pt``, so a death in the final save costs the save and not the run; a finished run leaves none behind."""

    def dies_in_the_final_save(self, name: str, *extra: str) -> Path:
        MODEL = _STATE["MODEL"]
        real = MODEL.JointModel.save

        def dies(self, directory, **kwargs):
            real(self, directory, **kwargs)
            raise OSError("the machine was preempted while the checkpoint was written")

        with mock.patch.object(MODEL.JointModel, "save", dies):
            with self.assertRaises(OSError):
                train(name, *extra, cache=False)
        return _STATE["root"] / "runs" / name

    def test_a_run_shorter_than_resume_every_can_resume_after_a_death_in_the_final_save(self) -> None:
        import torch

        out = self.dies_in_the_final_save("short_cut", *NO_EVERY, *FULL)
        self.assertEqual(training_steps(out), list(range(1, 13)))
        self.assertFalse((out / "final").exists())
        self.assertTrue((out / "resume.pt").is_file(), "twelve steps and --resume-every 100: without a save at the last step there is nothing to resume from")
        self.assertEqual(int(torch.load(out / "resume.pt", map_location="cpu", weights_only=True)["step"]), 12)
        self.assertEqual(resume("short_cut", *NO_EVERY, *FULL), 0)
        self.assertEqual(training_steps(out), list(range(1, 13)), "not one step was trained again")
        self.assertTrue(summary(out)["complete"])
        whole = model_tensors(train("short_whole", *NO_EVERY, *FULL, cache=False) / "final")
        resumed = model_tensors(out / "final")
        self.assertEqual(set(whole), set(resumed))
        self.assertLess(max(float((whole[name] - resumed[name]).abs().max()) for name in whole), 1e-6, "the resumed checkpoint is the one the whole run makes")

    def test_a_finished_run_leaves_no_resume_state_whatever_resume_every_is(self) -> None:
        for name, args in (("end_every2", ARGS), ("end_default", NO_EVERY)):
            out = train(name, *args, *FULL, cache=False)
            self.assertTrue(summary(out)["complete"])
            self.assertFalse((out / "resume.pt").exists(), name)
            self.assertFalse((out / "resume.pt.pending").exists(), name)

    def test_a_run_that_resumed_to_the_end_leaves_none_either(self) -> None:
        out = self.dies_in_the_final_save("cut_then_done", *ARGS, *FULL)
        self.assertTrue((out / "resume.pt").is_file())
        (out / "resume.pt.pending").write_bytes(b"a resume state that was being written when the machine died")
        self.assertEqual(resume("cut_then_done", *ARGS, *FULL), 0)
        self.assertFalse((out / "resume.pt").exists())
        self.assertFalse((out / "resume.pt.pending").exists(), "what a killed save of the resume state left is not kept either")

    def test_a_run_stopped_by_max_steps_keeps_its_resume_state(self) -> None:
        out = train("stopped_early", *NO_EVERY, *FULL, "--max-steps", "5", cache=False)
        self.assertTrue((out / "resume.pt").is_file())
        self.assertFalse((out / "final").exists())


class DtypeFlag(unittest.TestCase):
    """``--dtype`` decides the precision of the model (a T4 has no native bf16), and it is part of the record of the run."""

    def stored_dtype(self, final: Path):
        with _STATE["safe_open"](str(final / "model.safetensors"), "pt") as handle:
            return handle.get_tensor(next(iter(handle.keys()))).dtype

    def test_the_flag_reaches_the_model_and_the_checkpoint(self) -> None:
        import torch

        MODEL = _STATE["MODEL"]
        seen = []
        real = MODEL.JointModel.load

        def spy(*args, **kwargs):
            seen.append(kwargs.get("dtype"))
            return real(*args, **kwargs)

        with mock.patch.object(MODEL.JointModel, "load", spy):
            fp32 = train("dtype_fp32", *SHORT, *FULL, "--dtype", "fp32", cache=False)
            bf16 = train("dtype_bf16", *SHORT, *FULL, "--dtype", "bf16", cache=False)
            auto = train("dtype_auto", *SHORT, *FULL, cache=False)
        self.assertEqual(seen, [torch.float32, torch.bfloat16, torch.float32], "auto is fp32 on a CPU, bf16 on a GPU")
        self.assertEqual([config(run / "final")["training"]["dtype"] for run in (fp32, bf16, auto)], ["fp32", "bf16", "fp32"])
        self.assertEqual(self.stored_dtype(fp32 / "final"), torch.float32)
        self.assertEqual(self.stored_dtype(bf16 / "final"), torch.bfloat16)

    def test_the_parser_knows_the_precisions_and_nothing_else(self) -> None:
        parser = _STATE["TRAIN"].build_parser()
        base = ["--targets", "t", "--checkpoint", "c", "--out", "o"]
        for name in ("auto", "bf16", "fp32", "fp16"):
            self.assertEqual(parser.parse_args(base + ["--dtype", name]).dtype, name)
        self.assertEqual(parser.parse_args(base).dtype, "auto")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(base + ["--dtype", "int8"])

    def test_auto_is_bf16_on_a_gpu_and_fp32_on_a_cpu_and_an_explicit_precision_is_kept(self) -> None:
        """The flow tests run on a CPU, so only this call sees the GPU half of ``auto``: a rented B200 trains in bf16, not in fp32 at twice the memory."""
        resolve = _STATE["TRAIN"].resolve_dtype
        self.assertEqual(resolve("auto", True), "bf16")
        self.assertEqual(resolve("auto", False), "fp32")
        for name in ("bf16", "fp32", "fp16"):
            for cuda in (True, False):
                self.assertEqual(resolve(name, cuda), name, (name, cuda))

    def test_fp16_training_is_refused_because_nothing_scales_the_loss(self) -> None:
        """A gradient that underflows to zero in fp16 is a finite number: the check for nonfinite values never sees it, and the run learns less than it looks."""
        for extra in ((), ("--dry-run",)):
            with self.assertRaises(ValueError, msg=extra) as caught:
                train("dtype_fp16", *SHORT, *FULL, "--dtype", "fp16", *extra, cache=False)
            text = str(caught.exception)
            for word in ("--dtype fp16", "loss scal", "underflow", "bf16", "fp32"):
                self.assertIn(word, text)
        self.assertFalse((_STATE["root"] / "runs" / "dtype_fp16").exists(), "refused before a run folder, a model or a log exists")

    def test_bf16_and_fp32_are_not_affected_by_the_refusal(self) -> None:
        for name in ("bf16", "fp32"):
            train(f"dtype_ok_{name}", *SHORT, *FULL, "--dtype", name, "--dry-run", cache=False)       # the plan prints, nothing raises

    def test_a_resume_cannot_change_the_dtype(self) -> None:
        train("dtype_resume", *ARGS, *FULL, "--max-steps", "3", "--dtype", "fp32", cache=False)
        with self.assertRaises(ValueError) as caught:
            resume("dtype_resume", *ARGS, *FULL, "--dtype", "bf16")
        self.assertIn("dtype", str(caught.exception))


class DroppedTooLong(unittest.TestCase):
    """The requests the length limit leaves out are a number the person who reads the run must see (every request here has 104 tokens)."""

    def test_the_summary_and_the_checkpoint_count_them_by_request_and_by_example(self) -> None:
        out = train("dropped", *THREE, *FULL, "--max-length", "50", cache=False)
        done = summary(out)
        self.assertEqual(done["dropped_too_long"], 24, "three steps of eight requests")
        self.assertEqual(done["dropped_too_long_examples"], 8, "the same eight examples, three times")
        self.assertEqual(done["evaluation"]["plain"]["skipped_too_long"], 6, "all six dev examples")
        self.assertEqual(config(out / "final")["training"]["dropped_too_long"], 24)
        self.assertEqual([e["dropped_too_long"] for e in events(out) if e["kind"] == "training_step"], [8, 8, 8])

    def test_a_resumed_run_counts_what_the_part_before_the_stop_dropped(self) -> None:
        limit = ("--max-length", "120")                  # the examples with 144 tokens go: two of the eight, once per epoch
        whole = train("mixed_whole", *ARGS, *FULL, *limit, targets=_STATE["mixed"], cache=False)
        cut = train("mixed_cut", *ARGS, *FULL, *limit, "--max-steps", "5", targets=_STATE["mixed"], cache=False)
        self.assertEqual(resume("mixed_cut", *ARGS, *FULL, *limit, targets=_STATE["mixed"]), 0)
        for out in (whole, cut):
            done = summary(out)
            self.assertEqual((done["dropped_too_long"], done["dropped_too_long_examples"]), (12, 2), out.name)
            self.assertEqual(done["evaluation"]["plain"]["skipped_too_long"], 2, "two of the six dev examples are long")
            self.assertEqual(config(out / "final")["calibration"]["skipped_too_long"], 2)

    def test_a_run_that_drops_nothing_says_zero(self) -> None:
        out = train("dropped_none", *SHORT, *FULL)
        self.assertEqual((summary(out)["dropped_too_long"], summary(out)["dropped_too_long_examples"]), (0, 0))

    def test_the_calibration_record_says_how_many_dev_examples_were_left_out(self) -> None:
        plain = {"examples": 5, "questions": 5, "skipped_too_long": 2, "temperature": {"all": 1.2, "counts": {"choice": 5}}}
        fields = _STATE["TEMPERATURE"].calibration_fields(plain)
        self.assertEqual(fields["calibration"]["skipped_too_long"], 2)


class NoticesTravel(unittest.TestCase):
    """A derived checkpoint is distributed with the LICENSE and the NOTICE of the one it comes from, and says what was changed (Apache-2.0, 4(a), 4(b), 4(d))."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.first = train("notice_a", *SHORT, *FULL, "--seed", "1")
        cls.second = train("notice_b", *SHORT, *FULL, "--seed", "2")

    def test_final_has_the_license_and_the_notice_with_a_modifications_line(self) -> None:
        final, base = self.first / "final", _STATE["base"]
        self.assertEqual((final / "LICENSE").read_bytes(), (base / "LICENSE").read_bytes())
        notice = (final / "NOTICE").read_text(encoding="utf-8")
        self.assertTrue(notice.startswith((base / "NOTICE").read_text(encoding="utf-8").rstrip("\n")), "the original notice comes first, unchanged")
        last = [line for line in notice.splitlines() if line.strip()][-1]
        self.assertTrue(last.startswith("Modifications:"), last)
        for word in ("decide.train.train", "full", "2 steps"):
            self.assertIn(word, last)
        self.assertFalse((final / "README.md").exists(), "a model card describes one model; it is not copied")

    def test_the_soup_has_them_too(self) -> None:
        soup = _STATE["root"] / "soup_notice"
        with contextlib.redirect_stdout(io.StringIO()):
            _STATE["SOUP"].main(["--inputs", str(self.first / "final"), str(self.second / "final"), "--out", str(soup)])
        self.assertEqual((soup / "LICENSE").read_bytes(), (_STATE["base"] / "LICENSE").read_bytes())
        notice = (soup / "NOTICE").read_text(encoding="utf-8")
        self.assertTrue(notice.startswith((self.first / "final" / "NOTICE").read_text(encoding="utf-8").rstrip("\n")),
                        "the notice of the first input, with its own Modifications line, comes first")
        last = [line for line in notice.splitlines() if line.strip()][-1]
        self.assertTrue(last.startswith("Modifications:"), last)
        for word in ("decide.train.soup", "2 checkpoints", "average"):
            self.assertIn(word, last)

    def test_the_notice_of_the_directory_has_the_flash_notice(self) -> None:
        text = (ROOT / "decide" / "train" / "NOTICE").read_text(encoding="utf-8")
        for needle in ("Original NOTICE of pplx-decider-flash-v1.2", "Derived from Qwen/Qwen3.5-4B, revision",
                       "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, under Apache License 2.0."):
            self.assertIn(needle, text)


class SoupWorkFolder(unittest.TestCase):
    """``<out>.souping`` is deleted by the next soup only when the soup made it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.a = train("soup_a", *SHORT, *FULL, "--seed", "1") / "final"
        cls.b = train("soup_b", *SHORT, *FULL, "--seed", "2") / "final"

    def test_an_interrupted_soup_leaves_a_folder_the_next_run_replaces(self) -> None:
        SOUP = _STATE["SOUP"]
        out = _STATE["root"] / "soup_cut"
        work = out.with_name(out.name + ".souping")
        with mock.patch("safetensors.torch.save_file", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                SOUP.average_checkpoints([self.a, self.b], out)
        self.assertTrue(work.is_dir())
        self.assertFalse(out.exists())
        SOUP.average_checkpoints([self.a, self.b], out)
        self.assertFalse(work.exists())
        self.assertTrue((out / "readout.safetensors").is_file())
        self.assertEqual(sorted(p.name for p in out.glob(".*")), [], "the marker of the work folder does not travel into the soup")

    def test_a_folder_the_soup_did_not_make_is_not_deleted(self) -> None:
        SOUP = _STATE["SOUP"]
        out = _STATE["root"] / "soup_alien"
        work = out.with_name(out.name + ".souping")
        work.mkdir()
        (work / "mine.txt").write_text("keep me", encoding="utf-8")
        with self.assertRaises(SOUP.SoupError) as caught:
            SOUP.average_checkpoints([self.a, self.b], out)
        self.assertIn(str(work), str(caught.exception))
        self.assertEqual((work / "mine.txt").read_text(encoding="utf-8"), "keep me")
        self.assertFalse(out.exists())


class TheItemIgnoresAReasoningThatCannotBeShown(unittest.TestCase):
    """A state that already holds a ``Raciocínio:`` block cannot take a second one: the example trains without it instead of stopping the run."""

    def example(self, state):
        from decide.train.data import Example

        return Example(example_id="t:1", source="t", split="train", lang="en", request=choice_request("Why?", ["a", "b", "c"], keys=["x", "y", "z"], state=state),
                       gold_index={"q1": 0}, combined={"q1": None}, teachers=(), reasoning="a thought")

    def test_the_reasoning_is_left_out_when_the_state_has_the_block(self) -> None:
        from decide.train.batching import ReasoningPolicy, make_item
        from decide.train.codes import ANSWER_CODES
        from decide.train.protocol import RequestEncoder

        encoder = RequestEncoder(FakeTokenizer(), ANSWER_CODES)
        always = ReasoningPolicy("always")
        self.assertTrue(make_item(self.example("clean state"), encoder, seed=1, reasoning=always).reasoning)
        item = make_item(self.example("s\n\nRaciocínio:\nalready here"), encoder, seed=1, reasoning=always)
        self.assertFalse(item.reasoning)


class DocsSayWhatTheCodeDoes(unittest.TestCase):
    def test_the_package_doc_is_the_plan_of_adr_0004(self) -> None:
        import decide.train as package

        self.assertNotIn("distillation of a teacher ensemble", package.__doc__.lower(), "ADR 0004, decision 1: no ensemble; the target is the gold label mixed with one teacher")
        self.assertIn("ADR 0004", package.__doc__)

    def test_the_model_doc_says_which_layers_lora_adapts_and_why(self) -> None:
        from decide.train import model

        for word in ("DeltaNet", "in_proj_qkv", "in_proj_a", "decide/PLAN.md"):
            self.assertIn(word, model.__doc__)


if __name__ == "__main__":
    unittest.main()
