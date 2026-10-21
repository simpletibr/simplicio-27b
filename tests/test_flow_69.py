"""Flows of decide/train (issue #69, F4) on a tiny random Qwen3.5 and a stand-in tokenizer.

train (full weights and LoRA) -> resume -> soup -> temperature, on a CPU, offline, in seconds.  The backbone is four layers
wide 64 with a 600-token vocabulary (``decide_b_support.make_tiny_checkpoint``) and the corpus is eight training and six dev
questions whose right option is named in the state, so a few dozen steps can learn them.

What this cannot say: whether the 27B model trains on a GPU (throughput, memory, DDP across eight cards, LoRA of rank 128 on
27B parameters).  Those are listed as UNVERIFIED in the PR; the code paths are the same, the sizes are not.

The tests skip cleanly when torch, transformers or pyarrow is missing (LoRA ones when peft is).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from decide_b_support import HAVE_PEFT, HAVE_PYARROW, HAVE_TRANSFORMERS, FakeTokenizer, make_targets_dataset, make_tiny_checkpoint

_STATE: Dict[str, Any] = {}

#: The cross-entropy of a one-hot gold mixed 50/50 with a teacher that puts 0.8 on it is at best the entropy of the mixed target
#: (0.9, 0.05, 0.05): a model that has learned the training questions cannot go lower.
FLOOR = 0.394

#: Eight training examples in one step per epoch, 25 epochs: enough to learn them with these rates.
RUN = ("--batch-rows", "8", "--token-budget", "8192", "--max-length", "1024", "--cpu-threads", "2", "--device", "cpu", "--no-permute")
COMMON = ("--epochs", "25", "--effective-batch", "8", *RUN)
SHORT = ("--epochs", "1", "--effective-batch", "4", *RUN)                 # two steps
THREE = ("--epochs", "3", "--effective-batch", "8", *RUN)                 # three steps
FULL = ("--mode", "full", "--lr", "5e-3")
LORA = ("--mode", "lora", "--lora-rank", "8", "--lr", "2e-2")


def setUpModule() -> None:
    if not (HAVE_TRANSFORMERS and HAVE_PYARROW):
        raise unittest.SkipTest("torch, transformers and pyarrow are needed")
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    tokenizer = FakeTokenizer()
    make_tiny_checkpoint(root / "base", tokenizer)
    make_targets_dataset(root / "data", train=8, dev=6, traces_for=range(14))
    from safetensors.torch import load_file

    from decide.train import model as MODEL
    from decide.train import soup as SOUP
    from decide.train import temperature as TEMPERATURE
    from decide.train import train as TRAIN

    _STATE.update(tmp=tmp, root=root, base=root / "base", targets=root / "data" / "targets", runs={}, load_file=load_file,
                  MODEL=MODEL, SOUP=SOUP, TEMPERATURE=TEMPERATURE, TRAIN=TRAIN)


def tearDownModule() -> None:
    if "tmp" in _STATE:
        _STATE["tmp"].cleanup()
    _STATE.clear()


def train(name: str, *extra: str, cache: bool = True, keep_env: bool = False) -> Path:
    """Run ``decide.train.train`` in this process on the shared tiny checkpoint and targets; the run folder is ``runs/<name>``.

    Unless ``keep_env``, ``DECIDE_EVENTS`` is cleared first, so a test about something else does not depend on how an earlier run left it."""
    runs = _STATE["runs"]
    if cache and name in runs:
        return runs[name]
    if not keep_env:
        os.environ.pop("DECIDE_EVENTS", None)
    out = _STATE["root"] / "runs" / name
    argv = ["--targets", str(_STATE["targets"]), "--checkpoint", str(_STATE["base"]), "--out", str(out), *extra]
    with contextlib.redirect_stdout(io.StringIO()):
        rc = _STATE["TRAIN"].main(argv, tokenizer=FakeTokenizer())
    assert rc == 0
    runs[name] = out
    return out


def events(out: Path) -> List[Dict[str, Any]]:
    path = out / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


def steps(out: Path) -> List[Dict[str, Any]]:
    return [e for e in events(out) if e["kind"] == "training_step"]


def losses(out: Path) -> List[float]:
    return [e["loss"] for e in steps(out)]


def config(directory: Path) -> Dict[str, Any]:
    return json.loads((directory / "decision_config.json").read_text(encoding="utf-8"))


def weights(directory: Path) -> Dict[str, Any]:
    tensors: Dict[str, Any] = {}
    for file in sorted(directory.glob("*.safetensors")):
        tensors.update({f"{file.name}:{key}": value for key, value in _STATE["load_file"](str(file)).items()})
    return tensors


def largest_difference(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    assert set(a) == set(b), sorted(set(a) ^ set(b))[:5]
    return max(float((a[key] - b[key]).abs().max()) for key in a)


class EventLog(unittest.TestCase):
    """Each run writes its events into its own folder, however many runs one process makes (a notebook trains several seeds in one kernel)."""

    def setUp(self) -> None:
        self.addCleanup(os.environ.pop, "DECIDE_EVENTS", None)       # keep a failure here from leaking into the other tests
        os.environ.pop("DECIDE_EVENTS", None)

    def test_two_runs_in_one_process_keep_their_own_events(self) -> None:
        first = train("events_a", *SHORT, *FULL, cache=False, keep_env=True)
        second = train("events_b", *SHORT, *FULL, "--seed", "7", cache=False, keep_env=True)
        for out in (first, second):
            kinds = [event["kind"] for event in events(out)]
            self.assertEqual(kinds.count("training_started"), 1, out.name)
            self.assertEqual(kinds.count("training_step"), 2, out.name)
            self.assertEqual(kinds.count("checkpoint_saved"), 1, out.name)
            self.assertEqual(kinds.count("training_finished"), 1, out.name)
        self.assertNotIn("DECIDE_EVENTS", os.environ, "the path of one run must not stay in the environment of the next")

    def test_a_log_the_caller_asked_for_wins_and_is_left_in_the_environment(self) -> None:
        elsewhere = _STATE["root"] / "elsewhere" / "log.jsonl"
        os.environ["DECIDE_EVENTS"] = str(elsewhere)
        out = train("events_c", *SHORT, *FULL, cache=False, keep_env=True)
        self.assertFalse((out / "events.jsonl").exists())
        self.assertEqual(sum(json.loads(line)["kind"] == "training_step" for line in elsewhere.read_text(encoding="utf-8").splitlines()), 2)
        self.assertEqual(os.environ["DECIDE_EVENTS"], str(elsewhere))


class FullFineTuneFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.pop("DECIDE_EVENTS", None)
        cls.out = train("full", *COMMON, *FULL)

    def test_the_loss_falls_to_the_entropy_of_the_target(self) -> None:
        values = losses(self.out)
        self.assertEqual(len(values), 25)
        self.assertGreater(sum(values[:3]) / 3, 1.0, "a random model starts near ln(3)")
        self.assertLess(sum(values[-3:]) / 3, FLOOR + 0.05, "the eight training questions are learned")

    def test_the_run_folder_and_the_event_log(self) -> None:
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["events.jsonl", "final", "run.json", "summary.json"])
        kinds = [event["kind"] for event in events(self.out)]
        self.assertEqual(kinds, ["training_started"] + ["training_step"] * 25 + ["evaluation", "checkpoint_saved", "training_finished"])
        summary = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["complete"])
        self.assertEqual((summary["steps"], summary["planned_steps"]), (25, 25))

    def test_the_checkpoint_loads_and_carries_the_fit(self) -> None:
        final = self.out / "final"
        MODEL = _STATE["MODEL"]
        cfg = MODEL.read_config(final)
        summary = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(cfg["temperature"], summary["temperature"], places=9)
        self.assertFalse(cfg["temperature_is_placeholder"])
        self.assertEqual(cfg["calibration"]["split"], "dev")
        self.assertEqual(cfg["calibration"]["questions"], 6)
        self.assertEqual(cfg["temperature_fallback"], ["choice"], "six questions are fewer than the 50 a type needs for its own temperature")
        self.assertEqual(cfg["training"]["method"], "full")
        self.assertEqual(cfg["training"]["steps"], 25)
        self.assertIn("run_fingerprint", cfg["training"])
        model = MODEL.JointModel.load(final, mode="none")
        self.assertAlmostEqual(model.temperature, cfg["temperature"], places=9)
        self.assertEqual(sorted(p.name for p in final.glob("*.safetensors")), ["model.safetensors", "readout.safetensors"])

    def test_the_weights_moved_and_the_base_checkpoint_did_not(self) -> None:
        base, final = weights(_STATE["base"]), weights(self.out / "final")
        self.assertEqual(set(base), set(final))
        self.assertGreater(largest_difference(base, final), 1e-3)
        self.assertGreater(float((base["readout.safetensors:weight"] - final["readout.safetensors:weight"]).abs().max()), 1e-3, "the readout trains too")
        self.assertEqual(config(_STATE["base"])["temperature"], 1.0)

    def test_a_finished_run_leaves_no_resume_state_and_refuses_to_start_over(self) -> None:
        self.assertFalse((self.out / "resume.pt").exists())
        with self.assertRaises(ValueError) as caught, contextlib.redirect_stdout(io.StringIO()):
            _STATE["TRAIN"].main(["--targets", str(_STATE["targets"]), "--checkpoint", str(_STATE["base"]), "--out", str(self.out), *COMMON, *FULL],
                                 tokenizer=FakeTokenizer())
        self.assertIn("--resume", str(caught.exception))


@unittest.skipUnless(HAVE_PEFT, "peft is not installed")
class LoraFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.pop("DECIDE_EVENTS", None)
        cls.out = train("lora", *COMMON, *LORA)

    def test_the_loss_falls_to_the_entropy_of_the_target(self) -> None:
        values = losses(self.out)
        self.assertGreater(sum(values[:3]) / 3, 1.0)
        self.assertLess(sum(values[-3:]) / 3, FLOOR + 0.05)

    def test_only_the_adapters_and_the_readout_train(self) -> None:
        full = events(train("full", *COMMON, *FULL))[0]["trainable_parameters"]
        lora = events(self.out)[0]["trainable_parameters"]
        self.assertLess(lora, full / 3, f"rank 8 trains {lora} parameters against {full} for the full weights")

    def test_the_saved_checkpoint_has_the_adapters_folded_in(self) -> None:
        final = weights(self.out / "final")
        self.assertFalse([key for key in final if "lora" in key.lower()], "no adapter tensors in the checkpoint")
        base = weights(_STATE["base"])
        self.assertEqual(set(base), set(final), "the layout of the input checkpoint")
        moved = sorted(key for key in base if float((base[key] - final[key]).abs().max()) > 1e-6)
        self.assertTrue(any("q_proj" in key or "in_proj_qkv" in key or "down_proj" in key for key in moved), moved[:5])
        self.assertTrue(any("embed_tokens" in key for key in base) and not any("embed_tokens" in key for key in moved), "embeddings are frozen")
        self.assertEqual(config(self.out / "final")["training"]["method"], "lora")
        _STATE["MODEL"].JointModel.load(self.out / "final", mode="none")


class ResumeFlow(unittest.TestCase):
    """A run stopped by ``--max-steps`` (or killed) and continued with ``--resume`` ends where an uninterrupted run ends."""

    ARGS = ("--epochs", "6", "--effective-batch", "4", "--batch-rows", "4", "--token-budget", "4096", "--max-length", "1024",
            "--cpu-threads", "2", "--device", "cpu", "--resume-every", "2")

    def stopped_and_resumed(self, name: str, *mode: str) -> Path:
        out = train(name, *self.ARGS, *mode, "--max-steps", "5", cache=False)
        self.assertEqual(len(steps(out)), 5)
        self.assertTrue((out / "resume.pt").is_file())
        self.assertFalse((out / "final").exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(_STATE["TRAIN"].main(["--targets", str(_STATE["targets"]), "--checkpoint", str(_STATE["base"]), "--out", str(out), *self.ARGS,
                                                   *mode, "--resume"], tokenizer=FakeTokenizer()), 0)
        return out

    def assertSameRun(self, whole: Path, resumed: Path) -> None:
        self.assertEqual([e["step"] for e in steps(resumed)], list(range(1, 13)))
        for a, b in zip(losses(whole), losses(resumed)):
            self.assertAlmostEqual(a, b, places=6)
        self.assertLess(largest_difference(weights(whole / "final"), weights(resumed / "final")), 1e-6)
        self.assertAlmostEqual(config(whole / "final")["temperature"], config(resumed / "final")["temperature"], places=5)

    def test_full_weights(self) -> None:
        whole = train("resume_full_whole", *self.ARGS, *FULL, cache=False)
        self.assertEqual(len(steps(whole)), 12)
        self.assertSameRun(whole, self.stopped_and_resumed("resume_full_cut", *FULL))

    @unittest.skipUnless(HAVE_PEFT, "peft is not installed")
    def test_lora_with_dropout(self) -> None:
        # the dropout masks must depend on (seed, step) only, not on how many steps this process has already drawn
        mode = ("--mode", "lora", "--lora-rank", "8", "--lr", "2e-2", "--lora-dropout", "0.2")
        whole = train("resume_lora_whole", *self.ARGS, *mode, cache=False)
        self.assertSameRun(whole, self.stopped_and_resumed("resume_lora_cut", *mode))

    def test_a_resume_that_does_not_match_the_run_is_refused(self) -> None:
        out = train("resume_guard", *self.ARGS, *FULL, "--max-steps", "3", cache=False)
        base = ["--targets", str(_STATE["targets"]), "--checkpoint", str(_STATE["base"]), "--out", str(out)]
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError) as caught:
                _STATE["TRAIN"].main(base + list(self.ARGS) + ["--mode", "full", "--lr", "1e-3", "--resume"], tokenizer=FakeTokenizer())
            self.assertIn("lr", str(caught.exception))
            with self.assertRaises(ValueError):
                _STATE["TRAIN"].main(base + list(self.ARGS) + list(FULL), tokenizer=FakeTokenizer())          # no --resume: the folder holds a run
            (out / "resume.pt").unlink()
            with self.assertRaises(ValueError):
                _STATE["TRAIN"].main(base + list(self.ARGS) + list(FULL) + ["--resume"], tokenizer=FakeTokenizer())


class SoupAndTemperatureFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.pop("DECIDE_EVENTS", None)
        cls.first = train("full", *COMMON, *FULL)
        cls.second = train("full_seed2", *COMMON[:-1], *FULL, "--seed", "2")      # another seed, and the options shuffled: another path to other weights
        cls.soup = _STATE["root"] / "soup"
        with contextlib.redirect_stdout(io.StringIO()):
            _STATE["SOUP"].main(["--inputs", str(cls.first / "final"), str(cls.second / "final"), "--out", str(cls.soup)])

    def test_the_soup_is_the_mean_and_its_temperature_is_a_placeholder(self) -> None:
        a, b, mixed = weights(self.first / "final"), weights(self.second / "final"), weights(self.soup)
        self.assertGreater(largest_difference(a, b), 1e-3, "two seeds end in different weights")
        for key in ("readout.safetensors:weight", next(k for k in a if "q_proj" in k)):
            self.assertLess(float((mixed[key] - (a[key] + b[key]) / 2).abs().max()), 1e-6, key)
        cfg = config(self.soup)
        self.assertTrue(cfg["soup"]["temperature_is_placeholder"])
        self.assertAlmostEqual(cfg["temperature"], (config(self.first / "final")["temperature"] + config(self.second / "final")["temperature"]) / 2, places=9)
        self.assertNotIn("temperature_is_placeholder", cfg)
        self.assertNotIn("calibration", cfg, "the fit of one run does not describe the average")
        self.assertNotIn("training", cfg)
        self.assertNotIn("temperature_by_type", cfg)

    def run_temperature(self, checkpoint: Path, *extra: str) -> Dict[str, Any]:
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = _STATE["TEMPERATURE"].main(["--checkpoint", str(checkpoint), "--targets", str(_STATE["targets"]), "--cpu-threads", "2", "--device", "cpu", *extra],
                                            tokenizer=FakeTokenizer())
        self.assertEqual(rc, 0)
        return json.loads(printed.getvalue())

    def test_calibrating_the_soup_writes_the_fit_and_nothing_else(self) -> None:
        soup = _STATE["root"] / "soup_to_fit"
        shutil.copytree(self.soup, soup)
        before = {p.name: p.read_bytes() for p in soup.iterdir() if p.is_file() and p.name != "decision_config.json"}
        report = self.run_temperature(soup, "--write")
        self.assertTrue(report["written"])
        cfg = config(soup)
        self.assertFalse(cfg["temperature_is_placeholder"])
        self.assertEqual(cfg["temperature"], report["fields"]["temperature"])
        self.assertEqual(cfg["calibration"], {"split": "dev", "examples": 6, "questions": 6, "questions_by_type": {"choice": 6}, "with_reasoning": {"examples": 6, "questions": 6}})
        self.assertIn("temperature_with_reasoning", cfg, "the dev examples carry thinker traces")
        self.assertFalse(cfg["soup"]["temperature_is_placeholder"])
        self.assertEqual({p.name: p.read_bytes() for p in soup.iterdir() if p.is_file() and p.name != "decision_config.json"}, before, "weights and tokenizer files untouched")

    def test_without_write_the_checkpoint_is_not_modified(self) -> None:
        soup = _STATE["root"] / "soup_report_only"
        shutil.copytree(self.soup, soup)
        before = (soup / "decision_config.json").read_bytes()
        report = self.run_temperature(soup, "--report", str(_STATE["root"] / "report.json"))
        self.assertFalse(report["written"])
        self.assertEqual((soup / "decision_config.json").read_bytes(), before)
        self.assertEqual(json.loads((_STATE["root"] / "report.json").read_text(encoding="utf-8"))["fields"], report["fields"])

    def test_the_command_repeats_the_evaluation_a_run_makes_at_its_end(self) -> None:
        report = self.run_temperature(self.first / "final")
        at_the_end = next(e for e in events(self.first) if e["kind"] == "evaluation")["plain"]
        for key in ("nll", "accuracy", "brier"):
            self.assertAlmostEqual(report["plain"]["raw"][key], at_the_end["raw"][key], places=4, msg=key)
        self.assertEqual(report["fields"]["temperature_fallback"], config(self.first / "final")["temperature_fallback"])
        self.assertAlmostEqual(report["fields"]["temperature"], config(self.first / "final")["temperature"], delta=1e-3 * config(self.first / "final")["temperature"])


class ReasoningFlow(unittest.TestCase):
    """The thinker's reasoning (SCHEMA.md, section 10) is shown to the model during training when the policy says so."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.pop("DECIDE_EVENTS", None)

    def run_with(self, name: str, *policy: str) -> Path:
        return train(name, *THREE, *FULL, *policy)

    def test_off_never_shows_it_and_always_shows_it_to_every_example(self) -> None:
        off = self.run_with("reason_off")
        always = self.run_with("reason_always", "--reasoning", "always")
        self.assertEqual([e["with_reasoning"] for e in steps(off)], [0, 0, 0])
        self.assertEqual([e["with_reasoning"] for e in steps(always)], [8, 8, 8])
        self.assertEqual([e["reasoning_fell_back"] for e in steps(always)], [0, 0, 0])
        self.assertNotIn("temperature_with_reasoning", config(off / "final"))
        self.assertIn("temperature_with_reasoning", config(always / "final"))
        self.assertGreater(config(always / "final")["temperature_with_reasoning"], 0)

    def test_mixed_shows_it_to_some_and_depends_only_on_the_seed(self) -> None:
        mixed = self.run_with("reason_mixed", "--reasoning", "mixed", "--reasoning-fraction", "0.5")
        counts = [e["with_reasoning"] for e in steps(mixed)]
        self.assertTrue(all(0 < n < 8 for n in counts), counts)
        again = train("reason_mixed_again", *THREE, *FULL, "--reasoning", "mixed", "--reasoning-fraction", "0.5", cache=False)
        self.assertEqual([e["with_reasoning"] for e in steps(again)], counts)
        self.assertEqual(losses(again), losses(mixed), "same seed, same data order, same draws, same run")

    def test_the_reasoning_changes_what_the_model_sees(self) -> None:
        off, always = self.run_with("reason_off"), self.run_with("reason_always", "--reasoning", "always")
        self.assertGreater(steps(always)[0]["input_tokens"], steps(off)[0]["input_tokens"])
        self.assertNotEqual(losses(off), losses(always))


class DryRun(unittest.TestCase):
    def test_the_plan_is_printed_and_nothing_is_loaded_or_written(self) -> None:
        out = _STATE["root"] / "runs" / "dry"
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = _STATE["TRAIN"].main(["--targets", str(_STATE["targets"]), "--checkpoint", str(_STATE["base"]), "--out", str(out), *COMMON, *FULL, "--dry-run"])
        self.assertEqual(rc, 0)
        plan = json.loads(printed.getvalue())
        self.assertEqual((plan["train_examples"], plan["dev_examples"], plan["steps"], plan["mode"]), (8, 6, 25, "full"))
        self.assertEqual((plan["train_with_reasoning"], plan["dev_with_reasoning"]), (8, 6))
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
