"""Flows of decide/rental (issue #71, F6): the whole rented day, rehearsed on a CPU in a few minutes.

The Hub is a folder (``LocalHub``), the kit is a stand-in that answers the proxy and the public requests by a fixed rule, the
thinker is a stand-in, and the trainer is the real ``decide.train`` run in this process on a tiny random Qwen3.5 with a stand-in
tokenizer.  What this exercises is the orchestration: phases in order, state and resume, the gates (labels, GT, G4), the backup
to the work repo and the restore on a new machine, the upload to the private weights repo, and what the child processes can see.

What it cannot say: anything about a GPU or the real 27B model (time, memory, DDP, the real kit loading 52 GB), the real Hub, or
Modal.  Those stay UNVERIFIED in the PR.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import shutil
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from typing import Any, Dict, List

from decide_b_support import HAVE_PEFT, HAVE_PYARROW, HAVE_TRANSFORMERS, HAVE_YAML, FakeTokenizer, make_targets_dataset, make_tiny_checkpoint

TOKEN = "hf_" + "z" * 30


def tiny_profile(**train: Any):
    from decide.rental.profile import profile_from_dict

    options = {"mode": "lora" if HAVE_PEFT else "full", "epochs": 3, "effective_batch": 8, "batch_rows": 8, "token_budget": 8192, "max_length": 1024,
               "cpu_threads": 2, "no_permute": True, "alpha": 0.5, "warmup_ratio": 0.2, "reasoning": "off", "resume_every": 2,
               "lr": 2e-2 if HAVE_PEFT else 5e-3}
    if HAVE_PEFT:
        options.update(lora_rank=8, lora_alpha=8)
    options.update(train)
    return profile_from_dict({
        "version": 1, "name": "tiny_rehearsal", "title": "Rehearsal on a CPU with a random 4-layer model", "plan": {"hours": "0", "usd": "0"},
        "hardware": {"gpu": "B200", "gpus": 1, "host_ram_gb": 1, "disk_gb": 1}, "workload": {"examples": 14, "tokens_per_example": 100, "seeds": 1},
        "train": options, "train_with_traces": {"reasoning": "mixed", "reasoning_fraction": 0.5}, "thinker": {"examples": 14, "mean_tokens": 50}})


def tiny_config():
    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "decide" / "config.rental.yaml").read_text(encoding="utf-8"))["rental"]
    cfg["hub"].update(work_repo="me/day", data_repo="me/data", weights_repo="me/weights")
    return cfg


class InProcessExecutor:
    """What the day would send to a subprocess, run in this process with the tiny tokenizer; remembers every call."""

    def __init__(self) -> None:
        self.calls: List[List[str]] = []
        self.envs: List[Dict[str, str]] = []

    def modules(self) -> List[str]:
        return [argv[argv.index("-m") + 1] for argv in self.calls]

    def run(self, argv, *, log_path, env) -> int:
        self.calls.append(list(argv))
        self.envs.append(dict(env))
        module = argv[argv.index("-m") + 1]
        main = importlib.import_module(module).main
        extra = {"tokenizer": FakeTokenizer()} if module in ("decide.train.train", "decide.train.temperature") else {}
        out = io.StringIO()
        code = 0
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                code = main(argv[argv.index(module) + 1:], **extra) or 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        except Exception:                                  # noqa: BLE001 - a failing command is a return code, as with a subprocess
            out.write(traceback.format_exc())
            code = 1
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(out.getvalue(), encoding="utf-8")
        return code


class FakeKit:
    """Answers a requests file by a fixed rule: the first ``rates[model]`` of every four examples get the gold option."""

    def __init__(self, rates=None, fail_on=()) -> None:
        self.rates = rates or {"baseline": 2, "new": 3}
        self.fail_on = set(fail_on)
        self.calls: List[Dict[str, Any]] = []

    def run(self, model, requests, out, *, budget=None) -> Dict[str, Any]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from decide.data import schema as S

        self.calls.append({"model": model.name, "path": model.path, "requests": str(requests), "out": str(out)})
        if model.name in self.fail_on:
            self.fail_on.discard(model.name)
            raise RuntimeError(f"the kit lost the GPU while running {model.name}")
        rows, count = [], 0
        for line in Path(requests).read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            count += 1
            index = int(item["example_id"].rsplit(":", 1)[1])
            for qid, key in (item.get("gold") or {}).items():
                keys = S.option_keys(item["request"]["questions"][qid])
                right = index % 4 < self.rates.get(model.name, 2)
                favoured = keys.index(key) if right else (keys.index(key) + 1) % len(keys)
                probs = [0.7 if i == favoured else 0.3 / (len(keys) - 1) for i in range(len(keys))]
                rows.append({"run_id": item["example_id"], "question_id": qid, "probs": probs, "status": "ok"})
        Path(out).mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows or [{"run_id": "", "question_id": "", "probs": [0.0], "status": "ok"}]), str(Path(out) / "results.parquet"))
        return {"rows": count, "total": count, "tokens": 10 * count, "seconds": 1.5, "status": "complete"}


def build_world(root: Path, *, label_every=1):
    """A LocalHub holding the private data repo and the start checkpoint; returns (hub, proxy_requests_file)."""
    from decide.data import schema as S
    from decide.rental import gates as G
    from decide.rental import hub as H

    data = root / "make"
    make_targets_dataset(data, train=8, dev=6)
    if label_every != 1:
        part = next((data / "labels").rglob("*.parquet"))
        rows = list(S.read_teacher(part))
        part.unlink()
        S.write_teacher_part(part, rows[::label_every])
    proxy_rows = []
    for source in ("alpha", "beta"):
        for i in range(40):
            proxy_rows.append(S.corpus_row(example_id=f"{source}:{i}", source=source, license="mit", split="dev", lang="en",
                                           request={"state": f"case {i}", "questions": {"q1": {"type": "choice", "instructions": "pick", "criteria": {k: k.lower() for k in "ABCD"}}}},
                                           gold={"q1": "A"}))
    S.write_corpus_part(data / "proxy" / "proxy.parquet", proxy_rows)
    S.seal_directory(data / "proxy", kind="proxy")
    staging = root / "staging"
    shutil.copytree(data / "corpus", staging / "corpus-limpo")
    shutil.copytree(data / "labels", staging / "labels" / "pplx-decider-v1.2")
    shutil.copytree(data / "proxy", staging / "proxy")
    hub = H.LocalHub(root / "hub")
    hub.ensure_repo("me/data", kind="dataset")
    hub.upload("me/data", staging, kind="dataset")
    start = make_tiny_checkpoint(root / "start-src", FakeTokenizer())
    hub.ensure_repo("perplexity-ai/pplx-decider-v1.2", kind="model")
    hub.upload("perplexity-ai/pplx-decider-v1.2", start, kind="model")
    requests = root / "public-requests.jsonl"
    G.proxy_to_requests(data / "proxy", requests)
    return hub, requests


def make_day(workdir, hub, kit, executor, *, profile=None, cfg=None, **kw):
    from decide.rental import preflight as PF
    from decide.rental import run as R

    probes = PF.Probes(gpus=lambda: [], free_gb=lambda path: 10_000.0, ram_gb=lambda: 1_000.0)
    kw.setdefault("probes", probes)
    kw.setdefault("allow_cpu", True)
    kw.setdefault("expected_start", None)
    kw.setdefault("log", lambda text: None)
    return R.Day(profile=profile or tiny_profile(), cfg=cfg or tiny_config(), workdir=workdir, hub=hub, kit=kit, executor=executor, **kw)


def setUpModule() -> None:
    if not (HAVE_TRANSFORMERS and HAVE_PYARROW and HAVE_YAML):
        raise unittest.SkipTest("torch, transformers, pyarrow and PyYAML are needed")


class WholeDay(unittest.TestCase):
    """One decider-only day, in the stages a real one goes through: a stop at G4, a failure in the public round, completion, a rerun, a new machine."""

    @classmethod
    def setUpClass(cls) -> None:
        from decide.rental import run as R

        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        saved = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = TOKEN
        cls.addClassCleanup(lambda: os.environ.pop("HF_TOKEN", None) if saved is None else os.environ.__setitem__("HF_TOKEN", saved))
        root = Path(cls.tmp.name)
        cls.hub, cls.requests = build_world(root)
        cls.workdir = root / "day"
        cls.executor = InProcessExecutor()

        cls.kit_a = FakeKit()
        cls.a = make_day(cls.workdir, cls.hub, cls.kit_a, cls.executor).run(until="calibrate")
        cls.calls_after_a = list(cls.executor.modules())
        final, run = cls.workdir / "final", cls.workdir / "runs" / "seed1" / "final"
        cls.linked_after_a = all(os.path.samefile(path, run / path.name) for path in final.glob("*.safetensors"))
        cls.config_after_a = json.loads((final / "decision_config.json").read_text(encoding="utf-8"))
        cls.config_is_a_copy = not os.path.samefile(final / "decision_config.json", run / "decision_config.json")

        cls.kit_b = FakeKit(rates={"baseline": 2, "new": 2})
        cls.b = make_day(cls.workdir, cls.hub, cls.kit_b, cls.executor, public_requests=cls.requests).run()
        cls.weights_after_b = cls.hub.files("me/weights", kind="model") if (Path(cls.tmp.name) / "hub" / "model" / "me" / "weights").is_dir() else []
        cls.verdict_b = json.loads((cls.workdir / "g4" / "verdict.json").read_text(encoding="utf-8"))

        cls.kit_c = FakeKit(fail_on=("final",))
        try:
            make_day(cls.workdir, cls.hub, cls.kit_c, cls.executor, public_requests=cls.requests).run()
            cls.c_error = None
        except R.DayError as exc:
            cls.c_error = exc
        cls.state_after_c = make_day(cls.workdir, cls.hub, FakeKit(), cls.executor).state

        cls.kit_d = FakeKit()
        cls.d = make_day(cls.workdir, cls.hub, cls.kit_d, cls.executor, public_requests=cls.requests).run()
        cls.calls_after_d = list(cls.executor.modules())

        cls.kit_e = FakeKit()
        cls.e = make_day(cls.workdir, cls.hub, cls.kit_e, cls.executor, public_requests=cls.requests).run()
        cls.calls_after_e = list(cls.executor.modules())

        shutil.rmtree(cls.workdir)
        cls.kit_f = FakeKit()
        cls.f = make_day(cls.workdir, cls.hub, cls.kit_f, cls.executor, public_requests=cls.requests).run()
        cls.calls_after_f = list(cls.executor.modules())

    def test_the_first_stage_runs_the_cpu_steps_in_order_and_stops_where_asked(self) -> None:
        self.assertEqual(self.calls_after_a, ["decide.train.build_targets", "decide.train.train", "decide.train.temperature"])
        self.assertEqual((self.a.status, self.a.stopped_at), ("partial", None))
        self.assertEqual(self.a.phases["calibrate"], "done")
        self.assertEqual((self.a.phases["g4"], self.a.phases["publish"]), ("pending", "pending"))
        self.assertEqual(self.a.phases["traces"], "skipped")

    def test_the_trainer_got_the_orchestrators_paths_and_seed_and_the_profiles_options(self) -> None:
        argv = self.executor.calls[1]
        self.assertEqual(argv[argv.index("--seed") + 1], "20261021")
        self.assertEqual(Path(argv[argv.index("--out") + 1]), self.workdir / "runs" / "seed1")
        self.assertEqual(Path(argv[argv.index("--checkpoint") + 1]), self.workdir / "start")
        self.assertEqual(argv[argv.index("--reasoning") + 1], "off")
        self.assertEqual(Path(argv[argv.index("--targets") + 1]), self.workdir / "targets")

    def test_the_calibrated_checkpoint_is_a_staged_copy_and_the_run_output_is_untouched(self) -> None:
        self.assertTrue(self.linked_after_a, "the weights of final/ are links to the run's, not a second copy of 50 GB")
        self.assertFalse(self.config_after_a["temperature_is_placeholder"])
        self.assertIn("calibration", self.config_after_a)
        self.assertTrue(self.config_is_a_copy, "the config is a copy: fitting the temperature in final/ cannot change the run's output")
        self.assertTrue((self.workdir / "final" / ".staged-by-the-day").is_file())

    def test_g4_stops_the_day_when_the_new_checkpoint_is_not_better_and_nothing_is_published(self) -> None:
        self.assertEqual((self.b.status, self.b.stopped_at), ("stopped", "g4"))
        self.assertEqual(self.b.phases["public_round"], "pending")
        self.assertEqual(self.b.phases["publish"], "pending")
        self.assertEqual([call["model"] for call in self.kit_b.calls], ["baseline", "new"])
        self.assertEqual(self.weights_after_b, [])
        self.assertEqual((self.verdict_b["status"], self.verdict_b["reasons"]), ("FAIL", ["delta_below_margin"]))
        self.assertIn("delta_below_margin", self.b.reason)

    def test_a_failure_in_the_public_round_is_recorded_and_the_gate_result_is_kept(self) -> None:
        self.assertIsNotNone(self.c_error)
        self.assertIn("public_round", str(self.c_error))
        self.assertEqual(self.state_after_c.status("public_round"), "failed")
        self.assertEqual(self.state_after_c.status("g4"), "done")
        self.assertIn("lost the GPU", self.state_after_c.record("public_round")["reason"])

    def test_the_rerun_finishes_the_day_asking_the_kit_only_for_the_public_round(self) -> None:
        self.assertEqual((self.d.status, self.d.stopped_at), ("complete", None))
        self.assertEqual([call["model"] for call in self.kit_d.calls], ["final"])
        self.assertEqual(self.calls_after_d, self.calls_after_a)

    def test_the_gate_result_is_written_with_its_scope(self) -> None:
        verdict = json.loads((self.workdir / "g4" / "verdict.json").read_text(encoding="utf-8"))
        self.assertEqual(verdict["status"], "PASS")
        self.assertIn("not the board", verdict["scope"])
        self.assertGreaterEqual(verdict["delta_points"], 1.5)

    def test_publish_sends_the_weights_the_results_and_the_report_and_not_the_training_state(self) -> None:
        names = self.hub.files("me/weights", kind="model")
        self.assertTrue(any(name.endswith(".safetensors") for name in names))
        self.assertIn("decision_config.json", names)
        self.assertTrue(any(name.startswith("public-round/") for name in names), names)
        self.assertIn("day/day-report.json", names)
        for name in names:
            self.assertNotIn("resume.pt", name)
            self.assertNotIn("events.jsonl", name)
            self.assertNotIn(".staged-by-the-day", name)
        self.assertTrue(self.hub.is_private("me/weights", kind="model"))
        self.assertTrue(self.hub.is_private("me/day", kind="model"))

    def test_the_work_repo_keeps_the_state_and_the_outputs_for_a_new_machine(self) -> None:
        names = self.hub.files("me/day", kind="model")
        self.assertIn("state/day.json", names)
        self.assertTrue(any(name.startswith("outputs/train/") for name in names), names)
        self.assertTrue(any(name.startswith("outputs/calibrate/") for name in names), names)

    def test_a_third_run_changes_nothing(self) -> None:
        self.assertEqual(self.e.status, "complete")
        self.assertEqual(self.kit_e.calls, [])
        self.assertEqual(self.calls_after_e, self.calls_after_d)

    def test_on_a_new_machine_the_state_and_the_outputs_come_back_from_the_work_repo(self) -> None:
        self.assertEqual(self.f.status, "complete")
        self.assertEqual(self.kit_f.calls, [])
        self.assertEqual(self.calls_after_f, self.calls_after_d, "nothing was trained, labeled or calibrated again")
        self.assertTrue((self.workdir / "final" / "decision_config.json").is_file())
        self.assertTrue((self.workdir / "start" / "decision_config.json").is_file(), "the inputs are downloaded again, they are not backed up")

    def test_the_children_never_saw_the_token_that_the_parent_had(self) -> None:
        self.assertEqual(os.environ.get("HF_TOKEN"), TOKEN)
        for env in self.executor.envs:
            self.assertNotIn("HF_TOKEN", env)
            self.assertNotIn(TOKEN, json.dumps(env))

    def test_the_report_has_measured_seconds_and_the_unverified_estimate(self) -> None:
        report = json.loads((self.workdir / "report" / "day-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["phases"]["train"]["status"], "done")
        self.assertGreater(report["phases"]["train"]["seconds_measured"], 0)
        self.assertEqual(report["phases"]["train"]["estimate"]["status"], "UNVERIFIED")
        self.assertEqual(report["gates"]["g4"]["status"], "PASS")
        self.assertNotIn(TOKEN, json.dumps(report))

    def test_no_log_holds_the_token(self) -> None:
        for log in (self.workdir / "logs").rglob("*"):
            if log.is_file():
                self.assertNotIn(TOKEN, log.read_text(encoding="utf-8", errors="replace"), log)


class ThinkerDay(unittest.TestCase):
    """With a GT report that passed, the thinker's traces are made, attached, and the trainer shows the reasoning."""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(Path(__file__).parent))
        from test_contract_71 import FakeThinker, gt_report

        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        cls.hub, _ = build_world(root)
        cls.workdir = root / "day"
        cls.executor = InProcessExecutor()
        cls.thinker = FakeThinker()
        report = root / "gt-report.json"
        report.write_text(json.dumps(gt_report()), encoding="utf-8")
        cls.day = make_day(cls.workdir, cls.hub, FakeKit(), cls.executor, profile=tiny_profile(epochs=1, effective_batch=4), thinker=cls.thinker, gt_report=report)
        cls.result = cls.day.run(until="train")

    def test_traces_are_generated_attached_and_trained_on(self) -> None:
        self.assertEqual(self.result.status, "partial")
        self.assertEqual(self.executor.modules(), ["decide.train.build_targets", "decide.train.build_targets", "decide.train.train"])
        second, train = self.executor.calls[1], self.executor.calls[2]
        self.assertIn("--traces", second)
        self.assertEqual(second[second.index("--thinker") + 1], "Qwen3.8-27B")
        self.assertEqual(Path(train[train.index("--targets") + 1]), self.workdir / "targets-traces")
        self.assertEqual(train[train.index("--reasoning") + 1], "mixed")
        self.assertEqual(len(self.thinker.calls), 14)

    def test_the_evidence_and_the_counts_are_in_the_state(self) -> None:
        record = self.day.state.record("traces")
        self.assertEqual(record["status"], "done")
        self.assertEqual(record["detail"]["evidence"]["model"], "Qwen/Qwen3.8-27B")
        self.assertRegex(record["detail"]["evidence"]["report_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual((record["detail"]["written"], record["detail"]["errors"]), (14, 0))
        self.assertGreater(self.day.state.record("targets_traces")["detail"]["reasoning_rows"], 0)

    def test_the_gold_label_never_reached_the_thinker(self) -> None:
        self.assertNotIn("gold", json.dumps(self.thinker.calls))


class WithoutAGtReport(unittest.TestCase):
    def test_the_thinker_step_is_skipped_with_its_reason_and_nothing_is_asked_of_the_thinker(self) -> None:
        sys.path.insert(0, str(Path(__file__).parent))
        from test_contract_71 import FakeThinker

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            thinker = FakeThinker()
            executor = InProcessExecutor()
            day = make_day(Path(tmp) / "day", hub, FakeKit(), executor, thinker=thinker)
            day.run(until="targets_traces")
            self.assertEqual(thinker.calls, [])
            self.assertEqual(day.state.status("traces"), "skipped")
            self.assertIn("no GT report", day.state.record("traces")["reason"])
            self.assertEqual(day.state.status("targets_traces"), "skipped")


class LabelCoverageGate(unittest.TestCase):
    def test_too_few_labels_stop_the_day_before_any_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp), label_every=2)
            executor = InProcessExecutor()
            result = make_day(Path(tmp) / "day", hub, FakeKit(), executor).run()
            self.assertEqual((result.status, result.stopped_at), ("stopped", "targets"))
            self.assertEqual(executor.modules(), ["decide.train.build_targets"])
            self.assertIn("labeled", result.reason)


class RefusalsBeforeTheMoney(unittest.TestCase):
    def test_a_machine_that_does_not_match_stops_the_day_before_any_download(self) -> None:
        from decide.rental import preflight as PF
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            executor = InProcessExecutor()
            probes = PF.Probes(gpus=lambda: [], free_gb=lambda path: 10_000.0, ram_gb=lambda: 1_000.0)
            day = make_day(Path(tmp) / "day", hub, FakeKit(), executor, probes=probes, allow_cpu=False)
            with self.assertRaises(R.DayError) as caught:
                day.run()
            self.assertIn("B200", str(caught.exception))
            self.assertEqual(executor.calls, [])
            self.assertFalse((Path(tmp) / "day" / "start").exists())

    def test_a_dry_run_prints_the_plan_and_touches_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            executor, kit = InProcessExecutor(), FakeKit()
            day = make_day(Path(tmp) / "day", hub, kit, executor)
            text = day.plan_text()
            for word in ("fetch", "build_targets", "decide.train.train", "decide.train.temperature", "g4", "UNVERIFIED"):
                self.assertIn(word, text)
            self.assertEqual((executor.calls, kit.calls), ([], []))
            self.assertFalse((Path(tmp) / "day").exists())
            self.assertNotIn(TOKEN, text)

    def test_a_state_of_another_profile_is_refused(self) -> None:
        from decide.rental import run as R
        from decide.rental import state as ST

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            work = Path(tmp) / "day"
            ST.DayState(work / "state", identity={"profile": "other", "config": "other"}).begin("fetch")
            with self.assertRaises((R.DayError, ST.StateError)):
                make_day(work, hub, FakeKit(), InProcessExecutor()).run()


if __name__ == "__main__":
    unittest.main()
