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
OTHER_KEY = "sk-" + "q" * 30                    # a credential of another service that the rented machine happens to export


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
    cfg["gates"]["g4_min_rows"] = 10                                        # the rehearsal proxy has 80 questions; the shipped floor is for the real one
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


def build_world(root: Path, *, label_every=1, screened=True, labels_sealed=True, proxy_checked=True, proxy_declared_rows=None, corpus_tool=None):
    """A LocalHub holding the private data repo and the start checkpoint; returns (hub, proxy_requests_file).

    ``screened=False`` uploads a corpus sealed by hand (not the output of decontaminate); ``labels_sealed=False`` uploads labels without a manifest;
    ``proxy_checked=False`` uploads a proxy whose second pass (against the clean corpus) never ran; ``proxy_declared_rows`` makes its manifest declare another number;
    ``corpus_tool`` stamps the corpus manifest with another tool but leaves ``decontamination.json`` (and its reference for this proxy) where it is."""
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
        S.seal_directory(data / "labels", kind="labels")                       # build_targets reads a labels folder through its manifest
    proxy_rows = []
    for source in ("alpha", "beta"):
        for i in range(40):
            proxy_rows.append(S.corpus_row(example_id=f"{source}:{i}", source=source, license="mit", split="dev", lang="en",
                                           request={"state": f"case {i}", "questions": {"q1": {"type": "choice", "instructions": "pick", "criteria": {k: k.lower() for k in "ABCD"}}}},
                                           gold={"q1": "A"}))
    S.write_corpus_part(data / "proxy" / "proxy.parquet", proxy_rows)
    S.seal_directory(data / "proxy", kind="proxy", extra={"sources": {"alpha": {"rows": 40}, "beta": {"rows": 40}},
                                                           "corpus_leak_check": ({"checked": True, "leaks": 0, "examples": []} if proxy_checked
                                                                                 else {"checked": False, "sources_only": True})})     # as the second pass of proxy_private build leaves it
    if proxy_declared_rows is not None:
        sealed = json.loads((data / "proxy" / "manifest.json").read_text(encoding="utf-8"))
        S.write_manifest(data / "proxy" / "manifest.json", {**sealed, "rows": proxy_declared_rows})
    clean = json.loads((data / "corpus" / "manifest.json").read_text(encoding="utf-8"))                                            # and decontaminate, given the proxy
    S.write_manifest(data / "corpus" / "manifest.json", {**clean, "sources": {"toy": {"rows": clean["rows"]}}})
    report = json.loads((data / "corpus" / "decontamination.json").read_text(encoding="utf-8"))
    S.write_manifest(data / "corpus" / "decontamination.json", {**report, "references": [{"path": "work/proxy", "requests": len(proxy_rows), "rows": len(proxy_rows), "skipped": 0, "sha256": "0" * 64}]})
    staging = root / "staging"
    shutil.copytree(data / "corpus", staging / "corpus-limpo")
    shutil.copytree(data / "labels", staging / "labels" / "pplx-decider-v1.2")
    shutil.copytree(data / "proxy", staging / "proxy")
    if not screened:
        (staging / "corpus-limpo" / "decontamination.json").unlink()
        (staging / "corpus-limpo" / "manifest.json").unlink()
        S.seal_directory(staging / "corpus-limpo")
    if not labels_sealed:
        (staging / "labels" / "pplx-decider-v1.2" / "manifest.json").unlink()
    if corpus_tool is not None:
        stamped = json.loads((staging / "corpus-limpo" / "manifest.json").read_text(encoding="utf-8"))
        S.write_manifest(staging / "corpus-limpo" / "manifest.json", {**stamped, "tool": corpus_tool})
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
        saved_other = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = OTHER_KEY
        cls.addClassCleanup(lambda: os.environ.pop("OPENAI_API_KEY", None) if saved_other is None else os.environ.__setitem__("OPENAI_API_KEY", saved_other))
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

    def test_the_children_never_saw_the_other_credentials_of_the_machine_either(self) -> None:
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), OTHER_KEY)
        self.assertTrue(self.executor.envs)
        for env in self.executor.envs:
            self.assertFalse("OPENAI_API_KEY" in env, "a credential of another service reached a child")       # not assertNotIn: it would print the whole environment
            self.assertFalse(OTHER_KEY in json.dumps(env), "the value of a credential of another service reached a child")

    def test_the_day_log_names_the_variables_it_left_out_once_and_never_their_values(self) -> None:
        text = (self.workdir / "logs" / "day.log").read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if "child environment" in line]
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("OPENAI_API_KEY", lines[0])
        self.assertIn("HF_TOKEN", lines[0])
        self.assertNotIn(OTHER_KEY, text)
        self.assertNotIn(TOKEN, text)

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

    def test_a_corpus_that_did_not_come_out_of_decontaminate_stops_the_day_at_the_fetch(self) -> None:
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp), screened=False)
            executor = InProcessExecutor()
            with self.assertRaises(R.DayError) as caught:
                make_day(Path(tmp) / "day", hub, FakeKit(), executor).run()
            self.assertIn("fetch", str(caught.exception))
            self.assertIn("decontaminate", str(caught.exception))
            self.assertEqual(executor.calls, [], "no command ran, so no GPU hour was spent on rows nobody screened")
            self.assertFalse((Path(tmp) / "day" / "start").exists(), "the 52 GB checkpoint is not downloaded for a day that cannot start")

    def test_a_corpus_not_stamped_by_decontaminate_stops_the_day_even_when_a_decontamination_report_and_the_proxys_reference_are_there(self) -> None:
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp), corpus_tool="build_corpus")
            executor = InProcessExecutor()
            with self.assertRaises(R.DayError) as caught:
                make_day(Path(tmp) / "day", hub, FakeKit(), executor).run()
            self.assertIn("fetch", str(caught.exception))
            self.assertIn("build_corpus", str(caught.exception))
            self.assertEqual(executor.calls, [])
            self.assertFalse((Path(tmp) / "day" / "start").exists())

    def test_labels_without_a_manifest_stop_the_day_at_the_fetch(self) -> None:
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp), labels_sealed=False)
            executor = InProcessExecutor()
            with self.assertRaises(R.DayError) as caught:
                make_day(Path(tmp) / "day", hub, FakeKit(), executor).run()
            self.assertIn("fetch", str(caught.exception))
            self.assertIn("labels", str(caught.exception))
            self.assertIn("seal_directory", str(caught.exception), "the message says how to make the manifest")
            self.assertEqual(executor.calls, [])

    def test_a_proxy_the_clean_corpus_was_not_checked_against_stops_the_day_at_the_fetch(self) -> None:
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp), proxy_checked=False)
            executor = InProcessExecutor()
            day = make_day(Path(tmp) / "day", hub, FakeKit(), executor)
            with self.assertRaises(R.DayError) as caught:
                day.run()
            self.assertIn("proxy", str(caught.exception))
            self.assertIn("corpus_leak_check", str(caught.exception))
            self.assertFalse((Path(tmp) / "day" / "start").exists(), "stopped before the 52 GB checkpoint was downloaded")
            self.assertEqual(executor.calls, [])

    def test_a_proxy_with_fewer_questions_than_the_floor_stops_the_day_at_the_fetch_not_after_the_training(self) -> None:
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            cfg = tiny_config()
            cfg["gates"]["g4_min_rows"] = 81                                    # the proxy of the rehearsal has 80
            executor = InProcessExecutor()
            with self.assertRaises(R.DayError) as caught:
                make_day(Path(tmp) / "day", hub, FakeKit(), executor, cfg=cfg).run()
            self.assertIn("g4_min_rows", str(caught.exception))
            self.assertIn("80", str(caught.exception))
            self.assertFalse((Path(tmp) / "day" / "start").exists())
            self.assertEqual(executor.calls, [])

    def test_a_proxy_whose_parts_hold_other_rows_than_its_manifest_declares_stops_the_day_at_the_fetch(self) -> None:
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp), proxy_declared_rows=79)
            executor = InProcessExecutor()
            with self.assertRaises(R.DayError) as caught:
                make_day(Path(tmp) / "day", hub, FakeKit(), executor).run()
            self.assertIn("declares 79", str(caught.exception))
            self.assertIn("hold 80", str(caught.exception))
            self.assertEqual(executor.calls, [])

    def test_g4_scores_the_whole_proxy_the_manifest_declares(self) -> None:
        from decide.data import schema as S
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            day = make_day(Path(tmp) / "day", hub, FakeKit(), InProcessExecutor())
            day.run(until="calibrate")
            manifest = json.loads((day.paths.proxy / "manifest.json").read_text(encoding="utf-8"))
            S.write_manifest(day.paths.proxy / "manifest.json", {**manifest, "rows": 81})          # a manifest that declares one row more than the parts hold
            with self.assertRaises(R._Stop) as caught:
                day._do_g4()
            self.assertIn("proxy_rows_differ_from_manifest", str(caught.exception))

    def test_the_shipped_floor_is_not_met_by_the_tiny_rehearsal_proxy_so_the_default_config_is_not_a_rehearsal_config(self) -> None:
        import yaml

        shipped = yaml.safe_load((Path(__file__).resolve().parents[1] / "decide" / "config.rental.yaml").read_text(encoding="utf-8"))["rental"]
        self.assertEqual(shipped["gates"]["g4_min_rows"], 100)
        self.assertEqual(tiny_config()["gates"]["g4_min_rows"], 10)

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


class IncompleteKit(FakeKit):
    """A kit that ends one named run early: the answers are there, the status is not ``complete``."""

    def __init__(self, unfinished: str, **kw: Any) -> None:
        super().__init__(**kw)
        self.unfinished = unfinished

    def run(self, model, requests, out, *, budget=None) -> Dict[str, Any]:
        info = super().run(model, requests, out, budget=budget)
        return dict(info, status="interrupted") if model.name == self.unfinished else info


class FailingBackupHub:
    """A LocalHub whose uploads fail the way a full quota, a dropped connection or a revoked token would."""

    def __new__(cls, root: Path, error: Exception):
        from decide.rental import hub as H

        class Failing(H.LocalHub):
            def upload(self, *args, **kwargs):
                raise error

        return Failing(root)


class GuardsOfTheDay(unittest.TestCase):
    """The orders of the day that no rehearsal reaches by itself: what each phase refuses to do before its gate, and what a failing backup does."""

    def world(self, tmp, kit=None, hub=None, **kw):
        from decide.data import schema as S
        from decide.rental import gates as G
        from decide.rental import hub as H

        hub = hub or H.LocalHub(Path(tmp) / "hub")
        day = make_day(Path(tmp) / "day", hub, kit or FakeKit(), InProcessExecutor(), **kw)
        rows = [S.corpus_row(example_id=f"{source}:{i}", source=source, license="mit", split="dev", lang="en",
                             request={"state": f"case {i}", "questions": {"q1": {"type": "choice", "instructions": "pick", "criteria": {k: k.lower() for k in "ABCD"}}}},
                             gold={"q1": "A"}) for source in ("alpha", "beta") for i in range(8)]
        S.write_corpus_part(day.paths.proxy / "proxy.parquet", rows)
        S.seal_directory(day.paths.proxy, kind="proxy")
        requests = Path(tmp) / "public-requests.jsonl"
        G.proxy_to_requests(day.paths.proxy, requests)
        return day, hub, requests

    @staticmethod
    def settle(day, phase, detail=None) -> None:
        """Record ``phase`` as finished, with what it was built on, as the orchestrator does (the unit tests below set states up by hand)."""
        day.state.begin(phase)
        day.state.done(phase, outputs={}, detail=detail or {}, built_on=day._inputs(phase))

    def g4_passed(self, day) -> None:
        """A checkpoint on the disk and a G4 that measured exactly it."""
        from decide.rental import state as ST

        day.paths.final.mkdir(parents=True, exist_ok=True)
        (day.paths.final / "model.safetensors").write_bytes(b"weights")
        self.settle(day, "g4", {"final_digest": ST.digest_tree(day.paths.final), "start_fingerprint": day._start_fingerprint()})

    def test_the_public_round_does_not_start_before_g4_passed(self) -> None:
        from decide.rental import run as R

        for label, prepare in (("g4 never ran", lambda state: None), ("g4 stopped the day", lambda state: state.stop("g4", "delta 0.4 < 1.5"))):
            with tempfile.TemporaryDirectory() as tmp:
                kit = FakeKit()
                day, _, requests = self.world(tmp, kit=kit, public_requests=None)
                day.public_requests = requests
                prepare(day.state)
                with self.assertRaises(R.DayError, msg=label) as caught:
                    day._do_public_round()
                self.assertIn("needs G4 passed first", str(caught.exception), label)
                self.assertEqual(kit.calls, [], f"{label}: no GPU time is spent on a round the gate did not allow")

    def test_nothing_is_published_before_g4_passed(self) -> None:
        from decide.rental import hub as H
        from decide.rental import run as R

        for label, prepare in (("g4 never ran", lambda state: None), ("g4 stopped the day", lambda state: state.stop("g4", "delta 0.4 < 1.5"))):
            with tempfile.TemporaryDirectory() as tmp:
                day, hub, _ = self.world(tmp)
                prepare(day.state)
                with self.assertRaises(R.DayError, msg=label) as caught:
                    day._do_publish()
                self.assertIn("nothing is published before G4 passes", str(caught.exception), label)
                with self.assertRaises(H.HubError, msg="the weights repo was not even created"):
                    hub.is_private("me/weights", kind="model")

    def test_a_run_the_kit_did_not_finish_is_not_scored_by_g4(self) -> None:
        from decide.rental import run as R

        for unfinished in ("baseline", "new"):
            with tempfile.TemporaryDirectory() as tmp:
                kit = IncompleteKit(unfinished)
                day, _, _ = self.world(tmp, kit=kit)
                day.paths.final.mkdir(parents=True)
                (day.paths.final / "model.safetensors").write_bytes(b"weights")
                with self.assertRaises(R.DayError, msg=unfinished) as caught:
                    day._do_g4()
                self.assertIn(f"ended the {unfinished} run with status 'interrupted'", str(caught.exception))
                self.assertIn("run again", str(caught.exception))
                self.assertFalse((day.paths.g4 / "verdict.json").exists(), "no verdict from a half run")

    def test_g4_itself_refuses_a_proxy_below_the_floor_even_if_the_fetch_let_it_by(self) -> None:
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            kit = FakeKit()
            day, _, _ = self.world(tmp, kit=kit)                                     # 16 questions
            day.cfg["gates"]["g4_min_rows"] = 17
            day.paths.final.mkdir(parents=True)
            (day.paths.final / "model.safetensors").write_bytes(b"weights")
            with self.assertRaises(R._Stop) as caught:
                day._do_g4()
            self.assertIn("too_few_rows", str(caught.exception))
            self.assertIn("INVALID", str(caught.exception))

    def test_a_public_round_the_kit_did_not_finish_is_not_recorded_as_done(self) -> None:
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            kit = IncompleteKit("final")
            day, _, requests = self.world(tmp, kit=kit)
            day.public_requests = requests
            self.g4_passed(day)
            with self.assertRaises(R.DayError) as caught:
                day._do_public_round()
            self.assertIn("ended the public round with status 'interrupted'", str(caught.exception))
            self.assertEqual([call["model"] for call in kit.calls], ["final"])

    def test_a_public_round_skipped_for_lack_of_the_file_runs_when_the_file_comes(self) -> None:
        """The owner runs the day without ``--public-requests`` and then again with it: the second run must not say "skipped already"."""
        with tempfile.TemporaryDirectory() as tmp:
            kit = FakeKit()
            day, _, requests = self.world(tmp, kit=kit)
            self.g4_passed(day)
            day.state.skip("public_round", "no public requests file")
            self.assertEqual(day.run(only=["public_round"]).phases["public_round"], "skipped", "still no file: still skipped")
            self.assertEqual(kit.calls, [])
            day.public_requests = requests
            result = day.run(only=["public_round"])
            self.assertEqual(result.phases["public_round"], "done")
            self.assertEqual([call["model"] for call in kit.calls], ["final"])

    def test_publish_is_done_again_when_the_public_round_finished_after_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day, _, _ = self.world(tmp)
            for phase, detail in (("g4", {}), ("public_round", {}), ("publish", {"uploaded": ["weights", "report"]})):
                self.settle(day, phase, detail)
            self.assertIn("public round", day._stale("publish") or "")
            self.settle(day, "publish", {"uploaded": ["weights", "public round", "report"]})
            self.assertIsNone(day._stale("publish"))

    def test_a_publish_without_a_public_round_stays_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day, _, _ = self.world(tmp)
            self.settle(day, "g4")
            day.state.skip("public_round", "no public requests file")
            self.settle(day, "publish", {"uploaded": ["weights", "report"]})
            self.assertIsNone(day._stale("publish"))

    def test_a_gt_report_that_comes_after_the_training_is_refused_not_ignored(self) -> None:
        """Traces skipped (no GT report), the model trained without them, and now a passing report is given: say so, do not carry on as if it counted."""
        from decide.rental import run as R

        sys.path.insert(0, str(Path(__file__).parent))
        from test_contract_71 import FakeThinker, gt_report

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "gt-report.json"
            report.write_text(json.dumps(gt_report()), encoding="utf-8")
            day, _, _ = self.world(tmp, thinker=FakeThinker(), gt_report=report)
            day.state.skip("traces", "thinker disabled: no GT report")
            day.state.skip("targets_traces", "no traces")
            self.assertIn("GT report", day._stale("traces") or "", "nothing was trained yet: the traces can still be made")
            day.state.begin("train")
            day.state.done("train", outputs={}, detail={})
            with self.assertRaises(R.DayError) as caught:
                day._stale("traces")
            self.assertIn("new --workdir", str(caught.exception))
            self.assertIn("without them", str(caught.exception))

    def test_the_targets_with_traces_are_made_once_the_traces_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day, _, _ = self.world(tmp)
            day.state.begin("traces")
            day.state.done("traces", outputs={}, detail={})
            day.state.skip("targets_traces", "no traces")
            self.assertIn("traces were made", day._stale("targets_traces") or "")

    def test_traces_stay_skipped_while_there_is_still_no_passing_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day, _, _ = self.world(tmp)
            day.state.skip("traces", "thinker disabled: no GT report")
            day.state.skip("targets_traces", "no traces")
            day.state.begin("train")
            day.state.done("train", outputs={}, detail={})
            self.assertIsNone(day._stale("traces"))
            self.assertIsNone(day._stale("targets_traces"))

    def test_a_backup_that_fails_is_a_warning_and_the_day_goes_on(self) -> None:
        from decide.rental import hub as H

        for error in (H.HubError("quota exceeded"), OSError("no space left on device"), ConnectionError("connection reset")):
            with tempfile.TemporaryDirectory() as tmp:
                seen: List[str] = []
                day, _, _ = self.world(tmp, hub=FailingBackupHub(Path(tmp) / "hub", error), log=seen.append)
                day.state.begin("train")
                day.state.done("train", outputs={}, detail={})
                day._backup("train")                                  # must not raise
                self.assertEqual(len(day._backup_warnings), 1, error)
                self.assertIn("backup of train to me/day failed", day._backup_warnings[0])
                self.assertIn(str(error), day._backup_warnings[0])
                self.assertIn("a lost machine would lose this phase", day._backup_warnings[0])
                self.assertTrue(any(line.startswith("WARNING: backup of train") for line in seen), seen)
                self.assertEqual(day.state.status("train"), "done", "the phase itself stays done")


    def test_what_g4_measured_is_every_byte_not_the_edges_of_a_big_weight_file(self) -> None:
        from decide.rental import state as ST

        with tempfile.TemporaryDirectory() as tmp:
            day, _, _ = self.world(tmp)
            weights = day.paths.final / "model.safetensors"
            day.paths.final.mkdir(parents=True)
            with open(weights, "wb") as handle:
                handle.truncate(ST.FULL_LIMIT + (6 << 20))                                   # sparse: cheap to make and to read
            self.settle(day, "g4", {"final_digest": ST.digest_tree(day.paths.final), "start_fingerprint": day._start_fingerprint()})
            self.assertIsNone(day._measured_problem())
            with open(weights, "r+b") as handle:
                handle.seek(ST.FULL_LIMIT // 2)
                handle.write(b"a flipped byte in the middle of the weights")
            day._digests = {}
            self.assertIn("not the one G4 measured", day._measured_problem() or "")

    def test_the_digest_is_read_again_after_a_phase_that_may_have_changed_the_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day, _, _ = self.world(tmp)
            weights = day.paths.final / "model.safetensors"
            day.paths.final.mkdir(parents=True)
            weights.write_bytes(b"one")
            before = day._final_digest()
            self.assertEqual(day._final_digest(), before, "read once while nothing ran")

            def changes_the_checkpoint():
                weights.write_bytes(b"two")
                return {}, {}

            day._do_preflight = changes_the_checkpoint                                       # any phase will do for the rule
            day._phase("preflight")
            self.assertNotEqual(day._final_digest(), before)

    def test_the_start_checkpoint_changing_after_g4_is_noticed_because_the_baseline_was_measured_on_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day, _, _ = self.world(tmp)
            day.paths.start.mkdir(parents=True)
            (day.paths.start / "model.safetensors").write_bytes(b"start")
            self.g4_passed(day)
            self.assertIsNone(day._measured_problem())
            (day.paths.start / "model.safetensors").write_bytes(b"another start checkpoint")
            self.assertIn("start checkpoint", day._measured_problem() or "")

    def test_publish_and_the_public_round_check_the_bytes_themselves_not_only_the_gate_before_them(self) -> None:
        from decide.rental import run as R

        for name, call in (("publish", lambda day: day._do_publish()), ("public round", lambda day: day._do_public_round())):
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                kit = FakeKit()
                day, hub, requests = self.world(tmp, kit=kit)
                day.public_requests = requests
                self.g4_passed(day)
                (day.paths.final / "model.safetensors").write_bytes(b"retrained after G4")
                with self.assertRaises(R.DayError) as caught:
                    call(day)
                self.assertIn("G4", str(caught.exception))
                self.assertEqual(kit.calls, [])

    #: What each phase needs, written out here on purpose: a test that only loops over ``NEEDS`` would pass when a line is taken out of it.
    EXPECTED_NEEDS = {
        "preflight": set(), "fetch": set(), "targets": {"fetch"}, "traces": {"fetch", "targets"}, "targets_traces": {"fetch", "targets", "traces"},
        "train": {"fetch", "targets", "traces", "targets_traces"}, "soup": {"train"}, "calibrate": {"targets", "targets_traces", "train", "soup"},
        "g4": {"fetch", "calibrate"}, "public_round": {"g4"}, "publish": {"g4", "public_round"}}

    def test_the_table_of_what_each_phase_needs_is_the_one_the_gates_rely_on(self) -> None:
        from decide.rental import run as R

        self.assertEqual({phase: {need.partition(":")[0] for need in needs} for phase, needs in R.NEEDS.items()}, self.EXPECTED_NEEDS)
        self.assertEqual(list(R.NEEDS), list(R.PHASES), "one line per phase, in the order of the day")

    def test_every_phase_needs_each_of_the_phases_it_names(self) -> None:
        """For every phase and every phase it names, with all the others settled and that one not, the phase is refused naming it."""
        from decide.rental import run as R

        for phase, needs in R.NEEDS.items():
            names = [need.partition(":")[0] for need in needs]
            for missing in names:
                with self.subTest(phase=phase, missing=missing), tempfile.TemporaryDirectory() as tmp:
                    day, _, _ = self.world(tmp)
                    for dep in (name for name in R.PHASES if name in names and name != missing):
                        if dep == "g4":
                            self.g4_passed(day)
                        else:
                            self.settle(day, dep)
                    with self.assertRaises(R.DayError) as caught:
                        day._require(phase)
                    self.assertIn(missing, str(caught.exception))
            with self.subTest(phase=phase, settled="all"), tempfile.TemporaryDirectory() as tmp:
                day, _, _ = self.world(tmp)
                for dep in (name for name in R.PHASES if name in names):
                    self.g4_passed(day) if dep == "g4" else self.settle(day, dep)
                self.assertEqual(set(day._require(phase)), set(names), "with everything settled the phase may run, and says what it is built on")

    def test_a_phase_that_was_skipped_settles_only_the_phases_that_may_be_skipped(self) -> None:
        from decide.rental import run as R

        for dep in R.PHASES:
            users = [phase for phase, needs in R.NEEDS.items() if dep in [need.partition(":")[0] for need in needs]]
            if not users:
                continue
            with self.subTest(dep=dep), tempfile.TemporaryDirectory() as tmp:
                day, _, _ = self.world(tmp)
                day.state.skip(dep, "for the test")
                if dep in R.MAY_BE_SKIPPED:
                    self.assertIsNone(day._unsettled(dep), dep)
                else:
                    self.assertIn(dep, day._unsettled(dep) or "")

    def test_built_on_follows_exactly_the_outputs_a_phase_reads(self) -> None:
        """targets reads corpus and labels of the fetch, not the proxy: a new proxy leaves it, new labels make it stale."""
        with tempfile.TemporaryDirectory() as tmp:
            day, _, _ = self.world(tmp)
            folders = {}
            for name in ("corpus", "labels", "proxy", "start"):
                folders[name] = Path(tmp) / "fetched" / name
                folders[name].mkdir(parents=True)
                (folders[name] / "part.txt").write_text(name, encoding="utf-8")

            def fetch_again() -> None:
                day.state.begin("fetch")
                day.state.done("fetch", outputs=folders, detail={}, built_on=day._inputs("fetch"))

            fetch_again()
            self.settle(day, "targets")
            self.assertIsNone(day._stale("targets"))
            (folders["proxy"] / "part.txt").write_text("a new proxy", encoding="utf-8")
            fetch_again()
            self.assertIsNone(day._stale("targets"), "the proxy is not an input of targets")
            (folders["labels"] / "part.txt").write_text("new labels", encoding="utf-8")
            fetch_again()
            self.assertIn("fetch", day._stale("targets") or "")


class NoisyTrainer(InProcessExecutor):
    """A trainer that is not bit-for-bit repeatable, as a GPU run is: every training leaves one extra file in the final of its seed."""

    def __init__(self) -> None:
        super().__init__()
        self.trainings = 0

    def run(self, argv, *, log_path, env) -> int:
        code = super().run(argv, log_path=log_path, env=env)
        if code == 0 and "decide.train.train" in argv:
            self.trainings += 1
            out = Path(argv[argv.index("--out") + 1])
            (out / "final" / f"noise-{self.trainings}.txt").write_text(f"run {self.trainings}\n", encoding="utf-8")
        return code


class RecordingKit(FakeKit):
    """A FakeKit that also says whether the output folder of a run already existed when the run began (a real kit resumes from it)."""

    def run(self, model, requests, out, *, budget=None) -> Dict[str, Any]:
        existed = Path(out).exists()
        info = super().run(model, requests, out, budget=budget)
        self.calls[-1]["out_existed"] = existed
        return info


def weights_in(hub, tmp) -> List[str]:
    return hub.files("me/weights", kind="model") if (Path(tmp) / "hub" / "model" / "me" / "weights").is_dir() else []


class GatesCannotBeSkipped(unittest.TestCase):
    """Review of the PR of F6: no way to a trained, calibrated or published model that goes around the label gate or around what G4 measured."""

    LATER = ("train", "soup", "calibrate", "g4", "publish")

    def test_only_cannot_go_around_the_label_gate(self) -> None:
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp), label_every=2)                            # half of the corpus is labeled: F3 says no
            executor, kit = InProcessExecutor(), FakeKit()
            first = make_day(Path(tmp) / "day", hub, kit, executor).run()
            self.assertEqual((first.status, first.stopped_at), ("stopped", "targets"))
            with self.assertRaises(R.DayError) as caught:
                make_day(Path(tmp) / "day", hub, kit, executor).run(only=list(self.LATER))
            self.assertIn("targets", str(caught.exception))
            self.assertEqual(executor.modules(), ["decide.train.build_targets"], "nothing was trained or calibrated")
            self.assertEqual(kit.calls, [], "the kit was not asked for G4")
            self.assertEqual(weights_in(hub, tmp), [], "nothing reached the weights repo")

    def test_each_phase_refuses_to_run_alone_before_what_it_needs(self) -> None:
        from decide.rental import run as R

        for phase in ("targets", "traces", "targets_traces", "train", "soup", "calibrate", "g4", "public_round", "publish"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                hub, requests = build_world(Path(tmp))
                executor, kit = InProcessExecutor(), FakeKit()
                with self.assertRaises(R.DayError):
                    make_day(Path(tmp) / "day", hub, kit, executor, public_requests=requests).run(only=[phase])
                self.assertEqual((executor.calls, kit.calls), ([], []), "no command ran and the kit was not asked")
                self.assertEqual(weights_in(hub, tmp), [])

    def test_the_phases_can_still_be_run_one_by_one_in_their_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            executor, kit = InProcessExecutor(), FakeKit()
            for phase in ("fetch", "targets", "traces", "targets_traces", "train", "soup", "calibrate", "g4", "public_round", "publish"):
                result = make_day(Path(tmp) / "day", hub, kit, executor).run(only=[phase])
                self.assertIsNone(result.stopped_at, phase)
                self.assertIn(result.phases[phase], ("done", "skipped"), phase)
            self.assertTrue(any(name.endswith(".safetensors") for name in weights_in(hub, tmp)))

    def test_a_targets_folder_that_changed_after_its_gate_is_not_trusted(self) -> None:
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            executor = InProcessExecutor()
            work = Path(tmp) / "day"
            make_day(work, hub, FakeKit(), executor).run(until="targets_traces")           # traces and targets_traces are settled (skipped): only targets can refuse
            (work / "targets" / "edited-by-hand.txt").write_text("x", encoding="utf-8")
            calls = len(executor.calls)
            with self.assertRaises(R.DayError) as caught:
                make_day(work, hub, FakeKit(), executor).run(only=["train"])
            self.assertIn("targets", str(caught.exception))
            self.assertEqual(len(executor.calls), calls)

    def test_train_alone_refuses_while_the_traces_are_due(self) -> None:
        """Traces were skipped (no GT report); the report that passes is given now; ``--only train`` must not train without the reasoning the report asks for."""
        from decide.rental import run as R

        sys.path.insert(0, str(Path(__file__).parent))
        from test_contract_71 import FakeThinker, gt_report

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            work, executor = Path(tmp) / "day", InProcessExecutor()
            make_day(work, hub, FakeKit(), executor).run(until="targets_traces")             # no report: traces and targets_traces are skipped
            report = Path(tmp) / "gt-report.json"
            report.write_text(json.dumps(gt_report()), encoding="utf-8")
            calls = len(executor.calls)
            with self.assertRaises(R.DayError) as caught:
                make_day(work, hub, FakeKit(), executor, thinker=FakeThinker(), gt_report=report).run(only=["train"])
            self.assertIn("traces", str(caught.exception))
            self.assertIn("GT report", str(caught.exception))
            self.assertEqual(len(executor.calls), calls, "nothing was trained")

    def test_g4_writes_the_digest_of_the_checkpoint_it_measured(self) -> None:
        from decide.rental import state as ST

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            work = Path(tmp) / "day"
            day = make_day(work, hub, FakeKit(), InProcessExecutor())
            day.run(until="g4")
            detail = day.state.record("g4")["detail"]
            self.assertEqual(detail["final_digest"], ST.digest_tree(work / "final"))
            self.assertEqual(detail["start_fingerprint"], ST.fingerprint(work / "start"))
            verdict = json.loads((work / "g4" / "verdict.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["measured"]["final_digest"], detail["final_digest"])

    def test_publish_and_the_public_round_refuse_a_final_that_g4_did_not_measure(self) -> None:
        from decide.rental import run as R

        for phase in ("publish", "public_round"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                hub, requests = build_world(Path(tmp))
                work = Path(tmp) / "day"
                make_day(work, hub, FakeKit(), InProcessExecutor(), public_requests=requests).run(until="g4")
                (work / "final" / "not-what-g4-saw.txt").write_text("a retrain, a recalibration, a hand edit\n", encoding="utf-8")
                kit = FakeKit()
                with self.assertRaises(R.DayError) as caught:
                    make_day(work, hub, kit, InProcessExecutor(), public_requests=requests).run(only=[phase])
                self.assertIn("G4", str(caught.exception))
                self.assertEqual(kit.calls, [], "no GPU time on a checkpoint nobody measured")
                self.assertEqual(weights_in(hub, tmp), [])

    def test_a_retrain_after_g4_makes_g4_run_again_on_the_new_bytes_before_anything_is_published(self) -> None:
        from decide.rental import state as ST

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            work = Path(tmp) / "day"
            executor = NoisyTrainer()
            first_kit = RecordingKit()
            first = make_day(work, hub, first_kit, executor)
            first.run(until="g4")
            before = first.state.record("g4")["detail"]["final_digest"]
            shutil.rmtree(work / "runs")                                              # the machine was lost and the training state with it
            shutil.rmtree(work / "final")
            kit = RecordingKit()
            second = make_day(work, hub, kit, executor)
            result = second.run()
            self.assertEqual(result.status, "complete")
            self.assertEqual(executor.trainings, 2, "the seed was trained again")
            self.assertEqual([call["model"] for call in kit.calls], ["baseline", "new"], "G4 measured the new bytes")
            calls = {call["model"]: call for call in kit.calls}
            self.assertFalse(calls["new"]["out_existed"], "the results of the old checkpoint were set aside, not resumed as the new one's")
            self.assertTrue(calls["baseline"]["out_existed"], "the baseline did not change, so its results are kept")
            after = second.state.record("g4")["detail"]["final_digest"]
            self.assertNotEqual(before, after)
            self.assertEqual(after, ST.digest_tree(work / "final"))
            self.assertEqual(second.state.status("publish"), "done")

    def test_a_retrain_that_gives_the_same_bytes_leaves_g4_and_the_upload_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            work = Path(tmp) / "day"
            executor = InProcessExecutor()
            make_day(work, hub, FakeKit(), executor).run()
            shutil.rmtree(work / "runs")
            shutil.rmtree(work / "final")
            kit = FakeKit()
            result = make_day(work, hub, kit, executor).run()
            self.assertEqual(result.status, "complete")
            self.assertEqual(kit.calls, [], "the CPU run is bit for bit the same, so what G4 measured is what is on the disk")


class RestoreStaysInsideTheWorkdir(unittest.TestCase):
    """Review of the PR of F6: the state that comes back from the work repo is checked before one byte is written."""

    def setUp(self) -> None:
        from decide.rental import hub as H

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.hub, _ = build_world(self.root)
        self.work = self.root / "day"
        self.outside = self.root / "outside"
        self.day = make_day(self.work, self.hub, FakeKit(), InProcessExecutor())
        self.H = H

    def plant(self, outputs, *, identity=None, phase="targets") -> None:
        """Put a state in the work repo whose ``phase`` is done and wrote ``outputs`` ({name: path}), with one file under each output."""
        record = {"status": "done", "started": 1.0, "finished": 2.0, "seconds": 1.0, "detail": {},
                  "outputs": {name: {"path": path, "fingerprint": "0" * 64} for name, path in outputs.items()}}
        state = {"version": 1, "identity": identity or self.day.identity, "phases": {phase: record}}
        staging = self.root / "staging-state"
        staging.mkdir()
        (staging / "day.json").write_text(json.dumps(state), encoding="utf-8")
        self.hub.ensure_repo("me/day", kind="model")
        self.hub.upload("me/day", staging, kind="model", path_in_repo="state")
        for index, name in enumerate(outputs):
            if "/" in name:
                continue                                              # a name like that cannot be put in the folder repo by the test; the state alone is the attack
            folder = self.root / f"staging-{index}"
            folder.mkdir()
            (folder / "payload.txt").write_text("from the work repo\n", encoding="utf-8")
            self.hub.upload("me/day", folder, kind="model", path_in_repo=f"outputs/{phase}/{name}")

    def assert_refused_and_nothing_written(self, where_not) -> None:
        from decide.rental import run as R
        from decide.rental import state as ST

        with self.assertRaises((ST.StateError, R.DayError)):
            self.day.run(only=["preflight"])
        for path in where_not:
            self.assertFalse(Path(path).exists(), f"{path} was written")
        self.assertFalse((self.work / "state" / "day.json").exists(), "the state was put in place before it was checked")
        self.assertFalse((self.work / ".restore").exists(), "the scratch folder was left behind")

    def test_an_absolute_path_outside_the_workdir_is_refused_before_anything_is_written(self) -> None:
        self.plant({"targets": str(self.outside / "targets")})
        self.assert_refused_and_nothing_written([self.outside])

    def test_a_path_that_climbs_out_with_dotdot_is_refused(self) -> None:
        self.plant({"targets": str(self.work / "x" / ".." / ".." / "outside" / "targets")})
        self.assert_refused_and_nothing_written([self.outside])

    def test_a_backslash_that_climbs_out_is_refused(self) -> None:
        self.plant({"targets": "..\\outside\\targets"})
        self.assert_refused_and_nothing_written([self.outside])

    def test_a_drive_letter_is_refused(self) -> None:
        self.plant({"targets": "C:\\outside\\targets"})
        self.assert_refused_and_nothing_written([self.outside])

    def test_a_backslash_in_the_middle_of_a_path_is_refused(self) -> None:
        self.plant({"targets": str(self.work) + "/targets\\..\\..\\x"})
        self.assert_refused_and_nothing_written([self.outside])

    def test_a_symlink_inside_the_workdir_that_leaves_it_is_refused(self) -> None:
        self.work.mkdir()
        self.outside.mkdir()
        os.symlink(self.outside, self.work / "link")
        self.plant({"targets": str(self.work / "link" / "targets")})
        self.assert_refused_and_nothing_written([self.outside / "targets"])

    def test_a_state_of_another_profile_is_refused_before_its_outputs_are_moved(self) -> None:
        self.plant({"targets": str(self.work / "targets")}, identity={"profile": "other", "config": "other"})
        self.assert_refused_and_nothing_written([self.work / "targets"])

    def test_the_name_of_an_output_cannot_climb_either(self) -> None:
        self.plant({"../../escape": str(self.work / "targets")})
        self.assert_refused_and_nothing_written([self.work / "targets", self.root / "escape"])

    def test_a_state_that_is_all_inside_the_workdir_is_restored(self) -> None:
        self.plant({"targets": str(self.work / "targets")})
        self.day.run(only=["preflight"])
        self.assertTrue((self.work / "targets" / "payload.txt").is_file())
        self.assertTrue((self.work / "state" / "day.json").is_file())


    def test_a_restore_that_stops_halfway_leaves_no_state_file_so_the_next_run_finishes_it(self) -> None:
        self.plant({"targets": str(self.work / "targets")})
        real = self.hub.download

        def failing(repo, dest, **kw):
            if any(str(pattern).startswith("outputs/") for pattern in kw.get("patterns") or ()):
                raise self.H.HubError("the connection dropped")
            return real(repo, dest, **kw)

        self.hub.download = failing
        with self.assertRaises(self.H.HubError):
            self.day.run(only=["preflight"])
        self.assertFalse((self.work / "state" / "day.json").exists(), "a state without its outputs would make the next run believe they are there")
        self.hub.download = real
        self.day.run(only=["preflight"])
        self.assertTrue((self.work / "targets" / "payload.txt").is_file())
        self.assertTrue((self.work / "state" / "day.json").is_file())

    def test_what_an_earlier_stopped_restore_left_in_the_scratch_folder_is_not_trusted(self) -> None:
        self.plant({"targets": str(self.work / "targets")})
        left = self.work / ".restore" / "outputs" / "targets" / "targets"
        left.mkdir(parents=True)
        (left / "left-by-someone.txt").write_text("not from the work repo\n", encoding="utf-8")
        self.day.run(only=["preflight"])
        self.assertTrue((self.work / "targets" / "payload.txt").is_file())
        self.assertFalse((self.work / "targets" / "left-by-someone.txt").exists())

    def test_a_state_already_on_the_disk_is_checked_too(self) -> None:
        from decide.rental import run as R
        from decide.rental import state as ST

        (self.work / "state").mkdir(parents=True)
        record = {"status": "done", "outputs": {"targets": {"path": str(self.outside / "targets"), "fingerprint": "0" * 64}}}
        (self.work / "state" / "day.json").write_text(json.dumps({"version": 1, "identity": self.day.identity, "phases": {"targets": record}}), encoding="utf-8")
        with self.assertRaises((ST.StateError, R.DayError)):
            self.day.run(only=["preflight"])


class OneOrchestratorPerWorkdir(unittest.TestCase):
    def test_a_second_orchestrator_on_the_same_workdir_is_refused(self) -> None:
        from decide.rental import run as R

        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            work = Path(tmp) / "day"
            seen: List[str] = []

            class Rival(InProcessExecutor):
                def run(inner, argv, *, log_path, env) -> int:                  # noqa: N805 - the outer test case is `self`
                    try:
                        make_day(work, hub, FakeKit(), InProcessExecutor()).run(only=["preflight"])
                    except R.DayError as exc:
                        seen.append(str(exc))
                    return super().run(argv, log_path=log_path, env=env)

            make_day(work, hub, FakeKit(), Rival()).run(until="targets")
            self.assertTrue(seen and "another" in seen[0], seen)
            self.assertFalse((work / "state" / "day.lock").exists(), "the lock is released when the run ends")

    def test_a_lock_left_by_a_process_that_is_gone_is_taken_over(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub, _ = build_world(Path(tmp))
            work = Path(tmp) / "day"
            (work / "state").mkdir(parents=True)
            import socket

            (work / "state" / "day.lock").write_text(json.dumps({"pid": 2 ** 22 + 12345, "host": socket.gethostname(), "started": 1.0}), encoding="utf-8")
            result = make_day(work, hub, FakeKit(), InProcessExecutor()).run(until="fetch")
            self.assertEqual(result.phases["fetch"], "done")


class RealKitDay(unittest.TestCase):
    """The day over the REAL ``decide.sd`` (no substitute for the kit): up to G4, on a CPU, with the tiny random checkpoint.

    Everything the other classes run through ``FakeKit`` runs here through the kit that the rented machine will use: the spec check, the reading of
    the checkpoint folder (``dir:``), the weights, the engine, the requests and results files.  The model is random, so the verdict is not the
    point: that both models were really run over every question, and that G4 read them, is.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from decide_b_support import real_kit, use_the_kit

        use_the_kit()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        saved = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = TOKEN
        cls.addClassCleanup(lambda: os.environ.pop("HF_TOKEN", None) if saved is None else os.environ.__setitem__("HF_TOKEN", saved))
        root = Path(cls.tmp.name)
        cls.hub, cls.requests = build_world(root)
        cls.workdir = root / "day"
        cls.result = make_day(cls.workdir, cls.hub, real_kit(), InProcessExecutor()).run(until="g4")
        cls.verdict = json.loads((cls.workdir / "g4" / "verdict.json").read_text(encoding="utf-8"))

    def test_the_day_reaches_g4_and_g4_read_every_question_of_both_models(self) -> None:
        for phase in ("preflight", "fetch", "targets", "train", "calibrate"):
            self.assertEqual(self.result.phases[phase], "done", self.result.phases)
        self.assertIn(self.result.phases["g4"], ("done", "stopped"))                           # a random model may or may not pass the gate
        for who in ("baseline", "new"):
            self.assertEqual((self.verdict[who]["n"], self.verdict[who]["bad"], self.verdict[who]["missing"]), (80, 0, 0), who)
            status = json.loads((self.workdir / "g4" / who / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((status["event"], status["total"]), ("complete", 80))
        self.assertEqual(self.verdict["expected_rows"], 80)

    def test_the_two_models_that_the_kit_ran_are_two_different_checkpoints(self) -> None:
        self.assertNotEqual(self.verdict["baseline"]["nll"], self.verdict["new"]["nll"])

    def test_g4_recorded_the_bytes_it_measured_and_they_are_the_final_checkpoint_on_the_disk(self) -> None:
        from decide.rental import state as ST

        measured = self.verdict["measured"]
        self.assertEqual(measured["final_digest"], ST.digest_tree(self.workdir / "final"))
        self.assertRegex(measured["final_digest"], r"^[0-9a-f]{64}$")
        self.assertRegex(measured["start_fingerprint"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
