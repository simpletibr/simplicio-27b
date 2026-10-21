"""Flow tests for decide/sd/evaluate.py (issue #65): a real engine over a tiny Qwen3.5 backbone, rows in, results and scores out.

The backbone has random weights (tests/tiny_model.py), so the numbers mean nothing; what is checked is the plumbing:
that what lands in the files is what the engine computes, that resuming changes nothing, and that the sample is
shrunk, run and scored the way the notebooks do it.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from decide_support import HermeticTestCase, needs_kit, needs_transformers

from decide.sd import budget as budget_mod
from decide.sd import calibrate, sample


def tiny_rows(count=12):
    rows = []
    for i in range(count):
        catalog = 28 if i % 2 == 0 else 29
        keys = ["book", "ask", "refuse", "defer"] if catalog == 29 else ["book", "ask"]
        rows.append({
            "_evaluation": {"run_id": f"{catalog}:T:T:{i}", "catalog_id": catalog, "dataset": "D", "group_id": f"g{i}", "track": "T",
                            "proxy_tokens": 60 + 10 * i},
            "state": "alpha beta gamma " * (1 + i % 3),
            "questions": {"route": {"type": "choice", "instructions": "Pick one", "criteria": {k: v for k, v in zip(keys, ["yes", "no", "image", "0"])}},
                          "urgent": {"type": "noul", "instructions": "Is it urgent"}},
            "expected": {"route": keys[i % len(keys)], "urgent": bool(i % 2)},
            "family": "f", "id": f"id{i}", "metadata": {}, "split": "test",
        })
    return rows


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@needs_transformers
class EvaluateFlow(HermeticTestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        import tiny_model as tm
        from decide.sd import engine, evaluate

        cls.torch, cls.tm, cls.E, cls.ev = torch, tm, engine, evaluate
        cls.state = tm.backbone_state(1)
        cls.head = tm.head_weight(1)

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.rows = tiny_rows()
        self.req = write_rows(self.dir / "req.jsonl", self.rows)

    def engine(self, **kw):
        return self.E.DecisionEngine(self.tm.dense_decider(self.state, self.head), **kw)

    def read(self, path):
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]

    def test_the_rows_in_the_file_are_what_the_engine_answers(self):
        engine = self.engine(tta_rotations=2)
        out = self.ev.run_requests(engine, self.req, self.dir / "run", chunk=5, log=None)
        self.assertEqual((out["status"], out["total"], out["counts"]), ("complete", 12, {"ok": 12}))
        results = {r["run_id"]: r for r in self.read(self.dir / "run" / "results.jsonl")}
        raws = {r["run_id"]: r for r in self.read(self.dir / "run" / "raw.jsonl")}
        for row in self.rows[:4]:
            run_id = row["_evaluation"]["run_id"]
            response, raw = engine(row["state"], row["questions"])
            got = results[run_id]["response"]["answers"]
            for qid, answer in response["answers"].items():
                if answer["type"] == "choice":
                    self.assertEqual(got[qid]["choice"], answer["choice"])
                    for key, p in answer["probabilities"].items():
                        self.assertAlmostEqual(got[qid]["probabilities"][key], p, places=6)
                else:
                    self.assertAlmostEqual(got[qid]["noul"], answer["noul"], places=6)
            self.assertEqual(raws[run_id]["views"], 2)
            self.assertEqual(len(raws[run_id]["logits"]["route"]), 2)             # one row of logits per TTA view
            for view_a, view_b in zip(raws[run_id]["logits"]["route"], raw["logits"]["route"]):
                for a, b in zip(view_a, view_b):
                    self.assertAlmostEqual(a, b, places=5)

    def test_chunk_size_does_not_change_the_answers(self):
        small = self.engine()
        self.ev.run_requests(small, self.req, self.dir / "a", chunk=1, log=None)
        big = self.engine()
        self.ev.run_requests(big, self.req, self.dir / "b", chunk=12, log=None)
        a = {r["run_id"]: r["response"]["answers"] for r in self.read(self.dir / "a" / "results.jsonl")}
        b = {r["run_id"]: r["response"]["answers"] for r in self.read(self.dir / "b" / "results.jsonl")}
        self.assertEqual(a.keys(), b.keys())
        for run_id in a:
            self.assertEqual(a[run_id]["route"]["choice"], b[run_id]["route"]["choice"])
            for key, p in a[run_id]["route"]["probabilities"].items():
                self.assertAlmostEqual(b[run_id]["route"]["probabilities"][key], p, places=5)

    def test_resume_after_a_stop_gives_the_same_file_as_one_straight_run(self):
        straight = self.dir / "straight"
        self.ev.run_requests(self.engine(), self.req, straight, chunk=4, log=None)
        stopped = self.dir / "stopped"
        first = self.ev.run_requests(self.engine(), self.req, stopped, chunk=4, limit=6, log=None)
        self.assertEqual((first["status"], first["total"]), ("limit", 6))
        again = self.ev.run_requests(self.engine(), self.req, stopped, chunk=4, log=None)
        self.assertEqual((again["rows"], again["total"], again["skipped"]), (6, 12, 6))
        a = {r["run_id"]: r["response"]["answers"]["route"]["choice"] for r in self.read(straight / "results.jsonl")}
        b = {r["run_id"]: r["response"]["answers"]["route"]["choice"] for r in self.read(stopped / "results.jsonl")}
        self.assertEqual(a, b)

    def test_requests_the_kit_cannot_score_are_unsupported_rows_not_crashes(self):
        rows = tiny_rows(4)
        rows[1]["questions"]["extra"] = {"type": "score", "criteria": ["low", "high"]}
        write_rows(self.dir / "mixed.jsonl", rows)
        out = self.ev.run_requests(self.engine(), self.dir / "mixed.jsonl", self.dir / "mixed", log=None)
        self.assertEqual(out["counts"], {"ok": 3, "unsupported": 1})

    def test_a_row_too_long_for_the_memory_is_the_only_unsupported_row_of_its_chunk(self):
        rows = tiny_rows(3)
        rows[1]["state"] = "x " * 60                              # the only one that does not fit
        write_rows(self.dir / "oom.jsonl", rows)
        decider = self.tm.dense_decider(self.state, self.head)
        sizes = [max(c.length for c in decider.encoder.plan(r, decider.max_length)) for r in
                 ({"state": rows[0]["state"], "questions": rows[0]["questions"]}, {"state": rows[1]["state"], "questions": rows[1]["questions"]})]
        limit = (sizes[0] + sizes[1]) // 2
        real = decider.reference_logits

        def flaky(chunks):
            if any(c.length > limit for c in chunks):
                raise self.E.OOM_ERRORS[0]("CUDA out of memory (simulated)")
            return real(chunks)

        decider.reference_logits = flaky
        out = self.ev.run_requests(self.E.DecisionEngine(decider), self.dir / "oom.jsonl", self.dir / "oom", chunk=16, log=None)
        self.assertEqual(out["counts"], {"ok": 2, "unsupported": 1})
        resumed = self.ev.run_requests(self.E.DecisionEngine(decider), self.dir / "oom.jsonl", self.dir / "oom", chunk=16, log=None)
        self.assertEqual(resumed["rows"], 0)                      # the unsupported row is final, the two good ones are done

    @needs_kit
    def test_the_results_file_is_read_by_the_kits_own_loader(self):
        from decision_index.scoring.report import load_results

        self.ev.run_requests(self.engine(), self.req, self.dir / "run", log=None)
        loaded = load_results(str(self.dir / "run" / "results.jsonl"))
        self.assertEqual(len(loaded), 12)
        self.assertTrue(all(r["status"] == "ok" and "total_wall_ms" in r for r in loaded.values()))

    def test_calibration_items_from_a_real_run_fit_a_temperature_per_question_type(self):
        engine = self.engine(tta_rotations=2)
        self.ev.run_requests(engine, self.req, self.dir / "run", log=None)
        items = self.ev.calibration_items(self.dir / "run", self.rows)
        self.assertEqual(len(items), 24)                                         # 12 choice + 12 noul questions
        self.assertEqual({len(i.logits) for i in items if i.qtype == "choice"}, {2})
        fit = calibrate.fit_groups(items, by="question_type", holdout=0.5, seed=1, baseline=engine.decider.temperature, min_items=5)
        self.assertEqual(set(fit["temperatures"]), {"choice", "noul"})
        for t in fit["temperatures"].values():
            self.assertGreater(t, 0)

    def test_evaluate_sample_shrinks_runs_and_resumes_with_the_same_rows(self):
        """Scoring is left out on purpose (it has its own test below, which needs the kit): this one holds in an interpreter without the kit."""
        selection = sample.select(self.rows, per_benchmark=6, seed=7, edition="0.3")
        sample.write(selection, self.dir / "sample")
        budget = budget_mod.Budget(session_hours=0.1, reserve_minutes=0)         # 360 s
        slow = {"tokens_per_second": 1.0, "n": 8, "seconds": 1.0, "proxy_tokens": 1, "requests_per_second": 1.0, "run_ids": [], "failed": 0}
        with mock.patch.object(self.ev, "measure_throughput", return_value=slow):
            first = self.ev.evaluate_sample(self.engine(), self.dir / "sample", self.dir / "eval", budget=budget, safety=0.5, log=None, name="tiny", score=False)
        # 1 token/s x 360 s x 0.5 = 180 proxy tokens: one group per benchmark (60..70 + 70..80 tokens) fits, two do not.
        self.assertEqual(first["plan"]["per_benchmark"], 1)
        self.assertIn("shrunk", first["plan"]["message"])
        self.assertEqual(first["sample"]["groups"], 2)
        self.assertEqual(first["run"]["status"], "complete")
        self.assertEqual(first["run"]["total"], 2)
        subset_rows, subset_manifest = sample.load(self.dir / "eval" / "subset")
        self.assertEqual(len(subset_rows), 2)
        self.assertEqual(subset_manifest["per_benchmark"], 1)
        metrics = json.loads((self.dir / "eval" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["plan"]["per_benchmark"], 1)
        # the second call must not measure again and must keep the same rows
        with mock.patch.object(self.ev, "measure_throughput", side_effect=AssertionError("measured again")):
            second = self.ev.evaluate_sample(self.engine(), self.dir / "sample", self.dir / "eval", budget=budget, log=None, name="tiny", score=False)
        self.assertEqual(second["plan"], first["plan"])
        self.assertEqual(second["run"]["rows"], 0)

    def test_the_measured_throughput_of_the_plan_prices_the_first_chunk_of_every_call(self):
        sample.write(sample.select(self.rows, per_benchmark=6, seed=7, edition="0.3"), self.dir / "sample")
        budget = budget_mod.Budget(session_hours=0.1, reserve_minutes=0)
        slow = {"tokens_per_second": 4.0, "n": 8, "seconds": 1.0, "proxy_tokens": 4, "requests_per_second": 1.0, "run_ids": [], "failed": 0}
        seen = []
        real = self.ev.run_requests

        def spy(*args, **kw):
            seen.append(kw.get("tokens_per_second"))
            return real(*args, **kw)

        with mock.patch.object(self.ev, "measure_throughput", return_value=slow), mock.patch.object(self.ev, "run_requests", side_effect=spy):
            self.ev.evaluate_sample(self.engine(), self.dir / "sample", self.dir / "eval", budget=budget, safety=0.5, log=None, name="tiny", score=False)
            self.ev.evaluate_sample(self.engine(), self.dir / "sample", self.dir / "eval", budget=budget, log=None, name="tiny", score=False)
            self.ev.evaluate_sample(self.engine(), self.dir / "sample", self.dir / "forced", per_benchmark=1, log=None, name="forced", score=False)
        self.assertEqual(seen, [4.0, 4.0, None])                                  # a forced size has no measurement to price with

    @needs_kit
    def test_evaluate_sample_scores_with_the_kit(self):
        sample.write(sample.select(self.rows, per_benchmark=6, seed=7, edition="0.3"), self.dir / "sample")
        metrics = self.ev.evaluate_sample(self.engine(), self.dir / "sample", self.dir / "eval", log=None, name="tiny")
        scores = metrics["scores"]
        self.assertTrue(0.0 <= scores["public"] <= 100.0)
        self.assertEqual(scores["benchmarks_present"], [28, 29])
        self.assertTrue(scores["partial"])
        self.assertEqual(scores["rows"], 12)
        again = self.ev.score_sample(self.dir / "eval" / "run" / "results.jsonl", self.dir / "eval" / "subset")
        self.assertAlmostEqual(again["public"], scores["public"], places=9)

    def test_a_forced_size_skips_the_measurement(self):
        sample.write(sample.select(self.rows, per_benchmark=6, seed=7, edition="0.3"), self.dir / "sample")
        with mock.patch.object(self.ev, "measure_throughput", side_effect=AssertionError("should not measure")):
            metrics = self.ev.evaluate_sample(self.engine(), self.dir / "sample", self.dir / "eval", per_benchmark=2, score=False, log=None)
        self.assertEqual(metrics["plan"]["per_benchmark"], 2)
        self.assertTrue(metrics["plan"]["forced"])
        self.assertEqual(metrics["run"]["total"], 4)
        self.assertNotIn("scores", metrics)


if __name__ == "__main__":
    unittest.main()
