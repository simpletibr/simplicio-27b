"""Flow tests for the engine extras and decide/sd/api.py (issue #65): the interface end to end over a tiny checkpoint.

The checkpoint has random weights and is served by the fake Hub of tests/decide_support.py, so what is checked is the
plumbing: that a name in the config becomes the merge it promises, that the numbers in the files are the engine's, and
that fitting a temperature from the probability table gives what fitting from the raw logits gives.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from decide_support import FakeHub, HermeticTestCase, needs_kit, needs_pyarrow, needs_transformers

from decide.sd import api, env, remote
from decide.sd import config as C


A, D, L, X = "a" * 40, "b" * 40, "c" * 40, "d" * 40         # commit hashes of the repos of these tests (the fake Hub ignores the revision in the URL)


def make_cfg(**overrides):
    return C.load_config(C.DEFAULT_PATH, overrides=overrides or None)


def hub_repos(hub):
    return lambda repo_id, revision=None: remote.HubRepo(repo_id, revision, endpoint=hub.base_url, sleep=lambda s: None, probe=4096)


@needs_transformers
class EngineExtrasFlow(HermeticTestCase):
    """Question types, `type:k` temperatures, answer codes and the fast-path routing of `DecisionEngine`."""

    @classmethod
    def setUpClass(cls):
        import torch

        import tiny_model as tm
        from decide.sd import engine, tta

        cls.torch, cls.tm, cls.E, cls.tta = torch, tm, engine, tta
        cls.state, cls.head = tm.backbone_state(1), tm.head_weight(1)

    def decider(self):
        return self.tm.dense_decider(self.state, self.head)

    def engine(self, decider=None, **kw):
        return self.E.DecisionEngine(decider or self.decider(), **kw)

    @staticmethod
    def with_score():
        return {"state": "alpha beta gamma", "questions": {
            "route": {"type": "choice", "instructions": "Pick one", "criteria": {"book": "yes", "ask": "no", "refuse": "image"}},
            "level": {"type": "score", "instructions": "How much", "criteria": ["no", "yes", "image"]}}}

    def test_score_questions_are_answered_when_the_engine_is_told_so(self):
        eng = self.engine(question_types=self.E.ALL_TYPES)
        response, raw = eng(**self.with_score())
        answer = response["answers"]["level"]
        self.assertEqual(answer["type"], "score")
        self.assertEqual(list(answer["probabilities"]), ["0", "1", "2"])
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=9)
        self.assertEqual(len(raw["logits"]["level"][0]), 3)

    def test_a_score_question_keeps_its_level_order_under_tta(self):
        request = self.with_score()
        plans = self.tta.plan(request, 2)
        self.assertEqual([p["level"] for p in plans], [[], []])        # an ordered scale is never permuted
        self.assertEqual(plans[1]["route"], [2, 1, 0])
        for plan in plans:
            self.assertEqual(self.tta.permute_request(request, plan)["questions"]["level"], request["questions"]["level"])
        eng = self.engine(question_types=self.E.ALL_TYPES, tta_rotations=2)
        response, raw = eng(**request)
        self.assertEqual(raw["views"], 2)
        views = raw["logits"]["level"]
        self.assertEqual(len(views), 2)                           # the other question changed the shared context, so the logits may differ
        mean = [sum(self.torch.tensor(v).div(self.tm.TEMPERATURE).softmax(-1)[i].item() for v in views) / 2 for i in range(3)]
        for got, want in zip(response["answers"]["level"]["probabilities"].values(), mean):
            self.assertAlmostEqual(got, want, places=6)

    def test_a_request_of_only_a_score_question_runs_one_view(self):
        only = {"state": "alpha", "questions": {"level": self.with_score()["questions"]["level"]}}
        _response, raw = self.engine(question_types=self.E.ALL_TYPES, tta_rotations=2)(**only)
        self.assertEqual(raw["views"], 1)

    def test_the_default_engine_still_serves_only_what_the_kit_scores(self):
        with self.assertRaises(self.E.Unsupported):
            self.engine()(**self.with_score())

    def test_an_unknown_question_type_in_the_engine_options_is_refused(self):
        with self.assertRaises(ValueError):
            self.engine(question_types=("choice", "essay"))

    def test_temperatures_can_be_set_per_type_and_option_count(self):
        eng = self.engine(temperatures={"choice:3-4": 2.0, "choice": 1.0, "default": 1.5})
        _response, raw = eng(**self.tm.REQUEST)                  # route has four options, urgent is a noul
        self.assertEqual(raw["applied_temperatures"], {"route": 2.0, "urgent": 1.5})
        two = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "1", "b": "2"}}}}
        self.assertEqual(eng(**two)[1]["applied_temperatures"], {"q": 1.0})        # two options: the plain choice entry
        plain = self.engine()
        self.assertEqual(plain(**two)[1]["applied_temperatures"], {"q": self.tm.TEMPERATURE})

    def test_raw_output_names_the_answer_codes_of_the_valid_options(self):
        d = self.decider()
        _response, raw = self.engine(d)(**self.tm.REQUEST)
        self.assertEqual(raw["codes"]["route"], d.codes[:4])
        self.assertEqual(raw["codes"]["urgent"], d.codes[:2])
        self.assertEqual(json.loads(json.dumps(raw, allow_nan=False)), raw)

    def test_an_attached_fast_path_replaces_the_reference_path_and_gets_the_view_requests(self):
        d = self.decider()
        reference = self.engine(d, tta_rotations=2)(**self.tm.REQUEST)

        class FakeFast:
            """Stands in for the packed path: answers by the reference computation, and says whether it was running."""

            def __init__(self, decider):
                self.decider, self.calls, self.active = decider, [], False

            def run(self, requests):
                self.calls.append(list(requests))
                self.active = True
                try:
                    for request in requests:
                        plan = self.decider.encoder.plan(request, self.decider.max_length)
                        yield request, self.decider.run_plans([plan])[0], plan
                finally:
                    self.active = False

        d.fast = FakeFast(d)
        real, outside = d.reference_logits, []

        def watch(chunks):
            if not d.fast.active:
                outside.append(len(chunks))
            return real(chunks)

        d.reference_logits = watch
        fast_result = self.E.DecisionEngine(d, tta_rotations=2)(**self.tm.REQUEST)
        self.assertEqual(outside, [])                              # nothing ran on the reference path except through the fast one
        self.assertEqual(len(d.fast.calls), 1)
        views = d.fast.calls[0]
        self.assertEqual(len(views), 2)
        self.assertEqual(list(views[0]["questions"]["route"]["criteria"]), ["book", "ask", "refuse", "defer"])
        self.assertEqual(list(views[1]["questions"]["route"]["criteria"]), ["defer", "refuse", "ask", "book"])
        for key in reference[1]["logits"]:
            for a, b in zip(reference[1]["logits"][key], fast_result[1]["logits"][key]):
                for x, y in zip(a, b):
                    self.assertAlmostEqual(x, y, places=5)


@needs_transformers
class ApiFlow(HermeticTestCase):
    """`load_engine` by name, `run_requests`, `fit_temperature`, `score`."""

    @classmethod
    def setUpClass(cls):
        import torch

        import tiny_model as tm
        from decide.sd import calibrate, engine, evaluate, loader, sample

        cls.torch, cls.tm, cls.E, cls.cal, cls.ev, cls.L, cls.sample = torch, tm, engine, calibrate, evaluate, loader, sample
        cls.hub = FakeHub()
        cls.hub.__enter__()
        cls.sa, cls.sd = tm.backbone_state(1), tm.backbone_state(2)
        cls.head = tm.head_weight(1)
        cls.pa, _ = tm.publish_backbone(cls.hub, "org/anchor", cls.sa, shards=3, head=cls.head, decision=True)
        cls.pd, _ = tm.publish_backbone(cls.hub, "org/donor", cls.sd, shards=2)
        cls.pairs = tm.publish_lora(cls.hub, "org/lora", cls.pa, rank=4, alpha=8.0)
        tm.publish_backbone(cls.hub, "org/d3x", tm.backbone_state(3), shards=2, head=tm.head_weight(3), decision=True, protocol="d3")
        cls.factories = {"repo": hub_repos(cls.hub), "processor": tm.processor_factory}
        cls.reference_head = cls.head.to(torch.bfloat16).float()                     # the head is published as bf16

    @classmethod
    def tearDownClass(cls):
        cls.hub.__exit__(None, None, None)

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.hub.log.clear()

    def cfg(self, **over):
        base = {"hub": {"endpoint": self.hub.base_url},
                "models": {"anchor": {"repo": "org/anchor", "revision": A},
                           "donors": [{"name": "d3", "repo": "org/donor", "weight": 1.0, "enabled": True, "revision": D}],
                           "lora": {"repo": "org/lora", "revision": L, "weight": 0.1, "weight_grid": [0.1], "enabled": True},
                           "rehearsal": {"d3_mini": {"repo": "org/d3x", "revision": X, "protocol": "d3"}, "pplx_flash": {"repo": "org/flash", "revision": X}},
                           "d3": {"repo": "org/d3x", "revision": X, "protocol": "d3"}}}
        C._merge(base, over)
        return make_cfg(**base)

    def engine(self, name="anchor", cfg=None, **kw):
        cfg = cfg or self.cfg()
        spec = name if isinstance(name, dict) else api.spec_for(cfg, name)
        kw.setdefault("quant", "none")
        kw.setdefault("device", "cpu")
        return api.load_engine(spec, cfg=cfg, factories=self.factories, **kw)

    def dense(self, state):
        return self.tm.dense_decider(state, self.reference_head)

    def merged_state(self, weight=0.0, lora=0.0):
        out = {}
        for name, a in self.pa.items():
            a64 = a.double()
            if "embed_tokens" in name or name.startswith("visual."):
                out[name] = a64.float()
                continue
            m = a64.clone()
            m += weight * (self.pd[name].double() - a64)
            if lora and name in self.pairs:
                A, B = self.pairs[name]
                m += (8.0 / 4) * lora * (B.double() @ A.double())
            out[name] = m.float() if a.is_floating_point() else a
        return out

    def assert_logits(self, got, want, places=4):
        for key in want:
            for g, w in zip(got[key][0], want[key]):
                self.assertAlmostEqual(g, w, places=places)

    def test_the_anchor_engine_matches_the_reference_decider(self):
        eng = self.engine("anchor")
        _response, raw = eng(**self.tm.REQUEST)
        self.assert_logits(raw["logits"], self.dense(self.pa).logits([self.tm.REQUEST])[0])
        self.assertEqual(raw["temperature"], self.tm.TEMPERATURE)

    def test_a_merge_candidate_is_the_merge_its_name_promises(self):
        eng = self.engine("ta_l0.5")
        _response, raw = eng(**self.tm.REQUEST)
        self.assert_logits(raw["logits"], self.dense(self.merged_state(weight=0.5)).logits([self.tm.REQUEST])[0])
        plain = self.engine("anchor")(**self.tm.REQUEST)[1]["logits"]["route"][0]
        self.assertNotEqual(plain, raw["logits"]["route"][0])

    def test_the_lora_weight_of_a_candidate_reaches_the_loader(self):
        eng = self.engine("anchor+lora0.1")
        _response, raw = eng(**self.tm.REQUEST)
        self.assert_logits(raw["logits"], self.dense(self.merged_state(lora=0.1)).logits([self.tm.REQUEST])[0])

    def test_provenance_records_the_recipe_and_never_the_token(self):
        secret = "hf_" + "a1b2c3d4e5f6a7b8c9d0"
        eng = self.engine("ta_l0.5", tta_rotations=2, temperatures={"choice": 1.7})
        text = json.dumps(eng.provenance)
        self.assertEqual(eng.provenance["spec"]["donors"], [{"repo": "org/donor", "weight": 0.5, "revision": D}])
        self.assertEqual(eng.provenance["spec"]["anchor_revision"], A)                  # the commit that was read, not "main"
        self.assertEqual(eng.provenance["options"]["quant"], "none")
        self.assertEqual(eng.provenance["fast"], {"enabled": False, "fp8": False, "changes_the_model": False})
        self.assertEqual(eng.provenance["tta"], {"views": 2, "mode": "reverse"})
        self.assertEqual(eng.name, "ta_l0.5")
        self.assertNotIn("hf_", text)
        # the default repo factory (no test hook) reads with the token it was given, and the token goes nowhere else
        keyed = api.load_engine(api.spec_for(self.cfg(), "anchor"), cfg=self.cfg(), quant="none", device="cpu", token=secret,
                                factories={"processor": self.tm.processor_factory})
        self.assertIn(f"Bearer {secret}", self.hub.auth_seen)
        self.assertNotIn(secret, json.dumps(keyed.provenance))

    def test_tta_temperatures_and_question_types_pass_through(self):
        eng = self.engine("anchor", tta_rotations=2, temperatures={"choice": 2.0}, question_types=("choice", "noul", "score"))
        _response, raw = eng(**self.tm.REQUEST)
        self.assertEqual((raw["views"], raw["applied_temperatures"]["route"]), (2, 2.0))
        self.assertEqual(eng.question_types, ("choice", "noul", "score"))

    def test_the_d3_name_loads_the_d3_protocol(self):
        eng = self.engine("d3")
        self.assertEqual(eng.decider.temperature, 1.0)
        self.assertEqual(type(eng.decider.encoder).__name__, "D3Encoder")
        self.assertEqual(eng.provenance["spec"]["protocol"], "d3")

    def test_a_donor_that_may_not_ship_is_refused_before_anything_is_downloaded(self):
        with self.assertRaisesRegex(api.ApiError, "not allowed"):
            api.load_engine({"anchor": "org/anchor", "anchor_revision": A, "donors": [{"repo": "x/lychee-27b", "weight": 1.0, "revision": D}]}, cfg=self.cfg(),
                            factories=self.factories, quant="none", device="cpu")
        self.assertEqual(self.hub.log, [])

    def test_auto_device_on_a_machine_without_a_gpu_loads_on_the_cpu(self):
        nothing = env.Hardware("generic", str(self.dir), (), (), 8.0, 10.0, ())
        eng = self.engine("anchor", device="auto", hardware=nothing)
        self.assertEqual(eng.decider.device.type, "cpu")

    def test_the_fast_path_and_fp8_are_refused_where_they_cannot_work(self):
        with self.assertRaisesRegex(api.ApiError, "fast=True"):
            self.engine("anchor", fp8=True)
        with self.assertRaisesRegex(api.ApiError, "pplx joint"):
            self.engine("d3", fast=True)
        with self.assertRaisesRegex(api.ApiError, "CUDA|fast path"):
            self.engine("anchor", fast=True)

    def test_stored_bytes_counts_the_text_tower_only(self):
        repo = hub_repos(self.hub)("org/anchor")
        want = sum(t.numel() * t.element_size() for n, t in self.pa.items() if not n.startswith("visual."))
        self.assertEqual(self.L.stored_bytes(repo), want)
        self.assertGreater(self.L.stored_bytes(repo, skip=()), want)

    # ---- running, calibrating, scoring

    def gold_rows(self, count=30):
        rows = []
        for i in range(count):
            catalog = 28 if i % 2 == 0 else 29
            keys = ["book", "ask"] if catalog == 28 else ["book", "ask", "refuse", "defer"]
            rows.append({
                "_evaluation": {"run_id": f"{catalog}:T:T:{i}", "catalog_id": catalog, "dataset": "D", "group_id": f"g{i // 2}", "track": "T",
                                "proxy_tokens": 60 + i},
                "state": "alpha beta gamma " * (1 + i % 3),
                "questions": {"route": {"type": "choice", "instructions": "Pick one", "criteria": dict(zip(keys, ["yes", "no", "image", "0"]))},
                              "urgent": {"type": "noul", "instructions": "Is it urgent"}},
                "expected": {"route": keys[(i * 7) % len(keys)], "urgent": bool((i * 5) % 3 == 0)},
                "family": "f", "id": f"id{i}", "metadata": {}, "split": "test",
            })
        return rows

    def write_requests(self, rows, name="req.jsonl"):
        path = self.dir / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    @needs_pyarrow
    def test_run_requests_writes_rows_logits_and_a_probability_table_with_codes(self):
        import pyarrow.parquet as pq

        rows = self.gold_rows(6)
        out = api.run_requests(self.engine("anchor", tta_rotations=2), self.write_requests(rows), self.dir / "run", log=None)
        self.assertEqual((out["rows"], out["status"]), (6, "complete"))
        self.assertGreater(out["tokens"], 0)
        self.assertIn("seconds", out)
        table = pq.read_table(self.dir / "run" / self.ev.TABLE_NAME).to_pylist()
        self.assertEqual(len(table), 12)
        route = next(t for t in table if t["question_id"] == "route" and t["catalog_id"] == 29)
        self.assertEqual(route["option_keys"], ["book", "ask", "refuse", "defer"])
        self.assertEqual(len(route["codes"]), 4)
        self.assertEqual(route["views"], 2)
        self.assertEqual(len(route["logits"]), 2)
        self.assertIsNotNone(route["gold"])

    @needs_pyarrow
    def test_a_second_call_resumes_and_does_no_work(self):
        rows = self.gold_rows(6)
        eng = self.engine("anchor")
        path = self.write_requests(rows)
        api.run_requests(eng, path, self.dir / "run", log=None)
        again = api.run_requests(eng, path, self.dir / "run", log=None)
        self.assertEqual((again["rows"], again["skipped"]), (0, 6))

    @needs_pyarrow
    def test_fitting_from_the_table_gives_what_fitting_from_the_raw_logits_gives(self):
        rows = self.gold_rows(40)
        eng = self.engine("anchor", tta_rotations=2)
        api.run_requests(eng, self.write_requests(rows), self.dir / "run", log=None)
        fit = api.fit_temperature(self.dir / "run" / self.ev.TABLE_NAME, min_items=5, seed=3, allow_evaluation_data=True)      # suite rows
        items = self.ev.calibration_items(self.dir / "run", rows)
        direct = self.cal.fit_groups(items, by="question_type", holdout=0.5, seed=3, baseline=self.tm.TEMPERATURE, grid=(0.5, 4.0), min_items=5)
        for key in ("temperatures", "default", "n_fit", "n_holdout", "fit_counts", "holdout", "baseline"):
            self.assertEqual(fit[key], direct[key], key)
        self.assertEqual(sorted(fit["temperatures"]), ["choice", "noul"])
        self.assertEqual(fit["n_items"], len(items))
        self.assertTrue(fit["source"].endswith(self.ev.TABLE_NAME))
        self.assertEqual(api.fit_temperature(self.dir / "run", min_items=5, seed=3, allow_evaluation_data=True)["temperatures"],
                         fit["temperatures"])
        self.assertEqual((fit["fitted_on"], fit["evaluation_rows"]), ({"public-suite": 80}, 80))

    def dev_rows(self, count=30):
        """The same questions as plain request rows (a dev split of a training corpus), not rows of the public suite."""
        return [{"example_id": f"dev{i}", "request": {k: row[k] for k in ("state", "questions")}, "gold": row["expected"]}
                for i, row in enumerate(self.gold_rows(count))]

    @needs_pyarrow
    def test_a_run_on_the_public_suite_cannot_be_fitted_but_a_run_on_dev_rows_can(self):
        api.run_requests(self.engine("anchor"), self.write_requests(self.gold_rows(12), "suite.jsonl"), self.dir / "suite", log=None)
        api.run_requests(self.engine("anchor"), self.write_requests(self.dev_rows(12), "dev.jsonl"), self.dir / "dev", log=None)
        with self.assertRaisesRegex(api.ApiError, "public-suite"):
            api.fit_temperature(self.dir / "suite", min_items=3)
        fit = api.fit_temperature(self.dir / "dev", min_items=3)
        self.assertEqual((fit["fitted_on"], fit["evaluation_rows"]), ({"requests": 24}, 0))
        env = json.loads((self.dir / "suite" / self.ev.ENVIRONMENT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(env["origins"], {"public-suite": 12})

    @needs_pyarrow
    def test_the_fitted_temperatures_go_back_into_an_engine(self):
        rows = self.dev_rows(40)
        api.run_requests(self.engine("anchor"), self.write_requests(rows), self.dir / "run", log=None)
        fit = api.fit_temperature(self.dir / "run", min_items=5)
        eng = self.engine("anchor", temperatures=fit["temperatures"])
        raw = eng(**self.tm.REQUEST)[1]
        self.assertAlmostEqual(raw["applied_temperatures"]["route"], fit["temperatures"]["choice"])
        self.assertAlmostEqual(raw["applied_temperatures"]["urgent"], fit["temperatures"]["noul"])

    @needs_pyarrow
    def test_a_run_without_gold_labels_cannot_be_fitted(self):
        rows = [{k: v for k, v in r.items() if k != "expected"} for r in self.gold_rows(4)]
        api.run_requests(self.engine("anchor"), self.write_requests(rows), self.dir / "run", log=None)
        with self.assertRaisesRegex(api.ApiError, "gold"):
            api.fit_temperature(self.dir / "run")

    @needs_kit
    @needs_pyarrow
    def test_score_finds_the_sample_that_sits_next_to_the_run(self):
        rows = self.gold_rows(24)
        selection = self.sample.select(rows, per_benchmark=6, seed=4, edition="0.3")
        self.sample.write(selection, self.dir / "sample")
        metrics = self.ev.evaluate_sample(self.engine("anchor"), self.dir / "sample", self.dir / "out", log=None)
        scored = api.score(self.dir / "out" / "run")
        self.assertAlmostEqual(scored["public"], metrics["scores"]["public"])
        for key in ("public", "areas", "per_benchmark"):
            self.assertIn(key, scored)
        explicit = api.score(self.dir / "out" / "run" / self.ev.RESULTS_NAME, subset_manifest=self.dir / "out" / "subset")
        self.assertAlmostEqual(explicit["public"], scored["public"])

    def test_score_without_a_sample_says_what_to_pass(self):
        run = self.dir / "elsewhere"
        run.mkdir()
        (run / self.ev.RESULTS_NAME).write_text("", encoding="utf-8")
        with self.assertRaisesRegex(api.ApiError, "subset_manifest"):
            api.score(run)
        with self.assertRaisesRegex(api.ApiError, "no results file"):
            api.score(self.dir / "missing")


if __name__ == "__main__":
    unittest.main()
