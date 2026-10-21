"""Flow tests for issue #66 (F1, fidelity): the d3 prompt path on a tiny backbone, and a fidelity check end to end.

The tiny checkpoint has random weights, so the numbers mean nothing.  What is checked: that the d3 path feeds the d3
prompt through the backbone and reads the last token in float32 (against a reference written independently in the
test), and that `check_fidelity` compares the right rows, applies the gate and survives a stop and a resume.
"""
from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from decide_support import FakeHub, HermeticTestCase, needs_transformers

from decide.sd import remote


def kit_rows(count=10):
    """Suite-style rows.  A group is two rows of one benchmark; benchmark 28 has two options, benchmark 29 has four."""
    rows = []
    for i in range(count):
        group = i // 2
        catalog = 28 if group % 2 == 0 else 29
        keys = ["book", "ask"] if catalog == 28 else ["book", "ask", "refuse", "defer"]
        rows.append({
            "_evaluation": {"run_id": f"{catalog}:T:T:{i}", "catalog_id": catalog, "dataset": "D", "group_id": f"g{group}", "track": "T",
                            "proxy_tokens": 80 + i, "payload_sha256": f"hash{i}"},
            "state": "alpha beta gamma " * (1 + i % 3),
            "questions": {"route": {"type": "choice", "instructions": "Pick one", "criteria": dict(zip(keys, ["yes", "no", "image", "0"]))},
                          "urgent": {"type": "noul", "instructions": "Is it urgent"}},
            "expected": {"route": keys[0], "urgent": True}, "family": "f", "id": f"id{i}", "metadata": {}, "split": "test",
        })
    return rows


def hub_repos(hub):
    return lambda repo_id, revision: remote.HubRepo(repo_id, revision, endpoint=hub.base_url, sleep=lambda s: None, probe=4096)


@needs_transformers
class D3Flow(HermeticTestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        import tiny_model as tm
        from decide.sd import engine, loader
        from decide.sd.vendor.d3 import d3_format

        cls.torch, cls.tm, cls.E, cls.L, cls.fmt = torch, tm, engine, loader, d3_format
        cls.hub = FakeHub()
        cls.hub.__enter__()
        cls.state, cls.head = tm.backbone_state(4), tm.head_weight(4)
        cls.published, _ = tm.publish_backbone(cls.hub, "org/d3", cls.state, shards=2, head=cls.head, decision=True, protocol="d3")
        cls.tok, cls.codes, _ids = tm.tokenizer_and_codes()
        cls.dense = tm.dense_decider(cls.published, cls.head.to(torch.bfloat16).float())          # the head is published as bf16

    @classmethod
    def tearDownClass(cls):
        cls.hub.__exit__(None, None, None)

    def load(self, spec=None, **opts):
        opts = {"quant": "none", "compute_dtype": "float32", "workers": 2, **opts}
        spec = {"anchor_revision": "d" * 40, **(spec or {"anchor": "org/d3", "protocol": "d3"})}               # a repo is read at a commit, never at "main"
        return self.L.load_decider(spec, self.L.LoadOptions(**opts),
                                   repo_factory=hub_repos(self.hub), processor_factory=self.tm.processor_factory)

    def reference_logits(self, state_text, question, n):
        """The d3 recipe written out independently: render, tokenize, backbone, last hidden state, float32 head."""
        torch = self.torch
        text = self.fmt.render(self.tok, state_text, question, self.codes)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        hidden = self.dense.backbone(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                     use_cache=False).last_hidden_state[0, -1]
        head = self.head.to(torch.bfloat16).float()
        return (hidden.float() @ head.t())[:n].tolist()

    @staticmethod
    def request():
        return {"state": "alpha beta gamma", "questions": {
            "route": {"type": "choice", "instructions": "Pick one", "criteria": {"book": "yes", "ask": "no", "refuse": "image"}},
            "urgent": {"type": "noul", "instructions": "Is it urgent"}}}

    def test_the_loader_builds_a_d3_decider_with_the_checkpoints_codes_and_temperature_one(self):
        decider = self.load()
        self.assertEqual(decider.temperature, 1.0)
        self.assertEqual(len(decider.codes), 255)
        self.assertEqual(decider.readout.weight.dtype, self.torch.float32)
        self.assertIsNone(decider.max_length)

    def test_each_question_is_its_own_sequence_with_the_readout_on_the_last_token(self):
        decider = self.load()
        request = self.request()
        chunks = decider.encoder.plan(request, decider.max_length)
        self.assertEqual([c.keys for c in chunks], [("route",), ("urgent",)])
        self.assertEqual([c.counts for c in chunks], [(3,), (2,)])
        for chunk, question in zip(chunks, request["questions"].values()):
            text = self.fmt.render(self.tok, request["state"], question, self.codes)
            self.assertEqual(chunk.input_ids.tolist(), self.tok(text, add_special_tokens=False)["input_ids"])
            self.assertEqual(chunk.markers.tolist(), [chunk.length - 1])

    def test_logits_equal_an_independent_last_token_float32_readout(self):
        engine = self.E.DecisionEngine(self.load())
        request = self.request()
        response, raw = engine(request["state"], request["questions"])
        for key, n in (("route", 3), ("urgent", 2)):
            want = self.reference_logits(request["state"], request["questions"][key], n)
            got = raw["logits"][key][0]
            self.assertEqual(len(got), n)
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=4)
        want = self.torch.tensor(self.reference_logits(request["state"], request["questions"]["route"], 3)).softmax(-1).tolist()
        for got, expected in zip(response["answers"]["route"]["probabilities"].values(), want):         # temperature 1: the plain softmax
            self.assertAlmostEqual(got, expected, places=5)

    def test_tta_reverse_is_d3s_permutation_average(self):
        engine = self.E.DecisionEngine(self.load(), tta_rotations=2, tta_mode="reverse")
        request = self.request()
        response, _raw = engine(request["state"], request["questions"])
        question = request["questions"]["route"]
        forward = self.torch.tensor(self.reference_logits(request["state"], question, 3)).softmax(-1).tolist()
        reversed_question = dict(question, criteria={k: question["criteria"][k] for k in reversed(list(question["criteria"]))})
        backward = self.torch.tensor(self.reference_logits(request["state"], reversed_question, 3)).softmax(-1).tolist()
        mean = [(a + b) / 2 for a, b in zip(forward, reversed(backward))]                   # d3's average over the original and the reversed order
        for got, want in zip(response["answers"]["route"]["probabilities"].values(), mean):
            self.assertAlmostEqual(got, want, places=5)

    def test_bfloat16_compute_keeps_the_head_in_float32(self):
        decider = self.load(compute_dtype="bfloat16")
        self.assertEqual(decider.readout.weight.dtype, self.torch.float32)
        rows = decider.reference_logits(decider.encoder.plan(self.request(), None)[:1])
        self.assertEqual(len(rows[0][0]), 3)
        self.assertTrue(all(abs(x) < 1e4 for x in rows[0][0]))

    def test_nf4_weights_still_answer_through_the_d3_path(self):
        dense = self.E.DecisionEngine(self.load())
        quant = self.E.DecisionEngine(self.load(quant="nf4", absmax_dtype="float32"))
        request = self.request()
        a = dense(request["state"], request["questions"])[1]["logits"]["route"][0]
        b = quant(request["state"], request["questions"])[1]["logits"]["route"][0]
        self.assertEqual(len(a), len(b))
        self.assertTrue(all(abs(x) < 1e3 for x in b))

    def test_a_prompt_over_the_limit_is_unsupported_and_nothing_is_truncated(self):
        engine = self.E.DecisionEngine(self.load(max_length=12))
        request = self.request()
        with self.assertRaises(self.E.Unsupported):
            engine(request["state"], request["questions"])

    def test_other_attention_modes_and_formats_are_refused_by_name(self):
        good = json.loads(self.hub.files["org/d3/decision_config.json"])
        for repo, change, name in (("org/turbo", {"attention_mode": "prefix_bidir"}, "prefix_bidir"),
                                   ("org/other", {"format_id": "something-else"}, "d3-code-readout-v1")):
            self.hub.add(repo, "decision_config.json", json.dumps({**good, **change}).encode())
            with self.subTest(repo), self.assertRaisesRegex(self.L.LoaderError, name):
                self.load({"anchor": repo, "protocol": "d3"})
        with self.assertRaisesRegex(self.L.LoaderError, "protocol"):
            self.load({"anchor": "org/d3", "protocol": "nope"})

    def test_a_d3_checkpoint_loaded_as_pplx_says_what_is_wrong(self):
        with self.assertRaisesRegex(self.L.LoaderError, "joint_config.json.*'d3'"):
            self.load({"anchor": "org/d3"})

    def test_questions_other_than_choice_and_noul_are_unsupported(self):
        engine = self.E.DecisionEngine(self.load())
        with self.assertRaises(self.E.Unsupported):
            engine("s", {"q": {"type": "score", "criteria": ["a", "b"]}})


@needs_transformers
class FidelityFlow(HermeticTestCase):
    """Engines hold no state between requests, so the two tiny models are built once and every test shares them."""

    @classmethod
    def setUpClass(cls):
        import tiny_model as tm
        from decide.sd import engine, evaluate, fidelity, loader

        cls.tm, cls.E, cls.ev, cls.fid, cls.L = tm, engine, evaluate, fidelity, loader
        cls.hub = FakeHub()
        cls.hub.__enter__()
        state = tm.backbone_state(5)
        tm.publish_backbone(cls.hub, "org/d3", state, shards=2, head=tm.head_weight(5), decision=True, protocol="d3")
        tm.publish_backbone(cls.hub, "org/d3-other-head", state, shards=2, head=tm.head_weight(99), decision=True, protocol="d3")
        cls.rows = kit_rows(10)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.req = cls.dir / "req.jsonl"
        cls.req.write_text("".join(json.dumps(r) + "\n" for r in cls.rows), encoding="utf-8")
        cls.engine = cls.make_engine("org/d3")
        cls.results = {"org/d3": cls.run_rows(cls.engine, "same"),
                       "org/d3-other-head": cls.run_rows(cls.make_engine("org/d3-other-head"), "other")}

    @classmethod
    def tearDownClass(cls):
        cls.hub.__exit__(None, None, None)
        cls.tmp.cleanup()

    @classmethod
    def make_engine(cls, repo):
        decider = cls.L.load_decider({"anchor": repo, "anchor_revision": "d" * 40, "protocol": "d3"}, cls.L.LoadOptions(quant="none", compute_dtype="float32", workers=2),
                                     repo_factory=hub_repos(cls.hub), processor_factory=cls.tm.processor_factory)
        return cls.E.DecisionEngine(decider, name=f"tiny:{repo}")

    @classmethod
    def run_rows(cls, engine, name):
        run = cls.dir / f"{name}-run"
        cls.ev.run_requests(engine, cls.req, run, log=None, table=False)
        return [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()]

    def setUp(self):
        super().setUp()
        self._out = tempfile.TemporaryDirectory()
        self.addCleanup(self._out.cleanup)
        self.out = Path(self._out.name)

    def publish(self, repo="org/d3", keep=None, change=None):
        """What the authors would have published: a results.jsonl.gz made by the model of `repo` over the same requests."""
        rows = [change(r) if change else r for r in self.results[repo] if keep is None or keep(r)]
        path = self.out / "published.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for r in rows:
                handle.write(json.dumps(r) + "\n")
        return path

    def check(self, published, engine=None, **kw):
        args = dict(threshold=0.98, min_rows=10, chunk=4, log=None, replicates=20)
        args.update(kw)
        return self.fid.check_fidelity(engine or self.engine, self.req, published, self.out / "fid", **args)

    def test_the_same_model_agrees_with_itself_on_every_question(self):
        report = self.check(self.publish())
        self.assertEqual((report["compare"]["rows"], report["compare"]["fields"]), (10, 20))
        self.assertEqual(report["compare"]["field_agreement"], 1.0)
        self.assertEqual(report["compare"]["row_agreement"], 1.0)
        self.assertLess(report["compare"]["prob"]["max_abs"], 1e-4)
        self.assertEqual(report["gate"]["status"], "pass")
        self.assertTrue(report["complete"])
        self.assertEqual(report["gate"]["coverage"], {"requested": 10, "ok": 10, "error": 0, "unsupported": 0, "abstained": 0, "missing": 0, "complete": True})
        self.assertIn("10 of 10", report["gate"]["reason"])
        saved = json.loads((self.out / "fid" / "fidelity.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["gate"]["status"], "pass")
        self.assertEqual(saved["gate"]["coverage"], report["gate"]["coverage"])
        self.assertEqual(saved["engine"]["tta"], {"views": 1, "mode": "reverse"})

    def test_a_model_with_another_head_disagrees_and_the_gate_fails(self):
        report = self.check(self.publish("org/d3-other-head"), threshold=0.99)
        self.assertLess(report["compare"]["field_agreement"], 0.99)
        self.assertEqual(report["gate"]["status"], "fail")
        self.assertTrue(report["compare"]["disagreements"])
        self.assertFalse(report["gate"]["firm"])

    def test_rows_with_another_payload_hash_or_no_published_answer_are_left_out_before_any_work(self):
        published = self.publish(keep=lambda r: not r["run_id"].endswith(":3"),
                                 change=lambda r: dict(r, payload_sha256="different") if r["run_id"].endswith(":2") else r)
        seen, original = [], self.engine.batch
        with mock.patch.object(self.engine, "batch", side_effect=lambda requests: (seen.extend(r["state"] for r in requests), original(requests))[1]):
            report = self.check(published, threshold=0.9, min_rows=5)
        self.assertEqual(report["requests"]["skipped"], {"not_published": 1, "published_not_ok": 0, "payload_mismatch": 1, "unmatched": 0})
        self.assertEqual((report["compare"]["rows"], report["run"]["total"], len(seen)), (8, 8, 8))

    def test_a_row_with_no_payload_hash_on_the_published_side_is_left_out_before_any_work_and_counted(self):
        published = self.publish(change=lambda r: {k: v for k, v in r.items() if k != "payload_sha256"} if r["run_id"].endswith((":2", ":6")) else r)
        seen, original = [], self.engine.batch
        with mock.patch.object(self.engine, "batch", side_effect=lambda requests: (seen.extend(r["state"] for r in requests), original(requests))[1]):
            report = self.check(published, threshold=0.9, min_rows=5)
        self.assertEqual(report["requests"]["skipped"], {"not_published": 0, "published_not_ok": 0, "payload_mismatch": 0, "unmatched": 2})
        self.assertEqual((report["compare"]["rows"], report["run"]["total"], len(seen)), (8, 8, 8))

    def test_a_row_with_no_payload_hash_on_our_side_is_left_out_too(self):
        rows = [dict(r, _evaluation={k: v for k, v in r["_evaluation"].items() if k != "payload_sha256"}) if i < 3 else r for i, r in enumerate(self.rows)]
        req = self.out / "unhashed.jsonl"
        req.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        report = self.fid.check_fidelity(self.engine, req, self.publish(), self.out / "fid-unhashed", threshold=0.9, min_rows=5, chunk=4, log=None, replicates=10)
        self.assertEqual(report["requests"]["skipped"]["unmatched"], 3)
        self.assertEqual((report["compare"]["rows"], report["run"]["total"]), (7, 7))

    def test_requests_that_end_in_error_withhold_the_verdict_even_when_the_rest_agrees(self):
        original, failed = self.engine.batch, []

        def flaky(requests):
            if len(requests) > 1:
                raise RuntimeError("the batch broke")                  # the run then asks again one request at a time
            if len(failed) < 4:
                failed.append(1)
                raise RuntimeError("this request broke")
            return original(requests)

        with mock.patch.object(self.engine, "batch", side_effect=flaky):
            report = self.check(self.publish(), threshold=0.98, min_rows=5)
        self.assertEqual((report["compare"]["rows"], report["compare"]["field_agreement"]), (6, 1.0))           # the 6 that answered all agree
        self.assertEqual(report["gate"]["status"], "insufficient")
        self.assertFalse(report["gate"]["firm"])
        self.assertEqual(report["gate"]["coverage"], {"requested": 10, "ok": 6, "error": 4, "unsupported": 0, "abstained": 0, "missing": 0, "complete": True})
        self.assertIn("4 request(s) ended in error", report["gate"]["reason"])
        self.assertIn("6 of 10", report["gate"]["reason"])
        saved = json.loads((self.out / "fid" / "fidelity.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["gate"]["coverage"]["error"], 4)

    def test_a_second_call_retries_the_errors_and_then_gives_the_verdict(self):
        original, failed = self.engine.batch, []

        def flaky(requests):
            if len(requests) > 1 or len(failed) < 4:
                failed.append(1)
                raise RuntimeError("broken")
            return original(requests)

        published = self.publish()
        with mock.patch.object(self.engine, "batch", side_effect=flaky):
            first = self.check(published, threshold=0.98, min_rows=5)
        self.assertEqual(first["gate"]["status"], "insufficient")
        second = self.check(published, threshold=0.98, min_rows=5)
        self.assertEqual((second["gate"]["status"], second["gate"]["coverage"]["error"], second["compare"]["rows"]), ("pass", 0, 10))

    def test_a_run_stopped_by_its_budget_has_no_verdict_whatever_the_rows_it_did_answer_say(self):
        class OneChunk:
            def __init__(self):
                self.calls = 0

            def ok(self, next_cost=0.0):
                self.calls += 1
                return self.calls <= 1

        report = self.check(self.publish(), budget=OneChunk(), threshold=0.98, min_rows=3)
        self.assertEqual((report["complete"], report["compare"]["rows"], report["compare"]["field_agreement"]), (False, 4, 1.0))
        self.assertEqual(report["gate"]["status"], "insufficient")
        self.assertFalse(report["gate"]["firm"])
        self.assertEqual(report["gate"]["coverage"], {"requested": 10, "ok": 4, "error": 0, "unsupported": 0, "abstained": 0, "missing": 6, "complete": False})
        self.assertIn("stopped", report["gate"]["reason"])
        self.assertIn("4 of 10", report["gate"]["reason"])

    def test_requests_the_engine_would_not_take_stay_out_of_the_agreement_and_are_counted(self):
        original, taken = self.engine.batch, []

        def picky(requests):
            if len(requests) > 1:
                raise RuntimeError("the batch broke")
            if len(taken) < 3:
                taken.append(1)
                raise self.ev.Unsupported("over the limit")
            return original(requests)

        with mock.patch.object(self.engine, "batch", side_effect=picky):
            report = self.check(self.publish(), threshold=0.98, min_rows=5)
        self.assertEqual((report["gate"]["status"], report["compare"]["rows"]), ("pass", 7))
        self.assertEqual(report["gate"]["coverage"]["unsupported"], 3)
        self.assertIn("3 unsupported", report["gate"]["reason"])

    def test_a_published_row_that_is_not_ok_is_not_compared(self):
        report = self.check(self.publish(change=lambda r: dict(r, status="error") if r["run_id"].endswith(":4") else r), min_rows=5)
        self.assertEqual(report["requests"]["skipped"]["published_not_ok"], 1)
        self.assertEqual(report["compare"]["rows"], 9)

    def test_fewer_rows_than_the_gate_needs_is_reported_as_insufficient(self):
        report = self.check(self.publish(), min_rows=500, limit=6)
        self.assertEqual(report["compare"]["rows"], 6)
        self.assertEqual(report["gate"]["status"], "insufficient")
        self.assertIn("500", report["gate"]["reason"])

    def test_nothing_in_common_is_an_error_not_a_pass(self):
        path = self.out / "elsewhere.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"run_id": "elsewhere", "status": "ok", "response": {"answers": {}}}) + "\n")
        with self.assertRaisesRegex(self.fid.FidelityError, "none of the 10 requests"):
            self.check(path)

    def test_a_stopped_run_can_be_resumed_and_gives_the_whole_report(self):
        published = self.publish()

        class OneChunk:
            def __init__(self):
                self.calls = 0

            def ok(self, next_cost=0.0):
                self.calls += 1
                return self.calls <= 1

        first = self.check(published, budget=OneChunk())
        self.assertFalse(first["complete"])
        self.assertEqual(first["compare"]["rows"], 4)
        second = self.check(published)
        self.assertTrue(second["complete"])
        self.assertEqual((second["compare"]["rows"], second["compare"]["field_agreement"]), (10, 1.0))

    def test_common_sample_keeps_only_groups_the_published_run_answered(self):
        published = self.fid.load_published(self.publish(keep=lambda r: not r["run_id"].endswith(":3")))
        selection = self.fid.common_sample(self.rows, published, per_benchmark=10, seed=1, edition="0.3")
        ids = {r["_evaluation"]["run_id"] for r in selection.rows}
        self.assertEqual(len(ids), 8)
        self.assertNotIn("29:T:T:3", ids)
        self.assertNotIn("29:T:T:2", ids)                           # the other row of the same group: a group is all or nothing

    def test_common_sample_drops_a_group_with_a_row_that_has_no_hash_on_either_side(self):
        loaded = self.fid.load_published(self.publish(change=lambda r: {k: v for k, v in r.items() if k != "payload_sha256"} if r["run_id"].endswith(":5") else r))
        selection = self.fid.common_sample(self.rows, loaded, per_benchmark=10, seed=1, edition="0.3")
        ids = {r["_evaluation"]["run_id"] for r in selection.rows}
        self.assertEqual(len(ids), 8)
        self.assertNotIn("29:T:T:5", ids)
        self.assertNotIn("29:T:T:4", ids)                           # the other row of the same group: a group is all or nothing
        unhashed = [dict(r, _evaluation={k: v for k, v in r["_evaluation"].items() if k != "payload_sha256"}) if i == 0 else r for i, r in enumerate(self.rows)]
        selection = self.fid.common_sample(unhashed, self.fid.load_published(self.publish()), per_benchmark=10, seed=1, edition="0.3")
        self.assertEqual(len({r["_evaluation"]["run_id"] for r in selection.rows}), 8)

    def test_common_sample_drops_a_group_whose_published_row_was_not_ok(self):
        published = self.fid.load_published(self.publish(change=lambda r: {**r, "status": "error"} if r["run_id"].endswith(":5") else r))
        selection = self.fid.common_sample(self.rows, published, per_benchmark=10, seed=1, edition="0.3")
        ids = {r["_evaluation"]["run_id"] for r in selection.rows}
        self.assertEqual(len(ids), 8)
        self.assertNotIn("29:T:T:5", ids)
        self.assertNotIn("29:T:T:4", ids)


if __name__ == "__main__":
    unittest.main()
