"""Flow tests for the d3 prompt path (issue #66 adds the fidelity check on top of it in this same file).

The tiny checkpoint has random weights, so the numbers mean nothing.  What is checked: that the d3 path feeds the d3
prompt through the backbone and reads the last token in float32, against a reference written independently in the test.
"""
from __future__ import annotations

import json

from decide_support import FakeHub, HermeticTestCase, needs_transformers

from decide.sd import remote


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
