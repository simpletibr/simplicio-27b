"""Flow tests for issue #65 (F0, kit core): bytes and tensors travel from a fake Hub to the merge.

A real HTTP server (tests/decide_support.FakeHub) serves real safetensors bytes with Range support and
injected faults.  Pure standard library plus numpy; the tests that turn bytes into torch tensors skip
cleanly when torch is missing.
"""

from __future__ import annotations

import json
import random
import struct
import tempfile
import unittest
from pathlib import Path

from decide_support import FakeHub, HermeticTestCase, build_safetensors, needs_kit, needs_requests, needs_torch, needs_transformers, numpy_tensors

from decide.sd import remote


@needs_requests
class RemoteFlow(HermeticTestCase):
    def setUp(self):
        super().setUp()
        self.tensors = numpy_tensors()
        self.blob = build_safetensors(self.tensors, metadata={"format": "pt"})
        self.hub = FakeHub()
        self.hub.__enter__()
        self.addCleanup(self.hub.__exit__, None, None, None)
        h = self.hub
        h.add("org/model", "model.safetensors", self.blob)
        names = list(self.tensors)
        s1 = {k: self.tensors[k] for k in names[:3]}
        s2 = {k: self.tensors[k] for k in names[3:]}
        h.add("org/sharded", "model-00001-of-00002.safetensors", build_safetensors(s1))
        h.add("org/sharded", "model-00002-of-00002.safetensors", build_safetensors(s2))
        wm = {k: "model-00001-of-00002.safetensors" for k in s1} | {k: "model-00002-of-00002.safetensors" for k in s2}
        h.add("org/sharded", "model.safetensors.index.json", json.dumps({"metadata": {}, "weight_map": wm}).encode())
        h.add("org/model", "config.json", json.dumps({"hidden_size": 16}).encode())

    def repo(self, name="org/model", **kw):
        return remote.HubRepo(name, endpoint=self.hub.base_url, sleep=lambda s: None, **kw)

    def raw(self, r, name):
        return bytes(r.shard(r.shard_of(name)).read_raw(name))

    def test_header_parse_matches_safetensors_layout(self):
        sf = self.repo().shard("model.safetensors")
        n = struct.unpack("<Q", self.blob[:8])[0]
        real = json.loads(self.blob[8:8 + n])
        self.assertEqual(set(sf.names()), set(self.tensors))
        for name, (dtype, shape, _raw) in self.tensors.items():
            info = sf.info(name)
            self.assertEqual((info.dtype, info.shape), (dtype, tuple(shape)))
            self.assertEqual((info.start, info.end), (8 + n + real[name]["data_offsets"][0], 8 + n + real[name]["data_offsets"][1]))
        self.assertEqual(sf.metadata, {"format": "pt"})

    def test_read_every_tensor_exactly(self):
        r = self.repo()
        for name, (_dtype, _shape, data) in self.tensors.items():
            with self.subTest(name):
                self.assertEqual(self.raw(r, name), data)

    def test_reading_one_tensor_does_not_download_the_shard(self):
        r = self.repo()
        r.shard("model.safetensors")
        self.hub.log.clear()
        r.shard("model.safetensors").read_raw("language_model.layers.0.input_layernorm.weight")
        served = self.hub.bytes_served("org/model/model.safetensors")
        self.assertLess(served, len(self.blob) / 2)
        self.assertTrue(all(rng.startswith("bytes=") for _k, rng, _n in self.hub.log))

    def test_row_slice_of_a_matrix_is_contiguous_and_exact(self):
        name = "language_model.layers.0.mlp.gate_proj.weight"
        _dtype, shape, data = self.tensors[name]
        row = len(data) // shape[0]
        part = self.repo().shard("model.safetensors").read_raw(name, row_start=5, row_end=11)
        self.assertEqual(bytes(part), data[5 * row:11 * row])

    def test_sharded_repo_weight_map_and_cross_shard_reads(self):
        r = self.repo("org/sharded")
        self.assertEqual(set(r.names()), set(self.tensors))
        self.assertTrue(r.shard_of("visual.blocks.0.attn.proj.bias").endswith("00002-of-00002.safetensors"))
        for name, (_d, _s, data) in self.tensors.items():
            self.assertEqual(self.raw(r, name), data)

    def test_stream_coalesces_adjacent_tensors_into_few_requests(self):
        r = self.repo()
        r.shard("model.safetensors")
        self.hub.log.clear()
        names = list(self.tensors)
        got = {n: bytes(t) for n, t in self._stream_raw(r, names, window_bytes=1 << 20)}
        self.assertEqual(set(got), set(self.tensors))
        data_requests = [x for x in self.hub.log if x[0].endswith("model.safetensors")]
        self.assertLessEqual(len(data_requests), 4)    # one coalesced window or two, not one per tensor

    def _stream_raw(self, r, names, **kw):
        """stream() builds torch tensors; this variant drives the same windows but keeps raw bytes."""
        by_shard = {}
        for n in names:
            by_shard.setdefault(r.shard_of(n), []).append(n)
        for shard_name, members in by_shard.items():
            sf = r.shard(shard_name)
            for window in sf.plan_windows(members, kw.get("window_bytes", 1 << 20), 1 << 20):
                s, e, group = window
                buf = sf.transport.fetch(sf.url, s, e - 1)
                for n in group:
                    i = sf.info(n)
                    yield n, bytes(buf[i.start - s:i.end - s])

    def test_retries_on_503_with_backoff(self):
        self.hub.fail_first["org/model/model.safetensors"] = 3
        sleeps = []
        r = remote.HubRepo("org/model", endpoint=self.hub.base_url, sleep=sleeps.append)
        self.assertEqual(self.raw(r, "mask"), self.tensors["mask"][2])
        self.assertEqual(len(sleeps), 3)
        self.assertTrue(all(x > 0 for x in sleeps))
        self.assertEqual(sleeps, sorted(sleeps))

    def test_gives_up_after_max_retries(self):
        self.hub.fail_first["org/model/model.safetensors"] = 99
        r = remote.HubRepo("org/model", endpoint=self.hub.base_url, sleep=lambda s: None, max_retries=2)
        with self.assertRaisesRegex(remote.RemoteError, "503"):
            r.shard("model.safetensors")

    def test_truncated_body_is_resumed_not_returned_short(self):
        r = self.repo()
        r.shard("model.safetensors")           # header first
        self.hub.truncate_once["org/model/model.safetensors"] = 7
        name = "language_model.embed_tokens.weight"
        self.assertEqual(self.raw(r, name), self.tensors[name][2])

    def test_server_ignoring_range_is_an_error_not_a_full_download(self):
        self.hub.ignore_range.add("org/model/model.safetensors")
        with self.assertRaisesRegex(remote.RemoteError, "Range"):
            self.repo().shard("model.safetensors")

    def test_redirect_keeps_range_and_token_is_checked_at_the_origin(self):
        self.hub.redirect.add("org/model/model.safetensors")
        self.hub.require_token = "tok"
        r = remote.HubRepo("org/model", endpoint=self.hub.base_url, token="tok", sleep=lambda s: None)
        name = "language_model.embed_tokens.weight"
        self.assertEqual(self.raw(r, name), self.tensors[name][2])
        r2 = remote.HubRepo("org/model", endpoint=self.hub.base_url, token="wrong", sleep=lambda s: None)
        with self.assertRaisesRegex(remote.RemoteError, "401"):
            r2.shard("model.safetensors")

    def test_missing_file_and_unknown_tensor(self):
        r = self.repo()
        with self.assertRaisesRegex(remote.RemoteError, "404"):
            r.read_json("nope.json")
        with self.assertRaises(KeyError):
            r.shard("model.safetensors").read_raw("does.not.exist")
        self.assertEqual(r.read_json("config.json"), {"hidden_size": 16})

    def test_local_transport_on_a_real_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.safetensors"
            path.write_bytes(self.blob)
            sf = remote.SafetensorsFile(str(path), remote.LocalTransport())
            for name, (_d, _s, data) in self.tensors.items():
                self.assertEqual(bytes(sf.read_raw(name)), data)

    def test_dtype_table_covers_safetensors_dtypes(self):
        for k in ("F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"):
            self.assertIn(k, remote.DTYPE_SIZES)
        self.assertEqual((remote.DTYPE_SIZES["BF16"], remote.DTYPE_SIZES["F32"]), (2, 4))

    def test_importing_the_reader_does_not_import_torch(self):
        import subprocess
        import sys

        from decide_support import ROOT

        code = "import sys; import decide.sd.remote; sys.exit(1 if 'torch' in sys.modules else 0)"
        run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": ""},
                             capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)


@needs_torch
class RemoteTensorFlow(HermeticTestCase):
    """The tensor-level API of the reader (needs torch)."""

    def setUp(self):
        super().setUp()
        import torch

        self.torch = torch
        self.tensors = numpy_tensors(3)
        self.blob = build_safetensors(self.tensors)
        self.hub = FakeHub()
        self.hub.__enter__()
        self.addCleanup(self.hub.__exit__, None, None, None)
        self.hub.add("org/model", "model.safetensors", self.blob)
        self.r = remote.HubRepo("org/model", endpoint=self.hub.base_url, sleep=lambda s: None)

    def expect(self, name):
        dtype, shape, data = self.tensors[name]
        t = self.torch.frombuffer(bytearray(data), dtype=remote.torch_dtype(dtype))
        return t.reshape(shape)

    def same(self, a, b):
        self.assertEqual((a.dtype, tuple(a.shape)), (b.dtype, tuple(b.shape)))
        self.assertTrue(self.torch.equal(a.contiguous().view(self.torch.uint8), b.contiguous().view(self.torch.uint8)))

    def test_read_every_tensor_as_the_right_torch_dtype(self):
        for name in self.tensors:
            with self.subTest(name):
                self.same(self.r.read_tensor(name), self.expect(name))

    def test_row_slice_matches_the_full_tensor(self):
        name = "language_model.layers.0.mlp.gate_proj.weight"
        part = self.r.shard("model.safetensors").read_tensor(name, row_start=5, row_end=11)
        self.same(part, self.expect(name)[5:11])

    def test_stream_order_follows_requested_names_for_any_worker_count(self):
        names = list(self.tensors)
        for workers in (1, 2, 4):
            with self.subTest(workers=workers):
                random.Random(workers).shuffle(names)
                seen = []
                for n, t in self.r.stream(names, window_bytes=4096, workers=workers, max_gap=64):
                    seen.append(n)
                    self.same(t, self.expect(n))
                self.assertEqual(seen, names)

    def test_stream_abandoned_midway_does_not_hang(self):
        it = self.r.stream(list(self.tensors), workers=2)
        next(it)
        it.close()


@needs_transformers
class EngineFlow(HermeticTestCase):
    """The engine around a dense tiny backbone: logits, temperature, TTA mapping, batching, failure modes."""

    @classmethod
    def setUpClass(cls):
        import torch

        import tiny_model as tm
        from decide.sd import calibrate, engine, tta

        cls.torch, cls.tm, cls.E, cls.tta, cls.cal = torch, tm, engine, tta, calibrate
        cls.state = tm.backbone_state(1)
        cls.head = tm.head_weight(1)

    def decider(self, **kw):
        return self.tm.dense_decider(self.state, self.head, **kw)

    def engine(self, decider=None, **kw):
        return self.E.DecisionEngine(decider or self.decider(), **kw)

    def softmax(self, row, t):
        return self.cal.softmax(row, t).tolist()

    def test_logits_equal_the_upstream_reference_path(self):
        d = self.decider()
        eng = self.engine(d)
        response, raw = eng(**self.tm.REQUEST)
        upstream = d.logits([self.tm.REQUEST])[0]
        for key in upstream:
            for got, want in zip(raw["logits"][key][0], upstream[key]):
                self.assertAlmostEqual(got, want, places=6)

    def test_probabilities_are_softmax_of_logits_over_the_saved_temperature(self):
        eng = self.engine()
        response, raw = eng(**self.tm.REQUEST)
        t = self.tm.TEMPERATURE
        probs = response["answers"]["route"]["probabilities"]
        want = self.softmax(raw["logits"]["route"][0], t)
        self.assertEqual(list(probs), ["book", "ask", "refuse", "defer"])
        for got, w in zip(probs.values(), want):
            self.assertAlmostEqual(got, w, places=9)
        self.assertAlmostEqual(response["answers"]["urgent"]["noul"], self.softmax(raw["logits"]["urgent"][0], t)[1], places=9)
        self.assertEqual(raw["temperature"], t)
        self.assertEqual(raw["views"], 1)

    def test_the_choice_is_the_argmax_and_the_response_has_what_the_kit_checks(self):
        eng = self.engine()
        response, _raw = eng(**self.tm.REQUEST)
        a = response["answers"]["route"]
        self.assertEqual(a["choice"], max(a["probabilities"], key=a["probabilities"].get))
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0, places=9)

    @needs_kit
    def test_responses_pass_the_kits_validator(self):
        from decision_index.engines import validate

        response, _ = self.engine(tta_rotations=2)(**self.tm.REQUEST)
        validate(self.tm.REQUEST["questions"], response)

    def test_tta_maps_each_view_back_to_the_original_option_order(self):
        d = self.decider()
        eng = self.engine(d, tta_rotations=2, tta_mode="reverse")
        _response, raw = eng(**self.tm.REQUEST)
        self.assertEqual(raw["views"], 2)
        reversed_request = self.tta.permute_request(self.tm.REQUEST, {"route": [3, 2, 1, 0]})
        rev = d.logits([reversed_request])[0]["route"]
        for got, want in zip(raw["logits"]["route"][0], d.logits([self.tm.REQUEST])[0]["route"]):
            self.assertAlmostEqual(got, want, places=6)
        for got, want in zip(raw["logits"]["route"][1], list(reversed(rev))):
            self.assertAlmostEqual(got, want, places=6)

    def test_tta_probabilities_are_the_mean_of_the_per_view_softmaxes(self):
        eng = self.engine(tta_rotations=2)
        response, raw = eng(**self.tm.REQUEST)
        t = self.tm.TEMPERATURE
        v0, v1 = (self.softmax(row, t) for row in raw["logits"]["route"])
        probs = list(response["answers"]["route"]["probabilities"].values())
        for got, a, b in zip(probs, v0, v1):
            self.assertAlmostEqual(got, (a + b) / 2, places=9)

    def test_tta_does_not_change_which_options_exist(self):
        response, _ = self.engine(tta_rotations=2)(**self.tm.REQUEST)
        self.assertEqual(list(response["answers"]["route"]["probabilities"]), ["book", "ask", "refuse", "defer"])

    def test_a_batch_gives_the_same_logits_as_one_call_at_a_time(self):
        eng = self.engine()
        reqs = [self.tm.REQUEST, self.tm.second_request(), self.tm.REQUEST]
        batched = eng.batch(reqs)
        for req, (resp, raw) in zip(reqs, batched):
            solo_resp, solo_raw = eng(**req)
            for key in raw["logits"]:
                for a, b in zip(raw["logits"][key][0], solo_raw["logits"][key][0]):
                    self.assertAlmostEqual(a, b, places=5)

    def test_padding_does_not_leak_into_a_short_request(self):
        eng = self.engine()
        long_one = {"state": "x " * 60, "questions": {"q": {"type": "choice", "criteria": {"a": "1", "b": "2"}}}}
        (_r1, mixed), _ = eng.batch([self.tm.REQUEST, long_one])
        _r2, alone = eng(**self.tm.REQUEST)
        for key in alone["logits"]:
            for a, b in zip(mixed["logits"][key][0], alone["logits"][key][0]):
                self.assertAlmostEqual(a, b, places=5)

    def test_questions_the_kit_cannot_score_are_unsupported(self):
        eng = self.engine()
        req = {"state": "s", "questions": {"q": {"type": "score", "criteria": ["low", "high"]}}}
        with self.assertRaises(self.E.Unsupported):
            eng(**req)
        out = eng.batch([req, self.tm.REQUEST])
        self.assertIsInstance(out[0], self.E.Unsupported)
        self.assertIsInstance(out[1], tuple)                     # the other request still ran

    def test_over_length_requests_are_unsupported_not_truncated(self):
        eng = self.engine(self.decider(max_length=20))
        with self.assertRaises(self.E.Unsupported) as ctx:
            eng(**self.tm.REQUEST)
        self.assertIn("20", str(ctx.exception))

    def test_non_finite_logits_are_an_error_not_a_probability(self):
        d = self.decider()
        d.readout.weight.data[:] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.engine(d)(**self.tm.REQUEST)

    def test_out_of_memory_splits_the_batch_and_a_lone_sequence_is_unsupported(self):
        d = self.decider(batch_tokens=10_000)
        real = d.reference_logits
        calls = []

        def flaky(chunks):
            calls.append(len(chunks))
            if len(chunks) > 1:
                raise self.E.OOM_ERRORS[0]("CUDA out of memory (simulated)")
            return real(chunks)

        d.reference_logits = flaky
        eng = self.engine(d)
        out = eng.batch([self.tm.REQUEST, self.tm.second_request()])
        self.assertTrue(all(isinstance(o, tuple) for o in out))
        self.assertIn(2, calls)
        self.assertLessEqual(d.batch_tokens, 10_000)

        def always(chunks):
            raise self.E.OOM_ERRORS[0]("CUDA out of memory (simulated)")

        d.reference_logits = always
        with self.assertRaisesRegex(self.E.Unsupported, "out of memory"):
            eng(**self.tm.REQUEST)

    def oom_above(self, decider, limit):
        """Make `reference_logits` raise a simulated out-of-memory error for any chunk longer than `limit` tokens."""
        real = decider.reference_logits

        def flaky(chunks):
            if any(c.length > limit for c in chunks):
                raise self.E.OOM_ERRORS[0]("CUDA out of memory (simulated)")
            return real(chunks)

        decider.reference_logits = flaky

    def short_and_long(self, decider):
        """A request that fits and one that does not, plus the token count that separates them."""
        short = self.tm.REQUEST
        long_one = {"state": "x " * 60, "questions": {"q": {"type": "choice", "criteria": {"a": "1", "b": "2"}}}}
        sizes = [max(c.length for c in decider.encoder.plan(r, decider.max_length)) for r in (short, long_one)]
        self.assertLess(sizes[0], sizes[1])
        return short, long_one, (sizes[0] + sizes[1]) // 2

    def test_one_request_that_runs_out_of_memory_alone_does_not_condemn_the_batch(self):
        d = self.decider()
        short, long_one, limit = self.short_and_long(d)
        self.oom_above(d, limit)
        eng = self.engine(d)
        out = eng.batch([short, long_one, short])
        self.assertEqual([type(o).__name__ for o in out], ["tuple", "Unsupported", "tuple"])
        self.assertIn("out of memory", str(out[1]))
        self.assertEqual(out[0][1]["tokens"], out[2][1]["tokens"])

    def test_the_requests_saved_from_the_batch_have_the_logits_they_have_alone(self):
        d = self.decider()
        short, long_one, limit = self.short_and_long(d)
        self.oom_above(d, limit)
        eng = self.engine(d)
        (_r, mixed), _bad = eng.batch([short, long_one])
        _r2, alone = eng(**short)
        for a, b in zip(mixed["logits"]["route"][0], alone["logits"]["route"][0]):
            self.assertAlmostEqual(a, b, places=5)

    def test_a_request_with_two_views_that_runs_out_of_memory_is_unsupported_as_a_whole(self):
        d = self.decider()
        short, long_one, limit = self.short_and_long(d)
        self.oom_above(d, limit)
        out = self.engine(d, tta_rotations=2).batch([long_one, short])
        self.assertEqual([type(o).__name__ for o in out], ["Unsupported", "tuple"])

    def test_temperatures_per_question_type(self):
        eng = self.engine(temperatures={"choice": 2.0, "noul": 0.5})
        response, raw = eng(**self.tm.REQUEST)
        probs = list(response["answers"]["route"]["probabilities"].values())
        for got, want in zip(probs, self.softmax(raw["logits"]["route"][0], 2.0)):
            self.assertAlmostEqual(got, want, places=9)
        self.assertAlmostEqual(response["answers"]["urgent"]["noul"], self.softmax(raw["logits"]["urgent"][0], 0.5)[1], places=9)
        self.assertEqual(raw["applied_temperatures"], {"route": 2.0, "urgent": 0.5})
        self.assertEqual(raw["temperature"], self.tm.TEMPERATURE)             # the raw record keeps the model's own

    def test_raw_output_is_plain_json(self):
        _response, raw = self.engine(tta_rotations=2)(**self.tm.REQUEST)
        self.assertEqual(json.loads(json.dumps(raw, allow_nan=False)), raw)

    def test_a_wrong_tta_mode_is_refused(self):
        with self.assertRaises(ValueError):
            self.engine(tta_mode="shuffle")


@needs_transformers
class LoaderFlow(HermeticTestCase):
    """Stream + merge + quantise + assemble, from a fake Hub that serves real safetensors over HTTP Range requests."""

    @classmethod
    def setUpClass(cls):
        import torch

        import tiny_model as tm
        from decide.sd import loader, merge, nf4

        cls.torch, cls.tm, cls.L, cls.M, cls.N = torch, tm, loader, merge, nf4
        cls.hub = FakeHub()
        cls.hub.__enter__()
        cls.sa, cls.sd = tm.backbone_state(1), tm.backbone_state(2)
        cls.head = tm.head_weight(1)
        cls.pa, _ = tm.publish_backbone(cls.hub, "org/anchor", cls.sa, shards=3, head=cls.head, decision=True)
        cls.pd, _ = tm.publish_backbone(cls.hub, "org/donor", cls.sd, shards=2)
        cls.pd2, _ = tm.publish_backbone(cls.hub, "org/donor2", tm.backbone_state(3), shards=2)
        cls.pairs = tm.publish_lora(cls.hub, "org/lora", cls.pa, rank=4, alpha=8.0)
        cls.hub.log.clear()

    @classmethod
    def tearDownClass(cls):
        cls.hub.__exit__(None, None, None)

    def setUp(self):
        super().setUp()
        self.hub.log.clear()

    def repo(self, repo_id, revision=None):
        return remote.HubRepo(repo_id, revision, endpoint=self.hub.base_url, sleep=lambda s: None, probe=4096)

    def spec(self, donors=(("org/donor", 0.5),), lora=None, method="task_arithmetic", **extra):
        return {"anchor": "org/anchor", "donors": [{"repo": r, "weight": w} for r, w in donors],
                "lora": None if lora is None else {"repo": "org/lora", "weight": lora}, "method": method, **extra}

    def load(self, spec=None, progress=None, **opts):
        opts = {"quant": "none", "compute_dtype": "float32", "workers": 2, **opts}
        return self.L.load_decider(spec or self.spec(), self.L.LoadOptions(**opts), repo_factory=self.repo,
                                   processor_factory=self.tm.processor_factory, progress=progress)

    def expected(self, donors=((0.5, "pd"),), lora=None, skip=("embed_tokens", "visual.")):
        out = {}
        for name, a in self.pa.items():
            a64 = a.double()
            if any(s in name for s in skip):
                out[name] = a64
                continue
            m = a64.clone()
            for w, which in donors:
                m += w * (getattr(self, which)[name].double() - a64)
            if lora is not None and name in self.pairs:
                A, B = self.pairs[name]
                m += (8.0 / 4) * lora * (B.double() @ A.double())
            out[name] = m
        return out

    def assert_state(self, decider, expected, tol=1e-5, skip_prefix="visual."):
        got = decider.backbone.state_dict()
        for name, want in expected.items():
            if name.startswith(skip_prefix):
                continue
            key = name.replace("language_model.embed_tokens.weight", "language_model.embed_tokens.table.weight")
            self.assertIn(key, got, name)
            self.assertLess(float((got[key].double() - want).abs().max()), tol, name)

    def test_task_arithmetic_load_equals_a_hand_computed_merge(self):
        d = self.load()
        self.assert_state(d, self.expected())

    def spans(self, key):
        blob = self.hub.files[key]
        (n,) = struct.unpack("<Q", blob[:8])
        base = 8 + n
        return {k: (base + v["data_offsets"][0], base + v["data_offsets"][1])
                for k, v in json.loads(blob[8:8 + n]).items() if k != "__metadata__"}

    def requested(self, key):
        out = []
        for k, rng, _n in self.hub.log:
            if k == key and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                out.append((int(a), int(b)))
        return out

    def test_embeddings_and_vision_are_not_merged_and_vision_is_not_read(self):
        d = self.load()
        got = d.backbone.state_dict()["language_model.embed_tokens.table.weight"].double()
        self.assertEqual(float((got - self.pa["language_model.embed_tokens.weight"].double()).abs().max()), 0.0)
        self.assertFalse(any(k.startswith("visual.") for k in d.backbone.state_dict()))
        for repo, skipped in (("org/anchor", lambda n: n.startswith("visual.")),
                              ("org/donor", lambda n: n.startswith("visual.") or "embed_tokens" in n)):
            for key in (k for k in self.hub.files if k.startswith(repo + "/") and k.endswith(".safetensors")):
                asked = self.requested(key)
                for name, (a, b) in self.spans(key).items():
                    if skipped(name):
                        self.assertFalse(any(x < b and y >= a for x, y in asked), f"{key}: {name} was downloaded")

    def test_anchor_only_spec_reproduces_the_anchor(self):
        d = self.load(self.spec(donors=()))
        self.assert_state(d, self.expected(donors=()))

    def test_lora_delta_uses_alpha_over_r_and_the_weight(self):
        d = self.load(self.spec(donors=(), lora=0.1))
        self.assert_state(d, self.expected(donors=(), lora=0.1))

    def test_donors_and_lora_together(self):
        d = self.load(self.spec(donors=(("org/donor", 0.3), ("org/donor2", 0.2)), lora=0.25))
        self.assert_state(d, self.expected(donors=((0.3, "pd"), (0.2, "pd2")), lora=0.25))

    def test_ties_routes_through_the_merge_module(self):
        d = self.load(self.spec(method="ties", density=0.5))
        name = "language_model.layers.1.mlp.up_proj.weight"
        want = self.M.merge_one(self.pa[name], [self.pd[name]], [0.5], "ties", density=0.5)
        got = d.backbone.state_dict()[name]
        self.assertLess(float((got.double() - want.double()).abs().max()), 1e-6)

    def test_reference_base_uses_the_base_repo_with_prefixed_names(self):
        prefixed = {("model." + k if k.startswith(("language_model.", "visual.")) else k): v for k, v in self.sa.items()}
        self.tm.publish_backbone(self.hub, "org/base", prefixed, shards=2)
        d = self.load(self.spec(reference="base", base={"repo": "org/base"}))
        name = "language_model.layers.0.mlp.gate_proj.weight"
        base = self.sa[name].to(self.torch.bfloat16).double()
        want = base + 1.0 * (self.pa[name].double() - base) + 0.5 * (self.pd[name].double() - base)
        self.assertLess(float((d.backbone.state_dict()[name].double() - want).abs().max()), 1e-5)

    def test_logits_equal_a_dense_decider_built_from_the_expected_state(self):
        d = self.load(self.spec(lora=0.1))
        want = {k: v.float() for k, v in self.expected(lora=0.1).items()}
        dense = self.tm.dense_decider({k: v for k, v in want.items() if not k.startswith("visual.")} | {k: v.float() for k, v in self.sa.items() if k.startswith("visual.")}, self.pa_head())
        for key in ("route", "urgent"):
            got = d.logits([self.tm.REQUEST])[0][key]
            ref = dense.logits([self.tm.REQUEST])[0][key]
            for a, b in zip(got, ref):
                self.assertAlmostEqual(a, b, places=4)

    def pa_head(self):
        return self.head.to(self.torch.bfloat16).float()

    def test_nf4_load_equals_dense_with_round_tripped_linears(self):
        d = self.load(quant="nf4", absmax_dtype="float32")
        want = {k: v.float() for k, v in self.expected().items()}
        quantised = 0
        for name, tensor in list(want.items()):
            if name.startswith("language_model.") and name.endswith(".weight") and tensor.ndim == 2 and "embed_tokens" not in name and not any(s in name for s in ("in_proj_a", "in_proj_b")):
                packed, absmax, shape = self.N.quantize_nf4(tensor)
                want[name] = self.N.dequantize_nf4(packed, absmax, shape, dtype=self.torch.float32)
                quantised += 1
        self.assertGreater(quantised, 10)
        layers = [m for m in d.backbone.modules() if isinstance(m, self.N.Nf4Linear)]
        self.assertEqual(len(layers), quantised)
        dense = self.tm.dense_decider({k: v for k, v in want.items() if not k.startswith("visual.")} | {k: v.float() for k, v in self.sa.items() if k.startswith("visual.")}, self.pa_head())
        for key in ("route", "urgent"):
            got = d.logits([self.tm.REQUEST])[0][key]
            ref = dense.logits([self.tm.REQUEST])[0][key]
            for a, b in zip(got, ref):
                self.assertAlmostEqual(a, b, places=4)

    def test_gating_projections_stay_dense_by_default(self):
        d = self.load(quant="nf4")
        gate = d.backbone.language_model.layers[0].linear_attn
        self.assertIsInstance(gate.in_proj_a, self.torch.nn.Linear)
        self.assertIsInstance(gate.in_proj_qkv, self.N.Nf4Linear)
        everything = self.load(quant="nf4", nf4_skip=())
        self.assertIsInstance(everything.backbone.language_model.layers[0].linear_attn.in_proj_a, self.N.Nf4Linear)

    def test_offloaded_layers_give_the_same_logits(self):
        plain = self.load(quant="nf4", absmax_dtype="float32")
        off = self.load(quant="nf4", absmax_dtype="float32", offload_layers=2, embeddings_device="cpu")
        a = plain.logits([self.tm.REQUEST])[0]["route"]
        b = off.logits([self.tm.REQUEST])[0]["route"]
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=5)

    def test_a_donor_without_a_tensor_leaves_the_anchor_value(self):
        partial = {k: v for k, v in self.sd.items() if k != "language_model.layers.0.input_layernorm.weight"}
        self.tm.publish_backbone(self.hub, "org/partial", partial, shards=2)
        d = self.load(self.spec(donors=(("org/partial", 0.5),)))
        name = "language_model.layers.0.input_layernorm.weight"
        self.assertLess(float((d.backbone.state_dict()[name].double() - self.pa[name].double()).abs().max()), 1e-6)

    def test_a_donor_with_another_shape_is_refused_with_the_tensor_name(self):
        bad = dict(self.sd)
        bad["language_model.layers.0.mlp.gate_proj.weight"] = self.torch.zeros(10, 64)
        self.tm.publish_backbone(self.hub, "org/badshape", bad, shards=2)
        with self.assertRaisesRegex(self.L.LoaderError, "layers.0.mlp.gate_proj"):
            self.load(self.spec(donors=(("org/badshape", 0.5),)))

    def test_missing_anchor_tensors_are_reported(self):
        partial = {k: v for k, v in self.sa.items() if k != "language_model.norm.weight"}
        self.tm.publish_backbone(self.hub, "org/halfanchor", partial, shards=2, decision=True, head=self.head)
        spec = self.spec(donors=())
        spec["anchor"] = "org/halfanchor"
        with self.assertRaisesRegex(self.L.LoaderError, "language_model.norm.weight"):
            self.load(spec)

    def test_lora_modules_that_match_nothing_are_refused(self):
        self.hub.add("org/stray", "adapter_config.json", json.dumps({"r": 2, "lora_alpha": 2}).encode())
        blob = self.tm.safetensors_blob({"base_model.model.model.nowhere.q_proj.lora_A.weight": self.torch.zeros(2, 8),
                                         "base_model.model.model.nowhere.q_proj.lora_B.weight": self.torch.zeros(8, 2)})
        self.hub.add("org/stray", "adapter_model.safetensors", blob)
        spec = self.spec(donors=())
        spec["lora"] = {"repo": "org/stray", "weight": 0.1}
        with self.assertRaisesRegex(self.L.LoaderError, "match no anchor tensor"):
            self.load(spec)

    def test_progress_reports_every_tensor_once(self):
        seen = []
        self.load(progress=lambda done, total, name: seen.append((done, total, name)))
        self.assertEqual([s[0] for s in seen], list(range(1, len(seen) + 1)))
        self.assertEqual(seen[-1][0], seen[-1][1])
        self.assertEqual(len({s[2] for s in seen}), len(seen))

    def test_the_head_is_the_anchors_and_a_refit_head_replaces_it(self):
        d = self.load()
        self.assertLess(float((d.readout.weight.double() - self.head.to(self.torch.bfloat16).double()).abs().max()), 1e-6)
        import tempfile

        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "head.safetensors")
            save_file({"weight": self.torch.ones(255, self.tm.HIDDEN)}, path)
            d2 = self.load(head_path=path)
        self.assertTrue(bool((d2.readout.weight == 1).all()))

    def test_a_checkpoint_with_another_protocol_is_refused(self):
        self.tm.publish_backbone(self.hub, "org/other", self.sa, shards=2, decision=True, head=self.head)
        self.hub.add("org/other", "joint_config.json", json.dumps({"version": 2}).encode())
        spec = self.spec(donors=())
        spec["anchor"] = "org/other"
        with self.assertRaisesRegex(self.L.LoaderError, "protocol"):
            self.load(spec)

    def test_end_to_end_engine_on_a_loaded_decider(self):
        from decide.sd.engine import DecisionEngine

        eng = DecisionEngine(self.load(self.spec(lora=0.1), quant="nf4"), tta_rotations=2)
        response, raw = eng(**self.tm.REQUEST)
        self.assertEqual(raw["views"], 2)
        self.assertAlmostEqual(sum(response["answers"]["route"]["probabilities"].values()), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
