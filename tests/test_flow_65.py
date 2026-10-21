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

from decide_support import FakeHub, HermeticTestCase, build_safetensors, needs_torch, numpy_tensors

from decide.sd import remote


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


if __name__ == "__main__":
    unittest.main()
