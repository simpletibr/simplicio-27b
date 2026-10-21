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
import urllib.parse
from unittest import mock
from pathlib import Path

from decide_support import FakeHub, HermeticTestCase, build_safetensors, needs_kit, needs_requests, needs_torch, needs_transformers, numpy_tensors

from decide.sd import remote

PIN = "1" * 40                                                   # a commit hash: the fake Hub serves any revision, and records which one it was asked for


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
        return remote.HubRepo(name, kw.pop("revision", PIN), endpoint=self.hub.base_url, sleep=lambda s: None, **kw)

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
        r = remote.HubRepo("org/model", PIN, endpoint=self.hub.base_url, sleep=sleeps.append)
        self.assertEqual(self.raw(r, "mask"), self.tensors["mask"][2])
        self.assertEqual(len(sleeps), 3)
        self.assertTrue(all(x > 0 for x in sleeps))
        self.assertEqual(sleeps, sorted(sleeps))

    def test_gives_up_after_max_retries(self):
        self.hub.fail_first["org/model/model.safetensors"] = 99
        r = remote.HubRepo("org/model", PIN, endpoint=self.hub.base_url, sleep=lambda s: None, max_retries=2)
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
        r = remote.HubRepo("org/model", PIN, endpoint=self.hub.base_url, token="tok", sleep=lambda s: None)
        name = "language_model.embed_tokens.weight"
        self.assertEqual(self.raw(r, name), self.tensors[name][2])
        r2 = remote.HubRepo("org/model", PIN, endpoint=self.hub.base_url, token="wrong", sleep=lambda s: None)
        with self.assertRaisesRegex(remote.RemoteError, "401"):
            r2.shard("model.safetensors")

    def test_a_token_is_never_sent_over_plain_http_to_another_machine(self):
        for endpoint in ("http://example.com", "http://huggingface.co", "ftp://example.com", "https://"):
            with self.subTest(endpoint), self.assertRaisesRegex(remote.RemoteError, "endpoint"):
                remote.HubRepo("org/model", PIN, endpoint=endpoint, token="tok")
        for endpoint in ("https://huggingface.co", "http://127.0.0.1:1", "http://localhost:8080", "http://[::1]:9"):
            remote.HubRepo("org/model", PIN, endpoint=endpoint, token="tok")
        remote.HubRepo("org/model", PIN, endpoint="http://example.com")                      # nothing secret to leak without a token

    def second_origin(self):
        """A second server holding the same files, reached as `localhost` while the first is `127.0.0.1`: another origin, as a CDN is."""
        cdn = FakeHub()
        cdn.files = self.hub.files
        cdn.__enter__()
        self.addCleanup(cdn.__exit__, None, None, None)
        self.hub.redirect_base = f"http://localhost:{cdn.server.server_address[1]}"
        return cdn

    def test_the_token_does_not_follow_a_redirect_to_another_origin(self):
        cdn = self.second_origin()
        self.hub.redirect.add("org/model/model.safetensors")
        self.hub.require_token = "tok"
        r = remote.HubRepo("org/model", PIN, endpoint=self.hub.base_url, token="tok", sleep=lambda s: None)
        name = "language_model.embed_tokens.weight"
        self.assertEqual(self.raw(r, name), self.tensors[name][2])                     # the Range request was carried to the other origin
        self.assertIn("Bearer tok", self.hub.auth_seen)
        self.assertGreater(len(cdn.auth_seen), 0)                                       # the bytes did come from the second server
        self.assertEqual([a for a in cdn.auth_seen if a], [])                           # ...and it was never shown the token

    def test_the_token_stays_on_a_redirect_inside_the_same_origin(self):
        self.hub.redirect.add("org/model/model.safetensors")
        r = remote.HubRepo("org/model", PIN, endpoint=self.hub.base_url, token="tok", sleep=lambda s: None)
        self.hub.auth_seen.clear()
        r.shard("model.safetensors")
        self.assertEqual(set(self.hub.auth_seen), {"Bearer tok"})

    def test_a_redirect_from_https_to_http_or_to_another_scheme_is_refused_and_loops_end(self):
        class Reply:
            def __init__(self, code, location=None):
                self.status_code, self.headers = code, ({"Location": location} if location else {})

            def close(self):
                pass

        class Session:
            def __init__(self, replies):
                self.replies, self.asked = list(replies), []

            def request(self, method, url, **kw):
                self.asked.append((url, kw["headers"].get("Authorization")))
                return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

        def transport(replies):
            t = remote.HttpTransport(token="tok", sleep=lambda s: None)
            t._session = lambda: Session(replies)
            return t

        for target in ("http://cdn.example/file", "ftp://cdn.example/file", "file:///etc/passwd"):
            with self.subTest(target), self.assertRaisesRegex(remote.RemoteError, "refusing a redirect"):
                transport([Reply(302, target)]).get("https://huggingface.co/x")
        with self.assertRaisesRegex(remote.RemoteError, "too many redirects"):
            transport([Reply(302, "https://huggingface.co/x")]).get("https://huggingface.co/x")
        session = Session([Reply(302, "/other/path"), Reply(200)])
        t = remote.HttpTransport(token="tok", sleep=lambda s: None)
        t._session = lambda: session
        reply = t._follow("GET", "https://huggingface.co/x", {"Authorization": "Bearer tok"})
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(session.asked, [("https://huggingface.co/x", "Bearer tok"), ("https://huggingface.co/other/path", "Bearer tok")])

    # ---- a whole-file read has a ceiling: a config.json is not a shard
    def test_a_small_file_over_the_cap_is_refused_by_the_length_the_server_declares(self):
        self.hub.add("org/big", "config.json", b"{" + b" " * 4096 + b"}")
        with self.assertRaisesRegex(remote.RemoteError, "server says 4098 bytes, more than 1024 bytes"):
            self.repo("org/big", max_bytes=1024).read_json("config.json")
        self.assertEqual(self.repo("org/big", max_bytes=8192).read_json("config.json"), {})
        self.assertEqual(len(self.repo("org/big", max_bytes=4098).read_bytes("config.json")), 4098)       # exactly at the cap is fine
        with self.assertRaisesRegex(remote.RemoteError, "server says 4098 bytes, more than 4097 bytes"):
            self.repo("org/big", max_bytes=4097).read_bytes("config.json")

    def test_a_body_with_no_length_is_cut_at_the_cap_and_not_read_to_the_end(self):
        class Endless:
            status_code, headers = 200, {}

            def __init__(self):
                self.chunks = 0
                self.closed = False

            def iter_content(self, chunk_size=1):
                for _ in range(40):                                  # "endless", but a test that fails must end
                    self.chunks += 1
                    yield b"x" * chunk_size

            def close(self):
                self.closed = True

        reply = Endless()

        class Session:
            def request(self, method, url, **kw):
                return reply

        t = remote.HttpTransport(max_bytes=100_000, sleep=lambda s: None)
        t._session = lambda: Session()
        with self.assertRaisesRegex(remote.RemoteError, "more than 100000 bytes"):
            t.get("https://huggingface.co/x")
        self.assertEqual(reply.chunks, 2)                                                # it stopped at the cap, it did not drain the stream
        self.assertTrue(reply.closed)

    def test_the_cap_is_a_positive_whole_number_and_the_default_leaves_room_for_tokenizer_json(self):
        for bad in (0, -1, 1.5, True, "8"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "max_bytes"):
                remote.HttpTransport(max_bytes=bad)
        self.assertGreaterEqual(remote.MAX_SMALL_FILE, 32 << 20)
        self.assertLess(remote.MAX_SMALL_FILE, 1 << 30)
        self.assertEqual(remote.HttpTransport().max_bytes, remote.MAX_SMALL_FILE)

    def test_a_local_file_has_the_same_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_bytes(b"x" * 600)
            with self.assertRaisesRegex(remote.RemoteError, "more than 500 bytes"):
                remote.LocalTransport(max_bytes=500).get(str(path))
            self.assertEqual(len(remote.LocalTransport(max_bytes=600).get(str(path))), 600)

    def test_who_keeps_the_token_after_a_redirect_case_by_case(self):
        """`_follow` against a scripted session: for each Location, was the next request sent with the Authorization header, or refused."""
        class Reply:
            def __init__(self, code, location=None):
                self.status_code, self.headers = code, ({"Location": location} if location is not None else {})

            def close(self):
                pass

        class Session:
            def __init__(self, replies):
                self.replies, self.asked = list(replies), []

            def request(self, method, url, **kw):
                self.asked.append((url, kw["headers"].get("Authorization")))
                return self.replies.pop(0)

        def follow(location, start="https://huggingface.co/org/m/resolve/abc/file.bin"):
            session = Session([Reply(302, location), Reply(200)])
            t = remote.HttpTransport(token="tok", sleep=lambda s: None)
            t._session = lambda: session
            t._follow("GET", start, {"Authorization": "Bearer tok"})
            return session.asked[1]

        keeps = {
            "a relative path": "/other/path",
            "a relative file": "file2.bin",
            "the same host written in capitals": "HTTPS://HuggingFace.CO/x",
            "the same host with the default port spelled out": "https://huggingface.co:443/x",
            "the same host with a query": "/x?download=true&token=NOT-OURS",
            "an empty host after the scheme, which joins to ours": "https:///x",
        }
        for name, location in keeps.items():
            with self.subTest(f"keeps: {name}"):
                url, auth = follow(location)
                self.assertEqual(auth, "Bearer tok", url)
                self.assertTrue(urllib.parse.urlsplit(url).hostname == "huggingface.co", url)
        drops = {
            "another host": "https://cdn-lfs.example.com/blob",
            "a subdomain": "https://cdn.huggingface.co/blob",
            "the same host on another port": "https://huggingface.co:8443/x",
            "a host that only ends like ours": "https://evilhuggingface.co/x",
            "a host that starts like ours": "https://huggingface.co.evil.example/x",
            "a protocol-relative location": "//cdn-lfs.example.com/blob",
        }
        for name, location in drops.items():
            with self.subTest(f"drops: {name}"):
                url, auth = follow(location)
                self.assertIsNone(auth, url)
        url, auth = follow("https://127.0.0.1:8080/x", start="http://127.0.0.1:8080/x")             # http to https on the same host and port
        self.assertIsNone(auth, "another scheme is another origin")
        refused = {
            "https to http": "http://huggingface.co/x",
            "https to http on another host": "http://cdn.example.com/x",
            "another scheme": "ftp://cdn.example.com/x",
            "a local file": "file:///etc/passwd",
            "a host written as userinfo": "https://huggingface.co@cdn.example.com/x",
            "userinfo on our own host": "https://user:pass@huggingface.co/x",
            "a backslash before the at sign": "https://cdn.example.com\\@huggingface.co/x",
            "a backslash inside the host": "https://huggingface.co\\.cdn.example.com/x",
            "a tab inside the host": "https://huggingface.co\t.cdn.example.com/x",
            "a line feed before the at sign": "https://cdn.example.com\n@huggingface.co/x",
            "a control character in the path": "/x\x01y",
            "a NUL byte": "https://cdn.example.com\x00.huggingface.co/x",
            "DEL": "/x\x7fy",
            "a port without a host": "https://:8443/x",
            "a port that is not a number": "https://huggingface.co:80x/x",
        }
        for name, location in refused.items():
            with self.subTest(f"refused: {name}"), self.assertRaisesRegex(remote.RemoteError, "refusing a redirect"):
                follow(location)
        with self.assertRaisesRegex(remote.RemoteError, "without a Location"):
            follow(None)
        with self.assertRaisesRegex(remote.RemoteError, "without a Location"):
            follow("")

    def test_the_request_goes_to_the_url_that_was_checked_not_to_the_text_of_the_location(self):
        """`urlsplit` decides the origin, `urllib3` makes the connection: what is sent is rebuilt from the fields `urlsplit` gave, so the host that was
        checked is the host that is connected to, and a second parser (urllib3's) must agree with it or the redirect is refused."""
        class Reply:
            def __init__(self, code, location=None):
                self.status_code, self.headers = code, ({"Location": location} if location is not None else {})

            def close(self):
                pass

        class Session:
            def __init__(self, replies):
                self.replies, self.asked = list(replies), []

            def request(self, method, url, **kw):
                self.asked.append((url, kw["headers"].get("Authorization")))
                return self.replies.pop(0)

        def follow(location, start="https://huggingface.co/org/m/resolve/abc/file.bin"):
            session = Session([Reply(302, location), Reply(200)])
            t = remote.HttpTransport(token="tok", sleep=lambda s: None)
            t._session = lambda: session
            t._follow("GET", start, {"Authorization": "Bearer tok"})
            return session.asked[1]

        rebuilt = {
            "HTTPS://HuggingFace.CO:443/x?y=1#frag": ("https://huggingface.co:443/x?y=1", "Bearer tok"),
            "/other/path?a=b": ("https://huggingface.co/other/path?a=b", "Bearer tok"),
            "https://cdn.example.com/blob#part": ("https://cdn.example.com/blob", None),
            "//cdn.example.com:8443/blob": ("https://cdn.example.com:8443/blob", None),
            "https://[::1]:8443/blob": ("https://[::1]:8443/blob", None),
        }
        for location, (want, auth) in rebuilt.items():
            with self.subTest(location):
                self.assertEqual(follow(location), (want, auth))
        import urllib3.util

        real = urllib3.util.parse_url
        for lie, why in ((lambda u: real(u.replace("huggingface.co", "evil.example")), "another host"),
                         (lambda u: real(u.replace("huggingface.co", "huggingface.co:9999")), "another port")):
            with self.subTest(f"urllib3 reads {why}"), mock.patch.object(urllib3.util, "parse_url", lie), \
                    self.assertRaisesRegex(remote.RemoteError, "refusing a redirect"):
                follow("/x")
        with mock.patch.object(urllib3.util, "parse_url", real):
            self.assertEqual(follow("/x")[1], "Bearer tok")                  # the check does not trip on an honest location

    def test_a_redirect_that_two_parsers_read_differently_never_gets_the_token(self):
        """Two real servers.  `A` is the Hub, `B` is somebody else's: a Location that `urllib.parse` reads as `A` and `urllib3` reads as `B`
        (a backslash before the at sign) used to keep the Authorization header and then connect to `B`.  `B` must never see a Bearer."""
        other = FakeHub()
        other.__enter__()
        self.addCleanup(other.__exit__, None, None, None)
        other.add("org/model", "config.json", b"{}")
        a_port, b_port = urllib.parse.urlsplit(self.hub.base_url).port, urllib.parse.urlsplit(other.base_url).port
        self.hub.redirect.add("org/model/config.json")
        url = f"{self.hub.base_url}/org/model/resolve/{'a' * 40}/config.json"
        transport = remote.HttpTransport(token="hf_SECRET", max_retries=0, sleep=lambda s: None)

        def follow(base):
            self.hub.redirect_base = base
            return transport._follow("GET", url, transport._headers())

        hostile = {
            "the host before a backslash and the Hub's after the at sign": f"http://127.0.0.1:{b_port}\\@127.0.0.1:{a_port}",
            "the same with localhost": f"http://localhost:{b_port}\\@127.0.0.1:{a_port}",
            "the same with a letter after the backslash": f"http://127.0.0.1:{b_port}\\x@127.0.0.1:{a_port}",
            "the other host as userinfo of ours": f"http://127.0.0.1:{a_port}@127.0.0.1:{b_port}",
            "userinfo naming the other host": f"http://user:pass@127.0.0.1:{b_port}",
        }
        for name, base in hostile.items():
            with self.subTest(name), self.assertRaisesRegex(remote.RemoteError, "refusing a redirect"):
                follow(base)
            self.assertEqual(other.auth_seen, [], name)               # nobody even asked B
        self.assertNotIn("Bearer hf_SECRET", other.auth_seen)

        resp = follow(other.base_url)                                    # control: a plain redirect to another origin works, and drops the token
        self.assertEqual((resp.status_code, other.auth_seen), (200, [""]))
        resp.close()
        before = len(self.hub.auth_seen)
        resp = follow("")                                                # control: back to the same origin keeps it
        self.assertEqual(resp.status_code, 200)
        resp.close()
        self.assertEqual(self.hub.auth_seen[before:], ["Bearer hf_SECRET", "Bearer hf_SECRET"])

    def test_a_repo_is_read_at_a_full_commit_hash_and_nothing_else(self):
        for bad in (None, "main", "", "abc123", "A" * 40, "z" * 40, "1" * 39, "1" * 41, " " + "1" * 40, "1" * 40 + "\n", 7):
            with self.subTest(repr(bad)), self.assertRaisesRegex(remote.RemoteError, "revision"):
                remote.HubRepo("org/model", bad, endpoint=self.hub.base_url)
        with self.assertRaises(TypeError):
            remote.HubRepo("org/model", endpoint=self.hub.base_url)                  # there is no default revision to fall back to
        self.assertEqual(remote.HubRepo("org/model", PIN, endpoint=self.hub.base_url).revision, PIN)

    def test_the_pinned_revision_is_what_goes_on_the_wire(self):
        r = self.repo(revision="2" * 40)
        self.assertEqual(r.url("config.json"), f"{self.hub.base_url}/org/model/resolve/{'2' * 40}/config.json")
        r.read_bytes("config.json")
        r.shard("model.safetensors")
        self.assertEqual(self.hub.requested, {("org/model", "2" * 40)})

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

    # ---- integrity: the size the server states, and the manifest the publisher ships
    def test_a_file_longer_than_its_header_says_is_refused(self):
        self.hub.add("org/padded", "model.safetensors", self.blob + b"\0" * 16)
        with self.assertRaisesRegex(remote.RemoteError, "header describes"):
            self.repo("org/padded").shard("model.safetensors")

    def test_a_file_shorter_than_its_header_says_is_refused(self):
        self.hub.add("org/cut", "model.safetensors", self.blob[:-64])
        with self.assertRaisesRegex(remote.RemoteError, "header describes"):
            self.repo("org/cut").shard("model.safetensors")

    def test_a_sound_file_records_its_size(self):
        sf = self.repo().shard("model.safetensors")
        self.assertEqual((sf.size, sf.implied_size), (len(self.blob), len(self.blob)))

    def signed(self, repo="org/signed", **changes):
        """A repo whose release-manifest.json lists model.safetensors; the Hub states the listed sha256 unless `changes` say otherwise."""
        import hashlib

        digest = hashlib.sha256(self.blob).hexdigest()
        entry = {"bytes": len(self.blob), "sha256": digest}
        entry.update({k: v for k, v in changes.items() if k in entry})
        self.hub.add(repo, "model.safetensors", self.blob)
        self.hub.add(repo, "release-manifest.json", json.dumps({"repo_id": repo, "files": {"model.safetensors": entry}}).encode())
        self.hub.linked_etag[f"{repo}/model.safetensors"] = changes.get("hub_sha", digest)
        return self.repo(repo)

    def test_a_shard_the_manifest_lists_opens_when_size_and_hub_sha256_agree(self):
        repo = self.signed()
        self.assertEqual(repo.shard("model.safetensors").size, len(self.blob))
        self.assertIn("org/signed/model.safetensors", self.hub.heads)            # the sha256 was asked of the Hub, not computed from a download

    def test_a_shard_whose_size_differs_from_the_manifest_is_refused(self):
        with self.assertRaisesRegex(remote.RemoteError, "release-manifest.json lists"):
            self.signed(**{"bytes": len(self.blob) + 8}).shard("model.safetensors")

    def test_a_shard_whose_hub_sha256_differs_from_the_manifest_is_refused(self):
        with self.assertRaisesRegex(remote.RemoteError, "the Hub holds sha256"):
            self.signed(hub_sha="0" * 64).shard("model.safetensors")

    def test_a_hub_that_does_not_state_a_sha256_leaves_the_size_check_alone(self):
        repo = self.signed()
        self.hub.linked_etag.clear()
        self.assertEqual(repo.shard("model.safetensors").size, len(self.blob))

    def test_a_malformed_manifest_is_an_error_not_a_pass(self):
        for text in ('{"files": []}', '{"files": {"model.safetensors": {"bytes": "x", "sha256": "ab"}}}', "[]"):
            with self.subTest(text):
                self.hub.add("org/odd", "model.safetensors", self.blob)
                self.hub.add("org/odd", "release-manifest.json", text.encode())
                with self.assertRaisesRegex(remote.RemoteError, "release-manifest.json"):
                    self.repo("org/odd").shard("model.safetensors")

    def test_a_repo_without_a_manifest_is_read_as_before(self):
        repo = self.repo()
        self.assertIsNone(repo.manifest())
        self.assertEqual(self.hub.heads, [])

    def test_files_the_manifest_does_not_list_are_not_checked_against_it(self):
        repo = self.signed()
        self.hub.add("org/signed", "extra.safetensors", self.blob)
        self.assertEqual(repo.shard("extra.safetensors").size, len(self.blob))

    # ---- a header that lies: every offset is checked before any tensor is read
    @staticmethod
    def crafted(header, data=b""):
        text = json.dumps(header).encode("utf-8")
        text += b" " * (-len(text) % 8)
        return struct.pack("<Q", len(text)) + text + data

    def assert_refused(self, blob, pattern):
        self.hub.add("org/crafted", "model.safetensors", blob)
        with self.assertRaisesRegex(remote.RemoteError, pattern):
            self.repo("org/crafted").shard("model.safetensors")

    def test_a_header_that_lies_about_where_the_bytes_are_is_refused(self):
        u8 = lambda a, b, n=None: {"dtype": "U8", "shape": [(b - a) if n is None else n], "data_offsets": [a, b]}  # noqa: E731
        cases = {
            "a negative offset (it would read the header as a tensor)": (self.crafted({"a": u8(-4, 0, 4)}), "offsets"),
            "an end before the start": (self.crafted({"a": {"dtype": "U8", "shape": [0], "data_offsets": [8, 4]}}, b"\0" * 8), "offsets"),
            "a size that is not shape x dtype": (self.crafted({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}, b"\0" * 8), "shape"),
            "two tensors over the same bytes": (self.crafted({"a": u8(0, 4), "b": u8(0, 4)}, b"\0" * 4), "contiguous"),
            "a gap between tensors": (self.crafted({"a": u8(0, 4), "b": u8(8, 12)}, b"\0" * 12), "contiguous"),
            "data that does not start at zero": (self.crafted({"a": u8(4, 8)}, b"\0" * 8), "contiguous"),
            "a header longer than the file": (struct.pack("<Q", 5000) + b"{}", "header"),
            "a header that is not JSON": (struct.pack("<Q", 8) + b"not json", "JSON"),
            "a header that is not an object": (struct.pack("<Q", 8) + b"[1, 2,  ]"[:8], "object|JSON"),
            "a tensor without offsets": (self.crafted({"a": {"dtype": "U8", "shape": [1]}}, b"\0"), "offsets"),
            "a shape with a negative size": (self.crafted({"a": {"dtype": "U8", "shape": [-1], "data_offsets": [0, 0]}}), "shape"),
            "a shape that is not whole numbers": (self.crafted({"a": {"dtype": "U8", "shape": [1.5], "data_offsets": [0, 1]}}, b"\0"), "shape"),
        }
        for label, (blob, pattern) in cases.items():
            with self.subTest(label):
                self.assert_refused(blob, pattern)

    def test_a_transport_that_cannot_say_the_file_size_still_notices_a_header_cut_short(self):
        blob = struct.pack("<Q", 5000) + b"{}"

        class Clipping:                                           # a bare `fetch`, as a test double or a local mirror has it: no Content-Range
            def fetch(self, url, start, end):
                return bytearray(blob[start:end + 1])

        with self.assertRaisesRegex(remote.RemoteError, "cut short"):
            remote.SafetensorsFile("x", Clipping())
        good = self.crafted({"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}, b"\0\0")

        class Plain:
            def fetch(self, url, start, end):
                return bytearray(good[start:end + 1])

        self.assertEqual(remote.SafetensorsFile("x", Plain()).size, None)                 # no size to check, and nothing wrong with the file

    def test_an_empty_header_and_a_scalar_are_sound_files(self):
        self.hub.add("org/odd", "empty.safetensors", self.crafted({}))
        self.assertEqual(self.repo("org/odd").shard("empty.safetensors").names(), [])
        self.hub.add("org/odd", "scalar.safetensors", self.crafted({"s": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}, b"\0" * 4))
        self.assertEqual(self.repo("org/odd").shard("scalar.safetensors").info("s").numel, 1)

    def test_a_shard_name_from_the_index_cannot_leave_the_repo_or_change_the_url(self):
        for name in ("../evil.safetensors", "/etc/passwd", "a/../b.safetensors", "a?x=1", "a#frag", "a\\b", "", "http://x/y", "a//b", " a", "a\n", "../404.safetensors"):
            with self.subTest(repr(name)):
                self.hub.add("org/hostile", "model.safetensors.index.json", json.dumps({"weight_map": {"w": name}}).encode())
                with self.assertRaisesRegex(remote.RemoteError, "shard name"):
                    self.repo("org/hostile").weight_map()
        self.hub.add("org/fine", "model.safetensors.index.json", json.dumps({"weight_map": {"w": "sub/model-00001-of-00002.safetensors"}}).encode())
        self.assertEqual(self.repo("org/fine").weight_map(), {"w": "sub/model-00001-of-00002.safetensors"})

    def test_the_url_of_a_file_is_built_only_from_a_safe_path(self):
        for name in ("../x", "/x", "a?b", "a#b"):
            with self.subTest(name), self.assertRaisesRegex(remote.RemoteError, "file name"):
                self.repo().url(name)
        self.assertTrue(self.repo().url("evaluation/inference-verification.json").endswith(f"/resolve/{PIN}/evaluation/inference-verification.json"))

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
        self.r = remote.HubRepo("org/model", PIN, endpoint=self.hub.base_url, sleep=lambda s: None)

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

    def test_a_request_with_reasoning_is_refused_not_answered_without_it(self):
        # the joint prompt of v1.2 has no place for a reasoning text; the vendored encoder would drop the key without a word,
        # and an answer that never saw the reasoning would be scored as if it had
        eng = self.engine()
        with_reasoning = {**self.tm.REQUEST, "reasoning": "The customer already paid, so booking is safe."}
        with self.assertRaisesRegex(self.E.Unsupported, "reasoning"):
            eng(**with_reasoning)
        out = eng.batch([self.tm.REQUEST, with_reasoning, self.tm.REQUEST])
        self.assertEqual([type(o).__name__ for o in out], ["tuple", "Unsupported", "tuple"])
        self.assertIn("thinking track", str(out[1]))
        self.assertIsInstance(eng.batch([{**self.tm.REQUEST, "reasoning": None}])[0], tuple)         # an empty key says nothing to leave out

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
        self.hub.requested.clear()

    PINS = {"org/anchor": "a" * 40, "org/donor": "b" * 40, "org/donor2": "e" * 40, "org/lora": "c" * 40}          # the commit each repo of these tests is read at

    def repo(self, repo_id, revision):
        return remote.HubRepo(repo_id, revision, endpoint=self.hub.base_url, sleep=lambda s: None, probe=4096)

    def spec(self, donors=(("org/donor", 0.5),), lora=None, method="task_arithmetic", **extra):
        return {"anchor": "org/anchor", "anchor_revision": self.PINS["org/anchor"],
                "donors": [{"repo": r, "weight": w, "revision": self.PINS.get(r, "9" * 40)} for r, w in donors],
                "lora": None if lora is None else {"repo": "org/lora", "weight": lora, "revision": self.PINS["org/lora"]}, "method": method, **extra}

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

    def test_each_repo_is_read_at_the_commit_the_spec_names_and_at_no_other(self):
        """The fake Hub serves any revision, so this used to go unnoticed when the loader asked for `None` (read as `main`) for a donor."""
        self.load(self.spec(donors=(("org/donor", 0.5), ("org/donor2", 0.25)), lora=0.1))
        self.assertEqual(self.hub.requested, {("org/anchor", "a" * 40), ("org/donor", "b" * 40), ("org/donor2", "e" * 40), ("org/lora", "c" * 40)})

    def test_a_spec_without_a_commit_for_a_repo_stops_before_anything_is_read(self):
        for name, spec in (("anchor", {**self.spec(), "anchor_revision": None}),
                           ("donor", {**self.spec(), "donors": [{"repo": "org/donor", "weight": 0.5}]}),
                           ("lora", {**self.spec(lora=0.1), "lora": {"repo": "org/lora", "weight": 0.1}})):
            with self.subTest(name), self.assertRaisesRegex(remote.RemoteError, "revision"):
                self.load(spec)
        self.assertEqual(self.hub.requested - {("org/anchor", "a" * 40)}, set())            # at most the anchor's shards were asked for

    def test_anchor_only_spec_reproduces_the_anchor(self):
        d = self.load(self.spec(donors=()))
        self.assert_state(d, self.expected(donors=()))

    def test_an_adapter_with_a_rank_or_alpha_per_module_is_refused_not_merged_with_the_wrong_scale(self):
        weights = self.hub.files["org/lora/adapter_model.safetensors"]

        def publish(name, **extra):
            self.hub.add(name, "adapter_model.safetensors", weights)
            self.hub.add(name, "adapter_config.json", json.dumps({"peft_type": "LORA", "r": 4, "lora_alpha": 8.0, **extra}).encode())
            return self.repo(name, "c" * 40)

        for key in ("alpha_pattern", "rank_pattern"):
            with self.subTest(key), self.assertRaisesRegex(self.L.LoaderError, key):
                self.L.LoraAdapter(publish(f"org/with-{key}", **{key: {"layers.0.self_attn.q_proj": 16}}), 0.1)
        adapter = self.L.LoraAdapter(publish("org/empty-patterns", alpha_pattern={}, rank_pattern={}), 0.1)       # what peft writes by default
        self.assertEqual((adapter.r, adapter.alpha), (4, 8.0))
        self.assertEqual(self.L.LoraAdapter(publish("org/null-patterns", alpha_pattern=None, rank_pattern=None), 0.1).r, 4)

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
        d = self.load(self.spec(reference="base", base={"repo": "org/base", "revision": "f" * 40}))
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

    def publish_partial(self, repo="org/partial", drop="language_model.layers.0.input_layernorm.weight"):
        self.tm.publish_backbone(self.hub, repo, {k: v for k, v in self.sd.items() if k != drop}, shards=2)
        return drop

    def test_a_donor_that_lacks_a_tensor_is_refused_with_the_count_and_the_first_missing_name(self):
        drop = self.publish_partial()
        with self.assertRaises(self.L.LoaderError) as ctx:
            self.load(self.spec(donors=(("org/partial", 0.5),)))
        message = str(ctx.exception)
        total = len([n for n in self.pa if not n.startswith("visual.") and "embed_tokens" not in n])
        self.assertIn(f"covers {total - 1} of {total}", message)
        self.assertIn(drop, message)
        self.assertIn("org/partial", message)

    def test_a_donor_that_ships_its_own_readout_never_lends_it(self):
        other = self.tm.head_weight(7)
        self.assertGreater(float((other - self.head).abs().max()), 0.1)
        self.tm.publish_backbone(self.hub, "org/donorhead", self.sd, shards=2, head=other, decision=True)
        d = self.load(self.spec(donors=(("org/donorhead", 0.5),)))
        want = self.head.to(self.torch.bfloat16).double()
        self.assertLess(float((d.readout.weight.double() - want).abs().max()), 1e-6)         # the anchor's head, not a mix, not the donor's
        self.assertGreater(float((d.readout.weight.double() - other.to(self.torch.bfloat16).double()).abs().max()), 0.1)

    def test_a_donor_without_a_tensor_leaves_the_anchor_value_when_partial_donors_are_allowed(self):
        drop = self.publish_partial()
        d = self.load(self.spec(donors=(("org/partial", 0.5),)), allow_partial=True)
        self.assertLess(float((d.backbone.state_dict()[drop].double() - self.pa[drop].double()).abs().max()), 1e-6)
        merged = "language_model.layers.1.mlp.up_proj.weight"
        want = self.pa[merged].double() + 0.5 * (self.pd[merged].double() - self.pa[merged].double())
        self.assertLess(float((d.backbone.state_dict()[merged].double() - want).abs().max()), 1e-5)

    def test_a_donor_whose_names_match_none_of_the_anchors_is_not_a_merge_that_equals_the_anchor(self):
        # the reviewer's probe: published with the base model's "model." prefix, 0 names in common, and the result was the anchor bit for bit
        other = {("model." + k if k.startswith(("language_model.", "visual.")) else k): v for k, v in self.sd.items()}
        self.tm.publish_backbone(self.hub, "org/othernames", other, shards=2)
        with self.assertRaisesRegex(self.L.LoaderError, "org/othernames"):
            self.load(self.spec(donors=(("org/othernames", 0.5),)))
        with self.assertRaisesRegex(self.L.LoaderError, "base-model naming"):
            self.load(self.spec(donors=(("org/othernames", 0.5),)), allow_partial=True)       # allowing a partial donor does not allow this

    def test_a_donor_with_an_lm_head_or_mtp_tensor_is_in_base_model_naming(self):
        for extra in ("lm_head.weight", "mtp.layers.0.weight"):
            with self.subTest(extra):
                donor = dict(self.sd)
                donor[extra] = self.torch.zeros(4, 4)
                self.tm.publish_backbone(self.hub, "org/withhead", donor, shards=2)
                with self.assertRaisesRegex(self.L.LoaderError, "base-model naming"):
                    self.load(self.spec(donors=(("org/withhead", 0.5),)))

    def test_the_same_repo_as_donor_and_as_lora_would_count_the_simplicio_delta_twice(self):
        spec = self.spec(donors=(("org/lora", 0.5),), lora=0.1)
        with self.assertRaisesRegex(self.L.LoaderError, "both a donor and the LoRA"):
            self.load(spec)

    def test_a_donor_that_is_the_anchor_itself_is_refused(self):
        with self.assertRaisesRegex(self.L.LoaderError, "is the anchor"):
            self.load(self.spec(donors=(("org/anchor", 0.5),)))

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
        spec["lora"] = {"repo": "org/stray", "weight": 0.1, "revision": "c" * 40}
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
