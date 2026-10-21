"""Flows of decide/think (issue #75, F2b): the pieces working together on disk.

A synthetic suite in the kit's file layout goes through `sample build/verify`, a fake OpenAI-compatible
server (stdlib, localhost) answers the thinker by a script hidden in each state, and `compare` reads
what the thinker wrote.  The script lets the test know the thinker's exact score without reading the
code under test.  A tiny random Qwen2 model (built in the test, no download) drives the transformers
backend when torch is installed; without torch that class is skipped.

Everything is synthetic: no benchmark text is written to this repository.  CPU only, no network
beyond 127.0.0.1.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import http.server
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decide.think import common as C  # noqa: E402
from decide.think import compare as K  # noqa: E402
from decide.think import prompt as P  # noqa: E402
from decide.think import sample as S  # noqa: E402
from decide.think import thinker as T  # noqa: E402

SECRET = "sk-test-secret-123"
MODELS = {"m_good": 85, "m_mid": 60, "m_bad": 30, "m_flaky": 70}          # model -> % of rows it answers right


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def flaky(run_id: str) -> bool:
    """The rows where m_flaky has an error instead of an answer."""
    return int(sha(run_id), 16) % 5 == 2


def mode_of(i: int) -> str:
    """What the fake server does with the i-th row of a benchmark."""
    if i % 6 == 0:
        return "wrong"
    if i % 11 == 5:
        return "garbage"
    if i % 13 == 7:
        return "cut"
    return "right"


def make_row(n, run_id, group, track, questions, expected, mode, scoring=None):
    """A kit row.  The state carries the reply the fake server will give (labels as the prompt shows them)."""
    right, wrong = [], []
    accepted = (scoring or {}).get("accepted") or []
    for qid, crit in questions.items():
        labels = {o.key: o.label for o in P.option_list(crit)}
        other = next(k for k in crit if k != expected[qid] and k not in accepted)
        right.append(f"{qid}={labels[expected[qid]]}")
        wrong.append(f"{qid}={labels[other]}")
    row = {"state": f"SYNTH {run_id}. HINT[{';'.join(right)}] WRONG[{';'.join(wrong)}] MODE[{mode}]",
           "questions": {qid: {"type": "choice", "instructions": f"SYNTH pick one for {qid}.", "criteria": crit} for qid, crit in questions.items()},
           "expected": expected,
           "_evaluation": {"run_id": run_id, "catalog_id": n, "group_id": group, "track": track, "payload_sha256": sha(run_id), "dataset": track}}
    if scoring:
        row["scoring"] = scoring
    return row


def write_gz(path: Path, rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_suite(root: Path) -> None:
    """GPQA-like (25), BBH-like (58), SATA-like (33), chess (31) and GSM8K (30), plus the rows the kit drops or the thinker cannot answer."""
    sel, added, gsm = [], [], []
    opts4 = {f"option_{k}": f"SYNTH text of option {k}" for k in range(1, 5)}
    for i in range(30):
        sel.append(make_row(25, f"25:{i}", f"GPQA-{i}", "GPQA-Diamond", {"answer": opts4}, {"answer": f"option_{i % 4 + 1}"}, mode_of(i)))
    for i in range(30):
        sel.append(make_row(58, f"58:{i}", f"BBH-{i}", "web_of_lies", {"answer": {"True": "True", "False": "False"}}, {"answer": "True" if i % 2 else "False"}, mode_of(i)))
    sel.append({"state": "SYNTH noul", "questions": {"x": {"type": "noul", "instructions": "i"}}, "expected": {"x": True},
                "_evaluation": {"run_id": "58:noul", "catalog_id": 58, "group_id": "BBH-noul", "track": "t", "payload_sha256": sha("58:noul")}})
    for i in range(16):
        qs = {f"item_{k}": {"no": "No", "yes": "Yes"} for k in range(1, 3 + i % 3)}
        sel.append(make_row(33, f"33:{i}", f"SATA-{i}", "SATA-Bench", qs, {q: ("yes" if (i + j) % 2 else "no") for j, q in enumerate(qs)}, mode_of(i)))
    moves = {f"{c}2{c}3": {"uci": f"{c}2{c}3", "san": f"{c}3"} for c in "abcdefgh"}
    moves.update({f"{c}2{c}4": {"uci": f"{c}2{c}4", "san": f"{c}4"} for c in "abcdefgh"})
    moves.update({f"{c}7{c}6": {"uci": f"{c}7{c}6", "san": f"{c}6"} for c in "abcdefgh"})
    moves.update({f"{c}7{c}5": {"uci": f"{c}7{c}5", "san": f"{c}5"} for c in "abcd"})
    keys = list(moves)
    for i in range(12):
        gold, tie = keys[i], keys[i + 1]
        added.append(make_row(31, f"31:{i}", f"CHESS-{i}", "ChessBench-legal-move", {"best": moves}, {"best": gold}, mode_of(i),
                              scoring={"type": "chess_reference_value", "accepted": [gold, tie], "values": {k: 0.5 for k in keys}}))
    for i in range(8):
        sel.append(make_row(30, f"gsm-old:{i}", f"GSM8K-{i}", "GSM8K-4choice", {"answer": opts4}, {"answer": "option_1"}, "right"))
        for k in range(4):
            n = 4 if k < 2 else 10
            crit = {f"option_{j}": f"SYNTH {j}" for j in range(1, n + 1)}
            gsm.append(make_row(30, f"v0.3-gsm8k:{i}:{k}", f"GSM8K-{i}", f"GSM8K-{n}choice", {"answer": crit}, {"answer": f"option_{(i + k) % n + 1}"}, mode_of(i + k)))
    write_gz(root / S.ROWS_FILE, sel)
    write_gz(root / S.ADDED_FILE, added)
    write_gz(root / S.GSM8K_FILE, gsm)
    (root / S.EXCLUSIONS_FILE).write_text(json.dumps({"rows": ["25:3", "25:4", "58:9"]}), encoding="utf-8")


def all_rows(root: Path):
    out = []
    for name in (S.ROWS_FILE, S.ADDED_FILE, S.GSM8K_FILE):
        with gzip.open(root / name, "rt", encoding="utf-8") as f:
            out += [json.loads(line) for line in f]
    return out


def write_results(path: Path, model: str, rows, flaky_status: bool = False) -> None:
    """The kit's per-row results of a decision model: probabilities per option, right in `pct` % of the rows."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            e = row["_evaluation"]
            answers, status = {}, "ok"
            if flaky_status and flaky(e["run_id"]):
                status = "error"
            for qid, q in row["questions"].items():
                if q["type"] != "choice":
                    continue
                keys = list(q["criteria"])
                gold = row["expected"][qid]
                right = int(sha(f"{model}|{e['run_id']}|{qid}"), 16) % 100 < MODELS[model]
                pick = gold if right else next(k for k in keys if k != gold)
                probs = {k: (0.7 if k == pick else 0.3 / (len(keys) - 1)) for k in keys}
                answers[qid] = {"type": "choice", "choice": pick, "probabilities": probs, "confidence": 0.7}
            rec = {"run_id": e["run_id"], "catalog_id": e["catalog_id"], "status": status, "payload_sha256": e["payload_sha256"], "total_wall_ms": 12}
            if status == "ok":
                rec["response"] = {"answers": answers}
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------------- the fake server

def scripted(body, count):
    prompt = body["messages"][0]["content"]
    m = re.search(r"MODE\[(\w+)\]", prompt)
    if not m:                                                           # the preflight
        return {"content": "A", "finish": "stop", "usage": (5, 3)}
    mode = m.group(1)
    right = re.search(r"HINT\[([^\]]*)\]", prompt).group(1)
    wrong = re.search(r"WRONG\[([^\]]*)\]", prompt).group(1)

    def lines(spec):
        return "\n".join(f"ANSWER {kv.split('=')[0]}: {kv.split('=')[1]}" for kv in spec.split(";"))
    if mode == "cut" or (mode == "long" and body.get("max_tokens", 0) < 200):
        return {"content": "hmm hmm hmm", "finish": "length", "usage": (50, body.get("max_tokens", 0))}
    final = {"right": lines(right), "wrong": lines(wrong), "garbage": "I am not sure.", "long": lines(right)}[mode]
    think = "Let me think about this carefully. Maybe the first option, no, the second."
    if count % 2 == 0:
        return {"reasoning": think, "content": final, "finish": "stop", "usage": (50, 80)}
    return {"content": think + "\n</think>\n\n" + final, "finish": "stop", "usage": (50, 80)}


class FakeServer:
    def __init__(self, behavior=scripted, require_key=None):
        self.behavior, self.require_key = behavior, require_key
        self.requests, self.lock, self.count = [], threading.Lock(), 0

    def __enter__(self):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_json(self, code, obj, headers=None):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with outer.lock:
                    outer.count += 1
                    k = outer.count
                    outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body, "ua": self.headers.get("User-Agent")})
                if outer.require_key and self.headers.get("Authorization") != f"Bearer {outer.require_key}":
                    return self.send_json(401, {"error": {"message": "bad key"}})
                act = outer.behavior(body, k)
                if "status" in act:
                    return self.send_json(act["status"], {"error": {"message": act.get("text", "error")}}, act.get("headers"))
                usage = {"prompt_tokens": act["usage"][0], "completion_tokens": act["usage"][1]}
                if not body.get("stream"):
                    msg = {"role": "assistant", "content": act["content"]}
                    if act.get("reasoning") is not None:
                        msg["reasoning"] = act["reasoning"]
                    return self.send_json(200, {"choices": [{"message": msg, "finish_reason": act["finish"]}], "usage": usage})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def ev(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                ev({"choices": [{"delta": {"role": "assistant"}}]})
                if act.get("reasoning"):
                    half = len(act["reasoning"]) // 2
                    ev({"choices": [{"delta": {"reasoning_content": act["reasoning"][:half]}}]})
                    ev({"choices": [{"delta": {"reasoning_content": act["reasoning"][half:]}}]})
                half = len(act["content"]) // 2
                ev({"choices": [{"delta": {"content": act["content"][:half]}}]})
                ev({"choices": [{"delta": {"content": act["content"][half:]}, "finish_reason": act["finish"]}]})
                ev({"choices": [], "usage": usage})
                self.wfile.write(b"data: [DONE]\n\n")

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(5)
        return False


def run_cli(module, argv, env=None):
    """Run a module's main() with a clean environment; return (exit code, stdout)."""
    out = io.StringIO()
    clean = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "THINK_API_KEY", "THINK_BASE_URL")}
    clean.update(env or {})
    with mock.patch.dict(os.environ, clean, clear=True), contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = module.main(argv)
    return code, out.getvalue()


# ------------------------------------------------------------------------------------------- tests

class FlowBase(unittest.TestCase):
    PER = 6
    BENCH = (25, 30, 31, 33, 58)

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.suite = cls.root / "suite"
        cls.suite.mkdir()
        build_suite(cls.suite)
        rows = all_rows(cls.suite)
        cls.results = {}
        for m in MODELS:
            p = cls.root / f"{m}.jsonl"
            write_results(p, m, rows, flaky_status=(m == "m_flaky"))
            cls.results[m] = p
        cls.out = cls.root / "sample"
        cls.info = S.build(cls.suite, cls.results, cls.out, per_benchmark=cls.PER, benchmarks=cls.BENCH)
        cls.pack = cls.out / "pack.jsonl.gz"
        cls.manifest = cls.out / "manifest.json"
        cls.rows = S.read_pack(cls.pack)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def results_path(self, name="results.jsonl") -> Path:
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        return Path(d.name) / name

    def thinker(self, server, out, *extra, env=None):
        argv = ["--pack", str(self.pack), "--out", str(out), "--base-url", server.url, "--max-tokens", "400", "--concurrency", "3", *extra]
        return run_cli(T, argv, env={"OPENAI_API_KEY": SECRET, **(env or {})})

    def expected_hits(self, bench):
        return sum(1 for r in self.rows if r["benchmark"] == bench and re.search(r"MODE\[(\w+)\]", r["state"]).group(1) == "right")


class SampleFlow(FlowBase):
    def test_build_reports_what_it_wrote(self) -> None:
        self.assertEqual(self.info["rows"], len(self.rows))
        self.assertEqual(self.info["per_benchmark"], {25: 6, 30: 6, 31: 6, 33: 6, 58: 6})
        self.assertEqual(self.info["missing_benchmarks"], [])
        self.assertEqual(S.verify(self.manifest, self.pack), [])

    def test_rows_are_the_kit_rows_after_exclusions_and_one_request_per_case(self) -> None:
        ids = [r["id"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse({"25:3", "25:4", "58:9", "58:noul"} & set(ids))
        gsm = [r for r in self.rows if r["benchmark"] == 30]
        self.assertTrue(all(r["id"].startswith(S.GSM8K_V3_PREFIX) for r in gsm))
        self.assertEqual(len({r["group"] for r in gsm}), len(gsm))
        self.assertFalse([i for i in ids if flaky(i)])               # m_flaky has an error there: the case is not eligible

    def test_pack_rows_have_gold_and_every_models_probabilities(self) -> None:
        for r in self.rows:
            self.assertEqual(sorted(r["decisions"]), sorted(MODELS))
            for qid, q in r["questions"].items():
                self.assertIn(r["gold"][qid][0], q["criteria"])
                for m in MODELS:
                    probs = r["decisions"][m][qid]["probs"]
                    self.assertEqual(list(probs), list(q["criteria"]))
                    self.assertAlmostEqual(sum(probs.values()), 1.0)
        chess = next(r for r in self.rows if r["benchmark"] == 31)
        self.assertEqual(len(chess["gold"]["best"]), 2)                # the tie set of the kit
        self.assertGreater(len(chess["questions"]["best"]["criteria"]), 26)
        sata = [r for r in self.rows if r["benchmark"] == 33]
        self.assertTrue(all(len(r["questions"]) >= 2 for r in sata))

    def test_same_inputs_same_bytes(self) -> None:
        out2 = self.root / "sample2"
        S.build(self.suite, dict(reversed(list(self.results.items()))), out2, per_benchmark=self.PER, benchmarks=self.BENCH)
        self.assertEqual((out2 / "pack.jsonl.gz").read_bytes(), self.pack.read_bytes())
        self.assertEqual((out2 / "manifest.json").read_bytes(), self.manifest.read_bytes())

    def test_a_smaller_sample_is_a_prefix_of_a_bigger_one(self) -> None:
        out3 = self.root / "sample3"
        S.build(self.suite, self.results, out3, per_benchmark=3, benchmarks=self.BENCH)
        small = json.loads((out3 / "manifest.json").read_text())["benchmarks"]
        big = json.loads(self.manifest.read_text())["benchmarks"]
        for b in small:
            self.assertEqual(small[b]["run_ids"], big[b]["run_ids"][:3])

    def test_manifest_names_its_inputs_by_stored_and_by_content_hash(self) -> None:
        import hashlib
        m = json.loads(self.manifest.read_text())
        self.assertEqual(sorted(m["inputs"]), ["results", "suite_files"])
        self.assertEqual(sorted(m["inputs"]["results"]), sorted(MODELS))
        for name, path in self.results.items():
            self.assertEqual(m["inputs"]["results"][name], {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        rows_file = m["inputs"]["suite_files"]["selected-rows.jsonl.gz"]
        with gzip.open(self.suite / "selected-rows.jsonl.gz", "rb") as f:
            self.assertEqual(rows_file["uncompressed_sha256"], hashlib.sha256(f.read()).hexdigest())
        self.assertEqual(rows_file["sha256"], hashlib.sha256((self.suite / "selected-rows.jsonl.gz").read_bytes()).hexdigest())

    def test_manifest_has_ids_and_hashes_never_question_text(self) -> None:
        text = self.manifest.read_text()
        for marker in ("SYNTH", "HINT[", "option_1", "text of"):
            self.assertNotIn(marker, text)
        m = json.loads(text)
        self.assertEqual(m["format"], C.MANIFEST_FORMAT)
        self.assertEqual(m["seed"], C.SEED)
        self.assertEqual(m["decision_models"], sorted(MODELS))
        self.assertEqual(m["use"], C.EVAL_ONLY_NOTICE)                # the sample is evaluation data, never training data
        self.assertEqual(len(m["benchmarks"]["25"]["run_ids"]), 6)
        self.assertEqual(len(m["benchmarks"]["25"]["payload_sha256"]), 6)
        with gzip.open(self.pack, "rt", encoding="utf-8") as f:
            self.assertIn("SYNTH", f.read())                         # the text is in the private pack only

    def test_pack_rows_are_round_robin_across_benchmarks(self) -> None:
        self.assertEqual([r["benchmark"] for r in self.rows[:5]], [25, 30, 31, 33, 58])

    def test_verify_catches_a_pack_that_is_not_the_manifests(self) -> None:
        bad = self.root / "bad.jsonl.gz"
        rows = S.read_pack(self.pack)
        rows[0]["payload_sha256"] = "0" * 64
        S.write_pack(bad, rows)
        problems = S.verify(self.manifest, bad)
        self.assertTrue(any("payload sha256" in p for p in problems), problems)
        self.assertTrue(any("content sha256" in p for p in problems), problems)
        swapped = self.root / "swapped.jsonl.gz"
        S.write_pack(swapped, list(reversed(S.read_pack(self.pack))))
        self.assertTrue(any("order" in p for p in S.verify(self.manifest, swapped)))

    def test_the_cli_builds_and_verifies(self) -> None:
        out = self.root / "cli"
        args = ["build", "--suite", str(self.suite), "--out", str(out), "--per-benchmark", "4", "--benchmarks", "25,58"]
        for m, p in self.results.items():
            args += ["--results", f"{m}={p}"]
        code, text = run_cli(S, args)
        self.assertEqual(code, 0, text)
        code, text = run_cli(S, ["verify", "--manifest", str(out / "manifest.json"), "--pack", str(out / "pack.jsonl.gz")])
        self.assertEqual(code, 0, text)
        self.assertIn("OK", text)

    def test_a_benchmark_without_rows_is_reported(self) -> None:
        out = self.root / "with45"
        info = S.build(self.suite, self.results, out, per_benchmark=2, benchmarks=(25, 45))
        self.assertEqual(info["missing_benchmarks"], [45])
        self.assertEqual(info["per_benchmark"], {25: 2})


class ThinkerFlow(FlowBase):
    def test_openai_end_to_end_with_json_replies(self) -> None:
        out = self.results_path()
        with FakeServer(require_key=SECRET) as srv:
            code, text = self.thinker(srv, out, "--no-stream")
        self.assertEqual(code, 0, text)
        header, recs = T.ResultStore.read(out)
        self.assertEqual(header["config"]["model"], C.QWEN_MODEL)
        self.assertEqual(header["config"]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(header["pack_content_sha256"], S.pack_content_sha256(self.pack))
        self.assertEqual(header["notice"], C.EVAL_ONLY_NOTICE)
        self.assertEqual(sorted(r["id"] for r in recs), sorted(r["id"] for r in self.rows))
        by_id = {r["id"]: r for r in recs}
        for row in self.rows:
            mode = re.search(r"MODE\[(\w+)\]", row["state"]).group(1)
            rec = by_id[row["id"]]
            self.assertEqual(rec["status"], "ok")
            hit = all(rec["answers"].get(q) in row["gold"][q] for q in row["questions"])
            self.assertEqual(hit, mode == "right", (row["id"], mode, rec["answers"]))
            if mode in ("garbage", "cut"):
                self.assertTrue(all(v is None for v in rec["answers"].values()))
            self.assertEqual(rec["truncated"], mode == "cut")
            self.assertTrue(rec["reasoning"])
        self.assertTrue(all(r["auth"] == f"Bearer {SECRET}" for r in srv.requests))
        self.assertTrue(all(r["ua"] and "Python-urllib" not in r["ua"] for r in srv.requests))
        first = next(r for r in srv.requests if "MODE[" in r["body"]["messages"][0]["content"])["body"]
        self.assertEqual((first["temperature"], first["top_p"], first["top_k"], first["max_tokens"]), (1.0, 0.95, 20, 400))
        self.assertEqual(first["chat_template_kwargs"], {"enable_thinking": True})

    def test_the_key_and_the_questions_never_reach_the_log_or_the_file(self) -> None:
        out = self.results_path()
        with FakeServer(require_key=SECRET) as srv:
            code, text = self.thinker(srv, out)
        self.assertEqual(code, 0, text)
        self.assertNotIn(SECRET, text)
        self.assertNotIn(SECRET, out.read_text(encoding="utf-8"))
        for marker in ("SYNTH", "HINT[", "text of"):
            self.assertNotIn(marker, text)
        self.assertIn("preflight ok", text)

    def test_streaming_replies_give_the_same_results_as_json_replies(self) -> None:
        a, b = self.results_path("a.jsonl"), self.results_path("b.jsonl")
        with FakeServer() as srv:
            self.assertEqual(self.thinker(srv, a, "--no-stream")[0], 0)
            self.assertEqual(self.thinker(srv, b)[0], 0)
        ra = {r["id"]: r["answers"] for r in T.ResultStore.read(a)[1]}
        rb = {r["id"]: r["answers"] for r in T.ResultStore.read(b)[1]}
        self.assertEqual(ra, rb)
        self.assertEqual(len(ra), len(self.rows))

    def test_resume_after_a_limit_a_cut_line_and_a_second_full_run(self) -> None:
        out = self.results_path()
        with FakeServer() as srv:
            self.assertEqual(self.thinker(srv, out, "--limit", "7")[0], 0)
            self.assertEqual(len(T.ResultStore.read(out)[1]), 7)
            with open(out, "a", encoding="utf-8") as f:
                f.write('{"kind":"row","id":"zz","sta')                              # a crash in the middle of a write
            before = srv.count
            code, text = self.thinker(srv, out)
            self.assertEqual(code, 0, text)
            self.assertEqual(srv.count - before, len(self.rows) - 7 + 1)              # the rest of the rows + one preflight
            again = srv.count
            self.assertEqual(self.thinker(srv, out)[0], 0)
            self.assertEqual(srv.count - again, 1)                                    # nothing left to run: just the preflight
        recs = T.ResultStore.read(out)[1]
        self.assertEqual(sorted(r["id"] for r in recs), sorted(r["id"] for r in self.rows))
        self.assertEqual(len(recs), len(self.rows))

    def test_a_wrong_key_stops_at_the_preflight_without_writing_rows(self) -> None:
        out = self.results_path()
        with FakeServer(require_key="the-right-key") as srv:
            code, text = self.thinker(srv, out)
        self.assertEqual(code, 2, text)
        self.assertIn("401", text)
        self.assertEqual(T.ResultStore.read(out)[1], [])

    def test_a_dead_endpoint_aborts_after_consecutive_errors_and_resumes_when_it_is_back(self) -> None:
        out = self.results_path()
        broken = {"on": True}

        def behavior(body, count):
            if broken["on"] and "MODE[" in body["messages"][0]["content"]:
                return {"status": 503, "text": "overloaded"}
            return scripted(body, count)
        with FakeServer(behavior) as srv:
            code, text = self.thinker(srv, out, "--retries", "0", "--max-consecutive-errors", "3", "--concurrency", "1")
            self.assertEqual(code, 3, text)
            recs = T.ResultStore.read(out)[1]
            self.assertEqual([r["status"] for r in recs], ["error"] * 3)
            broken["on"] = False
            code, text = self.thinker(srv, out)
            self.assertEqual(code, 0, text)
        latest = T.ResultStore(out, {}, {}, 0).latest()
        self.assertEqual(sorted(latest), sorted(r["id"] for r in self.rows))
        self.assertTrue(all(r["status"] == "ok" for r in latest.values()))
        self.assertEqual({r["attempt"] for r in latest.values() if r["id"] in {x["id"] for x in recs}}, {2})

    def test_a_429_with_retry_after_is_retried(self) -> None:
        out = self.results_path()
        seen = {"n": 0}

        def behavior(body, count):
            if "MODE[" in body["messages"][0]["content"]:
                seen["n"] += 1
                if seen["n"] in (1, 2):
                    return {"status": 429, "text": "slow down", "headers": {"Retry-After": "0"}}
            return scripted(body, count)
        with FakeServer(behavior) as srv:
            code, text = self.thinker(srv, out, "--limit", "3", "--concurrency", "1")
        self.assertEqual(code, 0, text)
        recs = T.ResultStore.read(out)[1]
        self.assertEqual([r["status"] for r in recs], ["ok"] * 3)
        self.assertIn("retry", text)

    def test_a_bigger_budget_redoes_only_the_rows_the_budget_cut(self) -> None:
        out = self.results_path()

        def needs_more(row_id):                                   # rows that only finish with a budget of 200+ tokens
            return int(sha(row_id), 16) % 3 == 0

        def behavior(body, count):
            prompt = body["messages"][0]["content"]
            m = re.search(r"SYNTH (\S+?)\. HINT", prompt)
            if m and needs_more(m.group(1)) and body["max_tokens"] < 200:
                return {"content": "still thinking", "finish": "length", "usage": (5, body["max_tokens"])}
            return scripted(body, count)
        modes = {r["id"]: re.search(r"MODE\[(\w+)\]", r["state"]).group(1) for r in self.rows}
        with FakeServer(behavior) as srv:
            self.assertEqual(self.thinker(srv, out, "--max-tokens", "100")[0], 0)
            first = T.ResultStore(out, {}, {}, 0).latest()
            cut = {i for i, r in first.items() if r["truncated"]}
            self.assertEqual(cut, {i for i in modes if needs_more(i) or modes[i] == "cut"})
            self.assertTrue(any(needs_more(i) and modes[i] != "cut" for i in modes))      # the sample really has rows this test is about
            before = srv.count
            self.assertEqual(self.thinker(srv, out, "--max-tokens", "300")[0], 0)
            self.assertEqual(srv.count - before, len(cut) + 1)                              # the cut rows again, plus the preflight
        second = T.ResultStore(out, {}, {}, 0).latest()
        for rid, rec in second.items():
            self.assertEqual(rec["attempt"], 2 if rid in cut else 1, rid)
            self.assertEqual(rec["truncated"], modes[rid] == "cut", rid)                    # only the rows the server always cuts stay cut
            self.assertEqual(rec["max_tokens"], 300 if rid in cut else 100, rid)

    def test_the_cli_refuses_another_config_or_a_lower_budget_on_the_same_file(self) -> None:
        out = self.results_path()
        with FakeServer() as srv:
            self.assertEqual(self.thinker(srv, out, "--limit", "2")[0], 0)
            code, text = self.thinker(srv, out, "--temperature", "0.6")
            self.assertEqual(code, 2, text)
            self.assertIn("temperature", text)
            code, text = self.thinker(srv, out, "--max-tokens", "100")
            self.assertEqual(code, 2, text)
            self.assertIn("max_tokens", text)
            code, text = self.thinker(srv, out, "--overwrite", "--limit", "1", "--temperature", "0.6")
            self.assertEqual(code, 0, text)
        self.assertEqual(len(T.ResultStore.read(out)[1]), 1)

    def test_a_context_overflow_fails_that_row_only(self) -> None:
        out = self.results_path()

        target = self.rows[0]["id"]

        def behavior(body, count):
            if f"SYNTH {target}." in body["messages"][0]["content"]:
                return {"status": 400, "text": "This model's maximum context length is 4096 tokens"}
            return scripted(body, count)
        with FakeServer(behavior) as srv:
            code, text = self.thinker(srv, out, "--limit", "4", "--concurrency", "1")
        self.assertEqual(code, 0, text)
        latest = T.ResultStore(out, {}, {}, 0).latest()
        self.assertEqual(latest[target]["status"], "failed")
        self.assertIn("context length", latest[target]["error"])
        self.assertEqual([latest[i]["status"] for i in latest if i != target], ["ok"] * 3)

    def test_thinking_off_is_a_different_protocol(self) -> None:
        out = self.results_path()
        with FakeServer() as srv:
            self.assertEqual(self.thinker(srv, out, "--thinking", "off", "--limit", "1")[0], 0)
            body = next(r for r in srv.requests if "MODE[" in r["body"]["messages"][0]["content"])["body"]
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual((body["temperature"], body["presence_penalty"]), (0.7, 1.5))

    def test_a_manifest_that_does_not_match_the_pack_stops_the_run(self) -> None:
        out = self.results_path()
        other = self.root / "other"
        S.build(self.suite, self.results, other, per_benchmark=2, benchmarks=(25,))
        with FakeServer() as srv:
            code, text = self.thinker(srv, out, "--manifest", str(other / "manifest.json"))
        self.assertEqual(code, 2, text)
        self.assertEqual(srv.count, 0)


class CompareFlow(FlowBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.res = cls.root / "thinker.jsonl"
        with FakeServer() as srv:
            code, text = run_cli(T, ["--pack", str(cls.pack), "--out", str(cls.res), "--base-url", srv.url, "--max-tokens", "400", "--concurrency", "3"])
        assert code == 0, text

    def test_report_numbers_match_a_plain_count(self) -> None:
        out = self.root / "report"
        code, text = run_cli(K, ["--pack", str(self.pack), "--thinker", str(self.res), "--out", str(out), "--bootstrap", "300"])
        self.assertEqual(code, 0, text)
        rep = json.loads((out / "report.json").read_text())
        self.assertEqual(rep["format"], K.REPORT_FORMAT)
        self.assertTrue(rep["coverage"]["complete"])
        self.assertEqual(rep["coverage"]["rows_scored"], len(self.rows))
        self.assertEqual(rep["models"], sorted(MODELS))
        for b in self.BENCH:
            row = rep["benchmarks"][str(b)]
            rows = [r for r in self.rows if r["benchmark"] == b]
            self.assertEqual(row["n"], len(rows))
            self.assertEqual(row["systems"]["thinker"]["correct"], self.expected_hits(b), b)
            for m in MODELS:
                n_ok = 0
                for r in rows:
                    ok = True
                    for qid, q in r["questions"].items():
                        probs = r["decisions"][m][qid]["probs"]
                        best = max(probs.values())
                        pick = next(k for k in q["criteria"] if probs[k] == best)
                        ok = ok and pick in r["gold"][qid]
                    n_ok += ok
                self.assertEqual(row["systems"][m]["correct"], n_ok, (b, m))
            self.assertEqual(row["chance"], C.chance_of(b))
        self.assertIn(rep["reference"], MODELS)
        self.assertEqual(rep["verdict"]["status"], "FAIL")             # the scripted thinker is right on about 70% of the rows: not 10 points above m_good
        md = (out / "report.md").read_text()
        self.assertIn("GT VERDICT:", md)
        self.assertIn("GPQA Diamond", md)
        self.assertIn("GT VERDICT:", text)

    def test_rows_the_thinker_has_not_done_are_left_out_for_every_system(self) -> None:
        part = self.root / "partial.jsonl"
        header, recs = T.ResultStore.read(self.res)
        with open(part, "w", encoding="utf-8") as f:
            for rec in [header, *recs[:12]]:
                f.write(json.dumps(rec) + "\n")
        out = self.root / "report_partial"
        code, text = run_cli(K, ["--pack", str(self.pack), "--thinker", str(part), "--out", str(out), "--bootstrap", "100"])
        self.assertEqual(code, 0, text)
        rep = json.loads((out / "report.json").read_text())
        self.assertFalse(rep["coverage"]["complete"])
        self.assertEqual(rep["coverage"]["rows_scored"], 12)
        self.assertIn("PARTIAL", text)
        self.assertEqual(sum(b["n"] for b in rep["benchmarks"].values()), 12)

    def test_results_made_from_another_pack_are_refused(self) -> None:
        other = self.root / "other_pack"
        S.build(self.suite, self.results, other, per_benchmark=2, benchmarks=(25, 58))
        code, text = run_cli(K, ["--pack", str(other / "pack.jsonl.gz"), "--thinker", str(self.res)])
        self.assertEqual(code, 2, text)
        self.assertIn("another pack", text)

    def test_a_model_name_that_is_not_in_the_pack_is_an_error_not_a_traceback(self) -> None:
        code, text = run_cli(K, ["--pack", str(self.pack), "--thinker", str(self.res), "--models", "nope"])
        self.assertEqual(code, 2, text)
        self.assertIn("nope", text)

    def test_a_planted_win_passes_the_gate_and_a_planted_tie_does_not(self) -> None:
        # Rewrite the thinker's answers so it is right on every row: it must beat every scripted model by 10+ points.
        rows = {r["id"]: r for r in self.rows}
        header, recs = T.ResultStore.read(self.res)
        win = self.root / "win.jsonl"
        with open(win, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            for rec in recs:
                rec["answers"] = {q: rows[rec["id"]]["gold"][q][0] for q in rows[rec["id"]]["questions"]}
                f.write(json.dumps(rec) + "\n")
        code, text = run_cli(K, ["--pack", str(self.pack), "--thinker", str(win), "--bootstrap", "500"])
        self.assertEqual(code, 0, text)
        self.assertIn("GT VERDICT: PASS", text)
        # And copy the best model's own picks: delta 0, so no pass.
        rep = K.build_report(self.rows, [], B=10)
        self.assertEqual(rep["verdict"]["status"], "INCONCLUSIVE")


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("torch", "transformers", "tokenizers")), "torch, transformers and tokenizers are needed")
class TinyTransformersFlow(unittest.TestCase):
    """The transformers backend on a random 2-layer Qwen2 built here: batching, padding, stop detection, OOM splitting, resume."""

    TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n"
                "{% else %}<think>\n{% endif %}{% endif %}")

    @classmethod
    def tiny_tokenizer(cls, d):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast
        tk = Tokenizer(models.BPE())
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
        trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tk.train_from_iterator(["ANSWER q: A the quick brown fox jumps over the lazy dog", "Options: A. one B. two C. three D. four"] * 20, trainer)
        tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
        tok.chat_template = cls.TEMPLATE
        tok.save_pretrained(d)
        return tok

    @classmethod
    def setUpClass(cls) -> None:
        import torch
        from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name) / "tiny"
        tok = cls.tiny_tokenizer(d)
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=1024, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        model = Qwen2ForCausalLM(cfg)
        model.generation_config = GenerationConfig(eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        model.save_pretrained(d)
        cls.dir = str(d)
        cls.vocab = len(tok)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def backend(self, **kw):
        kw.setdefault("sampling", {"temperature": 0.0})
        return T.TransformersBackend(model=self.dir, quant="none", dtype="float32", device_map="cpu", max_tokens=kw.pop("max_tokens", 10), **kw)

    def items(self, n):
        return [T.Item(f"i{k}", "Options: A. one B. two. " * (k + 1) + "Pick one.") for k in range(n)]

    def test_template_gets_the_thinking_switch(self) -> None:
        b = self.backend()
        self.assertTrue(b.render("hi").endswith("<|im_start|>assistant\n<think>\n"))
        off = self.backend(chat_template_kwargs={"enable_thinking": False})
        self.assertTrue(off.render("hi").endswith("</think>\n\n"))
        self.assertTrue(self.backend(reasoning_effort="low").render("hi").endswith("<think>\n"))

    def test_batch_of_different_lengths_gives_one_result_per_item(self) -> None:
        b = self.backend(batch_size=3)
        out = b.generate(self.items(3))
        self.assertEqual(len(out), 3)
        for g in out:
            self.assertIn(g.finish_reason, ("stop", "length"))
            self.assertLessEqual(g.completion_tokens, 10)
            self.assertGreater(g.prompt_tokens, 0)
            self.assertIsInstance(g.content, str)
            self.assertIsNone(g.error)
        self.assertEqual(len({g.prompt_tokens for g in out}), 3)          # each item keeps its own prompt length despite the left padding

    def test_greedy_decoding_is_repeatable_and_does_not_depend_on_the_padding(self) -> None:
        b = self.backend()
        items = self.items(2)
        together = b.generate(items)
        again = b.generate(items)
        alone = b.generate(items[:1])
        self.assertEqual([g.content for g in together], [g.content for g in again])
        self.assertEqual(together[0].content, alone[0].content)

    def test_a_budget_hit_is_reported_as_length_and_reads_as_unparseable(self) -> None:
        b = self.backend(max_tokens=3)
        g = b.generate(self.items(1))[0]
        self.assertEqual(g.finish_reason, "length")
        self.assertEqual(g.completion_tokens, 3)
        p = T.interpret(g, P.build_prompt({"state": "s", "questions": {"q": {"type": "choice", "criteria": {"A": "a", "B": "b"}}}}).options)
        self.assertEqual((p.answers, p.truncated), ({"q": None}, True))

    def test_out_of_memory_splits_the_batch_and_a_single_oom_is_a_retryable_error(self) -> None:
        import torch
        b = self.backend(batch_size=4)
        real = b._generate
        sizes = []

        def flaky(items):
            sizes.append(len(items))
            if len(items) > 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
            return real(items)
        b._generate = flaky
        out = b.generate(self.items(4))
        self.assertEqual(len(out), 4)
        self.assertEqual(sizes, [4, 2, 1, 1, 2, 1, 1])
        self.assertTrue(all(g.error is None for g in out))

        def always(items):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
        b._generate = always
        g = b.generate(self.items(1))[0]
        self.assertTrue(g.transient)
        self.assertIn("out of memory", g.error)

    def test_runner_end_to_end_with_resume_on_the_tiny_model(self) -> None:
        rows = [{"id": f"25:{i}", "benchmark": 25, "state": f"s{i}", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"A": "a", "B": "b", "C": "c"}}},
                 "gold": {"q": ["A"]}, "decisions": {}} for i in range(5)]
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = Path(d.name) / "r.jsonl"
        cfg = {"backend": "transformers", "model": "tiny", "revision": None, "prompt_version": C.PROMPT_VERSION, "sampling": {"temperature": 0.0},
               "chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": None, "extra_body": {}, "quantization": None, "max_tokens": 6, "seed": None, "endpoint": None}
        b = self.backend(batch_size=2, max_tokens=6)
        st = T.run(rows, b, T.ResultStore.open(path, cfg, "p" * 64, 5), max_tokens=6, limit=3, log=lambda s: None)
        self.assertEqual(st.done, 3)
        st = T.run(rows, b, T.ResultStore.open(path, cfg, "p" * 64, 5), max_tokens=6, log=lambda s: None)
        self.assertEqual(st.done, 2)
        recs = T.ResultStore.read(path)[1]
        self.assertEqual(sorted(r["id"] for r in recs), [r["id"] for r in rows])
        self.assertTrue(all(r["status"] == "ok" and r["tokens"]["completion"] <= 6 for r in recs))


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("torch", "transformers", "tokenizers")), "torch, transformers and tokenizers are needed")
class TinyHybridFlow(unittest.TestCase):
    """The architecture of the real model, tiny and random: Gated DeltaNet layers plus a full-attention layer, with the vision tower config.

    It runs the torch fallback of the DeltaNet kernels (no fla, no causal-conv1d on this CPU), so it says nothing about speed on a T4;
    it says that the backend loads this class, batches with left padding, and gives the same text as one row at a time.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from transformers import GenerationConfig, Qwen3_5Config, Qwen3_5ForConditionalGeneration
        except ImportError:
            raise unittest.SkipTest("this transformers has no Qwen3_5")
        import torch
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name) / "hybrid"
        tok = TinyTransformersFlow.tiny_tokenizer.__func__(TinyTransformersFlow, d)
        torch.manual_seed(0)
        text = dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=16, layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                    linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                    max_position_embeddings=512, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, full_attention_interval=4)
        vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=4, spatial_merge_size=2,
                      temporal_patch_size=2, in_channels=3, num_position_embeddings=16)
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision))
        model.generation_config = GenerationConfig(eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        model.save_pretrained(d)
        cls.dir = str(d)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def backend(self, **kw):
        return T.TransformersBackend(model=self.dir, quant="none", dtype="float32", device_map="cpu", max_tokens=12,
                                     sampling={"temperature": 0.0}, **kw)

    @staticmethod
    def items(n):
        return [T.Item(f"i{k}", "Options: A. one B. two. " * (k + 1) + "Pick one.") for k in range(n)]

    def test_the_hybrid_class_loads_and_generates(self) -> None:
        out = self.backend(batch_size=3).generate(self.items(3))
        self.assertEqual(len(out), 3)
        for g in out:
            self.assertIsNone(g.error)
            self.assertEqual((g.completion_tokens, g.finish_reason), (12, "length"))

    def test_a_padded_batch_gives_the_text_of_one_row_at_a_time(self) -> None:
        items = self.items(3)                                         # three lengths: the short ones are left-padded in the batch
        batched = self.backend(batch_size=3).generate(items)
        single = self.backend(batch_size=1)
        for it, got in zip(items, batched):
            want = single.generate([it])[0]
            self.assertEqual((got.reasoning, got.content, got.completion_tokens), (want.reasoning, want.content, want.completion_tokens), it.id)


if __name__ == "__main__":
    unittest.main()
