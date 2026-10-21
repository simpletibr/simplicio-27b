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
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from decide.think import common as C  # noqa: E402
from decide.think import compare as K  # noqa: E402
from decide.think import prompt as P  # noqa: E402
from decide.think import sample as S  # noqa: E402
from decide.think import thinker as T  # noqa: E402
import think_nb_support as NB  # noqa: E402

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
        return {"reasoning": think, "content": final, "finish": "stop", "usage": (50, 400)}
    return {"content": think + "\n</think>\n\n" + final, "finish": "stop", "usage": (50, 400)}


def terse(body, count):
    """An endpoint that ignored the request for thinking: the answer line and nothing else."""
    prompt = body["messages"][0]["content"]
    if "MODE[" not in prompt:
        return {"content": "A", "finish": "stop", "usage": (5, 3)}
    right = re.search(r"HINT\[([^\]]*)\]", prompt).group(1)
    final = "\n".join(f"ANSWER {kv.split('=')[0]}: {kv.split('=')[1]}" for kv in right.split(";"))
    return {"content": final, "finish": "stop", "usage": (50, 12)}


def hidden(body, count):
    """A provider that thinks and returns only the final reply: 3000 tokens were generated and none of them is visible."""
    out = terse(body, count)
    return {**out, "usage": (50, 3000)} if "MODE[" in body["messages"][0]["content"] else out


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
        argv = ["--pack", str(self.pack), "--out", str(out), "--base-url", server.url, "--max-tokens", "400", "--concurrency", "3",
                "--revision", C.QWEN_REVISION, *extra]
        return run_cli(T, argv, env={"OPENAI_API_KEY": SECRET, **(env or {})})

    def compare(self, thinker, *extra, out=None, manifest=True):
        """compare.py as the gate runs it: the manifest of this synthetic sample stands for the committed one."""
        argv = ["--pack", str(self.pack), "--thinker", str(thinker), *(["--manifest", str(self.manifest)] if manifest else []), *extra]
        if out is not None:
            argv += ["--out", str(out)]
        return run_cli(K, argv)

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
            self.assertNotIn("reasoning", rec, "the thinking text is evaluation data the verdict does not need: it is not stored unless asked")
            self.assertIs(rec["thought"], True)
            self.assertEqual((rec["no_close"], rec["garbled"]), (False, False))
            if mode == "right":
                self.assertEqual(set(rec["read_by"].values()), {"marker"})
            self.assertEqual(rec["endpoint"], T.sanitize_url(srv.url))
        self.assertTrue(all(r["auth"] == f"Bearer {SECRET}" for r in srv.requests))
        self.assertTrue(all(r["ua"] and "Python-urllib" not in r["ua"] for r in srv.requests))
        first = next(r for r in srv.requests if "MODE[" in r["body"]["messages"][0]["content"])["body"]
        self.assertEqual((first["temperature"], first["top_p"], first["top_k"], first["max_tokens"]), (1.0, 0.95, 20, 400))
        self.assertEqual(first["chat_template_kwargs"], {"enable_thinking": True})

    def test_an_endpoint_that_does_not_think_is_called_out_in_the_log_and_in_the_report(self) -> None:
        out = self.results_path()
        with FakeServer(behavior=terse) as srv:
            code, text = self.thinker(srv, out, "--limit", "4")
        self.assertEqual(code, 0, text)
        self.assertIn("WARNING: 12 tokens per reply on average: the model barely thought", text)
        rep = self.root / "terse-report"
        code, md = self.compare(out, out=rep)
        self.assertEqual(code, 0, md)
        self.assertIn("AVISO: a mediana e de 12 tokens por linha: o modelo quase nao pensou", md)
        last = md.strip().splitlines()[-1]
        self.assertTrue(last.startswith("GT VERDICT: INCONCLUSIVE"), last)         # the line `hosted` and the Modal app show is the one that says it
        self.assertIn("thinking not detected", last)
        self.assertEqual(json.loads((rep / "report.json").read_text())["verdict"]["reasons"], ["thinking_not_detected"])
        rows = T.ResultStore.read(out)[1]
        self.assertTrue(rows)
        self.assertTrue(all(r["thought"] is False and r["no_close"] is False and r["reasoning_hidden"] is False for r in rows),
                        "an endpoint is not the kit's own template: the bare reply is read as it is, and 12 tokens is too short to be hidden thinking")
        self.assertTrue(all(None not in r["answers"].values() for r in rows), "the answer line is read")

    def test_an_endpoint_that_thinks_and_hides_it_is_never_an_official_verdict(self) -> None:
        # Review of #78, second round: the reviewer's probe read this as a thinker at 0% and printed an official FAIL.
        out = self.results_path()
        with FakeServer(behavior=hidden) as srv:
            code, text = self.thinker(srv, out, "--limit", "4")
        self.assertEqual(code, 0, text)
        self.assertIn("show no reasoning: the endpoint probably thinks and hides it", text)
        rows = T.ResultStore.read(out)[1]
        self.assertTrue(rows and all(r["reasoning_hidden"] is True and r["no_close"] is False and r["thought"] is True for r in rows))
        self.assertTrue(all(None not in r["answers"].values() for r in rows), "the final reply is read, not thrown away")
        rep = self.root / "hidden-report"
        code, md = self.compare(out, out=rep)
        self.assertEqual(code, 0, md)
        last = md.strip().splitlines()[-1]
        self.assertTrue(last.startswith("GT VERDICT: INCONCLUSIVE"), last)
        self.assertFalse(last.startswith(("GT VERDICT: PASS", "GT VERDICT: FAIL")), last)
        self.assertIn("hid their reasoning", last)
        verdict = json.loads((rep / "report.json").read_text())["verdict"]
        self.assertEqual(verdict["reasons"], ["reasoning_hidden_or_not_closed"])
        self.assertIn("reasoning_hidden_or_not_closed", md)
        self.assertIn("pensa e esconde o raciocinio", md)

    def test_a_thinking_endpoint_and_a_thinking_off_control_get_no_such_warning(self) -> None:
        out = self.results_path()
        with FakeServer() as srv:
            code, text = self.thinker(srv, out, "--limit", "4")
        self.assertEqual(code, 0, text)
        self.assertNotIn("barely thought", text)
        code, md = self.compare(out)
        self.assertNotIn("quase nao pensou", md)
        self.assertNotIn("thinking not detected", md)
        control = self.results_path("off.jsonl")
        with FakeServer(behavior=terse) as srv:
            code, text = self.thinker(srv, control, "--limit", "4", "--thinking", "off")
        self.assertEqual(code, 0, text)
        self.assertNotIn("barely thought", text, "a control without thinking is supposed to be terse")
        code, md = self.compare(control)
        self.assertNotIn("quase nao pensou", md)
        last = md.strip().splitlines()[-1]
        self.assertTrue(last.startswith("GT VERDICT: PARTIAL") and "thinking_off" in last, last)            # a control is read whole, and it is not the gate
        self.assertNotIn("INCONCLUSIVE", last)

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

    def test_the_thinking_text_is_stored_only_with_keep_reasoning(self) -> None:
        out, kept = self.results_path(), self.results_path("kept.jsonl")
        with FakeServer() as srv:
            self.assertEqual(self.thinker(srv, out, "--limit", "2")[0], 0)
            self.assertEqual(self.thinker(srv, kept, "--limit", "2", "--keep-reasoning")[0], 0)
            self.assertEqual(self.thinker(srv, self.results_path("old.jsonl"), "--limit", "1", "--no-reasoning")[0], 0, "the old flag is still accepted")
        self.assertTrue(all("reasoning" not in r for r in T.ResultStore.read(out)[1]))
        self.assertTrue(all(r["reasoning"] for r in T.ResultStore.read(kept)[1]))

    def test_a_row_that_keeps_failing_is_given_up_after_max_attempts_and_counts_as_wrong(self) -> None:
        out = self.results_path()
        target = self.rows[0]["id"]

        def behavior(body, count):
            if f"SYNTH {target}." in body["messages"][0]["content"]:
                return {"status": 503, "text": "overloaded"}
            return scripted(body, count)
        with FakeServer(behavior) as srv:
            for k in (1, 2):
                code, text = self.thinker(srv, out, "--retries", "0", "--max-attempts", "3", "--limit", "3", "--concurrency", "1")
                self.assertEqual(code, 0, text)
                self.assertEqual(T.ResultStore(out, {}, {}, 0).latest()[target]["status"], "error", k)
            code, text = self.thinker(srv, out, "--retries", "0", "--max-attempts", "3", "--limit", "3", "--concurrency", "1")
            self.assertEqual(code, 0, text)
            self.assertIn("1 failed", text)
            last = T.ResultStore(out, {}, {}, 0).latest()[target]
            self.assertEqual((last["status"], last["gave_up"]), ("failed", 3))
            self.assertEqual(self.thinker(srv, out, "--retries", "0")[0], 0)                           # the rest of the pack goes on
            code, text = self.compare(out)
        report = text.strip().splitlines()[-1]
        self.assertTrue(report.startswith("GT VERDICT: FAIL") or report.startswith("GT VERDICT: PASS"), report)        # complete: nothing was dropped
        scored = K.score_pack(self.rows, T.ResultStore.read(out)[1])
        self.assertEqual(scored.complete, True)
        self.assertEqual(scored.hits["thinker"][self.rows[0]["benchmark"]][0], 0)
        self.assertIn("1 linhas desistidas", text)

    def test_retry_failed_and_the_attempt_limit_are_options(self) -> None:
        out = self.results_path()
        target = self.rows[0]["id"]
        down = {"on": True}

        def behavior(body, count):
            if down["on"] and f"SYNTH {target}." in body["messages"][0]["content"]:
                return {"status": 503, "text": "overloaded"}
            return scripted(body, count)
        with FakeServer(behavior) as srv:
            self.assertEqual(self.thinker(srv, out, "--retries", "0", "--max-attempts", "1", "--limit", "2")[0], 0)
            self.assertEqual(T.ResultStore(out, {}, {}, 0).latest()[target]["status"], "failed")
            down["on"] = False
            self.assertEqual(self.thinker(srv, out, "--limit", "2")[0], 0)
            self.assertEqual(T.ResultStore(out, {}, {}, 0).latest()[target]["status"], "failed", "failed rows are not run again unless asked")
            self.assertEqual(self.thinker(srv, out, "--limit", "2", "--retry-failed")[0], 0)
        self.assertEqual(T.ResultStore(out, {}, {}, 0).latest()[target]["status"], "ok")
        for bad in ("0", "-1", "x"):
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    T.build_parser().parse_args(["--pack", "p", "--out", "o", "--max-attempts", bad])

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


class Sink:
    """A server that records every request it gets, of any method, with the headers that matter."""

    def __init__(self):
        self.seen = []

    def __enter__(self):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def handle_any(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                outer.seen.append({"method": self.command, "path": self.path, "auth": self.headers.get("Authorization")})
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
            do_GET = do_POST = handle_any

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(5)
        return False


class Redirector:
    """An endpoint that answers every request with `code` and a Location that points at `target`."""

    def __init__(self, code, target):
        self.code, self.target = code, target

    def __enter__(self):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def handle_any(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                self.send_response(outer.code)
                self.send_header("Location", outer.target)
                self.send_header("Content-Length", "0")
                self.end_headers()
            do_GET = do_POST = handle_any

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


class KeyLeakFlow(FlowBase):
    """Review of #78, blocker 3: urllib re-sends Authorization to the host a 301/302/303 names.  Two real servers on two ports."""

    CODES = (301, 302, 303, 307, 308)

    def test_the_harness_sees_a_leak_when_the_default_opener_is_used(self) -> None:
        import urllib.request
        for code in (301, 302, 303):
            with Sink() as sink, Redirector(code, "PLACEHOLDER") as redirect:
                redirect.target = sink.url
                req = urllib.request.Request(redirect.url + "/chat/completions", data=b"{}", method="POST",
                                             headers={"Authorization": f"Bearer {SECRET}", "Content-Type": "application/json"})
                with contextlib.suppress(Exception), urllib.request.urlopen(req, timeout=5):
                    pass
                self.assertEqual([x["auth"] for x in sink.seen], [f"Bearer {SECRET}"], f"control for {code}: the sink must be able to see a leak")

    def test_the_key_goes_nowhere_the_endpoint_redirects_to(self) -> None:
        for code in self.CODES:
            with Sink() as sink, Redirector(code, "PLACEHOLDER") as redirect:
                redirect.target = sink.url
                backend = T.OpenAIBackend(base_url=redirect.url, model="m", api_key=SECRET, stream=False, retries=0, sleep=lambda s: None)
                with self.assertRaises(T.BackendError, msg=code) as cm:
                    backend.generate([T.Item("a", "p")])
                self.assertIn("redirect", str(cm.exception))
                with self.assertRaises(T.BackendError, msg=code):
                    backend.preflight()
                self.assertEqual(sink.seen, [], f"{code}: the second server got a request")

    def test_the_cli_stops_at_the_preflight_and_writes_no_row(self) -> None:
        out = self.results_path()
        with Sink() as sink, Redirector(302, "PLACEHOLDER") as redirect:
            redirect.target = sink.url
            code, text = run_cli(T, ["--pack", str(self.pack), "--out", str(out), "--base-url", redirect.url, "--max-tokens", "400"],
                                 env={"OPENAI_API_KEY": SECRET})
            self.assertEqual(sink.seen, [])
        self.assertEqual(code, 2, text)
        self.assertIn("redirect", text)
        self.assertNotIn(SECRET, text)
        self.assertEqual(T.ResultStore.read(out)[1], [])


class CompareFlow(FlowBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.res = cls.root / "thinker.jsonl"
        with FakeServer() as srv:
            code, text = run_cli(T, ["--pack", str(cls.pack), "--out", str(cls.res), "--base-url", srv.url, "--max-tokens", "400", "--concurrency", "3",
                                     "--revision", C.QWEN_REVISION])
        assert code == 0, text

    def test_report_numbers_match_a_plain_count(self) -> None:
        out = self.root / "report"
        code, text = self.compare(self.res, out=out)
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
        self.assertEqual((rep["config"]["bootstrap_b"], rep["verdict"]["exploratory"]), (C.BOOTSTRAP_B, []))
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
        code, text = self.compare(part, out=out)
        self.assertEqual(code, 0, text)
        rep = json.loads((out / "report.json").read_text())
        self.assertFalse(rep["coverage"]["complete"])
        self.assertEqual(rep["coverage"]["rows_scored"], 12)
        self.assertEqual(rep["verdict"]["status"], "PARTIAL")
        last = text.strip().splitlines()[-1]
        self.assertTrue(last.startswith("GT VERDICT: PARTIAL"), last)
        self.assertNotIn("GT VERDICT: PASS", text)
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
        code, text = self.compare(win)
        self.assertEqual(code, 0, text)
        self.assertEqual(text.strip().splitlines()[-1][:16], "GT VERDICT: PASS", text)
        rep = K.build_report(self.rows, [], B=10)
        self.assertEqual(rep["verdict"]["status"], "INCONCLUSIVE")

    def test_the_same_win_is_exploratory_the_moment_the_comparison_is_changed(self) -> None:
        rows = {r["id"]: r for r in self.rows}
        header, recs = T.ResultStore.read(self.res)
        win = self.root / "win2.jsonl"
        with open(win, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            for rec in recs:
                rec["answers"] = {q: rows[rec["id"]]["gold"][q][0] for q in rows[rec["id"]]["questions"]}
                f.write(json.dumps(rec) + "\n")
        cases = {"bootstrap_b": ["--bootstrap", "500"], "seed": ["--seed", "1"], "min_delta": ["--min-delta", "5"], "models_subset": ["--models", "m_good,m_mid"]}
        for code_name, extra in cases.items():
            code, text = self.compare(win, *extra)
            last = text.strip().splitlines()[-1]
            self.assertEqual(code, 0, text)
            self.assertTrue(last.startswith("GT VERDICT: EXPLORATORY") and code_name in last, last)
            self.assertNotIn("GT VERDICT: PASS", text)
        code, text = self.compare(win, "--models", ",".join(sorted(MODELS)))
        self.assertTrue(text.strip().splitlines()[-1].startswith("GT VERDICT: PASS"), "naming all the models is the registered comparison")

    def test_a_pack_that_is_not_the_manifests_is_exploratory_and_the_committed_manifest_is_the_default(self) -> None:
        other = self.root / "other_manifest"
        S.build(self.suite, self.results, other, per_benchmark=2, benchmarks=(25, 58))
        code, text = self.compare(self.res, "--manifest", str(other / "manifest.json"))
        self.assertEqual(code, 0, text)
        last = text.strip().splitlines()[-1]
        self.assertTrue(last.startswith("GT VERDICT: EXPLORATORY") and "pack_not_manifest" in last, last)
        self.assertIn("manifest:", text)
        code, text = self.compare(self.res, manifest=False)                           # no --manifest: the committed one, and this sample is synthetic
        self.assertIn("pack_not_manifest", text.strip().splitlines()[-1])
        code, text = self.compare(self.res, "--manifest", str(self.root / "nope.json"))
        self.assertIn("manifest_not_checked", text.strip().splitlines()[-1])

    def test_ignoring_a_pack_mismatch_is_exploratory(self) -> None:
        other = self.root / "other_pack2"
        S.build(self.suite, self.results, other, per_benchmark=2, benchmarks=(25, 58))
        code, text = run_cli(K, ["--pack", str(other / "pack.jsonl.gz"), "--thinker", str(self.res), "--manifest", str(other / "manifest.json"), "--ignore-pack-mismatch"])
        self.assertEqual(code, 0, text)
        last = text.strip().splitlines()[-1]
        self.assertTrue(last.startswith("GT VERDICT: EXPLORATORY") and "pack_mismatch_ignored" in last, last)

    def test_hosted_hands_its_manifest_to_the_comparison(self) -> None:
        seen = []
        real = K.main
        with mock.patch.object(K, "main", side_effect=lambda argv=None: seen.append(list(argv)) or real(argv)):
            from decide.think import hosted as H
            with FakeServer() as srv:
                class Up:
                    url = srv.url

                    def alive(self):
                        return True

                    def stop(self):
                        pass

                    def tail(self):
                        return ""
                work = self.root / "hosted-manifest"
                with contextlib.redirect_stdout(io.StringIO()):
                    H.run_gate(pack=str(self.pack), work=str(work), manifest=str(self.manifest), start_server=lambda w: Up(), wait=lambda *a, **k: True,
                               limit=2, log=lambda s: None, concurrency=1)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][seen[0].index("--manifest") + 1], str(self.manifest))


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

    def test_nf4_is_asked_for_in_the_dtype_that_was_chosen(self) -> None:
        """bitsandbytes is not installed here: a stand-in config object and a loader that sees what the backend asks for."""
        import torch
        import transformers
        real = transformers.AutoModelForCausalLM.from_pretrained
        for name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16), ("float32", torch.float32)):
            seen = {}

            def fake_bnb(**kw):
                seen["bnb"] = kw
                return "the-config"

            def fake_load(model, **kw):
                seen["load"] = dict(kw)
                kw.pop("quantization_config", None)                    # the random model cannot be quantized here
                return real(model, **kw)
            with mock.patch.object(transformers, "BitsAndBytesConfig", fake_bnb), \
                    mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained", side_effect=fake_load):
                T.TransformersBackend(model=self.dir, quant="nf4", dtype=name, device_map="cpu", max_tokens=4)
            self.assertIs(seen["bnb"]["bnb_4bit_compute_dtype"], dtype, name)
            self.assertEqual(seen["bnb"]["bnb_4bit_quant_type"], "nf4")
            self.assertEqual(seen["load"]["quantization_config"], "the-config")
            self.assertIs(seen["load"]["dtype"], dtype, name)

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

    def test_out_of_memory_splits_the_batch_and_a_single_oom_is_terminal(self) -> None:
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
        self.assertFalse(g.transient, "one sequence that does not fit will not fit the second time either: the row is terminal")
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


# ------------------------------------------------------------------------------------------- notebooks

def git_env() -> dict:
    """The environment for a git command that must touch only the throwaway repository it is aimed at (decide.think.common.clean_git_env)."""
    return C.clean_git_env()


class NotebookFlow(FlowBase):
    """The notebooks of decide/think, cell by cell.

    The shell lines of the API notebook run for real (a subprocess, like `!python -m ...`) against the fake
    endpoint; the Python cells run with `exec` on fakes for what a CPU box does not have (Colab and Kaggle
    secrets, nvidia-smi, the free disk, github.com, the vLLM server).  What cannot run here at all is the
    install of the pins and the model itself: the README lists it as UNVERIFIED.
    """

    def setUp(self) -> None:
        self.addCleanup(os.chdir, os.getcwd())                          # the fetch cell changes directory, like a notebook does

    def shell(self, command, ns, env=None):
        argv = [str(self.manifest) if a == NB.MANIFEST_PATH else a for a in NB.argv_of(command, ns)]      # the synthetic manifest
        clean = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY") and k not in ("THINK_BASE_URL", "OPENAI_BASE_URL")}
        return subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=600, env={**clean, **(env or {})})

    def workdir(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        return d

    def run_cell(self, key, needle, ns, replace=()):
        """exec the one code cell that holds `needle`, with its `!` lines dropped (the other tests run those)."""
        text = NB.code_cell_with(key, needle)
        for old, new in replace:
            self.assertEqual(text.count(old), 1, old)
            text = text.replace(old, new)
        src = "\n".join(line[: len(line) - len(line.lstrip())] + "pass" if line.lstrip().startswith("!") else line for line in text.split("\n"))
        exec(compile(src, f"{key}:{needle[:24]}", "exec"), ns)
        return ns

    # ---------------------------------------------------------------- the three commands, in the order of the notebook

    def test_the_api_notebook_commands_run_in_order_and_the_second_run_resumes(self) -> None:
        work = self.workdir()
        outputs = []
        with FakeServer(require_key=SECRET) as srv:
            def prompts():
                return sum(1 for q in srv.requests if "MODE[" in q["body"]["messages"][0]["content"])

            def go(module, ns, env=None):
                r = self.shell(NB.commands_of("api", f"decide.think.{module}")[0], ns, env)
                outputs.append(r.stdout + r.stderr)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                return r.stdout.strip().splitlines()

            probe = NB.parameters("api", PACK=str(self.pack), WORK=str(work), BASE_URL=srv.url, LIMIT="3")
            self.assertIn("OK", go("sample", probe)[-1])
            go("thinker", probe, {"THINK_API_KEY": SECRET})
            self.assertEqual(prompts(), 3)
            last = go("compare", probe)[-1]
            self.assertTrue(last.startswith("GT VERDICT:"), last)
            self.assertIn("PARTIAL", last)

            full = NB.parameters("api", PACK=str(self.pack), WORK=str(work), BASE_URL=srv.url, LIMIT="")
            go("thinker", full, {"THINK_API_KEY": SECRET})
            self.assertEqual(prompts(), len(self.rows), "a row was asked twice or not at all")
            last = go("compare", full)[-1]
            self.assertTrue(last.startswith("GT VERDICT:"), last)
            self.assertNotIn("PARTIAL", last)
        report = json.loads((work / "report" / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["coverage"]["complete"])
        self.assertEqual(report["coverage"]["rows_scored"], len(self.rows))
        self.assertTrue((work / "report" / "report.md").is_file())
        self.assertEqual(len(T.ResultStore.read(work / "results.jsonl")[1]), len(self.rows))
        for text in outputs + [(work / "results.jsonl").read_text(encoding="utf-8")]:
            self.assertNotIn(SECRET, text)
        for marker in ("SYNTH", "HINT["):
            self.assertFalse(any(marker in o for o in outputs), f"question text reached the log: {marker}")

    def test_the_report_says_which_model_and_revision_the_notebook_declared(self) -> None:
        for revision, warned in (("", True), (C.QWEN_FP8_REVISION, False)):
            with self.subTest(revision=revision or "none"):
                work = self.workdir()
                with FakeServer() as srv:
                    ns = NB.parameters("api", PACK=str(self.pack), WORK=str(work), BASE_URL=srv.url, LIMIT="2", REVISION=revision)
                    for module in ("thinker", "compare"):
                        r = self.shell(NB.commands_of("api", f"decide.think.{module}")[0], ns)
                        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                md = (work / "report" / "report.md").read_text(encoding="utf-8")
                self.assertIn(f"Thinker: {C.QWEN_FP8_MODEL}", md)
                self.assertEqual("AVISO: modelo ou revisao diferente do par fixado" in md, warned)

    def test_a_different_budget_on_the_same_file_is_refused_not_mixed(self) -> None:
        work = self.workdir()
        with FakeServer() as srv:
            kw = dict(PACK=str(self.pack), WORK=str(work), BASE_URL=srv.url, LIMIT="2")
            cmd = NB.commands_of("api", "decide.think.thinker")[0]
            self.assertEqual(self.shell(cmd, NB.parameters("api", **kw)).returncode, 0)
            r = self.shell(cmd, NB.parameters("api", MAX_TOKENS=8192, **kw))
        self.assertNotEqual(r.returncode, 0, "a lower budget would mix two protocols in one file")

    # ---------------------------------------------------------------- the cells that need Colab, Kaggle or github

    def test_the_key_cell_reads_the_secret_without_showing_it(self) -> None:
        needle = "from google.colab import userdata"
        base = dict(os=os, KEY_NAME="MY_THINK_KEY", MODE="endpoint", BASE_URL="https://example.org/v1", PACK=str(self.pack), WORK=str(self.workdir()))
        for source in ("env", "none"):
            with self.subTest(source=source), mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("THINK_API_KEY", None)
                os.environ.pop("MY_THINK_KEY", None)
                if source == "env":
                    os.environ["MY_THINK_KEY"] = SECRET
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    ns = self.run_cell("api", needle, dict(base))
                self.assertNotIn(SECRET, out.getvalue())
                self.assertNotIn("key", ns, "the key stays in a variable of the notebook")
                self.assertEqual(os.environ.get("THINK_API_KEY"), SECRET if source == "env" else None)
                self.assertIn("chave encontrada" if source == "env" else "sem chave", out.getvalue())

    def test_the_key_cell_keeps_the_key_of_an_endpoint_out_of_a_local_vllm(self) -> None:
        # Review of #78, second round: the vLLM the notebook starts has no key, so no key goes to it, whatever is in the environment.
        needle = "from google.colab import userdata"
        base = dict(os=os, KEY_NAME="MY_THINK_KEY", MODE="local-vllm", BASE_URL="https://SEU-ENDPOINT/v1", PACK=str(self.pack), WORK=str(self.workdir()))
        with mock.patch.dict(os.environ, {"MY_THINK_KEY": SECRET, "THINK_API_KEY": SECRET, "OPENAI_API_KEY": SECRET}, clear=False):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.run_cell("api", needle, dict(base))
            self.assertNotIn(SECRET, out.getvalue())
            self.assertIn("ela não é usada", out.getvalue())
            self.assertNotIn("THINK_API_KEY", os.environ)
            self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_the_checks_refuse_the_placeholder_endpoint_a_wrong_mode_and_a_missing_pack(self) -> None:
        needle = "from google.colab import userdata"
        base = dict(os=os, KEY_NAME="MY_THINK_KEY", MODE="endpoint", BASE_URL="https://example.org/v1", PACK=str(self.pack), WORK=str(self.workdir()))
        for bad, why in (({"BASE_URL": "https://SEU-ENDPOINT/v1"}, "BASE_URL"), ({"MODE": "local"}, "MODE"), ({"PACK": "/nonexistent/pack.jsonl.gz"}, "pack")):
            with self.subTest(why), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(AssertionError):
                    self.run_cell("api", needle, {**base, **bad})
        with contextlib.redirect_stdout(io.StringIO()):
            self.run_cell("api", needle, {**base, "MODE": "local-vllm", "BASE_URL": "https://SEU-ENDPOINT/v1"})        # filled in later by the vLLM cell

    def test_the_vllm_cell_starts_the_pinned_fp8_server_on_loopback(self) -> None:
        from decide.think import serve as serve_module
        work = self.workdir()
        calls = []

        class FakePopen:
            def __init__(self, cmd, **kw):
                calls.append(("popen", cmd, kw))

        def fake_run(cmd, **kw):
            calls.append(("run", cmd, kw))

        ns = dict(MODE="local-vllm", MODEL=C.QWEN_FP8_MODEL, REVISION="", WORK=str(work), sys=sys, os=os,
                  subprocess=type("SP", (), {"run": staticmethod(fake_run), "Popen": FakePopen, "STDOUT": -2}), REV="", q=shlex.quote)
        with mock.patch.object(serve_module, "wait_ready", return_value=True) as ready, contextlib.redirect_stdout(io.StringIO()):
            self.run_cell("api", 'if MODE == "local-vllm":', ns)
        install = next(c for c in calls if c[0] == "run")[1]
        self.assertEqual(install[-1], f"vllm=={serve_module.VLLM_VERSION}")
        started = next(c for c in calls if c[0] == "popen")[1]
        self.assertEqual(started, serve_module.vllm_command(C.QWEN_FP8_MODEL, C.QWEN_FP8_REVISION))
        self.assertEqual(started[started.index("--host") + 1], "127.0.0.1")
        ns["log"].close()
        self.assertEqual(ns["BASE_URL"], serve_module.base_url())
        self.assertEqual(ns["REV"], f"--revision {C.QWEN_FP8_REVISION}")
        self.assertEqual(ready.call_args[0][:2], (ns["BASE_URL"], 1800))
        self.assertIsNone(ready.call_args[1].get("api_key"), "the local server takes no key")
        self.assertTrue((work / "vllm.log").is_file())

    def test_the_vllm_cell_does_not_send_a_key_to_the_local_server_and_drops_it_from_the_environment(self) -> None:
        from decide.think import serve as serve_module
        ns = dict(MODE="local-vllm", MODEL=C.QWEN_FP8_MODEL, REVISION="", WORK=str(self.workdir()), sys=sys, os=os,
                  subprocess=type("SP", (), {"run": staticmethod(lambda cmd, **kw: None), "Popen": lambda cmd, **kw: None, "STDOUT": -2}),
                  REV="", q=shlex.quote)
        with mock.patch.dict(os.environ, {"THINK_API_KEY": SECRET, "OPENAI_API_KEY": SECRET}, clear=False):
            with mock.patch.object(serve_module, "wait_ready", return_value=True) as ready, contextlib.redirect_stdout(io.StringIO()):
                self.run_cell("api", 'if MODE == "local-vllm":', ns)
            ns["log"].close()
            self.assertIsNone(ready.call_args[1].get("api_key"))
            self.assertNotIn(SECRET, repr(ready.call_args))
            self.assertNotIn("THINK_API_KEY", os.environ)
            self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_the_vllm_cell_leaves_the_key_of_an_endpoint_alone(self) -> None:
        ns = dict(MODE="endpoint", MODEL=C.QWEN_FP8_MODEL, REVISION="", WORK=str(self.workdir()), sys=sys, os=os, subprocess=None, REV="", BASE_URL="https://example.org/v1")
        with mock.patch.dict(os.environ, {"THINK_API_KEY": SECRET}, clear=False):
            self.run_cell("api", 'if MODE == "local-vllm":', ns)
            self.assertEqual(os.environ.get("THINK_API_KEY"), SECRET)

    def test_the_vllm_cell_does_nothing_for_an_endpoint(self) -> None:
        ns = dict(MODE="endpoint", MODEL=C.QWEN_FP8_MODEL, REVISION="", WORK=str(self.workdir()), sys=sys, os=os, subprocess=None, REV="", BASE_URL="https://example.org/v1")
        self.run_cell("api", 'if MODE == "local-vllm":', ns)
        self.assertIsNone(ns["server"])
        self.assertEqual(ns["BASE_URL"], "https://example.org/v1")

    def test_the_last_api_cell_stops_the_server_it_started(self) -> None:
        class Proc:
            def __init__(self):
                self.log = []

            def terminate(self):
                self.log.append("terminate")

            def wait(self, timeout=None):
                self.log.append("wait")

        proc = Proc()
        with contextlib.redirect_stdout(io.StringIO()):
            self.run_cell("api", "server.terminate()", {"server": proc})
            self.run_cell("api", "server.terminate()", {"server": None})
        self.assertEqual(proc.log, ["terminate", "wait"])

    def origin(self):
        """A local repository with two commits, standing in for github.com; returns (path, sha of the first commit)."""
        path = self.workdir() / "origin"
        path.mkdir()

        def git(*args):
            return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True, env=git_env()).stdout.strip()
        git("init", "-q")
        self.assertEqual(os.path.realpath(git("rev-parse", "--show-toplevel")), os.path.realpath(path), "git must work on the throwaway repository")
        git("config", "user.email", "t@example.org")
        git("config", "user.name", "t")
        git("config", "commit.gpgsign", "false")
        git("config", "uploadpack.allowAnySHA1InWant", "true")
        (path / "a.txt").write_text("one", encoding="utf-8")
        git("add", ".")
        git("commit", "-q", "-m", "one")
        first = git("rev-parse", "HEAD")
        (path / "a.txt").write_text("two", encoding="utf-8")
        git("commit", "-q", "-am", "two")
        return path, first

    def run_fetch_cell(self, key, sha, origin, repo):
        """The fetch cell with github.com swapped for the local repository; returns the checkout path."""
        real = subprocess.run

        def to_local(cmd, *a, **kw):
            cmd = [f"file://{origin}" if str(x).startswith("https://github.com/simpletibr/") else x for x in cmd]
            kw = {**kw, "env": git_env()}
            if "capture_output" not in kw:
                kw.setdefault("stderr", subprocess.PIPE)                              # git's complaint about a missing commit is expected here
            return real(cmd, *a, **kw)
        needle = 'subprocess.run(["git", "init", "-q", REPO], check=True)'
        replace = [(m.group(0), f'REPO = {str(repo)!r}') for m in re.finditer(r"(?m)^REPO = .*$", NB.code_cell_with(key, needle))]
        with mock.patch("subprocess.run", side_effect=to_local), contextlib.redirect_stdout(io.StringIO()) as out:
            self.run_cell(key, needle, {"SIMPLICIO_SHA": sha}, replace=replace)
        return out.getvalue()

    @unittest.skipUnless(shutil.which("git"), "git is needed")
    def test_the_fetch_cell_checks_out_exactly_the_pinned_commit(self) -> None:
        origin, first = self.origin()
        for key in ("kaggle", "api"):
            with self.subTest(key):
                repo = self.workdir() / "checkout"
                out = self.run_fetch_cell(key, first, origin, repo)
                head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, env=git_env()).stdout.strip()
                self.assertEqual(head, first)
                self.assertEqual((repo / "a.txt").read_text(encoding="utf-8"), "one", "the later commit must not leak in")
                self.assertIn(first, out)
                self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(repo))
                self.run_fetch_cell(key, first, origin, repo)                       # running the cell again is harmless

    @unittest.skipUnless(shutil.which("git"), "git is needed")
    def test_the_fetch_cell_stops_on_a_commit_that_does_not_exist(self) -> None:
        origin, _ = self.origin()
        repo = self.workdir() / "checkout"
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_fetch_cell("kaggle", "0" * 40, origin, repo)

    @unittest.skipUnless(shutil.which("git"), "git is needed")
    def test_the_git_commands_of_these_tests_ignore_a_repository_named_by_the_environment(self) -> None:
        """A pre-push hook in a linked worktree exports GIT_DIR; the fixtures of these tests once landed in that repository."""
        decoy = self.workdir() / "decoy"
        decoy.mkdir()
        env = git_env()

        def git(*args):
            return subprocess.run(["git", *args], cwd=decoy, check=True, capture_output=True, text=True, env=env).stdout.strip()
        git("init", "-q")
        for key, value in (("user.email", "d@example.org"), ("user.name", "d"), ("commit.gpgsign", "false")):
            git("config", key, value)
        (decoy / "d.txt").write_text("d", encoding="utf-8")
        git("add", ".")
        git("commit", "-q", "-m", "decoy")

        def state():
            return git("rev-parse", "HEAD"), git("status", "--porcelain"), (decoy / ".git" / "config").read_text(encoding="utf-8"), git("for-each-ref")
        before = state()
        leak = {"GIT_DIR": str(decoy / ".git"), "GIT_WORK_TREE": str(decoy), "GIT_INDEX_FILE": str(decoy / ".git" / "index"), "GIT_PREFIX": ""}
        with mock.patch.dict(os.environ, leak):
            origin, first = self.origin()
            for key in ("kaggle", "api"):
                self.run_fetch_cell(key, first, origin, self.workdir() / f"checkout_{key}")
        self.assertEqual(state(), before, "a git command of these tests worked on the repository the environment named")

    # ---------------------------------------------------------------- the Kaggle checks

    def kaggle_checks(self, gpus, free_gib, *, pack_exists=True, found=(), resume=None, results=None):
        work = self.workdir()
        pack = work / "pack.jsonl.gz"
        if pack_exists:
            pack.write_bytes(b"x" * 2000)
        ns = dict(os=os, subprocess=subprocess, PACK=str(pack) if pack_exists else str(work / "missing.gz"), WORK=str(work / "w"),
                  HF_CACHE=str(work / "hf"), RESUME_FROM=str(work / "previous.jsonl") if resume is not None else str(work / "none.jsonl"))
        if resume is not None:
            Path(ns["RESUME_FROM"]).write_text(resume, encoding="utf-8")
        if results is not None:
            (work / "w").mkdir()
            (work / "w" / "results.jsonl").write_text(results, encoding="utf-8")
        done = subprocess.CompletedProcess([], 0, stdout="".join(f"Tesla T4, {m} MiB\n" for m in gpus))
        with mock.patch("subprocess.run", return_value=done), mock.patch("shutil.disk_usage", return_value=type("U", (), {"free": int(free_gib * 2**30)})), \
                mock.patch("glob.glob", return_value=list(found)), mock.patch.dict(os.environ, {}), contextlib.redirect_stdout(io.StringIO()):
            self.run_cell("kaggle", "nvidia-smi", ns)
            ns["hf_home"] = os.environ.get("HF_HOME")
        return ns, work

    def test_kaggle_checks_accept_two_t4_and_set_the_cache(self) -> None:
        ns, work = self.kaggle_checks([15360, 15360], 100)
        self.assertEqual(ns["total_mib"], 30720)
        self.assertEqual(ns["hf_home"], ns["HF_CACHE"])
        self.assertTrue(Path(ns["WORK"]).is_dir() and Path(ns["HF_CACHE"]).is_dir())

    def test_kaggle_checks_refuse_one_gpu_a_full_disk_and_no_pack(self) -> None:
        for args, kw in ((([15360], 100), {}), (([15360, 15360], 20), {}), (([15360, 15360], 100), {"pack_exists": False}), (([], 100), {})):
            with self.subTest(args=args, **kw), self.assertRaises(AssertionError):
                self.kaggle_checks(*args, **kw)

    def test_kaggle_checks_find_the_pack_under_input_when_the_path_is_wrong(self) -> None:
        elsewhere = self.workdir() / "pack.jsonl.gz"
        elsewhere.write_bytes(b"y" * 10)
        ns, _ = self.kaggle_checks([15360, 15360], 100, pack_exists=False, found=[str(elsewhere)])
        self.assertEqual(ns["PACK"], str(elsewhere))

    def test_kaggle_resume_copies_a_previous_session_once_and_never_overwrites(self) -> None:
        ns, _ = self.kaggle_checks([15360, 15360], 100, resume="previous rows\n")
        self.assertEqual((Path(ns["WORK"]) / "results.jsonl").read_text(encoding="utf-8"), "previous rows\n")
        ns, _ = self.kaggle_checks([15360, 15360], 100, resume="OLDER rows\n", results="this session's rows\n")
        self.assertEqual((Path(ns["WORK"]) / "results.jsonl").read_text(encoding="utf-8"), "this session's rows\n")


# ------------------------------------------------------------------------------------------- hosted + the Modal app

class FakeProc:
    """A `vllm serve` process that stays up until it is terminated."""

    def __init__(self, code=None):
        self.returncode, self.log = code, []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.log.append("terminate")
        self.returncode = 0

    def kill(self):
        self.log.append("kill")
        self.returncode = -9

    def wait(self, timeout=None):
        self.log.append("wait")
        return self.returncode


def quiet(fn, *args, **kwargs):
    """(result, what it printed): the thinker and the report print their logs."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(*args, **kwargs), buf.getvalue()


class HostedFlow(FlowBase):
    """`hosted.run_gate` strings serve, think and compare together; the fake endpoint stands for the server vLLM would start."""

    def workdir(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        return d

    def gate(self, srv, work, *, proc=None, started=None, **kw):
        from decide.think import hosted as H
        proc = proc or FakeProc()

        def start(w):
            if started is not None:
                started.append(w)
            return H.Server(proc, srv.url)
        args = dict(pack=str(self.pack), work=str(work), manifest=str(self.manifest), max_tokens=400, concurrency=3,
                    start_server=start, wait=lambda server, timeout: True, log=lambda s: None)
        args.update(kw)
        return quiet(H.run_gate, **args)[0], proc

    def test_serve_think_compare_in_order_and_the_server_is_stopped(self) -> None:
        work, started = self.workdir(), []
        with FakeServer() as srv:
            code, proc = self.gate(srv, work, started=started)
        self.assertEqual(code, 0)
        self.assertEqual(started, [work])
        self.assertEqual(proc.log, ["terminate", "wait"])
        header, recs = T.ResultStore.read(work / "results.jsonl")
        self.assertEqual((header["config"]["model"], header["config"]["revision"]), (C.QWEN_FP8_MODEL, C.QWEN_FP8_REVISION))
        self.assertTrue(header["config"]["endpoint"].startswith("http://127.0.0.1:"))
        self.assertEqual(len(recs), len(self.rows))
        report = json.loads((work / "report" / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["coverage"]["complete"])
        self.assertIn("GT VERDICT:", (work / "report" / "report.md").read_text(encoding="utf-8"))

    def test_the_pinned_revision_follows_the_model_and_none_reads_the_branch(self) -> None:
        for model, revision, expected in ((C.QWEN_MODEL, None, C.QWEN_REVISION), (C.QWEN_FP8_MODEL, "none", None),
                                          (C.QWEN_FP8_MODEL, "abc1234", "abc1234")):
            with self.subTest(model=model, revision=revision):
                work = self.workdir()
                with FakeServer() as srv:
                    self.assertEqual(self.gate(srv, work, model=model, revision=revision, limit=1)[0], 0)
                self.assertEqual(T.ResultStore.read(work / "results.jsonl")[0]["config"]["revision"], expected)

    def test_a_manifest_that_does_not_match_starts_nothing(self) -> None:
        work, started = self.workdir(), []
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        first = next(iter(manifest["benchmarks"].values()))
        first["run_ids"][0] = first["run_ids"][0] + "-changed"
        bad = work / "other-manifest.json"
        bad.write_text(json.dumps(manifest), encoding="utf-8")
        with FakeServer() as srv:
            code, _ = self.gate(srv, work, manifest=str(bad), started=started)
        self.assertEqual(code, 2)
        self.assertEqual(started, [], "no GPU box and no download for a pack that is not the sample")
        self.assertFalse((work / "results.jsonl").exists())

    def test_a_server_that_dies_gives_exit_4_and_the_tail_of_its_log(self) -> None:
        from decide.think import hosted as H
        work, logs = self.workdir(), []
        (work / "vllm.log").write_text("loading weights\nCUDA out of memory\n", encoding="utf-8")
        with FakeServer() as srv:
            dead = H.Server(FakeProc(code=1), srv.url, work / "vllm.log")
            code, proc = self.gate(srv, work, start_server=lambda w: dead, wait=H.wait_until_up, log=logs.append)
        self.assertEqual(code, H.EXIT_NO_SERVER)
        self.assertTrue(any("CUDA out of memory" in line for line in logs), logs)
        self.assertFalse((work / "results.jsonl").exists())

    def test_a_second_run_resumes_and_another_protocol_is_refused_without_a_new_report(self) -> None:
        work = self.workdir()
        with FakeServer() as srv:
            def prompts():
                return sum(1 for q in srv.requests if "MODE[" in q["body"]["messages"][0]["content"])
            self.assertEqual(self.gate(srv, work, limit=3)[0], 0)
            self.assertEqual(prompts(), 3)
            self.assertEqual(self.gate(srv, work)[0], 0)
            self.assertEqual(prompts(), len(self.rows), "a row was asked twice or not at all")
            stamp = (work / "report" / "report.json").stat().st_mtime_ns
            code, proc = self.gate(srv, work, thinker_extra=["--thinking", "off"])
        self.assertEqual(code, 2)
        self.assertEqual((work / "report" / "report.json").stat().st_mtime_ns, stamp, "a refused run must not rewrite the report")
        self.assertEqual(proc.log, ["terminate", "wait"])

    def test_consecutive_errors_exit_3_and_the_report_is_still_written(self) -> None:
        work = self.workdir()

        def down(body, count):
            return {"status": 503, "text": "overloaded"} if "MODE[" in body["messages"][0]["content"] else scripted(body, count)
        with FakeServer(down) as srv:
            code, _ = self.gate(srv, work, concurrency=1, thinker_extra=["--retries", "0", "--max-consecutive-errors", "2"])
        self.assertEqual(code, 3)
        self.assertIn("INCONCLUSIVE", (work / "report" / "report.md").read_text(encoding="utf-8"))

    def test_the_server_is_stopped_even_when_the_thinker_raises(self) -> None:
        from unittest import mock
        from decide.think import hosted as H
        work = self.workdir()
        proc = FakeProc()
        with FakeServer() as srv, mock.patch.object(H.thinker, "main", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.gate(srv, work, proc=proc)
        self.assertEqual(proc.log, ["terminate", "wait"])

    def test_the_cli_runs_the_whole_gate_with_a_server_that_is_already_up(self) -> None:
        """`main` starts vLLM itself; here the start is replaced by the endpoint of the fake server."""
        from unittest import mock
        from decide.think import hosted as H
        work = self.workdir()
        with FakeServer() as srv, mock.patch.object(H, "start_vllm", side_effect=lambda w, *a, **k: H.Server(FakeProc(), srv.url)), \
                mock.patch.object(H, "wait_until_up", return_value=True):
            code, text = quiet(H.main, ["--pack", str(self.pack), "--work", str(work), "--manifest", str(self.manifest), "--max-tokens", "400",
                                        "--concurrency", "3", "--limit", "4"])
        self.assertEqual(code, 0, text)
        self.assertEqual(len(T.ResultStore.read(work / "results.jsonl")[1]), 4)
        self.assertIn("GT VERDICT:", text)


def fake_modal():
    """A stand-in for the `modal` client that records how the app is declared (the real client is not installed in CI)."""
    import types
    mod, rec = types.ModuleType("modal"), {"app": [], "functions": [], "entrypoints": [], "volumes": []}

    class Image:
        def __init__(self, calls=()):
            self.calls = tuple(calls)

        @classmethod
        def debian_slim(cls, python_version=None):
            return cls([("debian_slim", python_version)])

        def pip_install(self, *packages, **kw):
            return Image(self.calls + (("pip_install", packages),))

        def env(self, values):
            return Image(self.calls + (("env", dict(values)),))

        def add_local_dir(self, source, remote, **kw):
            return Image(self.calls + (("add_local_dir", Path(source), remote, tuple(kw.get("ignore", ()))),))

    class Volume:
        def __init__(self, name):
            self.name, self.commits = name, 0

        @classmethod
        def from_name(cls, name, create_if_missing=False):
            rec["volumes"].append((name, create_if_missing))
            return cls(name)

        def commit(self):
            self.commits += 1

    class Retries:
        def __init__(self, **kw):
            self.kw = kw

    class Called:
        def __init__(self, fn):
            self.fn = fn

        def remote(self, *a, **k):
            return self.fn(*a, **k)

        def __call__(self, *a, **k):
            return self.fn(*a, **k)

    class App:
        def __init__(self, name, **kw):
            rec["app"].append((name, kw))

        def function(self, **kw):
            def deco(fn):
                rec["functions"].append((fn.__name__, kw))
                return Called(fn)
            return deco

        def local_entrypoint(self, **kw):
            def deco(fn):
                rec["entrypoints"].append(fn.__name__)
                return fn
            return deco
    for cls in (Image, Volume, Retries, App):
        setattr(mod, cls.__name__, cls)
    return mod, rec


class ModalAppFlow(FlowBase):
    """modal_app.py against a stand-in client: how it declares the function, and what the function and the entrypoint do.
    That it builds, finds a GPU and runs on a Modal account is UNVERIFIED (README)."""

    def load(self, gpu=None):
        from unittest import mock
        stub, rec = fake_modal()
        env = {"SIMPLICIO_THINK_GPU": gpu} if gpu else {}
        spec = importlib.util.spec_from_file_location("decide.think.modal_app", ROOT / "decide" / "think" / "modal_app.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"modal": stub}), mock.patch.dict(os.environ, env):
            if not gpu:
                os.environ.pop("SIMPLICIO_THINK_GPU", None)
            spec.loader.exec_module(module)
        return module, rec

    def workdir(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        return d

    def test_the_function_is_declared_with_the_gpu_volumes_and_pinned_image(self) -> None:
        module, rec = self.load()
        self.assertEqual(rec["app"], [("simplicio-think-gt", {"include_source": False})])
        (name, kw), = rec["functions"]
        self.assertEqual(name, "gate")
        self.assertEqual((kw["gpu"], kw["timeout"], kw["max_containers"]), ("H100", 12 * 3600, 1))
        self.assertEqual(kw["retries"].kw["max_retries"], 2)
        self.assertEqual({path: vol.name for path, vol in kw["volumes"].items()},
                         {"/root/.cache/huggingface": "simplicio-think-hf-cache", "/state": "simplicio-think-gt-state"})
        self.assertEqual(sorted(rec["volumes"]), [("simplicio-think-gt-state", True), ("simplicio-think-hf-cache", True)])
        calls = dict((c[0], c[1:]) for c in kw["image"].calls)
        self.assertEqual(calls["debian_slim"], ("3.12",))
        self.assertEqual(calls["pip_install"], (("vllm==0.31.0",),))
        self.assertEqual(calls["env"][0]["PYTHONPATH"], "/root")
        source, remote, ignore = calls["add_local_dir"]
        self.assertEqual((source, remote), (ROOT / "decide" / "think", "/root/decide/think"))
        self.assertTrue((source / "gt-sample-manifest.json").is_file(), "the container needs the manifest")
        self.assertIn("**/__pycache__", ignore)
        self.assertEqual(rec["entrypoints"], ["main"])

    def test_the_image_leaves_out_what_the_folder_keeps_out_of_git(self) -> None:
        """The pack and the results hold benchmark text; `decide/think/.gitignore` names them and the image must not carry them either."""
        module, _ = self.load()
        private = [line.strip() for line in (ROOT / "decide" / "think" / ".gitignore").read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.startswith("#")]
        self.assertGreaterEqual(len(private), 5)
        for line in private:
            self.assertIn("**/" + line.rstrip("/"), module.IGNORE, f"{line} is private (.gitignore) but would be copied into the image")
        self.assertNotIn("**/*.json", module.IGNORE, "the container needs gt-sample-manifest.json")

    def test_the_gpu_comes_from_the_environment(self) -> None:
        _, rec = self.load(gpu="A100-80GB")
        self.assertEqual(rec["functions"][0][1]["gpu"], "A100-80GB")

    def test_the_script_mode_is_refused_with_the_command_to_use(self) -> None:
        from unittest import mock
        stub, _ = fake_modal()
        spec = importlib.util.spec_from_file_location("modal_app_as_script", ROOT / "decide" / "think" / "modal_app.py")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = ""
        with mock.patch.dict(sys.modules, {"modal": stub}):
            with self.assertRaises(RuntimeError) as cm:
                spec.loader.exec_module(module)
        self.assertIn("modal run -m decide.think.modal_app", str(cm.exception))

    def run_function(self, module, tmp, *, result=0, boom=False, **kw):
        from unittest import mock
        from decide.think import hosted as H
        calls = {}

        def fake_run_gate(**kwargs):
            calls.update(kwargs)
            calls["pack_bytes"] = Path(kwargs["pack"]).read_bytes()
            report = Path(kwargs["work"]) / "report"
            report.mkdir(parents=True, exist_ok=True)
            (report / "report.md").write_text("# Gate GT\n\nGT VERDICT: FAIL (delta_below_min) | thinker 40.0%\n\n- other line\n", encoding="utf-8")
            (report / "report.json").write_text('{"x": 1}', encoding="utf-8")
            if boom:
                raise RuntimeError("boom")
            return result
        with mock.patch.object(module, "STATE", str(tmp)), mock.patch.object(module, "PACK_PATH", str(tmp / "pack.jsonl.gz")), \
                mock.patch.object(H, "run_gate", side_effect=fake_run_gate):
            out = module.gate(b"PACKBYTES", **kw)
        return out, calls

    def test_the_function_runs_the_gate_in_the_volume_and_returns_the_small_things(self) -> None:
        module, _ = self.load()
        tmp = self.workdir()
        out, calls = self.run_function(module, tmp, run="mine", limit=3, max_tokens=8192, concurrency=4)
        self.assertEqual(calls["pack_bytes"], b"PACKBYTES")
        self.assertEqual(calls["work"], str(tmp / "mine"))
        self.assertEqual((calls["limit"], calls["max_tokens"], calls["concurrency"]), (3, 8192, 4))
        self.assertEqual(calls["time_budget_s"], float(12 * 3600 - 3600))
        self.assertNotIn("manifest", calls, "the committed manifest of the container is the default")
        self.assertEqual(out["exit"], 0)
        self.assertEqual(out["verdict"], "GT VERDICT: FAIL (delta_below_min) | thinker 40.0%")
        self.assertEqual(out["report_json"], '{"x": 1}')
        self.assertEqual(out["results_in_volume"], "mine/results.jsonl")
        self.assertNotIn("results", out, "the results (questions and reasoning) stay in the Volume")
        self.assertIn("# Gate GT", out["report_md"])

    def test_both_volumes_are_committed_at_the_end_and_when_the_gate_raises(self) -> None:
        module, _ = self.load()
        before = (module.state_volume.commits, module.cache_volume.commits)
        self.run_function(module, self.workdir())
        self.assertGreater(module.state_volume.commits, before[0])
        self.assertGreater(module.cache_volume.commits, before[1])
        after = (module.state_volume.commits, module.cache_volume.commits)
        with self.assertRaises(RuntimeError):
            self.run_function(module, self.workdir(), boom=True)
        self.assertGreater(module.state_volume.commits, after[0])
        self.assertGreater(module.cache_volume.commits, after[1])

    def test_a_failed_commit_does_not_stop_the_run(self) -> None:
        module, _ = self.load()

        def refuse():
            raise OSError("volume busy")
        module.state_volume.commit = refuse
        out, _ = quiet(self.run_function, module, self.workdir())[0]
        self.assertEqual(out["exit"], 0)

    def entrypoint(self, module, pack, out_dir, *, remote=None, manifest=None, **kw):
        from unittest import mock
        from decide.think import hosted as H
        canned = {"exit": 0, "report_md": "# r\n\nGT VERDICT: PASS | x\n", "report_json": "{}", "verdict": "GT VERDICT: PASS | x",
                  "results_in_volume": "gt/results.jsonl"}
        seen = []
        module.gate = type("G", (), {"remote": staticmethod(lambda *a: (seen.append(a), remote or canned)[1])})
        with mock.patch.object(H, "DEFAULT_MANIFEST", Path(manifest or self.manifest)):
            try:
                quiet(module.main, pack=str(pack), out=str(out_dir), **kw)
            except SystemExit as e:                                   # the entrypoint always ends with the exit code of the gate
                return e.code, seen
        raise AssertionError("the entrypoint returned instead of exiting")

    def test_the_entrypoint_checks_the_pack_before_renting_anything(self) -> None:
        module, _ = self.load()
        work = self.workdir()
        bad = work / "bad.jsonl.gz"
        bad.write_bytes(b"not a pack")
        code, seen = self.entrypoint(module, bad, work / "out")
        self.assertIn("nothing was rented", str(code))
        code, seen2 = self.entrypoint(module, work / "missing.jsonl.gz", work / "out")
        self.assertIn("nothing was rented", str(code))
        self.assertEqual(seen + seen2, [], "the GPU function must not be called for a pack that is not the sample")

    def test_the_entrypoint_sends_the_pack_writes_the_report_and_exits_with_the_gate_code(self) -> None:
        module, _ = self.load()
        work = self.workdir()
        code, seen = self.entrypoint(module, self.pack, work / "out", limit=3, max_tokens=4096, concurrency=2, run="r1")
        self.assertEqual(code, 0)
        self.assertEqual(seen, [(self.pack.read_bytes(), "r1", 3, 4096, 2)])
        self.assertIn("GT VERDICT: PASS", (work / "out" / "report.md").read_text(encoding="utf-8"))
        self.assertEqual((work / "out" / "report.json").read_text(encoding="utf-8"), "{}")
        failing = {"exit": 3, "report_md": "", "report_json": "", "verdict": "", "results_in_volume": "gt/results.jsonl"}
        code, _ = self.entrypoint(module, self.pack, work / "out2", remote=failing)
        self.assertEqual(code, 3)
        self.assertFalse((work / "out2" / "report.md").exists())

    def test_the_entrypoint_says_how_to_fetch_the_private_results(self) -> None:
        from unittest import mock
        from decide.think import hosted as H
        module, _ = self.load()
        module.gate = type("G", (), {"remote": staticmethod(lambda *a: {"exit": 0, "report_md": "", "report_json": "", "verdict": "",
                                                                       "results_in_volume": "gt/results.jsonl"})})
        out = io.StringIO()
        with mock.patch.object(H, "DEFAULT_MANIFEST", Path(self.manifest)), contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                module.main(pack=str(self.pack), out=str(self.workdir() / "o"))
        self.assertIn("modal volume get simplicio-think-gt-state gt/results.jsonl", out.getvalue())
        self.assertIn("no report", out.getvalue())

    @unittest.skipUnless(importlib.util.find_spec("modal"), "modal is not installed: the app was read against modal 1.6.2 only")
    def test_the_installed_client_accepts_the_declarations(self) -> None:
        import warnings
        spec = importlib.util.spec_from_file_location("decide.think.modal_app_real", ROOT / "decide" / "think" / "modal_app.py")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "decide.think"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec.loader.exec_module(module)
            self.assertEqual(list(module.app.registered_functions), ["gate"])
            self.assertEqual(list(module.app.registered_entrypoints), ["main"])


if __name__ == "__main__":
    unittest.main()
