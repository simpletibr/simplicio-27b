"""Contract tests for decide/sd/evaluate.py (issue #65, F0 kit core): requests in, resumable per-row results out.

The engine is a test double that follows the engine contract (`batch(list) -> [(response, raw) | exception]`); no model,
no torch.  The tests that need the scoring kit or pyarrow skip cleanly when those are missing.
"""
from __future__ import annotations

import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

from decide_support import HermeticTestCase, needs_kit, needs_pyarrow

from decide.sd import evaluate as ev
from decide.sd import sample
from decide.sd.kitapi import NativeAbstention, Unsupported
from decide.sd.state import MemoryStore, State


def kit_row(i, *, catalog_id=28, group=None, gold="A", options=("A", "B"), proxy=100, qtype="choice", state=None):
    criteria = {k: f"option {k}" for k in options}
    question = {"type": qtype, "instructions": "pick", **({"criteria": criteria} if qtype == "choice" else {})}
    expected = {"q1": gold} if qtype == "choice" else {"q1": bool(gold)}
    return {
        "_evaluation": {"run_id": f"{catalog_id}:T:T:{i}", "catalog_id": catalog_id, "dataset": "D", "group_id": group or f"g{i}",
                        "track": "T", "proxy_tokens": proxy},
        "state": state if state is not None else f"state-{i}", "questions": {"q1": question}, "expected": expected,
        "family": "f", "id": f"id{i}", "metadata": {}, "split": "test",
    }


def write_jsonl(path, rows):
    text = "".join(json.dumps(r) + "\n" for r in rows)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        Path(path).write_text(text, encoding="utf-8")
    return path


def softmax(values):
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class StubEngine:
    """Favours the first option.  `modes` maps a request's state text to what the double does with it."""

    name = "stub"
    provenance = {"kind": "test-double"}

    def __init__(self, modes=None, has_batch=True):
        self.modes = dict(modes or {})
        self.seen = []                      # state text of every request the engine was asked for, in order
        self.batches = []
        self.closed = False
        if not has_batch:
            self.batch = None               # type: ignore[assignment]

    def _answer(self, request):
        mode = self.modes.get(request["state"])
        if mode == "unsupported":
            return Unsupported("too long for this double")
        if mode == "error":
            return RuntimeError("boom")
        if mode == "abstain":
            return NativeAbstention({"why": "test"})
        answers, logits = {}, {}
        for key, question in request["questions"].items():
            if question["type"] == "choice":
                keys = list(question["criteria"])
                values = [float(len(keys) - i) for i in range(len(keys))]
                probs = softmax(values)
                if mode == "bad":
                    probs = [p * 0.5 for p in probs]
                answers[key] = {"type": "choice", "choice": keys[0], "probabilities": dict(zip(keys, probs))}
            else:
                values = [0.0, 1.0]
                answers[key] = {"type": "noul", "noul": softmax(values)[1]}
            logits[key] = [values]
        response = {"model": "stub", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0}}
        raw = {"logits": logits, "temperature": 1.0, "applied_temperatures": {k: 1.0 for k in logits}, "views": 1, "tokens": 10}
        return response, raw

    def batch(self, requests):
        self.batches.append(len(requests))
        out = []
        for request in requests:
            self.seen.append(request["state"])
            out.append(self._answer(request))
        return out

    def __call__(self, state, questions, **inputs):
        self.seen.append(state)
        got = self._answer({"state": state, "questions": questions})
        if isinstance(got, BaseException):
            raise got
        return got

    def synchronize(self):
        pass

    def runtime(self):
        return {"device": "none"}

    def close(self):
        self.closed = True


class Clock:
    def __init__(self, step=0.0):
        self.t = 1000.0
        self.step = step

    def __call__(self):
        self.t += self.step
        return self.t


class RequestsContract(HermeticTestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_kit_rows_give_run_id_inputs_meta_and_gold(self):
        rows = [kit_row(0), kit_row(1, gold="B")]
        for name in ("rows.jsonl", "rows.jsonl.gz"):
            items = list(ev.iter_items(write_jsonl(self.dir / name, rows)))
            self.assertEqual([i.run_id for i in items], ["28:T:T:0", "28:T:T:1"])
            self.assertEqual(items[0].inputs, {"state": "state-0", "questions": rows[0]["questions"]})
            self.assertEqual(items[1].gold, {"q1": "B"})
            self.assertEqual(items[0].meta["catalog_id"], 28)
            self.assertEqual(items[0].proxy_tokens, 100)

    def test_plain_request_rows_with_a_json_string_or_a_dict(self):
        request = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"x": "1", "y": "2"}}}}
        write_jsonl(self.dir / "plain.jsonl", [
            {"example_id": "e1", "request": request, "source": "wiki", "split": "train"},
            {"run_id": "e2", "request": json.dumps(request), "gold": json.dumps({"q": "y"})},
        ])
        a, b = list(ev.iter_items(self.dir / "plain.jsonl"))
        self.assertEqual((a.run_id, a.inputs["state"], a.meta["source"]), ("e1", "s", "wiki"))
        self.assertEqual((b.run_id, b.gold), ("e2", {"q": "y"}))

    def test_blank_lines_are_ignored_and_a_bad_line_names_file_and_line(self):
        path = self.dir / "r.jsonl"
        path.write_text(json.dumps(kit_row(0)) + "\n\n" + "{not json\n", encoding="utf-8")
        with self.assertRaisesRegex(ev.EvaluateError, r"r\.jsonl.*line 3"):
            list(ev.iter_items(path))

    def test_a_row_without_questions_is_refused_by_name(self):
        row = kit_row(0)
        del row["questions"]
        with self.assertRaisesRegex(ev.EvaluateError, "28:T:T:0.*questions"):
            list(ev.iter_items(write_jsonl(self.dir / "r.jsonl", [row])))
        with self.assertRaisesRegex(ev.EvaluateError, "no run_id"):
            list(ev.iter_items(write_jsonl(self.dir / "s.jsonl", [{"request": {"state": "s", "questions": {"q": {}}}}])))

    def test_duplicate_run_ids_are_refused_before_any_work(self):
        engine = StubEngine()
        path = write_jsonl(self.dir / "r.jsonl", [kit_row(0), kit_row(0)])
        with self.assertRaisesRegex(ev.EvaluateError, "duplicate run_id"):
            ev.run_requests(engine, path, self.dir / "out")
        self.assertEqual(engine.seen, [])

    def test_unknown_suffix_is_refused(self):
        (self.dir / "r.csv").write_text("a,b\n", encoding="utf-8")
        with self.assertRaisesRegex(ev.EvaluateError, "suffix"):
            list(ev.iter_items(self.dir / "r.csv"))

    @needs_pyarrow
    def test_parquet_requests(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        request = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}
        table = pa.table({"example_id": ["a", "b"], "request": [json.dumps(request)] * 2, "gold": [json.dumps({"q": True}), None],
                          "source": ["s1", "s2"]})
        pq.write_table(table, self.dir / "r.parquet")
        a, b = list(ev.iter_items(self.dir / "r.parquet"))
        self.assertEqual((a.run_id, a.gold, a.meta["source"]), ("a", {"q": True}, "s1"))
        self.assertIsNone(b.gold)
        self.assertEqual(a.inputs["questions"]["q"]["type"], "noul")


class RunRequestsContract(HermeticTestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.rows = [kit_row(i, gold="A" if i % 2 else "B") for i in range(7)]
        self.req = write_jsonl(self.dir / "requests.jsonl", self.rows)
        self.out = self.dir / "run"

    def results(self):
        path = self.out / ev.RESULTS_NAME
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def raws(self):
        path = self.out / ev.RAW_NAME
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def go(self, engine=None, **kw):
        engine = engine or StubEngine()
        kw.setdefault("log", lambda text: None)
        return engine, ev.run_requests(engine, self.req, self.out, **kw)

    def test_writes_kit_compatible_result_rows_and_the_raw_logits_beside_them(self):
        engine, out = self.go(chunk=3)
        self.assertEqual((out["rows"], out["total"], out["status"]), (7, 7, "complete"))
        self.assertEqual(out["counts"], {"ok": 7})
        rows = self.results()
        self.assertEqual([r["run_id"] for r in rows], [r["_evaluation"]["run_id"] for r in self.rows])
        first = rows[0]
        for key in ("run_id", "catalog_id", "group_id", "track", "proxy_tokens", "status", "response", "engine", "started_utc",
                    "completed_utc", "total_wall_ms", "model_request_wall_ms"):
            self.assertIn(key, first)
        self.assertEqual(first["status"], "ok")
        self.assertNotIn("payload", first)                       # compact: the request is not copied into the result
        raws = self.raws()
        self.assertEqual([r["run_id"] for r in raws], [r["run_id"] for r in rows])
        self.assertEqual(raws[0]["logits"]["q1"], [[2.0, 1.0]])
        self.assertEqual(raws[0]["views"], 1)
        self.assertEqual(engine.batches, [3, 3, 1])
        self.assertTrue((self.out / ev.STATUS_NAME).exists() and (self.out / ev.ENVIRONMENT_NAME).exists())
        env = json.loads((self.out / ev.ENVIRONMENT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(env["engine"], "stub")
        self.assertEqual(env["model_source"], {"kind": "test-double"})
        self.assertEqual(out["tokens"], 70)

    def test_a_second_call_does_not_touch_the_engine(self):
        self.go()
        engine, out = self.go()
        self.assertEqual(engine.seen, [])
        self.assertEqual((out["rows"], out["total"], out["skipped"], out["status"]), (0, 7, 7, "complete"))
        self.assertEqual(len(self.results()), 7)

    def test_a_row_cut_in_half_by_a_crash_is_dropped_and_redone(self):
        self.go()
        path = self.out / ev.RESULTS_NAME
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[:5]) + lines[5][:40], encoding="utf-8")
        engine, out = self.go()
        self.assertEqual(engine.seen, ["state-5", "state-6"])
        self.assertEqual([r["run_id"] for r in self.results()], [r["_evaluation"]["run_id"] for r in self.rows])
        self.assertEqual(out["total"], 7)

    def test_error_rows_are_retried_and_unsupported_rows_are_final(self):
        engine, out = self.go(StubEngine({"state-2": "error", "state-4": "unsupported"}))
        self.assertEqual(out["counts"], {"ok": 5, "error": 1, "unsupported": 1})
        by = {r["run_id"]: r for r in self.results()}
        self.assertEqual(by["28:T:T:2"]["status"], "error")
        self.assertIn("boom", by["28:T:T:2"]["error"])
        self.assertEqual(by["28:T:T:2"]["exception"], "RuntimeError")
        self.assertEqual(by["28:T:T:4"]["status"], "unsupported")
        self.assertIn("too long", by["28:T:T:4"]["error"])
        engine2, out2 = self.go(StubEngine())
        self.assertEqual(engine2.seen, ["state-2"])              # only the error row runs again
        final = {}
        for row in self.results():                               # the last row of a run_id wins
            final[row["run_id"]] = row["status"]
        self.assertEqual(final["28:T:T:2"], "ok")
        self.assertEqual(final["28:T:T:4"], "unsupported")
        self.assertEqual(out2["total"], 7)

    def test_abstentions_and_invalid_responses_are_recorded_not_raised(self):
        engine, out = self.go(StubEngine({"state-1": "abstain", "state-3": "bad"}))
        by = {r["run_id"]: r for r in self.results()}
        self.assertEqual(by["28:T:T:1"]["status"], "abstained")
        self.assertEqual(by["28:T:T:3"]["status"], "error")
        self.assertIn("probability", by["28:T:T:3"]["error"].lower())
        self.assertEqual(out["counts"]["ok"], 5)

    def test_budget_stops_between_chunks_and_the_next_call_finishes_the_job(self):
        class Budget:
            def __init__(self, allowed):
                self.allowed = allowed
                self.asked = []

            def ok(self, cost=0.0):
                self.asked.append(cost)
                self.allowed -= 1
                return self.allowed >= 0

        engine, out = self.go(chunk=2, budget=Budget(2), clock=Clock(step=1.0))
        self.assertEqual((out["status"], out["rows"], out["total"]), ("budget", 4, 4))
        engine2, out2 = self.go(chunk=2)
        self.assertEqual((out2["status"], out2["rows"], out2["total"]), ("complete", 3, 7))
        self.assertEqual(engine.seen + engine2.seen, [f"state-{i}" for i in range(7)])

    def test_the_budget_is_asked_for_the_cost_of_the_next_chunk_once_throughput_is_known(self):
        asked = []

        class Budget:
            def ok(self, cost=0.0):
                asked.append(cost)
                return True

        self.go(chunk=2, budget=Budget(), clock=Clock(step=1.0))
        self.assertEqual(asked[0], 0.0)                          # nothing measured yet
        self.assertTrue(all(c > 0 for c in asked[1:]))           # afterwards: tokens of the chunk / measured tokens per second

    def test_limit_stops_after_that_many_new_rows(self):
        engine, out = self.go(limit=3, chunk=2)
        self.assertEqual((out["status"], out["rows"], out["total"]), ("limit", 3, 3))
        self.assertEqual(len(engine.seen), 3)

    def test_chunk_size_does_not_change_the_rows(self):
        self.go(chunk=1)
        a = [(r["run_id"], r["status"], r["response"]) for r in self.results()]
        other = ev.run_requests(StubEngine(), self.req, self.dir / "run2", chunk=16, log=lambda t: None)
        rows2 = [json.loads(line) for line in (self.dir / "run2" / ev.RESULTS_NAME).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(a, [(r["run_id"], r["status"], r["response"]) for r in rows2])
        self.assertEqual(other["total"], 7)

    def test_five_failures_without_a_success_stop_the_run(self):
        engine = StubEngine({f"state-{i}": "error" for i in range(7)})
        with self.assertRaisesRegex(ev.EvaluateError, "five failed"):
            ev.run_requests(engine, self.req, self.out, chunk=1, log=lambda t: None)
        self.assertEqual(len(engine.seen), 5)
        self.assertEqual(len(self.results()), 5)                 # what was done is on disk

    def test_a_request_that_makes_the_batch_call_itself_fail_is_isolated(self):
        class Flaky(StubEngine):
            def batch(self, requests):
                if any(r["state"] == "state-1" for r in requests) and len(requests) > 1:
                    raise ValueError("whole chunk rejected")
                return super().batch(requests)

        engine, out = self.go(Flaky({"state-1": "error"}), chunk=4)
        self.assertEqual(out["counts"], {"ok": 6, "error": 1})

    def test_device_errors_stop_before_the_context_is_reused_and_leave_the_chunk_to_resume(self):
        class Dead(StubEngine):
            def batch(self, requests):
                raise RuntimeError("CUDA error: device-side assert triggered")

        with self.assertRaisesRegex(ev.EvaluateError, "device"):
            ev.run_requests(Dead(), self.req, self.out, chunk=3, log=lambda t: None)
        self.assertEqual(self.results(), [])
        engine, out = self.go()
        self.assertEqual(out["total"], 7)

    def test_an_engine_without_batch_is_called_one_request_at_a_time(self):
        engine, out = self.go(StubEngine({"state-2": "unsupported"}, has_batch=False))
        self.assertEqual(out["counts"], {"ok": 6, "unsupported": 1})
        self.assertEqual(engine.seen, [f"state-{i}" for i in range(7)])

    def test_resume_false_moves_old_results_aside(self):
        self.go()
        engine, out = self.go(resume=False)
        self.assertEqual(len(engine.seen), 7)
        self.assertEqual(out["total"], 7)
        aside = [p.name for p in self.out.iterdir() if ".old" in p.name]
        self.assertTrue(any(name.startswith(ev.RESULTS_NAME) for name in aside), aside)

    def test_the_engine_is_not_closed_by_the_runner(self):
        engine, _ = self.go()
        self.assertFalse(engine.closed)

    def test_out_path_may_name_the_parquet_file(self):
        engine = StubEngine()
        ev.run_requests(engine, self.req, self.out / "table.parquet", log=lambda t: None)
        self.assertTrue((self.out / ev.RESULTS_NAME).exists())

    def test_state_mirrors_the_files_and_a_new_session_resumes_from_the_store(self):
        store = MemoryStore()
        work1 = self.dir / "s1"
        st1 = State.open(work1, store)
        out1 = work1 / "runs" / "stub"
        ev.run_requests(StubEngine(), self.req, out1, chunk=3, state=st1, log=lambda t: None, limit=4)
        self.assertIn("runs/stub/results.jsonl", store.list("runs"))
        self.assertIn("runs/stub/raw.jsonl", store.list("runs"))
        work2 = self.dir / "s2"                                   # a new session: empty disk, same store
        st2 = State.open(work2, store)
        engine = StubEngine()
        out = ev.run_requests(engine, self.req, work2 / "runs" / "stub", state=st2, log=lambda t: None)
        self.assertEqual(engine.seen, [f"state-{i}" for i in range(4, 7)])
        self.assertEqual(out["total"], 7)

    @needs_pyarrow
    def test_the_probability_table_has_one_row_per_question_with_gold(self):
        import pyarrow.parquet as pq

        engine, out = self.go(StubEngine({"state-3": "unsupported"}))
        table = pq.read_table(self.out / ev.TABLE_NAME).to_pylist()
        self.assertEqual(len(table), 7)
        ok = [t for t in table if t["status"] == "ok"]
        first = ok[0]
        self.assertEqual(first["question_id"], "q1")
        self.assertEqual(first["qtype"], "choice")
        self.assertEqual(first["option_keys"], ["A", "B"])
        self.assertAlmostEqual(sum(first["probs"]), 1.0)
        self.assertEqual(first["logits"], [[2.0, 1.0]])
        self.assertEqual(first["gold"], "B")
        self.assertEqual(first["catalog_id"], 28)
        bad = [t for t in table if t["status"] == "unsupported"][0]
        self.assertEqual(bad["probs"], [])
        self.assertTrue(out["files"]["table"].endswith(ev.TABLE_NAME))

    def test_noul_answers_become_a_two_option_row(self):
        write_jsonl(self.dir / "n.jsonl", [kit_row(0, qtype="noul", gold=True)])
        ev.run_requests(StubEngine(), self.dir / "n.jsonl", self.dir / "nout", log=lambda t: None)
        rows = list(ev.table_rows(self.dir / "nout", {"28:T:T:0": {"q1": True}}))
        self.assertEqual(rows[0]["option_keys"], ["false", "true"])
        self.assertAlmostEqual(rows[0]["probs"][1], softmax([0.0, 1.0])[1])
        self.assertEqual(rows[0]["gold"], "true")


class ThroughputContract(HermeticTestCase):
    def setUp(self):
        super().setUp()
        self.items = [ev.Item(f"r{i}", {"state": f"state-{i}", "questions": kit_row(i)["questions"]},
                              {"run_id": f"r{i}", "proxy_tokens": 100 * (i + 1)}) for i in range(9)]

    def test_measure_reports_proxy_tokens_per_second_after_a_warmup_request(self):
        engine = StubEngine()
        out = ev.measure_throughput(engine, self.items, n=3, clock=Clock(step=0.5))
        self.assertEqual(out["n"], 3)
        self.assertGreater(out["tokens_per_second"], 0)
        self.assertEqual(len(engine.seen), 1 + 3)                # one warm-up request, then the three measured ones
        picked = out["run_ids"]
        self.assertEqual(len(set(picked)), 3)
        self.assertEqual(out["proxy_tokens"], sum(i.proxy_tokens for i in self.items if i.run_id in picked))

    def test_measure_picks_requests_spread_over_the_length_range(self):
        out = ev.measure_throughput(StubEngine(), self.items, n=3, clock=Clock(step=1.0), warmup=0)
        sizes = sorted(i.proxy_tokens for i in self.items if i.run_id in out["run_ids"])
        self.assertEqual(sizes[0], 100)
        self.assertEqual(sizes[-1], 900)

    def test_measure_of_nothing_is_an_error(self):
        with self.assertRaisesRegex(ev.EvaluateError, "no requests"):
            ev.measure_throughput(StubEngine(), [], n=3)

    def test_fit_to_budget_shrinks_the_sample_and_says_so(self):
        rows = [kit_row(i, catalog_id=28 + (i % 2), group=f"g{i}", proxy=100) for i in range(20)]
        out = ev.fit_to_budget(rows, tokens_per_second=10.0, seconds=60.0, seed=1, minimum=1, safety=0.5)
        self.assertEqual(out["token_budget"], 300.0)
        self.assertTrue(out["fits"])
        self.assertEqual((out["per_benchmark"], out["tokens"], out["current"]), (1, 200, 10))
        self.assertIn("groups per benchmark", out["message"])

    def test_fit_to_budget_keeps_everything_when_it_fits(self):
        rows = [kit_row(i, catalog_id=28, group=f"g{i}", proxy=10) for i in range(5)]
        out = ev.fit_to_budget(rows, tokens_per_second=100.0, seconds=3600.0, seed=1, minimum=1)
        self.assertEqual(out["per_benchmark"], 5)
        self.assertTrue(out["fits"])

    def test_fit_to_budget_warns_when_even_the_floor_is_too_big(self):
        rows = [kit_row(i, catalog_id=28 + i, group=f"g{i}", proxy=1000) for i in range(6)]
        out = ev.fit_to_budget(rows, tokens_per_second=1.0, seconds=10.0, seed=1, minimum=2)
        self.assertFalse(out["fits"])
        self.assertEqual(out["per_benchmark"], 2)
        self.assertIn("does not fit", out["message"])

    def test_a_non_positive_throughput_is_refused(self):
        with self.assertRaisesRegex(ev.EvaluateError, "throughput"):
            ev.fit_to_budget([kit_row(0)], tokens_per_second=0.0, seconds=10.0, seed=1)


@needs_kit
class ScoreContract(HermeticTestCase):
    """The kit's own scorer through a sample (no metric is written here)."""

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.rows = []
        for i in range(12):
            self.rows.append(kit_row(i, catalog_id=28, gold="A" if i % 2 else "B"))
        for i in range(12):
            self.rows.append(kit_row(100 + i, catalog_id=29, gold="C", options=("A", "B", "C", "D")))

    def results_file(self, name, skill):
        rows = []
        for r in self.rows:
            keys = list(r["questions"]["q1"]["criteria"])
            gold = r["expected"]["q1"]
            right = skill(r)
            pick = gold if right else next(k for k in keys if k != gold)
            probs = {k: (0.7 if k == pick else 0.3 / (len(keys) - 1)) for k in keys}
            rows.append({**r["_evaluation"], "status": "ok", "engine": name, "total_wall_ms": 5.0, "model_request_wall_ms": 5.0,
                         "response": {"model": name, "answers": {"q1": {"type": "choice", "choice": pick, "probabilities": probs}}}})
        return write_jsonl(self.dir / f"{name}.jsonl", rows)

    def test_a_perfect_run_beats_a_bad_one_and_only_present_benchmarks_count(self):
        good = ev.score_results(self.results_file("good", lambda r: True), self.rows, edition="0.3")
        bad = ev.score_results(self.results_file("bad", lambda r: False), self.rows, edition="0.3")
        self.assertGreater(good["public"], 95.0)                 # both present areas perfect: weights renormalised over them
        self.assertLess(bad["public"], 5.0)
        self.assertEqual(sorted(good["per_benchmark"]), ["28", "29"])
        self.assertEqual(good["benchmarks_present"], [28, 29])
        self.assertIn(1, good["benchmarks_missing"])
        self.assertTrue(good["partial"])
        self.assertAlmostEqual(good["per_benchmark"]["28"]["coverage"], 1.0)
        self.assertEqual(good["rows"], 24)
        self.assertEqual(sorted(good["areas"]), ["knowledge", "language"])
        self.assertIn("sample", good["note"].lower())

    def test_unanswered_rows_count_as_wrong_through_coverage(self):
        path = self.results_file("half", lambda r: True)
        lines = path.read_text(encoding="utf-8").splitlines()[:12]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        half = ev.score_results(path, self.rows, edition="0.3")
        full = ev.score_results(self.results_file("good", lambda r: True), self.rows, edition="0.3")
        self.assertLess(half["per_benchmark"]["29"]["coverage"], 1.0)
        self.assertLess(half["public"], full["public"])

    def test_score_reads_the_sample_directory_and_its_manifest(self):
        selection = sample.select(self.rows, per_benchmark=5, seed=3, edition="0.3")
        sample.write(selection, self.dir / "sample")
        by_id = {r["_evaluation"]["run_id"]: r for r in self.rows}
        rows = []
        for r in selection.rows:
            gold = by_id[r["_evaluation"]["run_id"]]["expected"]["q1"]
            keys = list(r["questions"]["q1"]["criteria"])
            rows.append({**r["_evaluation"], "status": "ok", "total_wall_ms": 1.0,
                         "response": {"answers": {"q1": {"type": "choice", "choice": gold, "probabilities": {k: (1.0 if k == gold else 0.0) for k in keys}}}}})
        results = write_jsonl(self.dir / "res.jsonl", rows)
        for ref in (self.dir / "sample", self.dir / "sample" / sample.MANIFEST_NAME):
            out = ev.score_sample(results, ref, edition="0.3")
            self.assertEqual(out["rows"], len(selection.rows))
            self.assertEqual(out["sample"]["totals"]["rows"], len(selection.rows))

    def test_paired_bootstrap_of_a_clear_gain_excludes_zero(self):
        a = self.results_file("good", lambda r: True)
        b = self.results_file("bad", lambda r: False)
        out = ev.compare_runs(a, b, self.rows, edition="0.3", replicates=30, seed=5)
        self.assertGreater(out["delta"], 0)
        self.assertGreater(out["low"], 0)
        self.assertLessEqual(out["low"], out["delta"] + 1e-9)
        self.assertGreaterEqual(out["high"], out["delta"] - 1e-9)
        self.assertEqual(out["replicates"], 30)
        same = ev.compare_runs(a, a, self.rows, edition="0.3", replicates=10, seed=5)
        self.assertAlmostEqual(same["delta"], 0.0)
        self.assertAlmostEqual(same["low"], 0.0)
        self.assertAlmostEqual(same["high"], 0.0)

    def test_the_bootstrap_is_reproducible_for_a_seed(self):
        a = self.results_file("good", lambda r: r["_evaluation"]["run_id"][-1] in "02468")
        b = self.results_file("bad", lambda r: r["_evaluation"]["run_id"][-1] in "123")
        one = ev.compare_runs(a, b, self.rows, edition="0.3", replicates=12, seed=9)
        two = ev.compare_runs(a, b, self.rows, edition="0.3", replicates=12, seed=9)
        self.assertEqual(one, two)


class CalibrationItemsContract(HermeticTestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_items_carry_logits_per_view_and_the_gold_option_index(self):
        rows = [kit_row(0, gold="B"), kit_row(1, qtype="noul", gold=True, catalog_id=29), kit_row(2, gold="Z")]
        write_jsonl(self.dir / "req.jsonl", rows)
        ev.run_requests(StubEngine(), self.dir / "req.jsonl", self.dir / "out", log=lambda t: None)
        items = ev.calibration_items(self.dir / "out", rows)
        by = {i.run_id: i for i in items}
        self.assertEqual(by["28:T:T:0"].gold, 1)
        self.assertEqual(by["28:T:T:0"].qtype, "choice")
        self.assertEqual(by["28:T:T:0"].logits, [[2.0, 1.0]])
        self.assertEqual(by["29:T:T:1"].gold, 1)
        self.assertEqual(by["29:T:T:1"].qtype, "noul")
        self.assertNotIn("28:T:T:2", by)                          # gold "Z" is not one of the options: not usable, not guessed

    def test_rows_without_raw_logits_are_skipped(self):
        rows = [kit_row(0, gold="A")]
        write_jsonl(self.dir / "req.jsonl", rows)
        ev.run_requests(StubEngine({"state-0": "unsupported"}), self.dir / "req.jsonl", self.dir / "out", log=lambda t: None)
        self.assertEqual(ev.calibration_items(self.dir / "out", rows), [])


if __name__ == "__main__":
    unittest.main()
