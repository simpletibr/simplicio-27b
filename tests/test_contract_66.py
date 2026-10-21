"""Contract tests for issue #66 (F1, fidelity): our answers against the per-row results published for d3 and d3-mini.

Pure Python: the published files are built in the test, the "ours" side is a results dict.  The tests that need torch (the
d3 prompt path) live in test_flow_66.py.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from decide_support import FakeHub, HermeticTestCase, needs_requests

from decide.sd import fidelity as fid


def choice(keys, probs):
    best = max(range(len(keys)), key=lambda i: probs[i])
    return {"type": "choice", "choice": keys[best], "probabilities": dict(zip(keys, probs))}


def noul(p):
    return {"type": "noul", "noul": p}


def row(run_id, answers, *, catalog=28, group=None, status="ok", payload="h"):
    return {"run_id": run_id, "catalog_id": catalog, "group_id": group or run_id, "status": status, "payload_sha256": payload,
            "response": {"answers": answers} if answers is not None else None}


class CompareContract(unittest.TestCase):
    def pair(self, ours_probs, pub_probs, keys=("A", "B", "C")):
        return ({"r": row("r", {"q": choice(list(keys), ours_probs)})}, {"r": row("r", {"q": choice(list(keys), pub_probs)})})

    def test_identical_answers_agree_everywhere_with_zero_probability_difference(self):
        ours, pub = self.pair([0.6, 0.3, 0.1], [0.6, 0.3, 0.1])
        out = fid.compare(ours, pub, replicates=20)
        self.assertEqual((out["rows"], out["fields"], out["agree_fields"]), (1, 1, 1))
        self.assertEqual((out["field_agreement"], out["row_agreement"]), (1.0, 1.0))
        self.assertEqual(out["prob"]["max_abs"], 0.0)

    def test_a_different_argmax_is_a_disagreement_and_the_field_is_listed(self):
        ours, pub = self.pair([0.2, 0.7, 0.1], [0.6, 0.3, 0.1])
        out = fid.compare(ours, pub, replicates=20)
        self.assertEqual(out["agree_fields"], 0)
        self.assertEqual(out["field_agreement"], 0.0)
        self.assertEqual(out["disagreements"][0]["run_id"], "r")
        self.assertEqual((out["disagreements"][0]["ours"], out["disagreements"][0]["published"]), ("B", "A"))
        self.assertEqual(out["disagree_clear"], 1)
        self.assertEqual(out["disagree_near_tie"], 0)

    def test_a_flip_where_the_published_top_two_were_a_hair_apart_is_counted_as_a_near_tie(self):
        ours, pub = self.pair([0.34, 0.36, 0.30], [0.351, 0.349, 0.30])
        out = fid.compare(ours, pub, near_tie=0.02, replicates=10)
        self.assertEqual((out["agree_fields"], out["disagree_near_tie"], out["disagree_clear"]), (0, 1, 0))

    def test_noul_agrees_when_both_sides_are_on_the_same_side_of_one_half(self):
        ours = {"a": row("a", {"q": noul(0.9)}), "b": row("b", {"q": noul(0.4)}), "c": row("c", {"q": noul(0.51)})}
        pub = {"a": row("a", {"q": noul(0.6)}), "b": row("b", {"q": noul(0.55)}), "c": row("c", {"q": noul(0.49)})}
        out = fid.compare(ours, pub, replicates=10)
        self.assertEqual((out["fields"], out["agree_fields"]), (3, 1))
        self.assertAlmostEqual(out["prob"]["max_abs"], 0.3)

    def test_probability_differences_are_measured_per_option(self):
        ours, pub = self.pair([0.5, 0.3, 0.2], [0.4, 0.4, 0.2])
        out = fid.compare(ours, pub, replicates=10)
        self.assertAlmostEqual(out["prob"]["max_abs"], 0.1)
        self.assertAlmostEqual(out["prob"]["mean_abs"], 0.2 / 3)
        self.assertAlmostEqual(out["prob"]["mean_tv"], 0.1)

    def test_a_row_agrees_only_when_all_its_questions_do(self):
        ours = {"r": row("r", {"q1": choice(["A", "B"], [0.9, 0.1]), "q2": choice(["A", "B"], [0.2, 0.8])})}
        pub = {"r": row("r", {"q1": choice(["A", "B"], [0.8, 0.2]), "q2": choice(["A", "B"], [0.7, 0.3])})}
        out = fid.compare(ours, pub, replicates=10)
        self.assertEqual((out["fields"], out["agree_fields"], out["rows"], out["agree_rows"]), (2, 1, 1, 0))
        self.assertEqual((out["field_agreement"], out["row_agreement"]), (0.5, 0.0))

    def test_rows_that_are_not_ok_on_either_side_or_missing_are_counted_not_compared(self):
        ours = {"a": row("a", {"q": noul(0.9)}), "b": row("b", None, status="unsupported"), "c": row("c", {"q": noul(0.9)}),
                "d": row("d", {"q": noul(0.9)}, payload="x")}
        pub = {"a": row("a", {"q": noul(0.9)}), "b": row("b", {"q": noul(0.9)}), "d": row("d", {"q": noul(0.9)}, payload="y"),
               "e": row("e", {"q": noul(0.9)})}
        out = fid.compare(ours, pub, replicates=10)
        self.assertEqual(out["rows"], 1)
        self.assertEqual(out["not_ok"], 1)                         # b: unsupported on our side
        self.assertEqual(out["missing_in_published"], 1)           # c
        self.assertEqual(out["payload_mismatch"], 1)               # d: not the same request, so not compared
        self.assertEqual(out["missing_in_ours"], 1)                # e

    def test_other_option_sets_or_types_are_mismatches_and_count_as_disagreements(self):
        ours = {"a": row("a", {"q": choice(["A", "B"], [0.6, 0.4])}), "b": row("b", {"q": noul(0.9)})}
        pub = {"a": row("a", {"q": choice(["A", "C"], [0.6, 0.4])}), "b": row("b", {"q": choice(["A", "B"], [0.6, 0.4])})}
        out = fid.compare(ours, pub, replicates=10)
        self.assertEqual(out["fields"], 2)
        self.assertEqual(out["agree_fields"], 0)
        self.assertEqual(out["structure_mismatch"], 2)

    def test_per_benchmark_agreement_and_the_cluster_bootstrap(self):
        ours, pub = {}, {}
        for i in range(40):
            catalog = 28 if i < 20 else 29
            flip = i % 10 == 0                                    # 2 flips in each benchmark
            ours[f"r{i}"] = row(f"r{i}", {"q": choice(["A", "B"], [0.1, 0.9] if flip else [0.9, 0.1])}, catalog=catalog, group=f"g{i // 2}")
            pub[f"r{i}"] = row(f"r{i}", {"q": choice(["A", "B"], [0.9, 0.1])}, catalog=catalog, group=f"g{i // 2}")
        out = fid.compare(ours, pub, replicates=200, seed=3)
        self.assertEqual(out["agree_fields"], 36)
        self.assertAlmostEqual(out["field_agreement"], 0.9)
        self.assertEqual(out["per_benchmark"]["28"], {"rows": 20, "fields": 20, "agree_fields": 18, "agreement": 0.9})
        low, high = out["ci"]
        self.assertLess(low, 0.9)
        self.assertGreater(high, 0.9)
        self.assertLessEqual(high, 1.0)
        self.assertEqual(fid.compare(ours, pub, replicates=200, seed=3)["ci"], out["ci"])           # reproducible

    def test_nothing_in_common_gives_no_agreement_number(self):
        out = fid.compare({"a": row("a", {"q": noul(0.9)})}, {"b": row("b", {"q": noul(0.9)})}, replicates=5)
        self.assertEqual(out["rows"], 0)
        self.assertIsNone(out["field_agreement"])
        self.assertIsNone(out["ci"])

    def test_at_most_the_requested_number_of_examples_is_kept(self):
        ours = {f"r{i}": row(f"r{i}", {"q": noul(0.9)}) for i in range(30)}
        pub = {f"r{i}": row(f"r{i}", {"q": noul(0.1)}) for i in range(30)}
        out = fid.compare(ours, pub, replicates=5, examples=7)
        self.assertEqual(len(out["disagreements"]), 7)
        self.assertEqual(out["fields"] - out["agree_fields"], 30)


class GateContract(unittest.TestCase):
    def result(self, rows, agree, fields=None):
        fields = fields or rows
        return {"rows": rows, "fields": fields, "agree_fields": agree, "field_agreement": agree / fields if fields else None,
                "ci": [agree / fields - 0.02, min(1.0, agree / fields + 0.02)] if fields else None}

    def test_pass_fail_and_insufficient(self):
        ok = fid.gate(self.result(600, 590), threshold=0.98, min_rows=500)
        self.assertEqual((ok["status"], ok["rows"], ok["threshold"]), ("pass", 600, 0.98))
        bad = fid.gate(self.result(600, 560), threshold=0.98, min_rows=500)
        self.assertEqual(bad["status"], "fail")
        few = fid.gate(self.result(499, 499), threshold=0.98, min_rows=500)
        self.assertEqual(few["status"], "insufficient")
        self.assertIn("500", few["reason"])

    def test_the_threshold_is_inclusive_and_the_interval_is_reported(self):
        edge = fid.gate(self.result(1000, 950), threshold=0.95, min_rows=500)
        self.assertEqual(edge["status"], "pass")
        self.assertFalse(edge["firm"])                              # the interval's lower end is below the threshold
        firm = fid.gate(self.result(1000, 999), threshold=0.95, min_rows=500)
        self.assertTrue(firm["firm"])

    def test_no_comparable_rows_is_insufficient_not_a_crash(self):
        out = fid.gate({"rows": 0, "fields": 0, "agree_fields": 0, "field_agreement": None, "ci": None}, threshold=0.95, min_rows=500)
        self.assertEqual(out["status"], "insufficient")


class PublishedContract(HermeticTestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.rows = [{"run_id": f"r{i}", "catalog_id": 28, "group_id": f"g{i}", "track": "T", "status": "ok", "payload_sha256": f"h{i}",
                      "dataset": "D", "engine": "d3_engine:D3Engine", "started_utc": "x", "completed_utc": "y", "total_wall_ms": 1.0,
                      "source_path": "p", "proxy_tokens": 10,
                      "response": {"model": "d3", "answers": {"q": noul(0.7)}, "usage": {"input_tokens": 5}}} for i in range(6)]
        self.blob = gzip.compress("".join(json.dumps(r) + "\n" for r in self.rows).encode("utf-8"))

    def test_load_keeps_only_what_the_comparison_reads_and_only_the_wanted_rows(self):
        path = self.dir / "results.jsonl.gz"
        path.write_bytes(self.blob)
        got = fid.load_published(path, wanted={"r1", "r4", "nope"})
        self.assertEqual(sorted(got), ["r1", "r4"])
        self.assertEqual(set(got["r1"]), {"run_id", "catalog_id", "group_id", "status", "payload_sha256", "response"})
        self.assertEqual(got["r1"]["response"], {"answers": {"q": noul(0.7)}})
        self.assertEqual(len(fid.load_published(path)), 6)

    def test_checksum_line_parsing_and_verification(self):
        digest = hashlib.sha256(self.blob).hexdigest()
        sums = f"{'0' * 64}  RUN_NOTES.md\n{digest}  results.jsonl.gz\n"
        self.assertEqual(fid.parse_sha256sums(sums)["results.jsonl.gz"], digest)
        path = self.dir / "results.jsonl.gz"
        path.write_bytes(self.blob)
        fid.verify_sha256(path, sums)                               # no exception
        path.write_bytes(self.blob + b"x")
        with self.assertRaisesRegex(fid.FidelityError, "sha256"):
            fid.verify_sha256(path, sums)
        with self.assertRaisesRegex(fid.FidelityError, "no checksum"):
            fid.verify_sha256(path, f"{digest}  other.txt\n")

    @needs_requests
    def test_fetch_downloads_the_run_file_and_checks_it_against_sha256sums(self):
        with FakeHub() as hub:
            digest = hashlib.sha256(self.blob).hexdigest()
            hub.files["org/pub/runs/mini/results.jsonl.gz"] = self.blob
            hub.files["org/pub/runs/mini/SHA256SUMS"] = f"{digest}  results.jsonl.gz\n".encode()
            path = fid.fetch_published("org/pub", "mini", self.dir / "dl", endpoint=hub.base_url, sleep=lambda s: None)
            self.assertEqual(path.read_bytes(), self.blob)
            self.assertEqual(fid.fetch_published("org/pub", "mini", self.dir / "dl", endpoint=hub.base_url, sleep=lambda s: None), path)
            hub.files["org/pub/runs/mini/SHA256SUMS"] = f"{'1' * 64}  results.jsonl.gz\n".encode()
            with self.assertRaisesRegex(fid.FidelityError, "sha256"):
                fid.fetch_published("org/pub", "mini", self.dir / "dl2", endpoint=hub.base_url, sleep=lambda s: None)
            self.assertFalse((self.dir / "dl2" / "results.jsonl.gz").exists())                     # a bad file is not left behind

    def test_fetch_resolves_datasets_not_models(self):
        with FakeHub() as hub:
            self.assertIn("/datasets/org/pub/resolve/main/runs/mini/results.jsonl.gz", fid.published_url("org/pub", "mini", endpoint=hub.base_url))


if __name__ == "__main__":
    unittest.main()
