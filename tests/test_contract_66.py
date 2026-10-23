"""Contract tests for issue #66 (F1, fidelity): our answers against the per-row results published for d3 and d3-mini.

Pure Python: the published files are built in the test, the "ours" side is a results dict.  The tests that need torch (the
d3 prompt path) live in test_flow_66.py.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import posixpath
import random
import tempfile
import unittest
import urllib.parse
from pathlib import Path

from decide_support import FakeHub, HermeticTestCase, needs_requests

from decide.sd import config as sd_config
from decide.sd import evaluate
from decide.sd import fidelity as fid

PIN = "a1b2c3d4" * 5                                            # a full commit hash: the only revision the kit reads a dataset at


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

    def test_noul_at_exactly_one_half_is_the_true_side_on_both_sides(self):
        ours = {"a": row("a", {"q": noul(0.5)}), "b": row("b", {"q": noul(0.5)})}
        pub = {"a": row("a", {"q": noul(0.51)}), "b": row("b", {"q": noul(0.49)})}
        out = fid.compare(ours, pub, replicates=5)
        self.assertEqual((out["fields"], out["agree_fields"]), (2, 1))
        self.assertEqual(out["disagreements"][0]["run_id"], "b")
        self.assertEqual((out["disagreements"][0]["ours"], out["disagreements"][0]["published"]), ("true", "false"))

    def test_a_noul_flip_is_a_near_tie_only_when_the_published_p_is_within_the_margin_of_one_half(self):
        for published, near, clear in ((0.505, 1, 0), (0.515, 0, 1), (0.495, 1, 0), (0.485, 0, 1), (0.45, 0, 1)):
            with self.subTest(published=published):
                ours = {"r": row("r", {"q": noul(0.9 if published < 0.5 else 0.3)})}
                out = fid.compare(ours, {"r": row("r", {"q": noul(published)})}, near_tie=0.02, replicates=5)
                self.assertEqual((out["agree_fields"], out["disagree_near_tie"], out["disagree_clear"]), (0, near, clear))

    def test_the_margin_of_a_three_way_question_is_the_gap_to_the_runner_up_not_to_the_last(self):
        ours, pub = self.pair([0.2, 0.5, 0.3], [0.45, 0.43, 0.12])
        out = fid.compare(ours, pub, near_tie=0.05, replicates=5)
        self.assertEqual((out["disagree_near_tie"], out["disagree_clear"]), (1, 0))
        self.assertAlmostEqual(out["disagreements"][0]["published_margin"], 0.02)

    def test_a_published_row_that_is_not_ok_is_not_compared_even_when_ours_is(self):
        ours = {"a": row("a", {"q": noul(0.9)}), "b": row("b", {"q": noul(0.9)})}
        pub = {"a": row("a", {"q": noul(0.9)}), "b": row("b", None, status="error")}
        out = fid.compare(ours, pub, replicates=5)
        self.assertEqual((out["rows"], out["not_ok"]), (1, 1))

    def test_the_interval_is_the_95_percent_of_a_seeded_cluster_bootstrap_and_the_level_changes_it(self):
        ours, pub, tallies = {}, {}, []
        for i in range(40):
            flip = i % 2 == 0                                     # half of the 40 groups flip: a wide interval
            ours[f"r{i}"] = row(f"r{i}", {"q": choice(["A", "B"], [0.1, 0.9] if flip else [0.9, 0.1])}, group=f"g{i}")
            pub[f"r{i}"] = row(f"r{i}", {"q": choice(["A", "B"], [0.9, 0.1])}, group=f"g{i}")
            tallies.append([0 if flip else 1, 1])
        for level in (0.95, 0.9):
            rng, values = random.Random(11), []
            for _ in range(300):
                picks = [tallies[rng.randrange(len(tallies))] for _g in tallies]
                values.append(sum(p[0] for p in picks) / sum(p[1] for p in picks))
            tail = (1.0 - level) / 2.0
            with self.subTest(level=level):
                out = fid.compare(ours, pub, replicates=300, seed=11, level=level)
                self.assertEqual(out["ci"], [evaluate.percentile(values, tail), evaluate.percentile(values, 1.0 - tail)])
                self.assertEqual(out["ci_level"], level)

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

    def test_a_row_without_a_payload_hash_on_either_side_is_not_compared_it_is_unmatched(self):
        for mine_hash, their_hash in ((None, "h"), ("h", None), (None, None), ("", "h"), ("h", "")):
            with self.subTest(ours=mine_hash, published=their_hash):
                ours = {"a": row("a", {"q": noul(0.9)}, payload=mine_hash), "b": row("b", {"q": noul(0.9)})}
                pub = {"a": row("a", {"q": noul(0.1)}, payload=their_hash), "b": row("b", {"q": noul(0.9)})}
                out = fid.compare(ours, pub, replicates=5)
                self.assertEqual((out["rows"], out["fields"], out["unmatched"], out["payload_mismatch"]), (1, 1, 1, 0))
                self.assertEqual(out["field_agreement"], 1.0)      # "a" disagrees (0.9 against 0.1): compared, it would pull the number down

    def test_a_row_with_two_equal_hashes_is_compared_and_one_with_two_different_hashes_is_not(self):
        ours = {"a": row("a", {"q": noul(0.9)}, payload="x"), "b": row("b", {"q": noul(0.9)}, payload="x")}
        pub = {"a": row("a", {"q": noul(0.9)}, payload="x"), "b": row("b", {"q": noul(0.9)}, payload="y")}
        out = fid.compare(ours, pub, replicates=5)
        self.assertEqual((out["rows"], out["unmatched"], out["payload_mismatch"]), (1, 0, 1))

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


class PercentileContract(unittest.TestCase):
    """`evaluate.percentile` is what both bootstraps (the fidelity interval and the run comparison) take their ends from."""

    def test_linear_between_the_ranks_whatever_the_order_of_the_input(self):
        values = [30.0, 0.0, 40.0, 10.0, 20.0]
        for q, want in ((0.0, 0.0), (1.0, 40.0), (0.5, 20.0), (0.1, 4.0), (0.875, 35.0), (0.025, 1.0)):
            with self.subTest(q=q):
                self.assertAlmostEqual(evaluate.percentile(values, q), want)
        self.assertEqual(evaluate.percentile([3.0], 0.9), 3.0)

    def test_no_replicates_is_an_error_not_a_number(self):
        with self.assertRaises(evaluate.EvaluateError):
            evaluate.percentile([], 0.5)


class GateContract(unittest.TestCase):
    def result(self, rows, agree, fields=None):
        fields = fields or rows
        return {"rows": rows, "fields": fields, "agree_fields": agree, "field_agreement": agree / fields if fields else None,
                "ci": [agree / fields - 0.02, min(1.0, agree / fields + 0.02)] if fields else None}

    def covered(self, requested, **changes):
        """What a run that answered every request `ok` reports; `changes` says what went otherwise."""
        coverage = {"requested": requested, "ok": requested, "error": 0, "unsupported": 0, "abstained": 0, "missing": 0, "complete": True}
        coverage.update(changes)
        return coverage

    def test_pass_fail_and_insufficient(self):
        ok = fid.gate(self.result(600, 590), threshold=0.98, min_rows=500, coverage=self.covered(600))
        self.assertEqual((ok["status"], ok["rows"], ok["threshold"]), ("pass", 600, 0.98))
        bad = fid.gate(self.result(600, 560), threshold=0.98, min_rows=500, coverage=self.covered(600))
        self.assertEqual(bad["status"], "fail")
        few = fid.gate(self.result(499, 499), threshold=0.98, min_rows=500, coverage=self.covered(499))
        self.assertEqual(few["status"], "insufficient")
        self.assertIn("500", few["reason"])

    def test_the_threshold_is_inclusive_and_the_interval_is_reported(self):
        edge = fid.gate(self.result(1000, 950), threshold=0.95, min_rows=500, coverage=self.covered(1000))
        self.assertEqual(edge["status"], "pass")
        self.assertFalse(edge["firm"])                              # the interval's lower end is below the threshold
        firm = fid.gate(self.result(1000, 999), threshold=0.95, min_rows=500, coverage=self.covered(1000))
        self.assertTrue(firm["firm"])

    def test_no_comparable_rows_is_insufficient_not_a_crash(self):
        out = fid.gate({"rows": 0, "fields": 0, "agree_fields": 0, "field_agreement": None, "ci": None}, threshold=0.95, min_rows=500,
                       coverage=self.covered(0))
        self.assertEqual(out["status"], "insufficient")

    def test_the_gate_cannot_be_asked_without_saying_what_the_run_covered(self):
        with self.assertRaises(TypeError):
            fid.gate(self.result(600, 600), threshold=0.98, min_rows=500)               # type: ignore[call-arg]

    def test_requests_that_ended_in_error_are_no_verdict_even_when_the_rest_agrees(self):
        out = fid.gate(self.result(600, 600), threshold=0.98, min_rows=500, coverage=self.covered(1000, ok=600, error=400))
        self.assertEqual(out["status"], "insufficient")
        self.assertFalse(out["firm"])
        for text in ("400", "error", "600 of 1000"):
            self.assertIn(text, out["reason"])
        self.assertEqual((out["coverage"]["requested"], out["coverage"]["ok"], out["coverage"]["error"]), (1000, 600, 400))

    def test_a_single_error_is_enough_to_withhold_the_verdict(self):
        out = fid.gate(self.result(2000, 2000), threshold=0.98, min_rows=500, coverage=self.covered(2001, ok=2000, error=1))
        self.assertEqual((out["status"], out["firm"]), ("insufficient", False))

    def test_a_run_that_stopped_early_is_never_a_pass_and_never_firm(self):
        out = fid.gate(self.result(600, 600), threshold=0.98, min_rows=500, coverage=self.covered(1000, ok=600, missing=400, complete=False))
        self.assertEqual((out["status"], out["firm"]), ("insufficient", False))
        for text in ("stopped", "600 of 1000", "400 not run"):
            self.assertIn(text, out["reason"])
        flagged = fid.gate(self.result(1000, 1000), threshold=0.98, min_rows=500, coverage=self.covered(1000, complete=False))
        self.assertEqual(flagged["status"], "insufficient")                             # the flag alone is enough

    def test_requests_that_were_never_run_are_no_verdict(self):
        out = fid.gate(self.result(997, 997), threshold=0.98, min_rows=500, coverage=self.covered(1000, ok=997, missing=3))
        self.assertEqual(out["status"], "insufficient")
        self.assertIn("3 not run", out["reason"])

    def test_unsupported_and_abstained_requests_stay_out_of_the_agreement_but_are_counted_in_the_reason(self):
        out = fid.gate(self.result(600, 600), threshold=0.98, min_rows=500, coverage=self.covered(1000, ok=600, unsupported=300, abstained=100))
        self.assertEqual(out["status"], "pass")
        for text in ("300 unsupported", "100 abstained", "600 of 1000"):
            self.assertIn(text, out["reason"])
        self.assertEqual((out["coverage"]["unsupported"], out["coverage"]["abstained"]), (300, 100))

    def test_a_complete_clean_run_says_so_in_the_reason_too(self):
        out = fid.gate(self.result(600, 600), threshold=0.98, min_rows=500, coverage=self.covered(600))
        self.assertIn("600 of 600", out["reason"])

    def test_the_interim_agreement_of_a_run_without_a_verdict_is_still_shown(self):
        out = fid.gate(self.result(600, 300), threshold=0.98, min_rows=500, coverage=self.covered(1000, ok=600, missing=400, complete=False))
        self.assertEqual(out["status"], "insufficient")
        self.assertIn("0.5000", out["reason"])
        self.assertEqual(out["value"], 0.5)


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
            fetch = lambda dest: fid.fetch_published("org/pub", "mini", dest, endpoint=hub.base_url, revision=PIN, sleep=lambda s: None)
            path = fetch(self.dir / "dl")
            self.assertEqual(path.read_bytes(), self.blob)
            self.assertEqual(fetch(self.dir / "dl"), path)
            hub.files["org/pub/runs/mini/SHA256SUMS"] = f"{'1' * 64}  results.jsonl.gz\n".encode()
            with self.assertRaisesRegex(fid.FidelityError, "sha256"):
                fetch(self.dir / "dl2")
            self.assertFalse((self.dir / "dl2" / "results.jsonl.gz").exists())                     # a bad file is not left behind

    @needs_requests
    def test_a_copy_on_disk_that_does_not_match_the_checksums_is_downloaded_again(self):
        with FakeHub() as hub:
            digest = hashlib.sha256(self.blob).hexdigest()
            hub.files["org/pub/runs/mini/results.jsonl.gz"] = self.blob
            hub.files["org/pub/runs/mini/SHA256SUMS"] = f"{digest}  results.jsonl.gz\n".encode()
            (self.dir / "dl").mkdir()
            (self.dir / "dl" / "results.jsonl.gz").write_bytes(b"an old or damaged copy")
            path = fid.fetch_published("org/pub", "mini", self.dir / "dl", endpoint=hub.base_url, revision=PIN, sleep=lambda s: None)
            self.assertEqual(path.read_bytes(), self.blob)
            self.assertEqual(hub.requested, {("org/pub", PIN)})

    def test_fetch_resolves_datasets_not_models(self):
        with FakeHub() as hub:
            self.assertIn(f"/datasets/org/pub/resolve/{PIN}/runs/mini/results.jsonl.gz", fid.published_url("org/pub", "mini", endpoint=hub.base_url, revision=PIN))

    @needs_requests
    def test_both_files_are_read_at_the_pinned_commit_and_at_no_other(self):
        with FakeHub() as hub:
            digest = hashlib.sha256(self.blob).hexdigest()
            hub.files["org/pub/runs/mini/results.jsonl.gz"] = self.blob
            hub.files["org/pub/runs/mini/SHA256SUMS"] = f"{digest}  results.jsonl.gz\n".encode()
            fid.fetch_published("org/pub", "mini", self.dir / "dl", endpoint=hub.base_url, revision=PIN, sleep=lambda s: None)
            self.assertEqual(hub.requested, {("org/pub", PIN)})
            hub.requested.clear()
            fid.fetch_published("org/pub", "mini", self.dir / "dl-unchecked", endpoint=hub.base_url, revision=PIN, verify=False, sleep=lambda s: None)
            self.assertEqual(hub.requested, {("org/pub", PIN)})

    @needs_requests
    def test_a_missing_or_loose_revision_is_refused_before_anything_is_asked(self):
        bad = {"none": None, "empty": "", "a branch": "main", "a short hash": PIN[:7], "39 characters": PIN[:-1], "41 characters": PIN + "0",
               "capitals": PIN.upper(), "a trailing newline": PIN + "\n", "a tag": "v1.2", "a number": 12345}
        with FakeHub() as hub:
            for name, revision in bad.items():
                with self.subTest(name):
                    with self.assertRaisesRegex(fid.FidelityError, "40"):
                        fid.published_url("org/pub", "mini", endpoint=hub.base_url, revision=revision)
                    with self.assertRaisesRegex(fid.FidelityError, "40"):
                        fid.fetch_published("org/pub", "mini", self.dir / "dl", endpoint=hub.base_url, revision=revision, sleep=lambda s: None)
            self.assertEqual(hub.requested, set())
            self.assertFalse((self.dir / "dl" / "results.jsonl.gz").exists())
        with self.assertRaises(TypeError):                                       # there is no default to fall back on
            fid.published_url("org/pub", "mini")                                 # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            fid.fetch_published("org/pub", "mini", self.dir / "dl")              # type: ignore[call-arg]


    BAD_RUNS = {"dotdot": "..", "up and across": "../main", "a deep climb": "../../../../org/pub/resolve/main/runs/mini", "a slash": "a/b",
                "a backslash": "a\\b", "percent-encoded dots": "%2e%2e", "an encoded slash": "a%2fb", "a control character": "mini\x00",
                "a newline": "mini\n", "a tab": "a\tb", "empty": "", "blank": " ", "a space": "a b", "a dot": ".", "dots inside": "a..b",
                "a number": 5, "none": None}
    BAD_DATASETS = {"a climb in the name": "org/..", "a climb in the org": "../pub", "no org": "pub", "three parts": "org/pub/extra", "a leading slash": "/pub",
                    "a trailing slash": "org/", "a backslash": "org\\pub", "percent-encoded": "org/pub%2e", "empty": "", "none": None,
                    "a space": "org/pu b", "an empty part": "org//pub", "dots inside": "or..g/pub", "a newline": "org/pub\n"}

    def test_a_run_name_that_climbs_or_escapes_is_refused_before_a_url_exists(self):
        for name, run in self.BAD_RUNS.items():
            with self.subTest(name):
                with self.assertRaisesRegex(fid.FidelityError, "run"):
                    fid.published_url("org/pub", run, revision=PIN)

    def test_a_dataset_that_is_not_org_slash_name_is_refused_before_a_url_exists(self):
        for name, dataset in self.BAD_DATASETS.items():
            with self.subTest(name):
                with self.assertRaisesRegex(fid.FidelityError, "dataset"):
                    fid.published_url(dataset, "mini", revision=PIN)

    def test_a_good_name_gives_a_url_that_needs_no_resolving_and_stays_on_the_pinned_commit(self):
        url = fid.published_url("org/pub", "d3-mini.v2_x", endpoint="https://hub.example", revision=PIN)
        path = urllib.parse.urlsplit(url).path
        self.assertEqual(posixpath.normpath(path), path)                         # an HTTP client that resolves the dots would change nothing
        self.assertEqual(path, f"/datasets/org/pub/resolve/{PIN}/runs/d3-mini.v2_x/results.jsonl.gz")
        self.assertEqual(urllib.parse.urlsplit(url).netloc, "hub.example")

    @needs_requests
    def test_a_name_that_climbs_asks_the_network_for_nothing_and_leaves_nothing_behind(self):
        with FakeHub() as hub:
            for name, run in self.BAD_RUNS.items():
                with self.subTest(name):
                    with self.assertRaisesRegex(fid.FidelityError, "run"):
                        fid.fetch_published("org/pub", run, self.dir / "dl", endpoint=hub.base_url, revision=PIN, sleep=lambda s: None)
            for name, dataset in self.BAD_DATASETS.items():
                with self.subTest(name):
                    with self.assertRaisesRegex(fid.FidelityError, "dataset"):
                        fid.fetch_published(dataset, "mini", self.dir / "dl", endpoint=hub.base_url, revision=PIN, sleep=lambda s: None)
            self.assertEqual(hub.requested, set())
            self.assertFalse((self.dir / "dl").exists())                          # not even the folder for the download

    def test_an_endpoint_that_would_send_the_token_in_the_clear_is_refused_before_anything_is_asked(self):
        class Spy:
            def __init__(self):
                self.asked = []

            def get(self, url):
                self.asked.append(url)
                return b""

        for endpoint in ("http://hub.example", "http://huggingface.co", "http://localhost.example.com", "ftp://hub.example", "hub.example", "", None):
            with self.subTest(repr(endpoint)):
                spy = Spy()
                with self.assertRaisesRegex(fid.FidelityError, "endpoint"):
                    fid.fetch_published("org/pub", "mini", self.dir / "dl", revision=PIN, token="hf_secret", endpoint=endpoint, transport=spy)
                self.assertEqual(spy.asked, [])
        cfg = {"fidelity": {"published": {"mini": {"dataset": "org/pub", "run": "mini", "revision": PIN}}}, "hub": {"endpoint": "http://hub.example"}}
        spy = Spy()
        with self.assertRaisesRegex(fid.FidelityError, "endpoint"):
            fid.fetch_configured(cfg, "mini", self.dir / "dl", token="hf_secret", transport=spy)
        self.assertEqual(spy.asked, [])

    def test_an_https_endpoint_and_a_loopback_one_are_not_refused_for_their_scheme(self):
        class Gone:
            def get(self, url):
                raise fid.RemoteError("nothing here")

        for endpoint in ("https://hub.example", "http://127.0.0.1:9999", "http://localhost:9999"):
            with self.subTest(endpoint):
                with self.assertRaisesRegex(fid.FidelityError, "cannot read SHA256SUMS"):
                    fid.fetch_published("org/pub", "mini", self.dir / "dl", revision=PIN, endpoint=endpoint, transport=Gone())


class ConfiguredContract(HermeticTestCase):
    """The config names, for each published run, the dataset, the run and the commit it is read at; the fetch uses those and no other."""

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.cfg = sd_config.load_config()

    def test_the_shipped_config_gives_every_published_run_a_full_commit(self):
        entries = fid.published_entries(self.cfg)
        self.assertEqual(sorted(entries), sorted(self.cfg["fidelity"]["published"]))
        self.assertGreaterEqual(len(entries), 2)
        for name, entry in entries.items():
            with self.subTest(name):
                raw = self.cfg["fidelity"]["published"][name]
                self.assertEqual(entry, {"dataset": raw["dataset"], "run": raw["run"], "revision": raw["revision"]})
                self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")

    def test_a_name_the_config_does_not_have_says_which_it_has(self):
        with self.assertRaisesRegex(fid.FidelityError, r"d3_mini"):
            fid.published_entry(self.cfg, "d3_midi")

    def test_an_entry_without_a_full_commit_is_refused(self):
        for revision in (None, "main", PIN[:7], PIN.upper()):
            with self.subTest(repr(revision)):
                entry = {"dataset": "org/pub", "run": "mini"}
                if revision is not None:
                    entry["revision"] = revision
                with self.assertRaisesRegex(fid.FidelityError, "revision"):
                    fid.published_entry({"fidelity": {"published": {"x": entry}}}, "x")
        with self.assertRaisesRegex(fid.FidelityError, "dataset"):
            fid.published_entry({"fidelity": {"published": {"x": {"run": "mini", "revision": PIN}}}}, "x")
        with self.assertRaisesRegex(fid.FidelityError, "run"):
            fid.published_entry({"fidelity": {"published": {"x": {"dataset": "org/pub", "revision": PIN}}}}, "x")

    def test_a_run_or_dataset_that_climbs_is_refused_with_the_name_of_the_key(self):
        for key, bad in (("run", ".."), ("run", "../main"), ("run", "a/b"), ("run", "a\\b"), ("run", "%2e%2e"), ("run", "x\x00"), ("dataset", "org/.."),
                         ("dataset", "pub"), ("dataset", "../pub"), ("dataset", "org/pub/extra")):
            with self.subTest(f"{key}: {bad!r}"):
                entry = {"dataset": "org/pub", "run": "mini", "revision": PIN, key: bad}
                with self.assertRaisesRegex(fid.FidelityError, rf"fidelity\.published\.x\.{key}"):
                    fid.published_entry({"fidelity": {"published": {"x": entry}}}, "x")

    def test_the_config_refuses_a_published_run_or_dataset_that_climbs_before_any_check_runs(self):
        import yaml

        for key, bad in (("run", ".."), ("run", "../main"), ("run", "a/b"), ("run", "a\\b"), ("run", "%2e%2e"), ("run", ""), ("run", "x\x00"),
                         ("dataset", "org/.."), ("dataset", "pub"), ("dataset", "../pub"), ("dataset", "org/pub/extra")):
            for name in ("d3", "d3_mini"):
                with self.subTest(f"{name}.{key}: {bad!r}"):
                    raw = yaml.safe_load(sd_config.DEFAULT_PATH.read_text())
                    raw["fidelity"]["published"][name][key] = bad
                    with self.assertRaisesRegex(sd_config.ConfigError, rf"fidelity\.published\.{name}\.{key}"):
                        sd_config.validate(raw)
        sd_config.validate(yaml.safe_load(sd_config.DEFAULT_PATH.read_text()))          # the shipped names pass

    @needs_requests
    def test_the_endpoint_comes_from_the_config_unless_one_is_given(self):
        blob = gzip.compress(b'{"run_id": "r0", "status": "ok"}\n')
        digest = hashlib.sha256(blob).hexdigest()
        with FakeHub() as hub:
            hub.files["org/pub/runs/mini/results.jsonl.gz"] = blob
            hub.files["org/pub/runs/mini/SHA256SUMS"] = f"{digest}  results.jsonl.gz\n".encode()
            cfg = {"fidelity": {"published": {"mini": {"dataset": "org/pub", "run": "mini", "revision": PIN}}}, "hub": {"endpoint": hub.base_url}}
            path = fid.fetch_configured(cfg, "mini", self.dir / "dl", sleep=lambda s: None)
            self.assertEqual(path.read_bytes(), blob)
            self.assertEqual(hub.requested, {("org/pub", PIN)})

    @needs_requests
    def test_fetch_configured_reads_the_commit_the_config_names(self):
        blob = gzip.compress(b'{"run_id": "r0", "status": "ok"}\n')
        digest = hashlib.sha256(blob).hexdigest()
        cfg = {"fidelity": {"published": {"mini": {"dataset": "org/pub", "run": "mini", "revision": PIN}}}, "hub": {"endpoint": "unused"}}
        with FakeHub() as hub:
            hub.files["org/pub/runs/mini/results.jsonl.gz"] = blob
            hub.files["org/pub/runs/mini/SHA256SUMS"] = f"{digest}  results.jsonl.gz\n".encode()
            path = fid.fetch_configured(cfg, "mini", self.dir / "dl", endpoint=hub.base_url, sleep=lambda s: None)
            self.assertEqual(path.read_bytes(), blob)
            self.assertEqual(hub.requested, {("org/pub", PIN)})


if __name__ == "__main__":
    unittest.main()
