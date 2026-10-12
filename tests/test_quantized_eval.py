"""Tests for the quantization matrix quality gate (Marco 5 do Épico #63)."""

from __future__ import annotations

import unittest
from pathlib import Path

from benchmarks.quantized_eval import (
    evaluate_matrix,
    evaluate_quantization_variant,
    to_markdown_table,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parents[1]


class WilsonIntervalTest(unittest.TestCase):
    def test_zero_total_returns_zeros(self) -> None:
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_half_proportion(self) -> None:
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)

    def test_bounds_stay_in_unit_interval(self) -> None:
        low, high = wilson_interval(100, 100)
        self.assertGreaterEqual(low, 0.95)
        self.assertLessEqual(high, 1.0)


class QuantizationGateTest(unittest.TestCase):
    def test_pass_within_threshold(self) -> None:
        res = evaluate_quantization_variant(
            name="Q4_K_M",
            format_type="GGUF",
            quant_pass=55,
            quant_total=120,
            ref_pass=56,
            ref_total=120,
            max_drop_pp=3.0,
        )
        self.assertEqual(res["status"], "PASS")
        self.assertAlmostEqual(res["drop_percentage_points"], 0.83, places=2)

    def test_fail_exceeding_threshold(self) -> None:
        res = evaluate_quantization_variant(
            name="Q2_K",
            format_type="GGUF",
            quant_pass=40,
            quant_total=120,
            ref_pass=56,
            ref_total=120,
            max_drop_pp=3.0,
        )
        self.assertEqual(res["status"], "FAIL")
        self.assertGreater(res["drop_percentage_points"], 3.0)

    def test_matrix_evaluation_and_markdown(self) -> None:
        ref_data = {"loader": "BF16", "functional_pass": 56, "n": 120}
        variants = [
            {"variant": "FP8", "format": "FP8", "functional_pass": 55, "n": 120},
            {"variant": "Q4_K_M", "format": "GGUF", "functional_pass": 54, "n": 120},
        ]
        report = evaluate_matrix(ref_data, variants, max_drop_pp=3.0)
        self.assertEqual(report["gate_status"], "PASS")
        self.assertEqual(len(report["variants"]), 2)

        md = to_markdown_table(report)
        self.assertIn("Quantization Matrix Quality Gate Report", md)
        self.assertIn("FP8", md)
        self.assertIn("Q4_K_M", md)
        self.assertIn("PASS", md)


if __name__ == "__main__":
    unittest.main()
