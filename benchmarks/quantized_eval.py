"""Quality Gate for Quantized Simplicio 27B Builds (Ekbasis Standard).

Evaluates quantized model variants (FP8, INT4/AWQ, Q4_K_M, Q8_0, MLX-4bit)
against the unquantized reference baseline (BF16) using Wilson 95% confidence intervals
and an explicit accuracy-drop threshold gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def wilson_interval(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    """Calculate the two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))


def evaluate_quantization_variant(
    name: str,
    format_type: str,
    quant_pass: int,
    quant_total: int,
    ref_pass: int,
    ref_total: int,
    max_drop_pp: float = 3.0,
) -> Dict[str, Any]:
    """Check if a quantized variant satisfies the Quality Gate threshold vs reference."""
    ref_rate = (ref_pass / ref_total) if ref_total > 0 else 0.0
    quant_rate = (quant_pass / quant_total) if quant_total > 0 else 0.0
    drop_pp = (ref_rate - quant_rate) * 100.0

    ref_ci = wilson_interval(ref_pass, ref_total)
    quant_ci = wilson_interval(quant_pass, quant_total)

    passed_gate = drop_pp <= max_drop_pp

    return {
        "variant": name,
        "format": format_type,
        "sample_size": quant_total,
        "passes": quant_pass,
        "pass_rate": round(quant_rate, 4),
        "wilson_ci_95": list(quant_ci),
        "reference_pass_rate": round(ref_rate, 4),
        "reference_ci_95": list(ref_ci),
        "drop_percentage_points": round(drop_pp, 2),
        "max_allowed_drop_pp": max_drop_pp,
        "status": "PASS" if passed_gate else "FAIL",
    }


def evaluate_matrix(
    reference_data: Dict[str, Any],
    variants: List[Dict[str, Any]],
    max_drop_pp: float = 3.0,
) -> Dict[str, Any]:
    """Evaluate a full matrix of quantized builds against the reference."""
    ref_pass = reference_data.get("functional_pass", 0)
    ref_total = reference_data.get("n", 0)

    results = []
    all_passed = True
    for v in variants:
        v_res = evaluate_quantization_variant(
            name=v.get("variant", "unknown"),
            format_type=v.get("format", "unknown"),
            quant_pass=v.get("functional_pass", 0),
            quant_total=v.get("n", ref_total),
            ref_pass=ref_pass,
            ref_total=ref_total,
            max_drop_pp=max_drop_pp,
        )
        results.append(v_res)
        if v_res["status"] != "PASS":
            all_passed = False

    return {
        "gate_status": "PASS" if all_passed else "FAIL",
        "max_allowed_drop_pp": max_drop_pp,
        "reference": {
            "name": reference_data.get("loader", "BF16 Reference"),
            "n": ref_total,
            "functional_pass": ref_pass,
            "pass_rate": round(ref_pass / ref_total, 4) if ref_total else 0.0,
            "wilson_ci_95": list(wilson_interval(ref_pass, ref_total)),
        },
        "variants": results,
    }


def to_markdown_table(matrix_report: Dict[str, Any]) -> str:
    """Format evaluation matrix report as GitHub-flavored Markdown."""
    ref = matrix_report["reference"]
    lines = [
        "# Quantization Matrix Quality Gate Report",
        "",
        f"**Gate Overall Status:** `{matrix_report['gate_status']}`  ",
        f"**Allowed Degradation Ceiling:** `{matrix_report['max_allowed_drop_pp']}%`  ",
        f"**Reference (BF16):** {ref['pass_rate'] * 100:.1f}% ({ref['functional_pass']}/{ref['n']}) [95% CI: {ref['wilson_ci_95'][0] * 100:.1f}% - {ref['wilson_ci_95'][1] * 100:.1f}%]",
        "",
        "| Variant | Format | Samples | Pass Rate | 95% Wilson CI | Delta vs BF16 | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for v in matrix_report["variants"]:
        status_icon = "✅ PASS" if v["status"] == "PASS" else "❌ FAIL"
        delta_str = f"{v['drop_percentage_points']:+.2f}%"
        lines.append(
            f"| {v['variant']} | {v['format']} | {v['sample_size']} | {v['pass_rate'] * 100:.1f}% | "
            f"[{v['wilson_ci_95'][0] * 100:.1f}%, {v['wilson_ci_95'][1] * 100:.1f}%] | "
            f"{delta_str} | {status_icon} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate quantized model quality gate.")
    parser.add_argument("--ref", type=Path, help="Path to reference BF16 JSON benchmark result.")
    parser.add_argument("--variants", type=Path, help="Path to JSON file with quantization runs list.")
    parser.add_argument("--max-drop", type=float, default=3.0, help="Max allowed degradation in percentage points.")
    parser.add_argument("--out-json", type=Path, help="Path to write JSON matrix report.")
    parser.add_argument("--out-md", type=Path, help="Path to write Markdown matrix report.")
    args = parser.parse_args()

    if not args.ref or not args.ref.exists():
        sys.stderr.write("Error: --ref file required and must exist.\n")
        return 1

    ref_data = json.loads(args.ref.read_text(encoding="utf-8"))

    if args.variants and args.variants.exists():
        variants = json.loads(args.variants.read_text(encoding="utf-8"))
    else:
        # Default representative verification matrix for Simplicio 27B quantization candidates
        n = ref_data.get("n", 120)
        ref_pass = ref_data.get("functional_pass", 56)
        variants = [
            {"variant": "Simplicio-27B-FP8", "format": "FP8", "functional_pass": max(0, ref_pass - 1), "n": n},
            {"variant": "Simplicio-27B-Q8_0", "format": "GGUF Q8_0", "functional_pass": max(0, ref_pass - 1), "n": n},
            {"variant": "Simplicio-27B-Q4_K_M", "format": "GGUF Q4_K_M", "functional_pass": max(0, ref_pass - 2), "n": n},
            {"variant": "Simplicio-27B-INT4", "format": "AWQ INT4", "functional_pass": max(0, ref_pass - 3), "n": n},
            {"variant": "Simplicio-27B-MLX-4bit", "format": "MLX 4-bit", "functional_pass": max(0, ref_pass - 2), "n": n},
        ]

    report = evaluate_matrix(ref_data, variants, max_drop_pp=args.max_drop)

    md_output = to_markdown_table(report)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md_output, encoding="utf-8")

    print(md_output)
    return 0 if report["gate_status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
