"""Contract tests for issue #65 (F0, kit core): the pieces every other phase builds on.

Static or CPU-only: platform detection and secrets, config loading and overlays, the session budget,
the vendored pplx-decider and d3 files (licence, notice, byte-for-byte provenance), the public interface
in decide/sd/api.py, the notebooks, and the rule that no secret is ever written into the tree.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from decide_support import ROOT, HermeticTestCase, needs_torch

from decide.sd import budget as B
from decide.sd import config as C
from decide.sd import env

DECIDE = ROOT / "decide"


def noexist(_):
    return False


class EnvContract(HermeticTestCase):
    def test_detect_kaggle_by_dir_and_env(self):
        self.assertEqual(env.detect_platform({}, exists=lambda p: p == "/kaggle/working", module_available=lambda m: False), "kaggle")
        self.assertEqual(env.detect_platform({"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, exists=noexist, module_available=lambda m: False), "kaggle")

    def test_detect_colab_modal_generic_and_override(self):
        no_mod = lambda m: False  # noqa: E731
        self.assertEqual(env.detect_platform({"COLAB_RELEASE_TAG": "x"}, exists=noexist, module_available=no_mod), "colab")
        self.assertEqual(env.detect_platform({}, exists=noexist, module_available=lambda m: m == "google.colab"), "colab")
        self.assertEqual(env.detect_platform({"MODAL_TASK_ID": "ta-1"}, exists=noexist, module_available=no_mod), "modal")
        self.assertEqual(env.detect_platform({}, exists=noexist, module_available=no_mod), "generic")
        self.assertEqual(env.detect_platform({"SD_PLATFORM": "colab", "MODAL_TASK_ID": "t"}, exists=noexist), "colab")

    def test_workdir_defaults_and_override(self):
        self.assertEqual(str(env.default_workdir("kaggle", {})), "/kaggle/working")
        self.assertEqual(str(env.default_workdir("colab", {})), "/content/sd_work")
        self.assertEqual(str(env.default_workdir("modal", {}, exists=lambda p: p == "/cache")), "/cache/sd_work")
        self.assertEqual(str(env.default_workdir("generic", {"SD_WORKDIR": "/x/y"})), "/x/y")

    def test_get_secret_order_and_failures(self):
        def boom(_n):
            raise RuntimeError("no")

        self.assertEqual(env.get_secret("HF_TOKEN", {"HF_TOKEN": "env"}, kaggle_client=lambda n: "kag"), "env")
        self.assertEqual(env.get_secret("HF_TOKEN", {}, kaggle_client=lambda n: "kag"), "kag")
        self.assertEqual(env.get_secret("HF_TOKEN", {}, kaggle_client=boom, colab_userdata=lambda n: "col"), "col")
        self.assertIsNone(env.get_secret("HF_TOKEN", {}, kaggle_client=lambda n: None, colab_userdata=lambda n: None))

    def test_ensure_and_require_token(self):
        e = {}
        self.assertEqual(env.ensure_hf_token(e, kaggle_client=lambda n: "tok"), "tok")
        self.assertEqual(e["HF_TOKEN"], "tok")
        with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
            env.require_hf_token(environ={}, kaggle_client=lambda n: None, colab_userdata=lambda n: None)

    def test_parse_nvidia_smi(self):
        names, mem = env.parse_nvidia_smi("Tesla T4, 15360\nTesla T4, 15360\n")
        self.assertEqual(names, ("Tesla T4", "Tesla T4"))
        self.assertEqual(mem, (15.0, 15.0))
        self.assertEqual(env.parse_nvidia_smi("garbage"), ((), ()))

    def test_plan_devices_kaggle_t4x2_splits(self):
        p = env.plan_devices([15.0, 15.0], "nf4")
        self.assertTrue(p.ok)
        self.assertEqual((p.strategy, p.embeddings_device), ("split2", "cuda:0"))

    def test_plan_devices_colab_single_t4_is_tight_with_cpu_embeddings(self):
        p = env.plan_devices([15.0], "nf4")
        self.assertTrue(p.ok)
        self.assertEqual((p.strategy, p.embeddings_device), ("single_tight", "cpu"))
        self.assertTrue(any("marginal" in n for n in p.notes))

    def test_plan_devices_big_gpu_and_insufficient(self):
        self.assertEqual(env.plan_devices([48.0], "nf4").strategy, "single")
        self.assertEqual(env.plan_devices([96.0], "bf16").strategy, "bf16_single")
        bad = env.plan_devices([8.0], "nf4")
        self.assertFalse(bad.ok)
        self.assertEqual(bad.strategy, "insufficient")
        off = env.plan_devices([10.0], "nf4", allow_cpu_offload=True)
        self.assertTrue(off.ok)
        self.assertGreater(off.cpu_offload_layers, 0)
        self.assertTrue(any("UNVERIFIED" in n for n in off.notes))
        self.assertFalse(env.plan_devices([], "nf4").ok)


class ConfigContract(HermeticTestCase):
    def raw(self):
        import yaml

        return yaml.safe_load(C.DEFAULT_PATH.read_text())

    def test_default_config_loads_and_has_required_values(self):
        cfg = C.load_config()
        self.assertEqual(cfg.hub.user, "wesleysimplicio")
        self.assertEqual(cfg.hub.state_dataset, "wesleysimplicio/simplicio-decide-state")
        self.assertEqual(cfg.hub.winner_repo, "wesleysimplicio/Simplicio-Decide-27B")
        self.assertEqual((cfg.eval.seed, cfg.eval.quant, cfg.eval.compute_dtype), (20261021, "nf4", "float16"))
        self.assertEqual((cfg.budget.session_hours, cfg.budget.reserve_minutes), (11.5, 20))
        self.assertEqual((cfg.calibration.by, cfg.calibration.holdout), ("question_type", 0.5))
        self.assertEqual(cfg.tta.rotations, 2)
        self.assertEqual(cfg.models.anchor.repo, "perplexity-ai/pplx-decider-v1.2")
        self.assertEqual(cfg.models.lora.repo, "wesleysimplicio/Simplicio-27B")
        self.assertEqual(cfg.models.base.revision, "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")

    def test_gates_follow_the_issue_thresholds(self):
        cfg = C.load_config()
        self.assertEqual((cfg.gates.g0_agree_4b, cfg.gates.g0_agree_27b_nf4), (0.98, 0.95))
        self.assertEqual(cfg.fidelity.min_rows, 500)

    def test_overrides_merge_deeply(self):
        cfg = C.load_config(overrides={"eval": {"per_benchmark": 3}, "merge": {"method": "ties"}})
        self.assertEqual((cfg.eval.per_benchmark, cfg.eval.seed, cfg.merge.method), (3, 20261021, "ties"))

    def test_validation_rejects(self):
        mutations = {
            "unknown method": lambda r: r["merge"].update(method="slerp"),
            "empty lambda grid": lambda r: r["merge"].update(lambda_grid=[]),
            "negative lambda": lambda r: r["merge"].update(lambda_grid=[-1]),
            "zero density": lambda r: r["merge"]["ties"].update(density=0),
            "duplicate donor": lambda r: r["models"]["donors"].append(dict(r["models"]["donors"][0])),
            "bad quant": lambda r: r["eval"].update(quant="int3"),
            "holdout 1.0": lambda r: r["calibration"].update(holdout=1.0),
            "reserve > session": lambda r: r["budget"].update(reserve_minutes=10_000),
            "missing tta": lambda r: r.pop("tta"),
            "bad tta mode": lambda r: r["tta"].update(mode="shuffle"),
            "bad head": lambda r: r["merge"].update(head="average"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label):
                r = self.raw()
                mutate(r)
                with self.assertRaises(C.ConfigError):
                    C.validate(r)

    def test_license_and_readout_exclusions_cannot_be_enabled(self):
        for name, repo in (("lychee", "someone/Lychee-27B"), ("amos", "mukaddamzaid/amos-1-large")):
            with self.subTest(name):
                r = self.raw()
                r["models"]["donors"].append({"name": name, "repo": repo, "weight": 1.0, "enabled": True})
                with self.assertRaisesRegex(C.ConfigError, "not allowed"):
                    C.validate(r)
        r = self.raw()
        r["models"]["donors"].append({"name": "amos", "repo": "mukaddamzaid/amos-1-large", "weight": 1.0, "enabled": False})
        C.validate(r)      # present but disabled is tolerated

    def test_candidates_are_deterministic_and_normalised(self):
        cfg = C.load_config(overrides={"models": {"donors": [
            {"name": "d3", "repo": "a/b", "weight": 3.0, "enabled": True},
            {"name": "t", "repo": "a/c", "weight": 1.0, "enabled": True},
            {"name": "off", "repo": "a/d", "weight": 9.0, "enabled": False},
        ]}})
        cands = C.enumerate_candidates(cfg)
        self.assertEqual(cands, C.enumerate_candidates(cfg))
        self.assertEqual(cands[0].name, "anchor")
        self.assertTrue(cands[0].is_anchor)
        ta = [c for c in cands if c.method == "task_arithmetic"]
        self.assertEqual([c.name for c in ta], ["ta_l0.25", "ta_l0.5", "ta_l0.75"])
        d = dict(ta[0].donors)
        self.assertEqual(set(d), {"d3", "t"})
        self.assertAlmostEqual(d["d3"], 0.25 * 0.75)
        self.assertAlmostEqual(d["t"], 0.25 * 0.25)
        self.assertEqual([c.name for c in cands if c.lora_weight][-2:], ["anchor+lora0.1", "anchor+lora0.25"])
        self.assertEqual(C.with_lora(ta[0], 0.1).name, "ta_l0.25+lora0.1")

    def test_ties_candidate_names_carry_density(self):
        cfg = C.load_config(overrides={"merge": {"method": "ties"}})
        names = [c.name for c in C.enumerate_candidates(cfg, include_lora=False)]
        self.assertEqual(names, ["anchor", "ties_l0.25_d0.5", "ties_l0.5_d0.5", "ties_l0.75_d0.5"])

    def test_cfg_attribute_access(self):
        cfg = C.load_config()
        with self.assertRaises(AttributeError):
            cfg.nope
        cfg.eval.seed = 1
        self.assertEqual(cfg["eval"]["seed"], 1)

    def test_overlays_from_squad_b_files_are_merged_alphabetically(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "config.yaml"
            base.write_text(C.DEFAULT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            (Path(tmp) / "config.data.yaml").write_text("data:\n  corpus: x.parquet\neval:\n  per_benchmark: 7\n", encoding="utf-8")
            (Path(tmp) / "config.rental.yaml").write_text("rental:\n  gpus: 8\neval:\n  per_benchmark: 9\n", encoding="utf-8")
            alone = C.load_config(base)
            self.assertEqual(alone.eval.per_benchmark, 16)
            self.assertNotIn("data", alone)
            merged = C.load_config(base, overlays=True)
            self.assertEqual((merged.data.corpus, merged.rental.gpus), ("x.parquet", 8))
            self.assertEqual(merged.eval.per_benchmark, 9)            # config.rental.yaml comes after config.data.yaml
            self.assertEqual(merged.eval.seed, 20261021)              # keys the overlays do not mention survive
            self.assertEqual(C.load_config(base, overlays=True, overrides={"eval": {"per_benchmark": 2}}).eval.per_benchmark, 2)

    def test_the_default_file_never_matches_the_overlay_glob(self):
        self.assertNotIn(C.DEFAULT_PATH, C.overlay_paths(C.DEFAULT_PATH))


class BudgetContract(unittest.TestCase):
    class Clock:
        def __init__(self):
            self.t = 1000.0

        def __call__(self):
            return self.t

    def test_ok_respects_reserve(self):
        c = self.Clock()
        b = B.Budget(1.0, 10.0, clock=c)           # 3600 s total, 600 s reserve
        self.assertAlmostEqual(b.available(), 3000)
        self.assertTrue(b.ok(3000))
        self.assertFalse(b.ok(3000.1))
        c.t += 1000
        self.assertAlmostEqual(b.elapsed(), 1000)
        self.assertAlmostEqual(b.remaining(), 2600)
        self.assertAlmostEqual(b.available(), 2000)
        self.assertTrue(b.ok(1999))
        self.assertFalse(b.ok(2001))

    def test_never_negative_and_guard_message(self):
        c = self.Clock()
        b = B.Budget(1.0, 10.0, clock=c)
        c.t += 10_000
        self.assertEqual(b.available(), 0.0)
        self.assertLess(b.remaining(), 0)
        self.assertTrue(b.ok(0.0))
        self.assertFalse(b.ok(1.0))
        with self.assertRaisesRegex(B.BudgetExceeded, "step"):
            b.guard(5.0)

    def test_start_offset_counts_earlier_work(self):
        c = self.Clock()
        b = B.Budget(2.0, 0.0, clock=c, start=c.t - 3600)
        self.assertAlmostEqual(b.elapsed(), 3600)
        self.assertAlmostEqual(b.available(), 3600)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            B.Budget(0, 1)
        with self.assertRaises(ValueError):
            B.Budget(1, 60)

    def test_from_config_and_fraction(self):
        c = self.Clock()
        b = B.Budget.from_config(C.load_config(), clock=c)
        self.assertEqual((b.total, b.reserve), (11.5 * 3600, 20 * 60))
        self.assertAlmostEqual(b.fraction(0.5), 0.5 * (11.5 * 3600 - 1200))


@needs_torch
class MergeMathContract(unittest.TestCase):
    """Pure merge math on tiny tensors: the properties a merge must have, not a re-implementation of it."""

    @classmethod
    def setUpClass(cls):
        import torch

        from decide.sd import merge

        cls.torch, cls.M = torch, merge

    def rt(self, *shape, seed=0, dtype=None):
        g = self.torch.Generator().manual_seed(seed)
        return self.torch.randn(*shape, generator=g).to(dtype or self.torch.float32)

    def close(self, a, b, atol=1e-6):
        self.assertTrue(self.torch.allclose(a, b, atol=atol), f"{a} != {b}")

    # -- task arithmetic
    def test_zero_weights_returns_base(self):
        base, d1, d2 = self.rt(6, 5, seed=1), self.rt(6, 5, seed=2), self.rt(6, 5, seed=3)
        self.assertTrue(self.torch.equal(self.M.task_arithmetic(base, [d1, d2], [0.0, 0.0]), base))
        self.assertTrue(self.torch.equal(self.M.task_arithmetic(base, [], []), base))

    def test_single_donor_weight_one_is_donor(self):
        base, d = self.rt(6, 5, seed=1), self.rt(6, 5, seed=2)
        self.close(self.M.task_arithmetic(base, [d], [1.0]), d)

    def test_task_arithmetic_matches_float64_reference(self):
        t = self.torch
        base, d1, d2 = self.rt(7, 9, seed=1), self.rt(7, 9, seed=2), self.rt(7, 9, seed=3)
        w = [0.3, 0.45]
        ref = base.double() + w[0] * (d1.double() - base.double()) + w[1] * (d2.double() - base.double())
        out = self.M.task_arithmetic(base, [d1, d2], w)
        self.assertEqual(out.dtype, t.float32)
        self.assertTrue(t.allclose(out.double(), ref, atol=1e-6))

    def test_bf16_inputs_are_merged_in_float32(self):
        t = self.torch
        base, d = self.rt(8, 8, seed=1, dtype=t.bfloat16), self.rt(8, 8, seed=2, dtype=t.bfloat16)
        out = self.M.task_arithmetic(base, [d], [0.5])
        self.assertEqual(out.dtype, t.float32)
        self.assertTrue(t.equal(out, base.float() + 0.5 * (d.float() - base.float())))

    def test_task_arithmetic_rejects_bad_arguments(self):
        base = self.rt(3, 3)
        for models, weights in (([self.rt(3, 3)], [0.1, 0.2]), ([self.rt(3, 3)], [-0.1]), ([self.rt(3, 4)], [0.1])):
            with self.assertRaises(ValueError):
                self.M.task_arithmetic(base, models, weights)

    # -- TIES
    def test_trim_keeps_exactly_k_largest_with_deterministic_ties(self):
        t = self.torch.tensor([[3.0, -1.0, 2.0], [-2.0, 0.5, 2.0]])
        out = self.M.trim_topk(t, 0.5)                  # k = 3: |3|, then two of the three |2| (lowest flat index first)
        self.assertEqual((out != 0).flatten().nonzero().flatten().tolist(), [0, 2, 3])
        self.assertEqual((out[0, 0].item(), out[0, 2].item(), out[1, 0].item()), (3.0, 2.0, -2.0))

    def test_trim_density_one_is_identity_and_tiny_keeps_one(self):
        t = self.rt(4, 5, seed=4)
        self.assertTrue(self.torch.equal(self.M.trim_topk(t, 1.0), t))
        out = self.M.trim_topk(t, 1e-9)
        self.assertEqual(int((out != 0).sum()), 1)
        self.assertEqual(out.abs().max().item(), t.abs().max().item())

    def test_trim_validates_density(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                self.M.trim_topk(self.rt(2, 2), bad)

    def test_ties_sign_election_and_disjoint_mean(self):
        t = self.torch
        base = t.zeros(4)
        m1, m2 = t.tensor([3.0, 1.0, -2.0, 0.0]), t.tensor([1.0, -4.0, -2.0, 0.0])
        out = self.M.ties(base, [m1, m2], [0.5, 0.5], density=1.0)
        # pos0 both +, mean 2; pos1 elected sign - (sum -1.5), only m2 agrees -> -4; pos2 both -2; pos3 no signal -> 0
        self.close(out, t.tensor([2.0, -4.0, -2.0, 0.0]))
        ta = self.M.task_arithmetic(base, [m1, m2], [0.5, 0.5])
        self.assertFalse(t.allclose(out, ta))           # differs from plain averaging where signs conflict
        self.close(ta, t.tensor([2.0, -1.5, -2.0, 0.0]))

    def test_ties_weights_bias_the_election(self):
        t = self.torch
        base, m1, m2 = t.zeros(2), t.tensor([1.0, 1.0]), t.tensor([-1.0, -1.0])
        first = self.M.ties(base, [m1, m2], [0.9, 0.1], density=1.0)
        second = self.M.ties(base, [m1, m2], [0.1, 0.9], density=1.0)
        self.assertTrue(bool((first > 0).all()) and bool((second < 0).all()))
        self.close(first, t.tensor([1.0, 1.0]))         # the scale is the sum of weights: only the winner survives

    def test_ties_trims_before_electing(self):
        t = self.torch
        out = self.M.ties(t.zeros(4), [t.tensor([3.0, 0.2, -2.0, 0.1])], [1.0], density=0.5)
        self.close(out, t.tensor([3.0, 0.0, -2.0, 0.0]))

    def test_ties_zero_weights_is_base_and_single_donor_density1_is_donor(self):
        base, d = self.rt(5, 6, seed=1), self.rt(5, 6, seed=2)
        self.assertTrue(self.torch.equal(self.M.ties(base, [d], [0.0], density=0.5), base))
        self.close(self.M.ties(base, [d], [1.0], density=1.0), d)

    def test_ties_scale_is_sum_of_weights(self):
        base, d = self.rt(5, 6, seed=1), self.rt(5, 6, seed=2)
        self.close(self.M.ties(base, [d], [0.4], density=1.0), base + 0.4 * (d - base))

    def test_ties_all_agree_equals_task_arithmetic(self):
        t = self.torch
        base, m = t.zeros(3), t.tensor([1.0, 2.0, -3.0])
        out = self.M.ties(base, [m, m.clone()], [0.25, 0.25], density=1.0)
        self.close(out, 0.5 * m)
        self.close(out, self.M.task_arithmetic(base, [m, m.clone()], [0.25, 0.25]))

    # -- LoRA
    def test_lora_delta_scale_and_shape(self):
        t = self.torch
        A, B = self.rt(4, 6, seed=1), self.rt(5, 4, seed=2)
        d = self.M.lora_delta(A, B, alpha=8, r=4)
        self.assertEqual((tuple(d.shape), d.dtype), ((5, 6), t.float32))
        self.close(d, 2.0 * (B @ A))                                          # alpha/r = 2
        self.close(self.M.lora_delta(A, B, alpha=4, r=4, weight=0.25), 0.25 * (B @ A))
        self.close(self.M.lora_delta(A, B, alpha=8, r=4, rslora=True), (8 / math.sqrt(4)) * (B @ A))

    def test_lora_delta_rejects_inconsistent_shapes(self):
        with self.assertRaises(ValueError):
            self.M.lora_delta(self.rt(4, 6), self.rt(5, 3), alpha=4, r=4)
        with self.assertRaises(ValueError):
            self.M.lora_delta(self.rt(4, 6), self.rt(5, 4), alpha=4, r=8)    # r disagrees with A.shape[0]

    def test_apply_lora_shape_check_and_identity_when_weight_zero(self):
        W, A, B = self.rt(5, 6, seed=3), self.rt(4, 6, seed=1), self.rt(5, 4, seed=2)
        self.assertTrue(self.torch.equal(self.M.apply_lora(W, A, B, alpha=4, r=4, weight=0.0), W.float()))
        with self.assertRaises(ValueError):
            self.M.apply_lora(self.rt(7, 6), A, B, alpha=4, r=4, weight=1.0)

    def test_lora_bf16_roundtrip_matches_published_merge_rule(self):
        t = self.torch
        # fase0b: shipped bf16 weights == bf16(base_bf16 + B@A) with alpha/r = 1
        base = self.rt(16, 32, seed=5, dtype=t.bfloat16)
        A, B = self.rt(32, 32, seed=1) * 0.02, self.rt(16, 32, seed=2) * 0.02
        merged = self.M.apply_lora(base, A, B, alpha=32, r=32, weight=1.0).to(t.bfloat16)
        self.assertTrue(t.equal(merged, (base.float() + B @ A).to(t.bfloat16)))

    def test_parse_lora_key_variants(self):
        f = self.M.parse_lora_key
        self.assertEqual(f("base_model.model.model.language_model.layers.3.self_attn.k_proj.lora_A.weight"),
                         ("model.language_model.layers.3.self_attn.k_proj", "A"))
        self.assertEqual(f("base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_B.default.weight"),
                         ("model.language_model.layers.0.mlp.gate_proj", "B"))
        self.assertEqual(f("language_model.layers.1.mlp.up_proj.lora_A.weight"), ("language_model.layers.1.mlp.up_proj", "A"))
        self.assertIsNone(f("base_model.model.model.language_model.layers.0.input_layernorm.weight"))

    def test_name_mapper_adds_or_strips_model_prefix(self):
        m = self.M.NameMapper({"language_model.layers.3.self_attn.k_proj.weight", "visual.patch_embed.proj.weight"})
        want = "language_model.layers.3.self_attn.k_proj.weight"
        self.assertEqual(m.to_anchor("model.language_model.layers.3.self_attn.k_proj"), want)
        self.assertEqual(m.to_anchor("language_model.layers.3.self_attn.k_proj"), want)
        self.assertIsNone(m.to_anchor("model.language_model.layers.9.self_attn.k_proj"))
        self.assertEqual(m.to_base(want), "model." + want)
        self.assertEqual(m.to_base("lm_head.weight"), "lm_head.weight")

    def test_pair_lora_tensors_and_missing_half(self):
        keys = [
            "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_A.weight",
            "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_B.weight",
            "base_model.model.model.language_model.layers.3.self_attn.k_proj.lora_A.weight",
        ]
        with self.assertRaisesRegex(ValueError, "lora_B"):
            self.M.pair_lora_keys(keys)
        self.assertEqual(self.M.pair_lora_keys(keys[:2]), {"model.language_model.layers.0.mlp.gate_proj": (keys[0], keys[1])})

    # -- merge_one
    def test_merge_one_anchor_reference_with_lora(self):
        t = self.torch
        anchor, donor = self.rt(5, 6, seed=1, dtype=t.bfloat16), self.rt(5, 6, seed=2, dtype=t.bfloat16)
        A, B = self.rt(2, 6, seed=3), self.rt(5, 2, seed=4)
        out = self.M.merge_one(anchor, [donor], [0.5], "task_arithmetic", lora=(A, B, 2, 2, 0.1))
        self.close(out, anchor.float() + 0.5 * (donor.float() - anchor.float()) + 0.1 * (B @ A))

    def test_merge_one_base_reference_adds_anchor_with_weight_one(self):
        base, anchor, donor = self.rt(4, 4, seed=1), self.rt(4, 4, seed=2), self.rt(4, 4, seed=3)
        out = self.M.merge_one(anchor, [donor], [0.25], "task_arithmetic", base=base)
        self.close(out, base + 1.0 * (anchor - base) + 0.25 * (donor - base))
        self.close(self.M.merge_one(anchor, [donor], [0.0], "task_arithmetic", base=base), anchor)

    def test_merge_one_no_donors_is_anchor_and_ties_dispatch(self):
        anchor, donor = self.rt(4, 4, seed=1), self.rt(4, 4, seed=2)
        self.assertTrue(self.torch.equal(self.M.merge_one(anchor, [], [], "task_arithmetic"), anchor))
        self.close(self.M.merge_one(anchor, [donor], [1.0], "ties", density=1.0), donor)
        with self.assertRaises(ValueError):
            self.M.merge_one(anchor, [donor], [1.0], "slerp")

    def test_choose_head_prefers_refit_and_never_averages(self):
        anchor, refit = self.rt(3, 4, seed=1), self.rt(3, 4, seed=2)
        self.assertIs(self.M.choose_head(anchor), anchor)
        self.assertIs(self.M.choose_head(anchor, refit), refit)


class VendorContract(unittest.TestCase):
    """The vendored upstream files keep their licence, their notice and their exact upstream bytes."""

    VENDOR = DECIDE / "sd" / "vendor"

    def manifest(self, package: str) -> dict:
        return json.loads((self.VENDOR / package / "VENDOR.json").read_text(encoding="utf-8"))

    def test_licence_and_notice_ship_next_to_the_code(self):
        for package in ("pplx_decider", "d3"):
            with self.subTest(package):
                lic = (self.VENDOR / package / "LICENSE").read_text(encoding="utf-8")
                self.assertIn("Apache License", lic)
                self.assertIn("Version 2.0", lic)
                notice = (self.VENDOR / package / "NOTICE").read_text(encoding="utf-8")
                self.assertIn("Qwen3.8-27B", notice)
        self.assertEqual(self.manifest("pplx_decider")["license"], "Apache-2.0")
        self.assertEqual(self.manifest("d3")["license"], "Apache-2.0")

    def test_unmodified_files_equal_upstream_after_the_notice(self):
        for package in ("pplx_decider", "d3"):
            meta = self.manifest(package)
            sentinel = meta["sentinel"] + "\n"
            for name, info in meta["files"].items():
                if info["modified"]:
                    continue
                with self.subTest(f"{package}/{name}"):
                    raw = (self.VENDOR / package / name).read_bytes()
                    if name.endswith(".py"):
                        text = raw.decode("utf-8")
                        self.assertIn(sentinel, text)
                        self.assertTrue(text.startswith("# Vendored from "))
                        raw = text.split(sentinel, 1)[1].encode("utf-8")
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), info["upstream_sha256"])

    def test_modified_files_say_so_in_a_prominent_notice(self):
        for package in ("pplx_decider", "d3"):
            meta = self.manifest(package)
            for name, info in meta["files"].items():
                if not info["modified"]:
                    continue
                with self.subTest(f"{package}/{name}"):
                    head = (self.VENDOR / package / name).read_text(encoding="utf-8").splitlines()[:6]
                    self.assertTrue(any("Modifications by simplicio-decide:" in line for line in head))

    def test_protocol_constants_match_the_shipped_json(self):
        joint = json.loads((self.VENDOR / "pplx_decider" / "joint_config.json").read_text(encoding="utf-8"))
        tree = ast.parse((self.VENDOR / "pplx_decider" / "protocol.py").read_text(encoding="utf-8"))
        marker = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "MARKER")
        self.assertEqual(joint["marker"], marker)
        self.assertEqual(joint["version"], 3)
        decision = json.loads((self.VENDOR / "pplx_decider" / "decision_config.json").read_text(encoding="utf-8"))
        self.assertEqual(len(decision["codes"]), 255)
        self.assertEqual(decision["attention_mode"], "noncausal_full_attention")
        self.assertAlmostEqual(decision["temperature"], 1.3823494677922652)


class NoSecretsContract(unittest.TestCase):
    """HF_TOKEN comes from the environment; no token-looking string may be written into the tree."""

    TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")
    PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")

    def files(self):
        out = subprocess.run(["git", "ls-files", "-z", "--", "decide", "notebooks/decide"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        untracked = subprocess.run(["git", "ls-files", "-z", "--others", "--exclude-standard", "--", "decide", "notebooks/decide"],
                                   cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return sorted({p for p in (out + untracked).split("\0") if p})

    def test_no_token_or_private_key_in_the_kit(self):
        self.assertTrue(self.files())
        for rel in self.files():
            path = ROOT / rel
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            with self.subTest(rel):
                self.assertIsNone(self.TOKEN.search(text))
                self.assertIsNone(self.PRIVATE_KEY.search(text))

    def test_tls_verification_is_never_disabled(self):
        pattern = re.compile(r"verify\s*=\s*False|CERTIFICATE_VERIFY_FAILED|_create_unverified_context|PYTHONHTTPSVERIFY")
        for rel in self.files():
            if not rel.endswith((".py", ".ipynb", ".md", ".yaml")) or rel.startswith("decide/sd/vendor/"):
                continue
            text = (ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(rel):
                self.assertIsNone(pattern.search(text))


class PackageLayout(unittest.TestCase):
    def test_the_kit_package_exists_and_is_importable_without_heavy_dependencies(self):
        # decide/ itself stays a namespace package shared with Squad B (decide/data, decide/train, decide/rental)
        self.assertTrue((DECIDE / "sd" / "__init__.py").is_file())
        code = "import sys; import decide.sd; assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
        run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False,
                             env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
