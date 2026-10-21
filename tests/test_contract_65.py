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
from unittest import mock

from decide_support import ROOT, HermeticTestCase, needs_bnb, needs_kit, needs_numpy, needs_torch

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
            "empty sweep sample": lambda r: r["eval"].update(sweep_per_benchmark=0),
            "no candidates allowed": lambda r: r["merge"].update(max_candidates=0),
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
        self.assertEqual([c.name for c in ta], ["ta_l0.25", "ta_l0.5"])
        d = dict(ta[0].donors)
        self.assertEqual(set(d), {"d3", "t"})
        self.assertAlmostEqual(d["d3"], 0.25 * 0.75)
        self.assertAlmostEqual(d["t"], 0.25 * 0.25)
        self.assertEqual([c.name for c in cands if c.lora_weight][-2:], ["anchor+lora0.1", "anchor+lora0.25"])
        self.assertEqual(C.with_lora(ta[0], 0.1).name, "ta_l0.25+lora0.1")

    def test_ties_candidate_names_carry_density(self):
        cfg = C.load_config(overrides={"merge": {"method": "ties"}})
        names = [c.name for c in C.enumerate_candidates(cfg, include_lora=False)]
        self.assertEqual(names, ["anchor", "ties_l0.25_d0.5", "ties_l0.5_d0.5"])

    def test_the_merge_sweep_is_cheap_by_default(self):
        """Fase 0A: averaging top models did not beat the best member, so F5 is a low-prior experiment."""
        cfg = C.load_config()
        self.assertEqual(list(cfg.merge.lambda_grid), [0.25, 0.5])
        self.assertLessEqual(len(C.enumerate_candidates(cfg)), cfg.merge.max_candidates)
        self.assertLess(cfg.eval.sweep_per_benchmark, cfg.eval.per_benchmark)
        self.assertLessEqual(cfg.merge.max_candidates, 6)

    def test_the_sweep_sample_never_exceeds_the_final_sample(self):
        self.assertEqual(C.sweep_per_benchmark(C.load_config()), 8)
        small = C.load_config(overrides={"eval": {"per_benchmark": 3}})
        self.assertEqual(C.sweep_per_benchmark(small), 3)

    def test_a_big_grid_is_refused_instead_of_silently_running_for_days(self):
        cfg = C.load_config(overrides={"merge": {"lambda_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}})
        with self.assertRaisesRegex(C.ConfigError, "max_candidates"):
            C.enumerate_candidates(cfg)
        ok = C.load_config(overrides={"merge": {"lambda_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "max_candidates": 20}})
        self.assertEqual(len(C.enumerate_candidates(ok)), 1 + 8 + 2)

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


class TtaContract(unittest.TestCase):
    """Option-order TTA: the permutations are bijections and the inverse mapping is exact."""

    @classmethod
    def setUpClass(cls):
        from decide.sd import tta

        cls.T = tta

    def test_reverse_mode_orders(self):
        T = self.T
        self.assertEqual(T.permutations(1, 2), [[0]])
        self.assertEqual(T.permutations(2, 2), [[0, 1], [1, 0]])
        self.assertEqual(T.permutations(4, 2), [[0, 1, 2, 3], [3, 2, 1, 0]])
        self.assertEqual(T.permutations(4, 1), [[0, 1, 2, 3]])
        self.assertEqual(T.permutations(5, 3, "reverse"), [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [3, 4, 0, 1, 2]])

    def test_rotate_mode_shifts_are_evenly_spaced(self):
        T = self.T
        self.assertEqual(T.permutations(5, 2, "rotate"), [[0, 1, 2, 3, 4], [2, 3, 4, 0, 1]])
        self.assertEqual(T.permutations(4, 2, "rotate"), [[0, 1, 2, 3], [2, 3, 0, 1]])
        self.assertEqual(T.permutations(2, 2, "rotate"), [[0, 1], [1, 0]])

    def test_views_are_distinct_bijections_starting_with_the_identity(self):
        for mode in self.T.MODES:
            for n in range(0, 10):
                for views in (1, 2, 3, 4):
                    with self.subTest(mode=mode, n=n, views=views):
                        perms = self.T.permutations(n, views, mode)
                        self.assertEqual(perms[0], list(range(n)))
                        self.assertEqual(len({tuple(p) for p in perms}), len(perms))
                        self.assertLessEqual(len(perms), views)
                        for perm in perms:
                            self.assertEqual(sorted(perm), list(range(n)))

    def test_two_options_never_run_more_than_two_views(self):
        self.assertEqual(len(self.T.permutations(2, 5, "reverse")), 2)
        self.assertEqual(len(self.T.permutations(2, 5, "rotate")), 2)

    def test_unpermute_inverts_apply_exactly(self):
        import random

        rng = random.Random(7)
        for mode in self.T.MODES:
            for n in range(1, 10):
                for perm in self.T.permutations(n, 4, mode):
                    x = [rng.random() for _ in range(n)]
                    with self.subTest(mode=mode, n=n, perm=perm):
                        self.assertEqual(self.T.unpermute(self.T.apply(x, perm), perm), x)
                        self.assertEqual(self.T.apply(self.T.unpermute(x, perm), perm), x)
                        self.assertEqual(self.T.apply(x, self.T.invert(perm)), self.T.unpermute(x, perm))

    def test_the_inverse_is_not_the_forward_map_for_a_non_involution(self):
        perm = self.T.rotation(5, 2)              # [2, 3, 4, 0, 1]
        x = [10, 11, 12, 13, 14]
        shown = self.T.apply(x, perm)
        self.assertEqual(shown, [12, 13, 14, 10, 11])
        self.assertEqual(self.T.unpermute(shown, perm), x)
        self.assertNotEqual(self.T.apply(shown, perm), x)

    def test_bad_permutations_and_lengths_are_rejected(self):
        for bad in ([0, 0, 1], [1, 2, 3], [0, 2]):
            with self.assertRaises(ValueError):
                self.T.check(bad)
        with self.assertRaises(ValueError):
            self.T.apply([1, 2, 3], [0, 1])
        with self.assertRaises(ValueError):
            self.T.unpermute([1, 2], [0, 1, 2])
        with self.assertRaises(ValueError):
            self.T.permutations(3, 2, "shuffle")
        with self.assertRaises(ValueError):
            self.T.permutations(3, 0)

    def test_average_is_the_elementwise_mean(self):
        self.assertEqual(self.T.average([[1.0, 0.0], [0.0, 1.0]]), [0.5, 0.5])
        with self.assertRaises(ValueError):
            self.T.average([[1.0], [1.0, 2.0]])
        with self.assertRaises(ValueError):
            self.T.average([])

    def request(self):
        return {
            "state": "s",
            "questions": {
                "c": {"type": "choice", "criteria": {"a": "A", "b": None, "c": "C"}},
                "y": {"type": "noul", "criteria": {"false": "F", "true": "T"}},
                "one": {"type": "choice", "criteria": {"only": "x"}},
            },
        }

    def test_plan_permutes_choice_questions_only(self):
        views = self.T.plan(self.request(), 2, "reverse")
        self.assertEqual(len(views), 2)
        self.assertEqual(views[0], {"c": [0, 1, 2], "y": [], "one": []})
        self.assertEqual(views[1], {"c": [2, 1, 0], "y": [], "one": []})

    def test_permute_request_reorders_criteria_without_touching_the_original(self):
        req = self.request()
        before = repr(req)
        out = self.T.permute_request(req, {"c": [2, 0, 1], "y": [], "one": []})
        self.assertEqual(list(out["questions"]["c"]["criteria"]), ["c", "a", "b"])
        self.assertEqual(out["questions"]["c"]["criteria"]["b"], None)
        self.assertEqual(out["questions"]["y"], req["questions"]["y"])
        self.assertEqual(out["state"], "s")
        self.assertEqual(repr(req), before)

    def test_a_model_without_position_bias_gives_the_same_answer_in_every_view(self):
        """End to end on the request: score by option content, present permuted, map back, average."""
        content = {"a": 0.2, "b": 1.5, "c": -0.7, "d": 0.9, "e": 0.0}
        req = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {k: None for k in content}}}}

        def model(request):                      # probability per *presented* position, from content only
            keys = list(request["questions"]["q"]["criteria"])
            z = [content[k] for k in keys]
            e = [math.exp(v) for v in z]
            return [v / sum(e) for v in e]

        base = model(req)
        for mode in self.T.MODES:
            for views in (2, 3):
                with self.subTest(mode=mode, views=views):
                    rows = []
                    for perms in self.T.plan(req, views, mode):
                        shown = model(self.T.permute_request(req, perms))
                        rows.append(self.T.restore_question(shown, perms["q"]))
                    for row in rows:
                        for got, want in zip(row, base):
                            self.assertAlmostEqual(got, want, places=12)
                    for got, want in zip(self.T.average(rows), base):
                        self.assertAlmostEqual(got, want, places=12)

    def test_averaging_two_orders_cancels_a_pure_position_bias(self):
        """A model that only prefers the first presented option gives a flat answer after reverse-TTA."""
        req = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None, "c": None, "d": None}}}}

        def biased(_request):
            return [0.7, 0.1, 0.1, 0.1]

        rows = [self.T.restore_question(biased(self.T.permute_request(req, p)), p["q"]) for p in self.T.plan(req, 2, "reverse")]
        avg = self.T.average(rows)
        self.assertEqual(rows[0], [0.7, 0.1, 0.1, 0.1])
        self.assertEqual(rows[1], [0.1, 0.1, 0.1, 0.7])
        self.assertAlmostEqual(avg[0], avg[3])
        self.assertAlmostEqual(sum(avg), 1.0)


@needs_numpy
class CalibrateContract(unittest.TestCase):
    """Per-question-type temperature fitting, probability metrics and the holdout protocol."""

    @classmethod
    def setUpClass(cls):
        import numpy as np

        from decide.sd import calibrate

        cls.np, cls.K = np, calibrate

    def synth(self, n, k, temp, qtype="choice", seed=0, views=1, prefix="r"):
        """Items whose gold is drawn from softmax(logits / temp): the best temperature is `temp`."""
        rng = self.np.random.default_rng(seed)
        items = []
        for i in range(n):
            z = rng.normal(0.0, 2.0, size=k)
            p = self.np.exp(z / temp - (z / temp).max())
            p /= p.sum()
            gold = int(rng.choice(k, p=p))
            logits = [z.tolist()] + [(z + rng.normal(0, 0.0, size=k)).tolist() for _ in range(views - 1)]
            items.append(self.K.Item(f"{prefix}{i}", "q", qtype, logits, gold))
        return items

    def test_fit_recovers_a_known_temperature(self):
        for true in (0.7, 1.0, 1.8, 2.6):
            with self.subTest(true=true):
                fitted = self.K.fit_temperature_items(self.synth(6000, 5, true, seed=int(true * 10)))
                self.assertAlmostEqual(fitted, true, delta=0.06 * true)

    def test_fit_by_question_type_separates_the_types(self):
        items = self.synth(5000, 6, 1.8, "choice", seed=1, prefix="c") + self.synth(5000, 2, 0.8, "noul", seed=2, prefix="n")
        out = self.K.fit_groups(items, by="question_type", holdout=0.5, seed=3)
        temps = out["temperatures"]
        self.assertAlmostEqual(temps["choice"], 1.8, delta=0.15)
        self.assertAlmostEqual(temps["noul"], 0.8, delta=0.08)

    def test_the_search_stays_inside_the_grid(self):
        items = self.synth(3000, 4, 6.0, seed=5)
        self.assertLessEqual(self.K.fit_temperature_items(items, grid=(0.5, 3.0)), 3.0 + 1e-9)
        items = self.synth(3000, 4, 0.2, seed=6)
        self.assertGreaterEqual(self.K.fit_temperature_items(items, grid=(0.5, 3.0)), 0.5 - 1e-9)

    def test_nll_is_minimal_at_the_fitted_temperature(self):
        items = self.synth(2500, 4, 1.5, seed=9)
        best = self.K.fit_temperature_items(items)
        floor = self.K.nll(items, best)
        for t in (best * 0.8, best * 1.25, 0.5, 1.0, 3.0):
            self.assertLessEqual(floor, self.K.nll(items, t) + 1e-9)

    def test_softmax_matches_a_reference_and_is_stable(self):
        np = self.np
        z = [1.0, 2.0, 3.0]
        ref = np.exp(np.array(z) / 2.0)
        ref /= ref.sum()
        self.assertTrue(np.allclose(self.K.softmax(z, 2.0), ref))
        big = self.K.softmax([1e4, 1e4 + 1.0], 1.0)
        self.assertTrue(np.isfinite(big).all())
        self.assertAlmostEqual(float(big.sum()), 1.0)
        with self.assertRaises(ValueError):
            self.K.softmax([1.0], 0.0)

    def test_tta_probabilities_are_the_mean_of_the_view_softmaxes(self):
        np = self.np
        item = self.K.Item("r", "q", "choice", [[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]], 0)
        p = self.K.item_probs(item, 1.0)
        a, b = self.K.softmax([2.0, 0.0, 0.0], 1.0), self.K.softmax([0.0, 0.0, 2.0], 1.0)
        self.assertTrue(np.allclose(p, (a + b) / 2))
        self.assertTrue(np.allclose(self.K.item_probs(item, 1.0, views=[0]), a))
        self.assertTrue(np.allclose(self.K.item_probs(item, 1.0, views=[1]), b))

    def test_metrics_on_hand_computed_cases(self):
        sure_right = self.K.Item("a", "q", "choice", [[50.0, 0.0]], 0)
        sure_wrong = self.K.Item("b", "q", "choice", [[50.0, 0.0]], 1)
        m = self.K.metrics([sure_right], 1.0)
        self.assertAlmostEqual(m["nll"], 0.0, places=6)
        self.assertAlmostEqual(m["brier"], 0.0, places=6)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["ece"], 0.0, places=6)
        w = self.K.metrics([sure_wrong], 1.0)
        self.assertEqual(w["accuracy"], 0.0)
        self.assertAlmostEqual(w["brier"], 2.0, places=6)           # (1-0)^2 + (0-1)^2
        self.assertAlmostEqual(w["ece"], 1.0, places=6)             # confidence 1, accuracy 0
        flat = self.K.Item("c", "q", "choice", [[0.0, 0.0, 0.0, 0.0]], 2)
        f = self.K.metrics([flat], 1.0)
        self.assertAlmostEqual(f["nll"], math.log(4), places=9)
        self.assertAlmostEqual(f["brier"], 0.75, places=9)          # 3 * (1/4)^2 + (3/4)^2

    def test_items_without_a_gold_label_are_ignored(self):
        items = self.synth(200, 3, 1.0, seed=2) + [self.K.Item("x", "q", "choice", [[1.0, 2.0, 3.0]], None)]
        self.assertEqual(self.K.metrics(items, 1.0)["n"], 200)

    def test_holdout_split_is_deterministic_disjoint_and_by_request(self):
        items = self.synth(400, 3, 1.0, seed=4)
        items += [self.K.Item(it.run_id, "second", it.qtype, it.logits, it.gold) for it in items[:100]]   # same request, 2nd question
        fit, hold = self.K.split_holdout(items, 0.5, seed=11)
        self.assertEqual(len(fit) + len(hold), len(items))
        self.assertEqual({i.run_id for i in fit} & {i.run_id for i in hold}, set())
        fit2, hold2 = self.K.split_holdout(items, 0.5, seed=11)
        self.assertEqual([(i.run_id, i.question_id) for i in fit], [(i.run_id, i.question_id) for i in fit2])
        self.assertGreater(len(hold), 120)
        other, _ = self.K.split_holdout(items, 0.5, seed=12)
        self.assertNotEqual([i.run_id for i in fit], [i.run_id for i in other])

    def test_holdout_metrics_improve_when_the_model_is_miscalibrated(self):
        items = self.synth(8000, 5, 2.2, seed=7)                        # the model is over-confident at T=1
        out = self.K.fit_groups(items, by="question_type", holdout=0.5, seed=1, baseline=1.0)
        self.assertLess(out["holdout"]["after"]["nll"], out["holdout"]["before"]["nll"])
        self.assertLess(out["holdout"]["after"]["ece"], out["holdout"]["before"]["ece"])
        self.assertEqual(out["n_fit"] + out["n_holdout"], 8000)
        self.assertAlmostEqual(out["temperatures"]["choice"], 2.2, delta=0.25)

    def test_type_k_groups_split_by_option_count(self):
        K = self.K
        a = K.Item("r", "q", "choice", [[0.0, 0.0]], 0)
        b = K.Item("r", "q", "choice", [[0.0] * 3], 0)
        c = K.Item("r", "q", "choice", [[0.0] * 7], 0)
        d = K.Item("r", "q", "choice", [[0.0] * 20], 0)
        self.assertEqual([K.group_of(x, "type_k") for x in (a, b, c, d)], ["choice:2", "choice:3-4", "choice:5-8", "choice:9+"])
        self.assertEqual(K.group_of(a, "question_type"), "choice")
        with self.assertRaises(ValueError):
            K.group_of(a, "nope")

    def test_small_groups_fall_back_to_the_overall_temperature(self):
        items = self.synth(3000, 4, 1.6, "choice", seed=1, prefix="c") + self.synth(5, 2, 0.4, "noul", seed=2, prefix="n")
        out = self.K.fit_groups(items, by="question_type", holdout=0.5, seed=3, min_items=50)
        self.assertEqual(out["temperatures"]["noul"], out["default"])
        self.assertIn("noul", out["fallback"])


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class StateContract(HermeticTestCase):
    """Idempotent stage and row checkpoints, mirrored to a store; a dead session resumes where it stopped."""

    def setUp(self):
        super().setUp()
        from decide.sd import state

        self.S = state
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.clock = FakeClock()

    def open(self, store=None, sub="w", **kw):
        return self.S.State.open(self.dir / sub, store, clock=self.clock, now=self.clock, **kw)

    def test_a_fresh_state_is_empty_and_saved_atomically(self):
        st = self.open()
        self.assertFalse(st.is_done("fidelity"))
        self.assertIsNone(st.stage_status("fidelity"))
        st.save()
        data = json.loads((self.dir / "w" / "state.json").read_text())
        self.assertEqual((data["version"], data["stages"], data["rows"]), (self.S.SCHEMA_VERSION, {}, {}))
        self.assertEqual([p.name for p in (self.dir / "w").iterdir() if p.name.endswith(".tmp")], [])

    def test_run_stage_runs_once_and_returns_the_stored_result(self):
        st, calls = self.open(), []

        def work(x):
            calls.append(x)
            return {"score": x * 2}

        self.assertEqual(st.run_stage("a", work, 3), {"score": 6})
        self.assertEqual(st.run_stage("a", work, 99), {"score": 6})
        self.assertEqual(calls, [3])
        self.assertTrue(st.is_done("a"))
        self.assertEqual(st.run_stage("a", work, 4, force=True), {"score": 8})
        self.assertEqual(calls, [3, 4])

    def test_a_failed_stage_is_recorded_and_runs_again(self):
        st = self.open()

        def boom():
            raise RuntimeError("cuda out of memory")

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            st.run_stage("a", boom)
        self.assertEqual(st.stage_status("a"), "failed")
        self.assertIn("out of memory", st.data["stages"]["a"]["error"])
        self.assertEqual(st.run_stage("a", lambda: "fine"), "fine")
        self.assertEqual((st.stage_status("a"), st.data["stages"]["a"]["attempts"]), ("done", 2))

    def test_state_survives_a_restart_and_a_dead_running_stage_is_redone(self):
        st = self.open()
        st.run_stage("done_one", lambda: 1)
        st.begin("died_midway")                         # the session died here: status stays "running"
        again = self.open()
        self.assertTrue(again.is_done("done_one"))
        self.assertFalse(again.is_done("died_midway"))
        self.assertEqual(again.stage_status("died_midway"), "running")
        self.assertEqual(again.run_stage("died_midway", lambda: 2), 2)
        self.assertEqual(again.data["stages"]["died_midway"]["attempts"], 2)

    def test_rows_are_idempotent(self):
        st = self.open()
        for _ in range(3):
            st.mark_row("eval:anchor", "run-1", {"ms": 5})
        st.mark_row("eval:anchor", "run-2")
        self.assertEqual(st.rows("eval:anchor"), {"run-1": {"ms": 5}, "run-2": True})
        self.assertTrue(st.row_done("eval:anchor", "run-1"))
        self.assertFalse(st.row_done("eval:anchor", "run-3"))
        st.save()
        self.assertEqual(self.open().rows("eval:anchor"), st.rows("eval:anchor"))

    def test_a_truncated_state_file_is_moved_aside_and_the_store_copy_is_used(self):
        store = self.S.MemoryStore()
        st = self.open(store)
        st.run_stage("a", lambda: 1)
        st.push()
        (self.dir / "w" / "state.json").write_text('{"version": 1, "stag')
        again = self.open(store)
        self.assertTrue(again.is_done("a"))
        self.assertTrue((self.dir / "w" / "state.json.corrupt").exists())

    def test_a_truncated_state_file_without_a_store_starts_fresh_and_says_so(self):
        self.open().save()
        (self.dir / "w" / "state.json").write_text("{")
        again = self.open()
        self.assertEqual(again.data["stages"], {})
        self.assertIn("corrupt", " ".join(again.data["notes"]))

    def test_push_is_throttled_but_stage_end_always_pushes(self):
        store = self.S.MemoryStore()
        st = self.open(store, min_push_seconds=600)
        st.save()
        first = store.writes
        st.mark_row("s", "k1")
        st.save()                                      # inside the interval: local only
        self.assertEqual(store.writes, first)
        self.clock.advance(601)
        st.mark_row("s", "k2")
        st.save()
        self.assertEqual(store.writes, first + 1)
        st.run_stage("quick", lambda: 1)               # finishing a stage forces a push
        self.assertEqual(store.writes, first + 2)
        self.assertIn("quick", json.loads(store.read("state.json"))["stages"])

    def test_a_push_failure_never_kills_the_run(self):
        class Flaky(self.S.MemoryStore):
            fail = True

            def write(self, name, data):
                if self.fail:
                    raise OSError("503 from the Hub")
                super().write(name, data)

        store = Flaky()
        st = self.open(store)
        self.assertEqual(st.run_stage("a", lambda: 1), 1)
        self.assertIn("503", st.push_error)
        store.fail = False
        st.push()
        self.assertIsNone(st.push_error)
        self.assertIn("a", json.loads(store.read("state.json"))["stages"])

    def test_open_merges_local_and_remote_progress(self):
        store = self.S.MemoryStore()
        other = self.S.State.open(self.dir / "colab", store, clock=self.clock, now=self.clock)
        other.run_stage("merge:ta_l0.25", lambda: "from colab")
        other.mark_row("eval:anchor", "r1", 1)
        other.push()
        mine = self.open()                              # no store yet: local only
        mine.run_stage("fidelity", lambda: "from kaggle")
        mine.mark_row("eval:anchor", "r2", 2)
        mine.save()
        merged = self.open(store)
        self.assertTrue(merged.is_done("merge:ta_l0.25"))
        self.assertTrue(merged.is_done("fidelity"))
        self.assertEqual(merged.rows("eval:anchor"), {"r1": 1, "r2": 2})
        self.assertEqual(merged.result("merge:ta_l0.25"), "from colab")

    def test_done_beats_running_when_merging(self):
        a = {"version": 1, "seq": 3, "stages": {"x": {"status": "running", "attempts": 1}}, "rows": {}, "meta": {}, "files": {}, "notes": []}
        b = {"version": 1, "seq": 1, "stages": {"x": {"status": "done", "attempts": 1, "result": 5, "finished": 10.0}}, "rows": {}, "meta": {}, "files": {}, "notes": []}
        self.assertEqual(self.S.merge_states(a, b)["stages"]["x"]["status"], "done")
        self.assertEqual(self.S.merge_states(b, a)["stages"]["x"]["status"], "done")

    def test_tracked_files_are_mirrored_and_restored_after_a_dead_session(self):
        store = self.S.MemoryStore()
        st = self.open(store)
        results = self.dir / "w" / "runs" / "anchor" / "results.jsonl"
        results.parent.mkdir(parents=True)
        results.write_text('{"run_id": "a"}\n{"run_id": "b"}\n')
        st.track_file(results, "runs/anchor/results.jsonl")
        st.push(files_now=True)
        self.assertEqual(store.read("runs/anchor/results.jsonl"), results.read_bytes())
        writes = store.writes
        st.push(files_now=True)
        self.assertEqual(store.writes, writes + 1)      # only state.json again: the unchanged file is skipped
        # new session, empty disk
        fresh = self.S.State.open(self.dir / "fresh", store, clock=self.clock, now=self.clock)
        restored = fresh.restore_files()
        local = self.dir / "fresh" / "runs" / "anchor" / "results.jsonl"
        self.assertEqual(restored, ["runs/anchor/results.jsonl"])
        self.assertEqual(local.read_text(), results.read_text())

    def test_restore_keeps_a_longer_local_log(self):
        store = self.S.MemoryStore()
        st = self.open(store)
        log = self.dir / "w" / "results.jsonl"
        log.write_text("1\n")
        st.track_file(log, "results.jsonl")
        st.push(files_now=True)
        log.write_text("1\n2\n3\n")                    # the session kept going after the last push
        self.assertEqual(st.restore_files(), [])
        self.assertEqual(log.read_text(), "1\n2\n3\n")

    def test_tokens_are_redacted_from_recorded_errors(self):
        store = self.S.MemoryStore()
        st = self.open(store)

        def boom():
            raise RuntimeError("401 Client Error for Authorization: Bearer hf_SECRET_TOKEN_VALUE (token hf_SECRET_TOKEN_VALUE)")

        with self.assertRaises(RuntimeError):
            st.run_stage("a", boom)
        for text in ((self.dir / "w" / "state.json").read_text(), store.read("state.json").decode()):
            self.assertNotIn("hf_SECRET", text)
            self.assertIn("redacted", text)
        self.assertEqual(self.S.redact("x hf_abcdefgh12345 y"), "x hf_[redacted] y")
        self.assertNotIn("hf_", repr(self.S.HfDatasetStore("u/d", "hf_SECRET_TOKEN_VALUE")))

    def test_from_env_has_no_store_without_a_token(self):
        self.assertIsNone(self.S.store_from_env("user/dataset", environ={}))
        store = self.S.store_from_env("user/dataset", environ={"HF_TOKEN": "hf_x"})
        self.assertIsInstance(store, self.S.HfDatasetStore)
        self.assertEqual(store.repo_id, "user/dataset")

    def test_hf_dataset_store_uses_the_hub_api_for_a_private_dataset(self):
        import types

        calls = []

        class FakeErrors:
            class EntryNotFoundError(Exception):
                pass

            class RepositoryNotFoundError(Exception):
                pass

            class RevisionNotFoundError(Exception):
                pass

        class FakeApi:
            def __init__(self, token=None, endpoint=None):
                calls.append(("api", token, endpoint))

            def create_repo(self, repo_id, **kw):
                calls.append(("create", repo_id, kw))

            def upload_file(self, **kw):
                calls.append(("upload", kw))

            def list_repo_files(self, repo_id, repo_type=None):
                calls.append(("list", repo_id, repo_type))
                return ["state.json", "runs/a.jsonl", "other.txt"]

        served = {"state.json": b"{}"}

        def hf_hub_download(repo_id, filename, **kw):
            calls.append(("download", repo_id, filename, kw))
            if filename not in served:
                raise FakeErrors.EntryNotFoundError(filename)
            target = self.dir / "dl.bin"
            target.write_bytes(served[filename])
            return str(target)

        fake = types.SimpleNamespace(HfApi=FakeApi, hf_hub_download=hf_hub_download, errors=FakeErrors)
        store = self.S.HfDatasetStore("u/state", "hf_tok_for_test_only", endpoint="http://hub.test")
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake}):
            self.assertEqual(store.read("state.json"), b"{}")
            self.assertIsNone(store.read("missing.json"))
            store.write("state.json", b"[1]")
            store.write("runs/a.jsonl", b"x")
            self.assertEqual(store.list("runs/"), ["runs/a.jsonl"])
        creates = [c for c in calls if c[0] == "create"]
        self.assertEqual(len(creates), 1)                                  # the repo is created once, private
        self.assertEqual((creates[0][1], creates[0][2]["repo_type"], creates[0][2]["private"], creates[0][2]["exist_ok"]), ("u/state", "dataset", True, True))
        uploads = [c[1] for c in calls if c[0] == "upload"]
        self.assertEqual([(u["path_in_repo"], u["repo_type"], u["repo_id"], u["path_or_fileobj"]) for u in uploads],
                         [("state.json", "dataset", "u/state", b"[1]"), ("runs/a.jsonl", "dataset", "u/state", b"x")])
        downloads = [c for c in calls if c[0] == "download"]
        self.assertTrue(all(d[3]["repo_type"] == "dataset" and d[3]["force_download"] and d[3]["endpoint"] == "http://hub.test" for d in downloads))

    def test_hf_dataset_store_refuses_to_exist_without_a_token(self):
        with self.assertRaises(self.S.StateError):
            self.S.HfDatasetStore("u/state", "")

    def test_dir_store_round_trip_and_listing(self):
        store = self.S.DirStore(self.dir / "remote")
        self.assertIsNone(store.read("a/b.json"))
        store.write("a/b.json", b"{}")
        store.write("a/c.json", b"[]")
        self.assertEqual(store.read("a/b.json"), b"{}")
        self.assertEqual(store.list("a/"), ["a/b.json", "a/c.json"])
        with self.assertRaises(ValueError):
            store.write("../escape", b"x")


def synthetic_rows(benchmarks=(2, 5, 24), groups=30, rows_per_group=(1, 3), tokens=200, long_every=0):
    """A small stand-in for the suite: `groups` groups per benchmark, each with a few rows."""
    out = []
    for b in benchmarks:
        for g in range(groups):
            n = rows_per_group[0] + (g % (rows_per_group[1] - rows_per_group[0] + 1))
            for i in range(n):
                t = tokens + 10 * i
                if long_every and g % long_every == 0 and i == 0:
                    t = 100_000
                out.append({"_evaluation": {"run_id": f"b{b}:g{g}:r{i}", "catalog_id": b, "group_id": f"g{g}", "track": "all", "proxy_tokens": t},
                            "state": f"state {b} {g}", "questions": {"q": {"type": "choice", "criteria": {"a": "x", "b": "y"}}},
                            "expected": {"q": "a"}})
    return out


class SampleContract(HermeticTestCase):
    def setUp(self):
        super().setUp()
        from decide.sd import sample

        self.S = sample
        self.rows = synthetic_rows()

    def groups(self, sel):
        return sorted({(r["_evaluation"]["catalog_id"], r["_evaluation"]["group_id"]) for r in sel.rows})

    def test_quota_per_benchmark_and_complete_groups(self):
        sel = self.S.select(self.rows, per_benchmark=5, seed=1)
        per = {}
        for b, g in self.groups(sel):
            per[b] = per.get(b, 0) + 1
        self.assertEqual(per, {2: 5, 5: 5, 24: 5})
        for b, g in self.groups(sel):                       # every row of a chosen group is there
            want = {r["_evaluation"]["run_id"] for r in self.rows if r["_evaluation"]["catalog_id"] == b and r["_evaluation"]["group_id"] == g}
            got = {r["_evaluation"]["run_id"] for r in sel.rows if r["_evaluation"]["catalog_id"] == b and r["_evaluation"]["group_id"] == g}
            self.assertEqual(got, want)

    def test_selection_is_deterministic_and_independent_of_input_order(self):
        a = self.S.select(self.rows, per_benchmark=7, seed=11)
        shuffled = list(reversed(self.rows))
        b = self.S.select(shuffled, per_benchmark=7, seed=11)
        self.assertEqual(self.groups(a), self.groups(b))
        self.assertEqual(a.manifest["run_ids_sha256"], b.manifest["run_ids_sha256"])
        self.assertEqual([r["_evaluation"]["run_id"] for r in a.rows], [r["_evaluation"]["run_id"] for r in b.rows])
        c = self.S.select(self.rows, per_benchmark=7, seed=12)
        self.assertNotEqual(self.groups(a), self.groups(c))

    def test_a_smaller_sample_is_a_prefix_of_a_bigger_one(self):
        big = self.S.select(self.rows, per_benchmark=16, seed=3)
        small = self.S.select(self.rows, per_benchmark=6, seed=3)
        self.assertTrue(set(self.groups(small)) <= set(self.groups(big)))
        shrunk = self.S.shrink(big.rows, per_benchmark=6, seed=3)
        self.assertEqual([r["_evaluation"]["run_id"] for r in shrunk], [r["_evaluation"]["run_id"] for r in small.rows])

    def test_groups_with_a_long_row_are_dropped_whole_and_counted(self):
        rows = synthetic_rows(long_every=3)                   # groups 0, 3, 6, ... have one 100k-token row
        sel = self.S.select(rows, per_benchmark=30, seed=5, max_proxy_tokens=6000)
        chosen = {g for _b, g in self.groups(sel)}
        self.assertFalse(any(int(g[1:]) % 3 == 0 for g in chosen))
        self.assertEqual(max(r["_evaluation"]["proxy_tokens"] for r in sel.rows) <= 6000, True)
        m = sel.manifest["benchmarks"]["2"]
        self.assertEqual((m["available_groups"], m["excluded_long_groups"], m["groups"]), (30, 10, 20))
        self.assertEqual(sel.manifest["totals"]["excluded_long_groups"], 30)
        self.assertEqual(sel.manifest["max_proxy_tokens"], 6000)

    def test_excluded_long_groups_free_their_slot_for_the_next_group(self):
        rows = synthetic_rows(long_every=2)
        sel = self.S.select(rows, per_benchmark=5, seed=9, max_proxy_tokens=6000)
        self.assertEqual(sel.manifest["benchmarks"]["5"]["groups"], 5)

    def test_short_benchmarks_take_everything_and_say_so(self):
        rows = synthetic_rows(groups=3)
        sel = self.S.select(rows, per_benchmark=8, seed=1)
        self.assertEqual(sel.manifest["benchmarks"]["2"]["groups"], 3)
        self.assertTrue(sel.manifest["benchmarks"]["2"]["short"])
        full = self.S.select(synthetic_rows(groups=10), per_benchmark=8, seed=1)
        self.assertFalse(full.manifest["benchmarks"]["2"]["short"])

    def test_ranking_matches_the_kit_so_a_published_sample_can_be_reproduced(self):
        import hashlib

        key = self.S.rank_key(20261021, 24, "g7")
        self.assertEqual(key, hashlib.sha256(b"20261021:24:g7").hexdigest())

    @needs_kit
    def test_group_order_equals_the_scoring_kits_stratified_sample(self):
        from decision_index.suite.io import read_jsonl
        from decision_index.suite.sample import stratified_sample

        class Fake:
            def rows(self_inner, apply_exclusions=False):
                return iter(self.rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "kit.jsonl"
            stratified_sample(Fake(), 10**9, out, seed=20261021)
            order = {}
            for row in read_jsonl(out):
                e = row["_evaluation"]
                seen = order.setdefault(e["catalog_id"], [])
                if e["group_id"] not in seen:
                    seen.append(e["group_id"])
        mine = self.S.select(self.rows, per_benchmark=8, seed=20261021)
        for benchmark, groups in order.items():
            got = []
            for r in mine.rows:
                e = r["_evaluation"]
                if e["catalog_id"] == benchmark and e["group_id"] not in got:
                    got.append(e["group_id"])
            self.assertEqual(got, groups[:8])

    def test_manifest_describes_the_sample(self):
        sel = self.S.select(self.rows, per_benchmark=4, seed=2, edition="0.3")
        m = sel.manifest
        self.assertEqual((m["edition"], m["seed"], m["per_benchmark"]), ("0.3", 2, 4))
        self.assertEqual(m["totals"]["rows"], len(sel.rows))
        self.assertEqual(m["totals"]["groups"], 12)
        self.assertEqual(m["totals"]["tokens"], sum(r["_evaluation"]["proxy_tokens"] for r in sel.rows))
        self.assertEqual(sum(b["rows"] for b in m["benchmarks"].values()), len(sel.rows))
        json.dumps(m, allow_nan=False)

    def test_write_and_load_round_trip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            sel = self.S.select(self.rows, per_benchmark=4, seed=2)
            paths = self.S.write(sel, tmp)
            rows, manifest = self.S.load(tmp)
            self.assertEqual([r["_evaluation"]["run_id"] for r in rows], [r["_evaluation"]["run_id"] for r in sel.rows])
            self.assertEqual(manifest, sel.manifest)
            import gzip

            with gzip.open(paths["rows"], "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(sel.rows[0]) + "\n")
            with self.assertRaisesRegex(self.S.SampleError, "does not match"):
                self.S.load(tmp)

    def test_fit_to_budget_picks_the_largest_sample_that_fits(self):
        rows = self.S.select(self.rows, per_benchmark=20, seed=4).rows
        full = sum(r["_evaluation"]["proxy_tokens"] for r in rows)
        plan = self.S.fit(rows, seed=4, token_budget=full, minimum=2)
        self.assertEqual((plan["per_benchmark"], plan["fits"]), (20, True))
        half = self.S.fit(rows, seed=4, token_budget=full / 2, minimum=2)
        self.assertLess(half["per_benchmark"], 20)
        self.assertLessEqual(half["tokens"], full / 2)
        again = self.S.fit(rows, seed=4, token_budget=full / 2 + 1, minimum=2)
        self.assertGreaterEqual(again["per_benchmark"], half["per_benchmark"])
        tiny = self.S.fit(rows, seed=4, token_budget=1, minimum=3)
        self.assertEqual((tiny["per_benchmark"], tiny["fits"]), (3, False))       # the floor wins, and it says so

    def test_sample_suite_restricts_the_rows_a_scorer_sees(self):
        class Fake:
            edition = {"id": "0.3"}

            def __init__(self, rows):
                self._rows = rows

            def rows(self, apply_exclusions=False):
                return iter(self._rows)

        sel = self.S.select(self.rows, per_benchmark=3, seed=1)
        view = self.S.SampleSuite(Fake(self.rows), {r["_evaluation"]["run_id"] for r in sel.rows})
        self.assertEqual(view.edition, {"id": "0.3"})
        self.assertEqual({r["_evaluation"]["run_id"] for r in view.rows(apply_exclusions=True)}, {r["_evaluation"]["run_id"] for r in sel.rows})


@needs_torch
class Nf4Contract(HermeticTestCase):
    """The NF4 weight format: same bytes as bitsandbytes, bounded error, a Linear that dequantises on the fly."""

    @classmethod
    def setUpClass(cls):
        import torch

        from decide.sd import nf4

        cls.torch, cls.N = torch, nf4

    def weight(self, rows=48, cols=128, seed=0, scale=0.05):
        g = self.torch.Generator().manual_seed(seed)
        return self.torch.randn(rows, cols, generator=g) * scale

    def test_code_table_is_the_16_point_normal_float(self):
        code = self.N.NF4_CODE
        self.assertEqual(len(code), 16)
        self.assertEqual(list(code), sorted(code))
        self.assertEqual((code[0], code[7], code[15]), (-1.0, 0.0, 1.0))

    @needs_bnb
    def test_code_table_matches_bitsandbytes(self):
        import bitsandbytes.functional as bf

        self.assertEqual([round(float(x), 9) for x in bf.get_4bit_type("nf4", device="cpu")], [round(x, 9) for x in self.N.NF4_CODE])

    def test_round_trip_error_is_bounded_by_half_the_widest_code_gap(self):
        w = self.weight(64, 192)
        packed, absmax, shape = self.N.quantize_nf4(w)
        back = self.N.dequantize_nf4(packed, absmax, shape, dtype=self.torch.float32)
        gap = max(b - a for a, b in zip(self.N.NF4_CODE, self.N.NF4_CODE[1:]))
        per_block = w.reshape(-1, 64).abs().amax(dim=1)
        err = (back - w).reshape(-1, 64).abs().amax(dim=1)
        self.assertTrue(bool((err <= per_block * gap / 2 + 1e-7).all()))
        self.assertLess(float((back - w).abs().mean()), 0.12 * float(w.abs().mean()) + 1e-3)

    def test_the_largest_value_of_each_block_is_exact(self):
        w = self.weight(8, 64)
        packed, absmax, shape = self.N.quantize_nf4(w)
        back = self.N.dequantize_nf4(packed, absmax, shape, dtype=self.torch.float32)
        for i in range(8):
            j = int(w[i].abs().argmax())
            self.assertAlmostEqual(float(back[i, j]), float(w[i, j]), places=6)

    def test_quantising_a_dequantised_weight_changes_nothing(self):
        w = self.weight(32, 64, seed=3)
        p1, a1, shape = self.N.quantize_nf4(w)
        p2, a2, _ = self.N.quantize_nf4(self.N.dequantize_nf4(p1, a1, shape, dtype=self.torch.float32))
        self.assertTrue(self.torch.equal(p1, p2))
        self.assertTrue(self.torch.allclose(a1, a2))

    def test_pack_layout_first_value_in_the_high_nibble(self):
        w = self.torch.zeros(1, 64)
        w[0, 0], w[0, 1] = -1.0, 1.0                          # codes 0 and 15
        packed, absmax, _ = self.N.quantize_nf4(w)
        self.assertEqual(int(packed[0]), 0x0F)
        self.assertEqual(float(absmax[0]), 1.0)
        self.assertEqual(int(packed[1]), 0x77)                # zeros are code 7 in both nibbles

    def test_all_zero_blocks_stay_zero(self):
        packed, absmax, shape = self.N.quantize_nf4(self.torch.zeros(2, 64))
        back = self.N.dequantize_nf4(packed, absmax, shape, dtype=self.torch.float32)
        self.assertTrue(bool((back == 0).all()))
        self.assertTrue(bool(self.torch.isfinite(back).all()))

    def test_sizes_that_are_not_a_multiple_of_the_block_are_padded(self):
        w = self.weight(3, 50, seed=5)                         # 150 values -> 3 blocks of 64
        packed, absmax, shape = self.N.quantize_nf4(w)
        self.assertEqual((packed.numel(), absmax.numel()), (96, 3))
        back = self.N.dequantize_nf4(packed, absmax, shape, dtype=self.torch.float32)
        self.assertEqual(tuple(back.shape), (3, 50))
        self.assertLess(float((back - w).abs().max()), float(w.abs().max()) * 0.16)

    def test_memory_is_half_a_byte_per_weight_plus_the_block_statistics(self):
        w = self.weight(128, 256)
        packed, absmax, _ = self.N.quantize_nf4(w)
        self.assertEqual(packed.numel() * packed.element_size(), w.numel() // 2)
        self.assertEqual(absmax.numel(), w.numel() // 64)
        _, half, _ = self.N.quantize_nf4(w, absmax_dtype=self.torch.float16)
        self.assertEqual(half.dtype, self.torch.float16)

    def test_chunked_quantisation_gives_the_same_bytes(self):
        w = self.weight(40, 256, seed=7)
        whole = self.N.quantize_nf4(w)
        old = self.N.CHUNK_ELEMENTS
        self.N.CHUNK_ELEMENTS = 64 * 3
        try:
            chunked = self.N.quantize_nf4(w)
        finally:
            self.N.CHUNK_ELEMENTS = old
        self.assertTrue(self.torch.equal(whole[0], chunked[0]))
        self.assertTrue(self.torch.equal(whole[1], chunked[1]))

    @needs_bnb
    def test_bytes_and_values_are_identical_to_bitsandbytes(self):
        import bitsandbytes.functional as bf

        w = self.weight(32, 128, seed=11)
        packed, absmax, shape = self.N.quantize_nf4(w)
        q, state = bf.quantize_4bit(w, blocksize=64, compress_statistics=False, quant_type="nf4")
        self.assertTrue(self.torch.equal(packed.reshape(-1, 1), q))
        self.assertTrue(self.torch.equal(absmax, state.absmax))
        ours = self.N.dequantize_nf4(packed, absmax, shape, dtype=self.torch.float32)
        self.assertTrue(self.torch.allclose(ours, bf.dequantize_4bit(q, state), atol=0, rtol=0))

    def test_linear_forward_equals_a_dense_linear_on_the_dequantised_weight(self):
        w = self.weight(24, 128, seed=2)
        bias = self.torch.randn(24) * 0.1
        layer = self.N.Nf4Linear.from_weight(w, bias, compute_dtype=self.torch.float32)
        x = self.torch.randn(5, 7, 128)
        dense = self.torch.nn.functional.linear(x, layer.dequantized(self.torch.float32), bias)
        self.assertTrue(self.torch.allclose(layer(x), dense, atol=1e-5))
        plain = self.torch.nn.functional.linear(x, w, bias)
        self.assertLess(float((layer(x) - plain).abs().mean()), 0.12 * float(plain.abs().mean()) + 1e-3)

    def test_linear_runs_in_the_compute_dtype(self):
        layer = self.N.Nf4Linear.from_weight(self.weight(8, 64), compute_dtype=self.torch.bfloat16)
        out = layer(self.torch.randn(2, 64))
        self.assertEqual(out.dtype, self.torch.bfloat16)

    def test_linear_state_dict_round_trip(self):
        a = self.N.Nf4Linear.from_weight(self.weight(16, 64, seed=4), compute_dtype=self.torch.float32)
        b = self.N.Nf4Linear(64, 16, compute_dtype=self.torch.float32)
        b.load_state_dict(a.state_dict())
        x = self.torch.randn(3, 64)
        self.assertTrue(self.torch.equal(a(x), b(x)))

    def test_set_weight_refuses_a_wrong_shape(self):
        layer = self.N.Nf4Linear(64, 8)
        with self.assertRaises(ValueError):
            layer.set_weight(self.torch.zeros(9, 64))


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
