"""Contract tests for decide/sd/api.py (issue #65, F0 kit core): the interface Squad B and the notebooks call.

Everything here is pure (no download, no model).  The tests that build `LoadOptions` need torch and skip without it.
"""
from __future__ import annotations

import copy
import unittest

from decide_support import HermeticTestCase, needs_torch

from decide.sd import api, env
from decide.sd import config as C


def make_cfg(**overrides):
    """The shipped config, read alone (no overlays from other squads), with `overrides` deep-merged on top."""
    return C.load_config(C.DEFAULT_PATH, overrides=overrides or None)


def hardware(*gpus_gb, ram=29.0):
    names = tuple("Tesla T4" for _ in gpus_gb)
    return env.Hardware("kaggle", "/tmp/work", names, tuple(gpus_gb), ram, 50.0, tuple((7, 5) for _ in gpus_gb))


class NormalizeSpecContract(HermeticTestCase):
    def test_a_bare_anchor_gets_every_default(self):
        spec = api.normalize_spec({"anchor": "org/a"})
        self.assertEqual(spec, {"anchor": "org/a", "anchor_revision": None, "donors": [], "lora": None, "method": "anchor",
                                "density": None, "protocol": "pplx_joint", "reference": "anchor", "base": None, "name": None})

    def test_the_input_is_not_mutated(self):
        raw = {"anchor": "org/a", "donors": [{"repo": "org/d", "weight": 0.5}], "lora": {"repo": "org/l", "weight": 0.1}}
        before = copy.deepcopy(raw)
        api.normalize_spec(raw)
        self.assertEqual(raw, before)

    def test_donors_and_lora_are_carried_with_their_weights(self):
        spec = api.normalize_spec({"anchor": "org/a", "donors": [{"repo": "org/d", "weight": 0.5, "revision": "abc"}, {"repo": "org/e", "weight": 0.25}],
                                   "lora": {"repo": "org/l", "weight": 0.1}})
        self.assertEqual(spec["donors"], [{"repo": "org/d", "weight": 0.5, "revision": "abc"}, {"repo": "org/e", "weight": 0.25, "revision": None}])
        self.assertEqual(spec["lora"], {"repo": "org/l", "weight": 0.1, "revision": None})
        self.assertEqual(spec["method"], "task_arithmetic")

    def test_donors_with_weight_zero_are_dropped_and_nothing_left_means_the_anchor_alone(self):
        spec = api.normalize_spec({"anchor": "org/a", "donors": [{"repo": "org/d", "weight": 0}], "lora": {"repo": "org/l", "weight": 0}, "method": "ties"})
        self.assertEqual((spec["donors"], spec["lora"], spec["method"], spec["density"]), ([], None, "anchor", None))

    def test_ties_gets_a_density(self):
        base = {"anchor": "org/a", "donors": [{"repo": "org/d", "weight": 1.0}], "method": "ties"}
        self.assertEqual(api.normalize_spec(base)["density"], 0.5)
        self.assertEqual(api.normalize_spec({**base, "density": 0.2})["density"], 0.2)
        for bad in (0, 0.0, -0.1, 1.5, "half"):
            with self.subTest(bad), self.assertRaisesRegex(api.ApiError, "density"):
                api.normalize_spec({**base, "density": bad})

    def test_task_arithmetic_does_not_carry_a_density(self):
        spec = api.normalize_spec({"anchor": "org/a", "donors": [{"repo": "org/d", "weight": 1.0}], "density": 0.3})
        self.assertIsNone(spec["density"])

    def test_typos_are_errors_that_name_the_key(self):
        with self.assertRaisesRegex(api.ApiError, "donor"):
            api.normalize_spec({"anchor": "org/a", "donor": []})

    def test_the_anchor_is_required(self):
        for bad in ({}, {"anchor": ""}, {"anchor": 3}, "org/a", None):
            with self.subTest(bad), self.assertRaises(api.ApiError):
                api.normalize_spec(bad)

    def test_donor_mistakes_are_named(self):
        cases = (
            ({"donors": [{"weight": 1.0}]}, "repo"),
            ({"donors": [{"repo": "org/d"}]}, "weight"),
            ({"donors": [{"repo": "org/d", "weight": -1}]}, "weight"),
            ({"donors": [{"repo": "org/d", "weight": "x"}]}, "weight"),
            ({"donors": [{"repo": "org/d", "weight": 1}, {"repo": "org/d", "weight": 2}]}, "twice"),
            ({"donors": "org/d"}, "donors"),
            ({"lora": {"weight": 0.1}}, "repo"),
            ({"lora": {"repo": "org/l"}}, "weight"),
            ({"lora": {"repo": "org/l", "weight": -0.1}}, "weight"),
            ({"method": "average"}, "method"),
            ({"protocol": "mystery"}, "protocol"),
            ({"reference": "moon"}, "reference"),
        )
        for extra, word in cases:
            with self.subTest(extra), self.assertRaisesRegex(api.ApiError, word):
                api.normalize_spec({"anchor": "org/a", **extra})

    def test_licence_and_head_exclusions_apply_to_hand_written_specs_too(self):
        for repo in ("someone/Lychee-27B", "someone/amos-decider"):
            with self.subTest(repo), self.assertRaisesRegex(api.ApiError, "not allowed"):
                api.normalize_spec({"anchor": "org/a", "donors": [{"repo": repo, "weight": 1.0}]})

    def test_reference_base_takes_the_base_from_the_config(self):
        cfg = make_cfg()
        spec = api.normalize_spec({"anchor": "org/a", "reference": "base"}, cfg)
        self.assertEqual(spec["base"], {"repo": cfg.models.base.repo, "revision": cfg.models.base.revision})
        own = api.normalize_spec({"anchor": "org/a", "reference": "base", "base": {"repo": "org/b", "revision": "r1"}}, cfg)
        self.assertEqual(own["base"], {"repo": "org/b", "revision": "r1"})
        with self.assertRaisesRegex(api.ApiError, "base"):
            api.normalize_spec({"anchor": "org/a", "reference": "base"})                # no config, no base

    def test_a_normalised_spec_is_stable(self):
        once = api.normalize_spec({"anchor": "org/a", "donors": [{"repo": "org/d", "weight": 1}], "lora": {"repo": "org/l", "weight": 0.1}})
        self.assertEqual(api.normalize_spec(once), once)


class SpecForContract(HermeticTestCase):
    def test_anchor_is_the_pplx_checkpoint_alone(self):
        cfg = make_cfg()
        spec = api.spec_for(cfg, "anchor")
        self.assertEqual((spec["anchor"], spec["donors"], spec["lora"], spec["protocol"]), (cfg.models.anchor.repo, [], None, "pplx_joint"))
        self.assertEqual(spec["name"], "anchor")

    def test_a_merge_candidate_scales_the_donor_weights_by_lambda(self):
        cfg = make_cfg()
        spec = api.spec_for(cfg, "ta_l0.25")
        self.assertEqual([(d["repo"], d["weight"]) for d in spec["donors"]], [("vllm-sr/d3", 0.25)])
        self.assertEqual((spec["method"], spec["lora"]), ("task_arithmetic", None))

    def test_two_donors_split_lambda_in_proportion_to_their_weights(self):
        donors = [{"name": "d3", "repo": "vllm-sr/d3", "weight": 1.0, "enabled": True, "revision": None},
                  {"name": "d3-turbo", "repo": "vllm-sr/d3-turbo", "weight": 0.5, "enabled": True, "revision": None}]
        cfg = make_cfg(models={"donors": donors}, merge={"max_candidates": 8})
        weights = {d["repo"]: d["weight"] for d in api.spec_for(cfg, "ta_l0.5")["donors"]}
        self.assertAlmostEqual(weights["vllm-sr/d3"], 0.5 * 1.0 / 1.5, places=9)
        self.assertAlmostEqual(weights["vllm-sr/d3-turbo"], 0.5 * 0.5 / 1.5, places=9)

    def test_lora_candidates_carry_the_adapter_and_its_weight(self):
        cfg = make_cfg()
        spec = api.spec_for(cfg, "anchor+lora0.1")
        self.assertEqual(spec["lora"], {"repo": cfg.models.lora.repo, "weight": 0.1, "revision": None})
        self.assertEqual(spec["donors"], [])

    def test_a_merge_plus_lora_name_works_for_any_weight(self):
        cfg = make_cfg()
        spec = api.spec_for(cfg, "ta_l0.25+lora0.3")
        self.assertEqual((len(spec["donors"]), spec["lora"]["weight"]), (1, 0.3))

    def test_ties_candidates_carry_the_density(self):
        cfg = make_cfg(merge={"method": "ties", "ties": {"density": 0.4}})
        names = [c.name for c in C.enumerate_candidates(cfg)]
        tie = next(n for n in names if n.startswith("ties_"))
        spec = api.spec_for(cfg, tie)
        self.assertEqual((spec["method"], spec["density"]), ("ties", 0.4))

    def test_the_4b_rehearsal_models_and_the_d3_fidelity_model_have_their_own_protocols(self):
        cfg = make_cfg()
        flash = api.spec_for(cfg, "rehearsal:pplx_flash")
        mini = api.spec_for(cfg, "rehearsal:d3_mini")
        big = api.spec_for(cfg, "d3")
        self.assertEqual((flash["anchor"], flash["protocol"]), ("perplexity-ai/pplx-decider-flash-v1.2", "pplx_joint"))
        self.assertEqual((mini["anchor"], mini["protocol"]), ("vllm-sr/d3-mini", "d3"))
        self.assertEqual((big["anchor"], big["protocol"]), ("vllm-sr/d3", "d3"))
        for spec in (flash, mini, big):
            self.assertEqual((spec["donors"], spec["lora"]), ([], None))

    def test_unknown_names_list_the_known_ones(self):
        with self.assertRaises(api.ApiError) as ctx:
            api.spec_for(make_cfg(), "magic")
        text = str(ctx.exception)
        for known in ("anchor", "ta_l0.25", "anchor+lora0.1", "rehearsal:d3_mini", "d3"):
            self.assertIn(known, text)

    def test_every_candidate_of_the_config_makes_a_valid_spec(self):
        cfg = make_cfg()
        for candidate in C.enumerate_candidates(cfg):
            with self.subTest(candidate.name):
                spec = api.spec_for(cfg, candidate.name)
                self.assertEqual(api.normalize_spec(spec, cfg), spec)
                self.assertEqual(spec["lora"] is not None, candidate.lora_weight > 0)
                self.assertAlmostEqual(sum(d["weight"] for d in spec["donors"]), candidate.lam, places=9)

    def test_revisions_in_the_config_reach_the_spec(self):
        cfg = make_cfg(models={"anchor": {"revision": "a" * 40}})
        self.assertEqual(api.spec_for(cfg, "anchor")["anchor_revision"], "a" * 40)

    def test_a_disabled_lora_is_not_a_candidate(self):
        cfg = make_cfg(models={"lora": {"enabled": False}})
        with self.assertRaises(api.ApiError):
            api.spec_for(cfg, "anchor+lora0.1")


class ParseContract(unittest.TestCase):
    def test_devices(self):
        cases = {"auto": "auto", "cpu": ("cpu",), "cuda": ("cuda:0",), "cuda:1": ("cuda:1",),
                 "cuda:0,cuda:1": ("cuda:0", "cuda:1"), " cuda:0 , cpu ": ("cuda:0", "cpu")}
        for text, want in cases.items():
            with self.subTest(text):
                self.assertEqual(api.parse_device(text), want)
        self.assertEqual(api.parse_device(["cuda:0", "cuda:1"]), ("cuda:0", "cuda:1"))
        self.assertEqual(api.parse_device(("cpu",)), ("cpu",))
        for bad in ("gpu", "cuda:x", "", "cuda:-1", None, 3, []):
            with self.subTest(bad), self.assertRaises(api.ApiError):
                api.parse_device(bad)

    def test_quant_names_map_to_a_loader_quant_and_a_compute_dtype(self):
        self.assertEqual(api.quant_settings("nf4", "float16"), ("nf4", "float16"))
        self.assertEqual(api.quant_settings("fp16", "float32"), ("none", "float16"))
        self.assertEqual(api.quant_settings("bf16", "float16"), ("none", "bfloat16"))
        self.assertEqual(api.quant_settings("none", "float16"), ("none", "float16"))
        with self.assertRaisesRegex(api.ApiError, "quant"):
            api.quant_settings("int8", "float16")


class DensePlanContract(unittest.TestCase):
    def test_a_model_that_fits_one_gpu_goes_on_one_gpu(self):
        plan = env.plan_dense_devices([14.7], need_gb=8.7)
        self.assertEqual((plan.strategy, plan.ok), ("single", True))
        self.assertEqual(list(plan.max_memory_gb), [0])

    def test_a_model_that_needs_two_gpus_is_split_over_them(self):
        plan = env.plan_dense_devices([14.7, 14.7], need_gb=20.0)
        self.assertEqual((plan.strategy, plan.ok), ("split2", True))
        self.assertEqual(sorted(plan.max_memory_gb), [0, 1])

    def test_too_small_is_insufficient_and_says_how_much_is_needed(self):
        plan = env.plan_dense_devices([14.7], need_gb=40.0)
        self.assertFalse(plan.ok)
        self.assertEqual(plan.strategy, "insufficient")
        self.assertIn("40", " ".join(plan.notes))

    def test_no_gpu_is_insufficient(self):
        self.assertFalse(env.plan_dense_devices([], need_gb=1.0).ok)


@needs_torch
class LoadOptionsContract(HermeticTestCase):
    def options(self, hw=None, **kw):
        return api.load_options(kw.pop("cfg", None) or make_cfg(), hardware=hw, **kw)

    def test_two_t4s_split_the_nf4_layers_over_both(self):
        opt = self.options(hardware(14.74, 14.74))
        self.assertEqual((opt.quant, opt.compute_dtype), ("nf4", "float16"))
        self.assertEqual(tuple(opt.devices), ("cuda:0", "cuda:1"))
        self.assertEqual(len(opt.capacities), 2)
        self.assertEqual(opt.offload_layers, 0)

    def test_one_t4_keeps_the_embeddings_on_the_cpu(self):
        opt = self.options(hardware(14.74))
        self.assertEqual(tuple(opt.devices), ("cuda:0",))
        self.assertEqual(opt.embeddings_device, "cpu")
        self.assertEqual(opt.offload_layers, 0)

    def test_a_bigger_single_gpu_needs_no_tricks(self):
        opt = self.options(hardware(24.0))
        self.assertEqual((tuple(opt.devices), opt.embeddings_device, opt.offload_layers), (("cuda:0",), None, 0))

    def test_a_small_gpu_needs_the_cpu_offload_flag_and_then_offloads_layers(self):
        with self.assertRaisesRegex(api.ApiError, "NF4 needs"):
            self.options(hardware(9.0))
        opt = self.options(hardware(9.0), allow_cpu_offload=True)
        self.assertGreater(opt.offload_layers, 0)
        self.assertEqual(opt.embeddings_device, "cpu")

    def test_no_gpu_means_cpu_and_float32(self):
        opt = self.options(hardware())
        self.assertEqual((tuple(opt.devices), opt.compute_dtype, opt.quant), (("cpu",), "float32", "nf4"))

    def test_dense_4b_on_a_t4_stays_fp16_on_one_gpu(self):
        opt = self.options(hardware(14.74), quant="fp16", model_gb=8.7)
        self.assertEqual((opt.quant, opt.compute_dtype, tuple(opt.devices)), ("none", "float16", ("cuda:0",)))

    def test_dense_27b_does_not_fit_two_t4s_and_says_so(self):
        with self.assertRaisesRegex(api.ApiError, "needs about"):
            self.options(hardware(14.74, 14.74), quant="fp16", model_gb=47.7)

    def test_bf16_computes_in_bfloat16(self):
        opt = self.options(hardware(94.0), quant="bf16", model_gb=47.7)
        self.assertEqual((opt.quant, opt.compute_dtype, tuple(opt.devices)), ("none", "bfloat16", ("cuda:0",)))

    def test_an_explicit_device_skips_the_planner(self):
        opt = self.options(hardware(), device="cuda:1")
        self.assertEqual(tuple(opt.devices), ("cuda:1",))
        two = self.options(hardware(), device="cuda:0,cuda:1")
        self.assertEqual(tuple(two.devices), ("cuda:0", "cuda:1"))

    def test_the_config_decides_the_default_quant_and_overrides_win(self):
        cfg = make_cfg(eval={"quant": "fp16", "batch_tokens": 1000})
        opt = self.options(hardware(14.74), cfg=cfg, model_gb=8.7)
        self.assertEqual((opt.quant, opt.compute_dtype, opt.batch_tokens), ("none", "float16", 1000))
        over = self.options(hardware(14.74), cfg=cfg, model_gb=8.7, batch_tokens=77, window_mb=64)
        self.assertEqual((over.batch_tokens, over.window_mb), (77, 64))

    def test_an_unknown_override_is_an_error(self):
        with self.assertRaisesRegex(api.ApiError, "unknown"):
            self.options(hardware(14.74), no_such_option=1)


class FitTemperatureContract(HermeticTestCase):
    def test_a_missing_file_says_which_one(self):
        with self.assertRaisesRegex(api.ApiError, "nope"):
            api.fit_temperature("/tmp/definitely/nope/results.parquet")

    def test_by_must_be_a_known_grouping(self):
        with self.assertRaisesRegex(api.ApiError, "by"):
            api.fit_temperature("/tmp/x", by="colour")


if __name__ == "__main__":
    unittest.main()
