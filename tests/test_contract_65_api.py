"""Contract tests for decide/sd/api.py (issue #65, F0 kit core): the interface Squad B and the notebooks call.

Everything here is pure (no download, no model).  The tests that build `LoadOptions` need torch and skip without it.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from decide_support import HermeticTestCase, needs_pyarrow, needs_torch

from decide.sd import api, env
from decide.sd import config as C


A, D, E, L, B = ("a" * 40, "b" * 40, "e" * 40, "c" * 40, "d" * 40)       # commit hashes for the repos of these tests


def make_cfg(**overrides):
    """The shipped config, read alone (no overlays from other squads), with `overrides` deep-merged on top."""
    return C.load_config(C.DEFAULT_PATH, overrides=overrides or None)


def hardware(*gpus_gb, ram=29.0):
    names = tuple("Tesla T4" for _ in gpus_gb)
    return env.Hardware("kaggle", "/tmp/work", names, tuple(gpus_gb), ram, 50.0, tuple((7, 5) for _ in gpus_gb))


class NormalizeSpecContract(HermeticTestCase):
    def test_a_bare_anchor_gets_every_default(self):
        spec = api.normalize_spec({"anchor": "org/a", "anchor_revision": A})
        self.assertEqual(spec, {"anchor": "org/a", "anchor_revision": A, "donors": [], "lora": None, "method": "anchor",
                                "density": None, "protocol": "pplx_joint", "reference": "anchor", "base": None, "name": None})

    def test_a_revision_must_be_a_full_commit_hash_everywhere_a_repo_is_named(self):
        good = {"anchor": "org/a", "anchor_revision": "a" * 40, "donors": [{"repo": "org/d", "weight": 0.5, "revision": "b" * 40}],
                "lora": {"repo": "org/l", "weight": 0.1, "revision": "c" * 40}}
        api.normalize_spec(good)
        for where, make in (("anchor_revision", lambda bad: {**good, "anchor_revision": bad}),
                            ("donor", lambda bad: {**good, "donors": [{"repo": "org/d", "weight": 0.5, "revision": bad}]}),
                            ("lora", lambda bad: {**good, "lora": {"repo": "org/l", "weight": 0.1, "revision": bad}}),
                            ("base", lambda bad: {**good, "reference": "base", "base": {"repo": "org/b", "revision": bad}})):
            for bad in ("main", None, "", "abc123", "A" * 40, "z" * 40):
                with self.subTest(f"{where}: {bad!r}"), self.assertRaisesRegex(api.ApiError, "revision"):
                    api.normalize_spec(make(bad))

    def test_a_folder_on_this_machine_needs_no_revision_and_a_hub_repo_still_does(self):
        """`dir:PATH` is a checkpoint that is not on the Hub (the model that was just trained, which G4 and the public round load through the kit):
        it has no commit to pin.  Anything named like a Hub repo keeps needing 40 hex."""
        local = api.normalize_spec({"anchor": "dir:work/novo", "donors": [{"repo": "dir:/m/donor", "weight": 0.5}],
                                    "lora": {"repo": "dir:/m/lora", "weight": 0.1}})
        self.assertEqual((local["anchor"], local["anchor_revision"]), ("dir:work/novo", None))
        self.assertEqual([(d["repo"], d["revision"]) for d in local["donors"]], [("dir:/m/donor", None)])
        self.assertEqual((local["lora"]["repo"], local["lora"]["revision"]), ("dir:/m/lora", None))
        mixed = api.normalize_spec({"anchor": "org/a", "anchor_revision": A, "donors": [{"repo": "dir:/m/donor", "weight": 0.5}],
                                    "lora": {"repo": "org/l", "weight": 0.1, "revision": L}})
        self.assertEqual((mixed["anchor_revision"], mixed["donors"][0]["revision"], mixed["lora"]["revision"]), (A, None, L))
        for name, bad in (("anchor", {"anchor": "org/a"}),
                          ("anchor, text", {"anchor": "org/a", "anchor_revision": "main"}),
                          ("donor", {"anchor": "dir:/m/a", "donors": [{"repo": "org/d", "weight": 0.5}]}),
                          ("lora", {"anchor": "dir:/m/a", "lora": {"repo": "org/l", "weight": 0.1}}),
                          ("base", {"anchor": "dir:/m/a", "reference": "base", "base": {"repo": "org/b"}})):
            with self.subTest(f"a Hub repo without a commit: {name}"), self.assertRaisesRegex(api.ApiError, "40-hex"):
                api.normalize_spec(bad)

    def test_a_folder_with_a_revision_is_refused_because_a_folder_has_none(self):
        for name, bad in (("anchor", {"anchor": "dir:/m/a", "anchor_revision": A}),
                          ("donor", {"anchor": "dir:/m/a", "donors": [{"repo": "dir:/m/d", "weight": 0.5, "revision": D}]}),
                          ("lora", {"anchor": "dir:/m/a", "lora": {"repo": "dir:/m/l", "weight": 0.1, "revision": L}})):
            with self.subTest(name), self.assertRaisesRegex(api.ApiError, "no revision"):
                api.normalize_spec(bad)

    def folders(self):
        """A real folder `x`, a link `latest` to it, another folder `y`, and the root they sit in."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        for name in ("x", "y"):
            (root / name).mkdir()
        (root / "latest").symlink_to(root / "x")
        return root

    def spellings_of_x(self, root):
        return {"a trailing slash": f"dir:{root}/x/", "a dot and a dot-dot": f"dir:{root}/./x/../x", "a double slash": f"dir:{root}//x",
                "a link to it": f"dir:{root}/latest", "a link and a slash": f"dir:{root}/latest/"}

    def test_two_spellings_of_one_folder_are_one_repo_and_the_checks_for_duplicates_see_it(self):
        """A donor that is the anchor gives a merge that is the anchor (difference 0.0): for a folder the file system reads every spelling alike."""
        root = self.folders()
        for name, spelling in self.spellings_of_x(root).items():
            with self.subTest(f"donor is the anchor: {name}"), self.assertRaisesRegex(api.ApiError, "is the anchor"):
                api.normalize_spec({"anchor": f"dir:{root}/x", "donors": [{"repo": spelling, "weight": 0.5}]})
            with self.subTest(f"donor is the LoRA: {name}"), self.assertRaisesRegex(api.ApiError, "both a donor and the LoRA"):
                api.normalize_spec({"anchor": f"dir:{root}/y", "donors": [{"repo": f"dir:{root}/x", "weight": 0.5}],
                                    "lora": {"repo": spelling, "weight": 0.1}})
            with self.subTest(f"the same donor twice: {name}"), self.assertRaisesRegex(api.ApiError, "listed twice"):
                api.normalize_spec({"anchor": f"dir:{root}/y", "donors": [{"repo": f"dir:{root}/x", "weight": 0.5}, {"repo": spelling, "weight": 0.5}]})
        spec = api.normalize_spec({"anchor": f"dir:{root}/x", "donors": [{"repo": f"dir:{root}/y", "weight": 0.5}], "lora": {"repo": f"dir:{root}/other", "weight": 0.1}})
        self.assertEqual(len(spec["donors"]), 1)                                     # two different folders are two repos
        api.normalize_spec({"anchor": f"dir:{root}/y", "donors": [{"repo": f"dir:{root}/x", "weight": 0.5}], "lora": {"repo": f"dir:{root}/other", "weight": 0.1}})

    def test_the_case_of_a_folder_counts_and_the_case_of_a_hub_id_does_not(self):
        from decide.sd import remote

        self.assertNotEqual(remote.repo_key("dir:/models/A"), remote.repo_key("dir:/models/a"))
        self.assertEqual(remote.repo_key("Org/Repo"), remote.repo_key("org/repo"))
        self.assertNotEqual(remote.repo_key("dir:/models/org/repo"), remote.repo_key("org/repo"))

    def test_a_folder_named_without_a_path_is_refused(self):
        for bad in ("dir:", "dir:   "):
            with self.subTest(bad), self.assertRaises(api.ApiError):
                api.normalize_spec({"anchor": bad})

    def test_the_module_documents_that_reasoning_is_refused_for_now(self):
        self.assertIn("`reasoning`", api.__doc__)
        self.assertIn("fold_reasoning()", api.__doc__)

    def test_a_repo_cannot_be_a_donor_and_the_lora_or_a_donor_and_the_anchor(self):
        base = {"anchor": "org/a", "anchor_revision": A}
        both = {**base, "donors": [{"repo": "org/Simplicio-27B", "weight": 0.5, "revision": D}], "lora": {"repo": "org/simplicio-27b", "weight": 0.1, "revision": L}}
        with self.assertRaisesRegex(api.ApiError, "both a donor and the LoRA"):
            api.normalize_spec(both)
        with self.assertRaisesRegex(api.ApiError, "is the anchor"):
            api.normalize_spec({**base, "donors": [{"repo": "org/a", "weight": 0.5, "revision": D}]})
        api.normalize_spec({**both, "lora": {"repo": "org/other-lora", "weight": 0.1, "revision": L}})          # a different adapter is fine
        api.normalize_spec({**both, "lora": {"repo": "org/simplicio-27b", "weight": 0, "revision": L}})         # a LoRA with weight zero is dropped

    def test_a_zero_weight_part_is_dropped_before_its_revision_matters(self):
        spec = api.normalize_spec({"anchor": "org/a", "anchor_revision": "a" * 40, "donors": [{"repo": "org/d", "weight": 0}], "lora": {"repo": "org/l", "weight": 0}})
        self.assertEqual((spec["donors"], spec["lora"]), ([], None))

    def test_the_input_is_not_mutated(self):
        raw = {"anchor": "org/a", "anchor_revision": A, "donors": [{"repo": "org/d", "weight": 0.5, "revision": D}], "lora": {"repo": "org/l", "weight": 0.1, "revision": L}}
        before = copy.deepcopy(raw)
        api.normalize_spec(raw)
        self.assertEqual(raw, before)

    def test_donors_and_lora_are_carried_with_their_weights(self):
        spec = api.normalize_spec({"anchor": "org/a", "anchor_revision": A, "donors": [{"repo": "org/d", "weight": 0.5, "revision": D}, {"repo": "org/e", "weight": 0.25, "revision": E}],
                                   "lora": {"repo": "org/l", "weight": 0.1, "revision": L}})
        self.assertEqual(spec["donors"], [{"repo": "org/d", "weight": 0.5, "revision": D}, {"repo": "org/e", "weight": 0.25, "revision": E}])
        self.assertEqual(spec["lora"], {"repo": "org/l", "weight": 0.1, "revision": L})
        self.assertEqual(spec["method"], "task_arithmetic")

    def test_donors_with_weight_zero_are_dropped_and_nothing_left_means_the_anchor_alone(self):
        spec = api.normalize_spec({"anchor": "org/a", "anchor_revision": A, "donors": [{"repo": "org/d", "weight": 0}], "lora": {"repo": "org/l", "weight": 0}, "method": "ties"})
        self.assertEqual((spec["donors"], spec["lora"], spec["method"], spec["density"]), ([], None, "anchor", None))

    def test_ties_gets_a_density(self):
        base = {"anchor": "org/a", "anchor_revision": A, "donors": [{"repo": "org/d", "weight": 1.0, "revision": D}], "method": "ties"}
        self.assertEqual(api.normalize_spec(base)["density"], 0.5)
        self.assertEqual(api.normalize_spec({**base, "density": 0.2})["density"], 0.2)
        for bad in (0, 0.0, -0.1, 1.5, "half"):
            with self.subTest(bad), self.assertRaisesRegex(api.ApiError, "density"):
                api.normalize_spec({**base, "density": bad})

    def test_task_arithmetic_does_not_carry_a_density(self):
        spec = api.normalize_spec({"anchor": "org/a", "anchor_revision": A, "donors": [{"repo": "org/d", "weight": 1.0, "revision": D}], "density": 0.3})
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
            ({"donors": [{"repo": "org/d", "weight": 1, "revision": D}, {"repo": "org/d", "weight": 2, "revision": D}]}, "twice"),
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
                api.normalize_spec({"anchor": "org/a", "anchor_revision": A, **extra})

    def test_licence_and_head_exclusions_apply_to_hand_written_specs_too(self):
        for repo in ("someone/Lychee-27B", "someone/amos-decider"):
            with self.subTest(repo), self.assertRaisesRegex(api.ApiError, "not allowed"):
                api.normalize_spec({"anchor": "org/a", "anchor_revision": A, "donors": [{"repo": repo, "weight": 1.0, "revision": D}]})

    def test_reference_base_takes_the_base_from_the_config(self):
        cfg = make_cfg()
        spec = api.normalize_spec({"anchor": "org/a", "anchor_revision": A, "reference": "base"}, cfg)
        self.assertEqual(spec["base"], {"repo": cfg.models.base.repo, "revision": cfg.models.base.revision})
        own = api.normalize_spec({"anchor": "org/a", "anchor_revision": A, "reference": "base", "base": {"repo": "org/b", "revision": B}}, cfg)
        self.assertEqual(own["base"], {"repo": "org/b", "revision": B})
        with self.assertRaisesRegex(api.ApiError, "base"):
            api.normalize_spec({"anchor": "org/a", "anchor_revision": A, "reference": "base"})                # no config, no base

    def test_a_normalised_spec_is_stable(self):
        once = api.normalize_spec({"anchor": "org/a", "anchor_revision": A, "donors": [{"repo": "org/d", "weight": 1, "revision": D}], "lora": {"repo": "org/l", "weight": 0.1, "revision": L}})
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
        donors = [{"name": "d3", "repo": "vllm-sr/d3", "weight": 1.0, "enabled": True, "revision": D},
                  {"name": "d3-turbo", "repo": "vllm-sr/d3-turbo", "weight": 0.5, "enabled": True, "revision": E}]
        cfg = make_cfg(models={"donors": donors}, merge={"max_candidates": 8})
        weights = {d["repo"]: d["weight"] for d in api.spec_for(cfg, "ta_l0.5")["donors"]}
        self.assertAlmostEqual(weights["vllm-sr/d3"], 0.5 * 1.0 / 1.5, places=9)
        self.assertAlmostEqual(weights["vllm-sr/d3-turbo"], 0.5 * 0.5 / 1.5, places=9)

    def test_lora_candidates_carry_the_adapter_and_its_weight(self):
        cfg = make_cfg()
        spec = api.spec_for(cfg, "anchor+lora0.1")
        self.assertEqual(spec["lora"], {"repo": cfg.models.lora.repo, "weight": 0.1, "revision": cfg.models.lora.revision})
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


@needs_pyarrow
class FitTemperatureOriginContract(HermeticTestCase):
    """A temperature is not fitted on evaluation data (ADR 0004, item 6: the GT sample and the public suite are not training data)."""

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def table(self, origins, name="results.parquet", per=40):
        """A probability table with `per` questions for each origin in `origins` (None: no origin column at all)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = []
        for origin in origins:
            for i in range(per):
                gold = "A" if i % 2 else "B"
                wrong = i % 4 == 0
                top = (0 if gold == "A" else 1) if not wrong else (1 if gold == "A" else 0)
                logits = [[3.0, 0.0] if top == 0 else [0.0, 3.0]]
                row = {"run_id": f"{origin}-{i}", "question_id": "q", "qtype": "choice", "status": "ok", "option_keys": ["A", "B"],
                       "logits": logits, "gold": gold, "temperature": 1.0}
                if origin != "none":
                    row["origin"] = origin
                rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows), str(self.dir / name))
        return self.dir / name

    def test_a_table_of_the_public_suite_is_refused_and_the_refusal_says_why(self):
        with self.assertRaisesRegex(api.ApiError, r"public-suite.*ADR 0004"):
            api.fit_temperature(self.table(["public-suite"]), min_items=5)

    def test_the_gt_sample_is_evaluation_data_too(self):
        with self.assertRaisesRegex(api.ApiError, "gt-sample"):
            api.fit_temperature(self.table(["gt-sample"]), min_items=5)

    def test_a_few_evaluation_rows_among_clean_ones_are_still_refused(self):
        with self.assertRaisesRegex(api.ApiError, "public-suite"):
            api.fit_temperature(self.table(["requests", "public-suite"], per=20), min_items=5)

    def test_the_refusal_names_the_flag_that_overrides_it(self):
        with self.assertRaisesRegex(api.ApiError, "allow_evaluation_data"):
            api.fit_temperature(self.table(["public-suite"]), min_items=5)

    def test_rows_that_are_not_evaluation_data_are_fitted_and_the_result_says_what_they_were(self):
        fit = api.fit_temperature(self.table(["requests"]), min_items=5, seed=1)
        self.assertEqual(fit["fitted_on"], {"requests": 40})
        self.assertEqual(fit["evaluation_rows"], 0)
        self.assertNotIn("warning", fit)
        self.assertIn("choice", fit["temperatures"])

    def test_the_flag_allows_it_and_the_result_carries_the_mark(self):
        fit = api.fit_temperature(self.table(["public-suite", "requests"], per=20), min_items=5, seed=1, allow_evaluation_data=True)
        self.assertEqual(fit["fitted_on"], {"public-suite": 20, "requests": 20})
        self.assertEqual(fit["evaluation_rows"], 20)
        self.assertRegex(fit["warning"], "evaluation data")
        self.assertIn("choice", fit["temperatures"])

    def test_the_flag_must_be_a_real_true(self):
        for loose in (1, "yes", "False"):
            with self.assertRaisesRegex(api.ApiError, "public-suite"):
                api.fit_temperature(self.table(["public-suite"]), min_items=5, allow_evaluation_data=loose)

    def test_a_table_that_does_not_say_where_its_rows_came_from_is_fitted_and_marked_unknown(self):
        fit = api.fit_temperature(self.table(["none"]), min_items=5, seed=1)
        self.assertEqual(fit["fitted_on"], {"unknown": 40})
        self.assertEqual(fit["evaluation_rows"], 0)


if __name__ == "__main__":
    unittest.main()
