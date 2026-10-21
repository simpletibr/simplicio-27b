"""Contract of decide/rental (issue #71, F6 - the rented GPU day): hardware and price tables, the two profiles, the
estimator, the config overlay, the day's plan and state, gate G4, the thinker-trace step and the Modal variant.
Written before the code (see the PR for the red run).

Everything here runs on a CPU with no network, no Hugging Face account and no Modal account.  What needs torch,
pyarrow or PyYAML skips cleanly when the package is missing.  The numbers of the estimator are UNVERIFIED by
construction: the tests pin the formulas, not a measurement.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from decide_b_support import ROOT, needs_yaml

from decide.rental import estimate as E
from decide.rental import hardware as HW
from decide.rental import profile as PRF

PARAMS = 26_085_330_160            # model.safetensors.index.json of pplx-decider-v1.2: metadata.total_parameters
PUBLIC_TOKENS = 291_826_668        # billed input tokens of the full public round (PLAN.md section 2)


class HardwareContract(unittest.TestCase):
    def test_the_training_gpus_have_memory_flops_and_a_price(self) -> None:
        got = {name: (g.memory_gb, g.bf16_tflops, g.usd_per_second) for name, g in HW.GPUS.items()}
        self.assertEqual(got, {
            "B200": (180, 2250, 0.001736),
            "H200": (141, 989, 0.001261),
            "H100": (80, 989, 0.001097),
            "A100-80GB": (80, 312, 0.000694),
        })

    def test_the_sources_of_the_tables_are_written_down(self) -> None:
        self.assertIn("Modal", HW.PRICE_SOURCE)
        self.assertIn("2026-10-21", HW.PRICE_SOURCE)
        self.assertIn("dense", HW.SPEC_SOURCE)

    def test_the_parameter_count_is_the_one_in_the_checkpoint_index(self) -> None:
        self.assertEqual(HW.PARAMS, PARAMS)
        self.assertEqual(HW.WEIGHT_BYTES, 2 * PARAMS)         # bf16: 52,170,660,320 bytes, as in the index
        self.assertEqual(HW.PUBLIC_ROUND_INPUT_TOKENS, PUBLIC_TOKENS)

    def test_lora_counts_come_from_the_architecture(self) -> None:
        self.assertEqual(HW.LORA_UNITS_PER_RANK * 128, 870_318_080)
        self.assertEqual(HW.READOUT_PARAMS, 255 * 5120)
        self.assertEqual(HW.trainable_parameters("lora", 128), 870_318_080 + 255 * 5120)          # adapters + the readout head
        self.assertEqual(HW.trainable_parameters("full", 128), PARAMS + 255 * 5120)               # the backbone + the readout head

    def test_an_unknown_gpu_lists_the_known_ones(self) -> None:
        with self.assertRaises(KeyError) as caught:
            HW.gpu("H1000")
        for name in HW.GPUS:
            self.assertIn(name, str(caught.exception))

    def test_the_default_utilization_is_the_plans(self) -> None:
        self.assertEqual(HW.MFU, 0.40)


@needs_yaml
class ProfileContract(unittest.TestCase):
    def test_the_two_profiles_of_the_plan(self) -> None:
        self.assertEqual(PRF.profile_names(), ["econ_b200", "full_8xh100"])
        econ, full = PRF.load_profile("econ_b200"), PRF.load_profile("full_8xh100")
        self.assertEqual((econ.gpu, econ.gpus, econ.train["mode"], econ.train["lora_rank"], econ.seeds, econ.examples), ("B200", 1, "lora", 128, 1, 250_000))
        self.assertEqual((full.gpu, full.gpus, full.train["mode"], full.seeds, full.examples), ("H100", 8, "full", 2, 600_000))
        self.assertEqual((econ.plan["hours"], econ.plan["usd"]), ("~14", "~90"))
        self.assertEqual((full.plan["hours"], full.plan["usd"]), ("12-16", "380-510"))

    def test_a_profile_can_be_loaded_by_path(self) -> None:
        path = ROOT / "decide" / "rental" / "profiles" / "econ_b200.yaml"
        self.assertEqual(PRF.load_profile(str(path)).name, "econ_b200")

    def test_every_option_of_a_profile_is_one_the_trainer_has(self) -> None:
        # the allowlist the loader uses is a subset of the real parser, and the argv a profile builds parses with it
        try:
            from decide.train import train as TRAIN
        except ImportError:
            self.skipTest("torch is not installed")
        parser = TRAIN.build_parser()
        flags = {flag.lstrip("-").replace("-", "_") for action in parser._actions for flag in action.option_strings}
        flags |= {"gradient_checkpointing", "no_gradient_checkpointing"}
        self.assertLessEqual(set(PRF.TRAIN_OPTIONS), flags)
        for name in PRF.profile_names():
            profile = PRF.load_profile(name)
            for traces in (False, True):
                argv = ["--targets", "t", "--checkpoint", "c", "--out", "o", *profile.train_argv(with_traces=traces)]
                args = parser.parse_args(argv)
                TRAIN.validate(args)
                self.assertEqual(args.mode, profile.train["mode"])

    def test_the_orchestrator_owns_the_paths_the_seed_and_the_device(self) -> None:
        for forbidden in ("targets", "checkpoint", "out", "seed", "resume", "dry_run", "device"):
            self.assertNotIn(forbidden, PRF.TRAIN_OPTIONS)

    def test_options_become_flags(self) -> None:
        argv = PRF.cli_args({"mode": "lora", "lora_rank": 128, "lr": None, "no_permute": True, "gradient_checkpointing": False, "reasoning": "off", "alpha": 0.5})
        self.assertEqual(argv, ["--mode", "lora", "--lora-rank", "128", "--no-permute", "--no-gradient-checkpointing", "--reasoning", "off", "--alpha", "0.5"])
        self.assertEqual(PRF.cli_args({"no_permute": False, "gradient_checkpointing": True}), ["--gradient-checkpointing"])

    def test_traces_switch_on_the_reasoning_the_trainer_shows(self) -> None:
        for name in PRF.profile_names():
            profile = PRF.load_profile(name)
            self.assertEqual(profile.train["reasoning"], "off")
            plain = profile.train_argv(with_traces=False)
            traced = profile.train_argv(with_traces=True)
            self.assertEqual(plain[plain.index("--reasoning") + 1], "off")
            self.assertEqual(traced[traced.index("--reasoning") + 1], "mixed")
            self.assertIn("--reasoning-fraction", traced)

    def test_unknown_keys_and_bad_values_are_refused(self) -> None:
        good = PRF.load_profile("econ_b200").raw
        def broken(**changes):
            data = json.loads(json.dumps(good))
            for dotted, value in changes.items():
                node = data
                *path, last = dotted.split(".")
                for key in path:
                    node = node[key]
                if value is KeyError:
                    del node[last]
                else:
                    node[last] = value
            return data
        cases = {
            "extra top-level key": broken(surprise=1),
            "unknown train option": broken(**{"train.surprise": 1}),
            "a secret in a train option": broken(**{"train.targets": "x"}),
            "unknown gpu": broken(**{"hardware.gpu": "H1000"}),
            "zero gpus": broken(**{"hardware.gpus": 0}),
            "zero seeds": broken(**{"workload.seeds": 0}),
            "negative examples": broken(**{"workload.examples": -1}),
            "boolean as a number": broken(**{"workload.examples": True}),
            "bad mode": broken(**{"train.mode": "half"}),
            "missing name": broken(name=KeyError),
            "wrong version": broken(version=2),
        }
        for label, data in cases.items():
            with self.subTest(label), self.assertRaises(PRF.ProfileError):
                PRF.profile_from_dict(data)
        self.assertEqual(PRF.profile_from_dict(good).name, "econ_b200")

    def test_an_unknown_profile_name_lists_the_known_ones(self) -> None:
        with self.assertRaises(PRF.ProfileError) as caught:
            PRF.load_profile("nope")
        self.assertIn("econ_b200", str(caught.exception))

    def test_the_digest_follows_the_content(self) -> None:
        econ, full = PRF.load_profile("econ_b200"), PRF.load_profile("full_8xh100")
        self.assertEqual(econ.digest, PRF.load_profile("econ_b200").digest)
        self.assertNotEqual(econ.digest, full.digest)
        self.assertRegex(econ.digest, r"^[0-9a-f]{64}$")

    def test_a_profile_holds_no_secret(self) -> None:
        for path in sorted((ROOT / "decide" / "rental" / "profiles").glob("*.yaml")):
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"hf_[A-Za-z0-9]{10,}", path.name)
            self.assertNotRegex(text.lower(), r"(token|secret|password|api_key)\s*:", path.name)


@needs_yaml
class EstimateContract(unittest.TestCase):
    def setUp(self) -> None:
        self.econ = PRF.load_profile("econ_b200")
        self.full = PRF.load_profile("full_8xh100")

    @staticmethod
    def phases(report):
        return {phase["name"]: phase for phase in report["phases"]}

    def test_training_follows_the_formula_of_the_plan(self) -> None:
        report = E.estimate(self.econ)
        tokens = 250_000 * self.econ.tokens_per_example
        train = self.phases(report)["train"]
        self.assertEqual(train["flops"], 6 * PARAMS * tokens)
        self.assertAlmostEqual(train["seconds"], 6 * PARAMS * tokens / (2250e12 * 0.40), places=3)
        self.assertAlmostEqual(train["usd"], train["seconds"] * 1 * 0.001736, places=6)
        self.assertEqual(report["assumptions"]["tokens_trained"], tokens)

    def test_seeds_multiply_the_training_and_gpus_divide_it(self) -> None:
        report = E.estimate(self.full)
        tokens = 600_000 * self.full.tokens_per_example * 2
        train = self.phases(report)["train"]
        self.assertEqual(train["flops"], 6 * PARAMS * tokens)
        self.assertAlmostEqual(train["seconds"], 6 * PARAMS * tokens / (8 * 989e12 * 0.40), places=3)
        self.assertAlmostEqual(train["usd"], train["seconds"] * 8 * 0.001097, places=6)

    def test_the_public_round_is_inference_over_the_board_token_count(self) -> None:
        load = HW.WEIGHT_BYTES / (HW.ASSUMED_IO_GB_PER_S * 1e9)          # one model load (a guess about the disk), UNVERIFIED
        report = E.estimate(self.econ)
        public = self.phases(report)["public_round"]
        self.assertEqual(public["flops"], 2 * PARAMS * PUBLIC_TOKENS)
        self.assertAlmostEqual(public["seconds"], 2 * PARAMS * PUBLIC_TOKENS / (2250e12 * 0.40) + load, places=3)
        report = E.estimate(self.full)
        self.assertAlmostEqual(self.phases(report)["public_round"]["seconds"], 2 * PARAMS * PUBLIC_TOKENS / (8 * 989e12 * 0.40) + load, places=3)

    def test_totals_add_the_phases_up(self) -> None:
        report = E.estimate(self.econ)
        phases = self.phases(report)
        day = sum(phases[name]["seconds"] for name in ("setup", "train", "soup", "calibrate", "g4") if phases[name]["seconds"] is not None)
        self.assertAlmostEqual(report["totals"]["day_seconds"], day, places=3)
        self.assertAlmostEqual(report["totals"]["day_hours"], day / 3600, places=6)
        everything = day + phases["public_round"]["seconds"] + phases["publish"]["seconds"]
        self.assertAlmostEqual(report["totals"]["with_public_round_seconds"], everything, places=3)
        usd = sum(phases[name]["usd"] for name in ("setup", "train", "soup", "calibrate", "g4") if phases[name]["usd"] is not None)
        self.assertAlmostEqual(report["totals"]["day_usd"], usd, places=6)

    def test_without_the_public_round_it_is_not_counted(self) -> None:
        report = E.estimate(self.econ, public_round=False)
        self.assertNotIn("public_round", self.phases(report))
        self.assertNotIn("with_public_round_seconds", report["totals"])

    def test_every_number_is_marked_unverified(self) -> None:
        for profile in (self.econ, self.full):
            report = E.estimate(profile)
            self.assertEqual(report["status"], "UNVERIFIED")
            for phase in report["phases"]:
                self.assertEqual(phase["status"], "UNVERIFIED", phase["name"])
                self.assertTrue(phase["basis"], phase["name"])
            text = E.render_text(report)
            self.assertIn("UNVERIFIED", text.splitlines()[0])
            for line in text.splitlines():
                if re.search(r"\d+(\.\d+)? ?(h|USD|GB|s|TFLOPS)\b", line):
                    self.assertIn("UNVERIFIED", line, line)

    def test_the_plan_figures_are_shown_next_to_the_estimate(self) -> None:
        report = E.estimate(self.econ)
        self.assertEqual(report["plan_section_6"], {"hours": "~14", "usd": "~90"})
        self.assertIn("~14", E.render_text(report))

    def test_ddp_full_fine_tuning_does_not_fit_on_80_gb(self) -> None:
        report = E.estimate(self.full)
        codes = [w["code"] for w in report["warnings"]]
        self.assertIn("ddp_does_not_fit", codes)
        warning = next(w for w in report["warnings"] if w["code"] == "ddp_does_not_fit")
        for needle in ("104.3", "80", "H200", "B200", "FSDP", "UNVERIFIED"):
            self.assertIn(needle, warning["text"])
        self.assertAlmostEqual(report["memory"]["gpu_gb_needed"], (4 * PARAMS + 4 * 255 * 5120) / 1e9, places=3)     # bf16 weights + bf16 gradients (+ the head)
        self.assertFalse(report["memory"]["fits"])

    def test_bigger_gpus_clear_the_ddp_warning_but_not_the_activations(self) -> None:
        for gpu in ("H200", "B200"):
            report = E.estimate(self.full, gpu=gpu)
            codes = [w["code"] for w in report["warnings"]]
            self.assertNotIn("ddp_does_not_fit", codes, gpu)
            self.assertIn("activations_not_counted", codes, gpu)
            self.assertTrue(report["memory"]["fits"], gpu)

    def test_lora_on_one_b200_fits_and_needs_little_host_memory(self) -> None:
        report = E.estimate(self.econ)
        memory = report["memory"]
        self.assertTrue(memory["fits"])
        trainable = 870_318_080 + 255 * 5120
        self.assertAlmostEqual(memory["gpu_gb_needed"], (2 * PARAMS + 4 * trainable) / 1e9, places=3)     # bf16 backbone + adapters and their gradients
        self.assertEqual(memory["trainable_parameters"], trainable)
        self.assertAlmostEqual(memory["host_gb_optimizer"], 16 * (870_318_080 + 255 * 5120) / 1e9, places=3)
        self.assertNotIn("ddp_does_not_fit", [w["code"] for w in report["warnings"]])

    def test_the_cpu_optimizer_of_a_full_fine_tune_needs_hundreds_of_gb_of_host_memory(self) -> None:
        report = E.estimate(self.full, gpu="H200")
        self.assertAlmostEqual(report["memory"]["host_gb_optimizer"], 16 * (PARAMS + 255 * 5120) / 1e9, places=3)
        warning = next(w for w in report["warnings"] if w["code"] == "host_memory")
        self.assertIn("417", warning["text"])

    def test_the_host_memory_of_the_profile_is_checked_against_the_optimizer(self) -> None:
        for profile in (self.econ, self.full):
            report = E.estimate(profile, gpu="H200" if profile is self.full else None)
            needed = report["memory"]["host_gb_optimizer"]
            self.assertGreaterEqual(profile.host_ram_gb, needed, profile.name)

    def test_traces_are_not_estimated_without_a_measured_throughput(self) -> None:
        report = E.estimate(self.econ, traces=True)
        traces = self.phases(report)["traces"]
        self.assertIsNone(traces["seconds"])
        self.assertIn("throughput", traces["basis"])
        report = E.estimate(self.econ, traces=True, thinker_tokens_per_second=1000.0)
        traces = self.phases(report)["traces"]
        self.assertIsNotNone(traces["seconds"])
        thinker = self.econ.thinker
        generated = thinker["examples"] * thinker["mean_tokens"]
        self.assertAlmostEqual(traces["seconds"], generated / 1000.0 / self.econ.gpus, places=3)

    def test_overrides_replace_the_assumptions_and_say_so(self) -> None:
        report = E.estimate(self.econ, tokens_per_example=371, examples=100_000, mfu=0.3, usd_per_gpu_hour=4.0, seeds=2)
        a = report["assumptions"]
        self.assertEqual((a["tokens_per_example"], a["examples"], a["mfu"], a["seeds"]), (371, 100_000, 0.3, 2))
        self.assertAlmostEqual(a["usd_per_gpu_second"], 4.0 / 3600, places=9)
        self.assertEqual(self.phases(report)["train"]["flops"], 6 * PARAMS * 371 * 100_000 * 2)
        self.assertEqual(sorted(a["overridden"]), ["examples", "mfu", "seeds", "tokens_per_example", "usd_per_gpu_hour"])

    def test_bad_overrides_are_refused(self) -> None:
        for bad in ({"mfu": 0}, {"mfu": 1.5}, {"gpus": 0}, {"examples": 0}, {"tokens_per_example": -3}, {"seeds": 0}, {"usd_per_gpu_hour": -1}, {"gpu": "H1000"}):
            with self.subTest(bad), self.assertRaises((ValueError, KeyError)):
                E.estimate(self.econ, **bad)

    def test_the_command_line_prints_json_and_text(self) -> None:
        done = subprocess.run([sys.executable, "-m", "decide.rental.estimate", "--profile", "econ_b200", "--json"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-400:])
        report = json.loads(done.stdout)
        self.assertEqual(report["status"], "UNVERIFIED")
        done = subprocess.run([sys.executable, "-m", "decide.rental.estimate", "--profile", "full_8xh100"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-400:])
        self.assertIn("ddp_does_not_fit", done.stdout)
        done = subprocess.run([sys.executable, "-m", "decide.rental.estimate", "--profile", "nope"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("econ_b200", done.stderr)

    def test_the_estimate_needs_no_torch(self) -> None:
        code = ("import sys; sys.modules['torch'] = None; sys.modules['transformers'] = None; sys.modules['pyarrow'] = None\n"
                "import decide.rental.estimate as E, decide.rental.profile as P\n"
                "print(E.estimate(P.load_profile('econ_b200'))['status'])\n")
        done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "UNVERIFIED"), done.stderr[-400:])


class ArchitectureParity(unittest.TestCase):
    """Optional: DECIDE_PPLX_CONFIG_DIR=<folder with config.json and model.safetensors.index.json of pplx-decider-v1.2>
    recounts the parameters and the LoRA units on the meta device (no weights are read)."""

    def test_counts_match_the_real_architecture(self) -> None:
        folder = os.environ.get("DECIDE_PPLX_CONFIG_DIR")
        if not folder:
            self.skipTest("set DECIDE_PPLX_CONFIG_DIR=<folder with the config.json of pplx-decider-v1.2>")
        try:
            import torch
            from transformers import AutoConfig
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

            from decide.train.model import LoraSpec
        except ImportError as exc:
            self.skipTest(f"torch/transformers are not usable here: {exc}")
        config = AutoConfig.from_pretrained(folder)
        with torch.device("meta"):
            model = Qwen3_5Model(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), HW.PARAMS)
        pattern = re.compile(LoraSpec().regex())
        units = sum(m.in_features + m.out_features for name, m in model.named_modules() if isinstance(m, torch.nn.Linear) and pattern.fullmatch(name))
        self.assertEqual(units, HW.LORA_UNITS_PER_RANK)
        index = json.loads((Path(folder) / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["metadata"]["total_parameters"], HW.PARAMS)
        self.assertEqual(index["metadata"]["total_size"], HW.WEIGHT_BYTES)


@needs_yaml
class ConfigOverlayContract(unittest.TestCase):
    path = ROOT / "decide" / "config.rental.yaml"

    def load(self):
        import yaml

        return yaml.safe_load(self.path.read_text(encoding="utf-8"))

    def test_the_overlay_has_the_rental_key_and_nothing_else(self) -> None:
        self.assertEqual(set(self.load()), {"rental"})

    def test_the_loader_refuses_other_keys(self) -> None:
        from decide.rental import config as RC

        self.assertEqual(set(RC.load_rental_config()), {"rental"})
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "config.rental.yaml"
            bad.write_text("rental: {profile: econ_b200}\nhub: {}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                RC.load_rental_config(bad)

    def test_the_default_profile_exists(self) -> None:
        self.assertIn(self.load()["rental"]["profile"], PRF.profile_names())

    def test_no_secret_value_is_written_only_the_names_of_variables(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        self.assertNotRegex(text, r"hf_[A-Za-z0-9]{10,}")
        rental = self.load()["rental"]
        self.assertEqual(rental["hub"]["token_env"], "HF_TOKEN")
        self.assertEqual(rental["thinker"]["api_key_env"], "THINKER_API_KEY")

    def test_the_models_are_pinned_to_a_commit(self) -> None:
        rental = self.load()["rental"]
        self.assertRegex(rental["start"]["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(rental["gates"]["baseline"], rental["start"])        # G4 compares with the model the day starts from

    def test_gate_g4_and_coverage_are_the_plans(self) -> None:
        gates = self.load()["rental"]["gates"]
        self.assertEqual(gates["g4_margin_points"], 1.5)
        self.assertEqual(gates["label_coverage"], 0.95)
        self.assertEqual(gates["proxy_scale"], 100)

    def test_uploads_are_private_unless_the_owner_changes_it(self) -> None:
        self.assertIs(self.load()["rental"]["hub"]["private"], True)

    def test_the_thinker_runs_only_behind_the_gt_gate(self) -> None:
        thinker = self.load()["rental"]["thinker"]
        self.assertEqual(thinker["enabled"], "auto")                          # auto: only with a GT report that passed
        self.assertTrue(thinker["thinking"])
        self.assertGreater(thinker["budget_per_question"], 0)

    def test_it_agrees_with_the_kit_config_when_there_is_one(self) -> None:
        import yaml

        kit = ROOT / "decide" / "config.yaml"
        if not kit.is_file():
            self.skipTest("decide/config.yaml (the kit) is not in this checkout")
        base = yaml.safe_load(kit.read_text(encoding="utf-8"))
        rental = self.load()["rental"]
        self.assertEqual(base["models"]["anchor"]["repo"], rental["start"]["repo"])
        self.assertEqual(base["hub"]["winner_repo"], rental["hub"]["weights_repo"])


if __name__ == "__main__":
    unittest.main()
