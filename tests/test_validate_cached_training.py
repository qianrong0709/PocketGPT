"""CPU-only real-file technical validator tests; no native encoder/network."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from pocketgpt.pocket import file_sha256
from pocketgpt.tokenizer import SmilesTokenizer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_cached_training.py"
SPEC = importlib.util.spec_from_file_location("pocketgpt_validate_cached_training", SCRIPT)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class ValidateCachedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "source-recipe.json"
        self.output = self.root / "new-technical-validation"
        self.rows = {}
        for split, molecules in (("train", ("C", "CC", "CO", "CCO")),
                                 ("valid", ("CCC", "CCCO"))):
            self.rows[split] = [{"smiles": molecule, "complex_id": split + str(index),
                "pocket_embedding": [float(index + 1) / 128] * 128,
                "properties": {"logp": 1.2, "qed": 0.5, "tpsa": 20, "sas": 2},
                "fragment": "c1cccc1", "fragment_representation": "raw-smiles-v1",
                "ifp": [1, 0, 0, 0, 0, 0]} for index, molecule in enumerate(molecules)]
        self.recipe = {"stage": "finetune", "conditions": "all", "split_group": "complex_id",
            "fragment_representation": "raw-smiles-v1", "allow_unverified_pocket_embeddings": True,
            "train_data": "train.jsonl", "valid_data": "valid.jsonl", "vocabulary": "vocabulary.json",
            "output_dir": "runs/DO-NOT-TOUCH-production", "condition_init_checkpoint": "missing-production-init.pt",
            "pocket_encoder": {"repo": "/does-not-exist", "python": "/not-a-python"},
            "model": {"block_size": 16, "n_layer": 1, "n_head": 2, "n_embd": 16,
                      "fragment_layers": 1, "dropout": 0.1},
            "training": {"epochs": 10, "batch_size": 2, "learning_rate": 0.001,
                "save_every_steps": 500, "cpu_threads": 1, "seed": 42, "device": "cuda",
                "condition_dropout": 0.35, "property_dropout": 0.2}}
        SmilesTokenizer.build([row["smiles"] for row in self.rows["train"]] + ["c1cccc1"]).save(
            self.root / "vocabulary.json")
        self.write_sources()

    def write_sources(self):
        for split, rows in self.rows.items():
            (self.root / (split + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.config.write_text(json.dumps(self.recipe))

    def run_validator(self, **kwargs):
        # Reject every possible research encoder process/profile entry.
        with patch("pocketgpt.pocket.subprocess.run", side_effect=AssertionError("No subprocess allowed")), \
             patch("pocketgpt.pocket.PocketResolver.profile", side_effect=AssertionError("No profile probing allowed")), \
             contextlib.redirect_stdout(io.StringIO()):
            return VALIDATOR.validate_cached_training(self.config, self.output, device="cpu", **kwargs)

    def test_real_cached_full_preflight_training_exact_resume_and_subsets_on_cpu(self):
        before = {name: file_sha256(self.root / name) for name in
                  ("source-recipe.json", "train.jsonl", "valid.jsonl", "vocabulary.json")}
        old_determinism = torch.are_deterministic_algorithms_enabled()
        report = self.run_validator(steps=5)
        self.assertEqual(report["status"], "technical_validation_passed")
        self.assertTrue(report["validation_only"])
        self.assertFalse(report["training_converged"])
        self.assertFalse(report["scientific_quality_validated"])
        self.assertFalse(report["raw_pdb_inference_validated"])
        self.assertFalse(report["encoder_executed"])
        self.assertFalse(report["legacy_encoder_profile_verified"])
        self.assertEqual(report["pocket_provenance_status"], "unverified-legacy")
        self.assertTrue(report["real_cached_rows_all_preflighted"])
        self.assertEqual(report["preflight"]["rows"], {"train": 4, "valid": 2})
        self.assertEqual(report["preflight"]["synthetic_smoke"], "not_requested")
        self.assertEqual(report["preflight"]["validation_scope"], "complex_id_grouped_NOT_protein_heldout")
        self.assertEqual(report["production_settings_preserved"]["batch_size"], 2)
        for key in ("model", "optimizer", "rng"):
            self.assertTrue(report["resume"][key]["exact_equal"], key)
            self.assertGreater(report["resume"][key]["compared_tensors"], 0)
        self.assertEqual(report["resume"]["from_step"], 5)
        self.assertEqual(report["resume"]["to_step"], 6)
        self.assertTrue(report["resume"]["step_metric_exact_equal"])
        self.assertTrue(all(report["parameter_updates_by_branch"][name]["updated"] for name in
                            ("pocket", "properties", "fragment", "ifp", "fusion", "decoder")))
        self.assertTrue(report["parameter_change_evidence_includes_possible_weight_decay"])
        probe = report["gradient_probe"]
        self.assertTrue(probe["all_four_projectors_finite_nonzero"])
        self.assertFalse(probe["optimizer_step_executed"])
        self.assertFalse(probe["weight_decay_applied"])
        self.assertTrue(probe["probe_model_is_separate_loaded_copy"])
        self.assertTrue(probe["performed_after_resume_reproducibility_comparison"])
        self.assertEqual(probe["condition_mask"], [True] * 4)
        for name in ("pocket", "properties", "fragment", "ifp"):
            self.assertTrue(probe["branches"][name]["all_present_gradients_finite"], name)
            self.assertGreater(probe["branches"][name]["nonzero_gradient_tensors"], 0, name)
            self.assertTrue(probe["branches"][name]["finite_nonzero_gradient_evidence"], name)
        self.assertTrue(report["original_data_hashes_unchanged"])
        self.assertEqual(before, {name: file_sha256(self.root / name) for name in before})
        self.assertFalse((self.root / "runs").exists())
        local_recipe = json.loads((self.output / "validation-recipe.json").read_text())
        self.assertNotIn("condition_init_checkpoint", local_recipe)
        self.assertNotIn("pocket_encoder", local_recipe)
        self.assertEqual(local_recipe["training"], self.recipe["training"])
        self.assertEqual(local_recipe["model"], self.recipe["model"])
        run = self.output / "runs" / "short-training"
        for name in ("before-resume.pt", "reference-resume.pt", "replay-resume.pt"):
            self.assertTrue((run / name).is_file())
        requests = report["generation"]["requests"]
        self.assertEqual(len(requests), 4)
        self.assertEqual(requests[0]["conditions"], ["pocket", "properties", "fragment", "ifp"])
        self.assertTrue(all(request["attempted"] == 2 for request in requests))
        self.assertEqual(requests[0]["fragment_match_evaluation"], "unavailable_invalid_fragment_query")
        self.assertFalse(report["generation"]["scientific_valid_rate_claimed"])
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), old_determinism)
        self.assertEqual(json.loads((self.output / "verification.json").read_text())["status"], report["status"])

    def test_all_valid_rows_are_preflighted_not_just_selected_generation_row(self):
        self.rows["valid"][-1]["smiles"] = "CCN"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "absent from training vocabulary"):
            self.run_validator(steps=2)
        report = json.loads((self.output / "verification.json").read_text())
        self.assertEqual(report["status"], "technical_validation_failed")
        self.assertFalse(report["real_cached_rows_all_preflighted"])
        self.assertFalse(report["encoder_executed"])
        self.assertTrue(report["original_data_hashes_unchanged"])
        self.assertFalse((self.output / "runs").exists())

    def test_raw_pocket_is_rejected_before_any_encoder_or_training_call(self):
        self.rows["valid"][-1]["pocket"] = {"pocket_pdb": "/private/not-read.pdb"}
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "refuses every raw pocket"):
            self.run_validator(steps=2)
        report = json.loads((self.output / "verification.json").read_text())
        self.assertFalse(report["encoder_executed"])
        self.assertFalse(report["real_cached_rows_all_preflighted"])
        self.assertFalse((self.output / "runs").exists())

    def test_existing_output_and_insufficient_schedule_are_not_overwritten(self):
        self.output.mkdir()
        marker = self.output / "preserve.txt"
        marker.write_text("existing work")
        with self.assertRaises(FileExistsError):
            self.run_validator(steps=1)
        self.assertEqual(marker.read_text(), "existing work")
        # New output with the original schedule too short must fail explicitly.
        self.output = self.root / "another-new-output"
        self.recipe["training"]["epochs"] = 1
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "epochs do not permit"):
            self.run_validator(steps=5)
        self.assertFalse((self.output / "runs").exists())

    def test_exact_comparison_includes_optimizer_scalars_and_every_tensor(self):
        left = {"state": {0: {"exp_avg": torch.tensor([1.0, 2.0]), "step": torch.tensor(3.)}},
                "param_groups": [{"lr": 0.1, "params": [0]}]}
        self.assertTrue(VALIDATOR.compare_exact(left, left)["exact_equal"])
        for changed in ({"state": {0: {"exp_avg": torch.tensor([1.0, 2.01]), "step": torch.tensor(3.)}},
                         "param_groups": [{"lr": 0.1, "params": [0]}]},
                        {"state": left["state"], "param_groups": [{"lr": 0.2, "params": [0]}]}):
            result = VALIDATOR.compare_exact(left, changed)
            self.assertFalse(result["exact_equal"])
            self.assertGreater(result["different_values"], 0)


if __name__ == "__main__":
    unittest.main()
