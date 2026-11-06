import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/server_preflight.py"
SPEC = importlib.util.spec_from_file_location("pocketgpt_server_preflight", SCRIPT)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class ServerPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.train = self.project / "train.jsonl"
        self.valid = self.project / "valid.jsonl"
        self.train.write_text(json.dumps({"smiles": "CCO"}) + "\n")
        self.valid.write_text(json.dumps({"smiles": "CO"}) + "\n")
        self.config = self.project / "recipe.json"
        self.recipe = {"stage": "pretrain", "conditions": "none",
                       "train_data": "train.jsonl", "valid_data": "valid.jsonl",
                       "output_dir": "runs/new-test", "model": {"block_size": 8,
                       "n_layer": 1, "n_head": 2, "n_embd": 16, "fragment_layers": 1,
                       "dropout": 0.0}, "training": {"epochs": 1, "batch_size": 1,
                       "cpu_threads": 1, "device": "cpu"}}

    def check(self, **kwargs):
        self.config.write_text(json.dumps(self.recipe))
        with patch.dict("os.environ", {"SLURM_CPUS_PER_TASK": "4"}):
            return PREFLIGHT.check_recipe(self.config, device="cpu", project_root=self.project,
                                          **kwargs)

    def test_cpu_smoke_passes_without_creating_training_outputs(self):
        result = self.check()
        self.assertEqual(result["status"], "preflight_passed")
        self.assertEqual(result["synthetic_smoke"], "finite_forward_backward")
        self.assertEqual(result["architecture_version"], 2)
        self.assertFalse(result["research_encoder_executed"])
        self.assertFalse((self.project / "runs").exists())

    def test_wrong_environment_and_unallocated_gpu_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Wrong Python environment"):
            self.check(expected_python_prefix=self.project / "wrong-env", run_smoke=False)
        self.config.write_text(json.dumps(self.recipe))
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                PREFLIGHT.check_recipe(self.config, device="cuda", project_root=self.project)

    def test_run_outside_project_or_existing_artifacts_are_rejected(self):
        self.recipe["output_dir"] = "../outside-run"
        with self.assertRaisesRegex(ValueError, "dedicated child"):
            self.check(run_smoke=False)
        self.recipe["output_dir"] = "runs/old-run"
        run = self.project / "runs/old-run"
        run.mkdir(parents=True)
        (run / "metrics.jsonl").write_text("preserve me\n")
        with self.assertRaisesRegex(FileExistsError, "empty/new run"):
            self.check(run_smoke=False)
        self.assertEqual((run / "metrics.jsonl").read_text(), "preserve me\n")

    def test_canonical_molecule_overlap_is_rejected(self):
        self.valid.write_text(json.dumps({"smiles": "OCC"}) + "\n")
        with self.assertRaisesRegex(ValueError, "leakage"):
            self.check(run_smoke=False)

    def test_condition_pretrain_accepts_property_fragment_without_protein(self):
        self.recipe.update(stage="condition_pretrain", conditions=["properties", "fragment"])
        self.train.write_text(json.dumps({"smiles": "CCO", "properties": {"qed": 0.5}, "fragment": "CO"}) + "\n")
        self.valid.write_text(json.dumps({"smiles": "CO", "properties": {"qed": 0.6}, "fragment": "C"}) + "\n")
        result = self.check()
        self.assertEqual(result["stage"], "condition_pretrain")
        self.assertEqual(result["conditions"], ["properties", "fragment"])
        self.assertIsNone(result["pocket_contract"])
        self.assertFalse((self.project / "runs").exists())
        self.valid.write_text(json.dumps({"smiles": "OCC", "properties": {"qed": 0.6}, "fragment": "C"}) + "\n")
        with self.assertRaisesRegex(ValueError, "leakage"):
            self.check(run_smoke=False)

    def test_condition_pretrain_rejects_pocket_ifp_empty_and_missing_labels(self):
        for conditions in (["pocket"], ["ifp"], []):
            self.recipe.update(stage="condition_pretrain", conditions=conditions)
            with self.subTest(conditions=conditions), self.assertRaises(ValueError):
                self.check(run_smoke=False)
        self.recipe.update(stage="condition_pretrain", conditions=["properties", "fragment"])
        with self.assertRaisesRegex(ValueError, "lacks effective labels"):
            self.check(run_smoke=False)

    def test_cached_complex_raw_fragment_preflight_uses_no_encoder(self):
        self.recipe.update(stage="finetune", conditions="all", split_group="complex_id",
                           fragment_representation="raw-smiles-v1",
                           allow_unverified_pocket_embeddings=True)
        self.recipe["model"]["block_size"] = 16
        for path, smiles, identifier in ((self.train, "CCO", "complex_train"),
                                         (self.valid, "CO", "complex_valid")):
            path.write_text(json.dumps({"smiles": smiles, "complex_id": identifier,
                "pocket_embedding": [0.0] * 128, "properties": {"qed": 0.5},
                "fragment": "c1nccc1", "fragment_representation": "raw-smiles-v1",
                "ifp": [1, 0, 0, 0, 0, 0]}) + "\n")
        result = self.check(run_smoke=False)
        self.assertEqual(result["split_identity"], "complex_id")
        self.assertEqual(result["validation_scope"], "complex_id_grouped_NOT_protein_heldout")
        self.assertEqual(result["fragment_representation"], "raw-smiles-v1")
        self.assertFalse(result["research_encoder_executed"])
        self.assertEqual(result["pocket_contract"]["sources"], ["embedding"])
        self.assertFalse((self.project / "runs").exists())

    def test_context_overflow_and_unseen_tokens_are_rejected(self):
        self.recipe["model"]["block_size"] = 3
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            self.check(run_smoke=False)
        self.recipe["model"]["block_size"] = 8
        self.valid.write_text(json.dumps({"smiles": "CCN"}) + "\n")
        with self.assertRaisesRegex(ValueError, "absent from training vocabulary"):
            self.check(run_smoke=False)

    def test_empty_properties_and_old_architecture_are_rejected(self):
        self.recipe.update(stage="finetune", conditions=["properties"])
        self.train.write_text(json.dumps({"smiles": "CCO", "protein_id": "train", "properties": {"qed": None}}) + "\n")
        self.valid.write_text(json.dumps({"smiles": "CO", "protein_id": "valid", "properties": {"qed": None}}) + "\n")
        with self.assertRaisesRegex(ValueError, "lacks effective labels"):
            self.check(run_smoke=False)
        self.recipe.update(stage="pretrain", conditions="none")
        self.recipe["model"].update(architecture_version=1, fragment_encoder="mean-v1", fragment_layers=0)
        with self.assertRaisesRegex(ValueError, "architecture v2"):
            self.check(run_smoke=False)


if __name__ == "__main__":
    unittest.main()
