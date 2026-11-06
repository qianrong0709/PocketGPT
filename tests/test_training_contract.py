import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pocketgpt.checkpoint import load_checkpoint, warm_start_base
from pocketgpt.config import ModelConfig
from pocketgpt.model import PocketGPT
from pocketgpt.pocket import PocketResolver, file_sha256
from pocketgpt.tokenizer import SmilesTokenizer
from pocketgpt.training import train


class TrainingContractTests(unittest.TestCase):
    def setup_run(self, root, train_row, valid_row, conditions):
        for name, row in (("train", train_row), ("valid", valid_row)):
            (root / (name + ".jsonl")).write_text(json.dumps(row) + "\n")
        recipe = {"stage": "finetune", "conditions": conditions,
                  "train_data": "train.jsonl", "valid_data": "valid.jsonl", "output_dir": "run",
                  "model": {"block_size": 16, "n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0},
                  "training": {"epochs": 2, "batch_size": 1, "learning_rate": 0.001,
                               "cpu_threads": 1, "seed": 12}}
        return recipe

    def save_recipe(self, root, recipe):
        path = root / "recipe.json"
        path.write_text(json.dumps(recipe))
        return path

    def test_train_validation_encoder_mismatch_rejected_before_run_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = {"smiles": "CCO", "protein_id": "a", "pocket_embedding": [0.1] * 128,
                     "pocket_encoder_signature": "encoder-A"}
            second = dict(first, smiles="CCN", protein_id="b", pocket_encoder_signature="encoder-B")
            recipe = self.setup_run(root, first, second, ["pocket"])
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                train(self.save_recipe(root, recipe), max_steps=1)
            self.assertFalse((root / "run").exists())

    def test_unverified_vectors_require_explicit_legacy_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = {"smiles": "CCO", "protein_id": "a", "pocket_embedding": [0.1] * 128}
            recipe = self.setup_run(root, first, dict(first, smiles="CCN", protein_id="b"), ["pocket"])
            with self.assertRaisesRegex(ValueError, "explicit allow_unverified"):
                train(self.save_recipe(root, recipe), max_steps=1)

    def test_no_property_supervision_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = {"smiles": "CCO", "protein_id": "a", "properties": {"qed": None}}
            recipe = self.setup_run(root, first, dict(first, smiles="CCN", protein_id="b"), ["properties"])
            with self.assertRaisesRegex(ValueError, "Training labels absent"):
                train(self.save_recipe(root, recipe), max_steps=1)

    def test_legacy_mode_cannot_be_paired_with_raw_encoder_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = {"smiles": "CCO", "protein_id": "a", "pocket_embedding": [0.1] * 128}
            recipe = self.setup_run(root, first, dict(first, smiles="CCN", protein_id="b"), ["pocket"])
            recipe.update(allow_unverified_pocket_embeddings=True, pocket_encoder={"weights": "/absent"})
            with self.assertRaisesRegex(ValueError, "embedding-only mode"):
                train(self.save_recipe(root, recipe), max_steps=1)

    def test_smiles_alias_pretrain_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = self.setup_run(root, {"smiles": "CCO"}, {"smiles": "OCC"}, [])
            recipe["stage"] = "pretrain"
            with self.assertRaisesRegex(ValueError, "leakage"):
                train(self.save_recipe(root, recipe), max_steps=1)

    def test_raw_weight_or_structure_change_cannot_exactly_resume(self):
        for changed in ("weights.bin", "train.pdb"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ("weights.bin", "train.pdb", "valid.pdb"):
                    (root / name).write_text("synthetic-initial-" + name)
                first = {"smiles": "CCO", "protein_id": "a", "pocket": {"pocket_pdb": str(root / "train.pdb")}}
                second = {"smiles": "CCN", "protein_id": "b", "pocket": {"pocket_pdb": str(root / "valid.pdb")}}
                recipe = self.setup_run(root, first, second, ["pocket"])
                recipe["pocket_encoder"] = {"weights": str(root / "weights.bin")}
                config = self.save_recipe(root, recipe)
                def profile(resolver, refresh=False):
                    return {"synthetic_weight_sha256": file_sha256(resolver.config["weights"])}
                with patch.object(PocketResolver, "profile", profile), patch.object(
                        PocketResolver, "resolve", return_value=np.ones(128, dtype=np.float32)):
                    checkpoint = train(config, max_steps=1)
                    (root / changed).write_text("synthetic-changed-" + changed)
                    with self.assertRaisesRegex(ValueError, "Exact resume requires unchanged"):
                        train(config, resume=checkpoint)

    def test_legacy_architecture_requires_explicit_migration(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = SmilesTokenizer.build(["CCO", "CCN"])
            config = ModelConfig(vocab_size=len(tokenizer.tokens), block_size=16, n_layer=1,
                                 n_head=2, n_embd=16, architecture_version=1,
                                 fragment_encoder="mean-v1", fragment_layers=0)
            model = PocketGPT(config)
            old_config = config.to_dict()
            for key in ("architecture_version", "fragment_encoder", "fragment_layers"):
                old_config.pop(key)
            path = Path(tmp) / "old.pt"
            torch.save({"format_version": 1, "vocabulary": tokenizer.tokens,
                        "model_config": old_config, "model": model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "Legacy fragment architecture"):
                load_checkpoint(path)
            new_model = PocketGPT(ModelConfig(vocab_size=len(tokenizer.tokens), block_size=16,
                                              n_layer=1, n_head=2, n_embd=16))
            report = warm_start_base(new_model, path, tokenizer)
            self.assertFalse(report["exact_resume"])
            self.assertTrue(report["new_conditioners_require_training"])
            torch.testing.assert_close(new_model.tok_emb.weight, model.tok_emb.weight, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
