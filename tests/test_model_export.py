import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from pocketgpt.checkpoint import load_checkpoint
from pocketgpt.config import ModelConfig
from pocketgpt.model import PocketGPT
from pocketgpt.release import export_inference
from pocketgpt.tokenizer import SmilesTokenizer


class ModelExportTests(unittest.TestCase):
    def source(self, path, allow_legacy=True):
        tokenizer = SmilesTokenizer.build(["CCO", "c1ccccc1"])
        config = ModelConfig(vocab_size=len(tokenizer.tokens), n_layer=1, n_head=2,
                             n_embd=16, block_size=24, fragment_layers=1)
        model = PocketGPT(config)
        torch.save({"format_version": 1, "global_step": 12,
                    "model": model.state_dict(), "model_config": config.to_dict(),
                    "vocabulary": tokenizer.tokens,
                    "trained_conditions": ["pocket", "properties", "fragment", "ifp"],
                    "trained_properties": ["logp", "qed", "tpsa", "sas"],
                    "pocket_encoder_signature": "legacy-precomputed-unverified",
                    "fragment_representation": "raw-smiles-v1",
                    "recipe": {"fragment_representation": "raw-smiles-v1",
                               "allow_unverified_pocket_embeddings": allow_legacy,
                               "train_data": "/private/example-data.jsonl"},
                    "optimizer": {"private": "discard"}, "rng": {"private": "discard"},
                    "training_contract": {"fragment_representation": "raw-smiles-v1", "path": "/private/data"},
                    "warm_start": {"source": "/private/initializer.pt"}}, path)
        return model, tokenizer

    def test_lossless_export_removes_private_training_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.pt", Path(directory) / "model.pt"
            original, vocabulary = self.source(source)
            before = source.read_bytes()
            report = export_inference(source, output, "test-model")
            restored, tokens, state = load_checkpoint(output)
            self.assertEqual(tokens.tokens, vocabulary.tokens)
            for key, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            self.assertEqual(state["artifact_type"], "inference")
            for key in ("optimizer", "rng", "training_contract", "warm_start"):
                self.assertNotIn(key, state)
            self.assertEqual(set(state["recipe"]), {"fragment_representation", "allow_unverified_pocket_embeddings"})
            self.assertNotIn(b"/private/", output.read_bytes())
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(report["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                export_inference(source, output, "test-model")

    def test_legacy_permission_and_public_identifier_are_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.pt", Path(directory) / "model.pt"
            self.source(source, allow_legacy=False)
            with self.assertRaisesRegex(ValueError, "legacy"):
                export_inference(source, output, "test")
            with self.assertRaisesRegex(ValueError, "public identifier"):
                export_inference(source, output, "/private/path")


if __name__ == "__main__":
    unittest.main()
