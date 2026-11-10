import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pocketgpt.checkpoint import load_checkpoint
from pocketgpt.config import ModelConfig
from pocketgpt.model import PocketGPT
from pocketgpt.release import export_inference
from pocketgpt.tokenizer import SmilesTokenizer


class InferenceReleaseTests(unittest.TestCase):
    def source(self, path, *, allow_legacy=True):
        tokenizer = SmilesTokenizer.build(["CCO", "c1ccccc1"])
        config = ModelConfig(vocab_size=len(tokenizer.tokens), n_layer=1, n_head=2,
                             n_embd=16, block_size=24, fragment_layers=1)
        model = PocketGPT(config)
        payload = {"format_version": 1, "global_step": 12,
                   "model": model.state_dict(), "model_config": config.to_dict(),
                   "vocabulary": tokenizer.tokens,
                   "trained_conditions": ["pocket", "properties", "fragment", "ifp"],
                   "trained_properties": ["logp", "qed", "tpsa", "sas"],
                   "pocket_encoder_signature": "legacy-precomputed-unverified",
                   "fragment_representation": "raw-smiles-v1",
                   "recipe": {"fragment_representation": "raw-smiles-v1",
                              "allow_unverified_pocket_embeddings": allow_legacy,
                              "train_data": "/private/example-data.jsonl", "pocket_encoder": {"repo": "/private/encoder"}},
                   "optimizer": {"private": "discard"}, "rng": {"private": "discard"},
                   "training_contract": {"path": "/private/data", "fragment_representation": "raw-smiles-v1"},
                   "warm_start": {"source": "/private/model.pt"}}
        torch.save(payload, path)
        return model, tokenizer

    def test_export_preserves_weights_vocabulary_and_strips_private_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.pt", Path(directory) / "model.pt"
            original, tokenizer = self.source(source)
            before = source.read_bytes()
            report = export_inference(source, output, "pocketgpt-test")
            restored, tokens, state = load_checkpoint(output)
            self.assertEqual(tokens.tokens, tokenizer.tokens)
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
                export_inference(source, output, "pocketgpt-test")

    def test_export_does_not_promote_unapproved_legacy_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            self.source(source, allow_legacy=False)
            with self.assertRaisesRegex(ValueError, "legacy"):
                export_inference(source, Path(directory) / "model.pt", "test")
            with self.assertRaisesRegex(ValueError, "public identifier"):
                export_inference(source, Path(directory) / "model.pt", "/private/path")

    def test_downloader_checks_bytes_hash_and_never_overwrites(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "download_model.py"
        spec = importlib.util.spec_from_file_location("download_model_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = b"synthetic-weights-only-for-a-unit-test"
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            output = Path(directory) / "weights.pt"
            manifest.write_text(json.dumps({"url": "https://github.com/qianrong0709/PocketGPT/releases/download/test/model.pt",
                                            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}))
            with patch.object(module, "urlopen", return_value=io.BytesIO(payload)) as opened:
                module.download_model(output, manifest_file=manifest)
                module.download_model(output, verify_only=True, manifest_file=manifest)
                self.assertEqual(opened.call_count, 1)
            output.write_bytes(b"keep-existing-user-file")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                module.download_model(output, manifest_file=manifest)
            self.assertEqual(output.read_bytes(), b"keep-existing-user-file")
            other = Path(directory) / "bad.pt"
            with patch.object(module, "urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    module.download_model(other, manifest_file=manifest)
            self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
