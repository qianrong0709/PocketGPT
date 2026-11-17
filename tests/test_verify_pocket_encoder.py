"""Synthetic-only tests; these fixtures do not validate any real encoder."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pocketgpt.pocket import file_sha256


spec = importlib.util.spec_from_file_location(
    "verify_pocket_encoder_tool", Path(__file__).resolve().parents[1] / "scripts/verify_pocket_encoder.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class SyntheticResolver:
    """No DeeplyTough process or network access; returns explicit test arrays."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.encoder_signature = "deeplytough:v2:" + "a" * 64
        self.weights_sha256 = "b" * 64
        self.calls = []
        self.after_resolve = None

    def signature(self, refresh=False):
        return self.encoder_signature

    def profile(self):
        return {"weights_sha256": self.weights_sha256}

    def input_fingerprint(self, row):
        return {
            "source": "raw-pdb", "encoder": self.signature(),
            "protein_sha256": file_sha256(row["pocket"]["protein_pdb"]),
            "pocket_sha256": file_sha256(row["pocket"]["pocket_pdb"]),
            "context": "full-protein",
        }

    def resolve(self, row):
        if set(row) != {"pocket"}:
            raise AssertionError("Reference vectors must never be sent to the encoder")
        self.calls.append(copy.deepcopy(row))
        vector = self.vectors[Path(row["pocket"]["pocket_pdb"]).name]
        if self.after_resolve:
            self.after_resolve(row)
        return vector


class VerifyPocketEncoderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "synthetic.pt"
        self.checkpoint.write_bytes(b"synthetic loader mock fixture, not a real checkpoint")
        self.references = self.root / "references.jsonl"
        self.output = self.root / "approved.json"
        self.rows, self.vectors = [], {}
        for index in range(3):
            protein, pocket = f"protein{index}.pdb", f"pocket{index}.pdb"
            (self.root / protein).write_text(f"SYNTHETIC protein {index}\n")
            (self.root / pocket).write_text(f"SYNTHETIC pocket {index}\n")
            vector = np.linspace(0.25, 1.25, 128, dtype=np.float32) + index
            self.vectors[pocket] = vector.copy()
            self.rows.append({"pocket": {"protein_pdb": protein, "pocket_pdb": pocket},
                              "pocket_embedding": vector.tolist()})
        self.write_references()
        self.model = SimpleNamespace(config=SimpleNamespace(pocket_dim=128))
        self.state = {"trained_conditions": ["pocket"],
                      "pocket_encoder_signature": "legacy-precomputed-unverified"}
        self.resolver = SyntheticResolver(self.vectors)
        self.loader_patch = patch.object(tool, "load_checkpoint", side_effect=lambda _: (self.model, None, self.state))
        self.resolver_patch = patch.object(tool, "PocketResolver", return_value=self.resolver)
        self.loader = self.loader_patch.start()
        self.resolver_factory = self.resolver_patch.start()
        self.addCleanup(self.loader_patch.stop)
        self.addCleanup(self.resolver_patch.stop)

    def write_references(self):
        self.references.write_text("\n".join(json.dumps(row) for row in self.rows) + "\n")

    def verify(self, **kwargs):
        return tool.verify_pocket_encoder(self.checkpoint, {"synthetic": True},
                                          self.references, self.output, **kwargs)

    def assert_rejected(self, expression, exception=ValueError, **kwargs):
        with self.assertRaisesRegex(exception, expression):
            self.verify(**kwargs)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".pocket-compatibility-*")), [])

    def test_success_record_contains_only_comparison_metadata(self):
        result = self.verify()
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertEqual(result["checkpoint_sha256"], file_sha256(self.checkpoint))
        self.assertEqual(result["checkpoint_pocket_signature"], self.state["pocket_encoder_signature"])
        self.assertEqual(result["encoder_signature"], self.resolver.signature())
        self.assertEqual(result["encoder_weights_sha256"], self.resolver.weights_sha256)
        self.assertEqual(result["validation"]["sample_count"], 3)
        self.assertTrue(result["validation"]["allclose"])
        self.assertEqual(result["validation"]["max_abs_error"], 0)
        self.assertEqual(result["validation"]["max_norm_error"], 0)
        self.assertAlmostEqual(result["validation"]["min_cosine_similarity"], 1)
        self.assertEqual(result["validation"]["atol"], 1e-6)
        self.assertEqual(result["validation"]["rtol"], 1e-5)
        self.assertEqual(len(self.resolver.calls), 3)
        self.assertTrue(all(set(row) == {"pocket"} for row in self.resolver.calls))
        serialized = self.output.read_text()
        for private_value in (str(self.root), "pocket0.pdb", "pocket_embedding",
                              file_sha256(self.root / "protein0.pdb"), file_sha256(self.references)):
            self.assertNotIn(private_value, serialized)

    def test_no_reference_normalization_and_same_direction_is_not_enough(self):
        self.vectors["pocket0.pdb"] *= 2
        self.assert_rejected("Pocket vector mismatch")

    def test_small_nonzero_error_uses_explicit_tolerances(self):
        self.vectors["pocket0.pdb"][0] += 5e-5
        self.assert_rejected("Pocket vector mismatch")
        result = self.verify(atol=1e-4, rtol=0)
        self.assertGreater(result["validation"]["max_abs_error"], 0)
        self.assertGreater(result["validation"]["max_norm_error"], 0)

    def test_bad_tolerances_cannot_approve(self):
        for parameter, values in (("atol", [-1, float("inf"), float("nan"), True, 1.0001e-4]),
                                  ("rtol", [-1, float("inf"), float("nan"), True, 1.0001e-3])):
            for value in values:
                with self.subTest(parameter=parameter, value=value):
                    self.assert_rejected(parameter, **{parameter: value})

    def test_at_least_three_distinct_reference_inputs_required(self):
        self.rows = self.rows[:2]
        self.write_references()
        self.assert_rejected("At least three")
        self.rows = []
        self.write_references()
        self.assert_rejected("At least three")

    def test_duplicate_content_cannot_count_as_another_pocket(self):
        self.rows[2] = copy.deepcopy(self.rows[0])
        self.rows[2]["pocket"]["protein_pdb"] = "./protein0.pdb"
        self.write_references()
        self.assert_rejected("Duplicate pocket")
        self.assertEqual(self.resolver.calls, [])

    def test_invalid_reference_vectors_are_rejected(self):
        for vector in ([0] * 128, [1] * 127, [float("nan")] * 128, [float("inf")] * 128):
            with self.subTest(vector=vector[:1], length=len(vector)):
                self.rows[0]["pocket_embedding"] = vector
                self.write_references()
                self.assert_rejected("zero vector|128 finite numbers")

    def test_invalid_encoder_vectors_are_rejected(self):
        for vector in (np.zeros(128), np.ones(127), np.full(128, np.nan), np.full(128, np.inf)):
            with self.subTest(shape=vector.shape):
                self.vectors["pocket0.pdb"] = vector
                self.assert_rejected("zero vector|128 finite numbers")

    def test_missing_files_or_pocket_paths_are_rejected(self):
        self.rows[0]["pocket"].pop("protein_pdb")
        self.write_references()
        self.assert_rejected("requires protein_pdb")
        self.rows[0]["pocket"]["protein_pdb"] = "absent.pdb"
        self.write_references()
        self.assert_rejected("does not exist", FileNotFoundError)

    def test_checkpoint_must_have_pocket_training_signature_and_dimension(self):
        self.state["trained_conditions"] = ["properties"]
        self.assert_rejected("not trained with pocket")
        self.state["trained_conditions"] = ["pocket"]
        self.state.pop("pocket_encoder_signature")
        self.assert_rejected("no pocket encoder signature")
        self.state["pocket_encoder_signature"] = "legacy-precomputed-unverified"
        self.model.config.pocket_dim = 64
        self.assert_rejected("128-dimensional")

    def test_encoder_must_have_concrete_profile_and_weight_hash(self):
        self.resolver.encoder_signature = "legacy-precomputed-unverified"
        self.assert_rejected("concrete DeeplyTough")
        self.resolver.encoder_signature = "deeplytough:v2:" + "a" * 64
        self.resolver.weights_sha256 = "unavailable"
        self.assert_rejected("valid weights checksum")

    def test_existing_output_and_symlink_are_never_overwritten(self):
        self.output.write_text("original file")
        with self.assertRaises(FileExistsError):
            self.verify()
        self.assertEqual(self.output.read_text(), "original file")
        self.output.unlink()
        self.output.symlink_to(self.root / "absent.json")
        with self.assertRaises(FileExistsError):
            self.verify()
        self.assertTrue(self.output.is_symlink())

    def test_checkpoint_or_reference_mutation_cannot_approve(self):
        for path in (self.checkpoint, self.references):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                self.resolver.after_resolve = lambda _, path=path: path.write_bytes(original + b"\n")
                self.assert_rejected("Checkpoint or reference records changed", RuntimeError)
                path.write_bytes(original)

    def test_current_input_mutation_cannot_approve(self):
        self.resolver.after_resolve = lambda row: Path(row["pocket"]["pocket_pdb"]).write_text("changed")
        self.assert_rejected("Pocket inputs changed", RuntimeError)

    def test_already_compared_input_mutation_cannot_approve(self):
        def change_previous(row):
            if Path(row["pocket"]["pocket_pdb"]).name == "pocket2.pdb":
                (self.root / "pocket0.pdb").write_text("changed after comparison")
        self.resolver.after_resolve = change_previous
        self.assert_rejected("Pocket inputs changed", RuntimeError)

    def test_encoder_or_weights_mutation_cannot_approve(self):
        self.resolver.after_resolve = lambda _: setattr(self.resolver, "encoder_signature", "deeplytough:v2:" + "c" * 64)
        self.assert_rejected("Pocket inputs changed|Encoder weights", RuntimeError)
        self.resolver.encoder_signature = "deeplytough:v2:" + "a" * 64
        self.resolver.after_resolve = lambda _: setattr(self.resolver, "weights_sha256", "c" * 64)
        self.assert_rejected("Encoder weights", RuntimeError)

    def test_publish_is_exclusive_even_when_destination_appears_late(self):
        original_link = tool.os.link
        def raced_link(source, destination):
            Path(destination).write_text("concurrent writer")
            return original_link(source, destination)
        with patch.object(tool.os, "link", side_effect=raced_link), self.assertRaises(FileExistsError):
            self.verify()
        self.assertEqual(self.output.read_text(), "concurrent writer")
        self.assertEqual(list(self.root.glob(".pocket-compatibility-*")), [])


if __name__ == "__main__":
    unittest.main()
