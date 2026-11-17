"""Inference regression tests; synthetic vectors are not encoder verification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from pocketgpt.generation import generate
from pocketgpt.pocket import LEGACY_SIGNATURE, PocketResolver, file_sha256
from pocketgpt.pocket_inference import check_pocket_compatibility, load_pocket_vector

ROOT = Path(__file__).resolve().parents[1]
SIGNATURE = "deeplytough:v2:" + "a" * 64


class PocketInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "fixture.pt"
        self.checkpoint.write_bytes(b"synthetic identity only")
        self.state = {"pocket_encoder_signature": LEGACY_SIGNATURE,
                      "recipe": {"allow_unverified_pocket_embeddings": True}}
        self.request = {"pocket_embedding": (np.arange(128, dtype=np.float32) / 128).tolist(),
                        "pocket_encoder_signature": SIGNATURE}
        self.record = {"format_version": 1, "kind": "pocketgpt-encoder-compatibility",
                       "checkpoint_sha256": file_sha256(self.checkpoint),
                       "checkpoint_pocket_signature": LEGACY_SIGNATURE,
                       "encoder_signature": SIGNATURE, "encoder_weights_sha256": "b" * 64,
                       "validation": {"method": "paired-pocket-vectors-v1", "sample_count": 3,
                                      "allclose": True, "atol": 1e-6, "rtol": 1e-5,
                                      "max_abs_error": 0., "min_cosine_similarity": 1.}}

    def check(self, request=None, state=None, **kwargs):
        return check_pocket_compatibility(self.checkpoint, state or self.state,
                                         self.request if request is None else request, PocketResolver(), **kwargs)

    def save_record(self, record=None):
        path = self.root / "comparison.json"
        path.write_text(json.dumps(self.record if record is None else record))
        return path

    def test_legacy_requires_explicit_unverified_choice_and_preserves_identity(self):
        before = copy.deepcopy(self.request)
        with self.assertRaisesRegex(ValueError, "--allow-unverified-pocket"):
            self.check()
        with self.assertWarns(UserWarning):
            result = self.check(allow_unverified=True)
        self.assertEqual(result["status"], "legacy-user-accepted")
        self.assertEqual(result["input_signature"], SIGNATURE)
        self.assertFalse(result["feature_space_verified"])
        self.assertEqual(before, self.request)

    def test_bare_vector_does_not_gain_legacy_identity(self):
        request = {"pocket_embedding": self.request["pocket_embedding"]}
        with self.assertRaises(ValueError):
            self.check(request)
        with self.assertWarns(UserWarning):
            result = self.check(request, allow_unverified=True)
        self.assertIsNone(result["input_signature"])
        self.assertNotIn("pocket_encoder_signature", request)

    def test_explicit_historical_declaration_is_backward_compatible_not_verified(self):
        request = dict(self.request, pocket_encoder_signature=LEGACY_SIGNATURE)
        with self.assertWarns(UserWarning):
            result = self.check(request)
        self.assertEqual(result["status"], "legacy-declared")
        self.assertFalse(result["feature_space_verified"])

    def test_exact_profile_and_concrete_profile_conflict(self):
        state = dict(self.state, pocket_encoder_signature=SIGNATURE)
        self.assertEqual(self.check(state=state)["status"], "profile-matched")
        request = dict(self.request, pocket_encoder_signature="another-encoder")
        with self.assertRaises(ValueError):
            self.check(request, state, allow_unverified=True)

    def test_checkpoint_explicitly_forbidding_legacy_remains_forbidden(self):
        state = dict(self.state, recipe={}, training_contract={"version": 1})
        with self.assertRaisesRegex(ValueError, "does not explicitly allow"):
            self.check(state=state, allow_unverified=True)

    def test_valid_comparison_and_manifest_discovery(self):
        result = self.check(compatibility=self.save_record())
        self.assertEqual(result["status"], "reference-verified")
        self.assertTrue(result["feature_space_verified"])
        self.checkpoint.with_suffix(".json").write_text(json.dumps({"pocket_compatibility": [self.record]}))
        self.assertEqual(self.check()["status"], "reference-verified")

    def test_record_must_bind_checkpoint_bytes_expected_identity_and_input_identity(self):
        for field in ("checkpoint_sha256", "checkpoint_pocket_signature", "encoder_signature"):
            record = copy.deepcopy(self.record)
            record[field] = "deeplytough:v2:" + "c" * 64 if field == "encoder_signature" else "c" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "does not match"):
                self.check(compatibility=self.save_record(record), allow_unverified=True)
        path = self.save_record()
        self.checkpoint.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.check(compatibility=path)

    def test_explicit_wrong_record_is_not_ignored_for_exact_signature(self):
        state = dict(self.state, pocket_encoder_signature=SIGNATURE)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.check(state=state, compatibility=self.save_record())

    def test_malformed_comparison_cannot_be_accepted_by_allow_flag(self):
        cases = (("sample_count", 2), ("sample_count", True), ("allclose", False),
                 ("atol", 1.), ("rtol", float("nan")), ("method", "cosine-only"),
                 ("min_cosine_similarity", 2), ("max_abs_error", -1))
        for key, value in cases:
            record = copy.deepcopy(self.record)
            record["validation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(compatibility=self.save_record(record), allow_unverified=True)

    def test_invalid_vectors_never_reach_model(self):
        for vector in ([0.] * 127, [float("nan")] * 128, [float("inf")] * 128):
            with self.assertRaises(ValueError):
                self.check(dict(self.request, pocket_embedding=vector), allow_unverified=True)

    def test_configured_encoder_conflict_is_not_hidden_by_allow_flag(self):
        resolver = PocketResolver({"synthetic": True})
        with patch.object(resolver, "signature", return_value="different"), self.assertRaisesRegex(ValueError, "does not match --encoder-config"):
            check_pocket_compatibility(self.checkpoint, self.state, self.request, resolver, allow_unverified=True)

    def test_vector_loader_preserves_values_and_explicit_signature(self):
        path = self.root / "vector.json"
        path.write_text(json.dumps(self.request))
        self.assertEqual(load_pocket_vector(path), self.request)
        path.write_text(json.dumps(self.request["pocket_embedding"]))
        self.assertEqual(load_pocket_vector(path), {"pocket_embedding": self.request["pocket_embedding"]})

    def test_numpy_cache_loader_checks_sidecar_hash(self):
        path = self.root / "embedding.npy"
        np.save(path, np.asarray(self.request["pocket_embedding"], dtype=np.float32), allow_pickle=False)
        sidecar = path.with_name("provenance.json")
        sidecar.write_text(json.dumps({"input": {"encoder": SIGNATURE}, "output_sha256": file_sha256(path)}))
        self.assertEqual(load_pocket_vector(path), self.request)
        np.save(path, np.ones(128, dtype=np.float32), allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_pocket_vector(path)

    def test_published_model_accepts_explicit_choice_and_keeps_metadata(self):
        checkpoint = ROOT / "models" / "pocketgpt-v1.pt"
        original = file_sha256(checkpoint)
        with self.assertWarns(UserWarning):
            summary = generate(checkpoint, self.request, self.root / "generation.csv", conditions="pocket",
                               count=1, max_tokens=2, allow_unverified_pocket=True)
        self.assertEqual(summary["attempted"], 1)
        self.assertEqual(summary["pocket_input"]["encoder"], SIGNATURE)
        self.assertEqual(summary["pocket_provenance_status"], "legacy-user-accepted")
        self.assertEqual(file_sha256(checkpoint), original)

    def test_no_pocket_does_not_inspect_encoder_or_compatibility_file(self):
        summary = generate(ROOT / "models/pocketgpt-v1.pt", {"properties": {"qed": .7}},
                           self.root / "no-pocket.csv", conditions="properties", count=1, max_tokens=2,
                           encoder={"weights": "/absent.pt"}, pocket_compatibility="/absent.json")
        self.assertEqual(summary["pocket_provenance_status"], "unused")

    def test_published_checkpoint_all_condition_subsets(self):
        from pocketgpt.checkpoint import load_checkpoint
        from pocketgpt.config import CONDITIONS
        checkpoint = ROOT / "models/pocketgpt-v1.pt"
        loaded = load_checkpoint(checkpoint)
        request = dict(self.request, properties={"qed": .7}, fragment="c1ccccc1", ifp=[1, 0, 0, 0, 0, 0])
        with patch("pocketgpt.generation.load_checkpoint", return_value=loaded), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for mask in range(16):
                selected = [name for i, name in enumerate(CONDITIONS) if mask & (1 << i)]
                with self.subTest(conditions=selected):
                    result = generate(checkpoint, request, self.root / f"subset-{mask}.csv",
                                      conditions=selected, count=1, max_tokens=2, allow_unverified_pocket=True)
                    self.assertEqual(result["conditions"], selected)
                    self.assertEqual(result["attempted"], 1)

    def test_raw_and_cached_vectors_reach_projector_identically(self):
        # Replace only the encoder in this integration test. It does NOT test
        # real HTMD preprocessing or equivalence to the training CSV.
        from pocketgpt.checkpoint import load_checkpoint
        checkpoint = ROOT / "models/pocketgpt-v1.pt"
        model, tokenizer, state = load_checkpoint(checkpoint)
        observed = []
        hook = model.pocket_projector.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
        self.addCleanup(hook.remove)
        pocket = self.root / "synthetic.pdb"
        pocket.write_text("SYNTHETIC TEST INPUT\n")
        resolver = PocketResolver({"synthetic": True})
        vector = np.asarray(self.request["pocket_embedding"], dtype=np.float32)
        raw = {"pocket": {"protein_pdb": str(pocket), "pocket_pdb": str(pocket)}}
        with patch.object(resolver, "signature", return_value=SIGNATURE), \
                patch.object(resolver, "resolve", return_value=vector), \
                patch("pocketgpt.data.PocketResolver", return_value=resolver), \
                patch("pocketgpt.generation.load_checkpoint", return_value=(model, tokenizer, state)):
            for i, request in enumerate((raw, self.request)):
                with self.assertWarns(UserWarning):
                    generate(checkpoint, request, self.root / f"route-{i}.csv", conditions="pocket",
                             count=1, max_tokens=2, seed=13, allow_unverified_pocket=True)
        self.assertEqual((self.root / "route-0.csv").read_text(), (self.root / "route-1.csv").read_text())
        self.assertTrue(observed)
        for tensor in observed:
            np.testing.assert_array_equal(tensor.numpy()[0], vector)


if __name__ == "__main__":
    unittest.main()
