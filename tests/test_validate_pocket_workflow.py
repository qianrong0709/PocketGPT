"""Synthetic adapter fixtures only; these tests never run a research encoder."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pocketgpt.pocket import content_hash, file_sha256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_pocket_workflow.py"
SPEC = importlib.util.spec_from_file_location("validate_pocket_workflow_tested", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FixtureResolver:
    """A deterministic fixture, not a replacement for actual integration QA."""
    calls = 0
    dimension = 128

    def __init__(self, config=None):
        self.config = config or {}

    def signature(self, refresh=False):
        return "deeplytough:v2:synthetic-unit-fixture"

    def row_signature(self, row):
        return self.signature()

    def input_fingerprint(self, row):
        return {"encoder": self.signature(),
                "pocket": file_sha256(row["pocket"]["pocket_pdb"]),
                "protein": file_sha256(row["pocket"]["protein_pdb"]),
                "weights": file_sha256(self.config["weights"])}

    def resolve(self, row):
        fingerprint = self.input_fingerprint(row)
        path = Path(self.config["cache"]) / content_hash(fingerprint)
        output = path / "embedding.npy"
        if not output.exists():
            type(self).calls += 1
            path.mkdir(parents=True)
            vector = np.arange(type(self).dimension, dtype=np.float32) / 128
            np.save(output, vector)
            (path / "provenance.json").write_text(json.dumps(fingerprint))
            (path / "encoder.log").write_text("SYNTHETIC UNIT TEST FIXTURE; no encoder executed\n")
        return np.load(output, allow_pickle=False)


class PocketWorkflowValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pocket = self.root / "pocket.pdb"
        self.protein = self.root / "protein.pdb"
        self.weights = self.root / "fixture.weights"
        for path, text in ((self.pocket, "UNIT POCKET\n"), (self.protein, "UNIT PROTEIN\n"),
                           (self.weights, "UNIT WEIGHTS\n")):
            path.write_text(text)
        self.config = {"repo": str(self.root / "repo"), "python": "fixture-only",
                       "weights": str(self.weights), "mgltools_prefix": "fixture-only",
                       "cache": str(self.root / "never-touch-this-existing-cache")}
        self.output = self.root / "validation"
        self.settings = {"n_layer": 1, "n_head": 2, "n_embd": 16, "block_size": 8,
                         "fragment_layers": 1, "dropout": 0.1}
        self.request = {"smiles": "CCO", "properties": {"qed": 0.5, "logp": 0.1},
                        "fragment": "CO", "ifp": [1, 0, 0, 0, 0, 0],
                        "complex_id": "synthetic-unit-fixture"}
        FixtureResolver.calls = 0
        FixtureResolver.dimension = 128

    def validate(self, **kwargs):
        defaults = dict(pocket=self.pocket, protein=self.protein, encoder_config=self.config,
                        output_dir=self.output, request=self.request, device="cpu", steps=3,
                        model_settings=self.settings, count=2, max_tokens=4)
        defaults.update(kwargs)
        with patch.object(MODULE, "PocketResolver", FixtureResolver), \
                patch("pocketgpt.data.PocketResolver", FixtureResolver):
            return MODULE.validate_workflow(**defaults)

    def test_encoding_training_resume_and_explicit_all_four_generation_interfaces(self):
        before = {path: file_sha256(path) for path in (self.pocket, self.protein, self.weights)}
        result = self.validate()
        self.assertEqual(result["status"], "technical_smoke_passed")
        self.assertEqual(FixtureResolver.calls, 1)
        self.assertTrue(result["encoding"]["second_resolution_identical"])
        self.assertTrue(result["encoding"]["cache_artifacts_unchanged_on_second_resolution"])
        self.assertTrue(result["training"]["resume_continuation_model_optimizer_rng_exact"])
        self.assertEqual(result["training"]["steps"], 3)
        self.assertEqual(set(result["training"]["enabled_conditions"]), {"pocket", "properties", "fragment", "ifp"})
        for name in ("pocket", "properties", "fragment", "ifp", "gate", "decoder"):
            self.assertTrue(result["training"]["parameter_groups_changed"][name])
        samples = result["generation"]["interface_checks"]
        self.assertEqual(set(samples), {"pocket", "pocket_properties", "pocket_fragment", "all_supplied", "all_four"})
        self.assertTrue(all(value["attempted"] == 2 for value in samples.values()))
        self.assertFalse(result["full_corpus_training_completed"])
        self.assertFalse(result["scientific_control_efficacy_evaluated"])
        self.assertFalse(result["protein_identity_invented"])
        self.assertEqual({path: file_sha256(path) for path in before}, before)
        self.assertFalse(Path(self.config["cache"]).exists())
        state = torch.load(self.output / "smoke-only.pt", weights_only=True)
        self.assertTrue(state["validation_only"])
        self.assertNotIn("protein_id", state["recipe"])
        self.assertEqual(state["global_step"], 3)
        self.assertEqual(json.loads((self.output / "verification.json").read_text())["status"], result["status"])

    def test_missing_ifp_is_not_fabricated_and_all_four_is_skipped(self):
        request = {key: value for key, value in self.request.items() if key != "ifp"}
        result = self.validate(request=request)
        self.assertNotIn("ifp", result["training"]["enabled_conditions"])
        self.assertFalse(result["training"]["parameter_groups_changed"]["ifp"])
        self.assertIn("all_four", result["generation"]["skipped"])
        self.assertNotIn("all_four", result["generation"]["interface_checks"])

    def test_encode_only_needs_no_target_and_never_builds_a_generator(self):
        with patch.object(MODULE, "PocketGPT", side_effect=AssertionError("must not construct model")):
            result = self.validate(request=None, encode_only=True)
        self.assertEqual(result["status"], "encoding_passed")
        self.assertIsNone(result["training"])
        self.assertIsNone(result["generation"])
        self.assertFalse((self.output / "smoke-only.pt").exists())

    def test_existing_output_and_absent_target_fail_before_encoding(self):
        self.output.mkdir()
        (self.output / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(FileExistsError, "new output directory"):
            self.validate()
        self.assertEqual((self.output / "keep.txt").read_text(), "keep")
        with self.assertRaisesRegex(ValueError, "explicit real target"):
            self.validate(output_dir=self.root / "other", request=None)
        self.assertFalse((self.root / "other").exists())
        self.assertEqual(FixtureResolver.calls, 0)

    def test_request_cannot_override_pocket_or_keep_invalid_fragment(self):
        for update, pattern in (({"pocket_embedding": [0.0] * 128}, "override"),
                                ({"pocket": {"pocket_pdb": "another.pdb"}}, "override"),
                                ({"fragment": "not-a-smiles"}, "fragment label is invalid")):
            with self.subTest(update=list(update)), self.assertRaisesRegex(ValueError, pattern):
                self.validate(request={**self.request, **update})
        self.assertFalse(self.output.exists())
        self.assertEqual(FixtureResolver.calls, 0)

    def test_bad_encoder_output_records_failure_not_success(self):
        FixtureResolver.dimension = 127
        with self.assertRaisesRegex(ValueError, "128 finite"):
            self.validate()
        failure = json.loads((self.output / "failure.json").read_text())
        self.assertEqual(failure["stage"], "real_pocket_encoding")
        self.assertEqual(failure["status"], "failed")
        self.assertFalse((self.output / "verification.json").exists())

    def test_unallocated_cuda_does_not_execute_encoder(self):
        with patch.dict("os.environ", {}, clear=True), patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "independent Slurm"):
                self.validate(device="cuda")
        self.assertEqual(FixtureResolver.calls, 0)
        self.assertFalse(self.output.exists())

    def test_resume_guard_refuses_contract_changes(self):
        self.validate()
        state = torch.load(self.output / "before-resume.pt", weights_only=True)
        changed = {**state["validation_contract"], "request_sha256": "changed"}
        with self.assertRaisesRegex(ValueError, "contract changed"):
            MODULE._resume_smoke(self.output / "before-resume.pt", changed, "cpu")


if __name__ == "__main__":
    unittest.main()
