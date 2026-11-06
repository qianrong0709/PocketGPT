"""Synthetic-only regression tests for effective labels and canonical inputs."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from pocketgpt.data import (BatchBuilder, canonical_fragment, canonical_smiles,
                            inspect_manifest, property_vector)
from pocketgpt.preparation import prepare_dataset
from pocketgpt.tokenizer import SmilesTokenizer


class EffectivePropertyTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SmilesTokenizer.build(["CCO"])

    def test_all_missing_properties_are_not_supervision(self):
        for values in (None, {}, {"qed": None}, {"logp": "", "qed": "  "}):
            with self.subTest(values=values):
                row = {"smiles": "CCO", "protein_id": "synthetic-1", "properties": values}
                info = inspect_manifest([row], conditional=True)
                _, _, batch = BatchBuilder(self.tokenizer, 8, ["properties"])([row])
                self.assertEqual(info["properties"], set())
                self.assertNotIn("properties", info["modalities"])
                self.assertFalse(bool(batch["property_mask"].any()))
                self.assertFalse(bool(batch["condition_mask"].any()))

    def test_valid_zero_is_supervision_but_absent_values_are_not(self):
        row = {"smiles": "CCO", "properties": {"logp": 0.0, "qed": 0, "tpsa": None}}
        info = inspect_manifest([row])
        _, _, batch = BatchBuilder(self.tokenizer, 8, ["properties"])([row])
        self.assertEqual(info["properties"], {"logp", "qed"})
        self.assertIn("properties", info["modalities"])
        self.assertEqual(batch["property_mask"].tolist(), [[True, True, False, False]])
        self.assertTrue(bool(batch["condition_mask"][0, 1]))
        self.assertEqual(batch["properties"].tolist(), [[0.0] * 4])

    def test_inspection_and_batch_reject_same_invalid_property_values(self):
        cases = [{"qed": float("nan")}, {"logp": float("inf")}, {"qed": 1.1},
                 {"tpsa": -1}, {"sas": 0}, {"sas": 11}, {"qed": True},
                 {"qed": []}, {"logp": 1e300}, {"unknown": None}, []]
        for values in cases:
            row = {"smiles": "CCO", "properties": values}
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    property_vector(values)
                with self.assertRaises(ValueError):
                    inspect_manifest([row], conditions=["properties"])
                with self.assertRaises(ValueError):
                    BatchBuilder(self.tokenizer, 8, ["properties"])([row])

    def test_ignored_metadata_is_not_read_or_validated(self):
        row = {"smiles": "CCO", "properties": {"qed": float("nan")},
               "fragment": "not a molecule", "pocket": {"pocket_pdb": "/not-present.pdb"},
               "pocket_embedding": [float("nan")], "ifp": [100]}
        info = inspect_manifest([row], conditions=())
        self.assertEqual(info["modalities"], set())
        self.assertEqual(info["pocket_sources"], set())
        _, _, batch = BatchBuilder(self.tokenizer, 8, ())([row])
        self.assertFalse(bool(batch["condition_mask"].any()))


class CanonicalInputTests(unittest.TestCase):
    def test_molecule_aliases_share_leakage_identity(self):
        train = inspect_manifest([{"smiles": "CCO"}], conditions=())
        valid = inspect_manifest([{"smiles": "OCC"}], conditions=())
        self.assertEqual(train["molecules"] & valid["molecules"], {"CCO"})

    def test_fragment_aliases_use_same_tokens_at_training_and_inference(self):
        tokenizer = SmilesTokenizer.build(["CCO", canonical_fragment("OC(C)C")])
        builder = BatchBuilder(tokenizer, 16, ["fragment"])
        _, _, first = builder([{"smiles": "CCO", "fragment": "OC(C)C"}])
        _, _, second = builder([{"fragment": "CC(C)O"}], language_targets=False)
        torch.testing.assert_close(first["fragment_tokens"], second["fragment_tokens"], rtol=0, atol=0)
        torch.testing.assert_close(first["condition_mask"], second["condition_mask"], rtol=0, atol=0)

    def test_fragment_length_and_vocabulary_use_canonical_form(self):
        # Raw C(C)(C)O is 8 tokens, but canonical CC(C)O is only 6.
        fragment = canonical_fragment("C(C)(C)O")
        tokenizer = SmilesTokenizer.build([fragment])
        size = len(tokenizer.encode(fragment))
        _, _, batch = BatchBuilder(tokenizer, size, ["fragment"])(
            [{"fragment": "C(C)(C)O"}], language_targets=False)
        self.assertEqual(batch["fragment_tokens"].shape[1], size)
        with self.assertRaisesRegex(ValueError, "Fragment exceeds context"):
            BatchBuilder(tokenizer, size - 1, ["fragment"])(
                [{"fragment": "C(C)(C)O"}], language_targets=False)

    def test_fragment_stereochemistry_and_attachment_atoms_are_retained(self):
        first = canonical_fragment("C[C@H](O)F")
        second = canonical_fragment("C[C@@H](O)F")
        self.assertNotEqual(first, second)
        self.assertIn("*", canonical_fragment("[*]CO"))
        self.assertNotEqual(canonical_fragment("CCO"), canonical_fragment("COC"))

    def test_empty_and_non_string_smiles_are_rejected(self):
        for value in (None, "", "   ", 0, [], "this-is-not-smiles"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_smiles(value)


class PocketInputMetadataTests(unittest.TestCase):
    def test_conflicting_sources_fail_before_any_pdb_is_read(self):
        row = {"smiles": "CCO", "pocket_embedding": [0.0] * 128,
               "pocket": {"pocket_pdb": "/absent-synthetic.pdb"}}
        with self.assertRaisesRegex(ValueError, "[Cc]onflict"):
            inspect_manifest([row], conditions=["pocket"])
        tokenizer = SmilesTokenizer.build(["CCO"])
        with self.assertRaisesRegex(ValueError, "[Cc]onflict"):
            BatchBuilder(tokenizer, 8, ["pocket"])([row])

    def test_manifest_reports_raw_and_embedding_sources_without_opening_pdb(self):
        rows = [{"smiles": "CCO", "pocket_embedding": [0.0] * 128,
                 "pocket_encoder_signature": "synthetic-profile"},
                {"smiles": "CCN", "pocket": {"pocket_pdb": "/absent-synthetic.pdb"}}]
        info = inspect_manifest(rows, conditions=["pocket"])
        self.assertEqual(info["pocket_sources"], {"raw", "embedding"})
        self.assertEqual(info["encoder_signatures"], {"synthetic-profile"})


class PreparationContractTests(unittest.TestCase):
    def test_preparation_keeps_unlabelled_rows_and_canonicalizes_fragment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.jsonl"
            rows = [{"smiles": "OCC", "protein_id": f"synthetic-{i}",
                     "properties": {"qed": None}, "fragment": "OC(C)C"} for i in range(20)]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = prepare_dataset(source, root / "out", "finetune", validation_fraction=0.5)
            prepared = [json.loads(line) for split in ("train", "valid")
                        for line in (root / "out" / (split + ".jsonl")).read_text().splitlines()]
            self.assertEqual(sum(report["counts"].values()), 20)
            self.assertEqual(report["rejected"], {})
            self.assertEqual(len(prepared), 20)
            self.assertTrue(all(row["properties"] == {"qed": None} for row in prepared))
            self.assertTrue(all(row["smiles"] == "CCO" for row in prepared))
            self.assertTrue(all(row["fragment"] == canonical_fragment("OC(C)C") for row in prepared))

    def test_preparation_rejects_invalid_properties_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.jsonl"
            source.write_text(json.dumps({"smiles": "CCO", "protein_id": "synthetic-1",
                                          "properties": {"qed": 2}}) + "\n")
            with self.assertRaisesRegex(ValueError, "Invalid property"):
                prepare_dataset(source, root / "out", "finetune")

    def test_preparation_rejects_conflicting_pocket_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.jsonl"
            source.write_text(json.dumps({"smiles": "CCO", "protein_id": "synthetic-1",
                                          "pocket_embedding": [0.0] * 128,
                                          "pocket": {"pocket_pdb": "absent.pdb"}}) + "\n")
            with self.assertRaisesRegex(ValueError, "[Cc]onflict"):
                prepare_dataset(source, root / "out", "finetune")


if __name__ == "__main__":
    unittest.main()
