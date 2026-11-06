"""Explicit raw fragment semantics; synthetic fixtures, no research data/network."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from rdkit import Chem, rdBase

from pocketgpt.checkpoint import (checkpoint_fragment_representation, load_checkpoint,
                                  warm_start_conditioned)
from pocketgpt.conditional_preparation import prepare_condition_pretrain
from pocketgpt.config import DEFAULT_FRAGMENT_REPRESENTATION, validate_fragment_representation
from pocketgpt.data import BatchBuilder, fragment_text, inspect_manifest
from pocketgpt.generation import generate
from pocketgpt.model import PocketGPT
from pocketgpt.tokenizer import SmilesTokenizer
from pocketgpt.training import train


RAW = "raw-smiles-v1"
BAD_CHEMISTRY = "c1cccc1"  # Tokenizable, but an invalid aromatic five-carbon ring.


class RawFragmentRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tokenizer = SmilesTokenizer.build(["CCO", "OCC", "C", BAD_CHEMISTRY])

    def test_raw_text_is_preserved_without_canonicalizing_or_chemical_repair(self):
        with rdBase.BlockLogs():
            self.assertIsNone(Chem.MolFromSmiles(BAD_CHEMISTRY))
        self.assertEqual(fragment_text(BAD_CHEMISTRY, RAW), BAD_CHEMISTRY)
        self.assertEqual(fragment_text("  OCC  ", RAW), "  OCC  ")
        self.assertEqual(fragment_text("OCC"), "CCO")
        with rdBase.BlockLogs(), self.assertRaises(ValueError):
            fragment_text(BAD_CHEMISTRY)
        for invalid in (None, "", "   ", 1, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fragment_text(invalid, RAW)
        with self.assertRaisesRegex(ValueError, "unsupported tokenizer"):
            fragment_text("C!C", RAW)
        with self.assertRaises(ValueError):
            validate_fragment_representation("raw")

    def test_raw_labels_remain_active_and_default_canonical_stays_strict(self):
        row = {"smiles": "CCO", "fragment": BAD_CHEMISTRY,
               "fragment_representation": RAW, "fragment_chemical_valid": False}
        info = inspect_manifest([row], conditions=("fragment",), fragment_representation=RAW)
        self.assertIn("fragment", info["modalities"])
        _, _, batch = BatchBuilder(self.tokenizer, 16, ("fragment",),
                                   fragment_representation=RAW)([row, {"smiles": "C"}])
        self.assertEqual(batch["condition_mask"][:, 2].tolist(), [True, False])
        self.assertEqual(batch["fragment_tokens"][0].tolist(), self.tokenizer.encode(BAD_CHEMISTRY))
        with self.assertRaisesRegex(ValueError, "representation differs"):
            BatchBuilder(self.tokenizer, 16, ("fragment",))([row])
        with self.assertRaisesRegex(ValueError, "representation differs"):
            inspect_manifest([row], conditions=("fragment",))
        # Disabled fragments neither parse their value nor their representation.
        inactive = dict(row, fragment="C!C", fragment_representation="unknown")
        BatchBuilder(self.tokenizer, 16, ("properties",))([inactive])

    def test_raw_mode_still_rejects_unknown_tokens_context_and_invalid_parent(self):
        with self.assertRaisesRegex(ValueError, "absent from training vocabulary"):
            BatchBuilder(self.tokenizer, 16, ("fragment",), fragment_representation=RAW)(
                [{"smiles": "CCO", "fragment": "N"}])
        with self.assertRaisesRegex(ValueError, "Fragment exceeds context"):
            BatchBuilder(self.tokenizer, 3, ("fragment",), fragment_representation=RAW)(
                [{"smiles": "CCO", "fragment": BAD_CHEMISTRY}])
        with rdBase.BlockLogs(), self.assertRaisesRegex(ValueError, "Invalid SMILES"):
            inspect_manifest([{"smiles": BAD_CHEMISTRY, "fragment": "C"}],
                             conditions=("fragment",), fragment_representation=RAW)

    def make_csv(self, path, *, invalid_parent=False, unsupported_fragment=False):
        labels = (BAD_CHEMISTRY, "OCC", "  OCC  ", "C", "")
        rows = [["C" * i, "-1.25", "0", "0.5", "20", "2", labels[(i - 1) % len(labels)]]
                for i in range(1, 21)]
        if invalid_parent:
            rows[0][0] = BAD_CHEMISTRY
        if unsupported_fragment:
            rows[0][-1] = "C!C"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("smiles", "LogS", "logp", "qed", "tpsa", "sas", "frag"))
            writer.writerows(rows)
        return rows

    def test_preparation_keeps_every_raw_label_and_reports_chemistry_failures(self):
        source, output = self.root / "raw.csv", self.root / "prepared"
        originals = self.make_csv(source)
        before = source.read_bytes()
        report = prepare_condition_pretrain(source, output, validation_fraction=0.5,
                                             fragment_representation=RAW)
        self.assertEqual(source.read_bytes(), before)
        rows = [json.loads(line) for split in ("train", "valid")
                for line in (output / (split + ".jsonl")).read_text().splitlines()]
        self.assertEqual(len(rows), 20)
        for row in rows:
            original = originals[row["source_row"] - 1][-1]
            if original:
                self.assertEqual(row["fragment"], original)
                self.assertEqual(row["fragment_representation"], RAW)
                self.assertEqual(row["fragment_chemical_valid"], original != BAD_CHEMISTRY)
            else:
                self.assertNotIn("fragment", row)
        self.assertEqual(report["fragment_representation"], RAW)
        self.assertTrue(report["fragment_original_strings_preserved"])
        self.assertEqual(report["fragment_chemical_validation"]["invalid_rows_retained"], 4)
        self.assertEqual(report["fragment_chemical_validation"]["invalid_source_record_examples"], [1, 6, 11, 16])
        self.assertEqual(report["missing"]["fragment"], 4)
        self.assertFalse(report["fragment_labels_masked"])
        self.assertFalse(report["invalid_rows_skipped"])
        vocab = SmilesTokenizer.load(output / "vocabulary.json")
        self.assertEqual(vocab.decode(vocab.encode(BAD_CHEMISTRY)), BAD_CHEMISTRY)

    def test_raw_preparation_does_not_relax_parent_tokenizer_or_length_checks(self):
        for options, message, context in (({"invalid_parent": True}, "Invalid parent SMILES", 256),
                                           ({"unsupported_fragment": True}, "Unsupported raw fragment", 256),
                                           ({}, "no truncation or filtering", 3)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.make_csv(root / "source.csv", **options)
                with self.assertRaisesRegex(ValueError, message):
                    prepare_condition_pretrain(root / "source.csv", root / "result",
                                               validation_fraction=0.5, block_size=context,
                                               fragment_representation=RAW)
                self.assertFalse((root / "result").exists())

    def write_training(self, output_name):
        records = {"train": ["C", "CC", "CCO"], "valid": ["CO", "CCCO"]}
        for split, molecules in records.items():
            (self.root / (split + ".jsonl")).write_text("".join(json.dumps({
                "smiles": molecule, "properties": {"qed": 0.5},
                "fragment": BAD_CHEMISTRY, "fragment_representation": RAW}) + "\n"
                for molecule in molecules))
        recipe = {"stage": "condition_pretrain", "conditions": ["properties", "fragment"],
                  "fragment_representation": RAW, "train_data": "train.jsonl", "valid_data": "valid.jsonl",
                  "output_dir": output_name, "model": {"block_size": 16, "n_layer": 1, "n_head": 2,
                  "n_embd": 16, "fragment_layers": 1, "dropout": 0.1}, "training": {
                  "epochs": 2, "batch_size": 1, "learning_rate": 0.001,
                  "condition_dropout": 0.35, "property_dropout": 0.2,
                  "save_every_steps": 1, "cpu_threads": 1, "seed": 42, "device": "cpu"}}
        path = self.root / (output_name + ".json")
        path.write_text(json.dumps(recipe))
        return path, recipe

    def test_raw_training_resume_metadata_and_semantic_initialization_guard(self):
        recipe, _ = self.write_training("resumed")
        uninterrupted, _ = self.write_training("continuous")
        with contextlib.redirect_stdout(io.StringIO()):
            checkpoint = train(recipe, max_steps=2)
            checkpoint = train(recipe, resume=checkpoint, max_steps=4)
            other = train(uninterrupted, max_steps=4)
        model, tokenizer, state = load_checkpoint(checkpoint)
        _, _, continuous = load_checkpoint(other)
        self.assertEqual(state["fragment_representation"], RAW)
        self.assertEqual(state["training_contract"]["fragment_representation"], RAW)
        self.assertEqual(state["trained_conditions"], ["properties", "fragment"])
        self.assertEqual(state["global_step"], 4)
        for name, value in state["model"].items():
            self.assertTrue(torch.equal(value, continuous["model"][name]), name)
        target = PocketGPT(model.config)
        with self.assertRaisesRegex(ValueError, "silently change fragment representation"):
            warm_start_conditioned(target, checkpoint, tokenizer, ("properties", "fragment"))
        report = warm_start_conditioned(target, checkpoint, tokenizer, ("properties", "fragment"),
                                        fragment_representation=RAW)
        self.assertEqual(report["fragment_representation"], RAW)
        self.assertFalse(report["optimizer_restored"])

    def test_generation_marks_invalid_raw_query_unavailable_not_false(self):
        recipe, _ = self.write_training("generation")
        with contextlib.redirect_stdout(io.StringIO()):
            checkpoint = train(recipe, max_steps=1)
        _, tokenizer, _ = load_checkpoint(checkpoint)
        samples = [torch.full((2, 1), tokenizer.stoi["C"], dtype=torch.long),
                   torch.full((2, 1), tokenizer.eos_id, dtype=torch.long)]
        output = self.root / "generated.csv"
        with patch("pocketgpt.generation.torch.multinomial", side_effect=samples):
            summary = generate(checkpoint, {"fragment": BAD_CHEMISTRY}, output,
                               conditions="fragment", count=2, batch_size=2, max_tokens=4)
        with output.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(row["valid"] == "True" for row in rows))
        self.assertTrue(all(row["fragment_match"] == "unavailable" for row in rows))
        self.assertTrue(all(row["fragment_match_status"] == "unavailable_invalid_fragment_query" for row in rows))
        self.assertEqual(summary["fragment_representation"], RAW)
        self.assertFalse(summary["fragment_query_chemical_valid"])
        self.assertEqual(summary["fragment_match_evaluation"], "unavailable_invalid_fragment_query")

    def test_old_checkpoint_default_and_conflicting_policy_are_explicit(self):
        self.assertEqual(checkpoint_fragment_representation({"recipe": {}}), DEFAULT_FRAGMENT_REPRESENTATION)
        with self.assertRaisesRegex(ValueError, "conflicts with its recipe"):
            checkpoint_fragment_representation({"fragment_representation": RAW, "recipe": {}})
        with self.assertRaisesRegex(ValueError, "conflicts with its training contract"):
            checkpoint_fragment_representation({"fragment_representation": RAW,
                                                 "recipe": {"fragment_representation": RAW},
                                                 "training_contract": {"fragment_representation": DEFAULT_FRAGMENT_REPRESENTATION}})


if __name__ == "__main__":
    unittest.main()
