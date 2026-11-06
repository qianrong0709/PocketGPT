"""Explicit complex-group finetune is not protein-held-out validation."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from pocketgpt.checkpoint import load_checkpoint, save_checkpoint
from pocketgpt.config import CONDITIONS
from pocketgpt.data import inspect_manifest
from pocketgpt.training import train, training_stage_contract


def record(complex_id, fragment="n1cnnc1"):
    return {"smiles": "CCO", "complex_id": complex_id,
            "properties": {"logp": 0, "qed": 0.6, "tpsa": 20, "sas": 2},
            "pocket_embedding": [0.1] * 128, "ifp": [1, 1, 0, 0, 0, 0],
            "fragment": fragment, "fragment_representation": "raw-smiles-v1"}


class ComplexSplitTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        torch.set_num_threads(1)

    def recipe(self, run="run", **overrides):
        train_rows = [record("train-1"), record("train-1", "N"),
                      record("train-2"), record("train-2", "CO")]
        valid_rows = [record("valid-1"), record("valid-1", "C")]
        for name, rows in (("train", train_rows), ("valid", valid_rows)):
            (self.root / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
        value = {"stage": "finetune", "conditions": "all", "split_group": "complex_id",
                 "fragment_representation": "raw-smiles-v1", "allow_unverified_pocket_embeddings": True,
                 "train_data": "train.jsonl", "valid_data": "valid.jsonl", "output_dir": run,
                 "model": {"block_size": 16, "n_layer": 1, "n_head": 2, "n_embd": 16,
                           "fragment_layers": 1, "dropout": 0.1},
                 "training": {"epochs": 2, "batch_size": 2, "learning_rate": 0.001,
                              "seed": 5, "save_every_steps": 1, "log_every_steps": 1,
                              "cpu_threads": 1, "device": "cpu"}}
        value.update(overrides)
        path = self.root / (run + ".json")
        path.write_text(json.dumps(value))
        return path

    def train(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return train(*args, **kwargs)

    def test_manifest_uses_only_explicit_group_field_and_keeps_default(self):
        rows = [record("complex-A", "CO"), record("complex-A", "C")]
        info = inspect_manifest(rows, conditional=True, conditions=CONDITIONS,
                                fragment_representation="raw-smiles-v1", split_group="complex_id")
        self.assertEqual(info["groups"], {"complex-A"})
        self.assertFalse(any("protein_id" in row for row in rows))
        with self.assertRaisesRegex(ValueError, "protein_id"):
            inspect_manifest(rows, conditional=True, conditions=CONDITIONS,
                             fragment_representation="raw-smiles-v1")
        for value in (None, "", "   ", 1, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "complex_id"):
                inspect_manifest([record(value)], conditional=True, conditions=CONDITIONS,
                                 fragment_representation="raw-smiles-v1", split_group="complex_id")

    def test_stage_contract_records_semantics_and_rejects_wrong_stage(self):
        value = training_stage_contract("finetune", "all", split_group="complex_id")
        self.assertEqual(value["split_identity"], "complex_id")
        self.assertEqual(value["validation_scope"], "complex_id_grouped_NOT_protein_heldout")
        self.assertEqual(training_stage_contract("finetune", "all")["split_identity"], "protein_id")
        for stage in ("pretrain", "condition_pretrain"):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "only available for finetune"):
                training_stage_contract(stage, "none" if stage == "pretrain" else "fragment", split_group="complex_id")
        with self.assertRaisesRegex(ValueError, "split_group"):
            training_stage_contract("finetune", "all", split_group="pdb_prefix")

    def test_complex_overlap_rejected_even_when_fragments_differ(self):
        path = self.recipe()
        (self.root / "valid.jsonl").write_text(json.dumps(record("train-1", "C")) + "\n")
        with self.assertRaisesRegex(ValueError, "complex_id overlap"):
            self.train(path, max_steps=1)
        self.assertFalse((self.root / "run").exists())

    def test_explicit_complex_finetune_resume_is_exact_without_protein_ids(self):
        full = self.recipe("full")
        complete = self.train(full, max_steps=3)
        partial = self.recipe("resumed")
        stopped = self.train(partial, max_steps=2)
        resumed = self.train(partial, resume=stopped, max_steps=3)
        model_a, _, state_a = load_checkpoint(complete)
        model_b, _, state_b = load_checkpoint(resumed)
        self.assertEqual(state_b["global_step"], 3)
        self.assertEqual(state_b["split_group"], "complex_id")
        self.assertEqual(state_b["validation_scope"], "complex_id_grouped_NOT_protein_heldout")
        self.assertEqual(state_b["training_contract"]["training_stage"], training_stage_contract("finetune", "all", "complex_id"))
        self.assertEqual(state_b["fragment_representation"], "raw-smiles-v1")
        self.assertEqual(state_b["trained_conditions"], list(CONDITIONS))
        for key, value in model_a.state_dict().items():
            torch.testing.assert_close(value, model_b.state_dict()[key], rtol=0, atol=0)
        for name in ("train", "valid"):
            self.assertNotIn("protein_id", (self.root / (name + ".jsonl")).read_text())
        metrics = [json.loads(line) for line in (self.root / "resumed/metrics.jsonl").read_text().splitlines()]
        self.assertTrue(any(row.get("validation_scope") == "complex_id_grouped_NOT_protein_heldout" for row in metrics))

    def test_resume_contract_cannot_silently_switch_grouping_semantics(self):
        path = self.recipe()
        checkpoint = self.train(path, max_steps=1)
        _, _, saved = load_checkpoint(checkpoint)
        saved["training_contract"]["training_stage"]["split_identity"] = "protein_id"
        save_checkpoint(checkpoint, saved)
        with self.assertRaisesRegex(ValueError, "Exact resume requires unchanged"):
            self.train(path, resume=checkpoint, max_steps=2)


if __name__ == "__main__":
    unittest.main()
