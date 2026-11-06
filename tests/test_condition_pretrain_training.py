import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from pocketgpt.checkpoint import (load_checkpoint, seed_all, warm_start_base,
                                  warm_start_conditioned)
from pocketgpt.config import CONDITIONS, PROPERTIES, ModelConfig
from pocketgpt.generation import generate
from pocketgpt.model import PocketGPT
from pocketgpt.pocket import PocketResolver
from pocketgpt.tokenizer import SmilesTokenizer
from pocketgpt.training import train, training_stage_contract


class ConditionPretrainTrainingTests(unittest.TestCase):
    """Tiny synthetic ligand data only: no research data, encoder or GPU needed."""

    def write_data(self, root):
        source = root / "synthetic-v2.csv"
        molecules = ["CCO", "CCN", "CCC", "CO", "CNC", "C=O", "CCCO", "CNCO"]
        with source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["smiles", "LogS", *PROPERTIES, "frag"])
            writer.writeheader()
            for index, smiles in enumerate(molecules):
                writer.writerow({"smiles": smiles, "LogS": -2.0, "logp": index / 10,
                                 "qed": 0.6, "tpsa": 40.0, "sas": 2.0, "frag": "CO"})
        # Native-schema fixture conversion, independent of the production CSV preparer.
        with source.open(newline="") as handle:
            rows = [{"smiles": row["smiles"],
                     "properties": {name: float(row[name]) for name in PROPERTIES},
                     "fragment": row["frag"], "metadata": {"LogS": float(row["LogS"])}}
                    for row in csv.DictReader(handle)]
        self.write_rows(root / "train.jsonl", rows[:6])
        self.write_rows(root / "valid.jsonl", rows[6:])
        return rows

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    @staticmethod
    def recipe(output="run", conditions="properties,fragment"):
        return {"stage": "condition_pretrain", "conditions": conditions,
                "train_data": "train.jsonl", "valid_data": "valid.jsonl", "output_dir": output,
                "model": {"block_size": 16, "n_layer": 1, "n_head": 2, "n_embd": 16,
                          "dropout": 0.1, "fragment_layers": 1},
                "training": {"epochs": 2, "batch_size": 2, "learning_rate": 0.001,
                             "seed": 7, "condition_dropout": 0.35, "property_dropout": 0.2,
                             "cpu_threads": 1}}

    @staticmethod
    def run_train(root, recipe, **kwargs):
        config = root / (recipe["output_dir"] + ".json")
        config.write_text(json.dumps(recipe))
        with contextlib.redirect_stdout(io.StringIO()):
            checkpoint = train(config, **kwargs)
        return checkpoint, config

    def test_stage_policy_is_explicit_and_cannot_activate_pocket_or_ifp(self):
        for conditions in ("properties", "fragment", "properties,fragment"):
            contract = training_stage_contract("condition_pretrain", conditions)
            self.assertEqual(contract["split_identity"], "canonical_smiles")
        for conditions in ("none", "all", "pocket", "ifp", "properties,ifp"):
            with self.assertRaisesRegex(ValueError, "condition_pretrain"):
                training_stage_contract("condition_pretrain", conditions)
        with self.assertRaises(ValueError):
            training_stage_contract("pretrain", "properties")
        with self.assertRaises(ValueError):
            training_stage_contract("finetune", "none")
        self.assertEqual(training_stage_contract("pretrain", "none")["split_identity"], "canonical_smiles")
        self.assertEqual(training_stage_contract("finetune", "all")["split_identity"], "protein_id")

    def test_condition_pretrain_exact_resume_and_all_four_subset_generations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_data(root)
            complete, _ = self.run_train(root, self.recipe("continuous"))
            stopped, config = self.run_train(root, self.recipe("resumed"), max_steps=2)
            stopped_state = torch.load(stopped, weights_only=True)
            self.assertEqual(stopped_state["global_step"], 2)
            with contextlib.redirect_stdout(io.StringIO()):
                resumed = train(config, resume=stopped)
            final_a, final_b = [torch.load(path, weights_only=True) for path in (complete, resumed)]
            self.assertEqual(final_a["global_step"], 6)
            self.assertEqual(final_a["global_step"], final_b["global_step"])
            self.assertEqual(final_a["best_loss"], final_b["best_loss"])
            self.assertEqual(final_a["stage"], "condition_pretrain")
            self.assertEqual(final_a["stage_trained_conditions"], ["properties", "fragment"])
            self.assertEqual(final_a["inherited_trained_conditions"], [])
            self.assertEqual(final_a["trained_conditions"], ["properties", "fragment"])
            self.assertEqual(set(final_a["trained_properties"]), set(PROPERTIES))
            self.assertIsNone(final_a["pocket_encoder_signature"])
            self.assertEqual(final_a["training_contract"]["training_stage"],
                             training_stage_contract("condition_pretrain", "properties,fragment"))
            for key, value in final_a["model"].items():
                torch.testing.assert_close(value, final_b["model"][key], rtol=0, atol=0)
            for index, (conditions, request) in enumerate([
                ("none", {}), ("properties", {"properties": {"qed": 0.5}}),
                ("fragment", {"fragment": "CO"}),
                ("properties,fragment", {"properties": {"logp": 0.2}, "fragment": "CO"}),
            ]):
                summary = generate(complete, request, root / f"sample-{index}.csv", conditions,
                                   count=2, max_tokens=4, seed=11)
                self.assertEqual(summary["attempted"], 2)
                self.assertIsNone(summary["pocket_input"])
            for conditions, request in [("pocket", {"pocket_embedding": [0.] * 128}),
                                         ("ifp", {"ifp": [0] * 6})]:
                with self.assertRaisesRegex(ValueError, "not trained"):
                    generate(complete, request, root / (conditions + ".csv"), conditions, count=1)

    def test_disabled_pocket_never_reads_encoder_or_structures_and_subset_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.write_data(root)
            for row in rows:
                row["pocket"] = {"pocket_pdb": "/absent/protein.pdb"}
            self.write_rows(root / "train.jsonl", rows[:6])
            self.write_rows(root / "valid.jsonl", rows[6:])
            for conditions in ("properties", "fragment", "properties,fragment"):
                recipe = self.recipe(conditions.replace(",", "-"), conditions)
                recipe["pocket_encoder"] = {"repo": "/absent/encoder", "weights": "/absent/weights"}
                with patch.object(PocketResolver, "profile", side_effect=AssertionError("encoder read")), \
                     patch.object(PocketResolver, "input_fingerprint", side_effect=AssertionError("PDB read")), \
                     patch.object(PocketResolver, "resolve", side_effect=AssertionError("encoder run")):
                    checkpoint, _ = self.run_train(root, recipe, max_steps=1)
                state = torch.load(checkpoint, weights_only=True)
                self.assertEqual(state["trained_conditions"], conditions.split(","))
                self.assertEqual(bool(state["trained_properties"]), "properties" in conditions)
                self.assertIsNone(state["training_contract"]["pocket"])

    def test_canonical_parent_leakage_and_protein_identity_are_not_bypassed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.write_data(root)
            self.write_rows(root / "valid.jsonl", [dict(rows[0], smiles="OCC")])
            with self.assertRaisesRegex(ValueError, "leakage.*SMILES"):
                self.run_train(root, self.recipe())
            self.write_rows(root / "valid.jsonl", rows[6:])
            recipe = self.recipe()
            recipe.update(stage="finetune", conditions="properties,fragment")
            with self.assertRaisesRegex(ValueError, "protein_id"):
                self.run_train(root, recipe)

    def test_stage_is_in_exact_resume_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_data(root)
            checkpoint, config = self.run_train(root, self.recipe(), max_steps=1)
            state = torch.load(checkpoint, weights_only=True)
            state["training_contract"]["training_stage"]["stage"] = "pretrain"
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "Exact resume requires unchanged"):
                train(config, resume=checkpoint)

    def test_condition_aware_token_mapping_preserves_every_trained_semantic_branch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_data(root)
            checkpoint, _ = self.run_train(root, self.recipe(), max_steps=2)
            source, old_tokenizer, _ = load_checkpoint(checkpoint)
            seed_all(7)
            untrained = PocketGPT(source.config)
            # The fixture actually optimizes the semantic branches, not just LM weights.
            for key in ("property_projector.0.weight", "fragment_projector.0.weight",
                        "fragment_sequence_encoder.layers.0.self_attn.in_proj_weight", "gate.weight"):
                self.assertFalse(torch.equal(source.state_dict()[key], untrained.state_dict()[key]), key)
            new_tokenizer = SmilesTokenizer([*old_tokenizer.tokens[:3], *reversed(old_tokenizer.tokens[3:]), "Cl"])
            config = source.config.to_dict()
            config["vocab_size"] = len(new_tokenizer.tokens)
            seed_all(123)
            target = PocketGPT(ModelConfig(**config))
            fresh = {key: value.clone() for key, value in target.state_dict().items()}
            report = warm_start_conditioned(target, checkpoint, new_tokenizer, CONDITIONS)
            self.assertFalse(report["exact_resume"])
            self.assertFalse(report["optimizer_restored"])
            self.assertEqual(report["new_conditions"], ["pocket", "ifp"])
            self.assertEqual(report["new_tokens"], ["Cl"])
            self.assertIn("gate.weight", report["copied_tensors"])
            self.assertIn("null_prefix", report["copied_tensors"])
            old, new = source.state_dict(), target.state_dict()
            for key in old:
                if key in ("tok_emb.weight", "head.weight", "modality_emb"):
                    continue
                expected = fresh[key] if key.startswith(("pocket_projector.", "ifp_projector.")) else old[key]
                torch.testing.assert_close(new[key], expected, rtol=0, atol=0, msg=key)
            for token, old_id in old_tokenizer.stoi.items():
                for key in ("tok_emb.weight", "head.weight"):
                    torch.testing.assert_close(new[key][new_tokenizer.stoi[token]], old[key][old_id], rtol=0, atol=0)
            for key in ("tok_emb.weight", "head.weight"):
                torch.testing.assert_close(new[key][new_tokenizer.stoi["Cl"]], fresh[key][new_tokenizer.stoi["Cl"]], rtol=0, atol=0)
            for index, name in enumerate(CONDITIONS):
                torch.testing.assert_close(new["modality_emb"][index],
                                           old["modality_emb"][index] if name in ("properties", "fragment") else fresh["modality_emb"][index],
                                           rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "discard trained conditioning"):
                warm_start_base(target, checkpoint, new_tokenizer)
            bad_config = dict(config, fragment_layers=2)
            with self.assertRaisesRegex(ValueError, "same architecture"):
                warm_start_conditioned(PocketGPT(ModelConfig(**bad_config)), checkpoint, new_tokenizer, CONDITIONS)

    def test_four_condition_finetune_initializes_with_new_optimizer_and_capability_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.write_data(root)
            checkpoint, _ = self.run_train(root, self.recipe("pretrained"), max_steps=2)
            rows[0]["smiles"] = "CCCl"
            for index, row in enumerate(rows):
                row.update(protein_id=f"synthetic-protein-{index}", pocket_embedding=[0.01] * 128,
                           ifp=[1, 0, 1, 0, 0, 0])
            self.write_rows(root / "finetune-train.jsonl", rows[:6])
            self.write_rows(root / "finetune-valid.jsonl", rows[6:])
            recipe = self.recipe("finetuned", "all")
            recipe.update(stage="finetune", train_data="finetune-train.jsonl", valid_data="finetune-valid.jsonl",
                          condition_init_checkpoint=checkpoint, allow_unverified_pocket_embeddings=True)
            finetuned, _ = self.run_train(root, recipe, max_steps=1)
            state = torch.load(finetuned, weights_only=True)
            self.assertEqual(state["global_step"], 1)
            self.assertEqual(state["stage_trained_conditions"], list(CONDITIONS))
            self.assertEqual(state["inherited_trained_conditions"], ["properties", "fragment"])
            self.assertEqual(state["trained_conditions"], list(CONDITIONS))
            self.assertEqual(state["warm_start"]["source_step"], 2)
            self.assertIn("Cl", state["warm_start"]["new_tokens"])
            self.assertTrue((root / "finetuned" / "initialization_report.json").is_file())
            for optimizer_state in state["optimizer"]["state"].values():
                self.assertEqual(float(optimizer_state["step"]), 1.0)

    def test_inherited_conditions_remain_available_after_subset_finetune(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.write_data(root)
            checkpoint, _ = self.run_train(root, self.recipe("pretrained"), max_steps=1)
            tokenizer = load_checkpoint(checkpoint)[1]
            tokenizer.save(root / "pretrained-vocabulary.json")
            for index, row in enumerate(rows):
                row.update(protein_id=f"synthetic-protein-{index}", pocket_embedding=[0.01] * 128)
            self.write_rows(root / "target-train.jsonl", rows[:6])
            self.write_rows(root / "target-valid.jsonl", rows[6:])
            recipe = self.recipe("subset", "pocket")
            recipe.update(stage="finetune", train_data="target-train.jsonl", valid_data="target-valid.jsonl",
                          init_checkpoint=checkpoint, vocabulary="pretrained-vocabulary.json",
                          allow_unverified_pocket_embeddings=True)
            subset, _ = self.run_train(root, recipe, max_steps=1)
            state = torch.load(subset, weights_only=True)
            self.assertEqual(state["stage_trained_conditions"], ["pocket"])
            self.assertEqual(state["trained_conditions"], ["pocket", "properties", "fragment"])
            self.assertEqual(set(state["inherited_trained_properties"]), set(PROPERTIES))
            result = generate(subset, {"properties": {"qed": 0.5}, "fragment": "CO"},
                              root / "inherited.csv", "properties,fragment", count=1, max_tokens=4)
            self.assertEqual(result["conditions"], ["properties", "fragment"])


if __name__ == "__main__":
    unittest.main()
