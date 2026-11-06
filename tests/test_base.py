import csv
import hashlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from pocketgpt.checkpoint import load_checkpoint
from pocketgpt.config import CONDITIONS, IFP_NAMES, ModelConfig
from pocketgpt.data import BatchBuilder, inspect_manifest
from pocketgpt.generation import generate
from pocketgpt.ifp import LEGACY_IFP_COLUMNS, canonical_ifp, ifp_from_columns, validate_ifp_schema
from pocketgpt.model import PocketGPT
from pocketgpt.pocket import PocketResolver
from pocketgpt.preparation import prepare_dataset
from pocketgpt.tokenizer import SmilesTokenizer
from pocketgpt.training import train


def row(smiles="CCO", protein="train-1"):
    return {"smiles": smiles, "protein_id": protein,
            "pocket_embedding": [0.01 * i for i in range(128)],
            "properties": {"logp": 0.0, "qed": 0.6, "tpsa": 40.0, "sas": 2.0},
            "fragment": "CO", "ifp": [1, 1, 0, 0, 0, 0]}


# Deliberately non-contiguous synthetic ordering, NOT a claim about the old CSV.
TEST_IFP_SCHEMA = {"source_dim": 8, "indices": dict(zip(IFP_NAMES, [4, 1, 7, 0, 5, 2]))}


class IFPTests(unittest.TestCase):
    def test_author_header_order_and_named_column_selection_agree(self):
        schema_path = Path(__file__).resolve().parents[1] / "configs" / "legacy-ifp-schema.json"
        schema = json.loads(schema_path.read_text())
        source = [0, 1, 0, 1, 0, 1, 1, 1]
        expected = [1, 0, 1, 0, 0, 1]
        columns = dict(reversed(list(zip(LEGACY_IFP_COLUMNS, source))))
        self.assertEqual(canonical_ifp(source, schema), expected)
        self.assertEqual(canonical_ifp(columns), expected)
        columns.update(metal_complexes=float("nan"), water_bridges="not a model input")
        self.assertEqual(canonical_ifp(columns), expected)
        self.assertEqual(ifp_from_columns(dict(columns, smiles="CCO", metadata="ignored")), expected)

    def test_named_columns_reject_partial_and_duplicate_interactions(self):
        columns = dict.fromkeys(LEGACY_IFP_COLUMNS, 0)
        columns.pop("hbonds")
        with self.assertRaisesRegex(ValueError, "Missing IFP"):
            ifp_from_columns(columns)
        columns["hbonds"] = 1
        columns["hydrogen_bond"] = 1
        with self.assertRaisesRegex(ValueError, "Multiple columns"):
            ifp_from_columns(columns)

    def test_missing_named_labels_are_not_zero_labels(self):
        for missing in (None, "", "  ", float("nan")):
            columns = dict.fromkeys(LEGACY_IFP_COLUMNS[:6], missing)
            columns.update(metal_complexes=1, water_bridges=1)
            self.assertIsNone(ifp_from_columns(columns, allow_missing=True))
            with self.assertRaises(ValueError):
                canonical_ifp(columns)
            columns["hbonds"] = 0
            with self.assertRaises(ValueError):
                ifp_from_columns(columns, allow_missing=True)
        zeros = dict.fromkeys(LEGACY_IFP_COLUMNS, "0")
        self.assertEqual(ifp_from_columns(zeros, allow_missing=True), [0] * 6)

    def test_explicit_eight_to_six_mapping(self):
        source = [1, 0, 1, 0, 1, 0, 0, 1]
        expected = [1, 0, 1, 1, 0, 1]
        self.assertEqual(canonical_ifp(source, TEST_IFP_SCHEMA), expected)
        self.assertEqual(canonical_ifp(json.dumps(source), TEST_IFP_SCHEMA), expected)
        self.assertEqual(canonical_ifp(dict(zip(IFP_NAMES, expected))), expected)
        self.assertEqual(canonical_ifp(expected), expected)
        with self.assertRaisesRegex(ValueError, "do not guess"):
            canonical_ifp(source)

    def test_mapping_rejects_ambiguous_indices_and_nonbinary_labels(self):
        for value in (None, -1, 8, True, 1.5):
            bad = {"source_dim": 8, "indices": dict(TEST_IFP_SCHEMA["indices"])}
            bad["indices"][IFP_NAMES[0]] = value
            with self.assertRaises(ValueError):
                validate_ifp_schema(bad)
        duplicate = {"source_dim": 8, "indices": dict.fromkeys(IFP_NAMES, 0)}
        with self.assertRaises(ValueError):
            validate_ifp_schema(duplicate)
        for source in ([0] * 7, [0] * 9, [0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 1.00000001],
                       [0, 0, 0, 0, 0, float("nan")]):
            with self.assertRaises(ValueError):
                canonical_ifp(source)
        with self.assertRaises(ValueError):
            canonical_ifp([0] * 6, TEST_IFP_SCHEMA)

    def test_legacy_preparation_preserves_source_and_uses_ifp(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "legacy.csv"
            columns = ["smiles", "file", "logp", "qed", "tpsa", "sas", "evalue",
                       *[f"Unnamed: {i}" for i in range(3, 130)], "ifp"]
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for i in range(20):
                    sample = dict.fromkeys(columns, "0")
                    sample.update(smiles="CCO", file=f"complex-{i}", qed="0.6", sas="2.0",
                                  ifp="[1,0,1,0,1,0,0,1]")
                    writer.writerow(sample)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            protein_map = root / "proteins.json"
            protein_map.write_text(json.dumps({f"complex-{i}": f"protein-{i}" for i in range(20)}))
            schema_path = root / "ifp-schema.json"
            schema_path.write_text(json.dumps(TEST_IFP_SCHEMA))
            output = root / "prepared"
            report = prepare_dataset(source, output, "finetune", legacy=True, protein_map=protein_map,
                                     validation_fraction=0.5, ifp_schema=schema_path)
            self.assertEqual(report["ifp_labelled_rows"], 20)
            self.assertEqual(report["ifp_ignored_source_indices"], [3, 6])
            self.assertEqual(report["ifp_source_schema"], TEST_IFP_SCHEMA)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            for split in ("train", "valid"):
                rows = [json.loads(line) for line in (output / (split + ".jsonl")).read_text().splitlines()]
                self.assertTrue(rows)
                self.assertTrue(all(r["ifp"] == [1, 0, 1, 1, 0, 1] for r in rows))
            with self.assertRaisesRegex(ValueError, "--ifp-schema"):
                prepare_dataset(source, root / "unmapped", "finetune", legacy=True, protein_map=protein_map)

    def test_named_csv_preparation_selects_by_header_without_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "named.csv"
            columns = ["file", "smiles", *reversed(LEGACY_IFP_COLUMNS), "ifp", "evalue",
                       *[f"Unnamed: {i}" for i in range(3, 130)]]
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for i in range(20):
                    sample = dict.fromkeys(columns, "0")
                    sample.update(file=f"complex-{i}", smiles="CCO", hbonds="1", pistacking="1",
                                  halogen_bonds="1", metal_complexes="unused", water_bridges="unused",
                                  ifp="stale packed column must not override named values")
                    writer.writerow(sample)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            protein_map = root / "proteins.json"
            protein_map.write_text(json.dumps({f"complex-{i}": f"protein-{i}" for i in range(20)}))
            output = root / "prepared"
            report = prepare_dataset(source, output, "finetune", legacy=True,
                                     protein_map=protein_map, validation_fraction=0.5)
            self.assertEqual(report["ifp_input_formats"], {"named_csv_columns": 20})
            self.assertEqual(report["ifp_excluded_interactions"], ["metal_complexes", "water_bridges"])
            self.assertIsNone(report["ifp_source_schema"])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            for split in ("train", "valid"):
                rows = [json.loads(line) for line in (output / (split + ".jsonl")).read_text().splitlines()]
                self.assertTrue(rows)
                self.assertTrue(all(r["ifp"] == [1, 0, 1, 0, 0, 1] for r in rows))

    def test_preparation_preserves_rows_with_unavailable_named_ifp(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "missing-labels.csv"
            columns = ["file", "smiles", *LEGACY_IFP_COLUMNS, "ifp", "evalue",
                       *[f"Unnamed: {i}" for i in range(3, 130)]]
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for i in range(20):
                    sample = dict.fromkeys(columns, "0")
                    sample.update(file=f"complex-{i}", smiles="CCO", ifp="[1,1,1,1,1,1]")
                    if i < 3:
                        sample.update(dict.fromkeys(LEGACY_IFP_COLUMNS[:6], ""))
                    writer.writerow(sample)
            protein_map = root / "proteins.json"
            protein_map.write_text(json.dumps({f"complex-{i}": f"protein-{i}" for i in range(20)}))
            output = root / "prepared"
            report = prepare_dataset(source, output, "finetune", legacy=True,
                                     protein_map=protein_map, validation_fraction=0.5)
            rows = [json.loads(line) for name in ("train", "valid")
                    for line in (output / (name + ".jsonl")).read_text().splitlines()]
            self.assertEqual(len(rows), 20)
            self.assertEqual(report["ifp_labelled_rows"], 17)
            self.assertEqual(report["ifp_unlabelled_rows"], 3)
            self.assertEqual(report["rejected"], {})
            for sample in rows:
                if sample["complex_id"] in {"complex-0", "complex-1", "complex-2"}:
                    self.assertNotIn("ifp", sample)
                else:
                    self.assertEqual(sample["ifp"], [0] * 6)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.tokenizer = SmilesTokenizer.build(["CCO", "CCN", "CO", "C(=O)O", "ClCBr"])
        self.model = PocketGPT(ModelConfig(len(self.tokenizer.tokens), block_size=16,
                                          n_layer=2, n_head=2, n_embd=32, dropout=0.0)).eval()

    def test_all_sixteen_combinations_forward_and_backward(self):
        for flags in itertools.product([False, True], repeat=4):
            selected = [name for name, enabled in zip(CONDITIONS, flags) if enabled]
            x, y, batch = BatchBuilder(self.tokenizer, 16, selected)([row(), row("CCN")])
            self.model.zero_grad(set_to_none=True)
            out = self.model(x, batch, y)
            self.assertEqual(tuple(out["logits"].shape), (2, 16, len(self.tokenizer.tokens)))
            self.assertTrue(torch.isfinite(out["loss"]))
            out["loss"].backward()
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            for index, enabled in enumerate(flags):
                if not enabled:
                    self.assertEqual(float(out["gate_weights"][:, index].sum().detach()), 0.0)

    def test_disabled_values_do_not_affect_logits_even_nan(self):
        builder = BatchBuilder(self.tokenizer, 16)
        x, _, batch = builder([row()])
        for index, name in enumerate(CONDITIONS):
            first = {key: value.clone() for key, value in batch.items()}
            first["condition_mask"][:, index] = False
            second = {key: value.clone() for key, value in first.items()}
            if name == "fragment":
                second["fragment_tokens"][:] = self.tokenizer.stoi["N"]
            elif name == "properties":
                second["properties"][:] = float("nan")
                second["property_mask"][:] = ~second["property_mask"]
            else:
                second[name][:] = float("nan")
            torch.testing.assert_close(self.model(x, first)["logits"], self.model(x, second)["logits"], rtol=0, atol=0)

    def test_only_the_two_unselected_ifp_bits_are_ignored(self):
        source = [1, 0, 1, 0, 1, 0, 0, 1]
        modified = source.copy()
        modified[3] = modified[6] = 1
        builder = BatchBuilder(self.tokenizer, 16, ["ifp"])
        x, _, first = builder([dict(row(), ifp=canonical_ifp(source, TEST_IFP_SCHEMA))])
        _, _, second = builder([dict(row(), ifp=canonical_ifp(modified, TEST_IFP_SCHEMA))])
        self.assertTrue(bool(first["condition_mask"][0, 3]))
        self.assertEqual(int(first["condition_mask"].sum()), 1)
        torch.testing.assert_close(first["ifp"], second["ifp"], rtol=0, atol=0)
        torch.testing.assert_close(self.model(x, first)["logits"], self.model(x, second)["logits"], rtol=0, atol=0)

    def test_named_ifp_keeps_modality_active_but_ignores_metal_and_water(self):
        columns = dict(zip(LEGACY_IFP_COLUMNS, [0, 1, 0, 1, 0, 1, 0, 0]))
        builder = BatchBuilder(self.tokenizer, 16, ["ifp"])
        x, _, first = builder([dict(row(), ifp=columns)])
        columns = dict(columns, metal_complexes=float("nan"), water_bridges=1000)
        _, _, second = builder([dict(row(), ifp=columns)])
        self.assertTrue(bool(first["condition_mask"][0, 3]))
        self.assertEqual(int(first["condition_mask"].sum()), 1)
        torch.testing.assert_close(self.model(x, first)["logits"], self.model(x, second)["logits"], rtol=0, atol=0)

    def test_unlabelled_ifp_mask_differs_from_known_six_zeros(self):
        unlabelled = row()
        unlabelled.pop("ifp")
        _, _, batch = BatchBuilder(self.tokenizer, 16, ["ifp"])([unlabelled, dict(row(), ifp=[0] * 6)])
        self.assertFalse(bool(batch["condition_mask"][0, 3]))
        self.assertTrue(bool(batch["condition_mask"][1, 3]))
        torch.testing.assert_close(batch["ifp"][0], batch["ifp"][1], rtol=0, atol=0)

    def test_partial_property_masks_are_distinct_from_zero_targets(self):
        builder = BatchBuilder(self.tokenizer, 16, ["properties"])
        _, _, zero = builder([dict(row(), properties={"logp": 0.0})])
        _, _, missing = builder([dict(row(), properties={"qed": 0.5})])
        self.assertTrue(bool(zero["property_mask"][0, 0]))
        self.assertFalse(bool(missing["property_mask"][0, 0]))

    def test_padding_ignored_and_decoder_is_causal(self):
        x, y, batch = BatchBuilder(self.tokenizer, 16)([row()])
        result = self.model(x, batch, y)
        keep = y.ne(0)
        expected = F.cross_entropy(result["logits"][keep], y[keep])
        torch.testing.assert_close(result["loss"], expected)
        altered = x.clone()
        altered[:, 3:] = self.tokenizer.stoi["N"]
        torch.testing.assert_close(result["logits"][:, :3], self.model(altered, batch)["logits"][:, :3], rtol=0, atol=0)

    def test_missing_encoder_is_an_error_not_random_features(self):
        with self.assertRaises(ValueError):
            PocketResolver().resolve({"pocket": {"pocket_pdb": "/not-present.pdb"}})

    def test_disabled_pocket_does_not_read_a_file(self):
        request = dict(row(), pocket={"pocket_pdb": "/not-present.pdb"})
        request.pop("pocket_embedding")
        BatchBuilder(self.tokenizer, 16, ["properties"])([request])

    def test_strict_smiles_tokenization(self):
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode("ClCBr")), "ClCBr")
        with self.assertRaises(ValueError):
            self.tokenizer.encode("C invalid O")
        with self.assertRaises(ValueError):
            BatchBuilder(self.tokenizer, 3)([row("CCO")])

    def test_conditional_data_requires_protein_identity(self):
        sample = row()
        sample.pop("protein_id")
        with self.assertRaises(ValueError):
            inspect_manifest([sample], conditional=True)


class TrainingTests(unittest.TestCase):
    def test_exact_interrupted_resume_and_generation_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_rows = [row(s, f"train-{i}") for i, s in enumerate(["CCO", "CCN", "CCC", "CO", "CNC", "C=O"])]
            valid_rows = [row("CCCO", "valid-1"), row("CNCO", "valid-2")]
            for name, rows in [("train", train_rows), ("valid", valid_rows)]:
                (root / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
            recipe = {"stage": "finetune", "conditions": "all", "train_data": "train.jsonl", "valid_data": "valid.jsonl",
                      "allow_unverified_pocket_embeddings": True,
                      "output_dir": "continuous", "model": {"block_size": 16, "n_layer": 1, "n_head": 2, "n_embd": 32, "dropout": 0.1},
                      "training": {"epochs": 2, "batch_size": 2, "learning_rate": 0.001, "seed": 7,
                                   "condition_dropout": 0.35, "property_dropout": 0.2, "cpu_threads": 1}}
            config_a = root / "continuous.json"
            config_a.write_text(json.dumps(recipe))
            complete = train(config_a)
            recipe["output_dir"] = "resumed"
            config_b = root / "resumed.json"
            config_b.write_text(json.dumps(recipe))
            stopped = train(config_b, max_steps=2)
            resumed = train(config_b, resume=stopped)
            final_a = torch.load(complete, weights_only=True)
            final_b = torch.load(resumed, weights_only=True)
            self.assertEqual(final_a["global_step"], 6)
            self.assertEqual(final_a["global_step"], final_b["global_step"])
            for name, value in final_a["model"].items():
                torch.testing.assert_close(value, final_b["model"][name], rtol=0, atol=0)
            self.assertEqual(final_a["best_loss"], final_b["best_loss"])
            with self.assertRaises(ValueError):
                generate(complete, {"properties": {"unknown": 1}}, root / "bad.csv", "properties", count=1)
            with self.assertRaises(ValueError):
                generate(complete, {"pocket": {"pocket_pdb": "/absent.pdb"}}, root / "bad2.csv", "pocket", count=1)
            result = generate(complete, {"properties": {"qed": 0.5}}, root / "generated.csv", "properties", count=3, max_tokens=8)
            self.assertEqual(result["attempted"], 3)
            self.assertEqual(result["conditions"], ["properties"])
            with self.assertRaises(FileExistsError):
                generate(complete, {}, root / "generated.csv", "none", count=1)


if __name__ == "__main__":
    unittest.main()
