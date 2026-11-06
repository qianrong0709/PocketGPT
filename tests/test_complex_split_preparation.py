"""Synthetic explicit complex grouping, never fabricated protein identity."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from pocketgpt.preparation import prepare_dataset


def record(identifier, fragment="CO", protein=None):
    value = {"smiles": "CCO", "complex_id": identifier,
             "pocket_embedding": [0.1] * 128,
             "pocket_encoder_signature": "legacy-precomputed-unverified",
             "properties": {"logp": 0, "qed": 0.6, "tpsa": 20, "sas": 2},
             "fragment": fragment, "ifp": [1, 1, 0, 0, 0, 0]}
    if protein is not None:
        value["protein_id"] = protein
    return value


def source_rows():
    return [record(f"complex-{index}", fragment)
            for index in range(20) for fragment in ("C", "CO", "N")]


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def splits(output):
    return {name: [json.loads(line) for line in (output / (name + ".jsonl")).read_text().splitlines()]
            for name in ("train", "valid")}


class ComplexSplitPreparationTests(unittest.TestCase):
    def test_complex_group_preserves_all_rows_and_does_not_invent_proteins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = source_rows()
            rows[0]["fragment_is_substructure"] = False
            write_jsonl(root / "input.jsonl", rows)
            before = (root / "input.jsonl").read_bytes()
            result = prepare_dataset(root / "input.jsonl", root / "prepared", "finetune",
                                     split_group="complex_id", validation_fraction=0.5)
            output = splits(root / "prepared")
            all_rows = output["train"] + output["valid"]
            self.assertEqual(sum(result["counts"].values()), len(rows))
            self.assertEqual(len(all_rows), len(rows))
            self.assertEqual((root / "input.jsonl").read_bytes(), before)
            self.assertFalse(any("protein_id" in row for row in all_rows))
            self.assertTrue(any(row.get("fragment_is_substructure") is False for row in all_rows))
            self.assertEqual(result["split_group"], "complex_id")
            self.assertEqual(result["split"], "complex_id")
            self.assertEqual(result["validation_scope"], "complex_id_grouped_NOT_protein_heldout")
            self.assertFalse(result["protein_ids_invented"])
            self.assertEqual(result["group_overlap_count"], 0)
            self.assertEqual(sum(result["unique_groups"].values()), 20)
            self.assertFalse({row["complex_id"] for row in output["train"]} &
                             {row["complex_id"] for row in output["valid"]})
            self.assertEqual({row["fragment"] for row in all_rows}, {"C", "CO", "N"})

    def test_default_protein_group_remains_strict_and_actual_ids_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "without.jsonl", source_rows())
            with self.assertRaisesRegex(ValueError, "protein_id_required"):
                prepare_dataset(root / "without.jsonl", root / "reject", "finetune")
            rows = [record(f"complex-{index}", protein="real-protein-" + str(index // 2)) for index in range(20)]
            write_jsonl(root / "with.jsonl", rows)
            protein = prepare_dataset(root / "with.jsonl", root / "protein", "finetune", validation_fraction=0.5)
            self.assertEqual(protein["split_group"], "protein_id")
            output = splits(root / "protein")
            self.assertFalse({row["protein_id"] for row in output["train"]} &
                             {row["protein_id"] for row in output["valid"]})
            prepare_dataset(root / "with.jsonl", root / "complex", "finetune",
                            validation_fraction=0.5, split_group="complex_id")
            combined = [row for values in splits(root / "complex").values() for row in values]
            self.assertEqual({row["complex_id"]: row["protein_id"] for row in combined},
                             {row["complex_id"]: row["protein_id"] for row in rows})

    def test_grouping_is_exact_stable_under_order_and_never_derived_from_smiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = source_rows()
            for name, values in (("a", rows), ("b", list(reversed(rows)))):
                write_jsonl(root / (name + ".jsonl"), values)
                prepare_dataset(root / (name + ".jsonl"), root / name, "finetune",
                                split_group="complex_id", validation_fraction=0.5)
            for name in ("train", "valid"):
                self.assertEqual({row["complex_id"] for row in splits(root / "a")[name]},
                                 {row["complex_id"] for row in splits(root / "b")[name]})
            # All parents are identical. Both splits exist only because this is
            # explicitly a complex split, not a molecule- or protein-held-out one.
            for name in ("train", "valid"):
                self.assertEqual({row["smiles"] for row in splits(root / "a")[name]}, {"CCO"})

    def test_missing_unknown_or_wrong_stage_grouping_is_rejected(self):
        for value in (None, "", "   ", 1, True):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_jsonl(root / "input.jsonl", [record(value)])
                with self.assertRaisesRegex(ValueError, "complex_id_required"):
                    prepare_dataset(root / "input.jsonl", root / "prepared", "finetune", split_group="complex_id")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "input.jsonl", source_rows())
            with self.assertRaisesRegex(ValueError, "split_group"):
                prepare_dataset(root / "input.jsonl", root / "bad", "finetune", split_group="pdb_prefix")
            with self.assertRaisesRegex(ValueError, "only available for finetune"):
                prepare_dataset(root / "input.jsonl", root / "bad", "pretrain", split_group="complex_id")

    def test_raw_fragment_mode_preserves_invalid_chemical_text_without_masking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = source_rows()
            for row in rows:
                row["fragment"] = "n1cnnc1"
            write_jsonl(root / "input.jsonl", rows)
            result = prepare_dataset(root / "input.jsonl", root / "raw", "finetune",
                                     split_group="complex_id", fragment_representation="raw-smiles-v1",
                                     validation_fraction=0.5)
            combined = [row for values in splits(root / "raw").values() for row in values]
            self.assertEqual(len(combined), len(rows))
            self.assertTrue(all(row["fragment"] == "n1cnnc1" for row in combined))
            self.assertTrue(all(row["fragment_representation"] == "raw-smiles-v1" for row in combined))
            self.assertEqual(result["fragment_representation"], "raw-smiles-v1")
            self.assertFalse(any("protein_id" in row for row in combined))
            with self.assertRaises(ValueError):
                prepare_dataset(root / "input.jsonl", root / "canonical", "finetune", split_group="complex_id")

    def test_explicit_source_representation_cannot_be_silently_reinterpreted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = source_rows()
            rows[0]["fragment_representation"] = "raw-smiles-v1"
            write_jsonl(root / "input.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "conflicts with source record"):
                prepare_dataset(root / "input.jsonl", root / "prepared", "finetune", split_group="complex_id")

    def test_legacy_complex_split_uses_file_ids_without_required_protein_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields = ["smiles", "file", "qed", "sas", "evalue", *[f"Unnamed: {i}" for i in range(3, 130)],
                      "frag", "ifp"]
            with (root / "legacy.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for index in range(20):
                    value = dict.fromkeys(fields, "0")
                    value.update(smiles="CCO", file="complex-" + str(index), qed="0.6", sas="2",
                                 frag="n1cnnc1", ifp="[1,1,0,0,0,0]")
                    writer.writerow(value)
            result = prepare_dataset(root / "legacy.csv", root / "prepared", "finetune", legacy=True,
                                     split_group="complex_id", fragment_representation="raw-smiles-v1",
                                     validation_fraction=0.5)
            combined = [row for values in splits(root / "prepared").values() for row in values]
            self.assertEqual(len(combined), 20)
            self.assertFalse(any("protein_id" in row for row in combined))
            self.assertTrue(all(row["fragment"] == "n1cnnc1" for row in combined))
            self.assertEqual(result["ifp_labelled_rows"], 20)
            with self.assertRaisesRegex(ValueError, "--protein-map"):
                prepare_dataset(root / "legacy.csv", root / "need-map", "finetune", legacy=True)


if __name__ == "__main__":
    unittest.main()
