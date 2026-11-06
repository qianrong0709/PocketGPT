import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from pocketgpt.config import IFP_NAMES
from pocketgpt.data import BatchBuilder
from pocketgpt.ifp import LEGACY_IFP_COLUMNS, canonical_ifp
from pocketgpt.merging import merge_condition_csvs
from pocketgpt.tokenizer import SmilesTokenizer


BASE_COLUMNS = [
    "smiles", "file", "LogS", "logp", "qed", "tpsa", "sas", "evalue",
    *[f"Unnamed: {index}" for index in range(3, 130)],
    *LEGACY_IFP_COLUMNS,
]


def base_row(identifier, smiles="CCO", labels=None, embedding=0.125):
    row = {
        "smiles": smiles, "file": identifier, "LogS": "-2.5", "logp": "1.25",
        "qed": "0.75", "tpsa": "40.0", "sas": "2.0", "evalue": str(embedding),
        **{f"Unnamed: {index}": str(embedding) for index in range(3, 130)},
    }
    values = [1, 0, 1, 0, 1, 0, 1, 1] if labels is None else labels
    row.update(dict(zip(LEGACY_IFP_COLUMNS, values)))
    return row


def write_csv(path, columns, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_outputs(output):
    with (output / "merged.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    with (output / "merged.jsonl").open() as handle:
        json_rows = [json.loads(line) for line in handle if line.strip()]
    report = json.loads((output / "merge_report.json").read_text())
    return csv_rows, json_rows, report


class ConditionMergeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base.csv"
        self.fragments = self.root / "fragments.csv"
        self.output = self.root / "merged"

    def write_inputs(self, bases, fragments):
        write_csv(self.base, BASE_COLUMNS, bases)
        write_csv(self.fragments, ["file", "smiles", "frag"], fragments)

    def test_many_to_one_merge_preserves_rows_and_missing_conditions(self):
        self.write_inputs(
            [base_row("c1"), base_row("c2", "CCC", labels=[""] * 8),
             base_row("c3", "CCN", labels=[0] * 6 + [1, 1])],
            [
                {"file": "c1", "smiles": "OCC", "frag": "CO"},
                {"file": "c1", "smiles": "CCO", "frag": "OC"},
                {"file": "c2", "smiles": "CCC", "frag": ""},
                {"file": "c1", "smiles": "CCO", "frag": "OCC"},
                {"file": "c3", "smiles": "NCC", "frag": "N"},
            ],
        )
        source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (self.base, self.fragments)}
        report = merge_condition_csvs(self.base, self.fragments, self.output)
        csv_rows, rows, saved_report = read_outputs(self.output)
        self.assertEqual(len(csv_rows), 5)
        self.assertEqual(len(rows), 5)
        self.assertEqual(Counter(r["complex_id"] for r in rows), {"c1": 3, "c2": 1, "c3": 1})
        expected_counts = {
            "output_rows": 5, "base_complexes": 3, "missing_ifp_complexes": 1,
            "missing_ifp_rows": 1, "missing_fragment_rows": 1,
            "all_four_conditions_rows": 4, "duplicate_fragment_pairs": 1,
            "fallback_complexes": 0, "fragment_substructure_mismatch_rows": 0,
        }
        for name, expected in expected_counts.items():
            self.assertEqual(report[name], expected, name)
            self.assertEqual(saved_report[name], expected, name)
        for record in rows:
            self.assertNotIn("protein_id", record)
            self.assertEqual(record["pocket_encoder_signature"], "legacy-precomputed-unverified")
            self.assertEqual(record["pocket_embedding"], [0.125] * 128)
            self.assertEqual(record["properties"], {"logp": 1.25, "qed": 0.75, "tpsa": 40.0, "sas": 2.0})
            if record["complex_id"] == "c1":
                self.assertEqual(record["smiles"], "CCO")
                self.assertIn(record["fragment"], {"CO", "CCO"})
                self.assertEqual(canonical_ifp(record["ifp"]), [0, 1, 0, 1, 1, 0])
            elif record["complex_id"] == "c2":
                self.assertNotIn("ifp", record)
                self.assertNotIn("fragment", record)
            else:
                self.assertEqual(canonical_ifp(record["ifp"]), [0] * 6)
                self.assertEqual(record["fragment"], "N")
        for record in csv_rows:
            self.assertTrue(all(f"pocket_{index:03d}" in record for index in range(128)))
            self.assertNotIn("metal_complexes", record)
            self.assertNotIn("water_bridges", record)
            if record["complex_id"] == "c2":
                self.assertEqual(record["frag"], "")
                self.assertTrue(all(record[name] == "" for name in IFP_NAMES))
            elif record["complex_id"] == "c3":
                self.assertTrue(all(float(record[name]) == 0 for name in IFP_NAMES))
        for path, digest in source_hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        tokenizer = SmilesTokenizer.build([r["smiles"] for r in rows] +
                                          [r["fragment"] for r in rows if "fragment" in r])
        _, _, batch = BatchBuilder(tokenizer, 32)(rows)
        for index, record in enumerate(rows):
            if record["complex_id"] == "c2":
                self.assertEqual(batch["condition_mask"][index].tolist(), [True, True, False, False])
            else:
                self.assertEqual(batch["condition_mask"][index].tolist(), [True] * 4)
                if record["complex_id"] == "c3":
                    self.assertEqual(batch["ifp"][index].tolist(), [0] * 6)

    def test_missing_ifp_is_counted_at_both_complex_and_expanded_row_levels(self):
        self.write_inputs(
            [base_row("c1", labels=[""] * 8)],
            [{"file": "c1", "smiles": "CCO", "frag": "C"},
             {"file": "c1", "smiles": "CCO", "frag": "O"}],
        )
        report = merge_condition_csvs(self.base, self.fragments, self.output)
        _, rows, _ = read_outputs(self.output)
        self.assertEqual(report["output_rows"], 2)
        self.assertEqual(report["missing_ifp_complexes"], 1)
        self.assertEqual(report["missing_ifp_rows"], 2)
        self.assertEqual(report["all_four_conditions_rows"], 0)
        self.assertTrue(all("ifp" not in record for record in rows))

    def test_duplicate_base_ids_are_rejected_even_when_values_agree(self):
        self.write_inputs(
            [base_row("c1"), base_row("c1")],
            [{"file": "c1", "smiles": "CCO", "frag": "CO"}],
        )
        with self.assertRaises(ValueError):
            merge_condition_csvs(self.base, self.fragments, self.output)

    def test_unknown_fragment_complex_id_is_not_joined_by_smiles(self):
        self.write_inputs(
            [base_row("1abc_1")],
            [{"file": "1abc_1", "smiles": "CCO", "frag": "C"},
             {"file": "1abc_2", "smiles": "CCO", "frag": "CO"}],
        )
        with self.assertRaises(ValueError):
            merge_condition_csvs(self.base, self.fragments, self.output)
        self.assertFalse(self.output.exists(), "A failed join must not publish partial merged files")

    def test_parent_smiles_mismatch_is_rejected(self):
        self.write_inputs(
            [base_row("c1", "CCO")],
            [{"file": "c1", "smiles": "CCN", "frag": "C"}],
        )
        with self.assertRaises(ValueError):
            merge_condition_csvs(self.base, self.fragments, self.output)

    def test_base_complex_without_any_fragment_record_gets_one_fallback_row(self):
        self.write_inputs(
            [base_row("c1"), base_row("c2", "CCN", labels=[0] * 8)],
            [{"file": "c1", "smiles": "CCO", "frag": "CO"}],
        )
        report = merge_condition_csvs(self.base, self.fragments, self.output)
        _, rows, _ = read_outputs(self.output)
        self.assertEqual(report["output_rows"], 2)
        self.assertEqual(report["fallback_complexes"], 1)
        self.assertEqual(report["missing_fragment_rows"], 1)
        self.assertEqual(report["all_four_conditions_rows"], 1)
        fallback = next(r for r in rows if r["complex_id"] == "c2")
        self.assertEqual(fallback["smiles"], "CCN")
        self.assertNotIn("fragment", fallback)
        self.assertEqual(canonical_ifp(fallback["ifp"]), [0] * 6)

    def test_same_smiles_and_pdb_prefix_keep_distinct_complex_conditions(self):
        first = base_row("1abc_1", embedding=0.125)
        second = base_row("1abc_2", labels=[0] * 8, embedding=0.25)
        second["qed"] = "0.5"
        self.write_inputs(
            [first, second],
            [{"file": "1abc_2", "smiles": "CCO", "frag": "CO"},
             {"file": "1abc_1", "smiles": "CCO", "frag": "CO"}],
        )
        merge_condition_csvs(self.base, self.fragments, self.output)
        _, rows, _ = read_outputs(self.output)
        by_id = {record["complex_id"]: record for record in rows}
        self.assertEqual(by_id["1abc_1"]["pocket_embedding"], [0.125] * 128)
        self.assertEqual(by_id["1abc_2"]["pocket_embedding"], [0.25] * 128)
        self.assertEqual(by_id["1abc_1"]["properties"]["qed"], 0.75)
        self.assertEqual(by_id["1abc_2"]["properties"]["qed"], 0.5)
        self.assertNotEqual(canonical_ifp(by_id["1abc_1"]["ifp"]), canonical_ifp(by_id["1abc_2"]["ifp"]))

    def test_non_substructure_fragment_is_not_silently_dropped(self):
        self.write_inputs(
            [base_row("c1", "CCO")],
            [{"file": "c1", "smiles": "CCO", "frag": "N"}],
        )
        report = merge_condition_csvs(self.base, self.fragments, self.output)
        _, rows, _ = read_outputs(self.output)
        self.assertEqual(report["output_rows"], 1)
        self.assertEqual(rows[0]["fragment"], "N")
        self.assertIs(rows[0]["fragment_is_substructure"], False)
        self.assertEqual(report["fragment_substructure_mismatch_rows"], 1)
        self.assertEqual(report["fragment_substructure_mismatch_complexes"], 1)

    def test_repeated_ancillary_fields_in_fragment_table_cannot_overwrite_base(self):
        self.write_inputs(
            [base_row("c1")],
            [{"file": "c1", "smiles": "CCO", "frag": "CO"}],
        )
        write_csv(self.fragments, ["file", "smiles", "frag", "logp", "qed", "evalue", "hbonds"],
                  [{"file": "c1", "smiles": "CCO", "frag": "CO", "logp": "9.5",
                    "qed": "0.25", "evalue": "0.5", "hbonds": "1"}])
        report = merge_condition_csvs(self.base, self.fragments, self.output)
        _, rows, _ = read_outputs(self.output)
        self.assertEqual(rows[0]["properties"]["logp"], 1.25)
        self.assertEqual(rows[0]["properties"]["qed"], 0.75)
        self.assertEqual(rows[0]["pocket_embedding"], [0.125] * 128)
        self.assertEqual(canonical_ifp(rows[0]["ifp"]), [0, 1, 0, 1, 1, 0])
        for field in ("logp", "qed", "evalue"):
            self.assertEqual(report["fragment_source_ancillary_field_differences"][field], 1)

    def test_malformed_selected_fields_are_rejected(self):
        cases = [
            ({"smiles": ""}, "CO"),
            ({"smiles": "  "}, "CO"),
            ({"qed": "not numeric"}, "CO"),
            ({"evalue": "nan"}, "CO"),
            ({"hbonds": "2"}, "CO"),
            ({"hbonds": ""}, "CO"),
            ({}, "not a SMILES"),
        ]
        for number, (changes, fragment) in enumerate(cases):
            with self.subTest(changes=changes, fragment=fragment):
                base = dict(base_row("c1"), **changes)
                self.write_inputs([base], [{"file": "c1", "smiles": "CCO", "frag": fragment}])
                with self.assertRaises(ValueError):
                    merge_condition_csvs(self.base, self.fragments, self.root / f"bad-{number}")

    def test_existing_output_directory_is_never_overwritten(self):
        self.write_inputs(
            [base_row("c1")],
            [{"file": "c1", "smiles": "CCO", "frag": "CO"}],
        )
        for occupied in (False, True):
            with self.subTest(occupied=occupied):
                output = self.root / ("occupied" if occupied else "empty")
                output.mkdir()
                if occupied:
                    (output / "keep.txt").write_text("user-owned contents")
                with self.assertRaises(FileExistsError):
                    merge_condition_csvs(self.base, self.fragments, output)
                self.assertFalse((output / "merged.csv").exists())
                if occupied:
                    self.assertEqual((output / "keep.txt").read_text(), "user-owned contents")


if __name__ == "__main__":
    unittest.main()
