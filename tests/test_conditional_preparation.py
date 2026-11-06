"""Synthetic local fixtures only: no user corpus, server or network access."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pocketgpt.conditional_preparation import _publish_new_directory, prepare_condition_pretrain
from pocketgpt.data import canonical_smiles
from pocketgpt.tokenizer import SmilesTokenizer


HEADER = ("smiles", "LogS", "logp", "qed", "tpsa", "sas", "frag")


def fixture_rows():
    return [["C" * index, "-1.25", "0", "0.5", "20", "2", "C"]
            for index in range(1, 21)]


def write_csv(path, rows, header=HEADER):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def read_splits(directory):
    return {name: [json.loads(line) for line in (directory / f"{name}.jsonl").read_text().splitlines()]
            for name in ("train", "valid")}


class ConditionalPretrainingPreparationTests(unittest.TestCase):
    def test_rows_labels_parent_split_metadata_and_hashes_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "synthetic.csv", root / "prepared"
            rows = fixture_rows() + [["CCO", "-3.400", "1", "0.7", "30", "3", "CC"],
                                     ["OCC", "", "2", "0.6", "40", "4", "N"],
                                     ["CCO", "-3.400", "1", "0.7", "30", "3", "CC"]]
            write_csv(source, rows)
            before = source.read_bytes()
            report = prepare_condition_pretrain(source, output, validation_fraction=0.5)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(report["source_sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(report["source_sha256_after"], report["source_sha256"])
            self.assertEqual(report["source_total_rows"], len(rows))
            self.assertEqual(report["unique_parent_molecules"], 21)
            self.assertEqual(report["stage"], "condition_pretrain")
            self.assertFalse(report["row_deduplication_applied"])
            self.assertFalse(report["fragment_labels_masked"])
            self.assertEqual(report["fragment_substructure_check"], "not_performed")
            splits = read_splits(output)
            combined = splits["train"] + splits["valid"]
            self.assertEqual(len(combined), len(rows))
            self.assertEqual(sorted(row["source_row"] for row in combined), list(range(1, len(rows) + 1)))
            self.assertFalse({row["molecule_id"] for row in splits["train"]} &
                             {row["molecule_id"] for row in splits["valid"]})
            ethanol = [row for row in combined if row["smiles"] == "CCO"]
            self.assertEqual(len(ethanol), 3)
            self.assertEqual({row["fragment"] for row in ethanol}, {"CC", "N"})
            self.assertEqual(len({row["molecule_id"] for row in ethanol}), 1)
            self.assertEqual(ethanol[0]["molecule_id"], hashlib.sha256(b"CCO").hexdigest())
            for row in combined:
                self.assertFalse({"pocket", "ifp", "protein_id"} & row.keys())
                self.assertEqual(set(row["properties"]), {"logp", "qed", "tpsa", "sas"})
                self.assertNotIn("LogS", row["properties"])
                self.assertEqual(row["metadata"]["source_values"]["LogS"], rows[row["source_row"] - 1][1])
            self.assertEqual(report["max_context_tokens_including_eos"], 21)
            self.assertEqual(report["max_fragment_tokens"], 2)
            self.assertEqual(report["reject_reasons"], {})
            for name, digest in report["output_sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(json.loads((output / "preparation.json").read_text()), report)
            self.assertFalse(list(root.glob(".prepared.pending-*")))

    def test_missing_labels_are_not_invented_and_real_zero_survives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = fixture_rows()
            rows[0] = ["C", "", "0", "", "0", "", " "]
            rows[1] = ["CC", "", "", "", "", "", ""]
            write_csv(root / "data.csv", rows)
            report = prepare_condition_pretrain(root / "data.csv", root / "prepared", validation_fraction=0.5)
            by_source = {row["source_row"]: row for rows in read_splits(root / "prepared").values() for row in rows}
            self.assertEqual(by_source[1]["properties"], {"logp": 0.0, "tpsa": 0.0})
            self.assertEqual(by_source[2]["properties"], {})
            self.assertNotIn("fragment", by_source[1])
            self.assertNotIn("fragment", by_source[2])
            self.assertEqual(report["missing"]["properties"], {"logp": 1, "qed": 2, "tpsa": 1, "sas": 2})
            self.assertEqual(report["missing"]["fragment"], 2)
            self.assertEqual(report["missing"]["all_selected_conditions"], 1)

    def test_vocabulary_includes_canonical_fragments_and_keeps_stereo_attachment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = fixture_rows() + [["CCO", "0", "1", "0.5", "20", "2", "[SiH3]C[C@@H](F)*"]]
            write_csv(root / "data.csv", rows)
            prepare_condition_pretrain(root / "data.csv", root / "prepared", validation_fraction=0.5)
            tokenizer = SmilesTokenizer.load(root / "prepared" / "vocabulary.json")
            records = [row for rows in read_splits(root / "prepared").values() for row in rows]
            for row in records:
                for name in ("smiles", "fragment"):
                    self.assertEqual(tokenizer.decode(tokenizer.encode(row[name])), row[name])
            self.assertIn("[SiH3]", tokenizer.tokens)
            self.assertIn("*", tokenizer.tokens)
            self.assertTrue(any("@" in token for token in tokenizer.tokens))

    def test_subset_conditions_keep_other_labels_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_csv(root / "data.csv", fixture_rows())
            report = prepare_condition_pretrain(root / "data.csv", root / "prepared", validation_fraction=0.5,
                                                properties=("qed",), conditions=("properties",))
            self.assertEqual(report["conditions"], ["properties"])
            self.assertEqual(report["property_names"], ["qed"])
            row = read_splits(root / "prepared")["train"][0]
            self.assertEqual(row["properties"], {"qed": 0.5})
            self.assertEqual(row["fragment"], "C")
            self.assertEqual(set(row["metadata"]["source_values"]), {"LogS", "logp", "tpsa", "sas"})

    def test_fragment_only_preparation_can_have_no_property_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [["C" * index, "C"] for index in range(1, 21)]
            write_csv(root / "data.csv", rows, header=("smiles", "frag"))
            report = prepare_condition_pretrain(root / "data.csv", root / "prepared", validation_fraction=0.5,
                                                properties=(), conditions=("fragment",))
            self.assertEqual(report["property_names"], [])
            self.assertEqual(report["missing"]["properties"], {})
            self.assertEqual(report["missing"]["all_selected_conditions"], 0)
            for rows in read_splits(root / "prepared").values():
                self.assertTrue(all(row["properties"] == {} and row["fragment"] == "C" for row in rows))

    def test_parent_partition_does_not_depend_on_row_order_or_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = fixture_rows() + [["CCO", "0", "1", "0.5", "20", "2", "CC"]]
            write_csv(root / "a.csv", rows)
            write_csv(root / "b.csv", [*reversed(rows), ["OCC", "0", "1", "0.5", "20", "2", "N"]])
            prepare_condition_pretrain(root / "a.csv", root / "a", validation_fraction=0.5)
            prepare_condition_pretrain(root / "b.csv", root / "b", validation_fraction=0.5)
            a, b = read_splits(root / "a"), read_splits(root / "b")
            for split in ("train", "valid"):
                self.assertEqual({row["smiles"] for row in a[split]}, {row["smiles"] for row in b[split]})
                for row in b[split]:
                    self.assertEqual(row["molecule_id"], hashlib.sha256(canonical_smiles(row["smiles"]).encode()).hexdigest())

    def test_invalid_rows_abort_without_publication_and_keep_diagnostics(self):
        cases = [(0, "not-a-smiles", "Invalid parent SMILES"), (6, "not-a-fragment", "Invalid fragment SMILES"),
                 (2, "nan", "Non-finite property"), (3, "1.01", "Invalid property value"),
                 (4, "-1", "Invalid property value"), (5, "0", "Invalid property value")]
        for column, value, message in cases:
            with self.subTest(column=column), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows = fixture_rows()
                rows[1][column] = value
                write_csv(root / "data.csv", rows)
                before = (root / "data.csv").read_bytes()
                with self.assertRaisesRegex(ValueError, "Source record 2.*" + message):
                    prepare_condition_pretrain(root / "data.csv", root / "prepared", validation_fraction=0.5)
                self.assertFalse((root / "prepared").exists())
                failure = json.loads(next(root.glob(".prepared.pending-*/failure.json")).read_text())
                self.assertEqual(failure["source_total_rows"], 2)
                self.assertEqual(sum(failure["counts"].values()), 1)
                self.assertEqual(sum(failure["reject_reasons"].values()), 1)
                self.assertEqual(failure["status"], "failed_not_published")
                self.assertNotIn(value, failure["error"])
                self.assertEqual((root / "data.csv").read_bytes(), before)

    def test_parent_eos_and_fragment_lengths_are_never_truncated(self):
        for target in ("parent", "fragment"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows = fixture_rows()
                if target == "parent":
                    rows[0][0] = "C" * 21
                else:
                    rows[0][6] = "C" * 22
                write_csv(root / "data.csv", rows)
                with self.assertRaisesRegex(ValueError, "no truncation or filtering"):
                    prepare_condition_pretrain(root / "data.csv", root / "prepared", validation_fraction=0.5, block_size=21)
                self.assertFalse((root / "prepared").exists())
                failure = json.loads(next(root.glob(".prepared.pending-*/failure.json")).read_text())
                self.assertFalse(failure["truncation_applied"])

    def test_wrong_schema_is_not_guessed_and_bad_arguments_do_not_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_csv(root / "data.csv", fixture_rows(), header=(*HEADER[:-1], "two_frag"))
            with self.assertRaisesRegex(ValueError, "Required CSV columns.*frag"):
                prepare_condition_pretrain(root / "data.csv", root / "schema")
            for options in ({"conditions": ("pocket",)}, {"conditions": ()}, {"properties": ("LogS",)},
                            {"properties": ()}, {"block_size": True}, {"validation_fraction": float("nan")}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    prepare_condition_pretrain(root / "data.csv", root / "arguments", **options)
            self.assertFalse(list(root.glob(".arguments.pending-*")))

    def test_empty_malformed_and_duplicate_header_keep_failure_report(self):
        cases = [((), (), "no header"), (HEADER, (), "no molecule records"),
                 ((*HEADER, " QED "), (), "Duplicate normalized"),
                 (HEADER, [("CCO", "0")], "field count differs")]
        for header, rows, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "data.csv"
                if header:
                    write_csv(source, rows, header=header)
                else:
                    source.touch()
                with self.assertRaisesRegex(ValueError, message):
                    prepare_condition_pretrain(source, root / "prepared", validation_fraction=0.5)
                self.assertFalse((root / "prepared").exists())
                report = json.loads(next(root.glob(".prepared.pending-*/failure.json")).read_text())
                self.assertEqual(report["status"], "failed_not_published")

    def test_existing_directory_and_dangling_link_are_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_csv(root / "data.csv", fixture_rows())
            (root / "existing").mkdir()
            (root / "link").symlink_to(root / "missing-target")
            for destination in (root / "existing", root / "link"):
                with self.assertRaises(FileExistsError):
                    prepare_condition_pretrain(root / "data.csv", destination)
            staged = root / "staged"
            staged.mkdir()
            (staged / "sentinel").write_text("synthetic")
            with self.assertRaises(FileExistsError):
                _publish_new_directory(staged, root / "existing")
            self.assertEqual((staged / "sentinel").read_text(), "synthetic")
            self.assertEqual(list((root / "existing").iterdir()), [])
            self.assertTrue((root / "link").is_symlink())
            self.assertFalse((root / "missing-target").exists())

    def test_source_change_and_empty_partition_fail_without_resampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_csv(root / "data.csv", fixture_rows())
            with patch("pocketgpt.conditional_preparation.file_sha256", side_effect=["before", "after"]):
                with self.assertRaisesRegex(RuntimeError, "Input changed"):
                    prepare_condition_pretrain(root / "data.csv", root / "changed", validation_fraction=0.5)
            self.assertFalse((root / "changed").exists())
            write_csv(root / "same.csv", [["CCO", "0", "1", "0.5", "20", "2", "C"],
                                           ["OCC", "0", "1", "0.5", "20", "2", "CC"]])
            with self.assertRaisesRegex(ValueError, "empty partition"):
                prepare_condition_pretrain(root / "same.csv", root / "empty")
            self.assertFalse((root / "empty").exists())
            report = json.loads(next(root.glob(".empty.pending-*/failure.json")).read_text())
            self.assertEqual(report["unique_parent_molecules"], 1)
            self.assertEqual(sum(report["counts"].values()), 2)
            self.assertFalse(report["sampling_applied"])


if __name__ == "__main__":
    unittest.main()
