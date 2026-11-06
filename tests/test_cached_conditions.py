import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pocketgpt.cached_conditions import TARGET_COLUMNS, attach_ifp_targets, export_ifp_targets
from pocketgpt.config import IFP_NAMES
from pocketgpt.pocket import file_sha256


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.reader(stream))


class CachedConditionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = self.root / "base.csv"
        self.labels = self.root / "targets.csv"
        self.cached = self.root / "cached.csv"
        self.output = self.root / "with-ifp.csv"
        self.header = ["file", "smiles", "private_vector_000", "water_bridges", "saltbridge", "hbonds",
                       "pistacking", "hydrophobic_contacts", "metal_complexes", "pication", "halogen_bonds"]
        self.rows = [["3abc_A_2", "OCC", "private-vector-value", "ignored", "0", "1", "1", "0", "ignored", "1", "0"],
                     ["3abc_A_20", "CCN", "another-private-vector", "ignored", "", "", "", "", "ignored", "", ""]]
        write_csv(self.base, self.header, self.rows)
        self.cached_header = ["file", "smiles", "evalue", "Unnamed: 3", "frag", "comment"]
        self.cached_rows = [["3abc_A_2", "CCO", "1.2340E-09", "0.0001", "CO", "first"],
                            ["3abc_A_2", "OCC", "1.2340E-09", "0.0001", "n1cnnc1<", "chemical label left literal"],
                            ["3abc_A_20", "CCN", "-1e-06", "0.1", " C,\nO ", "quoted newline stays"]]
        write_csv(self.cached, self.cached_header, self.cached_rows)

    def export(self):
        return export_ifp_targets(self.base, self.labels)

    def test_export_is_minimal_named_ordered_and_missing_is_blank(self):
        original = file_sha256(self.base)
        report = self.export()
        rows = read_csv(self.labels)
        self.assertEqual(rows[0], list(TARGET_COLUMNS))
        self.assertEqual(rows[1][0], "3abc_A_2")
        self.assertEqual(rows[1][1], hashlib.sha256(b"CCO").hexdigest())
        self.assertEqual(rows[1][2:], ["1", "0", "1", "0", "1", "0"])
        self.assertEqual(rows[2][2:], [""] * 6)
        self.assertEqual(report["output_rows"], 2)
        self.assertEqual(report["ifp_missing_rows"], 1)
        self.assertEqual(report["ifp_order"], list(IFP_NAMES))
        self.assertEqual(file_sha256(self.base), original)
        self.assertEqual(report["source_sha256"], original)
        self.assertEqual(report["output_sha256"], file_sha256(self.labels))
        sidecar = self.labels.with_suffix(".report.json").read_text()
        self.assertNotIn(str(self.root), sidecar)
        self.assertNotIn("private-vector-value", self.labels.read_text())
        self.assertNotIn("smiles", rows[0])
        self.assertEqual(json.loads(sidecar), report)

    def test_attach_preserves_every_original_field_row_and_literal_fragment(self):
        self.export()
        before = {path: file_sha256(path) for path in (self.cached, self.labels)}
        report = attach_ifp_targets(self.cached, self.labels, self.output, before[self.cached])
        output = read_csv(self.output)
        self.assertEqual(output[0], self.cached_header + list(IFP_NAMES))
        self.assertEqual([row[:len(self.cached_header)] for row in output[1:]], self.cached_rows)
        self.assertEqual(output[1][-6:], ["1", "0", "1", "0", "1", "0"])
        self.assertEqual(output[2][-6:], output[1][-6:])
        self.assertEqual(output[3][-6:], [""] * 6)
        self.assertEqual(report["cached_rows"], 3)
        self.assertEqual(report["ifp_missing_rows"], 1)
        self.assertEqual(report["ifp_missing_complexes"], 1)
        self.assertTrue(report["original_field_values_and_row_order_verified"])
        self.assertFalse(report["fragment_labels_modified_or_masked"])
        self.assertEqual(before, {path: file_sha256(path) for path in before})
        self.assertNotIn(str(self.root), self.output.with_suffix(".report.json").read_text())

    def test_unknown_cached_id_is_not_matched_by_shared_pdb_prefix(self):
        self.export()
        rows = copy_rows(self.cached_rows)
        rows[0][0] = "3abc_A_999"
        write_csv(self.cached, self.cached_header, rows)
        with self.assertRaisesRegex(ValueError, "Unknown cached exact complex ID"):
            attach_ifp_targets(self.cached, self.labels, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".report.json").exists())

    def test_missing_target_complex_is_rejected(self):
        self.export()
        write_csv(self.cached, self.cached_header, self.cached_rows[:2])
        with self.assertRaisesRegex(ValueError, "target complex IDs are missing"):
            attach_ifp_targets(self.cached, self.labels, self.output)
        self.assertFalse(self.output.exists())

    def test_parent_identity_mismatch_rejected_but_aliases_match(self):
        self.export()
        rows = copy_rows(self.cached_rows)
        rows[0][1] = "CCC"
        write_csv(self.cached, self.cached_header, rows)
        with self.assertRaisesRegex(ValueError, "Canonical parent identity differs"):
            attach_ifp_targets(self.cached, self.labels, self.output)
        self.assertFalse(self.output.exists())

    def test_partial_invalid_and_duplicate_labels_do_not_publish(self):
        for changed in ("", "2", "nan"):
            rows = copy_rows(self.rows)
            rows[0][self.header.index("hbonds")] = changed
            write_csv(self.base, self.header, rows)
            with self.subTest(value=changed), self.assertRaisesRegex(ValueError, "invalid six-IFP"):
                self.export()
            self.assertFalse(self.labels.exists())
        write_csv(self.base, self.header, [self.rows[0], self.rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate exact complex ID"):
            self.export()

    def test_bad_smiles_and_malformed_field_count_are_rejected(self):
        rows = copy_rows(self.rows)
        rows[0][1] = "not-a-smiles"
        write_csv(self.base, self.header, rows)
        with self.assertRaisesRegex(ValueError, "Invalid parent SMILES"):
            self.export()
        write_csv(self.base, self.header, [self.rows[0][:-1]])
        with self.assertRaisesRegex(ValueError, "unexpected field count"):
            self.export()
        self.assertFalse(self.labels.exists())

    def test_duplicate_normalized_headers_and_missing_ifp_schema_rejected(self):
        write_csv(self.base, [*self.header, " HBONDS "], [[*self.rows[0], "1"]])
        with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
            self.export()
        write_csv(self.base, ["file", "smiles"], [["id", "CCO"]])
        with self.assertRaisesRegex(ValueError, "one named source column"):
            self.export()

    def test_no_existing_ifp_column_is_overwritten(self):
        self.export()
        for column in ("hydrogen_bond", "hbonds", "metal_complexes", "ifp"):
            write_csv(self.cached, [*self.cached_header, column], [[*row, "old-label"] for row in self.cached_rows])
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "already has IFP"):
                attach_ifp_targets(self.cached, self.labels, self.output)
            self.assertFalse(self.output.exists())

    def test_wrong_expected_cached_hash_fails_without_changing_any_file(self):
        self.export()
        before = file_sha256(self.cached)
        with self.assertRaisesRegex(ValueError, "explicitly expected"):
            attach_ifp_targets(self.cached, self.labels, self.output, "f" * 64)
        self.assertEqual(file_sha256(self.cached), before)
        self.assertFalse(self.output.exists())

    def test_output_csv_sidecar_and_symlinks_are_never_replaced(self):
        self.labels.write_text("keep CSV")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(self.labels.read_text(), "keep CSV")
        self.labels.unlink()
        sidecar = self.labels.with_suffix(".report.json")
        sidecar.write_text("keep report")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(sidecar.read_text(), "keep report")
        sidecar.unlink()
        self.labels.symlink_to(self.root / "absent")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertTrue(self.labels.is_symlink())

    def test_inputs_and_outputs_must_be_distinct(self):
        with self.assertRaises((ValueError, FileExistsError)):
            export_ifp_targets(self.base, self.base)
        self.export()
        with self.assertRaisesRegex(ValueError, "distinct files"):
            attach_ifp_targets(self.cached, self.cached, self.output)
        hardlink = self.root / "same-inode.csv"
        os.link(self.cached, hardlink)
        with self.assertRaisesRegex(ValueError, "distinct files"):
            attach_ifp_targets(self.cached, hardlink, self.output)

    def test_source_changes_are_not_published(self):
        original_hash = file_sha256(self.base)
        hashes = iter([original_hash, "f" * 64])
        with patch("pocketgpt.cached_conditions.file_sha256", side_effect=lambda path: next(hashes)):
            with self.assertRaisesRegex(RuntimeError, "changed during export"):
                self.export()
        self.assertFalse(self.labels.exists())
        self.assertFalse(self.labels.with_suffix(".report.json").exists())

    def test_attachment_rejects_changed_source_and_malformed_target_table(self):
        self.export()
        actual = file_sha256
        calls = {str(self.cached): 0}
        def hash_changed(path):
            path = Path(path)
            if path == self.cached:
                calls[str(self.cached)] += 1
                return "f" * 64 if calls[str(self.cached)] > 1 else actual(path)
            return actual(path)
        with patch("pocketgpt.cached_conditions.file_sha256", side_effect=hash_changed):
            with self.assertRaisesRegex(RuntimeError, "changed during attachment"):
                attach_ifp_targets(self.cached, self.labels, self.output)
        self.assertFalse(self.output.exists())
        target = read_csv(self.labels)
        target[1][1] = "invalid-hash"
        write_csv(self.labels, target[0], target[1:])
        with self.assertRaisesRegex(ValueError, "Invalid canonical-parent SHA256"):
            attach_ifp_targets(self.cached, self.labels, self.output)


def copy_rows(rows):
    return [list(row) for row in rows]


if __name__ == "__main__":
    unittest.main()
