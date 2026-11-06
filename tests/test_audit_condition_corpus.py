"""Synthetic corpus auditing only. No network, server or research data."""
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_condition_corpus import audit_condition_corpus, main
from pocketgpt.pocket import file_sha256


HEADER = ("smiles", "LogS", "logp", "qed", "tpsa", "sas", "frag")


def write_csv(path, rows, header=HEADER):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def row(smiles="CCO", fragment="CO"):
    return [smiles, "unused-LogS", "0", "0.5", "0", "2", fragment]


class ConditionCorpusAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.output = self.root / "source.csv", self.root / "audit"

    def audit(self, rows, **kwargs):
        write_csv(self.source, rows)
        self.before = self.source.read_bytes()
        result = audit_condition_corpus(self.source, self.output, progress_every=0, **kwargs)
        self.assertEqual(self.source.read_bytes(), self.before)
        self.assertEqual(json.loads((self.output / "summary.json").read_text()), result)
        self.assertEqual(set(path.name for path in self.output.iterdir()), {"summary.json", "issues.jsonl"})
        return result

    def issues(self):
        return [json.loads(line) for line in (self.output / "issues.jsonl").read_text().splitlines()]

    def test_full_table_counts_aliases_duplicates_and_missing_versus_zero(self):
        missing = ["CO", "", "", "", "0", "", ""]
        result = self.audit([row(), row("OCC"), row(), missing])
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["scan_completed"])
        self.assertTrue(result["strict_contract_passed"])
        self.assertFalse(result["training_ready"])
        self.assertEqual(result["source_total_records_read"], 4)
        self.assertEqual(result["unique_canonical_parent_molecules"], 2)
        self.assertEqual(result["missing"]["logp"], 1)
        self.assertEqual(result["missing"]["tpsa"], 0)
        self.assertEqual(result["missing"]["fragment"], 1)
        self.assertEqual(result["property_statistics"]["logp"]["min"], 0)
        self.assertEqual(result["property_statistics"]["tpsa"]["valid"], 4)
        self.assertEqual(result["effective_condition_rows"], {"fragment": 3, "properties": 4})
        self.assertGreater(result["cache"]["chemical_hits"], 0)
        self.assertEqual(result["source_sha256"], hashlib.sha256(self.before).hexdigest())
        self.assertEqual(result["source_sha256"], result["source_sha256_after"])
        self.assertFalse(self.issues())
        for name in ("labels_repaired", "labels_masked", "rows_filtered", "source_written", "training_data_published", "raw_representation_enabled"):
            self.assertFalse(result[name])

    def test_invalid_fragments_do_not_end_scan_and_are_not_raw_training_inputs(self):
        # Five aromatic n/c atoms without an [nH] cannot be kekulized, but are
        # tokenizable text. Test alongside a syntax error and a subsequent row.
        invalid = "n1cnnc1"
        result = self.audit([row(fragment=invalid), row(fragment="C)"),
                             row(fragment=invalid), row("CCN", "CN")])
        self.assertTrue(result["scan_completed"])
        self.assertFalse(result["strict_contract_passed"])
        self.assertEqual(result["counts"]["invalid_rows"], 3)
        self.assertEqual(result["counts"]["strict_row_contract_passed_rows"], 1)
        self.assertEqual(result["counts"]["fragment_raw_tokenizable_but_strict_invalid_rows"], 3)
        self.assertEqual(result["unique_invalid_fragment_raw_sha256"], 2)
        self.assertEqual([x["source_record"] for x in self.issues()], [1, 2, 3])
        self.assertTrue(any("kekulize" in key for key in result["issue_counts"]))
        self.assertIn("fragment_syntax_parse_failed", result["issue_counts"])
        self.assertEqual(self.issues()[0]["raw_sha256"], self.issues()[2]["raw_sha256"])
        self.assertNotIn(invalid, (self.output / "issues.jsonl").read_text())
        self.assertFalse(result["raw_representation_enabled"])

    def test_parent_errors_token_syntax_and_context_overflow_are_distinct(self):
        result = self.audit([row("C)"), row(fragment="C%XYZ"), row("CCCCCCCCCC", "CCCCCC")], block_size=5)
        self.assertEqual(result["counts"]["invalid_rows"], 3)
        self.assertEqual(result["unique_invalid_parent_raw_sha256"], 1)
        self.assertEqual(result["counts"]["fragment_raw_untokenizable_rows"], 1)
        self.assertEqual(result["max_context_tokens_including_eos"], 11)
        self.assertEqual(result["max_fragment_canonical_tokens"], 6)
        self.assertIn("parent_context_overflow", result["issue_counts"])
        self.assertIn("fragment_context_overflow", result["issue_counts"])
        self.assertEqual(result["counts"]["parent_raw_tokenizable_rows"], 3)

    def test_properties_are_checked_individually_without_nan_json_or_raw_values(self):
        bad = row()
        bad[2:6] = ["NaN", "1.1", "-1", "0"]
        missing = row("CO", "")
        missing[2:6] = ["", "", "", ""]
        result = self.audit([bad, missing, row("CCN", "CN")])
        self.assertEqual(result["counts"]["invalid_rows"], 1)
        self.assertEqual(result["counts"]["all_properties_missing_rows"], 1)
        for name in ("logp", "qed", "tpsa", "sas"):
            self.assertEqual(result["property_statistics"][name]["invalid"], 1)
            self.assertEqual(result["missing"][name], 1)
            self.assertIn("property_invalid_" + name, result["issue_counts"])
        self.assertEqual(len(self.issues()), 4)
        self.assertNotIn("NaN", (self.output / "issues.jsonl").read_text())
        json.loads((self.output / "summary.json").read_text(), parse_constant=lambda value: self.fail(value))

    def test_optional_substructure_mismatch_is_warning_not_invalid_or_mask(self):
        result = self.audit([row(fragment="N"), row()], check_substructure=True)
        self.assertTrue(result["strict_contract_passed"])
        self.assertEqual(result["counts"]["substructure_checked_rows"], 2)
        self.assertEqual(result["counts"]["substructure_mismatch_rows"], 1)
        self.assertEqual(self.issues()[0]["severity"], "warning")
        self.assertEqual(result["counts"].get("invalid_rows", 0), 0)
        self.assertFalse(result["labels_masked"])

    def test_width_error_is_located_and_scan_continues(self):
        result = self.audit([row(), ["wrong", "width"], row("CCN", "CN")])
        self.assertTrue(result["scan_completed"])
        self.assertEqual(result["source_total_records_read"], 3)
        self.assertEqual(result["counts"]["well_shaped_rows"], 2)
        self.assertEqual(result["counts"]["malformed_width_rows"], 1)
        self.assertEqual(self.issues()[0]["source_record"], 2)
        self.assertEqual(self.issues()[0]["csv_line"], 3)

    def test_csv_syntax_failure_is_incomplete_not_a_full_scan(self):
        # An unclosed quoted field consumes the remainder, so there is no safe
        # record boundary to resume from. Do not invent a successful row count.
        self.source.write_text(",".join(HEADER) + "\n" + ",".join(row()) + '\n"unclosed\n', encoding="utf-8")
        with self.assertRaises(csv.Error):
            audit_condition_corpus(self.source, self.output, progress_every=0)
        result = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["scan_completed"])
        self.assertFalse(result["strict_contract_passed"])
        self.assertEqual(result["source_total_records_read"], 1)
        self.assertEqual(result["first_failure"]["type"], "Error")

    def test_source_change_and_unexpected_failure_cannot_report_healthy(self):
        write_csv(self.source, [row()])
        actual = file_sha256(self.source)
        with patch("scripts.audit_condition_corpus.file_sha256", side_effect=[actual, "changed", "changed"]):
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                audit_condition_corpus(self.source, self.output, progress_every=0)
        result = json.loads((self.output / "summary.json").read_text())
        self.assertTrue(result["scan_completed"])
        self.assertFalse(result["source_unchanged"])
        self.assertFalse(result["strict_contract_passed"])
        self.assertEqual(result["status"], "incomplete")

        second = self.root / "second"
        with patch("scripts.audit_condition_corpus.canonical_smiles", side_effect=RuntimeError("unexpected")):
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                audit_condition_corpus(self.source, second, progress_every=0)
        result = json.loads((second / "summary.json").read_text())
        self.assertFalse(result["scan_completed"])
        self.assertFalse(result["strict_contract_passed"])
        self.assertEqual(result["first_failure"]["type"], "RuntimeError")

    def test_header_empty_data_existing_target_and_cli_exit_codes(self):
        write_csv(self.source, [], header=("smiles", "frag"))
        with self.assertRaises(ValueError):
            audit_condition_corpus(self.source, self.output, progress_every=0)
        result = json.loads((self.output / "summary.json").read_text())
        self.assertFalse(result["scan_completed"])
        self.assertFalse(result["strict_contract_passed"])
        before = (self.output / "summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            audit_condition_corpus(self.source, self.output, progress_every=0)
        self.assertEqual((self.output / "summary.json").read_bytes(), before)

        empty = self.root / "empty"
        write_csv(self.source, [])
        result = audit_condition_corpus(self.source, empty, progress_every=0)
        self.assertTrue(result["scan_completed"])
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["strict_contract_passed"])

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            write_csv(self.source, [row()])
            self.assertEqual(main(["--input", str(self.source), "--output-dir", str(self.root / "cli-ok")]), 0)
            write_csv(self.source, [row(fragment="n1cnnc1")])
            self.assertEqual(main(["--input", str(self.source), "--output-dir", str(self.root / "cli-bad")]), 2)
            self.assertEqual(main(["--input", str(self.source), "--output-dir", str(self.output)]), 1)


if __name__ == "__main__":
    unittest.main()
