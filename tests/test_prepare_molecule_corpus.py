"""Only synthetic CSVs: no private corpus reads or server access."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pocketgpt.data import canonical_smiles
from pocketgpt.tokenizer import SmilesTokenizer
from scripts.prepare_molecule_corpus import _publish_new_directory, prepare_molecule_corpus


def write_csv(path, smiles, column="smiles"):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([column, "frag", "unused_condition"])
        for index, text in enumerate(smiles):
            writer.writerow([text, f"synthetic-fragment-row-{index}", index])


def split_rows(directory):
    return {name: [json.loads(line)["smiles"] for line in (directory / (name + ".jsonl")).read_text().splitlines()]
            for name in ("train", "valid")}


class MoleculeCorpusTests(unittest.TestCase):
    def test_streaming_dedup_canonical_split_and_symbol_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "synthetic.csv", root / "prepared"
            unique = ["C" * i for i in range(1, 21)] + ["CCO", "C[C@H](O)F", "C[C@@H](O)F"]
            write_csv(source, unique + unique + ["OCC", " CCO "])
            source_before = source.read_bytes()
            with patch("scripts.prepare_molecule_corpus.canonical_smiles", wraps=canonical_smiles) as normalize:
                report = prepare_molecule_corpus(source, output, validation_fraction=0.5)
            self.assertEqual(normalize.call_count, 25)  # not all 48 fragment-expanded rows
            self.assertEqual(report["source_total_rows"], 48)
            self.assertEqual(report["unique_raw_smiles"], 25)
            self.assertEqual(report["unique_canonical_smiles"], 23)
            self.assertEqual(report["raw_duplicate_rows"], 23)
            self.assertEqual(report["canonical_duplicate_raw_strings"], 2)
            self.assertEqual(report["duplicate_rows"], 25)
            self.assertEqual(report["reject_reasons"], {})
            self.assertFalse(report["source_modified"])
            self.assertFalse(report["sampling_applied"])
            self.assertFalse(report["truncation_applied"])
            self.assertFalse(report["source_condition_rows_modified"])
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(report["source_sha256"], hashlib.sha256(source_before).hexdigest())
            self.assertEqual(report["source_sha256_after"], report["source_sha256"])
            self.assertEqual(report["max_smiles_tokens"], 20)
            self.assertEqual(report["max_context_tokens_including_eos"], 21)
            data = split_rows(output)
            self.assertFalse(set(data["train"]) & set(data["valid"]))
            self.assertEqual(set(data["train"]) | set(data["valid"]), {canonical_smiles(s) for s in unique})
            self.assertEqual(len(data["train"]), report["counts"]["train"])
            self.assertEqual(len(data["valid"]), report["counts"]["valid"])
            tokenizer = SmilesTokenizer.load(output / "vocabulary.json")
            for text in data["train"] + data["valid"]:
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            self.assertIn("no frequencies or weights learned", report["vocabulary_scope"])
            for name, digest in report["output_sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(json.loads((output / "preparation.json").read_text()), report)
            self.assertFalse(list(root.glob(".prepared.pending-*")))

    def test_split_is_independent_of_order_and_alias_repetition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            molecules = ["C" * i for i in range(1, 21)] + ["CCO"]
            write_csv(root / "a.csv", molecules)
            write_csv(root / "b.csv", ["OCC", *reversed(molecules), "OCC"])
            prepare_molecule_corpus(root / "a.csv", root / "a", validation_fraction=0.5)
            prepare_molecule_corpus(root / "b.csv", root / "b", validation_fraction=0.5)
            first, second = split_rows(root / "a"), split_rows(root / "b")
            for split in ("train", "valid"):
                self.assertEqual(set(first[split]), set(second[split]))

    def test_explicit_column_and_context_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.csv"
            write_csv(source, ["C" * i for i in range(1, 21)], column="Molecule")
            report = prepare_molecule_corpus(source, root / "ok", smiles_column="molecule",
                                              validation_fraction=0.5, max_context=21)
            self.assertEqual(report["max_context_requested"], 21)
            with self.assertRaisesRegex(ValueError, "SMILES column not found"):
                prepare_molecule_corpus(source, root / "wrong-column")
            with self.assertRaisesRegex(ValueError, "no truncation or filtering"):
                prepare_molecule_corpus(source, root / "too-short", smiles_column="Molecule",
                                         validation_fraction=0.5, max_context=5)
            self.assertFalse((root / "too-short").exists())
            failure = json.loads(next(root.glob(".too-short.pending-*/failure.json")).read_text())
            self.assertEqual(failure["status"], "failed_not_published")
            self.assertEqual(sum(failure["reject_reasons"].values()), 1)
            self.assertGreater(failure["max_context_tokens_including_eos"], 5)

    def test_invalid_molecule_fails_immediately_and_keeps_local_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.csv"
            write_csv(source, ["CCO", "not-a-smiles", "CCC"])
            before = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "Source record 2"):
                prepare_molecule_corpus(source, root / "failed", validation_fraction=0.5)
            self.assertFalse((root / "failed").exists())
            staged = next(root.glob(".failed.pending-*"))
            report = json.loads((staged / "failure.json").read_text())
            self.assertEqual(report["source_total_rows"], 2)
            self.assertEqual(sum(report["counts"].values()), 1)
            self.assertEqual(sum(report["reject_reasons"].values()), 1)
            self.assertFalse(report["invalid_rows_skipped"])
            self.assertEqual(source.read_bytes(), before)

    def test_existing_target_is_never_overwritten_even_when_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.csv"
            write_csv(source, ["CCO"])
            output = root / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_molecule_corpus(source, output)
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            staged = root / "staged"
            staged.mkdir()
            (staged / "sentinel").write_text("synthetic")
            with self.assertRaises(FileExistsError):
                _publish_new_directory(staged, output)
            self.assertEqual((staged / "sentinel").read_text(), "synthetic")
            self.assertEqual(list(output.iterdir()), [])

    def test_changed_source_is_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.csv"
            write_csv(source, ["C" * i for i in range(1, 21)])
            with patch("scripts.prepare_molecule_corpus.file_sha256", side_effect=["before", "after"]):
                with self.assertRaisesRegex(RuntimeError, "Input changed"):
                    prepare_molecule_corpus(source, root / "failed", validation_fraction=0.5)
            self.assertFalse((root / "failed").exists())

    def test_empty_partition_is_not_published_or_silently_resampled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.csv"
            write_csv(source, ["CCO", "OCC"])
            with self.assertRaisesRegex(ValueError, "empty partition"):
                prepare_molecule_corpus(source, root / "failed")
            self.assertFalse((root / "failed").exists())
            report = json.loads(next(root.glob(".failed.pending-*/failure.json")).read_text())
            self.assertEqual(report["unique_canonical_smiles"], 1)
            self.assertFalse(report["sampling_applied"])


if __name__ == "__main__":
    unittest.main()
