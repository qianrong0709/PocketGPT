"""Audit a named local IFP CSV and optionally export keyed labels, not a split.

Run from the project root with python -m scripts.validate_ifp_csv. No network
access, structure downloads, protein identity guesses, or training are involved.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from pocketgpt.config import IFP_NAMES
from pocketgpt.ifp import IGNORED_IFP_COLUMNS, ifp_from_columns


def validate(input_path, report_path, labels_path=None):
    source = Path(input_path).resolve()
    report_path = Path(report_path).resolve()
    labels_path = Path(labels_path).resolve() if labels_path else None
    outputs = [report_path] + ([labels_path] if labels_path else [])
    if len(set(outputs)) != len(outputs) or source in outputs:
        raise ValueError("Source, report and optional labels must be different files")
    if any(path.exists() for path in outputs):
        raise FileExistsError("Audit output already exists; choose new filenames")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    positive = Counter(dict.fromkeys(IFP_NAMES, 0))
    identifiers, smiles = set(), set()
    missing, records = [], []
    embedded_rows = 0
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = [name.strip().lower() for name in reader.fieldnames]
        if len(header) != len(set(header)):
            raise ValueError("Duplicate normalized CSV columns")
        if ifp_from_columns(dict.fromkeys(header, 0)) is None:
            raise ValueError("Input has no named IFP interaction columns")
        reader.fieldnames = header
        embedding_columns = ["evalue", *[f"unnamed: {i}" for i in range(3, 130)]]
        for number, row in enumerate(reader, 2):
            identifier = row.get("file", "").strip()
            if not identifier or identifier in identifiers:
                raise ValueError(f"Missing or duplicated file ID at source line {number}")
            identifiers.add(identifier)
            smiles.add(row.get("smiles", ""))
            values = ifp_from_columns(row, allow_missing=True)
            if values is None:
                missing.append(identifier)
            else:
                positive.update(dict(zip(IFP_NAMES, values)))
            # Excluded columns must not affect even missing-label decisions.
            changed = dict(row, metal_complexes="excluded", water_bridges="excluded")
            if ifp_from_columns(changed, allow_missing=True) != values:
                raise AssertionError("Excluded interaction leaked into selected IFP")
            embedded_rows += all(row.get(name, "").strip() for name in embedding_columns)
            record = {"complex_id": identifier}
            if values is not None:
                record["ifp"] = dict(zip(IFP_NAMES, values))
            records.append(record)
    if not records:
        raise ValueError("Input CSV has no records")
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Input changed during validation; discard this audit")
    result = {
        "source": str(source), "source_sha256": after, "rows": len(records),
        "unique_complex_ids": len(identifiers), "unique_smiles_strings": len(smiles),
        "ifp_labelled_rows": len(records) - len(missing), "ifp_unlabelled_rows": len(missing),
        "unlabelled_complex_ids": missing, "positive_counts": dict(positive),
        "model_order": list(IFP_NAMES), "excluded_interactions": list(IGNORED_IFP_COLUMNS),
        "rows_with_128_embedding_fields": embedded_rows, "fragment_column_present": "frag" in header,
        "excluded_columns_do_not_affect_inputs": True, "source_modified": False,
        "labels_output": str(labels_path) if labels_path else None,
        "protein_grouping_verified": False, "training_started": False,
        "note": "Keyed labels only; not a training split. Missing IFP is omitted, not zero-filled."
    }
    if labels_path:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        with labels_path.open("x") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--labels-output")
    args = parser.parse_args()
    print(json.dumps(validate(args.input, args.report, args.labels_output), indent=2))
