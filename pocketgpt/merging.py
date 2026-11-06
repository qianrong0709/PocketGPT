"""Local many-to-one join of single fragments to the authoritative complex table."""

import csv
import json
import math
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, rdBase

from .config import IFP_NAMES, PROPERTIES
from .data import property_vector
from .ifp import IGNORED_IFP_COLUMNS, ifp_from_columns
from .pocket import file_sha256
from .tokenizer import SmilesTokenizer

EMBEDDING_COLUMNS = ("evalue", *[f"unnamed: {i}" for i in range(3, 130)])
EMBEDDING_INDEX = {name: i for i, name in enumerate(EMBEDDING_COLUMNS)}
ENCODER_SIGNATURE = "legacy-precomputed-unverified"


def _reader(stream, required):
    reader = csv.DictReader(stream)
    if reader.fieldnames is None:
        raise ValueError("CSV has no header")
    reader.fieldnames = [name.strip().lower() for name in reader.fieldnames]
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("Duplicate normalized CSV columns")
    missing = set(required) - set(reader.fieldnames)
    if missing:
        raise ValueError("CSV missing required columns: " + repr(sorted(missing)))
    return reader


def merge_condition_csvs(base_csv, fragment_csv, output_dir, progress=None):
    """Keep all source fragment rows, omit unavailable conditions, never split.

    The base owns pocket/properties/IFP. Old values repeated in the fragment
    table are compared diagnostically but can never overwrite base values.
    """
    base_csv, fragment_csv = Path(base_csv).resolve(), Path(fragment_csv).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Merge output directory already exists; choose a new directory")
    if base_csv == fragment_csv:
        raise ValueError("Use separate base and fragment CSV inputs")
    source_hashes = {str(p): file_sha256(p) for p in (base_csv, fragment_csv)}

    @lru_cache(maxsize=65536)
    def molecule(text):
        if not text.strip():
            raise ValueError("SMILES must be nonempty")
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(text)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("Invalid SMILES: " + text[:80])
        return Chem.MolToSmiles(mol, canonical=True), mol

    def token_length(text):
        try:
            return len(SmilesTokenizer.split(text))
        except ValueError:
            return None

    base, parent_lengths = {}, {}
    with base_csv.open(encoding="utf-8-sig", newline="") as stream:
        reader = _reader(stream, ("file", "smiles", *PROPERTIES, *EMBEDDING_COLUMNS))
        if ifp_from_columns(dict.fromkeys(reader.fieldnames, 0)) is None:
            raise ValueError("Base CSV needs the six named IFP columns")
        for line, raw in enumerate(reader, 2):
            identifier = (raw.get("file") or "").strip()
            if not identifier or identifier in base:
                raise ValueError(f"Missing or duplicate base file ID at line {line}: {identifier}")
            smiles, _ = molecule(raw["smiles"].strip())
            props = {name: float(raw[name]) for name in PROPERTIES}
            property_vector(props)
            embedding = [float(raw[name]) for name in EMBEDDING_COLUMNS]
            if not all(math.isfinite(value) for value in embedding):
                raise ValueError("Non-finite pocket embedding: " + identifier)
            record = {"complex_id": identifier, "smiles": smiles, "properties": props,
                      "pocket_embedding": embedding, "pocket_encoder_signature": ENCODER_SIGNATURE}
            ifp = ifp_from_columns(raw, allow_missing=True)
            if ifp is not None:
                record["ifp"] = ifp
            base[identifier] = record
            parent_lengths[identifier] = token_length(smiles)
    if not base:
        raise ValueError("Base CSV contains no records")

    output.parent.mkdir(parents=True, exist_ok=True)
    counts, availability, differences = Counter(), Counter(), Counter()
    seen_ids, with_fragments, mismatched_substructure_ids = set(), set(), set()
    pairs = set()
    difference_max = Counter()
    fragment_max_tokens = 0
    missing_ifp_ids = [key for key, row in base.items() if "ifp" not in row]
    fallback_ids = []
    csv_columns = ["complex_id", "smiles", *PROPERTIES, "frag", *IFP_NAMES,
                   *[f"pocket_{i:03d}" for i in range(128)], "has_fragment", "has_ifp",
                   "fragment_is_substructure", "source_fragment_row"]

    # Only this newly created temporary directory is cleaned if validation fails.
    # No final-looking delivery directory is published until every check passes.
    with tempfile.TemporaryDirectory(prefix="." + output.name + "-", dir=output.parent) as temporary:
        stage = Path(temporary)
        delivery = stage / "delivery"
        delivery.mkdir()
        with (delivery / "merged.jsonl").open("x") as json_stream, \
                (delivery / "merged.csv").open("x", newline="") as csv_stream:
            writer = csv.DictWriter(csv_stream, fieldnames=csv_columns)
            writer.writeheader()

            def emit(identifier, fragment=None, source_line=None):
                nonlocal fragment_max_tokens
                row = dict(base[identifier])
                present = fragment is not None
                if present:
                    row["fragment"] = fragment
                    _, parent_mol = molecule(row["smiles"])
                    _, frag_mol = molecule(fragment)
                    match = bool(parent_mol.HasSubstructMatch(frag_mol))
                    row["fragment_is_substructure"] = match
                    if not match:
                        counts["fragment_substructure_mismatch_rows"] += 1
                        mismatched_substructure_ids.add(identifier)
                    with_fragments.add(identifier)
                    length = token_length(fragment)
                    if length is None:
                        counts["unsupported_fragment_tokenization_rows"] += 1
                    else:
                        fragment_max_tokens = max(fragment_max_tokens, length)
                        counts["fragment_rows_over_256_tokens"] += length > 256
                row["source_fragment_row"] = source_line
                known_ifp = "ifp" in row
                counts["output_rows"] += 1
                counts["missing_ifp_rows"] += not known_ifp
                counts["missing_fragment_rows"] += not present
                counts["all_four_conditions_rows"] += present and known_ifp
                parent_length = parent_lengths[identifier]
                counts["parent_rows_over_256_tokens"] += parent_length is not None and parent_length + 1 > 256
                availability[f"fragment={int(present)},ifp={int(known_ifp)}"] += 1
                pair = (identifier, fragment)
                if pair in pairs:
                    counts["duplicate_fragment_pairs"] += 1
                pairs.add(pair)
                flat = {"complex_id": identifier, "smiles": row["smiles"], **row["properties"],
                        "frag": fragment or "", "has_fragment": int(present), "has_ifp": int(known_ifp),
                        "fragment_is_substructure": row.get("fragment_is_substructure", ""),
                        "source_fragment_row": source_line if source_line is not None else ""}
                flat.update(zip(IFP_NAMES, row.get("ifp", [""] * 6)))
                flat.update((f"pocket_{i:03d}", value) for i, value in enumerate(row["pocket_embedding"]))
                writer.writerow(flat)
                json_stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                if progress is not None and counts["output_rows"] % 25000 == 0:
                    progress({"status": "merge_progress", "output_rows": counts["output_rows"]})

            with fragment_csv.open(encoding="utf-8-sig", newline="") as stream:
                reader = _reader(stream, ("file", "smiles", "frag"))
                check_fields = set(reader.fieldnames) & set((*PROPERTIES, *EMBEDDING_COLUMNS))
                for line, raw in enumerate(reader, 2):
                    identifier = (raw.get("file") or "").strip()
                    if identifier not in base:
                        raise ValueError(f"Unknown fragment file ID at line {line}: {identifier}")
                    smiles, _ = molecule(raw["smiles"].strip())
                    if smiles != base[identifier]["smiles"]:
                        raise ValueError(f"Parent SMILES mismatch at fragment line {line}: {identifier}")
                    for name in check_fields:
                        value = raw.get(name)
                        if value is None or not value.strip():
                            differences[name] += 1
                            continue
                        original = (base[identifier]["properties"][name] if name in PROPERTIES else
                                    base[identifier]["pocket_embedding"][EMBEDDING_INDEX[name]])
                        number = float(value)
                        if not math.isfinite(number) or not math.isclose(number, original, rel_tol=1e-7, abs_tol=1e-9):
                            differences[name] += 1
                            if math.isfinite(number):
                                difference_max[name] = max(difference_max[name], abs(number - original))
                    text = (raw.get("frag") or "").strip()
                    fragment = None
                    if text:
                        fragment, mol = molecule(text)
                        if len(Chem.GetMolFrags(mol)) != 1:
                            raise ValueError(f"Expected one connected fragment at line {line}")
                    counts["fragment_source_rows"] += 1
                    seen_ids.add(identifier)
                    emit(identifier, fragment, line)
            for identifier in base:
                if identifier not in seen_ids:
                    fallback_ids.append(identifier)
                    emit(identifier)

        if {str(p): file_sha256(p) for p in (base_csv, fragment_csv)} != source_hashes:
            raise RuntimeError("An input changed during merge; no merged files were published")
        # Explicit zero counters make the report stable for downstream audits.
        report = {key: counts[key] for key in (
            "output_rows", "fragment_source_rows", "missing_ifp_rows", "missing_fragment_rows",
            "all_four_conditions_rows", "duplicate_fragment_pairs", "fragment_substructure_mismatch_rows",
            "parent_rows_over_256_tokens", "fragment_rows_over_256_tokens", "unsupported_fragment_tokenization_rows")}
        report.update({"base_complexes": len(base), "fragment_source_complexes": len(seen_ids),
                       "missing_ifp_complexes": len(missing_ifp_ids), "missing_ifp_complex_ids": missing_ifp_ids,
                       "missing_fragment_complexes": len(set(base) - with_fragments),
                       "fallback_complexes": len(fallback_ids), "fallback_complex_ids": fallback_ids,
                       "fragment_substructure_mismatch_complexes": len(mismatched_substructure_ids),
                       "fragment_substructure_mismatch_id_examples": sorted(mismatched_substructure_ids)[:20],
                       "availability_rows": dict(availability), "source_hashes": source_hashes,
                       "source_files_modified": False, "join_key": "exact file ID", "join_validation": "many-to-one",
                       "field_authority": "base CSV owns properties, pocket embedding and six IFP types",
                       "fragment_source_ancillary_field_differences": dict(differences),
                       "fragment_source_ancillary_max_deltas": dict(difference_max),
                       "excluded_ifp_types": list(IGNORED_IFP_COLUMNS), "ifp_order": list(IFP_NAMES),
                       "pocket_encoder_signature": ENCODER_SIGNATURE,
                       "max_parent_tokens_including_eos": max((n + 1 for n in parent_lengths.values() if n is not None), default=0),
                       "max_fragment_tokens": fragment_max_tokens,
                       "unsupported_parent_tokenization_complexes": sum(n is None for n in parent_lengths.values()),
                       "protein_grouping_verified": False, "train_validation_split_created": False,
                       "training_started": False, "output_directory": str(output),
                       "note": "Rows are preserved, not deduplicated. Missing conditions are omitted, not zero-filled. "
                               "Substructure matches are diagnostics, not automatic exclusion rules. "
                               "Verify protein grouping and encoder provenance before training."})
        (delivery / "merge_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError("Output directory appeared during merge; refusing to overwrite")
        delivery.rename(output)
    return report
