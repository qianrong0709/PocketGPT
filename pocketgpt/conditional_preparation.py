"""Local, row-preserving preparation of ligand-only conditional pretraining.

The v2 CSV's molecular properties and single ``frag`` column are labels, not
extra independent molecules. Every source record is kept. Canonical parent
molecules, including different SMILES spellings and fragment expansions, are
assigned to one deterministic partition. No protein, pocket or IFP is inferred.
"""

import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, rdBase

from .config import PROPERTIES, DEFAULT_FRAGMENT_REPRESENTATION, validate_fragment_representation
from .data import fragment_text as represent_fragment_text, canonical_smiles, property_vector
from .pocket import file_sha256
from .tokenizer import SPECIAL, SmilesTokenizer


def _publish_new_directory(staged, destination):
    """Publish a whole corpus atomically, without replacing even an empty dir."""
    library = ctypes.CDLL(None, use_errno=True)
    arguments = (os.fsencode(staged), os.fsencode(destination))
    if sys.platform == "linux" and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, arguments[0], -100, arguments[1], 1)
    elif sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(*arguments, 0x00000004)
    else:
        raise RuntimeError("Atomic no-replace directory publication is unavailable on this platform")
    if result:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(error, "Corpus destination exists; no files replaced", str(destination))
        raise OSError(error, os.strerror(error), str(destination))


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _names(values, allowed, name, allow_empty=False):
    if isinstance(values, str) or not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a tuple or list of explicit names")
    result = tuple(values)
    if (not result and not allow_empty) or any(not isinstance(x, str) for x in result):
        raise ValueError(f"{name} must contain explicit names")
    if len(result) != len(set(result)) or set(result) - set(allowed):
        raise ValueError(f"Unknown or duplicate {name}: {result!r}")
    return result


def prepare_condition_pretrain(input_path, output_dir, validation_fraction=0.1,
                               block_size=256, properties=PROPERTIES,
                               conditions=("properties", "fragment"),
                               fragment_representation=DEFAULT_FRAGMENT_REPRESENTATION):
    """Stream a v2 CSV to new train/valid JSONL, vocabulary and audit report.

    ``conditions`` records the intended training combination; it does not erase
    available labels. ``properties`` selects raw-unit model property fields.
    Unselected CSV columns (including LogS) retain their exact source strings in
    ``metadata.source_values``. Missing labels are absent, never invented zeroes.
    Molecular/fragment stereochemistry and attachment-point atoms are retained.

    Every row must have valid parent SMILES. In explicit raw-smiles-v1 mode,
    fragment chemistry errors are reported without changing/removing labels.
    All labels must be tokenizable and fit context; no filtering/truncation.
    Parent length includes EOS, whereas the fragment encoder does not use EOS.
    Successful publication is atomic and never replaces an existing target.
    Failed staged output is retained locally with ``failure.json`` diagnostics.
    """
    if (isinstance(validation_fraction, bool) or
            not isinstance(validation_fraction, (int, float)) or
            not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be finite and between 0 and 1")
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    representation = validate_fragment_representation(fragment_representation)
    selected = _names(properties, PROPERTIES, "properties", allow_empty=True)
    enabled = _names(conditions, ("properties", "fragment"), "conditions")
    if "properties" in enabled and not selected:
        raise ValueError("The properties condition needs at least one selected property")
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")
    requested = Path(output_dir).absolute()
    destination = requested.parent.resolve() / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Corpus destination exists; choose a new output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{destination.name}.pending-", dir=destination.parent))
    parents, token_types = set(), set()
    counts = {"train": 0, "valid": 0}
    missing_properties = Counter({name: 0 for name in selected})
    available_properties = Counter({name: 0 for name in selected})
    rejects = Counter()
    rows = missing_fragment = missing_all = max_smiles = max_context = max_fragment = 0
    chemical_counts = Counter({"valid_rows": 0, "invalid_rows_retained": 0})
    chemical_invalid_examples = []
    source_before = source_after = None
    header = []

    # A bounded cache speeds fragment-expanded CSVs without loading the table.
    # Unique parent digests are retained only to report an exact parent count.
    @lru_cache(maxsize=65536)
    def molecule(text):
        try:
            with rdBase.BlockLogs():
                canonical = canonical_smiles(text)
        except (ValueError, TypeError):
            raise ValueError("Invalid parent SMILES") from None
        try:
            tokens = tuple(SmilesTokenizer.split(canonical))
        except ValueError:
            raise ValueError("Unsupported parent SMILES tokens") from None
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return canonical, tokens, digest

    @lru_cache(maxsize=65536)
    def fragment(text):
        try:
            with rdBase.BlockLogs():
                represented = represent_fragment_text(text, representation)
        except (ValueError, TypeError):
            message = ("Invalid fragment SMILES" if representation == DEFAULT_FRAGMENT_REPRESENTATION
                       else "Unsupported raw fragment tokenizer text")
            raise ValueError(message) from None
        try:
            tokens = tuple(SmilesTokenizer.split(represented))
        except ValueError:
            raise ValueError("Unsupported fragment SMILES tokens") from None
        if representation == "raw-smiles-v1":
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(text)
            chemically_valid = mol is not None and mol.GetNumAtoms() > 0
        else:
            chemically_valid = True
        return represented, tokens, chemically_valid

    def snapshot(status):
        return {
            "status": status, "stage": "condition_pretrain", "source": str(source),
            "source_sha256": source_before, "source_sha256_after": source_after,
            "source_total_rows": rows, "total_rows": rows, "counts": dict(counts),
            "train_rows": counts["train"], "valid_rows": counts["valid"],
            "unique_canonical_smiles": len(parents), "unique_parent_molecules": len(parents),
            "conditions": list(enabled), "property_names": list(selected),
            "fragment_representation": representation,
            "fragment_original_strings_preserved": representation == "raw-smiles-v1",
            "fragment_chemical_validation": {**dict(chemical_counts),
                                             "invalid_source_record_examples": list(chemical_invalid_examples),
                                             "invalid_labels_deleted_or_masked": False},
            "missing": {"properties": dict(missing_properties), "fragment": missing_fragment,
                        "all_selected_conditions": missing_all},
            "label_count_scope": "successfully written rows",
            "available_properties": dict(available_properties),
            "max_smiles_tokens": max_smiles, "max_context_tokens_including_eos": max_context,
            "max_fragment_tokens": max_fragment, "block_size": block_size,
            "reject_reasons": dict(rejects), "row_deduplication_applied": False,
            "fragment_substructure_check": "not_performed", "fragment_labels_masked": False,
            "sampling_applied": False, "truncation_applied": False,
            "invalid_rows_skipped": False,
            "source_modified": (source_before != source_after
                                if source_before is not None and source_after is not None else None),
            "stereochemistry_preserved": True,
            "split": "canonical_smiles_sha256_first_64_bits",
            "validation_fraction": validation_fraction,
            "vocabulary_scope": "parent and " + representation + " fragment token types in both partitions; no frequencies or weights learned",
            "vocabulary_size": len(token_types) + len(SPECIAL), "output_dir": str(destination),
            "schema": {"source_columns": list(header), "smiles_column": "smiles",
                       "fragment_column": "frag", "property_names": list(selected),
                       "property_units": "source raw values; model scaling occurs in BatchBuilder",
                       "model_fields": ["smiles", "properties", "fragment"],
                       "provenance_fields": ["source_row", "molecule_id", "metadata"],
                       "unselected_columns": "metadata.source_values; exact source strings",
                       "missing_labels": "absent field or absent property key, not zero",
                       "excluded_modalities": ["pocket", "ifp", "protein_id"]},
        }

    try:
        source_before = file_sha256(source)
        with source.open(encoding="utf-8-sig", newline="") as stream, \
                (staged / "train.jsonl").open("x", encoding="utf-8") as train, \
                (staged / "valid.jsonl").open("x", encoding="utf-8") as valid:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            if not header:
                raise ValueError("Input CSV has no header")
            normalized = [name.strip().lower() for name in header]
            if len(normalized) != len(set(normalized)):
                raise ValueError("Duplicate normalized CSV column names")
            indices = {name: index for index, name in enumerate(normalized)}
            required = {"smiles", *selected}
            if "fragment" in enabled:
                required.add("frag")
            if required - indices.keys():
                raise ValueError("Required CSV columns not found: " + repr(sorted(required - indices.keys())))
            modeled_columns = {"smiles", *selected}
            if "frag" in indices:
                modeled_columns.add("frag")
            for fields in reader:
                rows += 1
                if rows % 100000 == 0:
                    print(json.dumps({"status": "preparing", "stage": "condition_pretrain",
                                      "source_rows": rows, "unique_parent_molecules": len(parents),
                                      "max_context_tokens": max_context}), flush=True)
                try:
                    if len(fields) != len(header):
                        raise ValueError("CSV record field count differs from header")
                    canonical, parent_tokens, digest = molecule(fields[indices["smiles"]])
                    parent_length = len(parent_tokens) + 1
                    max_smiles = max(max_smiles, len(parent_tokens))
                    max_context = max(max_context, parent_length)
                    if parent_length > block_size:
                        raise ValueError(f"Molecule needs {parent_length} context tokens including EOS, above block_size {block_size}; no truncation or filtering performed")
                    values = {name: fields[indices[name]] for name in selected
                              if fields[indices[name]].strip()}
                    property_vector(values)  # shared range/finite/float32 checks
                    values = {name: float(value) for name, value in values.items()}
                    row = {"smiles": canonical, "properties": values,
                           "source_row": rows, "molecule_id": digest}
                    fragment_text = fields[indices["frag"]] if "frag" in indices else ""
                    fragment_tokens = ()
                    if fragment_text.strip():
                        represented_frag, fragment_tokens, chemically_valid = fragment(fragment_text)
                        max_fragment = max(max_fragment, len(fragment_tokens))
                        if len(fragment_tokens) > block_size:
                            raise ValueError(f"Fragment needs {len(fragment_tokens)} tokens, above block_size {block_size}; no truncation or filtering performed")
                        row["fragment"] = represented_frag
                        row["fragment_representation"] = representation
                        row["fragment_chemical_valid"] = chemically_valid
                        chemical_counts["valid_rows" if chemically_valid else "invalid_rows_retained"] += 1
                        if not chemically_valid and len(chemical_invalid_examples) < 20:
                            chemical_invalid_examples.append(rows)
                    metadata = {header[index]: value for index, value in enumerate(fields)
                                if normalized[index] not in modeled_columns}
                    if metadata:
                        row["metadata"] = {"source_values": metadata}
                    split = "valid" if int(digest[:16], 16) / 2**64 < validation_fraction else "train"
                    target = valid if split == "valid" else train
                    target.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    parents.add(digest)
                    token_types.update(parent_tokens)
                    token_types.update(fragment_tokens)
                    for name in selected:
                        if name in values:
                            available_properties[name] += 1
                        else:
                            missing_properties[name] += 1
                    missing_fragment += int(not fragment_tokens)
                    has_selected = (("properties" in enabled and bool(values)) or
                                    ("fragment" in enabled and bool(fragment_tokens)))
                    missing_all += int(not has_selected)
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    rejects[type(error).__name__ + ": " + str(error)] += 1
                    raise ValueError(f"Source record {rows} (CSV line {reader.line_num}): {error}") from error
        if not rows:
            raise ValueError("Input CSV contains no molecule records")
        if not counts["train"] or not counts["valid"]:
            raise ValueError("Canonical split produced an empty partition; choose an explicit validation fraction or inspect the source")
        source_after = file_sha256(source)
        if source_after != source_before:
            raise RuntimeError("Input changed during preparation; corpus will not be published")
        SmilesTokenizer([*SPECIAL, *sorted(token_types)]).save(staged / "vocabulary.json")
        report = snapshot("complete")
        report["output_sha256"] = {name: file_sha256(staged / name)
                                   for name in ("train.jsonl", "valid.jsonl", "vocabulary.json")}
        _write_json(staged / "preparation.json", report)
        _publish_new_directory(staged, destination)
        return report
    except BaseException as error:
        failure = snapshot("failed_not_published")
        failure.update(error=type(error).__name__ + ": " + str(error), staged_dir=str(staged))
        if staged.is_dir():
            try:
                _write_json(staged / "failure.json", failure)
            except OSError as diagnostic_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"Could not write local failure diagnostics: {diagnostic_error}")
        if hasattr(error, "add_note"):
            error.add_note(f"No corpus published. Local diagnostic directory: {staged}")
        raise
