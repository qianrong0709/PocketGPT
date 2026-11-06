"""Stream a local CSV into a deduplicated molecule-only pretraining corpus.

Run from the project root with ``python -m scripts.prepare_molecule_corpus``.
Only SMILES are used: no fragment/condition rows in the source are changed.
There is no sampling, truncation, implicit filtering, or network access.
"""

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from pocketgpt.data import canonical_smiles
from pocketgpt.pocket import file_sha256
from pocketgpt.tokenizer import SPECIAL, SmilesTokenizer


def _publish_new_directory(staged, destination):
    """Atomically rename a complete directory, never replacing any target.

    Plain os.rename can replace an existing *empty* directory on POSIX. Use
    the native no-replace flag on the two deployment platforms instead. If a
    filesystem does not support it, fail safely and retain the staged corpus.
    """
    library = ctypes.CDLL(None, use_errno=True)
    arguments = (os.fsencode(staged), os.fsencode(destination))
    if sys.platform == "linux" and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, arguments[0], -100, arguments[1], 1)  # AT_FDCWD, RENAME_NOREPLACE
    elif sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(*arguments, 0x00000004)  # RENAME_EXCL, from macOS sys/stdio.h
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


def prepare_molecule_corpus(input_path, output_dir, smiles_column="smiles",
                            validation_fraction=0.1, max_context=None):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if max_context is not None and (type(max_context) is not int or max_context < 1):
        raise ValueError("max_context must be a positive integer, including the EOS token")
    if not isinstance(smiles_column, str) or not smiles_column.strip():
        raise ValueError("smiles_column must be a nonempty column name")
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")
    # Do not resolve the last component: a dangling symlink is an existing,
    # protected destination too, rather than permission to replace its target.
    requested = Path(output_dir).absolute()
    destination = requested.parent.resolve() / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Corpus destination exists; choose a new output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{destination.name}.pending-", dir=destination.parent))
    seen_raw, seen_canonical, token_types = set(), set(), set()
    counts = {"train": 0, "valid": 0}
    rows = raw_duplicates = canonical_duplicates = max_tokens = max_context_tokens = 0
    rejects = Counter()
    source_before = None

    def snapshot(status):
        return {
            "status": status, "stage": "pretrain", "source": str(source),
            "source_sha256": source_before, "smiles_column": smiles_column,
            "source_total_rows": rows, "unique_raw_smiles": len(seen_raw),
            "unique_canonical_smiles": len(seen_canonical),
            "raw_duplicate_rows": raw_duplicates, "canonical_duplicate_raw_strings": canonical_duplicates,
            "duplicate_rows": raw_duplicates + canonical_duplicates,
            "counts": dict(counts), "reject_reasons": dict(rejects),
            "max_smiles_tokens": max_tokens, "max_context_tokens_including_eos": max_context_tokens,
            "max_context_requested": max_context, "truncation_applied": False,
            "sampling_applied": False, "invalid_rows_skipped": False,
            "split": "canonical_smiles_sha256_first_64_bits", "validation_fraction": validation_fraction,
            "stereochemistry_preserved": True,
            "vocabulary_scope": "token types across all unique canonical molecules; no frequencies or weights learned",
            "vocabulary_size": len(token_types) + len(SPECIAL),
            "source_condition_rows_modified": False, "output_dir": str(destination),
        }

    try:
        source_before = file_sha256(source)
        with source.open(encoding="utf-8-sig", newline="") as stream, \
                (staged / "train.jsonl").open("x", encoding="utf-8") as train, \
                (staged / "valid.jsonl").open("x", encoding="utf-8") as valid:
            reader = csv.reader(stream)
            header = next(reader, None)
            if not header:
                raise ValueError("Input CSV has no header")
            normalized = [name.strip().lower() for name in header]
            if len(normalized) != len(set(normalized)):
                raise ValueError("Duplicate normalized CSV column names")
            column = smiles_column.strip().lower()
            if column not in normalized:
                raise ValueError(f"SMILES column not found: {smiles_column}")
            index = normalized.index(column)
            for fields in reader:
                rows += 1
                if rows % 100000 == 0:
                    print(json.dumps({"status": "preparing", "source_rows": rows,
                                      "unique_molecules": len(seen_canonical),
                                      "max_context_tokens": max_context_tokens}), flush=True)
                try:
                    if len(fields) != len(header):
                        raise ValueError("CSV record field count differs from header")
                    raw = fields[index]
                    if raw in seen_raw:
                        raw_duplicates += 1
                        continue
                    seen_raw.add(raw)
                    canonical = canonical_smiles(raw)
                    if canonical in seen_canonical:
                        canonical_duplicates += 1
                        continue
                    tokens = SmilesTokenizer.split(canonical)
                    length = len(tokens) + 1
                    max_tokens = max(max_tokens, len(tokens))
                    max_context_tokens = max(max_context_tokens, length)
                    if max_context is not None and length > max_context:
                        raise ValueError(f"Molecule needs {length} context tokens including EOS, above --max-context {max_context}; no truncation or filtering performed")
                    seen_canonical.add(canonical)
                    token_types.update(tokens)
                    score = int(hashlib.sha256(canonical.encode()).hexdigest()[:16], 16)
                    split = "valid" if score / 2**64 < validation_fraction else "train"
                    target = valid if split == "valid" else train
                    target.write(json.dumps({"smiles": canonical}, allow_nan=False) + "\n")
                    counts[split] += 1
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    reason = type(error).__name__ + ": " + str(error)
                    rejects[reason] += 1
                    raise ValueError(f"Source record {rows} (CSV line {reader.line_num}): {error}") from error
        if not rows:
            raise ValueError("Input CSV contains no molecule records")
        if not counts["train"] or not counts["valid"]:
            raise ValueError("Canonical split produced an empty partition; choose an explicit validation fraction or inspect the source")
        source_after = file_sha256(source)
        if source_after != source_before:
            raise RuntimeError("Input changed during preparation; corpus will not be published")
        tokenizer = SmilesTokenizer([*SPECIAL, *sorted(token_types)])
        tokenizer.save(staged / "vocabulary.json")
        result = snapshot("complete")
        result.update(source_sha256_after=source_after, source_modified=False,
                      output_sha256={name: file_sha256(staged / name)
                                     for name in ("train.jsonl", "valid.jsonl", "vocabulary.json")})
        _write_json(staged / "preparation.json", result)
        _publish_new_directory(staged, destination)
        return result
    except BaseException as error:
        failure = snapshot("failed_not_published")
        failure.update(error=type(error).__name__ + ": " + str(error), staged_dir=str(staged))
        # Keep partial files private for diagnosis. Nothing appears under the
        # requested final corpus directory unless the whole run succeeds.
        if staged.is_dir():
            _write_json(staged / "failure.json", failure)
        if hasattr(error, "add_note"):
            error.add_note(f"No corpus published. Local diagnostic directory: {staged}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-context", type=int)
    args = parser.parse_args()
    result = prepare_molecule_corpus(args.input, args.output_dir, args.smiles_column,
                                     args.validation_fraction, args.max_context)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
