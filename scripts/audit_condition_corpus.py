"""Read-only, full-table audit of v2 property/single-fragment labels.

This does not prepare, filter, repair, mask or publish any training corpus.
It diagnoses both the current strict chemical contract and raw tokenizability;
the latter never enables a different representation in training. Outputs only
record row locations, reason codes and SHA256 identifiers, not source SMILES.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, rdBase

from pocketgpt.config import PROPERTIES
from pocketgpt.data import canonical_fragment, canonical_smiles, property_vector
from pocketgpt.pocket import file_sha256
from pocketgpt.tokenizer import SmilesTokenizer


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_condition_corpus(input_path, output_dir, block_size=256,
                           check_substructure=False, progress_every=100000,
                           cache_size=65536):
    """Scan every parseable CSV record without changing its data or labels.

    A complete audit may find invalid labels. In that case ``status`` is still
    ``complete`` but ``strict_contract_passed`` is false and the CLI exits 2.
    CSV syntax errors, changed input, unexpected exceptions or interruption
    produce an incomplete report and propagate, rather than silently skipping
    the remainder. Missing labels are counted separately from invalid labels.
    """
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if type(progress_every) is not int or progress_every < 0:
        raise ValueError("progress_every must be a nonnegative integer")
    if type(cache_size) is not int or cache_size < 1:
        raise ValueError("cache_size must be positive")
    if type(check_substructure) is not bool:
        raise ValueError("check_substructure must be boolean")
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError("Input CSV does not exist")
    requested = Path(output_dir).absolute()
    destination = requested.parent.resolve() / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Audit directory exists; use a new output directory")
    destination.mkdir(parents=True, exist_ok=False)

    counts = Counter()
    reasons = Counter()
    missing = Counter({name: 0 for name in ("smiles", "fragment", *PROPERTIES)})
    property_stats = {name: {"valid": 0, "invalid": 0, "min": None, "max": None}
                      for name in PROPERTIES}
    unique_parents, invalid_parents, invalid_fragments = set(), set(), set()
    source_before = source_after = None
    header = []
    scan_completed = False
    first_failure = None
    last_csv_line = 0
    reader = None

    @lru_cache(maxsize=cache_size)
    def chemistry(text, kind):
        canonical = None
        reason = None
        with rdBase.BlockLogs():
            try:
                canonical = (canonical_smiles if kind == "parent" else canonical_fragment)(text)
            except (ValueError, TypeError):
                # Diagnose without embedding the rejected text in a log/error.
                if not text.strip():
                    reason = "empty"
                else:
                    graph = Chem.MolFromSmiles(text.strip(), sanitize=False)
                    if graph is None:
                        reason = "syntax_parse_failed"
                    else:
                        flag = Chem.SanitizeMol(graph, catchErrors=True)
                        reason = ("sanitize_" + str(flag).lower()
                                  if flag != Chem.SanitizeFlags.SANITIZE_NONE
                                  else "canonicalization_rejected")
        raw_length = None
        try:
            raw_length = len(SmilesTokenizer.split(text))
        except ValueError:
            pass  # Expected diagnostic; chemical validity is reported separately.
        canonical_length = None
        if canonical is not None:
            try:
                canonical_length = len(SmilesTokenizer.split(canonical))
            except ValueError:
                reason = "canonical_tokens_unsupported"
        return canonical, canonical_length, raw_length, reason

    # These small caches exist only for the explicitly requested optional check.
    # Do not retain thousands of live RDKit graphs during a full-table audit.
    @lru_cache(maxsize=128)
    def graph(canonical):
        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(canonical)

    @lru_cache(maxsize=8192)
    def substructure(parent, fragment):
        return bool(graph(parent).HasSubstructMatch(graph(fragment)))

    def report():
        unchanged = (source_before == source_after
                     if source_before is not None and source_after is not None else None)
        complete = scan_completed and unchanged is True and first_failure is None
        effective = {name: counts[name + "_effective_rows"] for name in ("properties", "fragment")}
        strict = (complete and counts["records_read"] > 0 and
                  counts["invalid_rows"] == 0 and all(effective.values()))
        return {
            "status": "complete" if complete else "incomplete",
            "scan_completed": scan_completed,
            "source": str(source), "source_sha256": source_before,
            "source_sha256_after": source_after, "source_unchanged": unchanged,
            "source_total_records_read": counts["records_read"],
            "last_csv_line": last_csv_line, "header": header,
            "block_size": block_size, "counts": dict(counts),
            "missing": dict(missing), "property_statistics": property_stats,
            "issue_counts": dict(reasons),
            "unique_canonical_parent_molecules": len(unique_parents),
            "unique_invalid_parent_raw_sha256": len(invalid_parents),
            "unique_invalid_fragment_raw_sha256": len(invalid_fragments),
            "effective_condition_rows": effective,
            "strict_contract_passed": strict,
            "training_ready": False,
            "training_ready_note": "Audit only: split integrity, vocabulary, runtime, encoder and a training run require separate verification.",
            "raw_tokenization_diagnostic_only": True,
            "raw_representation_enabled": False,
            "max_context_tokens_including_eos": counts["max_context_tokens_including_eos"],
            "max_fragment_canonical_tokens": counts["max_fragment_canonical_tokens"],
            "max_parent_raw_tokens": counts["max_parent_raw_tokens"],
            "max_fragment_raw_tokens": counts["max_fragment_raw_tokens"],
            "fragment_substructure_check": "strict_rdkit_if_both_parse" if check_substructure else "not_performed",
            "fragment_substructure_failures_are_training_errors": False,
            "labels_repaired": False, "labels_masked": False,
            "rows_filtered": False, "source_written": False,
            "training_data_published": False,
            "first_failure": first_failure,
            "cache": {"max_chemical_entries": cache_size,
                      "chemical_hits": chemistry.cache_info().hits,
                      "chemical_misses": chemistry.cache_info().misses},
            "runtime": {"rdkit": rdBase.rdkitVersion},
        }

    def save_report():
        # This directory is exclusive to this run. Rewriting its own checkpoint
        # summary is not permitted to replace another existing audit directory.
        result = report()
        with (destination / "summary.json").open("w", encoding="utf-8") as output:
            json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
        return result

    try:
        source_before = file_sha256(source)
        with source.open(encoding="utf-8-sig", newline="") as handle, \
                (destination / "issues.jsonl").open("x", encoding="utf-8") as issues:
            reader = csv.reader(handle, strict=True)
            header = next(reader, [])
            last_csv_line = reader.line_num
            normalized = [name.strip().lower() for name in header]
            if not normalized or len(normalized) != len(set(normalized)):
                raise ValueError("Header missing or duplicate normalized column names")
            positions = {name: index for index, name in enumerate(normalized)}
            if {"smiles", "frag", *PROPERTIES} - positions.keys():
                raise ValueError("Required smiles/frag/four-property columns are absent")

            def issue(code, field, raw, severity="error", **detail):
                reasons[code] += 1
                counts[severity + "_issues"] += 1
                value = {"source_record": counts["records_read"], "csv_line": reader.line_num,
                         "field": field, "reason": code, "severity": severity,
                         "raw_sha256": _sha(raw), **detail}
                issues.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")

            for fields in reader:
                counts["records_read"] += 1
                last_csv_line = reader.line_num
                invalid = False
                if len(fields) != len(header):
                    issue("record_field_count", "record", json.dumps(fields, ensure_ascii=False),
                          expected_fields=len(header), observed_fields=len(fields))
                    counts["invalid_rows"] += 1
                    counts["malformed_width_rows"] += 1
                else:
                    counts["well_shaped_rows"] += 1
                    parent_text, fragment_text = fields[positions["smiles"]], fields[positions["frag"]]
                    parent, parent_length, raw_parent_length, parent_error = chemistry(parent_text, "parent")
                    if not parent_text.strip():
                        missing["smiles"] += 1
                    if raw_parent_length is None:
                        counts["parent_raw_untokenizable_rows"] += 1
                    else:
                        counts["parent_raw_tokenizable_rows"] += 1
                        counts["max_parent_raw_tokens"] = max(counts["max_parent_raw_tokens"], raw_parent_length)
                    if parent_error:
                        invalid = True
                        invalid_parents.add(_sha(parent_text))
                        issue("parent_" + parent_error, "smiles", parent_text)
                    if parent is not None:
                        unique_parents.add(hashlib.sha256(parent.encode("utf-8")).digest())
                    if parent_length is not None:
                        counts["parent_canonical_tokenizable_rows"] += 1
                        counts["max_context_tokens_including_eos"] = max(counts["max_context_tokens_including_eos"], parent_length + 1)
                        if parent_length + 1 > block_size:
                            invalid = True
                            issue("parent_context_overflow", "smiles", parent_text, tokens_with_eos=parent_length + 1)

                    fragment = None
                    if not fragment_text.strip():
                        missing["fragment"] += 1
                    else:
                        counts["fragment_present_rows"] += 1
                        fragment, fragment_length, raw_fragment_length, fragment_error = chemistry(fragment_text, "fragment")
                        if raw_fragment_length is None:
                            counts["fragment_raw_untokenizable_rows"] += 1
                        else:
                            counts["fragment_raw_tokenizable_rows"] += 1
                            counts["max_fragment_raw_tokens"] = max(counts["max_fragment_raw_tokens"], raw_fragment_length)
                        if fragment_error:
                            invalid = True
                            invalid_fragments.add(_sha(fragment_text))
                            issue("fragment_" + fragment_error, "frag", fragment_text)
                            if raw_fragment_length is not None:
                                counts["fragment_raw_tokenizable_but_strict_invalid_rows"] += 1
                        if fragment_length is not None:
                            counts["fragment_effective_rows"] += 1
                            counts["max_fragment_canonical_tokens"] = max(counts["max_fragment_canonical_tokens"], fragment_length)
                            if fragment_length > block_size:
                                invalid = True
                                issue("fragment_context_overflow", "frag", fragment_text, tokens=fragment_length)
                        if check_substructure and parent_length is not None and fragment_length is not None:
                            counts["substructure_checked_rows"] += 1
                            if not substructure(parent, fragment):
                                counts["substructure_mismatch_rows"] += 1
                                issue("fragment_not_parent_substructure", "frag", fragment_text, severity="warning")

                    values = {name: fields[positions[name]] for name in PROPERTIES}
                    present = {name: value for name, value in values.items() if value.strip()}
                    bad = set()
                    try:
                        property_vector(present)
                    except (ValueError, TypeError, OverflowError):
                        for name, value in present.items():
                            try:
                                property_vector({name: value})
                            except (ValueError, TypeError, OverflowError):
                                invalid = True
                                bad.add(name)
                                issue("property_invalid_" + name, name, value)
                    valid_properties = set(present) - bad
                    counts["properties_effective_rows"] += int(bool(valid_properties))
                    counts["all_properties_valid_present_rows"] += int(len(valid_properties) == len(PROPERTIES))
                    counts["all_properties_missing_rows"] += int(not present)
                    for name, value in values.items():
                        stats = property_stats[name]
                        if not value.strip():
                            missing[name] += 1
                        elif name in bad:
                            stats["invalid"] += 1
                        else:
                            number = float(value)
                            stats["valid"] += 1
                            stats["min"] = number if stats["min"] is None else min(stats["min"], number)
                            stats["max"] = number if stats["max"] is None else max(stats["max"], number)
                    counts["invalid_rows" if invalid else "strict_row_contract_passed_rows"] += 1
                if progress_every and counts["records_read"] % progress_every == 0:
                    print(json.dumps({"status": "auditing", "records_read": counts["records_read"],
                                      "invalid_rows": counts["invalid_rows"], "issues": sum(reasons.values())}), flush=True)
            scan_completed = True
            if not counts["records_read"]:
                reasons["empty_corpus"] += 1
                counts["error_issues"] += 1
            source_after = file_sha256(source)
            if source_after != source_before:
                raise RuntimeError("Source changed during audit")
        return save_report()
    except BaseException as error:
        first_failure = {"type": type(error).__name__,
                         "csv_line": reader.line_num if reader is not None else None,
                         "reason": "Audit did not finish successfully; see scan_completed and source_unchanged"}
        # Expected errors and interruption remain failures, never healthy output.
        # Do not print arbitrary exception text containing protected molecules.
        try:
            source_after = file_sha256(source)
        except OSError:
            source_after = None
        save_report()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--check-substructure", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args(argv)
    try:
        result = audit_condition_corpus(args.input, args.output_dir, args.block_size,
                                        args.check_substructure, args.progress_every)
    except (OSError, ValueError, RuntimeError, csv.Error, UnicodeError) as error:
        print(json.dumps({"status": "audit_failed_or_incomplete", "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["strict_contract_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
