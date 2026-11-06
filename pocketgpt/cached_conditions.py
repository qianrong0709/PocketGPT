"""Export minimal six-IFP targets and attach them to an existing cached CSV.

No encoder is run and no molecule/fragment/vector field is changed. Parent
canonicalization serves only identity hashing, never rewriting cached cells.
The minimal exported table contains no SMILES, properties or pocket vectors.
"""
import csv
import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from rdkit import rdBase

from .config import IFP_NAMES
from .data import canonical_smiles
from .ifp import IFP_COLUMN_ALIASES, LEGACY_IFP_COLUMNS, ifp_from_columns
from .pocket import file_sha256

TARGET_COLUMNS = ("complex_id", "parent_smiles_sha256", *IFP_NAMES)
_EXISTING_IFP = {"ifp", *LEGACY_IFP_COLUMNS,
                 *(alias for names in IFP_COLUMN_ALIASES.values() for alias in names)}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _source(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Required local CSV source is absent")
    return path


def _destination(output_csv, sources):
    requested = Path(output_csv).absolute()
    destination = requested.parent.resolve() / requested.name
    sidecar = destination.with_suffix(".report.json")
    if destination == sidecar:
        raise ValueError("CSV and report destinations must differ")
    source_ids = [(path.stat().st_dev, path.stat().st_ino) for path in sources]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Input CSV tables must be distinct files")
    for output in (destination, sidecar):
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output CSV/report already exists; no files replaced")
        if output.resolve() in sources:
            raise ValueError("Input and output paths must be distinct")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination, sidecar


def _reader(stream, required, *, exact=False):
    reader = csv.reader(stream, strict=True)
    header = next(reader, [])
    if not header or any(not name.strip() for name in header):
        raise ValueError("CSV needs nonempty column names")
    normalized = [name.strip().lower() for name in header]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate normalized CSV column names")
    positions = {name: index for index, name in enumerate(normalized)}
    if set(required) - set(positions) or (exact and set(positions) != set(required)):
        raise ValueError("CSV column schema does not match the required named fields")
    return reader, header, positions


def _rows(reader, width):
    for record, values in enumerate(reader, 1):
        if len(values) != width:
            raise ValueError("CSV record %d has an unexpected field count" % record)
        yield record, values


def _identifier(value, record):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("CSV record %d has an empty exact complex ID" % record)
    # Deliberately do not strip, lowercase or reduce to a PDB prefix.
    return value


@lru_cache(maxsize=65536)
def _parent_hash(smiles):
    try:
        with rdBase.BlockLogs():
            canonical = canonical_smiles(smiles)
    except (ValueError, TypeError):
        raise ValueError("Invalid parent SMILES; no input row changed or filtered") from None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _labels(values, record):
    try:
        return ifp_from_columns(values, allow_missing=True)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("CSV record %d has missing, partial or invalid six-IFP labels" % record) from None


def _temporary(destination, suffix):
    descriptor, name = tempfile.mkstemp(prefix="." + destination.name + ".pending-", suffix=suffix,
                                        dir=destination.parent)
    return descriptor, Path(name)


def _publish(csv_staged, report_staged, destination, sidecar):
    """No-replace publication, including races; undo only links we created."""
    published = []
    try:
        for staged, output in ((report_staged, sidecar), (csv_staged, destination)):
            os.link(staged, output)
            published.append((staged, output))
    except BaseException:
        for staged, output in reversed(published):
            try:
                if output.samefile(staged):
                    output.unlink()
            except FileNotFoundError:
                pass
        raise


def _publish_report(csv_staged, destination, sidecar, report):
    descriptor, staged = _temporary(destination, ".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        _publish(csv_staged, staged, destination, sidecar)
    finally:
        staged.unlink(missing_ok=True)


def _field_digest(digest, fields):
    digest.update(json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")


def export_ifp_targets(base_csv, output_csv):
    """Export exact file ID, canonical-parent SHA256 and six named IFP values.

    Missing all-six labels are six blank cells; partial observations are an
    error. Extra source fields (including vectors) never enter this output.
    Sidecar reports contain hashes/counts/order, not private source paths.
    """
    source = _source(base_csv)
    destination, sidecar = _destination(output_csv, (source,))
    source_hash = file_sha256(source)
    descriptor, staged = _temporary(destination, ".csv")
    rows = missing = 0
    identities = set()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output, \
                source.open(encoding="utf-8-sig", newline="") as original:
            reader, header, positions = _reader(original, ("file", "smiles"))
            # Header validation is independent of missing values in data rows.
            names = set(positions)
            for aliases in IFP_COLUMN_ALIASES.values():
                found = names.intersection(aliases)
                if len(found) != 1:
                    raise ValueError("Export requires one named source column for each of the six IFP types")
            writer = csv.writer(output)
            writer.writerow(TARGET_COLUMNS)
            for record, fields in _rows(reader, len(header)):
                identifier = _identifier(fields[positions["file"]], record)
                if identifier in identities:
                    raise ValueError("Duplicate exact complex ID in IFP source at record %d" % record)
                identities.add(identifier)
                parent = _parent_hash(fields[positions["smiles"]])
                labels = _labels(dict(zip(header, fields)), record)
                writer.writerow((identifier, parent, *(labels if labels is not None else [""] * 6)))
                rows += 1
                missing += labels is None
        if not rows:
            raise ValueError("IFP source has no data records")
        after = file_sha256(source)
        if after != source_hash:
            raise RuntimeError("IFP source changed during export; no output published")
        report = {"operation": "export_ifp_targets", "source_sha256": source_hash,
                  "source_sha256_after": after, "source_unchanged": True,
                  "source_rows": rows, "output_rows": rows, "unique_complexes": len(identities),
                  "ifp_observed_rows": rows - missing, "ifp_missing_rows": missing,
                  "ifp_order": list(IFP_NAMES), "output_sha256": file_sha256(staged)}
        _publish_report(staged, destination, sidecar, report)
        return report
    finally:
        staged.unlink(missing_ok=True)


def _target_map(source):
    targets = {}
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader, header, positions = _reader(stream, TARGET_COLUMNS, exact=True)
        for record, fields in _rows(reader, len(header)):
            identifier = _identifier(fields[positions["complex_id"]], record)
            if identifier in targets:
                raise ValueError("Duplicate exact complex ID in IFP target table at record %d" % record)
            parent = fields[positions["parent_smiles_sha256"]]
            if not _SHA256.fullmatch(parent):
                raise ValueError("Invalid canonical-parent SHA256 in IFP target record %d" % record)
            targets[identifier] = (parent, _labels(dict(zip(header, fields)), record))
    if not targets:
        raise ValueError("IFP target table has no data records")
    return targets


def _verify_preserved_prefix(staged, original_header, expected_rows, expected_digest):
    digest, rows = hashlib.sha256(), 0
    with staged.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if header != [*original_header, *IFP_NAMES]:
            raise RuntimeError("Staged output columns differ from source plus six IFP names")
        for _, fields in _rows(reader, len(header)):
            _field_digest(digest, fields[:len(original_header)])
            rows += 1
    if rows != expected_rows or digest.hexdigest() != expected_digest:
        raise RuntimeError("Cached source row/field preservation verification failed")


def attach_ifp_targets(cached_csv, ifp_csv, output_csv, expected_cached_sha256=None):
    """Append six IFP fields by exact complex ID AND canonical parent hash.

    All cached columns, row order, duplicate/expanded rows and fragment cell
    strings are preserved. Every target complex must occur in cached_csv, and
    every cached ID must be known to the target table. Existing IFP columns are
    rejected rather than overwritten. No raw structure or encoder is needed.
    """
    cached, labels = _source(cached_csv), _source(ifp_csv)
    destination, sidecar = _destination(output_csv, (cached, labels))
    source_hashes = {"cached": file_sha256(cached), "ifp_targets": file_sha256(labels)}
    if expected_cached_sha256 is not None:
        if not isinstance(expected_cached_sha256, str) or not _SHA256.fullmatch(expected_cached_sha256):
            raise ValueError("expected_cached_sha256 must be a lowercase 64-character SHA256")
        if source_hashes["cached"] != expected_cached_sha256:
            raise ValueError("Cached CSV SHA256 differs from the explicitly expected source")
    targets = _target_map(labels)
    descriptor, staged = _temporary(destination, ".csv")
    rows = missing = 0
    seen, missing_ids = set(), set()
    field_digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output, \
                cached.open(encoding="utf-8-sig", newline="") as original:
            reader, header, positions = _reader(original, ("file", "smiles"))
            if set(positions).intersection(_EXISTING_IFP):
                raise ValueError("Cached CSV already has IFP columns; refusing to overwrite or mix labels")
            writer = csv.writer(output)
            writer.writerow([*header, *IFP_NAMES])
            for record, fields in _rows(reader, len(header)):
                identifier = _identifier(fields[positions["file"]], record)
                if identifier not in targets:
                    raise ValueError("Unknown cached exact complex ID at record %d; no prefix matching is allowed" % record)
                parent, values = targets[identifier]
                if _parent_hash(fields[positions["smiles"]]) != parent:
                    raise ValueError("Canonical parent identity differs for cached record %d" % record)
                writer.writerow([*fields, *(values if values is not None else [""] * 6)])
                _field_digest(field_digest, fields)
                rows += 1
                missing += values is None
                seen.add(identifier)
                if values is None:
                    missing_ids.add(identifier)
        missing_target_ids = set(targets) - seen
        if missing_target_ids:
            raise ValueError("%d target complex IDs are missing from the cached CSV" % len(missing_target_ids))
        if not rows:
            raise ValueError("Cached CSV has no data records")
        _verify_preserved_prefix(staged, header, rows, field_digest.hexdigest())
        after = {"cached": file_sha256(cached), "ifp_targets": file_sha256(labels)}
        if after != source_hashes:
            raise RuntimeError("Cached CSV or IFP target table changed during attachment; no output published")
        report = {"operation": "attach_ifp_targets", "source_sha256": source_hashes,
                  "source_sha256_after": after, "sources_unchanged": True,
                  "cached_rows": rows, "output_rows": rows, "unique_cached_complexes": len(seen),
                  "ifp_target_complexes": len(targets), "unknown_cached_complexes": 0,
                  "missing_target_complexes": 0, "ifp_missing_rows": missing,
                  "ifp_missing_complexes": len(missing_ids), "ifp_observed_rows": rows - missing,
                  "ifp_order": list(IFP_NAMES), "original_column_count": len(header),
                  "output_column_count": len(header) + len(IFP_NAMES),
                  "original_field_values_and_row_order_verified": True,
                  "original_fields_sha256": field_digest.hexdigest(),
                  "fragment_column_present": "frag" in positions,
                  "fragment_labels_modified_or_masked": False,
                  "output_sha256": file_sha256(staged)}
        _publish_report(staged, destination, sidecar, report)
        return report
    finally:
        staged.unlink(missing_ok=True)
