import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .config import (PROPERTIES, DEFAULT_FRAGMENT_REPRESENTATION,
                     validate_fragment_representation)
from .data import fragment_text, canonical_smiles, property_vector
from .ifp import (IGNORED_IFP_COLUMNS, canonical_ifp, has_named_ifp_columns,
                  ifp_from_columns, validate_ifp_schema)
from .pocket import file_sha256, validate_embedding, validate_pocket_sources


def prepare_dataset(input_path, output_dir, stage, legacy=False, protein_map=None,
                    validation_fraction=0.1, limit=None, skip_invalid=False, ifp_schema=None,
                    *, split_group="protein_id",
                    fragment_representation=DEFAULT_FRAGMENT_REPRESENTATION):
    if stage not in ("pretrain", "finetune"):
        raise ValueError("stage must be pretrain or finetune")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if split_group not in ("protein_id", "complex_id"):
        raise ValueError("split_group must be protein_id or complex_id")
    if stage != "finetune" and split_group != "protein_id":
        raise ValueError("Explicit complex_id grouping is only available for finetune")
    fragment_representation = validate_fragment_representation(fragment_representation)
    schema = validate_ifp_schema(json.loads(Path(ifp_schema).read_text())) if ifp_schema else None
    if schema is not None and stage != "finetune":
        raise ValueError("IFP schema is only used in conditional finetune preparation")
    source = Path(input_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("train.jsonl", "valid.jsonl", "preparation.json")):
        raise FileExistsError("Prepared dataset already exists; choose a new output directory")
    mapping = json.loads(Path(protein_map).read_text()) if protein_map else {}
    if legacy and stage == "finetune" and split_group == "protein_id" and not mapping:
        raise ValueError("Legacy file/PDB ids do not identify proteins. Supply --protein-map (JSON: file -> protein_id)")
    counts, rejected = Counter(), Counter()
    split_groups = {"train": set(), "valid": set()}
    ifp_rows = 0
    ifp_formats = Counter()
    with source.open(encoding="utf-8-sig") as stream, (output / "train.jsonl").open("w") as train, (output / "valid.jsonl").open("w") as valid:
        if legacy:
            reader = csv.DictReader(stream)
            reader.fieldnames = [name.strip().lower() for name in reader.fieldnames]
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError("Duplicate normalized CSV column names")
            if "smiles" not in reader.fieldnames:
                raise ValueError("CSV has no smiles column")
            embedding_columns = ["evalue", *[f"unnamed: {i}" for i in range(3, 130)]]
            if stage == "finetune" and not set(embedding_columns).issubset(reader.fieldnames):
                raise ValueError("Unrecognized legacy 128-column pocket schema; specify a mapping instead of guessing")
        else:
            reader = (json.loads(line) for line in stream if line.strip())
        for index, original in enumerate(reader):
            if limit is not None and index >= limit:
                break
            try:
                smiles = canonical_smiles(original["smiles"])
                if stage == "pretrain":
                    row = {"smiles": smiles}
                elif legacy:
                    identifier = original["file"]
                    if split_group == "protein_id" and identifier not in mapping:
                        raise ValueError("protein_map_missing")
                    embedding = validate_embedding([float(original[name]) for name in embedding_columns])
                    row = {"smiles": smiles, "complex_id": identifier,
                           "pocket_embedding": embedding.tolist(),
                           "pocket_encoder_signature": "legacy-precomputed-unverified",
                           "properties": {key: float(original[key]) for key in PROPERTIES if original.get(key, "").strip()}}
                    if identifier in mapping:
                        row["protein_id"] = str(mapping[identifier])
                    if original.get("frag", "").strip():
                        text = original["frag"]
                        if fragment_representation == "canonical-smiles-v1":
                            text = text.strip().rstrip("<")
                        row["fragment"] = fragment_text(text, fragment_representation)
                else:
                    row = dict(original, smiles=smiles)
                    if row.get("fragment"):
                        declared = row.get("fragment_representation")
                        if declared is not None and declared != fragment_representation:
                            raise ValueError("fragment_representation conflicts with source record")
                        row["fragment"] = fragment_text(row["fragment"], fragment_representation)
                    if row.get("pocket"):
                        row["pocket"] = dict(row["pocket"])
                        for key in ("protein_pdb", "pocket_pdb"):
                            if key in row["pocket"]:
                                row["pocket"][key] = str((source.parent / row["pocket"][key]).resolve())
                ifp_format = None
                if stage == "finetune":
                    group_value = row.get(split_group)
                    if (not isinstance(group_value, str) or not group_value.strip()):
                        raise ValueError(split_group + "_required_nonempty_string")
                    # A complex identifier is a real complex grouping key, never
                    # a fabricated protein label. Preserve any actual protein_id.
                    row["fragment_representation"] = fragment_representation
                    validate_pocket_sources(row)
                    # Validate observed targets now rather than accepting a
                    # dataset which only fails after a training run starts.
                    # Missing targets remain missing; no zero-filled labels.
                    property_vector(row.get("properties"))
                    if "pocket_embedding" in row:
                        validate_embedding(row["pocket_embedding"])
                    if legacy and has_named_ifp_columns(original):
                        # Named columns remain authoritative even if all labels
                        # are unavailable and a stale packed ifp column exists.
                        named_ifp = ifp_from_columns(original, allow_missing=True)
                        if named_ifp is not None:
                            row["ifp"] = named_ifp
                            ifp_format = "named_csv_columns"
                    elif original.get("ifp") is not None and original["ifp"] != "":
                        # Packed legacy vectors use the author's documented eight-column order.
                        row["ifp"] = canonical_ifp(original["ifp"], schema)
                        ifp_format = "mapped_eight_bit_vector" if schema else "canonical_six_types"
                group = row["smiles"] if stage == "pretrain" else row[split_group]
                score = int(hashlib.sha256(str(group).encode()).hexdigest()[:16], 16) / 2**64
                split = "valid" if score < validation_fraction else "train"
                (valid if split == "valid" else train).write(json.dumps(row, allow_nan=False) + "\n")
                counts[split] += 1
                split_groups[split].add(group)
                ifp_rows += int(ifp_format is not None)
                if ifp_format:
                    ifp_formats[ifp_format] += 1
            except (ValueError, TypeError, KeyError) as error:
                if not skip_invalid:
                    raise ValueError(f"Source row {index + 1}: {error}") from error
                rejected[type(error).__name__ + ":" + str(error)[:60]] += 1
    report = {"source": str(source), "source_sha256": file_sha256(source), "stage": stage,
              "split": "canonical_smiles" if stage == "pretrain" else split_group,
              "split_group": "canonical_smiles" if stage == "pretrain" else split_group,
              "validation_scope": ("complex_id_grouped_NOT_protein_heldout"
                                   if stage == "finetune" and split_group == "complex_id"
                                   else "protein_id_grouped_requires_verified_identity"
                                   if stage == "finetune" else "canonical_parent_molecule_grouped"),
              "protein_ids_invented": False,
              "fragment_representation": fragment_representation,
              "unique_groups": {name: len(groups) for name, groups in split_groups.items()},
              "group_overlap_count": len(split_groups["train"] & split_groups["valid"]),
              "counts": dict(counts), "rejected": dict(rejected),
              "legacy_encoder_identity_verified": False if legacy and stage == "finetune" else None,
              "ifp_labels_created": False, "ifp_labelled_rows": ifp_rows,
              "ifp_unlabelled_rows": sum(counts.values()) - ifp_rows if stage == "finetune" else None,
              "ifp_input_formats": dict(ifp_formats),
              "ifp_source_schema": schema if ifp_formats["mapped_eight_bit_vector"] else None,
              "ifp_ignored_source_indices": sorted(set(range(8)) - set(schema["indices"].values()))
              if schema and ifp_formats["mapped_eight_bit_vector"] else [],
              "ifp_excluded_interactions": list(IGNORED_IFP_COLUMNS)}
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not counts["train"] or not counts["valid"]:
        raise ValueError("Split produced an empty partition; inspect preparation.json and prepare into a new directory")
    return report
