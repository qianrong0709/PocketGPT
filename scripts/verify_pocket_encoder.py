"""Compare local pocket encodings with matched reference vectors from training."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pocketgpt.checkpoint import load_checkpoint
from pocketgpt.pocket import PocketResolver, content_hash, file_sha256, validate_embedding


def _tolerance(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number between 0 and {maximum}")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a finite number between 0 and {maximum}")
    return value


def _references(path):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid reference JSON at line {line_number}") from error
            if not isinstance(row, dict) or not isinstance(row.get("pocket"), dict):
                raise ValueError(f"Reference line {line_number} requires a pocket object")
            pocket = row["pocket"]
            if set(pocket) != {"protein_pdb", "pocket_pdb"}:
                raise ValueError(f"Reference line {line_number} requires protein_pdb and pocket_pdb")
            raw = {"pocket": {}}
            for key in ("protein_pdb", "pocket_pdb"):
                if not isinstance(pocket[key], str) or not pocket[key].strip():
                    raise ValueError(f"Reference line {line_number} has an invalid {key}")
                source = (path.parent / pocket[key]).resolve()
                if not source.is_file():
                    raise FileNotFoundError(f"Reference line {line_number}: {key} does not exist")
                raw["pocket"][key] = str(source)
            if "pocket_embedding" not in row:
                raise ValueError(f"Reference line {line_number} requires pocket_embedding")
            vector = validate_embedding(row["pocket_embedding"])
            if float(np.linalg.norm(vector.astype(np.float64))) == 0:
                raise ValueError(f"Reference line {line_number} has a zero vector")
            records.append((line_number, raw, vector.copy()))
    if len(records) < 3:
        raise ValueError("At least three distinct matched pocket references are required")
    return records


def _publish_record(output, record):
    """Publish complete JSON without replacing an existing destination."""
    payload = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pocket-compatibility-", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def verify_pocket_encoder(checkpoint, encoder_config, references, output, atol=1e-6, rtol=1e-5):
    """Verify an encoder against paired local PDBs and training vectors.

    Each reference JSONL row contains ``pocket`` with ``protein_pdb`` and
    ``pocket_pdb`` paths plus a ``pocket_embedding`` array copied from the
    corresponding training record. Paths are relative to the JSONL file.
    Reference vectors are compared as float32, without further normalization.
    The returned record describes this comparison; reference selection and
    correspondence must be established by the dataset maintainer.
    """
    atol, rtol = _tolerance(atol, "atol", 1e-4), _tolerance(rtol, "rtol", 1e-3)
    checkpoint, references, output = Path(checkpoint).resolve(), Path(references).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Compatibility record already exists; choose a new output path")
    if not isinstance(encoder_config, dict) or not encoder_config:
        raise ValueError("A configured DeeplyTough encoder is required")
    checkpoint_digest, reference_digest = file_sha256(checkpoint), file_sha256(references)
    samples = _references(references)
    model, tokenizer, state = load_checkpoint(checkpoint)
    if "pocket" not in state.get("trained_conditions", ()):
        raise ValueError("Checkpoint was not trained with pocket conditioning")
    checkpoint_signature = state.get("pocket_encoder_signature")
    if not isinstance(checkpoint_signature, str) or not checkpoint_signature.strip():
        raise ValueError("Checkpoint has no pocket encoder signature")
    if model.config.pocket_dim != 128:
        raise ValueError("This comparison requires a checkpoint with 128-dimensional pockets")
    del model, tokenizer, state
    resolver = PocketResolver(copy.deepcopy(encoder_config))
    encoder_signature = resolver.signature(refresh=True)
    if not isinstance(encoder_signature, str) or not re.fullmatch(r"deeplytough:v2:[0-9a-f]{64}", encoder_signature):
        raise ValueError("Encoder must provide a concrete DeeplyTough profile signature")
    weights_digest = resolver.profile().get("weights_sha256")
    if not isinstance(weights_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", weights_digest):
        raise ValueError("Encoder profile has no valid weights checksum")

    fingerprints, seen = [], set()
    for line_number, raw, reference in samples:
        fingerprint = resolver.input_fingerprint(raw)
        if fingerprint.get("source") != "raw-pdb" or fingerprint.get("encoder") != encoder_signature:
            raise ValueError("Reference fingerprint does not identify the configured raw-PDB encoder")
        identity = content_hash(fingerprint)
        if identity in seen:
            raise ValueError(f"Duplicate pocket reference at line {line_number}")
        seen.add(identity)
        fingerprints.append(copy.deepcopy(fingerprint))

    max_abs_error, max_norm_error, minimum_cosine = 0.0, 0.0, 1.0
    for (line_number, raw, reference), expected_fingerprint in zip(samples, fingerprints):
        if resolver.input_fingerprint(raw) != expected_fingerprint:
            raise RuntimeError("Pocket inputs changed during verification")
        encoded = validate_embedding(resolver.resolve(raw)).astype(np.float64)
        target = reference.astype(np.float64)
        encoded_norm, target_norm = float(np.linalg.norm(encoded)), float(np.linalg.norm(target))
        if encoded_norm == 0:
            raise ValueError(f"Encoder returned a zero vector for reference line {line_number}")
        error = float(np.max(np.abs(encoded - target)))
        cosine = float(np.clip(np.dot(encoded, target) / (encoded_norm * target_norm), -1.0, 1.0))
        if not np.allclose(encoded, target, atol=atol, rtol=rtol):
            raise ValueError(f"Pocket vector mismatch at reference line {line_number}: max_abs_error={error:.8g}, cosine={cosine:.8g}")
        if resolver.input_fingerprint(raw) != expected_fingerprint:
            raise RuntimeError("Pocket inputs changed during verification")
        max_abs_error = max(max_abs_error, error)
        max_norm_error = max(max_norm_error, abs(encoded_norm - target_norm))
        minimum_cosine = min(minimum_cosine, cosine)

    if resolver.signature(refresh=True) != encoder_signature or resolver.profile().get("weights_sha256") != weights_digest:
        raise RuntimeError("Encoder weights, sources, or runtime changed during verification")
    if any(resolver.input_fingerprint(raw) != expected for (_, raw, _), expected in zip(samples, fingerprints)):
        raise RuntimeError("Pocket inputs changed during verification")
    if file_sha256(checkpoint) != checkpoint_digest or file_sha256(references) != reference_digest:
        raise RuntimeError("Checkpoint or reference records changed during verification")
    record = {
        "format_version": 1,
        "kind": "pocketgpt-encoder-compatibility",
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_pocket_signature": checkpoint_signature,
        "encoder_signature": encoder_signature,
        "encoder_weights_sha256": weights_digest,
        "validation": {
            "method": "paired-pocket-vectors-v1", "sample_count": len(samples),
            "atol": atol, "rtol": rtol, "allclose": True,
            "max_abs_error": max_abs_error, "max_norm_error": max_norm_error,
            "min_cosine_similarity": minimum_cosine,
        },
    }
    _publish_record(output, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--references", required=True, help="Local JSONL of matched PDB paths and training pocket vectors")
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    config = json.loads(Path(args.encoder_config).read_text(encoding="utf-8"))
    record = verify_pocket_encoder(args.checkpoint, config, args.references, args.output, args.atol, args.rtol)
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
