"""Inference-only pocket compatibility; training and cache identities stay exact."""
import hashlib
import json
import math
from pathlib import Path
import re
import warnings

import numpy as np

from .pocket import LEGACY_SIGNATURE, file_sha256, validate_embedding, validate_pocket_sources


def load_pocket_vector(path):
    """Read a descriptor without inventing an encoder identity or normalizing it."""
    path = Path(path)
    signature = None
    if path.suffix.lower() == ".npy":
        vector = validate_embedding(np.load(path, allow_pickle=False))
        sidecar = path.with_suffix(".provenance.json")
        if not sidecar.exists():
            sidecar = path.with_name("provenance.json")
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text())
            if metadata.get("output_sha256") != file_sha256(path):
                raise ValueError("Pocket vector checksum does not match its provenance sidecar")
            signature = metadata.get("input", {}).get("encoder")
            if not isinstance(signature, str) or not signature:
                raise ValueError("Pocket vector sidecar has no encoder signature")
    else:
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            if set(value) - {"pocket_embedding", "pocket_encoder_signature"}:
                raise ValueError("Pocket vector JSON accepts only pocket_embedding and pocket_encoder_signature; use --request for other conditions")
            if "pocket_encoder_signature" in value:
                signature = value["pocket_encoder_signature"]
                if not isinstance(signature, str) or not signature:
                    raise ValueError("Pocket encoder signature must be a nonempty string")
            value = value.get("pocket_embedding")
        vector = validate_embedding(value)
    result = {"pocket_embedding": vector.tolist()}
    if signature is not None:
        result["pocket_encoder_signature"] = signature
    return result


def _records(checkpoint, compatibility):
    if compatibility is not None:
        record = json.loads(Path(compatibility).read_text())
        return [record]
    manifest = Path(checkpoint).with_suffix(".json")
    if not manifest.exists():
        return []
    records = json.loads(manifest.read_text()).get("pocket_compatibility", [])
    if not isinstance(records, list):
        raise ValueError("Model manifest pocket_compatibility must be a list")
    return records


def _validate_record(record):
    if not isinstance(record, dict) or record.get("format_version") != 1 or record.get("kind") != "pocketgpt-encoder-compatibility":
        raise ValueError("Invalid pocket compatibility record")
    for key in ("checkpoint_sha256", "encoder_weights_sha256"):
        if not isinstance(record.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", record[key]):
            raise ValueError("Invalid compatibility checksum: " + key)
    if not isinstance(record.get("checkpoint_pocket_signature"), str) or not record["checkpoint_pocket_signature"]:
        raise ValueError("Compatibility record has no checkpoint pocket signature")
    if not isinstance(record.get("encoder_signature"), str) or not re.fullmatch(r"deeplytough:v2:[0-9a-f]{64}", record["encoder_signature"]):
        raise ValueError("Compatibility record has no concrete encoder profile")
    validation = record.get("validation", {})
    if not isinstance(validation, dict) or validation.get("method") != "paired-pocket-vectors-v1" or validation.get("allclose") is not True:
        raise ValueError("Compatibility record requires a passed paired-vector comparison")
    if type(validation.get("sample_count")) is not int or validation["sample_count"] < 3:
        raise ValueError("Compatibility record requires at least three matched pockets")
    for key, low, high in (("atol", 0, 1e-4), ("rtol", 0, 1e-3),
                           ("max_abs_error", 0, float("inf")), ("min_cosine_similarity", -1, 1)):
        value = validation.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Invalid compatibility comparison metric: " + key)


def check_pocket_compatibility(checkpoint, state, request, resolver, *,
                               compatibility=None, allow_unverified=False):
    """Return honest inference metadata, without relabelling the input vector.

    A paired-reference record is a maintainer's comparison report, not a
    cryptographic certificate. Use records from the model/encoder maintainer.
    """
    validate_pocket_sources(request)
    if type(allow_unverified) is not bool:
        raise TypeError("allow_unverified_pocket must be a boolean")
    expected = state.get("pocket_encoder_signature")
    if not isinstance(expected, str) or not expected:
        raise ValueError("Checkpoint has no pocket encoder identity")
    if "pocket_embedding" in request:
        validate_embedding(request["pocket_embedding"])
        actual = request.get("pocket_encoder_signature")
        if "pocket_encoder_signature" in request and (not isinstance(actual, str) or not actual):
            raise ValueError("Pocket encoder signature must be a nonempty string")
        if resolver.config and actual != resolver.signature():
            raise ValueError("Pocket vector metadata does not match --encoder-config; keep its original metadata and omit the encoder config for cached vectors")
    else:
        actual = resolver.row_signature(request)
    if (state.get("training_contract") and expected == LEGACY_SIGNATURE
            and not state.get("recipe", {}).get("allow_unverified_pocket_embeddings", False)):
        raise ValueError("Checkpoint does not explicitly allow unverified legacy pocket embeddings")
    status = None
    if actual == expected:
        status = "legacy-declared" if expected == LEGACY_SIGNATURE else "profile-matched"
    records = _records(checkpoint, compatibility)
    for record in records:
        _validate_record(record)
    if records and (status is None or compatibility is not None):
        digest = file_sha256(checkpoint)
        matched = False
        for record in records:
            if (record["checkpoint_sha256"] == digest and
                    record["checkpoint_pocket_signature"] == expected and record["encoder_signature"] == actual):
                if resolver.config and record["encoder_weights_sha256"] != resolver.profile()["weights_sha256"]:
                    raise ValueError("Compatibility record does not match encoder weights")
                status = "reference-verified"
                matched = True
                break
        if compatibility is not None and not matched:
            raise ValueError("Compatibility record does not match this checkpoint and input encoder; do not relabel the vector")
    if status is None and expected == LEGACY_SIGNATURE and allow_unverified:
        status = "legacy-user-accepted"
    if status is None:
        hint = (" This checkpoint used historical cached vectors. To run before a paired-vector comparison, "
                "explicitly pass --allow-unverified-pocket (Python: allow_unverified_pocket=True). "
                "This does not verify feature-space compatibility." if expected == LEGACY_SIGNATURE else "")
        raise ValueError(f"Pocket encoder compatibility is not established: checkpoint={expected!r}, input={actual!r}. "
                         "Use --pocket-compatibility with a matched comparison record." + hint)
    if status.startswith("legacy-"):
        warnings.warn("Using a historical-vector checkpoint without a verified encoder comparison. "
                      "The supplied 128 values are used unchanged; encoder compatibility is unverified.",
                      UserWarning, stacklevel=2)
    return {"status": status, "checkpoint_signature": expected, "input_signature": actual,
            "feature_space_verified": status in ("profile-matched", "reference-verified")}


def inference_fingerprint(request, resolver, metadata):
    """Unlike the training CSV reader, never label an unknown vector as legacy."""
    if "pocket_embedding" in request:
        vector = validate_embedding(request["pocket_embedding"])
        return {"source": "precomputed", "encoder": metadata["input_signature"],
                "embedding_sha256": hashlib.sha256(vector.tobytes()).hexdigest()}
    return resolver.input_fingerprint(request)
