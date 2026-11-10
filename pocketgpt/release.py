"""Create a portable inference artifact without private training metadata."""
import re
from pathlib import Path

import torch

from .checkpoint import checkpoint_fragment_representation, load_checkpoint, save_checkpoint
from .pocket import LEGACY_SIGNATURE, file_sha256


def export_inference(checkpoint, output, model_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", model_id):
        raise ValueError("model_id must be a short public identifier, not a path")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Inference artifact exists; choose a new output filename")
    model, tokenizer, source = load_checkpoint(checkpoint)
    representation = checkpoint_fragment_representation(source)
    conditions = list(source["trained_conditions"])
    legacy_allowed = bool(source.get("recipe", {}).get("allow_unverified_pocket_embeddings", False))
    if ("pocket" in conditions and source.get("pocket_encoder_signature") == LEGACY_SIGNATURE
            and not legacy_allowed):
        raise ValueError("Source checkpoint does not explicitly allow its legacy pocket descriptors")
    weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if not all(torch.isfinite(value).all().item() for value in weights.values()):
        raise ValueError("Model contains non-finite weights")
    # An allowlist, not a blacklist: no dataset rows, paths, encoder configuration,
    # optimizer, RNG, initialization reports or full training contracts are copied.
    payload = {
        "format_version": 1,
        "artifact_type": "inference",
        "model_id": model_id,
        "model_config": model.config.to_dict(),
        "model": weights,
        "vocabulary": list(tokenizer.tokens),
        "trained_conditions": conditions,
        "trained_properties": list(source["trained_properties"]),
        "pocket_encoder_signature": source.get("pocket_encoder_signature"),
        "fragment_representation": representation,
        "global_step": int(source["global_step"]),
        "recipe": {"fragment_representation": representation,
                   "allow_unverified_pocket_embeddings": legacy_allowed},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, payload)
    restored, restored_tokenizer, _ = load_checkpoint(output)
    if restored_tokenizer.tokens != tokenizer.tokens or any(
            not torch.equal(value, restored.state_dict()[name]) for name, value in weights.items()):
        raise RuntimeError("Inference export round-trip changed model parameters or vocabulary")
    return {"model_id": model_id, "bytes": output.stat().st_size, "sha256": file_sha256(output),
            "global_step": payload["global_step"], "exact_training_resume": False}
