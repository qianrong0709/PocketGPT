import math
import os
import random
import shutil
import tempfile
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch

from .config import (CONDITIONS, ModelConfig, parse_conditions,
                     DEFAULT_FRAGMENT_REPRESENTATION, validate_fragment_representation)
from .model import PocketGPT
from .tokenizer import SmilesTokenizer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_checkpoint(path, payload):
    """Publish only a flushed, readable archive, retaining the previous file.

    Use Python-owned binary streams rather than the path-based C++ file stream.
    Failed temporary files are retained for diagnosis and never used implicitly.
    This function must not draw from the training random-number generators.
    """
    path = Path(path)
    for attempt in range(2):
        temporary, phase = None, "create"
        try:
            descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
            temporary = Path(name)
            phase = "serialize/flush/fsync"
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            phase = "verify"
            with zipfile.ZipFile(temporary) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise OSError("Checkpoint CRC failure: " + bad_member)
            verified = torch.load(temporary, map_location="cpu", weights_only=True)
            if not isinstance(verified, dict) or verified.get("global_step") != payload.get("global_step"):
                raise ValueError("Checkpoint reload/step mismatch")
            del verified
            phase = "retain_previous"
            if path.exists():
                descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".previous.", dir=path.parent)
                previous_temporary = Path(name)
                with os.fdopen(descriptor, "wb") as handle, path.open("rb") as source:
                    shutil.copyfileobj(source, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(previous_temporary, path.with_name(path.name + ".previous"))
            phase = "publish"
            os.replace(temporary, path)
            phase = "directory_fsync"
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
            message = (f"Checkpoint save failed: phase={phase}, attempt={attempt + 1}/2, "
                       f"target={path}, temporary={temporary}, "
                       f"error={type(error).__name__}, errno={getattr(error, 'errno', None)}: {error}")
            if attempt == 1:
                raise RuntimeError(message) from error
            warnings.warn(message + "; retrying once; existing checkpoints retained", RuntimeWarning)
            time.sleep(0.25)


def checkpoint_fragment_representation(state):
    """Old checkpoints remain canonical; raw semantics must be explicit."""
    recipe_mode = state.get("recipe", {}).get("fragment_representation", DEFAULT_FRAGMENT_REPRESENTATION)
    mode = validate_fragment_representation(state.get("fragment_representation", recipe_mode))
    if mode != validate_fragment_representation(recipe_mode):
        raise ValueError("Checkpoint fragment representation conflicts with its recipe")
    contract = state.get("training_contract", {})
    if "fragment_representation" in contract and contract["fragment_representation"] != mode:
        raise ValueError("Checkpoint fragment representation conflicts with its training contract")
    return mode


def validate_fragment_initialization(state, representation):
    representation = validate_fragment_representation(representation)
    previous = checkpoint_fragment_representation(state)
    if "fragment" in state.get("trained_conditions", ()) and previous != representation:
        raise ValueError("Cannot silently change fragment representation while inheriting a trained fragment branch")
    return previous


def load_checkpoint(path, device="cpu", *, allow_legacy_architecture=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Not a PocketGPT-base checkpoint; use explicit legacy warm start")
    checkpoint_fragment_representation(state)
    tokenizer = SmilesTokenizer(state["vocabulary"])
    model = PocketGPT(ModelConfig.from_checkpoint_dict(
        state["model_config"], allow_legacy=allow_legacy_architecture))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device), tokenizer, state


def warm_start_base(model, path, tokenizer):
    """Explicit architecture migration; never restores optimizer or RNG state."""
    initial, old_tokenizer, state = load_checkpoint(path, allow_legacy_architecture=True)
    if state.get("trained_conditions"):
        raise ValueError("Backbone-only warm start would discard trained conditioning branches; use condition_init_checkpoint with the same architecture")
    old, current = initial.state_dict(), model.state_dict()
    loaded, skipped = [], []
    for key, value in old.items():
        if key in ("tok_emb.weight", "head.weight", "pos_emb"):
            continue
        if key.startswith(("blocks.", "ln_f.", "type_emb.")) and key in current and current[key].shape == value.shape:
            current[key].copy_(value)
            loaded.append(key)
        else:
            skipped.append(key)
    if initial.config.n_embd != model.config.n_embd or not loaded:
        raise ValueError("Warm start requires a compatible molecular decoder backbone")
    positions = min(initial.config.block_size, model.config.block_size)
    current["pos_emb"][:, :positions].copy_(old["pos_emb"][:, :positions])
    copied = []
    for token, old_id in old_tokenizer.stoi.items():
        if token in tokenizer.stoi:
            new_id = tokenizer.stoi[token]
            current["tok_emb.weight"][new_id].copy_(old["tok_emb.weight"][old_id])
            current["head.weight"][new_id].copy_(old["head.weight"][old_id])
            copied.append(token)
    model.load_state_dict(current, strict=True)
    return {"source": str(Path(path).resolve()), "source_step": state.get("global_step"),
            "backbone_tensors": loaded, "copied_token_count": len(copied),
            "copied_positions": positions, "skipped": skipped,
            "new_conditioners_require_training": True, "exact_resume": False}


def warm_start_conditioned(model, path, tokenizer, conditions,
                           fragment_representation=DEFAULT_FRAGMENT_REPRESENTATION):
    """Transfer a conditioned model to a new run without losing its semantics.

    This deliberately does not perform architecture migration or exact resume.
    Only the vocabulary may differ. All decoder/fusion parameters and every
    previously trained conditioning branch are retained; untrained branches
    keep the target model's fresh initialization instead of inheriting arbitrary
    source-checkpoint values. Token rows are matched by their literal token.
    """
    initial, old_tokenizer, state = load_checkpoint(path)
    previous = parse_conditions(state.get("trained_conditions", ()))
    requested = parse_conditions(conditions)
    previous_representation = validate_fragment_initialization(state, fragment_representation)
    if not previous:
        raise ValueError("condition_init_checkpoint requires a checkpoint with explicitly trained conditions")
    old_config, new_config = initial.config.to_dict(), model.config.to_dict()
    old_config.pop("vocab_size")
    new_config.pop("vocab_size")
    if old_config != new_config:
        raise ValueError("Condition-aware initialization requires the same architecture/configuration except vocabulary; no implicit architecture upgrade")

    old, current = initial.state_dict(), model.state_dict()
    if old.keys() != current.keys():
        raise ValueError("Condition-aware initialization requires identical parameter schemas")
    branches = {
        "pocket": ("pocket_projector.",),
        "properties": ("property_projector.",),
        "fragment": ("fragment_projector.", "fragment_pos_emb.", "fragment_sequence_encoder."),
        "ifp": ("ifp_projector.",),
    }
    copied, reinitialized = [], []
    for key, value in old.items():
        if key in ("tok_emb.weight", "head.weight", "modality_emb"):
            continue
        modality = next((name for name, prefixes in branches.items() if key.startswith(prefixes)), None)
        if modality is not None and modality not in previous:
            reinitialized.append(key)
            continue
        if current[key].shape != value.shape:
            raise ValueError("Condition-aware initialization cannot preserve parameter: " + key)
        current[key].copy_(value)
        copied.append(key)

    copied_tokens = []
    for token, old_id in old_tokenizer.stoi.items():
        if token in tokenizer.stoi:
            new_id = tokenizer.stoi[token]
            current["tok_emb.weight"][new_id].copy_(old["tok_emb.weight"][old_id])
            current["head.weight"][new_id].copy_(old["head.weight"][old_id])
            copied_tokens.append(token)
    for index, name in enumerate(CONDITIONS):
        if name in previous:
            current["modality_emb"][index].copy_(old["modality_emb"][index])
        else:
            reinitialized.append("modality_emb[" + name + "]")
    model.load_state_dict(current, strict=True)
    return {
        "method": "condition-aware-token-mapping-v1", "source": str(Path(path).resolve()),
        "source_step": state.get("global_step"), "source_stage": state.get("recipe", {}).get("stage"),
        "source_trained_conditions": list(previous),
        "source_trained_properties": list(state.get("trained_properties", ())),
        "source_pocket_encoder_signature": state.get("pocket_encoder_signature"),
        "source_fragment_representation": previous_representation,
        "fragment_representation": fragment_representation,
        "copied_tensors": copied, "copied_modality_rows": list(previous),
        "copied_token_count": len(copied_tokens), "copied_tokens": copied_tokens,
        "new_tokens": [token for token in tokenizer.tokens if token not in old_tokenizer.stoi],
        "source_only_tokens": [token for token in old_tokenizer.tokens if token not in tokenizer.stoi],
        "new_conditions": [name for name in requested if name not in previous],
        "reinitialized_tensors": reinitialized,
        "new_conditioners_require_training": bool(set(requested) - set(previous)),
        "optimizer_restored": False, "rng_restored": False, "exact_resume": False,
    }


def warm_start_legacy(model, old_path, old_vocabulary, new_tokenizer):
    """Transfers compatible backbone tensors, never silently invents vocab order."""
    old = torch.load(old_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(old, dict) or "tok_emb.weight" not in old:
        raise ValueError("Legacy import expects a tensor state_dict")
    vocab = old_vocabulary
    size = old["tok_emb.weight"].shape[0]
    if set(vocab.values()) != set(range(size)):
        raise ValueError("Legacy vocabulary does not match the checkpoint embedding rows")
    current = model.state_dict()
    loaded, skipped = [], []
    for key, value in old.items():
        if key in ("tok_emb.weight", "head.weight", "pos_emb"):
            continue
        if key.startswith(("blocks.", "ln_f.", "type_emb.")) and key in current and current[key].shape == value.shape:
            current[key].copy_(value)
            loaded.append(key)
        else:
            skipped.append(key)
    positions = 0
    if old["pos_emb"].shape[-1] == current["pos_emb"].shape[-1]:
        positions = min(old["pos_emb"].shape[1], current["pos_emb"].shape[1])
        current["pos_emb"][:, :positions].copy_(old["pos_emb"][:, :positions])
    copied_tokens = []
    new_vocab = new_tokenizer.stoi
    if old["tok_emb.weight"].shape[1] != model.config.n_embd:
        raise ValueError("Legacy embedding dimension differs from the new configuration")
    for token, old_id in vocab.items():
        if token in new_vocab and token != "<":
            new_id = new_vocab[token]
            current["tok_emb.weight"][new_id].copy_(old["tok_emb.weight"][old_id])
            current["head.weight"][new_id].copy_(old["head.weight"][old_id])
            copied_tokens.append(token)
    if not loaded or not copied_tokens:
        raise ValueError("No usable backbone weights or tokens were transferred")
    model.load_state_dict(current, strict=True)
    return {"source": str(Path(old_path).resolve()), "backbone_tensors": loaded,
            "copied_token_count": len(copied_tokens), "copied_positions": positions,
            "skipped": skipped, "new_conditioners_require_training": True,
            "note": "Warm start only; old checkpoints have no optimizer/RNG state and cannot be exactly resumed."}
