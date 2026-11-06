import json
import hashlib
import math
import platform
from pathlib import Path

import torch

from .checkpoint import (load_checkpoint, restore_rng, rng_state, save_checkpoint,
                         seed_all, warm_start_legacy, warm_start_base, warm_start_conditioned,
                         checkpoint_fragment_representation, validate_fragment_initialization)
from .config import (CONDITIONS, PROPERTIES, ModelConfig, parse_conditions,
                     DEFAULT_FRAGMENT_REPRESENTATION, validate_fragment_representation)
from .data import BatchBuilder, JsonlDataset, inspect_manifest, row_fragment_text
from .model import PocketGPT
from .pocket import PocketResolver, file_sha256
from .tokenizer import SmilesTokenizer


def recipe_path(recipe, value):
    path = Path(value)
    return str((Path(recipe).resolve().parent / path).resolve()) if not path.is_absolute() else str(path)


def training_stage_contract(stage, conditions, split_group="protein_id"):
    """Distinguish ligand conditional pretraining from target-based finetuning."""
    conditions = parse_conditions(conditions)
    if stage not in ("pretrain", "condition_pretrain", "finetune"):
        raise ValueError("stage must be pretrain, condition_pretrain or finetune")
    if stage == "pretrain" and conditions:
        raise ValueError("Pretrain is molecule-only; use condition_pretrain for properties/fragment labels")
    if stage == "condition_pretrain" and (not conditions or set(conditions) - {"properties", "fragment"}):
        raise ValueError("condition_pretrain needs a nonempty subset of properties,fragment; pocket/ifp require finetune with an explicit grouped split")
    if stage == "finetune" and not conditions:
        raise ValueError("Finetune needs explicit conditions")
    if split_group not in ("protein_id", "complex_id"):
        raise ValueError("split_group must be protein_id or complex_id")
    if stage != "finetune" and split_group != "protein_id":
        raise ValueError("Explicit complex_id grouping is only available for finetune")
    return {"stage": stage, "conditions": list(conditions),
            "split_identity": split_group if stage == "finetune" else "canonical_smiles",
            "validation_scope": ("complex_id_grouped_NOT_protein_heldout"
                                 if stage == "finetune" and split_group == "complex_id"
                                 else "protein_id_grouped_requires_verified_identity"
                                 if stage == "finetune" else "canonical_parent_molecule_grouped")}


def input_contract(train_data, valid_data, conditions, encoder_config, allow_unverified=False):
    """Hash actual inputs, not just paths; a cache hit is not a resume contract."""
    if "pocket" not in conditions:
        return {"version": 2, "pocket": None}
    resolver = PocketResolver(encoder_config)
    signatures, sources, digests = set(), set(), {}
    for name, dataset in (("train", train_data), ("valid", valid_data)):
        digest = hashlib.sha256()
        for row in dataset:
            if "pocket_embedding" not in row and not row.get("pocket"):
                continue
            signature = resolver.row_signature(row)
            signatures.add(signature)
            sources.add("embedding" if "pocket_embedding" in row else "raw")
            fingerprint = resolver.input_fingerprint(row)
            digest.update(json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            digest.update(b"\n")
        digests[name] = digest.hexdigest()
    if len(signatures) != 1:
        raise ValueError("Training/validation pocket encoder provenance differs; use one verified encoder profile")
    signature = next(iter(signatures))
    if signature == "legacy-precomputed-unverified":
        if sources != {"embedding"} or not allow_unverified or encoder_config:
            raise ValueError("Unverified legacy embeddings require explicit allow_unverified_pocket_embeddings and embedding-only mode")
    elif encoder_config and signature != resolver.signature():
        raise ValueError("Raw encoder configuration does not match training/validation embedding provenance")
    return {"version": 2, "pocket": {"signature": signature, "sources": sorted(sources),
                                    "input_fingerprints": digests,
                                    "encoder_profile": resolver.profile() if encoder_config and signature != "legacy-precomputed-unverified" else None}}


def implementation_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def runtime_contract(device):
    import numpy as np
    from rdkit import rdBase
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "rdkit": rdBase.rdkitVersion,
            "device": device, "cpu_threads": torch.get_num_threads(),
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def train(recipe_file, resume=None, max_steps=None, device_override=None):
    recipe = json.loads(Path(recipe_file).read_text())
    stage = recipe["stage"]
    conditions = parse_conditions(recipe.get("conditions", "none"))
    fragment_representation = validate_fragment_representation(
        recipe.get("fragment_representation", DEFAULT_FRAGMENT_REPRESENTATION))
    split_group = recipe.get("split_group", "protein_id")
    stage_contract = training_stage_contract(stage, conditions, split_group=split_group)
    settings = recipe["training"]
    for name, default in (("epochs", 10), ("batch_size", 32), ("save_every_steps", 1000), ("log_every_steps", 100)):
        if int(settings.get(name, default)) < 1:
            raise ValueError(name + " must be positive")
    for name, default in (("learning_rate", 0.0003), ("grad_clip", 1.0)):
        value = float(settings.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    device = device_override or settings.get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Submit a GPU job or use --device cpu for smoke tests")
    seed = settings.get("seed", 42)
    seed_all(seed)
    torch.set_num_threads(settings.get("cpu_threads", 2))
    train_path = recipe_path(recipe_file, recipe["train_data"])
    valid_path = recipe_path(recipe_file, recipe["valid_data"])
    train_data, valid_data = JsonlDataset(train_path), JsonlDataset(valid_path)
    train_info = inspect_manifest(train_data, conditional=stage == "finetune", conditions=conditions,
                                  fragment_representation=fragment_representation, split_group=split_group)
    valid_info = inspect_manifest(valid_data, conditional=stage == "finetune", conditions=conditions,
                                  fragment_representation=fragment_representation, split_group=split_group)
    overlap = (train_info["groups"] & valid_info["groups"]) if stage == "finetune" else (train_info["molecules"] & valid_info["molecules"])
    if overlap:
        raise ValueError("Training/validation leakage: " + (split_group if stage == "finetune" else "SMILES") + " overlap")
    missing = set(conditions) - train_info["modalities"]
    if missing:
        raise ValueError("Training labels absent for requested conditions: " + repr(sorted(missing)))
    missing_valid = set(conditions) - valid_info["modalities"]
    if missing_valid:
        raise ValueError("Validation labels absent for requested conditions: " + repr(sorted(missing_valid)))
    contract = input_contract(train_data, valid_data, conditions, recipe.get("pocket_encoder"),
                              recipe.get("allow_unverified_pocket_embeddings", False))
    contract["training_stage"] = stage_contract
    contract["fragment_representation"] = fragment_representation
    contract["implementation_sha256"] = implementation_fingerprint()
    contract["runtime"] = runtime_contract(device)
    data_fingerprints = {"train": file_sha256(train_path), "valid": file_sha256(valid_path)}
    run = Path(recipe_path(recipe_file, recipe["output_dir"]))
    run.mkdir(parents=True, exist_ok=True)
    if not resume and (run / "last.pt").exists():
        raise FileExistsError("Run already contains last.pt; explicitly resume or choose another output_dir")
    start_epoch, next_batch, global_step, best_loss = 0, 0, 0, math.inf
    warm_start = None
    state = None
    inherited_conditions, inherited_properties, inherited_encoder_signature = (), (), None
    if resume:
        model, tokenizer, state = load_checkpoint(resume, device)
        if checkpoint_fragment_representation(state) != fragment_representation:
            raise ValueError("Exact resume cannot change fragment representation")
        if state["recipe"] != recipe or state["data_fingerprints"] != data_fingerprints:
            raise ValueError("Resume requires the original recipe and unchanged datasets")
        if state.get("training_contract") != contract:
            raise ValueError("Exact resume requires unchanged encoder, referenced structures, implementation and runtime; create a new warm-start run instead")
        start_epoch, next_batch, global_step = state["epoch"], state["next_batch"], state["global_step"]
        best_loss = state["best_loss"]
        warm_start = state.get("warm_start")
        inherited_conditions = tuple(state.get("inherited_trained_conditions", ()))
        inherited_properties = tuple(state.get("inherited_trained_properties", ()))
        inherited_encoder_signature = state.get("inherited_pocket_encoder_signature")
    else:
        if recipe.get("vocabulary"):
            tokenizer = SmilesTokenizer.load(recipe_path(recipe_file, recipe["vocabulary"]))
        else:
            def texts():
                for row in train_data:
                    yield row["smiles"]
                    if "fragment" in conditions and row.get("fragment"):
                        yield row_fragment_text(row, fragment_representation)
            tokenizer = SmilesTokenizer.build(texts())
        model = PocketGPT(ModelConfig(vocab_size=len(tokenizer.tokens), **recipe.get("model", {})))
        initializers = [name for name in ("init_checkpoint", "condition_init_checkpoint", "warm_start_checkpoint", "legacy_checkpoint") if recipe.get(name)]
        if len(initializers) > 1:
            raise ValueError("Choose exactly one checkpoint initialization method")
        if recipe.get("init_checkpoint"):
            initial_path = recipe_path(recipe_file, recipe["init_checkpoint"])
            initial, initial_tokenizer, initial_state = load_checkpoint(initial_path)
            validate_fragment_initialization(initial_state, fragment_representation)
            if initial.config.to_dict() != model.config.to_dict() or initial_tokenizer.tokens != tokenizer.tokens:
                raise ValueError("init_checkpoint requires matching model configuration and vocabulary; use condition_init_checkpoint for conditioned token mapping")
            model.load_state_dict(initial.state_dict(), strict=True)
            inherited_conditions = parse_conditions(initial_state.get("trained_conditions", ()))
            inherited_properties = tuple(initial_state.get("trained_properties", ())) if "properties" in inherited_conditions else ()
            inherited_encoder_signature = initial_state.get("pocket_encoder_signature") if "pocket" in inherited_conditions else None
            warm_start = {"method": "strict-model-initialization-v1", "source": initial_path,
                          "source_step": initial_state.get("global_step"),
                          "source_stage": initial_state.get("recipe", {}).get("stage"),
                          "source_trained_conditions": list(inherited_conditions),
                          "source_trained_properties": list(inherited_properties),
                          "copied_tensors": list(model.state_dict()),
                          "optimizer_restored": False, "rng_restored": False, "exact_resume": False}
        if recipe.get("condition_init_checkpoint"):
            initial_path = recipe_path(recipe_file, recipe["condition_init_checkpoint"])
            warm_start = warm_start_conditioned(model, initial_path, tokenizer, conditions,
                                                fragment_representation=fragment_representation)
            inherited_conditions = tuple(warm_start["source_trained_conditions"])
            inherited_properties = tuple(warm_start["source_trained_properties"]) if "properties" in inherited_conditions else ()
            inherited_encoder_signature = warm_start.get("source_pocket_encoder_signature") if "pocket" in inherited_conditions else None
        if recipe.get("warm_start_checkpoint"):
            warm_start = warm_start_base(model, recipe_path(recipe_file, recipe["warm_start_checkpoint"]), tokenizer)
        if recipe.get("legacy_checkpoint"):
            old_vocab = json.loads(Path(recipe_path(recipe_file, recipe["legacy_vocabulary"])).read_text())
            warm_start = warm_start_legacy(model, recipe_path(recipe_file, recipe["legacy_checkpoint"]), old_vocab, tokenizer)
            (run / "legacy_import_report.json").write_text(json.dumps(warm_start, indent=2) + "\n")
        model.to(device)
        if warm_start:
            (run / "initialization_report.json").write_text(json.dumps(warm_start, indent=2) + "\n")
    tokenizer.save(run / "vocabulary.json")
    (run / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    # A ligand-only stage never resolves an old encoder profile or referenced PDB.
    builder = BatchBuilder(tokenizer, model.config.block_size, conditions,
                           recipe.get("pocket_encoder") if "pocket" in conditions else None,
                           fragment_representation=fragment_representation)
    encoder_signature = contract["pocket"]["signature"] if contract["pocket"] else inherited_encoder_signature
    if state and state.get("pocket_encoder_signature") != encoder_signature:
        raise ValueError("Checkpoint pocket encoder differs from current training inputs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.get("learning_rate", 0.0003)),
                                  weight_decay=float(settings.get("weight_decay", 0.01)))
    if state:
        optimizer.load_state_dict(state["optimizer"])
        # Optimizer tensors follow parameter devices after CPU checkpoint loading.
        for optim_state in optimizer.state.values():
            for key, value in optim_state.items():
                if torch.is_tensor(value) and key != "step":
                    optim_state[key] = value.to(device)
        restore_rng(state["rng"])
    epochs = int(settings.get("epochs", 10))
    batch_size = int(settings.get("batch_size", 32))
    if min(epochs, batch_size) < 1:
        raise ValueError("epochs and batch_size must be positive")
    batches_per_epoch = math.ceil(len(train_data) / batch_size)
    total_steps = epochs * batches_per_epoch
    warmup = int(settings.get("warmup_steps", 0))
    condition_dropout = float(settings.get("condition_dropout", 0.35 if conditions else 0.0))
    if not 0 <= condition_dropout < 1:
        raise ValueError("condition_dropout must be in [0, 1)")
    save_every = int(settings.get("save_every_steps", 1000))
    if save_every < 1:
        raise ValueError("save_every_steps must be positive")
    base_lr = float(settings.get("learning_rate", 0.0003))
    history = run / "metrics.jsonl"
    stage_properties = sorted(train_info["properties"]) if "properties" in conditions else []
    trained_conditions = [name for name in CONDITIONS if name in set(conditions) | set(inherited_conditions)]
    trained_properties = [name for name in PROPERTIES if name in set(stage_properties) | set(inherited_properties)]

    def payload(epoch, batch_index):
        return {"format_version": 1, "model_config": model.config.to_dict(),
                "vocabulary": tokenizer.tokens, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "recipe": recipe,
                "data_fingerprints": data_fingerprints, "global_step": global_step,
                "training_contract": contract,
                "fragment_representation": fragment_representation,
                "split_group": stage_contract["split_identity"],
                "validation_scope": stage_contract["validation_scope"],
                "epoch": epoch, "next_batch": batch_index, "best_loss": best_loss,
                "rng": rng_state(), "stage": stage,
                "stage_trained_conditions": list(conditions), "stage_trained_properties": stage_properties,
                "inherited_trained_conditions": list(inherited_conditions),
                "inherited_trained_properties": list(inherited_properties),
                "trained_conditions": trained_conditions, "trained_properties": trained_properties,
                "inherited_pocket_encoder_signature": inherited_encoder_signature,
                "pocket_encoder_signature": encoder_signature,
                "condition_dropout": condition_dropout,
                "runtime": {"torch": str(torch.__version__), "device": device},
                "initialization": warm_start, "warm_start": warm_start}

    for epoch in range(start_epoch, epochs):
        order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        model.train()
        start = next_batch if epoch == start_epoch else 0
        for batch_index in range(start, batches_per_epoch):
            rows = [train_data[index] for index in order[batch_index * batch_size:(batch_index + 1) * batch_size]]
            x, y, batch = builder(rows, device)
            # Train on partial property prompts as well as partially absent modalities.
            property_dropout = float(settings.get("property_dropout", 0.2 if "properties" in conditions else 0.0))
            if not 0 <= property_dropout < 1:
                raise ValueError("property_dropout must be in [0, 1)")
            if property_dropout:
                batch["property_mask"] &= torch.rand(batch["property_mask"].shape, device=device) >= property_dropout
                batch["condition_mask"][:, 1] &= batch["property_mask"].any(1)
            step = global_step + 1
            if warmup and step <= warmup:
                factor = step / warmup
            else:
                progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
                factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = base_lr * factor
            optimizer.zero_grad(set_to_none=True)
            result = model(x, batch, y, condition_dropout)
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; no checkpoint overwrite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("grad_clip", 1.0)))
            optimizer.step()
            global_step += 1
            metric = {"step": global_step, "epoch": epoch + 1, "train_loss": float(loss.detach()),
                      "learning_rate": base_lr * factor}
            with history.open("a") as handle:
                handle.write(json.dumps(metric) + "\n")
            if global_step == 1 or global_step % int(settings.get("log_every_steps", 100)) == 0:
                print(json.dumps(metric), flush=True)
            if global_step % save_every == 0 or (max_steps is not None and global_step >= max_steps):
                save_checkpoint(run / "last.pt", payload(epoch, batch_index + 1))
            if max_steps is not None and global_step >= max_steps:
                print(json.dumps({"status": "step_limit", "step": global_step, "last_loss": metric["train_loss"], "checkpoint": str(run / "last.pt")}))
                return str(run / "last.pt")
        # Full, token-weighted validation with no condition dropout.
        model.eval()
        sum_loss, sum_tokens = 0.0, 0
        with torch.no_grad():
            for offset in range(0, len(valid_data), batch_size):
                rows = [valid_data[i] for i in range(offset, min(offset + batch_size, len(valid_data)))]
                x, y, batch = builder(rows, device)
                result = model(x, batch, y)
                count = int(y.ne(0).sum())
                sum_loss += float(result["loss"]) * count
                sum_tokens += count
        validation_loss = sum_loss / sum_tokens
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
        saved = payload(epoch + 1, 0)
        save_checkpoint(run / "last.pt", saved)
        if improved:
            save_checkpoint(run / "best.pt", saved)
        summary = {"epoch": epoch + 1, "step": global_step, "validation_loss": validation_loss,
                   "best_loss": best_loss, "validation_scope": stage_contract["validation_scope"]}
        with history.open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
    return str(run / "last.pt")
