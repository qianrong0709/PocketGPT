"""Validate a real pocket workflow without calling it a completed model training.

Only the explicitly configured local DeeplyTough process is invoked. A new
output directory owns all caches, smoke checkpoints and generated samples.
Research input files are never changed. Optional labels must be supplied, not
fabricated: no protein identity, IFP, property or fragment is inferred here.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pocketgpt.checkpoint import (load_checkpoint, restore_rng, rng_state,
                                  save_checkpoint, seed_all)
from pocketgpt.config import CONDITIONS, PROPERTIES, ModelConfig
from pocketgpt.data import BatchBuilder, canonical_fragment, canonical_smiles, property_vector
from pocketgpt.generation import generate
from pocketgpt.ifp import canonical_ifp
from pocketgpt.model import PocketGPT
from pocketgpt.pocket import PocketResolver, content_hash, file_sha256, validate_embedding
from pocketgpt.tokenizer import SmilesTokenizer
from pocketgpt.training import implementation_fingerprint, runtime_contract


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _request_labels(request):
    """Validate explicit labels without turning missing labels into zeroes."""
    row = copy.deepcopy(request or {})
    if "pocket_embedding" in row or row.get("pocket"):
        raise ValueError("Request must not override the explicit raw pocket/protein inputs")
    if row.get("smiles"):
        try:
            with rdBase.BlockLogs():
                row["smiles"] = canonical_smiles(row["smiles"])
        except (ValueError, TypeError):
            raise ValueError("The supplied training target SMILES is invalid") from None
    elif "smiles" in row:
        raise ValueError("The supplied training target SMILES is empty")
    selected = ["pocket"]
    if row.get("properties") is not None:
        _, presence = property_vector(row["properties"])
        if any(presence):
            selected.append("properties")
    if row.get("fragment") not in (None, ""):
        try:
            with rdBase.BlockLogs():
                row["fragment"] = canonical_fragment(row["fragment"])
        except (ValueError, TypeError):
            raise ValueError("The supplied fragment label is invalid; no repair/masking performed") from None
        selected.append("fragment")
    if row.get("ifp") is not None:
        row["ifp"] = canonical_ifp(row["ifp"])
        selected.append("ifp")
    return row, tuple(selected)


def _equal_nested(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal_nested(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal_nested(a, b) for a, b in zip(left, right))
    return left == right


def _optimizer_on_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and key != "step":
                state[key] = value.to(device)


def _step(model, optimizer, x, y, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = model(x, batch, y)
    if not torch.isfinite(result["loss"]):
        raise FloatingPointError("Non-finite smoke training loss")
    result["loss"].backward()
    gradients = {}
    groups = {"pocket": "pocket_projector.", "properties": "property_projector.",
              "fragment": "fragment_sequence_encoder.", "ifp": "ifp_projector.",
              "gate": "gate.weight", "decoder": "blocks."}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Non-finite smoke training gradient")
            for group, prefix in groups.items():
                if name.startswith(prefix):
                    gradients[group] = gradients.get(group, 0.0) + float(parameter.grad.detach().abs().sum())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(result["loss"].detach()), "gradient_l1": gradients}


def _resume_smoke(checkpoint, expected_contract, device):
    model, tokenizer, saved = load_checkpoint(checkpoint, device)
    if saved.get("validation_contract") != expected_contract:
        raise ValueError("Smoke resume contract changed: inputs, code, model or runtime differ")
    if saved["model_config"] != expected_contract["model_config"]:
        raise ValueError("Smoke checkpoint architecture differs from its contract")
    optimizer = torch.optim.AdamW(model.parameters(), lr=expected_contract["learning_rate"],
                                  weight_decay=expected_contract["weight_decay"])
    optimizer.load_state_dict(saved["optimizer"])
    _optimizer_on_device(optimizer, device)
    restore_rng(saved["rng"])
    return model, tokenizer, optimizer, saved


def validate_workflow(pocket, encoder_config, output_dir, request=None, protein=None,
                      device="cuda", steps=5, model_settings=None, seed=42,
                      count=4, max_tokens=16, encode_only=False):
    """Real encoding plus a *single-example technical smoke*, never an efficacy test.

    The default generator has the production recipe's 8 layers / 256 dimensions.
    ``model_settings`` can explicitly request a smaller synthetic test model; its
    full configuration and scope are recorded. No train/valid split or protein
    identity is invented. Labels are used solely for this technical smoke.
    """
    if type(steps) is not int or steps < 2:
        raise ValueError("steps must be an integer of at least 2 to verify resume")
    if type(count) is not int or count < 1 or type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("count and max_tokens must be positive integers")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("Choose cpu or an allocated CUDA device")
    if device.startswith("cuda") and (not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID")):
        raise RuntimeError("CUDA validation requires an independent Slurm GPU allocation")
    row, selected = _request_labels(request)
    if not encode_only and not row.get("smiles"):
        raise ValueError("Training smoke needs an explicit real target smiles; use --encode-only otherwise")
    pocket = Path(pocket).resolve()
    protein = Path(protein).resolve() if protein else pocket
    if not pocket.is_file() or not protein.is_file():
        raise FileNotFoundError("Explicit pocket/protein PDB is absent")
    config = dict(encoder_config)
    if not config or not all(config.get(key) for key in ("repo", "python", "weights", "mgltools_prefix")):
        raise ValueError("Original repo/python/weights/MGLTools configuration is required")
    destination = Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Workflow validation requires a new output directory")
    destination.mkdir(parents=True, exist_ok=False)
    config["cache"] = str(destination / "pocket-cache")
    row["pocket"] = {"pocket_pdb": str(pocket), "protein_pdb": str(protein)}
    stage = "real_pocket_encoding"
    try:
        resolver = PocketResolver(config)
        signature = resolver.signature()
        fingerprint = resolver.input_fingerprint(row)
        embedding = validate_embedding(resolver.resolve(row))
        cache = Path(config["cache"]) / content_hash(fingerprint)
        cache_files = (cache / "embedding.npy", cache / "provenance.json", cache / "encoder.log")
        if not all(path.is_file() for path in cache_files):
            raise RuntimeError("Encoder output/cache evidence is incomplete")
        first_artifacts = {path.name: (path.stat().st_mtime_ns, file_sha256(path)) for path in cache_files}
        repeated = validate_embedding(PocketResolver(config).resolve(row))
        second_artifacts = {path.name: (path.stat().st_mtime_ns, file_sha256(path)) for path in cache_files}
        if not np.array_equal(embedding, repeated) or first_artifacts != second_artifacts:
            raise RuntimeError("Repeated encoding did not reuse identical validated cache artifacts")
        encoding = {"dimensions": int(embedding.size), "finite": bool(np.isfinite(embedding).all()),
                    "l2_norm": float(np.linalg.norm(embedding)), "signature": signature,
                    "input_fingerprint": fingerprint, "embedding_sha256": first_artifacts["embedding.npy"][1],
                    "second_resolution_identical": True, "cache_artifacts_unchanged_on_second_resolution": True,
                    "cache_directory": str(cache), "encoder_frozen": True}
        report = {"status": "encoding_passed" if encode_only else "in_progress",
                  "scope": "real-PDB technical workflow smoke; NOT full training or control efficacy",
                  "encoding": encoding, "source_labels_changed": False,
                  "protein_identity_invented": False, "scientific_control_efficacy_evaluated": False,
                  "full_corpus_training_completed": False, "trained_production_checkpoint": False,
                  "device": device, "training": None, "generation": None}
        _json(destination / "encoding.json", encoding)
        if encode_only:
            _json(destination / "verification.json", report)
            return report

        stage = "generator_forward_backward_and_resume"
        seed_all(seed)
        torch.set_num_threads(2)
        texts = [row["smiles"]]
        if "fragment" in selected:
            texts.append(row["fragment"])
        tokenizer = SmilesTokenizer.build(texts)
        settings = dict(model_settings or {})
        if "vocab_size" in settings:
            raise ValueError("Smoke vocabulary is built from explicit labels, not a supplied vocab_size")
        model_config = ModelConfig(vocab_size=len(tokenizer.tokens), **settings)
        if max_tokens > model_config.block_size:
            raise ValueError("Generation max_tokens exceeds model context")
        model = PocketGPT(model_config).to(device)
        initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        # Isolate data-driven updates. Weight decay would otherwise modify
        # inactive branches that receive zero gradients, confusing diagnostics.
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0)
        builder = BatchBuilder(tokenizer, model_config.block_size, selected, config)
        x, y, batch = builder([row], device)
        if batch["condition_mask"][0].tolist() != [name in selected for name in CONDITIONS]:
            raise RuntimeError("Explicit request labels and effective condition masks differ")

        def contract():
            return {"scope": "single-example-smoke-v1", "input": resolver.input_fingerprint(row),
                    "request_sha256": content_hash(row), "conditions": list(selected),
                    "encoder_signature": resolver.signature(refresh=True),
                    "implementation_sha256": implementation_fingerprint(),
                    "validator_sha256": file_sha256(__file__), "runtime": runtime_contract(device),
                    "model_config": model_config.to_dict(), "learning_rate": 0.0003, "weight_decay": 0.0}

        expected_contract = contract()
        metrics = [_step(model, optimizer, x, y, batch) for _ in range(steps - 1)]
        checkpoint = destination / "before-resume.pt"
        property_names = [name for name, present in zip(PROPERTIES, property_vector(row.get("properties"))[1]) if present]

        def payload(step):
            return {"format_version": 1, "model_config": model.config.to_dict(),
                    "vocabulary": tokenizer.tokens, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "rng": rng_state(),
                    "global_step": step, "recipe": {"stage": "technical_smoke", "conditions": list(selected),
                                                      "pocket_encoder": config},
                    "trained_conditions": list(selected), "trained_properties": property_names,
                    "pocket_encoder_signature": signature, "validation_only": True,
                    "validation_contract": expected_contract,
                    "capability_note": "Only a few single-example smoke updates; no scientific efficacy established."}

        save_checkpoint(checkpoint, payload(steps - 1))
        reference_metric = _step(model, optimizer, x, y, batch)
        reference_model = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        reference_optimizer = copy.deepcopy(optimizer.state_dict())
        if contract() != expected_contract:
            raise ValueError("Input/encoder/code/runtime changed during smoke validation")
        model, tokenizer, optimizer, saved = _resume_smoke(checkpoint, contract(), device)
        resumed_metric = _step(model, optimizer, x, y, batch)
        exact = (reference_metric == resumed_metric and _equal_nested(reference_model, model.state_dict())
                 and _equal_nested(reference_optimizer, optimizer.state_dict()))
        if not exact:
            raise RuntimeError("Single-step resumed continuation differs from uninterrupted reference")
        metrics.append(resumed_metric)
        final_checkpoint = destination / "smoke-only.pt"
        save_checkpoint(final_checkpoint, payload(steps))
        changed = {group: any(not torch.equal(initial[name], value.cpu()) for name, value in model.state_dict().items()
                              if name.startswith(prefix))
                   for group, prefix in (("pocket", "pocket_projector."), ("properties", "property_projector."),
                                         ("fragment", "fragment_sequence_encoder."), ("ifp", "ifp_projector."),
                                         ("gate", "gate.weight"), ("decoder", "blocks."))}
        for name in [*selected, "decoder"]:
            if not changed[name]:
                raise RuntimeError("Enabled generator branch did not update: " + name)
        if len(selected) > 1 and not changed["gate"]:
            raise RuntimeError("Multimodal fusion gate did not update")
        report["training"] = {"scope": "single real example technical optimization, no train/validation split",
                              "steps": steps, "metrics": metrics, "model_config": model_config.to_dict(),
                              "parameter_count": sum(p.numel() for p in model.parameters()),
                              "optimizer": {"name": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.0,
                                            "scope": "zero decay isolates data-driven gradient updates for diagnosis"},
                              "enabled_conditions": list(selected), "parameter_groups_changed": changed,
                              "checkpoint_roundtrip": True, "resume_continuation_model_optimizer_rng_exact": exact,
                              "resume_scope": "one uninterrupted versus resumed step on the same example",
                              "source_checkpoint": "fresh initialization", "production_trained": False,
                              "checkpoint": str(final_checkpoint)}
        stage = "conditional_sampling_interfaces"
        checks, skipped = {}, {}
        candidates = {"pocket": ("pocket",), "pocket_properties": ("pocket", "properties"),
                      "pocket_fragment": ("pocket", "fragment"),
                      "all_supplied": selected}
        if set(selected) == set(CONDITIONS):
            candidates["all_four"] = CONDITIONS
        else:
            skipped["all_four"] = "IFP/properties/fragment labels were not all explicitly supplied; none fabricated"
        for name, subset in candidates.items():
            if set(subset) - set(selected):
                skipped[name] = "Required real label not supplied"
                continue
            result = generate(final_checkpoint, row, destination / (name + ".csv"), subset,
                              count=count, max_tokens=max_tokens, batch_size=count, device=device,
                              encoder=config, seed=seed)
            if result["attempted"] != count or result["conditions"] != list(subset):
                raise RuntimeError("Sampling output count or selected conditions differ")
            checks[name] = {key: result[key] for key in ("attempted", "valid", "unique_valid", "conditions")}
        if contract() != expected_contract:
            raise ValueError("Input/encoder/code/runtime changed during generation")
        report["generation"] = {"interface_checks": checks, "skipped": skipped,
                                "validity_counts_are_not_an_efficacy_evaluation": True}
        report["status"] = "technical_smoke_passed"
        _json(destination / "verification.json", report)
        return report
    except BaseException as error:
        _json(destination / "failure.json", {"status": "failed", "stage": stage,
                                            "error_type": type(error).__name__, "error": str(error),
                                            "full_corpus_training_completed": False,
                                            "scientific_control_efficacy_evaluated": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pocket", required=True)
    parser.add_argument("--protein")
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--request", help="Local JSON with real smiles and optional properties/fragment/ifp labels")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--model-config", help="Local JSON: ModelConfig fields, or recipe containing model")
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()
    encoder = json.loads(Path(args.encoder_config).read_text())
    request = json.loads(Path(args.request).read_text()) if args.request else None
    model = json.loads(Path(args.model_config).read_text()) if args.model_config else None
    if model and "model" in model:
        model = model["model"]
    try:
        result = validate_workflow(args.pocket, encoder, args.output_dir, request, args.protein,
                                   args.device, args.steps, model, count=args.count,
                                   max_tokens=args.max_tokens, encode_only=args.encode_only)
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "output_dir": args.output_dir}), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "output_dir": args.output_dir,
                      "full_training_completed": False, "control_efficacy_evaluated": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
