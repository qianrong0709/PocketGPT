"""Read-only launch checks; never installs packages or runs a research encoder.

Use the dedicated environment's Python and an explicit absolute --config path.
The sbatch wrapper invokes this before unit tests and before creating a run.
"""
import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def check_recipe(config_file, *, device="cuda", project_root=PROJECT_ROOT,
                 expected_python_prefix=None, resume=None, run_smoke=True):
    import numpy as np
    import torch
    from rdkit import rdBase
    from rdkit.Contrib.SA_Score import sascorer  # noqa: F401: required by generation

    from pocketgpt.checkpoint import load_checkpoint
    from pocketgpt.config import (ModelConfig, parse_conditions,
                                 DEFAULT_FRAGMENT_REPRESENTATION, validate_fragment_representation)
    from pocketgpt.data import JsonlDataset, row_fragment_text, inspect_manifest
    from pocketgpt.model import PocketGPT
    from pocketgpt.pocket import file_sha256
    from pocketgpt.tokenizer import SmilesTokenizer
    from pocketgpt.training import (implementation_fingerprint, input_contract, recipe_path,
                                    runtime_contract, training_stage_contract)

    project_root = Path(project_root).resolve()
    if expected_python_prefix is not None:
        expected = Path(expected_python_prefix).resolve()
        if Path(sys.prefix).resolve() != expected:
            raise ValueError("Wrong Python environment; refusing to install or fall back")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: request an independent Slurm GPU allocation")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("Preflight supports cpu or an explicitly allocated CUDA device")
    config_file = Path(config_file)
    if not config_file.is_absolute() or not config_file.is_file():
        raise ValueError("--config must be an existing absolute recipe path")
    recipe = json.loads(config_file.read_text())
    stage = recipe["stage"]
    conditions = parse_conditions(recipe.get("conditions", "none"))
    split_group = recipe.get("split_group", "protein_id")
    fragment_representation = validate_fragment_representation(
        recipe.get("fragment_representation", DEFAULT_FRAGMENT_REPRESENTATION))
    stage_contract = training_stage_contract(stage, conditions, split_group=split_group)
    settings = recipe["training"]
    threads = int(settings.get("cpu_threads", 2))
    if threads < 1 or threads > int(os.environ.get("SLURM_CPUS_PER_TASK", threads)):
        raise ValueError("cpu_threads must fit the explicitly allocated CPU resources")
    torch.set_num_threads(threads)
    for name, default in (("epochs", 10), ("batch_size", 32), ("save_every_steps", 1000), ("log_every_steps", 100)):
        if int(settings.get(name, default)) < 1:
            raise ValueError(name + " must be positive")
    for name, default in (("learning_rate", 0.0003), ("grad_clip", 1.0)):
        value = float(settings.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive")
    for name, default in (("condition_dropout", 0.35 if conditions else 0),
                          ("property_dropout", 0.2 if "properties" in conditions else 0)):
        if not 0 <= float(settings.get(name, default)) < 1:
            raise ValueError(name + " must be in [0, 1)")

    run = Path(recipe_path(config_file, recipe["output_dir"])).resolve()
    runs_root = project_root / "runs"
    if run == runs_root or not run.is_relative_to(runs_root):
        raise ValueError("Training output must be a dedicated child of this project's runs directory")
    if run.exists() and (not run.is_dir() or (resume is None and any(run.iterdir()))):
        raise FileExistsError("New training must use an empty/new run directory; explicitly resume otherwise")
    for key in ("vocabulary", "init_checkpoint", "condition_init_checkpoint", "warm_start_checkpoint", "legacy_checkpoint", "legacy_vocabulary"):
        if recipe.get(key) and not Path(recipe_path(config_file, recipe[key])).is_file():
            raise FileNotFoundError("Required local initialization file is absent: " + key)
    if sum(bool(recipe.get(key)) for key in ("init_checkpoint", "condition_init_checkpoint", "warm_start_checkpoint", "legacy_checkpoint")) > 1:
        raise ValueError("Choose exactly one checkpoint initialization method")

    paths = {name: Path(recipe_path(config_file, recipe[name + "_data"])) for name in ("train", "valid")}
    if paths["train"].resolve() == paths["valid"].resolve():
        raise ValueError("Training and validation cannot reference the same file")
    with JsonlDataset(paths["train"]) as train, JsonlDataset(paths["valid"]) as valid:
        info = {name: inspect_manifest(dataset, conditional=stage == "finetune", conditions=conditions,
                                      fragment_representation=fragment_representation, split_group=split_group)
                for name, dataset in (("train", train), ("valid", valid))}
        identity = "groups" if stage == "finetune" else "molecules"
        if info["train"][identity] & info["valid"][identity]:
            raise ValueError("Training/validation leakage: " + stage_contract["split_identity"] + " overlaps")
        for name in ("train", "valid"):
            if set(conditions) - info[name]["modalities"]:
                raise ValueError(name + " lacks effective labels for a selected condition")
        contract = input_contract(train, valid, conditions, recipe.get("pocket_encoder"),
                                  recipe.get("allow_unverified_pocket_embeddings", False))
        contract["training_stage"] = stage_contract
        contract["fragment_representation"] = fragment_representation
        contract["implementation_sha256"] = implementation_fingerprint()
        contract["runtime"] = runtime_contract(device)
        fingerprints = {name: file_sha256(path) for name, path in paths.items()}
        if recipe.get("vocabulary"):
            tokenizer = SmilesTokenizer.load(recipe_path(config_file, recipe["vocabulary"]))
        else:
            def texts():
                for row in train:
                    yield row["smiles"]
                    if "fragment" in conditions and row.get("fragment"):
                        yield row_fragment_text(row, fragment_representation)
            tokenizer = SmilesTokenizer.build(texts())
        model_config = ModelConfig(vocab_size=len(tokenizer.tokens), **recipe.get("model", {}))
        if model_config.architecture_version != 2 or model_config.pocket_dim != 128:
            raise ValueError("New training requires architecture v2 and the original 128-dimensional pocket interface")
        lengths = {}
        for name, dataset in (("train", train), ("valid", valid)):
            max_molecule, max_fragment = 0, 0
            for row in dataset:
                max_molecule = max(max_molecule, len(tokenizer.encode(row["smiles"])) + 1)
                if "fragment" in conditions and row.get("fragment"):
                    max_fragment = max(max_fragment, len(tokenizer.encode(
                        row_fragment_text(row, fragment_representation))))
            if max(max_molecule, max_fragment) > model_config.block_size:
                raise ValueError(name + " exceeds context; do not truncate chemical tokens")
            lengths[name] = {"molecule_with_eos": max_molecule, "fragment": max_fragment}
        rows = {"train": len(train), "valid": len(valid)}

    if resume:
        resume_path = Path(resume).resolve()
        if resume_path.parent != run:
            raise ValueError("Resume checkpoint must belong to this recipe's dedicated run")
        old_model, old_tokenizer, saved = load_checkpoint(resume_path)
        if saved["recipe"] != recipe or saved["data_fingerprints"] != fingerprints or saved.get("training_contract") != contract:
            raise ValueError("Exact resume contract changed; use a new warm-start run")
        if old_tokenizer.tokens != tokenizer.tokens or old_model.config.to_dict() != model_config.to_dict():
            raise ValueError("Resume architecture or vocabulary mismatch")
        del old_model

    smoke = "not_requested"
    if run_smoke:
        # Synthetic tensors only: do not run HTMD, MGLTools or a research PDB here.
        model = PocketGPT(model_config).to(device).train()
        length = min(8, model_config.block_size)
        tokens = torch.full((1, length), tokenizer.bos_id, dtype=torch.long, device=device)
        batch = {"pocket": torch.zeros(1, 128, device=device),
                 "properties": torch.zeros(1, 4, device=device),
                 "property_mask": torch.ones(1, 4, dtype=torch.bool, device=device),
                 "ifp": torch.zeros(1, 6, device=device),
                 "fragment_tokens": tokens[:, :min(3, length)],
                 "condition_mask": torch.ones(1, 4, dtype=torch.bool, device=device)}
        result = model(tokens, batch, tokens)
        if not torch.isfinite(result["loss"]):
            raise FloatingPointError("Synthetic model forward is non-finite")
        result["loss"].backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError("Synthetic model backward is non-finite")
        torch.from_numpy(np.arange(4, dtype=np.float32)).numpy()
        smoke = "finite_forward_backward"

    return {"status": "preflight_passed", "stage": stage, "conditions": list(conditions),
            "split_identity": stage_contract["split_identity"],
            "validation_scope": stage_contract["validation_scope"],
            "fragment_representation": fragment_representation,
            "architecture_version": model_config.architecture_version, "rows": rows,
            "max_tokens": lengths, "output_dir": str(run), "data_sha256": fingerprints,
            "pocket_contract": contract["pocket"], "device": device,
            "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "rdkit": rdBase.rdkitVersion},
            "synthetic_smoke": smoke, "resume": bool(resume), "research_encoder_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--expected-python-prefix", type=Path)
    parser.add_argument("--resume")
    args = parser.parse_args()
    try:
        result = check_recipe(args.config, device=args.device, project_root=args.project_root,
                              expected_python_prefix=args.expected_python_prefix, resume=args.resume)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        print(json.dumps({"status": "preflight_failed", "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
