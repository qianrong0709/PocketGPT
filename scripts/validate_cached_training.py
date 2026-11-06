"""Technical-only short training/resume/generation check on real cached JSONL.

This is not convergence, molecular quality, protein-heldout, or raw-PDB encoder
validation. No initialization checkpoint is read and no research encoder runs.
Use a new --output-dir and an explicitly allocated --device cuda on the server.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
INITIALIZERS = ("init_checkpoint", "condition_init_checkpoint", "warm_start_checkpoint",
                "legacy_checkpoint", "legacy_vocabulary")
SCOPE = "technical_cached_training_resume_generation_NOT_convergence_or_encoder_validation"


def _preflight():
    spec = importlib.util.spec_from_file_location(
        "pocketgpt_cached_validation_preflight", Path(__file__).with_name("server_preflight.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_recipe


@contextlib.contextmanager
def deterministic_execution(device):
    """Force deterministic kernels, including CUDA SDPA's math implementation."""
    # main calls this before any CUDA operation; library callers must do so too.
    previous_environment = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    import torch
    if device.startswith("cuda") and torch.cuda.is_initialized() and previous_environment not in (":4096:8", ":16:8"):
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before initializing CUDA")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    old = {"algorithms": torch.are_deterministic_algorithms_enabled(),
           "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
           "benchmark": torch.backends.cudnn.benchmark,
           "cudnn_deterministic": torch.backends.cudnn.deterministic,
           "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn_tf32": torch.backends.cudnn.allow_tf32,
           "flash": torch.backends.cuda.flash_sdp_enabled(),
           "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
           "math": torch.backends.cuda.math_sdp_enabled()}
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        yield
    finally:
        torch.use_deterministic_algorithms(old["algorithms"], warn_only=old["warn_only"])
        torch.backends.cudnn.benchmark = old["benchmark"]
        torch.backends.cudnn.deterministic = old["cudnn_deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = old["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = old["cudnn_tf32"]
        torch.backends.cuda.enable_flash_sdp(old["flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(old["memory_efficient"])
        torch.backends.cuda.enable_math_sdp(old["math"])
        if previous_environment is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_environment


def _tensor_hash(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compare_exact(left, right):
    """Compare every tensor and scalar, not approximate parameter summaries."""
    import torch
    differences, tensor_count = [], 0

    def visit(a, b, path):
        nonlocal tensor_count
        if torch.is_tensor(a) or torch.is_tensor(b):
            tensor_count += 1
            equal = (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype
                     and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()))
            if not equal:
                differences.append(path)
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                differences.append(path + ".keys")
            for key in a.keys() & b.keys():
                visit(a[key], b[key], path + "." + str(key))
        elif isinstance(a, (list, tuple)) and type(a) is type(b):
            if len(a) != len(b):
                differences.append(path + ".length")
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + "[" + str(index) + "]")
        elif type(a) is not type(b) or a != b:
            differences.append(path)

    visit(left, right, "state")
    return {"exact_equal": not differences, "compared_tensors": tensor_count,
            "different_values": len(differences), "first_differences": differences[:20]}


def _all_finite(value):
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return math.isfinite(value) if isinstance(value, float) else True


def _branch_changes(before_hashes, after):
    groups = {"pocket": ("pocket_projector.",), "properties": ("property_projector.",),
              "fragment": ("fragment_projector.", "fragment_pos_emb.", "fragment_sequence_encoder."),
              "ifp": ("ifp_projector.",), "fusion": ("gate.", "modality_emb", "null_prefix"),
              "decoder": ("tok_emb.", "type_emb.", "pos_emb", "blocks.", "ln_f.", "head.")}
    report = {}
    for branch, prefixes in groups.items():
        keys = [key for key in before_hashes if key.startswith(prefixes)]
        changed = [key for key in keys if before_hashes[key] != _tensor_hash(after[key])]
        report[branch] = {"tensors": len(keys), "changed_tensors": len(changed), "updated": bool(changed)}
    return report


def _gradient_probe(checkpoint, row, builder, device):
    """Independent real-row backward; no optimizer/weight decay or weight step."""
    import torch
    from pocketgpt.checkpoint import load_checkpoint

    model, _, _ = load_checkpoint(checkpoint, device)
    model.train()
    model.zero_grad(set_to_none=True)
    input_ids, targets, batch = builder([row], device)
    if not bool(batch["condition_mask"].all()):
        raise AssertionError("Gradient probe requires one actual all-four-condition row")
    loss = model(input_ids, batch, targets, condition_dropout=0.0)["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("Real cached-row gradient probe loss is non-finite")
    loss.backward()
    # Four projector groups are intentionally separate from sequence encoding:
    # a sequence encoder gradient cannot stand in for the fragment projector.
    groups = {"pocket": ("pocket_projector.",), "properties": ("property_projector.",),
              "fragment": ("fragment_projector.",), "ifp": ("ifp_projector.",),
              "fragment_encoder": ("fragment_pos_emb.", "fragment_sequence_encoder."),
              "fusion": ("gate.", "modality_emb", "null_prefix"),
              "decoder": ("tok_emb.", "type_emb.", "pos_emb", "blocks.", "ln_f.", "head.")}
    gradients = {}
    for name, prefixes in groups.items():
        parameters = [(key, value) for key, value in model.named_parameters() if key.startswith(prefixes)]
        present = [value.grad for _, value in parameters if value.grad is not None]
        finite = bool(present) and all(bool(torch.isfinite(value).all()) for value in present)
        nonzero = sum(bool(value.ne(0).any()) for value in present)
        gradients[name] = {"parameter_tensors": len(parameters), "gradient_tensors": len(present),
                           "all_present_gradients_finite": finite,
                           "nonzero_gradient_tensors": nonzero,
                           "finite_nonzero_gradient_evidence": finite and nonzero > 0}
    result = {"source": "actual_all_four_cached_row_with_language_targets",
              "loss": float(loss.detach()), "condition_mask": batch["condition_mask"][0].tolist(),
              "condition_dropout": 0.0, "property_dropout": 0.0,
              "optimizer_step_executed": False, "weight_decay_applied": False,
              "probe_model_is_separate_loaded_copy": True,
              "performed_after_resume_reproducibility_comparison": True,
              "branches": gradients,
              "all_four_projectors_finite_nonzero": all(
                  gradients[name]["finite_nonzero_gradient_evidence"]
                  for name in ("pocket", "properties", "fragment", "ifp"))}
    del model
    if not result["all_four_projectors_finite_nonzero"]:
        raise AssertionError("Real all-four cached row did not supply finite nonzero gradients to every projector")
    return result


def _metric_after(path, offset, step):
    records = [json.loads(line) for line in path.read_text().splitlines()[offset:]]
    metrics = [row for row in records if row.get("step") == step and "train_loss" in row]
    if len(metrics) != 1 or not _all_finite(metrics[0]):
        raise AssertionError("Expected one finite technical resume-step metric")
    return metrics[0]


def validate_cached_training(config_file, output_dir, *, device="cuda", steps=5):
    """Use production batch/model/vocabulary and all input rows, never raw PDBs."""
    from pocketgpt.config import CONDITIONS, ModelConfig, parse_conditions
    from pocketgpt.data import BatchBuilder, JsonlDataset, row_fragment_text
    from pocketgpt.pocket import LEGACY_SIGNATURE, file_sha256
    from pocketgpt.tokenizer import SmilesTokenizer
    from pocketgpt.training import recipe_path

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be cpu or an explicitly allocated CUDA device")
    config_file = Path(config_file).resolve(strict=True)
    original_recipe = json.loads(config_file.read_text())
    if original_recipe.get("stage") != "finetune" or set(parse_conditions(original_recipe.get("conditions", "none"))) != set(CONDITIONS):
        raise ValueError("Cached validation requires a four-condition finetune recipe")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Validation output must not exist; never reuse a training run")
    recipe = json.loads(json.dumps(original_recipe))
    paths = {name: Path(recipe_path(config_file, recipe[name + "_data"])) for name in ("train", "valid")}
    for name, path in paths.items():
        recipe[name + "_data"] = str(path)
    if recipe.get("vocabulary"):
        recipe["vocabulary"] = recipe_path(config_file, recipe["vocabulary"])
    tracked_paths = dict(paths, source_config=config_file)
    if recipe.get("vocabulary"):
        tracked_paths["vocabulary"] = Path(recipe["vocabulary"])
    original_hashes = {key: file_sha256(path) for key, path in tracked_paths.items()}
    removed = {key: recipe.pop(key) for key in INITIALIZERS if key in recipe}
    removed_encoder = recipe.pop("pocket_encoder", None) is not None
    recipe["output_dir"] = str(output_dir / "runs" / "short-training")
    recipe["validation_only"] = True
    recipe["technical_validation_scope"] = SCOPE
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "validation_only": True, "scope": SCOPE,
              "scientific_quality_validated": False, "training_converged": False,
              "full_corpus_training_claimed": False, "gradient_updates_short_run_only": True,
              "raw_pdb_inference_validated": False, "encoder_executed": False,
              "legacy_encoder_profile_verified": False, "real_cached_rows_all_preflighted": False,
              "initialization": "fresh_random_parameters_for_technical_validation_only",
              "removed_initializer_fields": sorted(removed), "raw_encoder_config_removed": removed_encoder,
              "source_config": str(config_file), "source_sha256_before": original_hashes,
              "steps_requested": steps, "device": device}
    verification_file = output_dir / "verification.json"
    try:
        # Refuse raw structures before preflight can inspect an encoder profile.
        cached_rows, sizes = {}, {}
        for name, path in paths.items():
            cached = 0
            with JsonlDataset(path) as dataset:
                sizes[name] = len(dataset)
                for row in dataset:
                    if row.get("pocket"):
                        raise ValueError("Cached-only validation refuses every raw pocket structure")
                    cached += "pocket_embedding" in row
            cached_rows[name] = cached
        report["cached_embedding_rows"] = cached_rows
        settings = recipe["training"]
        available_steps = int(settings.get("epochs", 10)) * math.ceil(sizes["train"] / int(settings.get("batch_size", 32)))
        if available_steps < steps + 1:
            raise ValueError("Original recipe epochs do not permit the requested short run plus one resume step")
        validation_config = output_dir / "validation-recipe.json"
        validation_config.write_text(json.dumps(recipe, indent=2, allow_nan=False) + "\n")
        with deterministic_execution(device):
            import torch
            from pocketgpt.checkpoint import seed_all
            from pocketgpt.generation import generate
            from pocketgpt.model import PocketGPT
            from pocketgpt.training import train

            report["preflight"] = _preflight()(validation_config, device=device,
                                                  project_root=output_dir, run_smoke=False)
            report["real_cached_rows_all_preflighted"] = True
            report["pocket_provenance_status"] = ("unverified-legacy" if
                report["preflight"]["pocket_contract"]["signature"] == LEGACY_SIGNATURE else
                "cached-signature-identified-NOT-revalidated-against-original-encoder")
            report["production_settings_preserved"] = {
                "batch_size": int(settings.get("batch_size", 32)), "model": recipe.get("model", {}),
                "vocabulary_source": "explicit_source_vocabulary" if recipe.get("vocabulary") else "all_training_rows_only",
                "epochs": int(settings.get("epochs", 10))}
            with JsonlDataset(paths["train"]) as dataset:
                if recipe.get("vocabulary"):
                    tokenizer = SmilesTokenizer.load(recipe["vocabulary"])
                else:
                    def texts():
                        for row in dataset:
                            yield row["smiles"]
                            if row.get("fragment"):
                                yield row_fragment_text(row, recipe.get("fragment_representation", "canonical-smiles-v1"))
                    tokenizer = SmilesTokenizer.build(texts())
            seed_all(settings.get("seed", 42))
            initial_model = PocketGPT(ModelConfig(vocab_size=len(tokenizer.tokens), **recipe.get("model", {})))
            initial_hashes = {key: _tensor_hash(value) for key, value in initial_model.state_dict().items()}
            del initial_model
            builder = BatchBuilder(tokenizer, int(recipe.get("model", {}).get("block_size", 256)), CONDITIONS,
                                   fragment_representation=recipe.get("fragment_representation", "canonical-smiles-v1"))
            request, probe_row, origin = None, None, None
            for split, path in paths.items():
                with JsonlDataset(path) as dataset:
                    for index, row in enumerate(dataset):
                        _, _, batch = builder([row], language_targets=False)
                        if bool(batch["condition_mask"].all()):
                            request = {key: row[key] for key in ("pocket_embedding", "pocket_encoder_signature",
                                "properties", "fragment", "fragment_representation", "ifp") if key in row}
                            probe_row = row
                            origin = {"split": split, "row_index_zero_based": index}
                            break
                if request is not None:
                    break
            if request is None:
                raise ValueError("No actual cached row contains all four effective conditions")
            checkpoint = Path(train(validation_config, max_steps=steps, device_override=device))
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state["global_step"] != steps:
                raise AssertionError("Short training stopped at an unexpected step")
            before_resume = checkpoint.parent / "before-resume.pt"
            shutil.copyfile(checkpoint, before_resume)
            report["parameter_updates_by_branch"] = _branch_changes(initial_hashes, state["model"])
            report["all_conditioning_branches_updated"] = all(
                report["parameter_updates_by_branch"][name]["updated"] for name in CONDITIONS)
            if not report["all_conditioning_branches_updated"]:
                raise AssertionError("Short run did not update every conditioning branch; do not claim complete branch validation")
            if not _all_finite({"model": state["model"], "optimizer": state["optimizer"]}):
                raise FloatingPointError("Non-finite short-run checkpoint")
            del state
            metric_path = checkpoint.parent / "metrics.jsonl"
            offset = len(metric_path.read_text().splitlines())
            train(validation_config, resume=before_resume, max_steps=steps + 1, device_override=device)
            reference_metric = _metric_after(metric_path, offset, steps + 1)
            reference_path = checkpoint.parent / "reference-resume.pt"
            shutil.copyfile(checkpoint, reference_path)
            reference = torch.load(reference_path, map_location="cpu", weights_only=True)
            offset = len(metric_path.read_text().splitlines())
            train(validation_config, resume=before_resume, max_steps=steps + 1, device_override=device)
            replay_metric = _metric_after(metric_path, offset, steps + 1)
            replay_path = checkpoint.parent / "replay-resume.pt"
            shutil.copyfile(checkpoint, replay_path)
            replay = torch.load(replay_path, map_location="cpu", weights_only=True)
            report["resume"] = {"from_step": steps, "to_step": steps + 1,
                "model": compare_exact(reference["model"], replay["model"]),
                "optimizer": compare_exact(reference["optimizer"], replay["optimizer"]),
                "rng": compare_exact(reference["rng"], replay["rng"]),
                "reference_step_metric": reference_metric, "replay_step_metric": replay_metric,
                "step_metric_exact_equal": reference_metric == replay_metric,
                "metrics_log_has_intentional_replayed_step": True}
            if reference["global_step"] != steps + 1 or replay["global_step"] != steps + 1:
                raise AssertionError("Resume stopped at an unexpected step")
            if not all(report["resume"][key]["exact_equal"] for key in ("model", "optimizer", "rng")) or reference_metric != replay_metric:
                raise AssertionError("Same-checkpoint deterministic resume did not reproduce every model/optimizer value and step metric")
            if not _all_finite({"model": replay["model"], "optimizer": replay["optimizer"]}):
                raise FloatingPointError("Non-finite resumed checkpoint")
            del reference, replay
            # Parameter changes can include AdamW decay; establish actual loss
            # gradients independently, after the resume RNG comparison is done.
            report["gradient_probe"] = _gradient_probe(replay_path, probe_row, builder, device)
            report["parameter_change_evidence_includes_possible_weight_decay"] = True
            subsets = (CONDITIONS, ("pocket", "properties"), ("pocket", "fragment"), ("pocket", "ifp"))
            results = []
            for index, subset in enumerate(subsets):
                target = output_dir / "generation" / (str(index) + "-" + "-".join(subset) + ".csv")
                summary = generate(replay_path, request, target, conditions=subset, count=2,
                                   batch_size=2, max_tokens=min(32, builder.block_size),
                                   seed=int(settings.get("seed", 42)), device=device)
                if summary["attempted"] != 2 or summary["conditions"] != list(subset):
                    raise AssertionError("Generation entry did not execute the requested conditions/count")
                results.append({"conditions": list(subset), "attempted": summary["attempted"],
                    "output": str(target), "fragment_match_evaluation": summary["fragment_match_evaluation"],
                    "pocket_provenance_status": summary["pocket_provenance_status"],
                    "technical_entry_only": True, "molecular_quality_scored": False})
            report["generation"] = {"source_row": origin, "requests": results,
                "count_per_request": 2, "max_tokens": min(32, builder.block_size),
                "scientific_valid_rate_claimed": False}
            report["checkpoint"] = str(replay_path)
            report["determinism"] = {"algorithms": True, "cublas_workspace_config": ":4096:8",
                                     "sdpa_backend": "math", "tf32": False}
        report["status"] = "technical_validation_passed"
    except Exception as error:
        report["status"] = "technical_validation_failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        final_hashes = {key: file_sha256(path) for key, path in tracked_paths.items()}
        report["source_sha256_after"] = final_hashes
        report["original_data_hashes_unchanged"] = original_hashes == final_hashes
        if not report["original_data_hashes_unchanged"]:
            report["status"] = "technical_validation_failed"
            report["error"] = {"type": "SourceMutation", "message": "A tracked source changed during validation"}
        verification_file.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["original_data_hashes_unchanged"]:
        raise RuntimeError("Tracked source changed during validation")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", default=5, type=int)
    args = parser.parse_args()
    # Must precede imports which can initialize CUDA.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        result = validate_cached_training(args.config, args.output_dir, device=args.device, steps=args.steps)
    except (OSError, ValueError, RuntimeError, AssertionError, ImportError, KeyError, TypeError) as error:
        print(json.dumps({"status": "technical_validation_failed", "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "verification": str(args.output_dir.resolve() / "verification.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
