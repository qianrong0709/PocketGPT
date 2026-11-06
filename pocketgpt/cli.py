import argparse
import json
from pathlib import Path

from .config import CONDITIONS, PROPERTIES


def prepare_request_files(spec_file, output_file, encoder_file=None, resolve_pocket=False,
                          fragment_representation=None):
    """Local feature orchestration; JSON cannot select executable providers."""
    from .features import prepare_inference_request
    from .pocket import PocketResolver

    source, output = Path(spec_file).resolve(), Path(output_file).resolve()
    metadata = output.with_suffix(".provenance.json")
    if output == metadata or output.exists() or metadata.exists():
        raise FileExistsError("Request or provenance output already exists; choose a new filename")
    spec = json.loads(source.read_text())
    allowed = {"pocket_embedding", "pocket", "protein", "encoder_signature", "pocket_encoder_signature",
               "properties", "fragment", "ifp", "conditions", "reference_smiles", "complex_pdb",
               "compute_properties", "extract_fragment", "compute_ifp", "fragment_representation"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError("Feature spec must be an object containing only documented input fields")
    if "pocket_encoder_signature" in spec:
        if "encoder_signature" in spec:
            raise ValueError("Duplicate pocket encoder signature fields")
        spec["encoder_signature"] = spec.pop("pocket_encoder_signature")
    if isinstance(spec.get("pocket"), dict):
        pocket = spec.pop("pocket")
        if set(pocket) - {"pocket_pdb", "protein_pdb"} or not pocket.get("pocket_pdb"):
            raise ValueError("pocket must contain pocket_pdb and optionally protein_pdb")
        spec["pocket"] = pocket["pocket_pdb"]
        if "protein_pdb" in pocket:
            if "protein" in spec:
                raise ValueError("Duplicate protein paths")
            spec["protein"] = pocket["protein_pdb"]
    for name in ("pocket", "protein", "complex_pdb"):
        if spec.get(name):
            spec[name] = str((source.parent / spec[name]).resolve())
    if fragment_representation is not None:
        if "fragment_representation" in spec and spec["fragment_representation"] != fragment_representation:
            raise ValueError("Fragment representation differs between CLI and spec")
        spec["fragment_representation"] = fragment_representation
    if encoder_file and not resolve_pocket:
        raise ValueError("--encoder-config here requires --resolve-pocket; generation can resolve raw pockets later")
    if resolve_pocket and not encoder_file:
        raise ValueError("--resolve-pocket requires --encoder-config")
    resolver = PocketResolver(json.loads(Path(encoder_file).read_text())) if resolve_pocket else None
    result = prepare_inference_request(**spec, resolver=resolver, resolve_pocket=resolve_pocket)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result["request"], stream, indent=2, allow_nan=False)
        stream.write("\n")
    with metadata.open("x") as stream:
        json.dump(result["provenance"], stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"request": str(output), "provenance": str(metadata), "conditions": result["conditions"]}


def main():
    parser = argparse.ArgumentParser(prog="pocketgpt")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Split a local manifest or migrate an explicitly known legacy CSV")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    prepare.add_argument("--legacy-csv", action="store_true")
    prepare.add_argument("--protein-map")
    prepare.add_argument("--ifp-schema", help="Local JSON mapping six interaction names to confirmed legacy 8-bit indices")
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--skip-invalid", action="store_true")
    prepare.add_argument("--split-group", choices=["protein_id", "complex_id"], default="protein_id")
    prepare.add_argument("--fragment-representation", choices=["canonical-smiles-v1", "raw-smiles-v1"],
                         default="canonical-smiles-v1")
    conditional_prepare = sub.add_parser(
        "prepare-condition-pretrain",
        help="Stream a v2 smiles/properties/single-frag CSV, preserving all rows and grouping by canonical parent")
    conditional_prepare.add_argument("--input", required=True)
    conditional_prepare.add_argument("--output-dir", required=True)
    conditional_prepare.add_argument("--validation-fraction", type=float, default=0.1)
    conditional_prepare.add_argument("--block-size", type=int, default=256)
    conditional_prepare.add_argument("--properties", default=",".join(PROPERTIES),
                                     help="Comma-separated logp,qed,tpsa,sas subset; LogS is metadata only")
    conditional_prepare.add_argument("--conditions", default="properties,fragment",
                                     help="Nonempty subset of properties,fragment; labels are retained regardless")
    conditional_prepare.add_argument("--fragment-representation", choices=["canonical-smiles-v1", "raw-smiles-v1"],
                                     default="canonical-smiles-v1")
    feature = sub.add_parser("prepare-request", help="Assemble local inference features without retraining the pocket encoder")
    feature.add_argument("--spec", required=True, help="Local JSON; direct labels or reference-molecule property calculation")
    feature.add_argument("--output", required=True, help="New standard request JSON; a provenance sidecar is also written")
    feature.add_argument("--encoder-config")
    feature.add_argument("--resolve-pocket", action="store_true", help="Encode now and cache, rather than at generation time")
    feature.add_argument("--fragment-representation", choices=["canonical-smiles-v1", "raw-smiles-v1"])
    merge = sub.add_parser("merge-conditions", help="Join a single-frag CSV to local pocket/property/IFP records by exact file ID")
    merge.add_argument("--base-csv", required=True)
    merge.add_argument("--fragment-csv", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--progress", action="store_true", help="Print local row-count progress")
    export_ifp = sub.add_parser("export-ifp-targets", help="Export only exact complex IDs, parent hashes and six IFP targets")
    export_ifp.add_argument("--base-csv", required=True)
    export_ifp.add_argument("--output", required=True)
    attach_ifp = sub.add_parser("attach-ifp-targets", help="Attach six named IFP targets to an existing cached-feature CSV")
    attach_ifp.add_argument("--cached-csv", required=True)
    attach_ifp.add_argument("--ifp-table", required=True)
    attach_ifp.add_argument("--output", required=True)
    attach_ifp.add_argument("--expected-cached-sha256")
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--resume")
    train.add_argument("--max-steps", type=int)
    train.add_argument("--device")
    gen = sub.add_parser("generate")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--request", help="Local JSON containing pocket/properties/fragment/ifp")
    gen.add_argument("--pocket", help="Extracted pocket PDB")
    gen.add_argument("--protein", help="Optional full protein PDB providing the original structural context")
    gen.add_argument("--encoder-config", help="Local DeeplyTough repo/python/weights/cache JSON")
    gen.add_argument("--property", action="append", default=[], help="NAME=VALUE, repeatable")
    gen.add_argument("--fragment")
    gen.add_argument("--ifp", help="JSON list of six binary interaction presences, in documented order")
    gen.add_argument("--conditions", default=None, help="Explicit comma-separated subset; default: inputs supplied")
    gen.add_argument("--count", type=int, default=100)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=20)
    gen.add_argument("--max-tokens", type=int)
    gen.add_argument("--batch-size", type=int, default=32)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--device", default="cpu")
    gen.add_argument("--output", required=True)
    audit = sub.add_parser("inspect-checkpoint")
    audit.add_argument("path")
    args = parser.parse_args()
    if args.command == "prepare":
        from .preparation import prepare_dataset
        result = prepare_dataset(args.input, args.output_dir, args.stage, args.legacy_csv,
                                 args.protein_map, args.validation_fraction, args.limit, args.skip_invalid, args.ifp_schema,
                                 split_group=args.split_group, fragment_representation=args.fragment_representation)
    elif args.command == "prepare-condition-pretrain":
        from .conditional_preparation import prepare_condition_pretrain
        result = prepare_condition_pretrain(
            args.input, args.output_dir, args.validation_fraction, args.block_size,
            tuple(value.strip() for value in args.properties.split(",") if value.strip()),
            tuple(value.strip() for value in args.conditions.split(",") if value.strip()),
            fragment_representation=args.fragment_representation)
    elif args.command == "prepare-request":
        result = prepare_request_files(args.spec, args.output, args.encoder_config,
                                       args.resolve_pocket, args.fragment_representation)
    elif args.command == "merge-conditions":
        from .merging import merge_condition_csvs
        progress = (lambda value: print(json.dumps(value), flush=True)) if args.progress else None
        result = merge_condition_csvs(args.base_csv, args.fragment_csv, args.output_dir, progress=progress)
    elif args.command == "export-ifp-targets":
        from .cached_conditions import export_ifp_targets
        result = export_ifp_targets(args.base_csv, args.output)
    elif args.command == "attach-ifp-targets":
        from .cached_conditions import attach_ifp_targets
        result = attach_ifp_targets(args.cached_csv, args.ifp_table, args.output,
                                    expected_cached_sha256=args.expected_cached_sha256)
    elif args.command == "train":
        from .training import train
        result = {"checkpoint": train(args.config, args.resume, args.max_steps, args.device)}
    elif args.command == "generate":
        from .generation import generate
        request = json.loads(Path(args.request).read_text()) if args.request else {}
        if args.pocket:
            if "pocket_embedding" in request:
                raise ValueError("Pocket input conflict: --pocket cannot be combined with a request pocket_embedding")
            request["pocket"] = {"pocket_pdb": str(Path(args.pocket).resolve())}
            if args.protein:
                request["pocket"]["protein_pdb"] = str(Path(args.protein).resolve())
        elif args.protein:
            raise ValueError("--protein requires --pocket")
        from .pocket import validate_pocket_sources
        validate_pocket_sources(request)
        if args.property:
            props = dict(request.get("properties", {}))
            for value in args.property:
                key, number = value.split("=", 1)
                if key in props:
                    raise ValueError("Duplicate property: " + key)
                props[key] = float(number)
            request["properties"] = props
        if args.fragment:
            request["fragment"] = args.fragment
        if args.ifp:
            request["ifp"] = json.loads(args.ifp)
        selected = args.conditions
        if selected is None:
            selected = ",".join(name for name in CONDITIONS if
                                (name == "pocket" and ("pocket_embedding" in request or request.get("pocket"))) or
                                (name != "pocket" and name in request)) or "none"
        encoder = json.loads(Path(args.encoder_config).read_text()) if args.encoder_config else None
        result = generate(args.checkpoint, request, args.output, selected, args.count,
                          args.temperature, args.top_k, args.max_tokens, args.batch_size, args.seed, args.device, encoder)
    else:
        import torch
        state = torch.load(args.path, map_location="cpu", weights_only=True, mmap=True)
        if state.get("format_version") == 1:
            result = {key: state.get(key) for key in ("model_config", "global_step", "epoch", "next_batch",
                                                     "stage", "trained_conditions", "trained_properties",
                                                     "stage_trained_conditions", "stage_trained_properties",
                                                     "inherited_trained_conditions", "inherited_trained_properties",
                                                     "warm_start", "pocket_encoder_signature")}
        else:
            result = {"kind": "legacy-tensor-state", "exact_resume_available": False,
                      "shapes": {key: list(state[key].shape) for key in ("tok_emb.weight", "pos_emb", "prop_nn.weight", "scaff_mlp.0.weight") if key in state}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
