"""Generate molecules with the model included in this repository."""
import argparse
import json
from pathlib import Path

from pocketgpt.generation import generate
from pocketgpt.config import parse_conditions
from pocketgpt.pocket import validate_pocket_sources
from pocketgpt.pocket_inference import load_pocket_vector


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Generate molecules using selected conditions locally")
    parser.add_argument("--checkpoint", default=str(REPOSITORY_ROOT / "models" / "pocketgpt-v1.pt"),
                        help="Model path (default: models/pocketgpt-v1.pt in this repository)")
    parser.add_argument("--request", default=str(REPOSITORY_ROOT / "examples" / "property_fragment.json"))
    parser.add_argument("--pocket-vector", help="Pocket descriptor JSON or cached .npy")
    parser.add_argument("--pocket", help="Extracted pocket PDB")
    parser.add_argument("--protein", help="Full protein PDB corresponding to --pocket")
    parser.add_argument("--encoder-config", help="Local DeeplyTough encoder configuration JSON")
    parser.add_argument("--pocket-compatibility", help="Paired-vector comparison record")
    parser.add_argument("--allow-unverified-pocket", action="store_true")
    parser.add_argument("--conditions", help="Selected comma-separated conditions; default: inputs supplied")
    parser.add_argument("--ifp", help="Optional JSON containing the six named binary interaction targets")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/example.csv")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    if args.pocket_vector:
        if request.get("pocket") or "pocket_embedding" in request or args.pocket:
            raise ValueError("Supply only one pocket input")
        request.update(load_pocket_vector(args.pocket_vector))
    if args.pocket:
        if request.get("pocket") or "pocket_embedding" in request:
            raise ValueError("Supply only one pocket input")
        request["pocket"] = {"pocket_pdb": str(Path(args.pocket).resolve())}
        if args.protein:
            request["pocket"]["protein_pdb"] = str(Path(args.protein).resolve())
    elif args.protein:
        raise ValueError("--protein requires --pocket")
    validate_pocket_sources(request)
    if args.ifp:
        if "ifp" in request:
            raise ValueError("Supply only one IFP input")
        request["ifp"] = json.loads(Path(args.ifp).read_text())
    selected = [name for name in ("pocket", "properties", "fragment", "ifp") if
                (name == "pocket" and ("pocket_embedding" in request or request.get("pocket"))) or
                (name != "pocket" and name in request)]
    if args.conditions is not None:
        selected = list(parse_conditions(args.conditions))
    encoder = (json.loads(Path(args.encoder_config).read_text())
               if args.encoder_config and "pocket" in selected else None)
    result = generate(args.checkpoint, request, args.output, conditions=selected,
                      count=args.count, seed=args.seed, device=args.device, encoder=encoder,
                      pocket_compatibility=args.pocket_compatibility,
                      allow_unverified_pocket=args.allow_unverified_pocket)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
