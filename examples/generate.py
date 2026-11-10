"""Run from the repository root after installation and model download."""
import argparse
import json
from pathlib import Path

from pocketgpt.generation import generate
from pocketgpt.pocket import validate_embedding


def main():
    parser = argparse.ArgumentParser(description="Generate molecules using selected conditions locally")
    parser.add_argument("--checkpoint", default="weights/pocketgpt-v1.pt")
    parser.add_argument("--request", default="examples/property_fragment.json")
    parser.add_argument("--pocket-vector", help="Your own compatible vector JSON: a list of 128 finite numbers")
    parser.add_argument("--ifp", help="Optional JSON containing the six named binary interaction targets")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/example.csv")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    if args.pocket_vector:
        if request.get("pocket") or "pocket_embedding" in request:
            raise ValueError("Supply only one pocket input")
        request["pocket_embedding"] = validate_embedding(json.loads(Path(args.pocket_vector).read_text())).tolist()
        # Historical-vector interface, not proof that arbitrary encoders match.
        request["pocket_encoder_signature"] = "legacy-precomputed-unverified"
    if args.ifp:
        if "ifp" in request:
            raise ValueError("Supply only one IFP input")
        request["ifp"] = json.loads(Path(args.ifp).read_text())
    selected = [name for name in ("pocket", "properties", "fragment", "ifp") if
                (name == "pocket" and ("pocket_embedding" in request or request.get("pocket"))) or
                (name != "pocket" and name in request)]
    result = generate(args.checkpoint, request, args.output, conditions=selected,
                      count=args.count, seed=args.seed, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
