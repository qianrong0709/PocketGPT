"""Local one-step compatibility check using explicit, existing user files.

This writes validation artifacts only to --output-dir, does not contact services,
and does not claim a fully trained base model.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocketgpt.config import PROPERTIES
from pocketgpt.tokenizer import SPECIAL, SmilesTokenizer
from pocketgpt.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--legacy-vocabulary", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(args.legacy_vocabulary).read_text())
    vocabulary = [*SPECIAL, *sorted(set(source) - {"<"})]
    SmilesTokenizer(vocabulary).save(output / "vocabulary.json")
    data = Path(args.prepared_dir).resolve()
    recipe = {
        "stage": "pretrain", "conditions": "none",
        "train_data": str(data / "train.jsonl"), "valid_data": str(data / "valid.jsonl"),
        "output_dir": str(output / "run"), "vocabulary": str(output / "vocabulary.json"),
        "legacy_checkpoint": str(Path(args.checkpoint).resolve()),
        "legacy_vocabulary": str(Path(args.legacy_vocabulary).resolve()),
        "model": {"block_size": 256, "n_layer": 8, "n_head": 8, "n_embd": 256, "dropout": 0.1},
        "training": {"epochs": 2, "batch_size": 2, "learning_rate": 0.0001, "seed": 42,
                     "cpu_threads": 2, "device": "cpu", "save_every_steps": 100},
    }
    path = output / "recipe.json"
    path.write_text(json.dumps(recipe, indent=2) + "\n")
    checkpoint = train(path, max_steps=1)
    report = {"status": "one_real_data_step_passed", "checkpoint": checkpoint,
              "not_a_full_training_run": True, "raw_pocket_encoder_tested": False}
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
