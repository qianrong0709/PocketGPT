# Refactored package: installation and usage

[Back to PocketGPT](../README.md) · [Input contracts](INPUTS.md) · [Release status](RELEASE.md)

The root `pocketgpt/` package is the refactored implementation. The older `src/` scripts remain separate and unchanged. Do not mix their vocabularies, checkpoints, or command-line arguments. New trained checkpoints and research datasets are **not included**; the commands below require your own compatible local inputs and, for generation, a trained checkpoint.

## Install and test

Use Python 3.10 or newer in a separate environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pocketgpt --help
python -m unittest discover -s tests -v
```

`pyproject.toml` declares generator dependency ranges. `requirements-runtime.txt` records versions present in the inspected development runtime; it is not a complete cross-platform lockfile, does not select a CUDA build, and does not install the legacy pocket frontend. Select a PyTorch build suitable for your machine before training. With dependencies already installed, `python -m pip install -e . --no-deps --no-build-isolation` avoids dependency resolution; it still requires a suitable installed build backend.

Tests use temporary synthetic fixtures. Passing them does not demonstrate model convergence, molecular control quality, reproduction of paper metrics, or compatibility of newly encoded PDBs with historical vectors.

## 1. Property-plus-single-fragment pretraining

The input CSV needs `smiles`, `logp`, `qed`, `tpsa`, `sas`, and `frag` columns (column names are normalized case-insensitively). Each row contains a parent molecule and **one** fragment. Pocket vectors and IFP are not required or used at this stage. Extra source fields, such as LogS, remain metadata rather than becoming model conditions.

From the repository root:

```bash
python -m pocketgpt prepare-condition-pretrain \
  --input /absolute/path/molecules-with-single-fragment.csv \
  --output-dir data/condition-pretrain-v2-raw \
  --conditions properties,fragment --properties logp,qed,tpsa,sas \
  --fragment-representation raw-smiles-v1 \
  --block-size 256 --validation-fraction 0.1

python -m pocketgpt train --config configs/condition_pretrain.json --device cuda
```

The recipe expects the newly prepared files and vocabulary at those relative paths. Recipe paths resolve relative to the configuration file, not an arbitrary working directory. Use a new output directory or explicitly resume an existing compatible run.

Preparation groups canonical parent molecules before splitting; it does not sample, truncate, or deduplicate fragment-expanded rows. Explicit `raw-smiles-v1` preserves fragment text for tokenization, including chemically invalid fragment labels, and records chemical issues without certifying or repairing them. Parent-molecule, property, tokenizer, and context checks remain strict. `canonical-smiles-v1` is a separate policy and must not be silently substituted for a raw-mode checkpoint.

**Filesystem limitation:** corpus publication uses an atomic, no-replace directory operation. Some shared filesystems do not support it. A publication error can leave completed files and diagnostic reports in a hidden `.pending-*` directory; do not delete those files or blindly restart a large conversion. Use a supported local filesystem or obtain a separately verified, no-overwrite recovery. This release does not weaken the publication safeguard.

## 2. Fine-tuning with existing pocket vectors

You do not need to recompute every pocket or retrain DeeplyTough. A native cached-feature JSONL record contains parent `smiles`, `complex_id`, a finite 128-number `pocket_embedding`, and observed `properties`, `fragment`, and six-type `ifp` labels. Preserve encoder provenance when known. Missing labels are absent, not invented zeros; declare `fragment_representation: "raw-smiles-v1"` when using that policy.

```bash
python -m pocketgpt prepare \
  --input /absolute/path/cached-conditions.jsonl \
  --output-dir data/cached-complex-raw \
  --stage finetune --split-group complex_id \
  --fragment-representation raw-smiles-v1 --validation-fraction 0.1

python -m pocketgpt train --config configs/finetune.cached.json --device cuda
```

The example recipe initializes from `runs/condition-pretrain-v2-raw/best.pt`, enabling all four conditions. It deliberately opts into **unverified historical embedding provenance** through `allow_unverified_pocket_embeddings`; this enables a clearly labelled legacy-vector experiment, not proof of encoder identity. Do not use that option to make unrelated vectors interchangeable. New verified-encoder workflows should use a provenance-matched configuration instead.

Complex-grouped validation is not protein-held-out evaluation. A protein/sequence mapping and appropriate split are required for unseen-protein claims. The legacy CSV adapter recognizes a specific documented `evalue`/`Unnamed` layout, not arbitrary CSVs; prefer native records when preparing a new dataset. Six-label IFP export and attachment helpers are available through `python -m pocketgpt --help` and preserve explicit IDs and parent-molecule checks.

## 3. Submit training through Slurm

Do not run full GPU training on a shared login node. The portable wrapper has no hardcoded partition, node, user directory, or environment location. Export the absolute project directory and environment interpreter, then select resources according to your cluster policy:

```bash
export POCKETGPT_PROJECT=/absolute/path/PocketGPT
export POCKETGPT_PYTHON=/absolute/path/generator-environment/bin/python

sbatch --export=ALL --partition=YOUR_GPU_PARTITION \
  --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=24:00:00 \
  "$POCKETGPT_PROJECT/scripts/train.slurm" \
  "$POCKETGPT_PROJECT/configs/condition_pretrain.json"
```

Resource values above are examples, not a guarantee that your cluster can schedule them. Allocate at least the recipe's `cpu_threads`. The wrapper requires an allocation, checks the full input/configuration contract, runs regression tests, and only then starts GPU training. Full-data preflight can take time before the first loss is logged. Logs use Slurm's configured/default output location. It does not install software, run a research pocket encoder, submit other jobs, or upload telemetry.

For exact resume, append `--resume /absolute/path/to/the/same-run/last.pt`; for a bounded technical run, append `--max-steps 10`. These flags do not skip preflight. Do not submit multiple jobs writing the same run directory. Recipe, vocabulary, data, implementation, and runtime identity must match for exact resume; changing them requires a separately identified new warm-start run, not silently reusing optimizer state.

## 4. Generate with a trained compatible checkpoint

**An inference user does not need a training JSONL file.** Conditions can be passed directly. For a trained property/fragment checkpoint:

```bash
python -m pocketgpt generate \
  --checkpoint /absolute/path/to/compatible-checkpoint.pt \
  --property qed=0.7 --property logp=2.5 --fragment 'c1ccccc1' \
  --conditions properties,fragment --count 100 \
  --output runs/generated/property-fragment.csv
```

Only request conditions/properties that the checkpoint has learned. Raw pocket input additionally requires the separately installed original DeeplyTough frontend, compatible weights and preprocessing tools, and an edited local copy of `configs/encoder.example.json`:

```bash
python -m pocketgpt generate \
  --checkpoint /absolute/path/to/compatible-pocket-checkpoint.pt \
  --protein /absolute/path/protein.pdb --pocket /absolute/path/pocket.pdb \
  --encoder-config /absolute/path/encoder.local.json \
  --property qed=0.7 --fragment 'c1ccccc1' \
  --ifp '[1,1,0,0,0,0]' --conditions all --count 100 \
  --output runs/generated/four-conditions.csv
```

The array order is hydrogen bond, hydrophobic contact, π-stacking, salt bridge, π-cation interaction, halogen bond. The protein and pocket must correspond; protein alone does not select a binding site. Properties and IFP targets are user constraints, not quantities inferred from protein alone.

The raw-pocket command documents the implemented interface, **not a verified historical-vector-to-new-PDB release guarantee**. DeeplyTough remains frozen; generator training learns its pocket projector rather than backpropagating into that encoder. Historical cached embeddings and newly calculated embeddings must not be mixed until their numerical/provenance compatibility has been established. The current safeguards may intentionally reject such a request. Third-party code, weights and preprocessing tools are not bundled and retain their own installation/licensing terms.

Only load weights from a trusted source. The legacy DeeplyTough frontend uses an older PyTorch checkpoint deserializer, which may execute code embedded in an untrusted checkpoint; do not pass arbitrary downloaded weight files to it.

Generated CSVs retain sampled SMILES, including invalid samples, with available validity/property/fragment annotations. Learned constraints are not hard guarantees; IFP recovery and affinity need independent pose/interaction or experimental evaluation. New model weights and validated real-structure demonstrations will be announced separately after training and evaluation.
