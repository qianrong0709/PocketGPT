# Training and advanced usage

[Back to PocketGPT](../README.md) · [Installation](INSTALLATION.md) · [Generation examples](GENERATION.md) · [Input contracts](INPUTS.md)

The root `pocketgpt/` package is the implementation on `main`. Training commands below use your own local inputs. For generation with released weights, start with the [quick start](../README.md#installation-and-model-usage).

## Install and test

Use Python 3.10 or newer in a separate environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pocketgpt --help
python -m unittest discover -s tests -v
```

`pyproject.toml` defines dependency ranges; `requirements-runtime.txt` lists tested versions. Select a PyTorch build for your device using the [installation guide](INSTALLATION.md). Tests check the bundled model's checksum and use synthetic fixtures for training checks. With dependencies and the build backend already installed, `python -m pip install -e . --no-deps --no-build-isolation` installs the package without resolving dependencies again.

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

The recipe uses the prepared files and vocabulary in the directory above. Paths in a configuration file resolve relative to that file. Use a new output directory or resume an existing run with `--resume`.

Preparation retains all fragment-expanded rows and groups them by canonical parent molecule before splitting. `raw-smiles-v1` preserves fragment strings, including chemically invalid labels, and records parsing issues. Parent-molecule, property, token, and context checks remain strict. `canonical-smiles-v1` is a separate representation; use the representation expected by your checkpoint.

**Shared filesystems:** preparation finalizes data with an atomic directory operation that some shared filesystems do not support. If finalization fails, completed files and reports may remain in a hidden `.pending-*` directory. Preserve that directory and check the error before restarting preparation. Use a filesystem that supports the operation, or recover and verify the completed files separately.

## 2. Fine-tuning with existing pocket vectors

Fine-tuning uses cached descriptors directly, without rerunning DeeplyTough. Each JSONL record contains parent `smiles`, `complex_id`, a finite 128-number `pocket_embedding`, and available `properties`, `fragment`, and six-type `ifp` labels. Record the encoder signature and omit missing labels. Set `fragment_representation: "raw-smiles-v1"` for raw fragment strings.

```bash
python -m pocketgpt prepare \
  --input /absolute/path/cached-conditions.jsonl \
  --output-dir data/cached-complex-raw \
  --stage finetune --split-group complex_id \
  --fragment-representation raw-smiles-v1 --validation-fraction 0.1

python -m pocketgpt train --config configs/finetune.cached.json --device cuda
```

For frozen-backbone fine-tuning with early stopping, use `configs/finetune.frozen.json` and a new run directory. See [fine-tuning details](FINETUNING.md).

The recipe initializes from `runs/condition-pretrain-v2-raw/best.pt` and enables all four conditions. It sets `allow_unverified_pocket_embeddings` for the historical cached descriptors. For a new dataset, use one consistent encoder and record its signature; see [pocket compatibility](MODEL_CARD.md#pocket-compatibility).

The split is grouped by complex ID, not by protein identity. For protein-held-out validation, supply an appropriate protein or sequence-cluster grouping. Use native JSONL records for new datasets; the legacy CSV adapter expects the specific `evalue`/`Unnamed` layout. IFP export and attachment commands are listed in `python -m pocketgpt --help` and match records by complex ID and parent-molecule checks.

## 3. Submit training through Slurm

Submit training to a GPU compute node. Set the project directory and Python interpreter, then choose resources available on your cluster:

```bash
export POCKETGPT_PROJECT=/absolute/path/PocketGPT
export POCKETGPT_PYTHON=/absolute/path/generator-environment/bin/python

sbatch --export=ALL --partition=YOUR_GPU_PARTITION \
  --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=24:00:00 \
  "$POCKETGPT_PROJECT/scripts/train.slurm" \
  "$POCKETGPT_PROJECT/configs/condition_pretrain.json"
```

Adjust the partition, memory, and time limit to your cluster. Allocate at least the configured `cpu_threads`. The wrapper validates inputs and configuration and runs tests before GPU training, so a large dataset may take time to reach the first logged loss. Logs use Slurm's configured output location.

To resume, append `--resume /absolute/path/to/the/same-run/last.pt`; for a short test run, append `--max-steps 10`. Both run the input checks. Use only one job per run directory. Exact resume requires matching configuration, vocabulary, data, implementation, and runtime; changed settings require a new warm-start run.

## 4. Generate with a trained compatible checkpoint

For the included model, pass condition values directly:

```bash
python -m pocketgpt generate \
  --checkpoint models/pocketgpt-v1.pt \
  --property qed=0.7 --property logp=2.5 --fragment 'c1ccccc1' \
  --conditions properties,fragment --count 100 \
  --output runs/generated/property-fragment.csv
```

For a custom checkpoint, replace `--checkpoint` with its path and select conditions used during training. Raw PDB input requires a checkpoint matched to the separately installed DeeplyTough encoder and preprocessing pipeline. The included `pocketgpt-v1.pt` has not been verified with newly encoded vectors. With a matched checkpoint and a local copy of `configs/encoder.example.json`, the interface is:

```bash
python -m pocketgpt generate \
  --checkpoint /absolute/path/to/compatible-pocket-checkpoint.pt \
  --protein /absolute/path/protein.pdb --pocket /absolute/path/pocket.pdb \
  --encoder-config /absolute/path/encoder.local.json \
  --property qed=0.7 --fragment 'c1ccccc1' \
  --ifp '[1,1,0,0,0,0]' --conditions all --count 100 \
  --output runs/generated/four-conditions.csv
```

The IFP array order is hydrogen bond, hydrophobic contact, π-stacking, salt bridge, π-cation interaction, and halogen bond. Supply a protein and its corresponding pocket. Property and IFP targets are entered by the user; observed IFP requires a protein–ligand pose. DeeplyTough remains frozen. See [pocket compatibility](MODEL_CARD.md#pocket-compatibility) and [encoder setup](INSTALLATION.md#optional-pocket-encoding-environment).

The CSV retains sampled SMILES, including invalid samples, with validity, property, and fragment-match fields. See [generation outputs](GENERATION.md#output-and-sampling-settings).
