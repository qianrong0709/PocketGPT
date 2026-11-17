# PocketGPT

### Structure–function conditioned molecular generation

Code for **Structure–function conditioned molecular generation with PocketGPT enables prospective discovery of PARP1 inhibitors**.

Rong Qian, Jilong Duan, Xiaoping Liu, Wenxuan Wang, Tingfei Zhu, Dejun Jiang, Jin Liu, Youchao Deng, Aiping Lyu, and Dongsheng Cao. *The Innovation Drug Discovery* **1**(2), 100028 (2026).

[Paper](https://doi.org/10.59717/j.xinn-drugdisc.2026.100028) · [Project page](https://qianrong0709.github.io/PocketGPT_Page/) · [中文说明](docs/README.zh-CN.md)

PocketGPT generates molecular SMILES conditioned on protein pockets, molecular properties, fragments, and protein–ligand interaction fingerprints. It combines structural information with medicinal-chemistry targets in an autoregressive molecular language model, without explicitly constructing 3D ligand coordinates.

## Method

PocketGPT combines four sources of conditioning:

| Condition | Information supplied |
| --- | --- |
| Protein pocket | A representation of the selected binding pocket |
| Molecular properties | Targets for LogP, QED, TPSA, and synthetic accessibility score (SAS) |
| Single fragment | A molecular fragment expressed as SMILES |
| Interaction fingerprint (IFP) | Desired protein–ligand interaction types |

Each active condition is encoded and projected into a shared space. A masked, learnable fusion module produces a semantic prefix for the causal SMILES decoder. Conditions can be enabled individually or in combinations.

```text
Pocket representation ── projector ───────────────┐
Property targets ──────── projector ───────────────┤
Fragment SMILES ───────── sequence encoder ────────┼─ fusion ─ semantic prefix ─ SMILES decoder
Interaction fingerprint ─ projector ───────────────┘
```

Pocket encoding uses a frozen DeeplyTough frontend. It produces a 128-dimensional vector that is passed to the model's learned pocket projector. You can supply a stored vector or encode a pocket PDB during generation. The corresponding full protein PDB can also be supplied to retain the surrounding structural context.

## Prospective discovery

The paper studies joint structural and functional conditioning and reports a prospective PARP1 design–make–test campaign, including an inhibitor with a measured binding affinity of **K<sub>d</sub> = 4.88 nM** and favorable preliminary tolerability in mice. See the [paper](https://doi.org/10.59717/j.xinn-drugdisc.2026.100028) for the experimental setup and evaluation.

## Installation and model usage

The trained generator is included at [`models/pocketgpt-v1.pt`](models/pocketgpt-v1.pt). Generating from a new pocket also requires the DeeplyTough encoder and its preprocessing tools. The Linux setup below installs these in separate environments; no training data or retraining is needed.

### 1. Install on Linux

Use **Linux x86_64** with **glibc 2.28 or newer**, Conda, Git, a C/C++ compiler, Make, and zlib development headers. See [prerequisites](docs/INSTALLATION.md#linux-prerequisites) if these are not installed. No GPU is needed for this example.

The real-PDB workflow below was tested on Ubuntu 22.04 with CPU execution, using both a protein/pocket pair and a pocket PDB alone.

```bash
git clone --depth 1 https://github.com/qianrong0709/PocketGPT.git
cd PocketGPT
export PG_RUNTIME="$PWD/.pocketgpt-runtime"
```

Read the [dependency licenses](docs/INSTALLATION.md#licenses), including the terms bundled with MGLTools 1.5.6. Pass the acceptance flags only after reading and accepting the applicable terms:

```bash
bash scripts/setup_linux.sh --prefix "$PG_RUNTIME" \
  --accept-htmd-license --accept-mgltools-license
export PG_PYTHON="$PG_RUNTIME/envs/generator/bin/python"
"$PG_PYTHON" scripts/download_model.py --verify-only
```

The installer writes `encoder.local.json` with the encoder paths and cache location. Keep the installation directory in place. If a download is interrupted, rerun the installation command with `--resume`.

### 2. Generate from your pocket PDB

Replace `/absolute/path/pocket.pdb` with the path to your pocket file:

```bash
"$PG_PYTHON" -m pocketgpt generate \
  --checkpoint models/pocketgpt-v1.pt \
  --pocket /absolute/path/pocket.pdb \
  --encoder-config "$PG_RUNTIME/encoder.local.json" \
  --allow-unverified-pocket \
  --conditions pocket --count 10 --seed 42 --device cpu \
  --output outputs/pocket.csv
```

The pipeline calculates the 128-dimensional vector, caches it, and passes it to the generator. You do not need to create a CSV, JSON, or JSONL input file. `outputs/pocket.csv` contains the generated SMILES and calculated properties; `outputs/pocket.summary.json` contains sampling statistics.

The included model was trained with cached descriptors. `--allow-unverified-pocket` permits new descriptors while recording that their numerical compatibility with those training descriptors has not yet been established. It does not mean encoder weights are missing. See [pocket compatibility](docs/MODEL_CARD.md#pocket-compatibility).

### 3. Add protein context or other conditions

The command above uses only the atoms in the pocket file. If you also have the corresponding protein PDB, add `--protein /absolute/path/protein.pdb` to retain surrounding structural context. The two files must use the same coordinate frame. The pocket must contain protein atoms, not dummy points marking a cavity. Use a new `--output` filename for each run.

Add property, fragment, or IFP targets to combine conditions; see [four-condition generation](docs/GENERATION.md#four-condition-generation).

### Generate without a pocket

For property/fragment generation, DeeplyTough is not needed. You can use the generator environment above, or follow the [generator-only installation](docs/INSTALLATION.md#generator-only-installation).

```bash
"$PG_PYTHON" examples/generate.py --count 10 --output outputs/example.csv
```

This example requests **QED = 0.7**, **LogP = 2.5**, and a **benzene fragment**, using CPU. Equivalent command:

```bash
"$PG_PYTHON" -m pocketgpt generate \
  --checkpoint models/pocketgpt-v1.pt \
  --property qed=0.7 --property logp=2.5 \
  --fragment 'c1ccccc1' --conditions properties,fragment \
  --count 10 --seed 42 --device cpu \
  --output outputs/property-fragment.csv
```

The CSV retains attempted samples, including invalid SMILES, and reports validity, canonical SMILES, calculated properties, and fragment matches. GPU generation requires a [CUDA PyTorch installation](docs/INSTALLATION.md#nvidia-gpu-for-the-generator) and `--device cuda`.

## Training

Training has two stages: **SMILES with property and single-fragment conditioning**, followed by **pocket/property/fragment/IFP fine-tuning**. Cached pocket descriptors can be used directly without recomputing every structure or retraining DeeplyTough. The frozen-backbone recipe includes validation-based early stopping.

See [training commands](docs/USAGE.md), [input schemas](docs/INPUTS.md), and [frozen fine-tuning](docs/FINETUNING.md).

## Repository layout

```text
PocketGPT/
├── README.md
├── CITATION.cff
├── pyproject.toml
├── requirements-runtime.txt # Tested generator versions
├── pocketgpt/             # Generator, data handling, and feature adapters
├── configs/               # Portable configuration examples
├── environments/          # Pinned encoder dependencies
├── scripts/               # Preparation/validation tools and configurable Slurm wrapper
├── tests/                 # Regression tests
├── examples/              # Inference requests and Python example
├── models/
│   ├── pocketgpt-v1.pt     # Trained model, configuration, and vocabulary
│   ├── pocketgpt-v1.json   # Model metadata
│   └── SHA256SUMS         # Weight checksum
└── docs/
    ├── INPUTS.md          # Training and inference input contracts
    ├── INSTALLATION.md    # CPU/CUDA environment setup
    ├── GENERATION.md      # CLI/Python examples and condition combinations
    ├── MODEL_CARD.md      # Model specification
    ├── USAGE.md           # Training and resume commands
    └── README.zh-CN.md    # Chinese overview
```

## Citation

If you use PocketGPT, please cite:

```bibtex
@article{qian2026pocketgpt,
  title   = {Structure--function conditioned molecular generation with PocketGPT enables prospective discovery of PARP1 inhibitors},
  author  = {Qian, Rong and Duan, Jilong and Liu, Xiaoping and Wang, Wenxuan and Zhu, Tingfei and Jiang, Dejun and Liu, Jin and Deng, Youchao and Lyu, Aiping and Cao, Dongsheng},
  journal = {The Innovation Drug Discovery},
  volume  = {1},
  number  = {2},
  pages   = {100028},
  year    = {2026},
  doi     = {10.59717/j.xinn-drugdisc.2026.100028}
}
```

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## Dependencies

Pocket encoding builds on [DeeplyTough](https://github.com/BenevolentAI/DeeplyTough). The generator uses PyTorch, NumPy, and RDKit. Pocket-preprocessing tools require a separate environment. Third-party dependencies remain subject to their respective licenses.

## Contact

For research questions, contact Rong Qian at [24483877@life.hkbu.edu.hk](mailto:24483877@life.hkbu.edu.hk). For usage questions, open a repository issue.
