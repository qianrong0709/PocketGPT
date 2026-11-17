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

Pocket encoding uses a frozen DeeplyTough frontend. It produces a 128-dimensional vector that is passed to the model's learned pocket projector. You can supply a stored vector or encode a protein/pocket PDB pair during generation; see [pocket generation](docs/GENERATION.md#generate-from-a-protein-and-pocket).

## Prospective discovery

The paper studies joint structural and functional conditioning and reports a prospective PARP1 design–make–test campaign, including an inhibitor with a measured binding affinity of **K<sub>d</sub> = 4.88 nM** and favorable preliminary tolerability in mice. See the [paper](https://doi.org/10.59717/j.xinn-drugdisc.2026.100028) for the experimental setup and evaluation.

## Installation and model usage

The generator runs locally on CPU or an NVIDIA GPU. The [installation guide](docs/INSTALLATION.md) covers both environments; the [generation guide](docs/GENERATION.md) explains condition combinations and output fields.

### 1. Configure the environment

```bash
git clone https://github.com/qianrong0709/PocketGPT.git
cd PocketGPT
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 numpy==2.2.6 rdkit==2026.3.1
python -m pip install -e . --no-deps
```

For a CPU-only Linux installation or a specific CUDA build, use the commands in [environment setup](docs/INSTALLATION.md).

### 2. Generate molecules

The trained model is included at [`models/pocketgpt-v1.pt`](models/pocketgpt-v1.pt). The file contains the weights, model configuration, and vocabulary. After cloning the repository and installing the environment, run:

```bash
python examples/generate.py --count 10 --output outputs/example.csv
```

This example requests **QED = 0.7**, **LogP = 2.5**, and a **benzene fragment**, using CPU. The same request can be passed directly to the command line:

```bash
python -m pocketgpt generate \
  --checkpoint models/pocketgpt-v1.pt \
  --property qed=0.7 --property logp=2.5 \
  --fragment 'c1ccccc1' --conditions properties,fragment \
  --count 10 --seed 42 --device cpu \
  --output outputs/property-fragment.csv
```

The output CSV contains sampled SMILES, validity, canonical SMILES, calculated properties, and fragment-match annotations. A companion `.summary.json` records sampling counts and settings. Use `--device cuda` for an available NVIDIA GPU. Choose a new output name for each run.

For **pocket + properties + fragment + IFP**, supply a pocket vector or a protein/pocket PDB pair and the selected targets; see the [four-condition example](docs/GENERATION.md#four-condition-generation). The included model was trained with cached descriptors. When using new DeeplyTough vectors without a verified encoder match, explicitly select `--allow-unverified-pocket`; this runs generation and records that compatibility has not been established. See [pocket compatibility](docs/MODEL_CARD.md#pocket-compatibility).

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
