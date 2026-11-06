# PocketGPT

### Structure–function conditioned molecular generation

Code for **Structure–function conditioned molecular generation with PocketGPT enables prospective discovery of PARP1 inhibitors**.

[Paper](https://doi.org/10.59717/j.xinn-drugdisc.2026.100028) · [Project page](https://qianrong0709.github.io/PocketGPT_Page/) · [Input guide](docs/INPUTS.md) · [Usage](docs/USAGE.md) · [中文说明](docs/README.zh-CN.md)

PocketGPT brings protein-pocket context and medicinal-chemistry objectives into an autoregressive molecular language model. Pocket information, physicochemical property targets, a molecular fragment, and protein–ligand interaction fingerprints provide complementary conditions for generating molecular SMILES, without explicitly constructing 3D ligand coordinates.

## Method at a glance

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

The pocket frontend interfaces with a frozen DeeplyTough encoder. See the [input guide](docs/INPUTS.md) for cached pocket vectors and new-pocket encoding.

## Research context

The paper studies joint structural and functional conditioning and reports a prospective PARP1 design–make–test campaign, including an inhibitor with a measured binding affinity of **K<sub>d</sub> = 4.88 nM** and favorable preliminary tolerability in mice. See the [paper](https://doi.org/10.59717/j.xinn-drugdisc.2026.100028) for the experimental setup and evaluation.

## Repository layout

```text
PocketGPT/
├── README.md
├── CITATION.cff
├── pyproject.toml
├── requirements-runtime.txt # Exact versions used in the tested runtime
├── pocketgpt/             # Generator, data handling, and feature adapters
├── configs/               # Portable configuration examples
├── scripts/               # Preparation/validation tools and configurable Slurm wrapper
├── tests/                 # Regression tests
└── docs/
    ├── INPUTS.md          # Training and inference input contracts
    ├── USAGE.md           # Installation, training, and generation commands
    └── README.zh-CN.md    # Chinese overview
```

## Getting started

Clone and install PocketGPT in an isolated environment:

```bash
git clone https://github.com/qianrong0709/PocketGPT.git
cd PocketGPT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pocketgpt --help
python -m unittest discover -s tests -v
```

Python 3.10 or later is required. The generator uses PyTorch, NumPy, and RDKit. Select a PyTorch build appropriate for your hardware. The tested dependency versions are listed in `requirements-runtime.txt`.

The [usage guide](docs/USAGE.md) covers property-plus-single-fragment pretraining, cached-vector fine-tuning, and molecular generation. Generation uses a compatible model checkpoint and its tokenizer vocabulary.

For generation, specify a binding pocket and any optional property, fragment, and IFP conditions. When providing a full protein, identify the binding site explicitly. See the [input guide](docs/INPUTS.md) for details.

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

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff). Bibliographic details can also be checked against the [authors' institutional publication record](https://scholars.hkbu.edu.hk/en/publications/structurefunction-conditioned-molecular-generation-with-pocketgpt/).

## Dependencies

Pocket encoding builds on [DeeplyTough](https://github.com/BenevolentAI/DeeplyTough). The generator uses PyTorch, NumPy, and RDKit. Pocket-preprocessing tools require a separate environment. Third-party dependencies remain subject to their respective licenses.

## Contact

For research questions, contact Rong Qian at [24483877@life.hkbu.edu.hk](mailto:24483877@life.hkbu.edu.hk). For usage questions, open a repository issue. Please do not include credentials or confidential research materials in issues.
