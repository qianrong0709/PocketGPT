# PocketGPT

### Structure–function conditioned molecular generation

Research repository for **Structure–function conditioned molecular generation with PocketGPT enables prospective discovery of PARP1 inhibitors**.

[Paper](https://doi.org/10.59717/j.xinn-drugdisc.2026.100028) · [Project page](https://qianrong0709.github.io/PocketGPT_Page/) · [Input guide](docs/INPUTS.md) · [Release status](docs/RELEASE.md) · [中文说明](docs/README.zh-CN.md)

PocketGPT brings protein-pocket context and medicinal-chemistry objectives into an autoregressive molecular language model. Pocket information, physicochemical property targets, a molecular fragment, and protein–ligand interaction fingerprints provide complementary conditions for generating molecular SMILES, without explicitly constructing 3D ligand coordinates.

## Availability

**This repository currently contains the legacy research implementation. A refactored training and inference package is under validation and has not yet been uploaded here.** This documentation update does not replace the existing code or release new model weights.

| Component | Status |
| --- | --- |
| Published paper and project page | Available through the links above |
| Legacy research scripts | Available in [`src/`](src/) |
| Refactored package and portable configurations | Pending release after validation |
| Validated refactored model checkpoints | Not yet released |
| Reproducible quick-start example | Pending the matching code, environment, and checkpoint release |

The paper's reported results and the new implementation's validation status are separate. A successful code test is not evidence that a new checkpoint reproduces the paper.

## Method at a glance

PocketGPT combines four sources of conditioning:

| Condition | Information supplied |
| --- | --- |
| Protein pocket | A representation of the selected binding pocket |
| Molecular properties | Targets for LogP, QED, TPSA, and synthetic accessibility score (SAS) |
| Single fragment | A molecular fragment expressed as SMILES |
| Interaction fingerprint (IFP) | Desired protein–ligand interaction types |

In the refactored design, each active condition is encoded and projected into a shared space. A masked, learnable fusion module produces a semantic prefix for the causal SMILES decoder. Conditions can be enabled individually or in combinations; the achieved control depends on the corresponding training and evaluation.

```text
Pocket representation ── projector ───────────────┐
Property targets ──────── projector ───────────────┤
Fragment SMILES ───────── sequence encoder ────────┼─ fusion ─ semantic prefix ─ SMILES decoder
Interaction fingerprint ─ projector ───────────────┘
```

The pocket frontend uses a frozen DeeplyTough workflow in the refactored implementation. Automatic structure-to-generation orchestration does not mean joint backpropagation into that encoder. See the [input guide](docs/INPUTS.md) for the distinction between cached training vectors and new-pocket inference.

## Research context

The paper studies joint structural and functional conditioning and reports a prospective PARP1 design–make–test campaign, including an inhibitor with a measured binding affinity of **K<sub>d</sub> = 4.88 nM** and favorable preliminary tolerability in mice. These are results reported in the [paper](https://doi.org/10.59717/j.xinn-drugdisc.2026.100028), not performance guarantees for arbitrary generated molecules or unreleased checkpoints.

## Repository layout

```text
PocketGPT/
├── README.md
├── CITATION.cff
├── docs/
│   ├── INPUTS.md          # Training and inference input contracts
│   ├── RELEASE.md         # Availability and release-readiness checklist
│   └── README.zh-CN.md    # Chinese overview
└── src/                   # Existing legacy implementation; unchanged
    ├── model.py
    ├── dataset.py
    ├── train.py
    ├── trainer.py
    ├── generate.py
    ├── utils.py
    ├── vocabulary.json
    ├── frag_molgpt/       # Fragment-conditioned research scripts
    ├── case_study/        # Case-study research scripts
    ├── parp1_vs/          # Downstream screening and analysis
    └── result/            # Existing analysis figures
```

## Getting started

You can clone the repository to inspect the current research scripts:

```bash
git clone https://github.com/qianrong0709/PocketGPT.git
cd PocketGPT
```

Access is required while the repository is private. The legacy scripts are not yet packaged as a portable, one-command installation. There is currently no root `environment.yml`, packaged generation shell script, or validated checkpoint download to use as a complete quick start. Do not substitute an unrelated checkpoint or pocket encoder.

The forthcoming release will pair code, dependency instructions, configuration examples, model checkpoints, and a tested inference example. Until those artifacts are available together, see [release status](docs/RELEASE.md) rather than treating development-interface examples as commands supported by the current checkout.

**Inference users will not need to prepare a training JSONL dataset.** The intended interface accepts a selected pocket and optional property, fragment, and IFP conditions. A full protein alone does not identify which binding site to use. See [what users need to provide](docs/INPUTS.md).

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

## Dependencies and licensing

Pocket encoding builds on [DeeplyTough](https://github.com/BenevolentAI/DeeplyTough). The refactored workflow uses PyTorch, NumPy, and RDKit for the generator; legacy pocket-preprocessing tools have separate environment and licensing requirements.

This repository does not currently declare a software license. Do not infer a code, model, or dataset license from the paper's publication license. Third-party code, pretrained weights, and source datasets remain subject to their own terms; a software license and explicit artifact terms must be settled before an open-source release.

## Contact

For research questions, contact Rong Qian at [24483877@life.hkbu.edu.hk](mailto:24483877@life.hkbu.edu.hk). For reproducibility questions, open a repository issue when you have access. Please do not include credentials or confidential structures, datasets, or model files in issues.
