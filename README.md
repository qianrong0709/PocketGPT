# 🧠 PocketGPT: Structure-Function Dual-Driven Generative Framework for Drug Design

> Official repository for **PocketGPT**, a generative model designed for multi-objective molecule generation with structure-function synergy. Our framework integrates protein pocket geometry with pharmacologically relevant constraints, enabling accurate, scalable, and experimentally validated drug design.

---

## 🔬 Highlights
- **Structure–Function Integration**: PocketGPT simultaneously considers spatial compatibility with the binding site and functional constraints like ADMET, fragment retention, and interaction fidelity.
- **Transformer-Based Autoregressive Model**: Enables unified modeling of discrete (atom/bond types) and continuous (3D coordinates) molecular features under conditional controls.
- **Systematic Evaluation**: Benchmarked across 100 diverse protein targets with strong performance in property satisfaction, geometric alignment, and pharmacophoric relevance.
- **Wet-Lab Validation**: Generated inhibitors for PARP1 were synthesized and experimentally validated with nanomolar affinity and favorable in vivo PK profiles.

---

## 📁 Repository Structure

```bash
PocketGPT/
├── models/               # Core architecture (equivariant networks, Bayesian samplers)
├── data/                 # Pocket-ligand datasets & preprocessed inputs
├── utils/                # Training, evaluation, visualization tools
├── scripts/              # Run training, sampling, and docking
└── README.md             # You are here
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/qianrong0709/PocketGPT.git
cd PocketGPT
conda env create -f environment.yml
conda activate pocketgpt
```

### Run Example
```bash
bash scripts/run_generation.sh  # Generate molecules conditioned on protein pockets
bash scripts/run_docking.sh     # Perform docking and evaluate binding scores
```

---

## 🧪 Supported Targets
We validate PocketGPT on several pharmaceutically relevant targets:
- PTPN1/PTPN2 (immune oncology)
- PI3Kα (metabolic/cancer)
- PARP1/PARP2 (DNA repair)

Each target evaluation includes:
- Pocket-conditioned generation
- Binding affinity benchmarking
- Fragment retention & interaction recovery

---

## 📊 Benchmark Metrics
- **Docking Score** (AutoDock Vina)
- **Pharmacophore Coverage**
- **ADMET Compliance** (QED, TPSA, SAS, logP)
- **Interaction Recovery** (IFP similarity)

---

## 📄 Citation
Coming soon (manuscript under review at *Nature Machine Intelligence*)

---

## 📬 Contact
- Rong Qian (qianrong0709)  
  PhD Student @ HKBU  
  Email: [24483877@life.hkbu.edu.hk](mailto:24483877@hkbu.edu.hk)

---

## 🧠 Keywords
`molecular generation` · `structure-based drug design` · `Bayesian generative model` · `pocket-conditioned generation` · `protein–ligand interaction`
