# 🧠 PocketGPT: Structure-Function Dual-Driven Generative Framework for Drug Design

> Official repository for **PocketGPT**, a generative model designed for multi-objective molecule generation with structure-function synergy. Our framework integrates protein pocket geometry with pharmacologically relevant constraints, enabling accurate, scalable, and experimentally validated drug design.

---

## 🔬 Highlights
- **Dual-driven Generation**: Simultaneous control of structural (3D pocket compatibility) and functional (ADMET, fragments, interactions) objectives.
- **SE(3)-Equivariant Modeling**: Captures 3D molecular geometry with physically grounded constraints.
- **Bayesian Parameter-Space Sampling**: Enables efficient sampling over learnable distributions instead of noisy point-wise perturbation.
- **Wet-lab Validation**: Computational and experimental evaluation across multiple drug targets.

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
  Email: [qianrong0709@hkbu.edu.hk](mailto:qianrong0709@hkbu.edu.hk)

---

## 🧠 Keywords
`molecular generation` · `structure-based drug design` · `Bayesian generative model` · `pocket-conditioned generation` · `protein–ligand interaction`
