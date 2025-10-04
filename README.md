# 🧠 PocketGPT: Structure-Function Dual-Driven Generative Framework for Drug Design

**PocketGPT** is a structure–function dual-driven generative framework for molecular design. It unifies protein pocket geometry with multiple pharmacological constraints—such as drug-likeness, key fragments, and protein–ligand interaction fingerprints—into a single autoregressive molecular generation process.


---

## 🔬 Highlights
- **Structure–Function Integration**: PocketGPT simultaneously considers spatial compatibility with the binding site and functional constraints like ADMET, fragment retention, and interaction fidelity.
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

## 📊 Benchmark Metrics
- **Docking Score** (AutoDock Vina)
- **Pharmacophore Coverage**
- **ADMET Compliance** (QED, TPSA, SAS, logP)
- **Interaction Recovery** (IFP similarity)

---

## 📄 Citation
Coming soon 

---

## 📬 Contact
- Rong Qian  
  PhD Student @ HKBU  
  Email: [24483877@life.hkbu.edu.hk](mailto:24483877@hkbu.edu.hk)

---

## 🧠 Keywords
`molecular generation` · `structure-based drug design` · `pocket-conditioned generation` · `protein–ligand interaction`
