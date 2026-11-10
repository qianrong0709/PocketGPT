# Environment setup

[Overview](../README.md) · [Generate molecules](GENERATION.md) · [Training](USAGE.md)

## Generator environment

Use Python **3.11** for the quick start. Python 3.10+ is supported by the package, but dependency wheels also depend on your operating system and Python version. The tested generator versions are PyTorch **2.11.0**, NumPy **2.2.6**, and RDKit **2026.3.1**. Linux CUDA training used PyTorch's **cu126** build; local CPU checks used macOS/Python 3.11.

From the repository root, create one isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Alternatively, with Conda:

```bash
conda create -n pocketgpt python=3.11 pip -y
conda activate pocketgpt
```

Choose **one** PyTorch installation below.

### macOS

```bash
python -m pip install torch==2.11.0
```

Use `--device cpu` for the documented examples. The released workflow does not claim MPS validation.

### Linux / Windows, CPU only

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

### Linux / Windows, NVIDIA CUDA 12.6 build

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
```

This requires a compatible NVIDIA driver and GPU. Select a different build if needed using the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v2110); do not disable certificate verification to install packages. `torchvision` and `torchaudio` are not required.

### Install PocketGPT and check the environment

```bash
python -m pip install numpy==2.2.6 rdkit==2026.3.1
python -m pip install -e . --no-deps
python -m pip check
python -m pocketgpt --help
python -c "import torch; from rdkit import rdBase; print('PyTorch:', torch.__version__); print('RDKit:', rdBase.rdkitVersion); print('CUDA available:', torch.cuda.is_available())"
python -m unittest discover -s tests -q
```

For Windows PowerShell, activate a venv with `.venv\Scripts\Activate.ps1`; the package commands remain the same. Windows installation and the legacy encoder are not part of the tested release environments. On a cluster, check GPU visibility **inside a GPU allocation**, not only on the login node. Tests use temporary synthetic fixtures; they do not need research datasets or model downloads.

## Download and run

```bash
python scripts/download_model.py
python examples/generate.py --count 10 --output outputs/example.csv
```

All molecule generation is local. The downloader makes an explicit HTTPS request to this project's GitHub release; generation does not call a hosted model service.

## Optional pocket-encoding environment

**Skip this section for property/fragment/IFP generation or existing compatible pocket descriptors.** The generator environment does not need HTMD, MGLTools, PLIP, or DeeplyTough merely to load the released model.

Raw-PDB encoding is a separate adapter based on [original DeeplyTough](https://github.com/BenevolentAI/DeeplyTough). Its older Python/PyTorch/HTMD dependencies should remain in a separate environment rather than being installed into the modern generator environment. Third-party source, weights, preprocessing dependencies, licenses, and any telemetry behavior must be reviewed separately; they are not downloaded by the PocketGPT model downloader.

Configure a local copy of `configs/encoder.example.json` with the encoder repository, its interpreter, trusted weights, MGLTools prefix, and a local cache directory. The adapter checks provenance and caches descriptors by input/encoder identity. A full protein also needs an explicit corresponding binding-pocket PDB.

The released model uses historical cached descriptors labelled `legacy-precomputed-unverified`. A newly encoded PDB is **not automatically interchangeable** with these vectors, even if both have 128 dimensions. The provenance guard intentionally rejects a mismatched encoder. See [model compatibility](MODEL_CARD.md#pocket-compatibility) before using raw structures with a checkpoint.

## Common issues

- **`No module named pocketgpt`**: activate the intended environment and run `python -m pip install -e . --no-deps` from the repository root.
- **`CUDA available: False`**: use CPU, or check the PyTorch build, driver, and scheduler allocation. Installing a CUDA toolkit alone does not prove GPU visibility.
- **RDKit/NumPy wheel unavailable**: use the recommended Python version and a supported platform; do not mix packages from different environments.
- **Model download fails**: download `pocketgpt-v1.pt` from the linked GitHub release manually, place it under `weights/`, and run `python scripts/download_model.py --verify-only`.
- **Output already exists**: choose a new `--output` path. The generator does not silently replace previous results.
