# Environment setup

[Overview](../README.md) · [Generate molecules](GENERATION.md) · [Training](USAGE.md)

## Generator environment

Use Python **3.11** for the examples below. The tested versions are PyTorch **2.11.0**, NumPy **2.2.6**, and RDKit **2026.3.1**, with **cu126** for Linux GPU training and CPU execution on macOS. The package requires Python 3.10 or newer.

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

Use `--device cpu` for the examples.

### Linux / Windows, CPU only

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

### Linux / Windows, NVIDIA CUDA 12.6 build

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
```

This build requires a compatible NVIDIA driver and GPU. For other CUDA versions, use the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v2110). PocketGPT does not require `torchvision` or `torchaudio`.

### Install PocketGPT and check the environment

```bash
python -m pip install numpy==2.2.6 rdkit==2026.3.1
python -m pip install -e . --no-deps
python -m pip check
python -m pocketgpt --help
python -c "import torch; from rdkit import rdBase; print('PyTorch:', torch.__version__); print('RDKit:', rdBase.rdkitVersion); print('CUDA available:', torch.cuda.is_available())"
python -m unittest discover -s tests -q
```

For Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. Windows has not been tested. On a cluster, check GPU visibility inside a GPU allocation. Tests check the bundled model's checksum and use synthetic fixtures for training checks.

## Run the model

The repository includes `models/pocketgpt-v1.pt`, with the model configuration and vocabulary stored in the same file. After installation, run:

```bash
python examples/generate.py --count 10 --output outputs/example.csv
```

Generation runs locally on CPU or GPU. To check the model file against its recorded SHA-256 checksum:

```bash
python scripts/download_model.py --verify-only
```

If the file is missing, `python scripts/download_model.py` restores it from the [v1.0.0 release](https://github.com/qianrong0709/PocketGPT/releases/tag/v1.0.0) to `models/`.

## Optional pocket-encoding environment

This environment is only needed to encode new protein pockets. Property/fragment/IFP generation and generation from compatible cached pocket vectors use the generator environment above.

The PDB adapter uses [DeeplyTough](https://github.com/BenevolentAI/DeeplyTough), which requires older Python/PyTorch/HTMD dependencies in a separate environment. Install its source, weights, and preprocessing tools under their respective licenses. Review HTMD's usage-reporting settings before running it. The PocketGPT download script retrieves only the generator weights.

Configure a local copy of `configs/encoder.example.json` with the encoder repository, interpreter, weights, MGLTools prefix, and cache directory. Supply both the protein PDB and its corresponding binding-pocket PDB. The adapter caches descriptors by input and encoder identity. Load encoder weights only from a trusted source; its older PyTorch loader can execute code contained in a checkpoint.

`pocketgpt-v1.pt` was trained with cached descriptors labelled `legacy-precomputed-unverified`. Compatibility with newly encoded DeeplyTough vectors has not been verified. Use a checkpoint matched to the encoder before generating from raw PDB files; see [pocket compatibility](MODEL_CARD.md#pocket-compatibility).

## Common issues

- **`No module named pocketgpt`**: activate the intended environment and run `python -m pip install -e . --no-deps` from the repository root.
- **`CUDA available: False`**: use CPU, or check the PyTorch build, NVIDIA driver, and scheduler allocation.
- **RDKit/NumPy wheel unavailable**: use Python 3.11 and a platform supported by the selected versions.
- **Model file missing**: run `python scripts/download_model.py`, or download `pocketgpt-v1.pt` from the release into `models/` and run `python scripts/download_model.py --verify-only`.
- **Output already exists**: choose a new `--output` path. Existing results are preserved.

### Python HTTPS certificates

If an optional model download reports `CERTIFICATE_VERIFY_FAILED`, configure the trusted CA certificates for your Python installation. Keep HTTPS verification enabled.

On macOS, if the system-provided `/etc/ssl/cert.pem` is present, this command uses that existing trusted bundle for this download only:

```bash
SSL_CERT_FILE=/etc/ssl/cert.pem python scripts/download_model.py
```

On other systems, use the trusted CA bundle for that Python installation. See [Python's SSL reference](https://docs.python.org/3.11/library/ssl.html#ssl.get_default_verify_paths).
