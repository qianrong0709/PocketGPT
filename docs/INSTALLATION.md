# Environment setup

[Overview](../README.md) · [Generate molecules](GENERATION.md) · [Training](USAGE.md)

There are two installation routes:

- **New pocket PDB → molecules:** install the generator and the original DeeplyTough preprocessing/encoding environments on Linux.
- **Stored pocket vector or property/fragment/IFP targets → molecules:** install only the generator.

The trained generator is included in `models/pocketgpt-v1.pt`. No training dataset is needed for inference.

## Linux prerequisites

The full PDB workflow requires Linux x86_64, glibc 2.28 or newer, Conda, Git, GCC/G++, Make, and zlib development headers. Use a directory without spaces. The default installation and examples use CPU; a GPU is optional for the generator.

The real-pocket PDB workflow was run on Ubuntu 22.04 (glibc 2.35) in newly created CPU environments, using the released generator and original DeeplyTough weights. Both protein/pocket pairs and pocket-only inputs completed encoding and generation; four-condition generation also completed.

Check the system in your terminal:

```bash
uname -sm
getconf GNU_LIBC_VERSION
conda --version
git --version
gcc --version
g++ --version
make --version
test -f /usr/include/zlib.h && echo "zlib headers found"
```

If Conda is missing, install the Linux x86_64 version of [Miniforge](https://github.com/conda-forge/miniforge#install), then reopen the terminal. If compiler tools are missing on Ubuntu/Debian, an administrator can install them with:

```bash
sudo apt-get update
sudo apt-get install git build-essential zlib1g-dev
```

On a shared cluster, ask the administrator for missing system packages and run encoding/generation in an allocated compute job, not on the login node. The setup script does not need administrator permissions and does not modify existing Conda environments.

<a id="optional-pocket-encoding-environment"></a>

## Full Linux installation

Download the repository and choose a new installation directory:

```bash
git clone --depth 1 https://github.com/qianrong0709/PocketGPT.git
cd PocketGPT
export PG_RUNTIME="$PWD/.pocketgpt-runtime"
```

### Licenses

Before installation, read the [HTMD 1.13.10 license](https://github.com/Acellera/htmd/blob/1.13.10/HTMD_LICENCE.txt) and the [DeeplyTough license at the pinned revision](https://github.com/BenevolentAI/DeeplyTough/blob/7536c5bb7d4e1e7e5d4f6cfacd1a437db03596e9/LICENCE). HTMD's terms include reporting software-usage information.

For MGLTools, review `info/LICENSE.txt` inside the exact **1.5.6 build 1** [Bioconda package](https://conda.anaconda.org/bioconda/linux-64/mgltools-1.5.6-1.tar.bz2). That file contains the terms for the bundled components; a license page for a newer MGLTools release is not a substitute. If you have not already read this file, the following commands download the package, verify its checksum, and display its license. They do not install or execute MGLTools:

```bash
PG_LICENSE_DIR=$(mktemp -d)
curl --fail --location --proto '=https' \
  https://conda.anaconda.org/bioconda/linux-64/mgltools-1.5.6-1.tar.bz2 \
  --output "$PG_LICENSE_DIR/mgltools-1.5.6-1.tar.bz2"
printf '%s  %s\n' \
  fb2e70d2106e2bc644c9b152f131039b516af702d52f680902c3060c6d702a09 \
  "$PG_LICENSE_DIR/mgltools-1.5.6-1.tar.bz2" | sha256sum --check - \
  && tar -xOf "$PG_LICENSE_DIR/mgltools-1.5.6-1.tar.bz2" info/LICENSE.txt
```

If the checksum does not match, stop and check the download. Pass the following flags only after reading and accepting the applicable terms:

```bash
bash scripts/setup_linux.sh --prefix "$PG_RUNTIME" \
  --accept-htmd-license --accept-mgltools-license
export PG_PYTHON="$PG_RUNTIME/envs/generator/bin/python"
"$PG_PYTHON" scripts/download_model.py --verify-only
"$PG_PYTHON" -m pocketgpt --help
```

The script downloads dependencies from their official package indexes and source repositories. It installs separate runtimes because the original encoder and the generator require different Python versions:

| Directory under `PG_RUNTIME` | Purpose |
| --- | --- |
| `envs/generator/` | Python 3.10.22, PyTorch 2.11.0 CPU, NumPy 2.2.6, RDKit 2026.3.1 |
| `envs/encoder/` | Python 3.6.15, PyTorch 1.7.1 CPU, original HTMD/SE3CNN dependencies |
| `envs/mgltools/` | MGLTools 1.5.6 with its own Python runtime |
| `sources/` | Pinned upstream source checkouts and DeeplyTough weights |
| `encoder.local.json` | Interpreter, source, weight, and cache paths for the PDB adapter |
| `pocket_cache/` | Computed 128-dimensional descriptors and encoder logs |
| `logs/` | Installation log and installed package lists |

DeeplyTough is pinned to commit `7536c5bb7d4e1e7e5d4f6cfacd1a437db03596e9`. Its `deeplytough_toughm1_test.pth.tar` checkpoint is downloaded from the upstream repository and checked against SHA-256 `a8dfa2e19f24943f5faed170690d2bfc761782eebdf85477771d7fc16adcc558`. Do not substitute the Vertex or ProSPECCTs checkpoint: they produce different descriptors.

Keep the runtime and source directories in place after installation. No manual Conda activation is required when you use `"$PG_PYTHON"`. In a new terminal, return to the repository and set the two variables again:

```bash
cd /path/to/PocketGPT
export PG_RUNTIME="$PWD/.pocketgpt-runtime"
export PG_PYTHON="$PG_RUNTIME/envs/generator/bin/python"
```

Then supply [your own pocket PDB](GENERATION.md#generate-from-your-pocket-pdb). Installation alone does not run a structure through the encoder.

### Resume an interrupted installation

The installer preserves downloaded files. Read `"$PG_RUNTIME/logs/setup.log"`, resolve the reported problem, and rerun the same installation command with `--resume`:

```bash
bash scripts/setup_linux.sh --prefix "$PG_RUNTIME" --resume \
  --accept-htmd-license --accept-mgltools-license
```

`--resume` is for a directory created by this installer; it does not adopt an unrelated environment. Keep HTTPS certificate verification enabled.

## Generator-only installation

Use this route when you do not need to encode a new PDB. Python 3.10 or newer is required. The following example uses Python 3.11:

```bash
git clone --depth 1 https://github.com/qianrong0709/PocketGPT.git
cd PocketGPT
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.6 rdkit==2026.3.1
python -m pip install -e . --no-deps
python -m pip check
python scripts/download_model.py --verify-only
python examples/generate.py --count 10 --output outputs/example.csv
```

For macOS, install PyTorch with `python -m pip install torch==2.11.0` instead of the CPU-index command. For Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`; Windows has not been tested. The legacy PDB encoder setup above is Linux-only.

The example requests QED 0.7, LogP 2.5, and a benzene fragment. It writes sampled SMILES and calculated properties to `outputs/example.csv`, plus a companion summary JSON. If you want to use the README's `"$PG_PYTHON"` commands with this environment, set:

```bash
export PG_PYTHON="$PWD/.venv/bin/python"
```

### NVIDIA GPU for the generator

The Linux setup script installs a CPU generator. To use an NVIDIA GPU, install the CUDA 12.6 build in the generator environment:

```bash
"$PG_PYTHON" -m pip install --upgrade --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
"$PG_PYTHON" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

This requires a compatible NVIDIA driver and GPU. On a cluster, run the check inside a GPU allocation. See the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v2110) for other supported builds. Generate with `--device cuda`; the original DeeplyTough adapter still runs on CPU. `torchvision` and `torchaudio` are not required.

## Common issues

- **`conda: command not found`:** finish Conda installation and reopen the terminal before running the setup script.
- **Prefix already exists:** choose a new installation directory, or use `--resume` for an interrupted run of this installer.
- **`No module named pocketgpt`:** use `"$PG_PYTHON"` from the installed generator environment. For a manual installation, run `python -m pip install -e . --no-deps` from the repository root.
- **`DeeplyTough failed; see local log`:** open the named `encoder.log`. It records the preprocessing or encoder error; the last line of the outer traceback is not the underlying cause.
- **`CUDA available: False`:** use CPU, or check the PyTorch build, NVIDIA driver, and scheduler allocation.
- **RDKit wheel unavailable:** check the Python version, architecture, and glibc version. The selected Linux wheel requires glibc 2.28 or newer.
- **Model file missing:** run `"$PG_PYTHON" scripts/download_model.py` to restore it from the [v1.0.0 release](https://github.com/qianrong0709/PocketGPT/releases/tag/v1.0.0), then run with `--verify-only`.
- **Output already exists:** choose a new `--output` path. Existing results are preserved.
- **Pocket provenance mismatch:** with the included historical checkpoint, use `--allow-unverified-pocket` to proceed while recording unverified descriptor compatibility, or provide a verified `--pocket-compatibility` record. This error is not a missing-weights error. Do not rename the encoder signature to suppress it.

### Python HTTPS certificates

For `CERTIFICATE_VERIFY_FAILED`, configure the trusted CA bundle for the Python installation. Do not disable certificate checks. On macOS, if the system bundle `/etc/ssl/cert.pem` exists, it can be selected for a download:

```bash
SSL_CERT_FILE=/etc/ssl/cert.pem python scripts/download_model.py
```

See [Python's SSL reference](https://docs.python.org/3.11/library/ssl.html#ssl.get_default_verify_paths) for default certificate paths.
