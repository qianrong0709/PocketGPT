#!/usr/bin/env bash
# Install separate CPU runtimes; does not run training or pocket encoding.
set -Eeuo pipefail

usage() {
    cat <<'USAGE'
Usage: bash scripts/setup_linux.sh --prefix /absolute/new/directory [options]

  --stage all|generator|encoder      Default: all
  --deeplytough-commit SHA          Override the pinned upstream DeeplyTough commit
  --accept-htmd-license             You have read and accepted HTMD 1.13.10 terms
  --accept-mgltools-license         You have read and accepted MGLTools 1.5.6 terms
  --resume                         Continue this install after a failed download

Requires Linux x86_64, glibc >= 2.28, Conda, git, gcc, g++, make and zlib headers.
No administrator access is used. Existing environments are not modified.
HTMD license: https://github.com/Acellera/htmd/blob/1.13.10/HTMD_LICENCE.txt
MGLTools 1.5.6 package (contains the exact license):
https://conda.anaconda.org/bioconda/linux-64/mgltools-1.5.6-1.tar.bz2
Read without installing: tar -xOf mgltools-1.5.6-1.tar.bz2 info/LICENSE.txt
The HTMD license includes collection of software usage information.
Review the DeeplyTough license at https://github.com/BenevolentAI/DeeplyTough/blob/master/LICENCE
USAGE
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
prefix=''
stage=all
deeplytough_commit=7536c5bb7d4e1e7e5d4f6cfacd1a437db03596e9
accept_htmd=0
accept_mgl=0
resume=0
while (($#)); do
    case "$1" in
        --prefix|--stage|--deeplytough-commit)
            (($# >= 2)) || die "Missing value after $1"
            case "$1" in
                --prefix) prefix=$2;;
                --stage) stage=$2;;
                --deeplytough-commit) deeplytough_commit=$2;;
            esac
            shift 2;;
        --accept-htmd-license) accept_htmd=1; shift;;
        --accept-mgltools-license) accept_mgl=1; shift;;
        --resume) resume=1; shift;;
        -h|--help) usage; exit 0;;
        *) die "Unknown option: $1";;
    esac
done
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die 'This installer requires Linux x86_64.'
[[ $prefix == /* && $prefix != / && $prefix =~ ^/[a-zA-Z0-9_./-]+$ ]] || die 'Use an absolute installation path without spaces or shell characters.'
[[ $stage == all || $stage == generator || $stage == encoder ]] || die 'Invalid --stage.'
for program in conda git gcc g++ make sha256sum; do
    command -v "$program" >/dev/null || die "Required program not found: $program"
done
[[ -f /usr/include/zlib.h ]] || die 'Install the zlib development headers first (Ubuntu: zlib1g-dev).'
glibc_version=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
[[ $(printf '%s\n' 2.28 "$glibc_version" | sort -V | head -n 1) == 2.28 ]] || die 'glibc >= 2.28 is required by the generator wheels.'
if [[ $stage != generator ]]; then
    [[ $accept_htmd == 1 && $accept_mgl == 1 ]] || { usage; die 'Read the two licenses, then pass both acceptance flags.'; }
    [[ $deeplytough_commit =~ ^[a-f0-9]{40}$ ]] || die 'Supply the documented 40-character DeeplyTough commit.'
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
[[ -f "$repo_root/pyproject.toml" ]] || die 'Run the script from a full PocketGPT checkout.'
marker="$prefix/.pocketgpt-install-v1"
if [[ -e $prefix ]]; then
    [[ $resume == 1 && -f $marker ]] || die 'Prefix already exists. Choose a new directory, or use --resume for this installer only.'
else
    mkdir -p -- "$prefix"
    touch "$marker"
fi
prefix=$(cd -- "$prefix" && pwd -P)
generator="$prefix/envs/generator"
encoder="$prefix/envs/encoder"
mgltools="$prefix/envs/mgltools"
mkdir -p -- "$prefix/sources" "$prefix/logs" "$prefix/pocket_cache"
exec > >(tee -a "$prefix/logs/setup.log") 2>&1
trap 'printf "Installation stopped at line %s. Files are retained in %s; rerun the same command with --resume.\n" "$LINENO" "$prefix" >&2' ERR

# Do not inherit alternate package indexes or TLS bypasses from user pip settings.
pip_install() {
    local interpreter=$1
    shift
    env -u PIP_TRUSTED_HOST -u PIP_EXTRA_INDEX_URL -u PIP_INDEX_URL \
        PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1 \
        "$interpreter" -m pip install --index-url https://pypi.org/simple "$@"
}
create_env() {
    local destination=$1
    shift
    if [[ ! -f $destination/conda-meta/history ]]; then
        CONDA_SSL_VERIFY=true conda create --yes --prefix "$destination" \
            --override-channels -c conda-forge "$@"
    fi
}
checkout_source() {
    local url=$1 destination=$2 revision=$3
    if [[ ! -d $destination/.git ]]; then
        [[ ! -e $destination ]] || die "Incomplete source directory: $destination. Inspect it before resuming."
        git init "$destination"
        git -C "$destination" remote add origin "$url"
    fi
    [[ $(git -C "$destination" remote get-url origin) == "$url" ]] || die "Unexpected source origin: $destination"
    if ! git -C "$destination" rev-parse --verify HEAD >/dev/null 2>&1; then
        if [[ $# == 4 ]]; then
            git -c http.sslVerify=true -C "$destination" fetch --depth 1 origin "tag" "$4"
        else
            git -c http.sslVerify=true -C "$destination" fetch --depth 1 origin "$revision"
        fi
        git -C "$destination" checkout --detach "$revision"
    fi
    [[ $(git -C "$destination" rev-parse HEAD) == "$revision" ]] || die "Unexpected source revision: $destination"
}

if [[ $stage == all || $stage == generator ]]; then
    printf '\nInstalling PocketGPT generator runtime\n'
    create_env "$generator" python=3.10.22 pip setuptools
    pip_install "$generator/bin/python" numpy==2.2.6 rdkit==2026.3.1
    pip_install "$generator/bin/python" --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
    pip_install "$generator/bin/python" --no-deps -e "$repo_root"
    "$generator/bin/python" -m pip check
    "$generator/bin/python" -m pip freeze > "$prefix/logs/generator-freeze.txt"
    CONDA_SSL_VERIFY=true conda list --prefix "$generator" --explicit > "$prefix/logs/generator-conda-explicit.txt"
fi

if [[ $stage == all || $stage == encoder ]]; then
    printf '\nInstalling original DeeplyTough CPU runtime\n'
    create_env "$encoder" python=3.6.15 pip
    pip_install "$encoder/bin/python" pip==21.3.1 setuptools==59.6.0 wheel==0.37.1
    # Old Conda Python ships a distutils-installed certifi; do not uninstall it.
    pip_install "$encoder/bin/python" --ignore-installed certifi==2025.4.26
    # Source metadata/build steps need these available before pip resolves the
    # remaining legacy packages (not merely listed in the same transaction).
    pip_install "$encoder/bin/python" numpy==1.19.2 Cython==0.29.13
    pip_install "$encoder/bin/python" --no-build-isolation -r "$repo_root/environments/encoder-py36.txt"
    pip_install "$encoder/bin/python" --no-build-isolation mdtraj==1.9.7
    pip_install "$encoder/bin/python" --index-url https://download.pytorch.org/whl/cpu torch==1.7.1+cpu

    checkout_source https://github.com/BenevolentAI/DeeplyTough.git "$prefix/sources/DeeplyTough" "$deeplytough_commit"
    checkpoint="$prefix/sources/DeeplyTough/networks/deeplytough_toughm1_test.pth.tar"
    [[ $(sha256sum "$checkpoint" | awk '{print $1}') == a8dfa2e19f24943f5faed170690d2bfc761782eebdf85477771d7fc16adcc558 ]] || die 'Unexpected DeeplyTough weights. Do not replace them with another encoder checkpoint.'

    htmd_source="$prefix/sources/htmd"
    checkout_source https://github.com/Acellera/htmd.git "$htmd_source" 489fd55e535a9c9394f746b44dbeede2d17cccad 1.13.10
    for library in libvmdparser.so libxtc.so mindist_ext.so occupancy_ext.so tmalign.so; do
        [[ -s $htmd_source/htmd/lib/Linux/$library ]] || die "Official HTMD checkout is missing native library: $library"
    done
    git -C "$htmd_source" diff --exit-code -- htmd/lib/Linux
    # The official Git tree contains the precompiled Linux libraries. Editable
    # installation also retains logging.ini, omitted by the legacy MANIFEST.in.
    pip_install "$encoder/bin/python" --no-deps --no-build-isolation -e "$htmd_source"

    se3_source="$prefix/sources/se3cnn"
    checkout_source https://github.com/mariogeiger/se3cnn.git "$se3_source" 6b976bea4ea17e1bd5655f0f030c6e2bb1637b57
    # Upstream DeeplyTough requires these experiment modules inside the package.
    if [[ -d $se3_source/experiments && ! -d $se3_source/se3cnn/experiments ]]; then
        mv -- "$se3_source/experiments" "$se3_source/se3cnn/experiments"
    fi
    "$encoder/bin/python" - "$se3_source/setup.py" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
path.write_text(text.replace("exclude=['experiments*']", ""))
PY
    pip_install "$encoder/bin/python" --no-deps --no-build-isolation -e "$se3_source"

    if [[ ! -f $mgltools/conda-meta/history ]]; then
        # This exact Bioconda package has depends=[] and its own Python runtime.
        # Installing the verified archive avoids solving current channels for a
        # retired package, and does not add a different Python interpreter.
        "$encoder/bin/python" - "$prefix" <<'PY'
import hashlib, pathlib, shutil, ssl, sys, tempfile, urllib.request
import certifi
root = pathlib.Path(sys.argv[1]) / "downloads"
root.mkdir(exist_ok=True)
package = root / "mgltools-1.5.6-1.tar.bz2"
expected = "fb2e70d2106e2bc644c9b152f131039b516af702d52f680902c3060c6d702a09"
url = "https://conda.anaconda.org/bioconda/linux-64/mgltools-1.5.6-1.tar.bz2"

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if not package.exists():
    print("Downloading the pinned MGLTools package", flush=True)
    with tempfile.TemporaryDirectory(prefix="mgl-download-", dir=str(root)) as work:
        temporary = pathlib.Path(work) / package.name
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=60, context=context) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if digest(temporary) != expected:
            raise RuntimeError("MGLTools download SHA256 mismatch; nothing installed")
        temporary.replace(package)
if digest(package) != expected:
    raise RuntimeError("Cached MGLTools package SHA256 mismatch; inspect it before retrying")
print("MGLTools package SHA256 verified: " + expected)
PY
        CONDA_SSL_VERIFY=true conda create --yes --prefix "$mgltools" \
            --offline "$prefix/downloads/mgltools-1.5.6-1.tar.bz2"
    fi
    [[ -x $mgltools/bin/pythonsh ]] || die 'MGLTools pythonsh is missing.'
    # No extra Conda Python is installed: this package includes Python 2.5.6.
    "$mgltools/bin/pythonsh" -V
    "$encoder/bin/python" -m pip check
    "$encoder/bin/python" -m pip freeze > "$prefix/logs/encoder-freeze.txt"
    CONDA_SSL_VERIFY=true conda list --prefix "$encoder" --explicit > "$prefix/logs/encoder-conda-explicit.txt"
    CONDA_SSL_VERIFY=true conda list --prefix "$mgltools" --explicit > "$prefix/logs/mgltools-conda-explicit.txt"
    "$encoder/bin/python" - "$prefix" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
config = {
    "repo": str(root / "sources/DeeplyTough"),
    "python": str(root / "envs/encoder/bin/python"),
    "weights": str(root / "sources/DeeplyTough/networks/deeplytough_toughm1_test.pth.tar"),
    "mgltools_prefix": str(root / "envs/mgltools"),
    "cache": str(root / "pocket_cache"),
}
(root / "encoder.local.json").write_text(json.dumps(config, indent=2) + "\n")
PY
fi
printf '\nInstallation finished. This does not yet validate a PDB-to-molecule run.\n'
printf 'Generator Python: %s/bin/python\nEncoder configuration: %s/encoder.local.json\n' "$generator" "$prefix"
