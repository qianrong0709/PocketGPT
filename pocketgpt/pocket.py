"""Original DeeplyTough adapter with explicit inputs and content provenance."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

LEGACY_SIGNATURE = "legacy-precomputed-unverified"
PROFILE_VERSION = 2


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_embedding(value, dim=128):
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (dim,) or not np.isfinite(vector).all():
        raise ValueError(f"Pocket embedding must contain {dim} finite numbers, got {vector.shape}")
    return vector


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_pocket_sources(row):
    """Check exclusivity without opening a file, including during preparation."""
    if "pocket_embedding" in row and row.get("pocket"):
        raise ValueError("Pocket input conflict: provide pocket_embedding OR a pocket structure, not both")


def _atomic_json(path, value):
    handle, temporary = tempfile.mkstemp(prefix=".provenance-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# Python 3.6-compatible, metadata-only query: never import HTMD to inspect it.
_RUNTIME_PROBE = r'''
import importlib.util, json, sys
try:
    from importlib import metadata
    version = metadata.version
except ImportError:
    import pkg_resources
    version = lambda name: pkg_resources.get_distribution(name).version
packages = {}
for name in ('torch', 'numpy', 'scipy', 'biopython', 'htmd', 'se3cnn', 'lie-learn', 'mdtraj', 'numba'):
    try:
        packages[name] = version(name)
    except Exception:
        packages[name] = None
roots = {}
for name in ('htmd', 'se3cnn'):
    spec = importlib.util.find_spec(name)
    roots[name] = list(spec.submodule_search_locations or []) if spec else []
print(json.dumps({'python': list(sys.version_info[:3]), 'packages': packages, 'source_roots': roots}, sort_keys=True))
'''


class PocketResolver:
    def __init__(self, config=None):
        self.config = config or {}
        self._file_hashes = {}
        self._runtime = None
        self._profile = None

    def _hash(self, path):
        path = Path(path)
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._file_hashes.get(str(path))
        if cached is None or cached[0] != identity:
            self._file_hashes[str(path)] = (identity, file_sha256(path))
        return self._file_hashes[str(path)][1]

    def _source_hashes(self, root):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Encoder source directory not found: {root}")
        files = {str(path.relative_to(root)): self._hash(path)
                 for path in sorted(root.rglob("*.py")) if "__pycache__" not in path.parts}
        if not files:
            raise ValueError(f"No encoder source files found: {root}")
        return files

    def _runtime_environment(self):
        """Explicit PATH, no inherited Python/HTMD hooks or conda activation."""
        environment = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "HTMD_CONFIG", "BASH_ENV", "ENV",
                     "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "LD_PRELOAD", "LD_LIBRARY_PATH"):
            environment.pop(name, None)
        interpreter = Path(self.config.get("python", sys.executable)).resolve()
        environment.update(PATH=os.pathsep.join([str(interpreter.parent), "/usr/bin", "/bin"]),
                           PYTHONNOUSERSITE="1", WANDB_MODE="disabled", WANDB_DISABLED="true",
                           HF_HUB_OFFLINE="1", MPLBACKEND="Agg", OMP_NUM_THREADS="1")
        return environment

    def _probe_runtime(self):
        if self._runtime is None:
            interpreter = Path(self.config.get("python", sys.executable)).resolve()
            result = subprocess.run([str(interpreter), "-I", "-c", _RUNTIME_PROBE],
                                    env=self._runtime_environment(), capture_output=True, text=True,
                                    timeout=60, check=True)
            self._runtime = json.loads(result.stdout)
        return self._runtime

    def profile(self, refresh=False):
        """Content identity; installation/cache paths are deliberately excluded."""
        if not self.config:
            return {"profile_version": 0, "kind": LEGACY_SIGNATURE}
        required = ("repo", "python", "weights", "mgltools_prefix")
        missing = [name for name in required if not self.config.get(name)]
        if missing:
            raise ValueError("Raw encoder configuration is missing: " + ", ".join(missing))
        weights = Path(self.config["weights"]).resolve()
        if self._profile is not None and not refresh:
            # A profile is a source/runtime snapshot for this run; a new run or
            # explicit refresh reprobes it. Weight replacement is cheap to
            # detect immediately, including through an existing resolver.
            if self._hash(weights) == self._profile["weights_sha256"]:
                return self._profile
            refresh = True
        if refresh:
            self._runtime = None
        repo = Path(self.config["repo"]).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Original pocket encoder weights not found: {weights}")
        mgl = Path(self.config["mgltools_prefix"]).resolve()
        scripts = (mgl / "bin/pythonsh", mgl / "MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py")
        for path in scripts:
            if not path.is_file():
                raise FileNotFoundError(f"Original MGLTools helper not found: {path}")
        runtime = self._probe_runtime()
        dependency_sources = {}
        for name in ("htmd", "se3cnn"):
            roots = runtime["source_roots"].get(name, [])
            if len(roots) != 1:
                raise ValueError(f"Cannot identify installed {name} sources in the encoder environment")
            dependency_sources[name] = self._source_hashes(roots[0])
        profile = {
            "profile_version": PROFILE_VERSION, "kind": "deeplytough-original-htmd",
            "weights_sha256": self._hash(weights),
            "encoder_sources": self._source_hashes(repo / "deeplytough"),
            "bridge_sha256": self._hash(Path(__file__).with_name("deeplytough_bridge.py")),
            "dependency_sources": dependency_sources,
            "runtime": {"python": runtime["python"], "packages": runtime["packages"]},
            "mgltools": {"pythonsh_sha256": self._hash(scripts[0]),
                         "MolKit": self._source_hashes(mgl / "MGLToolsPckgs/MolKit"),
                         "AutoDockTools": self._source_hashes(mgl / "MGLToolsPckgs/AutoDockTools")},
            "preprocessing": {"remove": "nphs_lps_waters", "add": "hydrogens",
                              "protein_filter": "original-remove-water-and-hets",
                              "descriptor_dim": 128, "normalization": "original-L2",
                              "device": "cpu", "num_threads": 1, "mgl_invocation": "explicit-argv-v1"},
        }
        self._profile = profile
        return profile

    def signature(self, refresh=False):
        if not self.config:
            return LEGACY_SIGNATURE
        return "deeplytough:v2:" + content_hash(self.profile(refresh=refresh))

    def row_signature(self, row):
        validate_pocket_sources(row)
        if "pocket_embedding" in row:
            signature = row.get("pocket_encoder_signature", LEGACY_SIGNATURE)
            if not isinstance(signature, str) or not signature:
                raise ValueError("Pocket encoder signature must be a nonempty string")
            return signature
        if row.get("pocket"):
            if not self.config:
                raise ValueError("Raw-PDB encoding needs the original DeeplyTough repo/python/weights configuration")
            return self.signature()
        raise ValueError("A pocket condition needs pocket_embedding or a pocket structure")

    def input_fingerprint(self, row):
        """Bind continuation to actual inputs, not merely JSONL path strings."""
        signature = self.row_signature(row)
        if "pocket_embedding" in row:
            vector = validate_embedding(row["pocket_embedding"])
            return {"source": "precomputed", "encoder": signature,
                    "embedding_sha256": hashlib.sha256(vector.tobytes()).hexdigest()}
        spec = row["pocket"]
        pocket = Path(spec["pocket_pdb"]).resolve()
        protein = Path(spec.get("protein_pdb", pocket)).resolve()
        if not pocket.is_file() or not protein.is_file():
            raise FileNotFoundError("Protein/pocket PDB path does not exist")
        return {"source": "raw-pdb", "encoder": signature,
                "protein_sha256": self._hash(protein), "pocket_sha256": self._hash(pocket),
                "context": "cropped-pocket" if protein == pocket else "full-protein"}

    def resolve(self, row):
        validate_pocket_sources(row)
        if "pocket_embedding" in row:
            if self.config and self.row_signature(row) != self.signature():
                raise ValueError("Pocket embedding provenance does not match the configured encoder")
            return validate_embedding(row["pocket_embedding"])
        provenance = self.input_fingerprint(row)
        repo = Path(self.config["repo"]).resolve()
        interpreter = Path(self.config.get("python", sys.executable)).resolve()
        if not interpreter.is_file() or not (repo / "deeplytough/engine/predictor.py").is_file():
            raise FileNotFoundError("Original DeeplyTough interpreter or predictor.py not found")
        key = content_hash(provenance)
        cache_root = Path(self.config.get("cache", "cache/pockets")).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        work = cache_root / key
        output, manifest = work / "embedding.npy", work / "provenance.json"
        if output.is_file() and manifest.is_file():
            try:
                cached = json.loads(manifest.read_text())
                if cached["input"] == provenance and cached["output_sha256"] == self._hash(output):
                    return validate_embedding(np.load(output, allow_pickle=False))
            except (ValueError, KeyError, OSError):
                pass  # Incomplete/tampered cache must be recomputed.
        # Never reuse a stale protein.npz during recomputation. Publish only a
        # validated output from a private staging directory.
        with tempfile.TemporaryDirectory(prefix=".encode-", dir=cache_root) as staging:
            staging = Path(staging)
            staged_output = staging / "embedding.npy"
            environment = self._runtime_environment()
            isolated_home = staging / "home"
            isolated_home.mkdir()
            environment.update(HOME=str(isolated_home), TMPDIR=str(staging))
            spec = row["pocket"]
            command = [str(interpreter), "-I", str(Path(__file__).with_name("deeplytough_bridge.py")), "--repo", str(repo),
                       "--weights", str(Path(self.config["weights"]).resolve()),
                       "--protein", str(Path(spec.get("protein_pdb", spec["pocket_pdb"])).resolve()),
                       "--pocket", str(Path(spec["pocket_pdb"]).resolve()),
                       "--mgltools-prefix", str(Path(self.config["mgltools_prefix"]).resolve()),
                       "--work", str(staging), "--output", str(staged_output)]
            result = subprocess.run(command, cwd=staging, env=environment, capture_output=True, text=True, timeout=600)
            work.mkdir(parents=True, exist_ok=True)
            (work / "encoder.log").write_text(result.stdout + "\n" + result.stderr)
            if result.returncode != 0:
                raise RuntimeError(f"DeeplyTough failed; see local log: {work / 'encoder.log'}")
            vector = validate_embedding(np.load(staged_output, allow_pickle=False))
            self.signature(refresh=True)
            if self.input_fingerprint(row) != provenance:
                raise RuntimeError("Pocket inputs or encoder changed while encoding; refusing cached output")
            output_hash = file_sha256(staged_output)
            os.replace(staged_output, output)
            _atomic_json(manifest, {"input": provenance, "output_sha256": output_hash})
            return vector
