"""Synthetic/local-only regressions for the raw pocket input contract."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from pocketgpt.cli import main
from pocketgpt.deeplytough_bridge import ExplicitMGLSubprocess
from pocketgpt.generation import generate
from pocketgpt.pocket import PocketResolver, validate_pocket_sources


def fixture(root):
    paths = {
        "repo/deeplytough/engine/predictor.py": "# original synthetic encoder\n",
        "weights.pt": "synthetic weights",
        "mgl/bin/pythonsh": "#!/bin/sh\n",
        "mgl/MGLToolsPckgs/MolKit/__init__.py": "# synthetic MolKit\n",
        "mgl/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py": "# synthetic helper\n",
        "htmd/__init__.py": "# synthetic HTMD\n",
        "se3cnn/__init__.py": "# synthetic SE3CNN\n",
        "pocket.pdb": "SYNTHETIC PDB\n",
    }
    for name, text in paths.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    config = {"repo": str(root / "repo"), "weights": str(root / "weights.pt"),
              "python": sys.executable, "mgltools_prefix": str(root / "mgl"),
              "cache": str(root / "cache")}
    resolver = PocketResolver(config)
    resolver._probe_runtime = Mock(return_value={
        "python": [3, 6, 15], "packages": {"torch": "1.7.1", "numpy": "1.19.2"},
        "source_roots": {"htmd": [str(root / "htmd")], "se3cnn": [str(root / "se3cnn")]},
    })
    return resolver, {"pocket": {"pocket_pdb": str(root / "pocket.pdb")}}


class PocketContractTests(unittest.TestCase):
    def test_conflicting_sources_rejected_without_loading_anything(self):
        row = {"pocket_embedding": [7] * 128, "pocket": {"pocket_pdb": "/absent.pdb"}}
        with self.assertRaisesRegex(ValueError, "Pocket input conflict"):
            validate_pocket_sources(row)
        with self.assertRaisesRegex(ValueError, "Pocket input conflict"):
            PocketResolver().resolve(row)
        with self.assertRaisesRegex(ValueError, "Pocket input conflict"):
            generate("/absent-checkpoint.pt", row, "/absent-output.csv")

    def test_cli_explicit_pocket_cannot_keep_old_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            request.write_text(json.dumps({"pocket_embedding": [7] * 128}))
            argv = ["pocketgpt", "generate", "--checkpoint", "/absent.pt", "--request", str(request),
                    "--pocket", "/new.pdb", "--output", str(Path(directory) / "output.csv")]
            with patch.object(sys, "argv", argv), patch("pocketgpt.generation.generate") as run:
                with self.assertRaisesRegex(ValueError, "Pocket input conflict"):
                    main()
                run.assert_not_called()

    def test_profile_identity_is_content_not_path_and_refresh_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, _ = fixture(root / "one")
            second, _ = fixture(root / "two")
            initial = first.signature()
            self.assertEqual(initial, second.signature())
            self.assertTrue(initial.startswith("deeplytough:v2:"))
            (root / "one/weights.pt").write_text("different synthetic weights")
            self.assertNotEqual(initial, first.signature())
            original_source = second.signature()
            (root / "two/repo/deeplytough/engine/predictor.py").write_text("# different encoder\n")
            self.assertNotEqual(original_source, second.signature(refresh=True))

    def test_metadata_missing_or_inconsistent_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "signature must be a nonempty string"):
            PocketResolver().row_signature({"pocket_embedding": [0] * 128, "pocket_encoder_signature": None})
        with self.assertRaisesRegex(ValueError, "configuration is missing"):
            PocketResolver({"weights": "/absent.pt"}).signature()
        with tempfile.TemporaryDirectory() as directory:
            resolver, _ = fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "provenance does not match"):
                resolver.resolve({"pocket_embedding": [0] * 128})

    def test_fingerprints_bind_actual_pdb_and_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver, row = fixture(Path(directory))
            before = resolver.input_fingerprint(row)
            Path(row["pocket"]["pocket_pdb"]).write_text("CHANGED SYNTHETIC PDB\n")
            after = resolver.input_fingerprint(row)
            self.assertNotEqual(before["pocket_sha256"], after["pocket_sha256"])
            one = PocketResolver().input_fingerprint({"pocket_embedding": [0] * 128})
            two = PocketResolver().input_fingerprint({"pocket_embedding": [1] * 128})
            self.assertNotEqual(one["embedding_sha256"], two["embedding_sha256"])
            self.assertEqual(one["encoder"], "legacy-precomputed-unverified")

    def test_cache_checks_manifest_output_and_uses_isolated_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver, row = fixture(Path(directory))
            commands = []

            def encode(command, **kwargs):
                commands.append((command, kwargs))
                output = Path(command[command.index("--output") + 1])
                with output.open("wb") as stream:
                    np.save(stream, np.arange(128, dtype=np.float32), allow_pickle=False)
                return SimpleNamespace(returncode=0, stdout="synthetic only", stderr="")

            with patch.dict(os.environ, {"PYTHONPATH": "/wrong/python", "HTMD_CONFIG": "/wrong/htmd",
                                         "CONDA_PREFIX": "/wrong/conda", "BASH_ENV": "/wrong/bash"}):
                with patch("pocketgpt.pocket.subprocess.run", side_effect=encode):
                    first = resolver.resolve(row)
                    np.testing.assert_array_equal(first, resolver.resolve(row))
                    self.assertEqual(len(commands), 1)
                    environment = commands[0][1]["env"]
                    for name in ("PYTHONPATH", "HTMD_CONFIG", "CONDA_PREFIX", "BASH_ENV"):
                        self.assertNotIn(name, environment)
                    self.assertIn("--mgltools-prefix", commands[0][0])
                    output = next(Path(resolver.config["cache"]).glob("*/embedding.npy"))
                    output.write_bytes(b"corrupt output")
                    np.testing.assert_array_equal(first, resolver.resolve(row))
                    self.assertEqual(len(commands), 2)
                    manifest = output.with_name("provenance.json")
                    manifest.unlink()
                    resolver.resolve(row)
                    self.assertEqual(len(commands), 3)

    def test_original_mgl_command_uses_explicit_helper_without_shell(self):
        delegate = Mock()
        runner = ExplicitMGLSubprocess(delegate, "/synthetic/mgl")
        original = ["/bin/bash", "-ic", "source activate deeplytough_mgltools; pythonsh "
                    "$CONDA_PREFIX/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py "
                    "-r /synthetic/protein.pdb -U nphs_lps_waters -A hydrogens"]
        runner.run(original, cwd="/synthetic/output", check=True)
        args, kwargs = delegate.run.call_args
        self.assertEqual(args[0], ["/synthetic/mgl/bin/pythonsh",
                                  "/synthetic/mgl/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py",
                                  "-r", "/synthetic/protein.pdb", "-U", "nphs_lps_waters", "-A", "hydrogens"])
        self.assertEqual(kwargs["env"]["CONDA_PREFIX"], "/synthetic/mgl")
        self.assertTrue(kwargs["check"])
        # A changed/unrecognized shell command is not silently rewritten.
        unknown = ["/bin/bash", "-ic", "echo unrelated"]
        runner.run(unknown, check=True)
        self.assertEqual(delegate.run.call_args.args[0], unknown)
        unmatched = ["/bin/bash", "-ic", "echo 'unmatched"]
        runner.run(unmatched, check=True)
        self.assertEqual(delegate.run.call_args.args[0], unmatched)


if __name__ == "__main__":
    unittest.main()
