"""Regression for the original library's mixed import styles; no HTMD/native run."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BRIDGE = Path(__file__).resolve().parents[1] / "pocketgpt/deeplytough_bridge.py"


class DeeplyToughImportPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "original-repo"
        # The original engine uses `from datasets ...`; datasets/custom.py
        # also uses `from deeplytough.misc.utils ...`. Both roots are needed.
        fixtures = {
            "deeplytough/datasets/__init__.py": "from .custom import Custom\n",
            "deeplytough/datasets/custom.py": (
                "from deeplytough.misc.utils import htmd_featurizer\n"
                "class Custom: pass\n"),
            "deeplytough/misc/utils.py": (
                "def htmd_featurizer(*args, **kwargs):\n"
                "    raise AssertionError('Research encoder must not execute in import test')\n"),
            "deeplytough/engine/datasets.py": (
                "from datasets import Custom\n"
                "class PointOfInterestVoxelizedDataset: pass\n"),
            # Stop at import, before checkpoint access or any preprocessing.
            "deeplytough/engine/models.py": "raise RuntimeError('IMPORT_CLOSURE_VERIFIED')\n",
        }
        for name, text in fixtures.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def run_isolated(self, program):
        result = subprocess.run([sys.executable, "-I", "-c", program,
                                 str(self.repo), str(BRIDGE)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_original_single_root_reproduces_missing_namespace(self):
        result = self.run_isolated("""
import json, os, sys
sys.path.insert(0, os.path.join(sys.argv[1], 'deeplytough'))
try:
    import datasets
except ModuleNotFoundError as error:
    print(json.dumps({'missing': error.name}))
else:
    raise AssertionError('Expected original single-root import failure')
""")
        self.assertEqual(result["missing"], "deeplytough")

    def test_bridge_resolves_both_roots_before_any_encoder_execution(self):
        result = self.run_isolated("""
import importlib.util, json, os, sys, types
repo, path = sys.argv[1:]
# Prevent even ordinary heavy scientific-library imports; these stubs must
# never be used for numerical operations in this import-only regression.
sys.modules['numpy'] = types.ModuleType('numpy')
sys.modules['torch'] = types.ModuleType('torch')
spec = importlib.util.spec_from_file_location('isolated_bridge', path)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
sys.argv = ['bridge', '--repo', repo, '--weights', 'never-opened',
            '--protein', 'never-opened', '--pocket', 'never-opened',
            '--work', 'never-created', '--output', 'never-created',
            '--mgltools-prefix', 'never-opened']
try:
    bridge.main()
except RuntimeError as error:
    if str(error) != 'IMPORT_CLOSURE_VERIFIED':
        raise
    print(json.dumps({'status': str(error), 'roots': sys.path[:2],
                      'custom_module': sys.modules['datasets.custom'].__file__,
                      'utils_module': sys.modules['deeplytough.misc.utils'].__file__,
                      'htmd_imported': any(name == 'htmd' or name.startswith('htmd.')
                                           for name in sys.modules)}))
else:
    raise AssertionError('Import sentinel was not reached')
""")
        self.assertEqual(result["status"], "IMPORT_CLOSURE_VERIFIED")
        self.assertEqual(result["roots"], [str(self.repo / "deeplytough"), str(self.repo)])
        self.assertEqual(result["custom_module"], str(self.repo / "deeplytough/datasets/custom.py"))
        self.assertEqual(result["utils_module"], str(self.repo / "deeplytough/misc/utils.py"))
        self.assertFalse(result["htmd_imported"])
        self.assertFalse((self.repo / "never-created").exists())


if __name__ == "__main__":
    unittest.main()
