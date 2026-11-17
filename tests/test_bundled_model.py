import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BundledModelTests(unittest.TestCase):
    def test_model_matches_manifest_and_checksum_file(self):
        manifest = json.loads((ROOT / "models" / "pocketgpt-v1.json").read_text())
        model = ROOT / "models" / manifest["filename"]
        self.assertTrue(model.is_file())
        self.assertEqual(model.stat().st_size, manifest["bytes"])
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["sha256"])
        checksum, filename = (ROOT / "models" / "SHA256SUMS").read_text().split()
        self.assertEqual(checksum, digest)
        self.assertEqual(filename, model.name)

    def test_default_model_is_verified_without_network(self):
        module = load_script("scripts/download_model.py", "bundled_model_downloader")
        expected = ROOT / "models" / "pocketgpt-v1.pt"
        self.assertEqual(module.DEFAULT_MODEL, expected)
        with patch.object(module, "urlopen", side_effect=AssertionError("Unexpected network request")):
            for verify_only in (False, True):
                result = module.download_model(verify_only=verify_only)
                self.assertEqual(result["status"], "verified")
                self.assertEqual(Path(result["model"]), expected)

    def test_example_uses_bundled_model_and_request(self):
        module = load_script("examples/generate.py", "bundled_model_example")
        request = json.loads((ROOT / "examples" / "property_fragment.json").read_text())
        with patch("sys.argv", ["generate.py"]), \
                patch.object(module, "generate", return_value={"status": "test"}) as generate, \
                contextlib.redirect_stdout(io.StringIO()):
            module.main()
        self.assertEqual(Path(generate.call_args.args[0]), ROOT / "models" / "pocketgpt-v1.pt")
        self.assertEqual(generate.call_args.args[1], request)
        self.assertEqual(generate.call_args.kwargs["conditions"], ["properties", "fragment"])


if __name__ == "__main__":
    unittest.main()
