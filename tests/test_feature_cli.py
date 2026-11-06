import json
from pathlib import Path
import tempfile
import unittest

from pocketgpt.cli import prepare_request_files


class FeatureCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def spec(self, value):
        path = self.root / "spec.json"
        path.write_text(json.dumps(value))
        return path

    def test_local_reference_properties_and_exclusive_output(self):
        source = self.spec({"reference_smiles": "CCO", "compute_properties": True,
                            "conditions": ["properties"]})
        output = self.root / "request.json"
        result = prepare_request_files(source, output)
        request = json.loads(output.read_text())
        self.assertEqual(set(request["properties"]), {"logp", "qed", "tpsa", "sas"})
        self.assertEqual(result["conditions"], ["properties"])
        self.assertTrue(Path(result["provenance"]).is_file())
        with self.assertRaises(FileExistsError):
            prepare_request_files(source, output)

    def test_relative_structure_paths_do_not_run_encoder(self):
        (self.root / "pocket.pdb").write_text("END\n")
        (self.root / "protein.pdb").write_text("END\n")
        source = self.spec({"pocket": {"pocket_pdb": "pocket.pdb", "protein_pdb": "protein.pdb"},
                            "properties": {"qed": 0.6}})
        output = self.root / "request.json"
        result = prepare_request_files(source, output)
        request = json.loads(output.read_text())
        self.assertEqual(request["pocket"]["pocket_pdb"], str(self.root / "pocket.pdb"))
        self.assertEqual(request["pocket"]["protein_pdb"], str(self.root / "protein.pdb"))
        self.assertNotIn("pocket_embedding", request)
        self.assertEqual(result["conditions"], ["pocket", "properties"])

    def test_raw_fragment_preserved(self):
        source = self.spec({"fragment": "OCC", "conditions": ["fragment"],
                            "fragment_representation": "raw-smiles-v1"})
        output = self.root / "request.json"
        prepare_request_files(source, output)
        self.assertEqual(json.loads(output.read_text())["fragment"], "OCC")

    def test_json_cannot_select_executable_feature_provider(self):
        source = self.spec({"fragment_provider": "untrusted:callable"})
        output = self.root / "request.json"
        with self.assertRaisesRegex(ValueError, "documented input fields"):
            prepare_request_files(source, output)
        self.assertFalse(output.exists())

    def test_encoder_configuration_is_not_silently_ignored(self):
        source = self.spec({"properties": {"qed": 0.5}})
        with self.assertRaisesRegex(ValueError, "requires --resolve-pocket"):
            prepare_request_files(source, self.root / "request.json", self.root / "missing.json")


if __name__ == "__main__":
    unittest.main()
