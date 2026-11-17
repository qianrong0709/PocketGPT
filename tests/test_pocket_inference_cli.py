"""CLI wiring with synthetic vectors and mocked generation; no encoder execution."""
import contextlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pocketgpt.cli import main as cli_main
from pocketgpt.config import parse_conditions
from pocketgpt.generation import generate


ROOT = Path(__file__).resolve().parents[1]
GENERATE_SIGNATURE = inspect.signature(generate)
ENCODER_SIGNATURE = "deeplytough:v2:" + "a" * 64


class PocketInferenceCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        spec = importlib.util.spec_from_file_location(
            "pocket_inference_example", ROOT / "examples" / "generate.py")
        self.example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.example)
        self.vector = [float(index) / 8 for index in range(128)]

    def write_json(self, name, value):
        path = self.directory / name
        path.write_text(json.dumps(value))
        return path

    def invoke(self, entry, flags=(), request=None):
        request_path = self.write_json("request.json", {} if request is None else request)
        flags = [str(value) for value in flags]
        common = ["--checkpoint", str(self.directory / "absent-model.pt"),
                  "--request", str(request_path),
                  "--output", str(self.directory / "output.csv")]
        if entry == "cli":
            argv = ["pocketgpt", "generate", *common, *flags]
            function = cli_main
            replacement = patch("pocketgpt.generation.generate", return_value={"status": "fixture"})
        else:
            argv = ["generate.py", *common, *flags]
            function = self.example.main
            replacement = patch.object(self.example, "generate", return_value={"status": "fixture"})
        with replacement as mocked, patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            try:
                function()
            except Exception:
                mocked.assert_not_called()
                raise
            mocked.assert_called_once()
            bound = GENERATE_SIGNATURE.bind(*mocked.call_args.args, **mocked.call_args.kwargs)
            bound.apply_defaults()
        self.assertFalse((self.directory / "output.csv").exists())
        return bound.arguments

    def test_bare_vector_does_not_acquire_a_legacy_signature(self):
        vector_path = self.write_json("vector.json", self.vector)
        compatibility = self.directory / "comparison-not-read-by-cli.json"
        for entry in ("cli", "example"):
            with self.subTest(entry=entry):
                call = self.invoke(entry, ["--pocket-vector", vector_path,
                                          "--pocket-compatibility", compatibility,
                                          "--allow-unverified-pocket", "--conditions", "pocket",
                                          "--count", "7", "--seed", "21", "--device", "cpu"])
                self.assertEqual(call["request"], {"pocket_embedding": self.vector})
                self.assertNotIn("pocket_encoder_signature", call["request"])
                self.assertEqual(parse_conditions(call["conditions"]), ("pocket",))
                self.assertEqual(call["pocket_compatibility"], str(compatibility))
                self.assertIs(call["allow_unverified_pocket"], True)
                self.assertEqual((call["count"], call["seed"], call["device"]), (7, 21, "cpu"))
                self.assertIsNone(call["encoder"])

    def test_structured_vector_keeps_its_recorded_identity(self):
        payload = {"pocket_embedding": self.vector, "pocket_encoder_signature": ENCODER_SIGNATURE}
        vector_path = self.write_json("identified-vector.json", payload)
        for entry in ("cli", "example"):
            with self.subTest(entry=entry):
                call = self.invoke(entry, ["--pocket-vector", vector_path])
                self.assertEqual(call["request"], payload)
                self.assertEqual(parse_conditions(call["conditions"]), ("pocket",))
                self.assertIs(call["allow_unverified_pocket"], False)
                self.assertIsNone(call["pocket_compatibility"])

    def test_raw_pocket_pair_and_encoder_configuration_are_forwarded(self):
        config = {"repo": "/not-installed/deeplytough", "python": "/not-installed/python",
                  "weights": "/not-installed/weights.pt", "cache": str(self.directory / "unused-cache")}
        config_path = self.write_json("encoder.json", config)
        protein, pocket = self.directory / "protein.pdb", self.directory / "pocket.pdb"
        for entry in ("cli", "example"):
            with self.subTest(entry=entry):
                call = self.invoke(entry, ["--protein", protein, "--pocket", pocket,
                                          "--encoder-config", config_path,
                                          "--allow-unverified-pocket"])
                self.assertEqual(call["request"], {"pocket": {
                    "protein_pdb": str(protein.resolve()), "pocket_pdb": str(pocket.resolve())}})
                self.assertEqual(call["encoder"], config)
                self.assertEqual(parse_conditions(call["conditions"]), ("pocket",))
                self.assertIs(call["allow_unverified_pocket"], True)
        self.assertFalse((self.directory / "unused-cache").exists())

    def test_nonpocket_selection_does_not_open_encoder_configuration(self):
        request = {"properties": {"qed": 0.7}}
        for entry in ("cli", "example"):
            with self.subTest(entry=entry):
                call = self.invoke(entry, ["--pocket", self.directory / "absent-pocket.pdb",
                                          "--encoder-config", self.directory / "absent-encoder.json",
                                          "--conditions", "properties"], request)
                self.assertEqual(parse_conditions(call["conditions"]), ("properties",))
                self.assertIsNone(call["encoder"])
                self.assertEqual(call["request"]["properties"], request["properties"])

    def test_all_condition_selection_preserves_other_inputs(self):
        vector_path = self.write_json("vector.json", self.vector)
        request = {"properties": {"qed": 0.7}, "fragment": "c1ccccc1", "ifp": [1, 1, 0, 0, 0, 0]}
        for entry in ("cli", "example"):
            with self.subTest(entry=entry):
                call = self.invoke(entry, ["--pocket-vector", vector_path,
                                          "--allow-unverified-pocket", "--conditions", "all"], request)
                self.assertEqual(set(parse_conditions(call["conditions"])),
                                 {"pocket", "properties", "fragment", "ifp"})
                self.assertEqual(call["request"], {**request, "pocket_embedding": self.vector})

    def test_conflicting_vector_inputs_are_rejected_before_generation(self):
        vector_path = self.write_json("vector.json", self.vector)
        cases = [
            ({}, ["--pocket-vector", vector_path, "--pocket", "another.pdb"]),
            ({"pocket_embedding": self.vector}, ["--pocket-vector", vector_path]),
            ({"pocket": {"pocket_pdb": "old.pdb"}}, ["--pocket-vector", vector_path]),
            ({"pocket_embedding": self.vector}, ["--pocket", "new.pdb"]),
            ({"pocket_embedding": self.vector, "pocket": {"pocket_pdb": "old.pdb"}}, []),
        ]
        for entry in ("cli", "example"):
            for index, (request, flags) in enumerate(cases):
                for selection in ([], ["--conditions", "properties"]):
                    with self.subTest(entry=entry, case=index, selection=selection):
                        with self.assertRaisesRegex(ValueError, "(?i)pocket input"):
                            self.invoke(entry, [*flags, *selection], request)

    def test_two_raw_pocket_sources_are_not_silently_overwritten(self):
        for entry in ("cli", "example"):
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, "(?i)pocket input"):
                    self.invoke(entry, ["--pocket", "new.pdb"],
                                {"pocket": {"pocket_pdb": "old.pdb"}})

    def test_protein_without_raw_pocket_is_rejected(self):
        vector_path = self.write_json("vector.json", self.vector)
        for entry in ("cli", "example"):
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, "--protein requires --pocket"):
                    self.invoke(entry, ["--pocket-vector", vector_path, "--protein", "protein.pdb"])

    def test_invalid_vector_is_not_sent_to_generation(self):
        vector_path = self.write_json("short-vector.json", [1.0] * 127)
        for entry in ("cli", "example"):
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, "128 finite numbers"):
                    self.invoke(entry, ["--pocket-vector", vector_path, "--allow-unverified-pocket"])


if __name__ == "__main__":
    unittest.main()
