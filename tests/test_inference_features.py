"""Public example molecules and synthetic vectors; no research encoder/network."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from pocketgpt.config import CONDITIONS
from pocketgpt.features import prepare_inference_request, rdkit_properties
from pocketgpt.pocket import LEGACY_SIGNATURE, file_sha256


class FixtureResolver:
    def __init__(self):
        self.calls = 0
        self.fingerprint = {"source": "raw-pdb", "encoder": "deeplytough:v2:fixture", "pocket_sha256": "fixture"}

    def input_fingerprint(self, row):
        return copy.deepcopy(self.fingerprint)

    def row_signature(self, row):
        return "deeplytough:v2:fixture"

    def resolve(self, row):
        self.calls += 1
        return np.arange(128, dtype=np.float32) / 128


class InferenceFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pocket = self.root / "pocket.pdb"
        self.protein = self.root / "protein.pdb"
        self.complex = self.root / "complex.pdb"
        for path in (self.pocket, self.protein, self.complex):
            path.write_text("UNIT STRUCTURE FIXTURE; not a scientific complex\n")
        self.explicit = {"pocket_embedding": [0.1] * 128,
                         "properties": {"qed": 0.5, "logp": 0.0},
                         "fragment": "CO", "ifp": [0, 0, 0, 0, 0, 0]}

    def test_explicit_labels_remain_independent_and_legacy_signature_not_relabelled(self):
        original = copy.deepcopy(self.explicit)
        resolver = Mock()
        result = prepare_inference_request(**self.explicit, resolver=resolver)
        request = result["request"]
        self.assertEqual(result["conditions"], list(CONDITIONS))
        self.assertEqual(request["properties"], self.explicit["properties"])
        self.assertEqual(request["fragment"], self.explicit["fragment"])
        self.assertEqual(request["ifp"], [0] * 6)
        self.assertEqual(request["pocket_encoder_signature"], LEGACY_SIGNATURE)
        self.assertEqual(self.explicit, original)
        resolver.assert_not_called()
        resolver.resolve.assert_not_called()
        resolver.signature.assert_not_called()
        self.assertEqual(result["provenance"]["historical_training_embedding_equivalence"], "not_checked")
        json.dumps(result, allow_nan=False)

    def test_every_modality_subset_has_exact_request_fields(self):
        names = ("pocket_embedding", "properties", "fragment", "ifp")
        for mask in range(16):
            selected = [name for index, name in enumerate(CONDITIONS) if mask & (1 << index)]
            result = prepare_inference_request(**self.explicit, conditions=selected)
            expected = {field for field, name in zip(names, CONDITIONS) if name in selected}
            if "pocket" in selected:
                expected.add("pocket_encoder_signature")
            if "fragment" in selected:
                expected.add("fragment_representation")
            with self.subTest(mask=mask):
                self.assertEqual(result["conditions"], selected)
                self.assertEqual(set(result["request"]), expected)
                self.assertEqual(result["provenance"]["selected_conditions"], selected)

    def test_missing_properties_are_omitted_but_zero_is_a_real_target(self):
        result = prepare_inference_request(properties={"qed": 0.0, "sas": None, "tpsa": ""})
        self.assertEqual(result["request"]["properties"], {"qed": 0.0})
        self.assertEqual(result["conditions"], ["properties"])

    def test_none_skips_disabled_bad_inputs_providers_and_file_reads(self):
        provider = Mock(side_effect=AssertionError("must not compute disabled feature"))
        resolver = Mock(side_effect=AssertionError("must not encode disabled pocket"))
        with patch("pocketgpt.features._file_reference", side_effect=AssertionError("must not read disabled PDB")):
            result = prepare_inference_request(pocket="missing.pdb", protein="missing-protein.pdb",
                        properties={"qed": float("nan")}, fragment="invalid fragment", ifp=[1],
                        conditions="none", reference_smiles="invalid", complex_pdb="missing-complex.pdb",
                        compute_properties=True, extract_fragment=True, compute_ifp=True,
                        property_provider=provider, fragment_provider=provider, ifp_provider=provider,
                        resolver=resolver, resolve_pocket=True)
        self.assertEqual(result["request"], {})
        self.assertEqual(set(result["provenance"]["skipped_requested_computations"]), set(CONDITIONS))
        provider.assert_not_called()
        resolver.resolve.assert_not_called()

    def test_pocket_sources_remain_mutually_exclusive_even_when_disabled(self):
        with self.assertRaisesRegex(ValueError, "Pocket input conflict"):
            prepare_inference_request(pocket_embedding=[0.0] * 128, pocket=self.pocket, conditions="none")

    def test_raw_structure_is_deferred_without_loading_encoder(self):
        resolver = Mock(side_effect=AssertionError("must not load encoder"))
        before = {path: file_sha256(path) for path in (self.pocket, self.protein)}
        result = prepare_inference_request(pocket=self.pocket, protein=self.protein, resolver=resolver)
        self.assertEqual(result["request"], {"pocket": {"pocket_pdb": str(self.pocket), "protein_pdb": str(self.protein)}})
        source = result["provenance"]["sources"]["pocket"]
        self.assertEqual(source["cache"]["status"], "deferred_to_generation")
        self.assertEqual(source["pocket"]["sha256"], before[self.pocket])
        self.assertFalse(result["provenance"]["raw_structure_contents_copied"])
        resolver.resolve.assert_not_called()
        self.assertEqual(before, {path: file_sha256(path) for path in before})

    def test_explicit_raw_resolution_returns_only_embedding_and_profile_metadata(self):
        resolver = FixtureResolver()
        result = prepare_inference_request(pocket=self.pocket, protein=self.protein,
                                           resolver=resolver, resolve_pocket=True)
        self.assertEqual(resolver.calls, 1)
        self.assertNotIn("pocket", result["request"])
        self.assertEqual(len(result["request"]["pocket_embedding"]), 128)
        self.assertEqual(result["request"]["pocket_encoder_signature"], "deeplytough:v2:fixture")
        self.assertEqual(result["provenance"]["sources"]["pocket"]["cache"]["status"], "resolved_by_configured_adapter")
        self.assertEqual(result["provenance"]["historical_training_embedding_equivalence"], "not_checked")

    def test_raw_resolution_requires_resolver_and_stable_profile(self):
        with self.assertRaisesRegex(ValueError, "explicitly configured"):
            prepare_inference_request(pocket=self.pocket, resolve_pocket=True)
        resolver = FixtureResolver()
        def resolve_and_mutate(row):
            resolver.fingerprint["pocket_sha256"] = "changed"
            return np.ones(128, dtype=np.float32)
        resolver.resolve = resolve_and_mutate
        with self.assertRaisesRegex(ValueError, "changed during"):
            prepare_inference_request(pocket=self.pocket, resolver=resolver, resolve_pocket=True)

    def test_selected_conditions_without_values_raise_and_do_not_zero_fill(self):
        for name in CONDITIONS:
            with self.subTest(condition=name), self.assertRaises(ValueError):
                prepare_inference_request(conditions=[name])
        with self.assertRaisesRegex(ValueError, "no observed values"):
            prepare_inference_request(properties={"qed": None}, conditions="properties")
        with self.assertRaisesRegex(ValueError, "Missing IFP interaction columns"):
            prepare_inference_request(ifp={"hbonds": 1}, conditions="ifp")

    def test_reference_properties_use_ligand_and_shared_rdkit_definitions(self):
        result = prepare_inference_request(reference_smiles="OCC", compute_properties=True)
        self.assertEqual(result["conditions"], ["properties"])
        self.assertEqual(result["request"]["properties"], rdkit_properties("CCO"))
        source = result["provenance"]["sources"]["properties"]
        self.assertIn("rdkit_version", source)
        self.assertEqual(source["definitions"]["sas"], "Contrib.SA_Score.sascorer.calculateScore")
        self.assertEqual(source["historical_training_label_equivalence"], "not_asserted")
        with self.assertRaisesRegex(ValueError, "reference_smiles"):
            prepare_inference_request(protein=self.protein, compute_properties=True)

    def test_custom_property_provider_is_opt_in_and_canonical_reference_is_supplied(self):
        provider = Mock(return_value={"qed": 0.0})
        result = prepare_inference_request(reference_smiles="OCC", compute_properties=True, property_provider=provider)
        provider.assert_called_once_with("CCO")
        self.assertEqual(result["request"]["properties"], {"qed": 0.0})
        with self.assertRaisesRegex(ValueError, "Choose explicit"):
            prepare_inference_request(properties={"qed": 0.4}, reference_smiles="CCO", compute_properties=True)

    def test_fragment_extraction_needs_provider_and_an_explicit_single_choice(self):
        with self.assertRaisesRegex(ValueError, "no algorithm is selected automatically"):
            prepare_inference_request(reference_smiles="CCO", extract_fragment=True)
        provider = Mock(return_value="OC")
        result = prepare_inference_request(reference_smiles="OCC", extract_fragment=True, fragment_provider=provider)
        provider.assert_called_once_with("CCO")
        self.assertEqual(result["request"]["fragment"], "CO")
        with self.assertRaisesRegex(ValueError, "one nonempty string"):
            prepare_inference_request(reference_smiles="CCO", extract_fragment=True,
                                      fragment_provider=lambda smiles: ["C", "CO"])

    def test_ifp_calculation_needs_reference_complex_and_never_uses_protein_alone(self):
        provider = Mock(return_value=[0, 0, 0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "protein-ligand complex"):
            prepare_inference_request(protein=self.protein, compute_ifp=True, ifp_provider=provider)
        provider.assert_not_called()
        with self.assertRaisesRegex(ValueError, "explicit local ifp_provider"):
            prepare_inference_request(complex_pdb=self.complex, compute_ifp=True)
        result = prepare_inference_request(complex_pdb=self.complex, compute_ifp=True, ifp_provider=provider)
        provider.assert_called_once_with(str(self.complex))
        self.assertEqual(result["request"]["ifp"], [0] * 6)
        self.assertEqual(result["provenance"]["sources"]["ifp"]["complex"]["sha256"], file_sha256(self.complex))

    def test_command_strings_are_not_feature_providers(self):
        for flag, provider_key in (("compute_properties", "property_provider"), ("extract_fragment", "fragment_provider"),
                                   ("compute_ifp", "ifp_provider")):
            with self.subTest(flag=flag), self.assertRaisesRegex(TypeError, "local Python callable"):
                prepare_inference_request(reference_smiles="CCO", complex_pdb=self.complex,
                                          **{flag: True, provider_key: "python external-tool.py"})

    def test_raw_fragment_literal_is_kept_without_chemical_repair(self):
        text = "n1cnnc1SC1CC(=O)N(c2ccc(Cl)cc(Cl)c2)C1=O"
        result = prepare_inference_request(fragment=text, fragment_representation="raw-smiles-v1")
        self.assertEqual(result["request"]["fragment"], text)
        self.assertEqual(result["provenance"]["fragment_representation"], "raw-smiles-v1")
        self.assertEqual(result["provenance"]["sources"]["fragment"]["chemical_validity"], "not_asserted_literal_token_input")
        with self.assertRaisesRegex(ValueError, "invalid for its declared representation"):
            prepare_inference_request(fragment=text)
        with self.assertRaisesRegex(ValueError, "unsupported text"):
            prepare_inference_request(fragment="C<", fragment_representation="raw-smiles-v1")

    def test_missing_fields_are_not_inferred_from_provider_registration(self):
        provider = Mock(side_effect=AssertionError("provider registration does not enable it"))
        result = prepare_inference_request(reference_smiles="CCO", complex_pdb=self.complex,
                                           property_provider=provider, fragment_provider=provider, ifp_provider=provider)
        self.assertEqual(result["request"], {})
        self.assertEqual(result["conditions"], [])
        provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
