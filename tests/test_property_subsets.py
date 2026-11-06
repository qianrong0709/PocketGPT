import itertools
import unittest

import torch

from pocketgpt.config import PROPERTIES, ModelConfig
from pocketgpt.data import BatchBuilder
from pocketgpt.model import PocketGPT
from pocketgpt.tokenizer import SmilesTokenizer


class PropertySubsetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(32)
        self.tokenizer = SmilesTokenizer.build(["CCO", "CCN", "COC", "N"])
        config = ModelConfig(len(self.tokenizer.tokens), block_size=16,
                             n_layer=1, n_head=2, n_embd=16,
                             fragment_layers=1, dropout=0.0)
        self.model = PocketGPT(config).eval()
        # A genuine zero target must remain distinguishable from a missing value.
        self.properties = {"logp": 0.0, "qed": 0.5, "tpsa": 40.0, "sas": 2.0}

    def test_all_property_subsets_ignore_missing_values_including_nan(self):
        for include_fragment in (False, True):
            conditions = ["properties"] + (["fragment"] if include_fragment else [])
            builder = BatchBuilder(self.tokenizer, 16, conditions)
            for flags in itertools.product((False, True), repeat=len(PROPERTIES)):
                with self.subTest(flags=flags, include_fragment=include_fragment):
                    row = {"smiles": "CCN", "properties": {
                        name: self.properties[name]
                        for name, enabled in zip(PROPERTIES, flags) if enabled
                    }}
                    if include_fragment:
                        row["fragment"] = "COC"
                    x, y, batch = builder([row])
                    self.assertEqual(batch["property_mask"].tolist(), [list(flags)])
                    self.assertEqual(bool(batch["condition_mask"][0, 1]), any(flags))
                    self.assertFalse(bool(batch["condition_mask"][0, 0]))
                    self.assertFalse(bool(batch["condition_mask"][0, 3]))

                    with torch.no_grad():
                        expected = self.model(x, batch, y)
                        hidden_nan = {key: value.clone() for key, value in batch.items()}
                        hidden_nan["properties"][~hidden_nan["property_mask"]] = float("nan")
                        actual = self.model(x, hidden_nan, y)

                    self.assertTrue(torch.isfinite(expected["loss"]))
                    self.assertTrue(torch.isfinite(actual["loss"]))
                    self.assertTrue(torch.isfinite(actual["logits"]).all())
                    torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)
                    torch.testing.assert_close(actual["gate_weights"], expected["gate_weights"],
                                               rtol=0, atol=0)
                    if not any(flags):
                        self.assertEqual(float(actual["gate_weights"][0, 1]), 0.0)
                    if not any(flags) and not include_fragment:
                        self.assertEqual(float(actual["gate_weights"].sum()), 0.0)

    def test_properties_and_fragment_ce_train_both_branches_and_gate(self):
        rows = [
            {"smiles": "CCN", "properties": self.properties, "fragment": "COC"},
            {"smiles": "CCO", "properties": {
                "logp": 2.0, "qed": 0.7, "tpsa": 20.0, "sas": 3.0,
            }, "fragment": "N"},
        ]
        x, y, batch = BatchBuilder(self.tokenizer, 16, ["properties", "fragment"])(rows)
        self.model.train()
        result = self.model(x, batch, y)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertTrue((result["gate_weights"][:, 1:3] > 0).all())
        self.assertEqual(float(result["gate_weights"][:, (0, 3)].sum().detach()), 0.0)
        result["loss"].backward()

        parameters = dict(self.model.named_parameters())
        # The scalar gate bias cancels in softmax; only gate.weight must be nonzero.
        # Single-modality batches are intentionally not required to train the gate.
        for name in (
            "property_projector.0.weight", "property_projector.2.weight",
            "fragment_projector.0.weight", "fragment_pos_emb.weight",
            "fragment_sequence_encoder.layers.0.self_attn.in_proj_weight",
            "gate.weight",
        ):
            with self.subTest(parameter=name):
                gradient = parameters[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0.0)
        for name, parameter in parameters.items():
            if parameter.grad is not None:
                with self.subTest(finite_gradient=name):
                    self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
