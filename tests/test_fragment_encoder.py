import itertools
import unittest

import torch
from torch.nn import functional as F

from pocketgpt.config import CONDITIONS, ModelConfig
from pocketgpt.data import BatchBuilder
from pocketgpt.model import PocketGPT
from pocketgpt.tokenizer import SmilesTokenizer


class FragmentEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.tokenizer = SmilesTokenizer.build(["CCO", "COC", "CCN", "C(=O)O", "N"])
        self.config = ModelConfig(len(self.tokenizer.tokens), block_size=16,
                                  n_layer=1, n_head=2, n_embd=32, dropout=0.0)
        self.model = PocketGPT(self.config).eval()

    def fragment(self, text):
        tokens = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)
        return tokens, tokens.ne(0)

    def test_sequence_encoder_distinguishes_different_connectivity(self):
        alcohol, alcohol_mask = self.fragment("CCO")
        ether, ether_mask = self.fragment("COC")
        self.assertEqual(sorted(alcohol[0].tolist()), sorted(ether[0].tolist()))
        first = self.model.encode_fragment(alcohol, alcohol_mask)
        second = self.model.encode_fragment(ether, ether_mask)
        self.assertGreater(float((first - second).abs().max().detach()), 1e-4)
        builder = BatchBuilder(self.tokenizer, 16, ["fragment"])
        x, _, batch = builder([{"smiles": "CCN", "fragment": "CCO"},
                               {"smiles": "CCN", "fragment": "COC"}])
        prefix, _ = self.model.make_prefix(batch)
        self.assertGreater(float((prefix[0] - prefix[1]).abs().max().detach()), 1e-4)
        logits = self.model(x, batch)["logits"]
        self.assertGreater(float((logits[0] - logits[1]).abs().max().detach()), 1e-7)

    def test_padding_and_unrelated_batch_rows_do_not_change_representation(self):
        tokens, mask = self.fragment("CCO")
        expected = self.model.encode_fragment(tokens, mask)
        padded = F.pad(tokens, (0, 5))
        torch.testing.assert_close(expected, self.model.encode_fragment(padded, padded.ne(0)), rtol=1e-5, atol=1e-6)
        other, _ = self.fragment("C(=O)O")
        length = max(tokens.size(1), other.size(1))
        batch = torch.cat([F.pad(tokens, (0, length - tokens.size(1))),
                           F.pad(other, (0, length - other.size(1)))])
        torch.testing.assert_close(expected, self.model.encode_fragment(batch, batch.ne(0))[:1], rtol=1e-5, atol=1e-6)

    def test_missing_rows_do_not_enter_all_padding_attention(self):
        empty = torch.zeros(2, 1, dtype=torch.long)
        missing = self.model.encode_fragment(empty, empty.bool())
        self.assertTrue(torch.isfinite(missing).all())
        self.assertEqual(int(torch.count_nonzero(missing)), 0)
        builder = BatchBuilder(self.tokenizer, 16, ["fragment"])
        x, y, batch = builder([{"smiles": "CCN"}, {"smiles": "CCO", "fragment": "COC"}])
        self.model.train()
        result = self.model(x, batch, y)
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_disabled_fragment_does_not_read_its_tokens(self):
        x, _, batch = BatchBuilder(self.tokenizer, 16, [])([{"smiles": "CCN"}])
        expected = self.model(x, batch)["logits"]
        batch["fragment_tokens"] = torch.full((1, 3), len(self.tokenizer.tokens) + 100, dtype=torch.long)
        actual = self.model(x, batch)["logits"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sequence_parameters_receive_gradients(self):
        self.model.train()
        x, y, batch = BatchBuilder(self.tokenizer, 16, ["fragment"])(
            [{"smiles": "CCN", "fragment": "CCO"}])
        self.model(x, batch, y)["loss"].backward()
        gradients = dict(self.model.named_parameters())
        for name in ("fragment_pos_emb.weight", "fragment_sequence_encoder.layers.0.self_attn.in_proj_weight"):
            self.assertIsNotNone(gradients[name].grad)
            self.assertTrue(torch.isfinite(gradients[name].grad).all())
            self.assertGreater(float(gradients[name].grad.abs().sum()), 0.0)

    def test_all_condition_combinations_remain_safe(self):
        row = {"smiles": "CCO", "pocket_embedding": [0.1] * 128,
               "properties": {"logp": 0, "qed": 0.5}, "fragment": "COC", "ifp": [0] * 6}
        for flags in itertools.product((False, True), repeat=4):
            selected = [name for name, enabled in zip(CONDITIONS, flags) if enabled]
            x, y, batch = BatchBuilder(self.tokenizer, 16, selected)([row])
            result = self.model(x, batch, y)
            self.assertTrue(torch.isfinite(result["loss"]))
            for index, enabled in enumerate(flags):
                if not enabled:
                    self.assertEqual(float(result["gate_weights"][0, index].detach()), 0.0)

    def test_new_architecture_state_round_trip(self):
        restored = PocketGPT(ModelConfig.from_checkpoint_dict(self.config.to_dict())).eval()
        restored.load_state_dict(self.model.state_dict(), strict=True)
        x, _, batch = BatchBuilder(self.tokenizer, 16, ["fragment"])(
            [{"smiles": "CCN", "fragment": "COC"}])
        torch.testing.assert_close(self.model(x, batch)["logits"], restored(x, batch)["logits"], rtol=0, atol=0)

    def test_old_architecture_requires_explicit_legacy_mode(self):
        old = {key: value for key, value in self.config.to_dict().items()
               if key not in ("architecture_version", "fragment_encoder", "fragment_layers")}
        with self.assertRaisesRegex(ValueError, "explicit legacy"):
            ModelConfig.from_checkpoint_dict(old)
        legacy = ModelConfig.from_checkpoint_dict(old, allow_legacy=True)
        self.assertEqual(legacy.architecture_version, 1)
        self.assertEqual(legacy.fragment_encoder, "mean-v1")
        legacy_model = PocketGPT(legacy)
        self.assertFalse(any(key.startswith(("fragment_pos_emb.", "fragment_sequence_encoder."))
                             for key in legacy_model.state_dict()))
        tokens, mask = self.fragment("CCO")
        expected = (legacy_model.tok_emb(tokens) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        torch.testing.assert_close(legacy_model.encode_fragment(tokens, mask), expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "explicit legacy"):
            ModelConfig.from_checkpoint_dict(legacy.to_dict())

    def test_incomplete_or_inconsistent_architecture_is_rejected(self):
        incomplete = self.config.to_dict()
        incomplete.pop("fragment_layers")
        with self.assertRaisesRegex(ValueError, "missing architecture"):
            ModelConfig.from_checkpoint_dict(incomplete)
        for kwargs in ({"fragment_encoder": "mean-v1"}, {"fragment_layers": 0},
                       {"architecture_version": 1}, {"architecture_version": 3}):
            with self.assertRaises(ValueError):
                ModelConfig(self.config.vocab_size, **kwargs)


if __name__ == "__main__":
    unittest.main()
