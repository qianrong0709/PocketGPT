"""Synthetic-only regression checks for fixed decoders and resumable stopping."""
import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from pocketgpt.checkpoint import load_checkpoint, save_checkpoint, seed_all
from pocketgpt.config import CONDITIONS, ModelConfig
from pocketgpt.data import BatchBuilder
from pocketgpt.model import PocketGPT
from pocketgpt.tokenizer import SmilesTokenizer
from pocketgpt.training import train
from pocketgpt.training_control import (EarlyStopping, backbone_fingerprint, configure_parameters,
                                        is_backbone, set_training_mode, training_options,
                                        verify_frozen_backbone, optimizer_parameter_groups,
                                        verify_frozen_optimizer)


def tiny_model_config():
    return {"block_size": 16, "n_layer": 1, "n_head": 2, "n_embd": 16,
            "fragment_layers": 1, "dropout": 0.1}


def row(identifier, fragment="CO"):
    return {"smiles": "CCO", "complex_id": identifier, "pocket_embedding": [0.1] * 128,
            "properties": {"logp": 0.5, "qed": 0.7, "tpsa": 30, "sas": 2},
            "fragment": fragment, "fragment_representation": "raw-smiles-v1",
            "ifp": [1, 0, 1, 0, 0, 0]}


class TrainingControlTests(unittest.TestCase):
    def test_options_are_explicit_and_reject_untrained_freezing(self):
        base = {"stage": "finetune", "condition_init_checkpoint": "source.pt", "training": {"epochs": 3}}
        self.assertFalse(training_options(base)["freeze_backbone"])
        bad = [
            {"freeze_backbone": "true"}, {"validation_every_steps": True},
            {"train_new_token_rows": "true"}, {"train_new_token_rows": True},
            {"validation_every_steps": -1}, {"early_stopping": {}},
            {"early_stopping": {"patience": True}},
            {"early_stopping": {"patience": 0}},
            {"early_stopping": {"patience": 2, "min_delta": math.nan}},
            {"early_stopping": {"patience": 2, "min_delta": -1}},
            {"early_stopping": {"patience": 2, "min_epochs": 3}},
            {"early_stopping": {"patience": 2, "unknown": 3}},
        ]
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError):
                training_options(dict(base, training=dict(base["training"], **options)))
        for stage, initializer in (("pretrain", "source.pt"), ("finetune", None)):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                training_options({"stage": stage, "condition_init_checkpoint": initializer,
                                  "training": {"freeze_backbone": True}})

    def test_stopping_uses_validation_events_cumulative_delta_and_minimum_duration(self):
        config = {"patience": 2, "min_delta": 0.1, "min_epochs": 1}
        control = EarlyStopping(config, 10)
        for loss, step in ((1.0, 2), (1.2, 4), (1.3, 6), (1.1, 8)):
            self.assertFalse(control.observe(loss, step))
        self.assertEqual(control.bad_validations, 0)
        self.assertFalse(control.observe(0.95, 10))
        self.assertEqual(control.bad_validations, 1)
        # Two small improvements can add up to a significant improvement.
        self.assertFalse(control.observe(0.85, 12))
        self.assertEqual(control.bad_validations, 0)
        restored = EarlyStopping(config, 10, control.state_dict())
        self.assertFalse(restored.observe(0.9, 14))
        self.assertTrue(restored.observe(0.91, 16))
        self.assertEqual(restored.bad_validations, 2)
        self.assertTrue(EarlyStopping(config, 10, restored.state_dict()).stopped)
        with self.assertRaisesRegex(ValueError, "non-stopped"):
            restored.observe(0.5, 18)

    def test_stopping_rejects_nonfinite_and_changed_or_invalid_state(self):
        config = {"patience": 2, "min_delta": 0.0, "min_epochs": 0}
        control = EarlyStopping(config, 2)
        before = control.state_dict()
        for loss in (math.nan, math.inf):
            with self.assertRaises(FloatingPointError):
                control.observe(loss, 1)
            self.assertEqual(before, control.state_dict())
        control.observe(1.0, 1)
        with self.assertRaises(ValueError):
            control.observe(0.9, 1)
        saved = control.state_dict()
        for changed in (dict(saved, config=dict(config, patience=3)), dict(saved, start_step=4),
                        dict(saved, stopped=True), dict(saved, bad_validations=-1)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                EarlyStopping(config, 2, changed)

    def test_fixed_decoder_receives_no_weight_updates_but_passes_gradients(self):
        torch.set_num_threads(1)
        seed_all(42)
        tokenizer = SmilesTokenizer.build(["CCO", "CO"])
        model = PocketGPT(ModelConfig(vocab_size=len(tokenizer.tokens), **tiny_model_config()))
        report = {"method": "condition-aware-token-mapping-v1", "copied_token_count": len(tokenizer.tokens),
                  "new_tokens": [], "source_only_tokens": []}
        with self.assertRaisesRegex(ValueError, "verified checkpoint"):
            configure_parameters(model, True)
        with self.assertRaisesRegex(ValueError, "unchanged source vocabulary"):
            configure_parameters(model, True, dict(report, new_tokens=["Cl"]))
        policy = configure_parameters(model, True, report)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        parameters = [value for value in model.parameters() if value.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.1)
        set_training_mode(model, True)
        self.assertFalse(model.blocks.training)
        self.assertFalse(model.drop.training)
        self.assertFalse(model.tok_emb.training)
        self.assertTrue(model.fragment_sequence_encoder.training)
        self.assertTrue(model.training)
        builder = BatchBuilder(tokenizer, 16, CONDITIONS, fragment_representation="raw-smiles-v1")
        x, y, batch = builder([row("A"), row("B")], "cpu")
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(x, batch, y)["loss"].backward()
            for name in ("pocket_projector.0.weight", "ifp_projector.0.weight", "gate.weight",
                         "property_projector.0.weight", "fragment_projector.0.weight",
                         "fragment_sequence_encoder.layers.0.self_attn.in_proj_weight"):
                parameter = dict(model.named_parameters())[name]
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
            self.assertTrue(all(value.grad is None for name, value in model.named_parameters() if is_backbone(name)))
            optimizer.step()
        for name, value in model.named_parameters():
            if is_backbone(name):
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.pocket_projector[0].weight, before["pocket_projector.0.weight"]))
        self.assertFalse(torch.equal(model.ifp_projector[0].weight, before["ifp_projector.0.weight"]))
        self.assertEqual(policy["frozen_backbone_sha256"], backbone_fingerprint(model))
        self.assertEqual(sum(value.numel() for value in model.parameters()),
                         policy["frozen_numel"] + policy["trainable_numel"])
        self.assertEqual(policy, configure_parameters(model, True, saved_policy=policy))
        with torch.no_grad():
            model.tok_emb.weight[1, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "Frozen backbone changed"):
            verify_frozen_backbone(model, policy)
        with self.assertRaisesRegex(ValueError, "parameter policy"):
            configure_parameters(model, True, saved_policy=policy)


class FrozenFinetuneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        pretrain_rows = [dict(row("unused", "n1cnnc1"), smiles=value) for value in ("CCO", "CCN", "CCCl", "CCC", "CO")]
        for name, rows in (("pretrain", pretrain_rows[:4]), ("prevalid", pretrain_rows[4:]),
                           ("train", [row("train-" + str(i), fragment) for i, fragment in
                                      enumerate(("C", "CO", "N", "n1cnnc1", "CO", "C"))]),
                           ("valid", [row("valid-1", "C"), row("valid-2", "n1cnnc1")])):
            (self.root / (name + ".jsonl")).write_text("".join(json.dumps(value) + "\n" for value in rows))
        recipe = {"stage": "condition_pretrain", "conditions": "properties,fragment",
                  "fragment_representation": "raw-smiles-v1", "train_data": "pretrain.jsonl",
                  "valid_data": "prevalid.jsonl", "output_dir": "pretrained", "model": tiny_model_config(),
                  "training": {"epochs": 2, "batch_size": 2, "learning_rate": 0.001,
                               "cpu_threads": 1, "seed": 11}}
        config = self.root / "pretrain.json"
        config.write_text(json.dumps(recipe))
        self.source = self.quiet_train(config, max_steps=1)

    @staticmethod
    def quiet_train(*args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return train(*args, **kwargs)

    def recipe(self, name="frozen", **training_changes):
        value = {"stage": "finetune", "conditions": "all", "split_group": "complex_id",
                 "fragment_representation": "raw-smiles-v1", "allow_unverified_pocket_embeddings": True,
                 "train_data": "train.jsonl", "valid_data": "valid.jsonl", "output_dir": "runs/" + name,
                 "vocabulary": "pretrained/vocabulary.json", "condition_init_checkpoint": self.source,
                 "model": tiny_model_config(), "training": {"epochs": 4, "batch_size": 2,
                 "learning_rate": 0.001, "freeze_backbone": True, "validation_every_steps": 2,
                 "early_stopping": {"patience": 2, "min_delta": 0.0001, "min_epochs": 1},
                 "condition_dropout": 0.35, "property_dropout": 0.2, "save_every_steps": 2,
                 "cpu_threads": 1, "seed": 42}}
        value["training"].update(training_changes)
        path = self.root / (name + ".json")
        path.write_text(json.dumps(value))
        return path

    def test_actual_training_and_interval_resume_are_bitwise_exact(self):
        full = self.quiet_train(self.recipe("full"))
        config = self.recipe("resumed")
        partial = self.quiet_train(config, max_steps=2)
        full_model, _, full_state = load_checkpoint(full)
        self.quiet_train(config, resume=partial)
        resumed_model, _, resumed_state = load_checkpoint(partial)
        for key, tensor in full_model.state_dict().items():
            torch.testing.assert_close(tensor, resumed_model.state_dict()[key], rtol=0, atol=0, msg=key)
        for key in ("global_step", "best_loss", "early_stopping_state", "parameter_policy", "epoch", "next_batch"):
            self.assertEqual(full_state[key], resumed_state[key], key)
        torch.testing.assert_close(full_state["rng"]["torch"], resumed_state["rng"]["torch"], rtol=0, atol=0)
        for index, values in full_state["optimizer"]["state"].items():
            for key, tensor in values.items():
                torch.testing.assert_close(tensor, resumed_state["optimizer"]["state"][index][key], rtol=0, atol=0)
        source, tokenizer, _ = load_checkpoint(self.source)
        self.assertEqual(tokenizer.tokens, resumed_state["vocabulary"])
        self.assertEqual(backbone_fingerprint(source), backbone_fingerprint(resumed_model))
        policy = resumed_state["parameter_policy"]
        self.assertEqual(len(resumed_state["optimizer"]["param_groups"][0]["params"]), len(policy["trainable_parameter_names"]))
        self.assertNotIn("tok_emb.weight", policy["trainable_parameter_names"])
        metrics = [json.loads(line) for line in (self.root / "runs/resumed/metrics.jsonl").read_text().splitlines()]
        validations = [value for value in metrics if "validation_loss" in value]
        steps = [value["step"] for value in validations]
        self.assertEqual(len(steps), len(set(steps)))
        for value in validations:
            self.assertEqual(value["completed_epochs"], value["step"] // 3)
        self.assertEqual(steps[:4], [2, 3, 4, 6])

    def test_early_stop_persists_and_resume_never_resets_patience(self):
        config = self.recipe()
        with patch("pocketgpt.training.evaluate_validation", side_effect=[1.0, 1.05, 1.1]):
            checkpoint = self.quiet_train(config)
        _, _, saved = load_checkpoint(checkpoint)
        self.assertEqual(saved["global_step"], 4)
        self.assertEqual(saved["epoch"], 1)
        self.assertEqual(saved["next_batch"], 1)
        self.assertTrue(saved["early_stopping_state"]["stopped"])
        self.assertEqual(saved["early_stopping_state"]["bad_validations"], 2)
        _, _, best = load_checkpoint(Path(checkpoint).with_name("best.pt"))
        self.assertEqual(best["global_step"], 2)
        self.assertEqual(best["best_loss"], 1.0)
        with patch("pocketgpt.training.evaluate_validation", side_effect=AssertionError("should remain stopped")):
            self.quiet_train(config, resume=checkpoint)
        self.assertEqual(load_checkpoint(checkpoint)[2]["global_step"], 4)
        summary = json.loads((Path(checkpoint).parent / "training_summary.json").read_text())
        self.assertEqual(summary["status"], "early_stopped")

    def test_resuming_mid_patience_stops_at_same_point(self):
        early = {"patience": 3, "min_delta": 0.0001, "min_epochs": 0}
        reference = self.recipe("reference", early_stopping=early)
        with patch("pocketgpt.training.evaluate_validation", side_effect=[1.0, 1.1, 1.2, 1.3]):
            complete = self.quiet_train(reference)
        config = self.recipe(early_stopping=early)
        with patch("pocketgpt.training.evaluate_validation", side_effect=[1.0, 1.1, 1.2]):
            checkpoint = self.quiet_train(config, max_steps=4)
        self.assertEqual(load_checkpoint(checkpoint)[2]["early_stopping_state"]["bad_validations"], 2)
        with patch("pocketgpt.training.evaluate_validation", return_value=1.3):
            self.quiet_train(config, resume=checkpoint)
        expected, _, expected_state = load_checkpoint(complete)
        actual, _, actual_state = load_checkpoint(checkpoint)
        self.assertEqual(actual_state["global_step"], 6)
        self.assertEqual(actual_state["early_stopping_state"], expected_state["early_stopping_state"])
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(value, actual.state_dict()[name], rtol=0, atol=0)

    def test_best_checkpoint_keeps_small_improvement_even_when_stopping(self):
        config = self.recipe(early_stopping={"patience": 2, "min_delta": 0.1, "min_epochs": 0})
        with patch("pocketgpt.training.evaluate_validation", side_effect=[1.0, 0.99, 0.98]):
            checkpoint = self.quiet_train(config)
        _, _, saved = load_checkpoint(checkpoint)
        _, _, best = load_checkpoint(Path(checkpoint).with_name("best.pt"))
        self.assertTrue(saved["early_stopping_state"]["stopped"])
        self.assertEqual(saved["early_stopping_state"]["best_score"], 1.0)
        self.assertEqual(best["best_loss"], 0.98)
        self.assertEqual(best["global_step"], 4)

    def test_validation_nan_never_replaces_good_checkpoint(self):
        config = self.recipe()
        with patch("pocketgpt.training.evaluate_validation", side_effect=[1.0, math.nan]):
            with self.assertRaises(FloatingPointError):
                self.quiet_train(config)
        checkpoint = self.root / "runs/frozen/last.pt"
        self.assertEqual(load_checkpoint(checkpoint)[2]["global_step"], 2)

    def test_missing_or_tampered_policy_and_stopper_rejected_on_resume(self):
        config = self.recipe()
        checkpoint = self.quiet_train(config, max_steps=2)
        _, _, original = load_checkpoint(checkpoint)
        for key in ("parameter_policy", "early_stopping_state"):
            tampered = dict(original)
            del tampered[key]
            save_checkpoint(checkpoint, tampered)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.quiet_train(config, resume=checkpoint)
        tampered = dict(original, parameter_policy=dict(original["parameter_policy"], frozen_numel=0))
        save_checkpoint(checkpoint, tampered)
        with self.assertRaisesRegex(ValueError, "parameter policy"):
            self.quiet_train(config, resume=checkpoint)

    def test_frozen_preflight_checks_initializer_and_synthetic_gradient_without_run(self):
        path = Path(__file__).resolve().parents[1] / "scripts/server_preflight.py"
        spec = importlib.util.spec_from_file_location("frozen_preflight_test", path)
        preflight = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preflight)
        config = self.recipe()
        result = preflight.check_recipe(config, device="cpu", project_root=self.root)
        self.assertEqual(result["synthetic_smoke"], "finite_forward_backward")
        self.assertTrue(result["training_options"]["freeze_backbone"])
        self.assertGreater(result["parameter_counts"]["frozen_numel"], 0)
        self.assertFalse((self.root / "runs").exists())
        value = json.loads(config.read_text())
        value.pop("vocabulary")
        config.write_text(json.dumps(value))
        # Finetune-only vocabulary discards pretraining tokens; freezing must
        # reject that instead of freezing randomly initialized or missing rows.
        with self.assertRaisesRegex(ValueError, "unchanged source vocabulary"):
            preflight.check_recipe(config, device="cpu", project_root=self.root, run_smoke=False)
        self.assertFalse((self.root / "runs").exists())

    def add_new_token_rows(self):
        for name in ("train", "valid"):
            path = self.root / (name + ".jsonl")
            values = [json.loads(line) for line in path.read_text().splitlines()]
            values[0]["smiles"] = "C[N]"
            path.write_text("".join(json.dumps(value) + "\n" for value in values))

    def test_new_token_rows_train_while_every_inherited_backbone_bit_stays_fixed(self):
        self.add_new_token_rows()
        config = self.recipe(train_new_token_rows=True)
        captured = {}
        original = configure_parameters

        def capture(model, *args, **kwargs):
            policy = original(model, *args, **kwargs)
            captured.update({name: value.detach().clone() for name, value in model.named_parameters()
                             if name in policy["trainable_token_rows"]})
            return policy

        with patch("pocketgpt.training.configure_parameters", side_effect=capture):
            checkpoint = self.quiet_train(config, max_steps=2)
        model, tokenizer, saved = load_checkpoint(checkpoint)
        source, source_tokenizer, _ = load_checkpoint(self.source)
        self.assertEqual(tokenizer.tokens[:-1], source_tokenizer.tokens)
        self.assertEqual(tokenizer.tokens[-1], "[N]")
        policy = saved["parameter_policy"]
        self.assertEqual(policy["frozen_token_ids"], list(range(len(source_tokenizer.tokens))))
        self.assertEqual(policy["trainable_token_rows"],
                         {"tok_emb.weight": [len(source_tokenizer.tokens)], "head.weight": [len(source_tokenizer.tokens)]})
        self.assertEqual(backbone_fingerprint(source), backbone_fingerprint(model, policy["frozen_token_ids"]))
        self.assertEqual(policy["frozen_backbone_sha256"], backbone_fingerprint(source))
        for name, value in model.named_parameters():
            if is_backbone(name):
                expected = dict(source.named_parameters())[name]
                inherited = value[:len(source_tokenizer.tokens)] if name in policy["trainable_token_rows"] else value
                torch.testing.assert_close(inherited, expected, rtol=0, atol=0)
            if name in policy["trainable_token_rows"]:
                self.assertFalse(torch.equal(value[-1], captured[name][-1]), name)
        self.assertEqual(policy["frozen_numel"] + policy["trainable_numel"], sum(p.numel() for p in model.parameters()))
        self.assertEqual(saved["optimizer"]["param_groups"][1]["weight_decay"], 0.0)

    def test_new_token_rows_exact_resume_preserves_optimizer_masks_and_stopping(self):
        self.add_new_token_rows()
        complete = self.quiet_train(self.recipe("reference", train_new_token_rows=True))
        config = self.recipe(train_new_token_rows=True)
        checkpoint = self.quiet_train(config, max_steps=2)
        self.quiet_train(config, resume=checkpoint)
        model, _, saved = load_checkpoint(checkpoint)
        reference, _, expected = load_checkpoint(complete)
        self.assertEqual(saved["early_stopping_state"], expected["early_stopping_state"])
        self.assertEqual(saved["parameter_policy"], expected["parameter_policy"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
        for index, state in saved["optimizer"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, expected["optimizer"]["state"][index][name], rtol=0, atol=0)

    def test_new_token_optimizer_rejects_decay_and_nonzero_inherited_moments(self):
        self.add_new_token_rows()
        checkpoint = self.quiet_train(self.recipe(train_new_token_rows=True), max_steps=2)
        model, tokenizer, saved = load_checkpoint(checkpoint)
        policy = configure_parameters(model, True, saved["warm_start"], saved_policy=saved["parameter_policy"],
                                      tokenizer=tokenizer, train_new_token_rows=True)
        optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, policy, 0.01), lr=0.001)
        optimizer.load_state_dict(saved["optimizer"])
        verify_frozen_optimizer(model, optimizer, policy)
        optimizer.param_groups[1]["weight_decay"] = 0.01
        with self.assertRaisesRegex(ValueError, "AdamW decay"):
            verify_frozen_optimizer(model, optimizer, policy)
        optimizer.param_groups[1]["weight_decay"] = 0.0
        optimizer.state[model.tok_emb.weight]["exp_avg"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "optimizer moments"):
            verify_frozen_optimizer(model, optimizer, policy)

    def test_new_token_preflight_uses_training_union_and_checks_masked_gradients(self):
        self.add_new_token_rows()
        path = Path(__file__).resolve().parents[1] / "scripts/server_preflight.py"
        spec = importlib.util.spec_from_file_location("new_token_preflight_test", path)
        preflight = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preflight)
        result = preflight.check_recipe(self.recipe(train_new_token_rows=True), device="cpu", project_root=self.root)
        original_size = len(SmilesTokenizer.load(self.root / "pretrained/vocabulary.json").tokens)
        self.assertEqual(result["vocabulary_tokens"], original_size + 1)
        self.assertEqual(result["trainable_token_rows"],
                         {"tok_emb.weight": [original_size], "head.weight": [original_size]})
        self.assertEqual(result["synthetic_smoke"], "finite_forward_backward")
        self.assertFalse((self.root / "runs").exists())

    def test_new_tokens_are_training_only_and_disabled_option_stays_strict(self):
        self.add_new_token_rows()
        with self.assertRaisesRegex(ValueError, "absent from training vocabulary"):
            self.quiet_train(self.recipe())
        path = self.root / "valid.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()]
        values[0]["smiles"] = "C[SiH3]"
        path.write_text("".join(json.dumps(value) + "\n" for value in values))
        with self.assertRaisesRegex(ValueError, "absent from training vocabulary"):
            self.quiet_train(self.recipe("unseen", train_new_token_rows=True))


if __name__ == "__main__":
    unittest.main()
