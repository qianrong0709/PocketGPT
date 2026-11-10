"""Explicit, checkpointed freezing and validation-based stopping policies.

The molecular decoder is frozen by requires_grad, NOT by torch.no_grad:
gradients must still flow through it into the conditioning prefix.
"""
import hashlib
import math

import torch


BACKBONE_PREFIXES = ("tok_emb.", "type_emb.", "blocks.", "ln_f.", "head.")
BACKBONE_MODULES = ("tok_emb", "type_emb", "drop", "blocks", "ln_f", "head")
TOKEN_PARAMETERS = ("tok_emb.weight", "head.weight")


def is_backbone(name):
    # tok_emb is shared with fragment encoding and must also stay unchanged.
    return name == "pos_emb" or name.startswith(BACKBONE_PREFIXES)


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def training_options(recipe):
    settings = recipe["training"]
    freeze = settings.get("freeze_backbone", False)
    if type(freeze) is not bool:
        raise ValueError("freeze_backbone must be boolean")
    if freeze:
        if recipe["stage"] != "finetune":
            raise ValueError("freeze_backbone is only supported for finetune")
        if not any(recipe.get(key) for key in ("init_checkpoint", "condition_init_checkpoint")):
            raise ValueError("Frozen finetuning requires init_checkpoint or condition_init_checkpoint; never freeze a random backbone")
    interval = _integer(settings.get("validation_every_steps", 0), "validation_every_steps", 0)
    new_rows = settings.get("train_new_token_rows", False)
    if type(new_rows) is not bool:
        raise ValueError("train_new_token_rows must be boolean")
    if new_rows and (not freeze or not recipe.get("condition_init_checkpoint") or not recipe.get("vocabulary")):
        raise ValueError("train_new_token_rows requires frozen condition-aware finetuning and a source vocabulary")
    early = settings.get("early_stopping")
    if early is not None:
        if not isinstance(early, dict) or set(early) - {"patience", "min_delta", "min_epochs"}:
            raise ValueError("early_stopping needs patience, optional min_delta and min_epochs only")
        patience = _integer(early.get("patience"), "early_stopping.patience", 1)
        min_epochs = _integer(early.get("min_epochs", 0), "early_stopping.min_epochs", 0)
        if min_epochs >= int(settings.get("epochs", 10)):
            raise ValueError("early_stopping.min_epochs must be less than epochs")
        delta = early.get("min_delta", 0.0)
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
            raise ValueError("early_stopping.min_delta must be finite and nonnegative")
        early = {"patience": patience, "min_delta": float(delta), "min_epochs": min_epochs}
    return {"freeze_backbone": freeze, "validation_every_steps": interval, "early_stopping": early,
            "train_new_token_rows": new_rows}


def backbone_fingerprint(model, frozen_token_ids=None):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if is_backbone(name):
            value = parameter.detach().cpu().contiguous()
            if frozen_token_ids is not None and name in TOKEN_PARAMETERS:
                value = value[frozen_token_ids].contiguous()
            digest.update(name.encode() + b"\0")
            digest.update(str((tuple(value.shape), str(value.dtype))).encode() + b"\0")
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def configure_parameters(model, freeze, initialization=None, *, saved_policy=None,
                         tokenizer=None, train_new_token_rows=False):
    new_token_ids, frozen_token_ids = [], None
    if freeze and train_new_token_rows:
        if not initialization or initialization.get("method") != "condition-aware-token-mapping-v1" or tokenizer is None:
            raise ValueError("New-token row training requires verified condition-aware token mapping")
        copied = initialization.get("copied_tokens", [])
        new = initialization.get("new_tokens", [])
        if (initialization.get("source_only_tokens") or not copied or
                initialization.get("copied_token_count") != len(copied) or
                len(copied) + len(new) != model.config.vocab_size or
                len(set(copied) | set(new)) != model.config.vocab_size or
                set(copied) | set(new) != set(tokenizer.tokens)):
            raise ValueError("Frozen backbone must retain every source token exactly once")
        new_token_ids = [tokenizer.stoi[token] for token in new]
        if new_token_ids:
            frozen_token_ids = [tokenizer.stoi[token] for token in copied]
    if freeze and saved_policy is None:
        if not initialization:
            raise ValueError("Frozen finetuning requires verified checkpoint initialization")
        method = initialization.get("method")
        if method == "condition-aware-token-mapping-v1":
            if not train_new_token_rows and (initialization.get("new_tokens") or initialization.get("source_only_tokens") or
                    initialization.get("copied_token_count") != model.config.vocab_size):
                raise ValueError("Frozen backbone requires the unchanged source vocabulary; reuse its vocabulary.json")
        elif method != "strict-model-initialization-v1":
            raise ValueError("Frozen backbone requires same-architecture, fully inherited token embeddings")
    frozen, trainable = [], []
    frozen_numel = trainable_numel = 0
    partial = {}
    for name, parameter in model.named_parameters():
        old_hook = getattr(parameter, "_pocketgpt_row_mask_hook", None)
        if old_hook is not None:
            old_hook.remove()
            del parameter._pocketgpt_row_mask_hook
        parameter.requires_grad_(not (freeze and is_backbone(name)))
        if new_token_ids and name in TOKEN_PARAMETERS:
            parameter.requires_grad_(True)
            partial[name] = list(new_token_ids)
            # Only the appended rows learn. The mask follows the gradient
            # device so preflight may configure on CPU then move to CUDA.
            mask = torch.zeros(parameter.size(0), 1)
            mask[new_token_ids] = 1
            parameter._pocketgpt_row_mask_hook = parameter.register_hook(
                lambda gradient, mask=mask: gradient * mask.to(device=gradient.device, dtype=gradient.dtype))
            trainable.append(name)
            trainable_numel += len(new_token_ids) * parameter.size(1)
            frozen_numel += len(frozen_token_ids) * parameter.size(1)
        elif parameter.requires_grad:
            trainable.append(name)
            trainable_numel += parameter.numel()
        else:
            frozen.append(name)
            frozen_numel += parameter.numel()
    policy = {"version": 2, "freeze_backbone": freeze, "backbone_eval": freeze,
              "train_new_token_rows": train_new_token_rows, "trainable_token_rows": partial,
              "frozen_token_ids": frozen_token_ids,
              "frozen_parameter_names": frozen, "trainable_parameter_names": trainable,
              "frozen_numel": frozen_numel, "trainable_numel": trainable_numel,
              "frozen_backbone_sha256": backbone_fingerprint(model, frozen_token_ids) if freeze else None}
    if saved_policy is not None and saved_policy != policy:
        raise ValueError("Resume parameter policy or frozen backbone changed")
    if not trainable:
        raise ValueError("No trainable parameters remain")
    return policy


def set_training_mode(model, freeze):
    model.train()
    if freeze:
        # eval mode disables dropout in the fixed decoder, without disabling
        # autograd through it. Conditional encoders retain their own dropout.
        for name in BACKBONE_MODULES:
            getattr(model, name).eval()


def verify_frozen_backbone(model, policy):
    if policy["freeze_backbone"] and backbone_fingerprint(model, policy["frozen_token_ids"]) != policy["frozen_backbone_sha256"]:
        raise RuntimeError("Frozen backbone changed; refusing to publish a checkpoint")


def optimizer_parameter_groups(model, policy, weight_decay):
    partial = policy["trainable_token_rows"]
    groups = [{"params": [parameter for name, parameter in model.named_parameters()
                           if parameter.requires_grad and name not in partial], "weight_decay": weight_decay}]
    if partial:
        # AdamW decay would change old rows even when their gradients are zero.
        groups.append({"params": [parameter for name, parameter in model.named_parameters() if name in partial],
                       "weight_decay": 0.0})
    return groups


def verify_frozen_optimizer(model, optimizer, policy):
    parameters = dict(model.named_parameters())
    for name in policy["trainable_token_rows"]:
        parameter = parameters[name]
        for group in optimizer.param_groups:
            if any(value is parameter for value in group["params"]) and group["weight_decay"] != 0:
                raise ValueError("Partially frozen token parameters must not receive AdamW decay")
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = optimizer.state.get(parameter, {}).get(key)
            if value is not None and torch.count_nonzero(value[policy["frozen_token_ids"]]).item():
                raise ValueError("Frozen token rows have nonzero optimizer moments")


class EarlyStopping:
    """Count validation events, separately from the absolute best checkpoint.

    min_delta is cumulative relative to the last significant improvement.
    The minimum-duration guard never resets on resume.
    """
    def __init__(self, config, batches_per_epoch, state=None):
        self.config = config
        self.start_step = config["min_epochs"] * batches_per_epoch if config else 0
        self.best_score = None
        self.bad_validations = 0
        self.validations = 0
        self.last_step = 0
        self.stopped = False
        if state is not None:
            if state.get("config") != config or state.get("start_step") != self.start_step:
                raise ValueError("Resume early stopping configuration changed")
            for key in ("bad_validations", "validations", "last_step"):
                _integer(state.get(key), "early stopping state " + key, 0)
            score = state.get("best_score")
            if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
                raise ValueError("Invalid early stopping best_score")
            if type(state.get("stopped")) is not bool:
                raise ValueError("Invalid early stopping stopped flag")
            self.best_score, self.bad_validations = score, state["bad_validations"]
            self.validations, self.last_step = state["validations"], state["last_step"]
            self.stopped = state["stopped"]
            expected_stop = bool(config and self.last_step >= self.start_step and
                                 self.bad_validations >= config["patience"])
            if self.stopped != expected_stop or (self.validations == 0) != (self.best_score is None):
                raise ValueError("Inconsistent early stopping checkpoint state")

    def observe(self, loss, step):
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss; no checkpoint overwrite")
        if self.stopped or step <= self.last_step:
            raise ValueError("Early stopping requires a new validation step and a non-stopped run")
        self.validations += 1
        self.last_step = step
        delta = self.config["min_delta"] if self.config else 0.0
        if self.best_score is None or loss < self.best_score - delta:
            self.best_score = loss
            self.bad_validations = 0
        elif self.config and step >= self.start_step:
            self.bad_validations += 1
        else:
            self.bad_validations = 0
        self.stopped = bool(self.config and step >= self.start_step and
                            self.bad_validations >= self.config["patience"])
        return self.stopped

    def state_dict(self):
        return {"config": self.config, "start_step": self.start_step,
                "best_score": self.best_score, "bad_validations": self.bad_validations,
                "validations": self.validations, "last_step": self.last_step, "stopped": self.stopped}
