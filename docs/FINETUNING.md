# Frozen-backbone fine-tuning

[Training guide](USAGE.md) · [Input contracts](INPUTS.md)

1. Pretrain on parent SMILES with property and single-fragment conditioning, using `configs/condition_pretrain.json`.
2. Prepare complex-grouped fine-tuning records with cached pocket descriptors and observed property/fragment/IFP labels.
3. Fine-tune using `configs/finetune.frozen.json` in a **new** run directory.

```bash
python -m pocketgpt train --config configs/finetune.frozen.json --device cuda
```

The recipe freezes inherited decoder/backbone parameters, while training the conditioning projectors, fragment encoder, modality embeddings, and semantic fusion. The vocabulary starts from the pretraining vocabulary and appends tokens observed in the fine-tuning training split. New token rows can learn; gradient masks and a separate zero-decay optimizer group protect inherited token rows, including against AdamW weight decay.

Early stopping evaluates every 500 steps and at epoch ends. `patience: 5` counts validation events without an improvement greater than `min_delta: 0.0001`, after at least one epoch. `best.pt` tracks the actual minimum validation loss, even when a smaller improvement does not reset patience. `last.pt` retains continuation state, including optimizer, batch position, RNG, and early-stopping state.

For exact resume of this same run:

```bash
python -m pocketgpt train --config configs/finetune.frozen.json \
  --resume /absolute/path/to/finetune-frozen/last.pt --device cuda
```

Exact resume requires the same configuration, data, vocabulary, implementation, and runtime contract; it does not restart an already early-stopped run or reset patience. To change the schedule or training policy, start a separately named warm-start run. Avoid concurrent jobs writing to one output directory.

The recipe uses historical cached descriptors with the signature `legacy-precomputed-unverified`. For new data, use one consistent encoder and record its signature; see [pocket compatibility](MODEL_CARD.md#pocket-compatibility). Training reads stored descriptors and leaves DeeplyTough frozen. For cluster execution, see [Slurm submission](USAGE.md#3-submit-training-through-slurm).

To export a model for generation:

```bash
python -m pocketgpt export-model --checkpoint /absolute/path/best.pt \
  --output weights/my-model.pt --model-id my-pocketgpt-model
```

The export includes weights, vocabulary, and model configuration. Keep the original full checkpoint for resuming training; it contains the optimizer and continuation state.
