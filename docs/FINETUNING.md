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

The example recipe explicitly permits historical descriptors whose encoder identity is not verified. For new data, use a verified common encoder signature instead; a 128-dimensional shape alone is insufficient. Training uses stored descriptors and does not recompute all structures or update DeeplyTough. See [Slurm submission](USAGE.md#3-submit-training-through-slurm) for cluster execution.

To create a separate inference-only artifact from a trusted full checkpoint:

```bash
python -m pocketgpt export-model --checkpoint /absolute/path/best.pt \
  --output weights/my-model.pt --model-id my-pocketgpt-model
```

This exports weights and vocabulary without datasets, server paths, optimizer/RNG state, or private training reports. It does not alter the original checkpoint.
