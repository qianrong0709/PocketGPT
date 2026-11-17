# PocketGPT v1 model card

[Overview](../README.md) · [Model file](../models/pocketgpt-v1.pt) · [Generation](GENERATION.md)

## Model specification

- **Model ID:** `pocketgpt-v1`; file: `models/pocketgpt-v1.pt`.
- **Architecture:** causal SMILES Transformer, 8 decoder layers, 8 attention heads, hidden dimension 256, context 256 tokens; 2-layer Transformer fragment encoder.
- **Conditions:** 128-dimensional pocket descriptor; LogP/QED/TPSA/SAS targets; one fragment SMILES; six binary IFP targets.
- **Vocabulary:** 81 tokens, stored in the checkpoint.
- **Fragment representation:** `raw-smiles-v1` (literal fragment tokenization).
- **Training:** property-plus-single-fragment conditional pretraining, followed by frozen-backbone four-condition fine-tuning with condition/property dropout and validation-based early stopping. Newly introduced token rows are trainable; inherited backbone/token rows remain frozen.
- **Selected checkpoint:** minimum validation loss, fine-tuning step 42,664 after 8 completed epochs.

The file contains model parameters, configuration, vocabulary, condition names, fragment representation, and pocket signature. It is an inference checkpoint. Resuming interrupted training requires the original full checkpoint, which also stores optimizer and RNG state.

## Generation

Select conditions individually or in combination. The [quick start](../README.md#installation-and-model-usage) uses molecular properties and a fragment. Outputs are molecular SMILES with calculated properties and fragment-match annotations.

## Pocket compatibility

This model was fine-tuned with cached 128-dimensional pocket vectors labelled `legacy-precomputed-unverified`. The original encoder provenance is incomplete, and compatibility with newly calculated DeeplyTough vectors has not been verified.

A new pocket vector must match the training feature space: encoder weights, preprocessing, feature order, and normalization. Compare new descriptors with the training pipeline before using them with this checkpoint. The raw-PDB adapter checks the encoder signature and rejects mismatches; changing the label does not make vectors compatible.

See [input formats](INPUTS.md#pocket-requirements) for the pocket interface and [installation](INSTALLATION.md) for the separate DeeplyTough environment.

## Training validation

Fine-tuning validation loss decreased from approximately 0.3601 at the first epoch end to 0.3383 at the selected checkpoint. Training and validation were split by **complex ID**, keeping all fragment records from one complex together. This is a complex-level split, not a protein- or sequence-cluster-held-out split.

The exported weights and vocabulary match the selected training checkpoint. Regression tests cover condition masking, feature handling, freezing, early stopping, and checkpoint loading.

For the study's benchmark and prospective PARP1 experiments, see the [paper](https://doi.org/10.59717/j.xinn-drugdisc.2026.100028).

## Working with generated molecules

Conditions guide sampling rather than enforce exact constraints. Check generated structures, properties, and fragment retention. Binding activity requires experimental validation.

Desired IFP targets can be entered directly. Calculating an observed IFP requires a protein–ligand pose and interaction analysis.
