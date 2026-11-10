# PocketGPT v1 model card

[Overview](../README.md) · [Download](https://github.com/qianrong0709/PocketGPT/releases/tag/v1.0.0) · [Generation](GENERATION.md)

## Model specification

- **Model ID:** `pocketgpt-v1`; inference artifact: `pocketgpt-v1.pt`.
- **Architecture:** causal SMILES Transformer, 8 decoder layers, 8 attention heads, hidden dimension 256, context 256 tokens; 2-layer Transformer fragment encoder.
- **Conditions:** 128-dimensional pocket descriptor; LogP/QED/TPSA/SAS targets; one fragment SMILES; six binary IFP targets.
- **Vocabulary:** 81 tokens, embedded in the artifact and kept with the learned token rows.
- **Fragment representation:** `raw-smiles-v1` (literal fragment tokenization).
- **Training:** property-plus-single-fragment conditional pretraining, followed by frozen-backbone four-condition fine-tuning with condition/property dropout and validation-based early stopping. Newly introduced token rows are trainable; inherited backbone/token rows remain frozen.
- **Selected checkpoint:** minimum validation loss, fine-tuning step 42,664 after 8 completed epochs.

The artifact contains the learned generator parameters, vocabulary, architecture, condition names, fragment policy, and pocket signature. It does not contain optimizer state or RNG state and is **not an exact-resume training checkpoint**. Use the original full training checkpoint to resume an interrupted run.

## Intended use

Local molecular candidate generation for research. Users can select learned condition subsets; the quick start uses properties and a fragment, without requiring a pocket descriptor or external encoder. This model generates 2D molecular strings, not binding poses or measured activities.

## Pocket compatibility

This checkpoint was fine-tuned with existing 128-dimensional descriptors carrying the explicit signature `legacy-precomputed-unverified`. That label records incomplete historical encoder provenance; it does not verify that an arbitrary vector belongs to the same feature space.

Use descriptors from the corresponding historical pipeline only when their compatibility is established. Newly calculated DeeplyTough descriptors require a numerical/provenance comparison with the training feature space before they can be treated as interchangeable. The raw-PDB adapter is implemented separately, but its profile signature intentionally does not match this historical-vector label. Do not edit signatures to bypass that check.

For a new, fully provenance-identified pocket workflow, retain the encoder weights, preprocessing settings, feature order, normalization, and runtime/source profile with the training descriptors and resulting generator checkpoint.

## Evaluation scope

Fine-tuning validation loss decreased from approximately 0.3601 at the first epoch end to 0.3383 at the selected checkpoint. The split groups by **complex ID**, not protein identity or sequence cluster. It prevents the same complex's expanded fragment records from crossing the split; it does not establish unseen-protein generalization.

The model export is checked for finite parameters, strict loading, and exact tensor/vocabulary preservation. Regression tests check masking, feature contracts, freezing, early stopping, and checkpoint I/O. Small generation examples verify execution, not benchmark performance.

Results reported in the [PocketGPT paper](https://doi.org/10.59717/j.xinn-drugdisc.2026.100028), including the prospective PARP1 study, belong to the paper's experimental workflow. This versioned artifact should not be treated as independent reproduction of those experimental or benchmark results.

## Limitations

Property/fragment/IFP targets are learned guidance. Molecular validity, desired properties, fragment retention, novelty, affinity, selectivity, synthetic feasibility, and safety are not guaranteed. An observed IFP requires a protein–ligand pose and separate interaction analysis; it is not calculated from the protein alone. Independent validation is required before synthesis or biological use. No clinical use is established.

Only load checkpoints from trusted sources. PocketGPT loads this artifact using PyTorch's `weights_only=True`; legacy third-party encoders may use older deserialization behavior and must be reviewed separately.
