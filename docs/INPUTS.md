# Training and inference inputs

[Back to PocketGPT](../README.md)

This guide describes the refactored development package in `pocketgpt/`. Its new trained checkpoints and complete real-structure validation are not yet available. Legacy research scripts have different arguments and data layouts and are preserved only in the [archive branch](https://github.com/qianrong0709/PocketGPT/tree/codex/legacy-before-refactor-20261106/src). See [usage](USAGE.md) for commands supported by the refactored package.

## What will an inference user provide?

An inference request describes the molecule the user wants to generate. It is not a training dataset, and does not require preparing JSONL or assembling thousands of rows.

| Selected condition | User input | Processing inside the workflow |
| --- | --- | --- |
| Pocket | Protein PDB and the corresponding, explicitly selected pocket PDB; alternatively a compatible precomputed pocket vector with verified provenance | Preprocess and encode a new pocket with the configured DeeplyTough frontend, then cache the descriptor |
| Properties | One or more numerical targets: `logp`, `qed`, `tpsa`, `sas` | Validate the supplied values and apply the checkpoint's input scaling |
| Fragment | One fragment SMILES string | Tokenize and encode with the checkpoint's fragment representation |
| IFP | Six named binary interaction targets | Validate the schema and map names into the model's fixed order |

Only conditions selected for the request should participate in encoding and fusion. Supplying a property subset, such as QED alone, is different from supplying four targets with three zero values. Missing conditions are masked, not fabricated as observed zero-valued labels.

A usable checkpoint must have been trained for the requested conditions. Interface support for a combination does not establish its molecular control quality.

### Pocket requirements

- A complete protein file does not identify a unique binding pocket. Provide a corresponding pocket definition; automatic binding-site detection is not currently promised.
- Choose either raw pocket/protein files or a precomputed descriptor for one request, not both.
- A vector having the expected 128 dimensions is not sufficient evidence of compatibility. Encoder weights, preprocessing, feature order, and provenance must agree with training.
- Numerical compatibility between historical cached CSV vectors and newly encoded pockets remains a release gate for the refactored structure-input workflow.
- The pocket encoder is frozen. The generator learns its pocket projector and other conditional modules; it does not retrain DeeplyTough from the generation loss.

### Property targets

The four supported property names are `logp`, `qed`, `tpsa`, and `sas`. TPSA is expressed in square ångströms; QED and SAS are scores. Use raw target values, not hand-normalized embeddings. LogS is not an additional model condition in the current refactored configuration.

For example, a user can request QED = 0.7 and LogP = 2.5 while leaving TPSA and SAS unspecified. These are desired generation targets, not quantities inferred from the protein.

When a reference molecule is supplied, local feature tools may calculate its properties. Those measured/calculated reference features must remain distinguishable from user-selected targets.

### Interaction fingerprint

Use named fields to avoid ambiguous vector order:

```json
{
  "hydrogen_bond": 1,
  "hydrophobic": 1,
  "pi_stacking": 0,
  "salt_bridge": 0,
  "pi_cation": 0,
  "halogen_bond": 0
}
```

For array-based adapters, the fixed order is hydrogen bond, hydrophobic contact, π–π stacking, salt bridge, π–cation interaction, and halogen bond. Values are binary presence targets, not interaction counts or docking scores. A zero vector is an observed target pattern and must not be used to stand in for missing labels.

The historical source columns `hbonds`, `hydrophobic_contacts`, `pistacking`, `saltbridge`, `pication`, and `halogen_bonds` map to these six types. `metal_complexes` and `water_bridges` are excluded.

A desired IFP can be entered directly. Computing an observed IFP requires a ligand and its protein–ligand pose or complex; it cannot be determined from a protein structure alone. Fragment extraction likewise requires a reference molecule and a specified extraction rule, rather than an invented default.

## What data are used for training?

| Stage | Molecular sequence | Conditional labels | Not required |
| --- | --- | --- | --- |
| Current conditional pretraining | Parent-molecule SMILES | Molecular properties and one fragment | Protein structure, pocket vector, IFP |
| Structure–function fine-tuning | Parent-molecule SMILES | Cached pocket vector, properties, one fragment, six-type IFP, where observed | Recomputing every pocket or retraining DeeplyTough |

The current property-plus-fragment pretraining recipe is a development training choice. It should not be described as an independently verified, exact reproduction of every training detail in the publication.

Pretraining groups alternative SMILES/fragment records by canonical parent molecule before splitting. The current fine-tuning setup groups by complex ID; its validation results must not be described as unseen-protein or protein-cluster-held-out evaluation. Protein-level claims require an independently established identity/sequence grouping.

The selected pretraining policy preserves original fragment text for tokenization, including chemically problematic fragments, and records the issues. It does not silently repair, delete, or certify those labels. Parent-molecule, property, tokenizer, and context checks remain strict.

## Why JSON or JSONL may appear in developer documentation

- **CSV** can be the original tabular training source, including previously calculated pocket vectors.
- **JSONL** is an internal, row-wise training/cache representation in the refactored loader. Preparing it does not compute a new learned embedding or replace the model's embedding/projector modules.
- **JSON** can describe a programmatic inference request or configuration. A direct command-line interface can pass the same values without requiring users to write JSON themselves.

Training-cache preparation is a developer/data-maintainer step, not a prerequisite imposed on every inference user. The validated checkpoint release must provide a tested example that accepts ordinary structure files and optional target values.

## Outputs and interpretation

The intended output is sampled molecular SMILES with validity annotations and available calculated properties. Invalid samples must remain visible when measuring validity. Property, fragment, and IFP conditioning are learned guidance, not hard guarantees. Binding affinity and interaction recovery require independent evaluation; generated SMILES alone do not establish either.
