# Training and inference inputs

[Back to PocketGPT](../README.md)

This guide describes the `pocketgpt/` package. See the [generation guide](GENERATION.md) for runnable examples and the [model card](MODEL_CARD.md) for checkpoint compatibility.

## What will an inference user provide?

An inference request supplies the conditions for generating molecules. Pass them as command-line arguments or a request JSON file.

| Selected condition | User input | Processing inside the workflow |
| --- | --- | --- |
| Pocket | A 128-dimensional pocket vector, or a protein/pocket PDB pair plus an encoder configuration | Encode and cache raw structures if needed, then project the vector into the model's conditioning space |
| Properties | One or more numerical targets: `logp`, `qed`, `tpsa`, `sas` | Validate the supplied values and apply the checkpoint's input scaling |
| Fragment | One fragment SMILES string | Tokenize and encode with the checkpoint's fragment representation |
| IFP | Six named binary interaction targets | Validate the schema and map names into the model's fixed order |

Only selected conditions participate in encoding and fusion. Missing conditions and individual properties are masked. To request QED alone, supply its value and omit the other properties; zero is a target value, not a missing-value marker.

### Pocket requirements

`models/pocketgpt-v1.pt` uses historical cached pocket vectors with the signature `legacy-precomputed-unverified`. The signature is source metadata, not part of the 128-dimensional model input. Compatibility depends on the encoder weights, preprocessing, feature order, and normalization. It has not yet been established for newly encoded vectors. The historical checkpoint can be used with `--allow-unverified-pocket`, which records this status, or with a verified `--pocket-compatibility` record; see [pocket compatibility](MODEL_CARD.md#pocket-compatibility).

The PDB adapter accepts both a protein PDB and its corresponding pocket PDB. Choose this input pair or a cached vector, not both. The adapter encodes the selected pocket and caches its descriptor. Stored vectors can be supplied as a JSON array, a JSON object with `pocket_embedding` and its recorded `pocket_encoder_signature`, or a `.npy` file. A cache's adjacent `provenance.json` is checked when loading `.npy` vectors. Missing signatures remain unknown; the interface does not assign them a historical identity.

DeeplyTough is frozen. Training learns the pocket projector and conditional modules; generation only evaluates the trained model. No training JSONL conversion is required for generation.

### Property targets

The four property names are `logp`, `qed`, `tpsa`, and `sas`. TPSA is expressed in square ångströms; QED and SAS are scores. Supply raw target values; the model applies its stored scaling. LogS is not a model condition.

For example, request QED = 0.7 and LogP = 2.5 while leaving TPSA and SAS unspecified. Targets can be entered directly or calculated from a reference molecule with the local feature tools.

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

For arrays, the fixed order is hydrogen bond, hydrophobic contact, π–π stacking, salt bridge, π–cation interaction, and halogen bond. Values are binary presence targets, not counts or docking scores. An all-zero vector requests absence of all six types; omit the IFP condition when no target is specified.

The historical source columns `hbonds`, `hydrophobic_contacts`, `pistacking`, `saltbridge`, `pication`, and `halogen_bonds` map to these six types. `metal_complexes` and `water_bridges` are excluded.

A desired IFP can be entered directly. An observed IFP requires a protein–ligand pose or complex and interaction analysis. Fragment extraction requires a reference molecule and an extraction rule. Neither observed IFP nor ligand properties can be calculated from the protein alone.

## What data are used for training?

| Stage | Molecular sequence | Conditional labels | Not required |
| --- | --- | --- | --- |
| Conditional pretraining | Parent-molecule SMILES | Molecular properties and one fragment | Protein structure, pocket vector, IFP |
| Structure–function fine-tuning | Parent-molecule SMILES | Cached pocket vector, properties, one fragment, six-type IFP, where observed | Recomputing every pocket or retraining DeeplyTough |

Pretraining splits records by canonical parent molecule. Fine-tuning splits by complex ID, keeping records from the same complex together; this is not a protein-identity or sequence-cluster split.

The `raw-smiles-v1` policy preserves original fragment strings for tokenization and records chemical parsing issues. Parent molecules, properties, tokens, and context length are validated separately.

## File formats

- **CSV**: tabular training source, including precomputed pocket vectors.
- **JSONL**: row-wise training records prepared for the loader. Preparation preserves feature values; learned projections are applied by the model during training.
- **JSON**: configuration or inference request. Command-line arguments can supply the same inference values.

For generation, use the [command-line or Python examples](GENERATION.md); training-cache preparation is only needed when training a model.

## Outputs and interpretation

The output CSV contains sampled SMILES, validity, calculated properties, and fragment-match annotations. Invalid samples are retained. See [output fields and sampling settings](GENERATION.md#output-and-sampling-settings).
