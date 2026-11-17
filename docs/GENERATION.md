# Generate molecules

[Overview](../README.md) · [Installation](INSTALLATION.md) · [Model card](MODEL_CARD.md) · [Input details](INPUTS.md)

Run commands from the repository root after [installation](INSTALLATION.md). The included checkpoint, `models/pocketgpt-v1.pt`, contains the trained weights, vocabulary, and model configuration. Provide condition values directly on the command line or in a request JSON file.

## Property and fragment example

Request QED 0.7, LogP 2.5, and one benzene fragment:

```bash
python -m pocketgpt generate \
  --checkpoint models/pocketgpt-v1.pt \
  --property qed=0.7 --property logp=2.5 \
  --fragment 'c1ccccc1' --conditions properties,fragment \
  --count 10 --seed 42 --device cpu \
  --output outputs/property-fragment.csv
```

Equivalent request-file form:

```bash
python -m pocketgpt generate \
  --checkpoint models/pocketgpt-v1.pt \
  --request examples/property_fragment.json \
  --count 10 --seed 42 --device cpu \
  --output outputs/from-request.csv
```

Targets use raw values: LogP and QED are dimensionless, TPSA is in Å², and SAS uses the RDKit SA score convention. Omitted properties are masked. The fragment encoder uses `raw-smiles-v1` token input; supply a valid fragment SMILES supported by the checkpoint vocabulary.

## Choose condition combinations

`--conditions` accepts a comma-separated subset of `pocket,properties,fragment,ifp`, `all`, or `none`. When omitted, the CLI selects conditions for the inputs supplied. A selected condition must have an input value.

```bash
# QED only; the other property targets and modalities are omitted.
python -m pocketgpt generate --checkpoint models/pocketgpt-v1.pt \
  --property qed=0.8 --conditions properties --count 10 \
  --output outputs/qed.csv

# Fragment only.
python -m pocketgpt generate --checkpoint models/pocketgpt-v1.pt \
  --fragment 'c1ccccc1' --conditions fragment --count 10 \
  --output outputs/fragment.csv

# No conditioning.
python -m pocketgpt generate --checkpoint models/pocketgpt-v1.pt \
  --conditions none --count 10 --output outputs/unconditional.csv
```

## Four-condition generation

Save a compatible pocket descriptor as `my-pocket-vector.json`: a JSON array of 128 finite numbers from the same descriptor pipeline used for training. Compatibility between the checkpoint's historical cached descriptors and newly encoded DeeplyTough vectors has not been verified; see [pocket compatibility](MODEL_CARD.md#pocket-compatibility).

`examples/ifp.json` defines a desired interaction pattern using six names: `hydrogen_bond`, `hydrophobic`, `pi_stacking`, `salt_bridge`, `pi_cation`, `halogen_bond`. Edit the binary values to set your targets.

```bash
python examples/generate.py \
  --checkpoint models/pocketgpt-v1.pt \
  --request examples/property_fragment.json \
  --pocket-vector /absolute/path/my-pocket-vector.json \
  --ifp examples/ifp.json \
  --count 100 --device cuda \
  --output outputs/four-conditions.csv
```

Use `--device cpu` if no GPU is available. The example uses the checkpoint's `legacy-precomputed-unverified` descriptor signature. Keep the actual encoder identity when preparing new vectors.

For native request JSON, use `pocket_embedding`, `pocket_encoder_signature`, `properties`, `fragment`, `fragment_representation`, and `ifp` fields; then pass `--request my-request.json --conditions all` to the CLI. An IFP array, if used instead of names, follows the six-name order above. Metal complexes and water bridges are not model conditions.

## Python API

```python
from pocketgpt.generation import generate

summary = generate(
    checkpoint="models/pocketgpt-v1.pt",
    request={"properties": {"qed": 0.7, "logp": 2.5}, "fragment": "c1ccccc1"},
    output="outputs/python-api.csv",
    conditions="properties,fragment",
    count=10,
    seed=42,
    device="cpu",
)
print(summary["valid"], summary["attempted"])
```

`examples/generate.py` wraps this API and accepts a pocket vector and named IFP file. `prepare-request` can calculate reference-ligand properties with RDKit. Fragment extraction requires a reference molecule and a configured local provider; observed IFP calculation requires a protein–ligand pose and an interaction-analysis provider. Desired properties, fragments, and IFP targets can be entered directly.

## Output and sampling settings

The CSV retains **all attempted samples**, including invalid SMILES. Fields are `sample_id`, `smiles`, `valid`, `canonical_smiles`, `logp`, `qed`, `tpsa`, `sas`, `fragment_match`, and `fragment_match_status`. Missing calculated values for invalid molecules remain empty. Sample IDs start at zero.

The companion `.summary.json` records attempted/valid/unique-valid counts, selected conditions, seed, checkpoint step, descriptor signature, and local checkpoint path.

Useful controls:

- `--temperature 1.0`: sampling temperature.
- `--top-k 20`: restrict sampling to the highest-scoring tokens; `0` disables top-k.
- `--batch-size 32`: generation batch size; lower it if GPU memory is limited.
- `--max-tokens 256`: maximum generated length, bounded by the trained context.
- `--seed 42`: reproducibility seed. Results can differ across devices, library versions, and batch sizes.

Use the output fields to filter invalid or duplicate molecules and compare generated properties and fragment matches with your targets. Binding interactions and biological activity require separate evaluation.
