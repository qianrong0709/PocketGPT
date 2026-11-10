# Generate molecules

[Overview](../README.md) · [Installation](INSTALLATION.md) · [Model card](MODEL_CARD.md) · [Input details](INPUTS.md)

Run commands from the repository root after installation:

```bash
python scripts/download_model.py
```

No training CSV or JSONL conversion is needed for inference. The model artifact contains its vocabulary and configuration; do not substitute another tokenizer.

## Property and fragment example

Request QED 0.7, LogP 2.5, and one benzene fragment:

```bash
python -m pocketgpt generate \
  --checkpoint weights/pocketgpt-v1.pt \
  --property qed=0.7 --property logp=2.5 \
  --fragment 'c1ccccc1' --conditions properties,fragment \
  --count 10 --seed 42 --device cpu \
  --output outputs/property-fragment.csv
```

Equivalent request-file form:

```bash
python -m pocketgpt generate \
  --checkpoint weights/pocketgpt-v1.pt \
  --request examples/property_fragment.json \
  --count 10 --seed 42 --device cpu \
  --output outputs/from-request.csv
```

Targets use raw values: LogP and QED are dimensionless, TPSA is in Å², and SAS uses the RDKit SA score convention. Omitted properties are masked rather than set to zero. The released fragment branch uses literal `raw-smiles-v1` token input. For meaningful fragment matching, supply a chemically valid fragment SMILES using supported vocabulary tokens.

The ten-sample example above was run on CPU with Python 3.11, PyTorch 2.11.0, NumPy 2.2.6, and RDKit 2026.3.1. All 10 sampled SMILES parsed, all 10 canonical molecules were distinct, and 9 contained the requested benzene substructure. CLI and Python-example outputs matched in that environment. This is a small execution check, not a benchmark estimate or a guarantee of constraint satisfaction.

## Choose condition combinations

`--conditions` accepts a comma-separated subset of `pocket,properties,fragment,ifp`, `all`, or `none`. When omitted, the CLI selects conditions for the inputs supplied. A selected condition must have an input value.

```bash
# QED only; the other property targets and modalities are omitted.
python -m pocketgpt generate --checkpoint weights/pocketgpt-v1.pt \
  --property qed=0.8 --conditions properties --count 10 \
  --output outputs/qed.csv

# Fragment only.
python -m pocketgpt generate --checkpoint weights/pocketgpt-v1.pt \
  --fragment 'c1ccccc1' --conditions fragment --count 10 \
  --output outputs/fragment.csv

# No conditioning.
python -m pocketgpt generate --checkpoint weights/pocketgpt-v1.pt \
  --conditions none --count 10 --output outputs/unconditional.csv
```

Supported combinations describe interfaces, not a guarantee of equal performance for every subset or target range.

## Four-condition generation

Provide **your own compatible** 128-number descriptor in `my-pocket-vector.json`. This file is a JSON array of finite numbers obtained from the same historical descriptor pipeline used for training. No training pocket vectors are supplied as examples, and zero/random vectors are not substitutes.

The provided `examples/ifp.json` is an illustrative **desired interaction pattern**, not an observed label from a research complex. It uses the six explicit names: `hydrogen_bond`, `hydrophobic`, `pi_stacking`, `salt_bridge`, `pi_cation`, `halogen_bond`.

```bash
python examples/generate.py \
  --checkpoint weights/pocketgpt-v1.pt \
  --request examples/property_fragment.json \
  --pocket-vector /absolute/path/my-pocket-vector.json \
  --ifp examples/ifp.json \
  --count 100 --device cuda \
  --output outputs/four-conditions.csv
```

Use `--device cpu` if no GPU is available. The vector example retains the model's explicit unverified-legacy provenance label; supplying a file does not verify encoder identity. See [pocket compatibility](MODEL_CARD.md#pocket-compatibility). Do not relabel an unrelated new vector to bypass a provenance error.

For native request JSON, use `pocket_embedding`, `pocket_encoder_signature`, `properties`, `fragment`, `fragment_representation`, and `ifp` fields; then pass `--request my-request.json --conditions all` to the CLI. An IFP array, if used instead of names, follows the six-name order above. Metal complexes and water bridges are not model conditions.

## Python API

```python
from pocketgpt.generation import generate

summary = generate(
    checkpoint="weights/pocketgpt-v1.pt",
    request={"properties": {"qed": 0.7, "logp": 2.5}, "fragment": "c1ccccc1"},
    output="outputs/python-api.csv",
    conditions="properties,fragment",
    count=10,
    seed=42,
    device="cpu",
)
print(summary["valid"], summary["attempted"])
```

`examples/generate.py` wraps this API and can add a compatible pocket vector and named IFP file. Optional `prepare-request` orchestration can calculate reference-ligand properties locally with RDKit. Fragment extraction requires an explicitly selected local extraction provider; observed IFP calculation requires a protein–ligand pose/complex and a local provider. Protein alone does not determine ligand properties or interaction labels.

## Output and sampling settings

The CSV retains **all attempted samples**, including invalid SMILES. Fields are `sample_id`, `smiles`, `valid`, `canonical_smiles`, `logp`, `qed`, `tpsa`, `sas`, `fragment_match`, and `fragment_match_status`. Missing calculated values for invalid molecules remain empty. Sample IDs start at zero.

The companion `.summary.json` records attempted/valid/unique-valid counts, selected conditions, seed, checkpoint step, and provenance information. It also contains a **local checkpoint path**; review it before sharing results.

Useful controls:

- `--temperature 1.0`: sampling temperature.
- `--top-k 20`: restrict sampling to the highest-scoring tokens; `0` disables top-k.
- `--batch-size 32`: generation batch size; lower it if GPU memory is limited.
- `--max-tokens 256`: maximum generated length, bounded by the trained context.
- `--seed 42`: reproducibility seed. Results can differ across devices, library versions, and batch sizes.

Conditioning is learned guidance, not a hard constraint. Check validity, duplicates, fragment retention, and property deviations before downstream use. IFP compliance, affinity, selectivity, synthesis, and safety require independent structural or experimental evaluation; a generated SMILES is not evidence of an active or safe compound.
