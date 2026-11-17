# Generate molecules

[Overview](../README.md) · [Installation](INSTALLATION.md) · [Model card](MODEL_CARD.md) · [Input details](INPUTS.md)

Run commands from the repository root after [installation](INSTALLATION.md). The included checkpoint, `models/pocketgpt-v1.pt`, contains the trained weights, vocabulary, and model configuration. Provide condition values directly on the command line; a request JSON file is optional.

For the full Linux installation, set the interpreter and runtime paths in each new terminal:

```bash
export PG_RUNTIME="$PWD/.pocketgpt-runtime"
export PG_PYTHON="$PG_RUNTIME/envs/generator/bin/python"
```

For the generator-only `.venv` installation, use `export PG_PYTHON="$PWD/.venv/bin/python"` instead.

## Generate from your pocket PDB

Replace `/absolute/path/pocket.pdb` with your own pocket PDB path:

```bash
"$PG_PYTHON" -m pocketgpt generate \
  --checkpoint models/pocketgpt-v1.pt \
  --pocket /absolute/path/pocket.pdb \
  --encoder-config "$PG_RUNTIME/encoder.local.json" \
  --allow-unverified-pocket \
  --conditions pocket --count 10 --seed 42 --device cpu \
  --output outputs/pocket.csv
```

The generator writes `outputs/pocket.csv` and `outputs/pocket.summary.json`. The encoder stores the computed 128-dimensional vector under `"$PG_RUNTIME/pocket_cache"`; later runs with the same inputs and encoder reuse it. No training CSV or JSONL conversion is needed.

The pocket file must contain the protein atoms defining your intended binding site, not cavity-center points or an unselected whole protein. The command uses that cropped structure as its context. Add the corresponding full protein with `--protein /absolute/path/protein.pdb` to retain nearby atoms outside the crop; both files must use the same coordinate frame. The additional context can change the descriptor.

## Property and fragment example

Request QED 0.7, LogP 2.5, and one benzene fragment:

```bash
"$PG_PYTHON" -m pocketgpt generate \
  --checkpoint models/pocketgpt-v1.pt \
  --property qed=0.7 --property logp=2.5 \
  --fragment 'c1ccccc1' --conditions properties,fragment \
  --count 10 --seed 42 --device cpu \
  --output outputs/property-fragment.csv
```

Equivalent request-file form:

```bash
"$PG_PYTHON" -m pocketgpt generate \
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
"$PG_PYTHON" -m pocketgpt generate --checkpoint models/pocketgpt-v1.pt \
  --property qed=0.8 --conditions properties --count 10 \
  --output outputs/qed.csv

# Fragment only.
"$PG_PYTHON" -m pocketgpt generate --checkpoint models/pocketgpt-v1.pt \
  --fragment 'c1ccccc1' --conditions fragment --count 10 \
  --output outputs/fragment.csv

# No conditioning.
"$PG_PYTHON" -m pocketgpt generate --checkpoint models/pocketgpt-v1.pt \
  --conditions none --count 10 --output outputs/unconditional.csv
```

## Generate from a protein and pocket

After the [full Linux installation](INSTALLATION.md#full-linux-installation), provide the protein PDB, its corresponding pocket PDB, and the generated encoder configuration:

```bash
"$PG_PYTHON" -m pocketgpt generate \
  --checkpoint models/pocketgpt-v1.pt \
  --protein /absolute/path/protein.pdb \
  --pocket /absolute/path/pocket.pdb \
  --encoder-config "$PG_RUNTIME/encoder.local.json" \
  --allow-unverified-pocket \
  --conditions pocket --count 10 --device cpu \
  --output outputs/pocket.csv
```

The pipeline encodes the pocket once, caches the 128-dimensional vector, and passes it to the generator's pocket projector. It does not retrain DeeplyTough or prepare a training dataset. Add properties, a fragment, or IFP targets to use other condition combinations.

If you only have the pocket PDB, omit `--protein`. The encoder then uses the cropped pocket itself as structural context. Supplying the corresponding full protein retains nearby atoms outside the crop and can change the descriptor. The two files must use the same coordinate frame. A pocket file must contain protein atoms; a set of cavity-center points or an unselected whole protein does not define the intended binding pocket.

The included checkpoint uses historical cached descriptors whose exact encoder match has not been established. `--allow-unverified-pocket` explicitly allows inference in that case, emits a warning, and records the status in the output summary. It does not change the vector, relabel its source, or verify compatibility. A matched encoder signature or a [paired-vector verification record](#verify-an-encoder-against-training-vectors) allows generation without this flag. The flag does not override a conflict with a checkpoint that names a concrete encoder.

## Four-condition generation

For **pocket + properties + fragment + IFP**, add the targets to the real-pocket command:

```bash
"$PG_PYTHON" -m pocketgpt generate \
  --checkpoint models/pocketgpt-v1.pt \
  --pocket /absolute/path/pocket.pdb \
  --encoder-config "$PG_RUNTIME/encoder.local.json" \
  --property qed=0.7 --property logp=2.5 \
  --fragment 'c1ccccc1' \
  --ifp '{"hydrogen_bond":1,"hydrophobic":1,"pi_stacking":0,"salt_bridge":0,"pi_cation":0,"halogen_bond":0}' \
  --allow-unverified-pocket --conditions all \
  --count 10 --seed 42 --device cpu \
  --output outputs/pocket-four-conditions.csv
```

The IFP values specify a desired interaction pattern, not measured interactions. Edit the six binary values for your intended pattern. Omit an input and remove its name from `--conditions` to use a smaller combination. The main CLI reads `--ifp` as inline JSON; the `examples/generate.py` wrapper below reads it from a file.

### Use a stored pocket vector

If you already have a pocket descriptor, supply it with `--pocket-vector`. Accepted files are:

- A JSON array of 128 finite numbers. Its encoder identity is unknown.
- A JSON object containing `pocket_embedding` and, when known, `pocket_encoder_signature`. Preserve the signature produced by the encoder; do not invent one.
- A `.npy` vector. A matching adjacent `provenance.json` from the pocket cache supplies its encoder identity after the file checksum is checked.

The generator does not normalize or reorder the supplied vector. Values are read as float32 and passed to the pocket projector.

`examples/ifp.json` defines a desired interaction pattern using six names: `hydrogen_bond`, `hydrophobic`, `pi_stacking`, `salt_bridge`, `pi_cation`, `halogen_bond`. Edit the binary values to set your targets.

```bash
"$PG_PYTHON" examples/generate.py \
  --checkpoint models/pocketgpt-v1.pt \
  --request examples/property_fragment.json \
  --pocket-vector /absolute/path/my-pocket-vector.json \
  --ifp examples/ifp.json \
  --allow-unverified-pocket \
  --count 100 --device cuda \
  --output outputs/four-conditions.csv
```

Use `--device cpu` if no GPU is available. To encode a new pocket in this example, replace `--pocket-vector` with `--protein`, `--pocket`, and `--encoder-config` as above. The example's default request supplies properties and a fragment; pass `--conditions pocket` if you want only pocket conditioning.

For native request JSON, use `pocket_embedding`, `pocket_encoder_signature`, `properties`, `fragment`, `fragment_representation`, and `ifp` fields; then pass `--request my-request.json --conditions all` to the CLI. An IFP array, if used instead of names, follows the six-name order above. Metal complexes and water bridges are not model conditions.

## Verify an encoder against training vectors

This is a one-time check for the dataset maintainer, not a data-conversion step required for each user. Prepare a local JSONL file with at least three distinct matched pockets. Each row contains `pocket` with `protein_pdb` and `pocket_pdb` paths, and `pocket_embedding` copied from the corresponding original training record. Paths are relative to the JSONL file. Check the structure-to-vector correspondence before running:

```bash
"$PG_PYTHON" scripts/verify_pocket_encoder.py \
  --checkpoint models/pocketgpt-v1.pt \
  --encoder-config /absolute/path/encoder.local.json \
  --references /absolute/path/paired-pockets.jsonl \
  --output /absolute/path/pocket-compatibility.json
```

The script compares all 128 values without renormalization (`atol=1e-6`, `rtol=1e-5` by default). It writes a record only if every pair passes. The record binds the comparison to the checkpoint checksum and encoder profile; it contains no reference vectors or structure paths. Pass it to generation with `--pocket-compatibility /absolute/path/pocket-compatibility.json` instead of `--allow-unverified-pocket`. A maintainer can also include records under `pocket_compatibility` in the model's adjacent JSON metadata.

The current release does not include a verified encoder mapping. If a comparison fails, check the weights, pocket selection, preprocessing, and normalization before changing model parameters. Editing a signature is not a numerical compatibility check.

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

For pocket generation, use a request containing either `pocket_embedding` (with its recorded `pocket_encoder_signature`, if available) or `pocket={"protein_pdb": "...", "pocket_pdb": "..."}`. Pass the encoder configuration dictionary as `encoder` for raw structures. The API accepts `allow_unverified_pocket=True` for the historical checkpoint, or `pocket_compatibility="/absolute/path/pocket-compatibility.json"` for a verified mapping.

`examples/generate.py` wraps this API and accepts raw pocket inputs, a stored vector, and a named IFP file. `prepare-request` can calculate reference-ligand properties with RDKit. Fragment extraction requires a reference molecule and a configured local provider; observed IFP calculation requires a protein–ligand pose and an interaction-analysis provider. Desired properties, fragments, and IFP targets can be entered directly.

## Output and sampling settings

The CSV retains **all attempted samples**, including invalid SMILES. Fields are `sample_id`, `smiles`, `valid`, `canonical_smiles`, `logp`, `qed`, `tpsa`, `sas`, `fragment_match`, and `fragment_match_status`. Missing calculated values for invalid molecules remain empty. Sample IDs start at zero.

The companion `.summary.json` records attempted/valid/unique-valid counts, selected conditions, seed, checkpoint step, descriptor identity and compatibility status, and local checkpoint path.

Useful controls:

- `--temperature 1.0`: sampling temperature.
- `--top-k 20`: restrict sampling to the highest-scoring tokens; `0` disables top-k.
- `--batch-size 32`: generation batch size; lower it if GPU memory is limited.
- `--max-tokens 256`: maximum generated length, bounded by the trained context.
- `--seed 42`: reproducibility seed. Results can differ across devices, library versions, and batch sizes.

Use the output fields to filter invalid or duplicate molecules and compare generated properties and fragment matches with your targets. Binding interactions and biological activity require separate evaluation.
