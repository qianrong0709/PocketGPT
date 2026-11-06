"""Explicit conversion of interaction labels; never guess legacy bit order."""

import json

import numpy as np

from .config import IFP_NAMES


# Original eight-column order supplied by the author. The paper uses six types;
# metal coordination and water bridges are not model inputs.
LEGACY_IFP_COLUMNS = ("hydrophobic_contacts", "hbonds", "saltbridge", "pistacking",
                      "pication", "halogen_bonds", "metal_complexes", "water_bridges")
IGNORED_IFP_COLUMNS = ("metal_complexes", "water_bridges")
IFP_COLUMN_ALIASES = {
    "hydrogen_bond": ("hydrogen_bond", "hbonds"),
    "hydrophobic": ("hydrophobic", "hydrophobic_contacts"),
    "pi_stacking": ("pi_stacking", "pistacking"),
    "salt_bridge": ("salt_bridge", "saltbridge"),
    "pi_cation": ("pi_cation", "pication"),
    "halogen_bond": ("halogen_bond", "halogen_bonds"),
}


def has_named_ifp_columns(row):
    aliases = {alias for names in IFP_COLUMN_ALIASES.values() for alias in names}
    return any(isinstance(name, str) and name.strip().lower() in aliases for name in row)


def _is_missing_label(value):
    return (value is None or (isinstance(value, str) and not value.strip()) or
            (isinstance(value, (float, np.floating)) and bool(np.isnan(value))))


def ifp_from_columns(row, *, allow_missing=False):
    """Select the six paper interactions by column name, never by CSV position.

    Non-IFP metadata and the two excluded interaction columns are not read.
    With allow_missing, six unavailable labels return None; the caller must use
    has_named_ifp_columns to distinguish this from an absent column schema.
    Partial labels always raise rather than inventing absence labels.
    """
    normalized = {}
    for key, value in row.items():
        if not isinstance(key, str):
            continue
        name = key.strip().lower()
        if name in normalized:
            raise ValueError("Duplicate normalized CSV column: " + name)
        normalized[name] = value
    selected = {}
    for name, aliases in IFP_COLUMN_ALIASES.items():
        found = [alias for alias in aliases if alias in normalized]
        if len(found) > 1:
            raise ValueError("Multiple columns refer to the same IFP type: " + name)
        if found:
            selected[name] = normalized[found[0]]
    if not selected:
        return None
    missing = set(IFP_NAMES) - set(selected)
    if missing:
        raise ValueError("Missing IFP interaction columns: " + ", ".join(sorted(missing)))
    unavailable = [_is_missing_label(selected[name]) for name in IFP_NAMES]
    if all(unavailable) and allow_missing:
        return None
    if any(unavailable):
        raise ValueError("IFP labels must be all six observed values or all missing; do not zero-fill missing labels")
    return canonical_ifp([selected[name] for name in IFP_NAMES])


def validate_ifp_schema(schema):
    """Map each of our six named interactions to its zero-based legacy index."""
    if not isinstance(schema, dict) or set(schema) != {"source_dim", "indices"}:
        raise ValueError("IFP schema needs exactly source_dim and indices")
    if type(schema["source_dim"]) is not int or schema["source_dim"] != 8:
        raise ValueError("Legacy IFP source_dim must be 8")
    indices = schema["indices"]
    if not isinstance(indices, dict) or set(indices) != set(IFP_NAMES):
        raise ValueError("IFP indices must map all six named interaction types")
    ordered = [indices[name] for name in IFP_NAMES]
    if any(type(index) is not int or not 0 <= index < 8 for index in ordered):
        raise ValueError("IFP source indices must be integers from 0 to 7; confirm the old bit order first")
    if len(set(ordered)) != 6:
        raise ValueError("Each selected IFP source index must be distinct")
    return {"source_dim": 8, "indices": dict(zip(IFP_NAMES, ordered))}


def canonical_ifp(values, schema=None):
    """Return six binary values, discarding only explicitly unselected indices.

    Eight-bit arrays always require a schema. Six-bit arrays and dictionaries
    use the canonical named order and cannot be relabelled by a legacy schema.
    """
    if schema is not None:
        schema = validate_ifp_schema(schema)
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except json.JSONDecodeError as error:
            raise ValueError("IFP must be a JSON binary vector or named object") from error
    named = isinstance(values, dict)
    if named:
        if schema is not None:
            raise ValueError("An 8-bit IFP schema can only be applied to an 8-bit source vector")
        allowed = {alias for aliases in IFP_COLUMN_ALIASES.values() for alias in aliases} | set(IGNORED_IFP_COLUMNS)
        if any(not isinstance(name, str) or name.strip().lower() not in allowed for name in values):
            raise ValueError("Unknown IFP interaction name")
        selected = ifp_from_columns(values)
        if selected is None:
            raise ValueError("IFP dictionary needs all six named interaction types")
        return selected
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("IFP must contain binary presence/absence values") from error
    if vector.ndim != 1 or vector.size not in (6, 8) or not np.isin(vector, [0, 1]).all():
        raise ValueError("IFP must be a binary vector of length 6 or explicitly mapped length 8")
    if schema is not None:
        if named or vector.size != 8:
            raise ValueError("An 8-bit IFP schema can only be applied to an 8-bit source vector")
        vector = vector[[schema["indices"][name] for name in IFP_NAMES]]
    elif vector.size == 8:
        raise ValueError("8-bit IFP requires --ifp-schema; do not guess which two bits to drop")
    return vector.astype(int).tolist()
