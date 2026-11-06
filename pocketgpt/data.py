import hashlib
import json
import math
import mmap
from array import array
from pathlib import Path

import torch
from rdkit import Chem

from .config import (CONDITIONS, PROPERTIES, DEFAULT_FRAGMENT_REPRESENTATION,
                     validate_fragment_representation)
from .ifp import canonical_ifp
from .pocket import PocketResolver, validate_embedding, validate_pocket_sources
from .tokenizer import SmilesTokenizer

PROPERTY_SCALES = (5.0, 1.0, 100.0, 5.0)


class JsonlDataset:
    """Memory-mapped JSONL: large corpora do not become millions of Python dicts."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.offsets = array("Q")
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"Empty dataset: {self.path}")
        self._file = self.path.open("rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        start = self.offsets[index]
        end = self._map.find(b"\n", start)
        row = json.loads(self._map[start:None if end < 0 else end])
        if not isinstance(row, dict) or not row.get("smiles"):
            raise ValueError(f"Row {index} in {self.path} needs smiles")
        return row

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def close(self):
        if getattr(self, "_map", None) is not None:
            self._map.close()
            self._map = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


def canonical_smiles(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("SMILES must be a nonempty string")
    mol = Chem.MolFromSmiles(text.strip())
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Invalid SMILES: " + str(text)[:80])
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_fragment(text):
    """Use the same stereochemistry-preserving fragment form at every entry.

    This helper is also used when building the training vocabulary: a fragment
    must not acquire new token types only after batching or at inference time.
    Attachment-point atoms, if present in the supplied structure, are retained.
    """
    return canonical_smiles(text)


def fragment_text(text, representation=DEFAULT_FRAGMENT_REPRESENTATION):
    """One explicit fragment policy shared by preparation, train and inference.

    Raw mode preserves the supplied text and permits chemically invalid SMILES
    as symbolic conditioning labels. It does not repair, canonicalize or mask
    them. Tokenizer support, presence and context limits still apply.
    """
    representation = validate_fragment_representation(representation)
    if representation == DEFAULT_FRAGMENT_REPRESENTATION:
        return canonical_fragment(text)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Raw fragment must be a nonempty string")
    try:
        SmilesTokenizer.split(text)
    except ValueError:
        raise ValueError("Raw fragment contains unsupported tokenizer text") from None
    return text


def row_fragment_text(row, representation=DEFAULT_FRAGMENT_REPRESENTATION):
    representation = validate_fragment_representation(representation)
    value = row.get("fragment")
    if value in (None, ""):
        return None
    if "fragment_representation" in row:
        declared = validate_fragment_representation(row["fragment_representation"])
        if declared != representation:
            raise ValueError("Fragment record representation differs from the configured training/inference policy")
    return fragment_text(value, representation)


def property_vector(values):
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("Properties must be an object with named targets")
    if set(values) - set(PROPERTIES):
        raise ValueError("Unknown property names: " + repr(sorted(set(values) - set(PROPERTIES), key=str)))
    vector, mask = [], []
    for name, scale in zip(PROPERTIES, PROPERTY_SCALES):
        value = values.get(name)
        present = value is not None and not (isinstance(value, str) and not value.strip())
        if present and isinstance(value, bool):
            raise ValueError(f"Property targets must be numeric, not boolean: {name}")
        try:
            number = float(value) if present else 0.0
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"Invalid numeric property: {name}") from error
        if not math.isfinite(number):
            raise ValueError(f"Non-finite property: {name}")
        if present and ((name == "qed" and not 0 <= number <= 1) or
                        (name == "tpsa" and number < 0) or
                        (name == "sas" and not 1 <= number <= 10)):
            raise ValueError(f"Invalid property value: {name}")
        scaled = number / scale
        if abs(scaled) > torch.finfo(torch.float32).max:
            raise ValueError(f"Property exceeds finite float32 range: {name}")
        vector.append(scaled)
        mask.append(present)
    return vector, mask


class BatchBuilder:
    def __init__(self, tokenizer, block_size, conditions=CONDITIONS, encoder=None,
                 fragment_representation=DEFAULT_FRAGMENT_REPRESENTATION):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.conditions = set(conditions)
        self.resolver = PocketResolver(encoder)
        self.fragment_representation = validate_fragment_representation(fragment_representation)

    def __call__(self, rows, device="cpu", language_targets=True):
        n = len(rows)
        fragment_tokens = [None] * n
        if "fragment" in self.conditions:
            for index, row in enumerate(rows):
                text = row_fragment_text(row, self.fragment_representation)
                if text is not None:
                    fragment_tokens[index] = self.tokenizer.encode(text)
        fragment_size = max([1] + [len(tokens) for tokens in fragment_tokens if tokens is not None])
        if fragment_size > self.block_size:
            raise ValueError("Fragment exceeds context; do not truncate chemical tokens")
        batch = {
            "pocket": torch.zeros(n, 128), "properties": torch.zeros(n, 4),
            "property_mask": torch.zeros(n, 4, dtype=torch.bool),
            "ifp": torch.zeros(n, 6), "fragment_tokens": torch.zeros(n, fragment_size, dtype=torch.long),
            "condition_mask": torch.zeros(n, 4, dtype=torch.bool),
        }
        input_ids = torch.zeros(n, self.block_size, dtype=torch.long)
        targets = torch.zeros_like(input_ids)
        for i, row in enumerate(rows):
            if "pocket" in self.conditions:
                validate_pocket_sources(row)
            if language_targets:
                tokens = self.tokenizer.encode(row["smiles"])
                if len(tokens) + 1 > self.block_size:
                    raise ValueError("Molecule exceeds context; increase block_size rather than truncate SMILES")
                input_ids[i, :len(tokens) + 1] = torch.tensor([self.tokenizer.bos_id, *tokens])
                targets[i, :len(tokens) + 1] = torch.tensor([*tokens, self.tokenizer.eos_id])
            for j, name in enumerate(CONDITIONS):
                if name not in self.conditions:
                    continue
                if name == "pocket" and ("pocket_embedding" in row or row.get("pocket")):
                    batch["pocket"][i] = torch.from_numpy(self.resolver.resolve(row).copy())
                    batch["condition_mask"][i, j] = True
                elif name == "properties" and row.get("properties") is not None:
                    values, mask = property_vector(row["properties"])
                    batch["properties"][i] = torch.tensor(values)
                    batch["property_mask"][i] = torch.tensor(mask)
                    batch["condition_mask"][i, j] = any(mask)
                elif name == "fragment" and fragment_tokens[i] is not None:
                    tokens = fragment_tokens[i]
                    batch["fragment_tokens"][i, :len(tokens)] = torch.tensor(tokens)
                    batch["condition_mask"][i, j] = True
                elif name == "ifp" and row.get("ifp") is not None:
                    batch["ifp"][i] = torch.tensor(canonical_ifp(row["ifp"]), dtype=torch.float32)
                    batch["condition_mask"][i, j] = True
        batch = {key: value.to(device) for key, value in batch.items()}
        return input_ids.to(device), targets.to(device), batch


def inspect_manifest(dataset, conditional=False, conditions=None,
                     fragment_representation=DEFAULT_FRAGMENT_REPRESENTATION,
                     split_group="protein_id"):
    """Inspect effective labels using the same rules as BatchBuilder.

    ``conditions`` limits inspection to enabled inputs. In particular,
    molecule-only pretraining need not parse ignored conditional metadata, and
    disabling pockets never opens a structure or encoder file here.
    """
    selected = set(CONDITIONS if conditions is None else conditions)
    fragment_representation = validate_fragment_representation(fragment_representation)
    if split_group not in ("protein_id", "complex_id"):
        raise ValueError("split_group must be protein_id or complex_id")
    if selected - set(CONDITIONS):
        raise ValueError("Unknown conditions in manifest inspection")
    groups, molecules, properties, modalities, encoder_signatures = set(), set(), set(), set(), set()
    pocket_sources = set()
    for row in dataset:
        molecules.add(canonical_smiles(row["smiles"]))
        if conditional:
            group = row.get(split_group)
            if not isinstance(group, str) or not group.strip():
                if split_group == "protein_id":
                    raise ValueError("Conditional training requires protein_id for grouped splitting; PDB ids are not protein identities")
                raise ValueError("Conditional training with explicit complex_id grouping requires a nonempty complex_id; this is NOT protein-heldout validation")
            groups.add(group)
        if "properties" in selected:
            _, mask = property_vector(row.get("properties"))
            properties.update(name for name, present in zip(PROPERTIES, mask) if present)
            if any(mask):
                modalities.add("properties")
        if "pocket" in selected:
            validate_pocket_sources(row)
        if "pocket" in selected and ("pocket_embedding" in row or row.get("pocket")):
            modalities.add("pocket")
            if "pocket_embedding" in row:
                validate_embedding(row["pocket_embedding"])
                pocket_sources.add("embedding")
                encoder_signatures.add(row.get("pocket_encoder_signature", "legacy-precomputed-unverified"))
            else:
                pocket_sources.add("raw")
        if "fragment" in selected and row.get("fragment") not in (None, ""):
            row_fragment_text(row, fragment_representation)
            modalities.add("fragment")
        if "ifp" in selected and row.get("ifp") is not None:
            canonical_ifp(row["ifp"])
            modalities.add("ifp")
    return {"groups": groups, "molecules": molecules, "properties": properties,
            "modalities": modalities, "encoder_signatures": encoder_signatures,
            "pocket_sources": pocket_sources}
