import csv
import json
from pathlib import Path

import torch
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer

from .checkpoint import load_checkpoint, seed_all, checkpoint_fragment_representation
from .config import CONDITIONS, parse_conditions
from .data import BatchBuilder
from .pocket import LEGACY_SIGNATURE, validate_pocket_sources


def generate(checkpoint, request, output, conditions="all", count=100,
             temperature=1.0, top_k=20, max_tokens=None, batch_size=32, seed=42, device="cpu", encoder=None):
    validate_pocket_sources(request)
    if count < 1 or temperature <= 0 or batch_size < 1 or top_k < 0:
        raise ValueError("Invalid generation count/temperature/batch_size/top_k")
    output = Path(output)
    if output.exists() or output.with_suffix(".summary.json").exists():
        raise FileExistsError("Generation output exists; choose a new filename")
    model, tokenizer, state = load_checkpoint(checkpoint, device)
    seed_all(seed)
    torch.set_num_threads(2)
    model.eval()
    selected = parse_conditions(conditions)
    missing = set(selected) - set(state["trained_conditions"])
    if missing:
        raise ValueError("Checkpoint was not trained for these conditions: " + repr(sorted(missing)))
    if "properties" in selected:
        unknown = set(request.get("properties", {})) - set(state["trained_properties"])
        if unknown:
            raise ValueError("Property targets absent from checkpoint training: " + repr(sorted(unknown)))
    encoder_config = encoder or state["recipe"].get("pocket_encoder")
    fragment_representation = checkpoint_fragment_representation(state)
    builder = BatchBuilder(tokenizer, model.config.block_size, selected, encoder_config,
                           fragment_representation=fragment_representation)
    if "pocket" in selected:
        if builder.resolver.row_signature(request) != state.get("pocket_encoder_signature"):
            raise ValueError("Pocket input does not match checkpoint provenance. Verify against original CSV descriptors before using it")
        if (state.get("training_contract") and state.get("pocket_encoder_signature") == LEGACY_SIGNATURE
                and not state["recipe"].get("allow_unverified_pocket_embeddings", False)):
            raise ValueError("Checkpoint does not explicitly allow unverified legacy pocket embeddings")
    _, _, single = builder([request], device, language_targets=False)
    actual = {name for i, name in enumerate(CONDITIONS) if bool(single["condition_mask"][0, i])}
    if set(selected) - actual:
        raise ValueError("Selected conditions have no input values: " + repr(sorted(set(selected) - actual)))
    max_tokens = model.config.block_size if max_tokens is None else max_tokens
    if not 1 <= max_tokens <= model.config.block_size:
        raise ValueError("max_tokens exceeds the trained context")
    with rdBase.BlockLogs():
        query = Chem.MolFromSmiles(request["fragment"]) if "fragment" in selected else None
    if query is not None and not query.GetNumAtoms():
        query = None
    query_unavailable = "fragment" in selected and query is None
    records = []
    with torch.inference_mode():
        for offset in range(0, count, batch_size):
            size = min(batch_size, count - offset)
            batch = {key: value.expand(size, *value.shape[1:]).contiguous() for key, value in single.items()}
            ids = torch.full((size, 1), tokenizer.bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(size, dtype=torch.bool, device=device)
            for _ in range(max_tokens):
                logits = model(ids, batch)["logits"][:, -1, :] / temperature
                logits[:, tokenizer.pad_id] = -torch.inf
                logits[:, tokenizer.bos_id] = -torch.inf
                if top_k:
                    threshold = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                next_token = torch.multinomial(logits.softmax(-1), 1).squeeze(1)
                next_token = torch.where(finished, tokenizer.eos_id, next_token)
                ids = torch.cat([ids, next_token[:, None]], 1)
                finished |= next_token.eq(tokenizer.eos_id)
                if bool(finished.all()):
                    break
            for sequence in ids.cpu().tolist():
                text = tokenizer.decode(sequence[1:])
                with rdBase.BlockLogs():
                    mol = Chem.MolFromSmiles(text) if text else None
                record = {"sample_id": len(records), "smiles": text, "valid": mol is not None,
                          "canonical_smiles": Chem.MolToSmiles(mol) if mol is not None else "",
                          "fragment_match": "unavailable" if query_unavailable else "",
                          "fragment_match_status": ("unavailable_invalid_fragment_query" if query_unavailable else
                                                    "unavailable_invalid_generated_molecule" if query is not None and mol is None else
                                                    "available" if query is not None else "not_requested")}
                if mol is not None:
                    record.update(logp=Crippen.MolLogP(mol), qed=QED.qed(mol),
                                  tpsa=Descriptors.TPSA(mol), sas=sascorer.calculateScore(mol),
                                  fragment_match=(bool(mol.HasSubstructMatch(query)) if query is not None else
                                                  "unavailable" if query_unavailable else ""))
                records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["sample_id", "smiles", "valid", "canonical_smiles", "logp", "qed", "tpsa", "sas", "fragment_match", "fragment_match_status"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    valid = [row for row in records if row["valid"]]
    summary = {"requested": count, "attempted": len(records), "valid": len(valid),
               "unique_valid": len({r["canonical_smiles"] for r in valid}), "conditions": list(selected),
               "seed": seed, "checkpoint": str(Path(checkpoint).resolve()),
               "checkpoint_step": state["global_step"],
               "fragment_representation": fragment_representation,
               "fragment_query_chemical_valid": (query is not None) if "fragment" in selected else None,
               "fragment_match_evaluation": ("unavailable_invalid_fragment_query" if query_unavailable else
                                             "available_for_valid_generated_molecules" if query is not None else "not_requested"),
               "pocket_input": builder.resolver.input_fingerprint(request) if "pocket" in selected else None,
               "pocket_provenance_status": ("unverified-legacy" if state.get("pocket_encoder_signature") == LEGACY_SIGNATURE
                                             else "profile-identified") if "pocket" in selected else "unused",
               "note": "Conditioning is learned guidance. IFP compliance requires independent 3D pose/interaction evaluation."}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
