"""Local inference-input orchestration; this module never alters training data.

Numerical targets/fragment text/IFP labels can be explicit. Optional reference
tools are local Python callables, not commands, URLs or dynamic imports. A
protein does not determine ligand QED, a fragment, or ligand interaction labels.
"""
import copy
import hashlib
import json
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer

from .config import (CONDITIONS, DEFAULT_FRAGMENT_REPRESENTATION, PROPERTIES,
                     parse_conditions, validate_fragment_representation)
from .data import canonical_smiles, fragment_text, property_vector
from .ifp import canonical_ifp
from .pocket import LEGACY_SIGNATURE, content_hash, file_sha256, validate_embedding
from .tokenizer import SmilesTokenizer


def rdkit_properties(smiles):
    """Four ligand properties using the same definitions as generation output.

    This declares the current calculation, not historical CSV equivalence.
    No structure/coordinate or target-specific interaction property is inferred.
    """
    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(canonical_smiles(smiles))
    except (ValueError, TypeError):
        raise ValueError("Property calculation requires a valid reference molecule SMILES") from None
    return {"logp": float(Crippen.MolLogP(mol)), "qed": float(QED.qed(mol)),
            "tpsa": float(Descriptors.TPSA(mol)), "sas": float(sascorer.calculateScore(mol))}


def _callable_name(provider):
    if not callable(provider):
        raise TypeError("Feature provider must be a local Python callable, not a command/module/URL string")
    return getattr(provider, "__module__", type(provider).__module__) + "." + getattr(
        provider, "__qualname__", type(provider).__qualname__)


def _file_reference(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Selected local structure file does not exist: " + str(path))
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _property_presence(value):
    return value is not None and any(property_vector(value)[1])


def prepare_inference_request(*, pocket_embedding=None, pocket=None, protein=None,
                              encoder_signature=None, properties=None, fragment=None, ifp=None,
                              conditions=None, reference_smiles=None, complex_pdb=None,
                              compute_properties=False, extract_fragment=False, compute_ifp=False,
                              property_provider=None, fragment_provider=None, ifp_provider=None,
                              resolver=None, resolve_pocket=False,
                              fragment_representation=DEFAULT_FRAGMENT_REPRESENTATION):
    """Return ``request``, canonical ``conditions`` and JSON-safe ``provenance``.

    ``pocket`` and ``protein`` are local file paths. A standard pocket object
    should be normalized by its caller to these two keyword arguments. By
    default raw PDB references are preserved for the existing generation entry
    to encode/cache. ``resolve_pocket=True`` explicitly asks the supplied
    PocketResolver-compatible object to resolve them now; the request then
    contains only its embedding+signature, while provenance keeps the raw
    references. No model/encoder is loaded by this module merely to assemble a
    raw request. Explicitly disabled inputs are not read or validated.

    Tools are opt-in: ``compute_properties`` uses reference_smiles and defaults
    to rdkit_properties; ``extract_fragment`` needs reference_smiles and an
    explicit local fragment_provider(str)->str; ``compute_ifp`` needs a local
    protein-ligand complex PDB and an explicit ifp_provider(path)->six labels.
    No default fragment-selection, docking, protonation, PLIP version or ligand
    selection is invented. Only selected modalities invoke providers.

    Precomputed encoder_signature is a caller declaration, NOT proof of CSV
    provenance. Absent signatures remain LEGACY_SIGNATURE. Nothing here binds
    old embeddings to a newly configured encoder or disables generation's
    checkpoint/profile compatibility checks.
    """
    representation = validate_fragment_representation(fragment_representation)
    if pocket_embedding is not None and pocket is not None:
        raise ValueError("Pocket input conflict: provide embedding OR raw pocket, not both")
    for name, value in (("compute_properties", compute_properties), ("extract_fragment", extract_fragment),
                        ("compute_ifp", compute_ifp), ("resolve_pocket", resolve_pocket)):
        if type(value) is not bool:
            raise TypeError(name + " must be a boolean")
    supplied = {"pocket": pocket_embedding is not None or pocket is not None,
                "properties": properties is not None, "fragment": fragment is not None,
                "ifp": ifp is not None}
    if conditions is None:
        selected = tuple(name for name in CONDITIONS if
                         (name == "pocket" and supplied[name]) or
                         (name == "properties" and (compute_properties or _property_presence(properties))) or
                         (name == "fragment" and (extract_fragment or fragment not in (None, ""))) or
                         (name == "ifp" and (compute_ifp or ifp is not None)))
    else:
        parsed = parse_conditions(conditions)
        selected = tuple(name for name in CONDITIONS if name in parsed)
    request = {}
    provenance = {"format_version": 1, "purpose": "inference_input_preparation",
                  "selected_conditions": list(selected), "fragment_representation": representation,
                  "sources": {}, "ignored_supplied_modalities": [name for name in CONDITIONS
                                                                    if supplied[name] and name not in selected],
                  "provider_policy": "local callable only; no shell/network/dynamic import is performed by this module",
                  "raw_structure_contents_copied": False,
                  "historical_training_embedding_equivalence": "not_checked",
                  "checkpoint_encoder_compatibility_check": "still_required_by_generation"}
    reference = None

    def reference_molecule():
        nonlocal reference
        if reference is None:
            if reference_smiles is None:
                raise ValueError("Selected molecule-derived feature requires reference_smiles; protein is not a substitute")
            try:
                with rdBase.BlockLogs():
                    reference = canonical_smiles(reference_smiles)
            except (ValueError, TypeError):
                raise ValueError("The supplied reference molecule SMILES is invalid") from None
            provenance["reference_molecule_canonical_sha256"] = hashlib.sha256(reference.encode()).hexdigest()
        return reference

    if "pocket" in selected:
        if pocket_embedding is not None:
            vector = validate_embedding(pocket_embedding)
            signature = LEGACY_SIGNATURE if encoder_signature is None else encoder_signature
            if not isinstance(signature, str) or not signature:
                raise ValueError("Precomputed encoder_signature must be a nonempty string")
            request.update(pocket_embedding=vector.tolist(), pocket_encoder_signature=signature)
            provenance["sources"]["pocket"] = {
                "kind": "explicit_precomputed_embedding", "dimension": 128,
                "embedding_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                "encoder_signature": signature, "signature_evidence": "caller_declaration_not_verified_here",
                "cache": {"status": "not_used"}}
        elif pocket is not None:
            if encoder_signature is not None:
                raise ValueError("encoder_signature is only for a precomputed embedding; raw signature comes from the resolver")
            pocket_ref = _file_reference(pocket)
            protein_ref = _file_reference(protein if protein is not None else pocket)
            raw = {"pocket": {"pocket_pdb": pocket_ref["path"], "protein_pdb": protein_ref["path"]}}
            source = {"kind": "explicit_raw_structure", "pocket": pocket_ref, "protein": protein_ref,
                      "context": "cropped-pocket" if pocket_ref["path"] == protein_ref["path"] else "full-protein",
                      "cache": {"status": "deferred_to_generation"}}
            if resolve_pocket:
                if resolver is None:
                    raise ValueError("resolve_pocket requires an explicitly configured local resolver")
                fingerprint = resolver.input_fingerprint(raw)
                signature = resolver.row_signature(raw)
                if not isinstance(signature, str) or not signature or signature == LEGACY_SIGNATURE:
                    raise ValueError("A raw resolver must provide explicit non-legacy encoder provenance")
                vector = validate_embedding(resolver.resolve(raw))
                if resolver.input_fingerprint(raw) != fingerprint or resolver.row_signature(raw) != signature:
                    raise ValueError("Raw pocket/encoder changed during feature preparation")
                request.update(pocket_embedding=vector.tolist(), pocket_encoder_signature=signature)
                source.update(encoder_signature=signature, input_fingerprint=fingerprint,
                              embedding_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
                              cache={"status": "resolved_by_configured_adapter", "key": content_hash(fingerprint)})
            else:
                request.update(raw)
            if _file_reference(pocket) != pocket_ref or _file_reference(protein_ref["path"]) != protein_ref:
                raise ValueError("Raw structure changed while preparing inference inputs")
            provenance["sources"]["pocket"] = source
        else:
            raise ValueError("Selected pocket condition has no embedding or raw pocket input")
    if "properties" in selected:
        if compute_properties:
            if properties is not None:
                raise ValueError("Choose explicit properties OR reference-molecule calculation, not both")
            provider = rdkit_properties if property_provider is None else property_provider
            name = _callable_name(provider)
            value = provider(reference_molecule())
            source = {"kind": "reference_molecule_provider", "provider": name,
                      "historical_training_label_equivalence": "not_asserted"}
            if provider is rdkit_properties:
                source.update(rdkit_version=rdBase.rdkitVersion,
                              definitions={"logp": "Crippen.MolLogP", "qed": "QED.qed",
                                           "tpsa": "Descriptors.TPSA", "sas": "Contrib.SA_Score.sascorer.calculateScore"})
        else:
            value = copy.deepcopy(properties)
            source = {"kind": "explicit_targets"}
        _, observed = property_vector(value)
        if not any(observed):
            raise ValueError("Selected properties condition has no observed values")
        # Missing entries are not requested targets. Keep numerical zero, omit
        # None/blank, and use JSON numeric scalars rather than NumPy/Decimal.
        request["properties"] = {name: float(value[name]) for name, present in zip(PROPERTIES, observed) if present}
        provenance["sources"]["properties"] = source
    if "fragment" in selected:
        if extract_fragment:
            if fragment is not None:
                raise ValueError("Choose explicit fragment OR reference-molecule extraction, not both")
            if fragment_provider is None:
                raise ValueError("Fragment extraction requires an explicit local fragment_provider; no algorithm is selected automatically")
            name = _callable_name(fragment_provider)
            value = fragment_provider(reference_molecule())
            source = {"kind": "reference_molecule_provider", "provider": name}
        else:
            value = fragment
            source = {"kind": "explicit_fragment"}
        if not isinstance(value, str) or not value.strip():
            raise ValueError("A selected fragment must be one nonempty string; choose a fragment explicitly from any candidate list")
        try:
            with rdBase.BlockLogs():
                value = fragment_text(value, representation)
        except (ValueError, TypeError):
            if representation == "raw-smiles-v1":
                raise ValueError("Selected raw fragment contains unsupported text; no repair or masking performed") from None
            raise ValueError("Selected fragment is invalid for its declared representation; no repair or masking performed") from None
        SmilesTokenizer.split(value)
        request["fragment"] = value
        request["fragment_representation"] = representation
        source["representation"] = representation
        if representation == "raw-smiles-v1":
            source["chemical_validity"] = "not_asserted_literal_token_input"
        provenance["sources"]["fragment"] = source
    if "ifp" in selected:
        if compute_ifp:
            if ifp is not None:
                raise ValueError("Choose explicit IFP targets OR complex-based calculation, not both")
            if complex_pdb is None:
                raise ValueError("IFP calculation requires a protein-ligand complex PDB; protein/pocket alone is insufficient")
            if ifp_provider is None:
                raise ValueError("IFP calculation requires an explicit local ifp_provider with ligand selection/configuration")
            name = _callable_name(ifp_provider)
            complex_ref = _file_reference(complex_pdb)
            value = ifp_provider(complex_ref["path"])
            if _file_reference(complex_pdb) != complex_ref:
                raise ValueError("Reference complex changed during IFP calculation")
            source = {"kind": "reference_complex_provider", "provider": name, "complex": complex_ref}
        else:
            value = ifp
            source = {"kind": "explicit_interaction_targets"}
        if value is None:
            raise ValueError("Selected IFP condition has no observed values")
        request["ifp"] = canonical_ifp(value)
        provenance["sources"]["ifp"] = source
    provenance["skipped_requested_computations"] = [name for name, active in
        (("properties", compute_properties), ("fragment", extract_fragment), ("ifp", compute_ifp), ("pocket", resolve_pocket))
        if active and name not in selected]
    json.dumps({"request": request, "provenance": provenance}, allow_nan=False)
    return {"request": request, "conditions": list(selected), "provenance": provenance}
