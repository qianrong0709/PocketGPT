from dataclasses import asdict, dataclass

CONDITIONS = ("pocket", "properties", "fragment", "ifp")
PROPERTIES = ("logp", "qed", "tpsa", "sas")
IFP_NAMES = ("hydrogen_bond", "hydrophobic", "pi_stacking", "salt_bridge", "pi_cation", "halogen_bond")
DEFAULT_FRAGMENT_REPRESENTATION = "canonical-smiles-v1"
FRAGMENT_REPRESENTATIONS = (DEFAULT_FRAGMENT_REPRESENTATION, "raw-smiles-v1")


def validate_fragment_representation(value):
    if value not in FRAGMENT_REPRESENTATIONS:
        raise ValueError("fragment_representation must be canonical-smiles-v1 or raw-smiles-v1")
    return value


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    pocket_dim: int = 128
    architecture_version: int = 2
    fragment_encoder: str = "transformer-v1"
    fragment_layers: int = 2

    def __post_init__(self):
        if min(self.n_layer, self.n_head, self.block_size, self.n_embd, self.pocket_dim) < 1 or self.n_embd % self.n_head:
            raise ValueError("Invalid Transformer dimensions")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")
        if self.architecture_version == 2:
            if self.fragment_encoder != "transformer-v1" or self.fragment_layers < 1:
                raise ValueError("Architecture v2 requires the sequence fragment encoder")
        elif self.architecture_version == 1:
            if self.fragment_encoder != "mean-v1" or self.fragment_layers != 0:
                raise ValueError("Legacy architecture v1 requires explicit mean-v1 / zero layers")
        else:
            raise ValueError("Unsupported PocketGPT architecture version")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_checkpoint_dict(cls, value, *, allow_legacy=False):
        """Do not reinterpret an unversioned mean-pooling model as architecture v2."""
        parameters = dict(value)
        version = parameters.get("architecture_version", 1)
        if version == 1:
            if not allow_legacy:
                raise ValueError("Legacy fragment architecture requires explicit legacy inference or warm start; not exact resume")
            parameters.setdefault("architecture_version", 1)
            parameters.setdefault("fragment_encoder", "mean-v1")
            parameters.setdefault("fragment_layers", 0)
        elif version == 2:
            if "fragment_encoder" not in parameters or "fragment_layers" not in parameters:
                raise ValueError("Checkpoint is missing architecture v2 fragment metadata")
        return cls(**parameters)


def parse_conditions(value):
    names = tuple(value.split(",")) if isinstance(value, str) else tuple(value)
    if names == ("none",) or not names:
        return ()
    if names == ("all",):
        return CONDITIONS
    if len(set(names)) != len(names) or set(names) - set(CONDITIONS):
        raise ValueError("Conditions must be a unique subset of: " + ",".join(CONDITIONS))
    return names
