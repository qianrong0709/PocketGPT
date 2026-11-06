import json
import re
from pathlib import Path

PATTERN = re.compile(r"\[[^\]]+\]|Br|Cl|Si|Se|se|%[0-9]{2}|[A-Za-z0-9().=#+\\/\-:@~?*$]")
SPECIAL = ("<pad>", "<bos>", "<eos>")


class SmilesTokenizer:
    def __init__(self, tokens):
        tokens = list(tokens)
        if tokens[:3] != list(SPECIAL) or len(set(tokens)) != len(tokens):
            raise ValueError("Vocabulary must start with unique <pad>, <bos>, <eos> tokens")
        self.tokens = tokens
        self.stoi = {t: i for i, t in enumerate(tokens)}
        self.pad_id, self.bos_id, self.eos_id = range(3)

    @staticmethod
    def split(smiles):
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("SMILES must be a nonempty string")
        smiles = smiles.strip()
        tokens = PATTERN.findall(smiles)
        if "".join(tokens) != smiles:
            raise ValueError("SMILES contains unsupported text: " + smiles[:80])
        return tokens

    @classmethod
    def build(cls, smiles):
        found = set()
        for text in smiles:
            found.update(cls.split(text))
        return cls([*SPECIAL, *sorted(found)])

    def encode(self, smiles):
        tokens = self.split(smiles)
        missing = set(tokens) - self.stoi.keys()
        if missing:
            raise ValueError("Tokens absent from training vocabulary: " + repr(sorted(missing)))
        return [self.stoi[t] for t in tokens]

    def decode(self, ids):
        result = []
        for index in ids:
            index = int(index)
            if index == self.eos_id:
                break
            if index not in (self.pad_id, self.bos_id):
                result.append(self.tokens[index])
        return "".join(result)

    def save(self, path):
        Path(path).write_text(json.dumps(self.tokens, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))
