"""Character-level tokenizer: every unique character is one token."""
import json


class CharTokenizer:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str):
        return cls(text)

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.chars, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))
