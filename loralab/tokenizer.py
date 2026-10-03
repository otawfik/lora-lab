"""Word-level tokenizer with special tokens."""

import json
import re
from collections import Counter

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


class Tokenizer:
    """Simple lowercase word tokenizer backed by a fixed vocabulary."""

    def __init__(self, stoi):
        self.stoi = dict(stoi)
        self.itos = [None] * len(stoi)
        for tok, idx in self.stoi.items():
            self.itos[idx] = tok
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    @classmethod
    def build(cls, texts, min_freq=2, max_vocab=6000):
        """Build a vocabulary from raw text lines.

        Keeps words that appear at least ``min_freq`` times so rare words
        fall back to the <unk> token.
        """
        counts = Counter()
        for text in texts:
            counts.update(WORD_RE.findall(text.lower()))
        vocab = [w for w, c in counts.most_common(max_vocab) if c >= min_freq]
        stoi = {tok: i for i, tok in enumerate(SPECIALS)}
        for word in vocab:
            if word not in stoi:
                stoi[word] = len(stoi)
        return cls(stoi)

    def encode(self, text):
        """Turn raw text into a list of token ids."""
        return [self.stoi.get(w, self.unk_id) for w in WORD_RE.findall(text.lower())]

    def decode(self, ids):
        """Turn token ids back into readable text, dropping specials."""
        words = [self.itos[i] for i in ids if self.itos[i] not in SPECIALS]
        return " ".join(words)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.stoi, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            stoi = json.load(f)
        # JSON keys are strings already; values come back as ints.
        return cls({tok: int(idx) for tok, idx in stoi.items()})

    def __len__(self):
        return len(self.stoi)
