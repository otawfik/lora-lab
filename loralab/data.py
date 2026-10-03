"""Corpus loading, train/val splitting, and mini-batch iteration."""

import numpy as np


def load_sequences(path, tokenizer, context):
    """Load a corpus file into padded token-id sequences.

    Each line becomes ``[<bos>] * context + ids + [<eos>]`` so every
    target token has a full context window.
    """
    seqs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = [tokenizer.bos_id] * context + tokenizer.encode(line) + [tokenizer.eos_id]
            seqs.append(np.array(ids, dtype=np.int64))
    return seqs


def train_val_split(seqs, val_frac=0.15, seed=0):
    """Deterministic random split of a sequence list."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(seqs))
    n_val = max(1, int(len(seqs) * val_frac))
    val_idx = set(idx[:n_val].tolist())
    train = [s for i, s in enumerate(seqs) if i not in val_idx]
    val = [s for i, s in enumerate(seqs) if i in val_idx]
    return train, val


def iter_batches(seqs, context, batch_size, rng):
    """Yield ``(x, y)`` mini-batches of context windows and next tokens."""
    xs, ys = [], []
    for s in seqs:
        for t in range(context, len(s)):
            xs.append(s[t - context : t])
            ys.append(s[t])
    if not xs:
        return
    xs = np.stack(xs)
    ys = np.array(ys, dtype=np.int64)
    perm = rng.permutation(len(xs))
    for i in range(0, len(xs), batch_size):
        j = perm[i : i + batch_size]
        yield xs[j], ys[j]


def count_pairs(seqs, context):
    """Total number of (context, target) training pairs."""
    return sum(max(0, len(s) - context) for s in seqs)
