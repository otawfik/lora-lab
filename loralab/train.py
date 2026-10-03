"""Mini-batch Adam training loops for the base model and the adapter."""

import numpy as np

from .data import iter_batches


class Adam:
    """Minimal Adam optimizer that updates parameter dicts in place."""

    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for key, p in self.params.items():
            g = grads[key]
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * (g * g)
            m_hat = self.m[key] / (1 - self.b1 ** self.t)
            v_hat = self.v[key] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train_epoch(loss_grad_fn, params, seqs, context, batch_size, optimizer, seed):
    rng = np.random.default_rng(seed)
    total, n = 0.0, 0
    for x, y in iter_batches(seqs, context, batch_size, rng):
        loss, grads = loss_grad_fn(x, y)
        optimizer.step(grads)
        total += loss * len(x)
        n += len(x)
    return total / max(n, 1)


def fit(loss_grad_fn, params, seqs, context, epochs, batch_size, lr,
        seed=0, log_every=5, label=""):
    """Train for ``epochs`` passes, returning per-epoch mean loss history."""
    optimizer = Adam(params, lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(loss_grad_fn, params, seqs, context,
                           batch_size, optimizer, seed + epoch)
        history.append(loss)
        if log_every and epoch % log_every == 0:
            tag = f"[{label}] " if label else ""
            print(f"{tag}epoch {epoch}/{epochs}  loss {loss:.4f}", flush=True)
    return history
