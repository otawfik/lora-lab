"""LoRA-style low-rank adapter for TinyLM.

The core LoRA idea: freeze every base-model weight, then learn a small
update to one layer as a product of two skinny matrices,

    W_new = W_base + (alpha / rank) * B @ A,

where B is (hidden_dim, rank) and A is (rank, vocab_size). Only B and A
receive gradients, so fine-tuning touches a tiny fraction of the
parameters. B starts at zero, so the adapted model is exactly the base
model before training begins.
"""

import numpy as np


class LoRAAdapter:
    def __init__(self, base, rank=8, alpha=None, seed=1):
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha) if alpha is not None else float(rank)
        rng = np.random.default_rng(seed)
        # Zero-init B: the adapter is a no-op until trained.
        self.B = np.zeros((base.hidden_dim, self.rank))
        self.A = rng.normal(0, 0.02, size=(self.rank, base.vocab_size))

    def scaling(self):
        return self.alpha / self.rank

    def delta_W(self):
        """The learned low-rank update to the output head."""
        return self.scaling() * (self.B @ self.A)

    def effective_W2(self):
        return self.base.W2 + self.delta_W()

    def trainable_params(self):
        # Only the adapter matrices are ever passed to the optimizer,
        # so the base model stays frozen by construction.
        return {"B": self.B, "A": self.A}

    def trainable_count(self):
        return self.B.size + self.A.size

    def adapter_fraction(self):
        return self.trainable_count() / self.base.param_count()

    def loss_and_grads(self, x, y):
        """Loss plus gradients wrt B and A only (chain rule through B@A)."""
        W_eff = self.effective_W2()
        loss, grads = self.base.loss_and_grads(x, y, W2=W_eff)
        dW = grads["W2"]
        s = self.scaling()
        dB = s * (dW @ self.A.T)
        dA = s * (self.B.T @ dW)
        return loss, {"B": dB, "A": dA}

    def generate(self, prompt_ids, n_tokens=40, temperature=1.0, top_k=40,
                 seed=0, bos_id=1, suppress_ids=(), stop_on=()):
        return self.base.generate(
            prompt_ids, n_tokens=n_tokens, temperature=temperature,
            top_k=top_k, seed=seed, W2=self.effective_W2(), bos_id=bos_id,
            suppress_ids=suppress_ids, stop_on=stop_on)

    def save(self, path):
        np.savez(path, B=self.B, A=self.A, rank=self.rank, alpha=self.alpha,
                 E=self.base.E, W1=self.base.W1, b1=self.base.b1,
                 W2=self.base.W2, b2=self.base.b2,
                 vocab_size=self.base.vocab_size, emb_dim=self.base.emb_dim,
                 hidden_dim=self.base.hidden_dim, context=self.base.context)

    @classmethod
    def load(cls, path):
        from .model import TinyLM
        d = np.load(path, allow_pickle=False)
        base = TinyLM(int(d["vocab_size"]), int(d["emb_dim"]),
                      int(d["hidden_dim"]), int(d["context"]))
        base.E = d["E"]
        base.W1 = d["W1"]
        base.b1 = d["b1"]
        base.W2 = d["W2"]
        base.b2 = d["b2"]
        adapter = cls(base, rank=int(d["rank"]), alpha=float(d["alpha"]))
        adapter.B = d["B"]
        adapter.A = d["A"]
        return adapter
