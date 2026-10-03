"""Tiny neural n-gram language model, implemented from scratch in numpy.

Architecture: token embedding -> concat context embeddings -> tanh MLP ->
linear output head -> softmax. Small enough to train on CPU in seconds,
real enough to demonstrate fine-tuning with gradients.
"""

import numpy as np


class TinyLM:
    def __init__(self, vocab_size, emb_dim=24, hidden_dim=64, context=2, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = int(vocab_size)
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.context = int(context)
        self.E = rng.normal(0, 0.1, size=(self.vocab_size, self.emb_dim))
        self.W1 = rng.normal(
            0, np.sqrt(2.0 / (self.context * self.emb_dim)),
            size=(self.context * self.emb_dim, self.hidden_dim),
        )
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.normal(
            0, np.sqrt(1.0 / self.hidden_dim),
            size=(self.hidden_dim, self.vocab_size),
        )
        self.b2 = np.zeros(self.vocab_size)

    def params(self):
        return {"E": self.E, "W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2}

    def param_count(self):
        return sum(p.size for p in self.params().values())

    def forward(self, x, W2=None):
        """Return logits for a batch of context windows ``x`` (B, context).

        ``W2`` optionally overrides the output head, which is how the
        LoRA adapter injects its low-rank update without touching the
        base weights.
        """
        B = x.shape[0]
        emb = self.E[x].reshape(B, self.context * self.emb_dim)
        z = emb @ self.W1 + self.b1
        a = np.tanh(z)
        W_eff = self.W2 if W2 is None else W2
        logits = a @ W_eff + self.b2
        cache = (x, emb, z, a)
        return logits, cache

    def loss_and_grads(self, x, y, W2=None):
        """Mean softmax cross-entropy and gradients for a mini-batch."""
        B = x.shape[0]
        logits, (xx, emb, z, a) = self.forward(x, W2)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = float(-np.log(probs[np.arange(B), y] + 1e-12).mean())

        dlogits = probs
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B

        W_eff = self.W2 if W2 is None else W2
        dW2 = a.T @ dlogits
        db2 = dlogits.sum(axis=0)
        da = dlogits @ W_eff.T
        dz = da * (1.0 - np.tanh(z) ** 2)
        dW1 = emb.T @ dz
        db1 = dz.sum(axis=0)
        demb = dz @ self.W1.T
        dE = np.zeros_like(self.E)
        np.add.at(dE, xx, demb.reshape(B, self.context, self.emb_dim))

        grads = {"E": dE, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return loss, grads

    def predict_next(self, x, W2=None, temperature=1.0, top_k=40, rng=None,
                     suppress_ids=()):
        """Sample one next-token id per row of ``x``.

        ``suppress_ids`` are masked out (used to hide <unk>/<eos>).
        """
        rng = np.random.default_rng() if rng is None else rng
        logits, _ = self.forward(x, W2)
        logits = logits / max(float(temperature), 1e-6)
        if suppress_ids:
            logits[:, list(suppress_ids)] = -np.inf
        if top_k and top_k < logits.shape[1]:
            part = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]
            masked = np.full_like(logits, -np.inf)
            np.put_along_axis(
                masked, part, np.take_along_axis(logits, part, axis=1), axis=1
            )
            logits = masked
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return np.array([rng.choice(probs.shape[1], p=p) for p in probs])

    def generate(self, prompt_ids, n_tokens=40, temperature=1.0, top_k=40,
                 seed=0, W2=None, bos_id=1, suppress_ids=(), stop_on=(),
                 min_tokens=10):
        """Autoregressively generate up to ``n_tokens`` ids after ``prompt_ids``.

        Stops early if a token in ``stop_on`` is sampled, but only after
        ``min_tokens`` tokens have been generated.
        """
        rng = np.random.default_rng(seed)
        context = list(prompt_ids[-self.context :])
        while len(context) < self.context:
            context = [bos_id] + context
        out = []
        for _ in range(n_tokens):
            x = np.array([context[-self.context :]])
            nxt = int(self.predict_next(
                x, W2=W2, temperature=temperature, top_k=top_k, rng=rng,
                suppress_ids=suppress_ids)[0])
            if nxt in stop_on and len(out) >= min_tokens:
                break
            out.append(nxt)
            context.append(nxt)
        return out

    def save(self, path):
        np.savez(path, E=self.E, W1=self.W1, b1=self.b1, W2=self.W2,
                 b2=self.b2, vocab_size=self.vocab_size, emb_dim=self.emb_dim,
                 hidden_dim=self.hidden_dim, context=self.context)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        model = cls(int(d["vocab_size"]), int(d["emb_dim"]),
                    int(d["hidden_dim"]), int(d["context"]))
        model.E = d["E"]
        model.W1 = d["W1"]
        model.b1 = d["b1"]
        model.W2 = d["W2"]
        model.b2 = d["b2"]
        return model
