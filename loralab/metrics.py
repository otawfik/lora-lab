"""Evaluation metrics (perplexity) and chart helpers."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import iter_batches


def perplexity(loss_grad_fn, seqs, context, batch_size=512):
    """Perplexity = exp(mean negative log likelihood) over the sequences."""
    total, n = 0.0, 0
    rng = np.random.default_rng(0)
    for x, y in iter_batches(seqs, context, batch_size, rng):
        loss, _ = loss_grad_fn(x, y)
        total += loss * len(x)
        n += len(x)
    return float(np.exp(total / max(n, 1)))


def plot_loss_curves(histories, path, title="Training loss"):
    """Plot per-epoch loss for one or more runs (dict of name -> list)."""
    plt.figure(figsize=(8, 5))
    for name, hist in histories.items():
        plt.plot(range(1, len(hist) + 1), hist, marker="o",
                 markersize=3, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Mean cross-entropy loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_perplexity_comparison(labels, values, path,
                               title="Perplexity on held-out pirate lines (lower is better)"):
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=["#5b7fa6", "#c97b2d"])
    plt.ylabel("Perplexity")
    plt.title(title)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
