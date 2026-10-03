#!/usr/bin/env python3
"""Train the base TinyLM on generic English, then a LoRA-style adapter on
pirate speak. Saves weights, charts, and prints a perplexity comparison.

Usage:
    python train.py
    python train.py --epochs-base 25 --epochs-adapter 60 --rank 8
"""

import argparse
import os
import time

from loralab.adapter import LoRAAdapter
from loralab.data import count_pairs, load_sequences, train_val_split
from loralab.metrics import (
    perplexity,
    plot_loss_curves,
    plot_perplexity_comparison,
)
from loralab.model import TinyLM
from loralab.tokenizer import Tokenizer
from loralab.train import fit


def main():
    ap = argparse.ArgumentParser(description="Train base model + LoRA adapter.")
    ap.add_argument("--epochs-base", type=int, default=25)
    ap.add_argument("--epochs-adapter", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-base", type=float, default=3e-3)
    ap.add_argument("--lr-adapter", type=float, default=5e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--charts-dir", default="charts")
    args = ap.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.charts_dir, exist_ok=True)

    base_path = os.path.join(args.data_dir, "base_corpus.txt")
    pirate_path = os.path.join(args.data_dir, "pirate_corpus.txt")
    with open(base_path, encoding="utf-8") as f:
        base_texts = [l.strip() for l in f if l.strip()]
    with open(pirate_path, encoding="utf-8") as f:
        pirate_texts = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(base_texts)} base lines, {len(pirate_texts)} pirate lines.")

    tokenizer = Tokenizer.build(base_texts + pirate_texts, min_freq=2)
    print(f"Vocabulary size: {len(tokenizer)}")
    tokenizer.save(os.path.join(args.weights_dir, "vocab.json"))

    context = 2
    base_seqs = load_sequences(base_path, tokenizer, context)
    pirate_seqs = load_sequences(pirate_path, tokenizer, context)
    pirate_train, pirate_val = train_val_split(pirate_seqs, args.val_frac, args.seed)
    print(f"Base pairs: {count_pairs(base_seqs, context)}, "
          f"pirate train pairs: {count_pairs(pirate_train, context)}, "
          f"pirate val pairs: {count_pairs(pirate_val, context)}")

    # ---- Base model -----------------------------------------------------
    model = TinyLM(len(tokenizer), emb_dim=24, hidden_dim=64,
                   context=context, seed=args.seed)
    print(f"Base params: {model.param_count()}")
    t0 = time.time()
    hist_base = fit(
        model.loss_and_grads, model.params(), base_seqs, context,
        args.epochs_base, args.batch_size, args.lr_base,
        seed=args.seed, label="base",
    )
    print(f"Base training took {time.time() - t0:.1f}s")
    model.save(os.path.join(args.weights_dir, "base.npz"))

    ppl_base = perplexity(model.loss_and_grads, pirate_val, context)
    print(f"Base perplexity on held-out pirate lines: {ppl_base:.2f}")

    # ---- LoRA adapter ---------------------------------------------------
    adapter = LoRAAdapter(model, rank=args.rank, seed=args.seed + 1)
    print(f"Adapter trainable params: {adapter.trainable_count()} "
          f"({100 * adapter.adapter_fraction():.2f}% of base)")
    t0 = time.time()
    hist_adapter = fit(
        adapter.loss_and_grads, adapter.trainable_params(), pirate_train,
        context, args.epochs_adapter, args.batch_size, args.lr_adapter,
        seed=args.seed + 100, label="adapter",
    )
    print(f"Adapter training took {time.time() - t0:.1f}s")
    adapter.save(os.path.join(args.weights_dir, "lora_adapter.npz"))

    ppl_adapter = perplexity(adapter.loss_and_grads, pirate_val, context)
    print(f"Adapter perplexity on held-out pirate lines: {ppl_adapter:.2f}")
    print(f"Perplexity improvement: {100 * (ppl_base - ppl_adapter) / ppl_base:.1f}%")

    # ---- Charts ---------------------------------------------------------
    plot_loss_curves(
        {"base model": hist_base, "LoRA adapter": hist_adapter},
        os.path.join(args.charts_dir, "loss_curves.png"),
    )
    plot_perplexity_comparison(
        ["base model", "LoRA adapter"], [ppl_base, ppl_adapter],
        os.path.join(args.charts_dir, "perplexity.png"),
    )
    print(f"Weights -> {args.weights_dir}, charts -> {args.charts_dir}")


if __name__ == "__main__":
    main()
