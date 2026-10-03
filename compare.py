#!/usr/bin/env python3
"""A/B comparison: generate text from the base model and the LoRA-adapted
pirate model side by side.

Usage:
    python compare.py --prompt "the treasure is"
    python compare.py --prompt "good morning" --temperature 0.7 --seed 3
"""

import argparse
import os

from loralab.adapter import LoRAAdapter
from loralab.model import TinyLM
from loralab.tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def load_all(weights_dir):
    tokenizer = Tokenizer.load(os.path.join(weights_dir, "vocab.json"))
    base = TinyLM.load(os.path.join(weights_dir, "base.npz"))
    adapter = LoRAAdapter.load(os.path.join(weights_dir, "lora_adapter.npz"))
    return tokenizer, base, adapter


def main():
    ap = argparse.ArgumentParser(description="A/B the base model vs the pirate adapter.")
    ap.add_argument("--prompt", default="the treasure is",
                    help="Starting words for generation.")
    ap.add_argument("--n-tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights-dir", default=os.path.join(HERE, "weights"))
    args = ap.parse_args()

    tokenizer, base, adapter = load_all(args.weights_dir)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        print("Prompt produced no known words, try something else.")
        return

    base_ids = base.generate(prompt_ids, n_tokens=args.n_tokens,
                             temperature=args.temperature, seed=args.seed,
                             bos_id=tokenizer.bos_id,
                             suppress_ids=(tokenizer.unk_id,),
                             stop_on=(tokenizer.eos_id,))
    pirate_ids = adapter.generate(prompt_ids, n_tokens=args.n_tokens,
                                  temperature=args.temperature, seed=args.seed,
                                  bos_id=tokenizer.bos_id,
                                  suppress_ids=(tokenizer.unk_id,),
                                  stop_on=(tokenizer.eos_id,))

    print(f'Prompt: "{args.prompt}"\n')
    print("--- BASE MODEL ---")
    print(tokenizer.decode(prompt_ids + base_ids))
    print("\n--- LORA PIRATE MODEL ---")
    print(tokenizer.decode(prompt_ids + pirate_ids))


if __name__ == "__main__":
    main()
