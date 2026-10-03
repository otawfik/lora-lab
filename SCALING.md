# SCALING.md: from this toy to real LoRA

LoRA Lab demonstrates the LoRA mechanism on a tiny numpy model. The same
ideas scale directly to production fine-tuning with Hugging Face
`transformers` and the `peft` library. This guide shows the mapping.

## Concept mapping

| LoRA Lab (this repo) | Production equivalent |
|---|---|
| `TinyLM` (numpy n-gram LM) | Any HF causal LM (TinyLlama, Qwen2.5, Llama 3.1) |
| `LoRAAdapter`, rank 8 on the output head | `peft.LoraConfig(r=8, ...)` on attention projections |
| `W_new = W + (alpha/rank) * B @ A`, B zero-init | Exactly what `peft` implements |
| Base weights never passed to the optimizer | `model.requires_grad_(False)` except LoRA params, handled by `get_peft_model` |
| `train.py` with Adam | `transformers.Trainer` or `trl.SFTTrainer` |
| Perplexity on held-out pirate lines | Perplexity on a held-out slice of your domain data |
| `weights/lora_adapter.npz` | Adapter checkpoint (`adapter_model.safetensors`) |

The core loop is identical: freeze the base, train a low-rank delta,
compare against the base.

## Minimal real-LoRA recipe

```bash
pip install torch transformers peft datasets accelerate
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

base = "Qwen/Qwen2.5-0.5B-Instruct"  # small enough for a laptop GPU
model = AutoModelForCausalLM.from_pretrained(base, load_in_4bit=True)
tok = AutoTokenizer.from_pretrained(base)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()  # expect well under 1% of base params

data = load_dataset("text", data_files={"train": "pirate_train.txt",
                                        "eval": "pirate_eval.txt"})

trainer = SFTTrainer(
    model=model, train_dataset=data["train"], eval_dataset=data["eval"],
    args=TrainingArguments(
        output_dir="pirate-lora", per_device_train_batch_size=4,
        num_train_epochs=3, learning_rate=2e-4, logging_steps=10,
        eval_strategy="epoch", save_strategy="epoch",
    ),
)
trainer.train()
```

## Evaluate like LoRA Lab does

Perplexity before and after, on text the adapter never trained on:

```python
import torch, math
model.eval()
nll, n = 0.0, 0
with torch.no_grad():
    for batch in eval_loader:
        loss = model(**batch).loss
        nll += loss.item() * batch["input_ids"].numel()
        n += batch["input_ids"].numel()
print("perplexity:", math.exp(nll / n))
```

Run it once with the plain base model and once with the adapter
attached. The gap is your pirate score, same as `charts/perplexity.png`
in this repo.

## Merging and serving

```python
merged = model.merge_and_unload()   # fold B@A into W, like effective_W2()
merged.save_pretrained("pirate-model-merged")
```

Or keep the adapter separate and hot-swap styles at request time, which
is the real superpower of LoRA: one frozen base, many tiny adapters.

## Data tips (learned from the toy)

- Repeat your key vocabulary. In the toy corpus, iconic words that
  appeared only once fell out of the vocabulary entirely; a second pass
  of lines reusing core terms fixed it. Real tokenizers have the same
  issue in reverse: rare style words get split into pieces, so include
  enough examples for the adapter to learn them.
- Keep a clean held-out split from the start. The toy's 91.8%
  perplexity win is only meaningful because it was measured on lines
  the adapter never saw.
- Start with rank 8 or 16 on the attention projections. If the style
  does not shift, raise the rank or the learning rate before blaming
  the data.

## What changes at scale (and what does not)

What changes: distributed training, quantization (QLoRA), chat
templates, preference alignment. What does not: the update is still a
frozen base plus a learned low-rank delta, evaluated against the base
on held-out data. If you understood `loralab/adapter.py`, you
understand `peft`.
