"""LoRA Lab: fine-tune a tiny language model with LoRA-style adapters.

A from-scratch numpy language model plus a low-rank adapter that
demonstrates the core LoRA idea: freeze the base weights, train only a
small rank-decomposed update, and A/B the result against the base model.
"""

__version__ = "0.1.0"

from .tokenizer import Tokenizer
from .model import TinyLM
from .adapter import LoRAAdapter

__all__ = ["Tokenizer", "TinyLM", "LoRAAdapter", "__version__"]
