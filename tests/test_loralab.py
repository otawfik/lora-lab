"""Tests for the LoRA Lab package."""

import numpy as np
import pytest

from loralab.adapter import LoRAAdapter
from loralab.data import iter_batches, load_sequences, train_val_split
from loralab.metrics import perplexity
from loralab.model import TinyLM
from loralab.tokenizer import Tokenizer
from loralab.train import fit


@pytest.fixture()
def tiny_setup():
    texts = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "a cat and a dog play",
        "the mat is on the floor",
    ]
    tok = Tokenizer.build(texts, min_freq=1)
    model = TinyLM(len(tok), emb_dim=8, hidden_dim=16, context=2, seed=0)
    seqs = []
    for t in texts:
        ids = [tok.bos_id] * 2 + tok.encode(t) + [tok.eos_id]
        seqs.append(np.array(ids, dtype=np.int64))
    return tok, model, seqs


def test_tokenizer_roundtrip():
    tok = Tokenizer.build(["hello world", "hello there"], min_freq=1)
    ids = tok.encode("Hello World!")
    assert tok.decode(ids) == "hello world"
    assert tok.encode("zzzunknownzzz") == [tok.unk_id]


def test_tokenizer_save_load(tmp_path):
    tok = Tokenizer.build(["hello world"], min_freq=1)
    p = tmp_path / "vocab.json"
    tok.save(str(p))
    tok2 = Tokenizer.load(str(p))
    assert tok2.encode("hello world") == tok.encode("hello world")


def test_model_forward_shape(tiny_setup):
    tok, model, seqs = tiny_setup
    x = np.array([[tok.bos_id, tok.bos_id], [tok.bos_id, tok.encode("the")[0]]])
    logits, _ = model.forward(x)
    assert logits.shape == (2, len(tok))


def test_loss_and_grads_shapes(tiny_setup):
    _, model, seqs = tiny_setup
    rng = np.random.default_rng(0)
    x, y = next(iter_batches(seqs, 2, 8, rng))
    loss, grads = model.loss_and_grads(x, y)
    assert np.isfinite(loss)
    for k, p in model.params().items():
        assert grads[k].shape == p.shape, k


def test_training_reduces_loss(tiny_setup):
    _, model, seqs = tiny_setup
    x, y = next(iter_batches(seqs, 2, 16, np.random.default_rng(0)))
    before, _ = model.loss_and_grads(x, y)
    fit(model.loss_and_grads, model.params(), seqs, 2, epochs=10,
        batch_size=16, lr=5e-3, seed=0, log_every=0)
    after, _ = model.loss_and_grads(x, y)
    assert after < before


def test_lora_zero_init_matches_base(tiny_setup):
    _, model, seqs = tiny_setup
    adapter = LoRAAdapter(model, rank=4, seed=1)
    rng = np.random.default_rng(0)
    x, _ = next(iter_batches(seqs, 2, 8, rng))
    base_logits, _ = model.forward(x)
    adapted_logits, _ = model.forward(x, W2=adapter.effective_W2())
    np.testing.assert_allclose(base_logits, adapted_logits, rtol=1e-10)


def test_lora_only_trains_adapter(tiny_setup):
    _, model, seqs = tiny_setup
    adapter = LoRAAdapter(model, rank=4, seed=1)
    w2_before = model.W2.copy()
    fit(adapter.loss_and_grads, adapter.trainable_params(), seqs, 2,
        epochs=5, batch_size=16, lr=5e-3, seed=0, log_every=0)
    np.testing.assert_array_equal(model.W2, w2_before)  # base frozen
    assert np.abs(adapter.B).sum() > 0  # adapter actually learned


def test_perplexity_finite_and_positive(tiny_setup):
    _, model, seqs = tiny_setup
    ppl = perplexity(model.loss_and_grads, seqs, 2)
    assert np.isfinite(ppl) and ppl > 1.0


def test_generation_deterministic(tiny_setup):
    tok, model, _ = tiny_setup
    ids = tok.encode("the cat")
    out1 = model.generate(ids, n_tokens=10, seed=42, bos_id=tok.bos_id)
    out2 = model.generate(ids, n_tokens=10, seed=42, bos_id=tok.bos_id)
    assert out1 == out2
    assert len(out1) == 10


def test_train_val_split_sizes():
    seqs = [np.array([i]) for i in range(20)]
    train, val = train_val_split(seqs, val_frac=0.2, seed=0)
    assert len(train) == 16 and len(val) == 4
    assert len({id(s) for s in train + val}) == 20
