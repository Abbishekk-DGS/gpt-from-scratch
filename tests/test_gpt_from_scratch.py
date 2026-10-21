import numpy as np
import torch

from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.tokenizer import BPETokenizer
from gpt_from_scratch.train import TrainConfig, lr_at, train

TEXT = "the cat sat on the mat. the cat ate the rat. " * 50 + "வணக்கம் உலகம் 🚀"


def tiny_cfg(vocab: int = 300) -> GPTConfig:
    return GPTConfig(vocab_size=vocab, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_bpe_roundtrip_any_unicode():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    for s in [TEXT, "unseen words ok", "தமிழ் text 🙂", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_compresses_and_persists(tmp_path):
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    assert len(tok.encode(TEXT)) < len(TEXT.encode()) / 2
    tok.save(tmp_path / "tok.json")
    again = BPETokenizer.load(tmp_path / "tok.json")
    assert again.encode(TEXT) == tok.encode(TEXT)


def test_model_is_causal():
    torch.manual_seed(0)
    model = GPT(tiny_cfg()).eval()
    a = torch.randint(0, 300, (1, 16))
    b = a.clone()
    b[0, 10:] = (b[0, 10:] + 1) % 300  # change only the future
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-5)
    assert not torch.allclose(la[0, 10:], lb[0, 10:])


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = GPT(tiny_cfg()).eval()
    seq = torch.randint(0, 300, (1, 12))
    full, _ = model(seq)
    caches = [[] for _ in model.blocks]
    step_logits, _ = model(seq[:, :8], caches=caches)
    outs = [step_logits[:, -1]]
    for pos in range(8, 12):
        logit, _ = model(seq[:, pos : pos + 1], caches=caches, start_pos=pos)
        outs.append(logit[:, -1])
    assert torch.allclose(torch.stack(outs[:-1], 1), full[:, 7:11], atol=1e-4)


def test_generate_past_block_size():
    model = GPT(tiny_cfg())
    out = model.generate(torch.zeros((1, 4), dtype=torch.long), max_new_tokens=60)
    assert out.shape == (1, 64)


def test_lr_schedule_shape():
    tc = TrainConfig(max_steps=100, warmup_steps=10, lr=1.0, min_lr=0.1)
    assert lr_at(0, tc) < lr_at(9, tc) <= 1.0
    assert abs(lr_at(100, tc) - 0.1) < 1e-9


def test_training_reduces_loss():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    tokens = np.array(tok.encode(TEXT * 4), dtype=np.uint16)
    tc = TrainConfig(max_steps=60, batch_size=8, eval_every=60, eval_batches=2, warmup_steps=5)
    _, history = train(tokens, tiny_cfg(tok.vocab_size), tc, device="cpu", log=lambda _: None)
    assert history[-1]["train"] < history[0]["train"] - 1.0
