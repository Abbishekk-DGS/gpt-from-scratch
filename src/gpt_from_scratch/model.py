"""A compact Llama-style decoder: RMSNorm, rotary embeddings, SwiGLU, tied weights, KV cache."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    rope_base: float = 10_000.0

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(seq_len: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); rotate pairs (even, odd) by position-dependent angles.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache: list | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if cache:
                k = torch.cat([cache[0], k], dim=2)
                v = torch.cat([cache[1], v], dim=2)
            cache[:] = [k, v]
        # With a cache, a single new query attends to every cached key: no mask needed.
        causal = cache is None or T > 1
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3 / 64 + 1) * 64  # Llama sizing, rounded to 64
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.n_embd), Attention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.n_embd), SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        cos, sin = rope_cache(cfg.block_size, cfg.n_embd // cfg.n_head, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # scaled init for residual projections (GPT-2)
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None, start_pos: int = 0):
        T = idx.size(1)
        if start_pos + T > self.cfg.block_size:
            raise ValueError(
                f"sequence of {start_pos + T} exceeds block_size {self.cfg.block_size}"
            )
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        x = self.drop(self.embed(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[i] if caches is not None else None)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
    ) -> torch.Tensor:
        """Sample with a KV cache: each step feeds one token instead of the whole context."""
        self.eval()
        idx = idx[:, -self.cfg.block_size :]
        caches: list[list] = [[] for _ in self.blocks]
        logits, _ = self(idx, caches=caches)
        pos = idx.size(1)
        for _ in range(max_new_tokens):
            next_id = sample(logits[:, -1, :], temperature, top_k, top_p)
            idx = torch.cat([idx, next_id], dim=1)
            if pos >= self.cfg.block_size:  # context full: restart cache on the trailing window
                window = idx[:, -(self.cfg.block_size - 1) :]
                caches = [[] for _ in self.blocks]
                logits, _ = self(window, caches=caches)
                pos = window.size(1)
            else:
                logits, _ = self(next_id, caches=caches, start_pos=pos)
                pos += 1
        return idx


def sample(logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None):
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        drop = cum - sorted_logits.softmax(-1) > top_p
        sorted_logits[drop] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(1, order, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1)
