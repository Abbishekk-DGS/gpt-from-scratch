"""Training loop: AdamW, warmup + cosine LR, gradient clipping, periodic eval, checkpoints."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .model import GPT, GPTConfig


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 3000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    seed: int = 1337


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step: int, tc: TrainConfig) -> float:
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    progress = min(1.0, (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps))
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * progress))


def get_batch(data: np.ndarray, block: int, batch: int, device: str, rng: np.random.Generator):
    starts = rng.integers(0, len(data) - block - 1, size=batch)
    x = torch.from_numpy(np.stack([data[s : s + block] for s in starts]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[s + 1 : s + 1 + block] for s in starts]).astype(np.int64))
    return x.to(device), y.to(device)


def configure_optimizer(model: GPT, tc: TrainConfig) -> torch.optim.AdamW:
    # Decay matrices only; norms and (tied) embeddings stay undecayed.
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "embed" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "embed" in n]
    groups = [
        {"params": decay, "weight_decay": tc.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model, splits: dict[str, np.ndarray], tc: TrainConfig, device, rng) -> dict:
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [
            model(*get_batch(data, model.cfg.block_size, tc.batch_size, device, rng))[1].item()
            for _ in range(tc.eval_batches)
        ]
        out[name] = float(np.mean(losses))
    model.train()
    return out


def train(
    tokens: np.ndarray,
    cfg: GPTConfig,
    tc: TrainConfig,
    out_dir: Path | None = None,
    device: str | None = None,
    log=print,
) -> tuple[GPT, list[dict]]:
    device = device or pick_device()
    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)
    split = int(0.9 * len(tokens))
    splits = {"train": tokens[:split], "val": tokens[split:]}

    model = GPT(cfg).to(device)
    opt = configure_optimizer(model, tc)
    log(f"model: {model.num_params() / 1e6:.2f}M params on {device}")

    history: list[dict] = []
    best_val = float("inf")
    t0 = time.time()
    for step in range(tc.max_steps + 1):
        if step % tc.eval_every == 0 or step == tc.max_steps:
            losses = estimate_loss(model, splits, tc, device, rng)
            history.append({"step": step, **losses})
            log(
                f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                f"| ppl {math.exp(losses['val']):.1f} | {time.time() - t0:.0f}s"
            )
            if out_dir and losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(model, out_dir / "ckpt.pt", step, losses["val"])
        if step == tc.max_steps:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, tc)
        x, y = get_batch(splits["train"], cfg.block_size, tc.batch_size, device, rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
    return model, history


def save_checkpoint(model: GPT, path: Path, step: int, val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": model.cfg.to_dict(), "model": model.state_dict(), "step": step, "val": val_loss},
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> GPT:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device)
