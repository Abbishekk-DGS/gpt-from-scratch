"""gpt-from-scratch: prepare data, train, and sample from a GPT you built yourself.

gpt-from-scratch prepare  --input data/shakespeare.txt --vocab 2048
gpt-from-scratch train    --preset small --steps 3000
gpt-from-scratch sample   --prompt "ROMEO:" --tokens 300
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .model import GPTConfig
from .tokenizer import BPETokenizer
from .train import TrainConfig, load_checkpoint, pick_device, train

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

PRESETS = {
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=128),  # ~1M params, CPU minutes
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=256),  # ~11M params, laptop GPU
    "base": dict(n_layer=12, n_head=12, n_embd=768, block_size=512),  # ~85M, real GPU
}


def cmd_prepare(args) -> None:
    src = Path(args.input)
    if not src.exists():
        print(f"{src} not found, downloading Tiny Shakespeare...")
        src.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, src)
    text = src.read_text(encoding="utf-8")
    print(f"training BPE ({args.vocab} tokens) on {len(text):,} chars")
    tok = BPETokenizer.train(text, args.vocab, verbose=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "tokenizer.json")
    ids = np.array(tok.encode(text), dtype=np.uint16)
    ids.tofile(out / "tokens.bin")
    print(f"{len(ids):,} tokens ({len(text) / len(ids):.2f} chars/token) -> {out}")


def cmd_train(args) -> None:
    data = Path(args.data)
    tok = BPETokenizer.load(data / "tokenizer.json")
    tokens = np.fromfile(data / "tokens.bin", dtype=np.uint16)
    cfg = GPTConfig(vocab_size=tok.vocab_size, dropout=args.dropout, **PRESETS[args.preset])
    tc = TrainConfig(max_steps=args.steps, batch_size=args.batch, lr=args.lr)
    _, history = train(tokens, cfg, tc, out_dir=Path(args.out), device=args.device)
    (Path(args.out) / "history.json").write_text(json.dumps(history, indent=2))


def cmd_sample(args) -> None:
    device = args.device or pick_device()
    tok = BPETokenizer.load(Path(args.data) / "tokenizer.json")
    model = load_checkpoint(Path(args.out) / "ckpt.pt", device)
    torch.manual_seed(args.seed)
    idx = torch.tensor([tok.encode(args.prompt)], device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gpt-from-scratch",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} · by Abbishekk DGS · abbishekkdgs.tech",
    )
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--out", default="runs/latest")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (auto by default)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--input", default="data/shakespeare.txt")
    p.add_argument("--vocab", type=int, default=2048)
    p.set_defaults(fn=cmd_prepare, out_override=True)

    t = sub.add_parser("train")
    t.add_argument("--preset", choices=PRESETS, default="small")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--dropout", type=float, default=0.1)
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("sample")
    s.add_argument("--prompt", default="\n")
    s.add_argument("--tokens", type=int, default=300)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sample)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        args.out = args.data  # prepare writes into the data directory
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
