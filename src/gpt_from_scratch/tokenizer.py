"""Byte-level BPE tokenizer, trained from scratch.

Works on raw UTF-8 bytes, so it can encode any string (English, Tamil, emoji, code)
with no unknown tokens. Text is first pre-split with a GPT-2 style regex so merges
never cross word boundaries.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PRETOKENIZE = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")


def _merge(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = merges or []
        self._rebuild()

    def _rebuild(self) -> None:
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> BPETokenizer:
        """Learn merges with incremental pair counting: each merge touches only affected words."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the byte alphabet)")
        counted = Counter(chunk.encode() for chunk in PRETOKENIZE.findall(text))
        words = [list(w) for w in counted]
        freqs = list(counted.values())
        pair_counts: Counter[tuple[int, int]] = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, word in enumerate(words):
            for pair in zip(word, word[1:], strict=False):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)

        merges: list[tuple[int, int]] = []
        for step in range(vocab_size - 256):
            if not pair_counts:
                break
            pair, freq = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if freq < 2:
                break
            new_id = 256 + step
            merges.append(pair)
            for wi in list(where.pop(pair, ())):
                word, f = words[wi], freqs[wi]
                for old in zip(word, word[1:], strict=False):
                    pair_counts[old] -= f
                    if pair_counts[old] <= 0:
                        del pair_counts[old]
                word = list(_merge(tuple(word), pair, new_id))
                words[wi] = word
                for new in zip(word, word[1:], strict=False):
                    pair_counts[new] += f
                    where[new].add(wi)
            pair_counts.pop(pair, None)
            if verbose and step % 250 == 0:
                print(f"merge {step}: {pair} -> {new_id} (freq {freq})")
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode())
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:], strict=False))
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            ids = list(_merge(tuple(ids), best, 256 + self.ranks[best]))
        if len(self._cache) < 100_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for chunk in PRETOKENIZE.findall(text) for i in self._encode_chunk(chunk)]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"type": "byte-bpe", "merges": self.merges}))

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        data = json.loads(path.read_text())
        return cls([tuple(m) for m in data["merges"]])
