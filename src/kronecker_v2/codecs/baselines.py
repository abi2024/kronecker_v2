"""Learned-embedding baselines a referee will ask about.

Unlike the Kronecker codecs these are NOT frozen codes fed through a shared
projection — they hold per-token trainable parameters. They therefore do not
implement the ``Codec`` protocol; they are ``nn.Module`` replacements for
``wte``, like the dense arm.

The point of both is parameter parity with the codec arms, so the helpers below
solve for the setting that matches a target budget rather than leaving it to be
guessed:

    hash embeddings   B*d + V*k   (Svenstrup et al., arXiv 1709.03933)
    ALBERT            V*r + r*d   (Lan et al., ICLR 2020)

Note what the second formula says. ALBERT's cost carries a factor of V, so at
V = 131,072 a parameter-matched factorization is forced into a very tight rank
— 12 at d_model 384, 16 at 512. The codec's cost has no V in it at all. That
asymmetry is the structural argument this project rests on, and it is visible
here in the arithmetic before any model is trained.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

VOCAB_SIZE = 131_072


# ------------------------------------------------------------- budget maths --
def codec_budget(code_dim: int, d_model: int) -> int:
    """Trainable parameters in a codec arm: one Linear(code_dim, d_model)."""
    return code_dim * d_model


def matched_rank(target: int, d_model: int, vocab_size: int = VOCAB_SIZE) -> int:
    """ALBERT rank r solving r*(V + d) ~= target."""
    return max(1, round(target / (vocab_size + d_model)))


def matched_buckets(target: int, d_model: int, n_hashes: int = 2,
                    vocab_size: int = VOCAB_SIZE) -> int:
    """Hash-embedding bucket count B solving B*d + V*k ~= target."""
    return max(1, round((target - vocab_size * n_hashes) / d_model))


# ------------------------------------------------------------------ modules --
class ALBERTEmbedding(nn.Module):
    """V x r factor, then a shared r x d_model projection."""

    name = "albert"

    def __init__(self, vocab_size: int, d_model: int, rank: int) -> None:
        super().__init__()
        self.vocab_size, self.d_model, self.rank = vocab_size, d_model, rank
        self.factor = nn.Embedding(vocab_size, rank)
        self.projection = nn.Linear(rank, d_model, bias=False)
        nn.init.normal_(self.factor.weight, std=0.02)
        nn.init.normal_(self.projection.weight, std=1.0 / rank**0.5)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.factor(token_ids))

    def extra_repr(self) -> str:
        return f"vocab={self.vocab_size}, rank={self.rank}, d_model={self.d_model}"


class HashEmbedding(nn.Module):
    """Hash embeddings: k hashed lookups into a shared table, learned weights.

    Each token hashes to ``n_hashes`` rows of a B x d table; the token's vector
    is the weighted sum of those rows, with the weights learned per token. The
    hash assignment is fixed and generated from numpy's PCG64 — torch's CPU RNG
    is not reproducible across platforms, and a codec-like component whose
    identity changes between the machine that audits it and the machine that
    trains it is a silent correctness bug.
    """

    name = "hash"

    def __init__(self, vocab_size: int, d_model: int, n_buckets: int,
                 n_hashes: int = 2, seed: int = 0) -> None:
        super().__init__()
        self.vocab_size, self.d_model = vocab_size, d_model
        self.n_buckets, self.n_hashes = n_buckets, n_hashes

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_buckets, size=(vocab_size, n_hashes), dtype=np.int64)
        self.register_buffer("hash_idx", torch.from_numpy(idx), persistent=False)

        self.table = nn.Embedding(n_buckets, d_model)
        self.weights = nn.Embedding(vocab_size, n_hashes)
        nn.init.normal_(self.table.weight, std=0.02)
        nn.init.normal_(self.weights.weight, mean=1.0 / n_hashes, std=0.1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        idx = self.hash_idx[token_ids]                    # [..., k]
        vecs = self.table(idx)                            # [..., k, d]
        w = self.weights(token_ids).unsqueeze(-1)         # [..., k, 1]
        return (vecs * w).sum(dim=-2)

    def extra_repr(self) -> str:
        return (f"vocab={self.vocab_size}, buckets={self.n_buckets}, "
                f"hashes={self.n_hashes}, d_model={self.d_model}")


class DenseEmbedding(nn.Embedding):
    """Plain V x d_model table. Reference arm — NOT parameter-matched."""

    name = "dense"

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__(vocab_size, d_model)
        nn.init.normal_(self.weight, std=0.02)


# ----------------------------------------------------------------- factory --
def build_matched(name: str, d_model: int, code_dim: int = 4096,
                  vocab_size: int = VOCAB_SIZE, seed: int = 0) -> nn.Module:
    """Construct a baseline sized to the codec arms' parameter budget."""
    target = codec_budget(code_dim, d_model)
    if name == "albert":
        return ALBERTEmbedding(vocab_size, d_model,
                               matched_rank(target, d_model, vocab_size))
    if name == "hash":
        return HashEmbedding(vocab_size, d_model,
                             matched_buckets(target, d_model, 2, vocab_size),
                             n_hashes=2, seed=seed)
    if name == "dense":
        return DenseEmbedding(vocab_size, d_model)
    raise ValueError(f"unknown baseline {name!r}")


def budget_report(d_model: int, code_dim: int = 4096,
                  vocab_size: int = VOCAB_SIZE) -> str:
    """Human-readable parity table — paste into the paper, don't retype it."""
    target = codec_budget(code_dim, d_model)
    r = matched_rank(target, d_model, vocab_size)
    b = matched_buckets(target, d_model, 2, vocab_size)
    lines = [f"d_model={d_model}, code_dim={code_dim}, V={vocab_size:,}",
             f"  codec  (onehot / wave)  {target:>12,}   target"]
    for nm, n in (("albert", r * (vocab_size + d_model)),
                  ("hash", b * d_model + vocab_size * 2)):
        lines.append(f"  {nm:<22}{n:>12,}   {100*(n-target)/target:+.2f}%"
                     + (f"   rank={r}" if nm == "albert" else f"   buckets={b}"))
    lines.append(f"  dense (reference)     {vocab_size*d_model:>12,}   "
                 f"{vocab_size*d_model/target:.0f}x")
    return "\n".join(lines)
