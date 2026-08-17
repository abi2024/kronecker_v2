"""Mechanism ablations — what is the phase codec's advantage actually made of?

At pos_dim=32 the one-hot codec merges nothing, yet the wave codec still wins by
~0.061 nats. That advantage cannot be about collisions. Two candidate
explanations remain, and they are separable:

  DENSITY.  A one-hot code at pos_dim=32 is 8,192 dimensions holding at most 32
            non-zeros — 0.4% dense. The wave code is fully dense. The shared
            projection sees far richer gradient from a dense input, and that has
            nothing to do with what the code MEANS.

  BINDING.  The wave code multiplies each byte's phase by its position, so order
            is encoded. Remove that and the code becomes a bag of bytes — every
            anagram identical — but it stays just as dense.

Two ablations complete a 2x2:

                    sparse code          dense code
    no order        (not constructed)    BagOfBytes
    order encoded   OneHot               Wave, RandomProjectedOneHot

  * RandomProjectedOneHot carries EXACTLY the one-hot information — a square
    Gaussian matrix is invertible almost surely, so nothing is lost — but the
    code is dense. OneHot vs this isolates density with information held fixed.
  * BagOfBytes uses the SAME frozen phasors as the wave codec and the same
    bundling, but drops the position rotation. Wave vs this isolates binding
    with density held fixed.

If RandomProjectedOneHot closes most of the gap, the story is density and the
Fourier construction is incidental. If BagOfBytes collapses, order binding is
doing the work. Both are single overnight runs.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from .base import OneHotCodec
from .wave import WaveKroneckerCodec

CHAR_DIM = 256


def _post(z: torch.Tensor, length: int, normalize: str) -> torch.Tensor:
    """Mirrors WaveKroneckerCodec._post so the arms differ only where intended."""
    if normalize == "sqrt_len":
        z = z / max(length, 1) ** 0.5
    elif normalize == "l2":
        z = z / (z.abs().pow(2).sum().sqrt() + 1e-6)
    elif normalize != "znorm":
        raise ValueError(f"unknown normalize={normalize!r}")
    out = torch.cat([z.real, z.imag], dim=-1)
    if normalize == "znorm":
        out = (out - out.mean()) / (out.std() + 1e-6)
    return out


class BagOfBytesCodec:
    """The wave codec with the position rotation removed.

    Same frozen phasors, same bundling, same normalisation — order is simply not
    encoded, so `dog` and `god` receive identical codes and so does every other
    anagram. Density is unchanged, which is the point: this isolates binding.
    """

    name = "bag"

    def __init__(self, vocab_bytes: dict[int, bytes], d_complex: int = 2048,
                 seed: int = 0, normalize: str = "l2", device: str = "cpu") -> None:
        # borrow the frozen phasors so the only difference is the rotation
        ref = WaveKroneckerCodec(vocab_bytes={}, d_complex=d_complex, seed=seed,
                                 normalize=normalize, device=device)
        self._byte_phase = ref._byte_phase
        self.vocab_bytes = vocab_bytes
        self.d_complex, self.code_dim = d_complex, 2 * d_complex
        self.seed, self.normalize, self.device = seed, normalize, device

    def encode_bytes(self, raw: bytes) -> torch.Tensor:
        if len(raw) == 0:
            z = torch.zeros(self.d_complex, dtype=torch.complex64, device=self.device)
            return _post(z, 0, self.normalize)
        vals = torch.tensor(list(raw), dtype=torch.long, device=self.device)
        phase = self._byte_phase[vals]                 # NO position term
        z = torch.polar(torch.ones_like(phase), phase).sum(dim=0)
        return _post(z, len(raw), self.normalize)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.reshape(-1).tolist()
        return torch.stack([self.encode_bytes(self.vocab_bytes[t]) for t in ids])

    def build_table(self, vocab_size: int, chunk: int = 8192) -> torch.Tensor:
        rows = [self.encode(torch.arange(s, min(s + chunk, vocab_size)))
                for s in range(0, vocab_size, chunk)]
        return torch.cat(rows, dim=0)

    def fingerprint(self, vocab_size: int = 131_072) -> str:
        t = self.build_table(vocab_size)
        return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


class RandomProjectedOneHotCodec:
    """The one-hot code made dense, with its information preserved.

    A square Gaussian matrix is invertible almost surely, so this carries
    exactly what the one-hot grid carries — including its truncation and its
    collisions. Only the sparsity is gone. Comparing it against OneHot isolates
    density from information content.
    """

    name = "rp"

    def __init__(self, vocab_bytes: dict[int, bytes], pos_dim: int = 16,
                 seed: int = 0, normalize: str = "l2", device: str = "cpu") -> None:
        self._oh = OneHotCodec(vocab_bytes=vocab_bytes, pos_dim=pos_dim)
        self.vocab_bytes = vocab_bytes
        self.code_dim = self._oh.code_dim
        self.pos_dim, self.seed, self.normalize, self.device = (
            pos_dim, seed, normalize, device)
        # numpy PCG64, as everywhere else: torch's CPU RNG is not portable
        rng = np.random.default_rng(10_000 + seed)
        R = rng.normal(0.0, 1.0 / np.sqrt(self.code_dim),
                       size=(self.code_dim, self.code_dim))
        self.R = torch.from_numpy(R).to(torch.float32).to(device)

    def encode_bytes(self, raw: bytes) -> torch.Tensor:
        v = self._oh.encode_bytes(raw).to(self.device) @ self.R
        if self.normalize == "l2":
            v = v / (v.norm() + 1e-6)
        elif self.normalize == "znorm":
            v = (v - v.mean()) / (v.std() + 1e-6)
        return v

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.reshape(-1).tolist()
        return torch.stack([self.encode_bytes(self.vocab_bytes[t]) for t in ids])

    def build_table(self, vocab_size: int, chunk: int = 8192) -> torch.Tensor:
        rows = [self.encode(torch.arange(s, min(s + chunk, vocab_size)))
                for s in range(0, vocab_size, chunk)]
        return torch.cat(rows, dim=0)

    def fingerprint(self, vocab_size: int = 131_072) -> str:
        t = self.build_table(vocab_size)
        return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def density(code: torch.Tensor, eps: float = 1e-8) -> float:
    """Fraction of non-zero entries — the quantity these ablations vary."""
    return float((code.abs() > eps).float().mean())
