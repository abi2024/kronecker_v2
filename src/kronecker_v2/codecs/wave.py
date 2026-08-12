"""Phase-bound Fourier byte codec (FHRR + fractional power position encoding).

Each byte VALUE gets a fixed random phasor vector; byte POSITION enters as a
phase rotation z^p rather than a one-hot slot. A token's code is the bundle
(sum) of its bound (value, position) pairs:

    kappa_wave(b) = (1/sqrt(L)) * sum_{p=1..L}  c_{b_p} (*) z^p

where (*) is elementwise complex multiplication (= phase addition, the FHRR
bind) and z^p raises the position base phasor to the p-th power (fractional
power encoding). There is no position ceiling: p is an exponent, not an index,
so tokens of any length encode and long tokens degrade gracefully instead of
being cropped.

Engineering contract: build in complex64 ONCE, store as real (real ++ imag) so
nothing complex enters the training loop. d_complex=2048 -> 4096 real dims,
matching the paper's 124M setting (pos_dim=16, D=256*16=4096) exactly, so the
projection Linear(4096, d_model) is parameter-identical to the one-hot arm.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

CHAR_DIM = 256


class WaveKroneckerCodec:
    """Drop-in alternative to the one-hot Kronecker codec. Frozen, deterministic."""

    name = "wave"

    def __init__(
        self,
        vocab_bytes: dict[int, bytes],
        d_complex: int = 2048,
        seed: int = 0,
        normalize: str = "sqrt_len",   # sqrt_len | l2 | znorm
        device: str = "cpu",
    ) -> None:
        self.vocab_bytes = vocab_bytes
        self.d_complex = d_complex
        self.code_dim = 2 * d_complex          # real ++ imag
        self.seed = seed
        self.normalize = normalize
        self.device = device

        # numpy PCG64 rather than torch.rand: torch only guarantees CPU RNG
        # reproducibility within one platform and build, so a torch-seeded codec
        # differs between a Windows laptop and a Linux GPU box. numpy's PCG64
        # stream is identical everywhere, which is what a frozen codec requires.
        rng = np.random.default_rng(seed)
        byte_phase = rng.uniform(-np.pi, np.pi, size=(CHAR_DIM, d_complex))
        pos_phase = rng.uniform(-np.pi, np.pi, size=(d_complex,))
        self._byte_phase = torch.from_numpy(byte_phase).to(torch.float32).to(device)
        self._pos_phase = torch.from_numpy(pos_phase).to(torch.float32).to(device)

    def _code_from_bytes(self, raw: bytes) -> torch.Tensor:
        """One token's complex code. No length ceiling."""
        if len(raw) == 0:
            return torch.zeros(self.d_complex, dtype=torch.complex64, device=self.device)
        vals = torch.tensor(list(raw), dtype=torch.long, device=self.device)
        pos = torch.arange(1, len(raw) + 1, device=self.device, dtype=torch.float32)
        # bind: phase(value) + p * phase(position_base)
        phase = self._byte_phase[vals] + pos.unsqueeze(1) * self._pos_phase.unsqueeze(0)
        bound = torch.polar(torch.ones_like(phase), phase)
        return bound.sum(dim=0)                                   # bundle

    def _post(self, z: torch.Tensor, length: int) -> torch.Tensor:
        if self.normalize == "sqrt_len":
            z = z / max(length, 1) ** 0.5
        elif self.normalize == "l2":
            z = z / (z.abs().pow(2).sum().sqrt() + 1e-6)
        elif self.normalize != "znorm":
            raise ValueError(f"unknown normalize={self.normalize!r}")
        out = torch.cat([z.real, z.imag], dim=-1)
        if self.normalize == "znorm":
            out = (out - out.mean()) / (out.std() + 1e-6)
        return out

    def encode_bytes(self, raw: bytes) -> torch.Tensor:
        """Any byte string -> [code_dim] float32. Works for OOV and any length."""
        return self._post(self._code_from_bytes(raw), len(raw))

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """[N] int64 -> [N, code_dim] float32."""
        ids = token_ids.reshape(-1).tolist()
        return torch.stack([self.encode_bytes(self.vocab_bytes[t]) for t in ids])

    def build_table(self, vocab_size: int, chunk: int = 8192) -> torch.Tensor:
        """Precompute the full [V, code_dim] frozen table once, in chunks."""
        rows = []
        for s in range(0, vocab_size, chunk):
            ids = torch.arange(s, min(s + chunk, vocab_size))
            rows.append(self.encode(ids))
        return torch.cat(rows, dim=0)

    def fingerprint(self, vocab_size: int = 131_072) -> str:
        t = self.build_table(vocab_size)
        return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()