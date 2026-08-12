"""The Codec contract every arm satisfies.

The transformer body never learns which codec produced its codes. That is what
makes "we changed only the embedding" a property of the code rather than a
promise in the README.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Codec(Protocol):
    name: str
    code_dim: int

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """[N] int64 -> [N, code_dim] float32."""
        ...

    def fingerprint(self, vocab_size: int = 131_072) -> str:
        """sha256 over the full-vocab code table. Pinned by tests."""
        ...


class OneHotCodec:
    """V1 baseline. Thin adapter over the published reference implementation.

    Do not reimplement the one-hot codec: a reimplementation that differs from
    the reference by any detail turns every downstream comparison into a
    comparison against a strawman.
    """

    name = "onehot"

    def __init__(self, vocab_bytes: dict[int, bytes], pos_dim: int = 16,
                 char_dim: int = 256) -> None:
        from kronecker_embeddings import codec_output_dim

        self.vocab_bytes = vocab_bytes
        self.pos_dim = pos_dim
        self.char_dim = char_dim
        self.code_dim = codec_output_dim(char_dim, pos_dim)

    def encode_bytes(self, raw: bytes) -> torch.Tensor:
        from kronecker_embeddings import encode_single, utf8_safe_truncate

        if len(raw) > self.pos_dim:
            raw = utf8_safe_truncate(raw, self.pos_dim)
        return encode_single(raw, char_dim=self.char_dim, pos_dim=self.pos_dim)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.reshape(-1).tolist()
        return torch.stack([self.encode_bytes(self.vocab_bytes[t]) for t in ids])

    def build_table(self, vocab_size: int, chunk: int = 8192) -> torch.Tensor:
        rows = []
        for s in range(0, vocab_size, chunk):
            rows.append(self.encode(torch.arange(s, min(s + chunk, vocab_size))))
        return torch.cat(rows, dim=0)

    def fingerprint(self, vocab_size: int = 131_072) -> str:
        t = self.build_table(vocab_size)
        return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()