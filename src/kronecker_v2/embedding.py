"""The nn.Module every training run uses.

Frozen code table (a buffer, never trained) + one shared trainable projection.
The transformer body cannot tell which codec produced the codes — that is what
keeps "we changed only the embedding" true in code rather than in prose.

Drop-in for nn.Embedding: forward([B, T] int64) -> [B, T, d_model].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CodecEmbedding(nn.Module):
    """Wrap any Codec into a trainable embedding layer."""

    def __init__(self, codec, d_model: int, vocab_size: int = 131_072,
                 chunk: int = 8192) -> None:
        super().__init__()
        self.codec = codec
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.code_dim = codec.code_dim

        with torch.no_grad():
            table = codec.build_table(vocab_size, chunk=chunk)
        # persistent=False: the table is a pure function of the codec + vocab,
        # so it is rebuilt on load rather than shipped inside the checkpoint.
        self.register_buffer("codes", table, persistent=False)

        self.projection = nn.Linear(self.code_dim, d_model, bias=False)
        nn.init.normal_(self.projection.weight, mean=0.0,
                        std=1.0 / math.sqrt(self.code_dim))

    @property
    def num_embeddings(self) -> int:
        return self.vocab_size

    @property
    def embedding_dim(self) -> int:
        return self.d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        codes = self.codes.index_select(0, token_ids.reshape(-1))
        codes = codes.view(*token_ids.shape, self.code_dim)
        return self.projection(codes.to(self.projection.weight.dtype))

    def extra_repr(self) -> str:
        return (f"codec={self.codec.name!r}, vocab_size={self.vocab_size}, "
                f"code_dim={self.code_dim}, d_model={self.d_model}")