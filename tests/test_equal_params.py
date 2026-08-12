"""The equal-parameter contract, enforced rather than asserted in prose.

Every comparison in this project rests on the arms having identical trainable
parameter counts. If that ever stops being true, a loss difference stops meaning
anything, and the failure would be silent. So it is a test.

Fast and hermetic: code_dim does not depend on the vocabulary, so no tokenizer
is loaded here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from kronecker_v2.codecs.base import OneHotCodec
from kronecker_v2.codecs.wave import WaveKroneckerCodec

POS_DIM = 16          # the paper's 124M setting: D = 256 * 16 = 4096
D_COMPLEX = 2048      # 2048 complex -> 4096 real. Matched by construction.
D_MODEL = 768


def codecs():
    """One instance of each arm. Empty vocab: code_dim is vocab-independent."""
    return [
        OneHotCodec(vocab_bytes={}, pos_dim=POS_DIM),
        WaveKroneckerCodec(vocab_bytes={}, d_complex=D_COMPLEX),
    ]


def test_code_dims_match():
    dims = {c.name: c.code_dim for c in codecs()}
    assert len(set(dims.values())) == 1, f"code_dim mismatch: {dims}"


def test_projection_param_counts_match():
    counts = {
        c.name: sum(p.numel() for p in nn.Linear(c.code_dim, D_MODEL, bias=False).parameters())
        for c in codecs()
    }
    assert len(set(counts.values())) == 1, f"trainable param mismatch: {counts}"


def test_codecs_produce_the_declared_dim():
    for c in codecs():
        out = c.encode_bytes("कार्य".encode())
        assert out.shape == (c.code_dim,), f"{c.name}: {tuple(out.shape)} != ({c.code_dim},)"
        assert out.dtype == torch.float32


@pytest.mark.parametrize("pos_dim,d_complex", [(8, 1024), (16, 2048), (32, 4096)])
def test_contract_holds_across_settings(pos_dim, d_complex):
    """The pairing rule is d_complex = 128 * pos_dim. Verify it at three points."""
    a = OneHotCodec(vocab_bytes={}, pos_dim=pos_dim)
    b = WaveKroneckerCodec(vocab_bytes={}, d_complex=d_complex)
    assert a.code_dim == b.code_dim