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
from kronecker_v2.codecs.baselines import build_matched, codec_budget
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


# --- learned baselines --------------------------------------------------------
# hash and ALBERT hold per-token trainable parameters, so they are nn.Modules
# rather than Codecs. Integer rank/bucket counts cannot hit the codec budget
# exactly; the contract is parity within 1%, and the exact counts are reported
# by ``baselines.budget_report`` so the paper quotes measurements, not intents.

TOLERANCE = 0.01


@pytest.mark.parametrize("d_model", [384, 512])
def test_learned_baselines_match_codec_budget(d_model):
    target = codec_budget(4096, d_model)
    for name in ("albert", "hash"):
        n = sum(p.numel() for p in build_matched(name, d_model).parameters())
        assert abs(n - target) / target < TOLERANCE, (
            f"{name} at d_model={d_model}: {n:,} vs target {target:,}")


@pytest.mark.parametrize("d_model", [384, 512])
def test_dense_is_not_matched_and_we_say_so(d_model):
    """Dense is the reference arm. It must be ~32x, and that is the point."""
    n = sum(p.numel() for p in build_matched("dense", d_model).parameters())
    assert n / codec_budget(4096, d_model) > 30


def test_baselines_produce_d_model_output():
    for name in ("albert", "hash", "dense"):
        out = build_matched(name, D_MODEL)(torch.tensor([[0, 1, 2], [3, 4, 5]]))
        assert out.shape == (2, 3, D_MODEL), f"{name}: {tuple(out.shape)}"


def test_hash_assignment_is_deterministic():
    """Hash indices come from numpy PCG64 — identical on every platform."""
    a = build_matched("hash", D_MODEL, seed=0)
    b = build_matched("hash", D_MODEL, seed=0)
    assert torch.equal(a.hash_idx, b.hash_idx)
    assert not torch.equal(a.hash_idx, build_matched("hash", D_MODEL, seed=1).hash_idx)
