"""Shape and lookup contracts.

Codes are [N, code_dim]; embeddings are [B, T, d_model]; and a token embeds to
the same vector wherever it appears. That last one sounds trivial and is not:
it is the property that makes the embedding a lookup rather than a function of
context, and a codec bug that broke it would be invisible in the loss curve.
"""

from __future__ import annotations

import torch

from kronecker_v2.codecs.base import OneHotCodec
from kronecker_v2.codecs.wave import WaveKroneckerCodec
from kronecker_v2.embedding import CodecEmbedding

VOCAB = {
    0: b"the", 1: b" the", 2: "कार्य".encode(), 3: "कार्यक्रम".encode(),
    4: "తెలుగు".encode(), 5: b"", 6: b"a" * 40, 7: b"\xef\xbf\xbd",
}
D_MODEL = 64


def _arms():
    return [OneHotCodec(vocab_bytes=VOCAB, pos_dim=16),
            WaveKroneckerCodec(vocab_bytes=VOCAB, d_complex=2048)]


def test_code_shape():
    ids = torch.tensor([0, 2, 4, 6])
    for c in _arms():
        out = c.encode(ids)
        assert out.shape == (4, c.code_dim), f"{c.name}: {tuple(out.shape)}"
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all(), f"{c.name} produced non-finite values"


def test_embedding_shape():
    ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    for c in _arms():
        emb = CodecEmbedding(c, d_model=D_MODEL, vocab_size=len(VOCAB))
        out = emb(ids)
        assert out.shape == (2, 3, D_MODEL), f"{c.name}: {tuple(out.shape)}"


def test_same_token_same_code():
    """The LOOKUP is position-invariant. Exact, because it is an index_select."""
    ids = torch.tensor([[2, 0, 2], [2, 4, 2]])
    for c in _arms():
        emb = CodecEmbedding(c, d_model=D_MODEL, vocab_size=len(VOCAB))
        codes = emb.codes.index_select(0, ids.reshape(-1)).view(*ids.shape, c.code_dim)
        ref = codes[0, 0]
        for b, t in [(0, 2), (1, 0), (1, 2)]:
            assert torch.equal(codes[b, t], ref), f"{c.name}: code differs by position"


def test_same_token_same_vector():
    """And the embedding follows. Not bitwise: BLAS may schedule the GEMM
    differently for identical rows at different batch positions, which is
    numerical noise, not a codec property."""
    ids = torch.tensor([[2, 0, 2], [2, 4, 2]])
    for c in _arms():
        emb = CodecEmbedding(c, d_model=D_MODEL, vocab_size=len(VOCAB))
        out = emb(ids)
        ref = out[0, 0]
        for b, t in [(0, 2), (1, 0), (1, 2)]:
            assert torch.allclose(out[b, t], ref, atol=1e-6), \
                f"{c.name}: token 2 embedded differently"


def test_only_the_projection_is_trainable():
    """The codec contributes zero parameters. That is the whole point."""
    for c in _arms():
        emb = CodecEmbedding(c, d_model=D_MODEL, vocab_size=len(VOCAB))
        trainable = {n for n, p in emb.named_parameters() if p.requires_grad}
        assert trainable == {"projection.weight"}, f"{c.name}: {trainable}"


def test_long_token_survives_in_wave_but_not_onehot():
    """Token 6 is 40 bytes. One-hot at pos_dim=16 sees 16 of them; wave sees all."""
    long_bytes, prefix = VOCAB[6], b"a" * 16
    oh, wv = _arms()
    assert torch.equal(oh.encode_bytes(long_bytes), oh.encode_bytes(prefix))
    assert not torch.equal(wv.encode_bytes(long_bytes), wv.encode_bytes(prefix))