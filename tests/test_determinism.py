"""Codec fingerprints, at the right level of strictness.

A codec is a frozen function. If its output drifts mid-project, every earlier
result becomes uncomparable with every later one, and nothing would otherwise
warn you. But "identical" has to be defined carefully:

  * The PHASE TABLES are bit-stable everywhere. They come from numpy's PCG64
    stream plus IEEE-754 arithmetic, with no transcendental functions, so a
    Windows laptop and a Linux GPU box agree exactly. This is the pin that
    actually guards the RNG and the seed.

  * The CODES are not bit-stable across operating systems. Building them calls
    cos/sin via torch.polar, and libm differs in the last bits between MSVC and
    glibc. So they are pinned by value at a tolerance far tighter than any real
    logic change and far looser than libm noise.

Determinism WITHIN a machine is still exact, and is tested separately.
"""

from __future__ import annotations

import hashlib

import torch

from kronecker_v2.codecs.base import OneHotCodec
from kronecker_v2.codecs.wave import WaveKroneckerCodec

PROBE = [b"the", b" the", "कार्य".encode(), "తెలుగు".encode(),
         "കേരളം".encode(), b"", b"\xef\xbf\xbd", b"a" * 40]

# Regenerate with:  python tests/test_determinism.py
EXACT = {
    # platform-stable: integer RNG + IEEE arithmetic only
    "wave@2048.phases": "b4b03664bc0a805746bf911e8c4e851fae0615ca5028a3afdd0dc6dbff6d438c",
    # platform-stable: the one-hot codec uses no transcendental functions
    "onehot@16.codes": "2f3046d440bdef2f467129969c0c244250280e65f2df638864b9f8ed7c7ea445",
}
# platform-tolerant: first 6 values of the code for PROBE[2] ("कार्य")
WAVE_CODE_HEAD = [-0.080267, -0.434995, -0.056143, 0.215077, -1.569017, 2.598114]
TOL = 1e-4


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def _codes(codec) -> torch.Tensor:
    return torch.stack([codec.encode_bytes(b) for b in PROBE])


def _arms():
    return {
        "onehot@16": OneHotCodec(vocab_bytes={}, pos_dim=16),
        "wave@2048": WaveKroneckerCodec(vocab_bytes={}, d_complex=2048),
    }


def test_encoding_is_deterministic():
    """Same input twice on this machine: byte-identical. This one is exact."""
    for name, c in _arms().items():
        assert torch.equal(_codes(c), _codes(c)), f"{name} is not deterministic"


def test_phase_tables_pinned():
    """Guards the RNG and seed. Bit-stable on every platform."""
    if EXACT["wave@2048.phases"] == "PASTE_ME":
        import pytest
        pytest.skip("not pinned yet — run this file directly")
    w = _arms()["wave@2048"]
    got = _sha(torch.cat([w._byte_phase.reshape(-1), w._pos_phase.reshape(-1)]))
    assert got == EXACT["wave@2048.phases"], "wave phase table changed"


def test_onehot_codes_pinned():
    """The one-hot codec is pure scatter-add arithmetic, so this can be exact."""
    if EXACT["onehot@16.codes"] == "PASTE_ME":
        import pytest
        pytest.skip("not pinned yet — run this file directly")
    assert _sha(_codes(_arms()["onehot@16"])) == EXACT["onehot@16.codes"], \
        "one-hot codec changed"


def test_wave_codes_pinned_by_value():
    """Catches any logic change; tolerates cross-platform libm differences."""
    if WAVE_CODE_HEAD is None:
        import pytest
        pytest.skip("not pinned yet — run this file directly")
    got = _codes(_arms()["wave@2048"])[2, :6]
    assert torch.allclose(got, torch.tensor(WAVE_CODE_HEAD), atol=TOL), \
        f"wave codec changed: {got.tolist()} vs {WAVE_CODE_HEAD}"


if __name__ == "__main__":
    a = _arms()
    w = a["wave@2048"]
    print('EXACT = {')
    print(f'    "wave@2048.phases": "{_sha(torch.cat([w._byte_phase.reshape(-1), w._pos_phase.reshape(-1)]))}",')
    print(f'    "onehot@16.codes": "{_sha(_codes(a["onehot@16"]))}",')
    print('}')
    print(f"WAVE_CODE_HEAD = {[round(x, 6) for x in _codes(w)[2, :6].tolist()]}")