"""Analytic FHRR capacity curve, overlaid on the empirical histogram.

Independent phasor codes overlap with mean 0 and noise std ~ 1/sqrt(2D).
A bundle of M components has self-similarity ~ 1/sqrt(M), so retrieval SNR
~ sqrt(D/M) (Frady, Kleyko & Sommer, arXiv 1707.01429). Capacity grows
linearly in D — that is the theory the measurement should match.
"""

from __future__ import annotations


def noise_std(d_complex: int) -> float:
    raise NotImplementedError


def expected_similarity(shared_bindings: int, total_bindings: int) -> float:
    raise NotImplementedError


def collision_probability(d_complex: int, n_items: int, threshold: float) -> float:
    raise NotImplementedError
