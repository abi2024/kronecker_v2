"""Comparison arms for the matched-parameter experiments.

All must produce the same trainable parameter count as the Kronecker arms at a
given d_model — enforced by tests/test_equal_params.py.

TODO: implement the Codec protocol defined in .base
"""

from __future__ import annotations


class DenseTableCodec:
    """Plain V x d_model learned embedding table."""
    name = "dense"


class HashEmbeddingCodec:
    """Hash embeddings (arXiv 1709.03933)."""
    name = "hash"


class ALBERTFactorizedCodec:
    """V x r then r x d_model factorization (Lan et al., 2020)."""
    name = "albert"
