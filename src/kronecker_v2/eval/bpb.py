"""Bits per byte — the unit that makes scripts comparable.

Per-token loss is not comparable across writing systems. A Devanagari token
covers roughly three times the UTF-8 bytes of a Latin one, so a model can look
worse per token on Indic text while being no worse per unit of text. Dividing
the total loss by total bytes removes that distortion:

    BPB = (sum of per-token losses in nats) / (total UTF-8 bytes) / ln(2)

The tokenizer is held fixed across arms here, so BPB and mean loss give the same
ORDERING within a script; what BPB adds is the ability to read one script's cost
against another's on the same axis.
"""

from __future__ import annotations

import numpy as np

LN2 = float(np.log(2.0))


def bits_per_byte(losses: np.ndarray, byte_lengths: np.ndarray,
                  mask: np.ndarray | None = None) -> float:
    """losses: per-token nats. byte_lengths: UTF-8 length of each target token."""
    if mask is not None:
        losses, byte_lengths = losses[mask], byte_lengths[mask]
    total_bytes = float(byte_lengths.sum())
    if total_bytes == 0:
        return float("nan")
    return float(losses.sum()) / total_bytes / LN2


def vocab_byte_lengths(vocab_bytes: dict[int, bytes], vocab_size: int) -> np.ndarray:
    return np.array([len(vocab_bytes.get(i, b"")) for i in range(vocab_size)],
                    dtype=np.int64)
