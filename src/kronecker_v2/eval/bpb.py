"""Per-script bits-per-byte.

Per-BYTE, not per-token, so arms stay comparable regardless of how many tokens
a script costs.
"""

from __future__ import annotations


def bits_per_byte(model, dataloader, script: str | None = None) -> float:
    raise NotImplementedError
