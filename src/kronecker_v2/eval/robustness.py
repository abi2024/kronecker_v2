"""Typo robustness probe, matching the V1 protocol.

Clean/typo prompt pairs; report top-1-preserved rate, KL(clean || typo), and
final-hidden cosine. V1 reported 55.5% top-1-preserved for Kronecker vs 47.3%
for BPE — reproduce that metric set so the numbers sit side by side.
"""

from __future__ import annotations


def make_typo_pairs(prompts: list[str], seed: int = 0) -> list[tuple[str, str]]:
    raise NotImplementedError


def top1_preserved(model, pairs) -> float:
    raise NotImplementedError
