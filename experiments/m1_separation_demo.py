"""Milestone 1 companion — does the wave codec separate what one-hot merges?

At pos_dim=16 the one-hot codec emits byte-identical vectors for the Devanagari
groups found by the audit. The wave codec has no position ceiling, so it sees
every byte and separates them — while keeping them *near* each other, which is
the orthographic locality the method is supposed to provide.

Run after m1_collision_audit.py. No training, seconds.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from kronecker_embeddings import codec_output_dim, encode_single

from kronecker_v2.codecs.wave import WaveKroneckerCodec

GROUPS = [
    [" कार्य", " कार्यक्रम", " कार्यालय"],
    [" सरकार", " सरकारी", " सरकारले"],
    [" अधिकार", " अधिकारी", " अधिकारियों"],
]
POS_DIM = 16


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main() -> None:
    wave = WaveKroneckerCodec(vocab_bytes={}, d_complex=2048, seed=0)
    assert wave.code_dim == codec_output_dim(pos_dim=POS_DIM), "equal-dim contract broken"
    print(f"code_dim: one-hot@pos_dim={POS_DIM} = {codec_output_dim(pos_dim=POS_DIM)}, "
          f"wave = {wave.code_dim}  (equal by construction)\n")
    print(f"{'pair':<34}{'one-hot':>12}{'wave':>9}")
    for group in GROUPS:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i].encode(), group[j].encode()
                oa, ob = encode_single(a, pos_dim=POS_DIM), encode_single(b, pos_dim=POS_DIM)
                verdict = "IDENTICAL" if torch.equal(oa, ob) else f"{cos(oa, ob):.4f}"
                pair = f"{group[i].strip()} / {group[j].strip()}"
                print(f"{pair:<34}{verdict:>12}{cos(wave.encode_bytes(a), wave.encode_bytes(b)):>9.4f}")
    print("\nIDENTICAL means the two tokens are indistinguishable to the model, permanently.")


if __name__ == "__main__":
    main()
