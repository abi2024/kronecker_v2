"""Vectorized full-vocab code tables, verified against the frozen codecs.

The frozen per-token paths in ``codecs/`` define the ground truth but loop in
Python; building 131,072 rows that way is minutes. These builders produce the
same values in seconds and verify themselves against the frozen path on a
random sample of rows, chunk by chunk, BEFORE any precision cast — so speed
and storage dtype are never allowed to drift from the frozen definition.

Memory note: tables are built chunk-wise straight into the output dtype
(default bf16, ~1.07 GB per table at 131,072 x 4,096); the fp32 intermediate
only ever exists for one chunk (~134 MB).
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from kronecker_embeddings import kronecker_codec, utf8_safe_truncate

from .codecs.base import OneHotCodec
from .codecs.wave import WaveKroneckerCodec

VERIFY_SAMPLE = 256


def _byte_matrix(vocab_bytes: dict[int, bytes], max_len: int):
    V = len(vocab_bytes)
    buf = np.zeros((V, max_len), dtype=np.uint8)
    lens = np.zeros((V,), dtype=np.int64)
    for tid in range(V):
        raw = vocab_bytes[tid]
        if len(raw) > max_len:
            raw = utf8_safe_truncate(raw, max_len)
        L = len(raw)
        if L:
            buf[tid, :L] = np.frombuffer(raw, dtype=np.uint8, count=L)
        lens[tid] = L
    return torch.from_numpy(buf), torch.from_numpy(lens)


def _verify_ids(V: int, seed: int) -> set[int]:
    return set(np.random.default_rng(seed).choice(
        V, min(VERIFY_SAMPLE, V), replace=False).tolist())


def _check_chunk(chunk_fp32, s, ids, frozen_fn, vocab_bytes, exact, label):
    hit = [i for i in ids if s <= i < s + chunk_fp32.size(0)]
    if not hit:
        return
    ref = torch.stack([frozen_fn(vocab_bytes[i]) for i in hit])
    got = chunk_fp32[[i - s for i in hit]]
    ok = torch.equal(got, ref) if exact else torch.allclose(got, ref, atol=1e-4)
    assert ok, f"{label} table diverged from frozen codec at rows {hit[:3]}…"


def build_onehot_table(vocab_bytes: dict[int, bytes], pos_dim: int = 16,
                       chunk: int = 8192, verify: bool = True,
                       out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    codec = OneHotCodec(vocab_bytes=vocab_bytes, pos_dim=pos_dim)
    buf, lens = _byte_matrix(vocab_bytes, pos_dim)
    V = buf.size(0)
    ids = _verify_ids(V, 0) if verify else set()
    out = torch.empty((V, codec.code_dim), dtype=out_dtype)
    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        block = kronecker_codec(buf[s:e], lens[s:e], char_dim=256, pos_dim=pos_dim)
        if verify:
            _check_chunk(block, s, ids, codec.encode_bytes, vocab_bytes,
                         exact=True, label="one-hot")
        out[s:e] = block.to(out_dtype)
    return out


def build_wave_table(vocab_bytes: dict[int, bytes], d_complex: int = 2048,
                     seed: int = 0, normalize: str = "sqrt_len",
                     chunk: int = 8192, verify: bool = True,
                     out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    codec = WaveKroneckerCodec(vocab_bytes=vocab_bytes, d_complex=d_complex,
                               seed=seed, normalize=normalize)
    max_len = max((len(b) for b in vocab_bytes.values()), default=1)
    buf, lens = _byte_matrix(vocab_bytes, max_len)
    V = buf.size(0)
    ids = _verify_ids(V, 1) if verify else set()
    bp, pp = codec._byte_phase, codec._pos_phase
    out = torch.empty((V, codec.code_dim), dtype=out_dtype)

    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        b, l = buf[s:e].long(), lens[s:e]
        z = torch.zeros((e - s, d_complex), dtype=torch.complex64)
        for p in range(max_len):           # mirrors the frozen sum over p=1..L
            active = l > p
            if not active.any():
                break
            phase = bp[b[active, p]] + float(p + 1) * pp.unsqueeze(0)
            z[active] += torch.polar(torch.ones_like(phase), phase)
        if normalize == "sqrt_len":        # mirrors WaveKroneckerCodec._post
            z = z / l.clamp_min(1).float().sqrt().unsqueeze(1)
        elif normalize == "l2":
            z = z / (z.abs().pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-6)
        elif normalize != "znorm":
            raise ValueError(f"unknown normalize={normalize!r}")
        row = torch.cat([z.real, z.imag], dim=-1)
        if normalize == "znorm":
            mean = row.mean(dim=1, keepdim=True)
            std = row.std(dim=1, keepdim=True) + 1e-6
            row = (row - mean) / std
        if verify:
            _check_chunk(row, s, ids, codec.encode_bytes, vocab_bytes,
                         exact=False, label=f"wave/{normalize}")
        out[s:e] = row.to(out_dtype)
    return out


def table_hash(table: torch.Tensor) -> str:
    return hashlib.sha256(
        table.contiguous().view(torch.uint8).numpy().tobytes()
        if table.dtype == torch.bfloat16
        else table.contiguous().numpy().tobytes()
    ).hexdigest()
