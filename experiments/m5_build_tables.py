"""M5 — tables for the mechanism ablations and the wave efficiency sweep.

  bag2048        wave phasors, position rotation removed (isolates BINDING)
  rp16           one-hot @16 through a fixed dense Gaussian matrix. Invertible,
                 so the information is identical; only the spread changes
                 (isolates EFFECTIVE DIMENSIONALITY)
  wave1024/768   smaller wave codes — M4 found wave gets BETTER as the code
                 narrows, so this maps how far that goes

    python experiments/m5_build_tables.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import torch

from kronecker_v2.codecs.ablations import BagOfBytesCodec, RandomProjectedOneHotCodec
from kronecker_v2.tables import build_wave_table, table_hash
from kronecker_v2.vocab import load_tokenizer, vocab_bytes

VERIFY = 128


def build_bag_table(vb, d_complex, seed=0, normalize="l2", chunk=8192):
    """Vectorised, then spot-checked against the per-token codec."""
    codec = BagOfBytesCodec(vb, d_complex=d_complex, seed=seed, normalize=normalize)
    V = len(vb)
    max_len = max(len(b) for b in vb.values())
    buf = np.zeros((V, max_len), dtype=np.uint8)
    lens = np.zeros(V, dtype=np.int64)
    for t in range(V):
        r = vb[t]; lens[t] = len(r)
        if r:
            buf[t, :len(r)] = np.frombuffer(r, dtype=np.uint8)
    buf_t, lens_t = torch.from_numpy(buf).long(), torch.from_numpy(lens)
    bp = codec._byte_phase
    out = torch.empty((V, 2 * d_complex), dtype=torch.bfloat16)
    ids = set(np.random.default_rng(3).choice(V, VERIFY, replace=False).tolist())
    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        z = torch.zeros((e - s, d_complex), dtype=torch.complex64)
        l = lens_t[s:e]
        for p in range(max_len):
            act = l > p
            if not act.any():
                break
            ph = bp[buf_t[s:e][act, p]]          # no position term: that is the ablation
            z[act] += torch.polar(torch.ones_like(ph), ph)
        if normalize == "l2":
            z = z / (z.abs().pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-6)
        elif normalize == "sqrt_len":
            z = z / l.clamp_min(1).float().sqrt().unsqueeze(1)
        row = torch.cat([z.real, z.imag], dim=-1)
        if normalize == "znorm":
            row = (row - row.mean(1, keepdim=True)) / (row.std(1, keepdim=True) + 1e-6)
        hit = [i for i in ids if s <= i < e]
        if hit:
            ref = torch.stack([codec.encode_bytes(vb[i]) for i in hit])
            assert torch.allclose(row[[i - s for i in hit]], ref, atol=1e-4), \
                "bag table diverged from the frozen per-token codec"
        out[s:e] = row.to(torch.bfloat16)
    return out


def build_rp_table(vb, pos_dim, seed=0, normalize="l2", chunk=4096):
    """onehot_table @ R, one big matmul rather than 131k small ones."""
    codec = RandomProjectedOneHotCodec(vb, pos_dim=pos_dim, seed=seed,
                                       normalize=normalize)
    from kronecker_v2.tables import build_onehot_table
    base = build_onehot_table(vb, pos_dim=pos_dim, out_dtype=torch.float32)
    V, D = base.shape
    out = torch.empty((V, D), dtype=torch.bfloat16)
    ids = set(np.random.default_rng(4).choice(V, VERIFY, replace=False).tolist())
    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        row = base[s:e] @ codec.R
        if normalize == "l2":
            row = row / (row.norm(dim=1, keepdim=True) + 1e-6)
        elif normalize == "znorm":
            row = (row - row.mean(1, keepdim=True)) / (row.std(1, keepdim=True) + 1e-6)
        hit = [i for i in ids if s <= i < e]
        if hit:
            ref = torch.stack([codec.encode_bytes(vb[i]) for i in hit])
            assert torch.allclose(row[[i - s for i in hit]], ref, atol=1e-3), \
                "rp table diverged from the frozen per-token codec"
        out[s:e] = row.to(torch.bfloat16)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default="theschoolofai/BrahmicTokenizer-131K")
    ap.add_argument("--out", type=Path, default=Path("data/tables"))
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    vb = vocab_bytes(tok, source="raw")
    args.out.mkdir(parents=True, exist_ok=True)
    hp = args.out / "hashes.json"
    hashes = json.loads(hp.read_text()) if hp.exists() else {}

    jobs = [
        ("bag2048_l2",   lambda: build_bag_table(vb, 2048)),
        ("rp16_l2",      lambda: build_rp_table(vb, 16)),
        ("wave1024_l2",  lambda: build_wave_table(vb, 1024, 0, "l2")),
        ("wave768_l2",   lambda: build_wave_table(vb, 768, 0, "l2")),
    ]
    for name, fn in jobs:
        path = args.out / f"{name}.pt"
        if path.exists():
            print(f"{name}: exists, skipping")
            continue
        t0 = time.time()
        table = fn()
        torch.save(table, path)
        hashes[name] = table_hash(table)
        print(f"{name}: {tuple(table.shape)} bf16, "
              f"{table.numel()*2/2**30:.2f} GB, {time.time()-t0:.0f}s, "
              f"hash {hashes[name][:16]}…  (spot-checked vs frozen codec)")
        del table
    hp.write_text(json.dumps(hashes, indent=2))
    print(f"\n{len(hashes)} fingerprints recorded in {hp}")


if __name__ == "__main__":
    main()
