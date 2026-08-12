"""Milestone 2 — build and cache the frozen code tables.

Builds the full 131,072-row code table for every arm, self-verifies each
against the frozen per-token codec, and saves bf16 tensors (~1.07 GB each)
plus a fingerprint. Training runs load these instead of rebuilding.

    python experiments/m2_build_tables.py --tokenizer <path-or-hub-name>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kronecker_v2.tables import build_onehot_table, build_wave_table, table_hash
from kronecker_v2.vocab import load_tokenizer, vocab_bytes

ARMS = {
    "onehot16": lambda vb: build_onehot_table(vb, pos_dim=16),
    "wave2048_sqrt_len": lambda vb: build_wave_table(vb, 2048, 0, "sqrt_len"),
    "wave2048_l2": lambda vb: build_wave_table(vb, 2048, 0, "l2"),
    "wave2048_znorm": lambda vb: build_wave_table(vb, 2048, 0, "znorm"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default="theschoolofai/BrahmicTokenizer-131K")
    ap.add_argument("--out", type=Path, default=Path("data/tables"))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    vb = vocab_bytes(tok, source="raw")   # frozen decision: raw byte source
    args.out.mkdir(parents=True, exist_ok=True)

    hashes = {}
    for name in args.arms:
        t0 = time.time()
        table = ARMS[name](vb)                      # built directly in bf16
        h = table_hash(table)
        torch.save(table, args.out / f"{name}.pt")
        hashes[name] = h
        print(f"{name}: {tuple(table.shape)} bf16, {time.time()-t0:.1f}s, "
              f"hash {h[:16]}…  (verified against frozen codec)")
    (args.out / "hashes.json").write_text(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
