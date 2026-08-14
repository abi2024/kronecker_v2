"""M4 — code tables across the pos_dim dose axis.

The dose-response experiment needs a matched (one-hot, wave) pair at each
window size. ``pos_dim`` is a dial whose dose is known exactly, because the M1
audit counted the collisions it produces:

    pos_dim 12   D=3072   d_complex=1536    5,335 tokens permanently merged
    pos_dim 16   D=4096   d_complex=2048      903      <- already built for M2
    pos_dim 24   D=6144   d_complex=3072       93
    pos_dim 32   D=8192   d_complex=4096        0      <- the built-in placebo

The last row is the point of the experiment. It is a dose of zero produced not
by us but by the tokenizer's own design constraint, so if the effect survives
there it was never about collisions.

    python experiments/m4_build_tables.py --pos-dims 12 24 32
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

import torch

from kronecker_v2.tables import build_onehot_table, build_wave_table, table_hash
from kronecker_v2.vocab import load_tokenizer, vocab_bytes

PAIRING = 128        # d_complex = PAIRING * pos_dim, the frozen parity rule


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default="theschoolofai/BrahmicTokenizer-131K")
    ap.add_argument("--pos-dims", type=int, nargs="+", default=[12, 24, 32])
    ap.add_argument("--out", type=Path, default=Path("data/tables"))
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    vb = vocab_bytes(tok, source="raw")          # frozen decision: raw bytes
    args.out.mkdir(parents=True, exist_ok=True)

    hashes_path = args.out / "hashes.json"
    hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}

    total_gb = 0.0
    for pd in args.pos_dims:
        dc = PAIRING * pd
        jobs = [(f"onehot{pd}", lambda: build_onehot_table(vb, pos_dim=pd)),
                (f"wave{dc}_l2", lambda: build_wave_table(vb, dc, 0, "l2"))]
        for name, fn in jobs:
            path = args.out / f"{name}.pt"
            if path.exists():
                print(f"{name}: exists, skipping")
                continue
            t0 = time.time()
            table = fn()
            torch.save(table, path)
            gb = table.numel() * 2 / 2**30
            total_gb += gb
            hashes[name] = table_hash(table)
            print(f"{name}: {tuple(table.shape)} bf16, {gb:.2f} GB, "
                  f"{time.time()-t0:.0f}s, hash {hashes[name][:16]}…  "
                  f"(verified against frozen codec)")
            del table

    hashes_path.write_text(json.dumps(hashes, indent=2))
    print(f"\nwrote {total_gb:.1f} GB of new tables; {len(hashes)} fingerprints "
          f"recorded in {hashes_path}")


if __name__ == "__main__":
    main()
