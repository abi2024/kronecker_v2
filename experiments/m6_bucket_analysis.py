"""M6 — collision-context analysis and bits-per-byte for an EXTERNAL tokenizer.

Two jobs the frozen analyses cannot do for non-Brahmic vocabularies:

1. The input-side collision DiD (M2 Finding 7 / M4's corrected design), with
   buckets built from the GENERIC byte extractor instead of the Brahmic vocab
   path. Treatment = tokens merged by the window at pos_dim=16; control =
   cropped at 16 but distinct at pos_dim=12 (distinctness is monotone, so the
   control is provably untreated — the M4 lesson, applied from the start).

2. Aggregate bits per byte. Per-token loss is NOT comparable across
   tokenizers — Qwen's 2.75 vs Gemma's 4.40 says nothing about quality, only
   about how finely each splits text. Dividing by UTF-8 bytes puts every
   tokenizer on one axis.

One subtlety the audit forced: SentencePiece vocabularies contain DUPLICATE
byte strings (byte-fallback <0xNN> aliasing single-char pieces). Every byte
codec — one-hot AND wave — must merge those, so they carry no wave-vs-onehot
signal and are excluded from both treatment and control; their share is
reported separately.

    python experiments/m6_bucket_analysis.py --slug gemma-2-9b \
        --tokenizer google/gemma-2-9b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from experiments import m2_bucket_analysis as BA        # frozen: imported only
from experiments.m6_tokenizer_audit import extract_vocab_bytes
from kronecker_v2.collisions import exact_collisions
from kronecker_v2.eval.bpb import bits_per_byte, vocab_byte_lengths
from kronecker_v2.vocab import truncate

REF, STRICT = 16, 12          # treatment window / control-distinctness window


def masks(vb: dict[int, bytes], V: int):
    def merged(pd_):
        m = np.zeros(V, dtype=bool)
        for ids in exact_collisions(vb, pd_).values():
            m[np.asarray(ids, dtype=np.int64)] = True
        return m

    max_len = max((len(b) for b in vb.values()), default=0)
    dup = merged(max_len)                      # no truncation possible => exact
    coll = merged(REF)
    cropped = np.array([len(truncate(vb[i], REF)) < len(vb[i])
                        for i in range(V)], dtype=bool)
    treat = coll & ~dup
    ctrl = cropped & ~merged(STRICT) & ~dup
    return treat, ctrl, dup


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    root = args.root or (_ROOT / "results" / "m6")
    data = args.data or (_ROOT / "data" / "m6" / args.slug)

    runs = sorted(d for d in root.iterdir() if d.is_dir()
                  and d.name.startswith(f"m6_{args.slug}_")
                  and (d / "final.pt").exists())
    if not runs:
        raise SystemExit(f"no m6_{args.slug}_* checkpoints under {root}")
    meta = json.loads((data / "meta.json").read_text())
    V = meta["vocab_size"]
    block = json.loads((runs[0] / "manifest.json").read_text())["cfg"]["model"]["block_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    losses = {}
    for d in runs:
        cached = (d / "per_token_loss.npy").exists()
        print(f"{'cached' if cached else 'scoring'} {d.name}…", flush=True)
        losses[d.name] = BA.score_arm(d, device, data / "val.bin", block, args.batch)

    ids = BA.val_stream(data / "val.bin", block, root / f"val_inputs_{args.slug}.npy",
                        offset=0)
    tgt = BA.val_stream(data / "val.bin", block, root / f"val_targets_{args.slug}.npy",
                        offset=1)
    n = min(len(ids), len(tgt), *(len(v) for v in losses.values()))
    ids, tgt = ids[:n], tgt[:n]
    losses = {k: v[:n] for k, v in losses.items()}

    print(f"extracting bytes for {args.tokenizer}…")
    from transformers import AutoTokenizer
    vb, family, _ = extract_vocab_bytes(AutoTokenizer.from_pretrained(args.tokenizer))
    for i in range(V):
        vb.setdefault(i, b"")
    treat, ctrl, dup = masks(vb, V)
    blen = vocab_byte_lengths(vb, V)[tgt]      # bytes of each SCORED target
    m_t, m_c, m_d = treat[ids], ctrl[ids], dup[ids]
    print(f"  treatment {int(treat.sum()):,} tokens ({int(m_t.sum()):,} positions, "
          f"{100*m_t.mean():.2f}% of stream)  |  control {int(ctrl.sum()):,} tokens "
          f"({int(m_c.sum()):,} positions)  |  duplicate-alias {int(dup.sum()):,} tokens "
          f"({100*m_d.mean():.2f}% of stream, excluded)")

    arms = sorted(losses)
    base = next((a for a in arms if a.endswith("_onehot")), arms[0])
    rows = []
    for a in arms:
        row = {"arm": a,
               "val_loss": round(float(losses[a].mean()), 4),
               "bits_per_byte": round(bits_per_byte(losses[a], blen), 4)}
        if a != base and m_t.sum() >= 50 and m_c.sum() >= 50:
            d_t = losses[a][m_t] - losses[base][m_t]
            d_c = losses[a][m_c] - losses[base][m_c]
            did = float(d_t.mean() - d_c.mean())
            se = float(np.hypot(d_t.std(ddof=1) / np.sqrt(max(m_t.sum(), 2)),
                                d_c.std(ddof=1) / np.sqrt(max(m_c.sum(), 2))))
            row.update(after_treat=round(float(d_t.mean()), 4),
                       after_ctrl=round(float(d_c.mean()), 4),
                       DiD=round(did, 4), SE=round(se, 4),
                       sigma=round(abs(did) / se, 1) if se > 1e-12 else 0.0)
        rows.append(row)

    if m_t.sum() < 50 or m_c.sum() < 50:
        print(f"NOTE: too few bucket positions for a DiD "
              f"(treatment {int(m_t.sum())}, control {int(m_c.sum())}) — "
              f"aggregate and BPB are still valid")
    df = pd.DataFrame(rows)
    df.to_csv(root / f"bucket_{args.slug}.csv", index=False)
    print(f"\n=== {args.slug} ({family}) — aggregate, BPB, and input-side collision "
          f"DiD vs {base} ===")
    print(df.to_string(index=False))
    print(f"\nwrote {root}/bucket_{args.slug}.csv")
    print("note: bits_per_byte is the ONLY column comparable across tokenizers.")


if __name__ == "__main__":
    main()
