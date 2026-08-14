"""M3 — per-script loss and bits per byte.

The first DIRECT test of the Indic claim. Everything so far has argued from the
collision audit (a property of the vocabulary) or from buckets defined by
collision status. This measures the thing itself: how much does each writing
system cost each arm?

Reported in **bits per byte**, because per-token loss is not comparable across
scripts. A Devanagari token covers roughly three times the UTF-8 bytes of a
Latin one, so an arm can look worse per token on Indic text while being no worse
per unit of text:

    BPB = (sum of per-token losses in nats) / (total UTF-8 bytes) / ln(2)

Reuses M2's scoring and caching without modifying it — ``m2_bucket_analysis.py``
is frozen under the M2 declaration, so this imports from it rather than editing
it. If the per-token caches already exist, this runs in seconds with no GPU.

    python experiments/m3_script_analysis.py
"""

from __future__ import annotations

import argparse
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

from experiments import m2_bucket_analysis as BA      # frozen: imported, not edited
from kronecker_v2.eval.bpb import bits_per_byte, vocab_byte_lengths
from kronecker_v2.vocab import load_tokenizer, script_of, vocab_bytes

SCRIPTS = ["LATIN", "DEVANAGARI", "BENGALI", "TELUGU", "TAMIL", "MALAYALAM",
           "KANNADA", "GUJARATI", "ORIYA", "GURMUKHI"]
OTHER = "other / non-alpha"


def script_buckets(vb: dict[int, bytes], vocab_size: int):
    order = {s: i for i, s in enumerate(SCRIPTS)}
    bmap = np.full(vocab_size, len(SCRIPTS), dtype=np.int64)
    for t in range(vocab_size):
        bmap[t] = order.get(script_of(vb[t]), len(SCRIPTS))
    return bmap, SCRIPTS + [OTHER]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/m2"))
    ap.add_argument("--data", type=Path, default=Path("data/m2"))
    ap.add_argument("--tokenizer", default="theschoolofai/BrahmicTokenizer-131K")
    ap.add_argument("--baseline", default="m2_onehot")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--min-share", type=float, default=0.05,
                    help="drop buckets below this %% of validation tokens")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.root

    runs = sorted(d for d in args.root.iterdir()
                  if d.is_dir() and (d / "final.pt").exists())
    if not runs:
        raise SystemExit(f"no checkpoints under {args.root}")
    import json
    meta = json.loads((args.data / "meta.json").read_text())
    V = meta["vocab_size"]
    block = json.loads((runs[0] / "manifest.json").read_text())["cfg"]["model"]["block_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    losses = {}
    for d in runs:
        cached = (d / "per_token_loss.npy").exists()
        print(f"{'loading cached' if cached else 'scoring'} {d.name}…", flush=True)
        losses[d.name] = BA.score_arm(d, device, args.data / "val.bin",
                                      block, args.batch)
    ids = BA.val_stream(args.data / "val.bin", block,
                        args.root / "val_targets.npy", offset=1)
    n = min(len(ids), *(len(v) for v in losses.values()))
    ids, losses = ids[:n], {k: v[:n] for k, v in losses.items()}

    print("tagging vocabulary by Unicode script…")
    tok = load_tokenizer(args.tokenizer)
    vb = vocab_bytes(tok, V, source="raw")
    bmap, names = script_buckets(vb, V)
    blen = vocab_byte_lengths(vb, V)[ids]        # UTF-8 bytes per scored target
    bucket = bmap[ids]

    arms = sorted(losses)
    base = args.baseline if args.baseline in losses else arms[0]

    rows, drows = [], []
    for i, name in enumerate(names):
        m = bucket == i
        cnt = int(m.sum())
        if 100 * cnt / n < args.min_share:
            continue
        row = {"script": name, "vocab_tokens": int((bmap == i).sum()),
               "val_tokens": cnt, "val_share_%": round(100 * cnt / n, 2),
               "val_bytes": int(blen[m].sum()),
               "bytes_per_token": round(float(blen[m].mean()), 2)}
        for a in arms:
            row[a + "_bpb"] = round(bits_per_byte(losses[a], blen, m), 4)
        rows.append(row)

        d = {"script": name, "val_tokens": cnt}
        for a in arms:
            if a == base:
                continue
            diff = losses[a][m] - losses[base][m]          # paired: same tokens
            gap_bpb = (bits_per_byte(losses[a], blen, m)
                       - bits_per_byte(losses[base], blen, m))
            se_nats = float(diff.std(ddof=1) / np.sqrt(cnt)) if cnt > 1 else float("nan")
            d[a + "_bpb"] = round(gap_bpb, 4)
            d[a + "_nats"] = round(float(diff.mean()), 4)
            d[a + "_SE"] = round(se_nats, 4)
            d[a + "_sig"] = "yes" if abs(float(diff.mean())) > 2 * se_nats else "no"
        drows.append(d)

    df, ddf = pd.DataFrame(rows), pd.DataFrame(drows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "script_bpb.csv", index=False)
    ddf.to_csv(out / "script_bpb_gaps.csv", index=False)
    print("\n=== bits per byte, by script ===")
    print(df.to_string(index=False))
    print(f"\n=== gap vs {base}  (negative = better; sig on paired nats) ===")
    print(ddf.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colour = {a: palette[i % len(palette)] for i, a in enumerate(arms)}
    labels = [f"{r['script'].title()}\n({r['val_share_%']:.0f}% of val, "
              f"{r['bytes_per_token']:.1f} B/tok)" for r in rows]
    x = np.arange(len(rows))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
    w = 0.8 / len(arms)
    for j, a in enumerate(arms):
        axL.bar(x + j * w - 0.4 + w / 2, [r[a + "_bpb"] for r in rows], w,
                color=colour[a], label=a)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=8)
    axL.set_ylabel("bits per byte"); axL.legend(fontsize=8); axL.grid(alpha=.3, axis="y")
    axL.set_title("Cost per byte of text, by writing system", fontsize=10, weight="bold")

    others = [a for a in arms if a != base]
    w2 = 0.8 / max(len(others), 1)
    for j, a in enumerate(others):
        axR.bar(x + j * w2 - 0.4 + w2 / 2, [d[a + "_bpb"] for d in drows], w2,
                color=colour[a], label=a)
    axR.axhline(0, color=colour[base], lw=1.5, ls="--")
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=8)
    axR.set_ylabel(f"bits per byte minus {base}  (negative = better)")
    axR.legend(fontsize=8); axR.grid(alpha=.3, axis="y")
    axR.set_title(f"Gap vs {base}, per script", fontsize=10, weight="bold")
    fig.tight_layout(); fig.savefig(out / "script_bpb.png", dpi=150)
    print(f"\nwrote {out}/script_bpb.csv, script_bpb_gaps.csv, script_bpb.png")


if __name__ == "__main__":
    main()
