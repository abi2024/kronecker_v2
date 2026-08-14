"""M2 follow-up — where does each arm's loss actually live?

Aggregate validation loss says which arm wins. It does not say WHY. This
re-scores the finished checkpoints and splits the loss by how frequent each
target token is in the training set, which separates two mechanisms that the
aggregate number blends together:

  * A dense table gives every token private parameters. A rare token's row only
    moves on the steps where that token appears, so rare rows stay near their
    initialisation.
  * A codec has no per-token parameters. Every token's representation improves
    on every step, because the shared projection is what learns — but no token
    can be adjusted without moving its byte-neighbours too.

The first mechanism should favour the codec in the tail; the second should
favour dense on frequent tokens. If dense's aggregate win is concentrated in
frequent tokens and shrinks or inverts in the tail, that is a mechanism rather
than a scoreboard.

No training. Re-scores existing final.pt checkpoints; a few minutes for the grid.

    python experiments/m2_frequency_analysis.py
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
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from kronecker_v2.model import GPT, GPTConfig
from experiments import m2_tiny_train as T

# Rank ranges over tokens that actually occur in training, most frequent first.
BUCKETS = [
    ("top 1K",      0,      1_000),
    ("1K-10K",      1_000,  10_000),
    ("10K-100K",    10_000, 100_000),
    ("100K+ tail",  100_000, 10**9),
]
UNSEEN = "unseen in train"


def bucket_of_rank(rank: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Vocab-sized array of bucket indices; last index is the unseen bucket."""
    out = np.full(rank.shape, len(BUCKETS), dtype=np.int64)   # default: unseen
    for i, (_, lo, hi) in enumerate(BUCKETS):
        out[seen & (rank >= lo) & (rank < hi)] = i
    return out


def token_buckets(train_path: Path, vocab_size: int):
    counts = np.zeros(vocab_size, dtype=np.int64)
    arr = np.memmap(train_path, dtype=np.uint32, mode="r")
    for s in range(0, len(arr), 50_000_000):                  # chunked bincount
        chunk = np.asarray(arr[s:s + 50_000_000], dtype=np.int64)
        counts += np.bincount(chunk, minlength=vocab_size)
    seen = counts > 0
    order = np.argsort(-counts, kind="stable")                # most frequent first
    rank = np.empty(vocab_size, dtype=np.int64)
    rank[order] = np.arange(vocab_size)
    return bucket_of_rank(rank, seen), counts, int(seen.sum())


@torch.no_grad()
def score_arm(run_dir: Path, bucket_map: torch.Tensor, n_buckets: int,
              device: str, val_path: Path, block: int, batch: int):
    """Sum of per-token losses and token count, per bucket."""
    ckpt = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mcfg = GPTConfig(**cfg["model"])
    wte = T.build_wte(cfg, device)                 # rebuilds the frozen table
    model = GPT(mcfg, wte).to(device)
    # codes is a non-persistent buffer: absent from the checkpoint by design,
    # reconstructed above from the codec. Everything else must match exactly.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not unexpected, f"unexpected keys in {run_dir.name}: {unexpected[:3]}"
    assert all(k.endswith("codes") for k in missing), \
        f"missing non-buffer keys in {run_dir.name}: {missing[:3]}"
    model.eval()

    use_bf16 = device.startswith("cuda")
    ac = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16
          else (lambda: torch.autocast("cpu", enabled=False)))

    val = np.memmap(val_path, dtype=np.uint32, mode="r")
    n_seq = (len(val) - 1) // block
    loss_sum = torch.zeros(n_buckets + 1, dtype=torch.float64, device=device)
    tok_cnt = torch.zeros(n_buckets + 1, dtype=torch.float64, device=device)

    for s in range(0, n_seq, batch):
        idx = range(s, min(s + batch, n_seq))
        x = torch.stack([torch.from_numpy(
            np.asarray(val[i * block:(i + 1) * block], dtype=np.int64)) for i in idx]).to(device)
        y = torch.stack([torch.from_numpy(
            np.asarray(val[i * block + 1:(i + 1) * block + 1], dtype=np.int64)) for i in idx]).to(device)
        with ac():
            logits, _ = model(x)
        per_tok = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                  y.reshape(-1), reduction="none")
        b = bucket_map[y.reshape(-1)]
        loss_sum.index_add_(0, b, per_tok.double())
        tok_cnt.index_add_(0, b, torch.ones_like(per_tok, dtype=torch.float64))

    del model, wte
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return loss_sum.cpu().numpy(), tok_cnt.cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/m2"))
    ap.add_argument("--data", type=Path, default=Path("data/m2"))
    ap.add_argument("--baseline", default="m2_dense")
    ap.add_argument("--batch", type=int, default=2)   # scoring upcasts logits to
                                                      # fp32; 2 keeps peak VRAM
                                                      # near 6 GB at V=131K
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.root

    runs = sorted(d for d in args.root.iterdir()
                  if d.is_dir() and (d / "final.pt").exists())
    if not runs:
        raise SystemExit(f"no checkpoints under {args.root}")
    meta = json.loads((args.data / "meta.json").read_text())
    vocab_size = meta["vocab_size"]
    block = json.loads((runs[0] / "manifest.json").read_text())["cfg"]["model"]["block_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("counting training-set token frequencies…")
    buckets, counts, n_seen = token_buckets(args.data / "train.bin", vocab_size)
    names = [b[0] for b in BUCKETS] + [UNSEEN]
    bmap = torch.from_numpy(buckets).to(device)
    print(f"vocab {vocab_size:,} | seen in train {n_seen:,} "
          f"| unseen {vocab_size - n_seen:,}")

    sums, cnts = {}, None
    for d in runs:
        print(f"scoring {d.name}…", flush=True)
        ls, tc = score_arm(d, bmap, len(BUCKETS), device,
                           args.data / "val.bin", block, args.batch)
        sums[d.name] = ls
        cnts = tc                       # identical across arms: same val stream

    rows = []
    for i, name in enumerate(names):
        if cnts[i] == 0:
            continue
        row = {"bucket": name, "val_tokens": int(cnts[i]),
               "val_share_%": round(100 * cnts[i] / cnts.sum(), 2)}
        for arm in sums:
            row[arm] = round(sums[arm][i] / cnts[i], 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "frequency_buckets.csv", index=False)
    print("\n=== mean validation loss by target-token frequency ===")
    print(df.to_string(index=False))

    base = args.baseline
    if base in sums:
        print(f"\n=== gap vs {base}  (positive = worse than {base}) ===")
        drows = []
        for i, name in enumerate(names):
            if cnts[i] == 0:
                continue
            r = {"bucket": name, "val_share_%": round(100 * cnts[i] / cnts.sum(), 2)}
            for arm in sums:
                if arm == base:
                    continue
                gap = sums[arm][i] / cnts[i] - sums[base][i] / cnts[i]
                r[arm] = round(gap, 4)
                # how much of the arm's TOTAL gap this bucket accounts for
                r[arm + "_share_of_gap_%"] = round(
                    100 * (sums[arm][i] - sums[base][i])
                    / (sums[arm].sum() - sums[base].sum()), 1)
            drows.append(r)
        ddf = pd.DataFrame(drows)
        ddf.to_csv(out / "frequency_gaps.csv", index=False)
        print(ddf.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colour = {a: palette[i % len(palette)] for i, a in enumerate(sorted(sums))}
    keep = [i for i in range(len(names)) if cnts[i] > 0]
    labels = [f"{names[i]}\n({100*cnts[i]/cnts.sum():.0f}% of val)" for i in keep]
    x = np.arange(len(keep))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
    arms = sorted(sums)
    w = 0.8 / len(arms)
    for j, arm in enumerate(arms):
        axL.bar(x + j * w - 0.4 + w / 2, [sums[arm][i] / cnts[i] for i in keep],
                w, color=colour[arm], label=arm)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=8)
    axL.set_ylabel("mean val loss"); axL.legend(fontsize=8); axL.grid(alpha=.3, axis="y")
    axL.set_title("Loss by target-token training frequency")

    if base in sums:
        others = [a for a in arms if a != base]
        w2 = 0.8 / max(len(others), 1)
        for j, arm in enumerate(others):
            axR.bar(x + j * w2 - 0.4 + w2 / 2,
                    [sums[arm][i] / cnts[i] - sums[base][i] / cnts[i] for i in keep],
                    w2, color=colour[arm], label=arm)
        axR.axhline(0, color=colour[base], lw=1.5, ls="--")
        axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=8)
        axR.set_ylabel(f"loss minus {base}  (negative = better)")
        axR.legend(fontsize=8); axR.grid(alpha=.3, axis="y")
        axR.set_title(f"Where {base}'s advantage lives")
    fig.tight_layout()
    fig.savefig(out / "frequency_buckets.png", dpi=150)
    print(f"\nwrote {out}/frequency_buckets.csv, frequency_gaps.csv, "
          f"frequency_buckets.png")


if __name__ == "__main__":
    main()
