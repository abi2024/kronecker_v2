"""M2 follow-up — where does each arm's loss live? Three ways of asking.

Scores every finished checkpoint over the full validation set once, caches the
per-token losses, and then slices them by whichever partition you ask for:

  --bucket-by frequency   how often the target token appears in TRAINING.
                          Separates "this token had many gradient updates" from
                          "this token had few".

  --bucket-by length      the target token's UTF-8 byte length, cut at the
                          pos_dim=16 window. Frequency and length are
                          confounded (frequent tokens are short), so this is
                          what tells you which variable the effect tracks.

  --bucket-by collision   whether the TARGET is one of the ~903 tokens that
                          share a truncated byte string with another token at
                          pos_dim=16, and therefore receive an IDENTICAL
                          one-hot code.

  --bucket-by prev-collision
                          the same partition applied to the INPUT token whose
                          prediction we are scoring. This is the measurement
                          that matches the mechanism. A collision cannot hurt a
                          token as a TARGET: the lm_head is untied and dense, so
                          every token keeps a private output row and stays
                          perfectly scoreable. A collision corrupts a token as
                          CONTEXT — two sequences containing कार्य and कार्यक्रम
                          produce identical hidden states, so what degrades is
                          the prediction of whatever comes next. Bucketing by
                          target looks in the wrong place; bucketing by the
                          preceding input looks where the damage is.

Because losses are cached per token, only the first invocation touches the GPU;
later slicings are instant. Paired standard errors are reported, because a
bucket holding a few thousand tokens can produce a difference that is entirely
noise, and the collision bucket is small by construction.

    python experiments/m2_bucket_analysis.py --bucket-by collision
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

POS_DIM = 16          # the paper's 124M setting, and M2's frozen constant


# ----------------------------------------------------------------- scoring --
@torch.no_grad()
def score_arm(run_dir: Path, device: str, val_path: Path, block: int,
              batch: int) -> np.ndarray:
    """Per-token validation loss for one checkpoint, cached to disk."""
    cache = run_dir / "per_token_loss.npy"
    if cache.exists():
        return np.load(cache)

    ckpt = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mcfg = GPTConfig(**cfg["model"])
    wte = T.build_wte(cfg, device)                  # rebuilds the frozen table
    model = GPT(mcfg, wte).to(device)
    # `codes` is a non-persistent buffer — absent from the checkpoint by design
    # and reconstructed above. Everything else must load exactly.
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
    out = []
    for s in range(0, n_seq, batch):
        idx = range(s, min(s + batch, n_seq))
        x = torch.stack([torch.from_numpy(np.asarray(
            val[i * block:(i + 1) * block], dtype=np.int64)) for i in idx]).to(device)
        y = torch.stack([torch.from_numpy(np.asarray(
            val[i * block + 1:(i + 1) * block + 1], dtype=np.int64)) for i in idx]).to(device)
        with ac():
            logits, _ = model(x)
        per_tok = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                  y.reshape(-1), reduction="none")
        out.append(per_tok.float().cpu().numpy())

    del model, wte
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    arr = np.concatenate(out).astype(np.float32)
    np.save(cache, arr)
    return arr


def val_stream(val_path: Path, block: int, cache: Path, offset: int) -> np.ndarray:
    """Token ids aligned with score_arm's loss array. Model-independent.

    offset=1 gives the TARGET at each scored position; offset=0 gives the INPUT
    fed at that position, i.e. the token immediately preceding the target.
    """
    if cache.exists():
        return np.load(cache)
    val = np.memmap(val_path, dtype=np.uint32, mode="r")
    n_seq = (len(val) - 1) // block
    arr = np.concatenate([np.asarray(val[i * block + offset:(i + 1) * block + offset],
                                     dtype=np.int64) for i in range(n_seq)])
    np.save(cache, arr)
    return arr


# --------------------------------------------------------------- bucketing --
def buckets_by_frequency(train_path: Path, vocab_size: int):
    counts = np.zeros(vocab_size, dtype=np.int64)
    arr = np.memmap(train_path, dtype=np.uint32, mode="r")
    for s in range(0, len(arr), 50_000_000):
        counts += np.bincount(np.asarray(arr[s:s + 50_000_000], dtype=np.int64),
                              minlength=vocab_size)
    seen = counts > 0
    order = np.argsort(-counts, kind="stable")
    rank = np.empty(vocab_size, dtype=np.int64)
    rank[order] = np.arange(vocab_size)

    edges = [("top 1K", 0, 1_000), ("1K-10K", 1_000, 10_000),
             ("10K-100K", 10_000, 100_000), ("100K+ tail", 100_000, 10**9)]
    bmap = np.full(vocab_size, len(edges), dtype=np.int64)
    for i, (_, lo, hi) in enumerate(edges):
        bmap[seen & (rank >= lo) & (rank < hi)] = i
    return bmap, [e[0] for e in edges] + ["unseen in train"]


def _vocab_lengths(tokenizer, vocab_size: int):
    from kronecker_v2.vocab import vocab_bytes
    vb = vocab_bytes(tokenizer, vocab_size, source="raw")
    lens = np.array([len(vb[i]) for i in range(vocab_size)], dtype=np.int64)
    return vb, lens


def buckets_by_length(tokenizer, vocab_size: int):
    _, lens = _vocab_lengths(tokenizer, vocab_size)
    edges = [("1-4 B", 1, 5), ("5-8 B", 5, 9), ("9-12 B", 9, 13),
             ("13-16 B", 13, 17), (f"17+ B (cropped @{POS_DIM})", 17, 10**9)]
    bmap = np.full(vocab_size, len(edges), dtype=np.int64)
    for i, (_, lo, hi) in enumerate(edges):
        bmap[(lens >= lo) & (lens < hi)] = i
    return bmap, [e[0] for e in edges] + ["empty"]


def buckets_by_collision(tokenizer, vocab_size: int):
    """Tokens that receive an identical one-hot code, vs those that do not."""
    from kronecker_v2.collisions import exact_collisions
    from kronecker_v2.vocab import truncate
    vb, lens = _vocab_lengths(tokenizer, vocab_size)
    collided = np.zeros(vocab_size, dtype=bool)
    groups = exact_collisions(vb, POS_DIM)
    for ids in groups.values():
        collided[np.asarray(ids, dtype=np.int64)] = True
    truncated = np.array([len(truncate(vb[i], POS_DIM)) < lens[i]
                          for i in range(vocab_size)], dtype=bool)
    names = [f"collides @{POS_DIM}", "cropped, still distinct", "fits in window"]
    # "cropped, still distinct" is the control that makes the first bucket
    # readable: also long, also byte-cropped, but NOT collided. Comparing
    # against "fits in window" instead would confound collision with length.
    bmap = np.full(vocab_size, 2, dtype=np.int64)
    bmap[truncated & ~collided] = 1
    bmap[collided] = 0
    print(f"  collision groups {len(groups):,} | collided tokens "
          f"{int(collided.sum()):,} | cropped {int(truncated.sum()):,}")
    return bmap, names


# ------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket-by",
                    choices=["frequency", "length", "collision", "prev-collision"],
                    default="frequency")
    ap.add_argument("--root", type=Path, default=Path("results/m2"))
    ap.add_argument("--data", type=Path, default=Path("data/m2"))
    ap.add_argument("--tokenizer", default="theschoolofai/BrahmicTokenizer-131K")
    ap.add_argument("--baseline", default="m2_onehot")
    ap.add_argument("--batch", type=int, default=2)   # scoring upcasts logits
                                                      # to fp32; 2 keeps peak
                                                      # VRAM near 6 GB at V=131K
    args = ap.parse_args()

    runs = sorted(d for d in args.root.iterdir()
                  if d.is_dir() and (d / "final.pt").exists())
    if not runs:
        raise SystemExit(f"no checkpoints under {args.root}")
    meta = json.loads((args.data / "meta.json").read_text())
    V = meta["vocab_size"]
    block = json.loads((runs[0] / "manifest.json").read_text())["cfg"]["model"]["block_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    losses = {}
    for d in runs:
        cached = (d / "per_token_loss.npy").exists()
        print(f"{'loading cached' if cached else 'scoring'} {d.name}…", flush=True)
        losses[d.name] = score_arm(d, device, args.data / "val.bin", block, args.batch)
    # Which token decides a position's bucket: the target being predicted, or
    # the input token fed at that position (the mechanism for context damage).
    by_input = args.bucket_by.startswith("prev-")
    ids = val_stream(args.data / "val.bin", block,
                     args.root / ("val_inputs.npy" if by_input else "val_targets.npy"),
                     offset=0 if by_input else 1)
    n = min(len(ids), *(len(v) for v in losses.values()))
    ids = ids[:n]
    losses = {k: v[:n] for k, v in losses.items()}

    print(f"building buckets by {args.bucket_by}"
          f" ({'input token' if by_input else 'target token'})…")
    if args.bucket_by == "frequency":
        bmap, names = buckets_by_frequency(args.data / "train.bin", V)
    else:
        from kronecker_v2.vocab import load_tokenizer
        tok = load_tokenizer(args.tokenizer)
        bmap, names = (buckets_by_length(tok, V) if args.bucket_by == "length"
                       else buckets_by_collision(tok, V))
    if by_input:
        names = [f"after {x}" for x in names]
    tok_bucket = bmap[ids]

    base = args.baseline if args.baseline in losses else sorted(losses)[0]
    arms = sorted(losses)
    rows, drows = [], []
    for i, name in enumerate(names):
        m = tok_bucket == i
        cnt = int(m.sum())
        if cnt == 0:
            continue
        row = {"bucket": name, "vocab_tokens": int((bmap == i).sum()),
               "val_tokens": cnt, "val_share_%": round(100 * cnt / n, 2)}
        for a in arms:
            row[a] = round(float(losses[a][m].mean()), 4)
        rows.append(row)

        d = {"bucket": name, "val_tokens": cnt}
        for a in arms:
            if a == base:
                continue
            diff = losses[a][m] - losses[base][m]      # paired: same tokens
            se = float(diff.std(ddof=1) / np.sqrt(cnt)) if cnt > 1 else float("nan")
            gap = float(diff.mean())
            d[a] = round(gap, 4)
            d[a + "_SE"] = round(se, 4)
            d[a + "_sig"] = "yes" if abs(gap) > 2 * se else "no"
        drows.append(d)

    df, ddf = pd.DataFrame(rows), pd.DataFrame(drows)
    stem = f"buckets_{args.bucket_by}"
    args.root.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.root / f"{stem}.csv", index=False)
    ddf.to_csv(args.root / f"{stem}_gaps.csv", index=False)

    print(f"\n=== mean val loss by {args.bucket_by} ===")
    print(df.to_string(index=False))
    print(f"\n=== gap vs {base}  (negative = better than {base}; "
          f"sig = |gap| > 2 paired SE) ===")
    print(ddf.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colour = {a: palette[i % len(palette)] for i, a in enumerate(arms)}
    labels = [f"{r['bucket']}\n({r['val_share_%']:.1f}% of val)" for r in rows]
    x = np.arange(len(rows))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.5))
    w = 0.8 / len(arms)
    for j, a in enumerate(arms):
        axL.bar(x + j * w - 0.4 + w / 2, [r[a] for r in rows], w,
                color=colour[a], label=a)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=8)
    axL.set_ylabel("mean val loss"); axL.legend(fontsize=8); axL.grid(alpha=.3, axis="y")
    axL.set_title("Loss by "
                  + ("preceding-token collision status" if by_input
                     else f"target-token {args.bucket_by}"))

    others = [a for a in arms if a != base]
    w2 = 0.8 / max(len(others), 1)
    for j, a in enumerate(others):
        vals = [d[a] for d in drows]
        errs = [2 * d[a + "_SE"] for d in drows]
        axR.bar(x + j * w2 - 0.4 + w2 / 2, vals, w2, yerr=errs, capsize=2,
                color=colour[a], label=a)
    axR.axhline(0, color=colour[base], lw=1.5, ls="--")
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=8)
    axR.set_ylabel(f"loss minus {base}  (negative = better)")
    axR.legend(fontsize=8); axR.grid(alpha=.3, axis="y")
    axR.set_title(f"Gap vs {base}  (bars = ±2 paired SE)")
    fig.tight_layout()
    fig.savefig(args.root / f"{stem}.png", dpi=150)
    print(f"\nwrote {args.root}/{stem}.csv, {stem}_gaps.csv, {stem}.png")


if __name__ == "__main__":
    main()