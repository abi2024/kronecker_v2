"""M2 — regenerate every figure in the findings sheet from the CSVs on disk.

Four figures, in the order the argument runs:

  fig1_grid          what happened: five arms, one variable changed
  fig2_decomposition where the advantage lives — frequency vs byte length
  fig3_which_side    the null result and the real one, side by side
  fig4_attribution   difference-in-differences: the collision, isolated

No hand-editing. If a number changes, re-run this and the paper's figures
change with it.

    python experiments/m2_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "m2_onehot"          # the matched-parameter baseline everything is read against
HERO = "m2_wave_l2"         # the arm M2 selected
COLOUR = {"m2_dense": "#7f7f7f", "m2_onehot": "#d62728", "m2_wave_l2": "#1f77b4",
          "m2_wave_sqrtlen": "#9ecae1", "m2_wave_znorm": "#c6dbef"}
LABEL = {"m2_dense": "dense (32x params)", "m2_onehot": "one-hot @16 (baseline)",
         "m2_wave_l2": "wave / l2", "m2_wave_sqrtlen": "wave / sqrt_len",
         "m2_wave_znorm": "wave / znorm"}


def arms_in(gaps: pd.DataFrame) -> list[str]:
    return [c for c in gaps.columns
            if c in COLOUR and not c.endswith(("_SE", "_sig"))]


def bars(ax, df, gaps, title, ylabel, hero=HERO):
    """Grouped bars of gap-vs-baseline with +/-2 SE whiskers."""
    arms = arms_in(gaps)
    x = np.arange(len(gaps))
    w = 0.8 / len(arms)
    for j, a in enumerate(arms):
        ax.bar(x + j * w - 0.4 + w / 2, gaps[a], w,
               yerr=2 * gaps[a + "_SE"], capsize=2, color=COLOUR[a],
               edgecolor="black" if a == hero else "none",
               linewidth=1.2 if a == hero else 0, label=LABEL[a], zorder=3)
    ax.axhline(0, color=COLOUR[BASE], lw=1.5, ls="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{b}\n({s:.0f}% of val)" for b, s in zip(df["bucket"], df["val_share_%"])],
        fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.grid(alpha=.3, axis="y", zorder=0)


def fig1_grid(root: Path, out: Path) -> None:
    logs = {}
    for d in sorted(root.iterdir()):
        if (d / "log.csv").exists():
            lg = pd.read_csv(d / "log.csv")
            logs[d.name] = lg[lg["val_loss"].notna()].drop_duplicates("val_loss")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, lg in logs.items():
        a.plot(lg["step"], lg["val_loss"], marker="o", ms=3,
               color=COLOUR[name], label=LABEL[name],
               lw=2.2 if name == HERO else 1.4)
    a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("val loss (log)")
    a.set_title("Five arms, one variable: the embedding module", fontsize=10, weight="bold")
    a.legend(fontsize=8); a.grid(alpha=.3, which="both")

    base = logs[BASE].set_index("step")["val_loss"]
    for name, lg in logs.items():
        if name == BASE:
            continue
        s = lg.set_index("step")["val_loss"]
        k = base.index.intersection(s.index)
        b.plot(k, s[k] - base[k], marker="o", ms=3, color=COLOUR[name],
               label=LABEL[name], lw=2.2 if name == HERO else 1.4)
    b.axhline(0, color=COLOUR[BASE], lw=1.5, ls="--")
    b.set_xlim(left=200); b.set_xlabel("step")
    b.set_ylabel("val loss minus baseline (negative = better)")
    b.set_title("wave/l2 starts worse, crosses at ~step 1500, keeps widening",
                fontsize=10, weight="bold")
    b.legend(fontsize=8); b.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig1_grid.png", dpi=150); plt.close(fig)


def fig2_decomposition(root: Path, out: Path) -> None:
    fq, fg = (pd.read_csv(root / "buckets_frequency.csv"),
              pd.read_csv(root / "buckets_frequency_gaps.csv"))
    ln, lg = (pd.read_csv(root / "buckets_length.csv"),
              pd.read_csv(root / "buckets_length_gaps.csv"))
    fq, fg = fq[fq["val_share_%"] >= 0.1], fg[fq["val_share_%"].values >= 0.1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    bars(a, fq, fg, "By training frequency: the advantage VARIES",
         f"loss minus {LABEL[BASE]}")
    bars(b, ln, lg, "By byte length: the advantage is FLAT",
         f"loss minus {LABEL[BASE]}")
    lo = min(a.get_ylim()[0], b.get_ylim()[0]); hi = max(a.get_ylim()[1], b.get_ylim()[1])
    a.set_ylim(lo, hi); b.set_ylim(lo, hi)      # shared scale, or the eye is fooled
    a.legend(fontsize=7, loc="upper left")
    fig.suptitle("Frequency and length are confounded — separating them shows the "
                 "effect tracks frequency", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig2_decomposition.png", dpi=150); plt.close(fig)


def fig3_which_side(root: Path, out: Path) -> None:
    tg, tgg = (pd.read_csv(root / "buckets_collision.csv"),
               pd.read_csv(root / "buckets_collision_gaps.csv"))
    pv, pvg = (pd.read_csv(root / "buckets_prev-collision.csv"),
               pd.read_csv(root / "buckets_prev-collision_gaps.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    bars(a, tg, tgg, "Scored by TARGET token: no collision effect\n"
                     "(the lm_head is untied — targets are always distinguishable)",
         f"loss minus {LABEL[BASE]}")
    bars(b, pv, pvg, "Scored by PRECEDING token: the effect appears\n"
                     "(collisions corrupt context, not the answer)",
         f"loss minus {LABEL[BASE]}")
    lo = min(a.get_ylim()[0], b.get_ylim()[0]); hi = max(a.get_ylim()[1], b.get_ylim()[1])
    a.set_ylim(lo, hi); b.set_ylim(lo, hi)
    b.legend(fontsize=7, loc="lower right")
    fig.suptitle("Same models, same data, same buckets — only which token defines "
                 "the bucket changed", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig3_which_side.png", dpi=150); plt.close(fig)


def fig4_attribution(root: Path, out: Path) -> None:
    pvg = pd.read_csv(root / "buckets_prev-collision_gaps.csv")
    pv = pd.read_csv(root / "buckets_prev-collision.csv")
    arms = arms_in(pvg)
    treat, ctrl = pvg.iloc[0], pvg.iloc[1]

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(arms))
    did = [treat[m] - ctrl[m] for m in arms]
    err = [2 * np.hypot(treat[m + "_SE"], ctrl[m + "_SE"]) for m in arms]
    a.bar(x, did, 0.6, yerr=err, capsize=3,
          color=[COLOUR[m] for m in arms],
          edgecolor=["black" if m == HERO else "none" for m in arms], zorder=3)
    for i, (d, e) in enumerate(zip(did, err)):
        a.text(i, d / 2, f"{d:+.3f}\n{abs(d/(e/2)):.0f} sigma", ha="center",
               va="center", fontsize=9, color="white", weight="bold")
    a.axhline(0, color="k", lw=1)
    a.set_xticks(x); a.set_xticklabels([LABEL[m] for m in arms], fontsize=7, rotation=12)
    a.set_ylim(min(did) * 1.18, 0.02)
    a.set_ylabel("(after collided) minus (after cropped-but-distinct)")
    a.set_title("Collision effect, isolated by difference-in-differences\n"
                "identical across arms — it belongs to the codec, not the normalisation",
                fontsize=10, weight="bold")
    a.grid(alpha=.3, axis="y", zorder=0)

    every = [BASE] + arms
    inv = [pv.iloc[0][m] - pv.iloc[1][m] for m in every]
    b.bar(np.arange(len(every)), inv, 0.6,
          color=[COLOUR[m] for m in every],
          edgecolor=["black" if m == HERO else "none" for m in every], zorder=3)
    for i, v in enumerate(inv):
        b.text(i, v + (0.012 if v > 0 else -0.012), f"{v:+.3f}", ha="center",
               va="bottom" if v > 0 else "top", fontsize=8)
    b.axhline(0, color="k", lw=1)
    b.set_xticks(np.arange(len(every)))
    b.set_xticklabels([LABEL[m] for m in every], fontsize=7, rotation=12)
    b.set_ylim(min(inv) * 1.25, max(inv) * 1.6)
    b.set_ylabel("collision-context loss minus control-context loss")
    b.set_title("Collision contexts are EASIER for every arm that can tell the\n"
                "tokens apart — and HARDER only for one-hot", fontsize=10, weight="bold")
    b.grid(alpha=.3, axis="y", zorder=0)
    fig.tight_layout(); fig.savefig(out / "fig4_attribution.png", dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/m2"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fn in (fig1_grid, fig2_decomposition, fig3_which_side, fig4_attribution):
        fn(args.root, args.out)
        print(f"  {fn.__name__} ok")
    print(f"wrote 4 figures to {args.out}/")


if __name__ == "__main__":
    main()
