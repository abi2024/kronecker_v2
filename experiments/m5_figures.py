"""M4/M5 — the three figures the later findings rest on.

  fig5_dose_response   the collision effect against a known dose, with a placebo
  fig6_ablation        seed variance, and what the advantage is made of
  fig7_efficiency      loss against trainable embedding parameters

Assembles from the manifests already on disk across results/m2, m4 and m5, so
the figures cannot drift from the runs. Regenerate any time:

    python experiments/m5_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

C_OH, C_WV, C_RP, C_BAG, C_DENSE = "#d62728", "#1f77b4", "#9467bd", "#8c564b", "#7f7f7f"

# run directory -> (milestone root, label, d_complex or None, proj params)
WAVE_SWEEP = [
    ("m5", "m5_wave768",   768,   1536 * 384),
    ("m5", "m5_wave1024",  1024,  2048 * 384),
    ("m4", "m4_p12_wave",  1536,  3072 * 384),
    ("m2", "m2_wave_l2",   2048,  4096 * 384),
    ("m4", "m4_p24_wave",  3072,  6144 * 384),
    ("m4", "m4_p32_wave",  4096,  8192 * 384),
]
ONEHOT_SWEEP = [
    ("m4", "m4_p12_onehot", 12, 3072 * 384),
    ("m2", "m2_onehot",     16, 4096 * 384),
    ("m4", "m4_p24_onehot", 24, 6144 * 384),
    ("m4", "m4_p32_onehot", 32, 8192 * 384),
]
SEEDS = {"onehot": [("m2", "m2_onehot"), ("m5", "m5_s1338_onehot"), ("m5", "m5_s1339_onehot")],
         "wave":   [("m2", "m2_wave_l2"), ("m5", "m5_s1338_wave"), ("m5", "m5_s1339_wave")]}
ABLATION = [("m5", "m5_bag", "bag\nno order", C_BAG),
            ("m2", "m2_onehot", "one-hot\n~15 eff. dims", C_OH),
            ("m5", "m5_rp", "rp\nsame info, spread", C_RP),
            ("m2", "m2_wave_l2", "wave\nphase-bound", C_WV)]


def val(roots: dict[str, Path], milestone: str, run: str) -> float:
    mf = roots[milestone] / run / "manifest.json"
    if not mf.exists():
        raise SystemExit(f"missing {mf}")
    return float(json.loads(mf.read_text())["final_val_loss"])


def fig5(root_m4: Path, out: Path) -> None:
    df = pd.read_csv(root_m4 / "dose_response.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    treated = df["treated_tokens"]
    ax.errorbar(treated, df["DiD"], yerr=2 * df["SE"], fmt="o-", ms=8,
                capsize=4, color=C_WV, lw=2)
    for r in df.itertuples():
        ax.annotate(f"pos_dim {r.pos_dim}\n{r.DiD:+.3f} ({r.sigma:.0f}σ)",
                    (r.treated_tokens, r.DiD), textcoords="offset points",
                    xytext=(10, -18 if r.pos_dim == 16 else 8), fontsize=8)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("tokens of the FIXED treatment set this model merges")
    ax.set_ylabel("difference-in-differences (nats)")
    ax.set_title("Collision effect against a known dose\n"
                 "zero dose is a placebo the tokenizer built, not one we chose",
                 fontsize=11, weight="bold")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig5_dose_response.png", dpi=150)
    plt.close(fig)


def fig6(roots: dict[str, Path], out: Path) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    seeds = {k: np.array([val(roots, m, r) for m, r in v]) for k, v in SEEDS.items()}
    for i, (k, v, col) in enumerate((("one-hot", seeds["onehot"], C_OH),
                                     ("wave/l2", seeds["wave"], C_WV))):
        axL.errorbar([i], [v.mean()], yerr=[v.std(ddof=1)], fmt="o", ms=10,
                     capsize=6, color=col, lw=2)
        axL.scatter([i] * len(v), v, color=col, alpha=.45, s=45, zorder=3)
        axL.text(i + .12, v.mean(), f"{v.mean():.4f}\n±{v.std(ddof=1):.4f}",
                 va="center", fontsize=9)
    gaps = seeds["wave"] - seeds["onehot"]
    axL.set_xticks([0, 1]); axL.set_xticklabels(["one-hot", "wave/l2"])
    axL.set_xlim(-.5, 1.6); axL.set_ylabel("final val loss")
    axL.set_title(f"Seed variance, n=3 — the missing error bar\n"
                  f"paired gap {gaps.mean():+.4f} ± {gaps.std(ddof=1)/np.sqrt(3):.4f} SE"
                  f"   (t={abs(gaps.mean())/(gaps.std(ddof=1)/np.sqrt(3)):.1f}, 2 df)",
                  fontsize=11, weight="bold")
    axL.grid(alpha=.3, axis="y")

    vals = [val(roots, m, r) for m, r, _, _ in ABLATION]
    labels = [lab for _, _, lab, _ in ABLATION]
    cols = [c for _, _, _, c in ABLATION]
    noise = float(np.hypot(seeds["onehot"].std(ddof=1), seeds["wave"].std(ddof=1)))
    x = np.arange(len(vals))
    axR.bar(x, vals, .6, color=cols, zorder=3)
    axR.errorbar(x, vals, yerr=noise, fmt="none", ecolor="k", capsize=4, zorder=4)
    for i, v in enumerate(vals):
        axR.text(i, v + .004, f"{v:.4f}", ha="center", fontsize=9)
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=8)
    axR.set_ylim(min(vals) - .02, max(vals) + .02)
    axR.set_ylabel("final val loss")
    axR.set_title("What the advantage is made of\n"
                  "bars = ±1 seed sd; rp carries one-hot's exact information",
                  fontsize=11, weight="bold")
    axR.grid(alpha=.3, axis="y", zorder=0)
    fig.tight_layout(); fig.savefig(out / "fig6_ablation.png", dpi=150)
    plt.close(fig)


def fig7(roots: dict[str, Path], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for sweep, col, lab, key in ((WAVE_SWEEP, C_WV, "wave / l2", "d_complex"),
                                 (ONEHOT_SWEEP, C_OH, "one-hot", "pos_dim")):
        p = [s[3] for s in sweep]
        v = [val(roots, s[0], s[1]) for s in sweep]
        ax.plot(p, v, "o-", color=col, ms=7, lw=2, label=lab)
        for s, pp, vv in zip(sweep, p, v):
            ax.annotate(f"{key.split('_')[0]}={s[2]}", (pp, vv),
                        textcoords="offset points", xytext=(6, 6), fontsize=7.5)
    bw = min(WAVE_SWEEP, key=lambda s: val(roots, s[0], s[1]))
    bo = min(ONEHOT_SWEEP, key=lambda s: val(roots, s[0], s[1]))
    ax.annotate("", xy=(bw[3], val(roots, bw[0], bw[1])),
                xytext=(bo[3], val(roots, bo[0], bo[1])),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.4, ls=":"))
    ax.text((bw[3] + bo[3]) / 2, (val(roots, bw[0], bw[1]) + val(roots, bo[0], bo[1])) / 2,
            f"  {val(roots, bo[0], bo[1]) - val(roots, bw[0], bw[1]):+.4f} nats\n"
            f"  {bo[3]/bw[3]:.0f}x fewer params", fontsize=9, weight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("trainable embedding parameters (log scale)")
    ax.set_ylabel("final val loss")
    ax.set_title("Efficiency frontier: one-hot needs the width, wave does not",
                 fontsize=11, weight="bold")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(out / "fig7_efficiency.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m2", type=Path, default=Path("results/m2"))
    ap.add_argument("--m4", type=Path, default=Path("results/m4"))
    ap.add_argument("--m5", type=Path, default=Path("results/m5"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    roots = {"m2": args.m2, "m4": args.m4, "m5": args.m5}
    args.out.mkdir(parents=True, exist_ok=True)
    fig5(args.m4, args.out); print("  fig5_dose_response ok")
    fig6(roots, args.out);   print("  fig6_ablation ok")
    fig7(roots, args.out);   print("  fig7_efficiency ok")
    print(f"wrote 3 figures to {args.out}/")


if __name__ == "__main__":
    main()
