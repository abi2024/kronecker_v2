"""Stability curves — the day's actual result, runnable at ANY point.

Reads each arm's log.csv (written every step by the frozen trainer) and plots
val loss against tokens, plus each arm's gap to the one-hot baseline with the
measured d384 seed band shaded. Safe to run mid-training from the phone loop:
partial logs just produce shorter curves.

    python experiments/mstab_curves.py
"""

from __future__ import annotations

import argparse
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

SEED_SD = 0.0128            # d384 wave seed sd (M5, n=3) — the honesty band
TOK_PER_STEP = 32_768


def read_arm(d: Path) -> pd.DataFrame | None:
    f = d / "log.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[np.isfinite(df["val_loss"])]
    df = df.drop_duplicates(subset="step", keep="last")
    df = df[df["step"] % 250 == 0].sort_values("step")
    df["tokens"] = df["step"] * TOK_PER_STEP
    return df[["step", "tokens", "val_loss"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/mstab"))
    ap.add_argument("--baseline", default="mstab_onehot")
    args = ap.parse_args()

    arms = {d.name: read_arm(d) for d in sorted(args.root.iterdir())
            if d.is_dir()}
    arms = {k: v for k, v in arms.items() if v is not None and len(v)}
    if not arms:
        raise SystemExit(f"no logs under {args.root}")
    for k, v in arms.items():
        print(f"  {k:<16}{len(v):>4} evals, through step {int(v['step'].max()):>6,} "
              f"({v['tokens'].max()/1e6:,.0f}M tokens), latest val {v['val_loss'].iloc[-1]:.4f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5))
    for k, v in arms.items():
        axL.plot(v["tokens"] / 1e6, v["val_loss"], lw=1.6, label=k.replace("mstab_", ""))
    axL.set_xlabel("training tokens (M)"); axL.set_ylabel("val loss")
    axL.set_title("Long-horizon validation loss", weight="bold", fontsize=11)
    axL.legend(); axL.grid(alpha=.3)

    base = arms.get(args.baseline)
    if base is not None:
        for k, v in arms.items():
            if k == args.baseline:
                continue
            m = v.merge(base, on="step", suffixes=("", "_b"))
            axR.plot(m["tokens"] / 1e6, m["val_loss"] - m["val_loss_b"],
                     lw=1.6, label=f"{k.replace('mstab_','')} − onehot")
            tail = (m["val_loss"] - m["val_loss_b"]).tail(5)
            print(f"  gap {k.replace('mstab_',''):<10} last-5-evals mean "
                  f"{tail.mean():+.4f} (sd {tail.std(ddof=1):.4f})")
        axR.axhspan(-SEED_SD, SEED_SD, color="grey", alpha=.25,
                    label="±1 seed sd (d384)")
        axR.axhline(0, color="k", lw=1)
        axR.set_xlabel("training tokens (M)"); axR.set_ylabel("gap to one-hot (nats)")
        axR.set_title("Does the ordering survive the budget?\n"
                      "inside the grey band = not distinguishable from seed noise",
                      weight="bold", fontsize=11)
        axR.legend(); axR.grid(alpha=.3)
    fig.tight_layout()
    out = args.root / "stability_curves.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
