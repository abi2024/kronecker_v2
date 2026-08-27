"""README summary figures.

Two charts from frozen findings (values transcribed with their source sheet
cited inline — regeneration of the underlying numbers = rerun the cited
analysis), plus the single-panel stability gap figure rebuilt from the run
logs when they are present.

    python experiments/readme_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FIG = _ROOT / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_WAVE, C_DENSE, C_GRID, C_TIED = "#2ca02c", "#7f7f7f", "#d62728", "#1f77b4"


def fig_script_ladder() -> None:
    # Source: results/M3_FINDINGS.md (Finding 17) — relative BPB gain vs grid.
    scripts = ["Latin\n(0.02% collide)", "Devanagari\n(6.29%)", "Malayalam\n(10.13%)"]
    wave768 = [0.06, 3.92, 6.88]
    dense = [3.2, 6.0, 8.3]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - 0.19, wave768, 0.36, color=C_WAVE, label="wave768 (0.79M emb params)")
    ax.bar(x + 0.19, dense, 0.36, color=C_DENSE, label="dense (67M, reference)")
    for i, v in enumerate(wave768):
        ax.text(x[i] - 0.19, v + 0.12, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x, scripts)
    ax.set_ylabel("bits-per-byte improvement over the grid (%)")
    ax.set_title("The gain follows each script's collision rate\n"
                 "(registered as an ordering before the Malayalam data existed)",
                 fontsize=10.5, weight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig8_script_ladder.png", dpi=150)
    print("wrote figures/fig8_script_ladder.png")


def fig_window_audit() -> None:
    # Source: results/M6_FINDINGS.md (Findings 21–22) — collided@16 decomposition.
    # Stacked honestly: dark = collisions a 32-byte window would resolve;
    # light = permanent even at 32 (SP vocabs: duplicate aliases; byte-BPE:
    # tokens longer than 32 bytes). Cause attribution lives in Finding 22.
    toks = ["Gemma-2-9B", "Qwen2.5-7B", "Brahmic\n(co-designed)", "Mistral-7B", "GPT-2"]
    window = [1055, 859, 903, 4, 27]
    dup = [254, 209, 0, 250, 17]
    y = np.arange(len(toks))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(y, window, color=C_GRID, label="resolved by widening to 32 bytes")
    ax.barh(y, dup, left=window, color="#ff9896", label="permanent even at 32 bytes")
    ax.set_yticks(y, toks)
    ax.set_xlabel("tokens permanently merged at pos_dim = 16")
    ax.bar_label(ax.containers[1], labels=[f" {w+d:,}" for w, d in zip(window, dup)],
                 fontsize=9)
    ax.text(960, y[2] - 0.33, "only the co-designed vocabulary\nreaches zero at pos_dim = 32",
            fontsize=9, style="italic")
    ax.set_title("Every unconstrained tokenizer violates the byte window",
                 fontsize=10.5, weight="bold")
    ax.legend(frameon=False, loc="lower right"); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig9_window_audit.png", dpi=150)
    print("wrote figures/fig9_window_audit.png")


def fig_stability_gap() -> None:
    root = _ROOT / "results" / "mstab"
    arms = {}
    for name in ("mstab_wave768", "mstab_tied", "mstab_onehot"):
        f = root / name / "log.csv"
        if not f.exists():
            print(f"  {f} missing — skipping stability panel")
            return
        import pandas as pd
        d = pd.read_csv(f)
        d = d[np.isfinite(d["val_loss"])].drop_duplicates("step", keep="last")
        arms[name] = d[d["step"] % 250 == 0].set_index("step")["val_loss"]
    base = arms.pop("mstab_onehot")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, col, lab in (("mstab_wave768", C_WAVE, "wave768 − one-hot"),
                           ("mstab_tied", C_TIED, "tied − one-hot")):
        s = arms[name]
        steps = s.index.intersection(base.index)
        ax.plot(steps * 32768 / 1e6, (s[steps] - base[steps]).values,
                lw=1.6, color=col, label=lab)
    ax.axhspan(-0.0128, 0.0128, color="grey", alpha=.25, label="±1 seed sd")
    ax.axhline(0, color="k", lw=1)
    ax.annotate("plateau −0.041", xy=(700, -0.041), xytext=(430, -0.075),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("gap to one-hot (nats; below 0 = better)")
    ax.set_title("Gaps compress 2–3× under long training, then hold\n"
                 "the tied baseline wins the conventional architecture",
                 fontsize=10.5, weight="bold")
    ax.legend(frameon=False); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig10_stability_gap.png", dpi=150)
    print("wrote figures/fig10_stability_gap.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_script_ladder()
    fig_window_audit()
    fig_stability_gap()