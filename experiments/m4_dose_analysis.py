"""M4 — the dose-response test, done on the side the mechanism acts on.

M2 established that collisions cost the one-hot codec ~0.355 nats on positions
that FOLLOW a collided token, isolated against a length-matched control. That is
association at a single dose. This asks whether the effect turns off when the
dose does.

Design — a fixed treatment set, four models:

  The 903 tokens that collide at pos_dim=16 are held FIXED across all four
  models. What varies is whether each model actually merges them:

      pos_dim 12   merges them (and 4,432 more)     treatment ON
      pos_dim 16   merges them                      treatment ON
      pos_dim 24   mostly does not                  treatment mostly OFF
      pos_dim 32   cannot: zero collisions exist    treatment OFF  <- placebo

  Holding the token set fixed matters. Comparing each model against its own
  collision set would confound the treatment with which tokens are in the
  bucket; here the bucket never changes, only whether the codec can tell its
  members apart.

Control group: tokens that exceed the 16-byte window but whose truncated form
stays unique AT THE NARROWEST WINDOW IN THE SWEEP. That last clause is doing
real work. Defining the control as "distinct at pos_dim=16" looks natural and
is wrong: 588 of those 887 tokens are themselves merged at pos_dim=12, so the
control is two-thirds TREATED in the highest-dose model and the
difference-in-differences cancels most of the effect it is meant to isolate.

Distinctness is monotone in the window — a wider window can only separate more
— so requiring distinctness at 12 guarantees it at 16, 24 and 32. The result is
one fixed control set, provably untreated in every model, verified per model in
the `control_merged` column (which must read 0 everywhere).

Prediction registered before the runs: the difference-in-differences shrinks
with the number of the fixed set still merged, and reaches zero at pos_dim=32.

Reads the per-token losses cached by m2_bucket_analysis; no GPU if they exist.

    python experiments/m4_dose_analysis.py
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

from experiments import m2_bucket_analysis as BA     # frozen: imported, not edited
from kronecker_v2.collisions import exact_collisions
from kronecker_v2.vocab import load_tokenizer, truncate, vocab_bytes

REFERENCE_POS_DIM = 16      # the window whose collision set defines the treatment
STRICT_POS_DIM = 12         # narrowest window in the sweep; the control must
                            # survive it to be untreated everywhere


def pairs(m4: Path, m2: Path) -> dict[int, tuple[Path, Path]]:
    return {
        12: (m4 / "m4_p12_onehot", m4 / "m4_p12_wave"),
        16: (m2 / "m2_onehot",     m2 / "m2_wave_l2"),      # M2's pair
        24: (m4 / "m4_p24_onehot", m4 / "m4_p24_wave"),
        32: (m4 / "m4_p32_onehot", m4 / "m4_p32_wave"),
    }


def merged_mask(vb, pos_dim: int, vocab_size: int) -> np.ndarray:
    """Tokens this window merges with at least one other token."""
    m = np.zeros(vocab_size, dtype=bool)
    for ids in exact_collisions(vb, pos_dim).values():
        m[np.asarray(ids, dtype=np.int64)] = True
    return m


def fixed_sets(tok, vocab_size: int):
    """Treatment: merged at REFERENCE_POS_DIM. Control: cropped there, but
    distinct even at STRICT_POS_DIM, hence untreated in every model."""
    vb = vocab_bytes(tok, vocab_size, source="raw")
    collided = merged_mask(vb, REFERENCE_POS_DIM, vocab_size)
    cropped = np.array([len(truncate(vb[i], REFERENCE_POS_DIM)) < len(vb[i])
                        for i in range(vocab_size)], dtype=bool)
    control = cropped & ~merged_mask(vb, STRICT_POS_DIM, vocab_size)
    return vb, collided, control


def still_merged(vb, subset: np.ndarray, pos_dim: int) -> int:
    """How many tokens of a FIXED subset this model actually merges."""
    return int((merged_mask(vb, pos_dim, subset.size) & subset).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m4-root", type=Path, default=Path("results/m4"))
    ap.add_argument("--m2-root", type=Path, default=Path("results/m2"))
    ap.add_argument("--data", type=Path, default=Path("data/m2"))
    ap.add_argument("--tokenizer", default="theschoolofai/BrahmicTokenizer-131K")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("results/m4"))
    args = ap.parse_args()

    P = pairs(args.m4_root, args.m2_root)
    missing = [str(d) for pr in P.values() for d in pr if not (d / "final.pt").exists()]
    if missing:
        raise SystemExit("missing checkpoints:\n  " + "\n  ".join(missing))

    meta = json.loads((args.data / "meta.json").read_text())
    V = meta["vocab_size"]
    any_dir = P[16][0]
    block = json.loads((any_dir / "manifest.json").read_text())["cfg"]["model"]["block_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    losses: dict[Path, np.ndarray] = {}
    for pr in P.values():
        for d in pr:
            cached = (d / "per_token_loss.npy").exists()
            print(f"{'cached' if cached else 'scoring'} {d.name}…", flush=True)
            losses[d] = BA.score_arm(d, device, args.data / "val.bin",
                                     block, args.batch)

    # Bucket by the INPUT token: a collision corrupts a token as context, not as
    # a target — the lm_head is untied, so targets stay perfectly scoreable.
    ids = BA.val_stream(args.data / "val.bin", block,
                        args.m4_root / "val_inputs.npy", offset=0)
    n = min(len(ids), *(len(v) for v in losses.values()))
    ids = ids[:n]
    losses = {k: v[:n] for k, v in losses.items()}

    print(f"\nbuilding the fixed treatment set at pos_dim={REFERENCE_POS_DIM}…")
    tok = load_tokenizer(args.tokenizer)
    vb, collided, control = fixed_sets(tok, V)
    m_treat, m_ctrl = collided[ids], control[ids]
    blen = np.array([len(vb[i]) for i in range(V)])
    print(f"  treatment {int(collided.sum()):,} tokens "
          f"({int(m_treat.sum()):,} val positions, "
          f"mean {blen[collided].mean():.1f} B)")
    print(f"  control   {int(control.sum()):,} tokens "
          f"({int(m_ctrl.sum()):,} val positions, "
          f"mean {blen[control].mean():.1f} B) "
          f"— distinct at pos_dim={STRICT_POS_DIM}, so untreated everywhere")

    rows = []
    for window, (oh, wv) in P.items():
        merged = still_merged(vb, collided, window)
        ctrl_merged = still_merged(vb, control, window)
        d_treat = losses[wv][m_treat] - losses[oh][m_treat]      # paired
        d_ctrl = losses[wv][m_ctrl] - losses[oh][m_ctrl]
        did = float(d_treat.mean() - d_ctrl.mean())
        se = float(np.hypot(d_treat.std(ddof=1) / np.sqrt(m_treat.sum()),
                            d_ctrl.std(ddof=1) / np.sqrt(m_ctrl.sum())))
        rows.append({
            "pos_dim": window,
            "treated_tokens": merged,
            "treated_%": round(100 * merged / max(int(collided.sum()), 1), 1),
            "control_merged": ctrl_merged,     # must be 0 — the control's warrant
            "after_collided": round(float(d_treat.mean()), 4),
            "after_control": round(float(d_ctrl.mean()), 4),
            "DiD": round(did, 4),
            "SE": round(se, 4),
            # a degenerate SE only happens on synthetic fixtures, but a
            # divide-by-zero in the verdict path is not worth risking
            "sigma": round(min(abs(did / se), 9999.0), 1) if se > 1e-12 else 0.0,
            "aggregate_gap": round(
                float(losses[wv].mean() - losses[oh].mean()), 4),
        })

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "dose_response.csv", index=False)
    print("\n=== dose-response: wave minus one-hot, fixed treatment set ===")
    print(df.to_string(index=False))
    if df["control_merged"].sum() != 0:
        print("\nWARNING: the control set is merged in some model — the "
              "difference-in-differences cancels part of the real effect there.")

    placebo = df[df["pos_dim"] == 32].iloc[0]
    top = df[df["pos_dim"] == 12].iloc[0]
    print(f"\nPLACEBO (pos_dim=32, {placebo['treated_tokens']} of the set merged): "
          f"DiD {placebo['DiD']:+.4f} ± {placebo['SE']:.4f} "
          f"({placebo['sigma']:.1f} sigma)")
    print(f"HIGHEST DOSE (pos_dim=12, {top['treated_tokens']} merged): "
          f"DiD {top['DiD']:+.4f} ± {top['SE']:.4f} ({top['sigma']:.1f} sigma)")
    tol = max(2 * placebo["SE"], 1e-6)
    verdict = "CONFIRMED" if abs(placebo["DiD"]) < tol else "NOT CONFIRMED"
    print(f"\nPrediction (effect vanishes at zero dose): {verdict}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(df))
    axL.bar(x, df["DiD"], 0.55, yerr=2 * df["SE"], capsize=3,
            color=["#1f77b4" if t > 0 else "#7f7f7f"
                   for t in df["treated_tokens"]], zorder=3)
    for i, r in df.iterrows():
        axL.text(i, r["DiD"] / 2 if abs(r["DiD"]) > 0.05 else r["DiD"] - 0.02,
                 f"{r['DiD']:+.3f}\n{r['sigma']:.0f}σ", ha="center", va="center",
                 fontsize=9, color="white" if abs(r["DiD"]) > 0.05 else "black",
                 weight="bold")
    axL.axhline(0, color="k", lw=1)
    axL.set_xticks(x)
    axL.set_xticklabels([f"pos_dim {r.pos_dim}\n{r.treated_tokens} of "
                         f"{int(collided.sum())} merged" for r in df.itertuples()],
                        fontsize=8)
    axL.set_ylabel("difference-in-differences (nats)")
    axL.set_title("Collision effect vs dose — fixed treatment set,\n"
                  "grey = no tokens merged (placebo)", fontsize=10, weight="bold")
    axL.grid(alpha=.3, axis="y", zorder=0)

    axR.plot(df["treated_tokens"], df["DiD"], "o-", color="#1f77b4", ms=7)
    seen: dict[int, int] = {}
    for r in df.itertuples():
        k = seen.get(r.treated_tokens, 0)
        seen[r.treated_tokens] = k + 1
        axR.annotate(f"p{r.pos_dim}", (r.treated_tokens, r.DiD),
                     textcoords="offset points",
                     xytext=(-24 if k else 8, 8 if k else -4), fontsize=9)
    axR.axhline(0, color="k", lw=1, ls="--")
    axR.set_xlabel("tokens of the fixed set this model merges")
    axR.set_ylabel("difference-in-differences (nats)")
    axR.set_title("Dose axis: does the effect scale with how many\n"
                  "of the SAME tokens the codec cannot separate?",
                  fontsize=10, weight="bold")
    axR.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(args.out / "dose_response.png", dpi=150)
    print(f"\nwrote {args.out}/dose_response.csv, dose_response.png")


if __name__ == "__main__":
    main()