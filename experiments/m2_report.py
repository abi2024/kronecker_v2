"""Milestone 2 — the morning report.

Reads every run directory under --root, cross-checks the experimental controls,
scores the pre-registered claim, and renders the val-loss curves. One command
turns an overnight grid into a decision:

    python experiments/m2_report.py

Checks performed:
  * body_state_hash identical across arms (the "one variable" control)
  * status == ok for every run (the NaN watchdog result)
  * the claim: best WAVE arm final val within 5% of the one-hot baseline
  * stability: largest single-step train-loss jump per arm
Outputs: console verdict + results/m2/summary.csv + results/m2/val_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CLAIM_PCT = 5.0


def load_runs(root: Path) -> dict[str, dict]:
    runs = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        entry = {"manifest": json.loads(mf.read_text(encoding="utf-8")), "dir": d}
        lg = d / "log.csv"
        entry["log"] = pd.read_csv(lg) if lg.exists() else None
        runs[d.name] = entry
    return runs


def spike(log: pd.DataFrame | None) -> float:
    if log is None or len(log) < 3:
        return float("nan")
    return float(log["loss"].diff().max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/m2"))
    ap.add_argument("--baseline", default="m2_onehot")
    args = ap.parse_args()

    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"no runs with manifest.json under {args.root}")

    base = runs.get(args.baseline)
    base_val = base["manifest"].get("final_val_loss") if base else None

    rows = []
    for name, r in runs.items():
        m = r["manifest"]
        fv = m.get("final_val_loss")
        rows.append({
            "arm": name,
            "status": m.get("status"),
            "final_val": round(fv, 4) if fv is not None else None,
            "best_val": round(m.get("best_val_loss", float("nan")), 4),
            "Δ_vs_baseline_%": (round(100 * (fv - base_val) / base_val, 2)
                                 if fv is not None and base_val else None),
            "max_loss_jump": round(spike(r["log"]), 3),
            "emb_params": m.get("param_counts", {}).get("embedding"),
            "wall_min": round(m.get("wall_seconds", 0) / 60, 1),
            "body_hash": m.get("body_state_hash", "")[:12],
            "tokens_seen": m.get("tokens_seen"),
        })
    df = pd.DataFrame(rows)
    print("\n=== M2 grid ===")
    print(df.to_string(index=False))

    # ---- controls ----
    hashes = {r["manifest"].get("body_state_hash") for r in runs.values()}
    print(f"\nbody init identical across arms: "
          f"{'YES' if len(hashes) == 1 else 'NO — INVESTIGATE BEFORE READING ANY NUMBER'}")
    bad = [n for n, r in runs.items() if r["manifest"].get("status") != "ok"]
    print(f"all runs finished ok: {'YES' if not bad else 'NO: ' + ', '.join(bad)}")

    # ---- the pre-registered claim ----
    wave = {n: r for n, r in runs.items()
            if r["manifest"].get("cfg", {}).get("codec", {}).get("name") == "wave"
            and r["manifest"].get("status") == "ok"}
    if base_val and wave:
        best = min(wave.items(), key=lambda kv: kv[1]["manifest"]["final_val_loss"])
        bn, bv = best[0], best[1]["manifest"]["final_val_loss"]
        delta = 100 * (bv - base_val) / base_val
        verdict = "PASS" if delta <= CLAIM_PCT else "FAIL"
        print(f"\nCLAIM (best wave within {CLAIM_PCT}% of one-hot): {verdict}")
        print(f"  best wave arm: {bn}  val {bv:.4f}  vs one-hot {base_val:.4f}  "
              f"(Δ {delta:+.2f}%)")
        if delta < 0:
            print("  note: wave BEAT one-hot — report the sign, don't bury it")
    else:
        print("\nCLAIM: cannot score (missing one-hot baseline or wave arms)")

    # ---- artifacts ----
    args.root.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.root / "summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, r in runs.items():
        log = r["log"]
        if log is None:
            continue
        ev = log[log["val_loss"].notna()].drop_duplicates("val_loss", keep="first")
        ax.plot(ev["step"], ev["val_loss"], marker="o", ms=3, label=name)
    ax.set_xlabel("step"); ax.set_ylabel("val loss")
    ax.set_title("M2: validation loss (identical data order across arms)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(args.root / "val_curves.png", dpi=150)
    print(f"\nwrote {args.root/'summary.csv'} and {args.root/'val_curves.png'}")


if __name__ == "__main__":
    main()
