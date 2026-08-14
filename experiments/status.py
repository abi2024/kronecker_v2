"""One-screen run status — designed to be readable on a phone.

Overnight monitoring through Remote Control works badly if the assistant has to
tail raw logs: the output is long, the interesting numbers are scattered, and a
phone screen shows about fifteen lines. This prints the whole grid in one block.

    python experiments/status.py                    # defaults to M4
    python experiments/status.py --root results/m3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

M4_ARMS = ["m4_p12_onehot", "m4_p12_wave", "m4_p24_onehot",
           "m4_p24_wave", "m4_p32_onehot", "m4_p32_wave"]


def gpu_line() -> str:
    try:
        q = ("utilization.gpu,memory.used,memory.total,temperature.gpu")
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=8).decode().strip().split(", ")
        return (f"GPU {out[0]}% busy | {int(out[1])/1024:.1f}/{int(out[2])/1024:.1f} GB"
                f" | {out[3]}C")
    except Exception:
        return "GPU: nvidia-smi unavailable"


def tail_log(path: Path, max_steps: int):
    """Last step, loss, throughput, and an ETA from the log's own timestamps."""
    try:
        lines = path.read_text(errors="replace").strip().splitlines()
        if len(lines) < 2:
            return None
        cols = lines[0].split(",")
        last = dict(zip(cols, lines[-1].split(",")))
        step = int(float(last["step"]))
        tps = float(last.get("tokens_per_s", 0) or 0)
        age_s = time.time() - path.stat().st_mtime
        # steps/second measured over the run so far, not from the config
        elapsed = time.time() - path.stat().st_ctime
        rate = (step + 1) / max(elapsed, 1)
        eta_min = (max_steps - step) / rate / 60 if rate > 0 else float("nan")
        return dict(step=step, loss=float(last["loss"]),
                    val=float(last.get("val_loss", "nan") or "nan"),
                    tps=tps, stale_s=age_s, eta_min=eta_min)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/m4"))
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--max-steps", type=int, default=3000)
    args = ap.parse_args()

    arms = args.arms or (M4_ARMS if args.root.name == "m4" else
                         sorted(d.name for d in args.root.iterdir() if d.is_dir())
                         if args.root.exists() else [])

    print(f"{time.strftime('%H:%M:%S')}  {args.root}   {gpu_line()}")
    print("-" * 62)

    done = running = 0
    for arm in arms:
        d = args.root / arm
        mf, lg = d / "manifest.json", d / "log.csv"
        if mf.exists():
            m = json.loads(mf.read_text())
            ok = m.get("status") == "ok"
            done += 1
            print(f"{'DONE' if ok else 'FAIL':<5}{arm:<20}"
                  f"val {m.get('final_val_loss', float('nan')):.4f}  "
                  f"{m.get('wall_seconds', 0)/60:.0f} min"
                  + ("" if ok else f"  <-- {m.get('status')}"))
        elif lg.exists():
            t = tail_log(lg, args.max_steps)
            if t is None:
                print(f"{'START':<5}{arm:<20}log present, no rows yet")
                continue
            running += 1
            warn = "  <-- STALLED?" if t["stale_s"] > 300 else ""
            print(f"{'RUN':<5}{arm:<20}step {t['step']}/{args.max_steps}  "
                  f"loss {t['loss']:.3f}  {t['tps']:,.0f} tok/s  "
                  f"eta {t['eta_min']:.0f} min{warn}")
        else:
            print(f"{'WAIT':<5}{arm:<20}not started")

    print("-" * 62)
    left = len(arms) - done - running
    print(f"{done} done, {running} running, {left} queued")
    if running == 0 and left > 0:
        print("NOTHING IS RUNNING — the chain may have died. Check the console log.")


if __name__ == "__main__":
    main()
