"""Run manifests and the RUNS.md ledger.

A number without a manifest is not a result. Every training run writes a
manifest.json beside its outputs and appends one row to results/RUNS.md.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import torch


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_manifest(out_dir: Path, **fields) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = {
        "git_sha": git_sha(),
        "torch": torch.__version__,
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **fields,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    return path


def append_run_row(runs_md: Path, milestone: str, config: str, seed,
                   sha: str, table_hash: str, final_val, out_dir: str) -> None:
    runs_md.parent.mkdir(parents=True, exist_ok=True)
    if not runs_md.exists():
        runs_md.write_text(
            "# Run ledger\n\n"
            "| date | milestone | config | seed | git SHA | table hash | "
            "final val loss | outputs |\n|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8")
    row = (f"| {time.strftime('%Y-%m-%d')} | {milestone} | {config} | {seed} "
           f"| {sha} | {str(table_hash)[:12]} | {final_val} | {out_dir} |\n")
    with runs_md.open("a", encoding="utf-8") as f:
        f.write(row)
