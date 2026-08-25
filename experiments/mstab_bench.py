"""Bench wrapper for the stability arms — installs the tied patch, then hands
off to the frozen bench (which resolves build_wte and GPT at call time).

    python experiments/mstab_bench.py --config configs/mstab_tied.yaml
"""
from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "src"))
from experiments import mstab_train          # noqa: F401 — applies the patch
from experiments import m2_bench
if __name__ == "__main__":
    m2_bench.main()
