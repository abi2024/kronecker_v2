"""M5 — throughput/VRAM bench for the ablation arms.

``m2_bench.py`` resolves ``build_wte`` through the frozen trainer, which knows
nothing about the bag and rp codecs. Importing ``m5_train`` first installs the
patch; ``m2_bench`` then looks the function up on the module at call time and
picks it up. Nothing on disk changes.

Running this before the overnight grid checks two things at once: the memory
headroom, and that the patched code path actually constructs a model.

    python experiments/m5_bench.py --config configs/m5_bag.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from experiments import m5_train          # noqa: F401  — applies the patch
from experiments import m2_bench

if __name__ == "__main__":
    m2_bench.main()
