"""M3 — training entry point for the scale test.

Extends ``build_wte`` once more, on top of M5's extension, so the frozen M2
trainer keeps running every other code path unchanged. The chain is:

    m2_tiny_train.build_wte     dense / onehot / wave      (frozen)
      <- m5_train.build_wte     + bag / rp                 (patched at import)
        <- m3_train.build_wte   + hash / albert            (patched here)

Nothing on disk is edited; each layer falls through to the one below.

    python experiments/m3_train.py --config configs/m3_onehot.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import torch

from experiments import m2_tiny_train as T
from experiments import m5_train                      # noqa: F401 — installs bag/rp
from kronecker_v2.codecs.baselines import build_matched, budget_report

_prev_build_wte = T.build_wte                          # m5's, which wraps m2's


def build_wte(cfg: dict, device: str) -> torch.nn.Module:
    c = cfg["codec"]
    if c["name"] not in ("hash", "albert"):
        return _prev_build_wte(cfg, device)
    # Learned baselines: no frozen table, sized to the codec arms' budget.
    return build_matched(c["name"], cfg["model"]["d_model"],
                         code_dim=c.get("code_dim", 4096),
                         vocab_size=cfg["model"]["vocab_size"],
                         seed=c.get("seed", 0))


T.build_wte = build_wte

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", type=Path)
    known, _ = ap.parse_known_args()
    if known.config is not None:
        cfg = T.load_config(known.config)
        print(budget_report(cfg["model"]["d_model"],
                            cfg["codec"].get("code_dim", 4096),
                            cfg["model"]["vocab_size"]))
        print()
    T.main()
