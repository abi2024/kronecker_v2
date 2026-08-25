"""Stability run — training entry point, adding the TIED baseline.

The lead-researcher review named two threats the short runs cannot answer:
every ranking is measured at 3,000 steps with no arm converged, and the
industry-default competitor — tying the input embedding to the lm_head — was
never run. One long day answers both: three arms, ~7.7x the M2 token budget,
val logged every 250 steps so the gap-vs-tokens curve IS the result.

The tied arm shares one Parameter between the input embedding and the output
head. The frozen trainer builds ``GPT(mcfg, wte)`` and only afterwards
constructs the optimizer, so tying is installed by wrapping the GPT class: the
wte carries a marker, and the wrapped constructor reassigns
``lm_head.weight = wte.weight`` before any optimizer sees the model.
``named_parameters()`` deduplicates shared tensors, so the weight is counted
and optimized exactly once. Nothing on disk is edited.

Two accounting notes, stated up front so the morning numbers aren't puzzling:
- ``param_counts`` will show the tied arm's ~50M under BOTH lm_head and
  embedding; the model's true total is ~11M smaller than the sum of columns.
- ``body_state_hash`` skips ``wte.*``, and under tying the shared weight is
  registered AS ``wte.weight`` — so the head drops out of the tied arm's hash
  and it will not match the other arms. The blocks/wpe/ln_f portion is still
  initialized identically; only the hash's coverage changes.

    python experiments/mstab_train.py --config configs/mstab_tied.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import torch

from experiments import m3_train                    # noqa: F401 — full codec chain
from experiments import m2_tiny_train as T

_prev_build_wte = T.build_wte
_FrozenGPT = T.GPT


def build_wte(cfg: dict, device: str) -> torch.nn.Module:
    c = cfg["codec"]
    if c["name"] != "tied":
        return _prev_build_wte(cfg, device)
    d_model, V = cfg["model"]["d_model"], cfg["model"]["vocab_size"]
    emb = torch.nn.Embedding(V, d_model)
    torch.nn.init.normal_(emb.weight, mean=0.0, std=0.02)   # the dense-arm init
    emb.tie_lm_head = True                                  # marker for TiedGPT
    return emb


class TiedGPT(_FrozenGPT):
    def __init__(self, cfg, wte) -> None:
        super().__init__(cfg, wte)
        if getattr(wte, "tie_lm_head", False):
            self.lm_head.weight = self.wte.weight           # one shared Parameter


T.build_wte = build_wte
T.GPT = TiedGPT

if __name__ == "__main__":
    T.main()
