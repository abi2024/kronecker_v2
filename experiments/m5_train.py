"""M5 — training entry point for the ablation arms.

``m2_tiny_train.py`` is frozen under the M2 declaration and its ``build_wte``
knows only dense / onehot / wave. Rather than edit it, this extends it in place
at import time: the frozen file on disk is untouched, and every other code path
— schedule, logging, manifest, the reseed that keeps body init arm-independent —
is literally the same code M2 ran.

    python experiments/m5_train.py --config configs/m5_bag.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import torch

from experiments import m2_tiny_train as T
from kronecker_v2.codecs.ablations import BagOfBytesCodec, RandomProjectedOneHotCodec
from kronecker_v2.embedding import CodecEmbedding

_frozen_build_wte = T.build_wte


def build_wte(cfg: dict, device: str) -> torch.nn.Module:
    c = cfg["codec"]
    if c["name"] not in ("bag", "rp"):
        return _frozen_build_wte(cfg, device)

    d_model, V = cfg["model"]["d_model"], cfg["model"]["vocab_size"]
    codes = torch.load(Path(cfg["data"]["tables_dir"]) / f"{c['table']}.pt",
                       map_location="cpu")
    if c["name"] == "bag":
        codec = BagOfBytesCodec(vocab_bytes={}, d_complex=c["d_complex"],
                                seed=c.get("seed", 0), normalize=c["normalize"])
    else:
        codec = RandomProjectedOneHotCodec(vocab_bytes={}, pos_dim=c["pos_dim"],
                                           seed=c.get("seed", 0),
                                           normalize=c["normalize"])
    return CodecEmbedding(codec, d_model=d_model, vocab_size=V, codes=codes)


T.build_wte = build_wte          # patched before main() resolves the global

if __name__ == "__main__":
    T.main()
