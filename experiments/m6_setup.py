"""M6 — tables and configs for one external tokenizer.

Builds the three codec tables (one-hot@16, wave2048, wave768) for an arbitrary
vocabulary and writes three standalone arm configs. The frozen trainer then
runs unchanged: configs point `tables_dir` at this tokenizer's own directory
and carry the right `vocab_size`, so `build_wte` and the model head just
follow the config.

Configs are standalone (no `extends`) because `load_config` merges shallowly —
a lesson already paid for once.

    python experiments/m6_setup.py --tokenizer google/mt5-small --slug mt5-small
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from experiments.m6_tokenizer_audit import extract_vocab_bytes
from kronecker_v2.tables import build_onehot_table, build_wave_table, table_hash

CONFIG = """# M6 arm — external tokenizer {name} (V={V:,}, {family})
# Standalone config: load_config merges shallowly, so every block is complete.
model:
  block_size: 1024
  vocab_size: {V}
  n_layer: 6
  n_head: 6
  d_model: 384
  dropout: 0.0
data:
  dir: data/m6/{slug}
  tables_dir: data/tables/m6_{slug}
codec: {codec}
train:
  device: null
  batch_size: {bs}
  grad_accum: {ga}
  max_steps: 3000
  lr: 6.0e-4
  warmup: 200
  min_lr_frac: 0.1
  weight_decay: 0.1
  grad_clip: 1.0
  eval_every: 250
  eval_iters: 50
  log_every: 20
  seed: 1337
  data_seed: 42
out_root: results/m6
"""

ARMS = {
    "onehot":  "{{name: onehot, pos_dim: 16, table: onehot16}}",
    "wave":    "{{name: wave, d_complex: 2048, seed: 0, normalize: l2, table: wave2048_l2}}",
    "wave768": "{{name: wave, d_complex: 768, seed: 0, normalize: l2, table: wave768_l2}}",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--slug", required=True,
                    help="short id used in paths and config names")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vb, family, _ = extract_vocab_bytes(tok)
    V = max(max(vb) + 1, len(tok))     # some vocabs report added ids past the
    for i in range(V):                 # piece table; pad so configs and data
        vb.setdefault(i, b"")          # prep agree on one vocab_size
    print(f"{args.tokenizer}: V={V:,} ({family})")

    tdir = _ROOT / "data" / "tables" / f"m6_{args.slug}"
    tdir.mkdir(parents=True, exist_ok=True)
    hp = tdir / "hashes.json"
    hashes = json.loads(hp.read_text()) if hp.exists() else {}
    jobs = [("onehot16",    lambda: build_onehot_table(vb, pos_dim=16)),
            ("wave2048_l2", lambda: build_wave_table(vb, 2048, 0, "l2")),
            ("wave768_l2",  lambda: build_wave_table(vb, 768, 0, "l2"))]
    for name, fn in jobs:
        path = tdir / f"{name}.pt"
        if path.exists():
            print(f"  {name}: exists, skipping"); continue
        t0 = time.time()
        t = fn(); torch.save(t, path)
        hashes[name] = table_hash(t)
        print(f"  {name}: {tuple(t.shape)} bf16, {t.numel()*2/2**30:.2f} GB, "
              f"{time.time()-t0:.0f}s, hash {hashes[name][:16]}…")
        del t
    hp.write_text(json.dumps(hashes, indent=2))

    # bigger head -> smaller microbatch, same 32,768 tokens/step
    bs, ga = (2, 16) if V > 200_000 else (4, 8)
    for arm, codec in ARMS.items():
        p = _ROOT / "configs" / f"m6_{args.slug}_{arm}.yaml"
        p.write_text(CONFIG.format(name=args.tokenizer, V=V, family=family,
                                   slug=args.slug, codec=codec.format(),
                                   bs=bs, ga=ga))
        print(f"  wrote {p.relative_to(_ROOT)}")
    print(f"done: tables in data/tables/m6_{args.slug}/, "
          f"microbatch {bs}x{ga} for V={V:,}")


if __name__ == "__main__":
    main()
