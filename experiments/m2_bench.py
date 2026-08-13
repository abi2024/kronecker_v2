"""Throughput diagnostic — is it compute, data loading, or the wrong device?

Times three things separately so the bottleneck is identified rather than
guessed at:

  1. pure model fwd+bwd on a resident random batch   -> raw compute ceiling
  2. batch fetch alone (memmap -> torch -> device)   -> data-path cost
  3. the real training step, both combined           -> what you actually get

Run:  python experiments/m2_bench.py --config configs/m2_onehot.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))            # repo root, for `experiments.*`
sys.path.insert(0, str(_ROOT / "src"))    # src layout, if not pip-installed

import torch

from kronecker_v2.model import GPT, GPTConfig
from experiments import m2_tiny_train as T   # reuse the real code paths


def timeit(fn, n: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/m2_onehot.yaml"))
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    cfg = T.load_config(args.config)
    t = cfg["train"]
    device = t.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()} "
          f"| resolved device={device}")
    if device.startswith("cuda"):
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    use_bf16 = device.startswith("cuda")
    ac = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16
          else (lambda: torch.autocast("cpu", enabled=False)))

    mcfg = GPTConfig(**cfg["model"])
    torch.manual_seed(t["seed"])
    wte = T.build_wte(cfg, device)
    model = GPT(mcfg, wte).to(device)
    opt = torch.optim.AdamW(model.optimizer_groups(t["weight_decay"]), lr=1e-4)
    B, blk, ga = t["batch_size"], mcfg.block_size, t["grad_accum"]
    print(f"config: batch={B} block={blk} grad_accum={ga} "
          f"-> {B*ga*blk:,} tokens/step")
    if device.startswith("cuda"):
        print(f"vram after model load: "
              f"{torch.cuda.memory_allocated()/2**30:.2f} GB allocated")

    # 1 — pure compute, batch already resident
    x = torch.randint(0, mcfg.vocab_size, (B, blk), device=device)
    y = torch.randint(0, mcfg.vocab_size, (B, blk), device=device)

    def compute_only():
        opt.zero_grad(set_to_none=True)
        with ac():
            _, loss = model(x, y)
        loss.backward()

    c = timeit(compute_only, args.iters)
    print(f"\n1. compute only      {c*1000:8.1f} ms/microbatch "
          f"-> {B*blk/c:12,.0f} tok/s")

    # 2 — data path only
    bins = T.Bins(Path(cfg["data"]["dir"]), blk, device, t["data_seed"])
    d = timeit(lambda: bins.train_batch(B), args.iters)
    print(f"2. data fetch only   {d*1000:8.1f} ms/microbatch "
          f"-> {B*blk/d:12,.0f} tok/s")

    # 3 — the real step
    def full_step():
        opt.zero_grad(set_to_none=True)
        for _ in range(ga):
            xb, yb = bins.train_batch(B)
            with ac():
                _, loss = model(xb, yb)
            (loss / ga).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    s = timeit(full_step, max(args.iters // 2, 3))
    print(f"3. full train step   {s*1000:8.1f} ms/step        "
          f"-> {B*ga*blk/s:12,.0f} tok/s")
    if device.startswith("cuda"):
        print(f"   peak vram: {torch.cuda.max_memory_allocated()/2**30:.2f} GB")

    share = (d * ga) / s * 100
    print(f"\ndata path is {share:.0f}% of step time; "
          f"compute is {(c*ga)/s*100:.0f}%")
    print(f"projected 3,000 steps: {s*3000/3600:.2f} h/arm")


if __name__ == "__main__":
    main()