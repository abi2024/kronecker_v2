"""Milestone 2 — the training loop.

One script, config-driven; the ONLY thing that differs between arms is the
``codec:`` block in the YAML. Body init, data order, schedule, and every other
setting are identical by construction, and the body-init equality is verified
(``body_state_hash`` in the manifest must match across arms with the same seed).

Pass/fail claim (pre-registered in M1_FINDINGS/RUNS.md): best-norm wave reaches
within 5% of one-hot's final validation loss at matched steps, no divergence.

Usage:
    python experiments/m2_tiny_train.py --config configs/m2_onehot.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from kronecker_v2.codecs.base import OneHotCodec
from kronecker_v2.codecs.wave import WaveKroneckerCodec
from kronecker_v2.embedding import CodecEmbedding
from kronecker_v2.model import GPT, GPTConfig
from kronecker_v2.runlog import append_run_row, git_sha, write_manifest


# ---------------------------------------------------------------- config ----
def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    while "extends" in cfg:
        parent = yaml.safe_load((path.parent / cfg.pop("extends")).read_text())
        parent.update({k: v for k, v in cfg.items()})
        cfg = parent
    return cfg


# ------------------------------------------------------------------ data ----
class Bins:
    def __init__(self, data_dir: Path, block: int, device: str, seed: int):
        meta = json.loads((data_dir / "meta.json").read_text())
        assert meta["dtype"] == "uint32"
        self.train = np.memmap(data_dir / "train.bin", dtype=np.uint32, mode="r")
        self.val = np.memmap(data_dir / "val.bin", dtype=np.uint32, mode="r")
        self.block, self.device = block, device
        self.rng = np.random.default_rng(seed)   # SAME seed across arms
                                                 # => identical batch order

    def _make(self, arr, ix):
        x = torch.stack([torch.from_numpy(
            arr[i:i + self.block].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            arr[i + 1:i + 1 + self.block].astype(np.int64)) for i in ix])
        if self.device.startswith("cuda"):
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device)

    def train_batch(self, batch_size: int):
        ix = self.rng.integers(0, len(self.train) - self.block - 1, batch_size)
        return self._make(self.train, ix)

    def val_batches(self, batch_size: int, iters: int):
        """Deterministic strided positions: every arm evaluates the same data."""
        span = len(self.val) - self.block - 1
        starts = np.linspace(0, span, batch_size * iters).astype(np.int64)
        for k in range(iters):
            yield self._make(self.val, starts[k * batch_size:(k + 1) * batch_size])


# ------------------------------------------------------------- embedding ----
def build_wte(cfg: dict, device: str) -> torch.nn.Module:
    c = cfg["codec"]
    d_model, V = cfg["model"]["d_model"], cfg["model"]["vocab_size"]
    if c["name"] == "dense":
        emb = torch.nn.Embedding(V, d_model)
        torch.nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        return emb

    table_path = Path(cfg["data"]["tables_dir"]) / f"{c['table']}.pt"
    codes = torch.load(table_path, map_location="cpu")          # bf16
    if c["name"] == "onehot":
        codec = OneHotCodec(vocab_bytes={}, pos_dim=c["pos_dim"])
    elif c["name"] == "wave":
        codec = WaveKroneckerCodec(vocab_bytes={}, d_complex=c["d_complex"],
                                   seed=c.get("seed", 0),
                                   normalize=c["normalize"])
    else:
        raise ValueError(c["name"])
    emb = CodecEmbedding(codec, d_model=d_model, vocab_size=V, codes=codes)
    return emb


# ----------------------------------------------------------------- train ----
def lr_at(step: int, t: dict) -> float:
    if step < t["warmup"]:
        return t["lr"] * (step + 1) / t["warmup"]
    frac = (step - t["warmup"]) / max(t["max_steps"] - t["warmup"], 1)
    coeff = 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
    min_lr = t["lr"] * t["min_lr_frac"]
    return min_lr + coeff * (t["lr"] - min_lr)


@torch.no_grad()
def evaluate(model, bins, t, autocast_ctx) -> float:
    model.eval()
    losses = []
    for x, y in bins.val_batches(t["batch_size"], t["eval_iters"]):
        with autocast_ctx():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    t = cfg["train"]
    name = args.config.stem
    out = args.out or Path(cfg.get("out_root", "results/m2")) / name
    out.mkdir(parents=True, exist_ok=True)

    device = t.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.startswith("cuda")
    autocast_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) \
        if use_bf16 else (lambda: torch.autocast("cpu", enabled=False))

    torch.manual_seed(t["seed"])
    mcfg = GPTConfig(**cfg["model"])
    wte = build_wte(cfg, device)
    torch.manual_seed(t["seed"])                 # reseed AFTER wte build so the
                                                 # body init is identical across
                                                 # arms regardless of how much
                                                 # RNG the embedding consumed
    model = GPT(mcfg, wte).to(device)
    counts = model.param_counts()
    body_hash = model.body_state_hash()

    bins = Bins(Path(cfg["data"]["dir"]), mcfg.block_size, device, t["data_seed"])
    opt = torch.optim.AdamW(model.optimizer_groups(t["weight_decay"]),
                            lr=t["lr"], betas=(0.9, 0.95))

    table_hash = cfg["codec"].get("table", "dense")
    print(f"[{name}] device={device} params={counts} body_hash={body_hash[:12]}")

    log = (out / "log.csv").open("w", encoding="utf-8", buffering=1)
    log.write("step,loss,val_loss,lr,grad_norm,proj_grad_norm,tokens_per_s\n")

    status, val_loss, best_val = "ok", float("nan"), float("inf")
    tokens_per_step = t["batch_size"] * t["grad_accum"] * mcfg.block_size
    t_start = time.time()
    model.train()

    for step in range(t["max_steps"]):
        lr = lr_at(step, t)
        for g in opt.param_groups:
            g["lr"] = lr

        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(t["grad_accum"]):
            x, y = bins.train_batch(t["batch_size"])
            with autocast_ctx():
                _, loss = model(x, y)
            (loss / t["grad_accum"]).backward()
            loss_acc += loss.item() / t["grad_accum"]

        if not math.isfinite(loss_acc):                      # NaN watchdog
            status = f"NON_FINITE_LOSS at step {step}"
            print(f"[{name}] ABORT: {status}")
            break

        proj = getattr(model.wte, "projection", None)
        proj_gn = (proj.weight.grad.norm().item()
                   if proj is not None and proj.weight.grad is not None else 0.0)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            t["grad_clip"]).item()
        opt.step()

        if step % t["log_every"] == 0 or step == t["max_steps"] - 1:
            tps = tokens_per_step / max(time.time() - t0, 1e-9)
            print(f"[{name}] step {step:5d} loss {loss_acc:.4f} lr {lr:.2e} "
                  f"gn {gn:.2f} proj_gn {proj_gn:.2f} {tps:,.0f} tok/s",
                  flush=True)
        if step % t["eval_every"] == 0 or step == t["max_steps"] - 1:
            val_loss = evaluate(model, bins, t, autocast_ctx)
            best_val = min(best_val, val_loss)
            print(f"[{name}] step {step:5d} VAL {val_loss:.4f}", flush=True)
        log.write(f"{step},{loss_acc:.5f},{val_loss:.5f},{lr:.3e},"
                  f"{gn:.3f},{proj_gn:.3f},{tokens_per_step/max(time.time()-t0,1e-9):.0f}\n")

    log.close()
    wall = time.time() - t_start
    torch.save({"model": model.state_dict(), "config": cfg},
               out / "final.pt")
    write_manifest(out, milestone="m2", config=str(args.config), cfg=cfg,
                   seed=t["seed"], data_seed=t["data_seed"],
                   param_counts=counts, body_state_hash=body_hash,
                   table=table_hash, status=status,
                   final_val_loss=val_loss, best_val_loss=best_val,
                   wall_seconds=round(wall, 1),
                   tokens_seen=tokens_per_step * (step + 1))
    append_run_row(Path("results/RUNS.md"), "m2", name, t["seed"], git_sha(),
                   table_hash, f"{val_loss:.4f}", str(out))
    print(f"[{name}] done: status={status} final_val={val_loss:.4f} "
          f"wall={wall:.0f}s -> {out}")


if __name__ == "__main__":
    main()