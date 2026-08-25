"""Controlled stress test — the codec's limits, measured rather than asserted.

Two instruments:

CAPACITY CURVE (no checkpoints needed). Pairs of random byte strings differing
only in their LAST byte, across lengths 16→256. The one-hot grid at pos_dim=16
returns cosine **exactly 1.0** for every length past its window — provably
blind to suffix edits. The wave code separates them at every length, with
separation decaying smoothly as one byte among many: suffix-preserving,
capacity-limited, graceful degradation. Not losslessly unlimited.

SWAP PROBE (real checkpoints, real validation contexts). Take positions where
a collided token A appears as input; swap it for its collision partner B, and
separately for a length-matched NON-collided control C; measure the change in
logits and loss on all following positions.
  - An arm whose code table merges A and B must show max|Δlogits| = 0 on the
    collision swap — a bit-identical-forwards receipt, checked against the
    table rows themselves.
  - The control swap must move every arm — proof the probe can detect change.
  - Before any probing, each arm passes a reconstruction gate: its full-stream
    mean loss must sit within --gate of the manifest's final_val, proving the
    rebuilt model is the trained one.

    python experiments/stress_test.py --capacity-only
    python experiments/stress_test.py --root results/m2 --data data/m2 \
        --arms m2_onehot m2_wave_l2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from experiments import m3_train                    # noqa: F401 — codec chain
from experiments import m2_bucket_analysis as BA
from experiments import m2_tiny_train as T
from kronecker_v2.codecs.base import OneHotCodec
from kronecker_v2.codecs.wave import WaveKroneckerCodec
from kronecker_v2.collisions import exact_collisions
from kronecker_v2.model import GPT, GPTConfig
from kronecker_v2.vocab import truncate

LENGTHS = (16, 24, 32, 48, 64, 96, 128, 192, 256)


# ------------------------------------------------------------- capacity -----
def capacity_curve(out: Path, n: int = 64) -> None:
    # Text-range bytes only: the reference one-hot maps non-text bytes to
    # nothing, and an all-random string would zero out entirely — a real
    # behaviour, but not the experiment. Lowercase ASCII mirrors real tokens.
    rng = np.random.default_rng(7)
    codecs = {"onehot@16": OneHotCodec(vocab_bytes={}, pos_dim=16),
              "wave@2048": WaveKroneckerCodec(vocab_bytes={}, d_complex=2048),
              "wave@768": WaveKroneckerCodec(vocab_bytes={}, d_complex=768)}

    def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
        if torch.equal(u, v):
            return 1.0                       # exact, not fp-approximate
        a64, b64 = u.double().flatten(), v.double().flatten()
        na, nb = a64.norm(), b64.norm()
        assert na > 0 and nb > 0, "zero-norm code — invalid byte domain"
        return float((a64 @ b64) / (na * nb))

    rows = []
    for L in LENGTHS:
        cos = {k: [] for k in codecs}
        for _ in range(n):
            a = bytes(rng.integers(97, 123, L, dtype=np.uint8).tolist())
            last = 97 + (a[-1] - 97 + 1 + int(rng.integers(0, 24))) % 26
            b = a[:-1] + bytes([last])
            for k, c in codecs.items():
                cos[k].append(cosine(c.encode_bytes(a), c.encode_bytes(b)))
        rows.append({"length": L, **{k: float(np.mean(v))
                                     for k, v in cos.items()}})
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with (out / "capacity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print(f"{'len':>5}{'onehot@16':>11}{'wave@2048':>11}{'wave@768':>11}")
    for r in rows:
        print(f"{r['length']:>5}{r['onehot@16']:>11.4f}"
              f"{r['wave@2048']:>11.4f}{r['wave@768']:>11.4f}")
    blind = all(r["onehot@16"] == 1.0 for r in rows if r["length"] > 16)
    print(f"\none-hot cosine == 1.0 exactly at every length > 16: {blind}"
          f"  (provably blind to suffix edits past the window)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for k, col in (("onehot@16", "#d62728"), ("wave@2048", "#1f77b4"),
                   ("wave@768", "#2ca02c")):
        ax.plot([r["length"] for r in rows], [1 - r[k] for r in rows],
                "o-", color=col, label=k)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("token length (bytes), edit at the last byte")
    ax.set_ylabel("separation (1 − cosine)")
    ax.set_title("Suffix-edit separation vs length\n"
                 "grid: exactly zero past its window · wave: graceful decay",
                 fontsize=10, weight="bold")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "capacity_curve.png", dpi=150)
    print(f"wrote {out}/capacity.csv, capacity_curve.png")


# ----------------------------------------------------------- swap probe -----
def load_arm(run_dir: Path, device: str):
    ckpt = torch.load(run_dir / "final.pt", map_location="cpu",
                      weights_only=False)
    cfg = ckpt["config"]
    wte = T.build_wte(cfg, device)
    model = GPT(GPTConfig(**cfg["model"]), wte).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not unexpected and all(k.endswith("codes") for k in missing)
    model.eval()
    return model, cfg


def pick_tokens(tokenizer: str, V: int):
    from kronecker_v2.vocab import load_tokenizer, vocab_bytes
    vb = vocab_bytes(load_tokenizer(tokenizer), V, source="raw")
    groups = sorted(exact_collisions(vb, 16).values(), key=len, reverse=True)
    A, B = groups[0][0], groups[0][1]
    coll = {t for g in exact_collisions(vb, 12).values() for t in g}
    ctrl = [i for i in range(V)
            if len(truncate(vb[i], 16)) < len(vb[i]) and i not in coll]
    C = min(ctrl, key=lambda i: abs(len(vb[i]) - len(vb[A])))
    return A, B, C, vb


@torch.no_grad()
def probe(model, x, y, p: int, alt: int, device: str):
    """Return (max |Δlogit|, mean Δloss) on positions ≥ p after swapping x[p]."""
    xs = x.clone(); xs[0, p] = alt
    lo, _ = model(x.to(device), y.to(device))
    ls, _ = model(xs.to(device), y.to(device))
    dl = (ls - lo)[0, p:].abs().max().item()
    ce = lambda l: F.cross_entropy(l[0, p:], y[0, p:].to(device),
                                   reduction="mean").item()
    return dl, ce(ls) - ce(lo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_ROOT / "results" / "m2")
    ap.add_argument("--data", type=Path, default=_ROOT / "data" / "m2")
    ap.add_argument("--arms", nargs="+", default=["m2_onehot", "m2_wave_l2"])
    ap.add_argument("--tokenizer", default="theschoolofai/BrahmicTokenizer-131K")
    ap.add_argument("--pair", nargs=2, type=int, default=None,
                    help="explicit collided ids A B (testing hook)")
    ap.add_argument("--control", type=int, default=None)
    ap.add_argument("--capacity-only", action="store_true")
    ap.add_argument("--gate", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", type=Path, default=_ROOT / "results" / "stress")
    args = ap.parse_args()

    print("=== capacity curve ===")
    capacity_curve(args.out)
    if args.capacity_only:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((args.data / "meta.json").read_text())
    V = meta["vocab_size"]
    if args.pair:
        A, B = args.pair
        C = args.control
        assert C is not None, "--control required with --pair"
    else:
        A, B, C, _ = pick_tokens(args.tokenizer, V)
    print(f"\n=== swap probe ===  A={A} B={B} (collided@16)  C={C} (control)")

    report = {"A": A, "B": B, "C": C, "arms": {}}
    for arm in args.arms:
        d = args.root / arm
        mani = json.loads((d / "manifest.json").read_text())
        block = mani["cfg"]["model"]["block_size"]
        losses = BA.score_arm(d, device, args.data / "val.bin", block,
                              args.batch)
        gap = abs(float(losses.mean()) - mani["final_val_loss"])
        gate_ok = gap <= args.gate
        print(f"\n{arm}: reconstruction gate |{losses.mean():.4f} - "
              f"{mani['final_val_loss']:.4f}| = {gap:.4f} "
              f"{'PASS' if gate_ok else 'FAIL — skipping probe'}")
        if not gate_ok:
            report["arms"][arm] = {"gate": "FAIL"}
            continue

        model, _ = load_arm(d, device)
        merged = None
        codes = getattr(model.wte, "codes", None)
        if codes is not None:
            merged = bool(torch.equal(codes[A], codes[B]))
            print(f"  table rows A,B identical: {merged}")

        val = np.memmap(args.data / "val.bin", dtype=np.uint32, mode="r")
        n_seq = (len(val) - 1) // block
        hits = []
        for j in range(n_seq):
            seg = np.asarray(val[j * block:(j + 1) * block])
            for o in np.where(seg == A)[0]:
                if 32 <= o <= block - 64:
                    hits.append((j, int(o)))
        hits = hits[:args.k]
        if not hits:
            print("  no usable occurrences of A in val — probe skipped")
            report["arms"][arm] = {"gate": "PASS", "occurrences": 0}
            continue

        res = {"collision": [], "control": []}
        for j, o in hits:
            x = torch.from_numpy(np.asarray(
                val[j * block:(j + 1) * block], dtype=np.int64)).unsqueeze(0)
            y = torch.from_numpy(np.asarray(
                val[j * block + 1:(j + 1) * block + 1],
                dtype=np.int64)).unsqueeze(0)
            res["collision"].append(probe(model, x, y, o, B, device))
            res["control"].append(probe(model, x, y, o, C, device))
        summary = {}
        for k2, v in res.items():
            dls, dcs = zip(*v)
            summary[k2] = {"max_abs_dlogit": float(max(dls)),
                           "mean_dloss": float(np.mean(dcs))}
            print(f"  {k2:<10} swap A->{B if k2=='collision' else C}: "
                  f"max|Δlogit| {max(dls):.3e}   meanΔloss {np.mean(dcs):+.4f} "
                  f"  ({len(v)} contexts)")
        if merged:
            receipt = summary["collision"]["max_abs_dlogit"] == 0.0
            print(f"  RECEIPT (merged codes ⇒ bit-identical forwards): "
                  f"{'CONFIRMED' if receipt else 'VIOLATED'}")
            summary["receipt"] = receipt
        report["arms"][arm] = {"gate": "PASS", "occurrences": len(hits),
                               **summary}
        del model

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stress_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}/stress_report.json")


if __name__ == "__main__":
    main()
