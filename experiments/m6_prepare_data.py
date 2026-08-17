"""M6 — retokenize the corpus with an external tokenizer.

Same sources and streaming machinery as M3 (imported from the frozen module,
not copied), but tokenized by an arbitrary Hugging Face tokenizer and written
with that tokenizer's own vocab size in `meta.json`. The corpus mix defaults to
M2's language pair (50% English, 50% Hindi) so the M6 comparison differs from
M2 in exactly one thing: the vocabulary.

    python experiments/m6_prepare_data.py --hf --tokenizer google/mt5-small \\
        --slug mt5-small --tokens 120_000_000
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

from experiments.m3_prepare_data import (SEP_TEXT, iter_hf_texts,
                                         iter_local_texts, parse_mix)

FLUSH = 8_000_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", action="store_true")
    src.add_argument("--local-text", type=Path)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--mix", default="eng=0.5,hin=0.5")
    ap.add_argument("--tokens", type=int, default=120_000_000)
    ap.add_argument("--hindi-source", choices=["sangraha", "fineweb2"],
                    default="sangraha")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = _ROOT / "data" / "m6" / args.slug

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    V = len(tok)
    sep_ids = tok(SEP_TEXT, add_special_tokens=False)["input_ids"]

    mix = parse_mix(args.mix)
    texts = (iter_hf_texts(mix, args.hindi_source, args.seed) if args.hf
             else iter_local_texts(args.local_text))

    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "all.bin.tmp"
    buf: list[int] = []
    per_source: dict[str, int] = {}
    n_docs = n_written = 0
    max_id = 0

    def flush(fh, keep=None):
        nonlocal buf, max_id
        chunk = buf if keep is None else buf[:keep]
        if chunk:
            a = np.asarray(chunk, dtype=np.uint32)
            max_id = max(max_id, int(a.max()))
            a.tofile(fh)
        n = len(chunk); buf = []
        return n

    with tmp.open("wb") as fh:
        for code, text in texts:
            t = tok(text, add_special_tokens=False)["input_ids"]
            buf.extend(t); buf.extend(sep_ids)
            per_source[code] = per_source.get(code, 0) + len(t)
            n_docs += 1
            if n_written + len(buf) >= args.tokens:
                n_written += flush(fh, keep=args.tokens - n_written)
                break
            if len(buf) >= FLUSH:
                n_written += flush(fh)
                share = {k: f"{100*v/max(n_written,1):.0f}%"
                         for k, v in per_source.items()}
                print(f"  {n_docs:,} docs, {n_written:,} tokens  {share}",
                      flush=True)
        else:
            n_written += flush(fh)

    assert max_id < V, f"token id {max_id} >= vocab size {V}"
    all_ids = np.memmap(tmp, dtype=np.uint32, mode="r")
    n_val = max(int(len(all_ids) * args.val_fraction), 1024)
    with (out / "train.bin").open("wb") as fh:
        for s in range(0, len(all_ids) - n_val, FLUSH):
            np.asarray(all_ids[s:min(s + FLUSH, len(all_ids) - n_val)]).tofile(fh)
    np.asarray(all_ids[-n_val:]).tofile(out / "val.bin")
    train_n = len(all_ids) - n_val
    del all_ids
    tmp.unlink()

    (out / "meta.json").write_text(json.dumps({
        "dtype": "uint32", "vocab_size": V, "tokenizer": args.tokenizer,
        "train_tokens": int(train_n), "val_tokens": int(n_val),
        "n_docs": n_docs, "mix_requested": dict(mix) if args.hf else {"local": 1.0},
        "mix_realised_tokens": per_source, "seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {train_n:,} train + {n_val:,} val -> {out}/  (V={V:,})")
    tot = sum(per_source.values())
    for k, v in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<6}{v:>14,}  {100*v/tot:5.1f}%")


if __name__ == "__main__":
    main()
