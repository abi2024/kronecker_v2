"""Milestone 2 — data preparation.

Builds a bilingual (English + Hindi) token stream with the BrahmicTokenizer
and writes nanoGPT-style memmap shards:

    data/m2/train.bin, data/m2/val.bin   (uint32 — the vocab is 131,072,
                                          which does NOT fit in uint16; using
                                          uint16 here would silently corrupt
                                          every token id above 65,535)
    data/m2/meta.json

Two source modes:

  --hf            stream FineWeb-Edu (English) + Sangraha or FineWeb-2 (Hindi)
                  from the HuggingFace hub. Run this on your machine.
  --local-text F  tokenize a local text file instead. This is the CI/smoke
                  path and how the pipeline itself was tested.

Documents are joined with a separator sequence (the tokenizer ships no EOS in
its bare tokenizer.json), recorded in meta.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEP_TEXT = "\n\n"


def load_tok(path_or_name: str):
    from kronecker_v2.vocab import load_tokenizer
    return load_tokenizer(path_or_name)


def iter_hf_texts(english_fraction: float, hindi_source: str, seed: int):
    """Interleave English and Hindi documents by the requested fraction."""
    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    en = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                           split="train", streaming=True))
    if hindi_source == "fineweb2":
        hi = iter(load_dataset("HuggingFaceFW/fineweb-2", name="hin_Deva",
                               split="train", streaming=True))
    else:
        hi = iter(load_dataset("ai4bharat/sangraha", data_dir="verified/hin",
                               split="train", streaming=True))
    while True:
        src = en if rng.random() < english_fraction else hi
        try:
            yield next(src)["text"]
        except StopIteration:
            return


def iter_local_texts(path: Path):
    text = path.read_text(encoding="utf-8")
    for doc in text.split("\n\n"):
        if doc.strip():
            yield doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", action="store_true")
    src.add_argument("--local-text", type=Path)
    ap.add_argument("--tokenizer", default="theschoolofai/BrahmicTokenizer-131K")
    ap.add_argument("--tokens", type=int, default=100_000_000)
    ap.add_argument("--english-fraction", type=float, default=0.5)
    ap.add_argument("--hindi-source", choices=["sangraha", "fineweb2"],
                    default="sangraha")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/m2"))
    args = ap.parse_args()

    tok = load_tok(args.tokenizer)
    sep_ids = tok(SEP_TEXT, add_special_tokens=False)["input_ids"]

    texts = (iter_hf_texts(args.english_fraction, args.hindi_source, args.seed)
             if args.hf else iter_local_texts(args.local_text))

    ids: list[int] = []
    n_docs = 0
    for text in texts:
        ids.extend(tok(text, add_special_tokens=False)["input_ids"])
        ids.extend(sep_ids)
        n_docs += 1
        if len(ids) >= args.tokens:
            break
        if n_docs % 2000 == 0:
            print(f"  {n_docs} docs, {len(ids):,} tokens", flush=True)

    arr = np.array(ids[:args.tokens] if len(ids) >= args.tokens else ids,
                   dtype=np.uint32)
    assert int(arr.max(initial=0)) < 131_072
    n_val = max(int(len(arr) * args.val_fraction), 1024)
    train, val = arr[:-n_val], arr[-n_val:]

    args.out.mkdir(parents=True, exist_ok=True)
    train.tofile(args.out / "train.bin")
    val.tofile(args.out / "val.bin")
    (args.out / "meta.json").write_text(json.dumps({
        "dtype": "uint32", "vocab_size": 131_072,
        "train_tokens": int(len(train)), "val_tokens": int(len(val)),
        "n_docs": n_docs, "separator": SEP_TEXT,
        "english_fraction": args.english_fraction if args.hf else None,
        "hindi_source": args.hindi_source if args.hf else "local",
        "seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(train):,} train + {len(val):,} val tokens -> {args.out}/")


if __name__ == "__main__":
    main()
