"""M3 — three-language data preparation.

M2 trained on English + Hindi, which meant only one Indic script could be
measured. The collision audit says Malayalam is the worst-hit script in the
vocabulary — 182 of its 1,797 tokens receive identical one-hot codes at
pos_dim=16, a 10.13% rate against Devanagari's 6.29% and Latin's 0.02%.

Adding it turns the per-script analysis into a three-point dose gradient inside
a single run:

    Latin       0.02% of its tokens collide
    Devanagari  6.29%
    Malayalam  10.13%

with a falsifiable prediction attached — the wave codec's advantage over the
one-hot grid should order Malayalam > Devanagari > Latin. Every other Indic
script in the vocabulary sits within a point of Devanagari and would add a
second measurement at the same dose rather than a new one.

Writes nanoGPT-style memmap shards as uint32. The vocabulary is 131,072, so
uint16 would silently corrupt every id above 65,535.

Tokens are flushed to disk in blocks rather than accumulated in memory. A
Python list of 550M ids costs roughly 18 GB of RAM — M2's 100M run survived
that pattern at ~3.6 GB, and M3's would not.

    python experiments/m3_prepare_data.py --hf --tokens 550_000_000
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

SEP_TEXT = "\n\n"
DEFAULT_MIX = "eng=0.40,hin=0.30,mal=0.30"

# `eng` streams FineWeb-Edu; everything else streams Sangraha's verified split,
# which is keyed by ISO 639-3 code.
SANGRAHA = "ai4bharat/sangraha"


def open_stream(code: str, hindi_source: str):
    from datasets import load_dataset
    if code == "eng":
        if hindi_source == "fineweb2":       # keep both sides on the same family
            return iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                                     split="train", streaming=True))
        return iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                                 split="train", streaming=True))
    if hindi_source == "fineweb2":
        name = {"hin": "hin_Deva", "mal": "mal_Mlym", "tam": "tam_Taml",
                "tel": "tel_Telu", "ben": "ben_Beng"}[code]
        return iter(load_dataset("HuggingFaceFW/fineweb-2", name=name,
                                 split="train", streaming=True))
    return iter(load_dataset(SANGRAHA, data_dir=f"verified/{code}",
                             split="train", streaming=True))


def parse_mix(spec: str) -> list[tuple[str, float]]:
    parts = []
    for chunk in spec.split(","):
        code, _, frac = chunk.partition("=")
        parts.append((code.strip(), float(frac)))
    total = sum(f for _, f in parts)
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"mixture fractions sum to {total}, not 1.0")
    return parts


def iter_hf_texts(mix, hindi_source: str, seed: int):
    """Interleave sources by their requested fractions. A source that runs dry
    is dropped and the remaining fractions renormalised, so the run continues
    rather than silently ending early."""
    rng = np.random.default_rng(seed)
    live = {code: open_stream(code, hindi_source) for code, _ in mix}
    weights = {code: f for code, f in mix}
    while live:
        codes = list(live)
        w = np.array([weights[c] for c in codes], dtype=float)
        pick = codes[int(rng.choice(len(codes), p=w / w.sum()))]
        try:
            yield pick, next(live[pick])["text"]
        except StopIteration:
            print(f"  source '{pick}' exhausted — continuing without it", flush=True)
            del live[pick]


def iter_local_texts(path: Path):
    for doc in path.read_text(encoding="utf-8").split("\n\n"):
        if doc.strip():
            yield "local", doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", action="store_true")
    src.add_argument("--local-text", type=Path)
    ap.add_argument("--tokenizer", default="theschoolofai/BrahmicTokenizer-131K")
    ap.add_argument("--mix", default=DEFAULT_MIX,
                    help=f"comma-separated code=fraction (default: {DEFAULT_MIX})")
    ap.add_argument("--tokens", type=int, default=550_000_000)
    ap.add_argument("--hindi-source", choices=["sangraha", "fineweb2"],
                    default="sangraha")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/m3"))
    args = ap.parse_args()

    from kronecker_v2.vocab import load_tokenizer
    tok = load_tokenizer(args.tokenizer)
    sep_ids = tok(SEP_TEXT, add_special_tokens=False)["input_ids"]

    mix = parse_mix(args.mix)
    texts = (iter_hf_texts(mix, args.hindi_source, args.seed) if args.hf
             else iter_local_texts(args.local_text))

    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / "all.bin.tmp"
    FLUSH = 8_000_000            # ids buffered before a write (~32 MB)

    buf: list[int] = []
    per_source: dict[str, int] = {}
    n_docs = n_written = 0
    max_id = 0

    def flush(fh, keep: int | None = None) -> int:
        """Append the buffer to disk, optionally truncated to `keep` ids."""
        nonlocal buf, max_id
        chunk = buf if keep is None else buf[:keep]
        if chunk:
            a = np.asarray(chunk, dtype=np.uint32)
            max_id = max(max_id, int(a.max()))
            a.tofile(fh)
        n = len(chunk)
        buf = []
        return n

    with tmp.open("wb") as fh:
        for code, text in texts:
            toks = tok(text, add_special_tokens=False)["input_ids"]
            buf.extend(toks)
            buf.extend(sep_ids)
            per_source[code] = per_source.get(code, 0) + len(toks)
            n_docs += 1

            if n_written + len(buf) >= args.tokens:      # last partial block
                n_written += flush(fh, keep=args.tokens - n_written)
                break
            if len(buf) >= FLUSH:
                n_written += flush(fh)
                if n_docs % 5000 < 50:
                    share = {k: f"{100*v/max(n_written,1):.0f}%"
                             for k, v in per_source.items()}
                    print(f"  {n_docs:,} docs, {n_written:,} tokens  {share}",
                          flush=True)
        else:
            n_written += flush(fh)                       # sources ran dry

    assert max_id < 131_072, f"token id {max_id} exceeds the vocabulary"

    # Split without loading the whole stream: memmap the temp file, copy out.
    all_ids = np.memmap(tmp, dtype=np.uint32, mode="r")
    n_val = max(int(len(all_ids) * args.val_fraction), 1024)
    with (args.out / "train.bin").open("wb") as fh:
        for s0 in range(0, len(all_ids) - n_val, FLUSH):
            np.asarray(all_ids[s0:min(s0 + FLUSH, len(all_ids) - n_val)]).tofile(fh)
    np.asarray(all_ids[-n_val:]).tofile(args.out / "val.bin")
    train_n = len(all_ids) - n_val
    del all_ids
    tmp.unlink()


    (args.out / "meta.json").write_text(json.dumps({
        "dtype": "uint32", "vocab_size": 131_072,
        "train_tokens": int(train_n), "val_tokens": int(n_val),
        "n_docs": n_docs, "separator": SEP_TEXT,
        "mix_requested": dict(mix) if args.hf else {"local": 1.0},
        "mix_realised_tokens": per_source,
        "source": args.hindi_source if args.hf else "local",
        "seed": args.seed,
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {train_n:,} train + {n_val:,} val tokens -> {args.out}/")
    print("realised token share (differs from the request: fertility varies by "
          "script, so a fixed share of DOCUMENTS is not a fixed share of TOKENS):")
    tot = sum(per_source.values())
    for k, v in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<6}{v:>14,}  {100*v/tot:5.1f}%")


if __name__ == "__main__":
    main()