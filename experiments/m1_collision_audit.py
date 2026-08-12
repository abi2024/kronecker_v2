"""Milestone 1 — the collision audit.

Two mechanisms produce permanent collisions, and they must be reported apart:

  A. TRUNCATION. Bytes past ``pos_dim`` are dropped, so long tokens agreeing on
     their prefix become identical. Scales with bytes-per-character, so it is
     an Indic problem before it is anyone else's.

  B. DECODE LOSSINESS. The reference ``token_id_to_bytes`` routes through
     ``tokenizer.decode()``, which renders every partial-codepoint byte-fallback
     token as U+FFFD. Those tokens collapse together at ANY pos_dim.

``--byte-source raw`` isolates A. ``--byte-source decode`` reproduces what the
shipped pipeline actually feeds the codec, i.e. A + B.

No training. No GPU. This is the cheapest load-bearing result in the project.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kronecker_v2.collisions import audit, examples, exact_collisions, truncation_stats
from kronecker_v2.vocab import load_tokenizer, vocab_bytes

INDIC = ["DEVANAGARI", "BENGALI", "TELUGU", "TAMIL", "MALAYALAM",
         "KANNADA", "GUJARATI", "ORIYA", "GURMUKHI"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default="theschoolofai/BrahmicTokenizer-131K")
    ap.add_argument("--byte-source", choices=["raw", "decode", "both"], default="both")
    ap.add_argument("--pos-dims", type=int, nargs="+", default=[12, 16, 20, 24, 32])
    ap.add_argument("--out", default="results/m1_collisions")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(args.tokenizer)
    sources = ["raw", "decode"] if args.byte_source == "both" else [args.byte_source]

    summary = {}
    for src in sources:
        vb = vocab_bytes(tok, source=src)
        rows = audit(vb, tuple(args.pos_dims))
        with (out / f"collisions_{src}.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

        per_pd = {}
        for pd in args.pos_dims:
            coll = exact_collisions(vb, pd)
            n = sum(len(v) for v in coll.values())
            per_pd[pd] = {
                "groups": len(coll),
                "collided_tokens": n,
                "pct_of_vocab": round(100.0 * n / len(vb), 4),
                "truncation": truncation_stats(vb, pd),
            }
        summary[src] = per_pd

        print(f"\n=== byte-source={src} ===")
        print(f"{'pos_dim':>8}{'groups':>8}{'tokens':>8}{'% vocab':>9}{'% trunc':>9}")
        for pd, s in per_pd.items():
            print(f"{pd:>8}{s['groups']:>8}{s['collided_tokens']:>8}"
                  f"{s['pct_of_vocab']:>8.3f}%{s['truncation']['pct_truncated']:>8.3f}%")

        if src == "raw":
            ex = {s: examples(vb, 16, s, k=4) for s in INDIC}
            (out / "examples_pos16.json").write_text(
                json.dumps(ex, ensure_ascii=False, indent=2), encoding="utf-8")
            print("\ncollision groups at pos_dim=16 (Devanagari):")
            for g in ex.get("DEVANAGARI", [])[:4]:
                print("   ", " / ".join(x.strip() for x in g))

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()