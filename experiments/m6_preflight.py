"""M6 — preflight: prove byte extraction round-trips before training on it.

A whole night rides on tables built from ``extract_vocab_bytes``. The Brahmic
control already validates the byte-BPE path against M1's frozen numbers; this
validates whichever EXTERNAL tokenizer the audit picked, in its own family:
encode real text, reassemble it from the extracted per-token bytes, and demand
it match the tokenizer's own decode.

Exit 0 = safe to build tables. Nonzero = do not train on this tokenizer
(the overnight chain uses && so a failure skips that tokenizer's pipeline).

    python experiments/m6_preflight.py --tokenizer google/mt5-small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from experiments.m6_tokenizer_audit import extract_vocab_bytes

STRINGS = [
    "The quick brown fox jumps over the lazy dog.",
    "Internationalization and localization matter.",
    "नमस्ते दुनिया, यह एक परीक्षा है।",
    "सरकार ने अधिकारियों को कार्यालय भेजा।",
    "Price today: $12.50 (approx).",
]


def norm(s: str) -> str:
    return " ".join(s.split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vb, family, frac = extract_vocab_bytes(tok)
    print(f"{args.tokenizer}: family={family} (mappable {frac:.1%}), "
          f"V={max(vb)+1:,}")

    failures = 0
    for s in STRINGS:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        rebuilt = b"".join(vb[i] for i in ids).decode("utf-8", errors="replace")
        reference = tok.decode(ids, skip_special_tokens=True)
        ok = norm(rebuilt) == norm(reference) and "\ufffd" not in rebuilt
        print(f"  {'PASS' if ok else 'FAIL'}  {s[:44]!r}")
        if not ok:
            failures += 1
            print(f"        rebuilt:   {rebuilt[:80]!r}")
            print(f"        reference: {reference[:80]!r}")

    if failures:
        print(f"\nPREFLIGHT FAILED: {failures}/{len(STRINGS)} round-trips broke. "
              f"Byte extraction is wrong for this tokenizer — do NOT build "
              f"tables or train on it.")
        raise SystemExit(1)
    print("\npreflight ok — byte extraction round-trips; safe to build tables")


if __name__ == "__main__":
    main()
