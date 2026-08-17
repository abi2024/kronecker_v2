"""M6 — audit off-the-shelf tokenizers against the byte window.

M1 showed the 32-byte window is satisfied by BrahmicTokenizer-131K only because
that vocabulary was BUILT to satisfy it. This asks whether vocabularies nobody
constrained — Llama-3, Gemma, mT5, Qwen — violate it, and by how much.

Byte extraction is the part that must not be botched (M1's second collision
mechanism came from a lossy decode path), so it is explicit per family:

  byte-level BPE (GPT-2, Llama-3, Qwen)
      pieces are strings over the 256-char GPT-2 byte alphabet; invert that
      mapping exactly. Pieces containing out-of-alphabet chars are added
      specials — taken as literal UTF-8.

  SentencePiece (mT5, Gemma, Llama-2)
      '<0xNN>' pieces are single byte-fallback bytes; otherwise the piece with
      U+2581 restored to space, UTF-8-encoded.

The family is detected per vocabulary (>95% of pieces mappable through the
byte alphabet -> byte-BPE), never assumed from the model name.

Validation: run against BrahmicTokenizer and it must reproduce M1 exactly —
903 collided tokens at pos_dim=16, zero at 32, max token length 32 bytes.

    python experiments/m6_tokenizer_audit.py
    python experiments/m6_tokenizer_audit.py --tokenizers gpt2 google/mt5-small
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from kronecker_v2.collisions import exact_collisions
from kronecker_v2.vocab import script_of

DEFAULT_TOKENIZERS = [
    "theschoolofai/BrahmicTokenizer-131K",      # the co-designed control
    "gpt2",
    "Qwen/Qwen2.5-7B",
    "google/mt5-small",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Meta-Llama-3-8B",               # gated: skipped unless logged in
    "google/gemma-2-9b",                        # gated: skipped unless logged in
]
POS_DIMS = (12, 16, 24, 32)


# ---------------------------------------------------------------- byte maps --
def gpt2_byte_maps() -> tuple[dict[int, str], dict[str, int]]:
    """The standard GPT-2 bytes<->unicode bijection."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\u00a1"), ord("\u00ac") + 1)) +
          list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    enc = {b: chr(c) for b, c in zip(bs, cs)}
    dec = {v: k for k, v in enc.items()}
    return enc, dec


_, BYTE_DEC = gpt2_byte_maps()


def piece_bytes_bpe(piece: str) -> bytes:
    try:
        return bytes(BYTE_DEC[ch] for ch in piece)
    except KeyError:                      # added special (<|begin_of_text|>…)
        return piece.encode("utf-8")


def piece_bytes_sp(piece: str) -> bytes:
    if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
        try:
            return bytes([int(piece[3:5], 16)])
        except ValueError:
            pass
    return piece.replace("\u2581", " ").encode("utf-8")


def extract_vocab_bytes(tok) -> tuple[dict[int, bytes], str, float]:
    """(id -> raw bytes, detected family, fraction mappable through byte alphabet)."""
    vocab = tok.get_vocab()
    n_map = sum(all(ch in BYTE_DEC for ch in p) for p in vocab)
    frac = n_map / max(len(vocab), 1)
    family = "byte-bpe" if frac > 0.95 else "sentencepiece"
    fn = piece_bytes_bpe if family == "byte-bpe" else piece_bytes_sp
    vb: dict[int, bytes] = {}
    for piece, idx in vocab.items():
        vb[idx] = fn(piece)
    for idx in range(max(vb) + 1):        # holes -> empty (never collide)
        vb.setdefault(idx, b"")
    return vb, family, frac


# -------------------------------------------------------------------- audit --
def audit_one(name: str) -> dict | None:
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name)
    except Exception as e:                # gated / offline / missing
        print(f"  SKIP {name}: {type(e).__name__}: {str(e)[:90]}")
        return None

    vb, family, frac = extract_vocab_bytes(tok)
    V = max(vb) + 1
    lens = [len(vb[i]) for i in range(V)]
    over16 = sum(l > 16 for l in lens)
    over32 = sum(l > 32 for l in lens)

    row: dict = {"name": name, "vocab": V, "family": family,
                 "mappable": round(frac, 4), "max_bytes": max(lens),
                 "over16": over16, "over32": over32,
                 "n_special": len(getattr(tok, "all_special_ids", []) or [])}
    for pd in POS_DIMS:
        groups = exact_collisions(vb, pd)
        row[f"collided@{pd}"] = int(sum(len(g) for g in groups.values()))

    scripts: dict[str, int] = {}
    for g in exact_collisions(vb, 16).values():
        for t in g:
            s = script_of(vb[t]) or "OTHER"
            scripts[s] = scripts.get(s, 0) + 1
    row["top_scripts@16"] = sorted(scripts.items(), key=lambda kv: -kv[1])[:3]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizers", nargs="+", default=DEFAULT_TOKENIZERS)
    ap.add_argument("--out", type=Path, default=Path("results/m6"))
    args = ap.parse_args()

    rows = []
    for name in args.tokenizers:
        print(f"auditing {name}…", flush=True)
        r = audit_one(name)
        if r:
            rows.append(r)
            print(f"  {r['family']:<14} V={r['vocab']:>7,}  max {r['max_bytes']:>3} B"
                  f"  >16B {r['over16']:>6,}  >32B {r['over32']:>6,}"
                  f"  collided@16 {r['collided@16']:>6,}  @32 {r['collided@32']:>6,}")

    if not rows:
        raise SystemExit("no tokenizer could be loaded")

    print(f"\n=== ranked by permanently merged tokens at pos_dim=16 ===")
    hdr = (f"{'tokenizer':<38}{'family':<15}{'vocab':>9}{'max B':>7}"
           f"{'>32 B':>8}{'@12':>8}{'@16':>8}{'@24':>8}{'@32':>8}")
    print(hdr); print("-" * len(hdr))
    ranked = sorted(rows, key=lambda r: -r["collided@16"])
    for r in ranked:
        print(f"{r['name']:<38}{r['family']:<15}{r['vocab']:>9,}{r['max_bytes']:>7}"
              f"{r['over32']:>8,}{r['collided@12']:>8,}{r['collided@16']:>8,}"
              f"{r['collided@24']:>8,}{r['collided@32']:>8,}")
    print("\ntop collided scripts @16, per tokenizer:")
    for r in ranked:
        tops = ", ".join(f"{s} {n}" for s, n in r["top_scripts@16"]) or "—"
        print(f"  {r['name']:<38}{tops}")

    args.out.mkdir(parents=True, exist_ok=True)
    # the co-designed control is the yardstick, never the experiment target
    external = [r for r in ranked if "BrahmicTokenizer" not in r["name"]]
    (args.out / "audit.json").write_text(json.dumps(
        {"ranking": external, "control": [r for r in ranked if r not in external]},
        indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}/audit.json  "
          f"(ranking excludes the co-designed control)")


if __name__ == "__main__":
    main()
