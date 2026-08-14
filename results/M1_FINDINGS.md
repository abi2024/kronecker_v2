# Milestone 1 — Findings (FROZEN)

**Status:** CLOSED · 2026-08-12
**Reproduced independently on two machines** (Windows/MSVC and Linux/glibc): every audit
number matched exactly; codec phase hash matched exactly; codes agree within 1e-4.
**Cost:** zero training, zero GPU, ~2 minutes CPU.

Do not edit numbers in this file. If a codec changes, re-run the audit, re-pin the
fingerprints, add a RUNS.md entry, and write a new dated findings file.

---

## Environment

| component | value |
|---|---|
| tokenizer | BrahmicTokenizer-131K (`tokenizer.json`, 131,072 entries, 356 added) |
| reference codec | `kronecker-embeddings` 0.1.1 (PyPI) |
| VSA library | `torch-hd` 5.8.4 (imports as `torchhd`) |
| wave codec RNG | numpy PCG64, seed 0 (platform-stable by construction) |

**Pinned fingerprints** (see `tests/test_determinism.py`):

| pin | value | strictness |
|---|---|---|
| one-hot codes (pos_dim=16) | `2f3046d440bd…7c7ea445` | exact, platform-stable |
| wave phase tables (d=2048) | `b4b03664bc0a…ff6d438c` | exact, platform-stable |
| wave codes | 6 reference values, `atol=1e-4` | tolerant of libm noise |

---

## Finding 1 — The 32-byte window is a tokenizer design constraint, not a free parameter

Permanent collisions (raw byte source, UTF-8-safe truncation), full 131,072-token vocab:

| pos_dim | groups | collided tokens | % of vocab | % truncated |
|--------:|-------:|----------------:|-----------:|------------:|
| 12 | 1,877 | 5,335 | 4.070% | 5.015% |
| **16** | **380** | **903** | **0.689%** | **1.169%** |
| 20 | 131 | 303 | 0.231% | 0.542% |
| 24 | 39 | 93 | 0.071% | 0.220% |
| 32 | 0 | 0 | 0.000% | 0.000% |

Zero at 32 is **not a null result**: the tokenizer's own audit reports
`tokens > 32 bytes: 0` / `Kronecker constraints satisfied at POS_DIM=32`. The vocabulary
was built to obey the window. The one-hot codec therefore imposes a co-design rule —
**max token length ≤ pos_dim** — paid for upstream in the tokenizer and invisible in the
embedding's parameter accounting.

The row in bold is the paper's own 124M experimental setting (`pos_dim=16`, `D=4096`,
per §6.9 and `examples/02_nanogpt_integration.py`). There, the constraint is violated.

## Finding 2 — A second collision mechanism, independent of the window

The reference `token_id_to_bytes` resolves ids via `tokenizer.decode()`, which renders
every partial-codepoint byte-fallback token as U+FFFD. Those tokens collapse together at
**any** window:

| pos_dim | groups (decode) | tokens (decode) | % of vocab |
|--------:|----------------:|----------------:|-----------:|
| 16 | 391 | 2,056 | 1.569% |
| 32 | 10 | **1,151** | 0.878% |

At pos_dim=32 truncation contributes nothing — all 1,151 are decode-path casualties; 658
of them share the single byte string `b'\xef\xbf\xbd'`. Fixed by reading the byte-level
BPE piece directly (`--byte-source raw`). Both sources are reported because the decode
path is what the shipped pipeline actually feeds the codec.

## Finding 3 — The cost is not evenly distributed across scripts

Collided tokens at pos_dim=16 (raw source), per script:

| script | collided | of | % of own script |
|---|---:|---:|---:|
| Devanagari | 253 | 4,020 | 6.29% |
| Malayalam | 182 | 1,797 | 10.13% |
| Bengali | 101 | 2,166 | 4.66% |
| Telugu | 78 | 1,402 | 5.56% |
| Kannada | 77 | 1,422 | 5.41% |
| Tamil | 60 | 1,054 | 5.69% |
| **Latin** | **18** | **108,185** | **0.02%** |

Two orders of magnitude of asymmetry from one shared window: 3-byte-per-character
scripts exhaust it ~3× faster, and conjuncts (क्ष = 9 bytes for one visual symbol)
faster still. Representative groups: `संस्क / संस्कृति / संस्करण`,
`अधिकार / अधिकारी / अधिकारियों / अधिकांश`, `कार्य / कार्यक्रम / कार्यालय`.

## Finding 4 — The wave codec separates what one-hot merges, at equal width

Phase-bound Fourier codec (FHRR bind + fractional-power position), `d_complex=2048`
→ 4,096 real dims = one-hot `D` at pos_dim=16. Cosine similarity of colliding pairs:

| pair | one-hot @16 | wave @2048 |
|---|---|---:|
| कार्य / कार्यक्रम | IDENTICAL | 0.7609 |
| कार्य / कार्यालय | IDENTICAL | 0.7952 |
| कार्यक्रम / कार्यालय | IDENTICAL | 0.8001 |
| सरकार / सरकारी | IDENTICAL | 0.9130 |
| सरकार / सरकारले | IDENTICAL | 0.8416 |
| सरकारी / सरकारले | IDENTICAL | 0.8230 |

Separated **but still near** — orthographic locality (the reason Kronecker beat BPE)
is preserved; permanent merging is eliminated. Full table incl. the अधिकार family:
`python experiments/m1_separation_demo.py`.

## Reproducibility facts (discovered the hard way, now engineered around)

1. `torch.rand` is not portable across platforms/builds → phasors come from numpy PCG64.
2. `cos`/`sin` differ in last bits between MSVC and glibc → codes cannot be pinned
   bitwise cross-OS; phases are pinned exactly (no trig), codes by value at 1e-4.
3. BLAS GEMM is not row-order bitwise-stable → lookup asserted exact, projection
   asserted `allclose`.

Claim for the paper: *the wave codec is bit-reproducible in its phase tables across
platforms, and reproducible to ~1e-6 in its codes.*

## Reproduce

```bash
python experiments/m1_collision_audit.py --byte-source both   # findings 1–3
python experiments/m1_separation_demo.py                      # finding 4
pytest -q                                                     # 16 tests, all contracts
```

---

# Freeze declaration — what is immutable entering M2

**FROZEN (code).** Changing any of these invalidates M1 and requires: re-run audit →
re-pin fingerprints → RUNS.md entry → new findings file.

- `src/kronecker_v2/codecs/base.py` — Codec protocol + `OneHotCodec` (adapter over the
  published reference; never reimplemented)
- `src/kronecker_v2/codecs/wave.py` — phasor generation (numpy PCG64, seed 0), FHRR
  bind, FPE position rule. The `normalize=` argument stays exposed: **the mechanism is
  frozen; the normalization is M2's experimental variable, varied via config only.**
- `src/kronecker_v2/vocab.py`, `src/kronecker_v2/collisions.py` — the audit tools
- `tests/` — all 16, fingerprints pinned

**FROZEN (constants).**

- Experimental setting: `pos_dim=16` ↔ `d_complex=2048` (paper's 124M setting)
- Pairing rule: `d_complex = 128 × pos_dim` (enforced by `test_equal_params.py`)
- Tokenizer: BrahmicTokenizer-131K, untouched
- Byte source for training-facing tables: `raw`

**FROZEN (results).** `results/m1_collisions/*.csv`, `summary.json`,
`examples_pos16.json` — committed as-is.

**MUTABLE for M2.** `experiments/m2_tiny_train.py` (new), `configs/m2_*.yaml`, data
preparation scripts (new), tiny GPT model file (new, adapted from nanoGPT with
attribution), and — narrowly — `embedding.py` if device/dtype handling needs work for
the training loop (contract tests must keep passing).

**Action:** `git add -A && git commit -m "M1 closed: findings frozen" && git tag m1-closed`
