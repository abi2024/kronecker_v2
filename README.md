# Kronecker V2 — phase-bound Fourier byte codes

The V1 Kronecker embedding writes a token's bytes into 32 fixed slots and drops
everything past slot 32, so tokens agreeing on their first 32 bytes receive
identical vectors permanently. For 3-byte-per-character Indic scripts that
window is ~10 characters, and a single conjunct can cost 9 bytes.

This project (a) measures how many real tokens collide, per script, on the
131,072-token BrahmicTokenizer vocabulary, and (b) replaces the one-hot position
factor with a phase rotation, so length is unbounded and long tokens degrade
gracefully instead of being cropped — at identical parameter count.

## Status
- [X ] M0 — reproduce V1 codec, pin determinism fingerprint
- [X ] M1 — collision audit per script (pos_dim 32/48/64) + analytic overlay
- [ ] M2 — tiny-scale stability of the wave codec
- [ ] M3 — matched-parameter head-to-head, per-script bits-per-byte
- [ ] M4 — 124M / 2.5B tokens / 3 seeds, all arms + robustness probe
- [ ] M5 — long-token stress test, writeup

## Rules
1. Only the embedding module changes between arms; everything else is fixed.
2. Every claim is written down before the run that tests it.
3. Equal parameter count is enforced by `tests/test_equal_params.py`, not by assertion.
4. Nothing load-bearing lives in a notebook. Figures are regenerated, never hand-edited.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
```


## Finding 1: the 32-byte window is not a free parameter — it is a tokenizer design constraint

The Kronecker codec sees only a token's first `pos_dim` bytes. Two tokens
agreeing on that prefix receive the *same* vector for the life of the model:
the codec is a pure function of those bytes, so no amount of training separates
them. How often that happens is a property of the vocabulary, and nobody had
measured it on the production vocabulary. This section does.

**At `pos_dim = 32`, the collision count is exactly zero — because the
tokenizer was built to make it so.** BrahmicTokenizer-131K enforces a maximum
token length of 32 bytes (its own audit reports `tokens > 32 bytes: 0`,
`Kronecker constraints satisfied at POS_DIM=32`), and we reproduce that
independently from `tokenizer.json`: maximum observed byte length is 32, and
no token is truncated.

This is the finding, not a null result. The one-hot grid does not work with an
arbitrary tokenizer; it imposes a hard co-design constraint — *max token length
≤ pos_dim* — that must be paid for upstream, in the vocabulary. The constraint
is invisible in the embedding's own accounting and shows up as a restriction on
what merges a tokenizer is allowed to learn.

**At the paper's own experimental setting the constraint is violated.** The
124M runs use `pos_dim = 16` (`D = 256 × 16 = 4096`, per §6.9 and
`examples/02_nanogpt_integration.py`). There, 1.17% of the vocabulary is
truncated and 903 tokens across 380 groups are permanently collided:

| pos_dim | groups | collided tokens | % of vocab | % truncated |
|--------:|-------:|----------------:|-----------:|------------:|
| 12 | 1,877 | 5,335 | 4.070% | 5.015% |
| 16 | 380 | 903 | 0.689% | 1.169% |
| 20 | 131 | 303 | 0.231% | 0.542% |
| 24 | 39 | 93 | 0.071% | 0.220% |
| 32 | 0 | 0 | 0.000% | 0.000% |

The loss is not distributed evenly. At `pos_dim = 16`, 253 Devanagari, 182
Malayalam, 101 Bengali, 78 Telugu and 77 Kannada tokens collide, against 18 of
108,185 Latin tokens (0.02%). Three-byte-per-character scripts exhaust the
window roughly three times faster, so a budget that is generous for English is
not generous for Indic. Representative groups at `pos_dim = 16`:

```
संस्क / संस्कृति / संस्करण
अधिकार / अधिकारी / अधिकारियों / अधिकांश
मन्त्रालय / मन्त्री / मन्त्र
कार्य / कार्यक्रम / कार्यालय
```

`कार्य` (work), `कार्यक्रम` (programme) and `कार्यालय` (office) are three
unrelated words with one shared vector.

## Finding 2: a second collision mechanism, independent of the window

`token_id_to_bytes` resolves a token to bytes via `tokenizer.decode()`, which
renders every partial-codepoint byte-fallback token as U+FFFD. Those tokens
therefore collapse onto each other regardless of `pos_dim`: 1,151 tokens in 10
groups still collide at `pos_dim = 32`, where truncation contributes nothing
(658 of them share the single byte string `b'\xef\xbf\xbd'`).

This is a property of the byte-extraction path, not of the codec, and it is
fixed by reading the byte-level BPE piece directly rather than decoding it. We
report both: `--byte-source raw` isolates truncation, `--byte-source decode`
reproduces what the shipped pipeline actually feeds the codec.

## Reproduce

```bash
python experiments/m1_collision_audit.py --byte-source both
```

No GPU, no training, ~2 minutes.