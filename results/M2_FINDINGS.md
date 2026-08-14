# Milestone 2 — Findings (FROZEN)

**Status:** CLOSED · 2026-08-14
**Pre-registered claim:** *best-norm wave reaches within 5% of one-hot's final
validation loss at matched steps, with no divergence.*
**Verdict: PASS — and the sign is negative.** The best wave arm **beat** the
baseline by 1.24% at identical trainable parameter count.
**Cost:** 5 arms × 3,000 steps × 98.3M tokens, ~59 min each, ~4.9 h total on one
RTX 3060.

---

## Setup

| | |
|---|---|
| model | 6 layer, 6 head, d_model 384; 11.02M body + 50.33M lm_head |
| data | 98M train / 2M val tokens, ~50% FineWeb-Edu English + ~50% Sangraha Hindi |
| tokenizer | BrahmicTokenizer-131K, identical in every arm |
| schedule | 3,000 steps × 32,768 tokens (microbatch 4 × accum 8 × block 1024) |
| optimizer | AdamW, lr 6e-4, cosine to 10%, 200-step warmup, clip 1.0 |
| precision | bf16 autocast, fp32 master weights |

**Controls verified, not assumed.** All five arms report
`body_state_hash = 4a6392274148` — every parameter outside the embedding
initialised bit-identically. All five drew batches from the same RNG stream, so
the data order was identical. All five report `status: ok`. The four
matched-parameter arms each have exactly 1,572,864 trainable embedding
parameters (`Linear(4096 → 384)`), enforced by `tests/test_equal_params.py`.

## Result

| arm | final val | vs one-hot | embedding params |
|---|---:|---:|---:|
| dense (reference, **not** matched) | 4.5592 | −5.27% | 50,331,648 |
| **wave / l2** | **4.7532** | **−1.24%** | 1,572,864 |
| one-hot @ pos_dim 16 (baseline) | 4.8129 | — | 1,572,864 |
| wave / sqrt_len | 4.8612 | +1.00% | 1,572,864 |
| wave / znorm | 4.9023 | +1.86% | 1,572,864 |

All three normalizations landed inside 2% of the baseline. None diverged; no
run produced a non-finite loss.

## Finding 1 — The phase code beats the one-hot grid at equal width

At identical input width (4,096 real dims), identical projection size, identical
body init and identical data order, `wave/l2` reaches a lower validation loss
than the one-hot Kronecker codec. The only difference between those two runs is
how a token's bytes are turned into 4,096 numbers.

## Finding 2 — The advantage is late-arriving, and still growing at cutoff

`wave/l2` starts **worse** and overtakes near step 1500:

| step | 250 | 500 | 750 | 1000 | 1250 | **1500** | 2000 | 2500 | 2999 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| l2 − one-hot | +0.207 | +0.192 | +0.086 | +0.041 | +0.014 | **−0.011** | −0.039 | −0.052 | **−0.060** |

Two consequences, and the second must be stated in any writeup:

1. The gap is **widening monotonically** from the crossover to the end of
   training. Nothing has converged at 3,000 steps.
2. The result is therefore **budget-dependent**. A run stopped at 1,000 steps
   would have shown `wave/l2` losing. Do not report the final number without
   the trajectory.

## Finding 3 — l2 is also the most stable

Projection gradient norms across training:

| arm | typical proj_gn | max observed |
|---|---|---|
| one-hot | 0.8 – 1.3 | 2.53 (step 1380), 2.01 (1460), 2.01 (1700) |
| wave / sqrt_len | 0.4 – 0.6 | 3.24 (step 2040) |
| wave / znorm | 0.4 – 0.7 | ~0.95 |
| **wave / l2** | **0.13 – 0.20** | **~0.91** |

l2 normalises every row to unit norm before projection, so the gradient
reaching the shared matrix is far more uniform across tokens. It wins on final
loss, on stability, and on trajectory — three independent grounds, not one.

## Finding 4 — The honest counterweight: dense still wins

The dense table reaches 4.5592, **4.08% better than wave/l2** — using **32×
the embedding parameters** (50.3M vs 1.57M). Its advantage is roughly constant
through the second half of training, neither closing nor widening.

The defensible framing is therefore:

> At this scale the codec buys a 32× reduction in embedding parameters for
> about 4% of validation loss; and among codecs at equal width, the phase-bound
> Fourier code beats the one-hot grid outright.

Not: "our embedding is better than a dense table." It is not, here.

## Decision

**Normalization is locked to `l2` for M3 and M4.** Selected on the ablation
above, not by preference.

---

# Freeze declaration — what is immutable entering M3

**FROZEN (code).** Changing any of these invalidates M2:

- everything already frozen at M1 (codecs, audit tools, tests)
- `src/kronecker_v2/model.py` — the shared body; `wte` remains the only
  injected component
- `src/kronecker_v2/embedding.py` — frozen code buffer + single trainable
  projection
- `src/kronecker_v2/tables.py` — chunk-verified builders (every table
  self-verifies against the frozen per-token codec before the bf16 cast)
- `experiments/m2_tiny_train.py` — the training loop, including the reseed
  after `wte` construction that keeps body init arm-independent

**FROZEN (constants).**

- `pos_dim = 16` ↔ `d_complex = 2048`; pairing rule `d_complex = 128 × pos_dim`
- wave normalization: **`l2`** (M2's output)
- byte source for training tables: `raw`
- microbatch 4 × grad_accum 8 = 32,768 tokens/step (sized to 12 GB; changing
  the split changes nothing scientific but re-run `m2_bench.py` first)
- seed 1337 (body init), data_seed 42 (batch order)

**FROZEN (results).** `results/m2/summary.csv`, `val_curves.png`, every
`manifest.json` and `log.csv`, the five `console_*.log` files, and
`results/m2_onehot_paging_incident.csv` (evidence for why the microbatch
changed mid-milestone).

**MUTABLE for M3.** Model size (30–50M), token budget, `configs/m3_*.yaml`,
new baseline arms (hash embeddings, ALBERT factorization), and per-script
bits-per-byte evaluation (`src/kronecker_v2/eval/bpb.py`, still a stub).

## Engineering lessons worth carrying (all cost real time)

1. **A GPU can be 6× slow without erroring.** Peak allocation of 14.79 GB on a
   12 GB card made WDDM page to system RAM: `device=cuda`, no warning,
   4.5k tok/s. Always bench before an overnight grid.
2. **The launch script must gate on file existence.** One grid was lost because
   two scripts were never placed; the `&&` pre-flight now catches it.
3. **Write artifacts before printing them.** A `UnicodeEncodeError` on a
   Windows cp1252 console destroyed a completed report. `summary.csv` is now
   written first.
4. **Line-buffer the log.** 240 steps of `log.csv` were lost to an unflushed
   buffer when a run was killed.
5. **Fix the colour map across plot panels.** Excluding the baseline from one
   panel restarts matplotlib's colour cycle and silently relabels the lines.

## Reproduce

```bash
python experiments/m2_build_tables.py
python experiments/m2_prepare_data.py --hf --tokens 100_000_000
for a in onehot wave_sqrtlen wave_l2 wave_znorm dense; do
  python experiments/m2_tiny_train.py --config configs/m2_$a.yaml
done
python experiments/m2_report.py
```

**Action:** `git add -A && git commit -m "M2 closed: findings frozen" && git tag m2-closed`
