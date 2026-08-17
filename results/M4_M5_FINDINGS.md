# Milestones 4 & 5 — Findings (FROZEN)

**Status:** CLOSED · `git tag m5-closed`
**Cost:** 14 arms × 3,000 steps × 98.3M tokens, ~1.0 h each on one RTX 3060.
All analysis reuses cached per-token losses; no additional training.

Figures regenerate from the manifests and CSVs on disk:
`python experiments/m5_figures.py`. None is hand-edited.

---

## M4 — Is the collision mechanism causal?

**Pre-registered prediction** (written into `results/RUNS.md` before the runs):
*the wave-minus-one-hot gap shrinks monotonically as `pos_dim` rises, and
vanishes at `pos_dim=32` where the tokenizer guarantees zero collisions.*

### Design — a fixed treatment set, four models

`pos_dim` is a dial with a dose known exactly from the M1 audit. The treatment
set is the **903 tokens that collide at `pos_dim=16`**, held fixed across all
four models; only whether the codec merges them varies. Comparing each model
against its own collision set would confound the treatment with bucket
membership.

The control is long, Indic, cropped — but never ambiguous — so the
difference-in-differences cancels what the groups share. Everything is bucketed
on the **input** token, because the untied `lm_head` keeps targets perfectly
scoreable regardless of collisions (M2, Finding 6).

### Result

| pos_dim | of the set merged | after collided | after control | DiD | SE | σ |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 100% | −0.7935 | −0.0709 | **−0.7226** | 0.0156 | 46.3 |
| 16 | 100% | −0.3680 | −0.0244 | **−0.3436** | 0.0139 | 24.8 |
| 24 | 10.3% | −0.0060 | +0.0272 | −0.0333 | 0.0136 | 2.4 |
| **32** | **0%** | +0.0211 | +0.0166 | **+0.0045** | 0.0136 | **0.3** |

`control_merged = 0` on every row.

### Finding 11 — the effect vanishes at zero dose

**0.3σ at `pos_dim=32`.** The placebo is one the *tokenizer* built, not one we
chose: the vocabulary was constructed so no token exceeds 32 bytes, so there is
nothing there to fix.

### Finding 12 — and it is near-linear in dose

At 10.3% of the treatment set merged, the effect is 9.7% of full size — within
1.3 percentage points of exact proportionality. `pos_dim=12` overshoots
(−0.7226, 210% of the p16 effect) because merging is more severe there: the same
tokens fall into larger groups, and the surrounding context is degraded too
(its control gap is −0.0709 rather than ~0).

**Report −0.3436 ± 0.0139** as the effect at the reference setting.

### The methodological error, kept in the record

The first version defined the control as "cropped at 16 but distinct at 16".
**588 of those 887 tokens are themselves merged at `pos_dim=12`** — the control
was two-thirds *treated*, which suppressed p12's estimate (−0.3005 instead of
−0.7226) and produced a spurious placebo residual of +0.0238 at 3.7σ.

Distinctness is monotone in the window, so requiring it at the *narrowest*
window guarantees it everywhere. The corrected control is 299 tokens, matched on
length (mean 19.7 B against the treatment's 20.4 B, median 19 vs 19), and a
`control_merged` column now verifies per model that it is untreated.

**Both the wrong and the right numbers are preserved here.** The error is
instructive: a control that looks obviously correct can be silently treated, and
the failure mode is to *understate* your own effect.

---

## M5 — What is the advantage made of, and what is the noise floor?

### Finding 13 — seed variance, measured at last

Every number in this project was `n=1` until now.

| arm | seeds 1337 / 1338 / 1339 | mean | sd |
|---|---|---:|---:|
| one-hot | 4.8129 / 4.8122 / 4.8217 | 4.8156 | **0.0053** |
| wave / l2 | 4.7532 / 4.7343 / 4.7588 | 4.7488 | **0.0128** |

**Paired gap −0.0668 ± 0.0056 SE, t = 11.9 (2 df).** The M2 headline survives,
and the true gap is slightly larger than the single seed showed.

Recalibrating every prior claim against this noise floor:

| claim | effect | noise | sd | verdict |
|---|---:|---:|---:|---|
| wave beats one-hot | 0.0668 | 0.0056 | 11.9 | survives |
| one-hot depends on pos_dim | 0.0655 | 0.0053 | 12.4 | survives |
| wave768 beats one-hot's best | 0.0922 | 0.0139 | 6.6 | survives |
| bag is worse than wave | 0.0912 | 0.0139 | 6.6 | survives |
| rp recovers most of the gain | 0.0544 | 0.0139 | 3.9 | survives |
| **wave "insensitive to pos_dim"** | 0.0195 | 0.0128 | **1.5** | **not resolved** |
| **rp differs from wave at all** | 0.0124 | 0.0139 | **0.9** | **not resolved** |

**Withdrawn:** M4's unpredicted finding that wave is insensitive to `pos_dim`,
and that smaller wave codes are monotonically better. Wave's spread across the
four settings (0.0195) is 1.5× its own seed noise. The ordering was noise.

### Finding 14 — 81% of the aggregate gain is representational spread

The one-hot code is **not sparse** — z-normalisation makes every entry non-zero
— but it takes only **2 distinct values**, and its energy sits in ~15 effective
dimensions of 4,096 (participation ratio). Wave's sits in ~997.

Two ablations complete a 2×2:

- **`rp`** — the one-hot code through an invertible Gaussian matrix. Identical
  information, including its collisions and truncation; only the spread changes
  (PR ~1,436). Verified to inherit one-hot's collisions exactly.
- **`bag`** — the same frozen phasors and bundling, position rotation removed.
  Same spread, no order. Verified: anagram cosine is exactly 1.0000.

| arm | val | effective dims | order? | information |
|---|---:|---:|---|---|
| bag | 4.8400 | ~1,756 | **no** | byte multiset only |
| one-hot | 4.8156 | **~15** | yes | truncated bytes |
| **rp** | **4.7612** | ~1,436 | yes | **identical to one-hot** |
| wave | 4.7488 | ~997 | yes | untruncated bytes |

```
spread alone    one-hot → rp    −0.0544    81% of wave's total gain    3.9 sd
order alone     wave → bag      +0.0912    catastrophic                6.6 sd
residual        rp → wave       −0.0124    0.9 sd — NOT RESOLVED
```

**At matched width, `rp` and `wave` cannot be distinguished.** Most of the
aggregate advantage is not the Fourier construction; it is having the
information spread across dimensions instead of concentrated in fifteen.

Order binding is separately vindicated: strip it and the code lands *below* the
grid it was meant to improve on.

### Finding 15 — the phase code's real advantage is efficiency

| d_complex | proj params | val |
|---:|---:|---:|
| **768** | **589,824** | **4.7152** |
| 1024 | 786,432 | 4.7552 |
| 1536 | 1,179,648 | 4.7405 |
| 2048 | 1,572,864 | 4.7488 |
| 3072 | 2,359,296 | 4.7570 |
| 4096 | 3,145,728 | 4.7600 |

| | best setting | params | val |
|---|---|---:|---:|
| one-hot | pos_dim 24 | 2,359,296 | 4.8074 |
| **wave** | **d_complex 768** | **589,824** | **4.7152** |

**−0.0922 nats using 4× fewer parameters (6.6 sd).** And wave@768 beats
`rp`@4096 by 0.046 (3.3 sd): random projection buys spread but still needs 4,096
dimensions; the phase construction reaches it in 1,536. The two frontiers do not
overlap — one-hot's best is worse than wave's worst.

Note that the *ordering within* the wave sweep is not resolved (Finding 13); the
claim is the frontier's position, not its shape.

---

## Limits

- **Three seeds, one pair.** Seed variance is measured for one-hot and wave/l2
  at `pos_dim=16` only. The ablation and efficiency arms are single-seed and are
  judged against that noise estimate, which assumes it transfers.
- **One scale, one budget.** 11M body, 98M tokens, no convergence — all curves
  were still descending at 3,000 steps.
- **One language pair.** English + Hindi. Malayalam, the worst-hit script, is in
  the vocabulary but not the corpus.
- **One metric.** Validation loss is the axis most favourable to a dense table.
- **`rp` is a diagnostic, not a proposal.** It inherits every collision the grid
  has; it is in the design to isolate spread, not to be deployed.

---

## Freeze declaration — immutable entering M3

**FROZEN (code).** Everything frozen at M1 and M2, plus:

- `src/kronecker_v2/codecs/ablations.py` — BagOfBytes, RandomProjectedOneHot
- `src/kronecker_v2/codecs/baselines.py` — hash / ALBERT / dense with the
  matched-budget solvers
- `src/kronecker_v2/eval/bpb.py`
- `experiments/m3_script_analysis.py`, `m4_build_tables.py`,
  `m4_dose_analysis.py`, `m5_build_tables.py`, `m5_train.py`, `m5_bench.py`,
  `m5_figures.py`, `status.py`
- `tests/` — all 23

**FROZEN (constants).**

- treatment set: the 903 tokens merged at `pos_dim=16`
- control set: cropped at 16, distinct at 12 — 299 tokens, untreated everywhere
- pairing rule `d_complex = 128 × pos_dim` (for matched comparison only; the
  efficiency sweep deliberately departs from it)
- seeds 1337 / 1338 / 1339 with data_seeds 42 / 43 / 44
- microbatch 4 × grad_accum 8 = 32,768 tokens/step

**FROZEN (results).** `results/m4/*`, `results/m5/*`, every `manifest.json`,
`log.csv`, `per_token_loss.npy`, `dose_response.csv`, all `console_*.log`, and
`figures/fig5..fig7`.

**MUTABLE for M3.** Model size, token budget, `configs/m3_*.yaml`, the
three-language corpus (`data/m3/`), and a learned-dense-projection arm — which
is the experiment Finding 14 makes necessary.

---

## Engineering lessons added by M4/M5

1. **A control can be silently treated.** Defining it at the reference window
   rather than the narrowest one made it two-thirds treated at the highest dose,
   and the failure understated our own effect. The `control_merged` column now
   proves the control's warrant instead of assuming it.
2. **Measure your noise floor before interpreting fine structure.** Two claims
   died at 1.5σ and 0.9σ that a single seed had made look real.
3. **`load_config` merges shallowly.** A partial `train:` block silently drops
   `batch_size`, `lr` and `max_steps`. Found by reading the frozen trainer, not
   by running it.
4. **Extend frozen code by patching at import, not by editing.** `m5_train.py`
   installs a wider `build_wte` at import time; the frozen file on disk is
   untouched and every other code path is literally the code M2 ran.
5. **Check the framing before building on it.** "The one-hot code is sparse" was
   wrong — z-norm makes it dense. The right measure is the participation ratio
   (~15 of 4,096), and finding that out sharpened the ablation rather than
   invalidating it.

---

## Reproduce

```bash
python experiments/m4_build_tables.py --pos-dims 12 24 32
for c in m4_p12_onehot m4_p12_wave m4_p24_onehot m4_p24_wave m4_p32_onehot m4_p32_wave; do
  python experiments/m2_tiny_train.py --config configs/$c.yaml; done
python experiments/m5_build_tables.py
for c in m5_s1338_onehot m5_s1338_wave m5_s1339_onehot m5_s1339_wave; do
  python experiments/m2_tiny_train.py --config configs/$c.yaml; done
for c in m5_rp m5_bag m5_wave1024 m5_wave768; do
  python experiments/m5_train.py --config configs/$c.yaml; done
python experiments/m4_dose_analysis.py
python experiments/m5_figures.py
```

**Action:** `git add -A && git commit -m "M4/M5 closed: findings frozen" && git tag m5-closed`
