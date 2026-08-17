# Milestone 3 — Findings (FROZEN)

**Status:** CLOSED · `git tag m3-closed`
**Cost:** 7 arms × 3,000 steps × 98.3M tokens at d_model 512 (~1.4 h each,
dense 2.4 h), on one RTX 3060. Data: 539M train + 11M val tokens,
realised mix eng 49.5% / hin 26.1% / mal 24.2% (requested 40/30/30 — English
documents yield more tokens each; the realised figures are the ones reported).

**Two questions were registered before the runs:**

> **Q1.** Does `wave768` still beat `onehot` at 3.6× the body size?
> **Q2.** Do the per-script gaps order **Malayalam > Devanagari > Latin**,
> matching the collision rates from the M1 audit (10.13% / 6.29% / 0.02%)?

Both passed. A third result nobody designed turned out to be the strongest of
the milestone (Finding 18).

---

## Setup

d_model 512, 12 layers — 37.7M body (3.6× M2's) + 67.1M lm_head — on **M2's
exact schedule** (3,000 steps × 32,768 tokens), so model size is the only
variable against M2. Matched embedding budget at d512 is 2,097,152 params
(`Linear(4096→512)`); ALBERT lands at rank 16, hash at 3,584 buckets.
All seven arms share one `body_state_hash`. Single seed (1337); contrasts are
judged against the d384 noise floor (±0.0139 unpaired) on the assumption it
transfers.

## Result

| arm | emb params | final val | vs one-hot |
|---|---:|---:|---:|
| dense *(reference, not matched)* | 67,108,864 | 4.1225 | −5.18% |
| **wave768** | **786,432** | **4.2339** | **−2.62%** |
| albert (rank 16) | 2,105,344 | 4.2485 | −2.28% |
| rp @4096 | 2,097,152 | 4.2620 | −1.97% |
| wave @4096 | 2,097,152 | 4.2646 | −1.91% |
| one-hot @16 *(baseline)* | 2,097,152 | 4.3476 | — |
| hash (3,584 buckets) | 2,097,152 | 4.3697 | **+0.51%** |

### Finding 16 — Q1: the advantage grows with scale (registered, PASS)

`wave768` beats one-hot by **−0.1137** with **2.7× fewer embedding parameters**
than the baseline it beats. Every codec gap *grew* from M2 to M3 — matched wave
−1.39% → −1.91%, rp −1.13% → −1.97%, wave768 −2.08% → −2.62% — while dense's
lead stayed flat (−5.3% → −5.2%). And `wave768 > wave@4096` is now replicated at
two independent scales (−0.0336 at d384, −0.0307 at d512): what was borderline
at one scale is a pattern at two.

### Finding 17 — Q2: the script ladder (registered, PASS)

Relative bits-per-byte gap vs one-hot, against each script's collision rate:

| | Latin (0.02% collide) | Devanagari (6.29%) | Malayalam (10.13%) |
|---|---:|---:|---:|
| wave768 | −0.06% | −3.92% | **−6.88%** |
| wave @4096 | +0.48% | −2.85% | −5.97% |
| rp | +0.36% | −3.35% | −5.66% |
| albert | +0.41% | −3.46% | −6.82% |
| hash | +4.52% | −2.02% | −5.87% |
| dense | −3.20% | −5.96% | −8.32% |

**The registered ordering holds in every one of the six arms.** The honest
wrinkle is the first column: at matched width the frozen codecs are slightly
*worse than the grid on Latin* — the aggregate win is an Indic story. The
narrow `wave768` is the exception: it breaks even on Latin (−0.06%) while
keeping the largest Indic gains of any matched-budget arm.

### Finding 18 — the natural experiment: the rescue follows information, not format

Not designed; it fell out of arms run for other reasons. Input-side collision
rescue (gap after a collided token minus gap after the length-matched control):

| arm | rescue | can it distinguish the collided tokens? |
|---|---:|---|
| hash | −0.4802 | yes (its buckets break byte-prefix ties) |
| wave | −0.4786 | yes |
| wave768 | −0.4761 | yes |
| albert | −0.4690 | yes (per-token factors) |
| dense | −0.4556 | yes (per-token rows) |
| **rp** | **−0.0439** | **no — it carries one-hot's exact code, densified** |

Five architectures with nothing in common except the *ability* to tell the 903
tokens apart all rescue ~0.47 nats. The one arm that provably cannot — same
spread, same geometry class as wave, identical information to one-hot — rescues
a tenth of that. **The collision penalty is bound to the information in the
code, not to its format.** This is the sharpest single confirmation of the
mechanism in the project, and it also resolves M5's puzzle: rp matches wave on
the aggregate because rp is slightly better on the 96% of ordinary positions
while wave is ~0.43 nats better on the 2.17% of collision contexts — and the
two nearly cancel.

(Collision contexts are 2.17% of this corpus, up from M2's 1.00%, because the
Malayalam-heavy mixture hits collided tokens more often.)

### Finding 19 — ALBERT is the strongest matched-width arm, and why that's fine

ALBERT at −0.0991 beats both frozen codecs at width 4096. In hindsight it
should: its 2.1M parameters are *per-token learned* factors — sixteen numbers of
each token's own — while the codecs hold zero per-token state. `wave768` still
edges it (−0.0146) but at ~1.1 sd that is **unresolved at one seed**. The
defensible sentence: *at matched budget, a learned rank-16 factorization and the
narrow phase code are statistically indistinguishable as the best embedding* —
and the codec keeps what ALBERT cannot have: cost independent of V (ALBERT's is
`V·r`), compositional codes, no per-token state to store or shard.

### Finding 20 — hash embeddings lose overall while winning on collisions

Hash is the only arm below the baseline (+0.51%). Its random bucketing *does*
break the byte-prefix collisions (rescue −0.48, Finding 18) — but 131,072 tokens
into 3,584 buckets is a ~37-way collision everywhere by construction, and the
learned importance weights don't dig it out. It fixed the disease and infected
the rest of the vocabulary. Useful as a boundary: dense codes help only if they
don't destroy identity to get there.

---

## Limits

- **Single seed at this scale.** All contrasts assume the d384 noise floor
  (±0.0139) transfers to d512. The big gaps (wave768 vs one-hot, ~8 sd under
  that assumption) are safe; wave768-vs-albert (~1.1 sd) is not called.
- **BPB levels remain corpus-confounded across scripts** (Sangraha vs
  FineWeb-Edu); only the within-script *gaps*, paired on identical tokens, are
  interpreted.
- **No convergence** — all curves still descending at 3,000 steps; comparisons
  are budget-dependent as in M2.
- **One tokenizer.** The co-design claim (M6) is still untested beyond
  BrahmicTokenizer-131K.

---

## Freeze declaration — immutable entering M6

**FROZEN (code).** Everything frozen at M1/M2/M5, plus `experiments/m3_train.py`,
`experiments/m3_prepare_data.py`, `experiments/patched.py`, `experiments/status.py`,
and `configs/m3_*.yaml`.

**FROZEN (results).** `results/m3/*` in full — manifests, logs,
`per_token_loss.npy` caches, `buckets_prev-collision*.csv`, `script_bpb*.csv`,
console logs — and `data/m3/meta.json` (the corpus recipe of record).

**FROZEN (constants).** The three-language mixture as realised; the registered
predictions Q1/Q2 and their outcomes as stated here.

**MUTABLE.** M6 (other-tokenizer audit), M7 (124M × 2.5B × 3 seeds, rented
compute), the learned-dense-projection arm, and any multi-seed extension of M3.

## Engineering lesson added by M3

**A patch chain must be installed in every process that needs it.** The frozen
scorer rebuilds each arm's embedding via `T.build_wte`; analysis scripts never
imported `m3_train`, so scoring died on the first learned arm.
`experiments/patched.py` now runs any analysis with the full chain installed —
and rebuild-from-manifest stays correct for learned arms only because hash's
bucket table and rp's Gaussian regenerate from pinned PCG64 seeds before the
checkpoint restores trained weights. The numpy-RNG discipline paid for itself
here.

## Reproduce

```bash
python experiments/m3_prepare_data.py --hf --tokens 550_000_000
for c in m3_onehot m3_wave768 m3_wave m3_rp m3_hash m3_albert m3_dense; do
  python experiments/m3_train.py --config configs/$c.yaml; done
python experiments/patched.py m2_bucket_analysis --bucket-by prev-collision \
    --root results/m3 --data data/m3 --baseline m3_onehot
python experiments/patched.py m3_script_analysis --root results/m3 --data data/m3 --baseline m3_onehot
```

**Action:** `git add -A && git commit -m "M3 closed: findings frozen" && git tag m3-closed`
