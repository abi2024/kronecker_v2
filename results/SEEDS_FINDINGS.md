# Seed Completion — Findings (FROZEN)

**Status:** CLOSED · `git tag seeds-closed`
**Cost:** 4 runs (~4.8 h; one rerun after a degraded-GPU night). With these,
every core contrast in the project is a 3-seed paired measurement.

### Finding 30 — the d384 efficiency claim, now seeded and corrected

wave768 across seeds 1337/1338/1339: 4.7152 / 4.7257 / 4.7480
(mean 4.7296 ± 0.0168). Paired against one-hot@16:

**−0.0860 ± 0.0069 SE, t = 12.4 (2 df).**

The **winner's curse is now measured, not just flagged**: the original n=1
value (4.7152) was the minimum of a six-point sweep, and the 3-seed mean sits
**+0.0144 above it** — inside the 0.012–0.024 optimism predicted from
E[max-of-6] under the seed noise, before these runs existed. Against one-hot's
own best setting the corrected gap is **−0.0778 (~4.4 sd)**, replacing the
biased −0.0922 (6.6 sd). A quantitative prediction about our own selection
bias, confirmed.

### Finding 31 — the d512 baseline is 4× noisier than d384's

one-hot at d512 across seeds: 4.3476 / 4.3037 / 4.3179 —
**sd 0.0224**, against 0.0053 at d384. Baseline noise grows with scale far
faster than the treated arms' (wave768 0.0117, albert 0.0172). Paired
contrasts absorb it:

| contrast (d512, 3 seeds, paired) | mean ± SE | t |
|---|---:|---:|
| wave768 − one-hot | −0.0916 ± 0.0154 | 6.0 |
| albert − one-hot | −0.0907 ± 0.0045 | 20.1 |
| wave768 − albert | −0.0009 ± 0.0142 | 0.1 (tie, unchanged) |

Every prior claim judged against the assumed ~0.021 floor stands; claims near
it (hash-vs-onehot) stay unresolved with more reason than before.

### Finding 32 — the machine-noise bound, learned by accident

A same-seed, same-data-order rerun of `m5_s1339_wave768` (after the original
ran on a degraded, down-clocked GPU and was deleted) landed at **4.7480 vs the
original 4.7553 — |Δ| = 0.0073.** The determinism discipline pins
initialization, data order, and code tables (verified by hash); it does **not**
pin CUDA training trajectories, whose kernel reduction order is
non-deterministic and compounds over 3,000 steps. Consequences, stated
precisely:

- Same-machine, same-seed reruns vary by ~0.007. **No inference in this
  project ever used a floor that small** — all comparisons use the cross-seed
  floors (0.0128–0.0224), which contain it.
- "Bit-identical" claims are scoped to what is actually bit-identical: tables,
  codecs, initialization, batch order — and the one-hot collision receipt
  (identical embeddings ⇒ identical forwards *within* a single process).
- The prior expectation of near-exact reproduction of a training run was an
  overreach; this sheet replaces it with the measured bound.

## Freeze

`results/m5_seeds/*`, the two `m3_seeds` one-hot runs, `configs/m5_s133*_wave768.yaml`,
`configs/m3_s133*_onehot.yaml`, and the statements above.
The deleted degraded-GPU run survives only as its console log
(`console_m5_s1339_wave768.log` prior to relaunch) and the 4.7553 figure
recorded here.

## README v4 amendments (apply as a batch)

1. Abstract: "beating the grid's best setting at 4× fewer embedding
   parameters" → "beating the grid's best setting by 0.078 (~4.4 sd) at 4×
   fewer embedding parameters (3-seed, winner's-curse-corrected)".
2. Results §5 first paragraph: replace the d512 sentence with the paired
   3-seed forms: wave768 −0.0916 ± 0.0154 and the tie −0.0009 ± 0.0142; add
   the d384 paired headline −0.0860 ± 0.0069 (t = 12.4).
3. Method, seed sentence: "n=3 at both scales (d384 ±0.0053–0.0168; d512
   ±0.0117–0.0224)".
4. Negative results: add "The original d384 efficiency point overstated its
   gap by +0.014 — the winner's-curse bias predicted from the sweep design,
   then measured (Finding 30)."
5. Limitations: replace "Single seed on most individual arms (three seeds on
   the four core contrasts)" with "Three seeds on all five core contrasts;
   single seed on secondary arms" and add "same-seed CUDA reruns vary by
   ~0.007; all inference uses the larger cross-seed floors."

**Action:** `git add -A && git commit -m "seed completion: all core contrasts at n=3; winner's curse measured" && git tag seeds-closed`
