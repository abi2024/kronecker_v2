# Stability Day — Findings (FROZEN)

**Status:** CLOSED · `git tag mstab-closed`
**Cost:** 3 arms × 23,000 steps × 753.7M tokens (7.7× every prior budget),
~7.6 h each on one RTX 3060. Registered P1–P3 before launch; **P1 failed, P2
resolved against the codec, P3 passed** — and the failure is the most useful
result the project has produced since the seed floor.

Provenance note: the trainer's last in-log eval is step 22,750; each arm's
step-23,000 final lives in its manifest (onehot 3.6357 / wave768 3.5941 /
tied 3.5130 — consistent with the 22,750 values quoted below).

## Setup

d384 / 6L on the three-language corpus (`data/m3`, 1.4 epochs), one seed
(1337), val every 250 steps → 92 in-log curve points per arm. Third arm is the
previously missing industry baseline: **tied embeddings** — the input embedding
IS the lm_head weight (one shared Parameter, verified at the object level), a
fully learned input representation for ~zero additional total parameters.

## The trajectory (gap to one-hot, nats)

| tokens | onehot val | wave768 gap | tied gap |
|---:|---:|---:|---:|
| 8M | 6.2515 | +0.1565 | −0.0752 |
| 33M | 5.2908 | +0.0202 | −0.1830 |
| 49M | 5.0510 | −0.0325 | −0.2822 |
| 66M | 4.8383 | −0.0773 | **−0.3868** |
| 98M | 4.5101 | **−0.0995** | −0.3454 |
| 197M | 4.0688 | −0.0544 | −0.1724 |
| 393M | 3.8154 | −0.0415 | −0.1319 |
| 590M | 3.6860 | −0.0400 | −0.1245 |
| 745M | 3.6372 | −0.0414 | −0.1225 |

### Finding 26 — the crossover replicates on new data (P3: PASS)

wave768 starts behind and takes the lead between steps 1,000 and 1,500
(33–49M tokens), inside the registered ~2,500 and matching M2's pattern.
Consistency check for the harness: the 98M-token gap here (−0.0995, different
corpus) reproduces every prior d384 wave768-vs-onehot measurement.

### Finding 27 — the codec's edge contracts, then holds (P1: FAIL)

Registered criterion: the 750M-token gap retains ≥50% of the run's own
98M-token gap. **Retention is 41.6% — failed.** But the shape matters as much
as the verdict: the gap contracts −0.0995 → −0.0415 between 98M and 393M
tokens, then is **flat to ±0.0008 for the final 350M** (−0.0415 / −0.0400 /
−0.0414). It does not vanish; it plateaus at ≈ **−0.041**, about 3× the
measured d384 seed sd. Two honest sentences replace the old claim:

- Short-budget comparisons overstate the codec's edge by ~2.4×.
- The surviving long-horizon edge over the grid is real, stable, and modest.

### Finding 28 — tied embeddings win the conventional architecture (P2)

At 745M tokens, tied beats one-hot by **−0.1225** and beats wave768 by
**−0.0805** (~10 and ~6 seed-sd). The consequence registered before launch now
activates as written: **in a conventional untied architecture, the codec is
not the practical choice on aggregate loss**; its claims narrow to the
structural axes (no `V` in the embedding's cost, compositional codes, no
per-token state) plus the script-fairness and collision results.

The architectural reframe, stated precisely: tying wins by borrowing the 50M
lm_head. In an end-to-end codec design with a **free output head — the
consolidated paper's architecture — there is no head to borrow**, and "tied"
is not a definable arm. The decisive comparison in that setting is end-to-end
codec vs embedding+head, which is exactly the decoder workstream. This figure
is therefore an argument *for* that architecture direction, produced by trying
to falsify our own method.

Tied's own gap also peaks early (−0.387 at 66M tokens) and contracts ~3× to a
~−0.12 plateau — the contraction is a property of short-budget comparisons
generally, not of any one arm.

### Finding 29 — the contraction factor, as a portable warning

Every gap-to-baseline in this run compresses 2–3× from its early peak as the
baseline converges. Any embedding comparison made near 1:1 token:parameter
ratios — including proof-of-concept results at ~20M tokens — should expect its
gaps to contract by a factor of this order before drawing headline
conclusions. This is the project's most transferable methodological result.

## Limits

- One seed per arm. The plateau's flatness (within-run sd ~0.0007 over the
  last five evals) and the crossover replication substitute imperfectly for
  seed replicates.
- One scale, one corpus recipe, 1.4 data epochs; the cosine schedule is
  stretched with `max_steps`, so gaps are schedule-conditional.
- The tied arm's `body_state_hash` is not comparable by construction (the
  shared weight registers as `wte.weight`, removing the head from the hash).

## README v3 amendments (apply as a batch; v3 is otherwise current)

1. Step 5, ALBERT line: "a ~1σ call we won't make on one seed" → "resolved by
   two further seeds: a tie (−0.001 ± 0.014), reached at 37% of ALBERT's
   parameter budget."
2. Step 5, hash line: "lose overall" → "trailed in our single-seed run
   (within the measured d512 noise)"; keep the collision-rescue lesson.
3. Efficiency claim (Step 3 and "Claimed" §2): append "— an edge that
   contracts ~2.4× over long training and stabilizes near −0.04, and that a
   tied-embedding baseline beats outright in conventional untied
   architectures (see Stability findings)."
4. "Not claimed": add "That the codec beats weight tying in a conventional
   architecture. It does not (−0.08 at 745M tokens). Its parameter story is
   specific to designs with no output head."
5. Roadmap: M6 row → closed (audit + 12σ mechanism replication on gemma +
   BPB inversion); add row "Stability — long-horizon + tied baseline —
   closed: P1 fail (41.6% retention, plateau −0.041), tied wins conventional
   architecture."
6. "What would change the conclusion" item 1 (multi-seed wave768 ≤ ALBERT):
   mark **happened** — outcome recorded in M6_FINDINGS addendum; likewise the
   tied item now resolved here.

## Freeze declaration

**FROZEN.** `results/mstab/*` (manifests, log.csv ×3, `stability_curves.png`,
console logs), `configs/mstab_*.yaml`, `experiments/mstab_train.py`,
`mstab_bench.py`, `mstab_curves.py`, and predictions P1–P3 with outcomes as
stated. **MUTABLE.** The decoder workstream, seed replicates of the tied arm,
and any end-to-end (free-output-head) comparison.

## Reproduce

```bash
python experiments/mstab_bench.py --config configs/mstab_tied.yaml
for c in mstab_onehot mstab_wave768 mstab_tied; do
  python experiments/mstab_train.py --config configs/$c.yaml; done
python experiments/mstab_curves.py
```

**Action:** `git add -A && git commit -m "Stability day closed: P1 fail, tied wins conventional arch" && git tag mstab-closed`
