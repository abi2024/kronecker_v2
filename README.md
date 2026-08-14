# Kronecker V2 — Removing the Byte-Window Limit

**Assignment problem solved: #3** — *"Today Kronecker is limiting to presenting 32 positions for every word… That's a waste of space. How can it be dynamic and not force us to crop a word?"*

> **One sentence.** The one-hot byte×position grid does not merely waste space — it imposes a hidden design constraint on the tokenizer (*no token may exceed `pos_dim` bytes*), and where that constraint is violated it merges distinct words into a single vector permanently; this project measures the damage on the real 131,072-token vocabulary and replaces the grid with a phase-bound Fourier code that has no position ceiling, at identical parameter count.

| | |
|---|---|
| **Headline result** | At equal trainable parameters, the phase code beats the one-hot grid by **1.24%** validation loss (4.7532 vs 4.8129) |
| **Mechanism, isolated** | On positions following a collided token, the phase code gains **0.355 ± 0.008 nats** over one-hot — a difference-in-differences at **45σ** |
| **Honest magnitude** | Those positions are 1% of the stream, so the mechanism explains **6.3%** of the aggregate gain |
| **Compute** | One RTX 3060. ~5 GPU-hours for the headline grid. No frontier hardware anywhere in this repo. |

Detailed evidence: [`results/M1_FINDINGS.md`](results/M1_FINDINGS.md) · [`results/M2_FINDINGS.md`](results/M2_FINDINGS.md)

---

## 1. Why this is one problem and not two

Problem 4 asks for a Fourier alternative — *"represent each character like a Fourier wave, and just add them to make a word."*

Taken literally that fails on the first example you try. Addition is commutative, so `dog` and `god` produce the identical vector, and so does every anagram. To make character waves work at all, order has to enter through something other than the sum — and the natural choice is to rotate each character's wave by its position.

Rotation has no ceiling. Position 47 is simply "rotate 47 times as far." **The fix that makes Problem 4 work is the fix that solves Problem 3.** So this is a single submission answering Problem 3, whose method happens to be the construction Problem 4 gestures at. Nothing is mixed; the two questions have one answer.

---

## 2. What the 32-byte window actually costs

The Kronecker codec marks each of a token's UTF-8 bytes on a 256 × `pos_dim` grid and flattens it. Bytes past `pos_dim` are dropped. Two tokens sharing that prefix therefore produce **byte-identical codes**, and since the codec is frozen and the projection is shared, no amount of training can separate them. This is not "close in embedding space" — it is the same vector, forever.

### Finding 1 — the window is a tokenizer design constraint, not a free parameter

Measured on the production vocabulary. No training, no GPU, ~2 minutes:

| pos_dim | collision groups | collided tokens | % of vocab | % truncated |
|--------:|-----------------:|----------------:|-----------:|------------:|
| 12 | 1,877 | 5,335 | 4.070% | 5.015% |
| **16** | **380** | **903** | **0.689%** | **1.169%** |
| 20 | 131 | 303 | 0.231% | 0.542% |
| 24 | 39 | 93 | 0.071% | 0.220% |
| 32 | 0 | 0 | 0.000% | 0.000% |

Zero at 32 is **not a null result**. BrahmicTokenizer-131K's own audit reports `tokens > 32 bytes: 0` — the vocabulary was *built* to satisfy the window, and we reproduce that independently from `tokenizer.json` (max observed byte length: exactly 32).

So the constraint never disappeared. It was **paid for upstream, in the tokenizer**, as a restriction on which merges the vocabulary is allowed to learn — and it appears nowhere in the embedding's parameter accounting.

The bolded row is the setting the reference paper's own 124M experiments use (`pos_dim=16`, `D = 256 × 16 = 4096`). There, the constraint is violated.

### Finding 2 — the cost falls almost entirely on Indic scripts

At `pos_dim = 16`:

| script | collided | of | % of own script |
|---|---:|---:|---:|
| Malayalam | 182 | 1,797 | **10.13%** |
| Devanagari | 253 | 4,020 | 6.29% |
| Tamil | 60 | 1,054 | 5.69% |
| Telugu | 78 | 1,402 | 5.56% |
| Kannada | 77 | 1,422 | 5.41% |
| Bengali | 101 | 2,166 | 4.66% |
| **Latin** | **18** | **108,185** | **0.02%** |

UTF-8 charges 1 byte per Latin character and **3 per Indic character**; a conjunct such as क्ष is three codepoints, so nine bytes for one visual symbol. The same window is 32 English characters or ~10 Indic ones. Two orders of magnitude of asymmetry from one shared constant.

Representative collision groups — three unrelated words, one vector:

```
कार्य (work) / कार्यक्रम (programme) / कार्यालय (office)
अधिकार (right) / अधिकारी (officer) / अधिकारियों (officers)
संस्क / संस्कृति (culture) / संस्करण (edition)
```

### Finding 3 — a second collision mechanism, independent of the window

The reference `token_id_to_bytes` resolves ids through `tokenizer.decode()`, which renders every partial-codepoint byte-fallback token as U+FFFD. Those tokens collapse together at **any** `pos_dim`: **1,151 tokens still collide at `pos_dim = 32`**, where truncation contributes nothing (658 of them share the byte string `b'\xef\xbf\xbd'`). This is a defect in the byte-extraction path, not the codec; reading the byte-level BPE piece directly removes it. The audit reports both sources so the two are never conflated.

---

## 3. The proposed fix: a phase-bound Fourier byte code

Replace the question *"which of `pos_dim` slots?"* with *"how much rotation?"*

1. Every byte **value** (0–255) gets a fixed random vector of angles — a phasor.
2. Byte **position** `p` enters as a rotation: the position phasor raised to the power `p` (fractional power encoding), i.e. angles multiplied by `p`.
3. **Bind** value to position by adding angles (complex multiplication), **bundle** across the token's bytes by summing, then normalise.

Since `p` is an exponent rather than an index, there is no last slot to fall off. Long tokens degrade gracefully through phasor crosstalk instead of being cropped. Nothing about this is learned — the codec is frozen, exactly as in V1; only the shared projection trains.

**Parameter parity by construction.** 2048 complex dimensions = 4096 real dimensions = the one-hot code width at `pos_dim = 16`. Both arms feed an identical `Linear(4096 → d_model)`. This is asserted by [`tests/test_equal_params.py`](tests/test_equal_params.py), not claimed in prose — any difference measured later cannot be explained by one arm being larger.

Prior art this rests on: Plate's Holographic Reduced Representations (binding by circular convolution = phase addition in the Fourier domain) and fractional power encoding, implemented over `torchhd`. The novelty claimed here is narrow and specific: *a phase-bound Fourier byte code as the token-embedding replacement inside a transformer LM.*

---

## 4. How the solution is proved

Three layers of evidence, each falsifiable, each with its claim written down **before** the run.

### Layer 1 — measurement (no training)

```bash
python experiments/m1_collision_audit.py --byte-source both
python experiments/m1_separation_demo.py
```

The tables in §2, plus a direct demonstration that the phase code separates what the grid merges — at identical width:

| pair | one-hot @16 | wave @2048 |
|---|---|---:|
| कार्य / कार्यक्रम | **IDENTICAL** | 0.7609 |
| कार्य / कार्यालय | **IDENTICAL** | 0.7952 |
| सरकार / सरकारी | **IDENTICAL** | 0.9130 |
| अधिकार / अधिकारी | **IDENTICAL** | 0.9293 |

Separated but still **near** — orthographic locality, the property that made Kronecker work in the first place, is preserved. A codec that pushed these to zero similarity would have destroyed the thing worth keeping.

### Layer 2 — a controlled training comparison

Five arms. **Only the embedding module differs.** Same tokenizer, same data, same body, same schedule, same batch order.

| | |
|---|---|
| model | 6 layer, 6 head, d_model 384 (11.0M body + 50.3M lm_head) |
| data | 98M train / 2M val tokens, ~50% English (FineWeb-Edu) + ~50% Hindi (Sangraha) |
| schedule | 3,000 steps × 32,768 tokens, AdamW, cosine, bf16 |

**Controls verified, not assumed:** all five arms report `body_state_hash = 4a6392274148` — every parameter outside the embedding initialised bit-identically — and all five drew batches from the same RNG stream.

![Grid](figures/fig1_grid.png)

| arm | final val loss | vs baseline | embedding params |
|---|---:|---:|---:|
| dense table *(reference, **not** matched)* | 4.5592 | −5.27% | 50,331,648 |
| **wave / l2** | **4.7532** | **−1.24%** | 1,572,864 |
| one-hot @16 *(baseline)* | 4.8129 | — | 1,572,864 |
| wave / sqrt_len | 4.8612 | +1.00% | 1,572,864 |
| wave / znorm | 4.9023 | +1.86% | 1,572,864 |

Pre-registered claim — *"best wave arm within 5% of one-hot"* — **passed, with the sign negative.**

Two things the right-hand panel shows that the final number hides. The phase code starts **worse**, crosses over near step 1500, and the gap is **still widening at cutoff** — so the result is budget-dependent and a shorter run would have shown it losing. Both stated here rather than left for a reviewer to find.

### Layer 3 — mechanism, not just scoreboard

An aggregate cannot distinguish "slightly better everywhere" from "identical everywhere except one slice." Re-scoring the finished checkpoints separates them — no retraining, only re-reading the answers with the questions sorted.

![Decomposition](figures/fig2_decomposition.png)

**The advantage tracks frequency, not byte length.** Frequent tokens are short tokens, so these are confounded — and the distinction matters, because this project's thesis is about length. Sorted by frequency the gap varies and crosses zero (−0.084 / −0.031 / +0.021); sorted by byte length it is flat (−0.054 / −0.064 / −0.060 / −0.055 / −0.040). A prior prediction that the dense table's lead would shrink in the tail was **falsified**.

![Which side](figures/fig3_which_side.png)

**Measuring the right side of the model.** Bucketing by the *target* token showed no collision effect at all. That contradiction located an error in the measurement: the `lm_head` is untied and dense, so every token keeps a private output row and remains perfectly scoreable *as an answer*. A collision corrupts a token *as context* — two sequences containing कार्य and कार्यक्रम produce identical hidden states, so what degrades is the prediction of **whatever follows**. Re-bucketing by the preceding input token, the effect is **11× larger**.

![Attribution](figures/fig4_attribution.png)

**Attribution by difference-in-differences.** Positions after collided tokens are also after *long*, *Indic*, *common* tokens — so the raw −0.368 cannot be attributed. The control group is long Indic tokens that one-hot also crops, but whose truncated form is **unique** (e.g. दिल्ली 19 B → ' दिल्ल'; 887 such tokens). Subtracting it cancels everything the groups share:

| arm | after collided | after control | difference | σ |
|---|---:|---:|---:|---:|
| dense | −0.5121 | −0.1532 | −0.3589 | 41 |
| **wave / l2** | −0.3680 | −0.0134 | **−0.3546** | **45** |
| wave / sqrt_len | −0.3374 | +0.0137 | −0.3511 | 49 |
| wave / znorm | −0.3029 | +0.0480 | −0.3509 | 48 |

**The reported effect is −0.355, not −0.368.** The first is attributable; the second is merely observed.

The obvious objection — *"collided tokens are mostly Indic, maybe the codec is just better at Indic"* — is answered inside the data. `sqrt_len` and `znorm` **lose** to one-hot overall and on the control bucket, which is also long and Indic, yet win by −0.337 and −0.303 after collided tokens. All four arms land within 0.008 of each other on the difference-in-differences while their aggregate scores span 0.15 nats. **The collision effect belongs to the codec; the aggregate belongs to the normalization.** Two independent dials, cleanly separated.

And the cleanest single sentence in the project:

| | after collided | after control | |
|---|---:|---:|---|
| dense | 3.3531 | 3.6130 | −0.260 **easier** |
| wave / l2 | 3.4973 | 3.7528 | −0.256 **easier** |
| **one-hot** | **3.8652** | 3.7662 | **+0.099 harder** |

Collided tokens are common words, so what follows them is predictable. Every model that *can* tell them apart finds those positions easier. One-hot alone finds them **harder**.

---

## 5. What is claimed, and what is not

**Claimed.**
- At `pos_dim = 16`, 903 tokens of the production vocabulary receive permanently identical one-hot codes, ~98% of them Indic.
- The 32-byte window is satisfied only because the tokenizer was co-designed to satisfy it.
- A phase-bound Fourier code removes the ceiling and, at identical parameter count, reduces validation loss by 1.24%.
- Collisions cost the one-hot codec 0.355 nats on affected positions; the phase code eliminates that cost; because such positions are 1% of the stream, this explains 6.3% of the aggregate gain.

**Not claimed.**
- That this beats a dense table. **It does not** — dense is 4.08% better, using **32×** the embedding parameters. The trade is a 32× parameter reduction for ~4% loss.
- That it holds at frontier scale. One model size, one token budget, one seed per arm, one language pair, one metric.
- That validation loss is the right axis. It is the axis most favourable to a dense table; typo robustness, unseen-token handling and vocabulary-independent cost are invisible to it and untested here.

A note on reproducibility discovered the hard way: `torch.rand` is not portable across platforms, and `cos`/`sin` differ in their last bits between MSVC and glibc. The codec's phase tables are therefore generated from NumPy's PCG64 and pinned **exactly**; the resulting codes are pinned **by value at 1e-4**. Develop on one OS and train on another without this, and the trained model uses a different codec than the one audited — silently.

---

## 6. Reproduce

```bash
python -m venv .venv && source .venv/bin/activate     # Scripts/activate on Windows
pip install -e .
pytest -q                                              # 16 contract tests

# Layer 1 — no GPU, ~2 minutes
python experiments/m1_collision_audit.py --byte-source both
python experiments/m1_separation_demo.py

# Layer 2 — ~5 GPU-hours total on one RTX 3060
python experiments/m2_build_tables.py
python experiments/m2_prepare_data.py --hf --tokens 100_000_000
for a in onehot wave_sqrtlen wave_l2 wave_znorm dense; do
  python experiments/m2_tiny_train.py --config configs/m2_$a.yaml
done
python experiments/m2_report.py

# Layer 3 — no training; first call scores and caches, the rest are instant
python experiments/m2_bucket_analysis.py --bucket-by prev-collision
for b in collision length frequency; do
  python experiments/m2_bucket_analysis.py --bucket-by $b
done
python experiments/m2_figures.py                       # regenerates every figure
```

Every figure in this README is produced by `m2_figures.py` from the CSVs on disk. None is hand-edited.

---

## 7. Repository map

```
src/kronecker_v2/
  codecs/base.py        Codec protocol + OneHotCodec (adapter over the published
                        reference implementation — the baseline is never reimplemented)
  codecs/wave.py        the phase-bound Fourier codec (frozen, PCG64-seeded)
  vocab.py              token id -> UTF-8 bytes, raw and decode paths, script tagging
  collisions.py         the audit: exact collisions per script, UTF-8-safe truncation
  capacity.py           analytic FHRR overlap curve
  embedding.py          frozen code buffer + one trainable projection
  model.py              nanoGPT (MIT, attributed) with a pluggable wte — the only change
  tables.py             vectorised table builders, self-verified against the frozen codecs
experiments/            one runner per milestone; thin, all logic imported from src/
configs/                one YAML per arm; arms differ by the `codec:` block alone
tests/                  16 contract tests: parameter parity, determinism, shapes
results/                findings sheets, CSVs, run ledger, per-run manifests
figures/                regenerated, never hand-edited
```

Three rules the repo enforces mechanically: only the embedding changes between arms; parameter parity is a test; every run writes a `manifest.json` with its git SHA, seed, codec fingerprint and body-init hash.

---

## 8. Status and roadmap

| phase | question it settles | status |
|---|---|---|
| M1 | how much does the byte window actually cost, on the real vocabulary? | **closed** · `m1-closed` |
| M2 | does removing the ceiling help, at equal parameters — and through what mechanism? | **closed** · `m2-closed` |
| M3 | does it survive scale, and beat the baselines a reviewer will demand? | in progress |
| M4 | is the collision mechanism **causal**? | designed, ~1 night |
| M5 | does the co-design claim generalise beyond one tokenizer? | designed, ~1 night |
| M6 | does the vocabulary-independence advantage show up where it matters? | designed, ~1 night |
| M7 | does it replicate at the reference paper's own scale and protocol? | needs rented compute |

What follows is a proposal, not a promise. Each phase states the question, the
experiment, a **falsifiable prediction written before the run**, and what a
negative result would mean. Phases M4–M6 each fit in a single overnight session
on one RTX 3060, because they run at M2's measured ~1.0 h/arm.

---

### M3 — Scale and the baselines reviewers demand *(in progress)*

**Question.** M2 established the result at 11M body parameters on 98M tokens.
Does it hold at ~4× the body and 5× the data, and does it beat the two
parameter-matched baselines a referee will ask about?

**Experiment.** d_model 512, 12 layers (37.7M body), 500M tokens, five arms:
one-hot · wave/l2 · dense · **hash embeddings** · **ALBERT factorization**.
Plus **per-script bits-per-byte** — the first direct measurement of the Indic
claim, rather than inference from the audit. ~8.3 h/arm, ~42 h total.

**Prediction.** wave/l2 ≤ one-hot at the larger scale, and the crossover point
arrives *earlier* in token terms than M2's step 1500.

**Already worth reporting from the design alone.** At V = 131,072, a
parameter-matched ALBERT factorization gets a **rank-16 bottleneck per token**,
because ALBERT's cost is `V·r`. The codec's cost contains no `V` at all. That
asymmetry is the structural argument this whole line of work rests on, and M6
tests it directly.

---

### M4 — Dose-response: turning a mechanism into a cause

**This is the highest-value experiment remaining, and it costs one night.**

**Question.** M2 showed that positions following a collided token are 0.355 nats
worse under one-hot, isolated by difference-in-differences at 45σ. That is
strong evidence of association. It is not yet proof of cause, because the
comparison sits at a single `pos_dim`.

**Experiment.** `pos_dim` is a **dose dial with a known dose at every setting**:

| pos_dim | code width D | matched d_complex | collision groups | tokens merged |
|--------:|-------------:|------------------:|-----------------:|--------------:|
| 12 | 3,072 | 1,536 | 1,877 | **5,335** |
| 16 | 4,096 | 2,048 | 380 | **903** |
| 24 | 6,144 | 3,072 | 39 | **93** |
| 32 | 8,192 | 4,096 | 0 | **0** |

Train the matched pair (one-hot, wave/l2) at each setting — 8 runs at M2 scale,
~8 h — and plot the difference-in-differences against the collision count.

**Prediction.** The isolated effect scales monotonically with the number of
collided tokens, and **vanishes at `pos_dim = 32`**, where the tokenizer
guarantees zero collisions.

That last row is the part that matters. It is a **built-in placebo**: a dose of
zero, produced not by us but by the tokenizer's own design constraint. If the
effect persists at 32 where there is nothing to fix, the mechanism story is
wrong and M2's result is something else wearing its clothes. If it disappears
exactly where the collisions do, the causal claim is made on a dose-response
curve rather than a single contrast — which is the difference between "we
measured a correlation" and "we can turn the effect on and off."

**If it fails.** The paper loses the causal claim but keeps the audit, the
matched-parameter win, and the difference-in-differences as association. We
would then have to explain what else changes with `pos_dim` — and that question
is worth a section either way.

---

### M5 — Does the co-design claim generalise?

**Question.** M1's central reframe is that the 32-byte window is satisfied only
because BrahmicTokenizer-131K was *built* to satisfy it. That claim currently
rests on one tokenizer. If it is a general property of the codec, it should be
visible in any vocabulary nobody constrained.

**Experiment.** Audit off-the-shelf tokenizers that were never designed around a
byte window — Llama-3, Gemma, mT5 — and count collisions at `pos_dim` 16 and 32.
Then retokenize a slice of the same corpus and train the matched pair on the one
with the most collisions. 6 runs at M2 scale, ~6 h plus data prep.

**Prediction.** Unconstrained tokenizers show **non-zero collisions even at
`pos_dim = 32`**, because nothing stopped them learning long merges — and the
wave arm's advantage there is *larger* than on the co-designed vocabulary.

**Why this matters to the paper.** It converts a finding about one artifact into
a property of the method: *the one-hot Kronecker codec cannot be dropped into an
arbitrary tokenizer without an audit.* That is a claim a practitioner can act
on, and it is exactly the kind of constraint that should be documented before a
production model is built on top of it.

**If it fails** — if other tokenizers also happen to stay inside 32 bytes — then
the constraint is easier to satisfy than argued, and the honest conclusion is
that the window is a mild rather than a severe restriction. Worth knowing before
anyone scales it.

---

### M6 — Vocabulary independence, tested where it pays

**Question.** The codec's headline structural property is that its parameter
cost contains no `V`. Every measurement so far holds `V` fixed at 131,072, so
that property has never actually been exercised.

**Experiment.** Hold the model fixed and vary the vocabulary — 131K and a
~262K-token vocabulary over the same corpus — for one-hot, wave/l2, dense and
ALBERT. Report loss per byte (not per token, which is not comparable across
vocabularies) against total embedding parameters. 6 runs, ~6 h.

**Prediction.** Dense and ALBERT costs double with `V`; both codecs stay flat.
The interesting quantity is not who wins at 131K but **where the curves cross** —
and whether Indic fertility gains from the larger vocabulary offset the codec's
representational cost.

**Why this matters beyond the paper.** This is the decision the V5 model
actually faces: vocabulary size trades attention compute against embedding
memory, and a codec whose cost is independent of `V` changes where that optimum
sits. Section 4 of the session notes puts the V5 minimum near 101K under a
memory price that assumes a dense table. **If the embedding cost is flat in `V`,
that optimum moves — and this experiment says by how much.**

---

### M7 — Replication at the reference protocol

**Question.** Does any of this survive at the scale the V1 paper reports?

**Experiment.** The V1 protocol exactly — GPT-2 124M, 2.5B tokens, **3 seeds**,
five arms — but multilingual rather than English-only, plus the typo-robustness
probe the reference repo already ships (110 prompt pairs, top-1-preserved rate).

**Cost.** Not local. On one RTX 3060 this is roughly 4–6 weeks of continuous
compute; on rented A100 time it is on the order of $100–200 and a few days.
This is the one phase that needs a budget decision rather than a night.

**Prediction.** The aggregate gap narrows with scale (the dense table's
advantage is a capacity argument, and capacity matters less as the body grows),
while the collision effect **persists undiminished**, because it is a property
of the input representation rather than of model size.

Three seeds is what turns M2's single-seed direction into a significance claim
on the aggregate. The bucket-level results are already significant — paired
standard errors over ~2M positions — but the headline number is not, and the
paper should not pretend otherwise until this runs.

---

### Open questions these phases do **not** answer

Stated so they are not mistaken for oversights.

- **Why does the advantage track frequency rather than byte length?** M2
  established the fact (flat across 1–4 → 17+ bytes; varying and sign-changing
  across frequency bands) and no mechanism explains it. The natural next probe
  is representational: measure the rank and pairwise geometry of each codec's
  code matrix restricted to frequency bands. Cheap, and currently unplanned.
- **Where is the phasor-crosstalk ceiling?** VSA theory says bundling `M`
  components into `D` dimensions degrades retrieval as `√(D/M)`. Long tokens
  bundle more bytes. M2 hints at this — the wave advantage halves on
  cropped-but-distinct long tokens — but the capacity curve has not been
  measured against the analytic bound in `capacity.py`.
- **Does the codec help generation, not just likelihood?** Nothing here samples
  from the models. Loss is the only axis, and it is the axis most favourable to
  a dense table.
- **Is `l2` the right normalization at scale?** It was selected on a single
  ablation at 11M parameters. M3 re-tests it implicitly; nothing re-tests it
  deliberately.

---

### What would change the conclusion

The result stops being interesting if any of these hold:

1. The dose-response is flat, or the effect survives at `pos_dim = 32` where no
   collisions exist (M4).
2. The advantage vanishes at 38M parameters, i.e. it was a small-model artifact
   (M3).
3. Hash embeddings or ALBERT match the codec at equal parameters, making the
   phase construction unnecessary (M3).
4. Other tokenizers are all naturally inside the window, making the co-design
   constraint a curiosity rather than a limitation (M5).

Each of those is a run that has been designed, not a hypothetical. The point of
listing them is that the project is currently structured so that a negative
result is publishable rather than fatal — the audit stands regardless, and the
mechanism is measured against a control in every case.