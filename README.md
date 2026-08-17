# Kronecker V2 — Removing the Byte-Window Limit

**Assignment problem solved: #3** — *"Today Kronecker is limiting to presenting 32 positions for every word… That's a waste of space. How can it be dynamic and not force us to crop a word?"*

**[→ Interactive walkthrough](https://abi2024.github.io/kronecker_v2/)** — type a word, watch it get cropped, see why anagrams break plain addition.

---

## The problem, in plain words

Kronecker embeddings turn a word into a vector by marking each of its bytes on a grid: *which byte value, in which position*. The grid has a fixed number of position slots — 16 or 32 — and **bytes past the last slot are simply thrown away**.

Here is the part that isn't obvious: if two different words share the same first 16 bytes, they get **the exact same vector**. Not similar — identical. And because this codec is frozen (nothing about it is learned), no amount of training can ever pull them apart. The model literally cannot tell कार्य (*work*), कार्यक्रम (*programme*) and कार्यालय (*office*) apart at the input. They wear the same name tag forever.

**Who pays?** Byte counting is unfair across languages. One English letter costs 1 byte; one Devanagari or Malayalam character costs 3, and a conjunct like क्ष costs 9. So a 16-byte window is ~16 English letters but only ~5 Indic characters. On the real 131,072-token vocabulary:

| script | tokens that get merged at pos_dim=16 |
|---|---:|
| Malayalam | **10.13%** of its tokens |
| Devanagari | 6.29% |
| Latin (English) | 0.02% |

One more thing the audit found: **the 32-byte version only looks safe because the tokenizer was built to make it safe** — its vocabulary was constrained to never learn a token over 32 bytes. The cost didn't disappear; it was paid upstream, invisibly.

## The fix, in plain words

Stop asking *"which slot?"* and ask *"how much rotation?"*

Give every byte value a fixed random wave (a set of angles). To say "this byte is at position 7," rotate its wave 7 steps. Add up all the rotated waves — that's the word's code. Position is now a *multiplier*, not a *slot*, so there is no last slot to fall off. Position 47 just means "rotate 47 times as far."

The rotation is not decoration. Without it, addition alone makes `dog` and `god` identical (try it in the walkthrough). We trained that stripped version too — it scores **worse than the grid it was meant to replace**. Order is load-bearing.

Everything else stays exactly like V1: the codec is frozen, and only one shared projection matrix is trained.

## How to read our numbers

- **Loss** (in *nats*): how surprised the model is by the correct next token. Lower is better. A gap of 0.05 means the better model gives the right answer about 5% more probability.
- **Bits per byte (BPB)**: loss per *letter of text* instead of per token — the only fair way to compare across scripts, since Indic tokens cover ~3× the bytes.
- **Seed**: rerun the same experiment from a different random start. Our measured run-to-run wobble is about **±0.013**. Any difference smaller than that, we refuse to interpret.
- **sd / σ**: how many times bigger than the wobble a difference is. Below ~2, we call it "not resolved."

---

## The evidence, in five steps

Each step answers one question. Every claim was written down **before** the run that tested it, and every figure regenerates from a script — none is hand-edited.

### Step 1 — Count the damage (no training needed, ~2 minutes)

`python experiments/m1_collision_audit.py` reproduces the collision tables above from the raw tokenizer file. And a direct check: at identical width, the wave code separates every pair the grid merges — कार्य vs कार्यक्रम goes from **IDENTICAL** to similarity 0.76. Separated, but still *near*: related words should stay neighbours, and they do.

### Step 2 — A fair race (same everything, only the embedding differs)

Small model (11M body), 98M tokens of English+Hindi, five arms. Same tokenizer, same data order, same body initialization — verified by an identical hash of every non-embedding weight. Then repeated across **three seeds**:

| arm | mean loss (3 seeds) |
|---|---:|
| **wave** | **4.7488 ± 0.0128** |
| one-hot grid | 4.8156 ± 0.0053 |

**Paired gap −0.0668 ± 0.0056 — about 12× the noise.** The wave code wins the fair race. (A dense learned table with 32× the parameters still beats both by ~4%; we never claim otherwise.)

![Grid](figures/fig1_grid.png)

### Step 3 — Ask *why* it wins (and accept an uncomfortable answer)

Maybe it's not the wave math at all. Two ablations, each removing one ingredient:

- **`rp`** takes the grid's code — collisions, truncation and all — and just *spreads* it across dimensions (the grid concentrates its energy in ~15 of 4,096 dimensions; spreading is free and loses nothing).
- **`bag`** keeps the waves but removes the rotation (no order).

Result: **`rp` recovers 81% of wave's advantage.** At the same width, wave and rp are statistically indistinguishable. So most of the aggregate win is *spread*, not Fourier magic — we say so plainly. (`bag` collapses below the grid: order still matters.)

![Ablation](figures/fig6_ablation.png)

What rescues the wave code: it doesn't *need* the width. At width 1536 it beats the grid's best setting by **0.09 nats with 4× fewer parameters**. Spread can be bought by random projection only at full width; the wave construction gets it cheaply.

![Efficiency](figures/fig7_efficiency.png)

### Step 4 — Turn the effect on and off (the causal test)

If collisions are really the mechanism, the collision penalty should scale with the *number* of collisions and vanish when there are none. `pos_dim` is a dial with a known dose: 5,335 merged tokens at 12, then 903, 93, and **0 at 32** — a placebo built by the tokenizer itself, not by us.

We held the same 903 tokens fixed across all four models and measured the penalty on positions *right after* them (collisions corrupt a word as **context** — the output layer keeps every token scoreable as an answer, which fooled us for a full analysis round):

| dose (tokens merged) | penalty | σ |
|---:|---:|---:|
| 903 of 903 | −0.34 | 25 |
| 93 of 903 | −0.03 | 2.4 |
| **0 of 903** | **+0.005** | **0.3** |

**The effect is proportional to dose and is exactly zero at zero dose.** That's the difference between a correlation and a cause. (First attempt at this analysis had a contaminated control group that faked a placebo failure — the error and the fix are both preserved in `results/M4_M5_FINDINGS.md`.)

![Dose response](figures/fig5_dose_response.png)

### Step 5 — Scale it up and test the language claim (M3)

Bigger model (3.6× the body), 539M tokens, and a **third language: Malayalam** — chosen because the audit says it's the worst-hit script, giving a prediction we registered before the data existed: *the gain should order Malayalam > Devanagari > Latin.* Plus the baselines a reviewer demands: hash embeddings and ALBERT at exactly matched parameter budgets.

**The scoreboard** (single seed at this scale):

| arm | emb params | loss | vs grid |
|---|---:|---:|---:|
| dense (32× budget, reference) | 67.1M | 4.1225 | −5.2% |
| **wave, narrow (768)** | **0.79M** | **4.2339** | **−2.6%** |
| ALBERT (rank 16) | 2.1M | 4.2485 | −2.3% |
| rp | 2.1M | 4.2620 | −2.0% |
| wave, matched width | 2.1M | 4.2646 | −1.9% |
| one-hot grid | 2.1M | 4.3476 | — |
| hash embeddings | 2.1M | 4.3697 | **+0.5%** |

Three headlines:

**The advantage *grew* with scale.** Every codec gap is larger at 38M than at 11M, and narrow-beats-wide replicated at both scales. The efficiency claim — better loss at 2.7× fewer parameters than the baseline — survived its scale test.

**The script ladder came out exactly as registered, in every single arm** (relative BPB gain vs the grid):

| | Latin (0.02% collide) | Devanagari (6.29%) | Malayalam (10.13%) |
|---|---:|---:|---:|
| wave768 | −0.06% | −3.9% | **−6.9%** |
| dense | −3.2% | −6.0% | −8.3% |

Honest wrinkle: at matched width the frozen codecs are slightly *worse* than the grid on English — the win is an Indic story. The narrow wave768 breaks even on English while keeping the largest Indic gains of any matched arm.

**A natural experiment we never designed proved the mechanism best of all.** On positions right after a collided token, every arm that *can* tell those tokens apart — dense, ALBERT, hash, both waves, five unrelated architectures — gains ~0.47 nats. The one arm that provably *cannot* (`rp`, which carries the grid's exact code, just spread out) gains 0.04. **The penalty follows the information in the code, not its format.** It also explains why rp ties wave on the total score: rp is a touch better on the 96% of ordinary positions, wave is far better on the 2% of collision positions, and they cancel.

Two baseline lessons for free: ALBERT is genuinely strong (per-token learned factors — but its cost grows with vocabulary size, `V·r`, while the codec's doesn't contain `V` at all; whether wave768 beats it is a ~1σ call we won't make on one seed). Hash embeddings *fix* the collisions but lose overall — packing 131K tokens into 3,584 buckets creates a 37-way collision everywhere. Dense codes only help if they don't destroy identity to get there.

---

## What we claim — and what we don't

**Claimed.**
1. The byte window is a hidden tokenizer design constraint; where violated, it merges 903 real tokens permanently, ~98% of them Indic.
2. The phase code beats the grid by 0.067 ± 0.006 over three seeds, and by 0.09 nats using **4× fewer parameters** at each side's best setting — a gap that *grows* at 3.6× scale.
3. The collision penalty is **causal** (dose-response, zero at zero dose) and **information-bound** (the rp natural experiment).
4. The gain orders across scripts exactly as collision rates predict: Malayalam > Devanagari > Latin, in all six arms.

**Not claimed.**
- That the Fourier construction explains the matched-width win — 81% of it is representational spread, and rp ties wave there.
- That this beats a dense table (it doesn't; dense leads by ~5% with 32× the parameters), or that wave768 beats ALBERT (unresolved at one seed).
- That anything here transfers to frontier scale, or that validation loss — the metric most favourable to a dense table — is the final word.

**Withdrawn along the way** (kept in the findings sheets): "wave is insensitive to pos_dim" died at 1.5σ once we measured seed noise; a contaminated control briefly faked a placebo failure in the dose experiment.

---

## Reproduce

```bash
pip install -e . && pytest -q                     # 23 contract tests
python experiments/m1_collision_audit.py --byte-source both      # 2 min, no GPU

# training grids (~30 GPU-hours total on one RTX 3060) — see findings sheets
# M2: configs/m2_*  ·  M4: configs/m4_*  ·  M5: configs/m5_*  ·  M3: configs/m3_*
# frozen arms train via m2_tiny_train.py; ablations via m5_train.py; M3 via m3_train.py

# analysis (no training; first call caches, the rest are instant)
python experiments/patched.py m2_bucket_analysis --bucket-by prev-collision \
    --root results/m3 --data data/m3 --baseline m3_onehot
python experiments/patched.py m3_script_analysis --root results/m3 --data data/m3 --baseline m3_onehot
python experiments/m4_dose_analysis.py
python experiments/m2_figures.py && python experiments/m5_figures.py
```

Full commands per milestone: [M1](results/M1_FINDINGS.md) · [M2](results/M2_FINDINGS.md) · [M3](results/M3_FINDINGS.md) · [M4/M5](results/M4_M5_FINDINGS.md)

## Repository map

```
src/kronecker_v2/       the library: codecs (base, wave, ablations, baselines),
                        vocab, collisions, embedding, model, tables, eval/bpb
experiments/            one thin runner per milestone; patched.py runs any
                        analysis with the full codec chain installed
configs/                one YAML per arm; arms differ ONLY in the codec block
tests/                  23 contract tests (parameter parity is a test, not a claim)
results/                findings sheets + every manifest, log and cache
figures/                all regenerated by scripts; none hand-edited
```

## Status and roadmap

| phase | question | status |
|---|---|---|
| M1 | how much does the window cost, on the real vocabulary? | **closed** |
| M2 | does removing it help, at equal parameters? | **closed** — yes, 12× noise |
| M4 | is the collision mechanism causal? | **closed** — dose-response, placebo clean |
| M5 | what is the advantage made of, and what's the noise floor? | **closed** — 81% spread; ±0.013 |
| M3 | does it survive scale, three languages, and real baselines? | **closed** — registered predictions passed |
| M6 | do other tokenizers (Llama-3, Gemma, mT5) violate the window too? | designed, one night |
| M7 | 124M × 2.5B-token replication of the reference protocol, 3 seeds | needs ~$100–200 rented compute |

Plus one experiment M5 made necessary: a **learned** dense projection of the grid's code — the sharpest version of "how cheaply can spread be bought?"

### What would change the conclusion

Multi-seed M3 showing wave768 ≤ ALBERT; other tokenizers sitting naturally inside the window (M6); the efficiency gap closing at 124M (M7); or a learned projection matching wave768 at its width. Each is a designed run, not a hypothetical — the project is built so that a negative result is publishable rather than fatal.