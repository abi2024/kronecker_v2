# Kronecker V2 — The Milestone Map

**The project in one sentence:** the current Kronecker embedding stores a token's bytes in 32 fixed boxes and throws away everything past box 32, so some tokens become permanently identical; you're replacing boxes with rotating waves so nothing gets thrown away — and proving it works with fair measurements.

**Why the order below matters:** every milestone is cheap enough to fail safely, and each one buys you the right to spend money on the next. You never run the expensive experiment until the cheap ones have told you it's worth running.

---

## The 3 rules that make your results valid

Read these first. Every milestone obeys them.

1. **Change one thing.** Same tokenizer, same data, same model body, same training settings in every comparison. Only the embedding module swaps. Then any difference you measure was *caused* by your change — there's nothing else it could be.
2. **Say the claim before you run.** Each milestone below has a pass/fail sentence. Write it down first, then run. A result only counts as evidence because it was allowed to fail.
3. **Equal size by construction.** Your wave code uses 4,096 complex numbers = 8,192 real numbers — exactly the size of the old one-hot code, feeding the exact same Linear(8192 → d_model). Nobody can say "you just used a bigger model."

---

## Milestone 0 — Hold the existing thing in your hands (Days 1–3)

**Plain English goal:** before improving something, run it yourself and understand it.

**What you actually do:**
- Clone three repos: `theschoolofai/BrahmicTokenizer-131K` (the vocab), `theschoolofai/kronecker-embeddings` (the V1 codec), and `karpathy/nanoGPT` (or `KellerJordan/modded-nanogpt`). `pip install torchhd`.
- Run the V1 codec on a few tokens — `the`, `भारत`, `తెలుగు` — and look at the codes it produces.
- Write a tiny determinism test: encode the same token twice, hash both codes, confirm they're identical. Then encode the entire 131K vocab once without errors.

**What you're achieving inside this milestone:** you now understand the 3-step pipeline (bytes → grid → shared projection) well enough to replace step 2 later, and your environment provably works.

**You're done when:** your locally computed code for a token matches the reference repo exactly, and all 131,072 tokens encode cleanly.

**Cost:** laptop only. No GPU needed.

---

## Milestone 1 — The collision audit (Week 1) — the most important cheap step

**Plain English goal:** measure the disease before selling the cure. Nobody has ever counted how many real tokens collide.

**What you actually do:**
- For every token in the vocab: take its first 32 bytes, hash them, group identical hashes. Every group of 2+ tokens is a set the model *can never tell apart*, forever.
- Split the counts by script (Devanagari, Tamil, Telugu, Bengali, Latin…) using Unicode script properties.
- Repeat with windows of 48 and 64 bytes.
- Now build your wave code with torchhd: 256 fixed random waves (one per byte value), position = rotate the wave by its position number, add up all the byte-waves of a token. Encode all 131K tokens. Count exact collisions and near-collisions (cosine similarity above a threshold).

**What you're achieving inside this milestone:** two publishable facts, with zero training — (a) how bad the collision problem actually is today, per script, and (b) how much your fix reduces it at identical size. Plus a theory check: the math predicts random codes overlap by about 1/√(2D); your histogram should match that curve.

**You're done when:** you have a CSV, one chart (collisions vs window size, one line per script), and the theory curve overlaid on the measured histogram.

**Decision gate (write it down BEFORE looking at results):**
- If any major Indic script has ≥ ~0.01% permanent collisions → your headline is *"real collisions exist and we fix them."*
- If collisions are negligible → your headline pivots to *"no truncation ever, graceful handling of long tokens, better robustness."*
- Both are legitimate papers. The audit decides which one you're writing — you don't.

**Cost:** mostly CPU; a few GPU-hours for the similarity matrices.

---

## Milestone 2 — Does the wave code train at all? (Week 2)

**Plain English goal:** find the numerical surprises while they cost hours, not days.

**What you actually do:**
- Build `KroneckerWaveEmbedding` as a proper module: precompute all 131K wave codes **once** into a frozen table of real numbers (split each complex number into its real and imaginary parts → 8,192 real values), then a normal Linear(8192 → d_model). No complex numbers inside the training loop — they break compilation and bf16.
- Train two tiny ~10M-parameter models on TinyStories or a small multilingual slice: one with the old one-hot code, one with your wave code. Everything else identical.
- Quickly sweep the normalization choice (divide by √length vs L2 vs z-norm) here, at toy scale.

**What you're achieving inside this milestone:** proof the engineering works — a smooth loss curve — and a locked normalization choice, so you never debug numerics at 124M scale.

**Pass/fail claim:** "The wave code reaches within a few percent of one-hot's loss at this toy scale."

**Kill criterion:** if it diverges or badly plateaus, stop and fix scaling/normalization. Do not proceed to Milestone 3 until this passes.

**Cost:** 2–10 GPU-hours.

---

## Milestone 3 — The fair fight (Weeks 3–4)

**Plain English goal:** a head-to-head where the *only possible explanation* for a difference is your change.

**What you actually do:**
- Train 30–50M-parameter models on a multilingual mix (~half English FineWeb-Edu, ~half Indic from Sangraha / FineWeb-2 hi-ta-te-bn), with the same BrahmicTokenizer in every arm.
- Run the small ablation grid on the wave code: frozen vs learnable waves, normalization, bandwidth. Pick one winner config.
- Report **bits-per-byte per script** (per-byte, not per-token, so the comparison stays fair).

**What you're achieving inside this milestone:** your method's final recipe, chosen on evidence — and the first real quality signal, especially on the scripts Milestone 1 flagged as collision-heavy.

**Pass/fail claim (write it before running):** "At equal parameters, the wave code matches or beats one-hot on per-script loss for the high-collision scripts."

**You're done when:** one config is clearly the one to scale, and the claim is confirmed — or honestly rejected, in which case your paper is the audit + robustness story and you saved yourself the big run.

**Cost:** ~0.5–1 GPU-day.

---

## Milestone 4 — The headline run (Weeks 5–7)

**Plain English goal:** produce the number people will quote — under the *same protocol as the V1 paper*, so your result sits directly next to theirs.

**What you actually do:**
- 124M GPT-2 shape, 2.5B tokens, **3 random seeds each**, all arms at matched parameter count: dense table, one-hot Kronecker, wave Kronecker, hash embeddings, ALBERT factorization.
- Run the typo-robustness probe exactly as V1 did (clean/typo prompt pairs, "top-1 prediction preserved" rate — V1 reported 55.5% for Kronecker vs 47.3% for BPE).

**What you're achieving inside this milestone:** the load-bearing result of the whole project — mean±std across seeds, per-script loss table, robustness table. Three seeds is what stops you from publishing luck.

**Pass/fail claim:** "Wave ≤ one-hot on validation loss AND ≥ one-hot on typo robustness at 124M / 2.5B tokens."

**Cost:** the big one — roughly 1–4 GPU-days per arm on an A100-class card. This is why Milestones 2 and 3 exist: you only pay this once the story and the method are both proven cheap.

---

## Milestone 5 — Stress test + shipping (Weeks 8–10)

**Plain English goal:** turn results into something a stranger can verify.

**What you actually do:**
- **Long-token stress test:** collect real tokens longer than 32 bytes (conjunct-heavy Hindi words, German compounds, URLs, code identifiers). Show that one-hot literally outputs *identical vectors* for colliding pairs while your wave code separates them, then measure loss on text rich in those tokens.
- **Ship:** README that tells the story in order (problem → audit numbers → method → fair fight → headline → limits), public code, and the four load-bearing figures: collisions-vs-window per script; theory curve vs measured; matched-param loss curves; robustness table. Optional webapp with an interactive collision explorer, per the assignment spec.

**You're done when:** someone else can clone your repo and regenerate Figure 1 with one command.

---

## Why this ladder produces *valid* results (the whole logic in five lines)

- Each step is **cheap enough to be wrong**, so failing early is a feature, not a disaster.
- Only the embedding ever changes, so every measured gap is **caused by your work**.
- Every pass/fail line is written **before** the run, so a pass actually means something.
- You reuse V1's exact protocol, so your numbers are **directly comparable** without argument.
- Three seeds with mean±std means you report **signal, not luck**.

**What you can honestly claim at the end:** the audit facts, the matched-parameter 124M comparison, the robustness numbers, and the long-token behavior. **What you can't:** that any of it holds at frontier scale — say so in one sentence and you're bulletproof.

---

## Today, literally

1. Clone the three repos and `pip install torchhd`.
2. Encode `भारत` with the reference V1 codec and print the code.
3. Write the determinism test.

That's Milestone 0, step 1. Everything else follows from there.
