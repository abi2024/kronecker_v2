#!/usr/bin/env bash
# Kronecker V2 — scaffold boilerplate.
# Safe to re-run: never overwrites an existing file.
set -euo pipefail

w() {  # w <path>  — write stdin to path, skipping if it already exists
  if [ -e "$1" ]; then echo "  skip (exists): $1"; return 0; fi
  mkdir -p "$(dirname "$1")"
  cat > "$1"
  echo "  created: $1"
}

mkdir -p src/kronecker_v2/{codecs,eval} experiments configs tests \
         results/{m1_collisions,m4_headline} figures notebooks paper

# ---------------------------------------------------------------- packages ---
w src/kronecker_v2/__init__.py <<'EOF'
"""Kronecker V2 — phase-bound Fourier byte codes for transformer token embeddings."""

__version__ = "0.1.0"
EOF

w src/kronecker_v2/codecs/__init__.py <<'EOF'
"""Codec implementations. Every codec maps token ids -> [N, code_dim] float32."""
EOF

w src/kronecker_v2/eval/__init__.py <<'EOF'
"""Evaluation: per-script bits-per-byte and typo robustness."""
EOF

# ------------------------------------------------------------------ library ---
w src/kronecker_v2/vocab.py <<'EOF'
"""Vocabulary access: token id -> raw UTF-8 bytes, plus Unicode script tagging.

This is the input to the Milestone 1 collision audit. Exact collisions are
counted on truncated *bytes*, not on codes.
"""

from __future__ import annotations

VOCAB_SIZE = 131_072


def load_tokenizer(name: str = "theschoolofai/BrahmicTokenizer-131K"):
    """Return the BrahmicTokenizer."""
    raise NotImplementedError


def token_bytes(tokenizer, token_id: int) -> bytes:
    """Raw UTF-8 bytes for one token id."""
    raise NotImplementedError


def dump_vocab_bytes(tokenizer, path: str) -> None:
    """Write every token id -> its UTF-8 bytes. Milestone 0 deliverable."""
    raise NotImplementedError


def script_of(text: str) -> str:
    """Dominant Unicode script: 'Devanagari', 'Tamil', 'Telugu', 'Latin', ..."""
    raise NotImplementedError
EOF

w src/kronecker_v2/codecs/onehot.py <<'EOF'
"""V1 baseline codec: one-hot(byte value) (x) one-hot(byte position).

256 byte values x 32 positions = 8192 dims. Scale by 1/sqrt(L), then z-normalise.
Known failure: bytes past position `pos_dim` are dropped, so tokens agreeing on
their first `pos_dim` bytes collide permanently.

TODO: implement the Codec protocol defined in .base
"""

from __future__ import annotations

CHAR_DIM = 256


class OneHotKroneckerCodec:
    name = "onehot"

    def __init__(self, tokenizer, pos_dim: int = 32) -> None:
        self.tokenizer = tokenizer
        self.pos_dim = pos_dim
        self.code_dim = CHAR_DIM * pos_dim

    def encode(self, token_ids):
        """[N] int64 -> [N, code_dim] float32."""
        raise NotImplementedError

    def fingerprint(self) -> str:
        """sha256 over the full-vocab code table. Pinned by tests/test_determinism.py."""
        raise NotImplementedError
EOF

w src/kronecker_v2/codecs/wave.py <<'EOF'
"""Phase-bound Fourier byte codec (FHRR + fractional power encoding).

Each byte VALUE gets a fixed random phasor vector; byte POSITION enters as a
phase rotation (z^p), so length is unbounded and long tokens degrade gracefully
instead of being cropped. Token code = bundle (sum) of bound byte-position pairs.

Engineering contract: build codes in complex64 ONCE, then store as real
(concat real & imag -> 2*D real dims). No complex tensors in the training loop.
D = 4096 complex -> 8192 real, matching the one-hot code dim exactly.

TODO: implement the Codec protocol defined in .base
"""

from __future__ import annotations

CHAR_DIM = 256


class WaveKroneckerCodec:
    name = "wave"

    def __init__(
        self,
        tokenizer,
        d_complex: int = 4096,
        seed: int = 0,
        normalize: str = "sqrt_len",   # sqrt_len | l2 | znorm
        learnable_phases: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.d_complex = d_complex
        self.code_dim = 2 * d_complex
        self.seed = seed
        self.normalize = normalize
        self.learnable_phases = learnable_phases

    def _base_phasors(self):
        """[256, d_complex] complex64, phases uniform on (-pi, pi]. Frozen."""
        raise NotImplementedError

    def _position_phasor(self, p: int):
        """Fractional power encoding: z^p."""
        raise NotImplementedError

    def encode(self, token_ids):
        """[N] int64 -> [N, code_dim] float32 (real ++ imag)."""
        raise NotImplementedError

    def fingerprint(self) -> str:
        raise NotImplementedError
EOF

w src/kronecker_v2/codecs/baselines.py <<'EOF'
"""Comparison arms for the matched-parameter experiments.

All must produce the same trainable parameter count as the Kronecker arms at a
given d_model — enforced by tests/test_equal_params.py.

TODO: implement the Codec protocol defined in .base
"""

from __future__ import annotations


class DenseTableCodec:
    """Plain V x d_model learned embedding table."""
    name = "dense"


class HashEmbeddingCodec:
    """Hash embeddings (arXiv 1709.03933)."""
    name = "hash"


class ALBERTFactorizedCodec:
    """V x r then r x d_model factorization (Lan et al., 2020)."""
    name = "albert"
EOF

w src/kronecker_v2/embedding.py <<'EOF'
"""The nn.Module every training run uses.

Frozen code table (a buffer, never trained) + one shared trainable projection.
The transformer body never knows which codec produced the codes — that is what
keeps "change only the embedding" true.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KroneckerEmbedding(nn.Module):
    def __init__(self, codec, d_model: int, vocab_size: int = 131_072) -> None:
        super().__init__()
        self.codec = codec
        # TODO: precompute all codes once, in chunks, register as a buffer
        # self.register_buffer("codes", codec.encode(torch.arange(vocab_size)))
        self.proj = nn.Linear(codec.code_dim, d_model, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """[B, T] int64 -> [B, T, d_model]."""
        raise NotImplementedError
EOF

w src/kronecker_v2/collisions.py <<'EOF'
"""Milestone 1: the collision audit.

Exact collisions  -> hash the truncated byte string; any group of 2+ tokens is a
                     set the model can never distinguish, forever.
Near collisions   -> cosine similarity above threshold in code space.
Both are reported PER SCRIPT. This is the headline measurement of the project.
"""

from __future__ import annotations


def exact_collisions(vocab_bytes: dict[int, bytes], pos_dim: int) -> dict:
    """Group token ids whose first `pos_dim` bytes are identical."""
    raise NotImplementedError


def near_collisions(codes, threshold: float = 0.95, chunk: int = 8192):
    """Pairs above a cosine threshold. Chunked: 131k x 8192 fp32 is ~4.3 GB."""
    raise NotImplementedError


def collisions_by_script(collisions: dict, script_map: dict[int, str]) -> dict:
    raise NotImplementedError
EOF

w src/kronecker_v2/capacity.py <<'EOF'
"""Analytic FHRR capacity curve, overlaid on the empirical histogram.

Independent phasor codes overlap with mean 0 and noise std ~ 1/sqrt(2D).
A bundle of M components has self-similarity ~ 1/sqrt(M), so retrieval SNR
~ sqrt(D/M) (Frady, Kleyko & Sommer, arXiv 1707.01429). Capacity grows
linearly in D — that is the theory the measurement should match.
"""

from __future__ import annotations


def noise_std(d_complex: int) -> float:
    raise NotImplementedError


def expected_similarity(shared_bindings: int, total_bindings: int) -> float:
    raise NotImplementedError


def collision_probability(d_complex: int, n_items: int, threshold: float) -> float:
    raise NotImplementedError
EOF

w src/kronecker_v2/eval/bpb.py <<'EOF'
"""Per-script bits-per-byte.

Per-BYTE, not per-token, so arms stay comparable regardless of how many tokens
a script costs.
"""

from __future__ import annotations


def bits_per_byte(model, dataloader, script: str | None = None) -> float:
    raise NotImplementedError
EOF

w src/kronecker_v2/eval/robustness.py <<'EOF'
"""Typo robustness probe, matching the V1 protocol.

Clean/typo prompt pairs; report top-1-preserved rate, KL(clean || typo), and
final-hidden cosine. V1 reported 55.5% top-1-preserved for Kronecker vs 47.3%
for BPE — reproduce that metric set so the numbers sit side by side.
"""

from __future__ import annotations


def make_typo_pairs(prompts: list[str], seed: int = 0) -> list[tuple[str, str]]:
    raise NotImplementedError


def top1_preserved(model, pairs) -> float:
    raise NotImplementedError
EOF

# -------------------------------------------------------------- experiments ---
for m in \
  "m0_smoke:Milestone 0 — encode a few tokens, print byte counts and shapes, dump vocab bytes." \
  "m1_collision_audit:Milestone 1 — exact + near collisions per script at pos_dim 32/48/64, with the analytic overlay." \
  "m2_tiny_train:Milestone 2 — ~10M model, one-hot vs wave, does it train at all; sweep normalisation." \
  "m3_matched_param:Milestone 3 — 30-50M matched-parameter head-to-head, per-script bits-per-byte, ablation grid." \
  "m4_headline:Milestone 4 — 124M / 2.5B tokens / 3 seeds, all arms, plus the robustness probe." \
  "m5_long_tokens:Milestone 5 — tokens longer than 32 bytes: show one-hot emits identical vectors, wave does not."
do
  name="${m%%:*}"; desc="${m#*:}"
  w "experiments/${name}.py" <<EOF
"""${desc}"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/")
    args = ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
EOF
done

# ------------------------------------------------------------------ configs ---
w configs/_shared.yaml <<'EOF'
# Settings identical across every arm. If a value lives here, it is NOT the
# variable under test.
tokenizer: theschoolofai/BrahmicTokenizer-131K
vocab_size: 131072
code_dim: 8192
seed: 0
data:
  english: HuggingFaceFW/fineweb-edu
  indic: [ai4bharat/sangraha]
  english_fraction: 0.5
EOF

w configs/m2_tiny.yaml <<'EOF'
extends: _shared.yaml
milestone: m2
model: {n_layer: 6, n_head: 6, d_model: 384}
tokens: 100_000_000
codec: {name: onehot, pos_dim: 32}   # <- the ONLY line that differs between arms
EOF

for arm in onehot wave; do
  w "configs/m3_50m_${arm}.yaml" <<EOF
extends: _shared.yaml
milestone: m3
model: {n_layer: 12, n_head: 8, d_model: 512}
tokens: 500_000_000
codec: {name: ${arm}}
EOF
done

for arm in dense onehot wave hash albert; do
  w "configs/m4_124m_${arm}.yaml" <<EOF
extends: _shared.yaml
milestone: m4
model: {n_layer: 12, n_head: 12, d_model: 768}
tokens: 2_500_000_000
seeds: [0, 1, 2]
codec: {name: ${arm}}
EOF
done

# -------------------------------------------------------------------- tests ---
w tests/test_determinism.py <<'EOF'
"""The codec fingerprint. Pin it once; never let it change silently."""

import hashlib

import torch

EXPECTED = {
    # "onehot": "<paste the sha256 from your first full-vocab encode>",
    # "wave":   "<...>",
}


def _fingerprint(codes: torch.Tensor) -> str:
    return hashlib.sha256(codes.contiguous().numpy().tobytes()).hexdigest()


def test_encode_is_deterministic():
    """Same ids, twice, byte-identical."""
    raise NotImplementedError


def test_fingerprint_unchanged():
    """Guards against a refactor silently changing a code mid-project."""
    raise NotImplementedError
EOF

w tests/test_shapes.py <<'EOF'
"""Shape contracts: codes are [N, code_dim]; embeddings are [B, T, d_model]."""


def test_code_shape():
    raise NotImplementedError


def test_embedding_shape():
    raise NotImplementedError


def test_same_token_same_vector():
    """A token at two different positions in a batch must embed identically."""
    raise NotImplementedError
EOF

# ------------------------------------------------------------------- extras ---
w .gitignore <<'EOF'
.venv/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
*.pt
*.bin
*.safetensors
data/
results/**/*.npy
results/**/*.pt
wandb/
.DS_Store
EOF

w Makefile <<'EOF'
.PHONY: install test m0 m1 m2 m3 m4 m5 figures

install:
	pip install -e .

test:
	pytest -q

m0:
	python experiments/m0_smoke.py

m1:
	python experiments/m1_collision_audit.py --out results/m1_collisions

m2:
	python experiments/m2_tiny_train.py --config configs/m2_tiny.yaml

m3:
	python experiments/m3_matched_param.py --config configs/m3_50m_onehot.yaml
	python experiments/m3_matched_param.py --config configs/m3_50m_wave.yaml

m4:
	@for a in dense onehot wave hash albert; do \
	  for s in 0 1 2; do \
	    python experiments/m4_headline.py --config configs/m4_124m_$$a.yaml --seed $$s; \
	  done; \
	done

m5:
	python experiments/m5_long_tokens.py

figures:
	@echo "Regenerate every figure from results/. No hand-edited figures."
EOF

w results/RUNS.md <<'EOF'
# Run ledger

Append one row per run. This file becomes the methods section of the paper.
A number without a manifest.json is not a result.

| date | milestone | config | seed | git SHA | codec fingerprint | final loss | outputs |
|------|-----------|--------|------|---------|-------------------|-----------|---------|
EOF

w README.md <<'EOF'
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
- [ ] M0 — reproduce V1 codec, pin determinism fingerprint
- [ ] M1 — collision audit per script (pos_dim 32/48/64) + analytic overlay
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
EOF

w paper/draft.md <<'EOF'
# Kronecker V2 (draft)

## Abstract
## 1. The truncation bottleneck
## 2. Method: phase-bound Fourier byte codes
## 3. Collision audit (Milestone 1)
## 4. Matched-parameter comparison (Milestones 3-4)
## 5. Robustness and long tokens (Milestone 5)
## 6. Limitations
## 7. Related work
EOF

echo
echo "Done. Next: pip install -e . && pytest -q  (expect NotImplementedError — that is correct)"