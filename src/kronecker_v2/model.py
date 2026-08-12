"""Tiny GPT for the M2/M3 embedding comparisons.

Adapted from Andrej Karpathy's nanoGPT (github.com/karpathy/nanoGPT, MIT
license) with two deliberate changes and nothing else:

  1. ``wte`` is pluggable: a dense ``nn.Embedding`` or a ``CodecEmbedding``
     wrapping any frozen codec. This is the ONLY thing that differs between
     experimental arms.
  2. No weight tying — the codec arms have no V x d_model matrix to tie to
     (their input width is code_dim, not d_model), so no arm ties.

Everything downstream of the embedding — positions, blocks, ln_f, lm_head,
init, optimizer grouping — is identical across arms by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 131_072
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """``wte`` is injected, everything else fixed."""

    def __init__(self, cfg: GPTConfig, wte: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = wte
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # nanoGPT init on the shared body. The wte initialises itself.
        self.apply(self._init_body)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_body(self, module: nn.Module) -> None:
        if module is self.wte or any(m is module for m in self.wte.modules()):
            return  # the embedding arm owns its own init
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    # -------- reporting helpers --------
    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        head = sum(p.numel() for p in self.lm_head.parameters())
        emb = sum(p.numel() for p in self.wte.parameters())
        return {"total": total, "body": total - head - emb,
                "lm_head": head, "embedding": emb}

    def body_state_hash(self) -> str:
        """Hash of everything EXCEPT the embedding — must be identical across
        arms started from the same seed. That equality is the experimental
        control, so it is checkable, not assumed."""
        import hashlib
        h = hashlib.sha256()
        for name, p in sorted(self.named_parameters()):
            if name.startswith("wte."):
                continue
            h.update(name.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()

    def optimizer_groups(self, weight_decay: float):
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]
