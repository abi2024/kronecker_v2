"""Milestone 1: the collision audit.

A permanent collision is a set of >=2 token ids whose UTF-8-safe-truncated byte
sequences are identical at a given ``pos_dim``. The one-hot Kronecker codec is a
pure function of those bytes, so such tokens receive the SAME embedding vector
for the entire life of the model. No amount of training separates them.

Reported per Unicode script, because the failure is not uniform across scripts:
3-byte-per-character Indic text exhausts a byte window ~3x faster than Latin.
"""

from __future__ import annotations

import collections

from .vocab import script_of, truncate


def exact_collisions(vocab_bytes: dict[int, bytes], pos_dim: int) -> dict[bytes, list[int]]:
    """Groups of token ids sharing an identical truncated byte string."""
    groups: dict[bytes, list[int]] = collections.defaultdict(list)
    for tid, raw in vocab_bytes.items():
        groups[truncate(raw, pos_dim)].append(tid)
    return {k: v for k, v in groups.items() if len(v) > 1}


def truncation_stats(vocab_bytes: dict[int, bytes], pos_dim: int) -> dict:
    """How many tokens lose bytes at this window."""
    n = len(vocab_bytes)
    lost = [len(r) - len(truncate(r, pos_dim)) for r in vocab_bytes.values()]
    n_trunc = sum(1 for x in lost if x > 0)
    return {
        "pos_dim": pos_dim,
        "vocab_size": n,
        "n_truncated": n_trunc,
        "pct_truncated": 100.0 * n_trunc / max(n, 1),
        "max_bytes_lost": max(lost) if lost else 0,
    }


def audit(vocab_bytes: dict[int, bytes], pos_dims=(12, 16, 20, 24, 32)) -> list[dict]:
    """Per-(pos_dim, script) collision counts. One row per script per window."""
    smap = {tid: script_of(raw) for tid, raw in vocab_bytes.items()}
    totals = collections.Counter(smap.values())
    rows = []
    for pd in pos_dims:
        coll = exact_collisions(vocab_bytes, pd)
        per_script = collections.Counter()
        for ids in coll.values():
            for tid in ids:
                per_script[smap[tid]] += 1
        for script, total in totals.items():
            n = per_script.get(script, 0)
            rows.append({
                "pos_dim": pd,
                "script": script,
                "tokens_in_script": total,
                "collided_tokens": n,
                "pct_collided": 100.0 * n / max(total, 1),
            })
    return rows


def examples(vocab_bytes: dict[int, bytes], pos_dim: int, script: str, k: int = 5):
    """Human-readable collision groups for a script — the figure-1 material."""
    out = []
    for ids in exact_collisions(vocab_bytes, pos_dim).values():
        if script_of(vocab_bytes[ids[0]]) != script:
            continue
        out.append([vocab_bytes[t].decode("utf-8", "replace") for t in ids])
        if len(out) >= k:
            break
    return out