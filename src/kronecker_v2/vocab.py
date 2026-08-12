"""Vocabulary access: token id -> UNTRUNCATED UTF-8 bytes, plus script tagging.

Why untruncated: the reference ``build_byte_buffer`` truncates at ``pos_dim``,
which is exactly the information the collision audit needs to measure. We
truncate ourselves, per window, using the reference's UTF-8-safe rule.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

from kronecker_embeddings import token_id_to_bytes, utf8_safe_truncate

VOCAB_SIZE = 131_072
DEFAULT_TOKENIZER = "theschoolofai/BrahmicTokenizer-131K"


def load_tokenizer(name_or_file: str = DEFAULT_TOKENIZER):
    """Load the BrahmicTokenizer from the hub or from a local tokenizer.json."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if name_or_file.endswith(".json"):
        return PreTrainedTokenizerFast(tokenizer_file=name_or_file)
    return AutoTokenizer.from_pretrained(name_or_file)


def _byte_alphabet() -> dict[str, int]:
    """Inverse of the GPT-2 byte<->unicode alphabet used by byte-level BPE."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def vocab_bytes(
    tokenizer,
    vocab_size: int = VOCAB_SIZE,
    source: str = "decode",
) -> dict[int, bytes]:
    """token id -> full UTF-8 bytes, NOT truncated. The audit's raw material.

    source="decode"  reference behaviour: ``token_id_to_bytes`` -> ``decode()``.
                     This is what the shipped codec actually sees, so it is the
                     ground truth for "what collides in the deployed model".
    source="raw"     the byte-level BPE piece decoded through the GPT-2 byte
                     alphabet. Lossless for partial-codepoint tokens, which
                     ``decode()`` collapses to U+FFFD.

    Report both. Their difference is itself a finding.
    """
    if source == "decode":
        special = set(getattr(tokenizer, "all_special_ids", []) or [])
        out: dict[int, bytes] = {}
        for tid in range(vocab_size):
            try:
                out[tid] = token_id_to_bytes(tokenizer, tid, special)
            except Exception:
                out[tid] = b""
        return out

    if source != "raw":
        raise ValueError("source must be 'decode' or 'raw'")

    u2b = _byte_alphabet()
    added = {t.content for t in getattr(tokenizer, "added_tokens_decoder", {}).values()}
    out = {}
    for piece, tid in tokenizer.get_vocab().items():
        if piece in added:
            out[tid] = piece.encode("utf-8")
        else:
            try:
                out[tid] = bytes(u2b[c] for c in piece)
            except KeyError:
                out[tid] = piece.encode("utf-8")
    return out


def truncate(raw: bytes, pos_dim: int) -> bytes:
    """The reference's UTF-8-safe truncation — never splits a codepoint."""
    return utf8_safe_truncate(raw, pos_dim) if len(raw) > pos_dim else raw


@lru_cache(maxsize=None)
def _script_of_char(ch: str) -> str | None:
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return None


def script_of(raw: bytes) -> str:
    """Dominant Unicode script of a token's bytes."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "Invalid/Partial"
    for ch in text:
        if ch.isspace() or not ch.isalpha():
            continue
        name = _script_of_char(ch)
        return name if name else "Unknown"
    return "NonAlpha"


def script_map(vb: dict[int, bytes]) -> dict[int, str]:
    return {tid: script_of(raw) for tid, raw in vb.items()}