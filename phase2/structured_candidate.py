"""
Phase 2 — structured candidates and deterministic encoding.

This module is intentionally independent of the (frozen) semantic
extraction layer. Everything downstream in Phase 2 consumes Candidate
objects directly -- no natural-language extraction happens here.

ENCODING CHOICE (documented, not hidden):

x_t = E(C_t) is built with the "hashing trick" (feature hashing): each
candidate is turned into a bag of tokens (subject, predicate, object,
lowercased and split on non-alphanumeric characters), each token is
hashed deterministically (SHA-1, fixed across runs -- NOT Python's
randomized hash()) into one of D dimensions with a deterministic +/-1
sign, and the resulting vector is L2-normalized.

This is deterministic, requires no training and no model download, and
is a standard, well-understood technique (used e.g. in Vowpal Wabbit) --
not something invented for this benchmark. It is NOT a learned semantic
embedding: token overlap drives similarity, nothing more.

Two encodings are produced per candidate, on purpose:
  - full vector  = hash(subject) + hash(predicate) + hash(object) tokens
    together -- used for content-addressing (finding/creating a slot).
  - query vector = hash(subject) + hash(predicate) tokens ONLY (no
    object) -- used when recalling "what is subject's predicate?"
This split is what makes recall-by-partial-query possible at all; see
memory_update.py's docstring for the addressing/threshold rule this
implies.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

DIM = 64  # dimensionality of x_t; fixed, documented, not tuned per-test


@dataclass
class Candidate:
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    timestamp: float = 0.0


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if t]


def _hash_token(token: str, dim: int = DIM) -> tuple[int, float]:
    """Deterministic (index, sign) for one token, independent of process
    or PYTHONHASHSEED -- uses SHA-1, not Python's built-in hash()."""
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return idx, sign


def _bag_of_tokens_vector(tokens: list[str], dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for tok in tokens:
        idx, sign = _hash_token(tok, dim)
        v[idx] += sign
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


def encode_full(c: Candidate, dim: int = DIM) -> np.ndarray:
    """
    x_t for addressing/update -- subject + predicate + object.

    ATOMIC TOKENIZATION (Phase 2B change -- the ONLY thing altered in this
    experiment): each field is hashed as ONE token, not word-split. E.g.
    "person_000001" becomes the single token "subject:person_000001", not
    ["person", "000001"]. This removes the tokenizer-induced shared-token
    floor discovered in Phase 2A (all subjects secretly shared the literal
    token "person"). Fields are prefixed with their role ("subject:",
    "predicate:", "object:") only to prevent a value that happens to
    equal another field's value from colliding across roles -- this does
    not reintroduce word-splitting.
    """
    tokens = [f"subject:{c.subject.lower()}",
              f"predicate:{c.predicate.lower()}",
              f"object:{c.object.lower()}"]
    return _bag_of_tokens_vector(tokens, dim)


def encode_query(subject: str, predicate: str, dim: int = DIM) -> np.ndarray:
    """Partial encoding for recall -- subject + predicate only, no object
    (the object is what recall is trying to retrieve). Same atomic
    tokenization as encode_full."""
    tokens = [f"subject:{subject.lower()}", f"predicate:{predicate.lower()}"]
    return _bag_of_tokens_vector(tokens, dim)
