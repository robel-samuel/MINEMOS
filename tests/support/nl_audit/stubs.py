"""Deterministic stubs used by the fast audit tests.

These are mechanics-only substitutes.  They are never used for reported audit
numbers; the audit run always uses the real MiniLM encoder and the real NLI
cross-encoder unless ``--stub`` is passed explicitly.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Sequence

import numpy as np

WIDTH = 384


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", text).strip()


class HashEncoder:
    """Token-hash encoder: deterministic, dependency-free, no semantics."""

    def __init__(self, width: int = WIDTH):
        self.width = int(width)
        self.dim = self.width

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.width, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", normalize(text)):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.width
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        return vector.astype(np.float32)


class StubGate:
    """Placeholder gate; the verifier below is used instead."""


class EqualityVerifier:
    """SAME only for normalized-equal texts, NEUTRAL otherwise."""

    def verify_batch(self, pairs: Sequence[tuple], **kwargs) -> list[dict]:
        out = []
        for premise, hypothesis in pairs:
            same = normalize(premise) == normalize(hypothesis)
            out.append({
                "decision": "SAME" if same else "NEUTRAL",
                "confidence": 0.99,
                "p_entail": 0.99 if same else 0.01,
                "p_contra": 0.01,
                "p_neutral": 0.01 if same else 0.98,
                "nli_invocations": 1,
            })
        return out
