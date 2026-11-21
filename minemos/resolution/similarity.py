"""Deterministic semantic similarity for Stage 4.

The default provider is a hashed character n-gram cosine: local, dependency
free, stable across processes and platforms.  ``none`` keeps only exact
normalized matches.  ``local`` reuses the engine's sentence encoder; it never
contacts a remote service, and provisions the encoder from the Hugging Face
cache (downloading it once when absent) before falling back to the hash
provider.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Dict, Optional

from .config import ResolutionConfig

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_NORMALIZE = re.compile(r"[^\w\s]+", re.UNICODE)


@lru_cache(maxsize=4096)
def normalize_value(value: str) -> str:
    """Folds case, punctuation and redundant whitespace for comparison.

    Stage 4 compares one candidate value against every comparable prior fact,
    so a conversation with many repeats of the same value would otherwise
    re-run the same regular expressions per pair.  The fold is a pure
    function of its argument; the cache only bounds how many distinct values
    are remembered.
    """
    folded = _NORMALIZE.sub(" ", value.casefold())
    return " ".join(folded.split())


def edit_distance(left: str, right: str, limit: int = 2) -> int:
    """Bounded optimal-string-alignment distance (with transpositions).

    Returns ``limit + 1`` as soon as the distance is known to exceed the
    bound, so the cost stays linear for unrelated strings.  Only used for
    short tokens; no other stage depends on it.
    """
    if left == right:
        return 0
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    before_previous = None
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        row_minimum = row
        for column, right_char in enumerate(right, 1):
            cost = 0 if left_char == right_char else 1
            value = min(previous[column] + 1, current[column - 1] + 1,
                        previous[column - 1] + cost)
            if (before_previous is not None and row > 1 and column > 1
                    and left_char == right[column - 2]
                    and left[row - 2] == right_char):
                value = min(value, before_previous[column - 2] + 1)
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > limit:
            return limit + 1
        before_previous, previous = previous, current
    return previous[-1]


def is_transposition(left: str, right: str) -> bool:
    """True when two equal-length strings differ by one adjacent swap."""
    if len(left) != len(right) or left == right:
        return False
    differences = [index for index, (a, b) in enumerate(zip(left, right))
                   if a != b]
    if len(differences) != 2:
        return False
    first, second = differences
    return second == first + 1 and left[first] == right[second] \
        and left[second] == right[first]


def _ngrams(text: str, size: int = 3) -> Counter:
    padded = f" {text} "
    if len(padded) <= size:
        return Counter([padded])
    return Counter(padded[index:index + size]
                   for index in range(len(padded) - size + 1))


def _cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(sum(count * count for count in left.values()))
    right_norm = math.sqrt(sum(count * count for count in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _tokens(text: str) -> Counter:
    return Counter(token for token in _WORD.findall(text.casefold()))


class HashSimilarity:
    """Character n-gram cosine blended with token overlap."""

    name = "hash"

    def __call__(self, left: str, right: str) -> float:
        first = normalize_value(left)
        second = normalize_value(right)
        if not first or not second:
            return 0.0
        if first == second:
            return 1.0
        char_score = _cosine(_ngrams(first), _ngrams(second))
        left_tokens = _tokens(first)
        right_tokens = _tokens(second)
        union = sum((left_tokens | right_tokens).values())
        token_score = (sum((left_tokens & right_tokens).values()) / union
                       if union else 0.0)
        return round(max(char_score, token_score, 0.0), 4)


class ExactSimilarity:
    """Exact normalized equality only."""

    name = "none"

    def __call__(self, left: str, right: str) -> float:
        return 1.0 if normalize_value(left) == normalize_value(right) else 0.0


_LOCAL_ENCODER = None
_LOCAL_ENCODER_FAILED = False


def _local_encoder():
    """Load the engine's local sentence encoder once per process.

    Returns ``None`` when the engine package or its model files are
    unavailable.  The encoder is read from the Hugging Face cache and
    downloaded on first use when absent; when offline mode is enabled and the
    cache is empty the load fails fast.  Callers fall back to the hash
    provider and report that through :attr:`LocalEncoderSimilarity.name`.
    """
    global _LOCAL_ENCODER, _LOCAL_ENCODER_FAILED
    if _LOCAL_ENCODER is None and not _LOCAL_ENCODER_FAILED:
        try:
            from minemos_engine.intelligence.dense import DenseSemanticEncoder
            _LOCAL_ENCODER = DenseSemanticEncoder()
        except Exception:  # missing engine, missing model files, no torch
            _LOCAL_ENCODER_FAILED = True
    return _LOCAL_ENCODER


class LocalEncoderSimilarity:
    """Reuses the engine's local sentence encoder when available."""

    def __init__(self) -> None:
        self._fallback = HashSimilarity()

    @property
    def name(self) -> str:
        return "local" if _local_encoder() is not None else "hash"

    def _encode(self, text: str) -> Optional[Dict[int, float]]:
        encoder = _local_encoder()
        if encoder is None:
            return None
        try:
            vector = encoder.encode(text)
        except Exception:
            return None
        if isinstance(vector, dict):
            return {int(key): float(value) for key, value in vector.items()}
        items = {index: float(value) for index, value in enumerate(vector)
                 if value}
        return items

    def __call__(self, left: str, right: str) -> float:
        first = normalize_value(left)
        second = normalize_value(right)
        if not first or not second:
            return 0.0
        if first == second:
            return 1.0
        encoded_left = self._encode(first)
        encoded_right = self._encode(second)
        if encoded_left is None or encoded_right is None:
            return self._fallback(first, second)
        dot = sum(encoded_left[key] * encoded_right.get(key, 0.0)
                  for key in encoded_left)
        left_norm = math.sqrt(sum(value * value
                                  for value in encoded_left.values()))
        right_norm = math.sqrt(sum(value * value
                                   for value in encoded_right.values()))
        if not left_norm or not right_norm:
            return 0.0
        return round(max(0.0, min(1.0, dot / (left_norm * right_norm))), 4)


def build_similarity(config: Optional[ResolutionConfig] = None):
    cfg = config or ResolutionConfig()
    if cfg.semantic_provider == "none":
        return ExactSimilarity()
    if cfg.semantic_provider == "local":
        return LocalEncoderSimilarity()
    return HashSimilarity()


__all__ = [
    "ExactSimilarity",
    "HashSimilarity",
    "LocalEncoderSimilarity",
    "build_similarity",
    "edit_distance",
    "is_transposition",
    "normalize_value",
]
