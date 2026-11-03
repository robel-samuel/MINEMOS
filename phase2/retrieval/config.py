"""Configuration for the optional retrieval backends.

Canonical retrieval is the reference implementation and the default.  The
optimized backends are opt-in and reversible: switching ``backend`` back to
``canonical`` takes effect on the next query, and no snapshot state is
consulted on that path.

Limitations that callers must accept when opting in:

* a published snapshot is *stale-but-coherent*: it may not include mutations
  that happened after its capture, but every returned row belongs to one
  single engine state;
* ``slot_index`` is snapshot-relative, so it can differ from the live
  ``memory.slots`` index while structural mutations are in flight;
* direct out-of-band writes to slot fields or ``memory.slots`` bypass the
  engine's version counter and are unsupported for every backend.

``contended_fallback`` controls what happens after the optimistic build budget
is exhausted.  ``"canonical"`` (the default) hands the query back to the
engine's canonical path, which is exact and never holds the lock for a rebuild;
``"locked_build"`` keeps the original B5 behaviour and rebuilds under the lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

CANONICAL = "canonical"
OPTIMIZED = "optimized"
INCREMENTAL = "incremental"
QDRANT = "qdrant"
BACKENDS = (CANONICAL, OPTIMIZED, INCREMENTAL, QDRANT)
CONTENTION_FALLBACKS = ("canonical", "locked_build")

# Deployment profiles.  ``legacy`` preserves the historical default (canonical
# unless a backend is explicitly selected).  ``production`` makes Qdrant the
# standard backend and validates its server configuration eagerly.
# ``canonical_only`` never imports or contacts Qdrant, for development, tests,
# recovery and unchanged installations.
LEGACY = "legacy"
PRODUCTION = "production"
CANONICAL_ONLY = "canonical"
RETRIEVAL_MODES = (LEGACY, PRODUCTION, CANONICAL_ONLY)
DEFAULT_RETRIEVAL_MODE = LEGACY


class RetrievalConfigurationError(ValueError):
    """A retrieval deployment configuration is missing or unsafe."""


def normalize_mode(mode: Optional[str]) -> str:
    if mode is None:
        return DEFAULT_RETRIEVAL_MODE
    if not isinstance(mode, str):
        raise TypeError(
            f"retrieval_mode must be a string, got {type(mode)!r}")
    value = mode.strip().lower()
    if not value:
        return DEFAULT_RETRIEVAL_MODE
    if value not in RETRIEVAL_MODES:
        raise RetrievalConfigurationError(
            f"unknown retrieval_mode {mode!r}; expected one of "
            f"{RETRIEVAL_MODES}")
    return value


def normalize_backend(backend: Optional[str]) -> str:
    if backend is None:
        return CANONICAL
    if not isinstance(backend, str):
        raise TypeError(f"retrieval_backend must be a string, got {type(backend)!r}")
    value = backend.strip().lower()
    if value not in BACKENDS:
        raise ValueError(
            f"unknown retrieval_backend {backend!r}; expected one of {BACKENDS}")
    return value


@dataclass(frozen=True)
class RetrievalConfig:
    """Immutable retrieval configuration (never serialized in checkpoints)."""

    backend: str = CANONICAL
    mode: str = DEFAULT_RETRIEVAL_MODE
    max_optimistic_attempts: int = 3
    max_incremental_updates: int = 64
    fallback_to_canonical: bool = True
    verify_bound: bool = False
    contended_fallback: str = "canonical"

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", normalize_backend(self.backend))
        object.__setattr__(self, "mode", normalize_mode(self.mode))
        if int(self.max_optimistic_attempts) < 1:
            raise ValueError("max_optimistic_attempts must be >= 1")
        if int(self.max_incremental_updates) < 0:
            raise ValueError("max_incremental_updates must be >= 0")
        if self.contended_fallback not in CONTENTION_FALLBACKS:
            raise ValueError(
                f"contended_fallback must be one of {CONTENTION_FALLBACKS}, "
                f"got {self.contended_fallback!r}")
        object.__setattr__(self, "max_optimistic_attempts",
                           int(self.max_optimistic_attempts))
        object.__setattr__(self, "max_incremental_updates",
                           int(self.max_incremental_updates))
        object.__setattr__(self, "fallback_to_canonical",
                           bool(self.fallback_to_canonical))
        object.__setattr__(self, "verify_bound", bool(self.verify_bound))
        object.__setattr__(self, "contended_fallback",
                           str(self.contended_fallback))

    @classmethod
    def from_value(cls, value: Any) -> "RetrievalConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(backend=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(
            f"retrieval_config must be RetrievalConfig, str, dict or None, "
            f"got {type(value)!r}")
