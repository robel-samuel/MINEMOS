"""Reversible adapter between ``BlackHoleMemory.query()`` and the backends.

The engine keeps its canonical query loop untouched.  When a non-canonical
backend is selected, ``query()`` calls this adapter first; if the adapter
raises :class:`RetrievalUnavailable`, the engine falls through to the
canonical loop.  That makes a switch back to canonical or an internal backend
failure a safe, observable event rather than a correctness change.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from .config import (CANONICAL, INCREMENTAL, OPTIMIZED, QDRANT,
                     RetrievalConfig, normalize_backend)
from .incremental import IncrementalSnapshotRetriever
from .snapshot import SnapshotContended, SnapshotRetriever


class RetrievalUnavailable(RuntimeError):
    """The optimized backend could not answer; the engine should use canonical."""


class RetrievalAdapter:
    """Owns one retriever for the configured backend and exposes its stats."""

    def __init__(self, mem: Any, config: Optional[RetrievalConfig] = None,
                 backend: Optional[str] = None,
                 qdrant_config: Optional[Any] = None):
        self.mem = mem
        self.config = config if isinstance(config, RetrievalConfig) else (
            RetrievalConfig.from_value(config))
        self.qdrant_config = qdrant_config
        self._backend = normalize_backend(
            backend if backend is not None else self.config.backend)
        self._retriever: Optional[Any] = None
        self._lock = threading.Lock()
        self._errors = 0
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        return self._backend

    def _make_retriever(self) -> Any:
        if self._backend == OPTIMIZED:
            return SnapshotRetriever(
                self.mem,
                max_optimistic_attempts=self.config.max_optimistic_attempts,
                verify_bound=self.config.verify_bound,
                contended_fallback=self.config.contended_fallback)
        if self._backend == INCREMENTAL:
            return IncrementalSnapshotRetriever(
                self.mem,
                max_optimistic_attempts=self.config.max_optimistic_attempts,
                max_incremental_updates=self.config.max_incremental_updates,
                verify_bound=self.config.verify_bound,
                contended_fallback=self.config.contended_fallback)
        if self._backend == QDRANT:
            from .qdrant import QdrantRetriever
            from .qdrant_config import QdrantConfig

            if self.qdrant_config is None:
                raise RetrievalUnavailable(
                    "qdrant backend requires qdrant_config (or QDRANT_URL / "
                    "QDRANT_PATH environment variables)")
            return QdrantRetriever(
                self.mem, QdrantConfig.from_value(self.qdrant_config))
        raise ValueError(
            f"RetrievalAdapter does not serve backend {self._backend!r}")

    def _ensure(self) -> Any:
        with self._lock:
            if self._retriever is None:
                self._retriever = self._make_retriever()
            return self._retriever

    def ensure_retriever(self) -> Any:
        """Build the retriever if needed (used by write-path index sync)."""
        return self._ensure()

    @property
    def retriever(self) -> Optional[Any]:
        return self._retriever

    def set_backend(self, backend: str) -> str:
        value = normalize_backend(backend)
        previous = None
        with self._lock:
            if value != self._backend:
                previous = self._retriever
                self._backend = value
                self._retriever = None
        if previous is not None:
            close = getattr(previous, "close", None)
            if callable(close):
                close()
        return value

    def force_full_rebuild(self) -> None:
        retriever = self._retriever
        if retriever is not None:
            retriever.force_full_rebuild()

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def query(self, query_text: str, top_k: Any = 5,
              as_of: Optional[float] = None,
              active_only: bool = True,
              filters: Optional[Dict[str, Any]] = None,
              hybrid: bool = False):
        if self._backend == CANONICAL:
            # The engine handles this case before calling the adapter; this
            # guard keeps direct adapter use honest.
            raise RetrievalUnavailable("adapter is configured for canonical")
        if filters and self._backend != QDRANT:
            raise RetrievalUnavailable(
                "metadata filters require the qdrant backend or canonical path")
        if hybrid and self._backend != QDRANT:
            raise RetrievalUnavailable(
                "hybrid retrieval requires the qdrant backend")
        try:
            retriever = self._ensure()
            if self._backend == QDRANT:
                return retriever.query(
                    query_text, top_k=top_k, as_of=as_of,
                    active_only=active_only, filters=filters, hybrid=hybrid)
            return retriever.query(query_text, top_k=top_k, as_of=as_of,
                                   active_only=active_only)
        except SnapshotContended as exc:
            # Contention is an expected hand-off, not a backend defect: the
            # engine answers with its exact canonical loop.  Counting it as an
            # error would make normal write-heavy operation look broken.
            if self.config.fallback_to_canonical:
                raise RetrievalUnavailable(
                    f"{self._backend} snapshot contended: {exc}") from exc
            raise
        except RetrievalUnavailable:
            raise
        except Exception as exc:                # noqa: BLE001 - deliberate
            self._errors += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            if self.config.fallback_to_canonical:
                raise RetrievalUnavailable(
                    f"{self._backend} retrieval failed: "
                    f"{type(exc).__name__}: {exc}") from exc
            raise

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        retriever = self._retriever
        payload: Dict[str, Any] = {
            "backend": self._backend,
            "errors": self._errors,
            "last_error": self._last_error,
            "fallback_to_canonical": self.config.fallback_to_canonical,
            "contended_fallback": self.config.contended_fallback,
        }
        if retriever is None:
            payload.update({"strategy": None, "manager": None,
                            "published_version": None})
            return payload
        manager = retriever.manager
        with self.mem._lock:
            published = manager._snapshot
            payload["published_version"] = (
                None if published is None else published.version)
            payload["slots_in_snapshot"] = (
                None if published is None else published.n)
        payload["strategy"] = getattr(retriever, "strategy", None)
        payload["manager"] = dict(manager.stats)
        return payload

    def close(self) -> None:
        with self._lock:
            retriever = self._retriever
            self._retriever = None
        close = getattr(retriever, "close", None)
        if callable(close):
            close()

    def reset_stats(self) -> None:
        self._errors = 0
        self._last_error = None
        retriever = self._retriever
        if retriever is not None:
            retriever.manager.stats = _fresh_stats(retriever.manager.stats)


def _fresh_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    fresh: Dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            fresh[key] = {}
        elif isinstance(value, bool):
            fresh[key] = value
        elif isinstance(value, int):
            fresh[key] = 0
        elif isinstance(value, float):
            fresh[key] = 0.0
        else:
            fresh[key] = value
    return fresh
