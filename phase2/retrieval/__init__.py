"""Optional retrieval backends for MINEMOS.

Canonical ``BlackHoleMemory.query()`` is the reference implementation and the
default.  Only :mod:`phase2.retrieval.config` (stdlib only) is imported eagerly;
the retriever modules load lazily on first access, so constructing an engine
with the default canonical backend never imports the optimized stack.
"""

from .config import (BACKENDS, CANONICAL, CANONICAL_ONLY, CONTENTION_FALLBACKS,
                     DEFAULT_RETRIEVAL_MODE, INCREMENTAL, LEGACY, OPTIMIZED,
                     PRODUCTION, QDRANT, RETRIEVAL_MODES, RetrievalConfig,
                     RetrievalConfigurationError, normalize_backend,
                     normalize_mode)
from .qdrant_config import QdrantConfig

__all__ = [
    "BACKENDS",
    "CANONICAL",
    "CANONICAL_ONLY",
    "CONTENTION_FALLBACKS",
    "DEFAULT_RETRIEVAL_MODE",
    "INCREMENTAL",
    "LEGACY",
    "OPTIMIZED",
    "PRODUCTION",
    "QDRANT",
    "RETRIEVAL_MODES",
    "IncrementalSnapshotRetriever",
    "QdrantConfig",
    "QdrantRetriever",
    "QdrantUnavailable",
    "RetrievalAdapter",
    "RetrievalConfig",
    "RetrievalConfigurationError",
    "RetrievalUnavailable",
    "SnapshotContended",
    "SnapshotRetriever",
    "VectorizedRetriever",
    "normalize_backend",
    "normalize_mode",
]

_LAZY_EXPORTS = {
    "RetrievalAdapter": (".adapter", "RetrievalAdapter"),
    "RetrievalUnavailable": (".adapter", "RetrievalUnavailable"),
    "SnapshotRetriever": (".snapshot", "SnapshotRetriever"),
    "SnapshotContended": (".snapshot", "SnapshotContended"),
    "VectorizedRetriever": (".snapshot", "VectorizedRetriever"),
    "IncrementalSnapshotRetriever": (".incremental",
                                     "IncrementalSnapshotRetriever"),
    "QdrantRetriever": (".qdrant", "QdrantRetriever"),
    "QdrantUnavailable": (".qdrant", "QdrantUnavailable"),
    "_slice_count": (".snapshot", "_slice_count"),
}


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
