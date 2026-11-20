"""MINEMOS — persistent memory intelligence for AI applications.

Public developer surface:

    from minemos import BlackHoleMemory, Candidate

``BlackHoleMemory`` is imported lazily so that ``import minemos`` stays cheap;
the first attribute access loads the engine and its encoder models.
"""

from minemos_engine.memory_state import MemoryState, MemorySlot
from minemos_engine.structured_candidate import Candidate
from minemos_engine.answer import ANSWER_STATUSES, answer_contract
from minemos_engine.retrieval import (
    QdrantConfig,
    RetrievalConfig,
    RetrievalConfigurationError,
)

try:
    from importlib.metadata import version as _distribution_version
    __version__ = _distribution_version("minemos")
except Exception:  # running from a checkout before installation
    __version__ = "0.1.0"

__all__ = [
    "BlackHoleMemory",
    "Candidate",
    "MemoryState",
    "MemorySlot",
    "QdrantConfig",
    "RetrievalConfig",
    "RetrievalConfigurationError",
    "ANSWER_STATUSES",
    "answer_contract",
    "ResolutionConfig",
    "ResolutionRecord",
    "Resolver",
    "ShadowRunner",
    "SourceMessage",
    "EngineSink",
    "SemanticIngestionError",
    "SemanticIngestionReport",
    "SemanticIngestor",
    "ServiceSink",
    "__version__",
]

_RESOLUTION_EXPORTS = frozenset({
    "ResolutionConfig", "ResolutionRecord", "Resolver", "ShadowRunner",
    "SourceMessage",
})

_SEMANTIC_EXPORTS = frozenset({
    "EngineSink", "SemanticIngestionError", "SemanticIngestionReport",
    "SemanticIngestor", "ServiceSink",
})


def __getattr__(name: str):
    if name == "BlackHoleMemory":
        from minemos_engine.black_hole_memory import BlackHoleMemory
        return BlackHoleMemory
    if name in _RESOLUTION_EXPORTS:
        from minemos import resolution

        return getattr(resolution, name)
    if name in _SEMANTIC_EXPORTS:
        from minemos import semantic

        return getattr(semantic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")