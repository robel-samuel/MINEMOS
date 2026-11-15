"""
MINEMOS engine package.

The canonical memory engine, its decision pipeline, persistence, and retrieval
backends live here.  Most applications should import the public surface from
the ``minemos`` package instead of depending on this module layout directly.
"""

from minemos_engine.memory_state import MemoryState, MemorySlot
from minemos_engine.memory_update import absorb, address, novelty, cosine_similarity, MATCH_THRESHOLD
from minemos_engine.structured_candidate import Candidate

def __getattr__(name: str):
    if name == "BlackHoleMemory":
        from minemos_engine.black_hole_memory import BlackHoleMemory
        return BlackHoleMemory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MemoryState",
    "MemorySlot",
    "absorb",
    "address",
    "novelty",
    "cosine_similarity",
    "MATCH_THRESHOLD",
    "Candidate",
    "BlackHoleMemory",
]