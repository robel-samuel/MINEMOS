"""
MINEMOS: Experimental persistent-memory architecture for language-model systems.
"""

from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import absorb, address, novelty, cosine_similarity
from phase2.structured_candidate import Candidate

def __getattr__(name: str):
    if name == "BlackHoleMemory":
        from phase2.black_hole_memory import BlackHoleMemory
        return BlackHoleMemory
    if name == "MemoryStore":
        from core.memory import MemoryStore
        return MemoryStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BlackHoleMemory",
    "MemoryStore",
    "MemoryState",
    "MemorySlot",
    "absorb",
    "address",
    "novelty",
    "cosine_similarity",
    "Candidate",
]

