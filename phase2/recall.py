"""
Phase 2 — recall.

Given a partial query (subject, predicate), build the query encoding
(no object -- see structured_candidate.encode_query) and find the
best-matching slot by the same cosine similarity used for addressing.

DECODING NOTE (read before trusting any "exact recall accuracy" number):
the memory itself only stores vectors -- it has no way to turn a value
vector back into a human-readable object string on its own. To score
recall accuracy in the benchmarks below, we decode the returned value
vector by nearest-neighbor match against the known vocabulary of object
strings used to build that benchmark's dataset. This decode step is
PART OF EVALUATION ONLY. It is not persisted, not shipped, and not
counted in any memory-footprint number reported in the benchmarks or the
report. A deployed system would either return the raw vector for
downstream similarity use, or would need a real decode mechanism as a
separate, explicitly-costed component -- this experiment does not build
or claim credit for one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from phase2.structured_candidate import encode_query
from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import cosine_similarity


def recall(memory: MemoryState, subject: str, predicate: str) -> dict:
    q = encode_query(subject, predicate, dim=memory.dim)
    if not memory.slots:
        return dict(found=False, slot_index=None, similarity=0.0, slot=None)
    sims = [cosine_similarity(q, slot.key) for slot in memory.slots]
    best_idx = int(np.argmax(sims))
    return dict(found=True, slot_index=best_idx, similarity=sims[best_idx],
                slot=memory.slots[best_idx])


def decode_object(value_vector: np.ndarray, vocabulary: dict[str, np.ndarray]) -> Optional[str]:
    """EVALUATION-ONLY decoder -- see module docstring. `vocabulary` maps
    known object strings to their encode_full-style vectors (built by the
    benchmark from its own ground truth, not from the memory itself)."""
    if not vocabulary:
        return None
    best_key, best_sim = None, -1.0
    for obj_str, obj_vec in vocabulary.items():
        s = cosine_similarity(value_vector, obj_vec)
        if s > best_sim:
            best_sim, best_key = s, obj_str
    return best_key
