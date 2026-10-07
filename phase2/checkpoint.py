"""
Phase 2 — persistence.

Saves the memory state as a single .npz (numpy's binary archive format)
containing dense arrays for keys, values, confidences, timestamps,
created_at, and update_counts. This is a fair, honest representation of
"real persistent bytes" -- raw float arrays, no JSON ASCII bloat, no
extra compression algorithm doing invisible work. os.path.getsize() on
the resulting file is what's reported as checkpoint size; it will be
close to (but not identical to, due to .npz's small format overhead) the
MemoryState.footprint_bytes() total.
"""

from __future__ import annotations

import numpy as np

from phase2.memory_state import MemoryState, MemorySlot


def save_checkpoint(memory: MemoryState, path: str):
    n = len(memory.slots)
    keys = np.zeros((n, memory.dim), dtype=np.float32)
    values = np.zeros((n, memory.dim), dtype=np.float32)
    confidences = np.zeros(n, dtype=np.float32)
    timestamps = np.zeros(n, dtype=np.float64)
    created_ats = np.zeros(n, dtype=np.float64)
    update_counts = np.zeros(n, dtype=np.int32)

    for i, slot in enumerate(memory.slots):
        keys[i] = slot.key
        values[i] = slot.value
        confidences[i] = slot.confidence
        timestamps[i] = slot.timestamp
        created_ats[i] = slot.created_at
        update_counts[i] = slot.update_count

    np.savez(path, keys=keys, values=values, confidences=confidences,
              timestamps=timestamps, created_ats=created_ats,
              update_counts=update_counts, capacity=memory.capacity, dim=memory.dim)


def load_checkpoint(path: str) -> MemoryState:
    data = np.load(path)
    capacity = int(data["capacity"])
    dim = int(data["dim"])
    memory = MemoryState(capacity=capacity, dim=dim)
    n = data["keys"].shape[0]
    for i in range(n):
        slot = MemorySlot(
            key=data["keys"][i], value=data["values"][i],
            confidence=float(data["confidences"][i]),
            timestamp=float(data["timestamps"][i]),
            created_at=float(data["created_ats"][i]),
            update_count=int(data["update_counts"][i]),
        )
        memory.slots.append(slot)
    return memory
