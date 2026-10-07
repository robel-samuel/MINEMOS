"""
Phase 2 — memory state.

M_t = {m_1, ..., m_N}, m_i = (k_i, v_i, c_i, t_i)

  k_i = key/address, FROZEN at slot creation (full candidate encoding at
        the time the slot was created). Does not drift.
  v_i = stored value, DRIFTS over time via the update equation in
        memory_update.py as new candidates map to this slot.
  c_i = confidence (scalar)
  t_i = last-updated timestamp

Capacity N is enforced, not aspirational: MemoryState.add() raises
CapacityExceeded if the state is full and the caller doesn't supply an
eviction policy. memory_update.py supplies one (FIFO on oldest
timestamp) so capacity pressure is handled explicitly, not silently
ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from phase2.structured_candidate import DIM


class CapacityExceeded(Exception):
    pass


@dataclass
class MemorySlot:
    key: np.ndarray       # k_i, frozen at creation
    value: np.ndarray     # v_i, drifts via update
    confidence: float     # c_i
    timestamp: float      # t_i, last update time
    created_at: float     # for FIFO eviction / age-based recall tests
    update_count: int = 0
    # subject/predicate kept ONLY for evaluation/debugging -- NOT counted
    # as part of the persistent memory footprint (see MemoryState.footprint_bytes).
    debug_subject: Optional[str] = None
    debug_predicate: Optional[str] = None


class MemoryState:
    def __init__(self, capacity: int, dim: int = DIM):
        self.capacity = capacity
        self.dim = dim
        self.slots: list[MemorySlot] = []

    def is_full(self) -> bool:
        return len(self.slots) >= self.capacity

    def add(self, key: np.ndarray, value: np.ndarray, confidence: float,
            timestamp: float, debug_subject: str = None,
            debug_predicate: str = None) -> MemorySlot:
        if self.is_full():
            raise CapacityExceeded(
                f"memory is at capacity ({self.capacity} slots); "
                f"caller must evict before adding"
            )
        slot = MemorySlot(key=key.copy(), value=value.copy(),
                           confidence=confidence, timestamp=timestamp,
                           created_at=timestamp, update_count=1,
                           debug_subject=debug_subject, debug_predicate=debug_predicate)
        self.slots.append(slot)
        return slot

    def evict_oldest(self) -> Optional[MemorySlot]:
        """FIFO eviction policy: remove the slot with the oldest
        `created_at`. Simple and transparent -- explicitly NOT claimed to
        be optimal, just the documented v0.1 baseline policy."""
        if not self.slots:
            return None
        idx = min(range(len(self.slots)), key=lambda i: self.slots[i].created_at)
        return self.slots.pop(idx)

    # -- footprint measurement --------------------------------------------

    def footprint_bytes(self) -> dict:
        """
        Honest accounting of persistent bytes, broken out by component so
        nothing is hidden inside a single number. All vectors are stored
        as float32 (4 bytes/dim); confidence and timestamps as float32/
        float64 respectively for realistic accounting.
        """
        n = len(self.slots)
        key_bytes = n * self.dim * 4
        value_bytes = n * self.dim * 4
        confidence_bytes = n * 4
        timestamp_bytes = n * 8  # float64, matching checkpoint.py's storage
        # update_count + created_at are bookkeeping metadata, counted too
        metadata_bytes = n * (4 + 8)
        total = key_bytes + value_bytes + confidence_bytes + timestamp_bytes + metadata_bytes
        return {
            "n_slots": n,
            "key_bytes": key_bytes,
            "value_bytes": value_bytes,
            "confidence_bytes": confidence_bytes,
            "timestamp_bytes": timestamp_bytes,
            "metadata_bytes": metadata_bytes,
            "payload_bytes": value_bytes,  # "payload only" = the value vectors
            "total_bytes": total,           # everything: keys+values+conf+time+meta
        }
