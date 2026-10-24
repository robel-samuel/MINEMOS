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
import threading
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
    # Phase 25 additions: temporal validity window & semantic metadata
    valid_from: float = 0.0
    valid_until: Optional[float] = None
    text: str = ""
    dense_emb: Optional[np.ndarray] = None
    contra_count: int = 0
    superseded_by: Optional[int] = None
    supersedes: Optional[int] = None

    @property
    def is_active(self) -> bool:
        return self.valid_until is None

    @property
    def active(self) -> bool:
        return self.valid_until is None


class MemoryState:
    def __init__(self, capacity: int, dim: int = DIM):
        self.capacity = capacity
        self.dim = dim
        self.slots: list[MemorySlot] = []
        self._lock = threading.RLock()

    def is_full(self) -> bool:
        with self._lock:
            return len(self.slots) >= self.capacity

    def active_slots(self, as_of: Optional[float] = None) -> list[MemorySlot]:
        """Returns slots that were active at timestamp as_of (default: current active)."""
        with self._lock:
            if as_of is None:
                return [s for s in self.slots if s.is_active]
            return [
                s for s in self.slots
                if s.valid_from <= as_of and (s.valid_until is None or s.valid_until > as_of)
            ]

    def add(self, key: np.ndarray, value: np.ndarray, confidence: float,
            timestamp: float, debug_subject: str = None,
            debug_predicate: str = None,
            text: str = "", dense_emb: Optional[np.ndarray] = None,
            valid_from: Optional[float] = None,
            valid_until: Optional[float] = None,
            supersedes: Optional[int] = None) -> MemorySlot:
        with self._lock:
            if self.is_full():
                raise CapacityExceeded(
                    f"memory is at capacity ({self.capacity} slots); "
                    f"caller must evict before adding"
                )
            slot = MemorySlot(key=key.copy(), value=value.copy(),
                              confidence=confidence, timestamp=timestamp,
                              created_at=timestamp, update_count=1,
                              debug_subject=debug_subject, debug_predicate=debug_predicate,
                              valid_from=timestamp if valid_from is None else valid_from,
                              valid_until=valid_until,
                              text=text, dense_emb=dense_emb,
                              supersedes=supersedes)
            self.slots.append(slot)
            return slot

    def evict_oldest(self) -> Optional[MemorySlot]:
        """FIFO eviction policy: remove the slot with the oldest
        `created_at`. Simple and transparent -- explicitly NOT claimed to
        be optimal, just the documented v0.1 baseline policy."""
        with self._lock:
            if not self.slots:
                return None
            idx = min(range(len(self.slots)), key=lambda i: self.slots[i].created_at)
            return self.slots.pop(idx)

    def evict_tiered(self, current_time: float, max_longterm_ratio: float = 0.80) -> Optional[MemorySlot]:
        """
        Phase 25 Tiered Retention:
        Protects consolidated slots (update_count >= 2) by evicting provisional
        singletons (update_count == 1) first, subject to a long-term quota.
        """
        with self._lock:
            if not self.slots:
                return None
            import math
            tier0_indices = [idx for idx, s in enumerate(self.slots) if s.update_count <= 1]
            tier1_indices = [idx for idx, s in enumerate(self.slots) if s.update_count > 1]
            max_tier1_slots = int(math.floor(self.capacity * max_longterm_ratio))

            def slot_utility(s: MemorySlot) -> float:
                u_cons = min(float(s.update_count) / 3.0, 1.0) * 0.55
                dt = max(0.0, float(current_time) - float(s.timestamp))
                u_rec = float(np.exp(-dt / 50.0)) * 0.25
                u_conf = float(s.confidence) * 0.20
                score = u_cons + u_rec + u_conf
                if s.contra_count > 0 and s.update_count >= 2:
                    score += 0.15 * min(s.contra_count, 3)
                return score

            if tier0_indices and len(tier1_indices) <= max_tier1_slots:
                evict_idx = min(tier0_indices, key=lambda idx: slot_utility(self.slots[idx]))
            elif tier1_indices:
                evict_idx = min(tier1_indices, key=lambda idx: slot_utility(self.slots[idx]))
            else:
                evict_idx = min(range(len(self.slots)), key=lambda idx: slot_utility(self.slots[idx]))

            return self.slots.pop(evict_idx)

    def evict(self, policy: str = "fifo", current_time: float = 0.0) -> Optional[MemorySlot]:
        """Evicts a slot according to policy ('fifo', 'recency', 'tiered')."""
        with self._lock:
            if not self.slots:
                return None
            if policy == "fifo":
                return self.evict_oldest()
            elif policy == "recency":
                idx = min(range(len(self.slots)), key=lambda i: self.slots[i].timestamp)
                return self.slots.pop(idx)
            elif policy == "tiered":
                return self.evict_tiered(current_time=current_time)
            else:
                return self.evict_oldest()


    # -- footprint measurement --------------------------------------------

    def footprint_bytes(self) -> dict:
        """
        Honest accounting of persistent bytes, broken out by component so
        nothing is hidden inside a single number. All vectors are stored
        as float32 (4 bytes/dim); confidence and timestamps as float32/
        float64 respectively for realistic accounting.
        """
        with self._lock:
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

