"""
benchmarks/phase24_retention.py

Phase 24 Retention Policies:
Hardening memory retention against premature eviction under capacity pressure.

Compares:
  1. fifo: Oldest created_at baseline (unmodified FIFO).
  2. recency: Least-recently-updated (LRU on slot.timestamp).
  3. utility: Online utility score combining consolidation count, recency decay, and confidence.
  4. verification_aware: Explicitly prioritizes slots with corroborated verification history.
  5. tiered: Two-tier architecture combining provisional working memory and verified long-term memory.
  6. oracle: Diagnostic upper bound (uses target IDs for failure attribution only; never claimed as practical).

Strict constraints:
  - Zero future knowledge.
  - Zero test labels or evaluation target leakage into practical policies.
  - Strictly enforces capacity bounds (no unbounded growth).
"""

from __future__ import annotations
import math
from typing import Optional, Set
import numpy as np

from phase2.memory_state import MemoryState, MemorySlot
from benchmarks.phase18_experiment import utility_score
from benchmarks.phase19_experiment import contradiction_aware_utility


def verification_aware_utility(
    slot: MemorySlot,
    current_time: float,
    contradiction_count: int = 0,
    tau: float = 50.0
) -> float:
    """
    Utility metric that strongly weights empirical verification / consolidation history:
      - Corroboration score: min(slot.update_count / 3.0, 1.0) * 0.55
      - Recency decay: exp(-max(0, current_time - slot.timestamp) / tau) * 0.25
      - Confidence: slot.confidence * 0.20
      - Contradiction shielding bonus: slots with >= 2 updates that resist contradiction
    """
    # Verifications: slots with >= 2 updates have been corroborated by NLI
    u_verif = min(float(slot.update_count) / 3.0, 1.0) * 0.55
    dt = max(0.0, float(current_time) - float(slot.timestamp))
    u_recency = float(np.exp(-dt / tau)) * 0.25
    u_confidence = float(slot.confidence) * 0.20
    score = u_verif + u_recency + u_confidence

    if slot.update_count >= 2 and contradiction_count > 0:
        score += 0.15 * min(contradiction_count, 3)
    elif slot.update_count == 1 and contradiction_count > 0:
        score -= 0.05 * min(contradiction_count, 2)

    return max(0.0, score)


def evict_slot_phase24(
    memory: MemoryState,
    eviction_policy: str,
    current_time: float,
    slot_texts: dict[int, str],
    slot_dense_embs: list[np.ndarray],
    slot_contra_counts: Optional[dict[int, int]] = None,
    slot_contra_anchors: Optional[dict[int, list[tuple[np.ndarray, float]]]] = None,
    protected_slot_ids: Optional[Set[int]] = None,
    max_longterm_ratio: float = 0.80
) -> tuple[int, MemorySlot]:
    """
    Evicts a single slot from memory according to the specified policy.
    
    Returns:
      (evict_idx, evicted_slot)
    """
    if not memory.slots:
        raise RuntimeError("Cannot evict from empty memory state")

    n_slots = len(memory.slots)
    contra_counts = slot_contra_counts or {}

    # 1. Oracle policy (diagnostic upper bound only)
    if eviction_policy == "oracle":
        protected = protected_slot_ids or set()
        candidate_indices = [idx for idx, s in enumerate(memory.slots) if id(s) not in protected]
        if not candidate_indices:
            candidate_indices = list(range(n_slots))
        evict_idx = min(candidate_indices, key=lambda idx: memory.slots[idx].created_at)

    # 2. FIFO baseline
    elif eviction_policy == "fifo":
        evict_idx = min(range(n_slots), key=lambda idx: memory.slots[idx].created_at)

    # 3. Recency (LRU on slot.timestamp)
    elif eviction_policy == "recency":
        evict_idx = min(range(n_slots), key=lambda idx: memory.slots[idx].timestamp)

    # 4. Utility-based retention (Phase 18/19 utility)
    elif eviction_policy == "utility":
        evict_idx = min(
            range(n_slots),
            key=lambda idx: contradiction_aware_utility(
                memory.slots[idx], current_time, contra_counts.get(id(memory.slots[idx]), 0)
            )
        )

    # 5. Verification-aware retention
    elif eviction_policy == "verification_aware":
        evict_idx = min(
            range(n_slots),
            key=lambda idx: verification_aware_utility(
                memory.slots[idx], current_time, contra_counts.get(id(memory.slots[idx]), 0)
            )
        )

    # 6. Tiered retention policy
    # Partitions memory into Tier 0 (provisional: update_count <= 1) and Tier 1 (consolidated: update_count >= 2)
    # Never allows Tier 1 to exceed max_longterm_ratio * capacity, protecting working memory quota.
    elif eviction_policy == "tiered":
        tier0_indices = [idx for idx, s in enumerate(memory.slots) if s.update_count <= 1]
        tier1_indices = [idx for idx, s in enumerate(memory.slots) if s.update_count > 1]

        max_tier1_slots = int(math.floor(memory.capacity * max_longterm_ratio))

        if tier0_indices and len(tier1_indices) <= max_tier1_slots:
            # Evict lowest-utility provisional slot from Tier 0
            evict_idx = min(
                tier0_indices,
                key=lambda idx: verification_aware_utility(
                    memory.slots[idx], current_time, contra_counts.get(id(memory.slots[idx]), 0)
                )
            )
        elif tier1_indices:
            # Tier 0 is empty or Tier 1 exceeded quota: evict lowest-utility consolidated slot from Tier 1
            evict_idx = min(
                tier1_indices,
                key=lambda idx: verification_aware_utility(
                    memory.slots[idx], current_time, contra_counts.get(id(memory.slots[idx]), 0)
                )
            )
        else:
            # Fallback (all slots in Tier 0)
            evict_idx = min(
                range(n_slots),
                key=lambda idx: verification_aware_utility(
                    memory.slots[idx], current_time, contra_counts.get(id(memory.slots[idx]), 0)
                )
            )
    else:
        raise ValueError(f"Unknown eviction policy: '{eviction_policy}'")

    evicted_slot = memory.slots.pop(evict_idx)
    slot_id = id(evicted_slot)
    if slot_id in slot_texts:
        del slot_texts[slot_id]
    if slot_dense_embs and evict_idx < len(slot_dense_embs):
        slot_dense_embs.pop(evict_idx)
    if slot_contra_counts and slot_id in slot_contra_counts:
        del slot_contra_counts[slot_id]
    if slot_contra_anchors and slot_id in slot_contra_anchors:
        del slot_contra_anchors[slot_id]

    return evict_idx, evicted_slot
