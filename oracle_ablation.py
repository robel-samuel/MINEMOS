"""
benchmarks/oracle_ablation.py

Systems:
  A. Existing similarity-driven mechanism (unchanged absorb(), atomic
     encoder at D=4096 -- the repository's most-validated configuration;
     D=64, the production default, was proven collision-prone in Phase
     2A and would confound this experiment with a known-bad
     representation rather than isolating the update mechanism).
  B. No-update baseline (same as every prior phase's convention: always
     insert/evict, never merge).
  C. Oracle-update: identical machinery to A, EXCEPT the merge-vs-new
     decision comes from a concept_id lookup (SAME/DIFFERENT only) in
     place of the cosine-similarity threshold. The oracle never
     produces or sees a vector -- it only ever points at a slot that a
     perfect addressing mechanism could equally have found, because it
     is built directly from ground-truth concept identity, not from any
     external judgment about content.

The core update line, `slot.value = (1 - alpha) * slot.value + alpha * x_t`,
is not reimplemented independently here -- oracle_absorb() below is a
byte-for-byte copy of that single line from phase2/memory_update.py's
absorb(), verified identical by tests/test_oracle_ablation.py rather
than assumed.
"""

from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate, encode_full
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb, address, novelty, cosine_similarity, MATCH_THRESHOLD
from phase2.checkpoint import save_checkpoint
from benchmarks.phase11_dataset import generate_dataset

D = 4096  # repository's most-validated dimensionality (Phase 2A onward)


def oracle_absorb(memory: MemoryState, candidate: Candidate, oracle_slot,
                   dim: int = D) -> dict:
    """
    oracle_slot: an existing MemorySlot OBJECT (SAME), or None (DIFFERENT).
    Tracking the slot by object identity, not list index, is deliberate:
    MemoryState.evict_oldest() does list.pop(idx), which shifts every
    later slot's index down by one. A cache keyed by index would
    silently point at the WRONG slot after any eviction elsewhere in
    memory -- caught and fixed here before running anything, not after
    seeing corrupted results.
    """
    x_t = encode_full(candidate, dim=dim)

    if oracle_slot is not None:
        sim = cosine_similarity(x_t, oracle_slot.key)
        n_t = novelty(sim, memory_empty=False)
        alpha_t = n_t
        # --- byte-identical copy of memory_update.absorb()'s update line ---
        oracle_slot.value = (1 - alpha_t) * oracle_slot.value + alpha_t * x_t
        # ---------------------------------------------------------------
        oracle_slot.confidence = max(oracle_slot.confidence, candidate.confidence)
        oracle_slot.timestamp = candidate.timestamp
        oracle_slot.update_count += 1
        return dict(action="update", slot=oracle_slot, similarity=sim, novelty=n_t)

    if not memory.is_full():
        memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                   debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        return dict(action="insert", slot=memory.slots[-1], similarity=0.0, novelty=1.0)

    memory.evict_oldest()
    memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
               debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    return dict(action="evict_insert", slot=memory.slots[-1], similarity=0.0, novelty=1.0)


def run_system_A(observations, capacity: int, dim: int = D) -> dict:
    mem = MemoryState(capacity=capacity, dim=dim)
    t0 = time.perf_counter()
    n_updates = n_inserts = n_evicts = n_rejects = 0
    for obs in observations:
        r = absorb(mem, obs.candidate, threshold=MATCH_THRESHOLD)
        if r["action"] == "update": n_updates += 1
        elif r["action"] == "insert": n_inserts += 1
        elif r["action"] == "evict_insert": n_evicts += 1
        else: n_rejects += 1
    elapsed = time.perf_counter() - t0
    return dict(system="A_similarity", memory=mem, n_updates=n_updates, n_inserts=n_inserts,
                n_evicts=n_evicts, n_rejects=n_rejects, wall_seconds=elapsed)


def run_system_B(observations, capacity: int, dim: int = D) -> dict:
    mem = MemoryState(capacity=capacity, dim=dim)
    t0 = time.perf_counter()
    n_evicts = 0
    for obs in observations:
        x_t = encode_full(obs.candidate, dim=dim)
        if mem.is_full():
            mem.evict_oldest()
            n_evicts += 1
        mem.add(x_t, x_t.copy(), obs.candidate.confidence, obs.candidate.timestamp,
                 debug_subject=obs.candidate.subject, debug_predicate=obs.candidate.predicate)
    elapsed = time.perf_counter() - t0
    return dict(system="B_no_update", memory=mem, n_updates=0, n_inserts=len(observations) - n_evicts,
                n_evicts=n_evicts, n_rejects=0, wall_seconds=elapsed)


def run_system_C(observations, capacity: int, dim: int = D) -> dict:
    mem = MemoryState(capacity=capacity, dim=dim)
    concept_to_slot = {}  # oracle's ONLY state: concept_id -> MemorySlot object
    t0 = time.perf_counter()
    n_updates = n_inserts = n_evicts = n_rejects = 0
    for obs in observations:
        existing_slot = concept_to_slot.get(obs.concept_id)
        if existing_slot is not None and not any(existing_slot is s for s in mem.slots):
            existing_slot = None  # its slot was evicted -- treat as unseen again
        r = oracle_absorb(mem, obs.candidate, existing_slot, dim=dim)
        if r["action"] == "update":
            n_updates += 1
        elif r["action"] == "insert":
            n_inserts += 1
            concept_to_slot[obs.concept_id] = r["slot"]
        elif r["action"] == "evict_insert":
            n_evicts += 1
            concept_to_slot[obs.concept_id] = r["slot"]
    elapsed = time.perf_counter() - t0
    return dict(system="C_oracle", memory=mem, n_updates=n_updates, n_inserts=n_inserts,
                n_evicts=n_evicts, n_rejects=n_rejects, wall_seconds=elapsed,
                concept_to_slot=concept_to_slot)
