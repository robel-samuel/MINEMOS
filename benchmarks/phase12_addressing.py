"""
benchmarks/phase12_addressing.py

System A: Phase 11's oracle_absorb, unchanged (frozen-key control).
System B: identical update, but after a SAME-update the slot's KEY is
          replaced with the post-update VALUE (re-keyed to reflect the
          consolidated representation, not whichever wording arrived
          first).
System C: identical update, primary key stays frozen (like A), but the
          incoming candidate's encoding is additionally stored as an
          alias address for that slot. Aliases are pure addressing
          bookkeeping -- never blended into value, never counted as
          slots, never create new concepts.

All three reuse the exact same update line from phase2/memory_update.py
(verified via test, not re-derived) -- only what happens to the KEY
differs.
"""

from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate, encode_full, encode_query
from phase2.memory_state import MemoryState
from phase2.memory_update import novelty, cosine_similarity
from benchmarks.oracle_ablation import oracle_absorb, D


def system_A_absorb(memory, candidate, oracle_slot, dim=D):
    """Frozen-key control -- literally Phase 11's function, unmodified."""
    return oracle_absorb(memory, candidate, oracle_slot, dim=dim)


def system_B_absorb(memory, candidate, oracle_slot, dim=D):
    x_t = encode_full(candidate, dim=dim)
    if oracle_slot is not None:
        sim = cosine_similarity(x_t, oracle_slot.key)
        n_t = novelty(sim, memory_empty=False)
        alpha_t = n_t
        # --- identical update line to memory_update.absorb() ---
        oracle_slot.value = (1 - alpha_t) * oracle_slot.value + alpha_t * x_t
        # ---------------------------------------------------------------
        oracle_slot.confidence = max(oracle_slot.confidence, candidate.confidence)
        oracle_slot.timestamp = candidate.timestamp
        oracle_slot.update_count += 1
        # THE ONLY DIFFERENCE FROM SYSTEM A: re-key from the updated value
        oracle_slot.key = oracle_slot.value.copy()
        return dict(action="update", slot=oracle_slot, similarity=sim, novelty=n_t)

    if not memory.is_full():
        memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                   debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        return dict(action="insert", slot=memory.slots[-1], similarity=0.0, novelty=1.0)

    memory.evict_oldest()
    memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
               debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    return dict(action="evict_insert", slot=memory.slots[-1], similarity=0.0, novelty=1.0)


def system_C_absorb(memory, candidate, oracle_slot, alias_store: dict, dim=D):
    """alias_store: dict[id(slot), list[np.ndarray]] -- addressing-only
    bookkeeping, keyed by slot identity (consistent with the Phase 11
    fix for list-index instability under eviction)."""
    x_t = encode_full(candidate, dim=dim)
    if oracle_slot is not None:
        sim = cosine_similarity(x_t, oracle_slot.key)
        n_t = novelty(sim, memory_empty=False)
        alpha_t = n_t
        # --- identical update line to memory_update.absorb() ---
        oracle_slot.value = (1 - alpha_t) * oracle_slot.value + alpha_t * x_t
        # ---------------------------------------------------------------
        oracle_slot.confidence = max(oracle_slot.confidence, candidate.confidence)
        oracle_slot.timestamp = candidate.timestamp
        oracle_slot.update_count += 1
        # THE ONLY DIFFERENCE FROM SYSTEM A: primary key untouched, but
        # this wording's encoding is retained as an additional address
        alias_store.setdefault(id(oracle_slot), []).append(x_t.copy())
        return dict(action="update", slot=oracle_slot, similarity=sim, novelty=n_t)

    if not memory.is_full():
        memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                   debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        return dict(action="insert", slot=memory.slots[-1], similarity=0.0, novelty=1.0)

    memory.evict_oldest()
    memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
               debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    return dict(action="evict_insert", slot=memory.slots[-1], similarity=0.0, novelty=1.0)


def oracle_recall_scored(memory, subject, predicate, dim=D, alias_store: dict = None):
    """Generic scoring recall used for ALL THREE systems -- checks each
    slot's primary key, and (if alias_store given) every alias too,
    taking the max. This is the ONLY place addressing logic differs
    between systems; identical code path for A/B (alias_store=None) and
    C (alias_store populated)."""
    q = encode_query(subject, predicate, dim=dim)
    if not memory.slots:
        return None, 0.0
    best_slot, best_sim = None, -1.0
    for slot in memory.slots:
        sim = cosine_similarity(q, slot.key)
        if alias_store is not None:
            for a in alias_store.get(id(slot), []):
                asim = cosine_similarity(q, a)
                if asim > sim:
                    sim = asim
        if sim > best_sim:
            best_sim, best_slot = sim, slot
    return best_slot, best_sim


def run_system(observations, capacity: int, absorb_fn: str, dim: int = D):
    """absorb_fn in {'A', 'B', 'C'}."""
    mem = MemoryState(capacity=capacity, dim=dim)
    concept_to_slot = {}
    alias_store = {} if absorb_fn == "C" else None

    n_updates = n_inserts = n_evicts = 0
    t0 = time.perf_counter()
    for obs in observations:
        existing_slot = concept_to_slot.get(obs.concept_id)
        if existing_slot is not None and not any(existing_slot is s for s in mem.slots):
            existing_slot = None

        if absorb_fn == "A":
            r = system_A_absorb(mem, obs.candidate, existing_slot, dim=dim)
        elif absorb_fn == "B":
            r = system_B_absorb(mem, obs.candidate, existing_slot, dim=dim)
        else:
            r = system_C_absorb(mem, obs.candidate, existing_slot, alias_store, dim=dim)

        if r["action"] == "update":
            n_updates += 1
        elif r["action"] == "insert":
            n_inserts += 1
            concept_to_slot[obs.concept_id] = r["slot"]
        elif r["action"] == "evict_insert":
            n_evicts += 1
            concept_to_slot[obs.concept_id] = r["slot"]
    elapsed = time.perf_counter() - t0

    return dict(system=absorb_fn, memory=mem, concept_to_slot=concept_to_slot,
                alias_store=alias_store, n_updates=n_updates, n_inserts=n_inserts,
                n_evicts=n_evicts, wall_seconds=elapsed)
