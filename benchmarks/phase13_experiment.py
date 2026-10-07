"""
benchmarks/phase13_experiment.py

System A: frozen-key, real (imperfect) similarity signal -- literally
    memory_update.absorb(), unmodified, called with the field-aware
    lexical encoder and the unchanged repository default threshold (0.75).
System B: identical to A, except after a successful update the slot's
    key is replaced with the post-update value (same re-keying rule as
    Phase 12's System B, reimplemented here against REAL addressing
    instead of the oracle).
System C: oracle + frozen key (Phase 11's oracle_absorb, reused directly).
System D: oracle + re-key (Phase 12's system_B_absorb, reused directly).

ENCODER/THRESHOLD JUSTIFICATION (verified before running anything, not
assumed): field-aware lexical at MATCH_THRESHOLD=0.75 (the unchanged
repository default) was checked directly against one paraphrase pair,
one hard-negative pair, one contradiction pair, and one unrelated pair.
Result: paraphrase correctly merges (sim=0.866), contradiction
INCORRECTLY merges (sim=0.750, right at the threshold), hard-negative
correctly does not merge (0.667), unrelated correctly does not merge
(0.0). This is exactly the "correct/ambiguous/incorrect" mix the task
requires. Atomic hashing at the same unchanged threshold was checked
too and produces ZERO merges beyond exact duplicates (consistent with
Phase 6), which would make error-propagation untestable -- so atomic
was rejected as unsuitable for this specific experiment, not because
it is a worse encoder in general.
"""

from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate, encode_query
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb, address, novelty, cosine_similarity, MATCH_THRESHOLD
from phase2.encoders import FieldAwareLexicalEncoder
from benchmarks.oracle_ablation import oracle_absorb
from benchmarks.phase12_addressing import system_B_absorb as oracle_rekey_absorb

D = 4096
ENCODER = FieldAwareLexicalEncoder()


def real_rekey_absorb(memory, candidate, threshold=MATCH_THRESHOLD, encoder=ENCODER, dim=D):
    """System B: identical addressing/decision logic to absorb(), with
    one addition -- re-key on successful update. The merge/insert
    decision itself is NOT oracle-driven; it uses real cosine similarity
    against the real encoder, exactly like System A."""
    x_t = encoder.encode_full(candidate, dim=dim)
    best_idx, max_sim = address(memory, x_t)
    n_t = novelty(max_sim, memory_empty=(best_idx is None))
    alpha_t = n_t

    if best_idx is not None and max_sim >= threshold:
        slot = memory.slots[best_idx]
        # --- identical update line to memory_update.absorb() ---
        slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
        # ---------------------------------------------------------------
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = candidate.timestamp
        slot.update_count += 1
        slot.key = slot.value.copy()  # THE ONLY DIFFERENCE FROM SYSTEM A
        return dict(action="update", slot_index=best_idx, similarity=max_sim, novelty=n_t)

    if not memory.is_full():
        memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                   debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        return dict(action="insert", slot_index=len(memory.slots) - 1, similarity=max_sim, novelty=n_t)

    memory.evict_oldest()
    memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
               debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    return dict(action="evict_insert", slot_index=len(memory.slots) - 1, similarity=max_sim, novelty=n_t)


def run_system_A(observations, capacity: int, dim: int = D):
    mem = MemoryState(capacity=capacity, dim=dim)
    t0 = time.perf_counter()
    n_updates = n_inserts = n_evicts = 0
    for obs in observations:
        r = absorb(mem, obs.candidate, threshold=MATCH_THRESHOLD, encoder=ENCODER)
        if r["action"] == "update": n_updates += 1
        elif r["action"] == "insert": n_inserts += 1
        elif r["action"] == "evict_insert": n_evicts += 1
    return dict(system="A_frozen_real", memory=mem, n_updates=n_updates, n_inserts=n_inserts,
                n_evicts=n_evicts, wall_seconds=time.perf_counter() - t0)


def run_system_B(observations, capacity: int, dim: int = D):
    mem = MemoryState(capacity=capacity, dim=dim)
    t0 = time.perf_counter()
    n_updates = n_inserts = n_evicts = 0
    for obs in observations:
        r = real_rekey_absorb(mem, obs.candidate, dim=dim)
        if r["action"] == "update": n_updates += 1
        elif r["action"] == "insert": n_inserts += 1
        elif r["action"] == "evict_insert": n_evicts += 1
    return dict(system="B_rekey_real", memory=mem, n_updates=n_updates, n_inserts=n_inserts,
                n_evicts=n_evicts, wall_seconds=time.perf_counter() - t0)


def run_system_C_oracle_frozen(observations, capacity: int, dim: int = D):
    mem = MemoryState(capacity=capacity, dim=dim)
    concept_to_slot = {}
    t0 = time.perf_counter()
    n_updates = n_inserts = n_evicts = 0
    for obs in observations:
        existing = concept_to_slot.get(obs.concept_id)
        if existing is not None and not any(existing is s for s in mem.slots):
            existing = None
        r = oracle_absorb(mem, obs.candidate, existing, dim=dim)
        if r["action"] == "update": n_updates += 1
        elif r["action"] == "insert":
            n_inserts += 1
            concept_to_slot[obs.concept_id] = r["slot"]
        elif r["action"] == "evict_insert":
            n_evicts += 1
            concept_to_slot[obs.concept_id] = r["slot"]
    return dict(system="C_oracle_frozen", memory=mem, concept_to_slot=concept_to_slot,
                n_updates=n_updates, n_inserts=n_inserts, n_evicts=n_evicts,
                wall_seconds=time.perf_counter() - t0)


def run_system_D_oracle_rekey(observations, capacity: int, dim: int = D):
    mem = MemoryState(capacity=capacity, dim=dim)
    concept_to_slot = {}
    t0 = time.perf_counter()
    n_updates = n_inserts = n_evicts = 0
    for obs in observations:
        existing = concept_to_slot.get(obs.concept_id)
        if existing is not None and not any(existing is s for s in mem.slots):
            existing = None
        r = oracle_rekey_absorb(mem, obs.candidate, existing, dim=dim)
        if r["action"] == "update": n_updates += 1
        elif r["action"] == "insert":
            n_inserts += 1
            concept_to_slot[obs.concept_id] = r["slot"]
        elif r["action"] == "evict_insert":
            n_evicts += 1
            concept_to_slot[obs.concept_id] = r["slot"]
    return dict(system="D_oracle_rekey", memory=mem, concept_to_slot=concept_to_slot,
                n_updates=n_updates, n_inserts=n_inserts, n_evicts=n_evicts,
                wall_seconds=time.perf_counter() - t0)
