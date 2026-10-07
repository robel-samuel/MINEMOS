"""
benchmarks/phase14_experiment.py

System A: frozen key (memory_update.absorb(), unmodified).
System B: immediate re-key (Phase 13's real_rekey_absorb, reused).
System C: delayed re-key -- key changes only after N successful updates
    to that slot since its last re-key (or since creation, for the
    first re-key). N in {2, 3, 5}, predefined, not tuned on results.

The update equation's single blend line is identical across all three
(verified by test, reusing Phase 13's byte-identical proof pattern).
Only WHEN the key is allowed to change differs.
"""

from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState
from phase2.memory_update import address, novelty, cosine_similarity, MATCH_THRESHOLD
from phase2.encoders import FieldAwareLexicalEncoder

D = 4096
ENCODER = FieldAwareLexicalEncoder()


def delayed_rekey_absorb(memory, candidate, merge_counts: dict, N: int,
                          threshold=MATCH_THRESHOLD, encoder=ENCODER, dim=D):
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

        sid = id(slot)
        merge_counts[sid] = merge_counts.get(sid, 0) + 1
        rekeyed = False
        if merge_counts[sid] >= N:
            slot.key = slot.value.copy()
            merge_counts[sid] = 0
            rekeyed = True
        return dict(action="update", slot_index=best_idx, similarity=max_sim,
                    novelty=n_t, rekeyed=rekeyed)

    if not memory.is_full():
        memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                   debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        return dict(action="insert", slot_index=len(memory.slots) - 1, similarity=max_sim,
                    novelty=n_t, rekeyed=False)

    memory.evict_oldest()
    memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
               debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    return dict(action="evict_insert", slot_index=len(memory.slots) - 1, similarity=max_sim,
                novelty=n_t, rekeyed=False)


def run_system(observations, capacity: int, system: str, dim: int = D):
    """system in {'A', 'B', 'C2', 'C3', 'C5'}."""
    from phase2.memory_update import absorb
    from benchmarks.phase13_experiment import real_rekey_absorb

    mem = MemoryState(capacity=capacity, dim=dim)
    merge_counts = {}
    n_updates = n_inserts = n_evicts = n_rekeys = 0
    t0 = time.perf_counter()
    for obs in observations:
        candidate = obs.candidate if hasattr(obs, "candidate") else obs
        if system == "A":
            r = absorb(mem, candidate, threshold=MATCH_THRESHOLD, encoder=ENCODER)
        elif system == "B":
            r = real_rekey_absorb(mem, candidate, dim=dim)
            r["rekeyed"] = (r["action"] == "update")
        else:
            N = int(system[1:])
            r = delayed_rekey_absorb(mem, candidate, merge_counts, N, dim=dim)

        if r["action"] == "update": n_updates += 1
        elif r["action"] == "insert": n_inserts += 1
        elif r["action"] == "evict_insert": n_evicts += 1
        if r.get("rekeyed"): n_rekeys += 1
    elapsed = time.perf_counter() - t0

    return dict(system=system, memory=mem, n_updates=n_updates, n_inserts=n_inserts,
                n_evicts=n_evicts, n_rekeys=n_rekeys, wall_seconds=elapsed)
