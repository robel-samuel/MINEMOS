"""
benchmarks/phase15_experiment.py

System A: frozen key (absorb(), unmodified).
System B: immediate re-key (Phase 13's real_rekey_absorb, reused).
System C: confidence-gated re-key.

CONFIDENCE RULE (as specified): a slot's key only moves after
REKEY_CONFIRMATIONS consecutive accepted merges that "support the same
new addressing direction." Implemented as: track the encoding (x_t) of
the most recent accepted merge for that slot. A new merge "confirms"
the pending direction if its x_t is highly similar (>= MATCH_THRESHOLD,
the same bar used for admission) to the PREVIOUS merge's x_t. If
confirmed, the counter increments; if not, the counter resets to 1 and
the new merge becomes the new pending direction. When the counter
reaches REKEY_CONFIRMATIONS, the key is set to the slot's current value
and the counter resets to 0.

The update equation's single blend line is byte-identical to
memory_update.absorb()'s across all three systems.

KNOWN LIMITATION, diagnosed directly (see phase15_report.md): comparing
consecutive merge x_t's pairwise means "confirmation" is measured with
the SAME field-aware lexical similarity that caused the original
problem. For single-word predicates sharing subject+object, ANY two
predicates score exactly the repository default threshold (0.75)
regardless of semantic consistency -- verified: 'avoids' vs 'frequents',
'avoids' vs 'dislikes', and 'visits' vs 'frequents' all score exactly
0.75 despite very different semantic relationships.
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


def confidence_gated_absorb(memory, candidate, confirm_state: dict, rekey_confirmations: int,
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
        state = confirm_state.setdefault(sid, dict(count=0, last_x=None))
        rekeyed = False
        if state["last_x"] is not None and cosine_similarity(x_t, state["last_x"]) >= threshold:
            state["count"] += 1
        else:
            state["count"] = 1
        state["last_x"] = x_t.copy()

        if state["count"] >= rekey_confirmations:
            slot.key = slot.value.copy()
            state["count"] = 0
            state["last_x"] = None
            rekeyed = True

        return dict(action="update", slot_index=best_idx, similarity=max_sim,
                    novelty=n_t, rekeyed=rekeyed, confirm_count=state["count"])

    if not memory.is_full():
        memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                   debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        return dict(action="insert", slot_index=len(memory.slots) - 1, similarity=max_sim,
                    novelty=n_t, rekeyed=False, confirm_count=0)

    memory.evict_oldest()
    memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
               debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    return dict(action="evict_insert", slot_index=len(memory.slots) - 1, similarity=max_sim,
                novelty=n_t, rekeyed=False, confirm_count=0)


def run_system(observations, capacity: int, system: str, dim: int = D):
    from phase2.memory_update import absorb
    from benchmarks.phase13_experiment import real_rekey_absorb

    mem = MemoryState(capacity=capacity, dim=dim)
    confirm_state = {}
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
            n_confirm = int(system[1:])
            r = confidence_gated_absorb(mem, candidate, confirm_state, n_confirm, dim=dim)

        if r["action"] == "update": n_updates += 1
        elif r["action"] == "insert": n_inserts += 1
        elif r["action"] == "evict_insert": n_evicts += 1
        if r.get("rekeyed"): n_rekeys += 1
    elapsed = time.perf_counter() - t0

    return dict(system=system, memory=mem, n_updates=n_updates, n_inserts=n_inserts,
                n_evicts=n_evicts, n_rekeys=n_rekeys, wall_seconds=elapsed)
