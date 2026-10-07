"""
benchmarks/phase17_experiment.py

Phase 17 Experiment:
Long-Horizon Continual Memory Under Semantic Noise.

The Phase 16 two-stage architecture is unchanged:
  1. Cheap lexical/vector candidate retrieval (FieldAwareLexicalEncoder)
  2. NLI semantic verification (cross-encoder/nli-distilroberta-base)
  3. SAME / CONTRADICTION / NEUTRAL → merge / reject
  4. Existing update equation: slot.value = (1-alpha)*slot.value + alpha*x_t

Three systems:
  A = Phase 16 NLI-gated baseline (two-stage)
  B = Oracle upper bound
  C = Lexical baseline

The update equation is INVARIANT across all systems.
MATCH_THRESHOLD = 0.75 is UNCHANGED.
"""

from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState
from phase2.memory_update import address, novelty, cosine_similarity, MATCH_THRESHOLD
from phase2.encoders import FieldAwareLexicalEncoder
from benchmarks.phase16_experiment import NliSemanticGate

D = 4096
ENCODER = FieldAwareLexicalEncoder()


# ──────────────────────────────────────────────
#  SYSTEM A — NLI-gated (Phase 16 baseline)
# ──────────────────────────────────────────────
def nli_gated_absorb(memory: MemoryState, candidate: Candidate,
                     slot_texts: dict[int, str],
                     gate: NliSemanticGate,
                     entail_threshold: float = 0.50,
                     encoder=ENCODER, dim: int = D) -> dict:
    """Exact Phase 16 two-stage architecture. DO NOT MODIFY."""
    x_t = encoder.encode_full(candidate, dim=dim)
    best_idx, max_sim = address(memory, x_t)
    n_t = novelty(max_sim, memory_empty=(best_idx is None))
    alpha_t = n_t
    cand_text = f"{candidate.subject} {candidate.predicate} {candidate.object}."

    target_idx = None
    target_pred = None
    gate_reason = "new_slot"

    if best_idx is not None:
        slot = memory.slots[best_idx]
        premise_text = slot_texts[id(slot)]
        pred = gate.predict_pair(premise_text, cand_text, entail_threshold=entail_threshold)
        if pred["decision"] == "SAME":
            target_idx = best_idx
            target_pred = pred
        else:
            gate_reason = "rejected_contra" if pred["decision"] == "CONTRADICTION" else "rejected_neutral"

    if target_idx is not None:
        slot = memory.slots[target_idx]
        # --- identical update line to memory_update.absorb() ---
        slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
        # ---------------------------------------------------------------
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = candidate.timestamp
        slot.update_count += 1
        return dict(action="update", slot_index=target_idx,
                    similarity=max_sim, novelty=n_t,
                    gate="nli_accept", sem_pred=target_pred,
                    cand_text=cand_text, retrieved_idx=best_idx)

    if not memory.is_full():
        slot = memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                          debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        slot_texts[id(slot)] = cand_text
        return dict(action="insert", slot_index=len(memory.slots) - 1,
                    similarity=max_sim, novelty=n_t, gate=gate_reason, cand_text=cand_text,
                    retrieved_idx=best_idx)

    evicted = memory.evict_oldest()
    if id(evicted) in slot_texts:
        del slot_texts[id(evicted)]
    slot = memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                      debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    slot_texts[id(slot)] = cand_text
    return dict(action="evict_insert", slot_index=len(memory.slots) - 1,
                similarity=max_sim, novelty=n_t, gate=gate_reason, cand_text=cand_text,
                retrieved_idx=best_idx)


# ──────────────────────────────────────────────
#  SYSTEM B — Oracle
# ──────────────────────────────────────────────
def oracle_absorb(memory: MemoryState, candidate: Candidate,
                  slot_concept_ids: dict[int, str],
                  concept_id: str, category: str,
                  encoder=ENCODER, dim: int = D) -> dict:
    """Oracle upper bound (Phase 11 mechanism, unchanged)."""
    x_t = encoder.encode_full(candidate, dim=dim)
    best_idx, max_sim = address(memory, x_t)
    n_t = novelty(max_sim, memory_empty=(best_idx is None))
    alpha_t = n_t

    # Oracle grants a merge only if same concept and not an adversarial observation
    target_idx = None
    if category in ("canonical", "paraphrase", "recovery_canonical", "delayed_paraphrase", "drift_step"):
        for idx, slot in enumerate(memory.slots):
            if slot_concept_ids.get(id(slot)) == concept_id:
                target_idx = idx
                break

    if target_idx is not None:
        slot = memory.slots[target_idx]
        # --- identical update line to memory_update.absorb() ---
        slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
        # ---------------------------------------------------------------
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = candidate.timestamp
        slot.update_count += 1
        return dict(action="update", slot_index=target_idx,
                    similarity=max_sim, novelty=n_t, gate="oracle_accept",
                    retrieved_idx=best_idx)

    if not memory.is_full():
        slot = memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                          debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        slot_concept_ids[id(slot)] = concept_id
        return dict(action="insert", slot_index=len(memory.slots) - 1,
                    similarity=max_sim, novelty=n_t, gate="oracle_new",
                    retrieved_idx=best_idx)

    evicted = memory.evict_oldest()
    if id(evicted) in slot_concept_ids:
        del slot_concept_ids[id(evicted)]
    slot = memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                      debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    slot_concept_ids[id(slot)] = concept_id
    return dict(action="evict_insert", slot_index=len(memory.slots) - 1,
                similarity=max_sim, novelty=n_t, gate="oracle_evict",
                retrieved_idx=best_idx)


# ──────────────────────────────────────────────
#  SYSTEM C — Lexical baseline
# ──────────────────────────────────────────────
def lexical_absorb(memory: MemoryState, candidate: Candidate,
                   threshold: float = MATCH_THRESHOLD,
                   encoder=ENCODER, dim: int = D) -> dict:
    """Pure lexical baseline. Threshold = 0.75 unchanged."""
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
        return dict(action="update", slot_index=best_idx,
                    similarity=max_sim, novelty=n_t, gate="lexical_accept",
                    retrieved_idx=best_idx)

    if not memory.is_full():
        memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                   debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        return dict(action="insert", slot_index=len(memory.slots) - 1,
                    similarity=max_sim, novelty=n_t, gate="new_slot",
                    retrieved_idx=best_idx)

    memory.evict_oldest()
    memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
               debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    return dict(action="evict_insert", slot_index=len(memory.slots) - 1,
                similarity=max_sim, novelty=n_t, gate="evict_insert",
                retrieved_idx=best_idx)


# ──────────────────────────────────────────────
#  Generic stream runner
# ──────────────────────────────────────────────
def run_stream(observations, capacity: int, system: str,
               gate: NliSemanticGate = None,
               entail_threshold: float = 0.50,
               record_diagnostics: bool = False,
               dim: int = D) -> dict:
    """
    Runs a stream of Phase17Obs observations through the specified system.
    Returns summary counts plus per-observation trace if record_diagnostics=True.
    """
    mem = MemoryState(capacity=capacity, dim=dim)
    slot_texts: dict[int, str] = {}
    slot_concept_ids: dict[int, str] = {}

    n_updates = n_inserts = n_evicts = 0
    n_nli_calls = 0
    gate_counts: dict[str, int] = {}
    trace = []
    t0 = time.perf_counter()

    for obs in observations:
        cand = obs.candidate
        cid = obs.concept_id
        cat = obs.category

        if system == "A_nli":
            r = nli_gated_absorb(mem, cand, slot_texts, gate,
                                 entail_threshold=entail_threshold, dim=dim)
            if r["action"] in ("insert", "evict_insert") or r.get("gate") == "nli_accept":
                n_nli_calls += 1 if mem.slots else 0  # called if there was a candidate
        elif system == "B_oracle":
            r = oracle_absorb(mem, cand, slot_concept_ids, cid, cat, dim=dim)
        elif system == "C_lexical":
            r = lexical_absorb(mem, cand, dim=dim)
        else:
            raise ValueError(f"Unknown system: {system}")

        if r["action"] == "update":
            n_updates += 1
        elif r["action"] == "insert":
            n_inserts += 1
        elif r["action"] == "evict_insert":
            n_evicts += 1

        g = r.get("gate", "unknown")
        gate_counts[g] = gate_counts.get(g, 0) + 1

        if record_diagnostics:
            trace.append({
                "concept_id": cid,
                "category": cat,
                "scenario": getattr(obs, "scenario", ""),
                "is_probe": obs.is_probe,
                "action": r["action"],
                "slot_index": r.get("slot_index"),
                "retrieved_idx": r.get("retrieved_idx"),
                "gate": g,
                "similarity": r.get("similarity", None),
                "sem_pred": r.get("sem_pred", None),
            })

    elapsed = time.perf_counter() - t0
    return dict(
        system=system, memory=mem, capacity=capacity,
        n_updates=n_updates, n_inserts=n_inserts, n_evicts=n_evicts,
        n_nli_calls=n_nli_calls, gate_counts=gate_counts,
        wall_seconds=elapsed, trace=trace if record_diagnostics else None
    )
