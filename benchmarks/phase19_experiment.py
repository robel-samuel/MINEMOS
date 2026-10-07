"""
benchmarks/phase19_experiment.py

Phase 19 Experiment Architecture:
Contradiction-Aware Retrieval, Safety Shielding, and Adaptive Verification.

Systems:
  System A:  Lexical + FIFO + NLI (Phase 17 baseline)
  System B:  Dense + FIFO + NLI (Phase 18 baseline)
  System C:  Lexical + Utility + NLI (Phase 18 baseline)
  System D:  Dense + Utility + NLI (Phase 18 baseline, fixed k=3)
  System D1: Dense + Contradiction-Aware Utility/Shielding + NLI (fixed k=3)
  System D2: Dense + Adaptive Beam + NLI (standard utility)
  System D3: Dense + Contradiction-Aware Shielding + Adaptive Beam + NLI
  System D4: System D3 + Batched NLI Verification

Core Invariants:
  - MATCH_THRESHOLD: 0.75 (frozen production default)
  - Core Update Equation: slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t (identical across all systems)
  - NLI Gate: cross-encoder/nli-distilroberta-base (local_files_only=True)
  - Dense Encoder: sentence-transformers/all-MiniLM-L6-v2 (local_files_only=True)
  - Seed: 42
  - Causal / Online Operation: Zero future labels, zero oracle concept IDs.
"""

from __future__ import annotations
import os, sys, time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import address, novelty, cosine_similarity, MATCH_THRESHOLD
from phase2.encoders import FieldAwareLexicalEncoder
from benchmarks.phase16_experiment import NliSemanticGate
from benchmarks.phase18_experiment import (
    DenseSemanticEncoder,
    dense_address,
    utility_score,
    D_LEX,
    D_DENSE,
    LEXICAL_ENCODER,
    DENSE_MODEL_NAME
)


# ──────────────────────────────────────────────
#  1. Contradiction-Aware Utility & Shielding
# ──────────────────────────────────────────────

def contradiction_aware_utility(
    slot: MemorySlot,
    current_time: float,
    contradiction_count: int = 0,
    tau: float = 50.0
) -> float:
    """
    Online contradiction-aware utility function:
      - Base utility = U_cons + U_rec + U_conf (identical to Phase 18)
      - Factual Shielding Bonus:
          If an established slot (update_count >= 2) receives contradiction challenge,
          it receives a shielding bonus to prevent adversarial distractor floods
          from prematurely evicting established factual knowledge.
      - Fragile Claim Penalty:
          If a slot has zero updates (update_count == 0) and high contradiction challenge,
          its score is slightly discounted to prioritize proven slots.
    """
    u_base = utility_score(slot, current_time, tau=tau)
    
    # Shielding bonus for established facts challenged by contradiction
    if slot.update_count >= 2 and contradiction_count > 0:
        shield_bonus = 0.15 * min(contradiction_count, 3)
    elif slot.update_count == 0 and contradiction_count > 0:
        shield_bonus = -0.10 * min(contradiction_count, 2)
    else:
        shield_bonus = 0.0

    return max(0.0, u_base + shield_bonus)


def evict_slot_phase19(
    memory: MemoryState,
    eviction_policy: str,
    current_time: float,
    slot_texts: dict[int, str],
    slot_dense_embs: list[np.ndarray],
    slot_contra_counts: dict[int, int] = None,
    slot_contra_anchors: dict[int, list[np.ndarray]] = None,
    slot_concept_ids: dict[int, str] = None
) -> tuple[int, MemorySlot]:
    """
    Evicts a slot according to:
      - 'fifo': oldest created_at
      - 'utility': standard Phase 18 utility U(s)
      - 'contra_utility': contradiction-aware utility U_contra(s)
    Maintains all slot metadata dictionaries in sync.
    """
    if not memory.slots:
        raise RuntimeError("Cannot evict from empty memory")

    if eviction_policy == "fifo":
        evict_idx = int(np.argmin([s.created_at for s in memory.slots]))
    elif eviction_policy == "utility":
        scores = [utility_score(s, current_time) for s in memory.slots]
        evict_idx = int(np.argmin(scores))
    elif eviction_policy == "contra_utility":
        contra_map = slot_contra_counts or {}
        scores = [
            contradiction_aware_utility(s, current_time, contra_map.get(id(s), 0))
            for s in memory.slots
        ]
        evict_idx = int(np.argmin(scores))
    else:
        raise ValueError(f"Unknown eviction policy: {eviction_policy}")

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
    if slot_concept_ids and slot_id in slot_concept_ids:
        del slot_concept_ids[slot_id]

    return evict_idx, evicted_slot


# ──────────────────────────────────────────────
#  2. Adaptive Candidate Beam & Repulsion
# ──────────────────────────────────────────────

def adaptive_dense_address(
    slot_embeddings: list[np.ndarray],
    cand_embedding: np.ndarray,
    slot_contra_anchors: dict[int, list[np.ndarray]] = None,
    slots: list[MemorySlot] = None,
    tau_min: float = 0.40,
    tau_high: float = 0.75,
    delta_margin: float = 0.10,
    k_max: int = 3,
    use_repulsion: bool = False
) -> tuple[list[tuple[int, float]], str]:
    """
    Adaptive candidate beam selection:
      1. Ranks all slots by cosine similarity.
      2. If use_repulsion=True and slot has contradiction anchors, penalizes ranking
         if cand_embedding is strongly similar to known contradiction anchors.
      3. Prunes candidates:
         - If top-1 sim < tau_min: candidate pool is empty -> beam=[] (policy: 'prune_weak')
         - If top-1 sim >= tau_high and (sim_1 - sim_2 >= delta_margin): beam size = 1 (policy: 'dominant')
         - Otherwise: beam size = up to k_max filtered by sim >= tau_min (policy: 'ambiguous')
    
    Returns (selected_candidates, beam_policy_reason).
    """
    if not slot_embeddings:
        return [], "empty"

    matrix = np.stack(slot_embeddings)  # (N, 384)
    sims = np.dot(matrix, cand_embedding)  # (N,)

    # Apply contradiction anchor repulsion penalty if enabled
    if use_repulsion and slot_contra_anchors and slots:
        for idx, slot in enumerate(slots):
            anchors = slot_contra_anchors.get(id(slot), [])
            if anchors:
                anchor_sims = [float(np.dot(a, cand_embedding)) for a in anchors]
                max_a_sim = max(anchor_sims)
                if max_a_sim > 0.70:
                    sims[idx] -= 0.20 * max_a_sim

    ranked_indices = np.argsort(-sims)
    sim_1 = float(sims[ranked_indices[0]])

    # 1. Weak candidate pruning
    if sim_1 < tau_min:
        return [], "prune_weak"

    sim_2 = float(sims[ranked_indices[1]]) if len(ranked_indices) > 1 else -1.0

    # 2. Dominant candidate fast-path
    if sim_1 >= tau_high and (sim_1 - sim_2 >= delta_margin or len(ranked_indices) == 1):
        return [(int(ranked_indices[0]), sim_1)], "dominant"

    # 3. Ambiguous clustered beam (up to k_max with sim >= tau_min)
    candidates = []
    for idx in ranked_indices[:k_max]:
        s_val = float(sims[idx])
        if s_val >= tau_min:
            candidates.append((int(idx), s_val))
    return candidates, "ambiguous"


# ──────────────────────────────────────────────
#  3. Unified Phase 19 Absorb Pipeline
# ──────────────────────────────────────────────

def absorb_observation_phase19(
    memory: MemoryState,
    candidate: Candidate,
    system: str,  # 'A', 'B', 'C', 'D', 'D1', 'D2', 'D3', 'D4'
    slot_texts: dict[int, str],
    slot_dense_embs: list[np.ndarray],
    slot_contra_counts: dict[int, int],
    slot_contra_anchors: dict[int, list[np.ndarray]],
    nli_gate: NliSemanticGate,
    dense_encoder: DenseSemanticEncoder = None,
    top_k: int = 3,
    current_time: float = 0.0,
    entail_threshold: float = 0.50,
    lex_encoder: FieldAwareLexicalEncoder = LEXICAL_ENCODER,
    dim: int = D_LEX,
    # Adaptive beam parameters
    tau_min: float = 0.40,
    tau_high: float = 0.75,
    delta_margin: float = 0.10
) -> dict:
    """
    Unified Phase 19 Absorption Pipeline supporting Systems A, B, C, D, D1, D2, D3, D4.
    """
    # 1. Lexical encoding for core memory vector
    x_t = lex_encoder.encode_full(candidate, dim=dim)
    cand_text = f"{candidate.subject} {candidate.predicate} {candidate.object}."

    is_dense = system in ("B", "D", "D1", "D2", "D3", "D4")
    use_adaptive_beam = system in ("D2", "D3", "D4")
    use_contra_shielding = system in ("D1", "D3", "D4")
    use_batched_nli = (system == "D4")

    if system in ("A", "B"):
        eviction_policy = "fifo"
    elif system in ("C", "D", "D2"):
        eviction_policy = "utility"
    else:  # D1, D3, D4
        eviction_policy = "contra_utility"

    t_emb = t_retr = t_nli = 0.0
    target_idx = None
    target_pred = None
    gate_reason = "new_slot"
    candidates_examined = 0
    top_k_indices = []
    beam_policy = "fixed"
    cand_dense = None

    # 2. Candidate Retrieval (Stage 1)
    if memory.slots:
        if is_dense:
            t0_emb = time.perf_counter()
            cand_dense = dense_encoder.encode(cand_text)
            t_emb += (time.perf_counter() - t0_emb)

            t0_retr = time.perf_counter()
            if use_adaptive_beam:
                ranked_candidates, beam_policy = adaptive_dense_address(
                    slot_embeddings=slot_dense_embs,
                    cand_embedding=cand_dense,
                    slot_contra_anchors=slot_contra_anchors if use_contra_shielding else None,
                    slots=memory.slots if use_contra_shielding else None,
                    tau_min=tau_min,
                    tau_high=tau_high,
                    delta_margin=delta_margin,
                    k_max=top_k,
                    use_repulsion=use_contra_shielding
                )
            else:
                ranked_candidates = dense_address(slot_dense_embs, cand_dense, top_k=top_k)
                beam_policy = f"fixed_{top_k}"
            t_retr += (time.perf_counter() - t0_retr)

            top_k_indices = [idx for idx, _ in ranked_candidates]
            best_retrieved_idx = ranked_candidates[0][0] if ranked_candidates else None
            best_sim = ranked_candidates[0][1] if ranked_candidates else 0.0

            # 3. Stage 2: NLI Verification
            if ranked_candidates:
                t0_nli = time.perf_counter()
                if use_batched_nli and len(ranked_candidates) > 1:
                    pairs = [(slot_texts[id(memory.slots[idx])], cand_text) for idx, _ in ranked_candidates]
                    preds = nli_gate.predict_batch(pairs, entail_threshold=entail_threshold)
                    for (cand_idx, _), pred in zip(ranked_candidates, preds):
                        candidates_examined += 1
                        if pred["decision"] == "SAME":
                            target_idx = cand_idx
                            target_pred = pred
                            gate_reason = "nli_accept"
                            break
                        elif pred["decision"] == "CONTRADICTION":
                            if use_contra_shielding:
                                slot = memory.slots[cand_idx]
                                slot_id = id(slot)
                                slot_contra_counts[slot_id] = slot_contra_counts.get(slot_id, 0) + 1
                                if slot_id not in slot_contra_anchors:
                                    slot_contra_anchors[slot_id] = []
                                slot_contra_anchors[slot_id].append(cand_dense)
                            if gate_reason == "new_slot":
                                gate_reason = "rejected_contra"
                        elif gate_reason == "new_slot":
                            gate_reason = "rejected_neutral"
                else:
                    for cand_idx, sim in ranked_candidates:
                        candidates_examined += 1
                        slot = memory.slots[cand_idx]
                        premise_text = slot_texts[id(slot)]
                        pred = nli_gate.predict_pair(premise_text, cand_text, entail_threshold=entail_threshold)
                        if pred["decision"] == "SAME":
                            target_idx = cand_idx
                            target_pred = pred
                            gate_reason = "nli_accept"
                            break
                        elif pred["decision"] == "CONTRADICTION":
                            if use_contra_shielding:
                                slot_id = id(slot)
                                slot_contra_counts[slot_id] = slot_contra_counts.get(slot_id, 0) + 1
                                if slot_id not in slot_contra_anchors:
                                    slot_contra_anchors[slot_id] = []
                                slot_contra_anchors[slot_id].append(cand_dense)
                            if gate_reason == "new_slot":
                                gate_reason = "rejected_contra"
                        elif gate_reason == "new_slot":
                            gate_reason = "rejected_neutral"
                t_nli += (time.perf_counter() - t0_nli)
            else:
                gate_reason = "pruned_by_beam"

            n_t = novelty(best_sim, memory_empty=False)

        else:
            # Lexical sparse candidate generator (Systems A, C)
            t0_retr = time.perf_counter()
            best_idx, max_sim = address(memory, x_t)
            t_retr += (time.perf_counter() - t0_retr)

            best_retrieved_idx = best_idx
            best_sim = max_sim
            n_t = novelty(max_sim, memory_empty=(best_idx is None))
            top_k_indices = [best_idx] if best_idx is not None else []

            if best_idx is not None:
                candidates_examined = 1
                slot = memory.slots[best_idx]
                premise_text = slot_texts[id(slot)]
                t0_nli = time.perf_counter()
                pred = nli_gate.predict_pair(premise_text, cand_text, entail_threshold=entail_threshold)
                t_nli += (time.perf_counter() - t0_nli)
                if pred["decision"] == "SAME":
                    target_idx = best_idx
                    target_pred = pred
                    gate_reason = "nli_accept"
                else:
                    gate_reason = "rejected_contra" if pred["decision"] == "CONTRADICTION" else "rejected_neutral"
    else:
        best_retrieved_idx = None
        best_sim = 0.0
        n_t = 1.0

    alpha_t = n_t

    # 4. Memory Update Path (Core Update Equation IDENTICAL across all systems)
    if target_idx is not None:
        slot = memory.slots[target_idx]
        slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = current_time
        slot.update_count += 1
        return dict(
            action="update", slot_index=target_idx, similarity=best_sim, novelty=n_t,
            gate="nli_accept", sem_pred=target_pred, cand_text=cand_text,
            retrieved_idx=best_retrieved_idx, top_k_indices=top_k_indices,
            candidates_examined=candidates_examined, beam_policy=beam_policy,
            t_emb=t_emb, t_retr=t_retr, t_nli=t_nli
        )

    # 5. Allocate New Slot (or Evict if Full)
    if is_dense and cand_dense is None:
        t0_emb = time.perf_counter()
        cand_dense = dense_encoder.encode(cand_text)
        t_emb += (time.perf_counter() - t0_emb)

    if not memory.is_full():
        slot = memory.add(x_t, x_t.copy(), candidate.confidence, current_time,
                          debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        slot_texts[id(slot)] = cand_text
        if is_dense and cand_dense is not None:
            slot_dense_embs.append(cand_dense)
        return dict(
            action="insert", slot_index=len(memory.slots) - 1, similarity=best_sim,
            novelty=n_t, gate=gate_reason, cand_text=cand_text,
            retrieved_idx=best_retrieved_idx, top_k_indices=top_k_indices,
            candidates_examined=candidates_examined, beam_policy=beam_policy,
            t_emb=t_emb, t_retr=t_retr, t_nli=t_nli
        )

    # Eviction Required
    evict_idx, evicted_slot = evict_slot_phase19(
        memory=memory,
        eviction_policy=eviction_policy,
        current_time=current_time,
        slot_texts=slot_texts,
        slot_dense_embs=slot_dense_embs,
        slot_contra_counts=slot_contra_counts if use_contra_shielding else None,
        slot_contra_anchors=slot_contra_anchors if use_contra_shielding else None
    )

    slot = memory.add(x_t, x_t.copy(), candidate.confidence, current_time,
                      debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    slot_texts[id(slot)] = cand_text
    if is_dense and cand_dense is not None:
        slot_dense_embs.append(cand_dense)

    return dict(
        action="evict_insert", slot_index=len(memory.slots) - 1, evicted_slot_index=evict_idx,
        similarity=best_sim, novelty=n_t, gate=gate_reason, cand_text=cand_text,
        retrieved_idx=best_retrieved_idx, top_k_indices=top_k_indices,
        candidates_examined=candidates_examined, beam_policy=beam_policy,
        t_emb=t_emb, t_retr=t_retr, t_nli=t_nli
    )


# ──────────────────────────────────────────────
#  4. Generic Stream Runner for Phase 19
# ──────────────────────────────────────────────

def run_phase19_stream(
    observations,
    capacity: int,
    system: str,  # 'A', 'B', 'C', 'D', 'D1', 'D2', 'D3', 'D4'
    nli_gate: NliSemanticGate,
    dense_encoder: DenseSemanticEncoder = None,
    top_k: int = 3,
    entail_threshold: float = 0.50,
    dim: int = D_LEX,
    tau_min: float = 0.40,
    tau_high: float = 0.75,
    delta_margin: float = 0.10
) -> dict:
    """
    Executes a stream of Phase18Obs/Phase19Obs observations through the requested system.
    Returns summary statistics, per-observation trace, and final memory state.
    """
    mem = MemoryState(capacity=capacity, dim=dim)
    slot_texts: dict[int, str] = {}
    slot_dense_embs: list[np.ndarray] = []
    slot_contra_counts: dict[int, int] = {}
    slot_contra_anchors: dict[int, list[np.ndarray]] = {}

    trace = []
    n_updates = n_inserts = n_evicts = 0
    gate_counts: dict[str, int] = {}
    t_emb_total = t_retr_total = t_nli_total = 0.0

    t0 = time.perf_counter()

    for obs in observations:
        t_current = float(obs.step_index) if hasattr(obs, "step_index") else float(obs.candidate.timestamp)
        step_res = absorb_observation_phase19(
            memory=mem,
            candidate=obs.candidate,
            system=system,
            slot_texts=slot_texts,
            slot_dense_embs=slot_dense_embs,
            slot_contra_counts=slot_contra_counts,
            slot_contra_anchors=slot_contra_anchors,
            nli_gate=nli_gate,
            dense_encoder=dense_encoder,
            top_k=top_k,
            current_time=t_current,
            entail_threshold=entail_threshold,
            dim=dim,
            tau_min=tau_min,
            tau_high=tau_high,
            delta_margin=delta_margin
        )
        trace.append(step_res)

        act = step_res["action"]
        if act == "update":
            n_updates += 1
        elif act == "insert":
            n_inserts += 1
        elif act == "evict_insert":
            n_inserts += 1
            n_evicts += 1

        gate = step_res.get("gate", "unknown")
        gate_counts[gate] = gate_counts.get(gate, 0) + 1

        t_emb_total += step_res.get("t_emb", 0.0)
        t_retr_total += step_res.get("t_retr", 0.0)
        t_nli_total += step_res.get("t_nli", 0.0)

    wall_sec = time.perf_counter() - t0
    total_nli_calls = sum(tr.get("candidates_examined", 0) for tr in trace)

    return dict(
        system=system,
        capacity=capacity,
        top_k=top_k,
        n_obs=len(observations),
        final_slots=len(mem.slots),
        n_updates=n_updates,
        n_inserts=n_inserts,
        n_evicts=n_evicts,
        nli_calls=total_nli_calls,
        gate_counts=gate_counts,
        wall_seconds=wall_sec,
        obs_per_second=round(len(observations) / wall_sec, 2) if wall_sec > 0 else 0.0,
        time_embedding=round(t_emb_total, 3),
        time_retrieval=round(t_retr_total, 3),
        time_nli=round(t_nli_total, 3),
        trace=trace,
        memory=mem
    )
