"""
benchmarks/phase20_experiment.py

Phase 20 Experiment Architecture:
Adaptive Contradiction Anchor Management:
1. Temporal Decay: w_t = w_0 * exp(-lambda * delta_t)
2. Bounded Anchor Buffers: k in {1, 3, 5, 10, infinity} with deterministic FIFO eviction
3. Anchor Weight Parameterization: w_anchor in {0.0, 0.1, 0.2, 0.3, 0.5, 1.0}

Systems:
  System D:  Phase 18 baseline (Dense + Utility + fixed k=3, no anchors)
  System D4: Phase 19 batched baseline (Dense + Contra-Utility + Adaptive Beam + Batched NLI + static unbounded anchors)
  System D5: System D4 + Temporal Decay (w_0=0.20, lambda > 0, k=infinity)
  System D6: System D4 + Bounded Anchors (w_0=0.20, lambda=0, k in {1, 3, 5, 10})
  System D7: System D4 + Temporal Decay + Bounded Anchors (w_0=0.20, lambda > 0, k < infinity)
  System D8: Best experimentally justified combination
"""

from __future__ import annotations
import os, sys, time
from typing import Optional, Union

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
from benchmarks.phase19_experiment import contradiction_aware_utility


# ──────────────────────────────────────────────
#  1. Phase 20 Slot Eviction & Synchronization
# ──────────────────────────────────────────────

def evict_slot_phase20(
    memory: MemoryState,
    eviction_policy: str,
    current_time: float,
    slot_texts: dict[int, str],
    slot_dense_embs: list[np.ndarray],
    slot_contra_counts: dict[int, int] = None,
    slot_contra_anchors: dict[int, list[tuple[np.ndarray, float]]] = None,
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
#  2. Adaptive Dense Addressing with Anchor Decay & Telemetry
# ──────────────────────────────────────────────

def adaptive_dense_address_phase20(
    slot_embeddings: list[np.ndarray],
    cand_embedding: np.ndarray,
    slot_contra_anchors: dict[int, list[Union[tuple[np.ndarray, float], np.ndarray]]] = None,
    slots: list[MemorySlot] = None,
    current_time: float = 0.0,
    tau_min: float = 0.40,
    tau_high: float = 0.75,
    delta_margin: float = 0.10,
    k_max: int = 3,
    use_repulsion: bool = False,
    w_anchor: float = 0.20,
    decay_lambda: float = 0.0,
    anchor_sim_threshold: float = 0.70
) -> tuple[list[tuple[int, float]], str, float, int]:
    """
    Phase 20 Adaptive Candidate Beam with Temporal Anchor Decay:
      w_t = w_anchor * exp(-decay_lambda * delta_t)
      penalty = max_{a in slot} [ w_t * a_sim * I(a_sim > anchor_sim_threshold) ]
    
    Returns:
      (selected_candidates, beam_policy_reason, applied_repulsion_penalty, active_anchors_evaluated)
    """
    if not slot_embeddings:
        return [], "empty", 0.0, 0

    matrix = np.stack(slot_embeddings)  # (N, 384)
    sims = np.dot(matrix, cand_embedding)  # (N,)

    applied_penalty_total = 0.0
    active_anchors_evaluated = 0

    if use_repulsion and slot_contra_anchors and slots and w_anchor > 0.0:
        for idx, slot in enumerate(slots):
            anchors = slot_contra_anchors.get(id(slot), [])
            if anchors:
                active_anchors_evaluated += len(anchors)
                max_slot_penalty = 0.0
                for a_item in anchors:
                    if isinstance(a_item, tuple):
                        a_emb, a_time = a_item
                    else:
                        a_emb, a_time = a_item, 0.0
                    
                    a_sim = float(np.dot(a_emb, cand_embedding))
                    if a_sim > anchor_sim_threshold:
                        delta_t = max(0.0, current_time - a_time)
                        decay = np.exp(-decay_lambda * delta_t) if decay_lambda > 0.0 else 1.0
                        pen = w_anchor * decay * a_sim
                        if pen > max_slot_penalty:
                            max_slot_penalty = pen

                if max_slot_penalty > 0.0:
                    sims[idx] -= max_slot_penalty
                    applied_penalty_total += max_slot_penalty

    ranked_indices = np.argsort(-sims)
    sim_1 = float(sims[ranked_indices[0]])

    # 1. Weak candidate pruning
    if sim_1 < tau_min:
        return [], "prune_weak", applied_penalty_total, active_anchors_evaluated

    sim_2 = float(sims[ranked_indices[1]]) if len(ranked_indices) > 1 else -1.0

    # 2. Dominant candidate fast-path
    if sim_1 >= tau_high and (sim_1 - sim_2 >= delta_margin or len(ranked_indices) == 1):
        return [(int(ranked_indices[0]), sim_1)], "dominant", applied_penalty_total, active_anchors_evaluated

    # 3. Ambiguous clustered beam (up to k_max with sim >= tau_min)
    candidates = []
    for idx in ranked_indices[:k_max]:
        s_val = float(sims[idx])
        if s_val >= tau_min:
            candidates.append((int(idx), s_val))
    return candidates, "ambiguous", applied_penalty_total, active_anchors_evaluated


# ──────────────────────────────────────────────
#  3. Phase 20 Observation Absorption
# ──────────────────────────────────────────────

def absorb_observation_phase20(
    memory: MemoryState,
    candidate: Candidate,
    system: str,  # 'D', 'D4', 'D5', 'D6', 'D7', 'D8'
    slot_texts: dict[int, str],
    slot_dense_embs: list[np.ndarray],
    slot_contra_counts: dict[int, int],
    slot_contra_anchors: dict[int, list[tuple[np.ndarray, float]]],
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
    delta_margin: float = 0.10,
    # Phase 20 Anchor Management parameters
    w_anchor: float = 0.20,
    decay_lambda: float = 0.0,
    max_anchors: Optional[int] = None,
    anchor_sim_threshold: float = 0.70
) -> dict:
    """
    Absorption Pipeline for Phase 20 supporting Adaptive Anchor Management:
      - w_anchor: initial anchor weight (w_anchor=0.0 reproduces non-anchor baseline)
      - decay_lambda: temporal decay constant
      - max_anchors: maximum anchors stored per slot (FIFO eviction)
    """
    x_t = lex_encoder.encode_full(candidate, dim=dim)
    cand_text = f"{candidate.subject} {candidate.predicate} {candidate.object}."

    is_dense = True  # Phase 20 focuses on dense retrieval systems
    use_adaptive_beam = (system != "D")
    use_contra_shielding = (system != "D")
    use_batched_nli = (system != "D")
    eviction_policy = "utility" if system == "D" else "contra_utility"
    use_repulsion = (system != "D") and (w_anchor > 0.0)

    t_emb = t_retr = t_nli = 0.0
    target_idx = None
    target_pred = None
    gate_reason = "new_slot"
    candidates_examined = 0
    top_k_indices = []
    beam_policy = "fixed"
    applied_penalty = 0.0
    active_anchors_evaluated = 0
    cand_dense = None

    if memory.slots:
        t0_emb = time.perf_counter()
        cand_dense = dense_encoder.encode(cand_text)
        t_emb += (time.perf_counter() - t0_emb)

        t0_retr = time.perf_counter()
        if use_adaptive_beam:
            ranked_candidates, beam_policy, applied_penalty, active_anchors_evaluated = adaptive_dense_address_phase20(
                slot_embeddings=slot_dense_embs,
                cand_embedding=cand_dense,
                slot_contra_anchors=slot_contra_anchors if use_contra_shielding else None,
                slots=memory.slots if use_contra_shielding else None,
                current_time=current_time,
                tau_min=tau_min,
                tau_high=tau_high,
                delta_margin=delta_margin,
                k_max=top_k,
                use_repulsion=use_repulsion,
                w_anchor=w_anchor,
                decay_lambda=decay_lambda,
                anchor_sim_threshold=anchor_sim_threshold
            )
        else:
            ranked_candidates = dense_address(slot_dense_embs, cand_dense, top_k=top_k)
            beam_policy = f"fixed_{top_k}"
        t_retr += (time.perf_counter() - t0_retr)

        top_k_indices = [idx for idx, _ in ranked_candidates]
        best_retrieved_idx = ranked_candidates[0][0] if ranked_candidates else None
        best_sim = ranked_candidates[0][1] if ranked_candidates else 0.0

        # NLI Verification
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
                            anchors = slot_contra_anchors.setdefault(slot_id, [])
                            # Bounded anchor eviction: FIFO
                            if max_anchors is not None and len(anchors) >= max_anchors:
                                anchors.pop(0)
                            anchors.append((cand_dense, current_time))
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
                            anchors = slot_contra_anchors.setdefault(slot_id, [])
                            if max_anchors is not None and len(anchors) >= max_anchors:
                                anchors.pop(0)
                            anchors.append((cand_dense, current_time))
                        if gate_reason == "new_slot":
                            gate_reason = "rejected_contra"
                    elif gate_reason == "new_slot":
                        gate_reason = "rejected_neutral"
            t_nli += (time.perf_counter() - t0_nli)
        else:
            gate_reason = "pruned_by_beam"

        n_t = novelty(best_sim, memory_empty=False)
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
            applied_penalty=applied_penalty, active_anchors=active_anchors_evaluated,
            t_emb=t_emb, t_retr=t_retr, t_nli=t_nli
        )

    # 5. Allocate New Slot (or Evict if Full)
    if cand_dense is None:
        t0_emb = time.perf_counter()
        cand_dense = dense_encoder.encode(cand_text)
        t_emb += (time.perf_counter() - t0_emb)

    if not memory.is_full():
        slot = memory.add(x_t, x_t.copy(), candidate.confidence, current_time,
                          debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        slot_texts[id(slot)] = cand_text
        slot_dense_embs.append(cand_dense)
        return dict(
            action="insert", slot_index=len(memory.slots) - 1, similarity=best_sim,
            novelty=n_t, gate=gate_reason, cand_text=cand_text,
            retrieved_idx=best_retrieved_idx, top_k_indices=top_k_indices,
            candidates_examined=candidates_examined, beam_policy=beam_policy,
            applied_penalty=applied_penalty, active_anchors=active_anchors_evaluated,
            t_emb=t_emb, t_retr=t_retr, t_nli=t_nli
        )

    # Eviction Required
    evict_idx, evicted_slot = evict_slot_phase20(
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
    slot_dense_embs.append(cand_dense)

    return dict(
        action="evict_insert", slot_index=len(memory.slots) - 1, evicted_slot_index=evict_idx,
        similarity=best_sim, novelty=n_t, gate=gate_reason, cand_text=cand_text,
        retrieved_idx=best_retrieved_idx, top_k_indices=top_k_indices,
        candidates_examined=candidates_examined, beam_policy=beam_policy,
        applied_penalty=applied_penalty, active_anchors=active_anchors_evaluated,
        t_emb=t_emb, t_retr=t_retr, t_nli=t_nli
    )


# ──────────────────────────────────────────────
#  4. Phase 20 Stream Runner
# ──────────────────────────────────────────────

def run_phase20_stream(
    observations,
    capacity: int,
    system: str,  # 'D', 'D4', 'D5', 'D6', 'D7', 'D8'
    nli_gate: NliSemanticGate,
    dense_encoder: DenseSemanticEncoder = None,
    top_k: int = 3,
    entail_threshold: float = 0.50,
    dim: int = D_LEX,
    tau_min: float = 0.40,
    tau_high: float = 0.75,
    delta_margin: float = 0.10,
    w_anchor: float = 0.20,
    decay_lambda: float = 0.0,
    max_anchors: Optional[int] = None,
    anchor_sim_threshold: float = 0.70
) -> dict:
    """
    Executes a stream through the requested Phase 20 system configuration.
    """
    mem = MemoryState(capacity=capacity, dim=dim)
    slot_texts: dict[int, str] = {}
    slot_dense_embs: list[np.ndarray] = []
    slot_contra_counts: dict[int, int] = {}
    slot_contra_anchors: dict[int, list[tuple[np.ndarray, float]]] = {}

    trace = []
    n_updates = n_inserts = n_evicts = 0
    gate_counts: dict[str, int] = {}
    t_emb_total = t_retr_total = t_nli_total = 0.0
    total_penalty_applied = 0.0
    steps_with_penalty = 0

    t0 = time.perf_counter()

    for obs in observations:
        t_current = float(obs.step_index) if hasattr(obs, "step_index") else float(obs.candidate.timestamp)
        step_res = absorb_observation_phase20(
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
            delta_margin=delta_margin,
            w_anchor=w_anchor,
            decay_lambda=decay_lambda,
            max_anchors=max_anchors,
            anchor_sim_threshold=anchor_sim_threshold
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

        pen = step_res.get("applied_penalty", 0.0)
        if pen > 0.0:
            total_penalty_applied += pen
            steps_with_penalty += 1

        t_emb_total += step_res.get("t_emb", 0.0)
        t_retr_total += step_res.get("t_retr", 0.0)
        t_nli_total += step_res.get("t_nli", 0.0)

    wall_sec = time.perf_counter() - t0
    total_nli_calls = sum(tr.get("candidates_examined", 0) for tr in trace)

    # Anchor summary statistics
    all_anchors = []
    for anchor_list in slot_contra_anchors.values():
        all_anchors.extend(anchor_list)
    total_anchors_retained = len(all_anchors)
    final_time = float(observations[-1].candidate.timestamp) if observations and hasattr(observations[-1], "candidate") else float(len(observations))
    avg_anchor_age = (
        round(float(np.mean([max(0.0, final_time - a[1]) for a in all_anchors])), 2)
        if all_anchors else 0.0
    )
    avg_applied_penalty = (
        round(total_penalty_applied / steps_with_penalty, 4)
        if steps_with_penalty > 0 else 0.0
    )

    return dict(
        system=system,
        capacity=capacity,
        top_k=top_k,
        w_anchor=w_anchor,
        decay_lambda=decay_lambda,
        max_anchors=max_anchors,
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
        total_anchors_retained=total_anchors_retained,
        avg_anchor_age=avg_anchor_age,
        avg_applied_penalty=avg_applied_penalty,
        steps_with_penalty=steps_with_penalty,
        slot_contra_anchors=slot_contra_anchors,
        trace=trace,
        memory=mem
    )
