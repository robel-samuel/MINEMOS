"""
benchmarks/phase23_experiment.py

Phase 23 Architecture & Execution Engine:
Consolidation and Retention Bottleneck Investigation.

Investigates downstream failure mechanisms after candidate generation:
  1. NLI Verification Threshold Sweep (0.55 - 0.85)
  2. Oracle Retrieval (Forcible Target Injection)
  3. Retention Oracle (Protected Target Slots during Eviction)
  4. Consolidation vs Retention 2x2 Causal Matrix
  5. Embedding Drift Tracking
  6. Failure Attribution Instrumentation (A, B, C, D, E, F)

Frozen Baselines (Do NOT modify):
  D8: Dense only, w*=0.15, lambda*=0.005, k*=3, tau_anchor=0.70
  H1: Hybrid Dense + BM25, k_fusion=5, D8 anchors
  Core update equation: v_t = (1 - alpha_t)*v_{t-1} + alpha_t*x_t (FROZEN)
"""

from __future__ import annotations
import os, sys, time
from typing import Optional, Union, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import MATCH_THRESHOLD, cosine_similarity, novelty
from phase2.encoders import FieldAwareLexicalEncoder
from benchmarks.phase16_experiment import NliSemanticGate
from benchmarks.phase18_experiment import (
    DenseSemanticEncoder,
    D_LEX,
    LEXICAL_ENCODER,
    dense_address,
    utility_score,
)
from benchmarks.phase19_experiment import contradiction_aware_utility
from benchmarks.phase20_experiment import adaptive_dense_address_phase20
from benchmarks.phase22_experiment import (
    LocalBM25Index,
    fuse_candidates,
    FROZEN_W,
    FROZEN_LAMBDA,
    FROZEN_K,
    TAU_MIN,
    TAU_HIGH,
    DELTA_MARGIN,
    K_MAX,
    TAU_ANCHOR,
)
from benchmarks.phase23_diagnostics import Phase23DiagnosticTracker


def evict_slot_phase23(
    memory: MemoryState,
    eviction_policy: str,
    current_time: float,
    slot_texts: dict[int, str],
    slot_dense_embs: list[np.ndarray],
    slot_contra_counts: Optional[dict[int, int]] = None,
    slot_contra_anchors: Optional[dict[int, list[tuple[np.ndarray, float]]]] = None,
    protected_slot_ids: Optional[set[int]] = None
) -> tuple[int, MemorySlot]:
    """
    Evicts a slot according to policy, strictly respecting protected_slot_ids (Retention Oracle).
    """
    if not memory.slots:
        raise RuntimeError("Cannot evict from empty memory")

    protected = protected_slot_ids or set()
    candidate_indices = [idx for idx, s in enumerate(memory.slots) if id(s) not in protected]

    # If all slots are protected (rare edge case), fallback to all slots
    if not candidate_indices:
        candidate_indices = list(range(len(memory.slots)))

    if eviction_policy == "fifo":
        evict_idx = min(candidate_indices, key=lambda idx: memory.slots[idx].created_at)
    elif eviction_policy == "utility":
        evict_idx = min(candidate_indices, key=lambda idx: utility_score(memory.slots[idx], current_time))
    elif eviction_policy == "contra_utility":
        contra_map = slot_contra_counts or {}
        evict_idx = min(
            candidate_indices,
            key=lambda idx: contradiction_aware_utility(
                memory.slots[idx], current_time, contra_map.get(id(memory.slots[idx]), 0)
            )
        )
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

    return evict_idx, evicted_slot


def absorb_observation_phase23(
    memory: MemoryState,
    candidate: Candidate,
    system: str,  # 'D', 'D4', 'D8', 'H1'
    slot_texts: dict[int, str],
    slot_dense_embs: list[np.ndarray],
    slot_contra_counts: dict[int, int],
    slot_contra_anchors: dict[int, list[tuple[np.ndarray, float]]],
    bm25_index: LocalBM25Index,
    nli_gate: NliSemanticGate,
    dense_encoder: DenseSemanticEncoder,
    concept_id: Optional[str] = None,
    is_probe: bool = False,
    category: str = "canonical",
    target_id: Optional[str] = None,
    entail_threshold: float = 0.75,  # Phase 23 Threshold Sweep parameter
    oracle_retrieval: bool = False,  # Phase 23 Oracle Retrieval
    protected_slot_ids: Optional[set[int]] = None,  # Phase 23 Retention Oracle
    diagnostic_tracker: Optional[Phase23DiagnosticTracker] = None,
    top_k_dense: int = 3,
    top_k_bm25: int = 3,
    k_fusion: int = 5,
    current_time: float = 0.0,
    lex_encoder: FieldAwareLexicalEncoder = LEXICAL_ENCODER,
    dim: int = D_LEX,
    tau_min: float = TAU_MIN,
    tau_high: float = TAU_HIGH,
    delta_margin: float = DELTA_MARGIN,
    w_anchor: float = FROZEN_W,
    decay_lambda: float = FROZEN_LAMBDA,
    max_anchors: Optional[int] = FROZEN_K,
    anchor_sim_threshold: float = TAU_ANCHOR
) -> dict:
    """
    Core Phase 23 Absorption Pipeline with Full Diagnostic Attribution.
    """
    x_t = lex_encoder.encode_full(candidate, dim=dim)
    cand_text = f"{candidate.subject} {candidate.predicate} {candidate.object}."

    use_lexical = (system == "H1")
    use_dense = True
    use_adaptive_beam = (system != "D")
    use_repulsion = (system in ("D4", "D8", "H1")) and (w_anchor > 0.0)
    use_contra_shielding = (system != "D")
    eviction_policy = "utility" if system == "D" else "contra_utility"

    t_emb = t_retr_dense = t_retr_bm25 = t_nli = 0.0
    target_idx = None
    target_pred = None
    gate_reason = "new_slot"
    candidates_examined = 0
    dense_candidates = []
    lexical_candidates = []
    fused_candidates = []
    beam_policy = "none"
    applied_penalty = 0.0
    active_anchors_evaluated = 0
    fusion_meta = {}
    cand_dense = None

    if memory.slots:
        id_to_slot_index = {id(slot): idx for idx, slot in enumerate(memory.slots)}

        # 1. Dense Candidate Generation
        t0_emb = time.perf_counter()
        cand_dense = dense_encoder.encode(cand_text)
        t_emb += (time.perf_counter() - t0_emb)

        t0_retr_d = time.perf_counter()
        if use_adaptive_beam:
            dense_candidates, beam_policy, applied_penalty, active_anchors_evaluated = adaptive_dense_address_phase20(
                slot_embeddings=slot_dense_embs,
                cand_embedding=cand_dense,
                slot_contra_anchors=slot_contra_anchors if use_contra_shielding else None,
                slots=memory.slots if use_contra_shielding else None,
                current_time=current_time,
                tau_min=tau_min,
                tau_high=tau_high,
                delta_margin=delta_margin,
                k_max=top_k_dense,
                use_repulsion=use_repulsion,
                w_anchor=w_anchor,
                decay_lambda=decay_lambda,
                anchor_sim_threshold=anchor_sim_threshold
            )
        else:
            dense_candidates = dense_address(slot_dense_embs, cand_dense, top_k=top_k_dense)
            beam_policy = f"fixed_{top_k_dense}"
        t_retr_dense += (time.perf_counter() - t0_retr_d)

        # 2. Lexical BM25 Candidate Generation
        if use_lexical:
            t0_retr_bm25 = time.perf_counter()
            lexical_candidates = bm25_index.query(
                query_text=cand_text,
                top_k=top_k_bm25,
                id_to_slot_index=id_to_slot_index
            )
            t_retr_bm25 += (time.perf_counter() - t0_retr_bm25)

        # 3. Candidate Fusion
        if use_dense and use_lexical:
            fused_candidates, fusion_meta = fuse_candidates(
                dense_candidates=dense_candidates,
                lexical_candidates=lexical_candidates,
                k_fusion=k_fusion
            )
        else:
            fused_candidates = [idx for idx, _ in dense_candidates]
            fusion_meta = {"n_dense": len(dense_candidates), "n_lexical": 0, "n_fused": len(fused_candidates)}

        # 4. ORACLE RETRIEVAL INJECTION (Diagnostic condition only)
        if oracle_retrieval and target_id and diagnostic_tracker:
            t_rec = diagnostic_tracker.targets.get(target_id)
            if t_rec and not t_rec.evicted and t_rec.target_slot_id:
                # Target slot is alive in memory!
                target_live_idx = id_to_slot_index.get(t_rec.target_slot_id)
                if target_live_idx is not None and target_live_idx not in fused_candidates:
                    fused_candidates.insert(0, target_live_idx)
                    fusion_meta["oracle_injected"] = True

        best_sim = dense_candidates[0][1] if dense_candidates else 0.0
        best_retrieved_idx = fused_candidates[0] if fused_candidates else None

        # 5. CrossEncoder NLI Verification (entail_threshold sweepable)
        if fused_candidates:
            t0_nli = time.perf_counter()
            pairs = [(slot_texts[id(memory.slots[idx])], cand_text) for idx in fused_candidates]
            preds = nli_gate.predict_batch(pairs, entail_threshold=entail_threshold)
            for cand_idx, pred in zip(fused_candidates, preds):
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

    # 6. Memory Update Path (Core Vector Update Equation FROZEN)
    if target_idx is not None:
        slot = memory.slots[target_idx]
        slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = current_time
        slot.update_count += 1

        if diagnostic_tracker:
            diagnostic_tracker.record_update(
                slot=slot,
                step=int(current_time),
                update_text=cand_text,
                dense_emb=cand_dense
            )

        step_res = dict(
            action="update", slot_index=target_idx, similarity=best_sim, novelty=n_t,
            gate="nli_accept", sem_pred=target_pred, cand_text=cand_text,
            retrieved_idx=best_retrieved_idx, top_k_indices=fused_candidates,
            dense_candidates=dense_candidates, lexical_candidates=lexical_candidates,
            fusion_meta=fusion_meta, candidates_examined=candidates_examined,
            beam_policy=beam_policy, applied_penalty=applied_penalty,
            active_anchors=active_anchors_evaluated,
            t_emb=t_emb, t_retr_dense=t_retr_dense, t_retr_bm25=t_retr_bm25, t_nli=t_nli
        )

        if is_probe and diagnostic_tracker and target_id:
            diagnostic_tracker.evaluate_probe_step(
                probe_id=f"probe_{int(current_time)}",
                target_id=target_id,
                category=category,
                memory=memory,
                step_result=step_res,
                cand_dense=cand_dense,
                cand_lex=x_t
            )
        return step_res

    # 7. Allocate New Slot (Evict if Full)
    if cand_dense is None:
        t0_emb = time.perf_counter()
        cand_dense = dense_encoder.encode(cand_text)
        t_emb += (time.perf_counter() - t0_emb)

    if not memory.is_full():
        slot = memory.add(x_t, x_t.copy(), candidate.confidence, current_time,
                          debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        slot_texts[id(slot)] = cand_text
        slot_dense_embs.append(cand_dense)
        bm25_index.add_document(id(slot), cand_text)

        if diagnostic_tracker and not is_probe and target_id:
            diagnostic_tracker.register_target(
                target_id=target_id,
                concept_id=concept_id or target_id,
                slot=slot,
                step=int(current_time),
                canonical_text=cand_text,
                dense_emb=cand_dense
            )

        step_res = dict(
            action="insert", slot_index=len(memory.slots) - 1, similarity=best_sim,
            novelty=n_t, gate=gate_reason, cand_text=cand_text,
            retrieved_idx=best_retrieved_idx, top_k_indices=fused_candidates,
            dense_candidates=dense_candidates, lexical_candidates=lexical_candidates,
            fusion_meta=fusion_meta, candidates_examined=candidates_examined,
            beam_policy=beam_policy, applied_penalty=applied_penalty,
            active_anchors=active_anchors_evaluated,
            t_emb=t_emb, t_retr_dense=t_retr_dense, t_retr_bm25=t_retr_bm25, t_nli=t_nli
        )

        if is_probe and diagnostic_tracker and target_id:
            diagnostic_tracker.evaluate_probe_step(
                probe_id=f"probe_{int(current_time)}",
                target_id=target_id,
                category=category,
                memory=memory,
                step_result=step_res,
                cand_dense=cand_dense,
                cand_lex=x_t
            )
        return step_res

    # 8. Eviction Required
    evict_idx, evicted_slot = evict_slot_phase23(
        memory=memory,
        eviction_policy=eviction_policy,
        current_time=current_time,
        slot_texts=slot_texts,
        slot_dense_embs=slot_dense_embs,
        slot_contra_counts=slot_contra_counts if use_contra_shielding else None,
        slot_contra_anchors=slot_contra_anchors if use_contra_shielding else None,
        protected_slot_ids=protected_slot_ids
    )
    bm25_index.remove_document(id(evicted_slot))

    if diagnostic_tracker:
        diagnostic_tracker.record_eviction(evicted_slot, step=int(current_time))

    slot = memory.add(x_t, x_t.copy(), candidate.confidence, current_time,
                      debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    slot_texts[id(slot)] = cand_text
    slot_dense_embs.append(cand_dense)
    bm25_index.add_document(id(slot), cand_text)

    if diagnostic_tracker and not is_probe and target_id:
        diagnostic_tracker.register_target(
            target_id=target_id,
            concept_id=concept_id or target_id,
            slot=slot,
            step=int(current_time),
            canonical_text=cand_text,
            dense_emb=cand_dense
        )

    step_res = dict(
        action="evict_insert", slot_index=len(memory.slots) - 1, evicted_slot_index=evict_idx,
        similarity=best_sim, novelty=n_t, gate=gate_reason, cand_text=cand_text,
        retrieved_idx=best_retrieved_idx, top_k_indices=fused_candidates,
        dense_candidates=dense_candidates, lexical_candidates=lexical_candidates,
        fusion_meta=fusion_meta, candidates_examined=candidates_examined,
        beam_policy=beam_policy, applied_penalty=applied_penalty,
        active_anchors=active_anchors_evaluated,
        t_emb=t_emb, t_retr_dense=t_retr_dense, t_retr_bm25=t_retr_bm25, t_nli=t_nli
    )

    if is_probe and diagnostic_tracker and target_id:
        diagnostic_tracker.evaluate_probe_step(
            probe_id=f"probe_{int(current_time)}",
            target_id=target_id,
            category=category,
            memory=memory,
            step_result=step_res,
            cand_dense=cand_dense,
            cand_lex=x_t
        )
    return step_res


def run_phase23_stream(
    observations,
    capacity: int,
    system: str,  # 'D', 'D4', 'D8', 'H1'
    nli_gate: NliSemanticGate,
    dense_encoder: DenseSemanticEncoder,
    entail_threshold: float = 0.75,
    oracle_retrieval: bool = False,
    protected_retention: bool = False,
    top_k_dense: int = 3,
    top_k_bm25: int = 3,
    k_fusion: int = 5,
    w_anchor: float = FROZEN_W,
    decay_lambda: float = FROZEN_LAMBDA,
    max_anchors: Optional[int] = FROZEN_K,
    dim: int = D_LEX
) -> tuple[dict, Phase23DiagnosticTracker]:
    """Runs an observation stream under Phase 23 architecture with full diagnostics."""
    mem = MemoryState(capacity=capacity, dim=dim)
    slot_texts: dict[int, str] = {}
    slot_dense_embs: list[np.ndarray] = []
    slot_contra_counts: dict[int, int] = {}
    slot_contra_anchors: dict[int, list[tuple[np.ndarray, float]]] = {}
    bm25_index = LocalBM25Index()
    diagnostic_tracker = Phase23DiagnosticTracker()
    protected_slot_ids: set[int] = set()

    trace = []
    n_updates = n_inserts = n_evicts = 0
    gate_counts: dict[str, int] = {}
    t_emb_total = t_retr_dense_total = t_retr_bm25_total = t_nli_total = 0.0
    total_penalty_applied = 0.0
    steps_with_penalty = 0

    if system == "D":
        w = 0.0
        lmb = 0.0
        k = None
    elif system == "D4":
        w = 0.20
        lmb = 0.0
        k = None
    elif system in ("D8", "H1"):
        w = w_anchor
        lmb = decay_lambda
        k = max_anchors
    else:
        w = w_anchor
        lmb = decay_lambda
        k = max_anchors

    t0 = time.perf_counter()

    for obs in observations:
        t_current = float(obs.step_index) if hasattr(obs, "step_index") else float(obs.candidate.timestamp)
        cid = getattr(obs, "concept_id", None)
        is_p = getattr(obs, "is_probe", False)
        cat = getattr(obs, "category", "canonical")
        tgt_id = cid

        step_res = absorb_observation_phase23(
            memory=mem,
            candidate=obs.candidate,
            system=system,
            slot_texts=slot_texts,
            slot_dense_embs=slot_dense_embs,
            slot_contra_counts=slot_contra_counts,
            slot_contra_anchors=slot_contra_anchors,
            bm25_index=bm25_index,
            nli_gate=nli_gate,
            dense_encoder=dense_encoder,
            concept_id=cid,
            is_probe=is_p,
            category=cat,
            target_id=tgt_id,
            entail_threshold=entail_threshold,
            oracle_retrieval=oracle_retrieval,
            protected_slot_ids=protected_slot_ids if protected_retention else None,
            diagnostic_tracker=diagnostic_tracker,
            top_k_dense=top_k_dense,
            top_k_bm25=top_k_bm25,
            k_fusion=k_fusion,
            current_time=t_current,
            dim=dim,
            tau_min=TAU_MIN,
            tau_high=TAU_HIGH,
            delta_margin=DELTA_MARGIN,
            w_anchor=w,
            decay_lambda=lmb,
            max_anchors=k,
            anchor_sim_threshold=TAU_ANCHOR
        )
        trace.append(step_res)

        # Retention Oracle Protection Management:
        if protected_retention and not is_p and cat == "canonical" and tgt_id:
            # Slot created for target should be protected until probed
            t_rec = diagnostic_tracker.targets.get(tgt_id)
            if t_rec and t_rec.target_slot_id:
                protected_slot_ids.add(t_rec.target_slot_id)

        # Unprotect slot after probe evaluation
        if protected_retention and is_p and tgt_id:
            t_rec = diagnostic_tracker.targets.get(tgt_id)
            if t_rec and t_rec.target_slot_id in protected_slot_ids:
                protected_slot_ids.discard(t_rec.target_slot_id)

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
        t_retr_dense_total += step_res.get("t_retr_dense", 0.0)
        t_retr_bm25_total += step_res.get("t_retr_bm25", 0.0)
        t_nli_total += step_res.get("t_nli", 0.0)

    wall_total = time.perf_counter() - t0
    n_obs = len(observations)
    obs_sec = round(n_obs / wall_total, 2) if wall_total > 0 else 0.0

    total_anchors = sum(len(anchors) for anchors in slot_contra_anchors.values())
    avg_penalty = (total_penalty_applied / steps_with_penalty) if steps_with_penalty > 0 else 0.0

    res = {
        "system": system,
        "capacity": capacity,
        "final_slots": len(mem.slots),
        "n_updates": n_updates,
        "n_inserts": n_inserts,
        "n_evicts": n_evicts,
        "nli_calls": sum(s.get("candidates_examined", 0) for s in trace),
        "gate_counts": gate_counts,
        "wall_seconds": round(wall_total, 2),
        "obs_per_second": obs_sec,
        "time_embedding": round(t_emb_total, 3),
        "time_retrieval_dense": round(t_retr_dense_total, 3),
        "time_retrieval_bm25": round(t_retr_bm25_total, 3),
        "time_retrieval": round(t_retr_dense_total + t_retr_bm25_total, 3),
        "time_nli": round(t_nli_total, 3),
        "total_anchors_retained": total_anchors,
        "avg_applied_penalty": round(avg_penalty, 4),
        "steps_with_penalty": steps_with_penalty,
        "trace": trace,
        "memory": mem,
        "entail_threshold": entail_threshold,
        "oracle_retrieval": oracle_retrieval,
        "protected_retention": protected_retention
    }
    return res, diagnostic_tracker
