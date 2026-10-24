"""
benchmarks/phase24_experiment.py

Phase 24 Core Execution Engine:
Integrates retention hardening, semantic verification recalibration,
and drift-resistant consolidation into the Black Hole Memory architecture.

Configurable Dimensions:
  1. retention_policy: 'fifo', 'recency', 'utility', 'verification_aware', 'tiered', 'oracle'
  2. verification_mode: 'baseline', 'asymmetric', 'bidirectional'
  3. entail_threshold: float (0.55 - 0.75)
  4. consolidation_mode: 'baseline', 'bounded', 'anchor', 'hybrid'
  5. delta_max: float (displacement bound)
  6. lambda_anchor: float (anchor regularization weight)
"""

from __future__ import annotations
import os, sys, time
from typing import Optional, Set, Any
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
)
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
from benchmarks.phase24_retention import evict_slot_phase24
from benchmarks.phase24_consolidation import consolidate_vector_phase24
from benchmarks.phase24_verification import CalibratedSemanticVerifier
from benchmarks.phase24_diagnostics import Phase24DiagnosticTracker


def absorb_observation_phase24(
    memory: MemoryState,
    candidate: Candidate,
    system: str,  # 'D', 'D8', 'H1'
    slot_texts: dict[int, str],
    slot_dense_embs: list[np.ndarray],
    slot_contra_counts: dict[int, int],
    slot_contra_anchors: dict[int, list[tuple[np.ndarray, float]]],
    bm25_index: LocalBM25Index,
    calibrated_verifier: CalibratedSemanticVerifier,
    dense_encoder: DenseSemanticEncoder,
    concept_id: Optional[str] = None,
    is_probe: bool = False,
    category: str = "canonical",
    target_id: Optional[str] = None,
    # Phase 24 policy switches
    retention_policy: str = "fifo",
    verification_mode: str = "baseline",
    entail_threshold: float = 0.75,
    contra_threshold: float = 0.50,
    consolidation_mode: str = "baseline",
    delta_max: float = 0.25,
    lambda_anchor: float = 0.20,
    protected_slot_ids: Optional[Set[int]] = None,
    diagnostic_tracker: Optional[Phase24DiagnosticTracker] = None,
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
    Core Phase 24 absorption step with configurable retention, verification,
    and consolidation modules.
    """
    x_t = lex_encoder.encode_full(candidate, dim=dim)
    cand_text = f"{candidate.subject} {candidate.predicate} {candidate.object}."

    use_lexical = (system == "H1")
    use_dense = True
    use_adaptive_beam = (system != "D")
    use_repulsion = (system in ("D8", "H1")) and (w_anchor > 0.0)
    use_contra_shielding = (system != "D")

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
    cand_dense = None

    if memory.slots:
        id_to_slot_index = {id(slot): idx for idx, slot in enumerate(memory.slots)}

        # 1. Dense candidate retrieval
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

        # 2. Lexical BM25 retrieval
        if use_lexical:
            t0_retr_bm25 = time.perf_counter()
            lexical_candidates = bm25_index.query(
                query_text=cand_text,
                top_k=top_k_bm25,
                id_to_slot_index=id_to_slot_index
            )
            t_retr_bm25 += (time.perf_counter() - t0_retr_bm25)

        # 3. Candidate fusion
        if use_dense and use_lexical:
            fused_candidates, _ = fuse_candidates(
                dense_candidates=dense_candidates,
                lexical_candidates=lexical_candidates,
                k_fusion=k_fusion
            )
        else:
            fused_candidates = [idx for idx, _ in dense_candidates]

        best_sim = dense_candidates[0][1] if dense_candidates else 0.0
        best_retrieved_idx = fused_candidates[0] if fused_candidates else None

        # 4. Calibrated Semantic Verification
        if fused_candidates:
            t0_nli = time.perf_counter()
            pairs = [(slot_texts[id(memory.slots[idx])], cand_text) for idx in fused_candidates]
            preds = calibrated_verifier.verify_batch(
                pairs,
                mode=verification_mode,
                entail_threshold=entail_threshold,
                contra_threshold=contra_threshold
            )
            for cand_idx, pred in zip(fused_candidates, preds):
                candidates_examined += pred.get("nli_invocations", 1)
                dec = pred["decision"]
                if dec == "SAME":
                    target_idx = cand_idx
                    target_pred = pred
                    gate_reason = "nli_accept"
                    break
                elif dec == "CONTRADICTION":
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

    # 5. Consolidation Update Path
    if target_idx is not None:
        slot = memory.slots[target_idx]

        # Apply configurable consolidation operator
        slot.value = consolidate_vector_phase24(
            slot=slot,
            x_t=x_t,
            alpha_t=alpha_t,
            mode=consolidation_mode,
            delta_max=delta_max,
            lambda_anchor=lambda_anchor
        )
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = current_time
        slot.update_count += 1

        if diagnostic_tracker:
            diagnostic_tracker.record_update(
                slot=slot,
                step=int(current_time),
                update_text=cand_text,
                cand_lex=x_t
            )

        step_res = dict(
            action="update", slot_index=target_idx, similarity=best_sim, novelty=n_t,
            gate="nli_accept", sem_pred=target_pred, cand_text=cand_text,
            retrieved_idx=best_retrieved_idx, top_k_indices=fused_candidates,
            dense_candidates=dense_candidates, lexical_candidates=lexical_candidates,
            candidates_examined=candidates_examined, beam_policy=beam_policy,
            applied_penalty=applied_penalty, active_anchors=active_anchors_evaluated,
            t_emb=t_emb, t_retr_dense=t_retr_dense, t_retr_bm25=t_retr_bm25, t_nli=t_nli
        )

        if is_probe and diagnostic_tracker and target_id:
            diagnostic_tracker.evaluate_probe_step(
                probe_id=f"probe_{int(current_time)}",
                target_id=target_id,
                category=category,
                memory=memory,
                step_result=step_res,
                cand_lex=x_t
            )
        return step_res

    # 6. Allocate New Slot (Insert or Evict-Insert)
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
            candidates_examined=candidates_examined, beam_policy=beam_policy,
            applied_penalty=applied_penalty, active_anchors=active_anchors_evaluated,
            t_emb=t_emb, t_retr_dense=t_retr_dense, t_retr_bm25=t_retr_bm25, t_nli=t_nli
        )

        if is_probe and diagnostic_tracker and target_id:
            diagnostic_tracker.evaluate_probe_step(
                probe_id=f"probe_{int(current_time)}",
                target_id=target_id,
                category=category,
                memory=memory,
                step_result=step_res,
                cand_lex=x_t
            )
        return step_res

    # 7. Eviction Required under specified retention_policy
    evict_idx, evicted_slot = evict_slot_phase24(
        memory=memory,
        eviction_policy=retention_policy,
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
        candidates_examined=candidates_examined, beam_policy=beam_policy,
        applied_penalty=applied_penalty, active_anchors=active_anchors_evaluated,
        t_emb=t_emb, t_retr_dense=t_retr_dense, t_retr_bm25=t_retr_bm25, t_nli=t_nli
    )

    if is_probe and diagnostic_tracker and target_id:
        diagnostic_tracker.evaluate_probe_step(
            probe_id=f"probe_{int(current_time)}",
            target_id=target_id,
            category=category,
            memory=memory,
            step_result=step_res,
            cand_lex=x_t
        )
    return step_res


def run_phase24_stream(
    observations,
    capacity: int,
    system: str = "D8",
    retention_policy: str = "fifo",
    verification_mode: str = "baseline",
    entail_threshold: float = 0.75,
    contra_threshold: float = 0.50,
    consolidation_mode: str = "baseline",
    delta_max: float = 0.25,
    lambda_anchor: float = 0.20,
    calibrated_verifier: Optional[CalibratedSemanticVerifier] = None,
    dense_encoder: Optional[DenseSemanticEncoder] = None,
    protected_retention: bool = False,
    top_k_dense: int = 3,
    top_k_bm25: int = 3,
    k_fusion: int = 5,
    w_anchor: float = FROZEN_W,
    decay_lambda: float = FROZEN_LAMBDA,
    max_anchors: Optional[int] = FROZEN_K,
    dim: int = D_LEX
) -> tuple[dict, Phase24DiagnosticTracker]:
    """Runs a complete Phase 24 stream with chosen policy parameters."""
    mem = MemoryState(capacity=capacity, dim=dim)
    slot_texts: dict[int, str] = {}
    slot_dense_embs: list[np.ndarray] = []
    slot_contra_counts: dict[int, int] = {}
    slot_contra_anchors: dict[int, list[tuple[np.ndarray, float]]] = {}
    bm25_index = LocalBM25Index()
    diagnostic_tracker = Phase24DiagnosticTracker()
    protected_slot_ids: Set[int] = set()

    trace = []
    n_updates = n_inserts = n_evicts = 0
    t_emb_total = t_retr_dense_total = t_retr_bm25_total = t_nli_total = 0.0

    if system == "D":
        w = 0.0
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

        step_res = absorb_observation_phase24(
            memory=mem,
            candidate=obs.candidate,
            system=system,
            slot_texts=slot_texts,
            slot_dense_embs=slot_dense_embs,
            slot_contra_counts=slot_contra_counts,
            slot_contra_anchors=slot_contra_anchors,
            bm25_index=bm25_index,
            calibrated_verifier=calibrated_verifier,
            dense_encoder=dense_encoder,
            concept_id=cid,
            is_probe=is_p,
            category=cat,
            target_id=tgt_id,
            retention_policy=retention_policy,
            verification_mode=verification_mode,
            entail_threshold=entail_threshold,
            contra_threshold=contra_threshold,
            consolidation_mode=consolidation_mode,
            delta_max=delta_max,
            lambda_anchor=lambda_anchor,
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

        if protected_retention and not is_p and cat == "canonical" and tgt_id:
            t_rec = diagnostic_tracker.targets.get(tgt_id)
            if t_rec and t_rec.target_slot_id:
                protected_slot_ids.add(t_rec.target_slot_id)
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

        t_emb_total += step_res.get("t_emb", 0.0)
        t_retr_dense_total += step_res.get("t_retr_dense", 0.0)
        t_retr_bm25_total += step_res.get("t_retr_bm25", 0.0)
        t_nli_total += step_res.get("t_nli", 0.0)

    wall_total = time.perf_counter() - t0

    res = {
        "system": system,
        "capacity": capacity,
        "final_slots": len(mem.slots),
        "n_updates": n_updates,
        "n_inserts": n_inserts,
        "n_evicts": n_evicts,
        "nli_calls": sum(s.get("candidates_examined", 0) for s in trace),
        "wall_seconds": round(wall_total, 2),
        "time_embedding": round(t_emb_total, 3),
        "time_retrieval_dense": round(t_retr_dense_total, 3),
        "time_retrieval_bm25": round(t_retr_bm25_total, 3),
        "time_retrieval": round(t_retr_dense_total + t_retr_bm25_total, 3),
        "time_nli": round(t_nli_total, 3),
        "trace": trace,
        "memory": mem,
        "retention_policy": retention_policy,
        "verification_mode": verification_mode,
        "entail_threshold": entail_threshold,
        "contra_threshold": contra_threshold,
        "consolidation_mode": consolidation_mode
    }
    return res, diagnostic_tracker
