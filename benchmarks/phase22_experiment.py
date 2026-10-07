"""
benchmarks/phase22_experiment.py

Phase 22 Architecture: Hybrid Retrieval (Dense + Lexical BM25 Candidate Generation).

Core Principles:
1. Frozen Phase 20/21 System D8 Configuration is the Primary Baseline:
   - w* = 0.15, lambda* = 0.005, k* = 3
   - tau_min = 0.40, tau_high = 0.75, delta_margin = 0.10, k_max = 3
   - tau_anchor = 0.70
   - MATCH_THRESHOLD = 0.75 (FROZEN)
   - Core update equation: v_t = (1 - alpha_t)*v_{t-1} + alpha_t*x_t (FROZEN)

2. Local, Deterministic BM25 Implementation:
   - No external APIs or heavy third-party search engines.
   - Standard Okapi BM25: k1 = 1.5, b = 0.75.
   - Operates over stored memory slot text representations.
   - Dynamic document lifecycle: add, update, remove (on eviction).

3. Candidate Fusion:
   - Dense retrieval produces top-k_dense candidates.
   - BM25 produces top-k_bm25 candidates.
   - Fusion: Ordered deduplication, preserving rank order, bounded by k_fusion.
   - Crucial Safety Invariant: Lexical candidates NEVER bypass NLI verification.
     retrieval -> candidate generation -> NLI verification -> consolidation.

4. Evaluated Systems:
   - System D: Phase 18 Baseline (Dense only, utility eviction, w=0, fixed beam)
   - System D4: Phase 19 Baseline (Dense only, static anchors w=0.20, k=unlimited)
   - System D8: Phase 20/21 Frozen Baseline (Dense only, w=0.15, lambda=0.005, k=3)
   - System H1: Hybrid System (Dense + BM25 candidate fusion + D8 anchors)
   - Ablation variants:
     - H_dense_only (= D8)
     - H_bm25_only (BM25 only + NLI + D8 anchors)
     - H_dense_bm25_no_anchor (Dense + BM25, no anchors w=0)
"""

from __future__ import annotations
import os, sys, re, math, time
from typing import Optional, Union

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
    D_DENSE,
    LEXICAL_ENCODER,
    DENSE_MODEL_NAME,
    dense_address,
    utility_score,
)
from benchmarks.phase19_experiment import contradiction_aware_utility
from benchmarks.phase20_experiment import (
    adaptive_dense_address_phase20,
    evict_slot_phase20,
)

# ──────────────────────────────────────────────
#  Frozen Phase 20/21 D8 Configuration Constants
# ──────────────────────────────────────────────
FROZEN_W = 0.15
FROZEN_LAMBDA = 0.005
FROZEN_K = 3
TAU_MIN = 0.40
TAU_HIGH = 0.75
DELTA_MARGIN = 0.10
K_MAX = 3
TAU_ANCHOR = 0.70

# ──────────────────────────────────────────────
#  1. Local Deterministic Okapi BM25 Index
# ──────────────────────────────────────────────

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def bm25_tokenize(text: str) -> list[str]:
    """Deterministic tokenization: lowercase alphanumeric word extraction."""
    return _TOKEN_PATTERN.findall(text.lower())


class LocalBM25Index:
    """
    Deterministic in-memory Okapi BM25 index over slot representations.
    Parameters:
      k1: term frequency saturation parameter (default 1.5)
      b: document length normalization parameter (default 0.75)
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_texts: dict[int, str] = {}
        self.doc_tokens: dict[int, list[str]] = {}
        self.doc_lens: dict[int, int] = {}
        self.term_freqs: dict[int, dict[str, int]] = {}
        self.inv_index: dict[str, set[int]] = {}
        self.total_len: int = 0

    @property
    def num_docs(self) -> int:
        return len(self.doc_tokens)

    @property
    def avgdl(self) -> float:
        n = self.num_docs
        return (self.total_len / n) if n > 0 else 1.0

    def add_document(self, slot_id: int, text: str):
        """Indexes a newly created memory slot."""
        if slot_id in self.doc_tokens:
            self.remove_document(slot_id)

        tokens = bm25_tokenize(text)
        self.doc_texts[slot_id] = text
        self.doc_tokens[slot_id] = tokens
        doc_len = len(tokens)
        self.doc_lens[slot_id] = doc_len
        self.total_len += doc_len

        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
            if token not in self.inv_index:
                self.inv_index[token] = set()
            self.inv_index[token].add(slot_id)
        self.term_freqs[slot_id] = tf

    def update_document(self, slot_id: int, new_text: str):
        """Updates representation of a consolidated memory slot."""
        self.remove_document(slot_id)
        self.add_document(slot_id, new_text)

    def remove_document(self, slot_id: int):
        """Removes an evicted slot from the index."""
        if slot_id not in self.doc_tokens:
            return

        old_tokens = self.doc_tokens.pop(slot_id)
        self.doc_texts.pop(slot_id, None)
        old_len = self.doc_lens.pop(slot_id, 0)
        self.total_len -= old_len
        self.term_freqs.pop(slot_id, None)

        for token in set(old_tokens):
            if token in self.inv_index:
                self.inv_index[token].discard(slot_id)
                if not self.inv_index[token]:
                    del self.inv_index[token]

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        id_to_slot_index: Optional[dict[int, int]] = None
    ) -> list[tuple[int, float]]:
        """
        Calculates Okapi BM25 score for the query across all indexed slots.
        Returns: list of (slot_index, bm25_score) sorted descending by score.
        """
        tokens = bm25_tokenize(query_text)
        if not tokens or self.num_docs == 0:
            return []

        n_docs = self.num_docs
        avg_len = self.avgdl
        scores: dict[int, float] = {}

        for token in tokens:
            matching_ids = self.inv_index.get(token, set())
            df = len(matching_ids)
            if df == 0:
                continue

            # Standard Robertson-Spärck Jones IDF with smoothing
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            if idf <= 0.0:
                continue

            for doc_id in matching_ids:
                tf = self.term_freqs[doc_id].get(token, 0)
                doc_len = self.doc_lens[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_len))
                term_score = idf * (tf * (self.k1 + 1.0)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score

        if not scores:
            return []

        # Sort descending by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in ranked:
            if score <= 0.0:
                continue
            idx = id_to_slot_index.get(doc_id) if id_to_slot_index is not None else doc_id
            if idx is not None:
                results.append((int(idx), float(score)))

        return results


# ──────────────────────────────────────────────
#  2. Hybrid Candidate Fusion
# ──────────────────────────────────────────────

def fuse_candidates(
    dense_candidates: list[tuple[int, float]],
    lexical_candidates: list[tuple[int, float]],
    k_fusion: int = 5
) -> tuple[list[int], dict]:
    """
    Fuses dense and lexical candidate lists via ordered deduplication:
    1. Retains dense candidates (rank-ordered).
    2. Appends lexical candidates not already present in the dense beam.
    3. Truncates to k_fusion candidates.
    Returns:
      (fused_slot_indices, fusion_metadata)
    """
    fused_indices: list[int] = []
    seen: set[int] = set()
    dense_set = set()
    lex_set = set()

    for idx, _ in dense_candidates:
        dense_set.add(idx)
        if idx not in seen:
            fused_indices.append(idx)
            seen.add(idx)

    lex_added = 0
    for idx, _ in lexical_candidates:
        lex_set.add(idx)
        if idx not in seen:
            fused_indices.append(idx)
            seen.add(idx)
            lex_added += 1

    final_candidates = fused_indices[:k_fusion]
    overlap = len(dense_set.intersection(lex_set))

    meta = {
        "n_dense": len(dense_candidates),
        "n_lexical": len(lexical_candidates),
        "n_overlap": overlap,
        "n_lex_added": lex_added,
        "n_fused": len(final_candidates)
    }
    return final_candidates, meta


# ──────────────────────────────────────────────
#  3. Phase 22 Observation Absorption Pipeline
# ──────────────────────────────────────────────

def absorb_observation_phase22(
    memory: MemoryState,
    candidate: Candidate,
    system: str,  # 'D', 'D4', 'D8', 'H1', 'H_dense_only', 'H_bm25_only', 'H_dense_bm25_no_anchor'
    slot_texts: dict[int, str],
    slot_dense_embs: list[np.ndarray],
    slot_contra_counts: dict[int, int],
    slot_contra_anchors: dict[int, list[tuple[np.ndarray, float]]],
    bm25_index: LocalBM25Index,
    nli_gate: NliSemanticGate,
    dense_encoder: DenseSemanticEncoder,
    top_k_dense: int = 3,
    top_k_bm25: int = 3,
    k_fusion: int = 5,
    current_time: float = 0.0,
    entail_threshold: float = 0.50,
    lex_encoder: FieldAwareLexicalEncoder = LEXICAL_ENCODER,
    dim: int = D_LEX,
    # Adaptive beam parameters
    tau_min: float = TAU_MIN,
    tau_high: float = TAU_HIGH,
    delta_margin: float = DELTA_MARGIN,
    # Frozen D8 Anchor Management parameters
    w_anchor: float = FROZEN_W,
    decay_lambda: float = FROZEN_LAMBDA,
    max_anchors: Optional[int] = FROZEN_K,
    anchor_sim_threshold: float = TAU_ANCHOR
) -> dict:
    """
    Absorption Pipeline for Phase 22 supporting Hybrid Candidate Generation:
    - Retains full D8 anchor repulsion and safety mechanics.
    - Fuses dense + BM25 candidates before CrossEncoder NLI verification.
    """
    x_t = lex_encoder.encode_full(candidate, dim=dim)
    cand_text = f"{candidate.subject} {candidate.predicate} {candidate.object}."

    # System mode flags
    use_lexical = system in ("H1", "H_bm25_only", "H_dense_bm25_no_anchor")
    use_dense = system != "H_bm25_only"
    use_adaptive_beam = system not in ("D", "H_bm25_only")
    use_repulsion = system in ("D4", "D8", "H1", "H_dense_only", "H_bm25_only") and (w_anchor > 0.0)
    use_contra_shielding = system not in ("D", "H_dense_bm25_no_anchor")
    eviction_policy = "utility" if system in ("D", "H_dense_bm25_no_anchor") else "contra_utility"

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
        if use_dense:
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
        elif use_dense:
            fused_candidates = [idx for idx, _ in dense_candidates]
            fusion_meta = {"n_dense": len(dense_candidates), "n_lexical": 0, "n_fused": len(fused_candidates)}
        else:
            fused_candidates = [idx for idx, _ in lexical_candidates]
            fusion_meta = {"n_dense": 0, "n_lexical": len(lexical_candidates), "n_fused": len(fused_candidates)}

        # Best similarity score for novelty computation
        best_sim = dense_candidates[0][1] if dense_candidates else 0.0
        best_retrieved_idx = fused_candidates[0] if fused_candidates else None

        # 4. CrossEncoder NLI Verification (CRITICAL: Lexical candidates NEVER bypass NLI)
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
                        if cand_dense is None:
                            t0_emb = time.perf_counter()
                            cand_dense = dense_encoder.encode(cand_text)
                            t_emb += (time.perf_counter() - t0_emb)
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

    # 5. Memory Update Path (Core Vector Update Equation FROZEN)
    if target_idx is not None:
        slot = memory.slots[target_idx]
        slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = current_time
        slot.update_count += 1
        return dict(
            action="update", slot_index=target_idx, similarity=best_sim, novelty=n_t,
            gate="nli_accept", sem_pred=target_pred, cand_text=cand_text,
            retrieved_idx=best_retrieved_idx, top_k_indices=fused_candidates,
            dense_candidates=dense_candidates, lexical_candidates=lexical_candidates,
            fusion_meta=fusion_meta, candidates_examined=candidates_examined,
            beam_policy=beam_policy, applied_penalty=applied_penalty,
            active_anchors=active_anchors_evaluated,
            t_emb=t_emb, t_retr_dense=t_retr_dense, t_retr_bm25=t_retr_bm25, t_nli=t_nli
        )

    # 6. Allocate New Slot (Evict if Full)
    if cand_dense is None and use_dense:
        t0_emb = time.perf_counter()
        cand_dense = dense_encoder.encode(cand_text)
        t_emb += (time.perf_counter() - t0_emb)

    if not memory.is_full():
        slot = memory.add(x_t, x_t.copy(), candidate.confidence, current_time,
                          debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        slot_texts[id(slot)] = cand_text
        if cand_dense is not None:
            slot_dense_embs.append(cand_dense)
        bm25_index.add_document(id(slot), cand_text)
        return dict(
            action="insert", slot_index=len(memory.slots) - 1, similarity=best_sim,
            novelty=n_t, gate=gate_reason, cand_text=cand_text,
            retrieved_idx=best_retrieved_idx, top_k_indices=fused_candidates,
            dense_candidates=dense_candidates, lexical_candidates=lexical_candidates,
            fusion_meta=fusion_meta, candidates_examined=candidates_examined,
            beam_policy=beam_policy, applied_penalty=applied_penalty,
            active_anchors=active_anchors_evaluated,
            t_emb=t_emb, t_retr_dense=t_retr_dense, t_retr_bm25=t_retr_bm25, t_nli=t_nli
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
    # Synchronize BM25 index with eviction
    bm25_index.remove_document(id(evicted_slot))

    slot = memory.add(x_t, x_t.copy(), candidate.confidence, current_time,
                      debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    slot_texts[id(slot)] = cand_text
    if cand_dense is not None:
        slot_dense_embs.append(cand_dense)
    bm25_index.add_document(id(slot), cand_text)

    return dict(
        action="evict_insert", slot_index=len(memory.slots) - 1, evicted_slot_index=evict_idx,
        similarity=best_sim, novelty=n_t, gate=gate_reason, cand_text=cand_text,
        retrieved_idx=best_retrieved_idx, top_k_indices=fused_candidates,
        dense_candidates=dense_candidates, lexical_candidates=lexical_candidates,
        fusion_meta=fusion_meta, candidates_examined=candidates_examined,
        beam_policy=beam_policy, applied_penalty=applied_penalty,
        active_anchors=active_anchors_evaluated,
        t_emb=t_emb, t_retr_dense=t_retr_dense, t_retr_bm25=t_retr_bm25, t_nli=t_nli
    )


# ──────────────────────────────────────────────
#  4. Phase 22 Stream Runner
# ──────────────────────────────────────────────

def run_phase22_stream(
    observations,
    capacity: int,
    system: str,  # 'D', 'D4', 'D8', 'H1', 'H_dense_only', 'H_bm25_only', 'H_dense_bm25_no_anchor'
    nli_gate: NliSemanticGate,
    dense_encoder: DenseSemanticEncoder,
    top_k_dense: int = 3,
    top_k_bm25: int = 3,
    k_fusion: int = 5,
    # Parameters defaulted strictly to frozen D8 parameters
    w_anchor: float = FROZEN_W,
    decay_lambda: float = FROZEN_LAMBDA,
    max_anchors: Optional[int] = FROZEN_K,
    dim: int = D_LEX
) -> dict:
    """Runs an observation stream under Phase 22 architecture."""
    mem = MemoryState(capacity=capacity, dim=dim)
    slot_texts: dict[int, str] = {}
    slot_dense_embs: list[np.ndarray] = []
    slot_contra_counts: dict[int, int] = {}
    slot_contra_anchors: dict[int, list[tuple[np.ndarray, float]]] = {}
    bm25_index = LocalBM25Index()

    trace = []
    n_updates = n_inserts = n_evicts = 0
    gate_counts: dict[str, int] = {}
    t_emb_total = t_retr_dense_total = t_retr_bm25_total = t_nli_total = 0.0
    total_penalty_applied = 0.0
    steps_with_penalty = 0

    # System-specific parameter defaults
    if system == "D":
        w = 0.0
        lmb = 0.0
        k = None
    elif system == "D4":
        w = 0.20
        lmb = 0.0
        k = None
    elif system in ("D8", "H1", "H_dense_only", "H_bm25_only"):
        w = w_anchor
        lmb = decay_lambda
        k = max_anchors
    elif system == "H_dense_bm25_no_anchor":
        w = 0.0
        lmb = 0.0
        k = None
    else:
        w = w_anchor
        lmb = decay_lambda
        k = max_anchors

    t0 = time.perf_counter()

    for obs in observations:
        t_current = float(obs.step_index) if hasattr(obs, "step_index") else float(obs.candidate.timestamp)
        step_res = absorb_observation_phase22(
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

    wall_sec = time.perf_counter() - t0
    total_nli_calls = sum(tr.get("candidates_examined", 0) for tr in trace)

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
        time_retrieval=round(t_retr_dense_total + t_retr_bm25_total, 3),
        time_retrieval_dense=round(t_retr_dense_total, 3),
        time_retrieval_bm25=round(t_retr_bm25_total, 3),
        time_nli=round(t_nli_total, 3),
        total_anchors_retained=total_anchors_retained,
        avg_anchor_age=avg_anchor_age,
        avg_applied_penalty=avg_applied_penalty,
        steps_with_penalty=steps_with_penalty,
        trace=trace,
        memory=mem
    )
