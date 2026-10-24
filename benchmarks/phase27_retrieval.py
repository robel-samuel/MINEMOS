"""
benchmarks/phase27_retrieval.py

Retrieval methods and ranking algorithms for MINEMOS Phase 27:
Adaptive Memory Retrieval Under Long-Horizon Interference.

Implements:
  1. Dense Baseline (query() as existing in BlackHoleMemory)
  2. BM25-Only Retriever
  3. Relevance-Only Hybrid Retriever (Dense + BM25 RRF)
  4. Temporal-Aware Retriever (Hybrid + temporal decay & window gating)
  5. Contradiction-Temporal Aware Retriever (incorporating contradiction anchors)
  6. Proposed Full Adaptive Retriever (calibrated multi-factor fusion)

Supports modular evaluation over BlackHoleMemory instances.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from phase2.black_hole_memory import BlackHoleMemory
from phase2.memory_state import MemorySlot
from benchmarks.phase22_experiment import FROZEN_W, FROZEN_LAMBDA, TAU_ANCHOR


def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """Min-max normalizes scores to [0.0, 1.0]."""
    if not scores:
        return {}
    vals = list(scores.values())
    min_v = min(vals)
    max_v = max(vals)
    if max_v - min_v < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


class Phase27Retriever:
    """
    Modular retrieval engine operating over a BlackHoleMemory instance.
    """

    def __init__(
        self,
        memory_engine: BlackHoleMemory,
        rrf_k: int = 60,
        w_dense: float = 0.60,
        w_lex: float = 0.40,
        temporal_decay_lambda: float = 0.001,
        inactive_penalty: float = 0.05,
        w_anchor: float = 0.25,
        decay_anchor: float = 0.005,
        tau_anchor: float = 0.65,
        tier_boost_factor: float = 0.15
    ):
        self.mem = memory_engine
        self.rrf_k = rrf_k
        self.w_dense = w_dense
        self.w_lex = w_lex
        self.temporal_decay_lambda = temporal_decay_lambda
        self.inactive_penalty = inactive_penalty
        self.w_anchor = w_anchor
        self.decay_anchor = decay_anchor
        self.tau_anchor = tau_anchor
        self.tier_boost_factor = tier_boost_factor

    # ─────────────────────────────────────────────────────────
    # 1. Base Dense Retriever (Existing Baseline)
    # ─────────────────────────────────────────────────────────
    def retrieve_dense_baseline(
        self,
        query_text: str,
        top_k: int = 5,
        as_of: Optional[float] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Direct call to BlackHoleMemory.query()."""
        return self.mem.query(query_text=query_text, top_k=top_k, as_of=as_of, active_only=active_only)

    # ─────────────────────────────────────────────────────────
    # 2. BM25-Only Retriever
    # ─────────────────────────────────────────────────────────
    def retrieve_bm25_only(
        self,
        query_text: str,
        top_k: int = 5,
        as_of: Optional[float] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Pure lexical BM25 ranking."""
        if not self.mem.memory.slots:
            return []

        id_to_slot = {id(s): (idx, s) for idx, s in enumerate(self.mem.memory.slots)}
        bm25_raw = self.mem.bm25_index.query(
            query_text=query_text,
            top_k=len(self.mem.memory.slots),
            id_to_slot_index={id(s): idx for idx, s in enumerate(self.mem.memory.slots)}
        )

        results = []
        for idx, score in bm25_raw:
            if idx >= len(self.mem.memory.slots):
                continue
            slot = self.mem.memory.slots[idx]

            # Filtering
            if as_of is not None:
                if slot.valid_from > as_of or (slot.valid_until is not None and slot.valid_until <= as_of):
                    continue
            elif active_only and not slot.is_active:
                continue

            results.append({
                "slot_index": idx,
                "slot_id": id(slot),
                "score": round(float(score), 4),
                "similarity": round(float(score), 4),
                "text": self.mem.slot_texts.get(id(slot), slot.text),
                "subject": slot.debug_subject,
                "predicate": slot.debug_predicate,
                "is_active": slot.is_active,
                "valid_from": slot.valid_from,
                "valid_until": slot.valid_until,
                "update_count": slot.update_count
            })
            if len(results) >= top_k:
                break
        return results

    # ─────────────────────────────────────────────────────────
    # 3. Relevance-Only Hybrid Retriever (Dense + BM25 RRF)
    # ─────────────────────────────────────────────────────────
    def retrieve_hybrid_relevance(
        self,
        query_text: str,
        top_k: int = 5,
        as_of: Optional[float] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion over Dense and BM25 without temporal or contradiction terms."""
        if not self.mem.memory.slots:
            return []

        q_dense = self.mem.dense_encoder.encode(query_text)
        n_slots = len(self.mem.memory.slots)

        # 1. Dense ranking
        dense_scores = []
        for idx, slot in enumerate(self.mem.memory.slots):
            emb = self.mem.slot_dense_embs[idx] if idx < len(self.mem.slot_dense_embs) else slot.dense_emb
            sim = float(np.dot(emb, q_dense)) if emb is not None else 0.0
            dense_scores.append((idx, sim))
        dense_scores.sort(key=lambda x: -x[1])
        dense_rank = {idx: r + 1 for r, (idx, _) in enumerate(dense_scores)}

        # 2. BM25 ranking
        bm25_raw = self.mem.bm25_index.query(
            query_text=query_text,
            top_k=n_slots,
            id_to_slot_index={id(s): idx for idx, s in enumerate(self.mem.memory.slots)}
        )
        bm25_rank = {idx: r + 1 for r, (idx, _) in enumerate(bm25_raw)}

        # 3. RRF combination
        fused = []
        for idx, slot in enumerate(self.mem.memory.slots):
            # Filtering
            if as_of is not None:
                if slot.valid_from > as_of or (slot.valid_until is not None and slot.valid_until <= as_of):
                    continue
            elif active_only and not slot.is_active:
                continue

            r_dense = dense_rank.get(idx, n_slots + 10)
            r_lex = bm25_rank.get(idx, n_slots + 10)
            rrf_score = (self.w_dense / (self.rrf_k + r_dense)) + (self.w_lex / (self.rrf_k + r_lex))
            fused.append((rrf_score, idx, slot))

        fused.sort(key=lambda x: -x[0])
        results = []
        for rrf_score, idx, slot in fused[:top_k]:
            results.append({
                "slot_index": idx,
                "slot_id": id(slot),
                "score": round(float(rrf_score), 6),
                "similarity": round(float(rrf_score), 6),
                "text": self.mem.slot_texts.get(id(slot), slot.text),
                "subject": slot.debug_subject,
                "predicate": slot.debug_predicate,
                "is_active": slot.is_active,
                "valid_from": slot.valid_from,
                "valid_until": slot.valid_until,
                "update_count": slot.update_count
            })
        return results

    # ─────────────────────────────────────────────────────────
    # 4. Temporal-Aware Retriever
    # ─────────────────────────────────────────────────────────
    def retrieve_temporal_aware(
        self,
        query_text: str,
        top_k: int = 5,
        as_of: Optional[float] = None,
        query_time: float = 1000.0,
        active_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval modulated by temporal validity and recency decay.
        Evaluates inactive slots but heavily demotes them if as_of is None.
        """
        if not self.mem.memory.slots:
            return []

        q_dense = self.mem.dense_encoder.encode(query_text)
        n_slots = len(self.mem.memory.slots)

        dense_scores = {}
        for idx, slot in enumerate(self.mem.memory.slots):
            emb = self.mem.slot_dense_embs[idx] if idx < len(self.mem.slot_dense_embs) else slot.dense_emb
            dense_scores[idx] = float(np.dot(emb, q_dense)) if emb is not None else 0.0

        bm25_raw = self.mem.bm25_index.query(
            query_text=query_text,
            top_k=n_slots,
            id_to_slot_index={id(s): idx for idx, s in enumerate(self.mem.memory.slots)}
        )
        bm25_scores = {idx: score for idx, score in bm25_raw}

        norm_dense = normalize_scores(dense_scores)
        norm_bm25 = normalize_scores(bm25_scores)

        scored = []
        for idx, slot in enumerate(self.mem.memory.slots):
            # Point-in-time filtering if as_of is provided
            if as_of is not None:
                if slot.valid_from > as_of or (slot.valid_until is not None and slot.valid_until <= as_of):
                    continue
                temp_factor = 1.0
            else:
                # Querying latest state: active slots favored, inactive slots penalized
                if slot.is_active:
                    delta_t = max(0.0, query_time - slot.valid_from)
                    recency = math.exp(-self.temporal_decay_lambda * delta_t)
                    temp_factor = 1.0 + 0.10 * recency
                else:
                    temp_factor = self.inactive_penalty

            rel = (self.w_dense * norm_dense.get(idx, 0.0)) + (self.w_lex * norm_bm25.get(idx, 0.0))
            final_score = rel * temp_factor
            scored.append((final_score, idx, slot))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, idx, slot in scored[:top_k]:
            results.append({
                "slot_index": idx,
                "slot_id": id(slot),
                "score": round(float(score), 6),
                "similarity": round(float(score), 6),
                "text": self.mem.slot_texts.get(id(slot), slot.text),
                "subject": slot.debug_subject,
                "predicate": slot.debug_predicate,
                "is_active": slot.is_active,
                "valid_from": slot.valid_from,
                "valid_until": slot.valid_until,
                "update_count": slot.update_count
            })
        return results

    # ─────────────────────────────────────────────────────────
    # 5. Relevance + Temporal + Contradiction-Aware Retriever
    # ─────────────────────────────────────────────────────────
    def retrieve_contra_temporal_aware(
        self,
        query_text: str,
        top_k: int = 5,
        as_of: Optional[float] = None,
        query_time: float = 1000.0
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval with temporal validity and contradiction repulsion penalties.
        """
        if not self.mem.memory.slots:
            return []

        q_dense = self.mem.dense_encoder.encode(query_text)
        n_slots = len(self.mem.memory.slots)

        dense_scores = {}
        for idx, slot in enumerate(self.mem.memory.slots):
            emb = self.mem.slot_dense_embs[idx] if idx < len(self.mem.slot_dense_embs) else slot.dense_emb
            dense_scores[idx] = float(np.dot(emb, q_dense)) if emb is not None else 0.0

        bm25_raw = self.mem.bm25_index.query(
            query_text=query_text,
            top_k=n_slots,
            id_to_slot_index={id(s): idx for idx, s in enumerate(self.mem.memory.slots)}
        )
        bm25_scores = {idx: score for idx, score in bm25_raw}

        norm_dense = normalize_scores(dense_scores)
        norm_bm25 = normalize_scores(bm25_scores)

        scored = []
        for idx, slot in enumerate(self.mem.memory.slots):
            slot_id = id(slot)

            if as_of is not None:
                if slot.valid_from > as_of or (slot.valid_until is not None and slot.valid_until <= as_of):
                    continue
                temp_factor = 1.0
            else:
                if slot.is_active:
                    delta_t = max(0.0, query_time - slot.valid_from)
                    recency = math.exp(-self.temporal_decay_lambda * delta_t)
                    temp_factor = 1.0 + 0.10 * recency
                else:
                    temp_factor = self.inactive_penalty

            rel = (self.w_dense * norm_dense.get(idx, 0.0)) + (self.w_lex * norm_bm25.get(idx, 0.0))

            # Contradiction anchor repulsion penalty
            contra_penalty = 0.0
            anchors = self.mem.slot_contra_anchors.get(slot_id, [])
            for a_emb, a_t in anchors:
                a_sim = float(np.dot(a_emb, q_dense))
                if a_sim > self.tau_anchor:
                    decay = math.exp(-self.decay_anchor * max(0.0, query_time - a_t))
                    contra_penalty = max(contra_penalty, self.w_anchor * a_sim * decay)

            final_score = max(0.0, rel * temp_factor - contra_penalty)
            scored.append((final_score, idx, slot))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, idx, slot in scored[:top_k]:
            results.append({
                "slot_index": idx,
                "slot_id": id(slot),
                "score": round(float(score), 6),
                "similarity": round(float(score), 6),
                "text": self.mem.slot_texts.get(id(slot), slot.text),
                "subject": slot.debug_subject,
                "predicate": slot.debug_predicate,
                "is_active": slot.is_active,
                "valid_from": slot.valid_from,
                "valid_until": slot.valid_until,
                "update_count": slot.update_count
            })
        return results

    # ─────────────────────────────────────────────────────────
    # 6. Proposed Full Adaptive Retriever (Calibrated Multi-Factor)
    # ─────────────────────────────────────────────────────────
    def retrieve_adaptive(
        self,
        query_text: str,
        top_k: int = 5,
        as_of: Optional[float] = None,
        query_time: float = 1000.0
    ) -> List[Dict[str, Any]]:
        """
        Proposed Adaptive Retriever:
          - Balanced Dense + Lexical normalisation
          - Strict point-in-time window validation for historical queries
          - Temporal freshness boost for active state
          - Inactive fact demotion with supersession link verification
          - Contradiction anchor repulsion
          - Tiered consolidation boost for corroborated facts (update_count > 1)
        """
        if not self.mem.memory.slots:
            return []

        q_dense = self.mem.dense_encoder.encode(query_text)
        n_slots = len(self.mem.memory.slots)

        dense_scores = {}
        for idx, slot in enumerate(self.mem.memory.slots):
            emb = self.mem.slot_dense_embs[idx] if idx < len(self.mem.slot_dense_embs) else slot.dense_emb
            dense_scores[idx] = float(np.dot(emb, q_dense)) if emb is not None else 0.0

        bm25_raw = self.mem.bm25_index.query(
            query_text=query_text,
            top_k=n_slots,
            id_to_slot_index={id(s): idx for idx, s in enumerate(self.mem.memory.slots)}
        )
        bm25_scores = {idx: score for idx, score in bm25_raw}

        norm_dense = normalize_scores(dense_scores)
        norm_bm25 = normalize_scores(bm25_scores)

        scored = []
        for idx, slot in enumerate(self.mem.memory.slots):
            slot_id = id(slot)

            # 1. Point-in-time filtering
            if as_of is not None:
                if slot.valid_from > as_of or (slot.valid_until is not None and slot.valid_until <= as_of):
                    continue
                temporal_factor = 1.0
            else:
                # Evaluating current state
                if slot.is_active:
                    delta_t = max(0.0, query_time - slot.valid_from)
                    recency = math.exp(-self.temporal_decay_lambda * delta_t)
                    temporal_factor = 1.0 + 0.15 * recency
                else:
                    # Inactive slot: heavy penalty if querying active truth
                    temporal_factor = self.inactive_penalty

            # 2. Hybrid relevance
            s_dense = norm_dense.get(idx, 0.0)
            s_lex = norm_bm25.get(idx, 0.0)
            relevance = (self.w_dense * s_dense) + (self.w_lex * s_lex)

            # 3. Corroboration / Tier boost
            # Rewards consolidated slots (Tier 1) over uncorroborated singletons
            corroboration = 1.0 + self.tier_boost_factor * min(max(0, slot.update_count - 1), 3)

            # 4. Contradiction anchor repulsion
            contra_penalty = 0.0
            anchors = self.mem.slot_contra_anchors.get(slot_id, [])
            for a_emb, a_t in anchors:
                a_sim = float(np.dot(a_emb, q_dense))
                if a_sim > self.tau_anchor:
                    decay = math.exp(-self.decay_anchor * max(0.0, query_time - a_t))
                    contra_penalty = max(contra_penalty, self.w_anchor * a_sim * decay)

            # Composite Score
            composite = max(0.0, (relevance * temporal_factor * corroboration) - contra_penalty)
            scored.append((composite, idx, slot))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, idx, slot in scored[:top_k]:
            results.append({
                "slot_index": idx,
                "slot_id": id(slot),
                "score": round(float(score), 6),
                "similarity": round(float(score), 6),
                "text": self.mem.slot_texts.get(id(slot), slot.text),
                "subject": slot.debug_subject,
                "predicate": slot.debug_predicate,
                "is_active": slot.is_active,
                "valid_from": slot.valid_from,
                "valid_until": slot.valid_until,
                "update_count": slot.update_count
            })
        return results

    # ─────────────────────────────────────────────────────────
    # Dispatcher supporting 6 Ablation Modes
    # ─────────────────────────────────────────────────────────
    def retrieve(
        self,
        method: str,
        query_text: str,
        top_k: int = 5,
        as_of: Optional[float] = None,
        query_time: float = 1000.0
    ) -> List[Dict[str, Any]]:
        """
        Unified dispatcher supporting all methods and ablations:
          - 'dense_baseline'
          - 'bm25_only'
          - 'hybrid_relevance'
          - 'temporal_aware'
          - 'contra_temporal_aware'
          - 'adaptive'
          - 'hybrid_temporal_only' (Ablation 4)
          - 'hybrid_contra_only' (Ablation 5)
        """
        if method == "dense_baseline":
            return self.retrieve_dense_baseline(query_text=query_text, top_k=top_k, as_of=as_of)
        elif method == "bm25_only":
            return self.retrieve_bm25_only(query_text=query_text, top_k=top_k, as_of=as_of)
        elif method == "hybrid_relevance":
            return self.retrieve_hybrid_relevance(query_text=query_text, top_k=top_k, as_of=as_of)
        elif method in ("temporal_aware", "hybrid_temporal_only"):
            return self.retrieve_temporal_aware(query_text=query_text, top_k=top_k, as_of=as_of, query_time=query_time)
        elif method == "hybrid_contra_only":
            # Hybrid + Contradiction anchor without temporal decay
            saved_decay = self.temporal_decay_lambda
            saved_penalty = self.inactive_penalty
            self.temporal_decay_lambda = 0.0
            self.inactive_penalty = 1.0  # No temporal penalty
            res = self.retrieve_contra_temporal_aware(query_text=query_text, top_k=top_k, as_of=as_of, query_time=query_time)
            self.temporal_decay_lambda = saved_decay
            self.inactive_penalty = saved_penalty
            return res
        elif method == "contra_temporal_aware":
            return self.retrieve_contra_temporal_aware(query_text=query_text, top_k=top_k, as_of=as_of, query_time=query_time)
        elif method == "adaptive":
            return self.retrieve_adaptive(query_text=query_text, top_k=top_k, as_of=as_of, query_time=query_time)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
