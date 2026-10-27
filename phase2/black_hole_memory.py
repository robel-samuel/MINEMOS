"""
phase2/black_hole_memory.py

MINEMOS Phase 25 Unified Black Hole Memory Architecture.

Implements the complete, consolidated production-grade memory system:
  1. Hybrid Retrieval: Expanded candidate beam (Dense bi-encoder + Local BM25)
     with adaptive anchor repulsion to prevent retrieval misses.
  2. Calibrated Semantic Verification: Bidirectional NLI equivalence check
     with strict conservative contradiction veto (tau_contra = 0.40, tau_entail = 0.60).
  3. Temporal Fact Handling: Explicit validity windows (valid_from, valid_until),
     supersession tracking, and historical point-in-time recall.
  4. Drift-Resistant Consolidation: Bounded displacement (delta_max = 0.25)
     and anchor regularization (lambda_anchor = 0.20) toward root key v_0.
  5. Tiered Retention: Capacity-bounded memory with provisional working buffer
     (update_count == 1) and protected long-term storage (update_count >= 2).
"""

from __future__ import annotations
import math
import re
import threading
import time
from typing import Optional, Union, List, Dict, Any, Tuple

import numpy as np

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import cosine_similarity, novelty
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
    TAU_ANCHOR,
)
from benchmarks.phase24_retention import evict_slot_phase24
from benchmarks.phase24_consolidation import consolidate_vector_phase24
from benchmarks.phase24_verification import CalibratedSemanticVerifier


class BlackHoleMemory:
    """
    Unified Black Hole Memory System for streaming factual knowledge.
    
    Provides high-level APIs:
      - absorb(candidate, timestamp) -> dict
      - query(query_text, top_k, as_of) -> list[dict]
      - recall(subject, predicate, as_of) -> list[dict]
      - get_history(subject, predicate) -> list[dict]
      - status() -> dict
    """

    def __init__(
        self,
        capacity: int = 100,
        dim: int = D_LEX,
        retention_policy: str = "tiered",
        verification_mode: str = "bidirectional",
        entail_threshold: float = 0.60,
        contra_threshold: float = 0.40,
        consolidation_mode: str = "hybrid",
        delta_max: float = 0.25,
        lambda_anchor: float = 0.20,
        top_k_dense: int = 8,
        top_k_bm25: int = 8,
        k_fusion: int = 10,
        w_anchor: float = FROZEN_W,
        decay_lambda: float = FROZEN_LAMBDA,
        max_anchors: int = FROZEN_K,
        dense_encoder: Optional[DenseSemanticEncoder] = None,
        nli_gate: Optional[NliSemanticGate] = None,
        lex_encoder: Optional[FieldAwareLexicalEncoder] = None,
        verifier: Optional[Any] = None
    ):
        self.capacity = capacity
        self.dim = dim
        self.retention_policy = retention_policy
        self.verification_mode = verification_mode
        self.entail_threshold = entail_threshold
        self.contra_threshold = contra_threshold
        self.consolidation_mode = consolidation_mode
        self.delta_max = delta_max
        self.lambda_anchor = lambda_anchor
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.k_fusion = k_fusion
        self.w_anchor = w_anchor
        self.decay_lambda = decay_lambda
        self.max_anchors = max_anchors

        # Core State
        self.memory = MemoryState(capacity=capacity, dim=dim)
        self.slot_texts: Dict[int, str] = {}
        self.slot_dense_embs: List[np.ndarray] = []
        self.slot_contra_counts: Dict[int, int] = {}
        self.slot_contra_anchors: Dict[int, List[Tuple[np.ndarray, float]]] = {}
        self.bm25_index = LocalBM25Index()

        # Models
        self.dense_encoder = dense_encoder or DenseSemanticEncoder()
        self.nli_gate = nli_gate or NliSemanticGate()
        self.verifier = verifier or CalibratedSemanticVerifier(self.nli_gate)
        self.lex_encoder = lex_encoder or LEXICAL_ENCODER

        # Operational telemetry
        self.total_absorbed = 0
        self.total_updates = 0
        self.total_inserts = 0
        self.total_temporal_updates = 0
        self.total_evictions = 0
        self.total_nli_calls = 0

        # Thread safety: RLock because internal calls (absorb -> _evict,
        # get_history -> recall) re-enter on the same thread.
        self._lock = threading.RLock()

    def _parse_candidate(self, obs: Union[Candidate, Tuple[str, str, str], str], timestamp: float) -> Candidate:
        if isinstance(obs, Candidate):
            return obs
        elif isinstance(obs, tuple) and len(obs) == 3:
            return Candidate(subject=obs[0], predicate=obs[1], object=obs[2], timestamp=timestamp)
        elif isinstance(obs, str):
            # Parse simple natural language "Subject Predicate Object."
            parts = obs.strip().rstrip(".").split(" ", 2)
            if len(parts) == 3:
                return Candidate(subject=parts[0], predicate=parts[1], object=parts[2], timestamp=timestamp)
            else:
                return Candidate(subject="entity", predicate="states", object=obs, timestamp=timestamp)
        else:
            raise TypeError(f"Unsupported observation format: {type(obs)}")

    def _is_same_attribute_or_event(
        self,
        slot: MemorySlot,
        candidate: Candidate,
        slot_idx: int,
        cand_dense: np.ndarray
    ) -> bool:
        """
        Determines whether an incoming candidate with an NLI CONTRADICTION signal
        concerns the exact same functional attribute or state as an existing memory slot
        (requiring temporal revision / supersession), or represents an independent
        coexisting fact or cumulative action that must NOT deactivate the existing slot.
        """
        cand_text = candidate.object if not candidate.predicate else f"{candidate.subject} {candidate.predicate} {candidate.object}."
        slot_text = self.slot_texts.get(id(slot), slot.text or "")
        slot_pred = (slot.debug_predicate or "").strip().lower()
        cand_pred = (candidate.predicate or "").strip().lower()

        # Cumulative / non-exclusive predicates where multiple distinct facts co-exist
        CUMULATIVE_PREDICATES = {
            "published", "authored", "wrote", "won", "received", "visited",
            "attended", "completed", "graduated", "studied", "speaks", "likes",
            "enjoys", "owns", "built", "designed", "created", "invented",
            "discovered", "invested", "invested in", "translated", "earned",
            "volunteers", "volunteered", "practices", "collects", "repairs"
        }

        TEMPORAL_TRANSITION_MARKERS = {
            "relocated", "promoted", "switched", "transitioned", "resigned",
            "became", "is now", "now", "no longer", "moved to", "moved from",
            "transferred", "replaced", "retired", "sold", "stepped down", "divested"
        }

        EXCLUSIVE_SYNONYM_GROUPS = [
            {"residence", "lives in", "lives", "resides in", "resides", "location", "located in", "headquarters", "city"},
            {"employed by", "works at", "employer", "company"},
            {"role", "title", "position", "current_role", "current_title"},
            {"status", "state", "operational_state"},
            {"marital_status", "spouse"},
            {"office suite", "office", "assigned to"}
        ]

        from benchmarks.phase26_verification import extract_negation_markers, extract_numbers
        neg_slot = extract_negation_markers(slot_text)
        neg_cand = extract_negation_markers(cand_text)

        import numpy as np
        dense_sim = 0.0
        if cand_dense is not None and slot_idx < len(self.slot_dense_embs):
            slot_dense = self.slot_dense_embs[slot_idx]
            if slot_dense is not None and len(slot_dense) == len(cand_dense):
                dense_sim = float(np.dot(slot_dense, cand_dense))

        # 1. Direct negation on matching proposition
        if bool(neg_slot) != bool(neg_cand):
            import re
            s_clean = re.sub(r"\b(does not|did not|is not|was not|not|never|no longer)\b", "", slot_pred).strip()
            c_clean = re.sub(r"\b(does not|did not|is not|was not|not|never|no longer)\b", "", cand_pred).strip()
            if s_clean == c_clean or not slot_pred or not cand_pred or dense_sim >= 0.55:
                return True

        # 2. Explicit state transition marker
        cand_text_l = cand_text.lower()
        if any(marker in cand_text_l for marker in TEMPORAL_TRANSITION_MARKERS):
            if dense_sim >= 0.65:
                return True

        # 3. Exclusive attribute groups.  Predicate identity alone is not
        # sufficient: NLI often marks independent facts sharing a verb
        # (e.g. "leads team A" vs "leads team B") as contradictions.
        if slot_pred and cand_pred:
            for grp in EXCLUSIVE_SYNONYM_GROUPS:
                in_slot = any(m in slot_pred for m in grp)
                in_cand = any(m in cand_pred for m in grp)
                if in_slot and in_cand:
                    return True

        # 4. Antonyms
        ANTONYMS = {
            ("active", "inactive"), ("online", "offline"), ("alive", "dead"),
            ("passed", "failed"), ("acquitted", "convicted"), ("approved", "denied"),
            ("elected", "rejected"), ("gold", "disqualified"), ("fluent", "cannot")
        }
        for a1, a2 in ANTONYMS:
            if (a1 in slot_text.lower() and a2 in cand_text.lower()) or \
               (a2 in slot_text.lower() and a1 in cand_text.lower()):
                return True

        # 5. Differing quantities on same predicate
        if slot_pred and cand_pred and slot_pred == cand_pred and slot_pred not in CUMULATIVE_PREDICATES:
            nums_s = extract_numbers(slot_text)
            nums_c = extract_numbers(cand_text)
            if nums_s and nums_c and nums_s != nums_c:
                return True

        return False

    def absorb(
        self,
        observation: Union[Candidate, Tuple[str, str, str], str],
        timestamp: Optional[float] = None,
        confidence: float = 1.0
    ) -> dict:
        """
        Absorbs a new observation into memory.
        
        Lifecycle:
          1. Encodes observation (lexical + dense).
          2. Generates candidates via expanded hybrid beam (Dense + BM25).
          3. Semantically verifies candidates using calibrated bidirectional NLI.
          4. If SAME: consolidates memory representation using drift-resistant update.
          5. If CONTRADICTION of an active fact: executes temporal state update
             (closes old validity window, adds contradiction repulsion anchor, creates new active slot).
          6. If NEUTRAL or new: inserts new slot into provisional working memory.
          7. Enforces capacity pressure via Tiered Retention (evicting uncorroborated singletons first).
        """
        if timestamp is not None:
            t_now = float(timestamp)
        elif isinstance(observation, Candidate):
            t_now = float(observation.timestamp)
        else:
            t_now = float(time.time())
        candidate = self._parse_candidate(observation, timestamp=t_now)
        cand_text = f"{candidate.subject} {candidate.predicate} {candidate.object}."
        x_t = self.lex_encoder.encode_full(candidate, dim=self.dim)
        cand_dense = self.dense_encoder.encode(cand_text)

        with self._lock:
            self.total_absorbed += 1
            target_idx: Optional[int] = None
            target_pred: Optional[dict] = None
            action = "insert"
            fused_candidates: List[int] = []
            is_temporal_change = False
            superseded_slot: Optional[MemorySlot] = None
            is_historical_predecessor = False
            successor_slot: Optional[MemorySlot] = None

            if self.memory.slots:
                id_to_slot_index = {id(slot): idx for idx, slot in enumerate(self.memory.slots)}

                # 1. Dense candidate retrieval with adaptive anchor repulsion
                dense_candidates, beam_policy, _, _ = adaptive_dense_address_phase20(
                    slot_embeddings=self.slot_dense_embs,
                    cand_embedding=cand_dense,
                    slot_contra_anchors=self.slot_contra_anchors,
                    slots=self.memory.slots,
                    current_time=t_now,
                    tau_min=TAU_MIN,
                    tau_high=TAU_HIGH,
                    delta_margin=DELTA_MARGIN,
                    k_max=self.top_k_dense,
                    use_repulsion=True,
                    w_anchor=self.w_anchor,
                    decay_lambda=self.decay_lambda,
                    anchor_sim_threshold=TAU_ANCHOR
                )

                # 2. Lexical BM25 retrieval
                lexical_candidates = self.bm25_index.query(
                    query_text=cand_text,
                    top_k=self.top_k_bm25,
                    id_to_slot_index=id_to_slot_index
                )

                # 3. Candidate Fusion
                fused_candidates, _ = fuse_candidates(
                    dense_candidates=dense_candidates,
                    lexical_candidates=lexical_candidates,
                    k_fusion=self.k_fusion
                )

                # 4. Calibrated Semantic Verification
                if fused_candidates:
                    pairs = [(self.slot_texts[id(self.memory.slots[idx])], cand_text) for idx in fused_candidates]
                    preds = self.verifier.verify_batch(
                        pairs,
                        mode=self.verification_mode,
                        entail_threshold=self.entail_threshold,
                        contra_threshold=self.contra_threshold
                    )
                    self.total_nli_calls += sum(p.get("nli_invocations", 1) for p in preds)

                    for cand_idx, pred in zip(fused_candidates, preds):
                        dec = pred["decision"]
                        slot = self.memory.slots[cand_idx]
                        slot_id = id(slot)

                        if dec == "SAME":
                            # Entity grounding guard: do not merge facts across different distinct entities
                            slot_subj = (slot.debug_subject or "").strip().lower()
                            cand_subj = (candidate.subject or "").strip().lower()
                            generic_subjs = {"", "entity", "someone", "something", "unknown", "item"}
                            if slot_subj not in generic_subjs and cand_subj not in generic_subjs and slot_subj != cand_subj:
                                continue
                            target_idx = cand_idx
                            target_pred = pred
                            break

                        elif dec == "CONTRADICTION":
                            # Conservative contradiction handling: check if this is a temporal update
                            # for the same subject/predicate
                            fwd = pred.get("fwd") if isinstance(pred.get("fwd"), dict) else {}
                            p_c_fwd = float(fwd.get("p_contra", pred.get("p_contra", 0.0)) or 0.0)
                            p_n_fwd = float(fwd.get("p_neutral", pred.get("p_neutral", 0.0)) or 0.0)
                            conf = float(pred.get("confidence", 1.0) or 0.0)

                            # An asymmetric unilateral artifact where forward direction is overwhelmingly
                            # neutral (and not contradictory) does not represent a true conflict.
                            is_asym_artifact = (p_c_fwd < 0.20 and p_n_fwd > 0.50)
                            if conf < 0.40:
                                is_asym_artifact = True

                            same_subject = (slot.debug_subject == candidate.subject)
                            is_same_attr = self._is_same_attribute_or_event(slot, candidate, cand_idx, cand_dense)

                            if same_subject and is_same_attr and not is_asym_artifact:
                                # Form contradiction anchor ONLY for genuine contradictions on the same entity & attribute
                                self.slot_contra_counts[slot_id] = self.slot_contra_counts.get(slot_id, 0) + 1
                                anchors = self.slot_contra_anchors.setdefault(slot_id, [])
                                if len(anchors) >= self.max_anchors:
                                    anchors.pop(0)
                                anchors.append((cand_dense, t_now))

                                if slot.is_active:
                                    if t_now >= slot.valid_from:
                                        # Forward temporal revision: record intent but do NOT
                                        # close the old validity window yet.  The actual
                                        # slot.valid_until = t_now is deferred to the Case 2
                                        # commit block below so that a failure in memory.add()
                                        # cannot leave the old slot deactivated without a
                                        # replacement (atomic commit).
                                        superseded_slot = slot
                                        is_temporal_change = True
                                    else:
                                        # Backward historical report: this candidate is a predecessor of slot
                                        is_historical_predecessor = True
                                        successor_slot = slot

                            # Do not merge into this slot
                            continue

                best_sim = dense_candidates[0][1] if dense_candidates else 0.0
                n_t = novelty(best_sim, memory_empty=False)
            else:
                best_sim = 0.0
                n_t = 1.0

            alpha_t = n_t

            # ── Case 1: Consolidate Semantic Paraphrase into Existing Slot ──
            if target_idx is not None:
                slot = self.memory.slots[target_idx]
                slot.value = consolidate_vector_phase24(
                    slot=slot,
                    x_t=x_t,
                    alpha_t=alpha_t,
                    mode=self.consolidation_mode,
                    delta_max=self.delta_max,
                    lambda_anchor=self.lambda_anchor
                )
                slot.confidence = min(1.0, max(slot.confidence, candidate.confidence) + 0.05)
                slot.timestamp = t_now
                slot.update_count += 1
                # Update BM25 and text representation with the latest phrasing
                self.slot_texts[id(slot)] = cand_text
                self.slot_dense_embs[target_idx] = cand_dense
                self.bm25_index.update_document(id(slot), cand_text)

                self.total_updates += 1
                return {
                    "action": "consolidate",
                    "slot_index": target_idx,
                    "slot_id": id(slot),
                    "similarity": best_sim,
                    "decision": "SAME",
                    "update_count": slot.update_count,
                    "valid_from": slot.valid_from,
                    "valid_until": slot.valid_until,
                    "text": cand_text
                }

            # ── Case 2: Temporal State Revision (Contradiction of Active Fact) ──
            if is_temporal_change and superseded_slot is not None:
                superseded_id = id(superseded_slot)
                # Allocate a new slot for the new active state
                if self.memory.is_full():
                    evict_idx, evicted = self._evict(t_now)
                new_slot = self.memory.add(
                    key=x_t, value=x_t.copy(), confidence=candidate.confidence,
                    timestamp=t_now, debug_subject=candidate.subject,
                    debug_predicate=candidate.predicate, text=cand_text,
                    dense_emb=cand_dense, valid_from=t_now, valid_until=None,
                    supersedes=superseded_id
                )
                # ATOMIC COMMIT: only deactivate the old slot AFTER the new
                # slot has been successfully added.  If memory.add() raised
                # an exception, the old slot remains fully active.
                superseded_slot.valid_until = t_now
                superseded_slot.superseded_by = id(new_slot)
                self.slot_texts[id(new_slot)] = cand_text
                self.slot_dense_embs.append(cand_dense)
                self.bm25_index.add_document(id(new_slot), cand_text)

                self.total_temporal_updates += 1
                return {
                    "action": "temporal_update",
                    "slot_index": len(self.memory.slots) - 1,
                    "slot_id": id(new_slot),
                    "superseded_slot_id": superseded_id,
                    "valid_from": t_now,
                    "valid_until": None,
                    "text": cand_text
                }

            # ── Case 2b: Backward Historical Insertion (Predecessor of Active Fact) ──
            if is_historical_predecessor and successor_slot is not None:
                successor_id = id(successor_slot)
                if self.memory.is_full():
                    evict_idx, evicted = self._evict(t_now)
                new_slot = self.memory.add(
                    key=x_t, value=x_t.copy(), confidence=candidate.confidence,
                    timestamp=t_now, debug_subject=candidate.subject,
                    debug_predicate=candidate.predicate, text=cand_text,
                    dense_emb=cand_dense, valid_from=t_now, valid_until=successor_slot.valid_from,
                    supersedes=None
                )
                new_slot.superseded_by = successor_id
                successor_slot.supersedes = id(new_slot)
                self.slot_texts[id(new_slot)] = cand_text
                self.slot_dense_embs.append(cand_dense)
                self.bm25_index.add_document(id(new_slot), cand_text)

                self.total_temporal_updates += 1
                return {
                    "action": "historical_update",
                    "slot_index": len(self.memory.slots) - 1,
                    "slot_id": id(new_slot),
                    "superseded_by_slot_id": successor_id,
                    "valid_from": t_now,
                    "valid_until": successor_slot.valid_from,
                    "text": cand_text
                }

            # ── Case 3: Allocate New Slot (Insert or Evict-Insert) ──
            if self.memory.is_full():
                evict_idx, evicted = self._evict(t_now)
                action = "evict_insert"
            else:
                action = "insert"

            slot = self.memory.add(
                key=x_t, value=x_t.copy(), confidence=candidate.confidence,
                timestamp=t_now, debug_subject=candidate.subject,
                debug_predicate=candidate.predicate, text=cand_text,
                dense_emb=cand_dense, valid_from=t_now, valid_until=None
            )
            self.slot_texts[id(slot)] = cand_text
            self.slot_dense_embs.append(cand_dense)
            self.bm25_index.add_document(id(slot), cand_text)

            self.total_inserts += 1
            return {
                "action": action,
                "slot_index": len(self.memory.slots) - 1,
                "slot_id": id(slot),
                "similarity": best_sim,
                "update_count": 1,
                "valid_from": t_now,
                "valid_until": None,
                "text": cand_text
            }

    def _evict(self, current_time: float) -> Tuple[int, MemorySlot]:
        """Evicts a slot according to the configured retention policy."""
        evict_idx, evicted_slot = evict_slot_phase24(
            memory=self.memory,
            eviction_policy=self.retention_policy,
            current_time=current_time,
            slot_texts=self.slot_texts,
            slot_dense_embs=self.slot_dense_embs,
            slot_contra_counts=self.slot_contra_counts,
            slot_contra_anchors=self.slot_contra_anchors
        )
        self.bm25_index.remove_document(id(evicted_slot))
        self.total_evictions += 1
        return evict_idx, evicted_slot

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        as_of: Optional[float] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant memory records for a natural-language query.
        Supports point-in-time temporal queries via as_of.
        """
        if not self.memory.slots:
            return []

        q_dense = self.dense_encoder.encode(query_text)

        with self._lock:
            candidate_slots = []

            for idx, slot in enumerate(self.memory.slots):
                # Temporal filter
                if as_of is not None:
                    if slot.valid_from > as_of or (slot.valid_until is not None and slot.valid_until <= as_of):
                        continue
                elif active_only and not slot.is_active:
                    continue

                emb = self.slot_dense_embs[idx] if idx < len(self.slot_dense_embs) else slot.dense_emb
                sim = float(np.dot(emb, q_dense)) if emb is not None else 0.0
                candidate_slots.append((sim, slot, idx))

            candidate_slots.sort(key=lambda x: -x[0])
            results = []
            for sim, slot, idx in candidate_slots[:top_k]:
                results.append({
                    "slot_index": idx,
                    "slot_id": id(slot),
                    "similarity": round(sim, 4),
                    "subject": slot.debug_subject,
                    "predicate": slot.debug_predicate,
                    "text": self.slot_texts.get(id(slot), slot.text),
                    "confidence": round(float(slot.confidence), 4),
                    "update_count": slot.update_count,
                    "is_active": slot.is_active,
                    "valid_from": slot.valid_from,
                    "valid_until": slot.valid_until
                })
            return results

    def recall(
        self,
        subject: str,
        predicate: Optional[str] = None,
        as_of: Optional[float] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Exact semantic lookup for an entity/attribute.
        """
        with self._lock:
            results = []
            for idx, slot in enumerate(self.memory.slots):
                if slot.debug_subject != subject:
                    continue
                if predicate is not None and slot.debug_predicate != predicate:
                    continue

                # Temporal filtering
                if as_of is not None:
                    if slot.valid_from > as_of or (slot.valid_until is not None and slot.valid_until <= as_of):
                        continue
                elif active_only and not slot.is_active:
                    continue

                results.append({
                    "slot_index": idx,
                    "slot_id": id(slot),
                    "subject": slot.debug_subject,
                    "predicate": slot.debug_predicate,
                    "text": self.slot_texts.get(id(slot), slot.text),
                    "confidence": round(float(slot.confidence), 4),
                    "update_count": slot.update_count,
                    "is_active": slot.is_active,
                    "valid_from": slot.valid_from,
                    "valid_until": slot.valid_until
                })
            results.sort(key=lambda r: r["valid_from"])
            return results

    def delete_slot(self, slot_id: int) -> bool:
        """
        Explicitly removes a specific memory slot by its identifier (id(slot)),
        synchronously purging it from active memory, auxiliary BM25 index,
        dense vectors, contradiction counts, and repulsion anchors.
        
        Maintains valid temporal/supersession chains among remaining slots.
        Returns True if a slot with the given ID was found and removed, False otherwise.
        """
        with self._lock:
            target_idx = None
            target_slot = None
            for idx, slot in enumerate(self.memory.slots):
                if id(slot) == slot_id:
                    target_idx = idx
                    target_slot = slot
                    break

            if target_idx is None or target_slot is None:
                return False

            # Mend supersession chain around the deleted slot
            pred_id = target_slot.supersedes
            succ_id = target_slot.superseded_by

            for s in self.memory.slots:
                if id(s) == pred_id:
                    s.superseded_by = succ_id
                if id(s) == succ_id:
                    s.supersedes = pred_id

            # Remove from MemoryState
            self.memory.slots.pop(target_idx)

            # Clean up auxiliary stores
            if slot_id in self.slot_texts:
                del self.slot_texts[slot_id]

            if target_idx < len(self.slot_dense_embs):
                self.slot_dense_embs.pop(target_idx)

            if slot_id in self.slot_contra_counts:
                del self.slot_contra_counts[slot_id]

            if slot_id in self.slot_contra_anchors:
                del self.slot_contra_anchors[slot_id]

            # Remove from BM25 index
            self.bm25_index.remove_document(slot_id)

            return True

    def purge_subject(self, subject: str, exact_match: bool = True) -> int:
        """
        Purges all memory slots associated with a given subject (entity),
        including active and superseded historical states, auxiliary indexes,
        dense vectors, and contradiction anchors.

        Parameters:
          subject: The entity identifier string to purge.
          exact_match: If True, matches exact case-sensitive string; if False,
                       matches case-insensitively (stripped).

        Returns:
          Number of memory slots purged.
        """
        with self._lock:
            if not isinstance(subject, str) or not subject.strip():
                return 0

            match_target = subject.strip() if exact_match else subject.strip().lower()

            # Find matching slot IDs
            matching_ids = []
            for slot in self.memory.slots:
                s_subj = slot.debug_subject or ""
                s_eval = s_subj.strip() if exact_match else s_subj.strip().lower()
                if s_eval == match_target:
                    matching_ids.append(id(slot))

            count = 0
            for s_id in matching_ids:
                if self.delete_slot(s_id):
                    count += 1

            return count

    def get_history(self, subject: str, predicate: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the complete temporal history of an entity/attribute,
        including past superseded states and active states.
        """
        return self.recall(subject=subject, predicate=predicate, active_only=False)

    def status(self) -> Dict[str, Any]:
        """Returns diagnostic status and memory footprint."""
        with self._lock:
            active = sum(1 for s in self.memory.slots if s.is_active)
            provisional = sum(1 for s in self.memory.slots if s.update_count <= 1 and s.is_active)
            consolidated = sum(1 for s in self.memory.slots if s.update_count > 1 and s.is_active)
            superseded = sum(1 for s in self.memory.slots if not s.is_active)
            footprint = self.memory.footprint_bytes()

            return {
                "capacity": self.capacity,
                "total_slots": len(self.memory.slots),
                "active_slots": active,
                "provisional_slots": provisional,
                "consolidated_slots": consolidated,
                "superseded_slots": superseded,
                "total_absorbed": self.total_absorbed,
                "total_updates": self.total_updates,
                "total_inserts": self.total_inserts,
                "total_temporal_updates": self.total_temporal_updates,
                "total_evictions": self.total_evictions,
                "total_nli_calls": self.total_nli_calls,
                "footprint_bytes": footprint["total_bytes"],
            }

    def save_checkpoint(
        self,
        path: str,
        create_backup: bool = True,
        allowed_base_dir: Optional[str] = None
    ) -> str:
        """Persists the memory engine state to a crash-safe .npz archive."""
        from phase2.checkpoint import save_checkpoint
        return save_checkpoint(self, path, create_backup=create_backup, allowed_base_dir=allowed_base_dir)

    def load_checkpoint(
        self,
        path: str,
        allow_backup_recovery: bool = True,
        allowed_base_dir: Optional[str] = None
    ) -> "BlackHoleMemory":
        """Restores memory state and auxiliary indexes from a .npz archive."""
        from phase2.checkpoint import load_into_black_hole_memory
        return load_into_black_hole_memory(
            self,
            path,
            allow_backup_recovery=allow_backup_recovery,
            allowed_base_dir=allowed_base_dir
        )

    @classmethod
    def from_checkpoint(cls, path: str, allowed_base_dir: Optional[str] = None, **kwargs) -> "BlackHoleMemory":
        """Constructs a new BlackHoleMemory instance initialized from a checkpoint."""
        instance = cls(**kwargs)
        instance.load_checkpoint(path, allowed_base_dir=allowed_base_dir)
        return instance
