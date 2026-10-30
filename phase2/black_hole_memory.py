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

ANTONYMS = {
    ("active", "inactive"), ("online", "offline"), ("alive", "dead"),
    ("passed", "failed"), ("acquitted", "convicted"), ("approved", "denied"),
    ("elected", "rejected"), ("gold", "disqualified"), ("fluent", "cannot")
}


def _contains_marker(text: str, marker: str) -> bool:
    """Whole-token match: 'city' must not match 'capacity', 'now' not 'known'."""
    return re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", text) is not None


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
        self.total_value_verifications = 0

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

    @staticmethod
    def _normalize_subject(subject: Optional[str]) -> str:
        return (subject or "").strip().casefold()

    @staticmethod
    def _share_exclusive_group(pred_a: str, pred_b: str) -> bool:
        if not pred_a or not pred_b:
            return False
        for grp in EXCLUSIVE_SYNONYM_GROUPS:
            in_a = any(_contains_marker(pred_a, m) for m in grp)
            in_b = any(_contains_marker(pred_b, m) for m in grp)
            if in_a and in_b:
                return True
        return False

    @staticmethod
    def _exclusive_group_ids(predicate: str) -> set:
        pred = (predicate or "").strip().lower()
        if not pred:
            return set()
        return {
            idx for idx, group in enumerate(EXCLUSIVE_SYNONYM_GROUPS)
            if any(_contains_marker(pred, marker) for marker in group)
        }

    @staticmethod
    def _slot_predicates(slot: MemorySlot) -> set:
        return {
            p.strip().lower()
            for p in (slot.debug_predicate, slot.latest_predicate)
            if p and p.strip()
        }

    @staticmethod
    def _normalize_value(value: Optional[str]) -> str:
        if value is None:
            return ""
        collapsed = re.sub(r"\s+", " ", value.strip().casefold())
        return collapsed.strip(" \t\r\n.,;:!?\"'")

    def _slot_value_text(self, slot: MemorySlot) -> Optional[str]:
        """Best-known object text for a slot, surviving paraphrases."""
        if slot.latest_object:
            return slot.latest_object
        text = self.slot_texts.get(id(slot), slot.text or "")
        if not text:
            return None
        return self._slot_object_text(slot, text) or None

    def _values_are_equivalent(self, value_a: str, value_b: str) -> bool:
        """Whether two attribute values are paraphrases rather than a change.

        A whole-statement SAME can hide a genuine value change, so the values
        are compared on their own.  Identical or numerically changed values
        need no model call; otherwise the configured verifier decides.
        """
        if self._normalize_value(value_a) == self._normalize_value(value_b):
            return True
        from benchmarks.phase26_verification import extract_numbers
        numbers_a = extract_numbers(value_a)
        numbers_b = extract_numbers(value_b)
        if numbers_a and numbers_b and numbers_a != numbers_b:
            return False
        self.total_value_verifications += 1
        result = self.verifier.verify_batch(
            [(value_a, value_b)],
            mode=self.verification_mode,
            entail_threshold=self.entail_threshold,
            contra_threshold=self.contra_threshold,
        )
        decision = result[0].get("decision") if result else "NEUTRAL"
        return decision == "SAME"

    def _is_duplicate_active_fact(self, a: MemorySlot, b: MemorySlot) -> bool:
        """Two active slots asserting the same value of one exclusive attribute."""
        if not (a.is_active and b.is_active):
            return False
        if self._normalize_subject(a.debug_subject) != self._normalize_subject(b.debug_subject):
            return False
        same_attribute = any(
            (pa == pb and pa not in CUMULATIVE_PREDICATES)
            or self._share_exclusive_group(pa, pb)
            for pa in self._slot_predicates(a)
            for pb in self._slot_predicates(b)
        )
        if not same_attribute:
            return False
        value_a = self._slot_value_text(a)
        value_b = self._slot_value_text(b)
        if not value_a or not value_b:
            return False
        return self._normalize_value(value_a) == self._normalize_value(value_b)

    def _attribute_chain(self, candidate: Candidate) -> List[MemorySlot]:
        """Slots describing the same subject and attribute as the candidate."""
        cand_subject = self._normalize_subject(candidate.subject)
        cand_pred = (candidate.predicate or "").strip().lower()
        chain = []
        for slot in self.memory.slots:
            if self._normalize_subject(slot.debug_subject) != cand_subject:
                continue
            if any(
                slot_pred == cand_pred or self._share_exclusive_group(slot_pred, cand_pred)
                for slot_pred in self._slot_predicates(slot)
            ):
                chain.append(slot)
        return chain

    @staticmethod
    def _slot_object_text(slot: MemorySlot, slot_text: str) -> str:
        """Extracts the object from free-form text using the live predicate."""
        subject = slot.debug_subject or ""
        text = (slot_text or "").strip().rstrip(".")
        predicates = [slot.latest_predicate, slot.debug_predicate]
        for predicate in dict.fromkeys(p for p in predicates if p):
            prefix = f"{subject} {predicate} "
            if subject and text.lower().startswith(prefix.lower()):
                return text[len(prefix):]
        return ""

    @staticmethod
    def _content_tokens(text: str) -> set:
        from benchmarks.phase26_verification import extract_negation_markers
        negations = extract_negation_markers(text)
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in negations}

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

        from benchmarks.phase26_verification import extract_negation_markers, extract_numbers
        neg_slot = extract_negation_markers(slot_text)
        neg_cand = extract_negation_markers(cand_text)

        import numpy as np
        dense_sim = 0.0
        if cand_dense is not None and slot_idx < len(self.slot_dense_embs):
            slot_dense = self.slot_dense_embs[slot_idx]
            if slot_dense is not None and len(slot_dense) == len(cand_dense):
                dense_sim = float(np.dot(slot_dense, cand_dense))

        # 1. Direct negation on matching proposition.  The object must match
        # too: "did not approve north" vs "approved south" are independent
        # facts even though the sentences are lexically close.
        if bool(neg_slot) != bool(neg_cand):
            s_clean = re.sub(r"\b(does not|did not|is not|was not|not|never|no longer)\b", "", slot_pred).strip()
            c_clean = re.sub(r"\b(does not|did not|is not|was not|not|never|no longer)\b", "", cand_pred).strip()
            slot_object = slot.latest_object
            if slot_object is None:
                slot_object = self._slot_object_text(slot, slot_text)
            if slot_object and candidate.object:
                slot_obj_tokens = self._content_tokens(slot_object)
                cand_obj_tokens = self._content_tokens(candidate.object)
                objects_match = (
                    bool(cand_obj_tokens) and cand_obj_tokens <= slot_obj_tokens
                )
            else:
                # After paraphrase consolidation the slot text no longer
                # matches its debug predicate, so the object cannot be cut
                # out by prefix.  Fall back to containment in the live text.
                cand_obj_tokens = self._content_tokens(candidate.object or "")
                objects_match = (
                    bool(cand_obj_tokens)
                    and cand_obj_tokens <= self._content_tokens(slot_text)
                )
            if objects_match and (s_clean == c_clean or not slot_pred or not cand_pred or dense_sim >= 0.55):
                return True

        # 2. Explicit state transition marker
        cand_text_l = cand_text.lower()
        if any(_contains_marker(cand_text_l, marker) for marker in TEMPORAL_TRANSITION_MARKERS):
            if dense_sim >= 0.65:
                return True

        # 3. Exclusive attribute groups.  Predicate identity alone is not
        # sufficient: NLI often marks independent facts sharing a verb
        # (e.g. "leads team A" vs "leads team B") as contradictions.
        if any(self._share_exclusive_group(sp, cand_pred) for sp in self._slot_predicates(slot)):
            return True

        # 4. Antonyms
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
            closed_history_same: Optional[MemorySlot] = None
            closed_history_same_idx: Optional[int] = None
            insert_predecessor: Optional[MemorySlot] = None

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
                            slot_subj = self._normalize_subject(slot.debug_subject)
                            cand_subj = self._normalize_subject(candidate.subject)
                            generic_subjs = {"", "entity", "someone", "something", "unknown", "item"}
                            if slot_subj not in generic_subjs and cand_subj not in generic_subjs and slot_subj != cand_subj:
                                continue
                            if (not slot.is_active and slot.valid_until is not None
                                    and t_now >= slot.valid_until):
                                # The fact is reasserted after its validity window
                                # closed.  Merging into the historical slot would
                                # swallow the recurrence; let an active conflicting
                                # state be evaluated (and superseded) instead.
                                closed_history_same = slot
                                closed_history_same_idx = cand_idx
                                continue
                            cand_pred = (candidate.predicate or "").strip().lower()
                            same_attribute = any(
                                pred == cand_pred
                                or self._share_exclusive_group(pred, cand_pred)
                                for pred in self._slot_predicates(slot)
                            )
                            if same_attribute:
                                slot_value = self._slot_value_text(slot)
                                values_differ = (
                                    bool(slot_value) and bool(candidate.object)
                                    and not self._values_are_equivalent(
                                        slot_value, candidate.object
                                    )
                                )
                                if values_differ:
                                    # SAME over a different value of one attribute
                                    # is not a paraphrase: consolidating would
                                    # erase the prior state without recording an
                                    # interval.  Attributes known to be exclusive
                                    # become temporal updates; everything else is
                                    # kept as an independent active observation.
                                    if self._is_same_attribute_or_event(
                                            slot, candidate, cand_idx, cand_dense
                                        ):
                                        if slot.is_active:
                                            if t_now >= slot.valid_from:
                                                superseded_slot = slot
                                                is_temporal_change = True
                                            else:
                                                is_historical_predecessor = True
                                                successor_slot = slot
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

                            same_subject = (
                                self._normalize_subject(slot.debug_subject)
                                == self._normalize_subject(candidate.subject)
                            )
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

                    # A SAME verdict that landed on a closed historical slot is
                    # a recurrence.  If retrieval did not surface a conflicting
                    # active head, resolve it directly; otherwise the recurrence
                    # would be inserted as a second active version of the same
                    # attribute with no interval and no link.
                    if (target_idx is None and not is_temporal_change
                            and not is_historical_predecessor
                            and closed_history_same is not None
                            and closed_history_same_idx is not None):
                        chain = self._attribute_chain(candidate)
                        active_chain = [s for s in chain if s.is_active]
                        if self._is_same_attribute_or_event(
                                closed_history_same, candidate,
                                closed_history_same_idx, cand_dense
                            ):
                            if active_chain:
                                head = max(
                                    active_chain,
                                    key=lambda s: (s.valid_from, s.created_at)
                                )
                                head_idx = self._slot_index(head)
                                head_value = self._slot_value_text(head)
                                if (head_idx is not None and head_value
                                        and candidate.object
                                        and self._normalize_value(head_value)
                                        == self._normalize_value(candidate.object)):
                                    target_idx = head_idx
                                elif t_now >= head.valid_from:
                                    superseded_slot = head
                                    is_temporal_change = True
                                else:
                                    is_historical_predecessor = True
                                    successor_slot = head
                            else:
                                # Every version is closed: the recurrence is a
                                # new version after a gap.  Link it to the
                                # nearest prior version so the chain stays
                                # traversable.
                                closed = [
                                    s for s in chain
                                    if s.valid_until is not None
                                    and s.valid_from < t_now
                                    and s.valid_until <= t_now
                                ]
                                if closed:
                                    insert_predecessor = max(
                                        closed,
                                        key=lambda s: (s.valid_from, s.created_at)
                                    )

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
                # Update every representation of the phrasing, including the
                # slot fields that the checkpoint serializes.  Otherwise a
                # reload would silently restore the pre-consolidation wording.
                slot.text = cand_text
                slot.dense_emb = cand_dense.copy()
                slot.latest_predicate = candidate.predicate
                slot.latest_object = candidate.object
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

                # Equal-timestamp policy: a contradiction arriving exactly at
                # the active interval's valid_from cannot close an earlier
                # interval and must not create an overlapping one.  The last
                # writer at that instant replaces the state in place.
                if t_now == superseded_slot.valid_from:
                    return self._replace_state_in_place(
                        superseded_slot, candidate, cand_text, cand_dense, x_t, t_now,
                        action="temporal_update"
                    )

                chain_before = self._attribute_chain(candidate)
                evicted = None
                if self.memory.is_full():
                    evict_idx, evicted = self._evict(
                        t_now, protected_slot_ids={superseded_id}
                    )

                # With capacity 1 the active slot may have been evicted to
                # make room.  Install the new active state regardless, but
                # never leave a link pointing at the removed object.
                keep_predecessor = self._slot_index(superseded_slot) is not None
                history_truncated = evicted is not None and any(
                    evicted is slot for slot in chain_before
                )

                new_slot = self.memory.add(
                    key=x_t, value=x_t.copy(), confidence=candidate.confidence,
                    timestamp=t_now, debug_subject=candidate.subject,
                    debug_predicate=candidate.predicate, text=cand_text,
                    dense_emb=cand_dense, valid_from=t_now, valid_until=None,
                    supersedes=superseded_id if keep_predecessor else None,
                    latest_predicate=candidate.predicate,
                    latest_object=candidate.object
                )
                # ATOMIC COMMIT: only deactivate the old slot AFTER the new
                # slot has been successfully added.  If memory.add() raised
                # an exception, the old slot remains fully active.
                if keep_predecessor:
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
                    "superseded_slot_id": superseded_id if keep_predecessor else None,
                    "evicted_slot_id": id(evicted) if evicted is not None else None,
                    "history_truncated": history_truncated,
                    "valid_from": t_now,
                    "valid_until": None,
                    "text": cand_text
                }

            # ── Case 2b: Backward Historical Insertion (Predecessor of Active Fact) ──
            if is_historical_predecessor and successor_slot is not None:
                chain = self._attribute_chain(candidate)

                # Equal-timestamp policy (historical): if an interval already
                # starts exactly at t_now, replace that state in place.  The
                # interval and its links stay intact and nothing overlaps.
                equal = [s for s in chain if s.valid_from == t_now]
                if equal:
                    target = max(
                        equal,
                        key=lambda s: (s.created_at, self._slot_index(s) or 0)
                    )
                    for duplicate in equal:
                        if duplicate is not target:
                            self._remove_slot_identity(duplicate)
                    return self._replace_state_in_place(
                        target, candidate, cand_text, cand_dense, x_t, t_now,
                        action="historical_update"
                    )

                later = [s for s in chain if s.valid_from > t_now]
                if later:
                    nearest_successor = min(later, key=lambda s: s.valid_from)
                elif successor_slot.valid_from > t_now:
                    nearest_successor = successor_slot
                else:
                    # Every known interval starts before t_now, so this
                    # observation extends the newest state into the present.
                    nearest_successor = None
                earlier = [s for s in chain if s.valid_from < t_now]
                predecessor = max(earlier, key=lambda s: s.valid_from) if earlier else None
                if predecessor is not None and predecessor.is_active:
                    predecessor = None

                evicted = None
                if self.memory.is_full():
                    # Hard-protect the live successor: losing it would leave
                    # the attribute without its expected active state.  The
                    # nearest interval successor and predecessor are soft
                    # protections; if capacity cannot honor them the insertion
                    # still preserves the active state and reports truncation.
                    hard = {id(successor_slot)} if successor_slot.is_active else set()
                    soft = {
                        id(s) for s in (nearest_successor, predecessor) if s is not None
                    } - hard
                    protected = hard | soft
                    if not self._has_eviction_candidate(protected):
                        protected = hard
                    if not self._has_eviction_candidate(protected):
                        return {
                            "action": "historical_rejected",
                            "reason": "capacity_would_remove_active_state",
                            "slot_id": None,
                            "valid_from": t_now,
                            "valid_until": successor_slot.valid_from,
                            "text": cand_text,
                            "history_truncated": True
                        }
                    evict_idx, evicted = self._evict(
                        t_now, protected_slot_ids=protected
                    )

                # Recompute the chain after eviction: a soft-protected
                # predecessor/successor may no longer exist.
                chain_after = self._attribute_chain(candidate)
                later = [s for s in chain_after if s.valid_from > t_now]
                if later:
                    successor = min(later, key=lambda s: s.valid_from)
                elif (successor_slot.valid_from > t_now
                      and self._slot_index(successor_slot) is not None):
                    successor = successor_slot
                else:
                    successor = None
                earlier = [s for s in chain_after if s.valid_from < t_now]
                predecessor = max(earlier, key=lambda s: s.valid_from) if earlier else None
                if predecessor is not None and predecessor.is_active:
                    predecessor = None

                history_truncated = evicted is not None and any(
                    evicted is slot for slot in chain
                )
                successor_id = id(successor) if successor is not None else None
                valid_until = successor.valid_from if successor is not None else None
                new_slot = self.memory.add(
                    key=x_t, value=x_t.copy(), confidence=candidate.confidence,
                    timestamp=t_now, debug_subject=candidate.subject,
                    debug_predicate=candidate.predicate, text=cand_text,
                    dense_emb=cand_dense, valid_from=t_now, valid_until=valid_until,
                    supersedes=id(predecessor) if predecessor is not None else None,
                    latest_predicate=candidate.predicate,
                    latest_object=candidate.object
                )
                if predecessor is not None:
                    if predecessor.valid_until is not None and predecessor.valid_until > t_now:
                        predecessor.valid_until = t_now
                    predecessor.superseded_by = id(new_slot)
                if successor is not None:
                    new_slot.superseded_by = successor_id
                    successor.supersedes = id(new_slot)
                self.slot_texts[id(new_slot)] = cand_text
                self.slot_dense_embs.append(cand_dense)
                self.bm25_index.add_document(id(new_slot), cand_text)

                self.total_temporal_updates += 1
                return {
                    "action": "historical_update",
                    "slot_index": len(self.memory.slots) - 1,
                    "slot_id": id(new_slot),
                    "superseded_by_slot_id": successor_id,
                    "evicted_slot_id": id(evicted) if evicted is not None else None,
                    "history_truncated": history_truncated,
                    "valid_from": t_now,
                    "valid_until": valid_until,
                    "text": cand_text
                }

            # ── Case 3: Allocate New Slot (Insert or Evict-Insert) ──
            if self.memory.is_full():
                evict_idx, evicted = self._evict(t_now)
                action = "evict_insert"
            else:
                action = "insert"

            link_predecessor = (
                insert_predecessor is not None
                and self._slot_index(insert_predecessor) is not None
            )
            slot = self.memory.add(
                key=x_t, value=x_t.copy(), confidence=candidate.confidence,
                timestamp=t_now, debug_subject=candidate.subject,
                debug_predicate=candidate.predicate, text=cand_text,
                dense_emb=cand_dense, valid_from=t_now, valid_until=None,
                supersedes=id(insert_predecessor) if link_predecessor else None,
                latest_predicate=candidate.predicate,
                latest_object=candidate.object
            )
            if link_predecessor:
                insert_predecessor.superseded_by = id(slot)
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

    def _slot_index(self, slot: Optional[MemorySlot]) -> Optional[int]:
        if slot is None:
            return None
        for idx, candidate in enumerate(self.memory.slots):
            if candidate is slot:
                return idx
        return None

    def _has_eviction_candidate(self, protected_slot_ids) -> bool:
        protected = set(protected_slot_ids or ())
        return any(id(s) not in protected for s in self.memory.slots)

    def _replace_state_in_place(
        self,
        slot: MemorySlot,
        candidate: Candidate,
        cand_text: str,
        cand_dense: np.ndarray,
        x_t: np.ndarray,
        t_now: float,
        action: str
    ) -> dict:
        """Deterministic equal-timestamp policy: last writer wins.

        A contradiction arriving exactly at an existing interval's
        ``valid_from`` cannot close an earlier interval and must not create an
        overlapping one.  The interval (``[valid_from, valid_until)``) and the
        supersession links are preserved; only the represented state changes.
        Timestamps are never shifted.
        """
        idx = self._slot_index(slot)
        if idx is None:
            raise RuntimeError("Attempted in-place replacement of a slot that is not in memory")

        slot.value = x_t.copy()
        slot.confidence = min(1.0, max(slot.confidence, candidate.confidence) + 0.05)
        slot.timestamp = t_now
        slot.update_count += 1
        slot.text = cand_text
        slot.dense_emb = cand_dense.copy()
        slot.latest_predicate = candidate.predicate
        slot.latest_object = candidate.object

        self.slot_texts[id(slot)] = cand_text
        if idx < len(self.slot_dense_embs):
            self.slot_dense_embs[idx] = cand_dense
        self.bm25_index.update_document(id(slot), cand_text)
        self.total_temporal_updates += 1

        return {
            "action": action,
            "slot_index": idx,
            "slot_id": id(slot),
            "replaced_in_place": True,
            "valid_from": slot.valid_from,
            "valid_until": slot.valid_until,
            "text": cand_text,
            "update_count": slot.update_count,
        }

    def _remove_slot_identity(self, target_slot: MemorySlot) -> bool:
        """Removes a specific slot object and repairs its chain neighbours."""
        idx = self._slot_index(target_slot)
        if idx is None:
            return False

        pred_id = target_slot.supersedes
        succ_id = target_slot.superseded_by
        for other in self.memory.slots:
            if other is target_slot:
                continue
            if pred_id is not None and id(other) == pred_id:
                other.superseded_by = succ_id
            if succ_id is not None and id(other) == succ_id:
                other.supersedes = pred_id

        self.memory.slots.pop(idx)
        self.slot_texts.pop(id(target_slot), None)
        if idx < len(self.slot_dense_embs):
            self.slot_dense_embs.pop(idx)
        self.slot_contra_counts.pop(id(target_slot), None)
        self.slot_contra_anchors.pop(id(target_slot), None)
        self.bm25_index.remove_document(id(target_slot))
        return True

    def _repair_chain_after_removal(self, removed_slot: MemorySlot) -> None:
        pred_id = removed_slot.supersedes
        succ_id = removed_slot.superseded_by
        for other in self.memory.slots:
            if pred_id is not None and id(other) == pred_id:
                other.superseded_by = succ_id
            if succ_id is not None and id(other) == succ_id:
                other.supersedes = pred_id

    def _evict(
        self,
        current_time: float,
        protected_slot_ids: Optional[set] = None
    ) -> Tuple[int, MemorySlot]:
        """Evicts a slot according to the configured retention policy.

        ``protected_slot_ids`` are excluded from eviction by every policy.  If
        all remaining slots are protected the policy falls back to all slots
        (the caller decides whether that is acceptable).
        """
        protected = set(protected_slot_ids or ())
        evict_idx, evicted_slot = evict_slot_phase24(
            memory=self.memory,
            eviction_policy=self.retention_policy,
            current_time=current_time,
            slot_texts=self.slot_texts,
            slot_dense_embs=self.slot_dense_embs,
            slot_contra_counts=self.slot_contra_counts,
            slot_contra_anchors=self.slot_contra_anchors,
            protected_slot_ids=protected or None
        )
        self._repair_chain_after_removal(evicted_slot)
        self.bm25_index.remove_document(id(evicted_slot))
        self.total_evictions += 1
        return evict_idx, evicted_slot

    def temporal_history_violations(self) -> List[Dict[str, Any]]:
        """Checks that temporally exclusive facts cannot overlap.

        Only facts the engine itself treats as versions of one attribute are
        grouped: slots connected by supersession links, plus active slots that
        assert the identical value of the same non-cumulative attribute.
        Multi-valued and cumulative predicates, and same-predicate statements
        the engine conservatively retained as independent, are intentional
        coexistence and are not violations.

        Within each group, intervals are half-open ``[valid_from,
        valid_until)`` and must be disjoint; at most one slot may be active
        (``valid_until is None``).  Malformed intervals are reported globally.
        Links that connect different subjects, or attributes that belong to
        provably different exclusive groups, are reported as
        ``inconsistent_link`` and are not grouped.  Structurally broken links
        are reported as ``dangling_link`` (target object absent),
        ``self_link`` (slot points at itself), or ``non_reciprocal_link``
        (only one side confirms the relationship), and are never grouped.

        Returns a list of violation dicts; an empty list means the invariant
        holds.
        """
        slots = self.memory.slots
        n = len(slots)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        index_by_id = {id(slot): idx for idx, slot in enumerate(slots)}
        violations: List[Dict[str, Any]] = []
        seen_links = set()
        for i, slot in enumerate(slots):
            for link in (slot.supersedes, slot.superseded_by):
                if link is None:
                    continue
                j = index_by_id.get(link)
                if j is None:
                    violations.append({
                        "reason": "dangling_link",
                        "slot_index": i,
                        "link": link,
                    })
                    continue
                if j == i:
                    violations.append({
                        "reason": "self_link",
                        "slot_index": i,
                    })
                    continue
                pair = (min(i, j), max(i, j))
                if pair in seen_links:
                    continue
                seen_links.add(pair)
                a, b = slots[pair[0]], slots[pair[1]]
                reciprocal = (
                    (a.supersedes == id(b) and b.superseded_by == id(a))
                    or (b.supersedes == id(a) and a.superseded_by == id(b))
                )
                if not reciprocal:
                    violations.append({
                        "reason": "non_reciprocal_link",
                        "slot_indices": [pair[0], pair[1]],
                    })
                    continue
                subject_match = (
                    self._normalize_subject(a.debug_subject)
                    == self._normalize_subject(b.debug_subject)
                )
                groups_a: set = set()
                for pred in self._slot_predicates(a):
                    groups_a |= self._exclusive_group_ids(pred)
                groups_b: set = set()
                for pred in self._slot_predicates(b):
                    groups_b |= self._exclusive_group_ids(pred)
                # Only a provable clash between two known exclusive groups is
                # treated as an attribute mismatch.  Paraphrases and negated
                # restatements often have no group or a changed predicate, and
                # the engine legitimately links them.
                attribute_match = not (
                    groups_a and groups_b and groups_a.isdisjoint(groups_b)
                )
                if subject_match and attribute_match:
                    union(*pair)
                else:
                    # A supersession link asserts two versions of one fact.
                    # Grouping across subjects or attributes would hide that
                    # structural corruption inside an interval check.
                    violations.append({
                        "reason": "inconsistent_link",
                        "slot_indices": [pair[0], pair[1]],
                        "subject_match": subject_match,
                        "attribute_match": attribute_match,
                    })

        for i in range(n):
            for j in range(i + 1, n):
                if self._is_duplicate_active_fact(slots[i], slots[j]):
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        for idx, s in enumerate(slots):
            if s.valid_until is not None and s.valid_until < s.valid_from:
                violations.append({
                    "reason": "invalid_interval",
                    "slot_index": idx,
                    "valid_from": s.valid_from,
                    "valid_until": s.valid_until,
                })

        for indices in groups.values():
            ordered = sorted(
                indices,
                key=lambda i: (slots[i].valid_from, slots[i].created_at, i)
            )

            active = [idx for idx in ordered if slots[idx].is_active]
            if len(active) > 1:
                violations.append({
                    "reason": "multiple_active",
                    "slot_indices": active,
                })

            for left, right in zip(ordered, ordered[1:]):
                a, b = slots[left], slots[right]
                a_end = math.inf if a.valid_until is None else a.valid_until
                if a_end > b.valid_from:
                    violations.append({
                        "reason": "overlap",
                        "slot_indices": [left, right],
                        "valid_from": [a.valid_from, b.valid_from],
                        "valid_until": [a.valid_until, b.valid_until],
                    })
        return violations

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
            lookup_subject = self._normalize_subject(subject)
            for idx, slot in enumerate(self.memory.slots):
                if self._normalize_subject(slot.debug_subject) != lookup_subject:
                    continue
                if predicate is not None and predicate not in (
                    slot.debug_predicate, slot.latest_predicate
                ):
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
            target_slot = next(
                (s for s in self.memory.slots if id(s) == slot_id), None
            )
            if target_slot is None:
                return False
            # Removes the slot and re-links its predecessor to its successor.
            return self._remove_slot_identity(target_slot)

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
                "total_value_verifications": self.total_value_verifications,
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
