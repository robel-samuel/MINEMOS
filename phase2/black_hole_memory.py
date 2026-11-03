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
import logging
import math
import os
import re
import threading
import time
import unicodedata
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

_LOG = logging.getLogger(__name__)


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

# Predicate cardinality policy.  Exclusivity is never inferred from predicate
# identity alone: a predicate is EXCLUSIVE only when it matches a configured
# exclusive group (or an explicit override), MULTI_VALUED only when listed as
# cumulative (or overridden), and UNKNOWN otherwise.  UNKNOWN predicates are
# never treated as exclusive.
PREDICATE_POLICY_EXCLUSIVE = "exclusive"
PREDICATE_POLICY_MULTI_VALUED = "multi_valued"
PREDICATE_POLICY_UNKNOWN = "unknown"

UNLINKED_EXCLUSIVE_CONFLICT = "unlinked_exclusive_temporal_conflict"

# A statement carrying one of these markers describes a past state and must not
# be consolidated into (or supersede) a current one.  "no longer" is excluded
# on purpose: it asserts a current negation, not a historical state.
HISTORICAL_STATEMENT_MARKERS = {
    "used to", "previously", "formerly", "in the past", "at one time"
}

# Only unambiguous abbreviations are expanded.  "AA" is deliberately absent
# (Addis Ababa vs American Airlines), so it is never merged by normalization.
ABBREVIATION_ALIASES = {
    "nyc": "new york city",
    "uk": "united kingdom",
    "usa": "united states",
    "uae": "united arab emirates",
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
        verifier: Optional[Any] = None,
        exclusive_predicates: Optional[Any] = None,
        multi_valued_predicates: Optional[Any] = None,
        retrieval_backend: Optional[str] = None,
        retrieval_config: Optional[Any] = None,
        qdrant_config: Optional[Any] = None,
        retrieval_mode: Optional[str] = None
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

        # Optional per-instance policy overrides.  They are intentionally not
        # persisted in checkpoints: an archive stores facts, not configuration,
        # and a restored instance uses the policy of the process that loads it.
        self._exclusive_predicates = self._predicate_config_set(exclusive_predicates)
        self._multi_valued_predicates = self._predicate_config_set(multi_valued_predicates)

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

        # Monotonic, process-local version of everything retrieval derives
        # state from (slot membership, dense rows, validity metadata, texts).
        # It is bumped *before* any mutation so that a reader comparing versions
        # can never mistake stale derived state for current state.  Never
        # serialized: a restored engine starts a new epoch and derived state is
        # rebuilt from the restored slots.
        self._state_version = 0

        # Retrieval backend selection.  Canonical is the reference path and the
        # legacy default; production deployments opt into the strict Qdrant
        # profile, and canonical-only deployments never import or contact
        # Qdrant.  Like the predicate policy above, this is process
        # configuration and is not serialized into checkpoints.
        from phase2.retrieval.config import (
            CANONICAL, CANONICAL_ONLY, PRODUCTION, QDRANT, RetrievalConfig,
            RetrievalConfigurationError, normalize_backend, normalize_mode)
        self._retrieval_config = RetrievalConfig.from_value(retrieval_config)
        env_mode = os.environ.get("MINEMOS_RETRIEVAL_MODE")
        self._retrieval_mode = normalize_mode(
            retrieval_mode if retrieval_mode is not None
            else env_mode if env_mode is not None
            else self._retrieval_config.mode)

        requested_backend = retrieval_backend
        if (requested_backend is None
                and self._retrieval_config.backend != CANONICAL):
            requested_backend = self._retrieval_config.backend

        if self._retrieval_mode == PRODUCTION:
            if (requested_backend is not None
                    and normalize_backend(requested_backend) != QDRANT):
                raise RetrievalConfigurationError(
                    "retrieval_mode='production' requires the qdrant backend; "
                    f"got retrieval_backend={requested_backend!r}. Use "
                    "retrieval_mode='canonical' for canonical-only "
                    "deployments.")
            self._retrieval_backend = QDRANT
        elif self._retrieval_mode == CANONICAL_ONLY:
            if (requested_backend is not None
                    and normalize_backend(requested_backend) != CANONICAL):
                raise RetrievalConfigurationError(
                    "retrieval_mode='canonical' does not permit another "
                    f"backend; got retrieval_backend={requested_backend!r}.")
            self._retrieval_backend = CANONICAL
        else:
            self._retrieval_backend = normalize_backend(
                requested_backend
                if requested_backend is not None else CANONICAL)
        self._retrieval_adapter_obj = None
        self._retrieval_fallback_warned = False
        self._index_failure_warned = False

        # Qdrant settings are validated eagerly (without importing the client)
        # when the backend is selected or an explicit config is supplied.
        # Secrets stay inside QdrantConfig and are never serialized.
        if self._retrieval_mode == CANONICAL_ONLY:
            self._qdrant_config = None
        elif qdrant_config is not None or self._retrieval_backend == QDRANT:
            from phase2.retrieval.qdrant_config import QdrantConfig
            try:
                self._qdrant_config = QdrantConfig.from_value(qdrant_config)
            except (TypeError, ValueError) as exc:
                if self._retrieval_mode == PRODUCTION:
                    raise RetrievalConfigurationError(
                        "retrieval_mode='production' needs a valid Qdrant "
                        "server configuration. Set QDRANT_URL (and "
                        "QDRANT_API_KEY for remote hosts) or pass "
                        f"qdrant_config. Original error: {exc}") from exc
                raise
            if self._retrieval_mode == PRODUCTION:
                try:
                    self._qdrant_config.validate_production()
                except ValueError as exc:
                    raise RetrievalConfigurationError(
                        f"invalid production Qdrant configuration: {exc}"
                    ) from exc
        else:
            self._qdrant_config = None

    def _invalidate_retrieval_cache(self) -> int:
        """Mark retrieval-derived state stale and return the new state version.

        Every engine mutation pathway calls this before it changes slots or the
        auxiliary stores.  Bumping before the mutation is deliberate: if the
        mutation then raises or returns early, derived state is still stale
        (fail-safe over-invalidation), never incorrectly current.

        This counter is the only supported notification mechanism.  It cannot
        observe a direct field write on a ``MemorySlot`` or a direct
        ``memory.slots`` edit from outside the engine; such mutation is
        unsupported.  Test fixtures that must construct state out of band call
        this method explicitly after the edit.
        """
        with self._lock:
            self._state_version += 1
            return self._state_version

    # -- optional retrieval backend ------------------------------------- #
    @property
    def retrieval_backend(self) -> str:
        return self._retrieval_backend

    @property
    def retrieval_mode(self) -> str:
        return self._retrieval_mode

    def set_retrieval_backend(self, backend: str) -> str:
        """Switch retrieval backends at runtime.

        Switching to ``"canonical"`` is the rollback path: the next query runs
        the canonical loop below and never consults snapshot state.  Switching
        between the optimized and incremental backends builds a fresh
        retriever; a stale snapshot can never be served to the new backend.

        The production profile permits only the standard Qdrant backend or an
        emergency canonical rollback.  Canonical-only mode permits canonical
        exclusively.
        """
        from phase2.retrieval.config import (
            CANONICAL, CANONICAL_ONLY, PRODUCTION, QDRANT,
            RetrievalConfigurationError, normalize_backend)
        value = normalize_backend(backend)
        if self._retrieval_mode == CANONICAL_ONLY and value != CANONICAL:
            raise RetrievalConfigurationError(
                "retrieval_mode='canonical' permits only the canonical "
                f"backend, got {value!r}")
        if (self._retrieval_mode == PRODUCTION
                and value not in (QDRANT, CANONICAL)):
            raise RetrievalConfigurationError(
                "retrieval_mode='production' permits the standard qdrant "
                f"backend or an emergency canonical rollback, got {value!r}")
        if value == QDRANT and self._qdrant_config is None:
            from phase2.retrieval.qdrant_config import QdrantConfig
            try:
                self._qdrant_config = QdrantConfig.from_env()
            except (TypeError, ValueError) as exc:
                raise RetrievalConfigurationError(
                    "cannot switch to qdrant: set QDRANT_URL (server) or "
                    f"QDRANT_PATH (development) first. Original error: {exc}"
                ) from exc
            if self._retrieval_mode == PRODUCTION:
                try:
                    self._qdrant_config.validate_production()
                except ValueError as exc:
                    raise RetrievalConfigurationError(
                        f"invalid production Qdrant configuration: {exc}"
                    ) from exc
        with self._lock:
            self._retrieval_backend = value
            if self._retrieval_adapter_obj is not None:
                self._retrieval_adapter_obj.set_backend(value)
        return value

    def _get_retrieval_adapter(self):
        adapter = self._retrieval_adapter_obj
        if adapter is None:
            from phase2.retrieval.adapter import RetrievalAdapter
            with self._lock:
                if self._retrieval_adapter_obj is None:
                    self._retrieval_adapter_obj = RetrievalAdapter(
                        self, self._retrieval_config,
                        backend=self._retrieval_backend,
                        qdrant_config=self._qdrant_config)
                adapter = self._retrieval_adapter_obj
        return adapter

    def retrieval_stats(self) -> Dict[str, Any]:
        """Report the configured backend and whether it is actually serving.

        ``state`` is ``canonical``, ``not_initialized`` (selected but not yet
        contacted), ``ready``, or ``degraded`` (the last index operation failed
        and queries are falling back to canonical).  This prevents a selected
        production backend from being mistaken for a healthy one.
        """
        with self._lock:
            adapter = self._retrieval_adapter_obj
            backend = self._retrieval_backend
            mode = self._retrieval_mode
        if adapter is None:
            canonical = backend == "canonical"
            return {
                "backend": backend,
                "mode": mode,
                "active": not canonical,
                "initialized": False,
                "degraded": False,
                "state": "canonical" if canonical else "not_initialized",
                "strategy": None,
                "manager": None,
                "published_version": None,
            }
        stats = adapter.stats()
        stats["backend"] = backend
        stats["mode"] = mode
        retriever = adapter.retriever
        stats["initialized"] = retriever is not None
        manager = stats.get("manager") or {}
        if backend == "canonical":
            # A canonical rollback is a healthy state, regardless of errors
            # recorded by the previously active retriever.
            degraded = False
        else:
            degraded = bool(
                manager.get("dirty") or manager.get("last_error")
                or stats.get("last_error"))
        stats["degraded"] = degraded
        stats["state"] = (
            "canonical" if backend == "canonical"
            else "degraded" if degraded
            else "ready" if retriever is not None
            else "not_initialized")
        stats["active"] = backend != "canonical"
        return stats

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

    @classmethod
    def _normalize_subject(cls, subject: Optional[str]) -> str:
        return cls._normalize_text(subject)

    @staticmethod
    def _predicate_config_set(values: Optional[Any]) -> set:
        """Normalizes a predicate policy configuration to a lowercase set.

        A bare string is one predicate, not an iterable of characters: passing
        ``exclusive_predicates="telemetry mode"`` must not silently configure
        the letters t/e/l/m/o/r/y/d.
        """
        if values is None:
            return set()
        if isinstance(values, str):
            values = (values,)
        return {
            str(p).strip().casefold() for p in values if p and str(p).strip()
        }

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
    def _normalize_text(text: Optional[str]) -> str:
        """Case-, width-, whitespace-, and edge-punctuation-insensitive form.

        NFKC folds compatibility forms (full-width letters, ligatures) before
        casefolding, so noisy input compares against clean text without any
        model call.
        """
        if text is None:
            return ""
        normalized = unicodedata.normalize("NFKC", str(text)).casefold()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized.strip(" \t\r\n.,;:!?\"'")

    @classmethod
    def _normalize_value(cls, value: Optional[str]) -> str:
        return cls._normalize_text(value)

    @classmethod
    def _comparison_tokens(cls, text: Optional[str]) -> List[str]:
        """Token sequence used for lexical equivalence, after alias expansion."""
        normalized = cls._normalize_text(text)
        tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        expanded: List[str] = []
        for token in tokens:
            alias = ABBREVIATION_ALIASES.get(token)
            expanded.extend(alias.split() if alias else [token])
        return expanded

    @staticmethod
    def _single_edit_kind(a: str, b: str) -> Optional[str]:
        """Classify the edit between two tokens, or None when it exceeds one."""
        if a == b:
            return None
        if abs(len(a) - len(b)) > 1:
            return None
        if len(a) == len(b):
            diffs = [i for i in range(len(a)) if a[i] != b[i]]
            if len(diffs) == 1:
                return "substitution"
            if (len(diffs) == 2
                    and diffs[1] == diffs[0] + 1
                    and a[diffs[0]] == b[diffs[1]]
                    and a[diffs[1]] == b[diffs[0]]):
                return "transposition"
            return None
        shorter, longer = (a, b) if len(a) < len(b) else (b, a)
        i = 0
        while i < len(shorter) and shorter[i] == longer[i]:
            i += 1
        if shorter[i:] == longer[i + 1:]:
            return "indel"
        return None

    @classmethod
    def _tokens_are_typo_equivalent(
        cls, tokens_a: List[str], tokens_b: List[str]
    ) -> bool:
        """Exactly one token differs, by an insertion/deletion/transposition.

        Single-letter substitutions are deliberately excluded: they are the
        ambiguous case (Iran/Iraq, Rome/Roma), where merging two distinct
        names is worse than leaving a misspelling unconflated.
        """
        if len(tokens_a) != len(tokens_b):
            return False
        diffs = [(x, y) for x, y in zip(tokens_a, tokens_b) if x != y]
        if len(diffs) != 1:
            return False
        x, y = diffs[0]
        if not (x.isalpha() and y.isalpha()):
            return False
        if min(len(x), len(y)) < 4 or x[0] != y[0]:
            return False
        return cls._single_edit_kind(x, y) in {"indel", "transposition"}

    @classmethod
    def _lexically_equivalent_value(cls, a: Optional[str], b: Optional[str]) -> bool:
        tokens_a = cls._comparison_tokens(a)
        tokens_b = cls._comparison_tokens(b)
        if not tokens_a or not tokens_b:
            return False
        return tokens_a == tokens_b or cls._tokens_are_typo_equivalent(tokens_a, tokens_b)

    @classmethod
    def _subjects_are_equivalent(cls, a: Optional[str], b: Optional[str]) -> bool:
        norm_a = (a or "").strip().casefold()
        norm_b = (b or "").strip().casefold()
        if norm_a == norm_b:
            return True
        if not norm_a or not norm_b:
            return False
        tokens_a = cls._comparison_tokens(norm_a)
        tokens_b = cls._comparison_tokens(norm_b)
        if not tokens_a or not tokens_b:
            return False
        return tokens_a == tokens_b or cls._tokens_are_typo_equivalent(tokens_a, tokens_b)

    @staticmethod
    def _has_historical_marker(text: Optional[str]) -> bool:
        lowered = (text or "").casefold()
        return any(_contains_marker(lowered, marker) for marker in HISTORICAL_STATEMENT_MARKERS)

    @staticmethod
    def _has_negation(text: Optional[str]) -> bool:
        from benchmarks.phase26_verification import extract_negation_markers
        return bool(extract_negation_markers(text or ""))

    def _slot_has_negation(self, slot: MemorySlot) -> bool:
        statement = self.slot_texts.get(id(slot), slot.text or "")
        predicates = [p for p in (slot.debug_predicate, slot.latest_predicate) if p]
        return any(self._has_negation(text) for text in [statement, *predicates])

    def classify_predicate(self, predicate: Optional[str]) -> str:
        """Cardinality policy for one predicate.

        Explicit per-instance overrides win, so a caller can reclassify a
        default group member.  Otherwise cumulative predicates are
        multi-valued, configured exclusive-group members are exclusive, and
        everything else is unknown.  Unknown predicates are never treated as
        exclusive.
        """
        pred = (predicate or "").strip().casefold()
        if not pred:
            return PREDICATE_POLICY_UNKNOWN
        if pred in self._exclusive_predicates:
            return PREDICATE_POLICY_EXCLUSIVE
        if pred in self._multi_valued_predicates:
            return PREDICATE_POLICY_MULTI_VALUED
        if pred in CUMULATIVE_PREDICATES:
            return PREDICATE_POLICY_MULTI_VALUED
        if any(
            _contains_marker(pred, marker)
            for group in EXCLUSIVE_SYNONYM_GROUPS
            for marker in group
        ):
            return PREDICATE_POLICY_EXCLUSIVE
        return PREDICATE_POLICY_UNKNOWN

    def _predicates_share_exclusive_attribute(
        self, pred_a: Optional[str], pred_b: Optional[str]
    ) -> bool:
        pred_a = (pred_a or "").strip().casefold()
        pred_b = (pred_b or "").strip().casefold()
        if not pred_a or not pred_b:
            return False
        if not (
            self.classify_predicate(pred_a) == PREDICATE_POLICY_EXCLUSIVE
            and self.classify_predicate(pred_b) == PREDICATE_POLICY_EXCLUSIVE
        ):
            return False
        return pred_a == pred_b or self._share_exclusive_group(pred_a, pred_b)

    def _lexically_same_fact(
        self,
        slot: MemorySlot,
        candidate: Candidate,
        slot_text: Optional[str] = None,
        cand_text: Optional[str] = None
    ) -> bool:
        """Whether the candidate restates the slot up to surface noise.

        Deterministic: equivalent entity, compatible predicate, lexically
        equivalent values, identical negation presence, and no historical
        marker.  It never calls a model.
        """
        slot_text = (
            slot_text if slot_text is not None
            else self.slot_texts.get(id(slot), slot.text or "")
        )
        cand_text = cand_text if cand_text is not None else (
            f"{candidate.subject} {candidate.predicate} {candidate.object}."
        )
        if not slot.is_active:
            return False
        if self._has_historical_marker(cand_text):
            return False
        if not self._subjects_are_equivalent(slot.debug_subject, candidate.subject):
            return False
        cand_pred = (candidate.predicate or "").strip().casefold()
        if not any(
            slot_pred == cand_pred
            or self._predicates_share_exclusive_attribute(slot_pred, cand_pred)
            for slot_pred in self._slot_predicates(slot)
        ):
            return False
        slot_value = self._slot_value_text(slot)
        if not slot_value or not candidate.object:
            return False
        if not self._lexically_equivalent_value(slot_value, candidate.object):
            return False
        return self._has_negation(slot_text) == self._has_negation(cand_text)

    def _find_lexically_equivalent_slot(self, candidate: Candidate) -> Optional[int]:
        for idx, slot in enumerate(self.memory.slots):
            if slot.is_active and self._lexically_same_fact(slot, candidate):
                return idx
        return None

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
        # A typo or an unambiguous abbreviation is the same value; deciding
        # that lexically avoids spending a verifier call and avoids the
        # verifier's noisy behaviour on near-identical strings.
        if self._lexically_equivalent_value(value_a, value_b):
            return True
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
        """Absorbs an observation, then reconciles the optional Qdrant index.

        Index failure is isolated from the memory decision: the canonical slot
        mutation still commits, the index is marked dirty, and the next query
        falls back to canonical until a sync succeeds.
        """
        try:
            return self._absorb_impl(observation, timestamp, confidence)
        finally:
            self._sync_qdrant_if_active()

    def _warn_index_failure(self, stage: str, exc: BaseException) -> None:
        from phase2.retrieval.config import PRODUCTION
        if (self._retrieval_mode != PRODUCTION
                or self._index_failure_warned):
            return
        self._index_failure_warned = True
        _LOG.warning(
            "Qdrant %s failed; canonical memory is intact and queries fall "
            "back to canonical retrieval: %s", stage, exc)

    def _warn_backend_fallback(self, exc: BaseException) -> None:
        from phase2.retrieval.config import PRODUCTION
        if (self._retrieval_mode != PRODUCTION
                or self._retrieval_fallback_warned):
            return
        self._retrieval_fallback_warned = True
        _LOG.warning(
            "Qdrant is configured but unavailable; serving this query from "
            "canonical MINEMOS retrieval: %s", exc)

    def _sync_qdrant_if_active(self) -> None:
        """Reconcile the Qdrant index after a canonical mutation.

        The retriever is built eagerly so the memory is indexed at ingestion
        time rather than waiting for the first query.  Any failure is recorded
        on the retriever as a dirty index and swallowed here: the canonical
        mutation has already committed and must never be rolled back for a
        derived-index problem.
        """
        if self._retrieval_backend != "qdrant":
            return
        adapter = self._retrieval_adapter_obj
        if adapter is not None and adapter.backend != "qdrant":
            return
        try:
            if adapter is None:
                adapter = self._get_retrieval_adapter()
            adapter.ensure_retriever().sync_state()
        except Exception as exc:  # noqa: BLE001 - canonical state already committed
            self._warn_index_failure("write-path sync", exc)

    def _sync_qdrant_after_restore(self) -> None:
        """Force a full reconcile after canonical state is replaced on load."""
        if self._retrieval_backend != "qdrant":
            return
        try:
            adapter = self._retrieval_adapter_obj
            if adapter is None:
                adapter = self._get_retrieval_adapter()
            adapter.ensure_retriever().sync_state(
                force=True, use_retry=False)
        except Exception as exc:  # noqa: BLE001 - the next query falls back canonical
            self._warn_index_failure("checkpoint-restore sync", exc)

    def _absorb_impl(
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
        # A historical statement reports a past state.  It is recorded as a
        # closed observation and is never consolidated into, or allowed to
        # supersede, a current one.
        is_historical_observation = self._has_historical_marker(cand_text)
        x_t = self.lex_encoder.encode_full(candidate, dim=self.dim)
        cand_dense = self.dense_encoder.encode(cand_text)

        with self._lock:
            # Invalidate before the first possible mutation.  Every branch
            # below either mutates slots/auxiliary stores or returns a
            # rejection; over-invalidating a rejection is safe, failing to
            # invalidate a partial mutation is not.
            self._invalidate_retrieval_cache()
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

            if self.memory.slots and not is_historical_observation:
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
                            # A near-identical restatement (typo, case/space
                            # noise, unambiguous abbreviation) is the same fact
                            # even though the verifier marked it contradictory.
                            # Consolidating it prevents a false state change.
                            if self._lexically_same_fact(slot, candidate):
                                target_idx = cand_idx
                                target_pred = pred
                                break
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

                # Deterministic lexical fallback: retrieval/verification can
                # miss a near-identical restatement (for example a one-letter
                # subject typo).  If an active slot is the same fact up to
                # surface noise, consolidate instead of allocating a duplicate.
                if (target_idx is None and not is_temporal_change
                        and not is_historical_predecessor):
                    equivalent_idx = self._find_lexically_equivalent_slot(candidate)
                    if equivalent_idx is not None:
                        target_idx = equivalent_idx

                best_sim = dense_candidates[0][1] if dense_candidates else 0.0
                n_t = novelty(best_sim, memory_empty=False)
            else:
                best_sim = 0.0
                n_t = 1.0

            alpha_t = n_t

            # ── Case 1: Consolidate Semantic Paraphrase into Existing Slot ──
            if target_idx is not None:
                slot = self.memory.slots[target_idx]
                slot_value = self._slot_value_text(slot)
                # A misspelled entity/value or an abbreviation restates the
                # stored fact, but it is not a better canonical form.  Keep the
                # incumbent wording (and its index entries) rather than
                # rewriting the fact with the noisy surface form.
                surface_variant = (
                    (
                        self._normalize_subject(slot.debug_subject)
                        != self._normalize_subject(candidate.subject)
                        and self._subjects_are_equivalent(
                            slot.debug_subject, candidate.subject
                        )
                    )
                    or (
                        bool(slot_value) and bool(candidate.object)
                        and self._normalize_value(slot_value)
                        != self._normalize_value(candidate.object)
                        and self._lexically_equivalent_value(
                            slot_value, candidate.object
                        )
                    )
                )
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
                if not surface_variant:
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
                    "text": cand_text,
                    "surface_variant": surface_variant,
                    "stored_text": self.slot_texts.get(id(slot), slot.text),
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

            # ── Case 2c: Historical Statement (Past State, Unknown Window) ──
            if is_historical_observation:
                # The statement's true start/end are unknown.  Record it as a
                # closed observation at its report time: it neither merges
                # into nor deactivates the current state, and it is not linked
                # into the interval chain (which would imply known bounds).
                protected = {
                    id(s) for s in self.memory.slots
                    if s.is_active
                    and self._subjects_are_equivalent(s.debug_subject, candidate.subject)
                }
                evicted = None
                if self.memory.is_full():
                    if not self._has_eviction_candidate(protected):
                        return {
                            "action": "historical_rejected",
                            "reason": "capacity_would_remove_active_state",
                            "slot_id": None,
                            "valid_from": t_now,
                            "valid_until": t_now,
                            "text": cand_text,
                        }
                    evict_idx, evicted = self._evict(t_now, protected_slot_ids=protected)

                slot = self.memory.add(
                    key=x_t, value=x_t.copy(), confidence=candidate.confidence,
                    timestamp=t_now, debug_subject=candidate.subject,
                    debug_predicate=candidate.predicate, text=cand_text,
                    dense_emb=cand_dense, valid_from=t_now, valid_until=t_now,
                    supersedes=None, latest_predicate=candidate.predicate,
                    latest_object=candidate.object
                )
                self.slot_texts[id(slot)] = cand_text
                self.slot_dense_embs.append(cand_dense)
                self.bm25_index.add_document(id(slot), cand_text)
                self.total_inserts += 1
                return {
                    "action": "historical_observation",
                    "slot_index": len(self.memory.slots) - 1,
                    "slot_id": id(slot),
                    "evicted_slot_id": id(evicted) if evicted is not None else None,
                    "valid_from": t_now,
                    "valid_until": t_now,
                    "text": cand_text,
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

        self._invalidate_retrieval_cache()
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

        self._invalidate_retrieval_cache()
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
        self._invalidate_retrieval_cache()
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

    @staticmethod
    def _slots_are_linked(a: MemorySlot, b: MemorySlot) -> bool:
        return (
            a.supersedes == id(b) or a.superseded_by == id(b)
            or b.supersedes == id(a) or b.superseded_by == id(a)
        )

    def exclusive_temporal_conflicts(self) -> List[Dict[str, Any]]:
        """Reports active facts that cannot both hold for one exclusive attribute.

        This closes the Phase 29 U1 gap: two active slots can assert different
        values of the same exclusive attribute without ever being linked as
        versions (for example after importing legacy state, or when the
        verifier returned NEUTRAL).  The structural invariant cannot see that
        pair, because it only groups linked or identical-value slots.

        The check is read-only.  It never mutates slots, intervals, links,
        indexes, or counters, and it never calls a model, so it is safe to run
        repeatedly.  Value equivalence is therefore decided lexically
        (normalization, unambiguous abbreviation expansion, one edit); two
        different surface forms are reported rather than silently assumed
        equivalent.

        Exclusivity comes from the predicate policy: a pair is reported only
        when both predicates are ``EXCLUSIVE`` and either identical or in the
        same configured exclusive group.  Multi-valued and unknown predicates
        are never reported, so legitimate multi-valued facts stay silent.  A
        negated statement does not assert its object as a value, so a negated
        slot and a different value are not reported; same-value negations stay
        visible through the structural invariant (``multiple_active``).

        ``confidence`` is the policy determination (always 1.0 for a
        configured exclusive predicate) and is scoped by
        ``confidence_scope="predicate_policy_only"``.
        ``semantic_equivalence="lexical_only"`` records that values were
        compared lexically, and ``conflict_status="unresolved"`` records that
        the finding reports the conflict without choosing a winner.

        Returns one finding per conflicting active pair, in slot order.  An
        empty list means no unlinked exclusive conflict was found.
        """
        slots = self.memory.slots
        findings: List[Dict[str, Any]] = []
        for i in range(len(slots)):
            a = slots[i]
            if not a.is_active:
                continue
            for j in range(i + 1, len(slots)):
                b = slots[j]
                if not b.is_active:
                    continue
                if self._slots_are_linked(a, b):
                    continue
                if not self._subjects_are_equivalent(a.debug_subject, b.debug_subject):
                    continue

                attribute_pair = None
                for pred_a in self._slot_predicates(a):
                    for pred_b in self._slot_predicates(b):
                        if self._predicates_share_exclusive_attribute(pred_a, pred_b):
                            attribute_pair = (pred_a, pred_b)
                            break
                    if attribute_pair is not None:
                        break
                if attribute_pair is None:
                    continue

                value_a = self._slot_value_text(a)
                value_b = self._slot_value_text(b)
                if not value_a or not value_b:
                    continue
                if self._lexically_equivalent_value(value_a, value_b):
                    continue
                # A negated statement does not assert its object as the
                # current value: "no longer lives in Addis Ababa" is
                # compatible with "lives in Nairobi".  Same-value negation
                # pairs stay visible through the structural invariant, which
                # groups equal values and reports multiple_active.
                if self._slot_has_negation(a) or self._slot_has_negation(b):
                    continue

                a_end = math.inf if a.valid_until is None else a.valid_until
                b_end = math.inf if b.valid_until is None else b.valid_until
                overlap = a.valid_from < b_end and b.valid_from < a_end
                if not overlap:
                    continue

                findings.append({
                    "reason": UNLINKED_EXCLUSIVE_CONFLICT,
                    "slot_indices": [i, j],
                    "slot_ids": [id(a), id(b)],
                    "subject": a.debug_subject,
                    "predicates": [attribute_pair[0], attribute_pair[1]],
                    "values": [value_a, value_b],
                    "valid_from": [a.valid_from, b.valid_from],
                    "valid_until": [a.valid_until, b.valid_until],
                    "overlap": True,
                    "predicate_policy": PREDICATE_POLICY_EXCLUSIVE,
                    "detection_source": "active_unlinked_slots",
                    # Deterministic policy determination, not semantic truth:
                    # the checker never calls a verifier.
                    "confidence": 1.0,
                    # Explicitly scoped so consumers cannot read `confidence`
                    # as 100% semantic confidence in the conflict.
                    "confidence_scope": "predicate_policy_only",
                    "semantic_equivalence": "lexical_only",
                    "conflict_status": "unresolved",
                })
        return findings

    def temporal_history_violations(
        self, include_unlinked_conflicts: bool = False
    ) -> List[Dict[str, Any]]:
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

        ``include_unlinked_conflicts=True`` appends the findings of
        :meth:`exclusive_temporal_conflicts`, which covers unlinked active
        conflicts of exclusive attributes.  The default stays ``False`` so the
        structural invariant remains comparable with its Phase 29 results.

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

        if include_unlinked_conflicts:
            violations.extend(self.exclusive_temporal_conflicts())
        return violations

    @staticmethod
    def _normalize_query_filters(
        filters: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if filters is None:
            return None
        if not isinstance(filters, dict):
            raise TypeError(
                f"filters must be a dict or None, got {type(filters)!r}")
        allowed = {
            "subject", "predicate", "min_confidence", "min_update_count",
            "memory_ids",
        }
        unknown = set(filters) - allowed
        if unknown:
            raise ValueError(
                f"unknown query filter(s): {sorted(unknown)}; "
                f"expected {sorted(allowed)}")
        out: Dict[str, Any] = {}
        if filters.get("subject") is not None:
            out["subject"] = str(filters["subject"])
        if filters.get("predicate") is not None:
            out["predicate"] = str(filters["predicate"]).strip().lower()
        if filters.get("min_confidence") is not None:
            value = float(filters["min_confidence"])
            if not 0.0 <= value <= 1.0:
                raise ValueError("min_confidence must be in [0, 1]")
            out["min_confidence"] = value
        if filters.get("min_update_count") is not None:
            value = int(filters["min_update_count"])
            if value < 0:
                raise ValueError("min_update_count must be >= 0")
            out["min_update_count"] = value
        if filters.get("memory_ids") is not None:
            out["memory_ids"] = [str(m) for m in filters["memory_ids"]]
        return out or None

    def _slot_matches_filters(
        self, slot: MemorySlot, filters: Optional[Dict[str, Any]]
    ) -> bool:
        if not filters:
            return True
        if "subject" in filters:
            if (self._normalize_subject(slot.debug_subject)
                    != self._normalize_subject(filters["subject"])):
                return False
        if "predicate" in filters:
            wanted = filters["predicate"]
            predicates = {
                str(p).strip().lower()
                for p in (slot.debug_predicate, slot.latest_predicate)
                if p
            }
            if wanted not in predicates:
                return False
        if "min_confidence" in filters:
            if float(slot.confidence) < filters["min_confidence"]:
                return False
        if "min_update_count" in filters:
            if int(slot.update_count) < filters["min_update_count"]:
                return False
        if "memory_ids" in filters:
            if str(slot.memory_id) not in set(filters["memory_ids"]):
                return False
        return True

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        as_of: Optional[float] = None,
        active_only: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        hybrid: bool = False,
        fusion: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant memory records for a natural-language query.
        Supports point-in-time temporal queries via as_of.

        ``filters`` optionally restricts results by ``subject``, ``predicate``,
        ``min_confidence``, ``min_update_count`` or ``memory_ids``.  Metadata
        filters are pushed into Qdrant when that backend is selected, then
        re-applied exactly against the live engine.

        ``hybrid`` requests Qdrant dense candidates plus the existing BM25
        lexical index, merged and then re-ranked exactly by MINEMOS.  It
        requires the qdrant backend; if Qdrant is unavailable the answer falls
        through to canonical dense retrieval.

        ``fusion`` optionally selects the hybrid ordering experiment:
        ``"dense"`` (default) applies MINEMOS's exact dense re-ranking, while
        ``"rrf"`` orders the merged candidate union by reciprocal rank.  RRF is
        an opt-in experiment and is not used unless explicitly requested.

        Canonical retrieval is the reference implementation and the default.
        When another backend is explicitly selected, it answers first;
        ``RetrievalUnavailable`` falls through to the canonical loop below, so
        the canonical path is always the final safety net.
        """
        filters = self._normalize_query_filters(filters)
        if hybrid and self._retrieval_backend != "qdrant":
            raise ValueError("hybrid retrieval requires the qdrant backend")
        if fusion is not None:
            if not hybrid:
                raise ValueError("fusion requires hybrid=True")
            fusion = str(fusion).strip().lower()
            if fusion not in ("dense", "rrf"):
                raise ValueError(
                    f"fusion must be 'dense' or 'rrf', got {fusion!r}")
        if self._retrieval_backend != "canonical":
            from phase2.retrieval.adapter import RetrievalUnavailable
            if not filters or self._retrieval_backend == "qdrant":
                try:
                    return self._get_retrieval_adapter().query(
                        query_text, top_k=top_k, as_of=as_of,
                        active_only=active_only, filters=filters,
                        hybrid=hybrid, fusion=fusion)
                except RetrievalUnavailable as exc:
                    self._warn_backend_fallback(exc)

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

                if filters and not self._slot_matches_filters(slot, filters):
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
        """Deletes a slot and reconciles the optional Qdrant index."""
        try:
            return self._delete_slot_impl(slot_id)
        finally:
            self._sync_qdrant_if_active()

    def _delete_slot_impl(self, slot_id: int) -> bool:
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
                if self._delete_slot_impl(s_id):
                    count += 1

            if count:
                self._sync_qdrant_if_active()
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

    def close(self) -> None:
        """Release optional backend connections (Qdrant client, etc.)."""
        with self._lock:
            adapter = self._retrieval_adapter_obj
        if adapter is not None:
            adapter.close()

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
