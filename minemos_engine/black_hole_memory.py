"""
The MINEMOS memory engine.

A fixed-capacity persistent memory that absorbs observations, decides whether
each one restates, revises or adds to what it already knows, and answers queries
from canonical state. It provides:
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
import functools
import logging
import math
import numbers
import os
import re
import threading
import time
import unicodedata
from typing import Optional, Union, List, Dict, Any, Tuple, Iterable, Sequence

import numpy as np

from minemos_engine.structured_candidate import Candidate
from minemos_engine.memory_state import MemoryState, MemorySlot
from minemos_engine.memory_update import cosine_similarity, novelty
from minemos_engine.encoders import FieldAwareLexicalEncoder
from minemos_engine.intelligence.nli import NliSemanticGate, LazyNliSemanticGate
from minemos_engine.intelligence.dense import (
    DenseSemanticEncoder,
    LazyDenseSemanticEncoder,
    D_LEX,
    D_DENSE,
    LEXICAL_ENCODER,
    dense_address,
)
from minemos_engine.intelligence.addressing import adaptive_dense_address
from minemos_engine.intelligence.bm25 import (
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
from minemos_engine.intelligence.retention import evict_slot
from minemos_engine.intelligence.consolidation import consolidate_vector
from minemos_engine.intelligence.verification import CalibratedSemanticVerifier

_LOG = logging.getLogger(__name__)


# Automatic cross-conversation discovery is bounded.  It scans a dense
# candidate window, considers only document scopes whose rows carry at least
# two distinct query evidence terms, and evaluates a bounded number of scopes.
# One incidental matching word cannot make an unrelated conversation a search
# scope; the ordinary joint evidence gate still decides whether the selected
# scopes support an answer.
SCOPE_DISCOVERY_DEPTH = 1024
SCOPE_DISCOVERY_MAX_SCOPES = 8
SCOPE_DISCOVERY_MIN_TERMS = 2


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

# Placeholder subjects carry no identity claim, so a merge between one of
# them and a named subject is never blocked by the entity-grounding guard.
_GENERIC_SUBJECTS = {"", "entity", "someone", "something", "unknown", "item"}

# Engine-local guard against the bidirectional verifier's maximum-direction
# artifact.  The verifier still decides with max(fwd, rev); consolidation
# additionally requires that neither direction is effectively unsupported.
# 0.20 is deliberately far below the SAME threshold: it rejects one-way
# subsumption, not asymmetric paraphrase.
MUTUAL_ENTAILMENT_FLOOR = 0.20

# Explicit relation families for cross-predicate compatibility.  Two
# predicates from different, known families denote different relations and
# must not be merged merely because a whole-statement verifier says SAME.
# A predicate with no family is unclassified, never incompatible by guess.
# A copula does not name a content relation: in "Nathan is the chief
# architect", the relation is carried by the object.  It may inherit the role
# relation (and only that relation) when the other predicate is explicitly a
# role verb.  This never authorizes a copula-to-ownership, performance or
# preference merge.
COPULA_PREDICATES = frozenset({"is", "was", "are", "were"})
ROLE_RELATION_FAMILY = frozenset(EXCLUSIVE_SYNONYM_GROUPS[2]) | frozenset(
    {"works as", "serves as", "acts as", "serves", "serve", "serving"})

RELATION_FAMILY_GROUPS = (
    frozenset(EXCLUSIVE_SYNONYM_GROUPS[0])
    | frozenset({"living in", "based in"}),  # residence
    # Employer and work location are different attributes of one broad
    # employment relation.  Sharing the relation does not make "works at
    # Acme" interchangeable with "works in London".
    frozenset({"works at", "employed at", "employed by", "employer",
               "company", "job at"}),
    frozenset({"works in", "employed in", "job in", "work location",
               "office in"}),
    frozenset({"has", "have", "owns", "possesses", "possess"}),
    frozenset({"uses", "use", "utilizes", "utilises", "operates"}),
    frozenset({"likes", "like", "enjoys", "enjoy", "loves", "love",
               "prefers", "prefer"}),
    ROLE_RELATION_FAMILY,
    frozenset({"authored", "wrote", "written", "published", "penned"}),
    # "plays guitar" and "performs guitar" describe one performance relation.
    # Playing, owning and liking the same instrument do not.
    frozenset({"plays", "play", "played", "playing", "performs", "perform",
               "performed", "performing"}),
    # Leading an ensemble or a team is one leadership relation whichever verb
    # reports it.  Leading and directing the same object are not evidence that
    # an ownership or performance relation changed.
    frozenset({"directed", "directs", "directing", "conducted", "conducts",
               "conducting", "led", "leads", "lead", "leading"}),
    # Holding and earning a credential are one education relation.
    frozenset({
        "holds a phd in", "holds a doctorate in", "holds a degree in",
        "holds a master's in", "holds a bachelor's in",
        "earned a phd in", "earned a doctorate in", "earned a degree in",
        "earned a master's in", "earned a bachelor's in",
        "graduated with",
    }),
)
ROLE_FAMILY_INDEX = RELATION_FAMILY_GROUPS.index(ROLE_RELATION_FAMILY)

# Predicate cardinality policy.  Exclusivity is never inferred from predicate
# identity alone: a predicate is EXCLUSIVE only when it matches a configured
# exclusive group (or an explicit override), MULTI_VALUED only when listed as
# cumulative (or overridden), and UNKNOWN otherwise.  UNKNOWN predicates are
# never treated as exclusive.
PREDICATE_POLICY_EXCLUSIVE = "exclusive"
PREDICATE_POLICY_MULTI_VALUED = "multi_valued"
PREDICATE_POLICY_UNKNOWN = "unknown"

UNLINKED_EXCLUSIVE_CONFLICT = "unlinked_exclusive_temporal_conflict"
UNRESOLVED_TEMPORAL_CORRECTION = "unresolved_temporal_correction"
UNRESOLVED_STATE_CONFLICT = "unresolved_state_temporal_conflict"

# Verbs that report disposing of a previously held asset.  A disposal is a
# correction of an ownership state only for the same entity; it is never
# evidence that an unrelated relation ("works at", "plays") changed.
DISPOSAL_PREDICATES = frozenset({
    "sell", "sold", "sells", "selling", "divested", "divests", "divesting",
    "liquidated", "liquidates", "relinquished", "relinquishes",
    "dispose of", "disposed of", "disposes of",
    "give away", "gave away", "gives away",
})

# Verbs that report an ownership change without saying whether the asset was
# disposed of.  "Ada transferred Acme to Bob" can describe a sale, a gift or
# a move between holdings; "Ada lost Acme" can describe losing control or
# losing a contract.  They are never treated as proof of a correction: both
# observations stay active and the response reports an unresolved change.
OWNERSHIP_TRANSITION_PREDICATES = frozenset({
    "transferred", "transfers", "transferring", "transfer",
    "lost", "loses", "losing", "forfeited", "forfeits", "forfeiting",
})

# Scope words around a disposal.  An all/majority disposal removes the
# represented ownership, so a later one can correct an earlier ownership
# fact.  A partial disposal ("sold some shares") leaves ownership in place
# and is retained as an additional, compatible observation.
FULL_DISPOSAL_QUALIFIERS = frozenset({
    "all", "entire", "entirety", "complete", "completely", "whole", "full",
    "fully", "total", "totally", "majority", "controlling", "100", "100%",
})
PARTIAL_DISPOSAL_QUALIFIERS = frozenset({
    "some", "part", "partial", "partially", "portion", "minority", "half",
    "fraction", "percentage", "percent", "few", "several",
})
# Objects that describe a quantity of an entity rather than the entity itself.
# Selling "shares" without a quantity is not evidence of a total disposal.
QUANTITY_OBJECT_WORDS = frozenset({
    "share", "shares", "equity", "stake", "stakes", "interest", "interests",
    "stock", "position", "portion", "part", "ownership",
})

# Predicates whose value is a role held by the subject.  They currently have
# no explicit cardinality policy, but two different active values of the same
# role attribute cannot both be presented as the current settled state.
ROLE_STATE_PREDICATES = frozenset({
    "works as", "serves as", "acts as", "serves", "serve", "serving",
})

# Marker nouns whose value is one current state of the subject.  A bare marker
# predicate ("phase") and its copular variant ("phase is") name the same
# attribute; the list is also used to recognize "<modifier> <marker>" forms.
STATE_MARKER_PREDICATES = (
    "status", "state", "phase", "stage", "version", "revision", "edition",
)

# Leading modifiers that do not name a different attribute: "current phase"
# and "phase" assert the same state.  Deliberately narrow: in "review stage"
# the modifier *does* distinguish the attribute, so it is not listed.
STATE_NEUTRAL_MODIFIERS = frozenset({
    "current", "latest", "present",
})

# Verb-form state relations.  Surface variants of one relation share a key so
# two incompatible values are exposed; a different relation (residence vs
# employer) keeps its own key and never collides.  This is detection only:
# nothing here supersedes a prior value.
STATE_RELATION_KEYS = (
    ("employer", (
        "works at", "works for", "is employed at", "is employed by",
        "employed at", "employed by", "is working at", "is working for",
        "working at", "working for",
    )),
    ("workplace", ("works in", "is working in", "working in")),
    ("location", (
        "based in", "is based in", "based at", "is based at",
        "located in", "is located in", "situated in", "is situated in",
    )),
    ("residence", (
        "lives in", "is living in", "living in", "resides in",
        "is residing in", "residing in", "is from",
    )),
    ("spouse", (
        "married to", "is married to", "partnered with",
        "is partnered with",
    )),
)


# Grammatical glue that must not count as shared content when two object
# phrases are compared for compatibility.
VALUE_STOP_TOKENS = frozenset({
    "a", "an", "the", "of", "in", "at", "on", "for", "to", "from", "and",
    "or", "with", "by", "as", "is", "are", "was", "were", "be", "been",
    "his", "her", "their", "its", "my", "your", "our", "this", "that",
    "these", "those", "some", "any", "all",
})

# Words that describe how much of an entity is held rather than which entity
# it is.  They are removed before comparing the entity named by two object
# phrases ("the majority share in Acme" / "all equity in the Acme
# Corporation" both name Acme).
OBJECT_ENTITY_FILLER = frozenset({
    "a", "an", "the", "all", "any", "some", "most", "of", "in", "at", "on",
    "for", "to", "from", "and", "his", "her", "their", "its", "my", "your",
    "our", "majority", "minority", "controlling", "remaining", "share",
    "shares", "equity", "stake", "stakes", "interest", "interests",
    "stock", "position", "portion", "part", "ownership",
})

# Legal forms are compared instead of discarded.  "Acme" and "the Acme
# Corporation" can name one company, but "Acme Inc" and "Acme Ltd" are
# different registered entities: a disposal of one must not supersede
# ownership of the other.
ENTITY_LEGAL_FORM_ALIASES = {
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co",
    "limited": "ltd", "ltd": "ltd",
    "plc": "plc", "gmbh": "gmbh", "ag": "ag", "sa": "sa",
    "nv": "nv", "bv": "bv",
}

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


# Leading adverbs that do not change which state a predicate asserts:
# "currently lives in" is the same current-state relation as "lives in".
# Negations ("no longer", "not") are deliberately absent: they assert a
# different fact, not the same one with extra wording.
STATE_NEUTRAL_ADVERBS = frozenset({"currently", "presently", "still"})


def _normalise_state_predicate(predicate: Optional[str]) -> str:
    pred = (predicate or "").strip().casefold()
    if not pred:
        return ""
    pred = re.sub(r"\s+", " ", pred)
    pred = pred.strip(" \t.,!?;:")
    parts = pred.split()
    if len(parts) > 1 and parts[0] in STATE_NEUTRAL_ADVERBS:
        pred = " ".join(parts[1:])
    return pred


def _strip_leading_article(text: str) -> str:
    for article in ("the ", "a ", "an "):
        if text.startswith(article):
            return text[len(article):].strip()
    return text


def _state_relation_key(pred: str) -> Optional[str]:
    for key, forms in STATE_RELATION_KEYS:
        if pred in forms:
            return key
    return None


def _state_marker_key(head: str) -> Optional[str]:
    """The state marker named by ``head``, allowing neutral modifiers."""
    candidate = _strip_leading_article(head)
    if candidate in STATE_MARKER_PREDICATES:
        return candidate
    parts = candidate.split()
    if (len(parts) == 2 and parts[0] in STATE_NEUTRAL_MODIFIERS
            and parts[1] in STATE_MARKER_PREDICATES):
        return parts[1]
    return None



def _snapshot_locked(method):
    """Runs a read-only diagnostic under the engine lock.

    The temporal invariants walk several derived views of ``memory.slots``
    (sizing the union-find parent array, indexing links, then enumerating
    slots again).  Holding the engine RLock for the whole call makes those
    views one consistent snapshot, so a concurrent ``absorb`` cannot append
    between sizing the parent array and enumerating the slots.  It also means
    a diagnostic always reports the state of one instant rather than a
    mixture of before/after writes.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


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
        retrieval_mode: Optional[str] = None,
        strict_entity_identity: bool = False,
        model_cache_dir: Optional[str] = None,
    ):
        self.memory = MemoryState(capacity=capacity, dim=dim)
        self.capacity = self.memory.capacity
        self.dim = self.memory.dim
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

        # Opt-in identity conservatism.  In strict mode a one-edit difference
        # inside a multi-token name, title or value is treated as uncertain
        # identity rather than as a typo of the same entity.  Process
        # configuration, like the predicate policy above; not persisted.
        self.strict_entity_identity = bool(strict_entity_identity)

        # Optional per-instance policy overrides.  They are intentionally not
        # persisted in checkpoints: an archive stores facts, not configuration,
        # and a restored instance uses the policy of the process that loads it.
        self._exclusive_predicates = self._predicate_config_set(exclusive_predicates)
        self._multi_valued_predicates = self._predicate_config_set(multi_valued_predicates)

        # Core State (capacity/dim are validated by MemoryState)
        self.slot_texts: Dict[int, str] = {}
        self.slot_dense_embs: List[np.ndarray] = []
        self.slot_contra_counts: Dict[int, int] = {}
        self.slot_contra_anchors: Dict[int, List[Tuple[np.ndarray, float]]] = {}
        self.bm25_index = LocalBM25Index()

        # Models.  The default encoders are lazy: constructing a memory, or
        # restoring one, never downloads the Hugging Face checkpoints.  The
        # first operation that actually encodes or verifies provisions them
        # from the cache (or downloads them once) and reuses them afterwards.
        self.dense_encoder = dense_encoder or LazyDenseSemanticEncoder(
            cache_dir=model_cache_dir)
        self.nli_gate = nli_gate or LazyNliSemanticGate(
            cache_dir=model_cache_dir)
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
        self.total_suppressed_value_merges = 0
        self.total_near_miss_separations = 0
        self.total_incompatible_value_separations = 0
        self.total_unresolved_near_duplicates = 0

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

        # Candidate index for near-duplicate classification.  It is a pure
        # accelerator: the decision still re-reads live slot fields, so a
        # stale or missing entry can only change whether an unresolved
        # near-duplicate is *reported*, never whether a memory is merged.
        # ``None`` means "rebuild on next use"; it is invalidated by every
        # pathway that edits, evicts, redacts or restores slots.
        self._near_dup_index = None
        self._near_dup_order: Dict[int, int] = {}

        # Retrieval backend selection.  Canonical is the reference path and the
        # legacy default; production deployments opt into the strict Qdrant
        # profile, and canonical-only deployments never import or contact
        # Qdrant.  Like the predicate policy above, this is process
        # configuration and is not serialized into checkpoints.
        from minemos_engine.retrieval.config import (
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
            from minemos_engine.retrieval.qdrant_config import QdrantConfig
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
        from minemos_engine.retrieval.config import (
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
            from minemos_engine.retrieval.qdrant_config import QdrantConfig
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
            from minemos_engine.retrieval.adapter import RetrievalAdapter
            with self._lock:
                if self._retrieval_adapter_obj is None:
                    self._retrieval_adapter_obj = RetrievalAdapter(
                        self, self._retrieval_config,
                        backend=self._retrieval_backend,
                        qdrant_config=self._qdrant_config)
                adapter = self._retrieval_adapter_obj
        return adapter

    def retrieval_stats(self, probe: bool = True) -> Dict[str, Any]:
        """Report the configured backend and whether it is actually serving.

        ``state`` is ``canonical``, ``not_initialized`` (selected but not yet
        contacted), ``ready``, or ``degraded`` (the last index operation failed
        and queries are falling back to canonical).  This prevents a selected
        production backend from being mistaken for a healthy one.

        ``probe=False`` keeps the call free of backend I/O: the Qdrant point
        count is omitted rather than fetched, so status endpoints stay bounded
        while an external index is unreachable.
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
        snapshot = getattr(adapter, "stats_snapshot", None)
        stats = (snapshot(probe=probe) if callable(snapshot)
                 else adapter.stats())
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
                or manager.get("cleanup_pending")
                or stats.get("last_error"))
        stats["degraded"] = degraded
        stats["state"] = (
            "canonical" if backend == "canonical"
            else "degraded" if degraded
            else "ready" if retriever is not None
            else "not_initialized")
        stats["active"] = backend != "canonical"
        return stats

    def retrieval_readiness(self) -> Dict[str, Any]:
        """Bounded backend availability for readiness reporting.

        Canonical retrieval is the engine's own state and is always available.
        A selected external backend is *required*: queries can still fall back
        to canonical, but the deployment is not healthy if the backend cannot
        be reached, so ``available`` is False and ``state`` is ``unavailable``.
        The probe is cached for a few seconds and never waits on the network
        for longer than its own timeout.
        """
        with self._lock:
            adapter = self._retrieval_adapter_obj
            backend = self._retrieval_backend
        if backend == "canonical":
            return {
                "backend": backend,
                "required": False,
                "available": True,
                "state": "ok",
                "last_error": None,
            }
        if adapter is None:
            return {
                "backend": backend,
                "required": True,
                "available": False,
                "state": "not_initialized",
                "last_error": None,
            }
        readiness = getattr(adapter, "readiness", None)
        report = (dict(readiness()) if callable(readiness)
                  else {"available": adapter.retriever is not None,
                        "state": None, "last_error": None})
        report["backend"] = backend
        report["required"] = True
        if not report.get("available"):
            report["state"] = "unavailable"
        elif report.get("degraded") or report.get("last_error"):
            report["state"] = "degraded"
        else:
            report["state"] = "ok"
        return report

    @staticmethod
    def _validated_vector(
        value: Any, expected_dim: Optional[int], label: str
    ) -> np.ndarray:
        """Rejects malformed encoder output before it can enter memory.

        Encoders are an input boundary: a custom or corrupted encoder can
        return a wrong-rank, wrong-width, empty, or non-finite vector.  Storing
        such a row silently corrupts ranking (NaN sorts unpredictably) or
        crashes a later ``np.stack``/Qdrant upsert, so it is rejected here.
        """
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a numeric vector: {exc}") from exc
        if array.ndim != 1:
            raise ValueError(
                f"{label} must be a 1-D vector, got shape {array.shape}")
        if array.size == 0:
            raise ValueError(f"{label} must be non-empty")
        if expected_dim is not None and array.shape[0] != expected_dim:
            raise ValueError(
                f"{label} dimension {array.shape[0]} does not match the "
                f"expected dimension {expected_dim}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label} contains NaN or Inf values")
        return array

    def _expected_dense_dim(self) -> Optional[int]:
        """Dense width declared by the encoder, else the live row width.

        Encoders that expose neither ``dim`` nor ``width`` fall back to the
        architecture's dense dimension.  Structure-only test doubles can opt
        out with a ``width`` attribute matching their vectors.
        """
        for attribute in ("dim", "width"):
            declared = getattr(self.dense_encoder, attribute, None)
            if declared is not None:
                return int(declared)
        if self.slot_dense_embs:
            return int(self.slot_dense_embs[0].shape[0])
        return D_DENSE

    @staticmethod
    def _validated_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"confidence must be a number, got {value!r}") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"confidence must be finite and in [0, 1], got {value!r}")
        return confidence

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
    @functools.lru_cache(maxsize=8192)
    def _normalize_text(text: Optional[str]) -> str:
        """Case-, width-, whitespace-, and edge-punctuation-insensitive form.

        NFKC folds compatibility forms (full-width letters, ligatures) before
        casefolding, so noisy input compares against clean text without any
        model call.  The result is cached: temporal scans compare the same
        stored subjects thousands of times and normalization is pure.
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
    @functools.lru_cache(maxsize=8192)
    def _contains_turn(cls, text: Optional[str], turn_norm: str) -> bool:
        """Whether normalized ``text`` contains the current turn verbatim.

        The stored row carries its own subject/predicate prefix, so the turn
        is matched as a bounded phrase inside the normalized row rather than
        compared for equality.  The caller only supplies text it asserts is
        its own current message; a row repeating that message is the
        question, not evidence for it.
        """
        normalized = cls._normalize_text(text)
        if not normalized:
            return False
        return re.search(
            r"(?<!\w)" + re.escape(turn_norm) + r"(?!\w)",
            normalized) is not None

    def current_turn_matches(self, current_turn: str) -> int:
        """Rows whose text repeats ``current_turn`` verbatim.

        Used by the ungated retrieval profile to widen its window by exactly
        the rows it will drop, so the caller still receives ``top_k`` rows.
        """
        turn_norm = self._normalize_text(current_turn)
        if len(turn_norm) < 3:
            return 0
        with self._lock:
            return sum(
                1 for slot in self.memory.slots
                if not slot.redacted
                and self._contains_turn(
                    self.slot_texts.get(id(slot), slot.text), turn_norm))

    @classmethod
    @functools.lru_cache(maxsize=8192)
    def _comparison_tokens(cls, text: Optional[str]) -> Tuple[str, ...]:
        """Token sequence used for lexical equivalence, after alias expansion."""
        normalized = cls._normalize_text(text)
        tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        expanded: List[str] = []
        for token in tokens:
            alias = ABBREVIATION_ALIASES.get(token)
            expanded.extend(alias.split() if alias else [token])
        return tuple(expanded)

    @classmethod
    def _object_entity_core(
        cls, value: Optional[str]
    ) -> Tuple[frozenset, frozenset]:
        """The entity an object phrase names, ignoring quantity/filler words.

        "the majority share in Acme" and "all equity in the Acme Corporation"
        both reduce to ``{acme}`` with the same legal form.  The reduction is
        deliberately lexical: it never calls a model, and it keeps qualifiers
        such as "Acme North" distinct.  A misspelled filler or legal suffix
        ("Corporaton") is normalized too, so it does not hide the entity.
        Legal forms are returned separately so that two different registered
        forms are never treated as one entity.
        """
        core = []
        legal_forms = set()
        for token in cls._comparison_tokens(value):
            form = ENTITY_LEGAL_FORM_ALIASES.get(token)
            if form is None:
                for alias, alias_form in ENTITY_LEGAL_FORM_ALIASES.items():
                    if cls._tokens_are_typo_equivalent([token], [alias]):
                        form = alias_form
                        break
            if form is not None:
                legal_forms.add(form)
                continue
            if token in OBJECT_ENTITY_FILLER:
                continue
            if any(
                cls._tokens_are_typo_equivalent([token], [filler])
                for filler in OBJECT_ENTITY_FILLER
            ):
                continue
            core.append(token)
        return frozenset(core), frozenset(legal_forms)

    @classmethod
    def _entity_cores_match(
        cls, core_a: frozenset, forms_a: frozenset,
        core_b: frozenset, forms_b: frozenset,
    ) -> bool:
        """Equal entity cores, allowing one typo in a single-token name.

        A legal form present on only one side is compatible ("Acme" /
        "the Acme Corporation").  Two different legal forms are not: a
        disposal of "Acme Inc" says nothing about "Acme Ltd".
        """
        if not core_a or not core_b:
            return False
        if forms_a and forms_b and forms_a != forms_b:
            return False
        if core_a == core_b:
            return True
        if len(core_a) != len(core_b):
            return False
        return cls._tokens_are_typo_equivalent(
            sorted(core_a), sorted(core_b))

    @staticmethod
    def _tokens_are_derivationally_related(a: str, b: str) -> bool:
        """Whether two tokens are inflections/derivations of one stem.

        "biography" / "biographical" and "physics" / "physical" differ by a
        known suffix on an unchanged stem.  The test is deliberately narrow:
        it never relates distinct words that merely share a prefix such as
        "literature" / "literacy" or "quantum" / "data".  A plural is a pure
        inflection and is accepted separately.
        """
        if not a or not b or a == b:
            return False
        if (len(a) >= 2 and len(b) >= 2
                and (a + "s" == b or b + "s" == a)):
            # An English plural is a surface variant of its singular, even
            # when the stem is too short for the suffix rule below
            # ("Lab" / "Labs").
            return True
        longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
        if len(shorter) < 5:
            return False
        for suffix in ("ical", "ally", "ism", "ist", "ic", "al", "ous",
                       "ive", "ary"):
            if not longer.endswith(suffix) or len(longer) - len(suffix) < 5:
                continue
            stem = longer[:-len(suffix)]
            if not shorter.startswith(stem):
                continue
            if shorter[len(stem):] in ("", "e", "y", "s"):
                return True
        return False

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
    def _lexically_equivalent_value(
        cls, a: Optional[str], b: Optional[str], strict: bool = False
    ) -> bool:
        tokens_a = cls._comparison_tokens(a)
        tokens_b = cls._comparison_tokens(b)
        if not tokens_a or not tokens_b:
            return False
        if tokens_a == tokens_b:
            return True
        # A one-edit difference inside a multi-token name is structurally
        # indistinguishable from a typo of that name ("Fairbank" vs
        # "Fairbanks").  Strict identity refuses the guess; a single-token
        # typo or an unambiguous abbreviation is still conflated.
        if strict and (len(tokens_a) > 1 or len(tokens_b) > 1):
            return False
        return cls._tokens_are_typo_equivalent(tokens_a, tokens_b)

    @classmethod
    def _subjects_are_equivalent(
        cls, a: Optional[str], b: Optional[str], strict: bool = False
    ) -> bool:
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
        if tokens_a == tokens_b:
            return True
        if strict and (len(tokens_a) > 1 or len(tokens_b) > 1):
            return False
        return cls._tokens_are_typo_equivalent(tokens_a, tokens_b)

    @classmethod
    def _values_are_near_miss(cls, a: Optional[str], b: Optional[str]) -> bool:
        """One token differs by a single edit, including a substitution.

        Used only to refuse an identity judgement: when two values are this
        close but not equivalent, the engine cannot tell a misspelling from a
        genuinely different name, so it keeps both facts rather than letting
        NLI silently pick a winner.
        """
        tokens_a = cls._comparison_tokens(a)
        tokens_b = cls._comparison_tokens(b)
        if not tokens_a or not tokens_b or len(tokens_a) != len(tokens_b):
            return False
        diffs = [(x, y) for x, y in zip(tokens_a, tokens_b) if x != y]
        if len(diffs) != 1:
            return False
        x, y = diffs[0]
        if not (x.isalpha() and y.isalpha()):
            return False
        if min(len(x), len(y)) < 4 or x[0] != y[0]:
            return False
        return cls._single_edit_kind(x, y) is not None

    @staticmethod
    def _has_historical_marker(text: Optional[str]) -> bool:
        lowered = (text or "").casefold()
        return any(_contains_marker(lowered, marker) for marker in HISTORICAL_STATEMENT_MARKERS)

    @staticmethod
    def _has_negation(text: Optional[str]) -> bool:
        from minemos_engine.intelligence.verification_terms import extract_negation_markers
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

    @staticmethod
    def _is_state_predicate(predicate: Optional[str]) -> bool:
        """A copular/state predicate whose value is time-indexed.

        ``"Saturn project phase is X"`` asserts one current phase; a different
        phase supersedes it.  Instrument predicates such as ``"uses"`` are not
        state predicates: numbered tools name independent resources even
        though neither predicate has an explicit cardinality policy.
        """
        pred = _normalise_state_predicate(predicate)
        if not pred:
            return False
        if pred in {"is", "are", "was", "were"}:
            return True
        if pred.endswith((" is", " are", " was", " were")):
            return True
        parts = pred.split()
        if len(parts) == 2 and parts[0] in {"is", "are", "was", "were"}:
            # "is aged" / "is worth": an adjective complement names one
            # scalar attribute, so a differing number is a state change.
            # Prepositional complements ("is compatible with") are relations
            # that may legitimately hold several values and stay excluded.
            return True
        if _state_relation_key(pred) is not None:
            return True
        return any(
            _contains_marker(pred, marker)
            for marker in STATE_MARKER_PREDICATES
        )

    def _state_attribute_key(self, predicate: Optional[str]) -> Optional[str]:
        """The state attribute a predicate asserts, or ``None``.

        This is a *detection* key, never a cardinality policy.  It groups only
        predicate forms that assert one current state of the subject:

        * a bare copula ("is", "was", ...) is one state attribute;
        * a noun phrase ending in a copula ("goal is", "website is") is the
          attribute named by the phrase, so "goal is" and "website is" never
          collide and neither is assumed to share a bare ``is``;
        * a marker noun with optional article or neutral modifier ("phase",
          "phase is", "current phase is", "the goal is") is the marker;
        * an explicit role-state verb ("works as", "serves as") is the role
          attribute;
        * a known state relation ("works at", "is based in", "lives in",
          "is married to") maps to its relation family for surface variants.
          The relation key is returned even when the surface predicate also
          belongs to a configured exclusive group ("lives in", "works at").
          The two detections stay separate: the exclusive path runs first and
          reports its own reason, while the relation key only lets a known
          form pair with an unconfigured variant of the same relation
          ("works at" / "works for").  Returning the key is not a policy: it
          never turns an unknown predicate into an exclusive one.

        Predicates with an explicit policy and no recognized state relation
        are excluded: exclusive attributes are already reported by
        :meth:`exclusive_temporal_conflicts`, and a multi-valued relation is
        legitimate coexistence, not a conflict.  Unknown predicates that are
        not a state form return ``None``; this function never infers
        exclusivity from the presence of "is".
        """
        pred = _normalise_state_predicate(predicate)
        if not pred:
            return None
        relation = _state_relation_key(pred)
        if relation is not None:
            return relation
        if self.classify_predicate(pred) != PREDICATE_POLICY_UNKNOWN:
            return None
        if pred in {"is", "are", "was", "were"}:
            return "__copula__"
        for suffix in (" is", " are", " was", " were"):
            if pred.endswith(suffix):
                head = _strip_leading_article(pred[:-len(suffix)].strip())
                if not head:
                    return "__copula__"
                return _state_marker_key(head) or head
        if pred in ROLE_STATE_PREDICATES:
            return "role"
        if any(
            _contains_marker(pred, marker)
            for marker in STATE_MARKER_PREDICATES
        ):
            return _state_marker_key(pred) or pred
        return None

    def _slots_share_state_attribute(
        self, pred_a: Optional[str], pred_b: Optional[str]
    ) -> bool:
        key_a = self._state_attribute_key(pred_a)
        key_b = self._state_attribute_key(pred_b)
        return key_a is not None and key_a == key_b

    @staticmethod
    def _represented_predicate(slot: MemorySlot) -> str:
        """The relation a slot currently represents, not its diagnostic label."""
        return (slot.latest_predicate or slot.debug_predicate or "").strip().lower()

    @staticmethod
    def _directional_entailments(pred: dict) -> Optional[tuple]:
        """Per-direction entailment probabilities, when the verifier reports them.

        ``CalibratedSemanticVerifier`` nests ``fwd``/``rev``; the calibrated
        verifier reports flat ``p_entail_fwd``/``p_entail_rev``.  A verifier
        that only returns a decision has no directional evidence and is
        trusted as declared.
        """
        fwd = pred.get("fwd")
        rev = pred.get("rev")
        if isinstance(fwd, dict) and isinstance(rev, dict):
            a = fwd.get("p_entail")
            b = rev.get("p_entail")
            if a is not None and b is not None:
                return float(a), float(b)
        a = pred.get("p_entail_fwd")
        b = pred.get("p_entail_rev")
        if a is not None and b is not None:
            return float(a), float(b)
        return None

    def _mutually_supported(self, pred: dict) -> bool:
        """Reject consolidation supported in only one entailment direction."""
        directional = self._directional_entailments(pred)
        if directional is None:
            return True
        return min(directional) >= MUTUAL_ENTAILMENT_FLOOR

    @staticmethod
    def _relation_families(predicate: Optional[str]) -> set:
        pred = (predicate or "").strip().casefold()
        if not pred:
            return set()
        return {
            idx for idx, family in enumerate(RELATION_FAMILY_GROUPS)
            if any(_contains_marker(pred, marker) for marker in family)
        }

    def _predicates_relation_compatible(
        self, pred_a: Optional[str], pred_b: Optional[str]
    ) -> bool:
        """Whether two different predicates provably denote the same relation.

        Same/exclusive-attribute predicates are handled before this check.
        Two explicitly classified families that do not intersect are
        incompatible.  A predicate with no family is unclassified, and an
        unclassified relation is not evidence that two facts may be merged:
        mutual semantic support alone must not authorize a destructive
        cross-predicate merge.  The only inheritance is a bare copula taking
        the role family from an explicit role predicate, because there the
        relation is carried by the object rather than by the verb.
        """
        a = self._relation_families(pred_a)
        b = self._relation_families(pred_b)
        if a & b:
            return True
        # A bare copula has no content relation of its own; the relation is
        # the object.  It may inherit the role family from an explicit role
        # verb ("is the chief architect" / "serves as lead architect"), and
        # from no other family.
        pred_a_norm = (pred_a or "").strip().casefold()
        pred_b_norm = (pred_b or "").strip().casefold()
        if pred_a_norm in COPULA_PREDICATES:
            return ROLE_FAMILY_INDEX in b
        if pred_b_norm in COPULA_PREDICATES:
            return ROLE_FAMILY_INDEX in a
        return False

    def _is_exclusive_state_change(
        self, slot: MemorySlot, candidate: Candidate
    ) -> bool:
        """A different value of a configured exclusive attribute."""
        if self._normalize_subject(slot.debug_subject) != self._normalize_subject(
                candidate.subject):
            return False
        return self._predicates_share_exclusive_attribute(
            self._represented_predicate(slot), candidate.predicate)

    def _is_explicit_correction(
        self, slot: MemorySlot, candidate: Candidate
    ) -> bool:
        """Same-proposition negation or an explicit state-transition marker."""
        candidate_text = (
            candidate.object if not candidate.predicate
            else f"{candidate.subject} {candidate.predicate} {candidate.object}."
        )
        lowered = candidate_text.lower()
        if (any(_contains_marker(lowered, marker)
                for marker in TEMPORAL_TRANSITION_MARKERS)
                and self._predicates_share_exclusive_attribute(
                    self._represented_predicate(slot), candidate.predicate)):
            return True
        from minemos_engine.intelligence.verification_terms import extract_negation_markers
        slot_text = self.slot_texts.get(id(slot), slot.text or "")
        if bool(extract_negation_markers(slot_text)) == bool(
                extract_negation_markers(candidate_text)):
            return False
        slot_object = slot.latest_object
        if slot_object is None:
            slot_object = self._slot_object_text(slot, slot_text)
        if not slot_object or not candidate.object:
            return False
        candidate_tokens = self._content_tokens(candidate.object)
        slot_tokens = self._content_tokens(slot_object)
        return bool(candidate_tokens) and candidate_tokens <= slot_tokens

    def _slot_belongs_to_ownership(self, slot: MemorySlot) -> bool:
        ownership = self._relation_families("owns")
        families: set = set()
        for predicate in self._slot_predicates(slot):
            families |= self._relation_families(predicate)
        return bool(families & ownership)

    @staticmethod
    def _strip_recipient(value: Optional[str]) -> str:
        """The asset phrase of an ownership statement, without the recipient.

        "sold all equity in Acme to Bob" disposes of Acme; Bob is the
        recipient, not a second entity.  The recipient clause starts at a
        transfer preposition that follows the asset ("to", "from", "into",
        "onto", "for").  "in" is deliberately not a split point because it
        introduces the asset itself ("equity in Acme").
        """
        text = (value or "").strip().rstrip(".")
        if not text:
            return ""
        match = re.search(
            r"\s+(?:to|from|into|onto|for)\s+", text, flags=re.IGNORECASE)
        if match is None:
            return text
        return text[:match.start()].strip()

    @staticmethod
    def _disposal_scope(predicate: Optional[str],
                        object_text: Optional[str]) -> str:
        """Whether a disposal removed all ownership or only part of it.

        ``full`` requires positive evidence (an all/majority qualifier or an
        object that names the asset itself).  ``partial`` is an explicit
        quantity words.  Selling an unquantified share/equity/stake is
        ``unknown``: it is change evidence, not proof that ownership ended.
        """
        text = f"{predicate or ''} {object_text or ''}".casefold()
        tokens = set(re.findall(r"[^\W_]+", text))
        if tokens & PARTIAL_DISPOSAL_QUALIFIERS:
            return "partial"
        if tokens & FULL_DISPOSAL_QUALIFIERS:
            return "full"
        if tokens & QUANTITY_OBJECT_WORDS:
            return "unknown"
        return "full"

    def _ownership_transition_kind(
        self, old_slot: MemorySlot, new_predicate: Optional[str],
        new_object: Optional[str]
    ) -> Optional[str]:
        """Classify a later ownership statement relative to an older slot.

        ``disposal`` is a full disposal of the same entity: the older
        ownership state is corrected.  ``unspecified`` is an ownership
        transition verb ("transferred", "lost") or a disposal whose scope is
        unknown: the engine cannot prove that ownership ended, so both
        observations stay active and the response reports an unresolved
        change.  ``None`` means no ownership transition is established.

        The check requires the older slot to be an ownership relation, the
        same subject and the same entity after quantity words, a recipient
        clause and legal forms are removed, and no negation.  A recipient or
        a similar organisation name is never identity proof.
        """
        new_pred = (new_predicate or "").strip().casefold()
        if not new_pred:
            return None
        if self._has_negation(new_pred) or self._has_negation(new_object or ""):
            return None
        if not self._slot_belongs_to_ownership(old_slot):
            return None
        old_value = self._slot_value_text(old_slot)
        asset = self._strip_recipient(new_object)
        if not old_value or not asset:
            return None
        old_core, old_forms = self._object_entity_core(old_value)
        new_core, new_forms = self._object_entity_core(asset)
        if not self._entity_cores_match(
                old_core, old_forms, new_core, new_forms):
            return None
        if any(_contains_marker(new_pred, verb)
               for verb in OWNERSHIP_TRANSITION_PREDICATES):
            return "unspecified"
        if not any(_contains_marker(new_pred, verb)
                   for verb in DISPOSAL_PREDICATES):
            return None
        scope = self._disposal_scope(new_pred, new_object)
        if scope == "partial":
            return None
        if scope == "unknown":
            return "unspecified"
        return "disposal"

    def _disposal_transition_matches(
        self, old_slot: MemorySlot, new_predicate: Optional[str],
        new_object: Optional[str]
    ) -> bool:
        """A full disposal of the same entity corrects an ownership state.

        This is intentionally narrow.  It requires positive evidence on four
        independent axes: the older slot is an ownership relation, the newer
        predicate is an explicit disposal verb of full scope, both statements
        name the same subject and the same entity after quantity words and
        recipient clauses are removed, and the newer statement is not negated.
        Anything less keeps both facts and is reported as unresolved by A3
        rather than merged.
        """
        return self._ownership_transition_kind(
            old_slot, new_predicate, new_object) == "disposal"

    def _is_disposal_correction(
        self, slot: MemorySlot, candidate: Candidate
    ) -> bool:
        if self._normalize_subject(slot.debug_subject) != self._normalize_subject(
                candidate.subject):
            return False
        return self._disposal_transition_matches(
            slot, candidate.predicate, candidate.object)

    def _same_exclusive_attribute_as(
        self, other: MemorySlot, slot: MemorySlot
    ) -> bool:
        if self._normalize_subject(other.debug_subject) != self._normalize_subject(
                slot.debug_subject):
            return False
        return self._predicates_share_exclusive_attribute(
            self._represented_predicate(other), self._represented_predicate(slot))

    def _interval_containing(
        self, slot: MemorySlot, timestamp: float
    ) -> Optional[MemorySlot]:
        """An interval of the same exclusive attribute containing ``timestamp``."""
        for candidate in self.memory.slots:
            if candidate is slot or candidate.redacted:
                continue
            if candidate.valid_from is None:
                continue
            if not self._same_exclusive_attribute_as(candidate, slot):
                continue
            end = (candidate.valid_until if candidate.valid_until is not None
                   else float("inf"))
            if candidate.valid_from <= timestamp < end:
                return candidate
        return None

    def _interval_value_matches(
        self, interval: Optional[MemorySlot], candidate: Candidate
    ) -> bool:
        """Whether a containing interval itself asserts the candidate's value.

        Equivalence judged against a successor slot says nothing about an
        earlier interval of the same attribute.  Merging a backdated
        observation into that interval rewrites its represented value, so the
        interval's own value must also be equivalent.
        """
        if interval is None or not candidate.object:
            return False
        value = self._slot_value_text(interval)
        if not value:
            return False
        return self._values_are_equivalent(value, candidate.object)

    def _can_extend_valid_from(self, slot: MemorySlot, timestamp: float) -> bool:
        """Whether a backdated equivalent observation can open the interval.

        Refuses when another interval of the same exclusive attribute already
        covers part of ``[timestamp, slot.valid_from)``; extending then would
        span a recorded state.
        """
        if slot.valid_from is None or timestamp >= slot.valid_from:
            return True
        for other in self.memory.slots:
            if other is slot or other.redacted or other.valid_from is None:
                continue
            if not self._same_exclusive_attribute_as(other, slot):
                continue
            end = (other.valid_until if other.valid_until is not None
                   else float("inf"))
            if other.valid_from < slot.valid_from and end > timestamp:
                return False
        return True

    def _backdated_exclusive_route(
        self, candidate: Candidate, timestamp: float
    ) -> Tuple[Optional[MemorySlot], Optional[MemorySlot]]:
        """Interval-aware routing for a backdated exclusive observation.

        Returns ``(equivalent, successor)``:

        * ``equivalent`` is an interval that already covers ``timestamp``
          and asserts the candidate's value; the observation consolidates
          into it.
        * ``successor`` is the interval whose start bounds the new segment
          when the candidate asserts a different value inside a closed
          interval, in a gap, or before recorded history.  Without it, the
          observation is not backdated and normal insertion applies.

        The verifier's decision is deliberately not consulted here: a SAME
        verdict against a closed predecessor, a CONTRADICTION and a NEUTRAL
        observation all describe the same timeline.
        """
        chain = [
            slot for slot in self._attribute_chain(candidate)
            if slot.valid_from is not None
        ]
        if not chain:
            return None, None

        containing = None
        for slot in chain:
            end = math.inf if slot.valid_until is None else slot.valid_until
            if slot.valid_from <= timestamp < end:
                containing = slot
                break
        if containing is not None and self._interval_value_matches(
                containing, candidate):
            return containing, None
        if containing is not None and containing.is_active:
            # The open active interval already covers the timestamp.  A
            # differing value here is a forward observation, not a backdated
            # one: unverified evidence stays conservatively retained and the
            # same verifier-specific paths as before apply.
            return None, None

        later = [slot for slot in chain if slot.valid_from > timestamp]
        if later:
            return None, min(later, key=lambda slot: slot.valid_from)
        return None, containing

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
        if not self._subjects_are_equivalent(
                slot.debug_subject, candidate.subject,
                strict=self.strict_entity_identity):
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
        if not self._lexically_equivalent_value(
                slot_value, candidate.object,
                strict=self.strict_entity_identity):
            return False
        return self._has_negation(slot_text) == self._has_negation(cand_text)

    def _find_lexically_equivalent_slot(self, candidate: Candidate) -> Optional[int]:
        for idx, slot in enumerate(self.memory.slots):
            if slot.redacted:
                continue
            if slot.is_active and self._lexically_same_fact(slot, candidate):
                return idx
        return None

    @classmethod
    def _single_token_edit_kind(
        cls, value_a: Optional[str], value_b: Optional[str]
    ) -> Optional[str]:
        """Edit kind when two field values differ by one single-character edit.

        The values are compared as token sequences after alias expansion; only
        one token may differ, and that difference must be a substitution,
        transposition or indel.  Returns ``None`` when the values are equal or
        differ by more than that.
        """
        tokens_a = cls._comparison_tokens(value_a)
        tokens_b = cls._comparison_tokens(value_b)
        if not tokens_a or not tokens_b or tokens_a == tokens_b:
            return None
        if len(tokens_a) != len(tokens_b):
            return None
        diffs = [(x, y) for x, y in zip(tokens_a, tokens_b) if x != y]
        if len(diffs) != 1:
            return None
        return cls._single_edit_kind(*diffs[0])

    @classmethod
    def _surface_variant_kind(
        cls, value_a: Optional[str], value_b: Optional[str]
    ) -> Optional[str]:
        """Surface edit kind between two field values, or ``None``.

        Covers the single-character edits of ``_single_token_edit_kind`` plus
        one whole token appended, dropped or replaced by an additional token.
        Like the single-character case this is evidence of a possible
        restatement only: "Lisbon" / "Lisbon, Portugal" and "Foxglove" /
        "Foxglove Publishing" share this shape, but so do distinct facts.
        """
        single = cls._single_token_edit_kind(value_a, value_b)
        if single is not None:
            return single
        tokens_a = cls._comparison_tokens(value_a)
        tokens_b = cls._comparison_tokens(value_b)
        if not tokens_a or not tokens_b or tokens_a == tokens_b:
            return None
        shorter, longer = sorted((tokens_a, tokens_b), key=len)
        if len(longer) != len(shorter) + 1:
            return None
        for index in range(len(longer)):
            if longer[:index] + longer[index + 1:] == shorter:
                # Relative to the candidate: extra slot tokens mean the
                # candidate omitted a word.
                return "token_deletion" if longer is tokens_a else "token_insertion"
        return None

    def _near_duplicate_field_keys(
        self, slot: MemorySlot
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Index keys for the fields a near-duplicate must share exactly."""
        values = (
            ("subject", slot.debug_subject),
            ("predicate", slot.latest_predicate or slot.debug_predicate),
            ("object", self._slot_value_text(slot)),
        )
        return tuple(
            (name, tuple(self._comparison_tokens(value)))
            for name, value in values
        )

    def _invalidate_near_duplicate_index(self) -> None:
        self._near_dup_index = None
        self._near_dup_order = {}

    def _near_duplicate_index_add(self, slot: MemorySlot) -> None:
        """Extend the candidate index after a pure append.

        No-op when the index has not been built yet (or was invalidated): the
        next lookup rebuilds from live slots, which already includes this one.
        """
        if self._near_dup_index is None:
            return
        if not slot.is_active or slot.redacted:
            return
        self._near_dup_order[id(slot)] = len(self.memory.slots) - 1
        for key in self._near_duplicate_field_keys(slot):
            if key[1]:
                self._near_dup_index.setdefault(key, []).append(slot)

    def _build_near_duplicate_index(self) -> None:
        buckets: Dict[Tuple[str, Tuple[str, ...]], List[MemorySlot]] = {}
        order: Dict[int, int] = {}
        for idx, slot in enumerate(self.memory.slots):
            if not slot.is_active or slot.redacted:
                continue
            order[id(slot)] = idx
            for key in self._near_duplicate_field_keys(slot):
                if key[1]:
                    buckets.setdefault(key, []).append(slot)
        self._near_dup_index = buckets
        self._near_dup_order = order

    def _find_near_duplicate_slot(
        self, candidate: Candidate
    ) -> Optional[Tuple[MemorySlot, str, str]]:
        """An active slot that is one surface edit away from the candidate.

        Two of the three fields (subject, predicate, object) must be
        token-equal and the third must differ by a surface edit: one
        single-character edit or one whole token added or removed.
        This is *evidence of possible repetition only*.  The same surface
        shape is produced by distinct entities ("Mira Ashford" / "Mira
        Ahsford"), by distinct facts ("Northwind Labs" / "Northwind Lab") and
        by genuine relation or temporal revisions ("studies" / "studied"), so
        the caller must never consolidate on this signal alone.  It exists so
        the engine can report an unresolved near-duplicate while preserving
        both memories.

        Candidates come from three field-postings lists; a slot can only
        match if it appears in at least two of them.  The decision itself is
        still made against the slot's live fields, so this index affects
        coverage only, not consolidation.
        """
        if self._near_dup_index is None:
            self._build_near_duplicate_index()
        index = self._near_dup_index or {}
        keys = tuple(
            (name, tuple(self._comparison_tokens(value)))
            for name, value in (
                ("subject", candidate.subject),
                ("predicate", candidate.predicate),
                ("object", candidate.object),
            )
            if tuple(self._comparison_tokens(value))
        )
        counts: Dict[int, int] = {}
        slots: Dict[int, MemorySlot] = {}
        for key in keys:
            for slot in index.get(key, ()):
                slot_id = id(slot)
                counts[slot_id] = counts.get(slot_id, 0) + 1
                slots[slot_id] = slot

        ordered = sorted(
            (slot for slot_id, slot in slots.items() if counts[slot_id] >= 2),
            key=lambda slot: self._near_dup_order.get(id(slot), len(
                self.memory.slots)),
        )
        for slot in ordered:
            if not slot.is_active or slot.redacted:
                continue
            slot_subject = slot.debug_subject
            slot_predicate = slot.latest_predicate or slot.debug_predicate
            slot_object = self._slot_value_text(slot)
            fields = (
                ("subject", slot_subject, candidate.subject),
                ("predicate", slot_predicate, candidate.predicate),
                ("object", slot_object, candidate.object),
            )
            equal = 0
            edited: Optional[Tuple[str, str]] = None
            for name, value_a, value_b in fields:
                tokens_a = self._comparison_tokens(value_a)
                tokens_b = self._comparison_tokens(value_b)
                if tokens_a and tokens_a == tokens_b:
                    equal += 1
                    continue
                kind = self._surface_variant_kind(value_a, value_b)
                if kind is None or edited is not None:
                    edited = None
                    break
                edited = (name, kind)
            if equal == 2 and edited is not None:
                return slot, edited[0], edited[1]
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
        are compared on their own.  A pair the lexical guard can prove
        distinct stays apart even when the verifier calls it SAME.  An
        unequal pair the guard cannot separate still needs value-level SAME
        with mutual entailment support, so ordinary rewording keeps
        consolidating.  Strict identity never merges an unequal value on a
        model's opinion.
        """
        if self._normalize_value(value_a) == self._normalize_value(value_b):
            return True
        from minemos_engine.intelligence.verification_terms import extract_numbers
        numbers_a = extract_numbers(value_a)
        numbers_b = extract_numbers(value_b)
        if numbers_a and numbers_b and numbers_a != numbers_b:
            return False
        # A typo or an unambiguous abbreviation is the same value; deciding
        # that lexically avoids spending a verifier call and avoids the
        # verifier's noisy behaviour on near-identical strings.
        if self._lexically_equivalent_value(
                value_a, value_b, strict=self.strict_entity_identity):
            return True
        distinct = self._values_are_provably_distinct(value_a, value_b)
        if self.strict_entity_identity:
            # The whole-statement NLI gate is known to call distinct names
            # equivalent ("the Roman Republic" / "the Roman Empire").  In
            # strict mode an unequal value is never merged on NLI alone; the
            # two facts remain separately retrievable.
            if distinct:
                self.total_incompatible_value_separations += 1
            self.total_suppressed_value_merges += 1
            return False
        # Ask the configured verifier even for a provably distinct pair: the
        # call is telemetry for how often value-level semantics disagree with
        # the lexical guard, and its answer is never allowed to merge them.
        self.total_value_verifications += 1
        result = self.verifier.verify_batch(
            [(value_a, value_b)],
            mode=self.verification_mode,
            entail_threshold=self.entail_threshold,
            contra_threshold=self.contra_threshold,
        )
        if not result:
            return False
        equivalent = (
            result[0].get("decision") == "SAME"
            and self._mutually_supported(result[0])
        )
        if not equivalent:
            return False
        if distinct:
            self.total_incompatible_value_separations += 1
            return False
        return True

    def _values_compatible_for_merge(
        self, value_a: Optional[str], value_b: Optional[str]
    ) -> bool:
        """Whether a verifier SAME may consolidate two represented values.

        A whole-statement SAME is not evidence that two *values* agree.  A
        destructive cross-predicate merge of "plays guitar" and "performs
        piano" would erase a distinct fact even when the relation is proven
        identical.  This guard therefore rejects only values it can prove
        different: different numbers, different registered legal forms, a
        bare name against a different value, proper names sharing a core but
        naming different entities, a near-miss name pair, or the same generic
        head noun with a different modifier.  A shared generic head noun
        alone is not enough ("the quantum team" / "the data team" both
        contain "team" yet name different objects).  When no lexical evidence
        separates the values, the verifier decides; a SAME alone still never
        overrides a value this guard can prove distinct.
        """
        if not value_a or not value_b:
            return False
        return not self._values_are_provably_distinct(value_a, value_b)

    def _values_are_provably_distinct(
        self, value_a: str, value_b: str
    ) -> bool:
        """Whether two values name different objects on lexical evidence.

        The checks are high-precision and conservative.  They must never fire
        on an ordinary paraphrase ("orchestral ensemble" / "symphony
        orchestra", "leads ... operations" / "directs ... operations"), yet
        they must catch the adversarial near-misses a semantic verifier
        conflates.  Anything they cannot prove distinct is left to the
        verifier.
        """
        if self._normalize_value(value_a) == self._normalize_value(value_b):
            return False
        if self._lexically_equivalent_value(value_a, value_b):
            return False
        from minemos_engine.intelligence.verification_terms import extract_numbers
        numbers_a = extract_numbers(value_a)
        numbers_b = extract_numbers(value_b)
        if numbers_a and numbers_b and numbers_a != numbers_b:
            return True
        core_a, forms_a = self._object_entity_core(value_a)
        core_b, forms_b = self._object_entity_core(value_b)
        if forms_a and forms_b and forms_a != forms_b:
            # "Acme Inc" and "Acme Ltd" are different registered entities.
            return True
        if self._entity_cores_match(core_a, forms_a, core_b, forms_b):
            return False
        tokens_a = self._value_content_tokens(value_a)
        tokens_b = self._value_content_tokens(value_b)
        if not tokens_a or not tokens_b:
            return False
        unique_a = set(tokens_a) - set(tokens_b)
        unique_b = set(tokens_b) - set(tokens_a)
        if not unique_a or not unique_b:
            # One value only adds detail ("Paris" / "Paris, France").
            return False
        if any(
            self._tokens_are_derivationally_related(a, b)
            for a in unique_a for b in unique_b
        ):
            return False
        if len(tokens_a) == 1 or len(tokens_b) == 1:
            # A bare value ("guitar", "Boston") against a differently named
            # value is not a paraphrase.
            return True
        if self._shares_only_a_prefix(unique_a, unique_b):
            # "motorscooter" / "motorcycle": close names, different objects.
            return True
        if self._capitalized_name_conflict(
                value_a, value_b, tokens_a, tokens_b):
            return True
        return self._same_head_noun_conflict(tokens_a, tokens_b)

    @classmethod
    def _value_content_tokens(cls, value: str) -> List[str]:
        return [
            token for token in cls._comparison_tokens(value)
            if token not in VALUE_STOP_TOKENS and len(token) >= 2
        ]

    @staticmethod
    def _shares_only_a_prefix(unique_a: set, unique_b: set) -> bool:
        for a in unique_a:
            for b in unique_b:
                if len(a) < 5 or len(b) < 5 or a[0] != b[0]:
                    continue
                prefix = 0
                while (prefix < min(len(a), len(b))
                       and a[prefix] == b[prefix]):
                    prefix += 1
                if 4 <= prefix < min(len(a), len(b)):
                    return True
        return False

    @staticmethod
    def _capitalized_name_conflict(
        value_a: str, value_b: str, tokens_a: List[str], tokens_b: List[str]
    ) -> bool:
        """Proper names that share a core but are not one another's detail."""
        caps_a = {
            word.casefold() for word in re.findall(r"[^\W_]+", value_a)
            if word[:1].isupper()
        } - VALUE_STOP_TOKENS
        caps_b = {
            word.casefold() for word in re.findall(r"[^\W_]+", value_b)
            if word[:1].isupper()
        } - VALUE_STOP_TOKENS
        if not caps_a or not caps_b:
            return False
        if caps_a & caps_b and not (
                caps_a <= caps_b or caps_b <= caps_a):
            # "the Roman Empire" / "the Roman Republic".
            return True
        # Every content word capitalized reads as a proper name; two such
        # names that are not a detail form of one another are different
        # entities ("New York" / "Los Angeles").
        return (len(caps_a) == len(tokens_a)
                and len(caps_b) == len(tokens_b))

    @classmethod
    def _same_head_noun_conflict(
        cls, tokens_a: List[str], tokens_b: List[str]
    ) -> bool:
        """Same generic head noun with a different, non-verbal modifier."""
        if len(tokens_a) != len(tokens_b) or len(tokens_a) < 2:
            return False
        if tokens_a[-1] != tokens_b[-1]:
            return False
        differences = [
            (a, b) for a, b in zip(tokens_a, tokens_b) if a != b
        ]
        if len(differences) != 1:
            return False
        a, b = differences[0]
        if cls._tokens_are_derivationally_related(a, b):
            return False
        # "leads cybersecurity operations" / "directs cybersecurity
        # operations" differ in the verb; the object is the same.  A
        # non-verbal modifier ("quantum" / "data" team) changes the object.
        return not (cls._looks_verb_like(a) and cls._looks_verb_like(b))

    @staticmethod
    def _looks_verb_like(token: str) -> bool:
        if len(token) >= 5 and token.endswith("ing"):
            return True
        if len(token) >= 4 and token.endswith("ed"):
            return True
        return (len(token) >= 4 and token.endswith("s")
                and not token.endswith(("ss", "us", "is", "ous", "ics")))

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
        """Slots describing the same state-bearing attribute as the candidate.

        Only an exclusive attribute or an explicitly corrected proposition
        forms a temporal chain.  A predicate the policy classifies as unknown
        or multi-valued does not: sharing a surface verb and subject is not
        evidence that two values are competing states.
        """
        cand_subject = self._normalize_subject(candidate.subject)
        cand_pred = (candidate.predicate or "").strip().lower()
        chain = []
        for slot in self.memory.slots:
            if slot.redacted:
                continue
            if self._normalize_subject(slot.debug_subject) != cand_subject:
                continue
            represented = self._represented_predicate(slot)
            if self._predicates_share_exclusive_attribute(represented, cand_pred):
                chain.append(slot)
            elif represented == cand_pred and self._is_explicit_correction(
                    slot, candidate):
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
        from minemos_engine.intelligence.verification_terms import extract_negation_markers
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
        slot_pred = self._represented_predicate(slot)
        cand_pred = (candidate.predicate or "").strip().lower()

        from minemos_engine.intelligence.verification_terms import extract_negation_markers, extract_numbers
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
            # Negating a change is not negating the state.  "Ada did not sell
            # Acme" is compatible with "Ada owns Acme": the negated verb
            # disposes of the owned asset, so it is not evidence about the
            # ownership proposition and must not supersede it.  A negated
            # ownership predicate ("does not own") is not a transition verb
            # and still supersedes the ownership state.
            negated_predicate = c_clean if neg_cand else s_clean
            if (self._slot_belongs_to_ownership(slot)
                    and any(_contains_marker(negated_predicate, verb)
                            for verb in (DISPOSAL_PREDICATES
                                         | OWNERSHIP_TRANSITION_PREDICATES))):
                return False
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

        # 2. Explicit state transition marker.  A marker can correct a state
        # that the predicate policy declares exclusive.  It can also correct
        # a statement about the same object ("owns Apex" -> "sold Apex now"):
        # that is a change of state for one value, not a cumulative addition.
        # Different values under a multi-valued or unknown predicate ("likes
        # tea" -> "likes coffee now", "uses tool 1" -> "uses tool 2 now") add
        # a value and must not supersede the prior one.
        cand_text_l = cand_text.lower()
        if (any(_contains_marker(cand_text_l, marker)
                for marker in TEMPORAL_TRANSITION_MARKERS)
                and dense_sim >= 0.65):
            if self._predicates_share_exclusive_attribute(slot_pred, cand_pred):
                return True
            # Sharing an object is not evidence that a transition changed this
            # relation: "works at Acme" / "plays Acme" must not be replaced by
            # "sold all equity in Acme".  Only a disposal of the same entity
            # inside an ownership relation is a provable same-object
            # correction; everything else keeps both facts.
            if self._is_disposal_correction(slot, candidate):
                return True

        # 3. Exclusive attribute groups.  Predicate identity alone is not
        # sufficient: NLI often marks independent facts sharing a verb
        # (e.g. "leads team A" vs "leads team B") as contradictions.  This
        # uses the configured policy, so a caller-declared exclusive
        # predicate supersedes on contradiction like a built-in group.
        if self._predicates_share_exclusive_attribute(slot_pred, cand_pred):
            return True

        # 4. Antonyms
        for a1, a2 in ANTONYMS:
            if (a1 in slot_text.lower() and a2 in cand_text.lower()) or \
               (a2 in slot_text.lower() and a1 in cand_text.lower()):
                return True

        # 5. Differing quantities on the same *exclusive* attribute.  An
        # unknown predicate with numbers is not an exclusive state: two
        # numbered tools under ``uses`` are independent facts.  A copular
        # state predicate is time-indexed even without an explicit policy.
        if (slot_pred and cand_pred and slot_pred == cand_pred
                and (self.classify_predicate(slot_pred)
                     == PREDICATE_POLICY_EXCLUSIVE
                     or self._is_state_predicate(slot_pred))):
            nums_s = extract_numbers(slot_text)
            nums_c = extract_numbers(cand_text)
            if nums_s and nums_c and nums_s != nums_c:
                return True

        return False

    def _verification_is_inert(
        self,
        slot: MemorySlot,
        candidate: Candidate,
        slot_idx: int,
        cand_dense: np.ndarray
    ) -> bool:
        """Whether no verifier verdict could change memory for this pair.

        The SAME branch only acts through relation evidence: the same or an
        exclusive attribute, a provably shared relation family, or a
        near-identical lexical restatement.  The CONTRADICTION branch only
        acts for the same subject *and* the same attribute (or the
        near-miss separation counter).  When none of those preconditions
        holds, every verdict falls through to the same insertion path, so
        the verifier call is pure cost.  Skipping it cannot change which
        facts are merged, superseded, or inserted; it only reduces model
        work.  The check mirrors the guards in ``_absorb_impl`` exactly and
        must be kept in sync with them.
        """
        represented = self._represented_predicate(slot)
        cand_pred = (candidate.predicate or "").strip().lower()
        slot_subject = self._normalize_subject(slot.debug_subject)
        cand_subject = self._normalize_subject(candidate.subject)
        same_subject = slot_subject == cand_subject
        named_pair = (
            slot_subject not in _GENERIC_SUBJECTS
            and cand_subject not in _GENERIC_SUBJECTS
        )
        if not same_subject and named_pair:
            # SAME is rejected by the entity-grounding guard for two distinct
            # named subjects, and CONTRADICTION requires the same subject.
            # Only a near-identical lexical restatement of a surface variant
            # can still merge, and that check is independent of the verdict.
            return not self._lexically_same_fact(slot, candidate)
        if (represented == cand_pred
                or self._share_exclusive_group(represented, cand_pred)):
            return False
        if self._predicates_relation_compatible(represented, cand_pred):
            return False
        if self._lexically_same_fact(slot, candidate):
            return False
        # Also covers generic-subject pairs, which the entity guard admits:
        # a SAME can re-link a closed interval through the attribute chain,
        # and CONTRADICTION needs the same subject and attribute.
        if self._is_same_attribute_or_event(
                slot, candidate, slot_idx, cand_dense):
            return False
        if (self.strict_entity_identity and same_subject
                and self._values_are_near_miss(
                    self._slot_value_text(slot), candidate.object)):
            return False
        return True

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
            result = self._absorb_impl(observation, timestamp, confidence)
        finally:
            self._sync_qdrant_if_active()
        slot_id = result.get("slot_id")
        if slot_id is not None:
            with self._lock:
                for slot in self.memory.slots:
                    if id(slot) == slot_id:
                        result["memory_id"] = slot.memory_id
                        break
        return result

    def _warn_index_failure(self, stage: str, exc: BaseException) -> None:
        from minemos_engine.retrieval.config import PRODUCTION
        if (self._retrieval_mode != PRODUCTION
                or self._index_failure_warned):
            return
        self._index_failure_warned = True
        _LOG.warning(
            "Qdrant %s failed; canonical memory is intact and queries fall "
            "back to canonical retrieval: %s", stage, exc)

    def _warn_backend_fallback(self, exc: BaseException) -> None:
        from minemos_engine.retrieval.config import PRODUCTION
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

    def sync_retrieval_after_restore(self) -> None:
        """Reconcile the derived index after canonical state was replaced.

        Recovery calls this only after durable privacy intent has been applied
        to the restored state and that sanitized state has been persisted.
        """
        self._sync_qdrant_after_restore()

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
            try:
                t_now = float(timestamp)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"timestamp must be a number, got {timestamp!r}") from exc
        elif isinstance(observation, Candidate):
            t_now = float(observation.timestamp)
        else:
            t_now = float(time.time())
        if not math.isfinite(t_now):
            raise ValueError(f"timestamp must be finite, got {t_now!r}")
        candidate = self._parse_candidate(observation, timestamp=t_now)
        # The explicit confidence argument applies to tuple/string
        # observations.  A Candidate keeps its own confidence, preserving the
        # documented behaviour that a structured observation carries it.
        if not isinstance(observation, Candidate):
            candidate.confidence = self._validated_confidence(confidence)
        else:
            candidate.confidence = self._validated_confidence(
                candidate.confidence)
        cand_text = f"{candidate.subject} {candidate.predicate} {candidate.object}."
        # A historical statement reports a past state.  It is recorded as a
        # closed observation and is never consolidated into, or allowed to
        # supersede, a current one.  A document/conversation segment is not a
        # state claim at all: its subject is a search boundary and its
        # predicate is an ordinal position, so past-tense wording in the text
        # says nothing about an expired interval.  Treating such a segment as
        # historical closed it at ingestion and made it permanently
        # unretrievable, which lost real transcript content.
        subject_is_document = str(candidate.subject or "").strip().casefold()
        is_historical_observation = (
            not subject_is_document.startswith(("conversation:", "document:"))
            and self._has_historical_marker(cand_text))
        x_t = self._validated_vector(
            self.lex_encoder.encode_full(candidate, dim=self.dim),
            self.dim, "lexical embedding")
        cand_dense = self._validated_vector(
            self.dense_encoder.encode(cand_text),
            self._expected_dense_dim(), "dense embedding")

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
            backdate_extend = False

            if self.memory.slots and not is_historical_observation:
                id_to_slot_index = {id(slot): idx for idx, slot in enumerate(self.memory.slots)}

                # 1. Dense candidate retrieval with adaptive anchor repulsion
                dense_candidates, beam_policy, _, _ = adaptive_dense_address(
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
                    # Pairs whose verdict cannot change state are excluded
                    # before the model call (see _verification_is_inert).
                    verify_indices = [
                        idx for idx in fused_candidates
                        if not self._verification_is_inert(
                            self.memory.slots[idx], candidate, idx, cand_dense)
                    ]
                    preds = []
                    if verify_indices:
                        pairs = [
                            (self.slot_texts[id(self.memory.slots[idx])],
                             cand_text)
                            for idx in verify_indices
                        ]
                        preds = self.verifier.verify_batch(
                            pairs,
                            mode=self.verification_mode,
                            entail_threshold=self.entail_threshold,
                            contra_threshold=self.contra_threshold
                        )
                        self.total_nli_calls += sum(
                            p.get("nli_invocations", 1) for p in preds)

                    for cand_idx, pred in zip(verify_indices, preds):
                        dec = pred["decision"]
                        slot = self.memory.slots[cand_idx]
                        slot_id = id(slot)

                        if dec == "SAME":
                            # Entity grounding guard: do not merge facts across different distinct entities
                            slot_subj = self._normalize_subject(slot.debug_subject)
                            cand_subj = self._normalize_subject(candidate.subject)
                            if slot_subj not in _GENERIC_SUBJECTS and cand_subj not in _GENERIC_SUBJECTS and slot_subj != cand_subj:
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
                            represented = self._represented_predicate(slot)
                            same_attribute = (
                                represented == cand_pred
                                or self._share_exclusive_group(represented, cand_pred)
                            )
                            if not same_attribute:
                                # The predicates differ.  A whole-statement
                                # SAME is not authorization to merge different
                                # relations.  Consolidation here requires
                                # positive evidence: both predicates must be
                                # explicitly classified into the same relation
                                # family, and neither entailment direction may
                                # be unsupported.  Object equality alone is
                                # never evidence of relation identity ("plays
                                # guitar" / "owns guitar"), and an unclassified
                                # predicate cannot authorize a destructive
                                # merge.  Once the relation is proven, the
                                # verifier's SAME carries the paraphrase itself:
                                # "authored an influential biography monograph"
                                # and "wrote an important biographical treatise"
                                # are one authorship fact even though their
                                # object strings are not lexically equivalent.
                                if not self._predicates_relation_compatible(
                                        represented, cand_pred):
                                    continue
                                slot_value = self._slot_value_text(slot)
                                if not slot_value or not candidate.object:
                                    continue
                                if not self._mutually_supported(pred):
                                    continue
                                if not self._values_compatible_for_merge(
                                        slot_value, candidate.object):
                                    # The relation is shared but the values
                                    # name different objects.  A verifier SAME
                                    # is not evidence that they are one fact,
                                    # so keep both rather than erase one.
                                    self.total_incompatible_value_separations += 1
                                    continue
                                target_idx = cand_idx
                                target_pred = pred
                                break

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
                                # interval.  Only a configured exclusive
                                # attribute (or an explicit correction) is a
                                # state change; everything else is kept as an
                                # independent fact.
                                if (self.strict_entity_identity
                                        and self._values_are_near_miss(
                                            slot_value, candidate.object)):
                                    # A one-edit difference is not evidence
                                    # of a state change.  Keep both facts;
                                    # never let NLI pick a winner.
                                    self.total_near_miss_separations += 1
                                    continue
                                state_change = (
                                    self._is_same_attribute_or_event(
                                        slot, candidate, cand_idx, cand_dense)
                                    or self._is_exclusive_state_change(
                                        slot, candidate)
                                )
                                if not state_change:
                                    continue
                                if t_now < slot.valid_from:
                                    containing = self._interval_containing(
                                        slot, t_now)
                                    containing_idx = self._slot_index(containing)
                                    if (containing_idx is not None
                                            and self._interval_value_matches(
                                                containing, candidate)):
                                        target_idx = containing_idx
                                        target_pred = pred
                                        break
                                    # The containing interval records a
                                    # different value: the observation is an
                                    # independent fact inside that window,
                                    # not a restatement of it.
                                    is_historical_predecessor = True
                                    successor_slot = slot
                                    continue
                                if slot.is_active:
                                    superseded_slot = slot
                                    is_temporal_change = True
                                # A different value asserted inside an already
                                # closed interval is an unresolved conflict;
                                # it never rewrites the closed history.
                                continue

                            # Equivalent value.  A backdated observation must
                            # become visible at its observation time without
                            # spanning a recorded interval.
                            if t_now < slot.valid_from:
                                containing = self._interval_containing(
                                    slot, t_now)
                                containing_idx = self._slot_index(containing)
                                if (containing_idx is not None
                                        and self._interval_value_matches(
                                            containing, candidate)):
                                    target_idx = containing_idx
                                    target_pred = pred
                                    break
                                if (slot.is_active and slot.valid_until is None
                                        and self._can_extend_valid_from(
                                            slot, t_now)):
                                    backdate_extend = True
                                    target_idx = cand_idx
                                    target_pred = pred
                                    break
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

                            if (self.strict_entity_identity and same_subject
                                    and self._values_are_near_miss(
                                        self._slot_value_text(slot),
                                        candidate.object)):
                                # An uncertain identity on the same subject and
                                # attribute is not a temporal revision.
                                self.total_near_miss_separations += 1
                                continue

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

                    # A backdated observation of an exclusive attribute is
                    # interval-aware regardless of the verifier's decision.
                    # SAME against a closed predecessor, CONTRADICTION against
                    # a later version, and NEUTRAL all describe one timeline:
                    # consolidate only into an interval that already asserts
                    # the value, otherwise insert a bounded segment before the
                    # nearest recorded successor.  Never allocate a second
                    # open-ended active value beside a recorded interval.
                    if (target_idx is None and not is_temporal_change
                            and not is_historical_predecessor
                            and self.classify_predicate(candidate.predicate)
                            == PREDICATE_POLICY_EXCLUSIVE):
                        equivalent, successor = (
                            self._backdated_exclusive_route(candidate, t_now)
                        )
                        if equivalent is not None:
                            target_idx = self._slot_index(equivalent)
                        elif successor is not None:
                            is_historical_predecessor = True
                            successor_slot = successor

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
                            slot.debug_subject, candidate.subject,
                            strict=self.strict_entity_identity
                        )
                    )
                    or (
                        bool(slot_value) and bool(candidate.object)
                        and self._normalize_value(slot_value)
                        != self._normalize_value(candidate.object)
                        and self._lexically_equivalent_value(
                            slot_value, candidate.object,
                            strict=self.strict_entity_identity
                        )
                    )
                )
                # A shorter restatement of an equivalent value must not erase
                # details the incumbent representation carries.  Keep the
                # richer wording; the shorter observation still counts as
                # consolidation.
                if (not surface_variant and slot_value and candidate.object
                        and self._content_tokens(candidate.object)
                        < self._content_tokens(slot_value)):
                    surface_variant = True
                slot.value = consolidate_vector(
                    slot=slot,
                    x_t=x_t,
                    alpha_t=alpha_t,
                    mode=self.consolidation_mode,
                    delta_max=self.delta_max,
                    lambda_anchor=self.lambda_anchor
                )
                slot.confidence = min(1.0, max(slot.confidence, candidate.confidence) + 0.05)
                slot.timestamp = t_now
                if backdate_extend and t_now < slot.valid_from:
                    # The observation proves the represented value held at
                    # t_now; no recorded interval covers the extended range.
                    slot.valid_from = t_now
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
                    self.bm25_index.update_document(id(slot), cand_text, stable_id=slot.memory_id)
                    self._invalidate_near_duplicate_index()

                self.total_updates += 1
                return {
                    "action": "consolidate",
                    "consolidation_class": "duplicate",
                    "slot_index": target_idx,
                    "slot_id": id(slot),
                    "similarity": best_sim,
                    "decision": "SAME",
                    "update_count": slot.update_count,
                    "valid_from": slot.valid_from,
                    "valid_until": slot.valid_until,
                    "text": cand_text,
                    "surface_variant": surface_variant,
                    "valid_from_extended": bool(backdate_extend),
                    "stored_text": self.slot_texts.get(id(slot), slot.text),
                }

            # Equal-timestamp policy: a contradiction arriving exactly at the
            # active interval's valid_from cannot close an earlier interval
            # and must not create an overlapping one.  Only the same exclusive
            # attribute or an explicitly corrected proposition may replace
            # the state in place; an unrelated fact that merely shares
            # predicate and timestamp falls through to independent insertion.
            if (is_temporal_change and superseded_slot is not None
                    and t_now == superseded_slot.valid_from):
                same_exclusive = self._predicates_share_exclusive_attribute(
                    self._represented_predicate(superseded_slot),
                    candidate.predicate)
                if same_exclusive or self._is_explicit_correction(
                        superseded_slot, candidate):
                    return self._replace_state_in_place(
                        superseded_slot, candidate, cand_text, cand_dense,
                        x_t, t_now, action="temporal_update"
                    )
                is_temporal_change = False
                superseded_slot = None

            # ── Case 2: Temporal State Revision (Contradiction of Active Fact) ──
            if is_temporal_change and superseded_slot is not None:
                superseded_id = id(superseded_slot)

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
                self.bm25_index.add_document(id(new_slot), cand_text, stable_id=new_slot.memory_id)
                self._invalidate_near_duplicate_index()

                self.total_temporal_updates += 1
                return {
                    "action": "temporal_update",
                    "consolidation_class": "temporal_update",
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
                equal = [
                    s for s in chain
                    if s.valid_from == t_now and (
                        self._predicates_share_exclusive_attribute(
                            self._represented_predicate(s),
                            candidate.predicate)
                        or self._is_explicit_correction(s, candidate))
                ]
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
                self.bm25_index.add_document(id(new_slot), cand_text, stable_id=new_slot.memory_id)
                self._invalidate_near_duplicate_index()

                self.total_temporal_updates += 1
                return {
                    "action": "historical_update",
                    "consolidation_class": "historical_observation",
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
                self.bm25_index.add_document(id(slot), cand_text, stable_id=slot.memory_id)
                self._invalidate_near_duplicate_index()
                self.total_inserts += 1
                return {
                    "action": "historical_observation",
                    "consolidation_class": "historical_observation",
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
            self.bm25_index.add_document(id(slot), cand_text, stable_id=slot.memory_id)
            self._near_duplicate_index_add(slot)

            self.total_inserts += 1
            result = {
                "action": action,
                "slot_index": len(self.memory.slots) - 1,
                "slot_id": id(slot),
                "similarity": best_sim,
                "update_count": 1,
                "valid_from": t_now,
                "valid_until": None,
                "text": cand_text
            }
            near = self._find_near_duplicate_slot(candidate)
            if near is None:
                result["consolidation_class"] = "distinct_fact"
            else:
                near_slot, near_field, near_edit = near
                self.total_unresolved_near_duplicates += 1
                result["consolidation_class"] = "unresolved_near_duplicate"
                result["near_duplicate_of"] = near_slot.memory_id
                result["near_duplicate_field"] = near_field
                result["near_duplicate_edit"] = near_edit
            return result

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
        self.bm25_index.update_document(id(slot), cand_text, stable_id=slot.memory_id)
        self._invalidate_near_duplicate_index()
        self.total_temporal_updates += 1

        return {
            "action": action,
            "consolidation_class": (
                "historical_observation" if action == "historical_update"
                else "temporal_update"),
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
        self._invalidate_near_duplicate_index()
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
        self._invalidate_near_duplicate_index()
        evict_idx, evicted_slot = evict_slot(
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

    def _slots_in_temporal_scope(
        self, slots: List[MemorySlot], memory_ids: Iterable[str]
    ) -> List[MemorySlot]:
        """Slots that can participate in a finding touching ``memory_ids``.

        A temporal finding names two slots whose subjects are equivalent.  If
        either slot is one of the scoped ids, the other slot therefore has the
        same subject family.  Restricting the scan to that family keeps a
        query-scoped diagnostic proportional to the facts of the queried
        subject instead of the whole tenant.  Global diagnostics pass no scope
        and still inspect every slot.
        """
        scoped = set(memory_ids)
        subjects = list(dict.fromkeys(
            slot.debug_subject for slot in slots
            if slot.memory_id in scoped))
        if not subjects:
            return []
        keys = {(subject or "").strip().casefold() for subject in subjects}
        # Precompute the distinct scoped subject tokens once.  The scan below
        # runs over every slot and used to re-normalize the same few subjects
        # for each of them, which dominated discovery latency.
        subject_tokens: List[Tuple[str, ...]] = []
        for subject in subjects:
            tokens = self._comparison_tokens(subject)
            if tokens and tokens not in subject_tokens:
                subject_tokens.append(tokens)
        selected: List[MemorySlot] = []
        for slot in slots:
            normalized = (slot.debug_subject or "").strip().casefold()
            if normalized in keys:
                selected.append(slot)
                continue
            tokens = self._comparison_tokens(slot.debug_subject)
            if not tokens:
                continue
            if any(
                tokens == candidate
                or self._tokens_are_typo_equivalent(tokens, candidate)
                for candidate in subject_tokens
            ):
                selected.append(slot)
        return selected

    def _require_finite_temporal_intervals(
        self, slots: Sequence[MemorySlot]
    ) -> None:
        """Fail closed on intervals that cannot be compared or sorted.

        Only reachable through hand-built or mutated state.  A clear error is
        better than a ``TypeError`` from deep inside a comparison, which would
        make an integrity check look like a crash.
        """
        for idx, slot in enumerate(slots):
            for field in ("valid_from", "valid_until"):
                value = getattr(slot, field)
                if value is None and field == "valid_until":
                    continue
                if (isinstance(value, bool)
                        or not isinstance(value, numbers.Real)
                        or not math.isfinite(float(value))):
                    raise ValueError(
                        f"slot {idx} (memory_id={slot.memory_id!r}) has an "
                        f"invalid {field} {value!r}; temporal invariants "
                        f"require finite numeric intervals")

    @_snapshot_locked
    def exclusive_temporal_conflicts(
        self, _scope_memory_ids: Optional[Iterable[str]] = None
    ) -> List[Dict[str, Any]]:
        """Reports active facts that cannot both hold for one exclusive attribute.

        This catches unlinked exclusive conflicts: two active slots can assert
        different values of the same exclusive attribute without ever being
        linked as versions (for example after importing legacy state, or when
        the verifier returned NEUTRAL).  The structural invariant cannot see
        that pair, because it only groups linked or identical-value slots.

        The check is read-only.  It never mutates slots, intervals, links,
        indexes, or counters, and it never calls a model, so it is safe to run
        repeatedly.  Value equivalence is therefore decided lexically
        (normalization, unambiguous abbreviation expansion, one edit); two
        different surface forms are reported rather than silently assumed
        equivalent.

        Two independent detections are reported:

        * configured exclusive attributes (``reason`` is
          ``unlinked_exclusive_temporal_conflict``), scoped by
          ``confidence_scope="predicate_policy_only"``;
        * copular/state attributes that have no cardinality policy but assert
          one current state of the subject ("goal is", "website is",
          "works as", bare "is"), reported as
          ``unresolved_state_temporal_conflict`` and scoped by
          ``confidence_scope="state_predicate_form_only"``.  This is
          detection only: it never turns an unknown predicate into an
          exclusive one and never supersedes a fact.

        The check reads one consistent snapshot: it holds the engine lock for
        its whole duration and works on a copied slot list, so a concurrent
        ``absorb`` cannot change the slot population between passes.

        Exclusivity comes from the predicate policy: a pair is reported only
        when both predicates are ``EXCLUSIVE`` and either identical or in the
        same configured exclusive group.  Multi-valued predicates (and
        unknown predicates that are not a state form) are never reported, so
        legitimate multi-valued facts stay silent.  A negated statement does
        not assert its object as a value, so a negated slot and a different
        value are not reported; same-value negations stay visible through the
        structural invariant (``multiple_active``).

        ``confidence`` is the policy determination (always 1.0 for a
        configured exclusive predicate) and is scoped by the emitted
        ``confidence_scope``.  ``semantic_equivalence="lexical_only"``
        records that values were compared lexically, and
        ``conflict_status="unresolved"`` records that the finding reports the
        conflict without choosing a winner.

        Returns one finding per conflicting active pair, in slot order.  An
        empty list means no unlinked exclusive conflict was found.

        ``_scope_memory_ids`` is an internal restriction used by the response
        metadata path: only the subject families of those ids are inspected,
        and the emitted findings are the same ones the global scan would
        produce for that subset.  Omit it for the whole-tenant invariant.
        """
        slots = list(self.memory.slots)
        self._require_finite_temporal_intervals(slots)
        if _scope_memory_ids is not None:
            slots = self._slots_in_temporal_scope(slots, _scope_memory_ids)
        findings: List[Dict[str, Any]] = []
        for i in range(len(slots)):
            a = slots[i]
            if a.redacted or not a.is_active:
                continue
            for j in range(i + 1, len(slots)):
                b = slots[j]
                if b.redacted or not b.is_active:
                    continue
                if self._slots_are_linked(a, b):
                    continue
                if not self._subjects_are_equivalent(a.debug_subject, b.debug_subject):
                    continue

                attribute_pair = None
                attribute_reason = UNLINKED_EXCLUSIVE_CONFLICT
                attribute_scope = "predicate_policy_only"
                attribute_policy = PREDICATE_POLICY_EXCLUSIVE
                state_attribute = None
                for pred_a in self._slot_predicates(a):
                    for pred_b in self._slot_predicates(b):
                        if self._predicates_share_exclusive_attribute(pred_a, pred_b):
                            attribute_pair = (pred_a, pred_b)
                            break
                    if attribute_pair is not None:
                        break
                if attribute_pair is None:
                    # An unconfigured predicate that asserts one *state* of
                    # the subject ("goal is", "website is", "works as", bare
                    # "is") can still leave two incompatible active values.
                    # This is detection only: no cardinality policy is
                    # invented and nothing is superseded.
                    for pred_a in self._slot_predicates(a):
                        key = self._state_attribute_key(pred_a)
                        if key is None:
                            continue
                        for pred_b in self._slot_predicates(b):
                            if self._state_attribute_key(pred_b) != key:
                                continue
                            attribute_pair = (pred_a, pred_b)
                            attribute_reason = UNRESOLVED_STATE_CONFLICT
                            attribute_scope = "state_predicate_form_only"
                            attribute_policy = PREDICATE_POLICY_UNKNOWN
                            state_attribute = key
                            break
                        if attribute_pair is not None:
                            break
                    if attribute_pair is None:
                        continue

                value_a = self._slot_value_text(a)
                value_b = self._slot_value_text(b)
                if not value_a or not value_b:
                    continue
                # Only proven equality keeps a state pair settled.  The
                # compatibility gate answers "cannot prove different", which
                # is not equality: two active state values it cannot separate
                # are still unresolved and must not be presented as settled.
                # Non-state exact/exclusive pairs keep the older behaviour.
                equivalent = self._lexically_equivalent_value(value_a, value_b)
                if equivalent:
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
                    "reason": attribute_reason,
                    "slot_indices": [i, j],
                    "slot_ids": [id(a), id(b)],
                    "memory_ids": [a.memory_id, b.memory_id],
                    "subject": a.debug_subject,
                    "predicates": [attribute_pair[0], attribute_pair[1]],
                    "values": [value_a, value_b],
                    "valid_from": [a.valid_from, b.valid_from],
                    "valid_until": [a.valid_until, b.valid_until],
                    "overlap": True,
                    "predicate_policy": attribute_policy,
                    "state_attribute": state_attribute,
                    "detection_source": (
                        "active_unlinked_state_slots"
                        if state_attribute is not None
                        else "active_unlinked_slots"),
                    # Deterministic policy determination, not semantic truth:
                    # the checker never calls a verifier.
                    "confidence": 1.0,
                    # Explicitly scoped so consumers cannot read `confidence`
                    # as 100% semantic confidence in the conflict.
                    "confidence_scope": attribute_scope,
                    "semantic_equivalence": "lexical_only",
                    "conflict_status": "unresolved",
                })
        return findings

    @_snapshot_locked
    def unresolved_temporal_corrections(
        self, _scope_memory_ids: Optional[Iterable[str]] = None
    ) -> List[Dict[str, Any]]:
        """Reports active facts that look like a correction the engine could
        not safely apply.

        The canonical engine supersedes an ownership state when a later
        disposal names the same entity.  When the verifier or the lexical gate
        cannot establish that (for example a paraphrased object, a NEUTRAL
        verdict, or a reversed order), both facts stay active.  Retaining them
        is correct; serving the older fact as current is not.  This read-only
        check surfaces that state so the service can mark the answer
        unsettled.

        A finding requires: same subject, an unlinked newer active slot whose
        predicate is an ownership transition, an older ownership slot, and
        equal entity cores after quantity/legal-suffix words and a transfer
        recipient are removed.  ``transition_kind`` records the evidence:
        ``disposal`` is a full disposal of the same entity the engine chose
        not to apply; ``unspecified`` is a transfer/loss or an unquantified
        share disposal, where the engine cannot prove ownership ended.  Both
        observations are retained in either case.  A linked pair is a resolved
        history and is never reported, and a newer ownership statement is
        treated as a legitimate re-acquisition rather than a correction.  No
        model is called and nothing is mutated.

        ``_scope_memory_ids`` restricts the scan to those ids' subject
        families, exactly as in ``exclusive_temporal_conflicts``.  Omitting it
        checks the whole tenant.
        """
        slots = list(self.memory.slots)
        if _scope_memory_ids is not None:
            slots = self._slots_in_temporal_scope(slots, _scope_memory_ids)
        findings: List[Dict[str, Any]] = []
        for i, older in enumerate(slots):
            if older.redacted or not older.is_active:
                continue
            for j, newer in enumerate(slots):
                if j == i or newer.redacted or not newer.is_active:
                    continue
                if (newer.valid_from, newer.created_at) <= (
                        older.valid_from, older.created_at):
                    continue
                if self._slots_are_linked(older, newer):
                    continue
                if self._normalize_subject(older.debug_subject) != (
                        self._normalize_subject(newer.debug_subject)):
                    continue
                newer_predicate = self._represented_predicate(newer)
                newer_value = self._slot_value_text(newer)
                transition_kind = self._ownership_transition_kind(
                    older, newer_predicate, newer_value)
                if transition_kind is None:
                    continue
                findings.append({
                    "reason": UNRESOLVED_TEMPORAL_CORRECTION,
                    "slot_indices": [i, j],
                    "slot_ids": [id(older), id(newer)],
                    "memory_ids": [older.memory_id, newer.memory_id],
                    "older_memory_id": older.memory_id,
                    "newer_memory_id": newer.memory_id,
                    "predicates": [
                        self._represented_predicate(older), newer_predicate],
                    "valid_from": [older.valid_from, newer.valid_from],
                    "transition_kind": transition_kind,
                    "detection_source": (
                        "active_unlinked_ownership_transition"
                        if transition_kind == "unspecified"
                        else "active_unlinked_disposal"),
                    "confidence_scope": "lexical_entity_equivalence",
                    "semantic_equivalence": "lexical_entity_core",
                    "conflict_status": "unresolved",
                })
        return findings

    def answer_temporal_status(
        self, rows: List[Dict[str, Any]], status: str = "answered"
    ) -> Dict[str, Any]:
        """Describes whether ``rows`` may be presented as settled truth.

        This is response metadata, not an eligibility verdict: ``status``
        still reports whether the query named a subject and stored attribute.
        The returned schema separates four situations a caller must not
        conflate:

        * settled: ``answer_is_settled`` is true, ``conflict_status`` is
          ``"none"`` and ``confidence_scope`` is ``"retrieval_only"``;
        * unresolved temporal conflict: two active exclusive values overlap
          (``reason=unlinked_exclusive_temporal_conflict``), scoped by
          ``confidence_scope="predicate_policy_only"``;
        * unresolved state conflict: two active copular/state values overlap
          (``reason=unresolved_state_temporal_conflict``), scoped by
          ``confidence_scope="state_predicate_form_only"``;
        * unresolved correction: a newer ownership transition names the same
          entity as an older active ownership fact but the engine kept both,
          scoped by ``confidence_scope="lexical_entity_equivalence"``;
        * unverified/empty: no row can be settled, ``confidence_scope`` is
          ``"none"`` and the temporal lists stay empty because no answer was
          composed.

        Findings are filtered to the returned rows' ``memory_id`` values, so an
        unrelated conflict elsewhere in memory does not mark this answer
        unsettled.  Detection is additionally restricted to those rows'
        subject families: the cost of a query is proportional to the facts of
        the subject it names, not to the whole tenant.  Only identifiers,
        predicates and validity starts are echoed: the statement text stays in
        the rows the caller already received.  The check is read-only and
        never mutates canonical memory.
        """
        answered = status in ("answered", "disabled") and bool(rows)
        if not answered:
            return {
                "conflict_status": "none",
                "answer_is_settled": False,
                "uncertain": True,
                "confidence_scope": "none",
                "temporal_conflicts": [],
                "unresolved_corrections": [],
            }
        row_ids = {
            row.get("memory_id") for row in rows if row.get("memory_id")
        }
        conflicts: List[Dict[str, Any]] = []
        corrections: List[Dict[str, Any]] = []
        with self._lock:
            if row_ids:
                for finding in self.exclusive_temporal_conflicts(
                        _scope_memory_ids=row_ids):
                    if not row_ids.intersection(
                            finding.get("memory_ids") or ()):
                        continue
                    conflict = {
                        "reason": finding["reason"],
                        "memory_ids": list(finding["memory_ids"]),
                        "predicates": list(finding["predicates"]),
                        "valid_from": list(finding["valid_from"]),
                        "conflict_status": finding["conflict_status"],
                        "confidence_scope": finding["confidence_scope"],
                        "semantic_equivalence": finding[
                            "semantic_equivalence"],
                    }
                    if finding.get("state_attribute") is not None:
                        conflict["state_attribute"] = finding["state_attribute"]
                    conflicts.append(conflict)
                for finding in self.unresolved_temporal_corrections(
                        _scope_memory_ids=row_ids):
                    if not row_ids.intersection(
                            finding.get("memory_ids") or ()):
                        continue
                    correction = {
                        "reason": finding["reason"],
                        "memory_ids": list(finding["memory_ids"]),
                        "older_memory_id": finding["older_memory_id"],
                        "newer_memory_id": finding["newer_memory_id"],
                        "predicates": list(finding["predicates"]),
                        "valid_from": list(finding["valid_from"]),
                        "conflict_status": finding["conflict_status"],
                        "confidence_scope": finding["confidence_scope"],
                        "semantic_equivalence": finding[
                            "semantic_equivalence"],
                    }
                    if finding.get("transition_kind") is not None:
                        correction["transition_kind"] = finding[
                            "transition_kind"]
                    corrections.append(correction)
        resolved = bool(conflicts or corrections)
        if any(item["reason"] == UNLINKED_EXCLUSIVE_CONFLICT
               for item in conflicts):
            scope = "predicate_policy_only"
        elif conflicts:
            scope = "state_predicate_form_only"
        elif corrections:
            scope = "lexical_entity_equivalence"
        else:
            scope = "retrieval_only"
        return {
            "conflict_status": "unresolved" if resolved else "none",
            "answer_is_settled": not resolved,
            "uncertain": resolved,
            "confidence_scope": scope,
            "temporal_conflicts": conflicts,
            "unresolved_corrections": corrections,
        }

    @_snapshot_locked
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

        The check holds the engine lock and iterates a copied slot list, so
        the union-find indices, link targets and the slot enumeration all
        describe the same instant even while writers are appending.

        ``include_unlinked_conflicts=True`` appends the findings of
        :meth:`exclusive_temporal_conflicts`, which covers unlinked active
        conflicts of exclusive attributes.  The default stays ``False`` so
        existing structural-invariant results remain comparable.

        Returns a list of violation dicts; an empty list means the invariant
        holds.
        """
        slots = list(self.memory.slots)
        self._require_finite_temporal_intervals(slots)
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
                # A redacted tombstone intentionally has no subject or value,
                # so its link cannot be classified as consistent or
                # inconsistent.  Reciprocity was already checked above; leave
                # it out of the grouping pass rather than reporting a false
                # corruption.
                if a.redacted or b.redacted:
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
            "subject", "subjects", "predicate", "min_confidence",
            "min_update_count", "memory_ids",
        }
        unknown = set(filters) - allowed
        if unknown:
            raise ValueError(
                f"unknown query filter(s): {sorted(unknown)}; "
                f"expected {sorted(allowed)}")
        out: Dict[str, Any] = {}
        if filters.get("subject") is not None:
            out["subject"] = str(filters["subject"])
        if filters.get("subjects") is not None:
            value = filters["subjects"]
            if isinstance(value, (str, bytes)) or not isinstance(
                    value, (list, tuple, set, frozenset)):
                raise ValueError(
                    "subjects filter must be a sequence of subject strings")
            if filters.get("subject") is not None:
                raise ValueError(
                    "subject and subjects filters are mutually exclusive")
            subjects = [str(item) for item in value]
            if not subjects:
                raise ValueError("subjects filter must not be empty")
            out["subjects"] = subjects
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
        if "subjects" in filters:
            wanted = {
                self._normalize_subject(item) for item in filters["subjects"]}
            if self._normalize_subject(slot.debug_subject) not in wanted:
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
            from minemos_engine.retrieval.adapter import RetrievalUnavailable
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

        # Query-side non-finite handling is part of the pinned retrieval
        # contract: canonical and optimized backends must agree on NaN/Inf and
        # width-mismatch outcomes rather than rejecting the query here.
        q_dense = np.asarray(self.dense_encoder.encode(query_text),
                             dtype=np.float32)

        with self._lock:
            candidate_slots = []

            for idx, slot in enumerate(self.memory.slots):
                if slot.redacted:
                    continue
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
                    "memory_id": slot.memory_id,
                    "similarity": round(sim, 4),
                    "subject": slot.debug_subject,
                    "predicate": self._represented_predicate(slot),
                    "original_predicate": slot.debug_predicate,
                    "text": self.slot_texts.get(id(slot), slot.text),
                    "confidence": round(float(slot.confidence), 4),
                    "update_count": slot.update_count,
                    "is_active": slot.is_active,
                    "valid_from": slot.valid_from,
                    "valid_until": slot.valid_until
                })
            return results

    def _discover_scope_subjects(
        self,
        query_text: str,
        *,
        as_of: Optional[float],
        active_only: bool,
        filters: Optional[Dict[str, Any]],
        current_turn: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Discover document scopes whose visible rows carry the question.

        Discovery is a candidate-generation step, not an answer: it scans a
        bounded dense window, groups document subjects, and keeps scopes whose
        rows contain at least two distinct query evidence terms.  The ordinary
        joint evidence gate decides afterwards whether the selected scopes
        actually support the question.  The caller's tenant boundary is the
        engine itself; discovery cannot query another engine.
        """
        from minemos_engine.evidence import (
            evidence_terms,
            is_document_subject,
            present_terms,
        )

        rows = self.query(
            query_text, top_k=SCOPE_DISCOVERY_DEPTH, as_of=as_of,
            active_only=active_only, filters=filters)
        if current_turn:
            # The caller's own current message cannot be the scope that
            # grounds the question: its self-similarity is not evidence.
            rows = [
                candidate for candidate in rows
                if not self._contains_turn(
                    candidate.get("text") or "", current_turn)
            ]
        terms = evidence_terms(query_text)
        grouped: Dict[str, Dict[str, Any]] = {}
        for candidate in rows:
            subject = str(candidate.get("subject") or "").strip()
            if not is_document_subject(subject):
                continue
            entry = grouped.get(subject)
            if entry is None:
                entry = {"matched": set(), "best": 0.0}
                grouped[subject] = entry
            entry["matched"].update(present_terms(
                terms, candidate.get("text") or ""))
            entry["best"] = max(
                entry["best"], float(candidate.get("similarity") or 0.0))
        ranked = sorted(
            ((len(entry["matched"]), entry["best"], subject)
             for subject, entry in grouped.items()),
            key=lambda item: (-item[0], -item[1], item[2]))
        selected = [
            item[2] for item in ranked
            if item[0] >= SCOPE_DISCOVERY_MIN_TERMS
        ][:SCOPE_DISCOVERY_MAX_SCOPES]
        metadata = {
            "enabled": True,
            "depth": SCOPE_DISCOVERY_DEPTH,
            "min_terms": SCOPE_DISCOVERY_MIN_TERMS,
            "candidate_scopes": len(grouped),
            "selected_scopes": len(selected),
        }
        return selected, metadata

    def eligible_query(
        self,
        query_text: str,
        top_k: int = 5,
        as_of: Optional[float] = None,
        active_only: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: Optional[float] = None,
        subject: Optional[str] = None,
        scope_subjects: Optional[Sequence[str]] = None,
        context_text: Optional[str] = None,
        discover_scopes: bool = False,
        current_turn: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieval with an opt-in subject/attribute eligibility gate.

        This wraps the canonical ranking rather than replacing it.  A fact is
        returned only when the query names a subject that exists in memory and
        asks for an attribute that subject actually stores.  Otherwise the
        result is an abstention with a reason:

        ``subject_not_found``
            No stored entity matches the entity named by the query.
        ``attribute_not_stored``
            The entity exists, but the requested attribute is not stored.
        ``subject_ambiguous``
            Several stored entities match the named entity equally well, so
            answering would attribute one person's fact to another.
        ``document_scope_required``
            Nothing named by the query exists, but memory holds only
            document/conversation-scoped records.  A verbatim archive cannot
            identify people from a name; the caller must ask in document
            scope (explicit ``subject``) or use structured ingestion.
        ``low_similarity``
            The caller supplied ``min_similarity`` and the best eligible fact
            falls below it.  No threshold is applied unless supplied, because
            an uncalibrated global threshold would be arbitrary.
        ``scope_not_found``
            The caller requested one or more document scopes and at least one
            of them holds no visible record.  Answering across the remaining
            scopes would silently narrow the request, so the query abstains.

        ``subject`` is an explicit caller assertion, not a resolver hint: the
        assertion must match a stored subject token-for-token (case and
        punctuation aside) or the query abstains.  It never falls back to a
        fuzzy match, and it still passes the attribute gate.  This is the
        supported way to address a document-scoped record such as
        ``conversation:<key>``.

        ``scope_subjects`` is the joint form of that assertion: an explicit,
        non-empty list of document/conversation subjects to search as one
        surface.  Every scope must be a document boundary, duplicates are
        rejected, and rows are ranked globally across the requested scopes so
        evidence can be distributed between them.  A requested scope with no
        visible record aborts the answer (``scope_not_found``) rather than
        silently answering from the scopes that remain: the caller asked about
        all of them, and an unresolved scope could hold contradicting facts.
        ``subject`` and ``scope_subjects`` are mutually exclusive.

        ``context_text`` is caller-supplied previous-turn text for a follow-up
        such as "what about that?".  It is only accepted together with an
        explicit ``subject`` or ``scope_subjects``: a follow-up belongs to a
        session, and allowing contentless anaphora to search the whole tenant
        would turn an unqualified question into a broad guess.  Context is
        used to build the retrieval/evidence question, while focus and
        attribute extraction stay on ``query_text``.  It is never stored and
        never broadens the requested scope; the returned rows must still
        contain the combined question's content.

        ``discover_scopes`` is the explicit, opt-in form of the joint query
        for callers who do not know the stored conversation identifiers.  It
        discovers document scopes by scanning a bounded dense candidate
        window and keeping conversations whose rows carry at least two
        distinct query evidence terms.  The discovered scopes then pass
        through the same joint evidence gate as an explicit
        ``scope_subjects`` request; nothing is merged across scopes and no
        threshold is relaxed.  Discovery never leaves the caller's tenant,
        is mutually exclusive with every explicit scope form and with
        ``context_text``, and fails closed with ``scope_not_discovered`` when
        no scope carries enough of the question.  Discovered identifiers are
        echoed in ``scopes`` so the caller can issue a narrower follow-up.

        ``current_turn`` is the caller's own current message when it may
        already have been ingested (a replay or ingest-then-serve flow).  A
        stored row that repeats it verbatim is the question, not evidence for
        it: those rows are dropped from ranking, evidence and the returned
        window, and discovery does not treat their conversation as a match.
        The parameter is opt-in because a caller may also send a derived
        retrieval expression that is not a stored message; a query whose text
        merely occurs inside a target row must keep that row.  It is never
        stored, never echoed, and does not change any threshold or scan
        limit.  ``current_turn_excluded`` reports how many rows were dropped.

        When the query does not identify an entity (indirect or fragment
        queries), ``status`` is ``unverified`` and the ungated rows are
        returned instead of guessing.  Those rows are diagnostic material:
        the HTTP service withholds them from ordinary ``/v1/query``
        responses.  That case is distinct from an absent memory.

        Eligibility inspects canonical state, so it works with every retrieval
        backend without importing Qdrant.
        """
        from minemos_engine.eligibility import (
            attribute_ids,
            attributes_compatible,
            extract_query_focus,
            focus_match_score,
            has_possessive_marker,
            tokens as focus_tokens,
        )
        from minemos_engine.evidence import (
            EVIDENCE_SCAN_LIMIT,
            assess_evidence,
            assess_joint_evidence,
            evidence_terms,
            is_document_subject,
            present_terms,
        )

        filters = self._normalize_query_filters(filters)
        if not isinstance(discover_scopes, bool):
            raise ValueError("discover_scopes must be a boolean")
        turn_norm = ""
        if current_turn is not None:
            if not isinstance(current_turn, str):
                raise ValueError("current_turn must be a string")
            if len(current_turn) > 8192:
                raise ValueError(
                    "current_turn must not exceed 8192 characters")
            # A one- or two-character turn ("ok") would match as a word in
            # many unrelated rows; exclusion is for substantive messages.
            candidate_turn = self._normalize_text(current_turn)
            if len(candidate_turn) >= 3:
                turn_norm = candidate_turn
        if discover_scopes:
            if subject is not None or scope_subjects is not None:
                raise ValueError(
                    "discover_scopes is mutually exclusive with subject and "
                    "scope_subjects")
            if context_text is not None:
                raise ValueError(
                    "discover_scopes is mutually exclusive with "
                    "context_text")
            if filters is not None and (
                    "subject" in filters or "subjects" in filters):
                raise ValueError(
                    "discover_scopes and a subject filter are mutually "
                    "exclusive")
        joint = scope_subjects is not None
        scopes: List[str] = []
        scope_keys: List[str] = []
        scope_meta: Dict[str, Any] = {}
        if joint:
            if subject is not None:
                raise ValueError(
                    "subject and scope_subjects are mutually exclusive")
            if filters is not None and "subjects" in filters:
                raise ValueError(
                    "scope_subjects and a subjects filter are mutually "
                    "exclusive")
            if isinstance(scope_subjects, (str, bytes)) or not isinstance(
                    scope_subjects, (list, tuple)):
                raise ValueError(
                    "scope_subjects must be a list of document subjects")
            for raw in scope_subjects:
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError(
                        "scope_subjects entries must be non-empty strings")
                value = raw.strip()
                if not is_document_subject(value):
                    raise ValueError(
                        "scope_subjects entries must be document or "
                        "conversation subjects")
                scopes.append(value)
            if not scopes:
                raise ValueError("scope_subjects must not be empty")
            scope_keys = [self._normalize_subject(item) for item in scopes]
            if len(set(scope_keys)) != len(scope_keys):
                raise ValueError("scope_subjects must not contain duplicates")
            scope_meta = {
                "scopes": list(scopes),
                "scopes_present": [],
                "scopes_missing": [],
            }
            filters = dict(filters or {})
            filters["subjects"] = list(scopes)

        if context_text is not None:
            if not isinstance(context_text, str) or not context_text.strip():
                raise ValueError(
                    "context_text must be a non-empty string")
            if len(context_text) > 8192:
                raise ValueError(
                    "context_text must not exceed 8192 characters")
            if subject is None and not joint:
                raise ValueError(
                    "context_text requires an explicit subject or "
                    "scope_subjects")
            context_text = context_text.strip()
        evidence_text = (
            query_text if context_text is None
            else f"{query_text}\n{context_text}")

        threshold: Optional[float] = None
        if min_similarity is not None:
            try:
                threshold = float(min_similarity)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "min_similarity must be a number in [0, 1]") from exc
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "min_similarity must be a finite value in [0, 1]")
        discovery_meta: Optional[Dict[str, Any]] = None
        discovered_scopes: List[str] = []
        if discover_scopes:
            discovered, discovery_meta = self._discover_scope_subjects(
                query_text, as_of=as_of, active_only=active_only,
                filters=filters, current_turn=turn_norm or None)
            if discovered:
                discovered_scopes = discovered
                joint = True
                scopes = discovered
                scope_keys = [self._normalize_subject(item)
                              for item in scopes]
                scope_meta = {
                    "scopes": list(scopes),
                    "scopes_present": [],
                    "scopes_missing": [],
                }
                filters = dict(filters or {})
                filters["subjects"] = list(scopes)
        focus, focus_candidates = extract_query_focus(query_text)
        original_focus = focus
        explicit = subject is not None or joint
        possessive_focus = (
            not explicit and has_possessive_marker(query_text))
        if subject is not None:
            if not isinstance(subject, str) or not subject.strip():
                raise ValueError("subject must be a non-empty string")
            asserted = focus_tokens(subject)
            if not asserted:
                raise ValueError(
                    "subject must contain at least one identifier token")
            focus = " ".join(asserted)
            focus_candidates = [asserted]
        query_attributes = attribute_ids(query_text)
        base_focus_tokens = focus_tokens(focus or "")
        extends_focus = any(
            len(candidate) > len(base_focus_tokens)
            for candidate in focus_candidates)

        def row(slot: MemorySlot, idx: int, sim: float) -> Dict[str, Any]:
            return {
                "slot_index": idx,
                "slot_id": id(slot),
                "memory_id": slot.memory_id,
                "similarity": round(sim, 4),
                "subject": slot.debug_subject,
                "predicate": self._represented_predicate(slot),
                "original_predicate": slot.debug_predicate,
                "text": self.slot_texts.get(id(slot), slot.text),
                "confidence": round(float(slot.confidence), 4),
                "update_count": slot.update_count,
                "is_active": slot.is_active,
                "valid_from": slot.valid_from,
                "valid_until": slot.valid_until,
            }

        current_turn_excluded = 0

        def result(status: str, reason: Optional[str],
                   rows: List[Dict[str, Any]], eligible: int,
                   subject_matches: int,
                   document_scoped: int = 0,
                   evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            if evidence is None:
                state = "ambiguous" if (
                    status == "unverified"
                    or reason == "subject_ambiguous") else "missing"
                evidence = {
                    "state": state,
                    "reason": reason,
                    "sufficient": False,
                    "scope_kind": "none",
                    "supporting_memory_ids": [],
                    "summary": {},
                }
            return {
                "status": status,
                "reason": reason,
                "focus": focus,
                "subject_source": (
                    "discovered_scopes" if discovery_meta is not None
                    else ("explicit_scopes" if joint
                          else ("explicit" if explicit
                                else ("query_focus"
                                      if focus is not None else None)))),
                "query_attributes": sorted(query_attributes),
                "rows": rows,
                "top_similarity": (
                    rows[0]["similarity"] if rows else None),
                "threshold": threshold,
                "eligible_slots": eligible,
                "subject_matched_slots": subject_matches,
                "document_scoped_slots": document_scoped,
                # Whether the returned rows are evidence for the question,
                # and why not when they are not.  Independent of the
                # temporal verdict below: scope membership alone is not
                # relevance.
                "evidence": evidence,
                # Response metadata, not an eligibility signal: a query can be
                # eligible and still be contested.  See
                # ``answer_temporal_status``.
                "temporal": self.answer_temporal_status(rows, status),
                **scope_meta,
                "context_used": context_text is not None,
                "current_turn_excluded": current_turn_excluded,
                "discovery": discovery_meta,
            }

        if discovery_meta is not None and not discovered_scopes:
            # Discovery is fail-closed: a broad search that cannot ground the
            # question in at least one conversation answers nothing rather
            # than falling back to tenant-wide similarity.
            evidence = {
                "state": "missing",
                "reason": "scope_not_discovered",
                "sufficient": False,
                "scope_kind": "none",
                "supporting_memory_ids": [],
                "summary": {},
            }
            outcome = result("abstained", "scope_not_discovered", [], 0, 0,
                             evidence=evidence)
            outcome["scopes"] = []
            outcome["scopes_present"] = []
            outcome["scopes_missing"] = []
            return outcome

        if focus is None and not joint:
            rows = self.query(
                evidence_text, top_k=top_k, as_of=as_of,
                active_only=active_only, filters=filters)
            if turn_norm:
                kept_rows = []
                for item in rows:
                    if self._contains_turn(
                            item.get("text") or "", turn_norm):
                        current_turn_excluded += 1
                        continue
                    kept_rows.append(item)
                rows = kept_rows
            return result("unverified", "subject_not_identified", rows,
                          0, 0)

        with self._lock:
            candidates: List[Tuple[int, MemorySlot, float, int]] = []
            document_scoped = 0
            visible = 0
            scope_keys_set = set(scope_keys)
            scope_present: set = set()
            for idx, slot in enumerate(self.memory.slots):
                if slot.redacted:
                    continue
                if as_of is not None:
                    if (slot.valid_from > as_of
                            or (slot.valid_until is not None
                                and slot.valid_until <= as_of)):
                        continue
                elif active_only and not slot.is_active:
                    continue
                if filters and not self._slot_matches_filters(slot, filters):
                    continue
                visible += 1
                normalized_subject = self._normalize_subject(
                    slot.debug_subject)
                if (normalized_subject.startswith("conversation:")
                        or normalized_subject.startswith("document:")):
                    document_scoped += 1
                if joint:
                    if normalized_subject in scope_keys_set:
                        scope_present.add(normalized_subject)
                        candidates.append(
                            (idx, slot, 1.0, len(scope_keys)))
                    continue
                if subject is not None and focus_tokens(
                        normalized_subject) != asserted:
                    # An explicit assertion names one subject.  A stored
                    # subject that merely contains the asserted token
                    # ("user's team" for "user") is a different entity and
                    # must not make the named subject look ambiguous.
                    continue
                best: Optional[Tuple[float, int]] = None
                for candidate in focus_candidates:
                    score = focus_match_score(
                        slot.debug_subject, candidate, exact=explicit)
                    if score is None:
                        continue
                    rank = (score, len(candidate))
                    if best is None or rank > best:
                        best = rank
                if best is not None:
                    candidates.append((idx, slot, best[0], best[1]))

            if joint:
                scope_meta["scopes_present"] = [
                    scope for scope, key in zip(scopes, scope_keys)
                    if key in scope_present]
                scope_meta["scopes_missing"] = [
                    scope for scope, key in zip(scopes, scope_keys)
                    if key not in scope_present]
                if scope_meta["scopes_missing"]:
                    # An unresolved scope could hold contradicting evidence.
                    # Answering from the remaining scopes would narrow the
                    # caller's request without saying so, so fail closed.
                    missing = {
                        "state": "missing",
                        "reason": "scope_not_found",
                        "sufficient": False,
                        "scope_kind": "document",
                        "supporting_memory_ids": [],
                        "summary": {},
                    }
                    return result("abstained", "scope_not_found", [],
                                  len(candidates), len(candidates),
                                  document_scoped, evidence=missing)
            else:
                exact = [item for item in candidates if item[2] >= 1.0]
                if exact:
                    candidates = exact
                elif explicit:
                    # An explicit assertion is never downgraded to a fuzzy
                    # guess.
                    candidates = []
                elif candidates:
                    # Keep only the closest name(s).  A typo must not make
                    # every nearby name eligible to answer.
                    best_score = max(item[2] for item in candidates)
                    candidates = [
                        item for item in candidates
                        if item[2] >= best_score - 1e-9]
                if candidates:
                    # A subject that accounts for the whole focus -- including
                    # a possessive head noun ("user's team") -- is stronger
                    # evidence than one that only accounts for its prefix
                    # ("user").
                    best_width = max(item[3] for item in candidates)
                    candidates = [
                        item for item in candidates if item[3] == best_width]
                    if (possessive_focus and not extends_focus
                            and best_width == len(base_focus_tokens)):
                        # No stored relation extends the possessor's name
                        # ("user's favourite colour" asks about the user), so
                        # the entity that is exactly the focus is the subject;
                        # do not treat every "user's <relation>" record as
                        # ambiguous.
                        exact_subject = [
                            item for item in candidates
                            if focus_tokens(item[1].debug_subject)
                            == base_focus_tokens]
                        if exact_subject:
                            candidates = exact_subject
            if not candidates:
                document_exclusive = (
                    document_scoped > 0 and document_scoped == visible)
                return result(
                    "abstained",
                    ("document_scope_required" if document_exclusive
                     else "subject_not_found"),
                    [], 0, 0, document_scoped)
            if not joint and len({self._normalize_subject(slot.debug_subject)
                                  for _, slot, _, _ in candidates}) > 1:
                # A first name shared by two people cannot identify either.
                return result("abstained", "subject_ambiguous", [], 0,
                              len(candidates), document_scoped)

            eligible: List[Tuple[int, MemorySlot]] = []
            for idx, slot, _, _ in candidates:
                slot_attributes = set()
                for predicate in (slot.debug_predicate, slot.latest_predicate):
                    slot_attributes |= attribute_ids(predicate)
                if attributes_compatible(query_attributes, slot_attributes):
                    eligible.append((idx, slot))
            if not eligible:
                return result(
                    "abstained", "attribute_not_stored", [], 0,
                    len(candidates), document_scoped)

            q_dense = np.asarray(self.dense_encoder.encode(evidence_text),
                                 dtype=np.float32)
            ranked: List[Tuple[float, MemorySlot, int]] = []
            for idx, slot in eligible:
                emb = (self.slot_dense_embs[idx]
                       if idx < len(self.slot_dense_embs)
                       else slot.dense_emb)
                sim = float(np.dot(emb, q_dense)) if emb is not None else 0.0
                ranked.append((sim, slot, idx))
            ranked.sort(key=lambda item: -item[0])

            if turn_norm:
                kept: List[Tuple[float, MemorySlot, int]] = []
                for item in ranked:
                    text = (self.slot_texts.get(id(item[1]), item[1].text)
                            or "")
                    if self._contains_turn(text, turn_norm):
                        current_turn_excluded += 1
                        continue
                    kept.append(item)
                ranked = kept
                if not ranked:
                    # The only matching material is the caller's own turn.
                    no_evidence = {
                        "state": "insufficient",
                        "reason": "evidence_insufficient",
                        "sufficient": False,
                        "scope_kind": "none",
                        "supporting_memory_ids": [],
                        "summary": {},
                    }
                    return result(
                        "abstained", "evidence_insufficient", [],
                        len(eligible), len(candidates), document_scoped,
                        evidence=no_evidence)

            # Scope membership is not relevance.  A document-scoped subject
            # is a search boundary: the query's own content must occur in a
            # returned row before that row may be presented as evidence.  An
            # entity subject whose query adds no content beyond the entity is
            # established by the verified subject and attribute.
            if joint and len(scope_keys) > 1:
                # Cross-scope union support is meaningful only when more than
                # one conversation was requested; a one-item list must decide
                # exactly like the equivalent ``conversation_id`` request.
                by_scope: Dict[str, List[Dict[str, Any]]] = {
                    key: [] for key in scope_keys}
                for sim, slot, idx in ranked:
                    by_scope[self._normalize_subject(
                        slot.debug_subject)].append(row(slot, idx, sim))
                evidence = assess_joint_evidence(
                    evidence_text, original_focus,
                    [by_scope[key] for key in scope_keys],
                    explicit_scope=True)
            else:
                scope_kind = (
                    "document" if is_document_subject(
                        ranked[0][1].debug_subject) else "entity")
                evidence_focus = original_focus
                if (scope_kind == "document"
                        and not evidence_terms(original_focus or "")):
                    # A natural question frame ("what is the situation
                    # with ...") can defeat focus extraction.  The scope is
                    # already asserted, so the question's own content is the
                    # only evidence; requiring it as a whole keeps the same
                    # gate instead of silently dropping the focus branch.
                    evidence_focus = query_text
                if explicit and scope_kind == "entity" and not evidence_focus:
                    evidence_focus = subject
                scan_rows = [row(slot, idx, sim)
                             for sim, slot, idx
                             in ranked[:EVIDENCE_SCAN_LIMIT]]
                evidence = assess_evidence(
                    evidence_text, evidence_focus, scan_rows,
                    scope_kind=scope_kind, explicit_scope=explicit)
            if not evidence["sufficient"]:
                return result("abstained", evidence["reason"], [],
                              len(eligible), len(candidates),
                              document_scoped, evidence=evidence)

            # Keep supporting rows ahead of merely similar ones so the
            # evidence the verdict rests on is part of the returned window.
            supporting = set(evidence["supporting_memory_ids"])
            relation_ids: set = set()
            if query_attributes:
                # A relational question is verified from the whole scanned
                # material; the row that expresses the asked relation must
                # also be visible to the caller, even when its own content
                # overlap is too thin to count as a supporting row.
                for _, slot, _ in ranked:
                    text = self.slot_texts.get(id(slot), slot.text) or ""
                    if attribute_ids(text) & query_attributes:
                        relation_ids.add(slot.memory_id)
            if discovery_meta is not None:
                # Discovery selected these scopes by content match, so rows
                # that carry more of the question should lead the returned
                # window; dense similarity alone can bury a short, specific
                # message under generic conversation text.
                discovery_terms = evidence_terms(evidence_text)
                hit_counts = {
                    slot.memory_id: len(present_terms(
                        discovery_terms,
                        self.slot_texts.get(id(slot), slot.text)))
                    for _, slot, _ in ranked}
                ranked = sorted(
                    ranked,
                    key=lambda item: (
                        0 if item[1].memory_id in supporting else 1,
                        0 if item[1].memory_id in relation_ids else 1,
                        -hit_counts.get(item[1].memory_id, 0),
                        -item[0]))
            else:
                ranked = sorted(
                    ranked,
                    key=lambda item: (
                        0 if item[1].memory_id in supporting else 1,
                        0 if item[1].memory_id in relation_ids else 1,
                        -item[0]))
            rows = [row(slot, idx, sim)
                    for sim, slot, idx in ranked[:top_k]]

            if threshold is not None and (
                    not rows or rows[0]["similarity"] < threshold):
                # Report the score that failed the caller's threshold so the
                # threshold can be recalculated; no row is returned.
                gated = result("abstained", "low_similarity", [],
                               len(eligible), len(candidates),
                               document_scoped, evidence=evidence)
                gated["evidence"]["state"] = "insufficient"
                gated["evidence"]["reason"] = "low_similarity"
                gated["top_similarity"] = (
                    rows[0]["similarity"] if rows else None)
                return gated
            outcome = result("answered", None, rows, len(eligible),
                             len(candidates), document_scoped,
                             evidence=evidence)
            if outcome["temporal"]["conflict_status"] != "none":
                # Contradictory evidence is retained and reported, never
                # discarded or presented as settled.
                outcome["evidence"]["state"] = "contradicted"
            return outcome

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
                if slot.redacted:
                    continue
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
                    "memory_id": slot.memory_id,
                    "subject": slot.debug_subject,
                    "predicate": self._represented_predicate(slot),
                    "original_predicate": slot.debug_predicate,
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

    def delete_memory(self, memory_id: str) -> bool:
        """Deletes a slot by its stable ``memory_id`` instead of ``id(slot)``.

        ``delete_slot`` addresses slots by a process-local object identity that
        cannot be persisted or used across restarts; provenance records use the
        stable id so deletion remains correct after a checkpoint reload.
        """
        with self._lock:
            target = next(
                (s for s in self.memory.slots
                 if s.memory_id == memory_id), None)
            if target is None:
                return False
            return self.delete_slot(id(target))

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

    def redact_slot(self, slot_id: int, reason: Optional[str] = None) -> bool:
        """Irreversibly removes a slot's content while keeping its tombstone.

        The slot object stays in place so supersession links and index
        bookkeeping remain valid, but its key/value vectors, text, embeddings,
        subject/predicate labels, contradiction anchors, and BM25 document are
        destroyed.  Redacted slots are excluded from every read path and from
        the Qdrant index, and the checkpoint persists only the tombstone.

        ``reason`` is accepted for audit call sites but is deliberately not
        stored: recording a free-form reason could itself retain the personal
        data the caller asked to remove.
        """
        changed = False
        try:
            with self._lock:
                target = next(
                    (s for s in self.memory.slots if id(s) == slot_id), None)
                if target is None or target.redacted:
                    return False
                self._redact_slot(target)
                changed = True
        finally:
            if changed:
                self._sync_qdrant_if_active()
        return changed

    def redact_memory(self, memory_id: str, reason: Optional[str] = None) -> bool:
        """Redacts a slot by its stable ``memory_id`` instead of ``id(slot)``."""
        with self._lock:
            target = next(
                (s for s in self.memory.slots
                 if s.memory_id == memory_id), None)
            if target is None:
                return False
            return self.redact_slot(id(target), reason=reason)

    def apply_privacy_tombstones(
        self, memory_ids: Iterable[str], reason: Optional[str] = None
    ) -> List[str]:
        """Wipes restored slots covered by durable privacy intent.

        Recovery loads canonical state that predates a later redaction or
        deletion, so the affected slot may still be present.  Both intent kinds
        are applied as an in-place content-free tombstone: the slot remains
        structurally referenced by temporal links, but its text, vectors,
        labels and index entries are destroyed.  No derived-index sync happens
        here; the caller persists this sanitized state first and only then
        reconciles external storage.
        """
        wanted = {str(memory_id) for memory_id in memory_ids if memory_id}
        if not wanted:
            return []
        applied: List[str] = []
        with self._lock:
            targets = [
                slot for slot in self.memory.slots
                if slot.memory_id in wanted and not slot.redacted
            ]
            for slot in targets:
                self._redact_slot(slot)
                applied.append(slot.memory_id)
        return applied

    def redact_subject(self, subject: str, exact_match: bool = True,
                       reason: Optional[str] = None) -> int:
        """Redacts every slot belonging to ``subject``.

        Matching happens before any content is wiped.  ``exact_match=False``
        compares case-insensitively after stripping surrounding whitespace.
        Returns the number of slots redacted.
        """
        if not isinstance(subject, str) or not subject.strip():
            return 0
        match_target = subject.strip() if exact_match else subject.strip().lower()
        changed = 0
        try:
            with self._lock:
                targets = []
                for slot in self.memory.slots:
                    if slot.redacted:
                        continue
                    slot_subject = slot.debug_subject or ""
                    value = (slot_subject.strip() if exact_match
                             else slot_subject.strip().lower())
                    if value == match_target:
                        targets.append(slot)
                for slot in targets:
                    self._redact_slot(slot)
                    changed += 1
        finally:
            if changed:
                self._sync_qdrant_if_active()
        return changed

    def _redact_slot(self, slot: MemorySlot) -> None:
        """Destroys one slot's content in place.  Caller holds the lock."""
        self._invalidate_retrieval_cache()
        self._invalidate_near_duplicate_index()
        idx = self._slot_index(slot)
        slot.key = np.zeros_like(slot.key)
        slot.value = np.zeros_like(slot.value)
        slot.text = ""
        slot.debug_subject = None
        slot.debug_predicate = None
        slot.latest_predicate = None
        slot.latest_object = None
        slot.dense_emb = (
            np.zeros_like(slot.dense_emb)
            if slot.dense_emb is not None else None)
        slot.contra_count = 0
        slot.redacted = True

        self.slot_texts.pop(id(slot), None)
        if idx is not None and idx < len(self.slot_dense_embs):
            self.slot_dense_embs[idx] = np.zeros_like(
                self.slot_dense_embs[idx])
        self.slot_contra_counts.pop(id(slot), None)
        self.slot_contra_anchors.pop(id(slot), None)
        self.bm25_index.remove_document(id(slot))

    def purge_redacted(self) -> int:
        """Permanently removes redacted tombstones.  Returns the count."""
        with self._lock:
            ids = [id(s) for s in self.memory.slots if s.redacted]
            count = 0
            for slot_id in ids:
                if self._delete_slot_impl(slot_id):
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
            live = [s for s in self.memory.slots if not s.redacted]
            redacted = len(self.memory.slots) - len(live)
            active = sum(1 for s in live if s.is_active)
            provisional = sum(1 for s in live if s.update_count <= 1 and s.is_active)
            consolidated = sum(1 for s in live if s.update_count > 1 and s.is_active)
            superseded = sum(1 for s in live if not s.is_active)
            footprint = self.memory.footprint_bytes()

            return {
                "capacity": self.capacity,
                "total_slots": len(self.memory.slots),
                "live_slots": len(live),
                "redacted_slots": redacted,
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
                "total_suppressed_value_merges":
                    self.total_suppressed_value_merges,
                "total_near_miss_separations":
                    self.total_near_miss_separations,
                "total_incompatible_value_separations":
                    self.total_incompatible_value_separations,
                "total_unresolved_near_duplicates":
                    self.total_unresolved_near_duplicates,
                "strict_entity_identity": self.strict_entity_identity,
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
        from minemos_engine.checkpoint import save_checkpoint
        return save_checkpoint(self, path, create_backup=create_backup, allowed_base_dir=allowed_base_dir)

    def load_checkpoint(
        self,
        path: str,
        allow_backup_recovery: bool = True,
        allowed_base_dir: Optional[str] = None,
        sync_retrieval: bool = True
    ) -> "BlackHoleMemory":
        """Restores memory state and auxiliary indexes from a .npz archive."""
        from minemos_engine.checkpoint import load_into_black_hole_memory
        return load_into_black_hole_memory(
            self,
            path,
            allow_backup_recovery=allow_backup_recovery,
            allowed_base_dir=allowed_base_dir,
            sync_retrieval=sync_retrieval
        )

    @classmethod
    def from_checkpoint(cls, path: str, allowed_base_dir: Optional[str] = None, **kwargs) -> "BlackHoleMemory":
        """Constructs a new BlackHoleMemory instance initialized from a checkpoint."""
        instance = cls(**kwargs)
        instance.load_checkpoint(path, allowed_base_dir=allowed_base_dir)
        return instance
