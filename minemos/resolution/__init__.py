"""Semantic resolution layer: four deterministic stages in front of MINEMOS.

    raw message
      → Stage 1  normalize_text / detect_language
      → Stage 2  interpret
      → Stage 3  resolve_identity
      → Stage 4  classify_relationships
      → minemos.resolution/1 record

The layer is candidate-only.  It never writes canonical memory, never changes
retrieval and never consolidates: MINEMOS keeps every memory decision.  The
default mode is ``shadow``; handing candidates to the engine requires the
explicit ``active`` opt-in in :mod:`minemos.resolution.bridge`.
"""

from .bridge import (
    BridgeDecision,
    absorb_record,
    candidates_for,
    decide,
    eligibility_counts,
    subject_text,
)
from .config import ACTIVE_WARNING, DEFAULT_CONFIG, MODES, ResolutionConfig
from .contracts import (
    ATTRIBUTION_SCOPES,
    ATTRIBUTION_STATUSES,
    ENTITY_TYPES,
    RELATIONSHIP_LABELS,
    STAGE_NAMES,
    SUBJECT_KINDS,
    Attribution,
    Entity,
    Evidence,
    FactCandidate,
    IdentityTransition,
    LanguageInfo,
    LanguageSpan,
    NormalizedText,
    NormalizedToken,
    Provenance,
    Relationship,
    ResolutionContractError,
    ResolutionRecord,
    SourceMessage,
    SpeakerRef,
    SubjectRef,
    TemporalExpression,
    TextEdit,
    validate_record,
)
from .host import (
    HostShadowObserver,
    HostShadowReport,
    ShadowSettings,
    write_json_report,
)
from .identity import (
    IdentityRegistry,
    IdentityResult,
    IdentitySession,
    PersonRecord,
    resolve_identity,
)
from .ingest import (
    DEFAULT_POLICY,
    IngestedChunk,
    IngestedMessage,
    IngestionPolicy,
    MessageTooLongError,
    OVERFLOW_POLICIES,
    absorb_ingested,
    ingest_conversation,
    ingest_message,
)
from .interpret import Interpretation, interpret
from .language import detect_language
from .native import (
    NativeConversion,
    NativeConversionError,
    NativeObservation,
    RejectedFact,
    native_identifier,
    native_ingestion_id,
    native_observations,
    native_observations_many,
    resolve_timestamp,
)
from .normalize import normalize_text
from .pipeline import ResolutionResult, Resolver, resolve_messages
from .relationships import (
    FactIndex,
    IndexedFact,
    RelationshipRegistry,
    RelationshipResult,
    classify_relationships,
)
from .schema import SCHEMA, validate_with_schema
from .segments import MAX_MESSAGE_CHARS, MessageChunk, segment_message
from .shadow import ShadowReport, ShadowRunner
from .similarity import build_similarity

__all__ = [
    "ACTIVE_WARNING",
    "ATTRIBUTION_SCOPES",
    "ATTRIBUTION_STATUSES",
    "Attribution",
    "BridgeDecision",
    "DEFAULT_CONFIG",
    "DEFAULT_POLICY",
    "ENTITY_TYPES",
    "Entity",
    "Evidence",
    "FactCandidate",
    "FactIndex",
    "HostShadowObserver",
    "HostShadowReport",
    "IdentityRegistry",
    "IdentityResult",
    "IdentitySession",
    "IdentityTransition",
    "IndexedFact",
    "IngestedChunk",
    "IngestedMessage",
    "IngestionPolicy",
    "Interpretation",
    "LanguageInfo",
    "LanguageSpan",
    "MAX_MESSAGE_CHARS",
    "MODES",
    "MessageChunk",
    "MessageTooLongError",
    "NativeConversion",
    "NativeConversionError",
    "NativeObservation",
    "NormalizedText",
    "NormalizedToken",
    "OVERFLOW_POLICIES",
    "PersonRecord",
    "Provenance",
    "RELATIONSHIP_LABELS",
    "RejectedFact",
    "Relationship",
    "RelationshipRegistry",
    "RelationshipResult",
    "ResolutionConfig",
    "ResolutionContractError",
    "ResolutionRecord",
    "ResolutionResult",
    "Resolver",
    "SCHEMA",
    "STAGE_NAMES",
    "SUBJECT_KINDS",
    "ShadowReport",
    "ShadowRunner",
    "ShadowSettings",
    "SourceMessage",
    "SpeakerRef",
    "SubjectRef",
    "TemporalExpression",
    "TextEdit",
    "absorb_ingested",
    "absorb_record",
    "build_similarity",
    "candidates_for",
    "classify_relationships",
    "decide",
    "detect_language",
    "eligibility_counts",
    "ingest_conversation",
    "ingest_message",
    "interpret",
    "native_identifier",
    "native_ingestion_id",
    "native_observations",
    "native_observations_many",
    "normalize_text",
    "resolve_identity",
    "resolve_messages",
    "resolve_timestamp",
    "segment_message",
    "subject_text",
    "validate_record",
    "validate_with_schema",
    "write_json_report",
]
