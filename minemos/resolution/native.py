"""Conversion of resolution records into native MINEMOS observations.

This module is the V8 preprocessing boundary.  It turns a
:class:`~minemos.resolution.contracts.ResolutionRecord` into values that the
existing native ingestion contract accepts — ``Candidate`` objects for the
engine and plain observation dictionaries for the service — without changing
that contract, without writing memory, and without inventing native fields.

Only facts that pass the existing bridge eligibility rules are converted;
everything else is returned with an explicit reason.  The full record remains
the audit source for evidence spans, polarity, modality, temporal expressions,
language, entities, attributions and transitions, none of which the native
representation can store.

Timestamp policy (never ``0.0``):

1. the semantic observation timestamp (``record.source.timestamp``);
2. the original conversation segment timestamp supplied by the host;
3. the ingestion timestamp supplied by the host, or ``time.time()``.

Because a missing timestamp makes a replay non-identical, hosts must pass a
stable ingestion timestamp (for example the run start) and should pass the
segment timestamp whenever the source carries one.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .bridge import BridgeDecision, decide, subject_text
from .config import ResolutionConfig
from .contracts import FactCandidate, ResolutionRecord

# Mirrors of the native service contract
# (``minemos_service.security``); tests assert they stay in sync.
NATIVE_SUBJECT_MAX_CHARS = 256
NATIVE_PREDICATE_MAX_CHARS = 128
NATIVE_OBJECT_MAX_CHARS = 8192
NATIVE_ID_MAX_CHARS = 128

_NATIVE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_UNSAFE_ID_RE = re.compile(r"[^A-Za-z0-9._:-]+")

TIMESTAMP_SEMANTIC = "semantic"
TIMESTAMP_SEGMENT = "segment"
TIMESTAMP_INGESTION = "ingestion"


class NativeConversionError(ValueError):
    """A record cannot be converted into native observations."""


@dataclass(frozen=True)
class NativeObservation:
    """One native observation plus the identity needed to trace it back."""

    fact_id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    timestamp: float
    timestamp_source: str
    ingestion_id: str
    source_id: str
    conversation_id: str
    message_id: str

    def as_candidate(self):
        """Engine-native structured observation."""
        from minemos_engine.structured_candidate import Candidate

        return Candidate(
            subject=self.subject,
            predicate=self.predicate,
            object=self.object,
            confidence=self.confidence,
            timestamp=self.timestamp,
        )

    def to_service_observation(self) -> Dict[str, Any]:
        """Observation mapping accepted by ``validate_ingest_payload``."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "ingestion_id": self.ingestion_id,
            "source": {
                "source_id": self.source_id,
                "conversation_id": self.conversation_id,
            },
        }


@dataclass(frozen=True)
class RejectedFact:
    fact_id: str
    reason: str


@dataclass(frozen=True)
class NativeConversion:
    """Outcome of converting one record."""

    observations: Tuple[NativeObservation, ...]
    rejected: Tuple[RejectedFact, ...]
    timestamp_source: Optional[str] = None

    def decisions(self) -> Tuple[BridgeDecision, ...]:
        return tuple(
            [BridgeDecision(item.fact_id, True, "eligible")
             for item in self.observations]
            + [BridgeDecision(item.fact_id, False, item.reason)
               for item in self.rejected]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Content-free summary safe for logs and reports."""
        return {
            "observations": [{
                "fact_id": item.fact_id,
                "ingestion_id": item.ingestion_id,
                "timestamp_source": item.timestamp_source,
            } for item in self.observations],
            "rejected": [
                {"fact_id": item.fact_id, "reason": item.reason}
                for item in self.rejected
            ],
            "timestamp_source": self.timestamp_source,
        }


def native_identifier(value: str) -> str:
    """Deterministic identifier that satisfies the native contract.

    Values that already match are returned unchanged, so an existing native
    identifier stays traceable.  Anything else is sanitized and suffixed with
    a digest of the full original value, which keeps different inputs distinct
    even when their sanitized prefixes are identical or truncated.
    """
    if not isinstance(value, str):
        raise TypeError("identifier must be a string")
    if not value:
        raise ValueError("identifier cannot be empty")
    if len(value) <= NATIVE_ID_MAX_CHARS and _NATIVE_ID_RE.match(value):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    slug = _UNSAFE_ID_RE.sub("_", value)
    while slug and not re.match(r"[A-Za-z0-9]", slug[0]):
        slug = slug[1:]
    slug = slug.rstrip("_-.")
    budget = NATIVE_ID_MAX_CHARS - len(digest) - 1
    slug = slug[:budget] or "id"
    return f"{slug}-{digest}"


def _valid_timestamp(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def resolve_timestamp(record_timestamp: Optional[float],
                      segment_timestamp: Optional[float] = None,
                      ingestion_timestamp: Optional[float] = None
                      ) -> Tuple[float, str]:
    """Applies the documented fallback order; never returns ``0.0``."""
    for value, source in (
            (record_timestamp, TIMESTAMP_SEMANTIC),
            (segment_timestamp, TIMESTAMP_SEGMENT),
            (ingestion_timestamp, TIMESTAMP_INGESTION)):
        number = _valid_timestamp(value)
        if number is not None:
            return number, source
    raise NativeConversionError(
        "no valid timestamp: semantic, segment and ingestion values are all "
        "missing, non-finite or non-positive")


def native_ingestion_id(*, source_id: str, conversation_id: str,
                        message_id: str, fact_id: str, subject: str,
                        predicate: str, object_text: str, timestamp: float,
                        confidence: float) -> str:
    """Deterministic id derived from source identity and fact content."""
    payload = {
        "source_id": source_id,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "fact_id": fact_id,
        "subject": subject,
        "predicate": predicate,
        "object_sha256": hashlib.sha256(
            object_text.encode("utf-8")).hexdigest(),
        "timestamp": float(timestamp),
        "confidence": float(confidence),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _field_reason(value: str, name: str, limit: int) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return f"empty_{name}"
    if len(value) > limit:
        return f"{name}_too_long"
    if _CONTROL_RE.search(value):
        return f"{name}_control_characters"
    return None


def native_observations(record: ResolutionRecord,
                        config: Optional[ResolutionConfig] = None, *,
                        source_id: Optional[str] = None,
                        segment_timestamp: Optional[float] = None,
                        ingestion_timestamp: Optional[float] = None,
                        conversation_id: Optional[str] = None,
                        message_id: Optional[str] = None,
                        ingestion_fallback: bool = True,
                        ) -> NativeConversion:
    """Converts the eligible facts of one record into native observations.

    No memory is written and no engine object is needed.  The caller decides
    where the values go: :meth:`NativeObservation.as_candidate` for the engine
    or :meth:`NativeObservation.to_service_observation` for the service.

    ``ingestion_fallback`` controls the last timestamp step only.  When it is
    false and neither the record nor the segment carries a usable event
    timestamp, every otherwise-eligible fact is rejected with
    ``timestamp_unavailable`` instead of being stamped with ingestion time.
    """
    cfg = config or ResolutionConfig()
    conversation = native_identifier(
        conversation_id or record.source.conversation_id)
    message = native_identifier(message_id or record.source.message_id)
    source = native_identifier(source_id or conversation)

    decisions: list = []
    accepted: list = []
    seen: set = set()
    for fact in record.facts:
        decision = decide(record, fact, cfg)
        reason = decision.reason
        subject = subject_text(fact.subject, record.source.speaker)
        if decision.accepted:
            key = (str(subject).strip().casefold(),
                   str(fact.predicate).strip().casefold(),
                   str(fact.value).strip().casefold())
            if fact.confidence < cfg.extraction_confidence:
                reason = "below_confidence_threshold"
            elif key in seen:
                reason = "duplicate_candidate"
            elif subject is None:
                reason = "subject_unresolved"
        if not decision.accepted or reason != "eligible":
            decisions.append(RejectedFact(fact.fact_id, reason))
            continue
        reason = _field_reason(subject, "subject",
                               NATIVE_SUBJECT_MAX_CHARS)
        if reason is None:
            reason = _field_reason(fact.predicate, "predicate",
                                   NATIVE_PREDICATE_MAX_CHARS)
        if reason is None:
            reason = _field_reason(fact.value, "object",
                                   NATIVE_OBJECT_MAX_CHARS)
        if reason is not None:
            decisions.append(RejectedFact(fact.fact_id, reason))
            continue
        seen.add((str(subject).strip().casefold(),
                  str(fact.predicate).strip().casefold(),
                  str(fact.value).strip().casefold()))
        accepted.append((fact, subject))

    if not accepted:
        return NativeConversion((), tuple(decisions))

    if not ingestion_fallback and (
            _valid_timestamp(record.source.timestamp) is None
            and _valid_timestamp(segment_timestamp) is None):
        rejected = tuple(
            RejectedFact(fact.fact_id, "timestamp_unavailable")
            for fact, _subject in accepted
        )
        return NativeConversion((), tuple(decisions) + rejected)

    try:
        timestamp, stamp_source = resolve_timestamp(
            record.source.timestamp, segment_timestamp, ingestion_timestamp)
    except NativeConversionError:
        rejected = tuple(
            RejectedFact(fact.fact_id, "invalid_timestamp")
            for fact, _subject in accepted
        )
        return NativeConversion((), tuple(decisions) + rejected)

    observations: list = []
    for fact, subject in accepted:
        confidence = round(float(fact.confidence), 4)
        observations.append(NativeObservation(
            fact_id=fact.fact_id,
            subject=subject,
            predicate=fact.predicate,
            object=fact.value,
            confidence=confidence,
            timestamp=timestamp,
            timestamp_source=stamp_source,
            ingestion_id=native_ingestion_id(
                source_id=source, conversation_id=conversation,
                message_id=message, fact_id=fact.fact_id, subject=subject,
                predicate=fact.predicate, object_text=fact.value,
                timestamp=timestamp, confidence=confidence),
            source_id=source,
            conversation_id=conversation,
            message_id=message,
        ))
    return NativeConversion(tuple(observations), tuple(decisions),
                            stamp_source)


def native_observations_many(records, config: Optional[ResolutionConfig] = None,
                             *, source_id: Optional[str] = None,
                             **kwargs) -> Tuple[list, list]:
    """Converts many records; returns ``(observations, conversions)``."""
    observations: list = []
    conversions: list = []
    for record in records:
        conversion = native_observations(
            record, config, source_id=source_id, **kwargs)
        conversions.append(conversion)
        observations.extend(conversion.observations)
    return observations, conversions


__all__ = [
    "NATIVE_ID_MAX_CHARS",
    "NATIVE_OBJECT_MAX_CHARS",
    "NATIVE_PREDICATE_MAX_CHARS",
    "NATIVE_SUBJECT_MAX_CHARS",
    "NativeConversion",
    "NativeConversionError",
    "NativeObservation",
    "RejectedFact",
    "TIMESTAMP_INGESTION",
    "TIMESTAMP_SEGMENT",
    "TIMESTAMP_SEMANTIC",
    "native_identifier",
    "native_ingestion_id",
    "native_observations",
    "native_observations_many",
    "resolve_timestamp",
]
