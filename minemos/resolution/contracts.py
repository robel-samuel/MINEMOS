"""Structured contracts for the semantic resolution layer.

Every value that leaves the layer is one of these immutable records.  The
contracts carry the original message text, the normalized form, the language
view, extracted entities and fact candidates, speaker/subject attribution,
source provenance, relationship classifications, per-stage confidence,
ambiguities and abstention reasons.

Two invariants are enforced by :func:`validate_record`:

* the original text is never replaced -- ``normalized.original`` must equal
  ``record.original_text`` and every evidence span must fall inside it;
* every fact is traceable to the message that produced it -- evidence carries
  the source message and conversation identifiers.

The records are candidates.  Nothing here writes canonical memory, changes a
validity window, consolidates a value or evicts a slot; those decisions stay
in :mod:`minemos_engine`.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

RECORD_VERSION = "minemos.resolution/1"
SCHEMA_VERSION = "1.0"

RELATIONSHIP_LABELS = (
    "duplicate",
    "temporal_update",
    "historical_observation",
    "distinct_fact",
    "unresolved",
)

ATTRIBUTION_STATUSES = ("resolved", "ambiguous", "unknown")

ATTRIBUTION_SCOPES = ("message", "conversation", "cross_conversation")

IDENTITY_TRANSITION_KINDS = (
    "relation_linked",
    "speaker_change",
    "name_change",
    "ownership_change",
)

SUBJECT_KINDS = ("speaker", "named", "derived", "unknown")

ENTITY_TYPES = (
    "person",
    "place",
    "organization",
    "product",
    "language",
    "date",
    "number",
    "unknown",
)

STAGE_NAMES = ("normalization", "interpretation", "identity", "relationships")

TEMPORAL_KINDS = (
    "absolute",
    "range",
    "relative",
    "now",
    "past",
    "future",
    "duration",
    "unknown",
)

_PREDICATE = re.compile(r"^[a-z][a-z0-9_]*$")


class ResolutionContractError(ValueError):
    """A resolution record violates the published output contract."""


@dataclass(frozen=True)
class SourceMessage:
    """One conversational message as supplied by the host application.

    ``text`` is retained exactly.  ``speaker`` is the host's own identity for
    the author (``"user"``, ``"assistant"`` or an application-specific id); it
    is never inferred from the message body.
    """

    conversation_id: str
    message_id: str
    text: str
    speaker: Optional[str] = None
    timestamp: Optional[float] = None
    order: int = 0
    message_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("SourceMessage.text must be a non-empty string")
        if not self.conversation_id:
            raise ValueError("SourceMessage.conversation_id is required")
        if not self.message_id:
            raise ValueError("SourceMessage.message_id is required")

    @property
    def provenance_key(self) -> str:
        return self.message_key or self.message_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "message_key": self.message_key,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "order": self.order,
            "text": self.text,
        }


@dataclass(frozen=True)
class NormalizedToken:
    """One normalized token and the source characters it came from."""

    text: str
    source_start: int
    source_end: int
    edited: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "edited": self.edited,
        }


@dataclass(frozen=True)
class TextEdit:
    """A reversible description of one normalization decision.

    ``reverse`` restores the original spelling at the recorded span, so a
    consumer can reconstruct the source tokens without storing a second copy.
    """

    kind: str
    source_start: int
    source_end: int
    original: str
    replacement: str
    confidence: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "original": self.original,
            "replacement": self.replacement,
            "confidence": round(self.confidence, 4),
            "reverse": {
                "source_start": self.source_start,
                "source_end": self.source_end,
                "text": self.original,
            },
            "reason": self.reason,
        }


_NO_SPACE_BEFORE = set(".,!?;:%)]}>'\"”’…።፣")
_NO_SPACE_AFTER = set("([{<\"'“‘")


def render_tokens(texts: Sequence[str]) -> str:
    """Renders normalized tokens with punctuation kept adjacent to words."""
    parts: list = []
    previous = ""
    for text in texts:
        if not parts:
            parts.append(text)
        elif text and text[0] in _NO_SPACE_BEFORE:
            parts.append(text)
        elif previous and previous[-1] in _NO_SPACE_AFTER:
            parts.append(text)
        else:
            parts.append(" " + text)
        previous = text
    return "".join(parts)


@dataclass(frozen=True)
class _TokenAlignment:
    """Precomputed normalized offsets for :meth:`NormalizedText.span_to_source`.

    Token layouts are built once per message, so mapping one of ``n`` value
    spans stays ``O(log n)`` instead of rescanning every token per span.
    """

    starts: Tuple[int, ...]
    ends: Tuple[int, ...]

    @classmethod
    def build(cls, tokens: Sequence["NormalizedToken"]) -> "_TokenAlignment":
        starts = []
        ends = []
        cursor = 0
        previous = ""
        for token in tokens:
            if not previous:
                token_start = 0
            elif token.text and token.text[0] in _NO_SPACE_BEFORE:
                token_start = cursor
            elif previous and previous[-1] in _NO_SPACE_AFTER:
                token_start = cursor
            else:
                token_start = cursor + 1
            token_end = token_start + len(token.text)
            cursor = token_end
            previous = token.text
            starts.append(token_start)
            ends.append(token_end)
        return cls(tuple(starts), tuple(ends))


@dataclass(frozen=True)
class NormalizedText:
    """Normalized text plus its alignment back to the original message.

    ``original`` is copied unchanged.  Mapping a normalized span to source
    characters goes through the token alignment, so a corrected typo still
    points at the original word rather than at invented characters.
    """

    original: str
    text: str
    tokens: Tuple[NormalizedToken, ...]
    edits: Tuple[TextEdit, ...] = ()
    sentence_terminated: bool = False
    ambiguities: Tuple[str, ...] = ()

    @cached_property
    def _alignment(self) -> _TokenAlignment:
        return _TokenAlignment.build(self.tokens)

    def span_to_source(self, start: int, end: int) -> Tuple[int, int]:
        """Maps a normalized character span to the original message."""
        if start >= end:
            raise ValueError("span must be non-empty")
        alignment = self._alignment
        # First token whose normalized end passes ``start``.
        first = bisect_right(alignment.ends, start)
        # Last token whose normalized start precedes ``end``.
        last = bisect_left(alignment.starts, end) - 1
        if first >= len(self.tokens):
            mapped_start = len(self.original)
        else:
            mapped_start = self.tokens[first].source_start
        if last < 0:
            mapped_end = len(self.original)
        else:
            mapped_end = self.tokens[last].source_end
        return mapped_start, max(mapped_end, mapped_start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "text": self.text,
            "tokens": [token.to_dict() for token in self.tokens],
            "edits": [edit.to_dict() for edit in self.edits],
            "sentence_terminated": self.sentence_terminated,
            "ambiguities": list(self.ambiguities),
            "reversible": True,
        }


@dataclass(frozen=True)
class LanguageSpan:
    start: int
    end: int
    language: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "language": self.language,
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class LanguageInfo:
    primary: str
    confidence: float
    mixed: bool = False
    alternates: Tuple[str, ...] = ()
    spans: Tuple[LanguageSpan, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary": self.primary,
            "confidence": round(self.confidence, 4),
            "mixed": self.mixed,
            "alternates": list(self.alternates),
            "spans": [span.to_dict() for span in self.spans],
        }


@dataclass(frozen=True)
class Entity:
    text: str
    type: str
    span: Tuple[int, int]
    confidence: float
    rule: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "span": list(self.span),
            "confidence": round(self.confidence, 4),
            "rule": self.rule,
        }


@dataclass(frozen=True)
class TemporalExpression:
    text: str
    kind: str
    span: Tuple[int, int]
    year: Optional[int] = None
    confidence: float = 0.0
    rule: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "span": list(self.span),
            "year": self.year,
            "confidence": round(self.confidence, 4),
            "rule": self.rule,
        }


@dataclass(frozen=True)
class SubjectRef:
    """Who or what a fact is about.

    ``kind`` is ``speaker`` (first person), ``named`` (an explicit name or a
    possessive-derived person), ``derived`` (``user's brother``) or
    ``unknown`` when the evidence does not identify the subject.

    Pronoun resolutions name the antecedent message and fact by id only, never
    by copying its text, so a record stays safe to log.
    """

    kind: str
    label: str
    confidence: float
    owner: Optional[str] = None
    relation: Optional[str] = None
    entity_key: Optional[str] = None
    resolved_from: str = ""
    status: str = "resolved"
    ambiguity: Optional[str] = None
    antecedent_message_id: Optional[str] = None
    antecedent_fact_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "owner": self.owner,
            "relation": self.relation,
            "entity_key": self.entity_key,
            "resolved_from": self.resolved_from,
            "status": self.status,
            "ambiguity": self.ambiguity,
            "antecedent_message_id": self.antecedent_message_id,
            "antecedent_fact_id": self.antecedent_fact_id,
        }


@dataclass(frozen=True)
class SpeakerRef:
    id: str
    label: str
    confidence: float
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }


@dataclass(frozen=True)
class Attribution:
    """Speaker and subject resolution for one candidate fact.

    ``evidence`` and ``referenced`` hold structural identifiers only — rule
    names, message ids and entity keys — so attribution can be logged without
    copying conversation content.
    """

    speaker: SpeakerRef
    subject: SubjectRef
    status: str
    evidence: Tuple[str, ...] = ()
    fact_id: Optional[str] = None
    scope: str = "conversation"
    referenced: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker.to_dict(),
            "subject": self.subject.to_dict(),
            "status": self.status,
            "evidence": list(self.evidence),
            "fact_id": self.fact_id,
            "scope": self.scope,
            "referenced": list(self.referenced),
        }


@dataclass(frozen=True)
class Evidence:
    message_id: str
    conversation_id: str
    start: int
    end: int
    text: str
    rule: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "rule": self.rule,
        }


@dataclass(frozen=True)
class FactCandidate:
    fact_id: str
    subject: SubjectRef
    predicate: str
    value: str
    value_span: Tuple[int, int]
    confidence: float
    evidence: Tuple[Evidence, ...]
    entities: Tuple[Entity, ...] = ()
    temporal: Tuple[TemporalExpression, ...] = ()
    polarity: str = "positive"
    modality: str = "asserted"
    rule: str = ""
    language: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject.to_dict(),
            "predicate": self.predicate,
            "value": self.value,
            "value_span": list(self.value_span),
            "confidence": round(self.confidence, 4),
            "evidence": [item.to_dict() for item in self.evidence],
            "entities": [entity.to_dict() for entity in self.entities],
            "temporal": [expression.to_dict() for expression in self.temporal],
            "polarity": self.polarity,
            "modality": self.modality,
            "rule": self.rule,
            "language": self.language,
        }


@dataclass(frozen=True)
class Provenance:
    conversation_id: str
    message_id: str
    message_key: Optional[str]
    speaker: Optional[str]
    timestamp: Optional[float]
    spans: Tuple[Tuple[int, int], ...] = ()
    channel: str = "message"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "message_key": self.message_key,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "spans": [list(span) for span in self.spans],
            "channel": self.channel,
        }


@dataclass(frozen=True)
class Relationship:
    """A relationship between a candidate fact and an earlier candidate.

    ``evidence`` holds structural signals only (for example
    ``"value_edit_distance=1"``), never the values themselves, so the record
    can be logged or diffed without copying conversation content.
    """

    label: str
    confidence: float
    target_fact_id: Optional[str] = None
    evidence: Tuple[str, ...] = ()
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "target_fact_id": self.target_fact_id,
            "evidence": list(self.evidence),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class IdentityTransition:
    """A change in how an identity is referenced, anchored to one message.

    Transitions are appended, never applied retroactively: facts recorded
    before the transition keep the subject they were resolved to at the time.
    """

    kind: str
    message_id: str
    subject_key: Optional[str] = None
    evidence: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "message_id": self.message_id,
            "subject_key": self.subject_key,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ResolutionRecord:
    source: Provenance
    original_text: str
    normalized: NormalizedText
    language: LanguageInfo
    entities: Tuple[Entity, ...]
    facts: Tuple[FactCandidate, ...]
    attributions: Tuple[Attribution, ...]
    relationships: Tuple[Relationship, ...]
    stage_confidence: Mapping[str, float]
    ambiguities: Tuple[str, ...] = ()
    abstentions: Tuple[str, ...] = ()
    transitions: Tuple[IdentityTransition, ...] = ()
    provider: str = "deterministic"
    version: str = RECORD_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source.to_dict(),
            "original_text": self.original_text,
            "normalized": self.normalized.to_dict(),
            "language": self.language.to_dict(),
            "entities": [entity.to_dict() for entity in self.entities],
            "facts": [fact.to_dict() for fact in self.facts],
            "attributions": [
                attribution.to_dict() for attribution in self.attributions
            ],
            "relationships": [
                relationship.to_dict() for relationship in self.relationships
            ],
            "stage_confidence": {
                key: round(float(value), 4)
                for key, value in self.stage_confidence.items()
            },
            "ambiguities": list(self.ambiguities),
            "abstentions": list(self.abstentions),
            "transitions": [
                transition.to_dict() for transition in self.transitions
            ],
            "provider": self.provider,
        }


def _check_confidence(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ResolutionContractError(f"{name} must be numeric")
    if not math.isfinite(value) or not 0.0 <= float(value) <= 1.0:
        raise ResolutionContractError(f"{name} must be within [0, 1]")


def _check_span(span: Sequence[int], limit: int, name: str) -> None:
    if len(span) != 2:
        raise ResolutionContractError(f"{name} must be a (start, end) pair")
    start, end = int(span[0]), int(span[1])
    if start < 0 or end < start or end > limit:
        raise ResolutionContractError(
            f"{name} {span} is outside the source message (0..{limit})")


def validate_record(record: ResolutionRecord,
                    earlier_fact_ids: Iterable[str] = ()) -> None:
    """Raises :class:`ResolutionContractError` when a record is malformed.

    ``earlier_fact_ids`` lets a record link a relationship to a fact that was
    resolved in an earlier message of the same conversation.
    """
    if not isinstance(record, ResolutionRecord):
        raise ResolutionContractError("expected a ResolutionRecord")
    known_fact_ids = set(earlier_fact_ids)
    source_limit = len(record.original_text)
    if record.normalized.original != record.original_text:
        raise ResolutionContractError(
            "normalized.original must be the untouched source text")
    if not record.original_text.strip():
        raise ResolutionContractError("original_text is empty")

    last_start = -1
    for token in record.normalized.tokens:
        if not token.text:
            raise ResolutionContractError("normalized token is empty")
        _check_span((token.source_start, token.source_end), source_limit,
                    "normalized token")
        if token.source_start < last_start:
            raise ResolutionContractError(
                "normalized tokens must be ordered by source offset")
        last_start = token.source_start
    for edit in record.normalized.edits:
        _check_span((edit.source_start, edit.source_end), source_limit, "edit")
        _check_confidence(edit.confidence, "edit confidence")
    if record.normalized.text != render_tokens(
            [token.text for token in record.normalized.tokens]):
        raise ResolutionContractError(
            "normalized.text must match the token rendering")
    # Entity, fact and temporal spans are expressed in normalized-text
    # coordinates; evidence spans stay in original-message coordinates.
    limit = len(record.normalized.text)

    _check_confidence(record.language.confidence, "language confidence")
    for span in record.language.spans:
        _check_span((span.start, span.end), len(record.normalized.text),
                    "language span")
        _check_confidence(span.confidence, "language span confidence")

    if not record.attributions:
        raise ResolutionContractError(
            "every record needs at least one attribution entry")
    attributed: set = set()
    for attribution in record.attributions:
        _check_confidence(attribution.speaker.confidence, "speaker confidence")
        _check_confidence(attribution.subject.confidence, "subject confidence")
        if attribution.status not in ATTRIBUTION_STATUSES:
            raise ResolutionContractError(
                f"attribution.status {attribution.status!r} is not allowed")
        if attribution.subject.kind not in SUBJECT_KINDS:
            raise ResolutionContractError(
                f"subject.kind {attribution.subject.kind!r} is not allowed")
        if attribution.subject.status not in ATTRIBUTION_STATUSES:
            raise ResolutionContractError(
                f"subject.status {attribution.subject.status!r} is not allowed")
        if attribution.status != attribution.subject.status:
            raise ResolutionContractError(
                "attribution.status must match the subject status")
        if attribution.status == "unknown" and \
                attribution.subject.kind != "unknown":
            raise ResolutionContractError(
                "an unknown attribution must carry an unknown subject kind")
        if attribution.status == "ambiguous" and not (
                attribution.subject.ambiguity or attribution.evidence):
            raise ResolutionContractError(
                "an ambiguous attribution needs an ambiguity or evidence entry")
        if attribution.scope not in ATTRIBUTION_SCOPES:
            raise ResolutionContractError(
                f"attribution.scope {attribution.scope!r} is not allowed")
        if attribution.fact_id is not None:
            if attribution.fact_id in attributed:
                raise ResolutionContractError(
                    f"fact {attribution.fact_id!r} has two attributions")
            attributed.add(attribution.fact_id)
        for reason in attribution.evidence:
            if not reason:
                raise ResolutionContractError(
                    "attribution evidence reasons cannot be empty")

    seen_ids = set()
    for entity in record.entities:
        if entity.type not in ENTITY_TYPES:
            raise ResolutionContractError(
                f"entity.type {entity.type!r} is not allowed")
        _check_span(entity.span, limit, "entity span")
        _check_confidence(entity.confidence, "entity confidence")

    for fact in record.facts:
        if fact.fact_id in seen_ids:
            raise ResolutionContractError(
                f"duplicate fact id {fact.fact_id!r}")
        seen_ids.add(fact.fact_id)
        if not _PREDICATE.match(fact.predicate):
            raise ResolutionContractError(
                f"predicate {fact.predicate!r} must be snake_case")
        if not fact.value.strip():
            raise ResolutionContractError("fact value is empty")
        _check_span(fact.value_span, limit, "fact value span")
        _check_confidence(fact.confidence, "fact confidence")
        if not fact.evidence:
            raise ResolutionContractError(
                "every fact must carry at least one evidence span")
        for item in fact.evidence:
            _check_span((item.start, item.end), source_limit,
                        "evidence span")
            if item.message_id != record.source.message_id:
                raise ResolutionContractError(
                    "fact evidence must come from the recorded message")
            if item.conversation_id != record.source.conversation_id:
                raise ResolutionContractError(
                    "fact evidence must stay inside the recorded conversation")
        if fact.subject.status not in ATTRIBUTION_STATUSES:
            raise ResolutionContractError(
                f"fact subject status {fact.subject.status!r} is not allowed")
        _check_confidence(fact.subject.confidence, "fact subject confidence")
        for expression in fact.temporal:
            if expression.kind not in TEMPORAL_KINDS:
                raise ResolutionContractError(
                    f"temporal kind {expression.kind!r} is not allowed")
            _check_span(expression.span, limit, "temporal span")
            _check_confidence(expression.confidence, "temporal confidence")
        if fact.polarity not in ("positive", "negative"):
            raise ResolutionContractError(
                f"polarity {fact.polarity!r} is not allowed")

    if seen_ids - attributed:
        raise ResolutionContractError(
            "every fact needs an attribution entry; missing "
            f"{sorted(seen_ids - attributed)}")
    if attributed - seen_ids:
        raise ResolutionContractError(
            "attributions must name a fact in the record; unknown "
            f"{sorted(attributed - seen_ids)}")
    if seen_ids and any(item.fact_id is None
                        for item in record.attributions):
        raise ResolutionContractError(
            "a message-level attribution cannot accompany per-fact ones")

    for relationship in record.relationships:
        if relationship.label not in RELATIONSHIP_LABELS:
            raise ResolutionContractError(
                f"relationship {relationship.label!r} is not allowed")
        _check_confidence(relationship.confidence,
                          "relationship confidence")
        if relationship.target_fact_id is not None and (
                relationship.target_fact_id not in seen_ids
                and relationship.target_fact_id not in known_fact_ids):
            raise ResolutionContractError(
                "relationship target_fact_id must name a fact in the record "
                "or an earlier fact supplied by the resolver")
        if relationship.label == "unresolved" and not relationship.reason:
            raise ResolutionContractError(
                "an unresolved relationship must record a reason")

    unknown = set(record.stage_confidence) - set(STAGE_NAMES)
    if unknown:
        raise ResolutionContractError(
            f"unknown stage confidence keys: {sorted(unknown)}")
    for name, value in record.stage_confidence.items():
        _check_confidence(value, f"stage confidence {name}")
    for reason in record.ambiguities:
        if not reason:
            raise ResolutionContractError("ambiguity reasons cannot be empty")
    for reason in record.abstentions:
        if not reason:
            raise ResolutionContractError("abstention reasons cannot be empty")
    for transition in record.transitions:
        if transition.kind not in IDENTITY_TRANSITION_KINDS:
            raise ResolutionContractError(
                f"transition kind {transition.kind!r} is not allowed")
        if not transition.message_id:
            raise ResolutionContractError(
                "a transition must name the message it was observed in")


def relationships_by_label(
        relationships: Iterable[Relationship]) -> Dict[str, Tuple[Relationship, ...]]:
    grouped: Dict[str, list] = {}
    for relationship in relationships:
        grouped.setdefault(relationship.label, []).append(relationship)
    return {label: tuple(items) for label, items in grouped.items()}


__all__ = [
    "ATTRIBUTION_SCOPES",
    "ATTRIBUTION_STATUSES",
    "ENTITY_TYPES",
    "Evidence",
    "FactCandidate",
    "LanguageInfo",
    "LanguageSpan",
    "NormalizedText",
    "NormalizedToken",
    "Provenance",
    "RECORD_VERSION",
    "RELATIONSHIP_LABELS",
    "Relationship",
    "ResolutionContractError",
    "ResolutionRecord",
    "SCHEMA_VERSION",
    "STAGE_NAMES",
    "SUBJECT_KINDS",
    "SourceMessage",
    "SpeakerRef",
    "SubjectRef",
    "TEMPORAL_KINDS",
    "TemporalExpression",
    "TextEdit",
    "Attribution",
    "IDENTITY_TRANSITION_KINDS",
    "IdentityTransition",
    "Entity",
    "relationships_by_label",
    "validate_record",
]
