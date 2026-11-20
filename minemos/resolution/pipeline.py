"""Pipeline — the four stages in order, with no memory side effects.

Raw message
  → Stage 1 ``normalize_text`` + ``detect_language``
  → Stage 2 ``interpret``
  → Stage 3 ``resolve_identity``
  → Stage 4 ``classify_relationships``
  → :class:`ResolutionRecord`

The resolver owns only conversation-scoped *resolution* state (identity
sessions and fact indexes).  It never imports a memory writer and never
mutates canonical memory, the derived index, checkpoints or retention.
Handing candidates to the engine is a separate, explicit opt-in step in
:mod:`minemos.resolution.bridge`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import ResolutionConfig
from .contracts import (
    Attribution,
    LanguageInfo,
    Provenance,
    ResolutionRecord,
    SourceMessage,
    SpeakerRef,
    SubjectRef,
    validate_record,
)
from .identity import IdentityRegistry, resolve_identity
from .interpret import interpret
from .language import detect_language
from .normalize import normalize_text
from .relationships import (
    RelationshipRegistry,
    RelationshipResult,
    classify_relationships,
)

_UNKNOWN_LANGUAGE = LanguageInfo(primary="und", confidence=0.0)


@dataclass(frozen=True)
class ResolutionResult:
    """A validated record plus a privacy-safe operator summary."""

    record: ResolutionRecord
    summary: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(
            item.startswith("pipeline_error:")
            for item in self.record.abstentions)


def _normalization_confidence(normalized, language: LanguageInfo) -> float:
    if not normalized.tokens:
        return 0.0
    score = 1.0
    score -= 0.05 * len(normalized.ambiguities)
    score -= 0.02 * sum(1 for edit in normalized.edits
                        if edit.kind in ("typo", "informal", "contraction"))
    if language.primary == "und":
        score -= 0.2
    if language.mixed:
        score -= 0.05
    return round(max(0.0, min(1.0, score)), 4)


def _evidence_spans(record_facts) -> Tuple[Tuple[int, int], ...]:
    spans: List[Tuple[int, int]] = []
    for fact in record_facts:
        for item in fact.evidence:
            span = (item.start, item.end)
            if span not in spans:
                spans.append(span)
    return tuple(spans)


def _message_attribution(source: SourceMessage, reason: str) -> Attribution:
    """The attribution a record carries when it produced no fact.

    Even an empty record names who spoke and what the message is about; when
    the speaker is unknown the entry says so rather than inventing one.
    """
    speaker = SpeakerRef(
        id=source.speaker or "unknown",
        label=source.speaker or "unknown",
        confidence=1.0 if source.speaker else 0.0,
        source="host" if source.speaker else "missing",
    )
    subject = SubjectRef(
        kind="speaker" if source.speaker else "unknown",
        label=speaker.label,
        confidence=speaker.confidence,
        owner=source.speaker,
        resolved_from="message_speaker",
        status="resolved" if source.speaker else "unknown",
        ambiguity=None if source.speaker else "speaker_missing",
    )
    return Attribution(
        speaker=speaker, subject=subject, status=subject.status,
        evidence=(reason,), scope="message")


class Resolver:
    """Stateful per-process resolver.  Thread-unsafe by construction."""

    def __init__(self, config: Optional[ResolutionConfig] = None) -> None:
        self.config = config or ResolutionConfig()
        self.identities = IdentityRegistry()
        self.relationships = RelationshipRegistry()
        self._records = 0
        self._failures = 0

    # -- single message --------------------------------------------------- #
    def resolve(self, source: SourceMessage,
                config: Optional[ResolutionConfig] = None) -> ResolutionResult:
        if not isinstance(source, SourceMessage):
            raise TypeError("resolve expects a SourceMessage")
        cfg = config or self.config
        if not cfg.enabled:
            record = self._empty_record(source, cfg, "disabled")
            validate_record(record)
            return ResolutionResult(record=record,
                                    summary={"status": "disabled"})

        try:
            record = self._run_stages(source, cfg)
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            self._failures += 1
            record = self._failure_record(source, cfg, type(exc).__name__)
        self._records += 1
        failed = any(item.startswith("pipeline_error:")
                     for item in record.abstentions)
        summary = {
            "status": "degraded" if failed else "ok",
            "conversation": source.conversation_id,
            "message": source.message_id,
            "facts": len(record.facts),
            "duplicates": sum(1 for item in record.relationships
                              if item.label == "duplicate"),
            "unresolved": len(record.abstentions),
            "language": record.language.primary,
            "mixed": record.language.mixed,
        }
        return ResolutionResult(record=record, summary=summary)

    # -- conversations ---------------------------------------------------- #
    def resolve_conversation(
            self, messages: Iterable[SourceMessage],
            config: Optional[ResolutionConfig] = None
    ) -> List[ResolutionResult]:
        ordered = sorted(messages, key=lambda item: (item.order, item.message_id))
        if not ordered:
            return []
        conversation = ordered[0].conversation_id
        if any(item.conversation_id != conversation for item in ordered):
            raise ValueError(
                "resolve_conversation expects messages from one conversation")
        return [self.resolve(item, config) for item in ordered]

    # -- internals -------------------------------------------------------- #
    def _run_stages(self, source: SourceMessage,
                    cfg: ResolutionConfig) -> ResolutionRecord:
        language = (detect_language(source.text) if cfg.detect_language
                    else _UNKNOWN_LANGUAGE)
        normalized = normalize_text(source.text, cfg, language=language.primary)
        # Spans must address the normalized text that later stages read, so
        # language is re-detected on the normalized view after normalization.
        if cfg.detect_language:
            language = detect_language(normalized.text)
        interpretation = interpret(source, normalized, language, cfg)
        session = self.identities.session(source.conversation_id)
        identity = resolve_identity(session, source, interpretation, cfg)
        index = self.relationships.index(source.conversation_id)
        earlier_ids = frozenset(item.fact_id for item in index.facts)
        relations: RelationshipResult = classify_relationships(
            index, source, identity.facts, cfg)

        # Stage 2 flags every pronoun as unresolved; Stage 3 may resolve it.
        # Keep only the abstentions that still apply to the resolved facts.
        still_unresolved = {
            fact.fact_id for fact in identity.facts
            if fact.subject.status != "resolved"}
        interpretation_abstentions = tuple(
            item for item in interpretation.abstentions
            if not item.startswith("unresolved_subject:")
            or item.split(":", 1)[1] in still_unresolved)

        record = ResolutionRecord(
            source=Provenance(
                conversation_id=source.conversation_id,
                message_id=source.message_id,
                message_key=source.message_key,
                speaker=source.speaker,
                timestamp=source.timestamp,
                spans=_evidence_spans(identity.facts),
            ),
            original_text=source.text,
            normalized=normalized,
            language=language,
            entities=interpretation.entities,
            facts=identity.facts,
            attributions=identity.attributions,
            relationships=relations.relationships,
            stage_confidence={
                "normalization": _normalization_confidence(normalized, language),
                "interpretation": interpretation.confidence,
                "identity": identity.confidence,
                "relationships": relations.confidence,
            },
            ambiguities=tuple(dict.fromkeys(
                normalized.ambiguities + interpretation.ambiguities
                + identity.ambiguities + relations.ambiguities)),
            abstentions=tuple(dict.fromkeys(
                interpretation_abstentions + identity.abstentions
                + relations.abstentions)),
            transitions=identity.transitions,
            provider=relations.provider,
        )
        validate_record(record, earlier_fact_ids=earlier_ids)
        return record

    def _empty_record(self, source: SourceMessage, cfg: ResolutionConfig,
                      reason: str) -> ResolutionRecord:
        return ResolutionRecord(
            source=Provenance(
                conversation_id=source.conversation_id,
                message_id=source.message_id,
                message_key=source.message_key,
                speaker=source.speaker,
                timestamp=source.timestamp,
            ),
            original_text=source.text,
            normalized=normalize_text(source.text, cfg, language="und"),
            language=_UNKNOWN_LANGUAGE,
            entities=(),
            facts=(),
            attributions=(_message_attribution(source, "resolution_disabled"),),
            relationships=(),
            stage_confidence={
                "normalization": 0.0,
                "interpretation": 0.0,
                "identity": 0.0,
                "relationships": 0.0,
            },
            abstentions=(f"resolution_disabled:{reason}",),
            provider=f"deterministic:{cfg.semantic_provider}",
        )

    def _failure_record(self, source: SourceMessage, cfg: ResolutionConfig,
                        error: str) -> ResolutionRecord:
        safe_error = "".join(
            char for char in error if char.isalnum() or char in "_")
        return ResolutionRecord(
            source=Provenance(
                conversation_id=source.conversation_id,
                message_id=source.message_id,
                message_key=source.message_key,
                speaker=source.speaker,
                timestamp=source.timestamp,
            ),
            original_text=source.text,
            normalized=normalize_text(source.text, cfg, language="und"),
            language=_UNKNOWN_LANGUAGE,
            entities=(),
            facts=(),
            attributions=(_message_attribution(source, "pipeline_error"),),
            relationships=(),
            stage_confidence={
                "normalization": 0.0,
                "interpretation": 0.0,
                "identity": 0.0,
                "relationships": 0.0,
            },
            abstentions=(f"pipeline_error:{safe_error or 'unknown'}",),
            provider=f"deterministic:{cfg.semantic_provider}",
        )

    # -- operator counters ------------------------------------------------- #
    def stats(self) -> Dict[str, object]:
        """Counters and identifiers only — never message content."""
        return {
            "records": self._records,
            "failures": self._failures,
            "conversations": len(self.identities.conversations()),
            "tracked_facts": sum(
                len(self.relationships.index(conversation).facts)
                for conversation in self.relationships.conversations()),
        }


def resolve_messages(messages: Sequence[SourceMessage],
                     config: Optional[ResolutionConfig] = None
                     ) -> List[ResolutionResult]:
    """Convenience wrapper: one resolver, messages grouped by conversation."""
    resolver = Resolver(config)
    grouped: Dict[str, List[SourceMessage]] = {}
    for message in messages:
        grouped.setdefault(message.conversation_id, []).append(message)
    results: List[ResolutionResult] = []
    for conversation in grouped:
        results.extend(resolver.resolve_conversation(grouped[conversation]))
    return results


__all__ = ["ResolutionResult", "Resolver", "resolve_messages"]
