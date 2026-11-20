"""Enforced host entry point for raw conversational messages.

The resolver is a pure text processor: it never truncates and it accepts a
message of any length.  A host that receives arbitrary-length messages must
segment before resolution, because the surrounding MINEMOS service caps one
fact value at 8,192 characters.  This module is that host-facing boundary.

* :func:`ingest_message` splits an oversized message at the published
  8,192-character boundary and resolves the chunks **in source order** on one
  resolver, so conversation-scoped identity state carries across boundaries.
* :func:`ingest_conversation` does the same for an ordered conversation and
  refuses a batch that mixes conversations.
* :func:`absorb_ingested` is the opt-in bridge to the engine.  It writes only
  when the caller explicitly passes ``ResolutionConfig(mode="active")``.

Segmentation is lossless: ``original_text[start:end]`` is the chunk text, the
first chunk keeps the original message id, later chunks are
``<message_id>#<index>``, and the original id, conversation id and offsets are
returned alongside every record.  Nothing is silently truncated.  A host that
would rather refuse an oversized message sets ``IngestionPolicy(overflow=
"reject")`` and gets :class:`MessageTooLongError` before any resolution or
write happens.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Tuple

from .bridge import absorb_record
from .config import ResolutionConfig
from .contracts import ResolutionRecord, SourceMessage
from .pipeline import ResolutionResult, Resolver
from .segments import MAX_MESSAGE_CHARS, MessageChunk, segment_message

OVERFLOW_POLICIES = ("segment", "reject")


class MessageTooLongError(ValueError):
    """An oversized message was refused instead of segmented or truncated."""


@dataclass(frozen=True)
class IngestionPolicy:
    """How a host treats a message longer than one resolvable segment."""

    max_chars: int = MAX_MESSAGE_CHARS
    overflow: str = "segment"

    def __post_init__(self) -> None:
        if isinstance(self.max_chars, bool) or not isinstance(
                self.max_chars, int):
            raise TypeError("max_chars must be an integer")
        if self.max_chars < 1:
            raise ValueError("max_chars must be positive")
        if self.overflow not in OVERFLOW_POLICIES:
            raise ValueError(
                f"overflow must be one of {OVERFLOW_POLICIES}")


DEFAULT_POLICY = IngestionPolicy()


@dataclass(frozen=True)
class IngestedChunk:
    """One resolved chunk plus the source coordinates it came from."""

    original_message_id: str
    conversation_id: str
    chunk: MessageChunk
    source: SourceMessage
    result: ResolutionResult

    @property
    def record(self) -> ResolutionRecord:
        return self.result.record

    @property
    def message_id(self) -> str:
        return self.source.message_id

    @property
    def char_start(self) -> int:
        return self.chunk.char_start

    @property
    def char_end(self) -> int:
        return self.chunk.char_end


@dataclass(frozen=True)
class IngestedMessage:
    """Every chunk of one source message, in source order."""

    original_message_id: str
    conversation_id: str
    chunks: Tuple[IngestedChunk, ...]

    @property
    def records(self) -> Tuple[ResolutionRecord, ...]:
        return tuple(item.record for item in self.chunks)

    @property
    def facts(self) -> int:
        return sum(len(record.facts) for record in self.records)

    @property
    def duplicates(self) -> int:
        return sum(
            1 for record in self.records
            for item in record.relationships if item.label == "duplicate")

    @property
    def max_chunk_chars(self) -> int:
        return max(len(item.chunk.text) for item in self.chunks)

    def summary(self) -> Dict[str, object]:
        """Counters only — never message content."""
        return {
            "conversation": self.conversation_id,
            "chunks": len(self.chunks),
            "max_chunk_chars": self.max_chunk_chars,
            "facts": self.facts,
            "duplicates": self.duplicates,
            "unresolved": sum(len(record.abstentions)
                              for record in self.records),
        }


def _chunk_key(message: SourceMessage, chunk: MessageChunk) -> Optional[str]:
    if chunk.chunk_count == 1:
        return message.message_key
    base = message.message_key or message.message_id
    return f"{base}#{chunk.chunk_index}"


def ingest_message(resolver: Resolver, message: SourceMessage, *,
                   policy: IngestionPolicy = DEFAULT_POLICY,
                   config: Optional[ResolutionConfig] = None
                   ) -> IngestedMessage:
    """Segments ``message`` if needed and resolves every chunk in order."""
    if not isinstance(resolver, Resolver):
        raise TypeError("ingest_message expects a Resolver")
    if not isinstance(message, SourceMessage):
        raise TypeError("ingest_message expects a SourceMessage")
    if not isinstance(policy, IngestionPolicy):
        raise TypeError("policy must be an IngestionPolicy")
    if len(message.text) > policy.max_chars and policy.overflow == "reject":
        raise MessageTooLongError(
            f"message is {len(message.text)} characters; the ingestion limit "
            f"is {policy.max_chars} characters and overflow policy is "
            "'reject'")

    chunks = segment_message(message.text, message.message_id,
                             policy.max_chars)
    ingested: List[IngestedChunk] = []
    for chunk in chunks:
        source = replace(
            message,
            message_id=chunk.identity,
            message_key=_chunk_key(message, chunk),
            text=chunk.text,
        )
        result = resolver.resolve(source, config)
        ingested.append(IngestedChunk(
            original_message_id=message.message_id,
            conversation_id=message.conversation_id,
            chunk=chunk,
            source=source,
            result=result,
        ))
    return IngestedMessage(
        original_message_id=message.message_id,
        conversation_id=message.conversation_id,
        chunks=tuple(ingested),
    )


def ingest_conversation(
        resolver: Resolver, messages: Iterable[SourceMessage], *,
        policy: IngestionPolicy = DEFAULT_POLICY,
        config: Optional[ResolutionConfig] = None
) -> List[IngestedMessage]:
    """Ingests one conversation in host order, segmenting each message."""
    ordered = sorted(messages, key=lambda item: (item.order, item.message_id))
    if not ordered:
        return []
    conversation = ordered[0].conversation_id
    if any(item.conversation_id != conversation for item in ordered):
        raise ValueError(
            "ingest_conversation expects messages from one conversation")
    return [ingest_message(resolver, item, policy=policy, config=config)
            for item in ordered]


def absorb_ingested(engine, ingested: IngestedMessage, *,
                    config: Optional[ResolutionConfig] = None
                    ) -> Dict[str, object]:
    """Absorbs an ingested message through the opt-in bridge.

    Every chunk is resolved before this call, so cross-chunk duplicate
    classification is already available and a repeated fact is never written
    twice.  Without ``active`` mode the engine is untouched.
    """
    if not isinstance(ingested, IngestedMessage):
        raise TypeError("absorb_ingested expects an IngestedMessage")
    cfg = config or ResolutionConfig()
    summary: Dict[str, object] = {
        "absorbed": 0,
        "skipped": 0,
        "chunks": len(ingested.chunks),
        "reason": "ok",
        "decisions": [],
    }
    blocked = False
    decisions: List[Dict[str, object]] = []
    for item in ingested.chunks:
        outcome = absorb_record(engine, item.record, cfg)
        summary["absorbed"] += int(outcome["absorbed"])
        summary["skipped"] += int(outcome["skipped"])
        decisions.extend(outcome["decisions"])
        if outcome.get("reason") == "mode_is_not_active":
            blocked = True
    if blocked:
        summary["reason"] = "mode_is_not_active"
    summary["decisions"] = decisions
    return summary


__all__ = [
    "DEFAULT_POLICY",
    "IngestedChunk",
    "IngestedMessage",
    "IngestionPolicy",
    "MessageTooLongError",
    "OVERFLOW_POLICIES",
    "absorb_ingested",
    "ingest_conversation",
    "ingest_message",
]
