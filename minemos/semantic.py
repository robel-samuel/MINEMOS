"""Opt-in semantic ingestion from raw conversations into canonical memory.

This is the supported host-facing entry point for the semantic resolution
layer.  It composes the existing pieces and adds nothing new to them:

    raw messages
      → :func:`minemos.resolution.ingest_conversation`      (segmentation + 4 stages)
      → :func:`minemos.resolution.native_observations`      (eligibility + native fields)
      → a caller-supplied sink                              (canonical ingestion)

The sink is where canonical memory lives.  Two adapters are provided:

* :class:`ServiceSink` forwards to ``EnginePool.absorb`` — the service path
  that writes provenance, deduplicates by ingestion id and persists a
  checkpoint.  This is the recommended production path.
* :class:`EngineSink` forwards to ``BlackHoleMemory.absorb`` for applications
  that embed the engine directly.  It has no provenance ledger; replay
  deduplication there is the engine's consolidation decision.

Semantic ingestion is **opt-in**: every write requires
``ResolutionConfig(mode="active")``.  In any other mode the ingestor raises
:class:`SemanticIngestionError` rather than silently skipping writes, so a
misconfigured host cannot believe it ingested something it did not.  Passing
``sink=None`` runs the full pipeline as a dry run and writes nothing.

Canonical memory is never corrupted by a semantic failure: a message whose
resolution raises is counted and skipped, and a single failed sink call is
recorded without stopping the remaining observations.  The original message is
always returned to the caller (:attr:`SemanticIngestionReport.ingested`), so a
host that wants to keep verbatim text can persist it independently.

Quality is not asserted here.  The extraction is heuristic and the repository
has not been validated against independent human labels; see the report that
ships with this release.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from minemos.resolution import (
    DEFAULT_POLICY,
    IngestionPolicy,
    IngestedMessage,
    MessageTooLongError,
    NativeConversion,
    ResolutionConfig,
    Resolver,
    SourceMessage,
    ingest_message,
    native_observations,
)

TIMESTAMP_MODES = ("empty", "timed", "mixed", "untimed")


class SemanticIngestionError(RuntimeError):
    """The caller asked for semantic ingestion that cannot be performed."""


@runtime_checkable
class ObservationSink(Protocol):
    """Minimal canonical ingestion interface.

    Any object with this method can receive observations.  ``source`` carries
    the native provenance fields (``source_id`` and optional identifiers); a
    sink that does not understand a field ignores it.
    """

    def absorb(self, *, subject: str, predicate: str, object: str,
               timestamp: float, confidence: float,
               source: Dict[str, Any],
               ingestion_id: Optional[str] = None) -> Dict[str, Any]:
        ...


class EngineSink:
    """Adapter for applications embedding :class:`BlackHoleMemory` directly."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def absorb(self, *, subject: str, predicate: str, object: str,
               timestamp: float, confidence: float,
               source: Dict[str, Any],
               ingestion_id: Optional[str] = None) -> Dict[str, Any]:
        from minemos_engine.structured_candidate import Candidate

        result = self.engine.absorb(
            Candidate(subject=subject, predicate=predicate, object=object,
                      confidence=confidence, timestamp=timestamp),
            timestamp=timestamp,
        )
        outcome = dict(result)
        outcome.setdefault("ingestion_id", ingestion_id)
        outcome.setdefault("skipped", False)
        return outcome


class ServiceSink:
    """Adapter for the MINEMOS service ``EnginePool``.

    ``pool`` is any object exposing
    ``absorb(tenant, subject=..., predicate=..., object=..., timestamp=...,
    confidence=..., source=..., ingestion_id=...)``.  The service pool records
    provenance, deduplicates a repeated ingestion id and persists the
    checkpoint; the adapter adds no behaviour of its own.
    """

    def __init__(self, pool: Any, tenant: str,
                 source_type: str = "conversation") -> None:
        if not isinstance(tenant, str) or not tenant:
            raise ValueError("tenant must be a non-empty string")
        self.pool = pool
        self.tenant = tenant
        self.source_type = source_type

    def absorb(self, *, subject: str, predicate: str, object: str,
               timestamp: float, confidence: float,
               source: Dict[str, Any],
               ingestion_id: Optional[str] = None) -> Dict[str, Any]:
        payload = dict(source)
        payload["source_type"] = payload.get("source_type",
                                             self.source_type)
        return self.pool.absorb(
            self.tenant,
            subject=subject,
            predicate=predicate,
            object=object,
            timestamp=timestamp,
            confidence=confidence,
            source=payload,
            ingestion_id=ingestion_id,
        )


@dataclass(frozen=True)
class SemanticIngestionReport:
    """Outcome of one conversation.  Every field is content-free."""

    conversation_id: str
    ingested: Tuple[IngestedMessage, ...]
    conversions: Tuple[NativeConversion, ...]
    absorbed: int
    skipped: int
    failed_observations: int
    failed_messages: int
    error_types: Tuple[str, ...]
    timestamp_mode: str
    timestamped_messages: int
    untimed_messages: int
    dry_run: bool

    @property
    def observations(self) -> int:
        return sum(len(item.observations) for item in self.conversions)

    @property
    def rejected(self) -> int:
        return sum(len(item.rejected) for item in self.conversions)

    def reasons(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for conversion in self.conversions:
            for item in conversion.rejected:
                counts[item.reason] = counts.get(item.reason, 0) + 1
        return counts

    def ingestion_ids(self) -> Tuple[str, ...]:
        return tuple(
            item.ingestion_id
            for conversion in self.conversions
            for item in conversion.observations
        )

    def summary(self) -> Dict[str, Any]:
        """Counters and identifiers only — never message content."""
        return {
            "conversation_id": self.conversation_id,
            "dry_run": self.dry_run,
            "messages": len(self.ingested),
            "chunks": sum(len(item.chunks) for item in self.ingested),
            "observations": self.observations,
            "absorbed": self.absorbed,
            "skipped": self.skipped,
            "rejected": self.rejected,
            "reasons": self.reasons(),
            "failed_observations": self.failed_observations,
            "failed_messages": self.failed_messages,
            "error_types": list(self.error_types),
            "timestamp_mode": self.timestamp_mode,
            "timestamped_messages": self.timestamped_messages,
            "untimed_messages": self.untimed_messages,
        }


def _usable_timestamp(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def conversation_timestamp_mode(messages: Sequence[SourceMessage]) -> str:
    """Availability of event timestamps across one conversation.

    ``timed``   – every message carries a usable event timestamp;
    ``untimed`` – none does (ingestion time may stand in only if allowed);
    ``mixed``   – some do and some do not; an untimed message cannot be placed
                  on the event timeline and its facts are abstained.
    """
    if not messages:
        return "empty"
    timed = sum(1 for message in messages
                if _usable_timestamp(message.timestamp))
    if timed == len(messages):
        return "timed"
    return "mixed" if timed else "untimed"


class SemanticIngestor:
    """Resolves raw conversation messages and hands observations to a sink.

    ``config`` must select ``mode="active"``; that is the explicit opt-in the
    resolution layer requires before any candidate may reach canonical memory.

    ``known_ingestion_ids`` supports deterministic replay: ids listed there
    are skipped without calling the sink.  The service sink already deduplicates
    durably; the set is useful for embedded deployments and for a host that
    wants to re-run a batch after a crash.

    An instance owns one stateful :class:`~minemos.resolution.Resolver` and is
    therefore **not thread-safe**.  Use one ingestor per conversation stream,
    or guard it with a lock.  Separate ingestors may run concurrently.
    """

    def __init__(self, sink: Optional[ObservationSink], *,
                 config: ResolutionConfig,
                 policy: IngestionPolicy = DEFAULT_POLICY,
                 ingestion_timestamp: Optional[float] = None,
                 allow_ingestion_timestamp: bool = False,
                 resolver: Optional[Resolver] = None,
                 known_ingestion_ids: Optional[Iterable[str]] = None) -> None:
        if not isinstance(config, ResolutionConfig):
            raise TypeError("config must be a ResolutionConfig")
        if config.mode != "active":
            raise SemanticIngestionError(
                "semantic ingestion requires ResolutionConfig(mode='active'); "
                "shadow and disabled modes never write memory")
        if not isinstance(policy, IngestionPolicy):
            raise TypeError("policy must be an IngestionPolicy")
        if ingestion_timestamp is not None:
            number = float(ingestion_timestamp)
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(
                    "ingestion_timestamp must be finite and positive")
            ingestion_timestamp = number
        self.config = config
        self.sink = sink
        self.policy = policy
        self.ingestion_timestamp = ingestion_timestamp
        self.allow_ingestion_timestamp = bool(allow_ingestion_timestamp)
        self.resolver = resolver or Resolver(config)
        self.known_ingestion_ids = set(known_ingestion_ids or ())

    # -- public API ------------------------------------------------------- #
    def ingest_conversation(
            self, messages: Iterable[SourceMessage]) -> SemanticIngestionReport:
        """Ingests one conversation in host order.

        Messages from different conversations are rejected before any write.
        """
        ordered = sorted(messages, key=lambda item: (item.order,
                                                     item.message_id))
        if not ordered:
            raise SemanticIngestionError("no messages to ingest")
        conversation = ordered[0].conversation_id
        if any(item.conversation_id != conversation for item in ordered):
            raise SemanticIngestionError(
                "ingest_conversation expects messages from one conversation")

        mode = conversation_timestamp_mode(ordered)
        stamp = (self.ingestion_timestamp
                 if self.ingestion_timestamp is not None else time.time())
        fallback = self.allow_ingestion_timestamp and mode in (
            "untimed", "empty")

        ingested: List[IngestedMessage] = []
        conversions: List[NativeConversion] = []
        absorbed = skipped = failed_observations = failed_messages = 0
        errors: set = set()

        for message in ordered:
            try:
                message_batch = ingest_message(
                    self.resolver, message, policy=self.policy,
                    config=self.config)
            except MessageTooLongError:
                # A refused oversized message is a caller configuration error:
                # nothing has been written for it, and silence would hide it.
                raise
            except Exception as exc:  # noqa: BLE001 - isolation
                failed_messages += 1
                errors.add(type(exc).__name__)
                continue
            ingested.append(message_batch)
            for chunk in message_batch.chunks:
                try:
                    conversion = native_observations(
                        chunk.record, self.config,
                        segment_timestamp=chunk.source.timestamp,
                        ingestion_timestamp=stamp if fallback else None,
                        ingestion_fallback=fallback,
                        conversation_id=chunk.conversation_id,
                        message_id=chunk.original_message_id,
                    )
                except Exception as exc:  # noqa: BLE001 - isolation
                    failed_messages += 1
                    errors.add(type(exc).__name__)
                    continue
                conversions.append(conversion)
                for observation in conversion.observations:
                    if observation.ingestion_id in self.known_ingestion_ids:
                        skipped += 1
                        continue
                    if self.sink is None:
                        continue
                    source = {
                        "source_id": observation.source_id,
                        "source_type": "conversation",
                        "conversation_id": observation.conversation_id,
                        "segment_index": chunk.source.order,
                        "original_timestamp": chunk.source.timestamp,
                    }
                    try:
                        result = self.sink.absorb(
                            subject=observation.subject,
                            predicate=observation.predicate,
                            object=observation.object,
                            timestamp=observation.timestamp,
                            confidence=observation.confidence,
                            source=source,
                            ingestion_id=observation.ingestion_id,
                        )
                    except Exception as exc:  # noqa: BLE001 - isolation
                        failed_observations += 1
                        errors.add(type(exc).__name__)
                        continue
                    self.known_ingestion_ids.add(observation.ingestion_id)
                    if result.get("skipped"):
                        skipped += 1
                    else:
                        absorbed += 1

        timestamped = sum(1 for message in ordered
                          if _usable_timestamp(message.timestamp))
        return SemanticIngestionReport(
            conversation_id=conversation,
            ingested=tuple(ingested),
            conversions=tuple(conversions),
            absorbed=absorbed,
            skipped=skipped,
            failed_observations=failed_observations,
            failed_messages=failed_messages,
            error_types=tuple(sorted(errors)),
            timestamp_mode=mode,
            timestamped_messages=timestamped,
            untimed_messages=len(ordered) - timestamped,
            dry_run=self.sink is None,
        )


__all__ = [
    "EngineSink",
    "ObservationSink",
    "SemanticIngestionError",
    "SemanticIngestionReport",
    "SemanticIngestor",
    "ServiceSink",
    "TIMESTAMP_MODES",
    "conversation_timestamp_mode",
]
