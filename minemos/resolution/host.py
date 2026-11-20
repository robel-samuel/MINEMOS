"""Host-side shadow observer for the semantic resolution layer.

A host that already ingests conversations has exactly one place where raw
message text exists.  This module is the adapter that host calls *beside* its
canonical write path: it resolves the same messages, produces counters and
returns.  It never receives an engine, never imports a writer, and cannot
absorb.

Guarantees, in the order they are enforced:

* **Disabled by default.**  :class:`ShadowSettings` starts disabled and
  :class:`HostShadowObserver` refuses to construct unless it is enabled.
* **Shadow only.**  The observer rejects any :class:`ResolutionConfig` whose
  mode is not ``shadow``.  Active absorption is not reachable from this
  module, which imports neither ``absorb_record`` nor ``absorb_ingested``.
* **Failure isolated.**  :meth:`HostShadowObserver.observe` catches every
  exception, counts it and returns.  A resolver, ingestion or reporting
  failure can never propagate into canonical ingestion.
* **Content-free output.**  Reports, logs and status contain counts, labels,
  language codes and digests only.  Message text and fact values never leave
  the record; :func:`write_json_report` serialises whatever counters the
  caller builds and nothing else.
* **Bounded.**  Repeated identical inputs are skipped, the per-conversation
  resolver set is capped, and no unbounded structure grows per message.

Fact-value tokens are available to the caller for comparison against canonical
rows (:meth:`HostShadowObserver.observe` with ``include_tokens=True``).  They
are casefolded strings derived from private text; a host must compare them and
discard them, never write them to a report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .bridge import decide, subject_text
from .config import ResolutionConfig
from .contracts import ResolutionRecord, SourceMessage
from .ingest import IngestedMessage, ingest_message
from .pipeline import Resolver

_LOG = logging.getLogger("minemos.resolution.host")

SHADOW_ENV = "MINEMOS_RESOLUTION_SHADOW"
REPORT_ENV = "MINEMOS_RESOLUTION_SHADOW_REPORT"
LOG_EVERY_ENV = "MINEMOS_RESOLUTION_SHADOW_LOG_EVERY"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

# Eligibility is a pure classification.  ``decide`` reads the mode only to
# short-circuit; passing an active config answers "would this fact be eligible
# if absorption were enabled" without touching the engine.  This object is only
# ever given to ``decide``; this module does not import any writer.
_ELIGIBILITY_PROBE = ResolutionConfig(mode="active")


def _env_bool(value: Optional[str], *, name: str) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE | _FALSE)}, got {value!r}")


def _env_int(value: Optional[str], *, name: str, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    return parsed


@dataclass(frozen=True)
class ShadowSettings:
    """Host configuration for shadow observation.

    ``enabled`` is the only switch that turns the observer on and it is false
    by default.  ``report_path`` is where an atomic counters-only JSON report is
    written after each observation; ``None`` disables file reporting.
    ``log_every`` emits one counters-only log line per this many observed
    messages (0 disables periodic logging).
    """

    enabled: bool = False
    report_path: Optional[Path] = None
    log_every: int = 50
    dedupe_capacity: int = 10_000
    max_conversations: int = 1_024

    def __post_init__(self) -> None:
        for name in ("enabled",):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in ("log_every", "dedupe_capacity", "max_conversations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.dedupe_capacity < 1:
            raise ValueError("dedupe_capacity must be at least 1")
        if self.max_conversations < 1:
            raise ValueError("max_conversations must be at least 1")

    @property
    def effective_mode(self) -> str:
        return "shadow" if self.enabled else "disabled"

    def describe(self) -> Dict[str, Any]:
        """Secret-free configuration description for logs and status."""
        return {
            "mode": self.effective_mode,
            "enabled": bool(self.enabled),
            "report_path": str(self.report_path) if self.report_path else None,
            "log_every": int(self.log_every),
            "dedupe_capacity": int(self.dedupe_capacity),
            "max_conversations": int(self.max_conversations),
            "activation": "explicit-opt-in",
            "canonical_writes": 0,
        }

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None
                 ) -> "ShadowSettings":
        raw = os.environ if env is None else env
        report = raw.get(REPORT_ENV)
        return cls(
            enabled=_env_bool(raw.get(SHADOW_ENV), name=SHADOW_ENV),
            report_path=Path(report).expanduser() if report else None,
            log_every=_env_int(raw.get(LOG_EVERY_ENV), name=LOG_EVERY_ENV,
                               default=50),
        )


@dataclass
class HostShadowReport:
    """Cumulative counters for one host.  Never holds message content."""

    messages: int = 0
    segments: int = 0
    facts: int = 0
    unresolved_subjects: int = 0
    ambiguous_identities: int = 0
    negated_facts: int = 0
    duplicate_candidates: int = 0
    resolver_failures: int = 0
    abstentions: int = 0
    transitions: int = 0
    inputs_skipped: int = 0
    conversations_evicted: int = 0
    decisions: Dict[str, int] = field(default_factory=dict)
    languages: Dict[str, int] = field(default_factory=dict)
    relationships: Dict[str, int] = field(default_factory=dict)
    subject_kinds: Dict[str, int] = field(default_factory=dict)
    latency_seconds: float = 0.0

    @property
    def would_accept(self) -> int:
        return int(self.decisions.get("eligible", 0))

    @property
    def declined(self) -> int:
        return max(0, self.facts - self.would_accept)

    def add(self, other: "HostShadowReport") -> None:
        self.messages += other.messages
        self.segments += other.segments
        self.facts += other.facts
        self.unresolved_subjects += other.unresolved_subjects
        self.ambiguous_identities += other.ambiguous_identities
        self.negated_facts += other.negated_facts
        self.duplicate_candidates += other.duplicate_candidates
        self.resolver_failures += other.resolver_failures
        self.abstentions += other.abstentions
        self.transitions += other.transitions
        self.inputs_skipped += other.inputs_skipped
        self.conversations_evicted += other.conversations_evicted
        self.latency_seconds = round(
            self.latency_seconds + other.latency_seconds, 6)
        for name in ("decisions", "languages", "relationships",
                     "subject_kinds"):
            target = getattr(self, name)
            for key, value in getattr(other, name).items():
                target[key] = target.get(key, 0) + value

    def snapshot(self) -> Dict[str, Any]:
        seen = self.messages
        return {
            "mode": "shadow",
            "messages": self.messages,
            "segments": self.segments,
            "facts_proposed": self.facts,
            "facts_eligible_if_active": self.would_accept,
            "facts_declined": self.declined,
            "declines": dict(sorted(self.decisions.items())),
            "unresolved_subjects": self.unresolved_subjects,
            "ambiguous_identities": self.ambiguous_identities,
            "negated_facts": self.negated_facts,
            "duplicate_candidates": self.duplicate_candidates,
            "resolver_failures": self.resolver_failures,
            "abstentions": self.abstentions,
            "transitions": self.transitions,
            "inputs_skipped": self.inputs_skipped,
            "conversations_evicted": self.conversations_evicted,
            "languages": dict(sorted(self.languages.items())),
            "relationships": dict(sorted(self.relationships.items())),
            "subject_kinds": dict(sorted(self.subject_kinds.items())),
            "latency_seconds_total": round(self.latency_seconds, 6),
            "latency_ms_mean": (
                round(1000.0 * self.latency_seconds / seen, 3)
                if seen else None),
            "active_absorption": False,
            "canonical_writes": 0,
        }

    def log_line(self) -> str:
        """A counters-only line safe for application logs."""
        return (
            "resolution shadow: "
            f"messages={self.messages} segments={self.segments} "
            f"facts={self.facts} eligible_if_active={self.would_accept} "
            f"declined={self.declined} unresolved={self.unresolved_subjects} "
            f"ambiguous={self.ambiguous_identities} "
            f"negated={self.negated_facts} failures={self.resolver_failures} "
            f"writes=0"
        )


class HostShadowObserver:
    """Resolves host messages in shadow mode and records counters only.

    The observer owns one :class:`Resolver` per conversation so identity state
    carries across messages and never leaks between conversations.  It is safe
    to call from a single host thread; internal mutations are additionally
    guarded by a lock.
    """

    def __init__(self, settings: Optional[ShadowSettings] = None,
                 config: Optional[ResolutionConfig] = None) -> None:
        self.settings = settings or ShadowSettings()
        if not self.settings.enabled:
            raise ValueError(
                "shadow observation is disabled; set ShadowSettings(enabled="
                "True) or MINEMOS_RESOLUTION_SHADOW=true")
        cfg = config or ResolutionConfig(mode="shadow")
        if cfg.mode != "shadow":
            raise ValueError(
                "host shadow observer requires ResolutionConfig(mode="
                f"'shadow'), got {cfg.mode!r}")
        self._config = cfg
        self._resolvers: "OrderedDict[str, Resolver]" = OrderedDict()
        self._dedupe: "OrderedDict[str, None]" = OrderedDict()
        self._report = HostShadowReport()
        self._lock = threading.RLock()
        self._logged_at = 0

    # -- introspection ---------------------------------------------------- #
    @property
    def config(self) -> ResolutionConfig:
        return self._config

    @property
    def report(self) -> HostShadowReport:
        return self._report

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._report.snapshot()

    # -- observation ------------------------------------------------------ #
    def observe(self, messages: Iterable[SourceMessage], *,
                include_tokens: bool = False) -> Dict[str, Any]:
        """Resolves ``messages`` in host order; never raises.

        Returns the counters for this call.  With ``include_tokens`` the
        result also carries ``tokens``: a list of ``(subject, predicate,
        value)`` casefolded strings for the caller to compare and discard.
        """
        delta = HostShadowReport()
        tokens: List[Tuple[str, str, str]] = []
        started = time.perf_counter()
        try:
            materialized = list(messages)
        except Exception:  # noqa: BLE001 - observer never interrupts the host
            with self._lock:
                self._report.resolver_failures += 1
            return delta.snapshot()
        ordered: List[SourceMessage] = []
        invalid = 0
        for message in materialized:
            if isinstance(message, SourceMessage):
                ordered.append(message)
            else:
                invalid += 1
        ordered.sort(key=lambda item: (item.order, item.message_id))
        delta.resolver_failures += invalid
        for message in ordered:
            try:
                key = self._dedupe_key(message)
                with self._lock:
                    if key in self._dedupe:
                        delta.inputs_skipped += 1
                        continue
                    self._remember(key)
            except Exception:  # noqa: BLE001
                delta.resolver_failures += 1
                continue
            try:
                resolver = self._resolver(message.conversation_id)
                ingested = ingest_message(
                    resolver, message, config=self._config)
            except Exception:  # noqa: BLE001 - one bad message, not the host
                delta.resolver_failures += 1
                continue
            try:
                self._accumulate(delta, ingested, tokens)
            except Exception:  # noqa: BLE001 - malformed record, keep counting
                delta.resolver_failures += 1

        delta.latency_seconds = round(time.perf_counter() - started, 6)
        with self._lock:
            self._report.add(delta)
            cumulative = self._report.snapshot()
            should_log = (
                self.settings.log_every > 0
                and self._report.messages - self._logged_at
                >= self.settings.log_every)
            if should_log:
                self._logged_at = self._report.messages
        if should_log:
            try:
                _LOG.info(self._report.log_line(), extra={
                    "shadow_messages": cumulative["messages"],
                    "shadow_facts": cumulative["facts_proposed"],
                    "shadow_eligible_if_active":
                        cumulative["facts_eligible_if_active"],
                    "shadow_failures": cumulative["resolver_failures"],
                    "canonical_writes": 0,
                })
            except Exception:  # noqa: BLE001 - logging never breaks the host
                pass

        payload = delta.snapshot()
        payload["cumulative"] = cumulative
        if include_tokens:
            payload["tokens"] = tokens
        return payload

    # -- internals -------------------------------------------------------- #
    def _resolver(self, conversation_id: str) -> Resolver:
        with self._lock:
            resolver = self._resolvers.get(conversation_id)
            if resolver is not None:
                self._resolvers.move_to_end(conversation_id)
                return resolver
            resolver = Resolver(self._config)
            self._resolvers[conversation_id] = resolver
            if len(self._resolvers) > self.settings.max_conversations:
                self._resolvers.popitem(last=False)
                self._report.conversations_evicted += 1
            return resolver

    def _dedupe_key(self, message: SourceMessage) -> str:
        digest = hashlib.sha256(
            f"{message.conversation_id}\x1f{message.provenance_key}\x1f"
            f"{message.text}".encode("utf-8")).hexdigest()
        return f"{message.conversation_id}:{digest}"

    def _remember(self, key: str) -> None:
        self._dedupe[key] = None
        self._dedupe.move_to_end(key)
        if len(self._dedupe) > self.settings.dedupe_capacity:
            self._dedupe.popitem(last=False)

    def _accumulate(self, delta: HostShadowReport,
                    ingested: IngestedMessage,
                    tokens: List[Tuple[str, str, str]]) -> None:
        delta.messages += 1
        for chunk in ingested.chunks:
            record: ResolutionRecord = chunk.record
            delta.segments += 1
            if not chunk.result.ok:
                delta.resolver_failures += 1
            language = record.language.primary
            delta.languages[language] = delta.languages.get(language, 0) + 1
            delta.abstentions += len(record.abstentions)
            delta.transitions += len(record.transitions)
            speaker = record.source.speaker
            for fact in record.facts:
                delta.facts += 1
                if fact.polarity == "negative":
                    delta.negated_facts += 1
                status = fact.subject.status
                if status == "ambiguous":
                    delta.ambiguous_identities += 1
                elif status != "resolved":
                    delta.unresolved_subjects += 1
                kind = fact.subject.kind
                delta.subject_kinds[kind] = (
                    delta.subject_kinds.get(kind, 0) + 1)
                decision = decide(record, fact, _ELIGIBILITY_PROBE)
                delta.decisions[decision.reason] = (
                    delta.decisions.get(decision.reason, 0) + 1)
                subject = subject_text(fact.subject, speaker)
                if subject:
                    tokens.append((
                        subject.casefold().strip(),
                        fact.predicate.casefold().strip(),
                        fact.value.casefold().strip(),
                    ))
            for relationship in record.relationships:
                label = relationship.label
                delta.relationships[label] = (
                    delta.relationships.get(label, 0) + 1)
                if label == "duplicate":
                    delta.duplicate_candidates += 1


def write_json_report(path: os.PathLike | str, payload: Mapping[str, Any]
                      ) -> bool:
    """Atomically writes ``payload`` as JSON; returns False on any failure.

    Reporting is a side concern of the host: a broken path, a permission error
    or a serialisation error must be observable (the return value) and must
    never propagate.
    """
    try:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=str(target.parent), prefix=f"{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True,
                          default=str)
                stream.write("\n")
            os.replace(temporary, target)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return True
    except Exception:  # noqa: BLE001 - reporting never interrupts ingestion
        return False


__all__ = [
    "HostShadowObserver",
    "HostShadowReport",
    "LOG_EVERY_ENV",
    "REPORT_ENV",
    "SHADOW_ENV",
    "ShadowSettings",
    "write_json_report",
]
