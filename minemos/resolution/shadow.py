"""Shadow mode: run the resolver beside the existing pipeline and compare.

Shadow mode never writes canonical memory, never touches retrieval and never
changes engine behaviour.  It records structured counts so an operator can see
what the layer *would* propose and how it differs from the current extraction.

Privacy rule: diagnostics contain counts, labels, stage names and hashes — no
message text, no fact values, no entity names.  :class:`ShadowReport.log_line`
is the only thing intended for application logs, and it is built from counters
alone.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .bridge import eligibility_counts
from .config import ResolutionConfig
from .contracts import FactCandidate, ResolutionRecord, SourceMessage
from .pipeline import ResolutionResult, Resolver

_LOG = logging.getLogger("minemos.resolution.shadow")


def value_token(value: str) -> str:
    """A stable opaque token for a value, safe to keep in diagnostics."""
    digest = hashlib.sha256(value.strip().casefold().encode("utf-8"))
    return digest.hexdigest()[:12]


def _fact_tokens(facts: Iterable[FactCandidate]) -> set:
    return {(fact.predicate, value_token(fact.value)) for fact in facts}


@dataclass
class ShadowReport:
    messages: int = 0
    facts: int = 0
    entities: int = 0
    by_relationship: Dict[str, int] = field(default_factory=dict)
    by_subject_kind: Dict[str, int] = field(default_factory=dict)
    by_language: Dict[str, int] = field(default_factory=dict)
    unresolved_subjects: int = 0
    abstentions: int = 0
    transitions: int = 0
    bridge: Dict[str, int] = field(default_factory=dict)
    stage_confidence: Dict[str, List[float]] = field(default_factory=dict)
    baseline_only: int = 0
    resolution_only: int = 0
    shared: int = 0

    def observe(self, result: ResolutionResult) -> None:
        record = result.record
        self.messages += 1
        self.facts += len(record.facts)
        self.entities += len(record.entities)
        for relationship in record.relationships:
            self.by_relationship[relationship.label] = (
                self.by_relationship.get(relationship.label, 0) + 1)
        for fact in record.facts:
            kind = fact.subject.kind
            self.by_subject_kind[kind] = self.by_subject_kind.get(kind, 0) + 1
            if fact.subject.status != "resolved":
                self.unresolved_subjects += 1
        self.by_language[record.language.primary] = (
            self.by_language.get(record.language.primary, 0) + 1)
        self.abstentions += len(record.abstentions)
        self.transitions += len(record.transitions)
        for name, value in record.stage_confidence.items():
            self.stage_confidence.setdefault(name, []).append(float(value))
        for reason, count in eligibility_counts(record).items():
            self.bridge[reason] = self.bridge.get(reason, 0) + count

    def compare(self, record: ResolutionRecord,
                baseline_facts: Sequence[FactCandidate]) -> None:
        ours = _fact_tokens(record.facts)
        theirs = _fact_tokens(baseline_facts)
        self.shared += len(ours & theirs)
        self.resolution_only += len(ours - theirs)
        self.baseline_only += len(theirs - ours)

    def to_dict(self) -> Dict[str, object]:
        means = {name: round(sum(values) / len(values), 4)
                 for name, values in self.stage_confidence.items() if values}
        return {
            "messages": self.messages,
            "facts": self.facts,
            "entities": self.entities,
            "by_relationship": dict(sorted(self.by_relationship.items())),
            "by_subject_kind": dict(sorted(self.by_subject_kind.items())),
            "by_language": dict(sorted(self.by_language.items())),
            "unresolved_subjects": self.unresolved_subjects,
            "abstentions": self.abstentions,
            "transitions": self.transitions,
            "bridge": dict(sorted(self.bridge.items())),
            "stage_confidence_mean": means,
            "baseline_only": self.baseline_only,
            "resolution_only": self.resolution_only,
            "shared": self.shared,
        }

    def log_line(self) -> str:
        """A one-line, content-free summary for application logs."""
        return json.dumps({
            "messages": self.messages,
            "facts": self.facts,
            "duplicates": self.by_relationship.get("duplicate", 0),
            "unresolved": self.by_relationship.get("unresolved", 0),
            "unresolved_subjects": self.unresolved_subjects,
            "resolution_only": self.resolution_only,
            "baseline_only": self.baseline_only,
        }, sort_keys=True)


class ShadowRunner:
    """Runs the resolver without side effects and keeps counters only."""

    def __init__(self, config: Optional[ResolutionConfig] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        cfg = config or ResolutionConfig()
        if cfg.mode != "shadow":
            raise ValueError(
                "ShadowRunner requires shadow mode; pass an explicit active "
                "resolver for write-side integration")
        self.resolver = Resolver(cfg)
        self.report = ShadowReport()
        self._logger = logger or _LOG
        self._log_every = 50

    def run(self, message: SourceMessage,
            baseline_facts: Sequence[FactCandidate] = ()
            ) -> ResolutionResult:
        result = self.resolver.resolve(message)
        self.report.observe(result)
        if baseline_facts:
            self.report.compare(result.record, baseline_facts)
        if self.report.messages % self._log_every == 0:
            self._logger.info("resolution shadow %s", self.report.log_line())
        return result

    def run_conversation(
            self, messages: Iterable[SourceMessage],
            baseline: Optional[Dict[str, Sequence[FactCandidate]]] = None
    ) -> List[ResolutionResult]:
        results: List[ResolutionResult] = []
        for message in sorted(messages,
                              key=lambda item: (item.order, item.message_id)):
            results.append(self.run(
                message, (baseline or {}).get(message.message_id, ())))
        return results

    def summary(self) -> Dict[str, object]:
        return self.report.to_dict()


__all__ = ["ShadowReport", "ShadowRunner", "value_token"]
