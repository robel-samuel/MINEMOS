"""Explicit, opt-in bridge from resolution records to the MINEMOS engine.

The resolution layer never writes memory on its own.  This module is the only
place that hands candidates to the engine, and it does so only when:

* the resolver is explicitly in ``active`` mode, **and**
* the caller calls :func:`absorb_record` / :func:`apply_records` directly.

Eligibility is deliberately narrow:

* the subject must be resolved (an unknown or ambiguous subject is never
  invented into a candidate);
* the fact must be asserted with positive polarity (the engine has no way to
  store a negated observation, so a negated candidate would become a false
  positive assertion);
* the fact must not be classified ``unresolved`` against its own conversation
  (an ambiguous conflict is left for a human or for the engine's own
  contradiction logic, not silently asserted);
* a relationship classified ``duplicate`` is not sent again — re-absorbing an
  identical fact would add no information.

The engine keeps every downstream decision: consolidation, contradiction,
supersession, temporal intervals, retention and eviction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional, Tuple

from .config import ResolutionConfig
from .contracts import FactCandidate, Relationship, ResolutionRecord, SubjectRef


@dataclass(frozen=True)
class BridgeDecision:
    fact_id: str
    accepted: bool
    reason: str


def subject_text(subject: SubjectRef, speaker: Optional[str]) -> Optional[str]:
    """Human-facing subject string, or None when it is not resolved."""
    if subject.status != "resolved":
        return None
    if subject.kind == "speaker":
        return subject.owner or speaker or "user"
    if subject.kind == "derived":
        if not subject.relation:
            return None
        return subject.label or f"user's {subject.relation}"
    if subject.kind == "named":
        return subject.label
    return None


def _relationship_for(record: ResolutionRecord,
                      fact_id: str) -> Optional[Relationship]:
    for index, fact in enumerate(record.facts):
        if fact.fact_id == fact_id and index < len(record.relationships):
            return record.relationships[index]
    return None


def decide(record: ResolutionRecord, fact: FactCandidate,
           config: Optional[ResolutionConfig] = None) -> BridgeDecision:
    cfg = config or ResolutionConfig()
    if cfg.mode != "active":
        return BridgeDecision(fact.fact_id, False, "mode_is_not_active")
    if fact.polarity != "positive":
        return BridgeDecision(fact.fact_id, False, "non_asserted_polarity")
    if fact.modality != "asserted":
        return BridgeDecision(fact.fact_id, False, "non_asserted_modality")
    if subject_text(fact.subject, record.source.speaker) is None:
        return BridgeDecision(fact.fact_id, False, "subject_unresolved")
    relationship = _relationship_for(record, fact.fact_id)
    if relationship is not None and relationship.label in (
            "duplicate", "unresolved"):
        return BridgeDecision(
            fact.fact_id, False, f"relationship_{relationship.label}")
    return BridgeDecision(fact.fact_id, True, "eligible")


def candidates_for(record: ResolutionRecord,
                   config: Optional[ResolutionConfig] = None
                   ) -> Tuple[List[object], List[BridgeDecision]]:
    """Builds engine ``Candidate`` objects for the eligible facts."""
    from minemos_engine.structured_candidate import Candidate

    cfg = config or ResolutionConfig()
    candidates: List[object] = []
    decisions: List[BridgeDecision] = []
    timestamp = record.source.timestamp or 0.0
    for fact in record.facts:
        decision = decide(record, fact, cfg)
        decisions.append(decision)
        if not decision.accepted:
            continue
        subject = subject_text(fact.subject, record.source.speaker)
        candidates.append(Candidate(
            subject=subject,
            predicate=fact.predicate,
            object=fact.value,
            confidence=round(fact.confidence, 4),
            timestamp=timestamp,
        ))
    return candidates, decisions


def absorb_record(engine, record: ResolutionRecord,
                  config: Optional[ResolutionConfig] = None
                  ) -> Dict[str, object]:
    """Absorbs the eligible candidates of one record into ``engine``.

    ``engine`` is a :class:`minemos_engine.BlackHoleMemory` instance.  The
    caller owns that object and its canonical state; this function performs no
    other mutation.
    """
    cfg = config or ResolutionConfig()
    if cfg.mode != "active":
        return {
            "absorbed": 0,
            "skipped": len(record.facts),
            "reason": "mode_is_not_active",
            "decisions": [],
        }
    candidates, decisions = candidates_for(record, cfg)
    absorbed = 0
    timestamp = record.source.timestamp
    for candidate in candidates:
        engine.absorb(candidate, timestamp=timestamp)
        absorbed += 1
    return {
        "absorbed": absorbed,
        "skipped": len(decisions) - absorbed,
        "reason": "ok",
        "decisions": [decision.__dict__ for decision in decisions],
    }


def eligibility_counts(record: ResolutionRecord,
                       config: Optional[ResolutionConfig] = None
                       ) -> Dict[str, int]:
    """Counts bridge decisions without building candidates."""
    cfg = config or ResolutionConfig()
    counts: Dict[str, int] = {}
    for fact in record.facts:
        decision = decide(record, fact, cfg)
        counts[decision.reason] = counts.get(decision.reason, 0) + 1
    return counts


__all__ = [
    "BridgeDecision",
    "absorb_record",
    "candidates_for",
    "decide",
    "eligibility_counts",
    "subject_text",
]
