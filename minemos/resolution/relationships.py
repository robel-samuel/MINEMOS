"""Stage 4 — fact relationship resolution.

For each candidate fact this stage answers one question: *how does this
candidate relate to facts already seen in the same conversation?*  It returns
a classification and the structural evidence behind it.  It deliberately does
**not** consolidate, supersede, resolve contradictions, retain or evict:
those decisions belong to the MINEMOS engine, which remains the only memory
authority.

Labels
------
``duplicate``
    Same subject, predicate and value (after normalization), same polarity.
    A value that only lost the capitalisation of a proper noun, or a known
    vocabulary word replaced by a non-word one edit away, also counts: that
    is a typing variant, not a new fact.
``temporal_update``
    Same functional predicate, a different value, and the new statement is
    anchored in the present/future, so it describes a newer state.
``historical_observation``
    Same functional predicate, a different value, and the new statement is
    explicitly anchored in the past or an earlier year.
``distinct_fact``
    A different fact: different subject, predicate, or a value that is
    additive rather than conflicting.
``unresolved``
    Comparable facts that conflict without enough evidence to say which
    classification applies.  The candidate abstains instead of guessing.

Comparisons never cross conversations, subjects or predicates, so a fact about
a brother can never be reported as a duplicate of a fact about the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Tuple

from .config import ResolutionConfig
from .contracts import (
    FactCandidate,
    Relationship,
    SourceMessage,
)
from .identity import subject_key
from .lexicon import is_vocabulary_inflection, is_vocabulary_word
from .similarity import (
    build_similarity,
    edit_distance,
    is_transposition,
    normalize_value,
)

FUNCTIONAL_PREDICATES = frozenset({
    "lives_in", "is_from", "born_in", "works_at", "works_as", "works_in",
    "studies_at", "attends", "age", "name", "uses",
    # Attributes a subject has one of at a time: a new value describes a new
    # state rather than an additional fact.
    "salary", "rent", "role", "job", "degree", "birthday", "hometown",
    "native_language", "allergic_to", "phone_number", "email_address",
    "favorite_color", "favorite_author", "favorite_film", "favorite_book",
    "favorite_movie", "favorite_food", "favorite_drink", "favorite_music",
    "favorite_song", "favorite_sport", "favorite_team", "favorite_language",
})

# A negated functional statement carrying one of these markers ends the prior
# state ("no longer works at ..."): the older value becomes historical, so the
# structural relationship is a temporal update rather than a contradiction.
# Without such a marker a polarity conflict stays unresolved.
_CESSATION_MARKERS = frozenset({"no longer", "no more", "anymore"})

_VALUE_PUNCT = " \t,;:.!?\"'`()[]{}"


@dataclass(frozen=True)
class IndexedFact:
    fact_id: str
    message_id: str
    order: int
    timestamp: Optional[float]
    subject_key: Optional[str]
    predicate: str
    value: str
    value_key: str
    polarity: str
    temporal_kinds: Tuple[str, ...]
    year: Optional[int]
    cessation: bool = False

    @classmethod
    def from_fact(cls, fact: FactCandidate, source: SourceMessage
                  ) -> "IndexedFact":
        years = [expression.year for expression in fact.temporal
                 if expression.year is not None]
        return cls(
            fact_id=fact.fact_id,
            message_id=source.message_id,
            order=source.order,
            timestamp=source.timestamp,
            subject_key=subject_key(fact.subject),
            predicate=fact.predicate,
            value=fact.value,
            value_key=normalize_value(fact.value),
            polarity=fact.polarity,
            temporal_kinds=tuple(sorted({item.kind for item in fact.temporal})),
            year=max(years) if years else None,
            cessation=any(
                expression.text.casefold() in _CESSATION_MARKERS
                for expression in fact.temporal),
        )


@dataclass
class FactIndex:
    """Conversation-scoped list of facts already classified."""

    conversation_id: str
    facts: List[IndexedFact] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.conversation_id:
            raise ValueError("FactIndex needs a conversation id")

    def comparable(self, fact: IndexedFact) -> List[IndexedFact]:
        if not fact.subject_key:
            return []
        return [item for item in self.facts
                if item.subject_key == fact.subject_key
                and item.predicate == fact.predicate]

    def add(self, fact: IndexedFact) -> None:
        self.facts.append(fact)


class RelationshipRegistry:
    """One fact index per conversation."""

    def __init__(self) -> None:
        self._indexes: Dict[str, FactIndex] = {}

    def index(self, conversation_id: str) -> FactIndex:
        if not conversation_id:
            raise ValueError("a conversation id is required")
        index = self._indexes.get(conversation_id)
        if index is None:
            index = FactIndex(conversation_id)
            self._indexes[conversation_id] = index
        return index

    def conversations(self) -> Tuple[str, ...]:
        return tuple(self._indexes)


@dataclass(frozen=True)
class RelationshipResult:
    relationships: Tuple[Relationship, ...]
    ambiguities: Tuple[str, ...] = ()
    abstentions: Tuple[str, ...] = ()
    confidence: float = 0.0
    provider: str = "deterministic"


def _similarity_band(value: float, config: ResolutionConfig) -> str:
    if value >= config.duplicate_similarity:
        return "duplicate"
    if value >= config.ambiguous_similarity:
        return "ambiguous"
    return "distinct"


def _has_now(fact: IndexedFact) -> bool:
    return "now" in fact.temporal_kinds or "future" in fact.temporal_kinds


def _has_past(fact: IndexedFact) -> bool:
    return "past" in fact.temporal_kinds


def _variant_duplicate(fact: IndexedFact, nearest: IndexedFact
                       ) -> Optional[int]:
    """Detects a typing variant of the same value; returns changed tokens.

    A near value is only accepted as the *same* fact when the rewrite carries
    a typo signature: a token that lost the capitalisation of a proper noun,
    or a token the vocabulary knows replaced by a form it does not know.
    Values with digits, short tokens and deliberate spelling variants that
    keep their capitals ("Harbor" / "Harbour") stay distinct or unresolved.
    """
    new_tokens = fact.value.split()
    old_tokens = nearest.value.split()
    if not new_tokens or len(new_tokens) != len(old_tokens):
        return None
    if any(char.isdigit() for char in fact.value):
        return None
    changed = 0
    for new_token, old_token in zip(new_tokens, old_tokens):
        new_clean = new_token.strip(_VALUE_PUNCT)
        old_clean = old_token.strip(_VALUE_PUNCT)
        new_key = new_clean.casefold()
        old_key = old_clean.casefold()
        if new_key == old_key:
            continue
        if len(new_clean) < 4 or len(old_clean) < 4:
            return None
        if edit_distance(new_key, old_key, 1) != 1:
            return None
        # Four-letter tokens are only accepted as an adjacent transposition,
        # which keeps short distinct names ("Iran" / "Iraq") apart.
        if len(new_clean) == 4 and not is_transposition(new_key, old_key):
            return None
        lost_capitals = (new_clean.isalpha() and new_clean.islower()
                         and old_clean[:1].isupper()
                         and old_clean[1:].islower())
        vocabulary_replacement = (
            is_vocabulary_word(old_clean)
            and not is_vocabulary_inflection(new_clean))
        if not (lost_capitals or vocabulary_replacement):
            return None
        changed += 1
    return changed or None


def _classify(index: FactIndex, fact: IndexedFact,
              config: ResolutionConfig, similarity) -> Relationship:
    candidates = index.comparable(fact)
    if not candidates:
        return Relationship(
            label="distinct_fact", confidence=0.9,
            evidence=("no_prior_fact",))

    ranked = sorted(
        ((similarity(fact.value, item.value), item) for item in candidates),
        key=lambda pair: (-pair[0], -pair[1].order, pair[1].fact_id))
    score, nearest = ranked[0]
    evidence = [f"prior_fact={nearest.fact_id}",
                f"value_similarity={score:.2f}"]

    # A duplicate needs an identical normalized value.  Similarity alone is
    # never enough: "The Quiet Harbor" and "The Silent Harbor" score high but
    # are different facts, and merging them would be a false merge.  Near
    # matches fall through to the unresolved branches instead.
    same_polarity = fact.polarity == nearest.polarity
    value_matches = fact.value_key == nearest.value_key

    if value_matches and same_polarity:
        return Relationship(
            label="duplicate", confidence=round(max(0.8, score), 4),
            target_fact_id=nearest.fact_id,
            evidence=tuple(evidence + [f"subject={fact.subject_key}"]))

    if value_matches and not same_polarity:
        if fact.cessation:
            return Relationship(
                label="temporal_update", confidence=0.74,
                target_fact_id=nearest.fact_id,
                evidence=tuple(evidence + ["polarity_differs",
                                           "cessation_marker"]))
        return Relationship(
            label="unresolved", confidence=0.6,
            target_fact_id=nearest.fact_id,
            evidence=tuple(evidence + ["polarity_differs"]),
            reason="polarity_conflict")

    if same_polarity:
        # The nearest candidate may itself be a lowercased repeat, so every
        # comparable value is checked for a typo signature.
        for prior_score, candidate in ranked:
            if candidate.value_key == fact.value_key:
                continue
            variant_tokens = _variant_duplicate(fact, candidate)
            if variant_tokens:
                return Relationship(
                    label="duplicate", confidence=0.8,
                    target_fact_id=candidate.fact_id,
                    evidence=(f"prior_fact={candidate.fact_id}",
                              f"value_similarity={prior_score:.2f}",
                              f"subject={fact.subject_key}",
                              f"value_variant_tokens={variant_tokens}"))

    functional = fact.predicate in FUNCTIONAL_PREDICATES

    if functional:
        if _has_now(fact) and not _has_past(fact):
            if nearest.year is not None and fact.year is not None \
                    and fact.year < nearest.year:
                return Relationship(
                    label="historical_observation", confidence=0.68,
                    target_fact_id=nearest.fact_id,
                    evidence=tuple(evidence + ["earlier_year"]),
                )
            return Relationship(
                label="temporal_update", confidence=0.78,
                target_fact_id=nearest.fact_id,
                evidence=tuple(evidence + ["temporal=now"]),
            )
        if _has_past(fact) and not _has_now(fact):
            return Relationship(
                label="historical_observation", confidence=0.72,
                target_fact_id=nearest.fact_id,
                evidence=tuple(evidence + ["temporal=past"]),
            )
        if fact.year is not None and nearest.year is not None:
            if fact.year < nearest.year:
                return Relationship(
                    label="historical_observation", confidence=0.7,
                    target_fact_id=nearest.fact_id,
                    evidence=tuple(evidence + ["earlier_year"]),
                )
            if fact.year > nearest.year:
                return Relationship(
                    label="temporal_update", confidence=0.74,
                    target_fact_id=nearest.fact_id,
                    evidence=tuple(evidence + ["later_year"]),
                )
        band = _similarity_band(score, config)
        if band == "ambiguous":
            return Relationship(
                label="unresolved", confidence=0.55,
                target_fact_id=nearest.fact_id,
                evidence=tuple(evidence + ["similarity_band=ambiguous"]),
                reason="ambiguous_value_similarity")
        return Relationship(
            label="unresolved", confidence=0.6,
            target_fact_id=nearest.fact_id,
            evidence=tuple(evidence + ["value_conflict_without_time"]),
            reason="value_conflict_without_time")

    return Relationship(
        label="distinct_fact", confidence=0.82,
        target_fact_id=nearest.fact_id,
        evidence=tuple(evidence + ["additive_value"]))


def classify_relationships(index: FactIndex, source: SourceMessage,
                           facts: Iterable[FactCandidate],
                           config: Optional[ResolutionConfig] = None
                           ) -> RelationshipResult:
    """Classifies each candidate against facts already in the conversation."""
    cfg = config or ResolutionConfig()
    if index.conversation_id != source.conversation_id:
        raise ValueError(
            "relationship indexes are conversation-scoped; refusing to "
            f"compare {source.conversation_id!r} against "
            f"{index.conversation_id!r}")

    similarity = build_similarity(cfg)
    relationships: List[Relationship] = []
    abstentions: List[str] = []
    ambiguities: List[str] = []
    confidences: List[float] = []

    for fact in facts:
        indexed = IndexedFact.from_fact(fact, source)
        relationship = _classify(index, indexed, cfg, similarity)
        relationships.append(relationship)
        confidences.append(relationship.confidence)
        if relationship.label == "unresolved":
            reason = relationship.reason or "unresolved"
            abstentions.append(f"relationship_unresolved:{fact.fact_id}:{reason}")
            ambiguities.append(f"{reason}:{fact.fact_id}")
        index.add(indexed)

    mean = (sum(confidences) / len(confidences)) if confidences else 0.0
    return RelationshipResult(
        relationships=tuple(relationships),
        ambiguities=tuple(dict.fromkeys(ambiguities)),
        abstentions=tuple(dict.fromkeys(abstentions)),
        confidence=round(mean, 4),
        provider=f"deterministic:{similarity.name}",
    )


__all__ = [
    "FUNCTIONAL_PREDICATES",
    "FactIndex",
    "IndexedFact",
    "RelationshipRegistry",
    "RelationshipResult",
    "classify_relationships",
]
