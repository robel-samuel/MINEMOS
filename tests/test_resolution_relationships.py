"""Stage 4: fact relationship classification.

Every case feeds two observations into one conversation and grades the label
the second one receives.  Stage 4 compares candidate facts; it never rewrites
the earlier record, so each test also checks that the first message's facts
are unchanged.
"""

from __future__ import annotations

import pytest

from minemos.resolution import Resolver
from minemos.resolution.contracts import SourceMessage
from minemos.resolution.relationships import (
    FactIndex,
    RelationshipRegistry,
    classify_relationships,
)

LABELS = {
    "duplicate",
    "temporal_update",
    "historical_observation",
    "distinct_fact",
    "unresolved",
}


def _message(text: str, order: int, conversation: str = "c") -> SourceMessage:
    return SourceMessage(
        conversation_id=conversation,
        message_id=f"m{order}",
        text=text,
        speaker="user",
        timestamp=float(order),
        order=order,
    )


def _resolve(conversation: str, texts) -> list:
    resolver = Resolver()
    return [resolver.resolve(_message(text, order, conversation)).record
            for order, text in enumerate(texts)]


def _only(record):
    assert len(record.facts) == 1, [fact.predicate for fact in record.facts]
    assert len(record.relationships) == 1
    return record.facts[0], record.relationships[0]


def test_identical_fact_is_a_duplicate():
    first, second = _resolve("c", ["I live in Berlin.", "I live in Berlin."])
    fact, relationship = _only(second)
    assert fact.predicate == "lives_in"
    assert relationship.label == "duplicate"
    assert relationship.reason is None
    assert relationship.target_fact_id == first.facts[0].fact_id
    assert second.abstentions == ()


def test_duplicate_despite_wording_case_and_punctuation():
    first, second = _resolve("c", ["I live in Berlin.", "i live in  berlin!!"])
    _, relationship = _only(second)
    assert relationship.label == "duplicate"
    assert relationship.target_fact_id == first.facts[0].fact_id


def test_repeat_is_never_recorded_as_a_new_fact():
    first, second = _resolve("c", ["I live in Berlin.", "I live in Berlin."])
    # Classification annotates candidates; it does not merge, drop or rewrite
    # them, and the earlier record stays exactly as produced.
    assert [fact.value for fact in first.facts] == ["Berlin"]
    assert [fact.value for fact in second.facts] == ["Berlin"]


def test_present_tense_marker_makes_a_temporal_update():
    _, second = _resolve("c", ["I live in Berlin.", "I now live in Paris."])
    fact, relationship = _only(second)
    assert fact.value == "Paris"
    assert relationship.label == "temporal_update"
    assert "temporal=now" in relationship.evidence


@pytest.mark.parametrize("marker", ["I currently live in Paris.",
                                    "I still live in Paris."])
def test_adverb_forms_extend_the_temporal_signal(marker):
    _, second = _resolve("c", ["I live in Berlin.", marker])
    _, relationship = _only(second)
    assert relationship.label == "temporal_update"


def test_past_tense_marker_makes_a_historical_observation():
    _, second = _resolve("c", ["I live in Berlin.", "I used to live in Paris."])
    fact, relationship = _only(second)
    assert fact.value == "Paris"
    assert relationship.label == "historical_observation"
    assert "temporal=past" in relationship.evidence


def test_newer_year_updates_and_older_year_is_historical():
    _, update = _resolve("c", ["I was born in 1990.", "I was born in 1995."])
    _, update_relationship = _only(update)
    assert update_relationship.label == "temporal_update"
    assert "later_year" in update_relationship.evidence

    _, historical = _resolve("c", ["I was born in 1995.",
                                   "I was born in 1990."])
    _, historical_relationship = _only(historical)
    assert historical_relationship.label == "historical_observation"
    assert "earlier_year" in historical_relationship.evidence


def test_additive_predicate_is_a_distinct_fact():
    _, second = _resolve("c", ["I like jazz.", "I like blues."])
    fact, relationship = _only(second)
    assert fact.predicate == "likes"
    assert relationship.label == "distinct_fact"
    assert "additive_value" in relationship.evidence


def test_similar_titles_are_not_merged():
    # The two titles score high on similarity but name different works;
    # merging them on similarity alone would be a false merge.
    _, second = _resolve("c", ["I read The Quiet Harbor.",
                               "I read The Silent Harbor."])
    _, relationship = _only(second)
    assert relationship.label == "distinct_fact"


def test_conflict_without_time_evidence_is_unresolved():
    first, second = _resolve("c", ["My favorite color is teal.",
                                   "My favorite color is green."])
    fact, relationship = _only(second)
    assert fact.predicate == "favorite_color"
    assert relationship.label == "unresolved"
    assert relationship.reason == "value_conflict_without_time"
    assert relationship.target_fact_id == first.facts[0].fact_id
    assert any(item.startswith("relationship_unresolved:")
               for item in second.abstentions)
    # Both conflicting values survive as separate candidates.
    assert first.facts[0].value == "teal"
    assert second.facts[0].value == "green"


def test_polarity_conflict_stays_unresolved():
    _, second = _resolve("c", ["I am allergic to peanuts.",
                               "I am not allergic to peanuts."])
    fact, relationship = _only(second)
    assert fact.polarity == "negative"
    assert relationship.label == "unresolved"
    assert relationship.reason == "polarity_conflict"


def test_cessation_marker_makes_a_polarity_change_temporal():
    _, second = _resolve("c", ["I work at Acme.", "I no longer work at Acme."])
    fact, relationship = _only(second)
    assert fact.polarity == "negative"
    assert relationship.label == "temporal_update"
    assert relationship.reason is None
    assert "cessation_marker" in relationship.evidence


def test_every_relationship_carries_evidence_and_confidence():
    records = _resolve("c", ["I live in Berlin.", "I now live in Paris.",
                             "I like jazz."])
    for record in records:
        for relationship in record.relationships:
            assert relationship.label in LABELS
            assert relationship.evidence
            assert 0.0 <= relationship.confidence <= 1.0


def test_evidence_never_repeats_the_conversation_value():
    _, second = _resolve("c", ["My favorite color is teal.",
                              "My favorite color is green."])
    joined = " ".join(second.relationships[0].evidence).casefold()
    assert "teal" not in joined
    assert "green" not in joined


def test_conversations_are_classified_in_isolation():
    resolver = Resolver()
    resolver.resolve(_message("I live in Berlin.", 0, "c1"))
    other = resolver.resolve(_message("I live in Berlin.", 0, "c2")).record
    _, relationship = _only(other)
    assert relationship.label == "distinct_fact"
    assert relationship.evidence == ("no_prior_fact",)


def test_index_refuses_a_conversation_mismatch():
    index = FactIndex("c1")
    with pytest.raises(ValueError):
        classify_relationships(index, _message("I live in Berlin.", 0, "c2"),
                               ())


def test_registry_keeps_one_index_per_conversation():
    registry = RelationshipRegistry()
    assert registry.index("c1") is registry.index("c1")
    assert registry.index("c1") is not registry.index("c2")
    assert registry.conversations() == ("c1", "c2")
    with pytest.raises(ValueError):
        registry.index("")
