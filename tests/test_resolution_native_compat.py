"""V7: converting resolved facts into the engine's native ingestion input.

These tests measure compatibility.  They do not activate semantic absorption
in the product: the bridge stays closed unless a caller constructs
``ResolutionConfig(mode="active")`` explicitly, and every engine used here is
a throwaway in-memory instance owned by the test.

The native engine input is ``BlackHoleMemory.absorb``, which accepts a
``Candidate``, a ``(subject, predicate, object)`` tuple, or a plain string.
These tests pin what survives that conversion and what does not.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from minemos.resolution import SourceMessage, resolve_messages
from minemos.resolution.bridge import candidates_for, decide
from minemos.resolution.config import DEFAULT_CONFIG, ResolutionConfig
from minemos_engine import BlackHoleMemory, Candidate

ACTIVE = ResolutionConfig(mode="active")


def _record(text: str, timestamp: float = 100.0, message_id: str = "m1",
            order: int = 0, conversation: str = "c1"):
    return resolve_messages([SourceMessage(
        conversation_id=conversation,
        message_id=message_id,
        text=text,
        speaker="user",
        timestamp=timestamp,
        order=order,
    )])[0].record


def _results(pairs):
    return resolve_messages([
        SourceMessage(conversation_id="c1", message_id=f"m{i}", text=text,
                      speaker="user", timestamp=stamp, order=i)
        for i, (text, stamp) in enumerate(pairs)
    ])


# -- field fidelity -----------------------------------------------------------


def test_eligible_fact_converts_to_a_native_candidate():
    record = _record("I live in Berlin.", timestamp=100.0)
    candidates, decisions = candidates_for(record, ACTIVE)

    assert len(record.facts) == 1
    assert decisions[0].accepted and decisions[0].reason == "eligible"
    assert len(candidates) == 1

    fact, candidate = record.facts[0], candidates[0]
    assert candidate.subject == "user"
    assert candidate.predicate == fact.predicate
    assert candidate.object == fact.value
    assert candidate.confidence == round(fact.confidence, 4)
    assert 0.0 <= candidate.confidence <= 1.0
    assert candidate.timestamp == record.source.timestamp


def test_native_candidate_carries_only_five_fields():
    record = _record("I moved to Berlin in 2019.", timestamp=100.0)
    fact = record.facts[0]
    candidate = candidates_for(record, ACTIVE)[0][0]

    assert {item.name for item in fields(Candidate)} == {
        "subject", "predicate", "object", "confidence", "timestamp"}

    # The record knows more than the native input can hold.
    assert fact.evidence and fact.value_span
    assert fact.temporal and fact.temporal[0].year == 2019
    assert fact.polarity == "positive" and fact.modality == "asserted"
    assert fact.language

    # None of those extras reach the candidate; the value year is not the
    # observation timestamp.
    assert candidate.timestamp == 100.0
    assert not hasattr(candidate, "evidence")
    assert not hasattr(candidate, "value_span")
    assert not hasattr(candidate, "temporal")
    assert not hasattr(candidate, "polarity")


def test_derived_subject_is_rendered_as_text():
    record = _record("My sister works at Acme.")
    candidate = candidates_for(record, ACTIVE)[0][0]

    # Identity structure collapses to a rendered string; kind/relation/
    # confidence/antecedent are not representable in a native candidate.
    assert candidate.subject == "user's sister"
    assert record.facts[0].subject.kind == "derived"
    assert record.facts[0].subject.relation == "sister"


# -- native forms and backward compatibility ---------------------------------


def test_candidate_and_tuple_forms_produce_the_same_decision():
    via_candidate = BlackHoleMemory()
    via_tuple = BlackHoleMemory()

    first = via_candidate.absorb(
        Candidate("user", "lives_in", "Berlin", confidence=0.87,
                  timestamp=10.0), timestamp=10.0)
    second = via_tuple.absorb(
        ("user", "lives_in", "Berlin"), timestamp=10.0, confidence=0.87)

    assert first["action"] == second["action"] == "insert"
    assert len(via_candidate.memory.slots) == len(via_tuple.memory.slots) == 1
    left, right = via_candidate.memory.slots[0], via_tuple.memory.slots[0]
    assert left.text == right.text == "user lives_in Berlin."
    assert left.confidence == right.confidence == pytest.approx(0.87)
    assert left.valid_from == right.valid_from == 10.0


def test_string_form_mangles_a_subject_that_contains_a_space():
    engine = BlackHoleMemory()
    engine.absorb("user's sister works_at Acme", timestamp=1.0)
    slot = engine.memory.slots[0]

    # The string form splits on the first two spaces, so the rendered text
    # happens to look right while the parsed identity is wrong.
    assert slot.text == "user's sister works_at Acme."
    assert slot.debug_subject == "user's"
    assert slot.debug_predicate == "sister"
    assert slot.latest_object == "works_at Acme"


# -- what the engine decides, and what the bridge pre-filters -----------------


def test_contradiction_is_redecided_by_the_engine():
    results = _results([("I live in Berlin.", 10.0),
                        ("I now live in Paris.", 20.0)])
    assert results[1].record.relationships[0].label == "temporal_update"

    engine = BlackHoleMemory()
    actions = []
    for result in results:
        candidates, _ = candidates_for(result.record, ACTIVE)
        assert len(candidates) == 1
        actions.append(engine.absorb(
            candidates[0], timestamp=candidates[0].timestamp)["action"])

    assert actions == ["insert", "temporal_update"]
    berlin, paris = engine.memory.slots
    assert berlin.latest_object == "Berlin" and berlin.valid_until == 20.0
    assert paris.latest_object == "Paris" and paris.valid_until is None


def test_duplicate_is_filtered_by_the_bridge_before_the_engine():
    results = _results([("I live in Berlin.", 10.0),
                        ("I live in Berlin.", 20.0)])
    second = results[1].record
    assert second.relationships[0].label == "duplicate"
    assert decide(second, second.facts[0], ACTIVE).reason == \
        "relationship_duplicate"
    assert candidates_for(second, ACTIVE)[0] == []

    # Given both observations, the engine would itself consolidate them.
    direct = BlackHoleMemory()
    direct.absorb(Candidate("user", "lives_in", "Berlin", timestamp=10.0),
                  timestamp=10.0)
    repeated = direct.absorb(
        Candidate("user", "lives_in", "Berlin", timestamp=20.0), timestamp=20.0)
    assert repeated["action"] == "consolidate"
    assert repeated["consolidation_class"] == "duplicate"
    assert len(direct.memory.slots) == 1


def test_shadow_mode_produces_no_native_candidates():
    record = _record("I live in Berlin.")
    assert DEFAULT_CONFIG.mode == "shadow"

    candidates, decisions = candidates_for(record, DEFAULT_CONFIG)
    assert candidates == []
    assert {decision.reason for decision in decisions} == {"mode_is_not_active"}


# -- rejection of unsupported facts -------------------------------------------


@pytest.mark.parametrize("text,reason", [
    ("I am not allergic to peanuts.", "non_asserted_polarity"),
    ("He lives in Rome.", "subject_unresolved"),
])
def test_unsupported_facts_are_rejected_not_reinterpreted(text, reason):
    record = _record(text)
    assert record.facts
    assert decide(record, record.facts[0], ACTIVE).reason == reason
    assert candidates_for(record, ACTIVE)[0] == []
