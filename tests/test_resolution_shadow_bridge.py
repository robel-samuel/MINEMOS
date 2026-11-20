"""Shadow mode and the opt-in bridge.

Shadow mode is the safe default: it observes and counts, never mutates memory,
and its diagnostics carry counters and hashes rather than message content.  The
bridge is the single place that can hand candidates to the engine, and only
when the caller opts into ``active`` mode.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from minemos.resolution import resolve_messages
from minemos.resolution.bridge import (
    absorb_record,
    candidates_for,
    decide,
    eligibility_counts,
    subject_text,
)
from minemos.resolution.config import DEFAULT_CONFIG, ResolutionConfig
from minemos.resolution.contracts import SourceMessage
from minemos.resolution.shadow import ShadowReport, ShadowRunner, value_token
from minemos_engine import BlackHoleMemory

_SECRET_NAME = "Marisol Quintanilla"
_SECRET_PLACE = "Valparaíso"
_SECRET_TEXT = f"I live in {_SECRET_PLACE} with {_SECRET_NAME}."


def _message(text: str, order: int = 0, conversation: str = "c",
             speaker: str = "user") -> SourceMessage:
    return SourceMessage(
        conversation_id=conversation,
        message_id=f"m{order}",
        text=text,
        speaker=speaker,
        timestamp=float(order),
        order=order,
    )


def _resolve(text: str = _SECRET_TEXT, conversation: str = "c"):
    return resolve_messages([_message(text, 0, conversation)])[0]


# -- shadow mode ---------------------------------------------------------- #

def test_default_configuration_is_shadow_mode():
    assert DEFAULT_CONFIG.mode == "shadow"
    assert DEFAULT_CONFIG.shadow
    assert not DEFAULT_CONFIG.active


def test_shadow_runner_rejects_active_mode():
    with pytest.raises(ValueError, match="shadow mode"):
        ShadowRunner(ResolutionConfig(mode="active"))


def test_shadow_runner_counts_without_side_effects():
    engine = BlackHoleMemory()
    before = len(engine.memory.slots)

    runner = ShadowRunner()
    result = runner.run(_message(_SECRET_TEXT))

    assert result.ok
    assert result.record.facts
    assert runner.report.messages == 1
    assert runner.report.facts == len(result.record.facts)
    assert len(engine.memory.slots) == before


def test_shadow_summary_carries_counters_only():
    runner = ShadowRunner()
    runner.run(_message(_SECRET_TEXT))
    summary = runner.summary()

    assert summary["messages"] == 1
    assert summary["facts"] >= 1
    assert "speaker" in summary["by_subject_kind"]
    assert summary["by_language"]["en"] == 1
    assert set(summary["stage_confidence_mean"]) == {
        "normalization", "interpretation", "identity", "relationships"}

    dumped = json.dumps(summary, ensure_ascii=False).casefold()
    for secret in (_SECRET_NAME, _SECRET_PLACE, "berlin"):
        assert secret.casefold() not in dumped


def test_shadow_log_line_is_content_free_json():
    runner = ShadowRunner()
    runner.run(_message(_SECRET_TEXT))
    line = runner.report.log_line()
    payload = json.loads(line)

    assert set(payload) == {"messages", "facts", "duplicates", "unresolved",
                            "unresolved_subjects", "resolution_only",
                            "baseline_only"}
    assert _SECRET_NAME not in line
    assert _SECRET_PLACE not in line


def test_shadow_compare_buckets_differences():
    record = _resolve("I live in Berlin and I like jazz.")
    baseline = record.record.facts
    assert len(baseline) == 2
    runner = ShadowRunner()
    runner.run(_message("I live in Berlin and I like jazz."), baseline)
    assert runner.report.shared == len(baseline)
    assert runner.report.resolution_only == 0
    assert runner.report.baseline_only == 0

    report = ShadowReport()
    report.compare(record.record,
                   [baseline[0], dataclasses.replace(baseline[1], value="Paris")])
    assert report.shared == 1
    assert report.baseline_only == 1


def test_value_tokens_are_stable_and_opaque():
    first = value_token("Berlin")
    assert first == value_token("  berlin ")
    assert first != value_token("Paris")
    assert "berlin" not in first


def test_run_conversation_orders_messages():
    runner = ShadowRunner()
    messages = [_message("I now live in Berlin.", 1),
                _message("I live in Paris.", 0)]
    results = runner.run_conversation(messages)
    assert [item.record.source.message_id for item in results] == ["m0", "m1"]
    # Sorted feeding means the second message is compared against the first.
    labels = [relationship.label
              for relationship in results[1].record.relationships]
    assert labels == ["temporal_update"]


def test_run_conversation_accepts_a_baseline_map():
    runner = ShadowRunner()
    records = resolve_messages([_message(_SECRET_TEXT, 0)])
    baseline = {records[0].record.source.message_id: records[0].record.facts}
    runner.run_conversation([_message(_SECRET_TEXT, 0)], baseline)
    assert runner.report.shared == len(records[0].record.facts)


# -- bridge --------------------------------------------------------------- #

def test_bridge_is_closed_in_shadow_mode():
    record = _resolve().record
    for fact in record.facts:
        assert decide(record, fact).reason == "mode_is_not_active"

    engine = BlackHoleMemory()
    outcome = absorb_record(engine, record)
    assert outcome["absorbed"] == 0
    assert outcome["skipped"] == len(record.facts)
    assert outcome["reason"] == "mode_is_not_active"
    assert engine.memory.slots == []


def test_active_bridge_builds_engine_candidates():
    record = _resolve().record
    config = ResolutionConfig(mode="active")
    candidates, decisions = candidates_for(record, config)

    assert decisions and all(item.accepted for item in decisions)
    assert all(item.reason == "eligible" for item in decisions)
    assert len(candidates) == len(record.facts)
    assert {candidate.subject for candidate in candidates} == {"user"}
    assert {candidate.predicate for candidate in candidates} == {
        fact.predicate for fact in record.facts}


def test_negated_fact_is_not_forwarded():
    record = _resolve("I am not allergic to peanuts.").record
    config = ResolutionConfig(mode="active")
    decisions = [decide(record, fact, config) for fact in record.facts]
    assert decisions[0].reason == "non_asserted_polarity"
    assert candidates_for(record, config)[0] == []


def test_unresolved_pronoun_subject_is_not_forwarded():
    record = _resolve("He lives in Rome.").record
    config = ResolutionConfig(mode="active")
    assert record.facts
    assert decide(record, record.facts[0], config).reason == \
        "subject_unresolved"
    assert subject_text(record.facts[0].subject, "user") is None


def test_duplicate_is_not_forwarded_again():
    records = resolve_messages([_message("I live in Berlin.", 0),
                                _message("I live in Berlin.", 1)])
    second = records[1].record
    config = ResolutionConfig(mode="active")
    assert second.relationships[0].label == "duplicate"
    assert decide(second, second.facts[0], config).reason == \
        "relationship_duplicate"


def test_unresolved_relationship_is_not_forwarded():
    records = resolve_messages([
        _message("My favorite color is teal.", 0),
        _message("My favorite color is green.", 1),
    ])
    second = records[1].record
    config = ResolutionConfig(mode="active")
    assert second.relationships[0].label == "unresolved"
    assert decide(second, second.facts[0], config).reason == \
        "relationship_unresolved"


def test_relation_subject_is_named_not_the_speaker():
    record = _resolve("My sister works at Acme.").record
    kinds = {fact.subject.kind for fact in record.facts}
    assert kinds == {"derived"}
    fact = record.facts[0]
    assert subject_text(fact.subject, "user") != "user"
    assert subject_text(fact.subject, "user").endswith("sister")


def test_active_absorption_writes_only_the_eligible_facts():
    record = _resolve().record
    config = ResolutionConfig(mode="active")
    engine = BlackHoleMemory()
    outcome = absorb_record(engine, record, config)

    assert outcome["absorbed"] == len(record.facts)
    assert len(engine.memory.slots) == len(record.facts)


def test_eligibility_counts_never_builds_candidates():
    record = _resolve("I am not allergic to peanuts.").record
    counts = eligibility_counts(record, ResolutionConfig(mode="active"))
    assert counts == {"non_asserted_polarity": 1}
