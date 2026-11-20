"""V8: conversion of resolution records into native MINEMOS observations.

These tests cover the shipped converter (``minemos.resolution.native``) only:
eligibility gating, timestamp fallback, deterministic identifiers and the two
native output forms.  No test activates absorption in the product; every
record is resolved in-process and every engine use is local.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

import pytest

from minemos.resolution import (
    ResolutionConfig,
    SourceMessage,
    SubjectRef,
    resolve_messages,
)
from minemos.resolution.native import (
    NATIVE_ID_MAX_CHARS,
    NATIVE_OBJECT_MAX_CHARS,
    NATIVE_PREDICATE_MAX_CHARS,
    NATIVE_SUBJECT_MAX_CHARS,
    native_identifier,
    native_ingestion_id,
    native_observations,
    resolve_timestamp,
)
from minemos_engine.structured_candidate import Candidate

ACTIVE = ResolutionConfig(mode="active")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _record(text: str = "I live in Berlin.", *, timestamp=100.0,
            message_id="m-1", conversation="c-1"):
    result = resolve_messages([SourceMessage(
        conversation_id=conversation,
        message_id=message_id,
        text=text,
        speaker="user",
        timestamp=timestamp,
    )])[0]
    return result.record


def _convert(record, **kwargs):
    kwargs.setdefault("source_id", "conv-2026-01")
    return native_observations(record, ACTIVE, **kwargs)


# -- happy path ------------------------------------------------------------


def test_eligible_fact_becomes_a_native_observation():
    conversion = _convert(_record())

    assert len(conversion.observations) == 1
    assert conversion.rejected == ()
    item = conversion.observations[0]
    assert item.subject == "user"
    assert item.predicate == "lives_in"
    assert item.object == "Berlin"
    assert item.confidence == 0.87
    assert item.timestamp == 100.0
    assert item.timestamp_source == "semantic"
    assert item.source_id == "conv-2026-01"
    assert item.conversation_id == "c-1"
    assert item.message_id == "m-1"
    assert ID_RE.match(item.ingestion_id)


def test_candidate_form_preserves_multi_word_subject():
    record = _record("My sister works at Acme.")
    conversion = _convert(record)
    assert len(conversion.observations) == 1
    candidate = conversion.observations[0].as_candidate()

    assert isinstance(candidate, Candidate)
    assert candidate.subject == "user's sister"
    assert " " in candidate.subject
    assert candidate.predicate == "works_at"
    assert candidate.object == "Acme"
    assert candidate.timestamp == 100.0


def test_service_observation_mapping_is_content_only():
    item = _convert(_record()).observations[0]
    payload = item.to_service_observation()

    assert payload["subject"] == "user"
    assert payload["predicate"] == "lives_in"
    assert payload["object"] == "Berlin"
    assert payload["ingestion_id"] == item.ingestion_id
    assert payload["source"] == {
        "source_id": "conv-2026-01", "conversation_id": "c-1"}


def test_conversion_never_writes_memory():
    from minemos_engine import BlackHoleMemory

    engine = BlackHoleMemory()
    _convert(_record())
    assert engine.memory.slots == []


# -- eligibility -----------------------------------------------------------


def test_shadow_mode_converts_nothing():
    record = _record()
    conversion = native_observations(
        record, ResolutionConfig(mode="shadow"), source_id="s-1")

    assert conversion.observations == ()
    assert [item.reason for item in conversion.rejected] == [
        "mode_is_not_active"]


def test_negative_polarity_is_rejected():
    record = _record("I do not live in Berlin.")
    conversion = _convert(record)

    assert conversion.observations == ()
    assert "non_asserted_polarity" in {
        item.reason for item in conversion.rejected}


def test_hypothetical_modality_is_rejected():
    record = _record()
    fact = replace(record.facts[0], modality="hypothetical")
    conversion = _convert(replace(record, facts=(fact,)))

    assert conversion.observations == ()
    assert {item.reason for item in conversion.rejected} == {
        "non_asserted_modality"}


def test_unresolved_subject_is_rejected():
    record = _record()
    fact = replace(record.facts[0], subject=SubjectRef(
        kind="unknown", label="somebody", confidence=0.3,
        status="unresolved"))
    conversion = _convert(replace(record, facts=(fact,)))

    assert conversion.observations == ()
    assert {item.reason for item in conversion.rejected} == {
        "subject_unresolved"}


def test_confidence_below_existing_threshold_is_rejected():
    record = _record()
    fact = replace(record.facts[0], confidence=0.3)
    conversion = _convert(replace(record, facts=(fact,)))

    assert conversion.observations == ()
    assert {item.reason for item in conversion.rejected} == {
        "below_confidence_threshold"}


def test_confidence_at_threshold_is_accepted():
    record = _record()
    threshold = ACTIVE.extraction_confidence
    fact = replace(record.facts[0], confidence=threshold)
    conversion = _convert(replace(record, facts=(fact,)))

    assert len(conversion.observations) == 1
    assert conversion.observations[0].confidence == round(threshold, 4)


def test_duplicate_candidate_within_one_record_is_rejected_once():
    record = _record()
    fact = record.facts[0]
    conversion = _convert(replace(record, facts=(fact, fact)))

    assert len(conversion.observations) == 1
    assert len(conversion.rejected) == 1
    assert conversion.rejected[0].reason == "duplicate_candidate"


def test_native_validation_limits_are_enforced_before_ingestion():
    record = _record()
    base = record.facts[0]

    too_long_object = _convert(replace(record, facts=(
        replace(base, value="x" * (NATIVE_OBJECT_MAX_CHARS + 1)),)))
    assert too_long_object.observations == ()
    assert too_long_object.rejected[0].reason == "object_too_long"

    too_long_predicate = _convert(replace(record, facts=(
        replace(base, predicate="p" * (NATIVE_PREDICATE_MAX_CHARS + 1)),)))
    assert too_long_predicate.rejected[0].reason == "predicate_too_long"

    too_long_subject = _convert(replace(record, facts=(
        replace(base, subject=SubjectRef(
            kind="named", label="A" * (NATIVE_SUBJECT_MAX_CHARS + 1),
            confidence=0.9)),)))
    assert too_long_subject.rejected[0].reason == "subject_too_long"


def test_control_characters_are_rejected():
    record = _record()
    fact = replace(record.facts[0], value="Ber\x00lin")
    conversion = _convert(replace(record, facts=(fact,)))

    assert conversion.observations == ()
    assert conversion.rejected[0].reason == "object_control_characters"


def test_record_with_no_facts_returns_empty_conversion():
    record = _record("The weather was nice yesterday.")
    conversion = _convert(record)

    assert conversion.observations == ()
    assert conversion.rejected == ()
    assert conversion.timestamp_source is None


# -- timestamps ------------------------------------------------------------


def test_semantic_timestamp_wins():
    assert resolve_timestamp(100.0, 200.0, 300.0) == (100.0, "semantic")


def test_segment_timestamp_is_the_first_fallback():
    assert resolve_timestamp(None, 200.0, 300.0) == (200.0, "segment")


def test_ingestion_timestamp_is_the_last_fallback():
    assert resolve_timestamp(None, None, 300.0) == (300.0, "ingestion")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"),
                                 float("-inf"), None, "later", True])
def test_invalid_values_are_skipped_in_fallback(bad):
    value, source = resolve_timestamp(bad, None, 300.0)
    assert value == 300.0
    assert source == "ingestion"


def test_no_valid_timestamp_rejects_the_record():
    with pytest.raises(ValueError, match="no valid timestamp"):
        resolve_timestamp(None, 0.0, float("nan"))


def test_missing_timestamps_reject_facts_with_specific_reason():
    record = _record(timestamp=None)
    conversion = _convert(record)

    assert conversion.observations == ()
    assert conversion.rejected[0].reason == "invalid_timestamp"


def test_record_timestamp_falls_back_to_segment():
    record = _record(timestamp=None)
    conversion = _convert(record, segment_timestamp=50.0)

    assert conversion.observations[0].timestamp == 50.0
    assert conversion.observations[0].timestamp_source == "segment"


def test_missing_timestamps_use_ingestion_time_never_zero():
    record = _record(timestamp=None)
    conversion = _convert(record, ingestion_timestamp=75.0)

    item = conversion.observations[0]
    assert item.timestamp == 75.0
    assert item.timestamp_source == "ingestion"
    assert item.timestamp != 0.0
    assert item.as_candidate().timestamp == 75.0


def test_disabled_ingestion_fallback_abstains_with_specific_reason():
    record = _record(timestamp=None)
    conversion = _convert(record, ingestion_timestamp=75.0,
                          ingestion_fallback=False)

    assert conversion.observations == ()
    assert conversion.timestamp_source is None
    assert [item.reason for item in conversion.rejected] == \
        ["timestamp_unavailable"]


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan")])
def test_disabled_fallback_treats_invalid_semantic_times_as_missing(bad):
    record = _record(timestamp=None)
    conversion = _convert(record, segment_timestamp=bad,
                          ingestion_timestamp=75.0,
                          ingestion_fallback=False)

    assert conversion.observations == ()
    assert conversion.rejected[0].reason == "timestamp_unavailable"


def test_disabled_fallback_is_not_needed_when_segment_time_exists():
    record = _record(timestamp=None)
    conversion = _convert(record, segment_timestamp=50.0,
                          ingestion_timestamp=75.0,
                          ingestion_fallback=False)

    assert conversion.observations[0].timestamp == 50.0
    assert conversion.observations[0].timestamp_source == "segment"


def test_field_rejection_keeps_precedence_over_timestamp_abstention():
    record = _record(timestamp=None)
    fact = replace(record.facts[0], value="Ber\x00lin")
    conversion = _convert(replace(record, facts=(fact,)),
                          ingestion_fallback=False)

    assert conversion.rejected[0].reason == "object_control_characters"


# -- identifiers -----------------------------------------------------------


def test_valid_identifiers_pass_through_unchanged():
    assert native_identifier("conv-2026.01:a") == "conv-2026.01:a"


@pytest.mark.parametrize("value", [
    "conv 2026/01", "Ünïcode ✓ id", "a" * 300, "x" * 5000,
    "café#1", "  spaced  ", "id\twith\ttabs",
])
def test_generated_identifiers_satisfy_the_native_contract(value):
    result = native_identifier(value)
    assert ID_RE.match(result)
    assert len(result) <= NATIVE_ID_MAX_CHARS


def test_identifier_mapping_is_deterministic():
    value = "conv 2026/01"
    assert native_identifier(value) == native_identifier(value)


def test_identifier_mapping_distinguishes_truncation_collisions():
    prefix = "a" * 200
    first = native_identifier(prefix + "one")
    second = native_identifier(prefix + "two")
    assert first != second


def test_identifier_mapping_distinguishes_unicode_and_ascii():
    assert native_identifier("café") != native_identifier("cafe")


@pytest.mark.parametrize("bad", ["", None, 7])
def test_invalid_identifier_inputs_raise(bad):
    with pytest.raises((TypeError, ValueError)):
        native_identifier(bad)


def test_identifier_keeps_existing_native_ids_traceable():
    record = _record(message_id="already.valid:id-1")
    conversion = _convert(record)
    assert conversion.observations[0].message_id == "already.valid:id-1"


# -- ingestion identifiers -------------------------------------------------


def test_ingestion_id_is_stable_for_the_same_input():
    record = _record()
    first = _convert(record).observations[0].ingestion_id
    second = _convert(record).observations[0].ingestion_id
    assert first == second


def test_ingestion_id_changes_with_content_and_source():
    record = _record()
    base = _convert(record).observations[0].ingestion_id
    other_value = _convert(replace(record, facts=(
        replace(record.facts[0], value="Paris"),))).observations[0]
    other_message = _convert(record, message_id="m-2").observations[0]

    assert base != other_value.ingestion_id
    assert base != other_message.ingestion_id


def test_ingestion_id_matches_the_native_identifier_contract():
    item = _convert(_record()).observations[0]
    assert ID_RE.match(item.ingestion_id)
    assert native_ingestion_id(
        source_id=item.source_id,
        conversation_id=item.conversation_id,
        message_id=item.message_id,
        fact_id=item.fact_id,
        subject=item.subject,
        predicate=item.predicate,
        object_text=item.object,
        timestamp=item.timestamp,
        confidence=item.confidence,
    ) == item.ingestion_id


def test_conversion_summary_is_content_free():
    conversion = _convert(_record())
    dumped = str(conversion.to_dict()).casefold()
    for secret in ("berlin", "lives_in", "user"):
        assert secret not in dumped


def test_converted_confidence_is_finite_and_bounded():
    item = _convert(_record()).observations[0]
    assert math.isfinite(item.confidence)
    assert 0.0 <= item.confidence <= 1.0
