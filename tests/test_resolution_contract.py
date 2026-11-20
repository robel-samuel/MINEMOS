"""Output contract: every resolution record is structurally valid.

The record is the only thing the rest of the system sees, so these tests
check its invariants directly: the source text is untouched, every span is in
range, every fact is attributed and evidenced, every relationship is labelled
and, when unresolved, explained, and the JSON form passes the schema gate.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from minemos.resolution import Resolver
from minemos.resolution.contracts import (
    RECORD_VERSION,
    RELATIONSHIP_LABELS,
    STAGE_NAMES,
    NormalizedText,
    NormalizedToken,
    ResolutionContractError,
    SourceMessage,
    validate_record,
)
from minemos.resolution.schema import validate_with_schema

_BASE_TEXT = "My sister now works at Acme and I live in Berlin."


def _record(text: str = _BASE_TEXT, conversation: str = "c"):
    resolver = Resolver()
    return resolver.resolve(SourceMessage(
        conversation_id=conversation,
        message_id="m0",
        text=text,
        speaker="user",
        timestamp=0.0,
        order=0,
    )).record


def _replace(record, **changes):
    return dataclasses.replace(record, **changes)


def test_resolved_record_satisfies_the_contract():
    record = _record()
    validate_record(record)
    assert record.original_text == _BASE_TEXT
    assert record.facts
    assert record.attributions


def test_json_form_passes_the_schema_gate():
    document = _record().to_dict()
    validate_with_schema(document)
    # The document must be plain JSON data, not Python objects.
    round_tripped = json.loads(json.dumps(document))
    assert round_tripped == document
    assert document["version"] == RECORD_VERSION


def test_schema_file_is_valid_json():
    schema_path = Path(__file__).resolve().parents[1] / \
        "minemos" / "resolution" / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["version"]["const"] == RECORD_VERSION


def test_every_fact_is_attributed_and_evidenced():
    record = _record()
    fact_ids = {fact.fact_id for fact in record.facts}
    attributed = {item.fact_id for item in record.attributions
                  if item.fact_id is not None}
    assert fact_ids == attributed
    for fact in record.facts:
        assert fact.evidence
        span = fact.evidence[0]
        assert span.message_id == record.source.message_id
        assert span.conversation_id == record.source.conversation_id
        assert 0 <= span.start <= span.end <= len(record.original_text)


def test_stage_confidences_cover_all_four_stages():
    record = _record()
    assert tuple(record.stage_confidence) == STAGE_NAMES
    for value in record.stage_confidence.values():
        assert 0.0 <= value <= 1.0


def test_normalized_text_must_not_rewrite_the_source():
    record = _record()
    tampered = dataclasses.replace(record.normalized, original="something else")
    with pytest.raises(ResolutionContractError, match="untouched source"):
        validate_record(_replace(record, normalized=tampered))


def test_normalized_text_must_match_its_tokens():
    record = _record()
    tokens = record.normalized.tokens[:-1]
    tampered = dataclasses.replace(record.normalized, tokens=tokens)
    with pytest.raises(ResolutionContractError, match="token rendering"):
        validate_record(_replace(record, normalized=tampered))


def test_tokens_must_be_ordered_by_source_offset():
    record = _record()
    tokens = tuple(reversed(record.normalized.tokens))
    tampered = NormalizedText(original=record.normalized.original,
                              text=record.normalized.text, tokens=tokens)
    with pytest.raises(ResolutionContractError, match="ordered"):
        validate_record(_replace(record, normalized=tampered))


def test_token_offsets_must_stay_inside_the_source():
    record = _record()
    bad = NormalizedToken("X", 0, len(record.original_text) + 5)
    tampered = NormalizedText(original=record.normalized.original,
                              text="X",
                              tokens=(bad,))
    with pytest.raises(ResolutionContractError, match="outside the source"):
        validate_record(_replace(record, normalized=tampered))


def test_empty_source_is_rejected():
    record = _record()
    empty = dataclasses.replace(record.normalized, original="   ", text="")
    with pytest.raises(ResolutionContractError, match="empty"):
        validate_record(_replace(record, original_text="   ",
                                 normalized=empty))


def test_duplicate_fact_ids_are_rejected():
    record = _record()
    assert len(record.facts) >= 2
    facts = (record.facts[0], record.facts[0])
    with pytest.raises(ResolutionContractError, match="duplicate fact id"):
        validate_record(_replace(record, facts=facts))


def test_unresolved_relationship_needs_a_reason():
    record = _record()
    unresolved = dataclasses.replace(record.relationships[0],
                                     label="unresolved", reason=None)
    with pytest.raises(ResolutionContractError, match="reason"):
        validate_record(_replace(record, relationships=(unresolved,)))


def test_relationship_label_must_be_known():
    record = _record()
    bad = dataclasses.replace(record.relationships[0], label="maybe")
    with pytest.raises(ResolutionContractError, match="not allowed"):
        validate_record(_replace(record, relationships=(bad,)))


def test_relationship_target_must_be_a_known_fact():
    record = _record()
    bad = dataclasses.replace(record.relationships[0],
                              target_fact_id="missing:fact:0")
    with pytest.raises(ResolutionContractError, match="target_fact_id"):
        validate_record(_replace(record, relationships=(bad,)))


def test_relationship_target_may_point_at_an_earlier_fact():
    record = _record()
    relation = record.relationships[0]
    earlier = dataclasses.replace(relation, target_fact_id="m-1:fact:0")
    validate_record(_replace(record, relationships=(earlier,)),
                    earlier_fact_ids=("m-1:fact:0",))


def test_fact_evidence_cannot_leave_the_message():
    record = _record()
    fact = record.facts[0]
    span = dataclasses.replace(fact.evidence[0], message_id="other")
    tampered = dataclasses.replace(fact, evidence=(span,))
    facts = (tampered,) + tuple(record.facts[1:])
    with pytest.raises(ResolutionContractError, match="recorded message"):
        validate_record(_replace(record, facts=facts))


def test_fact_evidence_cannot_leave_the_conversation():
    record = _record()
    fact = record.facts[0]
    span = dataclasses.replace(fact.evidence[0], conversation_id="other")
    tampered = dataclasses.replace(fact, evidence=(span,))
    facts = (tampered,) + tuple(record.facts[1:])
    with pytest.raises(ResolutionContractError, match="conversation"):
        validate_record(_replace(record, facts=facts))


def test_attribution_status_must_match_its_subject():
    record = _record()
    attribution = record.attributions[0]
    assert attribution.fact_id is not None
    tampered = dataclasses.replace(attribution, status="unknown")
    with pytest.raises(ResolutionContractError, match="match the subject"):
        validate_record(_replace(record,
                                 attributions=(tampered,)))


def test_unknown_attribution_needs_an_unknown_subject_kind():
    record = _record()
    attribution = record.attributions[0]
    subject = dataclasses.replace(attribution.subject, kind="speaker",
                                  status="unknown")
    tampered = dataclasses.replace(attribution, subject=subject,
                                   status="unknown")
    with pytest.raises(ResolutionContractError, match="unknown subject kind"):
        validate_record(_replace(record, attributions=(tampered,)))


def test_attributions_must_name_a_fact_in_the_record():
    record = _record()
    extra = dataclasses.replace(record.attributions[0],
                                fact_id="missing:fact:0")
    with pytest.raises(ResolutionContractError, match="unknown"):
        validate_record(_replace(record,
                                 attributions=record.attributions + (extra,)))


def test_one_attribution_per_fact():
    record = _record()
    duplicated = (record.attributions[0], record.attributions[0])
    with pytest.raises(ResolutionContractError, match="two attributions"):
        validate_record(_replace(record, attributions=duplicated))


def test_stage_confidence_keys_are_checked():
    record = _record()
    with pytest.raises(ResolutionContractError, match="unknown stage"):
        validate_record(_replace(record, stage_confidence={
            "normalization": 0.9, "invented": 0.5}))


def test_empty_abstention_reason_is_rejected():
    record = _record()
    with pytest.raises(ResolutionContractError, match="cannot be empty"):
        validate_record(_replace(record, abstentions=("",)))


def test_relationship_labels_and_stage_names_are_the_published_sets():
    assert set(RELATIONSHIP_LABELS) == {
        "duplicate", "temporal_update", "historical_observation",
        "distinct_fact", "unresolved"}
    assert STAGE_NAMES == ("normalization", "interpretation", "identity",
                           "relationships")


def test_schema_gate_rejects_unknown_and_missing_properties():
    document = _record().to_dict()
    with pytest.raises(ResolutionContractError, match="unknown top-level"):
        validate_with_schema(dict(document, extra=True))
    without_provider = dict(document)
    without_provider.pop("provider")
    with pytest.raises(ResolutionContractError, match="missing required"):
        validate_with_schema(without_provider)
    with pytest.raises(ResolutionContractError, match="version"):
        validate_with_schema(dict(document, version="minemos.resolution/0"))


def test_records_are_immutable():
    record = _record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.original_text = "changed"
