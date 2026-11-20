"""Contract tests for the supported semantic ingestion interface.

These tests exercise the composition of the resolution layer with a canonical
sink.  They use the deterministic encoder/verifier fixtures and never touch the
network or a real model.
"""

from __future__ import annotations

import json

import pytest

from minemos.resolution import ResolutionConfig, SourceMessage
from minemos.semantic import (
    EngineSink,
    SemanticIngestionError,
    SemanticIngestor,
    ServiceSink,
    conversation_timestamp_mode,
)
from tests._retrieval_helpers import make_memory

ACTIVE = ResolutionConfig(mode="active")
STAMP = 1_700_000_000.0


def _message(text: str, *, message_id: str, order: int = 0,
             timestamp=STAMP, speaker="user",
             conversation_id="conv-1") -> SourceMessage:
    return SourceMessage(
        conversation_id=conversation_id,
        message_id=message_id,
        text=text,
        speaker=speaker,
        timestamp=timestamp,
        order=order,
    )


class RecordingSink:
    def __init__(self) -> None:
        self.calls = []

    def absorb(self, *, subject, predicate, object, timestamp, confidence,
               source, ingestion_id=None):
        self.calls.append({
            "subject": subject, "predicate": predicate, "object": object,
            "timestamp": timestamp, "confidence": confidence,
            "source": dict(source), "ingestion_id": ingestion_id,
        })
        return {"action": "insert", "memory_id": f"m{len(self.calls)}"}


class FlakySink(RecordingSink):
    def absorb(self, **kwargs):
        if not self.calls:
            self.calls.append(kwargs)
            raise RuntimeError("sink unavailable")
        return super().absorb(**kwargs)


class FakePool:
    def __init__(self) -> None:
        self.calls = []

    def absorb(self, tenant, *, subject, predicate, object, timestamp,
               confidence, source, ingestion_id=None):
        self.calls.append({
            "tenant": tenant, "subject": subject, "predicate": predicate,
            "object": object, "timestamp": timestamp,
            "confidence": confidence, "source": dict(source),
            "ingestion_id": ingestion_id,
        })
        return {"action": "insert", "memory_id": f"m{len(self.calls)}",
                "skipped": False}


def test_non_active_mode_is_refused():
    with pytest.raises(SemanticIngestionError):
        SemanticIngestor(None, config=ResolutionConfig())
    with pytest.raises(SemanticIngestionError):
        SemanticIngestor(None, config=ResolutionConfig(mode="shadow"))


def test_dry_run_plans_without_writing():
    sink = RecordingSink()
    ingestor = SemanticIngestor(None, config=ACTIVE)
    report = ingestor.ingest_conversation(
        [_message("My name is Samuel and I work at Acme Corp.",
                  message_id="m1")])
    assert report.dry_run is True
    assert report.absorbed == 0
    assert sink.calls == []
    assert report.observations == 2
    assert report.ingested[0].chunks[0].chunk.text == (
        "My name is Samuel and I work at Acme Corp.")


def test_engine_sink_writes_through_canonical_engine():
    memory = make_memory(capacity=32)
    ingestor = SemanticIngestor(EngineSink(memory), config=ACTIVE)
    report = ingestor.ingest_conversation(
        [_message("My name is Samuel and I work at Acme Corp.",
                  message_id="m1")])
    assert report.absorbed == 2
    assert report.failed_observations == 0
    rows = memory.recall("user")
    predicates = {row["predicate"] for row in rows}
    assert {"name", "works_at"} <= predicates
    texts = {row["text"] for row in rows}
    assert "user name Samuel." in texts
    assert "user works_at Acme Corp." in texts


def test_service_sink_forwards_provenance_fields():
    pool = FakePool()
    ingestor = SemanticIngestor(ServiceSink(pool, "tenant-a"), config=ACTIVE)
    report = ingestor.ingest_conversation(
        [_message("I live in Berlin.", message_id="m1")])
    assert report.absorbed == 1
    call = pool.calls[0]
    assert call["tenant"] == "tenant-a"
    assert call["source"]["source_id"]
    assert call["source"]["source_type"] == "conversation"
    assert call["source"]["conversation_id"] == "conv-1"
    assert call["source"]["original_timestamp"] == STAMP
    assert call["ingestion_id"] == report.ingestion_ids()[0]


def test_known_ingestion_ids_make_replay_idempotent():
    pool = FakePool()
    first = SemanticIngestor(ServiceSink(pool, "t"), config=ACTIVE)
    report = first.ingest_conversation(
        [_message("I work at Acme Corp.", message_id="m1")])
    assert report.absorbed == 1
    ids = report.ingestion_ids()

    second = SemanticIngestor(
        ServiceSink(pool, "t"), config=ACTIVE, known_ingestion_ids=ids)
    replay = second.ingest_conversation(
        [_message("I work at Acme Corp.", message_id="m1")])
    assert replay.skipped == 1
    assert replay.absorbed == 0
    assert len(pool.calls) == 1


def test_sink_reported_skip_counts_as_skipped():
    class SkippingPool(FakePool):
        def absorb(self, *args, **kwargs):
            super().absorb(*args, **kwargs)
            return {"action": "duplicate", "memory_id": "m1",
                    "skipped": True}

    pool = SkippingPool()
    report = SemanticIngestor(
        ServiceSink(pool, "t"), config=ACTIVE).ingest_conversation(
        [_message("I work at Acme Corp.", message_id="m1")])
    assert report.skipped == 1
    assert report.absorbed == 0


def test_mixed_timestamps_abstain_on_untimed_message():
    sink = RecordingSink()
    report = SemanticIngestor(sink, config=ACTIVE).ingest_conversation([
        _message("I work at Acme Corp.", message_id="m1", order=0),
        _message("My name is Samuel.", message_id="m2", order=1,
                 timestamp=None),
    ])
    assert report.timestamp_mode == "mixed"
    assert report.timestamped_messages == 1
    assert report.untimed_messages == 1
    assert report.reasons().get("timestamp_unavailable", 0) >= 1
    assert all(call["source"]["original_timestamp"] is not None
               for call in sink.calls)


def test_untimed_conversation_requires_explicit_opt_in():
    sink = RecordingSink()
    report = SemanticIngestor(sink, config=ACTIVE).ingest_conversation(
        [_message("I work at Acme Corp.", message_id="m1",
                  timestamp=None)])
    assert report.timestamp_mode == "untimed"
    assert report.absorbed == 0
    assert sink.calls == []
    assert report.reasons().get("timestamp_unavailable")

    allowed = RecordingSink()
    report = SemanticIngestor(
        allowed, config=ACTIVE, ingestion_timestamp=STAMP,
        allow_ingestion_timestamp=True).ingest_conversation(
        [_message("I work at Acme Corp.", message_id="m1",
                  timestamp=None)])
    assert report.absorbed == 1
    assert allowed.calls[0]["timestamp"] == STAMP


def test_sink_failure_is_isolated_per_observation():
    sink = FlakySink()
    report = SemanticIngestor(sink, config=ACTIVE).ingest_conversation(
        [_message("My name is Samuel and I work at Acme Corp.",
                  message_id="m1")])
    assert report.failed_observations == 1
    assert report.absorbed == 1
    assert "RuntimeError" in report.error_types


def test_deterministic_ids_across_independent_runs():
    messages = [
        _message("My name is Samuel.", message_id="m1", order=0),
        _message("I work at Acme Corp.", message_id="m2", order=1),
    ]
    first = SemanticIngestor(
        RecordingSink(), config=ACTIVE).ingest_conversation(messages)
    second = SemanticIngestor(
        RecordingSink(), config=ACTIVE).ingest_conversation(messages)
    assert first.ingestion_ids() == second.ingestion_ids()
    assert len(set(first.ingestion_ids())) == len(first.ingestion_ids())


def test_mixed_conversations_rejected_before_any_write():
    sink = RecordingSink()
    with pytest.raises(SemanticIngestionError):
        SemanticIngestor(sink, config=ACTIVE).ingest_conversation([
            _message("I work at Acme Corp.", message_id="m1"),
            _message("My name is Samuel.", message_id="m2",
                     conversation_id="conv-2"),
        ])
    assert sink.calls == []


def test_messages_are_processed_in_host_order():
    sink = RecordingSink()
    report = SemanticIngestor(sink, config=ACTIVE).ingest_conversation([
        _message("I work at Acme Corp.", message_id="m2", order=1),
        _message("My name is Samuel.", message_id="m1", order=0),
    ])
    assert [item.original_message_id for item in report.ingested] == [
        "m1", "m2"]
    assert sink.calls[0]["predicate"] == "name"


def test_oversized_message_is_segmented_losslessly():
    filler = "x" * 9000
    text = f"My name is Samuel. {filler} I work at Acme Corp."
    report = SemanticIngestor(RecordingSink(), config=ACTIVE
                              ).ingest_conversation(
        [_message(text, message_id="m1")])
    chunks = report.ingested[0].chunks
    assert len(chunks) > 1
    assert max(len(chunk.chunk.text) for chunk in chunks) <= 8192
    assert "".join(chunk.chunk.text for chunk in chunks) == text


def test_summary_never_contains_message_content():
    report = SemanticIngestor(RecordingSink(), config=ACTIVE
                              ).ingest_conversation(
        [_message("My name is Samuel and I work at Acme Corp.",
                  message_id="m1")])
    encoded = json.dumps(report.summary())
    assert "Samuel" not in encoded
    assert "Acme" not in encoded


def test_negated_fact_is_never_written():
    sink = RecordingSink()
    report = SemanticIngestor(sink, config=ACTIVE).ingest_conversation(
        [_message("I do not work at Acme Corp.", message_id="n1")])
    assert report.absorbed == 0
    assert sink.calls == []
    assert report.reasons().get("non_asserted_polarity") == 1


def test_hypothetical_clause_is_abstained():
    sink = RecordingSink()
    report = SemanticIngestor(sink, config=ACTIVE).ingest_conversation(
        [_message("If I worked at Google I would move.", message_id="h1")])
    assert report.absorbed == 0
    assert report.observations == 0
    assert sink.calls == []


def test_repeated_fact_is_skipped_as_duplicate():
    sink = RecordingSink()
    report = SemanticIngestor(sink, config=ACTIVE).ingest_conversation([
        _message("I work at Acme Corp.", message_id="d1", order=0),
        _message("I work at Acme Corp.", message_id="d2", order=1),
    ])
    assert report.absorbed == 1
    assert report.reasons().get("relationship_duplicate") == 1


def test_pronoun_after_named_person_is_not_attributed_to_user():
    sink = RecordingSink()
    report = SemanticIngestor(sink, config=ACTIVE).ingest_conversation([
        _message("Daniel lives in Paris.", message_id="a1", order=0),
        _message("He works at Acme Corp.", message_id="a2", order=1),
    ])
    subjects = {call["subject"] for call in sink.calls}
    assert "user" not in subjects
    # "He ..." is either unresolved or below the extraction threshold; either
    # way it must not become an assertion about the speaker.
    assert (report.reasons().get("subject_unresolved", 0)
            + report.reasons().get("below_confidence_threshold", 0)) >= 1


def test_out_of_order_input_keeps_host_order_and_timestamps():
    first = _message("I work at Acme Corp.", message_id="m1", order=0,
                     timestamp=STAMP)
    second = _message("I live in Berlin.", message_id="m2", order=1,
                      timestamp=STAMP + 60)
    report = SemanticIngestor(RecordingSink(), config=ACTIVE
                              ).ingest_conversation([second, first])
    assert [item.original_message_id for item in report.ingested] == [
        "m1", "m2"]
    stamps = [item.timestamp for item in report.conversions[0].observations]
    assert stamps == [STAMP]


def test_timestamp_mode_helper():
    timed = [_message("a", message_id="m1", order=0)]
    untimed = [_message("a", message_id="m1", order=0, timestamp=None)]
    assert conversation_timestamp_mode([]) == "empty"
    assert conversation_timestamp_mode(timed) == "timed"
    assert conversation_timestamp_mode(untimed) == "untimed"
    assert conversation_timestamp_mode(timed + untimed) == "mixed"
