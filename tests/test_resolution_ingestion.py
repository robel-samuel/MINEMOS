"""End-to-end ingestion: segmentation, identity and the opt-in bridge.

The resolver itself is a pure text processor and accepts messages of any
length.  Hosts therefore need an enforced boundary that segments before
resolution; ``minemos.resolution.ingest`` is that boundary.  These tests use
the real entry points (``Resolver``, the bridge and ``BlackHoleMemory``) and
check that an oversized message is never truncated, that source identity and
offsets survive, that identity state carries across chunk boundaries, and that
a repeated fact is classified duplicate and not written twice.
"""

from __future__ import annotations

import json

import pytest

from minemos.resolution import (
    MAX_MESSAGE_CHARS,
    IngestionPolicy,
    MessageTooLongError,
    Resolver,
    SourceMessage,
    absorb_ingested,
    ingest_conversation,
    ingest_message,
)
from minemos.resolution.config import ResolutionConfig
from minemos_engine import BlackHoleMemory

FILLER = "Light rain fell on the empty street. "


def _message(text: str, message_id: str = "m1", order: int = 0,
             conversation: str = "c", speaker: str = "a",
             message_key: str | None = None) -> SourceMessage:
    return SourceMessage(
        conversation_id=conversation,
        message_id=message_id,
        text=text,
        speaker=speaker,
        timestamp=float(order),
        order=order,
        message_key=message_key,
    )


def _long(head: str, tail: str, fillers: int = 240) -> str:
    return head + FILLER * fillers + tail


# -- segmentation ---------------------------------------------------------


def test_short_message_keeps_original_identity():
    message = _message("I live in Berlin.", message_key="key-1")
    ingested = ingest_message(Resolver(), message)

    assert len(ingested.chunks) == 1
    chunk = ingested.chunks[0]
    assert chunk.original_message_id == "m1"
    assert chunk.conversation_id == "c"
    assert chunk.message_id == "m1"
    assert chunk.char_start == 0
    assert chunk.char_end == len("I live in Berlin.")
    assert chunk.record.source.message_id == "m1"
    assert chunk.record.source.message_key == "key-1"
    assert len(chunk.record.facts) == 1


def test_oversized_message_is_segmented_losslessly():
    text = _long("I live in Berlin. ", "I work at Acme.")
    assert len(text) > MAX_MESSAGE_CHARS

    ingested = ingest_message(Resolver(), _message(text, "long-1"))

    assert len(ingested.chunks) >= 2
    assert all(len(chunk.chunk.text) <= MAX_MESSAGE_CHARS
               for chunk in ingested.chunks)
    assert ingested.max_chunk_chars <= MAX_MESSAGE_CHARS
    assert [chunk.chunk.chunk_index for chunk in ingested.chunks] == \
        list(range(len(ingested.chunks)))
    assert [chunk.message_id for chunk in ingested.chunks] == [
        f"long-1#{index}" for index in range(len(ingested.chunks))]
    assert all(chunk.original_message_id == "long-1"
               for chunk in ingested.chunks)
    assert all(chunk.conversation_id == "c" for chunk in ingested.chunks)
    for chunk in ingested.chunks:
        assert text[chunk.char_start:chunk.char_end] == chunk.chunk.text
        assert chunk.record.original_text == chunk.chunk.text
    joined = "".join(chunk.chunk.text for chunk in ingested.chunks)
    assert "".join(joined.split()) == "".join(text.split())


def test_chunk_resolution_follows_source_order_past_ten_chunks():
    # A small policy cap exercises the ordering path without a long resolve.
    text = FILLER * 100 + "I live in Berlin."
    ingested = ingest_message(
        Resolver(), _message(text, "many"),
        policy=IngestionPolicy(max_chars=200))

    assert len(ingested.chunks) > 10
    assert [chunk.chunk.chunk_index for chunk in ingested.chunks] == \
        list(range(len(ingested.chunks)))
    assert ingested.chunks[-1].chunk.text.endswith("I live in Berlin.")
    for chunk in ingested.chunks:
        assert text[chunk.char_start:chunk.char_end] == chunk.chunk.text
    assert len(ingested.records) == len(ingested.chunks)


def test_reject_policy_refuses_oversized_message_before_any_work():
    resolver = Resolver(ResolutionConfig(mode="active"))
    text = _long("I live in Berlin. ", "I work at Acme.")

    with pytest.raises(MessageTooLongError, match="reject"):
        ingest_message(
            resolver, _message(text, "long-2"),
            policy=IngestionPolicy(overflow="reject"))

    assert resolver.stats()["records"] == 0
    assert resolver.stats()["failures"] == 0


def test_ingestion_policy_rejects_invalid_values():
    with pytest.raises(ValueError, match="overflow"):
        IngestionPolicy(overflow="truncate")
    with pytest.raises(ValueError, match="positive"):
        IngestionPolicy(max_chars=0)
    with pytest.raises(TypeError, match="integer"):
        IngestionPolicy(max_chars=True)


# -- identity across chunks ----------------------------------------------


def test_identity_context_carries_across_a_chunk_boundary():
    text = _long("My brother is called Tom. ", "He lives in Oslo.")
    ingested = ingest_message(Resolver(), _message(text, "long-3"))

    assert len(ingested.chunks) >= 2
    facts = [fact for record in ingested.records for fact in record.facts]
    name = [fact for fact in facts if fact.predicate == "name"]
    location = [fact for fact in facts if fact.predicate == "lives_in"]
    assert name and name[0].value == "Tom"
    assert location and location[0].value == "Oslo"
    # The pronoun was resolved against the name registered in an earlier
    # chunk of the same message, not attached to the speaker.
    assert location[0].subject.kind == "named"
    assert location[0].subject.label == "Tom"
    assert location[0].subject.status == "resolved"


def test_ingest_conversation_orders_messages_and_keeps_scope():
    resolver = Resolver()
    messages = [
        _message("He lives in Oslo.", "m1", order=1),
        _message("My brother is called Tom.", "m0", order=0),
    ]
    ingested = ingest_conversation(resolver, messages)

    assert [item.original_message_id for item in ingested] == ["m0", "m1"]
    fact = ingested[1].chunks[0].record.facts[0]
    assert fact.predicate == "lives_in"
    assert fact.subject.label == "Tom"
    assert fact.subject.status == "resolved"

    with pytest.raises(ValueError, match="one conversation"):
        ingest_conversation(resolver, [
            _message("I live in Berlin.", "a", conversation="c1"),
            _message("I work at Acme.", "b", conversation="c2"),
        ])


# -- duplicate facts and the engine bridge --------------------------------


def test_repeated_fact_across_chunks_is_duplicate_and_written_once():
    text = _long("I live in Berlin. ", "I live in Berlin.")
    resolver = Resolver(ResolutionConfig(mode="active"))
    ingested = ingest_message(resolver, _message(text, "long-4"))

    assert ingested.facts == 2
    assert ingested.duplicates == 1

    engine = BlackHoleMemory()
    outcome = absorb_ingested(
        engine, ingested, config=ResolutionConfig(mode="active"))

    assert outcome["absorbed"] == 1
    assert outcome["skipped"] == 1
    assert len(engine.memory.slots) == 1
    assert engine.memory.slots[0].latest_object == "Berlin"
    reasons = {item["reason"] for item in outcome["decisions"]}
    assert reasons == {"eligible", "relationship_duplicate"}


def test_fact_ids_are_unique_across_chunks():
    text = _long("I live in Berlin. ", "I work at Acme.")
    ingested = ingest_message(Resolver(), _message(text, "long-5"))
    fact_ids = [fact.fact_id for record in ingested.records
                for fact in record.facts]
    assert len(fact_ids) >= 2
    assert len(fact_ids) == len(set(fact_ids))


def test_active_absorption_reaches_the_real_engine():
    text = _long("I live in Berlin. ", "I work at Acme.")
    resolver = Resolver(ResolutionConfig(mode="active"))
    ingested = ingest_message(resolver, _message(text, "long-6"))
    engine = BlackHoleMemory()

    outcome = absorb_ingested(
        engine, ingested, config=ResolutionConfig(mode="active"))

    assert outcome["absorbed"] == ingested.facts
    assert len(engine.memory.slots) == outcome["absorbed"]
    texts = {slot.text for slot in engine.memory.slots}
    assert "a lives_in Berlin." in texts
    assert "a works_at Acme." in texts


def test_shadow_is_the_default_and_absorbs_nothing():
    ingested = ingest_message(
        Resolver(), _message(_long("I live in Berlin. ", "I work at Acme.")))
    engine = BlackHoleMemory()

    outcome = absorb_ingested(engine, ingested)

    assert outcome["absorbed"] == 0
    assert outcome["reason"] == "mode_is_not_active"
    assert engine.memory.slots == []


# -- diagnostics ----------------------------------------------------------


def test_summary_is_content_free():
    ingested = ingest_message(
        Resolver(), _message(_long("Marisol lives in Valparaiso. ",
                                   "Marisol works at Acme.")))
    summary = ingested.summary()

    assert set(summary) == {"conversation", "chunks", "max_chunk_chars",
                            "facts", "duplicates", "unresolved"}
    dumped = json.dumps(summary, ensure_ascii=False).casefold()
    for secret in ("marisol", "valparaiso", "acme"):
        assert secret not in dumped
