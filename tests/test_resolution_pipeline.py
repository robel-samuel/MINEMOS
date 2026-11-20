"""Pipeline: the four stages in order, with no memory side effects.

These tests cover what only the assembled pipeline can show: stage
composition, conversation-scoped state, failure isolation, operator counters
and the guarantee that a record validates before it leaves the layer.
"""

from __future__ import annotations

import pytest

from minemos.resolution import Resolver, resolve_messages
from minemos.resolution import pipeline as pipeline_module
from minemos.resolution.config import ResolutionConfig
from minemos.resolution.contracts import (
    STAGE_NAMES,
    SourceMessage,
    validate_record,
)


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


def _resolve(conversation: str, texts):
    return Resolver().resolve_conversation(
        [_message(text, order, conversation)
         for order, text in enumerate(texts)])


def test_resolved_record_is_valid_and_complete():
    result = _resolve("c", ["I live in Berlin."])[0]
    record = result.record

    validate_record(record)
    assert result.ok
    assert result.summary["status"] == "ok"
    assert tuple(record.stage_confidence) == STAGE_NAMES
    assert record.provider == "deterministic:hash"
    assert record.original_text == "I live in Berlin."
    assert record.facts[0].value == "Berlin"


def test_disabled_mode_returns_a_valid_empty_record():
    resolver = Resolver(ResolutionConfig(mode="disabled"))
    result = resolver.resolve(_message("I live in Berlin."))

    validate_record(result.record)
    assert result.summary == {"status": "disabled"}
    assert result.record.facts == ()
    assert result.record.abstentions == ("resolution_disabled:disabled",)
    assert result.record.attributions[0].fact_id is None
    assert result.record.attributions[0].subject.kind == "speaker"


def test_empty_text_is_rejected_at_the_boundary():
    for text in ("", "   "):
        with pytest.raises(ValueError, match="non-empty"):
            SourceMessage(conversation_id="c", message_id="m", text=text,
                          speaker="user", timestamp=0.0, order=0)


def test_resolve_rejects_non_messages():
    with pytest.raises(TypeError):
        Resolver().resolve("I live in Berlin.")


def test_resolution_is_deterministic():
    first = _resolve("c", ["i live in berllin", "My sister works at Acme."])
    second = _resolve("c", ["i live in berllin", "My sister works at Acme."])
    assert [item.record.to_dict() for item in first] == \
        [item.record.to_dict() for item in second]


def test_pronoun_resolves_only_inside_its_conversation():
    records = _resolve("c", ["My sister's name is Clara.",
                             "She lives in Rome."])
    resolved = records[1].record.facts[0]
    assert resolved.subject.status == "resolved"
    assert resolved.subject.label == "Clara"
    assert resolved.value == "Rome"

    elsewhere = Resolver().resolve(_message("She lives in Rome.", 0, "other"))
    unresolved = elsewhere.record.facts[0]
    assert unresolved.subject.status == "unknown"
    assert unresolved.subject.kind == "unknown"


def test_relationships_use_only_earlier_messages_of_the_conversation():
    records = _resolve("c", ["I live in Berlin.", "I now live in Paris."])
    assert records[1].record.relationships[0].label == "temporal_update"
    assert records[1].record.relationships[0].target_fact_id == \
        records[0].record.facts[0].fact_id


def test_resolve_conversation_orders_and_scopes_messages():
    resolver = Resolver()
    results = resolver.resolve_conversation([
        _message("I now live in Paris.", 1),
        _message("I live in Berlin.", 0),
    ])
    assert [item.record.source.message_id for item in results] == ["m0", "m1"]

    with pytest.raises(ValueError, match="one conversation"):
        resolver.resolve_conversation([_message("a", 0, "c1"),
                                       _message("b", 0, "c2")])


def test_resolve_messages_groups_by_conversation():
    results = resolve_messages([_message("I live in Berlin.", 0, "a"),
                                _message("I live in Paris.", 0, "b")])
    assert [item.record.source.conversation_id for item in results] == \
        ["a", "b"]
    for item in results:
        validate_record(item.record)
        assert item.record.relationships[0].label == "distinct_fact"


def test_stage_failure_becomes_a_degraded_record(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("stage exploded")

    monkeypatch.setattr(pipeline_module, "interpret", explode)

    resolver = Resolver()
    result = resolver.resolve(_message("I live in Berlin."))

    assert not result.ok
    assert result.summary["status"] == "degraded"
    assert result.record.abstentions == ("pipeline_error:RuntimeError",)
    assert result.record.facts == ()
    assert result.record.attributions[0].scope == "message"
    assert resolver.stats()["failures"] == 1


def test_pipeline_continues_after_a_stage_failure():
    resolver = Resolver()
    original = pipeline_module.interpret
    calls = {"n": 0}

    def first_call_fails(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first only")
        return original(*args, **kwargs)

    pipeline_module.interpret = first_call_fails
    try:
        first = resolver.resolve(_message("I live in Berlin.", 0))
        second = resolver.resolve(_message("I live in Paris.", 1))
    finally:
        pipeline_module.interpret = original

    assert not first.ok
    assert second.ok
    assert resolver.stats() == {"records": 2, "failures": 1,
                                "conversations": 1, "tracked_facts": 1}


def test_stats_track_conversations_and_facts():
    resolver = Resolver()
    resolver.resolve(_message("I live in Berlin.", 0, "a"))
    resolver.resolve(_message("I like jazz.", 0, "b"))
    resolver.resolve(_message("I live in Paris.", 1, "a"))

    stats = resolver.stats()
    assert stats["records"] == 3
    assert stats["failures"] == 0
    assert stats["conversations"] == 2
    assert stats["tracked_facts"] == 3


def test_summary_holds_no_conversation_content():
    result = Resolver().resolve(
        _message("I live in Valparaíso and I like jazz.", 0, "c"))
    dumped = repr(result.summary)
    assert "Valparaíso" not in dumped
    assert result.summary["message"] == "m0"
    assert result.summary["language"] == "en"
