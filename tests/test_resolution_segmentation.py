"""Segmentation: long messages are split before resolution, never truncated.

The resolver reads one message at a time.  A host that accepts long messages
must hand it chunks that fit the 8,192-character segment cap; these tests cover
the chunk boundary policy, offset accuracy, and the behaviour of the resolver
across chunks (facts, identity and duplicates).
"""

from __future__ import annotations

import pytest

from minemos.resolution import (
    MAX_MESSAGE_CHARS,
    Resolver,
    SourceMessage,
    segment_message,
)
from minemos.resolution.segments import MessageChunk

_FILLER = "The weather is mild today. "


def _filler(min_length: int) -> str:
    """Whole filler sentences, at least ``min_length`` characters."""
    return _FILLER * (min_length // len(_FILLER) + 1)


def _message(text: str, message_id: str, order: int = 0) -> SourceMessage:
    return SourceMessage(
        conversation_id="c",
        message_id=message_id,
        text=text,
        speaker="a",
        timestamp=float(order),
        order=order,
    )


def _non_whitespace(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def _assert_covers_source(text: str, chunks) -> None:
    for chunk in chunks:
        assert isinstance(chunk, MessageChunk)
        assert 0 < len(chunk.text) <= MAX_MESSAGE_CHARS
        assert text[chunk.char_start:chunk.char_end] == chunk.text
    positions = [chunk.chunk_index for chunk in chunks]
    assert positions == list(range(len(chunks)))
    assert all(chunk.chunk_count == len(chunks) for chunk in chunks)
    rebuilt = "".join(chunk.text for chunk in chunks)
    assert _non_whitespace(rebuilt) == _non_whitespace(text)


def test_exactly_8192_characters_is_one_chunk():
    text = "a" * MAX_MESSAGE_CHARS
    assert len(text) == MAX_MESSAGE_CHARS
    chunks = segment_message(text, "m0")
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert (chunks[0].char_start, chunks[0].char_end) == (0, len(text))
    assert chunks[0].identity == "m0"


def test_8193_characters_split_at_a_sentence_boundary():
    prefix = "My brother is called Sam. "
    text = prefix + "a" * (8193 - len(prefix))
    assert len(text) == 8193
    chunks = segment_message(text, "m1")
    assert len(chunks) == 2
    _assert_covers_source(text, chunks)
    assert chunks[0].text == prefix
    assert chunks[1].char_start == len(prefix)


def test_a_newline_is_used_when_no_sentence_boundary_fits():
    body = "a" * 7000
    text = body + "\n" + "b" * 3000
    chunks = segment_message(text, "m2")
    assert [chunk.text for chunk in chunks] == [body, "b" * 3000]
    assert chunks[0].char_end == 7000
    assert chunks[1].char_start == 7001


def test_unbroken_text_is_hard_cut_without_loss():
    text = "x" * 20000
    chunks = segment_message(text, "m3")
    assert [len(chunk.text) for chunk in chunks] == [8192, 8192, 3616]
    _assert_covers_source(text, chunks)
    assert "".join(chunk.text for chunk in chunks) == text


def test_unicode_text_is_measured_in_characters_not_bytes():
    text = ("ሰላም ዓለም! " * 4000)[:20000]
    chunks = segment_message(text, "m4")
    assert all(len(chunk.text) <= MAX_MESSAGE_CHARS for chunk in chunks)
    _assert_covers_source(text, chunks)


def test_blank_text_produces_no_chunks():
    assert segment_message("   \n\t ", "m5") == ()


@pytest.mark.parametrize("max_chars", [0, -1])
def test_non_positive_max_chars_is_rejected(max_chars):
    with pytest.raises(ValueError):
        segment_message("text", "m6", max_chars=max_chars)


def test_facts_on_both_sides_of_a_segment_boundary_are_kept():
    text = ("My brother is called Sam. " + _filler(8200)
            + "My sister lives in Oslo.")
    assert len(text) > MAX_MESSAGE_CHARS
    chunks = segment_message(text, "m7")
    resolver = Resolver()
    records = [resolver.resolve(_message(chunk.text, chunk.identity, order))
               .record for order, chunk in enumerate(chunks)]
    assert len(records) == len(chunks) >= 2
    predicates = [(fact.predicate, fact.value)
                  for record in records for fact in record.facts]
    assert ("name", "Sam") in predicates
    assert ("lives_in", "Oslo") in predicates
    assert records[0].source.message_id == "m7#0"
    assert records[-1].source.message_id == f"m7#{len(chunks) - 1}"


def test_identity_established_in_one_segment_resolves_in_the_next():
    text = ("My colleague is called Ivo. " + _filler(8200)
            + "He speaks Dutch.")
    chunks = segment_message(text, "m8")
    assert len(chunks) == 2
    resolver = Resolver()
    records = [resolver.resolve(_message(chunk.text, chunk.identity, order))
               .record for order, chunk in enumerate(chunks)]
    fact, = records[1].facts
    assert fact.predicate == "speaks"
    assert fact.subject.kind == "named"
    assert fact.subject.label == "Ivo"
    assert fact.subject.status == "resolved"
    assert fact.subject.resolved_from == "pronoun_antecedent"


def test_a_single_fact_is_not_duplicated_by_segmentation():
    text = ("My brother is called Sam. " + _filler(8200)
            + "The weather is mild today.")
    chunks = segment_message(text, "m9")
    resolver = Resolver()
    records = [resolver.resolve(_message(chunk.text, chunk.identity, order))
               .record for order, chunk in enumerate(chunks)]
    names = [(fact.predicate, fact.value) for record in records
             for fact in record.facts]
    assert names.count(("name", "Sam")) == 1


def test_a_repeated_fact_keeps_one_record_per_occurrence():
    text = ("My brother is called Sam. " + _filler(8200)
            + "My brother is called Sam.")
    chunks = segment_message(text, "m10")
    resolver = Resolver()
    records = [resolver.resolve(_message(chunk.text, chunk.identity, order))
               .record for order, chunk in enumerate(chunks)]
    names = [(fact.predicate, fact.value) for record in records
             for fact in record.facts]
    assert names.count(("name", "Sam")) == 2


def test_records_stay_inside_their_own_chunk():
    text = ("My brother is called Sam. " + _filler(9000)
            + "My sister lives in Oslo.")
    chunks = segment_message(text, "m11")
    resolver = Resolver()
    for order, chunk in enumerate(chunks):
        record = resolver.resolve(
            _message(chunk.text, chunk.identity, order)).record
        for fact in record.facts:
            for evidence in fact.evidence:
                assert 0 <= evidence.start < evidence.end \
                    <= len(chunk.text)
        assert record.original_text == chunk.text
