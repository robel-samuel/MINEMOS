"""V4 exact value extraction: the six V3 over-capture shapes and their guards.

Every case is a synthetic analogue of a real failure category recorded in the
V3 report.  The assertions are deliberately strict about the *value*: an
over-captured value is a wrong fact, so the expected value must match exactly
and the evidence span must address the same text in the original message.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from minemos.resolution import (
    Resolver,
    SourceMessage,
    normalize_text,
)


def _record(text: str, conversation: str = "conv"):
    return Resolver().resolve(
        SourceMessage(conversation, "m0", text, "owner", 1.0, 0)).record


def _pairs(text: str):
    return [(fact.predicate, fact.value, fact.polarity)
            for fact in _record(text).facts]


def _one(text: str):
    fact, = _record(text).facts
    return fact


# ── the six V3 over-capture cases, by shape ───────────────────────────────── #

@pytest.mark.parametrize("text, predicate, value", [
    # R06/R07 — running speech continues with a question.
    ("i live in ethiopia are they offering a visa for a job",
     "lives_in", "ethiopia"),
    ("i speak amharic what about you", "speaks", "amharic"),
    ("i work at acme do you know them", "works_at", "acme"),
    # R05 — a trailing interrogative after a counted value.
    ("i have 3k followers on youtube how much can i make per month",
     "has", "3 k followers on youtube"),
])
def test_trailing_question_is_not_part_of_the_value(text, predicate, value):
    assert _pairs(text) == [(predicate, value, "positive")]


@pytest.mark.parametrize("text, predicate, value", [
    # R09 — a new first-person clause without punctuation.
    ("my salary is crypto the i use p 2 p to change it to birr on binance",
     "salary", "crypto"),
    # R10 — the speaker continues with a need.
    ("i have 4 crates of mangoes i need 20 crates for the market i have 7 left",
     "has", "4 crates of mangoes"),
    ("i work as a nurse i want to move to dubai next year",
     "works_as", "nurse"),
    ("i have two kids i need a bigger flat", "has", "two kids"),
])
def test_trailing_new_clause_is_not_part_of_the_value(text, predicate, value):
    assert _pairs(text) == [(predicate, value, "positive")]


@pytest.mark.parametrize("text, predicate, value", [
    # R12 — a language adverbial after the comma.
    ("i have migrated to superbase, in english",
     "has", "migrated to superbase"),
    ("i work at acme, in english", "works_at", "acme"),
    ("i study medicine, in english", "studies", "medicine"),
    ("i live in addis ababa, in amharic", "lives_in", "addis ababa"),
])
def test_trailing_language_adverbial_is_not_part_of_the_value(
        text, predicate, value):
    assert _pairs(text) == [(predicate, value, "positive")]


def test_language_word_without_the_adverbial_comma_is_a_value():
    assert _pairs("my favorite language is english") == [
        ("favorite_language", "english", "positive")]


# ── guards: a relative clause belongs to the value ────────────────────────── #

@pytest.mark.parametrize("text, value", [
    ("my favorite book is the one i love most", "one i love most"),
    ("i have a tattoo that says i love you", "tattoo that says i love you"),
    ("i have a dog that lives with my parents",
     "dog that lives with my parents"),
    ("i have a car that my father gave me", "car that my father gave me"),
])
def test_relative_clause_keeps_its_tail(text, value):
    fact, = _record(text).facts
    assert fact.value == value
    assert fact.evidence[0].text == value


# ── one sentence, several facts ───────────────────────────────────────────── #

def test_coordinated_verb_without_a_subject_is_its_own_fact():
    assert _pairs("i study medicine and speak french") == [
        ("studies", "medicine", "positive"),
        ("speaks", "french", "positive"),
    ]


def test_coordinated_verb_keeps_the_established_relation_subject():
    facts = _record("my brother likes tea and plays tennis").facts
    assert [(fact.predicate, fact.value, fact.subject.kind)
            for fact in facts] == [
        ("likes", "tea", "derived"), ("plays", "tennis", "derived")]


def test_coordination_inside_a_noun_phrase_stays_one_value():
    assert _pairs("i have a dog and a cat") == [
        ("has", "dog and a cat", "positive")]


def test_role_and_employer_from_one_clause():
    assert _pairs("i work as a data analyst at acme") == [
        ("works_as", "data analyst", "positive"),
        ("works_at", "acme", "positive"),
    ]


# ── corrections, negation, possession, time ───────────────────────────────── #

def test_relation_name_correction_keeps_only_the_supplied_name():
    fact, = _record("my brother is called sam, i mean sam waters").facts
    assert (fact.predicate, fact.value) == ("name", "sam waters")
    assert fact.subject.kind == "derived"
    assert fact.subject.relation == "brother"


def test_speaker_name_correction_keeps_only_the_supplied_name():
    fact, = _record("my name is adam, i mean adam smith").facts
    assert (fact.predicate, fact.value) == ("name", "adam smith")
    assert fact.subject.kind == "speaker"


def test_negation_is_polarity_on_the_affirmative_predicate():
    assert _pairs("i don't drink coffee") == [
        ("drinks", "coffee", "negative")]


def test_possessive_subject_is_the_relative_not_the_speaker():
    facts = _record("my brother lives in oslo").facts
    fact, = facts
    assert fact.subject.kind == "derived"
    assert fact.subject.relation == "brother"
    assert fact.subject.label != "owner"


def test_temporal_modifier_is_recorded_not_stored_in_the_value():
    fact = _one("i live in berlin now")
    assert fact.value == "berlin"
    fact = _one("i moved to lisbon in 2023")
    assert fact.value == "lisbon"
    fact = _one("i used to live in rome")
    assert fact.value == "rome"
    assert any(item.kind == "past" for item in fact.temporal)


# ── value fidelity ────────────────────────────────────────────────────────── #

@pytest.mark.parametrize("text, value", [
    ("i work at the addis ababa city administration",
     "addis ababa city administration"),
    ("i work as a senior nurse", "senior nurse"),
    ("i work for the united nations", "united nations"),
    ("i have a silver flute", "silver flute"),
])
def test_organisation_place_and_role_keep_the_exact_wording(text, value):
    fact, = _record(text).facts
    assert fact.value == value


def test_evidence_span_addresses_the_value_in_the_source():
    text = "i live in berlin."
    fact = _one(text)
    evidence, = fact.evidence
    assert text[evidence.start:evidence.end] == evidence.text == "berlin"


def test_documented_place_alias_is_canonical_but_source_stays_in_evidence():
    fact = _one("i live in nyc")
    assert fact.value == "New York City"
    assert fact.evidence[0].text == "nyc"


def test_misspelled_place_is_never_repaired_into_a_guess():
    fact = _one("i live in addis abeba")
    assert fact.value == "addis abeba"
    assert fact.evidence[0].text == "addis abeba"


@pytest.mark.parametrize("name", ["rome", "bern", "bonn", "lome", "hayes",
                                 "acre", "porto", "oslo"])
def test_place_name_is_not_rewritten_by_the_typo_corrector(name):
    # "rome" is one adjacent swap from "role"; rewriting a city would put a
    # value in the fact that the speaker never wrote.
    assert normalize_text(f"i live in {name}").text == f"i live in {name}"
    assert _one(f"i live in {name}").value == name


def test_topic_typo_repair_still_works():
    assert normalize_text("i commutz to work").text == "i commute to work"


# ── assertion versus hypothesis, question, intention and opinion ──────────── #

@pytest.mark.parametrize("text", [
    "if i moved to berlin i would need a visa",
    "maybe i live in paris",
    "do i live in berlin?",
    "what is my name?",
    "i want to move to dubai",
    "i'm planning to study medicine",
    "i think paris is beautiful",
    "i have been asked this question before",
])
def test_non_asserted_clauses_emit_no_fact(text):
    assert _record(text).facts == ()


def test_stated_preference_is_an_asserted_fact_about_the_speaker():
    assert _pairs("i love coffee") == [("likes", "coffee", "positive")]


def test_contrast_tail_does_not_widen_the_value():
    assert _pairs("i have a website, not her") == [
        ("has", "website", "positive")]


# ── the frozen set itself ─────────────────────────────────────────────────── #

_SET = pathlib.Path(__file__).resolve().parents[1] / (
    "evaluation/resolution/v4_value_extraction/measure_value_extraction.py")


def _load_harness():
    spec = importlib.util.spec_from_file_location("v4_value_extraction", _SET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_v4_set_has_no_remaining_failures():
    # V4 declared two negation gaps (V4-028 negated third party, V4-029
    # negated possession).  V5 closes both, so the frozen set the resolver
    # was measured against now has no misses and no undeclared failures.
    harness = _load_harness()
    report = harness.evaluate(harness.load_cases())
    summary = report["summary"]
    assert summary["precision"] == 1.0
    assert summary["value_exactness"] == 1.0
    assert summary["evidence_exactness"] == 1.0
    assert summary["evidence_grounded"] == 1.0
    assert summary["no_fact_compliance"] == 1.0
    assert report["failures"] == []
