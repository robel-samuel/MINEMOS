"""Multilingual Stage 2/3 tests: Amharic, Spanish, French and code-switching.

Every rule is anchored on a construction whose function is explicit, so these
tests also pin the abstentions: a question, a hedge or a bare pronoun must
produce no asserted fact rather than a guess.
"""

from __future__ import annotations

from minemos.resolution import (
    Resolver,
    SourceMessage,
    detect_language,
    normalize_text,
)


def _run(turns, conversation: str = "conv"):
    resolver = Resolver()
    records = []
    for order, (message_id, text, speaker) in enumerate(turns):
        records.append(resolver.resolve(SourceMessage(
            conversation, message_id, text, speaker,
            float(order), order)).record)
    return resolver, records


def _record(text: str, speaker: str = "owner"):
    return _run([("m0", text, speaker)])[1][0]


def _facts(record):
    return {(fact.predicate, fact.value, fact.subject.kind,
             fact.subject.relation, fact.subject.status)
            for fact in record.facts}


# ── Amharic ──────────────────────────────────────────────────────────────── #

def test_amharic_first_person_state():
    _, records = _run([
        ("m0", "ስሜ ዳዊት ነው።", "a"),
        ("m1", "በአዲስ አበባ እኖራለሁ።", "a"),
        ("m2", "በአዲስ አበባ ውስጥ እሰራለሁ።", "a"),
        ("m3", "ዕድሜዬ 30 ዓመት ነው።", "a"),
    ])
    assert detect_language("ስሜ ዳዊት ነው።").primary == "am"
    assert _facts(records[0]) == {("name", "ዳዊት", "speaker", None,
                                   "resolved")}
    assert _facts(records[1]) == {("lives_in", "አዲስ አበባ", "speaker",
                                   None, "resolved")}
    assert _facts(records[2]) == {("works_in", "አዲስ አበባ", "speaker",
                                   None, "resolved")}
    assert _facts(records[3]) == {("age", "30", "speaker", None,
                                   "resolved")}


def test_amharic_relation_statement_never_becomes_first_person():
    _, records = _run([
        ("m0", "ወንድሜ ሳም ይባላል።", "a"),
        ("m1", "ወንድሜ በአዲስ አበባ ይኖራል።", "a"),
    ])
    fact, = records[0].facts
    assert fact.subject.kind == "derived"
    assert fact.subject.relation == "brother"
    assert fact.subject.owner == "a"
    assert fact.predicate == "name"
    assert fact.value == "ሳም"
    assert records[1].facts[0].predicate == "lives_in"
    # The brother was named in the previous message, so this mention links to
    # that recorded person instead of staying a bare relation.
    assert records[1].facts[0].subject.label == "ሳም"


def test_amharic_informal_copula_and_missing_punctuation_still_read():
    for text in ("ስሜ ማሪያም ነዉ።", "ስሜ ማሪያም", "ስሜ  ማሪያም   ነው።"):
        fact, = _record(text).facts
        assert (fact.predicate, fact.value) == ("name", "ማሪያም")


def test_amharic_question_and_hedge_assert_nothing():
    for text in ("ስሜ ማን ነው?", "ስሜ ማን ነው",
                 "ምናልባት ወንድሜ ሳም ይባላል።"):
        assert _record(text).facts == ()


def test_amharic_pronoun_without_antecedent_abstains():
    record = _record("እሷ በሐዋሳ ትኖራለች።")
    fact, = record.facts
    assert fact.subject.status == "unknown"
    assert any(item.startswith("unresolved_subject:")
               for item in record.abstentions)
    assert any(item.startswith("pronoun_needs_antecedent:")
               for item in record.ambiguities)


def test_amharic_pronoun_resolves_to_the_single_matching_relative():
    _, records = _run([
        ("m0", "እህቴ ሳራ ትባላለች።", "a"),
        ("m1", "እሷ በሐዋሳ ትኖራለች።", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.kind == "named"
    assert fact.subject.label == "ሳራ"
    assert fact.subject.status == "resolved"


# ── Spanish and French ───────────────────────────────────────────────────── #

def test_spanish_pronoun_needs_an_antecedent():
    record = _record("Ella vive en Madrid.")
    fact, = record.facts
    assert fact.subject.status == "unknown"
    assert any(item.startswith("unresolved_subject:")
               for item in record.abstentions)


def test_spanish_pronoun_resolves_after_the_relation_is_named():
    _, records = _run([
        ("m0", "Mi hermana se llama Ana.", "a"),
        ("m1", "Ella vive en Madrid.", "a"),
    ])
    name, = records[0].facts
    assert name.predicate == "name"
    assert name.subject.relation == "sister"
    fact, = records[1].facts
    assert fact.subject.kind == "named"
    assert fact.subject.label == "Ana"
    assert fact.predicate == "lives_in"


def test_french_relation_and_pronoun():
    _, records = _run([
        ("m0", "Ma sœur s'appelle Claire.", "a"),
        ("m1", "Elle habite à Lyon.", "a"),
    ])
    name, = records[0].facts
    assert name.value == "Claire"
    assert name.subject.relation == "sister"
    fact, = records[1].facts
    assert fact.value == "Lyon"
    assert fact.subject.label == "Claire"


def test_spanish_possessive_relation_can_head_a_predicate():
    fact, = _record("Mi madre vive en Lima.").facts
    assert (fact.predicate, fact.value) == ("lives_in", "Lima")
    assert (fact.subject.kind, fact.subject.relation) == ("derived", "mother")
    assert fact.subject.owner == "owner"


def test_french_possessive_relation_can_head_a_predicate():
    fact, = _record("Mon frère habite à Nantes.").facts
    assert (fact.predicate, fact.value) == ("lives_in", "Nantes")
    assert (fact.subject.kind, fact.subject.relation) == ("derived", "brother")


def test_french_relation_with_employer_stays_third_person():
    fact, = _record("Mon père travaille chez Renault.").facts
    assert (fact.predicate, fact.value) == ("works_at", "Renault")
    assert (fact.subject.kind, fact.subject.relation) == ("derived", "father")


def test_french_chez_names_the_employer_for_the_speaker():
    fact, = _record("je travaille chez Renault").facts
    assert (fact.predicate, fact.value, fact.subject.kind) == (
        "works_at", "Renault", "speaker")


def test_spanish_para_names_the_employer_for_the_speaker():
    fact, = _record("Trabajo para Acme.").facts
    assert (fact.predicate, fact.value, fact.subject.kind) == (
        "works_at", "Acme", "speaker")


def test_spanish_named_subject_keeps_its_identity():
    fact, = _record("Ana Lopez vive en Madrid.").facts
    assert fact.subject.kind == "named"
    assert fact.subject.label == "Ana Lopez"
    assert fact.predicate == "lives_in"
    assert fact.value == "Madrid"


# ── code-switching ───────────────────────────────────────────────────────── #

def test_english_frame_with_amharic_value():
    record = _record("I live in አዲስ አበባ.")
    fact, = record.facts
    assert (fact.predicate, fact.value) == ("lives_in", "አዲስ አበባ")
    assert fact.subject.kind == "speaker"
    assert detect_language("I live in አዲስ አበባ.").mixed is True


def test_amharic_relation_with_latin_name():
    fact, = _record("My brother is called ሳም.").facts
    assert fact.predicate == "name"
    assert fact.value == "ሳም"
    assert fact.subject.relation == "brother"


def test_english_and_spanish_clauses_in_one_message():
    record = _record("I live in Madrid pero mi hermana se llama Ana.")
    facts = {(fact.predicate, fact.value, fact.subject.kind)
             for fact in record.facts}
    assert facts == {("lives_in", "Madrid", "speaker"),
                     ("name", "Ana", "derived")}


def test_spanish_name_then_english_clause():
    record = _record("Me llamo Ana and I work at Acme.")
    facts = {(fact.predicate, fact.value, fact.subject.kind)
             for fact in record.facts}
    assert facts == {("name", "Ana", "speaker"),
                     ("works_at", "Acme", "speaker")}


# ── isolation and normalization ──────────────────────────────────────────── #

def test_same_name_in_two_conversations_is_not_shared():
    resolver = Resolver()
    resolver.resolve(SourceMessage(
        "one", "m0", "Mi hermana se llama Ana.", "a", 0.0, 0))
    resolver.resolve(SourceMessage(
        "two", "m0", "Ella vive en Madrid.", "a", 0.0, 0))
    assert "ana" in resolver.identities.session("one").people
    assert "ana" not in resolver.identities.session("two").people


def test_ethiopic_punctuation_is_preserved_and_terminates_the_sentence():
    normalized = normalize_text("ስሜ ዳዊት ነው።")
    assert normalized.text == "ስሜ ዳዊት ነው።"
    assert normalized.sentence_terminated is True
    assert normalized.edits == ()
