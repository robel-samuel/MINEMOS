"""Stage 2 tests: candidate facts, polarity, modality and evidence.

Interpretation may only produce candidates.  It must not decide truth, must
not write memory, and must abstain on hypothetical or interrogative clauses
instead of asserting them.
"""

from __future__ import annotations

import pytest

from minemos.resolution import (
    Resolver,
    SourceMessage,
    detect_language,
    interpret,
    normalize_text,
)


def _record(text: str, speaker: str = "owner"):
    return Resolver().resolve(
        SourceMessage("conv", "m0", text, speaker, 1.0, 0)).record


def _facts(text: str, speaker: str = "owner"):
    return {(fact.predicate, fact.value, fact.polarity)
            for fact in _record(text, speaker).facts}


def test_first_person_state():
    assert _facts("I live in Berlin.") == {("lives_in", "Berlin", "positive")}
    assert _facts("My name is Ada Lovelace.") == {
        ("name", "Ada Lovelace", "positive")}
    assert _facts("I am 42 years old.") == {("age", "42", "positive")}


def test_first_person_predicates_are_covered():
    assert _facts("I drink coffee.") == {("drinks", "coffee", "positive")}
    assert _facts("I'm allergic to peanuts.") == {
        ("allergic_to", "peanuts", "positive")}
    assert _facts("I'm fluent in French.") == {
        ("speaks", "French", "positive")}
    assert _facts("I commute by tram.") == {
        ("commutes_by", "tram", "positive")}


def test_negation_is_polarity_not_a_different_predicate():
    record = _record("I don't drink coffee.")
    assert [(fact.predicate, fact.value, fact.polarity)
            for fact in record.facts] == [("drinks", "coffee", "negative")]
    assert any(item.startswith("negated_clause:")
               for item in record.abstentions)


def test_relation_facts_belong_to_the_relative():
    fact, = _record("My brother lives in Oslo.").facts
    assert fact.subject.kind == "derived"
    assert fact.subject.relation == "brother"
    assert fact.subject.label == "user's brother"
    assert fact.value == "Oslo"


def test_named_subject_keeps_its_own_identity():
    fact, = _record("Nadia Ashford drinks lemonade.").facts
    assert fact.subject.kind == "named"
    assert fact.subject.label == "Nadia Ashford"
    assert fact.predicate == "drinks"


def test_two_clauses_yield_two_facts_with_the_same_subject():
    record = _record("Vero works at Acme and lives in Oslo.")
    assert [(fact.predicate, fact.value) for fact in record.facts] == [
        ("works_at", "Acme"), ("lives_in", "Oslo")]
    assert {fact.subject.label for fact in record.facts} == {"Vero"}


def test_elided_clause_inherits_the_previous_subject():
    record = _record("My sister is called Ada. She works at Acme.")
    works, = [fact for fact in record.facts if fact.predicate == "works_at"]
    assert works.subject.label == "Ada"


def test_hypothetical_clause_is_abstained():
    record = _record("I might move to Paris.")
    assert record.facts == ()
    assert any(item.startswith("hypothetical_clause:")
               for item in record.abstentions)


def test_question_is_not_asserted():
    record = _record("Do I live in Berlin?")
    assert record.facts == ()


# ── negation: third-party denial, possession denial and clause force ──────── #

@pytest.mark.parametrize("text, subject_label", [
    ("My brother doesn't live in Oslo.", "user's brother"),
    ("My brother does not live in Oslo.", "user's brother"),
    ("Zephyr Kingsley doesn't live in Oslo.", "Zephyr Kingsley"),
    ("Zephyr Kingsley does not live in Oslo.", "Zephyr Kingsley"),
    ("My sister doesn't work at Acme.", "user's sister"),
    ("Zephyr Kingsley doesn't work at Acme.", "Zephyr Kingsley"),
])
def test_negated_third_party_clause_keeps_its_subject_and_denial(
        text, subject_label):
    record = _record(text)
    fact, = record.facts
    assert fact.polarity == "negative"
    assert fact.subject.label == subject_label
    assert fact.subject.status == "resolved"
    assert any(item.startswith("negated_clause:")
               for item in record.abstentions)


@pytest.mark.parametrize("text", [
    "I don't have a car.",
    "I have no car.",
    "I don't own a car.",
    "I haven't got a car.",
    "I have not a car.",
    "I haven't got any car.",
    "I do not own any car.",
])
def test_every_negated_possession_form_records_a_denial(text):
    fact, = _record(text).facts
    assert (fact.predicate, fact.value, fact.polarity) == (
        "has", "car", "negative")
    assert (fact.subject.kind, fact.subject.status) == (
        "speaker", "resolved")


def test_negated_study_at_keeps_the_place_out_of_the_verb():
    # Denegation leaves the base form "study at"; it must parse like
    # "studies at" instead of falling through to bare "study" + "at MIT".
    for text, label in (
            ("Benedict Yarrow does not study at MIT.", "Benedict Yarrow"),
            ("My sister doesn't study at MIT.", "user's sister")):
        fact, = _record(text).facts
        assert (fact.predicate, fact.value, fact.polarity) == (
            "studies_at", "MIT", "negative")
        assert fact.subject.label == label


def test_negated_possession_of_a_relation_stays_with_the_relative():
    for text, label in (("My sister doesn't have a car.", "user's sister"),
                        ("My brother doesn't have a car.", "user's brother")):
        fact, = _record(text).facts
        assert (fact.predicate, fact.value, fact.polarity) == (
            "has", "car", "negative")
        assert (fact.subject.kind, fact.subject.label) == ("derived", label)


@pytest.mark.parametrize("text", [
    "I don't have a car.",
    "I have no car.",
    "I haven't got a car.",
    "I have not a car.",
    "I haven't got any car.",
    "I don't own any car.",
    "My sister doesn't have a car.",
    "My brother doesn't live in Oslo.",
    "Zephyr Kingsley does not speak French.",
])
def test_negation_never_leaks_into_the_value_or_the_offsets(text):
    record = _record(text)
    negators = {"not", "no", "any"}
    for fact in record.facts:
        assert not negators & set(fact.value.casefold().split())
        start, end = fact.value_span
        assert record.normalized.text[start:end] == fact.value
        evidence, = fact.evidence
        assert text[evidence.start:evidence.end] == evidence.text


def test_negated_fact_evidence_addresses_the_original_word():
    text = "I don't have a car."
    fact, = _record(text).facts
    evidence, = fact.evidence
    assert evidence.text == "car"
    assert text[evidence.start:evidence.end] == "car"
    start, end = fact.value_span
    assert _record(text).normalized.text[start:end] == "car"


@pytest.mark.parametrize("text", [
    "If I had a car, I would drive.",
    "If my brother lived in Oslo, he would be happy.",
    "Suppose I don't have a car.",
    "If Zephyr Kingsley doesn't live in Oslo, that is fine.",
    "I might not have a car.",
    "Maybe I have a car.",
    "Perhaps my brother lives in Oslo.",
])
def test_hypothetical_or_modal_clause_never_asserts(text):
    assert _record(text).facts == ()


@pytest.mark.parametrize("text", [
    "Do I have a car?",
    "Do I not have a car?",
    "Does my brother live in Oslo?",
    "Doesn't my brother live in Oslo?",
    "Is Zephyr Kingsley allergic to peanuts?",
])
def test_interrogative_clause_is_not_asserted(text):
    assert _record(text).facts == ()


def test_uncertain_hedged_claim_is_not_asserted():
    for text in ("I think I don't have a car.",
                 "I am not sure I have a car.",
                 "I'm not sure I have a car."):
        assert _record(text).facts == ()


def test_echo_question_keeps_the_denial_and_never_inverts_it():
    # A declarative with a question mark is split as a sentence, so the mark
    # does not suppress the clause: it stays negative.  The limitation is
    # that it is not classified as a question, never that it flips polarity.
    fact, = _record("I don't have a car?").facts
    assert fact.polarity == "negative"


def test_correction_prefix_keeps_the_denial():
    for text, expected, label in (
            ("Actually, I don't have a car.", ("has", "car"), "owner"),
            ("Sorry, my brother doesn't live in Oslo.",
             ("lives_in", "Oslo"), "user's brother")):
        fact, = _record(text).facts
        assert (fact.predicate, fact.value, fact.polarity) == (
            expected[0], expected[1], "negative")
        assert fact.subject.label == label


def test_pronoun_head_is_not_turned_into_a_name():
    for text in ("He didn't live in Oslo.", "She does not have a car.",
                 "They don't work at Acme."):
        record = _record(text)
        assert not [fact for fact in record.facts
                    if fact.subject.kind == "named"]
        assert all(fact.subject.label.casefold() not in {
            "he did not", "she does not", "they do not"} for fact in record.facts)


def test_pronoun_denial_keeps_an_unresolved_subject():
    fact, = _record("They don't work at Acme.").facts
    assert (fact.predicate, fact.value, fact.polarity) == (
        "works_at", "Acme", "negative")
    assert fact.subject.status != "resolved"


def test_negation_scope_stops_at_the_contrast_tail():
    fact, = _record("I don't have a car, but my brother does.").facts
    assert (fact.predicate, fact.value, fact.polarity) == (
        "has", "car", "negative")


def test_negated_preference_is_recorded_as_a_dislike():
    # "I don't like tea" is not a denial of a "likes" fact; the lexicon has a
    # dedicated dislikes predicate and the fact stays positive.
    assert _facts("I don't like tea.") == {("dislikes", "tea", "positive")}


def test_adverb_between_subject_and_possession_verb_abstains():
    # The affirmative form is also unmatched, so the denial is neither
    # asserted nor inverted; the gap is symmetric and not negation-specific.
    assert _record("I definitely don't have a car.").facts == ()
    assert _record("I definitely have a car.").facts == ()


def test_alternative_attribute_spellings_share_one_predicate():
    assert _facts("My favourite colour is teal.") == {
        ("favorite_color", "teal", "positive")}
    assert _facts("My favorite color is teal.") == {
        ("favorite_color", "teal", "positive")}


def test_multilingual_forms_are_understood():
    assert _facts("Me llamo Ana.") == {("name", "Ana", "positive")}
    assert _facts("Vivo en Madrid.") == {("lives_in", "Madrid", "positive")}
    assert _facts("Je m'appelle Marie.") == {("name", "Marie", "positive")}
    assert _facts("J'habite à Paris.") == {
        ("lives_in", "Paris", "positive")}


def test_entity_types_and_value_spans():
    fact, = _record("My brother lives in Oslo.").facts
    assert [entity.type for entity in fact.entities] == ["place"]
    start, end = fact.value_span
    record = _record("My brother lives in Oslo.")
    assert record.normalized.text[start:end] == "Oslo"


def test_evidence_points_back_at_the_original_message():
    text = "My   brother lives in Oslo."
    record = _record(text)
    fact, = record.facts
    for evidence in fact.evidence:
        assert 0 <= evidence.start < evidence.end <= len(text)
        assert text[evidence.start:evidence.end]
    assert any(text[item.start:item.end] == "Oslo"
               for item in fact.evidence)


def test_interpreter_is_deterministic_and_pure():
    text = "My brother lives in Oslo."
    source = SourceMessage("conv", "m0", text, "owner", 1.0, 0)
    normalized = normalize_text(text)
    first = interpret(source, normalized, detect_language(text))
    second = interpret(source, normalized, detect_language(text))
    assert first == second
    assert first.facts
    for fact in first.facts:
        assert fact.modality == "asserted"
        assert 0.0 <= fact.confidence <= 1.0

def test_relation_subject_uses_the_third_party_verb_table():
    # A relation subject must not be limited to a handful of verbs: the same
    # clause shape is read the same way for "my mother" and for a named person.
    record = _record("My mother studies botany.")
    fact, = record.facts
    assert (fact.predicate, fact.value) == ("studies", "botany")
    assert fact.subject.kind == "derived"
    assert fact.subject.relation == "mother"
    start, end = fact.value_span
    assert record.normalized.text[start:end] == "botany"


def test_relation_subject_covers_state_and_habit_predicates():
    cases = {
        "My sister was born in Lima.": ("born_in", "Lima"),
        "My neighbour commutes by bicycle.": ("commutes_by", "bicycle"),
        "My brother is allergic to peanuts.": ("allergic_to", "peanuts"),
        "My grandmother speaks Polish.": ("speaks", "Polish"),
    }
    for text, expected in cases.items():
        fact, = _record(text).facts
        assert (fact.predicate, fact.value) == expected
        assert fact.subject.kind == "derived"


def test_capitalised_clause_opening_is_not_a_person():
    # These shapes were observed in the frozen export: a clause happens to
    # begin with a word that reads like a name and is followed by a
    # third-person verb.  The fact survives, the invented subject does not.
    texts = {
        "Dose this works at the new place.": "works_at",
        "See your Google plays in the background.": "plays",
        "Let me reads it.": "reads",
        "Interleaved ABBA studies the data.": "studies",
        "Please create detailed studies of it.": "studies",
        "about plays a role here.": "plays",
        "result rendering never reads the file.": "reads",
    }
    for text, predicate in texts.items():
        fact, = _record(text).facts
        assert fact.predicate == predicate
        assert fact.subject.kind == "named"
        assert fact.subject.status == "ambiguous"
        assert fact.subject.ambiguity == "name_not_proper_noun"


def test_a_name_keeps_its_subject_in_title_and_lower_case():
    for text in ("Clara Whitlock lives in Almaty.",
                 "clara whitlock lives in helsinki."):
        fact, = _record(text).facts
        assert fact.subject.status == "resolved"
        assert fact.subject.label.casefold().startswith("clara")


def test_fluent_in_carries_the_language_not_the_preposition():
    fact, = _record("Ignatius Ashford is fluent in Vietnamese.").facts
    assert (fact.predicate, fact.value) == ("speaks", "Vietnamese")
    assert fact.subject.status == "resolved"


def test_an_adverb_between_name_and_verb_is_not_part_of_the_name():
    fact, = _record("Benedict Yarrow really likes the author Ismail Kadare.").facts
    assert fact.subject.label == "Benedict Yarrow"
    assert fact.subject.status == "resolved"
    assert fact.predicate == "likes"


def test_bare_copula_number_is_an_age_for_a_person():
    fact, = _record("Zephyr Harlow is 76.").facts
    assert (fact.predicate, fact.value) == ("age", "76")
    # A number with a unit is a different fact and must not become an age.
    assert not [item for item in _record("Vera is 5 miles away.").facts
                if item.predicate == "age"]


def test_knowing_a_language_is_speaking_it():
    fact, = _record("Darius Kingsley knows Turkish.").facts
    assert (fact.predicate, fact.value) == ("speaks", "Turkish")
    # The same verb about a person is not a language claim.
    other, = _record("Darius Kingsley knows my brother.").facts
    assert other.predicate == "knows"


def test_light_verb_and_filler_variants_keep_the_same_predicate():
    cases = {
        "Elowen Northcott works over at Lantern Bay Hotels.":
            ("works_at", "Lantern Bay Hotels"),
        "Lyra Ravenscroft does work as site reliability engineer.":
            ("role", "site reliability engineer"),
        "Elowen Whitlock is into reading art criticism.":
            ("reads", "art criticism"),
    }
    for text, expected in cases.items():
        fact, = _record(text).facts
        assert (fact.predicate, fact.value) == expected
        assert fact.subject.status == "resolved"


def test_interest_without_a_reading_verb_is_not_a_read():
    # "is really into vermilion" states an interest, not a colour fact, and
    # "has a problem with latex" is not an allergy; both stay unasserted.
    for text in ("Ulric Ingram is really into vermilion.",
                 "Elowen Merrivale has a problem with latex."):
        assert [fact.predicate for fact in _record(text).facts] == (
            [] if "vermilion" in text else ["has"])


@pytest.mark.parametrize("text, relation", [
    ("myy brther lives in Oslo.", "brother"),
    ("my brther lives in Oslo.", "brother"),
    ("Myy Brther lives in Oslo.", "brother"),
    ("myy sister lives in Oslo.", "sister"),
    ("my freind works at Acme.", "friend"),
    ("our motther studies at MIT.", "mother"),
])
def test_misspelled_possessive_relation_is_a_derived_subject(text, relation):
    fact, = _record(text).facts
    assert fact.subject.kind == "derived"
    assert fact.subject.relation == relation
    assert fact.subject.label == f"user's {relation}"
    assert fact.subject.status == "resolved"
    assert fact.subject.resolved_from == "possessive_near_match"
    assert fact.subject.confidence <= 0.75
    assert fact.confidence <= 0.75


def test_misspelled_relation_never_becomes_a_named_entity():
    # The pre-V3 behaviour read "myy brther" as a person because both tokens
    # were out of vocabulary; spelling evidence must win over position.
    for text in ("myy brther lives in Oslo.",
                 "Myy Brther lives in Oslo.",
                 "my brther lives in Oslo."):
        record = _record(text)
        assert record.facts
        assert all(fact.subject.kind != "named" for fact in record.facts)


def test_misspelled_relation_phrase_beyond_the_edit_bound_abstains():
    record = _record("myy broth lives in Oslo.")
    fact, = record.facts
    assert fact.subject.status != "resolved"
    assert any(item.startswith("unresolved_subject:")
               for item in record.abstentions)


def test_names_near_a_possessive_are_not_read_as_relations():
    # "May" and "Myra" are ordinary vocabulary or names, not typos of "my".
    for text in ("May brother lives in Oslo.",
                 "Myra brother lives in Oslo."):
        record = _record(text)
        fact, = record.facts
        assert fact.subject.kind == "named"
        assert fact.subject.status != "resolved"
        assert any(item.startswith("unresolved_subject:")
                   for item in record.abstentions)


def test_misspelled_relation_keeps_speaker_and_subject_separate():
    record = _record("myy brther lives in Oslo.", speaker="owner")
    fact, = record.facts
    assert fact.subject.owner == "owner"
    assert fact.subject.kind == "derived"
    assert fact.subject.label != "owner"
    assert all(fact.subject.kind != "speaker" for fact in record.facts)


def test_misspelled_relation_name_is_a_naming_fact():
    facts = _facts("myy brther's name is Sam.")
    assert facts == {("name", "Sam", "positive")}


def test_leading_discourse_marker_does_not_hide_the_speaker():
    # Real chat opens clauses with fillers.  Without peeling them the pronoun
    # is read as a name ("But i") or the clause is not recognised at all.
    assert _facts("But i live in Oslo.") == {
        ("lives_in", "Oslo", "positive")}
    assert _facts("And I work as a broker.") == {
        ("works_as", "broker", "positive")}
    assert _facts("Also I work as a broker.") == {
        ("works_as", "broker", "positive")}
    assert _facts("no I was 54.") == {("age", "54", "positive")}
    fact, = _record("So I live in Oslo.").facts
    assert fact.subject.kind == "speaker"


def test_discourse_marker_before_a_relation_keeps_the_relative():
    fact, = _record("Ok so my brother lives in Dubai.").facts
    assert fact.subject.kind == "derived"
    assert fact.subject.relation == "brother"
    assert fact.subject.status == "resolved"


def test_discourse_marker_does_not_touch_a_third_party_clause():
    fact, = _record("Mara lives in Oslo.").facts
    assert fact.subject.kind == "named"
    assert fact.subject.label == "Mara"


def test_first_person_head_never_becomes_a_third_party_subject():
    # "I don't own ..." is the speaker's denial.  Closing the negated
    # possession gap must not read it as a third-party candidate whose subject
    # is the literal string "I do not".
    for text in ("no I don't own the website.",
                 "I don't own the website."):
        fact, = _record(text).facts
        assert (fact.predicate, fact.value) == ("has", "website")
        assert fact.polarity == "negative"
        assert (fact.subject.kind, fact.subject.status) == (
            "speaker", "resolved")
        assert fact.subject.label == "owner"


def test_past_age_is_a_speaker_fact():
    fact, = _record("no i was 47 then i start drivng").facts
    assert (fact.predicate, fact.value) == ("age", "47")
    assert fact.subject.kind == "speaker"


def test_present_perfect_is_not_a_possession():
    # "I have been asked ..." is an experience, not an owned object.
    assert _facts("I have been asked this question before.") == set()


def test_prospective_plan_is_not_a_current_role():
    # A sentence that starts with a prospective marker states an intention, not
    # a job, even when it names a role and a workplace.
    assert _facts(
        "I'm about to apply for a job as a night baker at a small bakery."
    ) == set()


def test_demonstrative_is_not_a_possession():
    # A bare demonstrative points at the conversation, not an object.
    assert _facts("I have this, my colleague, for you.") == set()
    fact, = _record("I have a silver flute.").facts
    assert (fact.predicate, fact.value) == ("has", "silver flute")


def test_contrastive_tail_does_not_flip_the_speaker_fact():
    fact, = _record("I have a website, not her.").facts
    assert (fact.predicate, fact.value, fact.polarity) == (
        "has", "website", "positive")
    assert fact.subject.kind == "speaker"


def test_or_clause_is_not_part_of_the_previous_value():
    facts = _facts("I work as a broker, or I flip houses.")
    assert ("works_as", "broker", "positive") in facts
