"""Stage 1 tests: noise normalization must edit noise and only noise.

The layer is allowed to repair spelling, expand contractions and collapse
repeats.  It is not allowed to alter a name, a number, a date, a URL, an
e-mail address, a hyphenated compound or any content word that happens to be
absent from the closed vocabulary.  Every edit also has to be recorded so the
normalized view stays inspectable and reversible.
"""

from __future__ import annotations

import pytest

from minemos.resolution import normalize_text


def _kinds(normalized):
    return [edit.kind for edit in normalized.edits]


def test_original_text_is_never_overwritten():
    text = "I don't gonna lie  about the  typo"
    normalized = normalize_text(text)
    assert normalized.original == text
    assert normalized.text != text
    assert normalized.to_dict()["reversible"] is True


def test_contraction_is_expanded_and_recorded():
    normalized = normalize_text("I don't drink coffee")
    assert normalized.text == "I do not drink coffee"
    edit = normalized.edits[0]
    assert (edit.kind, edit.original, edit.replacement) == (
        "contraction", "don't", "do not")
    assert edit.to_dict()["reverse"]["text"] == "don't"


def test_informal_form_is_expanded():
    normalized = normalize_text("I gonna call you")
    assert "going to" in normalized.text
    assert "informal" in _kinds(normalized)


def test_repeated_word_is_collapsed():
    normalized = normalize_text("the the cat sat")
    assert normalized.text == "the cat sat"
    assert "repeated_word" in _kinds(normalized)


def test_whitespace_is_normalized():
    normalized = normalize_text("  she   wrks   at Acme  ")
    assert normalized.text == "she works at Acme"
    assert "whitespace" in _kinds(normalized)


def test_typo_and_shorthand_are_repaired():
    assert normalize_text("she wrks at Acme").text == "she works at Acme"
    assert normalize_text("i spks french").text == "i speaks french"
    # A vowel-dropped shorthand is only expanded when the consonants fit
    # exactly one functional word.
    assert normalize_text("he rds books").text == "he reads books"


def test_ambiguous_typo_is_reported_not_guessed():
    normalized = normalize_text("this is onw")
    assert "onw" in normalized.text
    assert any(item.startswith("typo_ambiguous:")
               for item in normalized.ambiguities)


def test_three_letter_deletion_never_rewrites_a_name():
    # "sam" and "tom" are one deletion from "am" and "to".  At this length a
    # deleted letter is a third of the token, so the evidence is too thin.
    assert normalize_text("sam waters lives in Oslo").text == \
        "sam waters lives in Oslo"
    assert normalize_text("tom lives in Oslo").text == "tom lives in Oslo"
    # A transposition keeps every letter and is still repaired.
    assert normalize_text("this is adn").text == "this is and"


def test_lowercase_name_before_a_verb_is_not_rewritten():
    # "lopes" is one substitution from the vocabulary verb "loves".  The token
    # in front of a lexical predicate is a subject, so it stays as written.
    normalized = normalize_text("ana lopes lives in Madrid.")
    assert normalized.text == "ana lopes lives in Madrid."
    assert normalized.edits == ()
    # A swap inside the verb itself is still repaired.
    assert normalize_text("Sam lieks jazz.").text == "Sam likes jazz."


def test_name_introduction_protects_a_lowercase_name_at_the_end():
    for text in ("my name is ana lopes.",
                 "my brother is called sam waters.",
                 "my name's ana lopes."):
        assert normalize_text(text).text == text
    # The cue ends at the first known word, so repairs after the name stand.
    assert normalize_text("my name is ana lopes and i wrks here.").text == \
        "my name is ana lopes and i works here."


def test_copula_does_not_hide_an_attribute_typo():
    # The subject-slot guard applies to lexical predicates only: attribute
    # nouns are followed by "is" ("favorite colof is scarlet").
    assert normalize_text("Hollis Merrivale favorite colof is scarlet.").text \
        == "Hollis Merrivale favorite color is scarlet."


@pytest.mark.parametrize("text", [
    "My name is N. K. Jemisin.",
    "I live at 555-0134.",
    "I moved on 2024-05-06.",
    "she is 30-something now",
    "I read sci-fi and well-known essays",
    "cost $12.50 or 13% off",
    "mail a.b-c@x.org today",
    "see https://example.com/a-b?q=1",
    "the record player is broken",
    "she commutes by tram",
    "my favorite color is wheat",
    "Clara Zeller plays the lute",
    "the leading candidate won",
])
def test_values_and_names_are_preserved(text):
    normalized = normalize_text(text)
    assert normalized.text == text
    assert normalized.edits == ()


def test_capitalized_tokens_are_never_corrected():
    # Capitalized tokens are name-like; Stage 3 may link a variant, but the
    # normalizer must not rewrite the spelling.
    normalized = normalize_text("Theron Merrivale works at Driftwood Studios")
    assert "Studios" in normalized.text
    assert normalized.text == "Theron Merrivale works at Driftwood Studios"


def test_english_lexicons_are_skipped_for_other_languages():
    text = "Grüße aus München"
    normalized = normalize_text(text, language="de")
    assert normalized.text == text
    assert normalized.edits == ()


def test_compatibility_forms_are_recorded():
    # A fullwidth letter is a Unicode compatibility form; the change is applied
    # but has to show up as an edit so an operator can diff it.
    normalized = normalize_text("Ａcme hired her")
    assert normalized.text.startswith("Acme")
    assert "compatibility" in _kinds(normalized)


def test_token_offsets_point_back_at_the_source():
    text = "she  wrks at Acme"
    normalized = normalize_text(text)
    for token in normalized.tokens:
        assert 0 <= token.source_start <= token.source_end <= len(text)
        assert text[token.source_start:token.source_end].strip()
    corrected = [token for token in normalized.tokens if token.edited]
    assert corrected and corrected[0].text == "works"
    assert text[corrected[0].source_start:corrected[0].source_end] == "wrks"


def test_span_to_source_maps_normalized_spans_back():
    text = "she  wrks at Acme"
    normalized = normalize_text(text)
    start = normalized.text.index("works")
    source_start, source_end = normalized.span_to_source(start, start + 5)
    assert text[source_start:source_end] == "wrks"


def test_edit_offsets_are_inside_the_source():
    text = "i spks french and rds books"
    normalized = normalize_text(text)
    for edit in normalized.edits:
        assert 0 <= edit.source_start <= edit.source_end <= len(text)
        assert text[edit.source_start:edit.source_end] == edit.original


def test_empty_message_stays_empty():
    normalized = normalize_text("   ")
    assert normalized.text == ""
    assert normalized.tokens == ()
    assert normalized.edits == ()


@pytest.mark.parametrize("text,expected", [
    ("Tamsin Fairbanks teachfes astronomy.", "teaches"),
    ("Hollis Ravenscroft plpys marimba.", "plays"),
    ("Clara Pemberton is agyd 61.", "aged"),
    ("her rloe is analyst", "role"),
    ("the favoriqe fclm is Dune", "film"),
    ("she was boren in Osh", "born"),
    ("he volunteeas at the shelter", "volunteers"),
])
def test_misspelled_structural_word_is_recovered(text, expected):
    # A verb or attribute name is structural: getting it wrong breaks the
    # parse, and the closed list makes the single reading evidence rather
    # than a guess.
    normalized = normalize_text(text)
    assert expected in normalized.text.split()
    assert "typo" in _kinds(normalized)


@pytest.mark.parametrize("text", [
    "valerie quill lives in recife",
    "she commutes by tram",
    "I bought lemonade",
    "Vance Sutherland called",
    "Alaric Harroow teaches",
    "Orion Kingsley is aged 82",
    "Greta Ainsley is allergic to soy",
    "theron merrivale works at driftwood studios",
    "her commute takes an hour",
])
def test_structural_recovery_never_touches_values_or_names(text):
    # Places, content nouns and proper names are one edit from ordinary words
    # ("recife"/"reside", "culler"/"called"); recovering those would rewrite a
    # fact, so they stay exactly as written.
    assert normalize_text(text).text == text


@pytest.mark.parametrize("word", [
    "teachfes", "commutz", "workd", "spks", "livse",
])
def test_misspelled_token_is_not_mistaken_for_an_inflection(word):
    # Suffixes used to be trimmed by length without checking the ending, so
    # "teachfes" lost three characters for "ing"/"est" and looked like the
    # valid form "teach"; that hid the typo from repair.
    from minemos.resolution.lexicon import is_vocabulary_inflection
    assert is_vocabulary_inflection(word) is False


@pytest.mark.parametrize("word", [
    "commutes", "worked", "roads", "studies", "teaching", "speaks",
    "teaches", "residing", "lives", "resides",
])
def test_genuine_inflections_are_never_rewritten(word):
    assert normalize_text(word).text == word


@pytest.mark.parametrize("word", [
    "commutes", "worked", "roads", "studies", "teaching", "speaks", "teaches",
])
def test_inflection_guard_recognises_vocabulary_forms(word):
    from minemos.resolution.lexicon import is_vocabulary_inflection
    assert is_vocabulary_inflection(word) is True


def test_span_to_source_maps_a_late_word_across_many_tokens():
    # The mapping is looked up from a precomputed token layout, so a span near
    # the end of a long message must still point at its source characters.
    clauses = [f"She wrks at Acme office {index}." for index in range(80)]
    text = " ".join(clauses)
    normalized = normalize_text(text)
    start = normalized.text.rindex("works")
    source_start, source_end = normalized.span_to_source(start, start + 5)
    assert text[source_start:source_end] == "wrks"
    assert source_start > len(text) // 2


def test_span_to_source_stops_at_punctuation_between_tokens():
    text = "He lives in Osh, then works at Acme."
    normalized = normalize_text(text)
    for word in ("Osh", "Acme"):
        start = normalized.text.index(word)
        source_start, source_end = normalized.span_to_source(
            start, start + len(word))
        assert text[source_start:source_end] == word
