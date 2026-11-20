"""Stage 1 language identification: script evidence, stopword evidence, abstention.

The detector is deterministic and local.  It reports ``"und"`` rather than
guessing when a message carries too little evidence, and it flags
code-switching only when a second language has real evidence of its own.
"""

from __future__ import annotations

import pytest

from minemos.resolution import detect_language


@pytest.mark.parametrize("text,expected", [
    ("I live in Berlin and work at Acme.", "en"),
    ("Hola, ¿cómo estás? Vivo en Madrid.", "es"),
    ("Bonjour, je m'appelle Marie.", "fr"),
    ("Ich heiße Anna und wohne in Berlin.", "de"),
    ("Привет, я живу в Москве.", "ru"),
    ("東京は大きい", "zh"),
])
def test_primary_language(text, expected):
    info = detect_language(text)
    assert info.primary == expected
    assert 0.0 < info.confidence <= 0.98
    assert info.mixed is False


def test_cyrillic_languages_are_told_apart_by_their_letters():
    assert detect_language("Я живу в Києві").primary == "uk"
    assert detect_language("Привет, я живу в Москве.").primary == "ru"


def test_little_evidence_abstains():
    info = detect_language("ok")
    assert info.primary == "und"
    assert info.confidence < 0.6


def test_short_ambiguous_word_is_not_guessed():
    # "sci-fi" is a borrowed compound with no stopwords; the detector says
    # und rather than assigning the message to whichever list sorts first.
    assert detect_language("sci-fi").primary == "und"


def test_urls_and_addresses_do_not_vote():
    # Masked text carries no words, so the detector reports und instead of
    # reading a language out of a bare URL.
    info = detect_language("https://example.com/path?q=1")
    assert info.primary == "und"
    assert info.spans == ()
    assert detect_language("mail me at a.b@example.com").primary == "en"


def test_code_switching_is_flagged():
    info = detect_language("I am here. Je suis ici.")
    assert info.primary == "en"
    assert info.mixed is True
    assert "fr" in info.alternates


def test_single_borrowed_word_is_not_code_switching():
    info = detect_language("I live in Berlin and work at Acme.")
    assert info.mixed is False
    assert info.alternates == ()


def test_shared_stopwords_do_not_fake_code_switching():
    # Italian "per" is an ordinary English preposition, and Vietnamese "thể"
    # folds to "the".  One borrowed-looking word is not a second language.
    info = detect_language("per trade like even if the deposit the total")
    assert info.primary == "en"
    assert info.mixed is False
    assert info.alternates == ()


def test_two_foreign_stopwords_still_flag_code_switching():
    info = detect_language("I read the file. Je suis ici.")
    assert info.primary == "en"
    assert info.mixed is True
    assert "fr" in info.alternates


def test_a_second_script_is_code_switching():
    info = detect_language("東京は大きいです。I live in Berlin and work at Acme.")
    assert info.primary == "en"
    assert info.mixed is True
    assert "zh" in info.alternates


def test_spans_cover_the_message_in_order():
    text = "Hola, ¿cómo estás? Vivo en Madrid."
    info = detect_language(text)
    assert info.spans
    last = -1
    for span in info.spans:
        assert 0 <= span.start < span.end <= len(text)
        assert span.start >= last
        last = span.start


def test_detection_is_deterministic():
    text = "I am from Berlin. Je suis de Paris."
    first = detect_language(text)
    second = detect_language(text)
    assert first == second
