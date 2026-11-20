"""Deterministic language and script identification.

The detector combines two cheap signals: Unicode script runs and stopword
overlap for Latin-script text.  It reports a primary language, a confidence, a
per-segment view and whether the message mixes languages.  No model or network
call is involved, and a message with too little evidence returns ``"und"``
rather than a guess.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from .contracts import LanguageInfo, LanguageSpan
from .lexicon import (
    CYRILLIC_MARKERS,
    EMAIL,
    LANGUAGE_STOPWORDS,
    SCRIPT_LANGUAGE,
    SCRIPT_RANGES,
    URL,
)

_SENTENCE = re.compile(r"[^.!?。！？,;\n]+[.!?。！？,;]?")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _script_of(char: str) -> Optional[str]:
    code = ord(char)
    for script, ranges in SCRIPT_RANGES.items():
        for low, high in ranges:
            if low <= code <= high:
                return script
    return None


def _script_runs(words: Sequence[Tuple[str, int, int]]
                 ) -> List[Tuple[str, List[str], int, int]]:
    """Groups located words into contiguous runs of one writing system."""
    runs: List[Tuple[str, List[str], int, int]] = []
    for word, start, end in words:
        script = _script_of(word[0]) or "latin"
        if runs and runs[-1][0] == script:
            runs[-1][1].append(word)
            runs[-1] = (script, runs[-1][1], runs[-1][2], end)
        else:
            runs.append((script, [word], start, end))
    return runs


def _cyrillic_language(words: Sequence[str]) -> str:
    """Chooses between Cyrillic languages from their distinguishing letters."""
    joined = "".join(words)
    for language, markers in CYRILLIC_MARKERS.items():
        if any(char in markers for char in joined):
            return language
    return "ru"


def _fold(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _folded_stopwords() -> Dict[str, frozenset]:
    """Stopword lists compared the same way as text: accent-free.

    ``_fold`` strips diacritics from the message, so an accented list entry
    such as ``está`` could never match.  Folding the lists too makes ``cómo``
    and ``como`` the same evidence.
    """
    return {language: frozenset(_fold(word) for word in words)
            for language, words in LANGUAGE_STOPWORDS.items()}


_FOLDED_STOPWORDS = _folded_stopwords()


def _stopword_weights() -> Dict[str, float]:
    """Weights a stopword by how many languages share it.

    Words like ``de`` or ``in`` occur in several languages and carry little
    evidence, so a message whose only match is a shared word stays ``"und"``
    instead of being assigned to whichever list sorts first.
    """
    document_frequency: Dict[str, int] = {}
    for stopwords in _FOLDED_STOPWORDS.values():
        for word in stopwords:
            document_frequency[word] = document_frequency.get(word, 0) + 1
    return {word: 1.0 / count for word, count in document_frequency.items()}


_STOPWORD_WEIGHT = _stopword_weights()


def _score_latin(words: Sequence[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for word in words:
        folded = _fold(word)
        weight = _STOPWORD_WEIGHT.get(folded)
        if weight is None:
            continue
        for language, stopwords in _FOLDED_STOPWORDS.items():
            if folded in stopwords:
                scores[language] = scores.get(language, 0.0) + weight
    return scores


def _second_language_evidence(words: Sequence[str], primary: str
                              ) -> Dict[str, Tuple[float, int]]:
    """Stopword evidence for every language other than the primary.

    A word that belongs to the primary language is no evidence of a switch, and
    one borrowed-looking word is not a second language: ``per`` is an Italian
    stopword but also an ordinary English preposition, and Vietnamese ``thể``
    folds to ``the``.  The caller decides how much evidence counts; this returns
    the weight and the number of distinct stopword types per language.
    """
    primary_words = _FOLDED_STOPWORDS.get(primary, frozenset())
    evidence: Dict[str, Tuple[float, int]] = {}
    types: Dict[str, set] = {}
    for word in words:
        folded = _fold(word)
        if folded in primary_words:
            continue
        weight = _STOPWORD_WEIGHT.get(folded)
        if weight is None:
            continue
        for language, stopwords in _FOLDED_STOPWORDS.items():
            if language == primary or folded not in stopwords:
                continue
            score = evidence.get(language, (0.0, 0))[0]
            seen = types.setdefault(language, set())
            seen.add(folded)
            evidence[language] = (score + weight, len(seen))
    return evidence


def _pick_language(scores: Dict[str, float], word_count: int
                   ) -> Tuple[str, float, Tuple[str, ...]]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ordered or ordered[0][1] <= 0.0:
        return "und", 0.15 if word_count else 0.0, ()
    best_language, best = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    coverage = best / max(word_count, 1)
    margin = (best - runner_up) / max(best, 1)
    confidence = min(0.95, 0.3 + 0.5 * coverage + 0.35 * margin)
    if best < 0.6:
        return "und", 0.2, ()
    if runner_up > 0 and runner_up >= best * 0.9:
        return "und", 0.25, ()
    alternates = tuple(
        language for language, score in ordered[1:]
        if score > 0 and score >= best * 0.6)
    if runner_up > 0 and runner_up >= best * 0.8:
        confidence = min(confidence, 0.55)
    return best_language, round(confidence, 4), alternates


def detect_language(text: str) -> LanguageInfo:
    """Classifies one message and returns per-segment spans."""
    spans: List[LanguageSpan] = []
    primary_votes: Dict[str, int] = {}
    word_count = 0
    # Weighted stopword evidence per language for the whole message.  A single
    # borrowed word is not code-switching, so a language needs 1.0 of weight
    # (one exclusive stopword or several shared ones) to count.
    evidence: Dict[str, float] = {}
    latin_words: List[str] = []
    script_languages: set = set()
    switched_script = False

    masked = text
    for pattern in (URL, EMAIL):
        masked = pattern.sub(lambda match: " " * (match.end() - match.start()),
                             masked)

    for match in _SENTENCE.finditer(masked):
        segment = match.group(0)
        if not segment.strip():
            continue
        located = [(item.group(0), item.start(), item.end())
                   for item in _WORD.finditer(segment)]
        if not located:
            continue
        word_count += len(located)
        words = [word for word, _, _ in located]
        runs = _script_runs(located)
        if len(runs) > 1:
            # A segment that changes writing system mid-sentence is
            # code-switched: "I live in አዲስ አበባ" is an English frame around an
            # Amharic value.  Describing each run on its own keeps the frame's
            # language and counts both scripts as present, instead of labelling
            # the whole segment with its majority script.
            switched_script = True
            for script, run_words, start, end in runs:
                if script == "latin":
                    scores = _score_latin(run_words)
                    for latin_language, score in scores.items():
                        if score > 0:
                            evidence[latin_language] = evidence.get(
                                latin_language, 0.0) + score
                    language, confidence, _ = _pick_language(
                        scores, len(run_words))
                    latin_words.extend(run_words)
                else:
                    language = (_cyrillic_language(run_words)
                                if script == "cyrillic"
                                else SCRIPT_LANGUAGE.get(script, "und"))
                    confidence = min(
                        0.95, 0.55 + 0.4 * (len(run_words) / len(located)))
                if language != "und":
                    script_languages.add(language)
                spans.append(LanguageSpan(
                    match.start() + start, match.start() + end,
                    language, round(confidence, 4)))
                primary_votes[language] = (
                    primary_votes.get(language, 0) + len(run_words))
            continue
        latin = [word for word in words
                 if (_script_of(word[0]) or "latin") == "latin"]
        script, run_words = runs[0][0], runs[0][1]
        if script != "latin":
            language = (_cyrillic_language(run_words)
                        if script == "cyrillic"
                        else SCRIPT_LANGUAGE.get(script, "und"))
            confidence = min(0.95, 0.55 + 0.4 * (len(run_words) / len(words)))
            spans.append(LanguageSpan(
                match.start(), match.end(), language, round(confidence, 4)))
            primary_votes[language] = (
                primary_votes.get(language, 0) + len(run_words))
            script_languages.add(language)
            continue
        latin_words.extend(latin)
        scores = _score_latin(latin)
        for language, score in scores.items():
            if score > 0:
                evidence[language] = evidence.get(language, 0.0) + score
        language, confidence, _ = _pick_language(scores, len(latin))
        spans.append(LanguageSpan(
            match.start(), match.end(), language, confidence))
        primary_votes[language] = primary_votes.get(language, 0) + len(latin)

    if not primary_votes:
        return LanguageInfo(primary="und", confidence=0.0, spans=())

    ordered = sorted(primary_votes.items(), key=lambda item: (-item[1], item[0]))
    primary = ordered[0][0]
    total = sum(primary_votes.values())
    share = ordered[0][1] / max(total, 1)

    # Code-switching needs evidence of its own for a second language: a word
    # shared with the primary language is not a switch, and one borrowed word
    # (Italian "per", Vietnamese "thể" which folds to "the") is not either.  A
    # language counts when it has at least two distinct stopword types or when
    # those words make up a fifth of the message, and a second script always
    # counts.
    spoken = set()
    if evidence.get(primary, 0.0) >= 1.0:
        spoken.add(primary)
    second = _second_language_evidence(latin_words, primary)
    for language, (score, distinct) in second.items():
        share = distinct / max(len(latin_words), 1)
        if score >= 1.0 and (distinct >= 2 or share >= 0.2):
            spoken.add(language)
    spoken |= script_languages
    # A message that switches writing system between words carries content in
    # two languages even when one side's stopwords are too sparse or too
    # language-shared to name it ("I live in አዲስ አበባ").
    mixed = switched_script or len(spoken) > 1

    if primary == "und":
        confidence = 0.2 if word_count else 0.0
        alternates = tuple(sorted(spoken))
    else:
        confidence = max(
            (span.confidence for span in spans if span.language == primary),
            default=0.6)
        alternates = tuple(
            sorted(language for language in spoken if language != primary))

    return LanguageInfo(
        primary=primary,
        confidence=round(min(0.98, max(0.0, share * 0.4 + confidence * 0.6)),
                         4),
        mixed=mixed,
        alternates=alternates,
        spans=tuple(spans),
    )


__all__ = ["detect_language"]
