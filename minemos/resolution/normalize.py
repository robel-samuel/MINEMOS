"""Stage 1 — noise normalization.

Human messages contain typos, missing punctuation, informal spellings and
mixed languages.  This stage produces a normalized view while keeping the
source message untouched:

* whitespace, control characters and compatibility forms are tidied;
* a closed set of contractions and informal spellings is expanded;
* misspellings are corrected only when a single vocabulary word is exactly one
  edit away;
* names, numbers, dates, URLs, e-mail addresses and any capitalized token are
  never rewritten;
* every change is recorded as a reversible :class:`TextEdit` tied to source
  offsets, and every normalized token points back at the characters it came
  from.

Anything the stage cannot resolve with confidence is left as written and
reported as an ambiguity.  Normalization never adds words that were not
already implied by a known contraction or spelling.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .config import ResolutionConfig
from .contracts import (
    NormalizedText,
    NormalizedToken,
    TextEdit,
    render_tokens,
)
from .lexicon import (
    COMMON_MISSPELLINGS,
    CONTRACTIONS,
    CONTENT_WORDS,
    DATE,
    EMAIL,
    INFORMAL,
    COMMON_PLACE_NAMES,
    NUMBER,
    STRUCTURAL_WORDS,
    TOKEN,
    TYPO_PROTECTED,
    TYPO_VOCABULARY,
    URL,
    is_vocabulary_inflection,
)
from .similarity import edit_distance, is_transposition

_WORD = frozenset(TYPO_VOCABULARY) | CONTENT_WORDS

# Misspellings are only repaired when the correction is a word the extraction
# rules must recognise.  Rewriting content values is forbidden: a lowercased
# place name like ``recife`` is one edit from ``recipe`` and a lowercased
# ``plays`` is one edit from ``pays``, and silently substituting either would
# change a fact.  Values stay exactly as the user wrote them.
_CORRECTABLE = frozenset("""
a an the my our i we you he she they it is are am was were be been being
has have had do does did not no never also still now then and but or so
in at on for from by with of to as if when where what which who how
favorite favourite colour color author book film movie food drink music song
sport team language native job title role salary rent age phone number email
address degree birthday hometown
lives live lived living resides reside resided works work worked working
studies study studied studying plays play played reads read reading speaks
speak speaks teaches teach taught owns own owned drinks drink drank eats eat
eaten allergic born employed commute commutes commuted volunteer volunteers
volunteered called named name likes like loves love hates hate uses use used
moved move relocated
""".split())

# A lowercase token directly followed by a lexical predicate is the last word
# of a subject phrase far more often than it is a misspelled verb: "ana lopes
# lives in Madrid" names a person, and "lopes" is one substitution from the
# vocabulary verb "loves".  Repairs there would destroy the name before Stage 3
# ever sees it.  Copulas and auxiliaries are excluded because attribute nouns
# sit before them ("favourite colof is scarlet"), and attribute nouns are absent
# so that "her favourit colour is teal" still repairs.
_PREDICATE_FOLLOWERS = frozenset("""
lives live lived residing resides reside
works worked working studies studied studying
plays played reads teaches taught speaks speak spoke
owns owned drinks drank eats ate born allergic
commutes commuted volunteers volunteered
likes loves hated hates uses used
move moved relocates relocated
enjoys prefers earns pays got
""".split())

# Constructions that introduce a person's name.  The name may end the
# sentence, where the predicate lookahead cannot see it, so the cue is the
# evidence that the following tokens are a name and not misspelt verbs.
_NAME_CUES = frozenset({"called", "named", "meet", "i'm", "name's"})


def _deletions(word: str) -> Set[str]:
    variants = {word}
    for index in range(len(word)):
        variants.add(word[:index] + word[index + 1:])
    return variants


def _build_index() -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for word in TYPO_VOCABULARY:
        for variant in _deletions(word):
            index.setdefault(variant, []).append(word)
    return index


_DELETION_INDEX = _build_index()

_VOWELS = str.maketrans({letter: None for letter in "aeiouy"})


def _skeleton(word: str) -> str:
    return word.translate(_VOWELS)


def _build_skeleton_index() -> Dict[str, List[str]]:
    """Vowel-stripped forms of the words extraction must recognise.

    Chat shorthand drops vowels (``spks``, ``tchs``, ``rds``).  Mapping such a
    token back needs the full word, and only the closed functional vocabulary
    is eligible: expanding a content value would invent information.
    """
    index: Dict[str, List[str]] = {}
    for word in _CORRECTABLE:
        skeleton = _skeleton(word)
        if len(skeleton) >= 2:
            index.setdefault(skeleton, []).append(word)
    return index


_SKELETON_INDEX = _build_skeleton_index()


def _protected_intervals(text: str) -> List[Tuple[int, int]]:
    """Non-overlapping spans that must never be edited or split.

    Numbers, dates and URLs overlap freely (``12/03/2021`` is both), so
    intervals are reduced to a greedy longest-first cover; every accepted span
    starts after the previous one ends.
    """
    found: List[Tuple[int, int]] = []
    for pattern in (URL, EMAIL, NUMBER, DATE):
        for match in pattern.finditer(text):
            start, end = match.span()
            if pattern is URL:
                while end > start and text[end - 1] in ".,;:!?)]}'\"":
                    end -= 1
            found.append((start, end))
    found.sort(key=lambda span: (span[0], span[0] - span[1]))
    intervals: List[Tuple[int, int]] = []
    for start, end in found:
        if start < end and (not intervals or start >= intervals[-1][1]):
            intervals.append((start, end))
    return intervals


def _covering(position: int, intervals: Sequence[Tuple[int, int]]
              ) -> Optional[Tuple[int, int]]:
    for start, end in intervals:
        if start <= position < end:
            return start, end
    return None


def _compat_char(char: str) -> str:
    normalized = unicodedata.normalize("NFKC", char)
    return normalized if len(normalized) == 1 else char


def _compatibility_view(text: str) -> Tuple[str, List[TextEdit]]:
    """NFKC-normalizes per character and records every changed run.

    Per-character normalization keeps every offset in step with the source, so
    the record can still point at the original bytes.  The changes are recorded
    as edits because a normalized view the operator cannot diff is not
    inspectable.
    """
    parts: List[str] = []
    edits: List[TextEdit] = []
    run_start: Optional[int] = None
    for index, char in enumerate(text):
        replacement = _compat_char(char)
        parts.append(replacement)
        if replacement != char:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            edits.append(TextEdit(
                "compatibility", run_start, index, text[run_start:index],
                "".join(parts[run_start:index]), confidence=0.99,
                reason="unicode compatibility form"))
            run_start = None
    if run_start is not None:
        edits.append(TextEdit(
            "compatibility", run_start, len(text), text[run_start:],
            "".join(parts[run_start:]), confidence=0.99,
            reason="unicode compatibility form"))
    return "".join(parts), edits


def _known_inflection(lower: str) -> bool:
    """True when the token is a regular inflection of a known word.

    Without this guard the typo corrector rewrites valid forms that happen to
    be absent from the vocabulary (``commutes`` → ``commute``, ``worked`` →
    ``work``), which would silently change tense and agreement.
    """
    return is_vocabulary_inflection(lower)


def _skeleton_candidate(lower: str) -> Optional[str]:
    """Expands vowel-dropped shorthand when exactly one word fits.

    ``spks`` keeps the consonants of ``speaks`` and nothing else in the
    functional vocabulary shares them, so the expansion is determined by the
    closed list rather than by a guess.
    """
    if len(lower) < 3:
        return None
    skeleton = _skeleton(lower)
    # Only a token with no vowels at all is treated as shorthand; ``clara``
    # shares "clr" with ``colour`` but is a name, not an abbreviation.
    if len(skeleton) < 2 or skeleton != lower:
        return None
    matches = [word for word in _SKELETON_INDEX.get(skeleton, ())
               if len(word) > len(lower)]
    if len(matches) != 1:
        return None
    return matches[0]


def _structural_candidate(lower: str, config: ResolutionConfig
                          ) -> Optional[str]:
    """Recovers a misspelled structural word from the closed predicate list.

    A misspelled verb or attribute name ("teachfes", "agyd", "rolq") breaks
    parsing without carrying content, so it is recovered from its own closed
    list under the ``max_structural_edits`` bound instead of the general
    vocabulary rules.  The token has to be out of vocabulary, keep its first
    letter and match exactly one structural word within the bound; a value,
    place, organisation or name is never eligible because the closed list does
    not contain them.
    """
    if len(lower) < 4:
        return None
    # A structural word that is already spelled correctly must not be pulled
    # towards a different inflection: "resides" is one edit from "reside" and
    # rewriting it would change agreement for no gain.
    if lower in STRUCTURAL_WORDS:
        return None
    limit = config.max_structural_edits
    if len(lower) <= 5:
        # Between short words a substitution is more often a different word
        # than a slip, so the tighter bound applies there.
        limit = min(limit, 1)
    if limit <= 0:
        return None
    matches: List[str] = []
    for word in STRUCTURAL_WORDS:
        if word[:1] != lower[:1] or abs(len(word) - len(lower)) > limit:
            continue
        if edit_distance(lower, word, limit) <= limit:
            matches.append(word)
    if len(matches) != 1:
        # "plys" is one substitution from "pays" and one insertion from
        # "plays".  A dropped letter is the commoner slip, so when exactly one
        # reading changes the length it wins; anything else stays ambiguous.
        changed_length = [word for word in matches
                          if len(word) != len(lower)]
        if len(changed_length) != 1:
            return None
        return changed_length[0]
    return matches[0]


def _transposition_candidate(lower: str) -> List[str]:
    """Vocabulary words one adjacent swap away from the token.

    A swap keeps every letter of the token, so it is the most specific repair
    available.  It is looked up directly because the deletion index only
    contains one part of the vocabulary: "lieks" must reach "likes", not the
    one-deletion neighbour "lives".
    """
    found: List[str] = []
    for index in range(len(lower) - 1):
        if lower[index] == lower[index + 1]:
            continue
        swapped = (lower[:index] + lower[index + 1] + lower[index]
                   + lower[index + 2:])
        if swapped in _CORRECTABLE:
            found.append(swapped)
    return found


def _typo_candidate(word: str, config: ResolutionConfig, next_word: str = "",
                    name_slot: bool = False
                    ) -> Tuple[Optional[str], Optional[str]]:
    """Returns (replacement, ambiguity_reason)."""
    lower = word.lower()
    if len(lower) < config.typo_min_length:
        return None, None
    if lower in _WORD or lower in STRUCTURAL_WORDS or lower in TYPO_PROTECTED \
            or lower in COMMON_PLACE_NAMES:
        return None, None
    if any(char.isdigit() for char in lower):
        return None, None
    if "'" in lower or "-" in lower:
        return None, None
    if config.preserve_capitalized and word[:1].isupper():
        # Capitalized tokens are name-like; a correction here would silently
        # rewrite an entity.  Stage 3 may link such a variant explicitly.
        return None, None
    if not lower.isalpha() or not lower.isascii():
        return None, None
    if lower in COMMON_MISSPELLINGS:
        replacement = COMMON_MISSPELLINGS[lower]
        if replacement in _CORRECTABLE:
            return replacement, None
        return None, None
    if _known_inflection(lower):
        return None, None
    if next_word in _PREDICATE_FOLLOWERS:
        # A lowercase token in the subject slot is left exactly as written.
        # The identity stage may still link a damaged name to a recorded one
        # from explicit evidence; Stage 1 must not decide it by itself.
        return None, None
    if name_slot:
        return None, None
    swaps = _transposition_candidate(lower)
    if len(swaps) > 1:
        return None, f"typo_ambiguous:{lower}"
    if swaps:
        return swaps[0], None
    candidates: Set[str] = set()
    for variant in _deletions(lower):
        for candidate in _DELETION_INDEX.get(variant, ()):
            if candidate in _CORRECTABLE and \
                    abs(len(candidate) - len(lower)) <= config.max_typo_edits:
                candidates.add(candidate)
    candidates.discard(lower)
    if candidates and len(lower) <= 4:
        # Between two four-letter words one substitution is far more often a
        # different word than a typo ("tram" is not "team"); only an adjacent
        # swap is repaired at that length.
        candidates = {candidate for candidate in candidates
                      if len(candidate) != len(lower)
                      or is_transposition(lower, candidate)}
        if len(lower) <= 3:
            # At three letters a deleted character is a third of the token:
            # "sam" → "am" and "tom" → "to" are names, not slips.  Leave the
            # token alone rather than rewrite a name into a function word.
            candidates = {candidate for candidate in candidates
                          if is_transposition(lower, candidate)}
    elif candidates:
        # For longer words a substitution is a plausible typo ("commutz" →
        # "commute"), but only after the first letter: a word whose opening
        # letter changed is far more often a different word ("leading" is not
        # "reading") than a slip.
        candidates = {candidate for candidate in candidates
                      if len(candidate) != len(lower)
                      or is_transposition(lower, candidate)
                      or lower[:1] == candidate[:1]}
    if not candidates:
        structural = _structural_candidate(lower, config)
        if structural is not None:
            return structural, None
        return _skeleton_candidate(lower), None
    if len(candidates) > 1:
        # "wrks" is one edit away from both "work" and "works".  When the only
        # competing readings are a stem and its own plural, the input's ending
        # is the evidence: keeping it preserves the author's agreement.
        if lower.endswith("s") and len(candidates) == 2:
            plural = [item for item in candidates
                      if item.endswith("s") and item[:-1] in candidates]
            if len(plural) == 1:
                return plural[0], None
        return None, f"typo_ambiguous:{lower}"
    return next(iter(candidates)), None


def normalize_text(text: str, config: Optional[ResolutionConfig] = None,
                   language: Optional[str] = None) -> NormalizedText:
    """Normalizes one message and keeps an editable link to the source.

    ``language`` is the detected primary language.  The contraction, informal
    and typo lexicons are English-only, so they are applied when no language
    was supplied (direct callers) and skipped for other detected languages;
    remaining stages still see the untouched text plus its spans.
    """
    cfg = config or ResolutionConfig()
    if not isinstance(text, str):
        raise TypeError("normalize_text expects a string")
    english = language in (None, "en", "und")

    working, compat_edits = _compatibility_view(text)
    protected = _protected_intervals(working)
    tokens: List[NormalizedToken] = []
    edits: List[TextEdit] = list(compat_edits)
    ambiguities: List[str] = []
    skip_until = 0

    matches = list(TOKEN.finditer(working))
    name_run = False
    previous_lower = ""
    for position, match in enumerate(matches):
        value = match.group(0)
        start, end = match.span()
        lower = value.casefold().replace("\u2019", "'")
        # Track whether the token sits in a name introduction.  The run ends at
        # the first token that is not name-shaped (a known vocabulary word,
        # punctuation, a number), so "my name is ana lopes and ..." protects
        # only the two name tokens.
        in_name = name_run
        if lower in _NAME_CUES or (lower == "is" and previous_lower == "name"):
            name_run = True
        elif name_run and (not lower.isalpha() or value[:1].isupper()
                           or lower in _WORD or lower in STRUCTURAL_WORDS
                           or lower in TYPO_PROTECTED):
            name_run = False
        previous_lower = lower
        if start < skip_until:
            continue
        covered = _covering(start, protected)
        if covered is not None:
            if start == covered[0] and end <= covered[1]:
                # URLs, e-mail addresses, numbers and dates stay byte-identical
                # and are emitted as one token so they can never be split or
                # edited.
                tokens.append(NormalizedToken(working[covered[0]:covered[1]],
                                              covered[0], covered[1]))
                skip_until = covered[1]
                continue
            if end > covered[1]:
                # A token that runs past a protected span ("30-something" over
                # a protected "30") keeps its whole text; trimming it to the
                # protected part would drop characters.
                tokens.append(NormalizedToken(value, start, end))
                continue
            continue
        if not any(char.isalpha() for char in value):
            tokens.append(NormalizedToken(value, start, end))
            continue
        lower = value.lower()
        if cfg.normalize_contractions and english and lower in CONTRACTIONS:
            replacement = CONTRACTIONS[lower]
            for part in replacement.split():
                tokens.append(NormalizedToken(part, start, end, edited=True))
            edits.append(TextEdit(
                "contraction", start, end, value, replacement,
                confidence=0.95, reason="closed contraction list"))
            continue
        if cfg.normalize_informal and english and lower in INFORMAL:
            replacement = INFORMAL[lower]
            for part in replacement.split():
                tokens.append(NormalizedToken(part, start, end, edited=True))
            edits.append(TextEdit(
                "informal", start, end, value, replacement,
                confidence=0.9, reason="unambiguous informal form"))
            continue
        if cfg.normalize_typos and english:
            next_word = (matches[position + 1].group(0).lower()
                         if position + 1 < len(matches) else "")
            replacement, ambiguity = _typo_candidate(
                value, cfg, next_word, in_name)
            if replacement:
                tokens.append(NormalizedToken(
                    replacement, start, end, edited=True))
                edits.append(TextEdit(
                    "typo", start, end, value, replacement,
                    confidence=cfg.typo_confidence,
                    reason="single vocabulary candidate at one edit"))
                continue
            if ambiguity:
                ambiguities.append(ambiguity)
        tokens.append(NormalizedToken(value, start, end))

    collapsed: List[NormalizedToken] = []
    for token in tokens:
        if (collapsed and token.text.lower() == collapsed[-1].text.lower()
                and token.text.isalpha()
                and token.source_start > collapsed[-1].source_end
                and token.text.lower() not in {"that", "had", "is"}):
            edits.append(TextEdit(
                "repeated_word", token.source_start, token.source_end,
                token.text, "", confidence=0.7,
                reason="immediately repeated token"))
            continue
        collapsed.append(token)

    normalized = render_tokens([token.text for token in collapsed])
    collapsed_source = " ".join(text.split())
    if cfg.normalize_whitespace and collapsed_source != text.strip():
        edits.append(TextEdit(
            "whitespace", 0, len(text), text, normalized, confidence=0.99,
            reason="whitespace and compatibility normalization"))
    terminated = bool(normalized.strip()) and normalized.strip()[-1] in \
        ".!?。！？።"
    return NormalizedText(
        original=text,
        text=normalized,
        tokens=tuple(collapsed),
        edits=tuple(edits),
        sentence_terminated=terminated,
        ambiguities=tuple(ambiguities),
    )


__all__ = ["normalize_text"]
