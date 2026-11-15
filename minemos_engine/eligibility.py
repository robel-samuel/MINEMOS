"""Opt-in query eligibility for retrieval abstention.

The helpers here are deliberately lexical and dependency-free.  They answer
two questions before a stored fact may be returned as an answer:

  1. Which entity does the query ask about, and is that entity actually in
     memory?
  2. Does the query ask for an attribute that the entity has stored?

Nothing here ranks or rewrites memory; ranking remains the engine's canonical
dense retrieval.  When the query does not identify an entity (indirect or
fragment queries), the caller reports ``unverified`` instead of guessing.  The
engine still returns its ungated ranking so callers can diagnose the miss, but
the HTTP boundary withholds those rows from ordinary query responses.

No similarity threshold is applied by default.  ``low_similarity`` is only
reported when a caller supplies a threshold calibrated for its own corpus.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

# Attribute families shared by stored predicates and natural-language queries.
# Matching is whole-token and contiguous, so "work" never matches "network".
ATTRIBUTE_MARKERS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "favorite_color": (
        ("favorite", "color"), ("favourite", "color"),
        ("color",), ("colour",)),
    "lives_in": (
        ("lives", "in"), ("live", "in"), ("lives",), ("live",),
        ("resides", "in"), ("resides",), ("reside",), ("residence",)),
    "works_at": (
        ("works", "at"), ("work", "at"), ("works",), ("work",),
        ("employer",), ("employed", "by"), ("company",)),
    "role": (
        ("role",), ("job", "title"), ("job", "role"), ("position",),
        ("profession",), ("do", "for", "work")),
    "drinks": (("drinks",), ("drink",), ("beverage",)),
    "studies": (("studies",), ("study",), ("field",)),
    "plays": (("plays",), ("play",), ("instrument",)),
    "allergic_to": (("allergic",), ("allergy",), ("allergen",)),
    "owns": (("owns",), ("own",), ("possession",), ("possessions",)),
    "born_in": (("born",), ("birth",)),
    "speaks": (("speaks",), ("speak",), ("language",)),
    "reads": (("reads",), ("read",), ("genre",)),
    "teaches": (("teaches",), ("teach",), ("subject",)),
    "salary": (("salary",), ("earn",), ("earns",)),
    "rent": (("rent",), ("pay",), ("pays",)),
    "favorite_author": (
        ("favorite", "author"), ("favourite", "author"), ("author",)),
    "favorite_film": (
        ("favorite", "film"), ("favourite", "film"), ("film",), ("movie",)),
    "commutes_by": (("commutes",), ("commute",)),
    "volunteers_at": (
        ("volunteers",), ("volunteer",), ("organization",)),
    "age": (("age",), ("aged",), ("old",)),
}

_MARKER_TOKENS: FrozenSet[str] = frozenset(
    token
    for phrases in ATTRIBUTE_MARKERS.values()
    for phrase in phrases
    for token in phrase
)

_STOP_TOKENS: FrozenSet[str] = frozenset({
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "has", "have", "had", "can", "could",
    "much", "many", "the", "a", "an", "of", "for", "to", "by", "in", "at",
    "on", "with", "and", "or", "s",
    "like", "likes", "best", "usually", "use", "uses", "using",
    "from", "really", "into", "based", "about",
    "he", "she", "it", "they", "him", "her", "them", "his", "hers",
    "their", "theirs", "its",
    "studying", "learning", "reading", "working", "living", "playing",
    "volunteering", "commuting", "teaching", "speaking", "drinking",
})

_AUXILIARIES = (
    "does", "do", "did", "is", "was", "are", "were", "has", "have", "had",
    "can",
)

_POSSESSIVE = re.compile(r"(?:'s|’s)(?![^\W_])")
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

# Tokens that end the possessive head noun phrase: the head is the noun right
# after the possessive ("user's team called X"), never the verb that follows.
_HEAD_STOP_TOKENS: FrozenSet[str] = _STOP_TOKENS | frozenset({
    "called", "named", "known", "mean", "means", "located", "based",
    "referred", "described", "titled",
})

# Polite lead-ins that would otherwise be mistaken for the entity.  They are
# stripped only when they form an exact leading phrase; the entity is still
# validated against stored subjects afterwards.
_FRAMING_PREFIXES: Tuple[Tuple[str, ...], ...] = (
    ("can", "you", "please", "tell", "me"),
    ("could", "you", "please", "tell", "me"),
    ("would", "you", "please", "tell", "me"),
    ("can", "you", "please", "show", "me"),
    ("could", "you", "please", "show", "me"),
    ("can", "you", "tell", "me"),
    ("could", "you", "tell", "me"),
    ("would", "you", "tell", "me"),
    ("will", "you", "tell", "me"),
    ("can", "you", "show", "me"),
    ("could", "you", "show", "me"),
    ("would", "you", "show", "me"),
    ("do", "you", "know"),
    ("did", "you", "know"),
    ("i", "was", "wondering"),
    ("i", "am", "wondering"),
    ("please", "tell", "me"),
    ("please", "show", "me"),
    ("tell", "me"),
    ("show", "me"),
    ("please",),
)

# Verbs that introduce a previously mentioned entity by name.  In a follow-up
# such as "earlier we discussed Vera Ravenscroft, who studies ...", the
# question's subject follows this verb.
_FOCUS_MARKERS: Tuple[str, ...] = ("discussed", "mentioned")


def normalize(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", value).strip()


def tokens(text: object) -> List[str]:
    return _TOKEN.findall(normalize(text))


def _contains_phrase(haystack: Sequence[str],
                     phrase: Sequence[str]) -> bool:
    width = len(phrase)
    if not width or width > len(haystack):
        return False
    for start in range(len(haystack) - width + 1):
        if tuple(haystack[start:start + width]) == tuple(phrase):
            return True
    return False


def attribute_ids_from_tokens(raw_tokens: Sequence[str]) -> Set[str]:
    """Every attribute family whose marker appears in a token sequence."""
    if not raw_tokens:
        return set()
    return {
        attribute
        for attribute, phrases in ATTRIBUTE_MARKERS.items()
        if any(_contains_phrase(raw_tokens, phrase) for phrase in phrases)
    }


def attribute_ids(text: object) -> Set[str]:
    """Every attribute family whose marker appears in ``text``."""
    return attribute_ids_from_tokens(tokens(text))


def attributes_compatible(query_attributes: Iterable[str],
                          slot_attributes: Iterable[str]) -> bool:
    """Unknown on either side is compatible; known-and-disjoint is not."""
    query_set = set(query_attributes)
    slot_set = set(slot_attributes)
    if not query_set or not slot_set:
        return True
    return bool(query_set & slot_set)


def _dedupe_adjacent(focus: Sequence[str]) -> List[str]:
    # Repeated-word noise ("Oscar Oscar Dunmore") would otherwise demand a
    # token the stored name cannot supply twice.
    result: List[str] = []
    for token in focus:
        if not result or result[-1] != token:
            result.append(token)
    return result


def _collect_focus(raw_tokens: Sequence[str], start: int,
                   limit: int = 3) -> List[str]:
    focus: List[str] = []
    for token in raw_tokens[start:]:
        if len(focus) >= limit:
            break
        if token in _STOP_TOKENS or token in _MARKER_TOKENS:
            break
        focus.append(token)
    return focus


def _possessive_focus(text: str) -> List[str]:
    # The last possessive is normally the question's subject, and a
    # contraction ("what's Lyra's film?") is filtered out by the stop tokens.
    # Repeated-word noise can leave a trailing surname possessive
    # ("Clara Everhart's Everhart's color"); when the last focus is only a
    # suffix of an earlier one, the earlier full name is the subject.
    focuses: List[List[str]] = []
    for match in _POSSESSIVE.finditer(text):
        prefix = tokens(text[:match.start()])
        focus: List[str] = []
        for token in reversed(prefix):
            if token in _STOP_TOKENS or token in _MARKER_TOKENS:
                break
            focus.append(token)
            if len(focus) >= 3:
                break
        if focus:
            focuses.append(list(reversed(focus)))
    if not focuses:
        return []
    chosen = focuses[-1]
    for candidate in focuses:
        if (len(candidate) > len(chosen)
                and candidate[-len(chosen):] == chosen):
            chosen = candidate
    return chosen


def _possessive_head(text: str, limit: int = 2) -> List[str]:
    """Content tokens that follow the last possessive.

    In "what is user's team using?" the possessive names the possessor
    (``user``) while ``team`` names the possessed relation.  Both are stored
    as one subject ("user's team"), so a candidate that includes the head
    distinguishes that relation from the possessor's other facts.  Attribute
    words are never collected: they are what the question asks about, not
    part of the entity.
    """
    matches = list(_POSSESSIVE.finditer(text))
    if not matches:
        return []
    head: List[str] = []
    for token in tokens(text[matches[-1].end():]):
        if token in _HEAD_STOP_TOKENS or token in _MARKER_TOKENS:
            break
        head.append(token)
        if len(head) >= limit:
            break
    return head


def has_possessive_marker(text: object) -> bool:
    return bool(_POSSESSIVE.search(normalize(text)))


def _last_index(raw_tokens: Sequence[str], token: str) -> Optional[int]:
    for index in range(len(raw_tokens) - 1, -1, -1):
        if raw_tokens[index] == token:
            return index
    return None


def _skip_framing(raw_tokens: Sequence[str]) -> int:
    index = 0
    while True:
        for prefix in _FRAMING_PREFIXES:
            width = len(prefix)
            if tuple(raw_tokens[index:index + width]) == prefix:
                index += width
                break
        else:
            return index


def extract_query_focus(query_text: object
                        ) -> Tuple[Optional[str], List[List[str]]]:
    """Returns the asked-about entity and progressively shorter candidates.

    ``candidates`` starts with the full focus and may include a two-token
    prefix so a trailing misspelled verb does not hide the entity.  A
    possessive question also offers the possessor plus its head noun
    ("user" and "user's team"), because both are distinct stored subjects.
    A one-token fallback is deliberately never offered: a surname alone must
    not select an unrelated person who shares it.
    """
    text = normalize(query_text)
    raw = tokens(text)
    if not raw:
        return None, []
    start = _skip_framing(raw)

    focus = _possessive_focus(text)
    possessive_focus = bool(focus)
    if not focus:
        for index in range(start, len(raw)):
            if raw[index] in _AUXILIARIES:
                focus = _collect_focus(raw, index + 1)
                if focus:
                    break
    if not focus:
        affects = _last_index(raw, "affects")
        if affects is None:
            affects = _last_index(raw, "affect")
        if affects is not None:
            focus = _collect_focus(raw, affects + 1)
    if not focus:
        for marker in _FOCUS_MARKERS:
            marker_index = _last_index(raw, marker)
            if marker_index is not None:
                focus = _collect_focus(raw, marker_index + 1)
                if focus:
                    break
    if not focus:
        of_index = _last_index(raw, "of")
        if of_index is not None:
            focus = _collect_focus(raw, of_index + 1)
    if not focus:
        # Word-removal noise drops the auxiliary ("What Priya Dunmore
        # study?").  Skip the leading question words and attribute markers,
        # but require at least two tokens: a single stray word is not
        # evidence of an entity, so it stays unverified.
        index = start
        while (index < len(raw)
               and (raw[index] in _STOP_TOKENS
                    or raw[index] in _MARKER_TOKENS)):
            index += 1
        fallback = _collect_focus(raw, index)
        if len(fallback) >= 2:
            focus = fallback
    if not focus:
        return None, []

    focus = _dedupe_adjacent(focus)
    candidates = [list(focus)]
    if possessive_focus:
        # Only the possessive form contributes a head: "user's team" is one
        # stored subject, while "which team does user use" is not.
        head = _possessive_head(text)
        if head:
            combined = _dedupe_adjacent(focus + head)
            if len(combined) > len(focus):
                candidates.append(combined)
    if len(focus) >= 3:
        candidates.append(list(focus[:2]))
    return " ".join(focus), candidates


def _token_matches(query_token: str, subject_token: str,
                   exact: bool) -> bool:
    if query_token == subject_token:
        return True
    if exact:
        return False
    if len(query_token) < 4 or len(subject_token) < 4:
        return False
    if query_token[0] != subject_token[0]:
        return False
    if abs(len(query_token) - len(subject_token)) > 2:
        return False
    return SequenceMatcher(
        None, query_token, subject_token).ratio() >= 0.8


def focus_match_score(subject: object, focus: Sequence[str],
                      exact: bool = False) -> Optional[float]:
    """How closely a stored subject accounts for every focus token.

    Returns the weakest per-token similarity, or ``None`` when any focus
    token has no counterpart.  The engine keeps only the best-scoring
    subject(s) for a fuzzy focus, so a typo does not drag in every person
    whose name happens to be nearby.
    """
    subject_tokens = tokens(subject)
    if not subject_tokens or not focus:
        return None
    used: Set[int] = set()
    scores: List[float] = []
    for query_token in focus:
        best: Optional[float] = None
        best_index: Optional[int] = None
        for index, subject_token in enumerate(subject_tokens):
            if index in used:
                continue
            if not _token_matches(query_token, subject_token, exact):
                continue
            if query_token == subject_token:
                ratio = 1.0
            else:
                ratio = SequenceMatcher(
                    None, query_token, subject_token).ratio()
            if best is None or ratio > best:
                best = ratio
                best_index = index
        if best_index is None:
            return None
        used.add(best_index)
        scores.append(best)
    return min(scores)


def focus_matches_subject(subject: object, focus: Sequence[str],
                          exact: bool = False) -> bool:
    """Whether every focus token is accounted for by the stored subject."""
    return focus_match_score(subject, focus, exact=exact) is not None
