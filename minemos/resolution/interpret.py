"""Stage 2 — semantic interpretation.

Turns one normalized message into *candidate* entities and facts.  Every
candidate carries the rule that produced it, the span it was read from, a
confidence and an evidence pointer back to the original message.  Nothing is
written to canonical memory and no truth decision is taken here.

The extractor is rule-based on purpose: it is deterministic, needs no model
download and cannot fabricate a value that is not present in the text.  A
clause that no rule recognises produces no candidate rather than a guess.

Rules are gated by the language span they fall in, so an English clause inside
a Spanish message is still read, while Spanish rules never fire on English
text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

from .config import ResolutionConfig
from .contracts import (
    Entity,
    Evidence,
    FactCandidate,
    LanguageInfo,
    NormalizedText,
    SourceMessage,
    SubjectRef,
    TemporalExpression,
)
from .lexicon import (
    ATTRIBUTE_PREDICATES,
    CONTENT_WORDS,
    CORRECTION_MARKERS,
    NEGATIONS,
    PLACE_ALIASES,
    RELATION_NOUNS,
    STOPWORDS,
    TEMPORAL_FUTURE,
    TEMPORAL_NOW,
    TEMPORAL_PAST,
    TYPO_VOCABULARY,
)
from .similarity import edit_distance

# A correction often opens with a discourse marker before the actual
# statement; the marker carries no fact and must not become a name token.
_LEADING_MARKER = re.compile(
    r"^(?:" + "|".join(
        re.escape(marker) for marker in
        sorted(CORRECTION_MARKERS, key=len, reverse=True)
    ) + r"|well|oh|btw|anyway|okay|ok|also|so)\b\s*[,:;-]?\s*",
    re.IGNORECASE)

_CLAUSE_SPLIT = re.compile(
    r"[!?。！？።;\n]+"
    # A period after a lone capital is an initial ("N. K. Jemisin"), not the
    # end of a clause; splitting there would cut an author's name in three.
    r"|(?<!\b[A-Z])\.+")
_JOINER = re.compile(
    r",\s*(?:(?:but|and|so|then|pero|y|e|et|mais|puis)\s+)?"
    r"|\s+(?:but|and|so|then|pero|y|e|et|mais|puis)\s+", re.IGNORECASE)
_JOIN_TAIL = re.compile(
    r"^(?:or\s+)?(?:i|i'm|im|we|my|he|she|they|it|yo|mi|vivo|trabajo|tengo|"
    r"je|j'|nous|on|mon|ma|mes|il|elle|ils|elles|vivo|vive|"
    # A coordinated verb keeps the subject the speaker already established:
    # "I study medicine and speak French" is two facts, not one long value.
    # Only verbs that take an object on their own are listed; "work" and
    # "live" need their preposition, and a bare one would let "and work is
    # busy" become a fact.
    r"works?|lives?|studies|likes|loves|has|"
    r"speaks?|drinks?|eats?|plays?|reads?|teaches?|stud(?:y|ies)|owns?|uses?|"
    r"needs?)\b", re.IGNORECASE)

# "I have a website, not her" contrasts the speaker with someone else; the
# negated tail belongs to its own fragment and must not flip the polarity of
# the fact before it or widen its value.
_CONTRAST_TAIL = re.compile(
    r"^(?:not|rather than|instead of)\b", re.IGNORECASE)

_QUESTION_START = re.compile(
    r"^(?:what|which|who|whom|whose|where|when|why|how|is|are|was|were|do|"
    r"does|did|can|could|would|should|will|have|has|am)\b", re.IGNORECASE)

_TRAILING_TEMPORAL = re.compile(
    r"\s+(?:now|currently|these days|at the moment|right now|any ?more|too|"
    r"as well|already|still|today|yesterday|tomorrow|"
    r"since\s+(?:19|20)\d{2}|since\s+\w+|used to|previously|formerly|"
    r"last (?:year|month|week)|next (?:year|month|week)|"
    r"in\s+(?:19|20)\d{2})\s*$",
    re.IGNORECASE)
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_TRAILING_CONTEXT = re.compile(
    r"\s+(?:at|from|for)\s+(?:work|home|school|college|university|"
    r"the office|the moment|now)\s*$",
    re.IGNORECASE)
# "I migrated the database, in English" attaches the language to the telling,
# not to the migration.  The comma is required so a language word inside a
# value ("my favorite language is English") is never touched, and only a
# closed list of language names and their ordinary prepositions qualifies.
_LANGUAGE_NAMES = (
    r"english|amharic|spanish|french|arabic|german|italian|portuguese|dutch|"
    r"turkish|swahili|indonesian|vietnamese|chinese|mandarin|cantonese|"
    r"japanese|korean|russian|hindi|urdu|hebrew|greek|thai|persian|farsi|"
    r"bengali|tamil|telugu|malayalam|kannada|marathi|gujarati|punjabi|"
    r"somali|hausa|yoruba|igbo|zulu|xhosa|afrikaans|tagalog|filipino|malay|"
    r"romanian|polish|ukrainian|czech|swedish|norwegian|danish|finnish|"
    r"hungarian|bulgarian|croatian|serbian|catalan|basque|galician|latin|"
    r"esperanto|ingl[eé]s|franc[eé]s|espa[nñ]ol|alem[aá]n|anglais|fran[cç]ais|"
    r"espagnol|allemand|amarico|italiano|portugu[eê]s|አማርኛ"
)
_TRAILING_LANGUAGE = re.compile(
    r",\s+(?:in|en|au|em|di|auf)\s+(?:" + _LANGUAGE_NAMES + r")\s*$",
    re.IGNORECASE)
_TRAILING_PUNCT = " \t,;:.!?።፣\"'`"

# Running speech starts the next sentence without punctuation:
# "i live in ethiopia are they offering a visa".  The asserted value is the
# prefix; a following interrogative or a new first-person clause is not part
# of it.  Only clause-opening shapes qualify, and a relative clause ("the one
# i love", "a tattoo that says i love you") keeps its tail.
_TRAILING_NEW_CLAUSE = re.compile(
    r"\s+(?:"
    r"(?:how|what|why|when|where|who|whom|whose|which)\b(?:\s+\w+){0,3}\s+"
    r"(?:i|we|you|they|he|she|it|there)\b"
    r"|(?:do|does|did|can|could|would|should|will|is|are|was|were|have|has|"
    r"am)\s+(?:i|we|you|they|he|she|it|there)\b"
    r"|(?:i|we)\s+(?:need|needed|want|wanted|use|used|using|have|had|give|"
    r"gave|live|lived|work|worked|study|studied|speak|spoke|make|made|earn|"
    r"earned|get|got|like|love|know|think|can|could|will|would|should|am|was|do|"
    r"did|don't|didn't|'m|'ve|'ll)\b"
    r")",
    re.IGNORECASE)
_RELATIVE_GUARD = frozenset({
    "that", "which", "who", "whom", "whose", "one", "ones", "something",
    "anything", "everything", "says", "said", "say", "than", "because", "if",
    "when", "while", "as", "and", "or", "but", "so", "like",
})
_TRAILING_DANGLING = re.compile(r"\s+(?:the|a|an)\s*$", re.IGNORECASE)

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_EMAIL_VALUE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
_URL_VALUE = re.compile(r"^(?:https?://|www\.)\S+$")

_EPHEMERAL = re.compile(
    r"^(?:question|problem|issue|idea|feeling|doubt|request|suggestion|"
    r"thought|concern|bug|error|hard time|lot of|no idea|nothing|"
    r"a lot|plans?|trouble|fun)\b", re.IGNORECASE)

# "I have been asked this question before" is a past experience, not
# possession.  Without this guard the perfect passive is stored as an object
# the speaker owns.
_NOT_A_POSSESSION = re.compile(r"^been\b", re.IGNORECASE)

# "I'm about to apply for a job" describes an intention, not a role.  The
# role rule would otherwise read "about to apply a job application" as the
# speaker's current job.
_PROSPECTIVE = re.compile(
    r"^(?:about to|going to|planning to|trying to|hoping to|looking to|"
    r"want to)\b", re.IGNORECASE)

# "I have this, my friend" points at something in the conversation; a bare
# demonstrative is not an object the speaker possesses.
_DEMONSTRATIVES = frozenset({
    "this", "that", "it", "these", "those", "them", "him", "her",
})

_LANGUAGE_NAMES = frozenset("""
amharic arabic bengali chinese czech danish dutch english farsi finnish
flemish french german greek hebrew hindi hungarian igbo indonesian italian
japanese kannada korean latin malay mandarin norwegian persian polish
portuguese punjabi romanian russian serbian somali spanish swahili swedish
tagalog tamil telugu thai tibetan turkish ukrainian urdu vietnamese welsh
yoruba zulu
""".split())

_RELATION = "|".join(sorted(RELATION_NOUNS, key=len, reverse=True))
_ATTR = "|".join(sorted(ATTRIBUTE_PREDICATES, key=len, reverse=True))

_ENGLISH_PREDICATES = {
    "lives in": "lives_in", "live in": "lives_in",
    "works at": "works_at", "work at": "works_at",
    "works for": "works_at", "work for": "works_at",
    "works in": "works_in", "work in": "works_in",
    "studies at": "studies_at", "study at": "studies_at", "goes to": "attends",
    "is from": "is_from", "comes from": "is_from",
    "likes": "likes", "loves": "likes", "has": "has",
    "drinks": "drinks", "drink": "drinks", "eats": "eats", "eat": "eats",
    "reads": "reads", "read": "reads", "plays": "plays", "play": "plays",
    "speaks": "speaks", "speak": "speaks",
    "teaches": "teaches", "teach": "teaches",
    "owns": "owns", "own": "owns",
    "studies": "studies", "study": "studies",
    "uses": "uses", "use": "uses", "needs": "needs", "need": "needs",
    "commutes by": "commutes_by", "commute by": "commutes_by",
    "volunteers at": "volunteers_at", "volunteer at": "volunteers_at",
    "was born in": "born_in", "were born in": "born_in",
    "is allergic to": "allergic_to", "are allergic to": "allergic_to",
    "is aged": "age", "are aged": "age",
    "resides in": "lives_in", "reside in": "lives_in",
    "is employed by": "works_at", "are employed by": "works_at",
    # One-step paraphrases of the predicates above.
    "is fluent in": "speaks", "are fluent in": "speaks",
    "is studying": "studies", "are studying": "studies",
    "is researching": "studies", "are researching": "studies",
    "instructs in": "teaches", "instruct in": "teaches",
    "travels by": "commutes_by", "travel by": "commutes_by",
    "enjoys reading": "reads", "enjoy reading": "reads",
    "volunteers with": "volunteers_at", "volunteer with": "volunteers_at",
    "prefers to drink": "drinks", "prefer to drink": "drinks",
    "likes to drink": "drinks", "loves to drink": "drinks",
    "has an allergy to": "allergic_to", "pays rent of": "rent",
    "pay rent of": "rent",
    "has got": "owns", "earns": "salary",
    # Filler words and light-verb forms that keep the same verb: "works over
    # at" and "does work as" name the same fact as "works at" / "work as".
    "works over at": "works_at", "work over at": "works_at",
    "does work as": "role", "do work as": "role",
    # "is into reading" carries the reading verb itself, so the fact is read,
    # not inferred from the interest.
    "is into reading": "reads", "are into reading": "reads",
    "is really into reading": "reads", "are really into reading": "reads",
    "knows": "knows", "know": "knows",
}

# Negative verb forms are recorded as the affirmative predicate plus negative
# polarity, so a later contradiction check compares like with like.
_NEGATED_PREDICATES = {
    "does not eat": ("eats", 0.85), "doesn't eat": ("eats", 0.85),
    "is not allergic to": ("allergic_to", 0.85),
    "are not allergic to": ("allergic_to", 0.85),
    "does not drink": ("drinks", 0.85), "doesn't drink": ("drinks", 0.85),
}

# A time adverb may sit between the subject and its verb ("I now live in
# Paris", "Vera currently works at ...").  Among them "no longer" is also a
# negation, which the polarity scan picks up separately, so the predicate
# itself stays affirmative.
_VERB_ADVERB = (
    r"(?:(?:now|currently|recently|previously|formerly|still|also|always|"
    r"really|just|often|usually|sometimes|mostly|definitely|"
    r"certainly|never|no longer|used to)\s+)?")

# One verb table is shared by clauses with a named subject and clauses with a
# possessive relation subject ("my mother studies botany"), so the same
# sentence shape yields the same predicate whichever subject it names.
_THIRD_PARTY_VERBS = (
    r"is not allergic to|are not allergic to|does not eat|doesn't eat|"
    r"does not drink|doesn't drink|"
    r"has got|is (?:fluent in|fluent|studying|researching)|"
    r"are (?:fluent in|fluent|studying|researching)|"
    r"instructs? in|travels? by|enjoys? reading|volunteers? with|"
    r"prefers? to drink|likes? to drink|loves? to drink|"
    r"has an allergy to|pays? rent of|"
    r"is (?:really\s+)?into reading|are (?:really\s+)?into reading|"
    r"does work as|do work as|works? over at|"
    r"lives? in|resides? in|works? at|works? for|works? in|stud(?:y|ies) at|"
    r"is employed by|are employed by|is from|comes from|likes|loves|has|"
    r"drinks?|eats?|reads?|plays?|speaks?|knows?|teach(?:es)?|owns?|"
    r"stud(?:y|ies)|commutes? by|volunteers? at|w(?:as|ere) born in|"
    r"is allergic to|are allergic to|is aged|are aged|earns?"
)

_ENGLISH_RULES: Tuple[Tuple[str, "re.Pattern[str]", str, str, str, float], ...] = (
    ("relation_name",
     re.compile(rf"^(?:my|our)\s+({_RELATION})(?:'s|s')?\s+name\s+(?:is|was)"
                r"\s+(?:called\s+|named\s+)?(.+)$", re.IGNORECASE),
     "name", "derived", "person", 0.9),
    ("relation_called",
     re.compile(rf"^(?:my|our)\s+({_RELATION})\s+(?:is|was)\s+"
                r"(?:called|named)\s+(.+)$", re.IGNORECASE),
     "name", "derived", "person", 0.88),
    ("relation_attribute",
     re.compile(rf"^(?:my|our)\s+({_RELATION})\s+{_VERB_ADVERB}"
                rf"(?P<verb>{_THIRD_PARTY_VERBS})\s+(.+)$",
                re.IGNORECASE),
     None, "derived", None, 0.82),
    ("speaker_name",
     re.compile(r"^my name is\s+(.+)$", re.IGNORECASE),
     "name", "speaker", "person", 0.9),
    ("speaker_attribute",
     re.compile(rf"^my ({_ATTR}|phone number|email address|e-mail|birthday|"
                r"hobby|job|age)\s+(?:is|are)\s+(.+)$", re.IGNORECASE),
     None, "speaker", "unknown", 0.85),
    ("speaker_lives",
     re.compile(rf"^i\s+{_VERB_ADVERB}(?:live|lives|am living|'m living|"
                r"stay|am staying)\s+in\s+(.+)$", re.IGNORECASE),
     "lives_in", "speaker", "place", 0.87),
    ("speaker_moved",
     re.compile(r"^i\s+(?:moved to|relocated to)\s+(.+)$", re.IGNORECASE),
     "lives_in", "speaker", "place", 0.82),
    ("speaker_from",
     re.compile(r"^i(?:'m| am)\s+(?:from|originally from)\s+(.+)$",
                re.IGNORECASE),
     "is_from", "speaker", "place", 0.85),
    ("speaker_born",
     re.compile(r"^i\s+was born\s+in\s+(.+)$", re.IGNORECASE),
     "born_in", "speaker", "place", 0.85),
    ("speaker_used_live",
     re.compile(r"^i\s+used to live\s+in\s+(.+)$", re.IGNORECASE),
     "lives_in", "speaker", "place", 0.8),
    ("speaker_used_work",
     re.compile(r"^i\s+used to work\s+(at|for|in)\s+(.+)$", re.IGNORECASE),
     None, "speaker", "organization", 0.8),
    ("speaker_work",
     re.compile(rf"^i\s+{_VERB_ADVERB}(?:work|worked|am working)\s+(at|for|in)"
                r"\s+(.+)$", re.IGNORECASE),
     None, "speaker", "organization", 0.85),
    ("speaker_work_as",
     re.compile(rf"^i\s+{_VERB_ADVERB}(?:work|worked|am working)\s+as\s+"
                r"(?:a\s+|an\s+)?(.+)$", re.IGNORECASE),
     "works_as", "speaker", None, 0.84),
    ("speaker_age",
     re.compile(r"^i(?:'m| am| was)\s+(\d{1,3})(?:\s+years?\s+old)?$",
                re.IGNORECASE),
     "age", "speaker", "number", 0.9),
    ("speaker_is",
     re.compile(r"^i(?:'m| am)\s+(?:a|an)\s+(.+)$", re.IGNORECASE),
     "is_a", "speaker", None, 0.72),
    ("speaker_has",
     re.compile(r"^i\s+(?:have|own)\s+(.+)$", re.IGNORECASE),
     "has", "speaker", None, 0.78),
    ("speaker_likes",
     re.compile(r"^i\s+(?:like|love|enjoy|prefer|am into)\s+(.+)$",
                re.IGNORECASE),
     "likes", "speaker", None, 0.8),
    ("speaker_dislikes",
     re.compile(r"^i\s+(?:hate|dislike|can't stand|cannot stand)\s+(.+)$",
                re.IGNORECASE),
     "dislikes", "speaker", None, 0.8),
    ("speaker_not_like",
     re.compile(r"^i\s+(?:do not|don't|did not)\s+(?:like|love|enjoy)\s+(.+)$",
                re.IGNORECASE),
     "dislikes", "speaker", None, 0.75),
    ("team_uses",
     re.compile(r"^(?:we|our team)\s+(?:use|uses|are using|switched to|"
                r"have switched to)\s+(.+)$", re.IGNORECASE),
     "uses", "derived", None, 0.78),
    ("speaker_uses",
     re.compile(rf"^i\s+{_VERB_ADVERB}(?:use|used|am using)\s+(.+)$",
                re.IGNORECASE),
     "uses", "speaker", "product", 0.8),
    ("speaker_studies",
     re.compile(rf"^i\s+{_VERB_ADVERB}(?:study|studied|am studying)\s+(.+)$",
                re.IGNORECASE),
     "studies", "speaker", None, 0.8),
    ("speaker_speaks",
     re.compile(rf"^i\s+{_VERB_ADVERB}(?:speak|know)\s+(.+)$",
                re.IGNORECASE),
     "speaks", "speaker", "language", 0.82),
    ("speaker_learning",
     re.compile(r"^i(?:'m| am)\s+(?:learning|studying|practicing)\s+(.+)$",
                re.IGNORECASE),
     "learning", "speaker", "language", 0.78),
    # The third-party rules cover these predicates for named and related
    # subjects; first-person speech is the common case in a conversation and
    # needs the same coverage.  Negation stays a polarity on the affirmative
    # predicate, exactly as for third-party clauses.
    ("speaker_allergic",
     re.compile(rf"^i(?:'m| am)\s+{_VERB_ADVERB}(?:not\s+)?allergic\s+to\s+"
                r"(.+)$", re.IGNORECASE),
     "allergic_to", "speaker", None, 0.85),
    ("speaker_drinks",
     re.compile(rf"^i\s+{_VERB_ADVERB}(?:(?:do not|did not|never)\s+)?"
                r"(?:drink|drank)\s+(.+)$", re.IGNORECASE),
     "drinks", "speaker", None, 0.8),
    ("speaker_eats",
     re.compile(rf"^i\s+{_VERB_ADVERB}(?:(?:do not|did not|never)\s+)?"
                r"(?:eat|ate)\s+(.+)$", re.IGNORECASE),
     "eats", "speaker", None, 0.8),
    ("speaker_plays",
     re.compile(rf"^i\s+{_VERB_ADVERB}(?:play|played)\s+(.+)$",
                re.IGNORECASE),
     "plays", "speaker", None, 0.78),
    ("speaker_reads",
     re.compile(rf"^i\s+{_VERB_ADVERB}(?:read|am reading)\s+(.+)$",
                re.IGNORECASE),
     "reads", "speaker", None, 0.76),
    ("speaker_teaches",
     re.compile(rf"^i\s+{_VERB_ADVERB}(?:teach|taught)\s+(.+)$",
                re.IGNORECASE),
     "teaches", "speaker", None, 0.8),
    ("speaker_earns",
     re.compile(rf"^i\s+{_VERB_ADVERB}earn\s+(.+)$", re.IGNORECASE),
     "salary", "speaker", "number", 0.8),
    ("speaker_fluent",
     re.compile(rf"^i(?:'m| am)\s+{_VERB_ADVERB}fluent\s+in\s+(.+)$",
                re.IGNORECASE),
     "speaks", "speaker", "language", 0.82),
    ("speaker_commutes",
     re.compile(rf"^i\s+{_VERB_ADVERB}commute\s+(?:by|on)\s+(.+)$",
                re.IGNORECASE),
     "commutes_by", "speaker", None, 0.78),
    ("speaker_volunteers",
     re.compile(rf"^i\s+{_VERB_ADVERB}volunteer\s+(?:at|with)\s+(.+)$",
                re.IGNORECASE),
     "volunteers_at", "speaker", None, 0.78),
)

_NAME_TOKEN = r"[^\W\d_][\w'’.-]*"
# Names are matched case-insensitively because chat text is often lowercased,
# but a clause may only open with a name if the first token is not a function
# word, pronoun or discourse marker — those belong to other rules or to none.
_NAME_STOP = frozenset("""
i we my our me us he she they it you your his her their its the a an this
that these those there here today yesterday tomorrow now then also actually
so but and or if when while because since someone anyone everyone nobody
something anything everything nothing was were is are has have had does do
did
""".split())
# Heads that belong to the speaker, used to stop the third-party rules from
# inventing a person out of a first-person clause they could not parse.
_FIRST_PERSON_HEADS = frozenset("""
i i'm im i've i'd me my mine we we're we've our ours us
""".split())
# Pronouns are resolved (or left unresolved) by the pronoun rule; the
# third-party rules must not turn "He did not" into a person's name when an
# auxiliary blocks the pronoun rule.
_NON_NAME_HEADS = _FIRST_PERSON_HEADS | frozenset("""
he him his she her hers it its they them their theirs you your yours
""".split())
# The extra name tokens may not be an adverb that separates the name from its
# verb: "Vera now works at ..." names Vera, not "Vera now".
_NAME_ADVERB = (
    r"now|currently|recently|previously|formerly|still|also|no|used|"
    r"really|just|often|always|usually|sometimes|then|mostly|"
    r"definitely|certainly")
_NAME = (
    rf"(?P<name>{_NAME_TOKEN}"
    rf"(?:\s+(?!(?:{_NAME_ADVERB})\b){_NAME_TOKEN}){{0,2}})")

# A possessive or relation noun may be misspelled in chat ("myy brther").
# Recovery is gated on closed lists and a one-edit bound, exactly like the
# Stage 1 structural vocabulary: a candidate has to be out of ordinary
# vocabulary, keep the first letter, stay at least four letters long and match
# exactly one relation noun.  No open-vocabulary word can become a relation.
_POSSESSIVE_NOUNS = ("my", "our")
_POSSESSIVE_SUFFIX = re.compile(r"['’]s$")
_NOISY_RELATION_EDITS = 1
_NOISY_RELATION_MIN_LENGTH = 4


def _noisy_possessive(token: str) -> Optional[str]:
    """The possessive a chat typo stands for, or None.

    Only two shapes are eligible: a same-length token one edit (including an
    adjacent transposition) away from "my"/"our", or the doubling typo the
    conversation actually produces ("myy", "ouur").  A token that is ordinary
    vocabulary ("may", "by", "or") is never read as a possessive, so a name
    like "May" keeps its reading.
    """
    folded = token.casefold()
    if folded in _POSSESSIVE_NOUNS:
        return folded
    if len(folded) < len(_POSSESSIVE_NOUNS[0]):
        return None
    if folded in TYPO_VOCABULARY or folded in CONTENT_WORDS:
        return None
    for target in _POSSESSIVE_NOUNS:
        if len(folded) == len(target) and edit_distance(
                folded, target, _NOISY_RELATION_EDITS) <= _NOISY_RELATION_EDITS:
            return target
        if len(folded) == len(target) + 1:
            for index in range(len(folded) - 1):
                if folded[index] == folded[index + 1] and \
                        folded[:index] + folded[index + 1:] == target:
                    return target
    return None


def _noisy_relation_noun(token: str) -> Tuple[Optional[str], bool]:
    """The relation noun a chat typo stands for, and whether it is ambiguous.

    Exact relation nouns are returned as themselves.  A misspelling is only
    recovered from the closed ``RELATION_NOUNS`` list under the one-edit bound
    with the first letter kept.  Two fits ("mather" is one edit from both
    mother and father) abstain instead of choosing.
    """
    folded = _POSSESSIVE_SUFFIX.sub(
        "", token.casefold().strip(_TRAILING_PUNCT))
    if not folded:
        return None, False
    if folded in RELATION_NOUNS:
        return folded, False
    if len(folded) < _NOISY_RELATION_MIN_LENGTH:
        return None, False
    if folded in TYPO_VOCABULARY or folded in CONTENT_WORDS:
        return None, False
    matches = [
        noun for noun in RELATION_NOUNS
        if "_" not in noun and len(noun) >= _NOISY_RELATION_MIN_LENGTH
        and noun[0] == folded[0]
        and edit_distance(folded, noun, _NOISY_RELATION_EDITS)
        <= _NOISY_RELATION_EDITS]
    if len(matches) == 1:
        return matches[0], False
    return None, len(matches) > 1


def _looks_like_name(name: str, sentence_initial: bool) -> bool:
    """Tells a person's name apart from a capitalised prose fragment.

    Free-form chat is full of clauses that begin with a word that happens to
    precede a third-person verb — "See your Google plays ..." or "Interleaved
    ABBA studies ..." — and treating those as a subject invents a person.  A
    candidate is a name when it contains no function word, is cased
    consistently, and, when written in lower case, opens the message and is not
    made entirely of ordinary English words.  Anything else keeps its fact but
    loses the named subject.

    A token one typo away from a possessive is not a name either: "myy brther"
    is a misspelled relation phrase, and reading it as a person would fabricate
    an entity the speaker never named.
    """
    tokens = [token for token in name.split() if token]
    if not tokens:
        return False
    folded = [token.strip("'’.-").casefold() for token in tokens]
    if any(not token for token in folded):
        return False
    if any(token in _NAME_STOP or token in STOPWORDS for token in folded):
        return False
    if _noisy_possessive(tokens[0]) is not None:
        return False
    capitalised = [token[0].isupper() for token in tokens]
    if all(capitalised):
        # Consistent title case, but an all-caps token is an acronym.
        return not any(token.isupper() and len(token) > 1 for token in tokens)
    if any(capitalised):
        # "Please create detailed" mixes case inside the candidate.
        return False
    # All lower case: chat text often is, so a lowercase candidate is accepted
    # only when it opens the message and none of its words is ordinary
    # vocabulary ("result rendering never" is prose).
    if not sentence_initial:
        return False
    return not any(token in TYPO_VOCABULARY or token in CONTENT_WORDS
                   for token in folded)

_THIRD_PARTY = re.compile(
    rf"^{_NAME}\s+{_VERB_ADVERB}(?P<verb>{_THIRD_PARTY_VERBS})"
    r"\s+(?P<value>.+)$", re.IGNORECASE)

_THIRD_PARTY_ATTRIBUTE = re.compile(
    rf"^{_NAME}\s+(?P<attr>{_ATTR})\s+(?:is|are|was|were)\s+"
    r"(?P<value>.+)$", re.IGNORECASE)

# "Zephyr Harlow is 76." states an age with no unit word; the copula plus a
# number and nothing else is the only reading that leaves the number a fact.
_THIRD_PARTY_AGE = re.compile(
    rf"^{_NAME}\s+(?:is|was|are|were)\s+"
    r"(?P<value>\d{1,3})(?:\s+years?\s+old)?$", re.IGNORECASE)

_PRONOUN = re.compile(
    r"^(?P<pronoun>he|she|they)\s+" + _VERB_ADVERB +
    rf"(?P<verb>{_THIRD_PARTY_VERBS})\s+(?P<value>.+)$",
    re.IGNORECASE)

# The remainder of a misspelled possessive clause reuses the verb table, so
# "myy brther lives in Oslo" yields the same predicate as "my brother lives in
# Oslo".  The naming shapes are matched too because "myy brther's name is Sam"
# is how a relation gets linked to a person in chat.
_RELATION_REST = re.compile(
    rf"^{_VERB_ADVERB}(?P<verb>{_THIRD_PARTY_VERBS})\s+(?P<value>.+)$",
    re.IGNORECASE)
_RELATION_NAME_REST = re.compile(
    r"^name\s+(?:is|was)\s+(?:called\s+|named\s+)?(?P<value>.+)$",
    re.IGNORECASE)
_RELATION_CALLED_REST = re.compile(
    r"^(?:is|was)\s+(?:called|named)\s+(?P<value>.+)$", re.IGNORECASE)


def _head_words(clause: str, count: int) -> Optional[Tuple[List[str], int]]:
    """Splits off the first ``count`` whitespace-delimited words.

    Returns the words and the offset where the remainder starts, so a fact
    value keeps its span in the message even though the candidate tokens were
    recovered from a misspelling.
    """
    words: List[str] = []
    position = 0
    for _ in range(count):
        while position < len(clause) and clause[position].isspace():
            position += 1
        start = position
        while position < len(clause) and not clause[position].isspace():
            position += 1
        if start == position:
            return None
        words.append(clause[start:position])
    rest_start = position
    while rest_start < len(clause) and clause[rest_start].isspace():
        rest_start += 1
    return words, rest_start


def _noisy_relation(clause: str, offset: int,
                    speaker: str) -> List[_Match]:
    """Reads a possessive relation whose possessive or noun is misspelled.

    The evidence is spelling, not position: the first token has to stand one
    typo away from "my"/"our", the second has to resolve to exactly one
    relation noun from the closed list, and at least one of the two has to be
    the misspelled form.  The subject is the user's relative at reduced
    confidence, never a named person; an ambiguous relation noun yields no
    candidate here, so the phrase is left to the unresolved-subject path
    instead of becoming a fabricated entity.
    """
    head = _head_words(clause, 2)
    if head is None:
        return []
    (possessive_token, relation_token), rest_start = head
    possessive = _noisy_possessive(possessive_token)
    if possessive is None:
        return []
    relation, ambiguous = _noisy_relation_noun(relation_token)
    if relation is None or ambiguous:
        return []
    exact_possessive = possessive_token.casefold() == possessive
    exact_relation = _POSSESSIVE_SUFFIX.sub(
        "", relation_token.casefold()) == relation
    if exact_possessive and exact_relation:
        # The ordinary rules already tried this spelling.
        return []
    rest = clause[rest_start:]
    if not rest:
        return []
    subject = replace(
        _subject_for_relation(relation, speaker),
        confidence=0.74, resolved_from="possessive_near_match")
    named = _RELATION_NAME_REST.match(rest)
    called = None if named else _RELATION_CALLED_REST.match(rest)
    predicate_rule = None if (named or called) else _RELATION_REST.match(rest)
    if not (named or called or predicate_rule):
        return []
    if named or called:
        match = named or called
        value, start, end = _trim(
            match.group("value"), offset + rest_start + match.start("value"))
        return [_Match(
            rule="relation_name_near_match", predicate="name", value=value,
            value_span=(start, end), subject=subject, entity_type="person",
            confidence=0.78)]
    verb = predicate_rule.group("verb").casefold()
    if verb in _NEGATED_PREDICATES:
        predicate = _NEGATED_PREDICATES[verb][0]
    else:
        predicate = _ENGLISH_PREDICATES.get(
            verb, _canonical_predicate(verb))
    value, start, end = _trim(
        predicate_rule.group("value"),
        offset + rest_start + predicate_rule.start("value"))
    if predicate in ("lives_in", "is_from", "born_in"):
        value = _canonical_place(value)
    return [_Match(
        rule="relation_attribute_near_match", predicate=predicate, value=value,
        value_span=(start, end), subject=subject, entity_type="",
        confidence=0.74)]

_SPANISH_RELATION = re.compile(
    r"^mi (?P<relation>hermana|hermano|madre|padre|mamá|papá|esposa|esposo|"
    r"hija|hijo|amiga|amigo|prima|primo) se llama (?P<value>.+)$",
    re.IGNORECASE)
_FRENCH_RELATION = re.compile(
    r"^(?:ma|mon)\s+(?P<relation>sœur|soeur|frère|frere|mère|mere|père|pere|"
    r"femme|mari|fille|fils|amie|ami|cousine|cousin)\s+s['’]appelle\s+"
    r"(?P<value>.+)$", re.IGNORECASE)

# A possessive relation can head an ordinary predicate too ("Mi madre vive en
# Lima"), and the verb plus preposition decide which one.  The subject stays
# derived, never the speaker, exactly as with the relation-name rules.
_SPANISH_RELATION_PREDICATE = re.compile(
    r"^mi\s+(?P<relation>hermana|hermano|madre|padre|mamá|papá|esposa|esposo|"
    r"hija|hijo|amiga|amigo|prima|primo)\s+(?P<verb>vive|trabaja)\s+"
    r"(?P<prep>en|para)\s+(?P<value>.+)$", re.IGNORECASE)
_FRENCH_RELATION_PREDICATE = re.compile(
    r"^(?:ma|mon)\s+(?P<relation>sœur|soeur|frère|frere|mère|mere|père|pere|"
    r"femme|mari|fille|fils|amie|ami|cousine|cousin)\s+"
    r"(?P<verb>habite|travaille)\s+(?P<prep>à|a|chez)\s+(?P<value>.+)$",
    re.IGNORECASE)

# The canonical relation is the English name, so the person's gender and the
# relation link behave the same whichever language introduced them.
_RELATION_CANON = {
    "hermana": "sister", "hermano": "brother", "madre": "mother",
    "padre": "father", "mamá": "mother", "mama": "mother",
    "papá": "father", "papa": "father", "esposa": "wife",
    "esposo": "husband", "hija": "daughter", "hijo": "son",
    "amiga": "friend", "amigo": "friend", "prima": "cousin",
    "primo": "cousin",
    "sœur": "sister", "soeur": "sister", "frère": "brother",
    "frere": "brother", "mère": "mother", "mere": "mother",
    "père": "father", "pere": "father", "femme": "wife",
    "mari": "husband", "fille": "daughter", "fils": "son",
    "amie": "friend", "ami": "friend", "cousine": "cousin",
    "cousin": "cousin",
}

_MULTILINGUAL_RULES = (
    ("es_name", re.compile(r"^me llamo\s+(.+)$", re.IGNORECASE),
     "name", "person", 0.88),
    ("es_lives", re.compile(r"^(?:yo )?vivo en\s+(.+)$", re.IGNORECASE),
     "lives_in", "place", 0.84),
    ("es_works", re.compile(r"^(?:yo )?trabajo en\s+(.+)$", re.IGNORECASE),
     "works_in", "organization", 0.82),
    ("es_works_at", re.compile(r"^(?:yo )?trabajo para\s+(.+)$",
                               re.IGNORECASE),
     "works_at", "organization", 0.82),
    ("es_age", re.compile(r"^tengo\s+(\d{1,3})\s+años$", re.IGNORECASE),
     "age", "number", 0.88),
    ("fr_name", re.compile(r"^je m'appelle\s+(.+)$", re.IGNORECASE),
     "name", "person", 0.88),
    ("fr_lives", re.compile(r"^j'habite\s+(?:à|a)\s+(.+)$", re.IGNORECASE),
     "lives_in", "place", 0.84),
    # "chez" names the employer while "à" names a place, the same distinction
    # the third-person French rule makes.
    ("fr_works_at", re.compile(r"^je travaille\s+chez\s+(.+)$",
                               re.IGNORECASE),
     "works_at", "organization", 0.84),
    ("fr_works", re.compile(r"^je travaille\s+(?:à|a)\s+(.+)$",
                            re.IGNORECASE),
     "works_in", "organization", 0.82),
)

# A named third party or a pronoun can head the same predicate in Spanish and
# French.  Only the capitalisation of the name (or the closed pronoun list)
# selects these rules, so a lowercased noun is not turned into a person.
_ES_NAMED = re.compile(
    r"^(?P<subject>[A-ZÁÉÍÓÚÜÑ][\w'’.-]*(?:\s+[A-ZÁÉÍÓÚÜÑ][\w'’.-]*){0,2})\s+"
    r"(?P<verb>vive|viven|trabaja|trabajan)\s+(?P<prep>en|para)\s+"
    r"(?P<value>.+)$")
_ES_PRONOUN = re.compile(
    r"^(?P<pronoun>él|ella|ellos|ellas)\s+"
    r"(?P<verb>vive|viven|trabaja|trabajan)\s+(?P<prep>en|para)\s+"
    r"(?P<value>.+)$", re.IGNORECASE)
_FR_NAMED = re.compile(
    r"^(?P<subject>[A-ZÀÂÇÉÈÊËÎÏÔÙÛÜŸ][\w'’.-]*"
    r"(?:\s+[A-ZÀÂÇÉÈÊËÎÏÔÙÛÜŸ][\w'’.-]*){0,2})\s+"
    r"(?P<verb>habite|habitent|travaille|travaillent)\s+"
    r"(?P<prep>à|a|chez)\s+(?P<value>.+)$")
_FR_PRONOUN = re.compile(
    r"^(?P<pronoun>il|elle|ils|elles)\s+"
    r"(?P<verb>habite|habitent|travaille|travaillent)\s+"
    r"(?P<prep>à|a|chez)\s+(?P<value>.+)$", re.IGNORECASE)

# Amharic puts the predicate at the end of the clause, so each rule anchors on
# the verb and takes what sits between the subject and it.  Only constructions
# whose function is carried by an unambiguous suffix are listed: the person is
# named by the possessive suffix (ወንድሜ "my brother"), and the verb form names
# the predicate.  Any other shape produces no fact and the clause abstains.
_AM_RELATIONS = {
    "ወንድሜ": "brother", "እህቴ": "sister", "እናቴ": "mother",
    "አባቴ": "father", "ሚስቴ": "wife", "ባለቤቴ": "spouse", "ልጄ": "child",
}
_AM_RELATION_PATTERN = "|".join(
    sorted(_AM_RELATIONS, key=len, reverse=True))
# ነዉ is the common informal spelling of the copula ነው; the variants are
# explicit, not guessed.
_AM_NAMED = r"(?:ይባላል|ትባላለች|ይባላሉ|ነው|ነዉ|ናት)"
_AM_RELATION_NAME = re.compile(
    rf"^(?P<relation>{_AM_RELATION_PATTERN})\s+(?P<value>.+?)\s+{_AM_NAMED}$")
_AM_RELATION_LIVES = re.compile(
    rf"^(?P<relation>{_AM_RELATION_PATTERN})\s+በ(?P<value>.+?)\s+"
    r"(?:ውስጥ\s+)?(?:ይኖራል|ትኖራለች|ይኖራሉ)$")
_AM_SPEAKER_RULES = (
    (re.compile(r"^ስሜ\s+(?P<value>.+?)(?:\s+(?:ነው|ነዉ|ናት|ይባላል))?$"),
     "am_speaker_name", "name", "person", 0.86),
    (re.compile(r"^በ(?P<value>.+?)\s+(?:ውስጥ\s+)?እኖራለሁ$"),
     "am_speaker_lives", "lives_in", "place", 0.84),
    (re.compile(r"^በ(?P<value>.+?)\s+(?:ውስጥ\s+)?እሰራለሁ$"),
     "am_speaker_works", "works_in", "organization", 0.82),
    (re.compile(r"^(?P<value>.+?)\s+እናገራለሁ$"),
     "am_speaker_speaks", "speaks", "language", 0.84),
    (re.compile(r"^(?P<value>.+?)\s+(?:እወዳለሁ|ወድጃለሁ)$"),
     "am_speaker_likes", "likes", "", 0.8),
    (re.compile(r"^ዕድሜዬ\s+(?P<value>\d{1,3})\s*ዓመት\s+ነው$"),
     "am_speaker_age", "age", "number", 0.86),
)
_AM_PRONOUN_LIVES = re.compile(
    r"^(?P<pronoun>እሱ|እሷ|እነሱ)\s+በ(?P<value>.+?)\s+"
    r"(?:ውስጥ\s+)?(?:ይኖራል|ትኖራለች|ይኖራሉ)$")

_ELIDED = re.compile(
    r"^(?:also\s+|then\s+|now\s+)?"
    r"(?P<verb>works? at|works? for|works? in|lives? in|stud(?:y|ies) at|"
    r"likes|loves|has|speaks?|drinks?|eats?|plays?|reads?|teaches?|"
    r"stud(?:y|ies)|owns?|uses?|needs?)\s+"
    # A coordinated clause can be followed by a state ("and work is busy"),
    # which is not an object of the verb; only an object continues the fact.
    r"(?!(?:is|are|was|were|be|been|being|do|does|did|can|could|will|would|"
    r"should|may|might|must|not)\b)"
    r"(?P<value>.+)$", re.IGNORECASE)

_TEMPORAL_MARKERS = (
    ("past", TEMPORAL_PAST, 0.85),
    ("now", TEMPORAL_NOW, 0.8),
    ("future", TEMPORAL_FUTURE, 0.75),
)
_TEMPORAL_PATTERNS = tuple(
    (kind, re.compile(
        "|".join(rf"\b{re.escape(marker)}\b" for marker in markers),
        re.IGNORECASE), confidence)
    for kind, markers, confidence in _TEMPORAL_MARKERS)
_NEGATION_PATTERN = re.compile(
    "|".join(rf"\b{re.escape(word)}\b" for word in NEGATIONS),
    re.IGNORECASE)
_HYPOTHETICAL_PATTERN = re.compile(
    r"\b(?:if|would|could|might|maybe|perhaps|suppose|imagine)\b"
    # Amharic hedges: ምናልባት "maybe", ከሆነ "if it is".
    r"|ምናልባት|ከሆነ",
    re.IGNORECASE)

# Negated auxiliaries block every rule anchored on the affirmative verb
# ("does not live in", "do not have"), so the fact was dropped instead of
# being stored with negative polarity.  The auxiliary is blanked at its
# exact length so value spans still address the normalized text; the
# original clause keeps the negation for the polarity scan.  Only verbs the
# English rules already accept in base form are listed: "have" needs the
# third-person form after "does", and "like"/"love" are deliberately absent
# so first-person "do not like" keeps its tested dislikes reading.
_NEGATABLE_VERBS = (
    "have", "own", "live", "work", "study", "speak", "read", "play",
    "teach", "eat", "drink", "commute", "volunteer",
)
_NEGATED_AUX = re.compile(
    r"\b(?P<aux>do|does)\s+not\s+(?P<verb>"
    + "|".join(_NEGATABLE_VERBS) + r")\b", re.IGNORECASE)
_HAVE_NOT_GOT = re.compile(
    r"\b(?:have|has)\s+(?P<tag>not\s+got)\b", re.IGNORECASE)
_HAVE_NOT_DETERMINER = re.compile(
    r"\b(?:have|has)\s+(?P<neg>not)(?=\s+(?:a|an|the|any|one|two)\b)",
    re.IGNORECASE)
_POSSESS_NO = re.compile(
    r"\b(?:have|has|own|owns)\s+(?P<neg>no)\b", re.IGNORECASE)
_POSSESS_ANY = re.compile(
    r"\b(?:have|has|own|owns)\s+(?P<neg>any)\b", re.IGNORECASE)


def _clause_negated(clause: str) -> bool:
    """True when the clause denies the fact, including "have no <object>"."""
    return bool(_NEGATION_PATTERN.search(clause)) or \
        bool(_POSSESS_NO.search(clause))


def _denegated_clause(clause: str) -> str:
    """Blanks a negated auxiliary so the affirmative rule can parse it.

    Length is preserved exactly, so every match offset in the returned string
    is also a valid offset in ``clause``.  The caller still reads polarity
    from the original clause, so the fact is recorded as negative and the
    value never contains the negation word.
    """
    chars = list(clause)

    def blank(start: int, end: int) -> None:
        for position in range(start, end):
            chars[position] = " "

    for found in _NEGATED_AUX.finditer(clause):
        blank(found.start(), found.start("verb"))
        if found.group("aux").casefold() == "does" and \
                found.group("verb").casefold() == "have":
            start, end = found.span("verb")
            chars[start:end] = list("has ")
    for found in _HAVE_NOT_GOT.finditer(clause):
        blank(found.start("tag"), found.end("tag"))
    for found in _HAVE_NOT_DETERMINER.finditer(clause):
        blank(found.start("neg"), found.end("neg"))
    for found in _POSSESS_NO.finditer(clause):
        blank(found.start("neg"), found.end("neg"))
    # "haven't got any car" only becomes "any car" adjacent to "have" once the
    # auxiliary is blanked, so this pass reads the partially rewritten text.
    if _clause_negated(clause):
        for found in _POSSESS_ANY.finditer("".join(chars)):
            blank(found.start("neg"), found.end("neg"))
    return "".join(chars)



_ROLE_AT = re.compile(
    r"^i\s+(?:(?:work|worked|am working)\s+as|(?:am|'m))\s+"
    r"(?:a\s+|an\s+)?(?P<role>.+?)\s+(?:at|for)\s+(?P<org>.+)$",
    re.IGNORECASE)
_ROLE_ORG_CONTEXT = frozenset({
    "work", "home", "the office", "office", "school", "college",
    "university", "the moment", "night", "least",
})


@dataclass(frozen=True)
class Interpretation:
    entities: Tuple[Entity, ...]
    facts: Tuple[FactCandidate, ...]
    ambiguities: Tuple[str, ...] = ()
    abstentions: Tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class _Match:
    rule: str
    predicate: str
    value: str
    value_span: Tuple[int, int]
    subject: SubjectRef
    entity_type: str
    confidence: float


# Real chat starts clauses with fillers: "But i live in ethiopia", "no I was
# 54", "Also I work as a broker".  Without peeling them the first-person
# pronoun is read as a name ("But i"), or the clause is not recognised at all.
# Only markers directly followed by a first-person or possessive token are
# removed, so a marker that opens a third-party clause is left intact.
_DISCOURSE_PREFIX = re.compile(
    r"^(?:but|and|so|then|also|no|yeah|yes|well|oh|okay|ok|like|now|"
    r"basically|actually|honestly|anyway|btw)\b\s*[,:;-]?\s*",
    re.IGNORECASE)
_DISCOURSE_TAIL = re.compile(
    r"^(?:i|i'm|im|i've|i'd|me|my|we|we're|we've|our|us)\b",
    re.IGNORECASE)


def _strip_discourse(clause: str, offset: int) -> Tuple[str, int]:
    while True:
        marker = _DISCOURSE_PREFIX.match(clause)
        if marker is None:
            return clause, offset
        remainder = clause[marker.end():]
        if not _DISCOURSE_TAIL.match(remainder):
            return clause, offset
        clause = remainder
        offset += marker.end()


def _clauses(text: str) -> List[Tuple[int, int]]:
    """Splits on sentence punctuation, then on clause joiners."""
    sentences: List[Tuple[int, int]] = []
    start = 0
    for match in _CLAUSE_SPLIT.finditer(text):
        if match.start() > start:
            sentences.append((start, match.start()))
        start = match.end()
    if start < len(text):
        sentences.append((start, len(text)))

    result: List[Tuple[int, int]] = []
    for sentence_start, sentence_end in sentences:
        cursor = sentence_start
        for joiner in _JOINER.finditer(text, sentence_start, sentence_end):
            tail = text[joiner.end():joiner.end() + 16]
            if not (_JOIN_TAIL.match(tail)
                    or _CONTRAST_TAIL.match(tail)):
                continue
            if joiner.start() > cursor:
                result.append((cursor, joiner.start()))
            cursor = joiner.end()
        if cursor < sentence_end:
            result.append((cursor, sentence_end))
    return [(left, right) for left, right in result
            if text[left:right].strip()]


def _language_at(language: LanguageInfo, offset: int) -> str:
    for span in language.spans:
        if span.start <= offset < span.end:
            return span.language
    return language.primary


def _strip_new_clause(value: str) -> str:
    """Keeps the asserted prefix of a value written without punctuation."""
    while True:
        match = _TRAILING_NEW_CLAUSE.search(value)
        if match is None:
            return value
        before = value[:match.start()].rstrip(_TRAILING_PUNCT)
        last = before.rsplit(None, 1)[-1].casefold() if before else ""
        if not before or last.strip(_TRAILING_PUNCT) in _RELATIVE_GUARD:
            return value
        value = before


def _trim(raw: str, start: int) -> Tuple[str, int, int]:
    """Strips padding, trailing context phrases and a leading article."""
    value = raw.strip(_TRAILING_PUNCT)
    start += len(raw) - len(raw.lstrip(_TRAILING_PUNCT))
    end = start + len(value)
    # The trims can unlock each other ("supabase now, in amharic"), so they run
    # until none of them changes the value.  Each keeps its own minimum length
    # guard; a strip that would leave one character is not a strip.
    for _ in range(3):
        changed = False
        stripped = _TRAILING_TEMPORAL.sub("", value)
        if stripped != value:
            value = stripped.rstrip(_TRAILING_PUNCT)
            end = start + len(value)
            changed = True
        for pattern in (_TRAILING_CONTEXT, _TRAILING_LANGUAGE):
            stripped = pattern.sub("", value)
            if stripped != value and len(stripped) >= 2:
                value = stripped.rstrip(_TRAILING_PUNCT)
                end = start + len(value)
                changed = True
        if not changed:
            break
    stripped = _strip_new_clause(value)
    if stripped != value and len(stripped) >= 2:
        value = stripped
        end = start + len(value)
    while True:
        stripped = _TRAILING_DANGLING.sub("", value)
        if stripped == value:
            break
        value = stripped.rstrip(_TRAILING_PUNCT)
        end = start + len(value)
    article = _LEADING_ARTICLE.match(value)
    if article and len(value) > article.end() + 1:
        start += article.end()
        value = value[article.end():]
        end = start + len(value)
    return value, start, end


# A speaker may correct a name in the same breath: "my brother is called Sam,
# I mean Sam Waters".  The corrected name is the one the speaker supplies, so
# it is the only value recorded; the rejected spelling is dropped rather than
# stored as a second person.
_NAME_CORRECTION = re.compile(
    r"[,;]?\s+(?:i|we)\s+(?:mean|meant|said)\s+(?:to say\s+)?",
    re.IGNORECASE)
_NAME_REJECTED_TAIL = re.compile(r"[,;]?\s+(?:and\s+)?not\s+.+$",
                                 re.IGNORECASE)
_NAME_RULES = frozenset({
    "relation_name", "relation_called", "speaker_name",
    "relation_name_near_match",
    # A correction clause may follow a Spanish or French naming clause.
    "es_name", "fr_name", "es_relation_name",
})


def _looks_like_corrected_name(value: str) -> bool:
    """True when a correction fragment can be a person's name.

    "I mean the weather is cold" is a sentence, not a name; the function words
    give it away.  Only a short fragment with no function word is accepted.
    """
    tokens = [token.strip(_TRAILING_PUNCT) for token in value.split()]
    if not tokens or len(tokens) > 4 or any(not token for token in tokens):
        return False
    folded = [token.casefold() for token in tokens]
    return not any(token in _NAME_STOP or token in STOPWORDS
                   for token in folded)


def _name_correction(raw: str) -> Optional[Tuple[int, str]]:
    """Returns (offset of the corrected name in ``raw``, corrected text)."""
    marker = _NAME_CORRECTION.search(raw)
    if marker is None:
        return None
    tail = raw[marker.end():]
    rejected = _NAME_REJECTED_TAIL.search(tail)
    if rejected is not None:
        tail = tail[:rejected.start()]
    if not _looks_like_corrected_name(tail):
        return None
    return marker.end(), tail


# A correction often arrives as its own clause because the comma splits the
# sentence: "my brother is called Sam, I mean Sam Waters".
_CORRECTION_CLAUSE = re.compile(
    r"^(?:(?:sorry|no|oops|wait)[,:]?\s+)?(?:i|we)\s+(?:mean|meant|said|"
    r"call(?:ed)?)\s+(?:to say\s+)?(?P<value>.+)$", re.IGNORECASE)


def _apply_name_correction(facts: List[FactCandidate],
                           entities: List[Entity],
                           corrected: Tuple[str, int, int],
                           source: SourceMessage,
                           normalized: NormalizedText) -> bool:
    """Replaces the preceding name fact's value with the speaker's correction.

    The speaker said which spelling is right, so the rejected one is dropped
    instead of becoming a second identity.  Only the immediately preceding
    fact is eligible, and only when it is a name.
    """
    if not facts or facts[-1].predicate != "name" \
            or facts[-1].rule not in _NAME_RULES:
        return False
    value, start, end = corrected
    previous = facts[-1]
    rule = f"{previous.rule}_correction"
    source_start, source_end = normalized.span_to_source(start, end)
    evidence = Evidence(
        message_id=source.message_id,
        conversation_id=source.conversation_id,
        start=source_start, end=source_end,
        text=source.text[source_start:source_end], rule=rule)
    facts[-1] = replace(previous, value=value, value_span=(start, end),
                        evidence=(evidence,), rule=rule)
    if entities:
        entities[-1] = Entity(
            text=value, type=entities[-1].type, span=(start, end),
            confidence=previous.confidence, rule=rule)
    return True


def _corrected_name(found: "re.Match[str]", clause_start: int
                    ) -> Optional[Tuple[str, int, int]]:
    value, start, end = _trim(
        found.group("value"), clause_start + found.start("value"))
    rejected = _NAME_REJECTED_TAIL.search(value)
    if rejected is not None:
        value = value[:rejected.start()].rstrip(_TRAILING_PUNCT)
        end = start + len(value)
    if not _valid_value(value) or not _looks_like_corrected_name(value):
        return None
    return value, start, end


def _correction_match(corrected: Tuple[str, int, int]) -> "_Match":
    """A correction with no target in this message.

    The target is the name the speaker used just before, which only the
    conversation-scoped identity stage can see; Stage 2 hands it over with an
    unresolved subject rather than guessing.
    """
    value, start, end = corrected
    return _Match(
        rule="name_correction", predicate="name", value=value,
        value_span=(start, end),
        subject=SubjectRef(
            kind="unknown", label="correction", confidence=0.4,
            resolved_from="speaker_correction", status="ambiguous",
            ambiguity="correction_target_unknown"),
        entity_type="person", confidence=0.8)


def _valid_value(value: str) -> bool:
    if not value or len(value) < 2 or len(value) > 80:
        return False
    return any(char.isalnum() for char in value)


def _canonical_predicate(value: str) -> str:
    key = value.casefold().strip()
    if key in ATTRIBUTE_PREDICATES:
        return ATTRIBUTE_PREDICATES[key]
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


def _canonical_place(value: str) -> str:
    key = value.casefold().strip()
    if key in PLACE_ALIASES and key not in {"la", "us"}:
        return PLACE_ALIASES[key]
    return value


def _value_entity_type(predicate: str, value: str) -> str:
    if predicate in ("lives_in", "is_from", "born_in"):
        return "place"
    if predicate in ("works_at", "works_in", "studies_at", "attends"):
        return "organization"
    if predicate in ("speaks", "learning"):
        return "language"
    if predicate == "age" or value.isdigit():
        return "number"
    if _EMAIL_VALUE.match(value):
        return "email"
    if _URL_VALUE.match(value):
        return "url"
    if value.casefold() in _LANGUAGE_NAMES:
        return "language"
    return "unknown"


def _temporal(clause: str, offset: int) -> List[TemporalExpression]:
    expressions: List[TemporalExpression] = []
    for kind, pattern, confidence in _TEMPORAL_PATTERNS:
        found = pattern.search(clause)
        if found:
            expressions.append(TemporalExpression(
                text=found.group(0), kind=kind,
                span=(offset + found.start(), offset + found.end()),
                confidence=confidence, rule=f"{kind}_marker"))
    for match in _YEAR.finditer(clause):
        expressions.append(TemporalExpression(
            text=match.group(0), kind="absolute", year=int(match.group(0)),
            span=(offset + match.start(), offset + match.end()),
            confidence=0.9, rule="year"))
    return expressions


def _subject_for_relation(relation: str, speaker: str) -> SubjectRef:
    relation = _RELATION_CANON.get(
        relation.casefold(), relation.casefold())
    return SubjectRef(
        kind="derived", label=f"user's {relation}", confidence=0.85,
        owner=speaker, relation=relation, resolved_from="possessive")


def interpret(source: SourceMessage, normalized: NormalizedText,
              language: LanguageInfo,
              config: Optional[ResolutionConfig] = None) -> Interpretation:
    """Extracts candidate facts and entities from one message."""
    cfg = config or ResolutionConfig()
    if not cfg.extract_facts:
        return Interpretation(entities=(), facts=())

    text = normalized.text
    speaker = source.speaker or "user"
    facts: List[FactCandidate] = []
    entities: List[Entity] = []
    ambiguities: List[str] = []
    abstentions: List[str] = []
    seen_spans = set()
    last_subject: Optional[SubjectRef] = None

    for clause_start, clause_end in _clauses(text):
        clause = text[clause_start:clause_end]
        leading = len(clause) - len(clause.lstrip())
        clause_start += leading
        clause = clause.strip().rstrip()
        if not clause:
            continue
        clause, clause_start = _strip_discourse(clause, clause_start)
        if not clause:
            continue
        if _QUESTION_START.match(clause) or clause.endswith("?"):
            continue
        if _ETHIOPIC_LETTER.search(clause) and _AM_QUESTION.search(clause):
            continue
        if _HYPOTHETICAL_PATTERN.search(clause):
            abstentions.append(f"hypothetical_clause:{clause_start}")
            continue
        language_here = _language_at(language, clause_start)
        correction = (_CORRECTION_CLAUSE.match(clause)
                      if language_here in ("en", "und") else None)
        corrected = None
        if correction is not None and not _match_clause(
                correction.group("value"),
                clause_start + correction.start("value"), "en", speaker):
            corrected = _corrected_name(correction, clause_start)
            if corrected is not None and _apply_name_correction(
                    facts, entities, corrected, source, normalized):
                continue
        matches = _match_clause(
            _denegated_clause(clause) if language_here in ("en", "und")
            else clause, clause_start, language_here, speaker)
        if not matches and corrected is not None:
            matches = [_correction_match(corrected)]
        if not matches and last_subject is not None and language_here in (
                "en", "und"):
            matches = _match_elided(clause, clause_start, last_subject)
        if matches and matches[0].subject.status == "resolved":
            last_subject = matches[0].subject
        for match in matches:
            if match.value_span in seen_spans:
                continue
            seen_spans.add(match.value_span)
            if not _valid_value(match.value):
                ambiguities.append(f"empty_fact_value:{match.rule}")
                continue

            negated = _clause_negated(clause) and \
                match.rule not in ("speaker_dislikes", "speaker_not_like")
            fact_entity = Entity(
                text=match.value,
                type=match.entity_type or _value_entity_type(
                    match.predicate, match.value),
                span=match.value_span,
                confidence=match.confidence,
                rule=match.rule)
            source_start, source_end = normalized.span_to_source(
                *match.value_span)
            evidence = Evidence(
                message_id=source.message_id,
                conversation_id=source.conversation_id,
                start=source_start, end=source_end,
                text=source.text[source_start:source_end], rule=match.rule)
            fact_id = f"{source.message_id}:fact:{len(facts)}"
            if match.subject.status != "resolved":
                abstentions.append(f"unresolved_subject:{fact_id}")
            if negated:
                abstentions.append(f"negated_clause:{fact_id}")
            facts.append(FactCandidate(
                fact_id=fact_id,
                subject=match.subject,
                predicate=match.predicate,
                value=match.value,
                value_span=match.value_span,
                confidence=round(
                    match.confidence * (0.85 if negated else 1.0), 4),
                evidence=(evidence,),
                entities=(fact_entity,),
                temporal=tuple(_temporal(clause, clause_start)),
                polarity="negative" if negated else "positive",
                modality="asserted",
                rule=match.rule,
                language=language_here,
            ))
            entities.append(fact_entity)
            if len(facts) >= cfg.max_facts_per_message:
                abstentions.append("fact_cap_reached")
                break

    _scan_entities(normalized, entities)
    return Interpretation(
        entities=tuple(_dedupe_entities(entities)),
        facts=tuple(facts),
        ambiguities=tuple(ambiguities),
        abstentions=tuple(abstentions),
        confidence=round(
            sum(fact.confidence for fact in facts) / len(facts), 4)
        if facts else 0.0,
    )


_LATIN_LETTER = re.compile(r"[A-Za-z\u00C0-\u024F]")
_ETHIOPIC_LETTER = re.compile(r"[\u1200-\u137F]")
# Amharic questions do not need the Latin question mark: the interrogative
# word carries the force.  ማን who, ምን what, የት where, መቼ when, ለምን why,
# እንዴት how, ስንት how many.
_AM_QUESTION = re.compile(r"(?:^|\s)(?:ማን|ምን|የት|መቼ|ለምን|እንዴት|ስንት)(?=\s|$)")


def _match_clause(clause: str, offset: int, language: str,
                  speaker: str) -> List[_Match]:
    # Script composition, not only the detected label, decides which packs a
    # clause is offered to.  In "I live in አዲስ አበባ" the Ethiopic words win the
    # label vote, yet the frame is English; each pack is anchored on a whole
    # construction, so offering a clause to a pack whose language did not win
    # cannot fabricate a fact, and a code-switched clause is read by both.
    latin = _LATIN_LETTER.search(clause) is not None
    ethiopic = _ETHIOPIC_LETTER.search(clause) is not None
    if language in ("en", "und") or latin:
        matches = _match_english(clause, offset, speaker)
        if matches:
            return matches
    if language in ("es", "fr", "und") or latin:
        matches = _match_multilingual(clause, offset, speaker)
        if matches:
            return matches
    if language == "am" or ethiopic:
        matches = _match_amharic(clause, offset, speaker)
        if matches:
            return matches
    # The third-party rules are anchored on an explicit English verb phrase,
    # so they are safe to try even when language detection picked another
    # Latin-script language from a single shared function word.
    return _match_third_party(clause, offset)


def _match_english(clause: str, offset: int,
                   speaker: str) -> List[_Match]:
    role_at = _ROLE_AT.match(clause)
    if role_at and role_at.group("org").strip(_TRAILING_PUNCT).casefold() \
            not in _ROLE_ORG_CONTEXT and \
            not _PROSPECTIVE.match(role_at.group("role").strip()):
        subject = SubjectRef(
            kind="speaker", label=speaker, confidence=1.0,
            owner=speaker, resolved_from="first_person")
        role, role_start, role_end = _trim(
            role_at.group("role"), offset + role_at.start("role"))
        organization, org_start, org_end = _trim(
            role_at.group("org"), offset + role_at.start("org"))
        return [
            _Match(rule="speaker_work_as", predicate="works_as", value=role,
                   value_span=(role_start, role_end), subject=subject,
                   entity_type="", confidence=0.84),
            _Match(rule="speaker_work", predicate="works_at",
                   value=organization, value_span=(org_start, org_end),
                   subject=subject, entity_type="organization",
                   confidence=0.85),
        ]
    for rule, pattern, predicate, subject_kind, entity_type, confidence in \
            _ENGLISH_RULES:
        found = pattern.match(clause)
        if not found:
            continue
        groups = found.groups()
        value_group = found.lastindex or len(groups)
        raw_value = groups[value_group - 1]
        if rule.startswith("relation_"):
            relation = groups[0]
            if relation.casefold() not in RELATION_NOUNS:
                continue
            if rule == "relation_attribute":
                verb = found.group("verb").casefold()
                if verb in _NEGATED_PREDICATES:
                    predicate = _NEGATED_PREDICATES[verb][0]
                else:
                    predicate = _ENGLISH_PREDICATES.get(
                        verb, _canonical_predicate(verb))
            subject = _subject_for_relation(relation, speaker)
        else:
            if rule == "speaker_attribute":
                predicate = _canonical_predicate(groups[0])
            elif rule == "speaker_work":
                predicate = "works_at" if groups[0].casefold() in (
                    "at", "for") else "works_in"
            elif rule == "speaker_used_work":
                predicate = "works_at" if groups[0].casefold() in (
                    "at", "for") else "works_in"
            if subject_kind == "speaker":
                subject = SubjectRef(
                    kind="speaker", label=speaker, confidence=1.0,
                    owner=speaker, resolved_from="first_person")
            elif rule == "team_uses":
                subject = SubjectRef(
                    kind="derived", label="user's team", confidence=0.75,
                    owner=speaker, relation="team",
                    resolved_from="collective_pronoun")
            else:
                subject = SubjectRef(
                    kind=subject_kind, label=raw_value, confidence=confidence)
        value_base = offset + found.start(value_group)
        corrected = _name_correction(raw_value) if rule in _NAME_RULES else None
        if corrected is not None:
            skipped, raw_value = corrected
            value_base += skipped
            rule = f"{rule}_correction"
        value, value_start, value_end = _trim(raw_value, value_base)
        if rule in _NAME_RULES:
            # "Sam Waters, not Sam Wallis" names one person and rejects one
            # spelling; the rejected tail is not part of the name.
            rejected = _NAME_REJECTED_TAIL.search(value)
            if rejected is not None:
                value = value[:rejected.start()].rstrip(_TRAILING_PUNCT)
                value_end = value_start + len(value)
                rule = f"{rule}_correction"
        if rule == "speaker_has":
            if _EPHEMERAL.match(value) or \
                    _NOT_A_POSSESSION.match(value) or \
                    value.casefold().strip(_TRAILING_PUNCT) in _DEMONSTRATIVES:
                continue
        if predicate in ("lives_in", "is_from", "born_in"):
            value = _canonical_place(value)
        return [_Match(
            rule=rule, predicate=predicate, value=value,
            value_span=(value_start, value_end), subject=subject,
            entity_type=entity_type or "", confidence=confidence)]

    noisy = _noisy_relation(clause, offset, speaker)
    if noisy:
        return noisy

    pronoun = _PRONOUN.match(clause)
    if pronoun:
        verb = pronoun.group("verb").casefold()
        if verb in _NEGATED_PREDICATES:
            predicate = _NEGATED_PREDICATES[verb][0]
        else:
            predicate = _ENGLISH_PREDICATES.get(
                verb, _canonical_predicate(verb))
        value, value_start, value_end = _trim(
            pronoun.group("value"), offset + pronoun.start("value"))
        if predicate in ("lives_in", "is_from"):
            value = _canonical_place(value)
        subject = SubjectRef(
            kind="unknown", label=pronoun.group("pronoun").casefold(),
            confidence=0.4, resolved_from="pronoun", status="ambiguous",
            ambiguity="pronoun_needs_antecedent")
        return [_Match(
            rule="pronoun_clause", predicate=predicate, value=value,
            value_span=(value_start, value_end), subject=subject,
            entity_type="", confidence=0.55)]
    return []


def _match_multilingual(clause: str, offset: int,
                        speaker: str) -> List[_Match]:
    for pattern, rule in ((_SPANISH_RELATION, "es_relation_name"),
                          (_FRENCH_RELATION, "fr_relation_name")):
        relation = pattern.match(clause)
        if relation:
            value, start, end = _trim(
                relation.group("value"), offset + relation.start("value"))
            return [_Match(
                rule=rule, predicate="name", value=value,
                value_span=(start, end),
                subject=_subject_for_relation(
                    relation.group("relation"), speaker),
                entity_type="person", confidence=0.85)]
    for pattern, rule, language in (
            (_SPANISH_RELATION_PREDICATE, "es_relation_predicate", "es"),
            (_FRENCH_RELATION_PREDICATE, "fr_relation_predicate", "fr")):
        relation = pattern.match(clause)
        if not relation:
            continue
        predicate = _es_fr_predicate(
            relation.group("verb").casefold(),
            relation.group("prep").casefold(), language)
        value, start, end = _trim(
            relation.group("value"), offset + relation.start("value"))
        if predicate in ("lives_in", "is_from"):
            value = _canonical_place(value)
        return [_Match(
            rule=rule, predicate=predicate, value=value,
            value_span=(start, end),
            subject=_subject_for_relation(
                relation.group("relation"), speaker),
            entity_type=("place" if predicate == "lives_in"
                         else "organization"),
            confidence=0.85)]
    for rule, pattern, predicate, entity_type, confidence in \
            _MULTILINGUAL_RULES:
        found = pattern.match(clause)
        if not found:
            continue
        value, start, end = _trim(
            found.group(1), offset + found.start(1))
        if predicate in ("lives_in", "is_from"):
            value = _canonical_place(value)
        return [_Match(
            rule=rule, predicate=predicate, value=value,
            value_span=(start, end),
            subject=SubjectRef(
                kind="speaker", label=speaker, confidence=1.0,
                owner=speaker, resolved_from="first_person"),
            entity_type=entity_type, confidence=confidence)]
    return _match_es_fr_third_party(clause, offset)


def _match_es_fr_third_party(clause: str, offset: int) -> List[_Match]:
    """Named and pronoun subjects in Spanish and French.

    A pronoun stays unresolved at this stage; Stage 3 decides from the recent
    conversation whether exactly one antecedent fits.  A name is accepted only
    when the clause defines it with a capital letter, which is the same
    evidence the English third-party rules use.
    """
    for pattern, rule, language in (
            (_ES_PRONOUN, "es_pronoun_clause", "es"),
            (_FR_PRONOUN, "fr_pronoun_clause", "fr")):
        found = pattern.match(clause)
        if not found:
            continue
        verb = found.group("verb").casefold()
        pronoun = found.group("pronoun").casefold()
        predicate = _es_fr_predicate(
            verb, found.group("prep").casefold(), language)
        value, start, end = _trim(
            found.group("value"), offset + found.start("value"))
        if predicate in ("lives_in", "is_from"):
            value = _canonical_place(value)
        return [_Match(
            rule=rule, predicate=predicate, value=value,
            value_span=(start, end),
            subject=SubjectRef(
                kind="unknown", label=pronoun, confidence=0.4,
                resolved_from="pronoun", status="ambiguous",
                ambiguity="pronoun_needs_antecedent"),
            entity_type="", confidence=0.55)]
    for pattern, rule, language in (
            (_ES_NAMED, "es_named_clause", "es"),
            (_FR_NAMED, "fr_named_clause", "fr")):
        found = pattern.match(clause)
        if not found:
            continue
        name = found.group("subject").strip()
        verb = found.group("verb").casefold()
        predicate = _es_fr_predicate(
            verb, found.group("prep").casefold(), language)
        value, start, end = _trim(
            found.group("value"), offset + found.start("value"))
        if predicate in ("lives_in", "is_from"):
            value = _canonical_place(value)
        return [_Match(
            rule=rule, predicate=predicate, value=value,
            value_span=(start, end),
            subject=SubjectRef(
                kind="named", label=name, confidence=0.7,
                entity_key=name.casefold(), resolved_from="named_mention"),
            entity_type="", confidence=0.7)]
    return []


def _es_fr_predicate(verb: str, preposition: str, language: str) -> str:
    if language == "es":
        if verb.startswith("trabaja"):
            return "works_at" if preposition == "para" else "works_in"
        return "lives_in"
    if verb.startswith("travaille"):
        return "works_at" if preposition == "chez" else "works_in"
    return "lives_in"


def _match_amharic(clause: str, offset: int,
                   speaker: str) -> List[_Match]:
    """Amharic clauses whose person and predicate carry explicit suffixes."""
    relation = _AM_RELATION_NAME.match(clause)
    if relation:
        value, start, end = _trim(
            relation.group("value"), offset + relation.start("value"))
        return [_Match(
            rule="am_relation_name", predicate="name", value=value,
            value_span=(start, end),
            subject=_subject_for_relation(
                _AM_RELATIONS[relation.group("relation")], speaker),
            entity_type="person", confidence=0.86)]
    relation = _AM_RELATION_LIVES.match(clause)
    if relation:
        value, start, end = _trim(
            relation.group("value"), offset + relation.start("value"))
        return [_Match(
            rule="am_relation_lives", predicate="lives_in",
            value=_canonical_place(value), value_span=(start, end),
            subject=_subject_for_relation(
                _AM_RELATIONS[relation.group("relation")], speaker),
            entity_type="place", confidence=0.84)]
    for pattern, rule, predicate, entity_type, confidence in _AM_SPEAKER_RULES:
        found = pattern.match(clause)
        if not found:
            continue
        value, start, end = _trim(
            found.group("value"), offset + found.start("value"))
        if predicate in ("lives_in", "is_from"):
            value = _canonical_place(value)
        return [_Match(
            rule=rule, predicate=predicate, value=value,
            value_span=(start, end),
            subject=SubjectRef(
                kind="speaker", label=speaker, confidence=1.0,
                owner=speaker, resolved_from="first_person"),
            entity_type=entity_type, confidence=confidence)]
    pronoun = _AM_PRONOUN_LIVES.match(clause)
    if pronoun:
        value, start, end = _trim(
            pronoun.group("value"), offset + pronoun.start("value"))
        return [_Match(
            rule="am_pronoun_lives", predicate="lives_in",
            value=_canonical_place(value), value_span=(start, end),
            subject=SubjectRef(
                kind="unknown", label=pronoun.group("pronoun").casefold(),
                confidence=0.4, resolved_from="pronoun", status="ambiguous",
                ambiguity="pronoun_needs_antecedent"),
            entity_type="", confidence=0.55)]
    return []


def _match_elided(clause: str, offset: int,
                  subject: SubjectRef) -> List[_Match]:
    """Binds a verb-first clause to the subject established just before it."""
    found = _ELIDED.match(clause)
    if not found or subject.status != "resolved":
        return []
    verb = found.group("verb").casefold()
    predicate = _ENGLISH_PREDICATES.get(verb, _canonical_predicate(verb))
    value, start, end = _trim(found.group("value"),
                              offset + found.start("value"))
    if predicate in ("lives_in", "is_from"):
        value = _canonical_place(value)
    return [_Match(
        rule="elided_clause", predicate=predicate, value=value,
        value_span=(start, end), subject=subject, entity_type="",
        confidence=0.7)]


def _match_third_party(clause: str, offset: int) -> List[_Match]:
    clause_offset = offset
    marker = _LEADING_MARKER.match(clause)
    if marker:
        offset += marker.end()
        clause = clause[marker.end():]
    found = _THIRD_PARTY.match(clause)
    attribute = None if found else _THIRD_PARTY_ATTRIBUTE.match(clause)
    aged = None if (found or attribute) else _THIRD_PARTY_AGE.match(clause)
    if not found and not attribute and not aged:
        return []
    match = found or attribute or aged
    name = match.group("name").strip()
    # A clause headed by a first-person pronoun is speaker material even when
    # no speaker rule covers its verb.  Falling through to the third-party
    # rules would label the subject "I"/"I do not", inventing a person out of
    # the speaker's own words; the correct outcome is no fact here.
    head = name.split()[0].strip("'’.-").casefold()
    if head in _NON_NAME_HEADS:
        return []
    if _looks_like_name(name, clause_offset == 0):
        subject = SubjectRef(kind="named", label=name, confidence=0.7,
                             entity_key=name.casefold(),
                             resolved_from="named_mention")
    else:
        # The clause is still worth a candidate fact, but the words are not
        # attested as a person, so the subject is left explicitly unresolved.
        subject = SubjectRef(
            kind="named", label=name, confidence=0.3, status="ambiguous",
            ambiguity="name_not_proper_noun",
            resolved_from="unverified_name")
    if found:
        verb = match.group("verb").casefold()
        if verb in _NEGATED_PREDICATES:
            predicate, confidence = _NEGATED_PREDICATES[verb]
        else:
            predicate = _ENGLISH_PREDICATES.get(
                verb, _canonical_predicate(verb))
            confidence = 0.7
        rule = "third_party"
    elif attribute:
        predicate = ATTRIBUTE_PREDICATES[match.group("attr").casefold()]
        confidence = 0.75
        rule = "third_party_attribute"
    else:
        predicate = "age"
        confidence = 0.7
        rule = "third_party_age"
    value, start, end = _trim(
        match.group("value"), offset + match.start("value"))
    if predicate in ("lives_in", "is_from"):
        value = _canonical_place(value)
    if predicate == "knows" and value.casefold() in _LANGUAGE_NAMES:
        # "knows Turkish" about a language is the same fact as "speaks
        # Turkish"; the closed language list supplies the evidence.
        predicate, confidence = "speaks", 0.75
    return [_Match(
        rule=rule, predicate=predicate, value=value,
        value_span=(start, end),
        subject=subject,
        entity_type="", confidence=confidence)]


def _scan_entities(normalized: NormalizedText,
                   entities: List[Entity]) -> None:
    """Adds date and number entities that no fact rule consumed."""
    text = normalized.text
    covered = {entity.span for entity in entities}
    for match in _YEAR.finditer(text):
        if match.span() in covered:
            continue
        entities.append(Entity(text=match.group(0), type="date",
                               span=match.span(), confidence=0.85,
                               rule="year"))
    for match in re.finditer(r"(?<![\w.])\d{1,4}(?![\w.])", text):
        if match.span() in covered:
            continue
        entities.append(Entity(text=match.group(0), type="number",
                               span=match.span(), confidence=0.6,
                               rule="number"))


def _dedupe_entities(entities: List[Entity]) -> List[Entity]:
    seen = set()
    unique: List[Entity] = []
    for entity in entities:
        key = (entity.span, entity.type, entity.text.casefold())
        if key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    return unique


__all__ = ["Interpretation", "interpret"]
