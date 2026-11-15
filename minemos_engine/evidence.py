"""Evidence sufficiency for the retrieval-to-answer decision.

Eligibility answers two questions structurally: does the query name a subject
that exists, and does that subject store the asked attribute?  Neither answer
establishes that a *retrieved row* supports the *question*.  A document-scoped
subject ("conversation:<key>") in particular is a search boundary, not an
entity: equality against it says only that the caller asked inside that
document, never that the document contains the queried content.

This module adds the missing semantic step.  It classifies the returned
candidate rows into the evidence states the answer contract must distinguish:

``supported``
    At least one row contains the query's substantive content (or, for an
    entity subject whose query adds no content beyond the entity, the
    verified subject and attribute are the evidence), and no unresolved
    contradiction was found.
``contradicted``
    Rows support the question but carry unresolved competing values.  The
    rows are kept; the answer must be presented as contested, never settled.
``insufficient``
    Some content matches, but not enough to establish that the row answers
    the question.  The query may also carry no verifiable content at all.
``irrelevant``
    The subject was verified but none of the returned rows addresses the
    query's substantive content.
``wrong_scope``
    The caller asserted a document scope, and no row in it contains the
    query's content.  The evidence, if it exists, lives in another scope.
``missing``
    No subject, attribute or row exists (reported by the eligibility gate).
``ambiguous``
    Several subjects or referents match equally well, so no single answer can
    be attributed (reported by the eligibility gate).

Two rules govern the decision:

* **Scope equality is not relevance.**  For a document scope the query's own
  content must appear in the rows.  Nothing is settled because the rows
  happen to live in the asserted conversation.
* **Similarity is supporting evidence, never proof.**  Dense similarity is
  reported for diagnostics and used only as one term in a conjunction; it
  can never make a row evidence on its own.

The lexical test is deliberately conservative and language-limited.  It
recognises exact tokens, near-miss typo variants, contiguous bigrams and
order-preserving subsequences.  It does not claim natural-language
understanding: a heavily rewritten paraphrase or a purely anaphoric follow-up
that shares no surface content abstains instead of being guessed.
"""

from __future__ import annotations

import functools
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from minemos_engine.eligibility import (
    _MARKER_TOKENS,
    _STOP_TOKENS,
    attribute_ids,
    attribute_ids_from_tokens,
    tokens,
)

# Question words never carry answer content.
_QUESTION_WORDS: frozenset = frozenset({
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "much", "many",
})

# Demonstratives, writers/readers and discourse fillers are not entities and
# appear in nearly any conversation.  They cannot make a row evidence.
_GENERIC_TERMS: frozenset = frozenset({
    "this", "that", "these", "those", "there", "here", "then", "than",
    "i", "you", "we", "us", "me", "my", "your", "our", "yours", "ours",
    "myself", "yourself", "ourselves", "u",
    "some", "any", "all", "more", "most", "very", "just", "also", "only",
    "even", "still", "again", "now", "new", "get", "got", "make", "made",
    "say", "said", "want", "need", "know", "think", "going", "gonna",
    "okay", "ok", "yes", "well", "thing", "things", "stuff",
    # question verbs: they name the act of asking, not stored content
    "tell", "told", "affect", "affects", "affected", "discuss",
    "discussed", "mention", "mentioned", "explain", "explained",
    "describe", "described", "happen", "happened",
    "call", "called", "calls", "refer", "referred", "referring",
    # meta-requests and scene-setting frames: they name the act of asking or
    # the discourse frame, so they can only dilute the question's content
    "remind", "reminds", "reminded", "reminding",
    "situation", "situations",
})

# Direct lexical equivalences used when testing whether a row contains the
# query's substantive content.  Paraphrase routinely swaps a small set of
# common verbs and nouns ("obtain" for "get", "repair" for "fix"); without an
# equivalence the row appears to lack content it plainly has.  The table is a
# closed set of attested direct synonyms, not a similarity score: a match still
# has to satisfy the same coverage/bigram/LCS conjunctions below, so it cannot
# make an otherwise unrelated row sufficient.  It is not used for ranking or
# for eligibility decisions.
_SYNONYM_FORMS: Tuple[Tuple[str, ...], ...] = (
    ("get", "gets", "got", "gotten", "getting",
     "obtain", "obtains", "obtained", "obtaining",
     "acquire", "acquires", "acquired", "acquiring"),
    ("give", "gives", "gave", "given", "giving",
     "provide", "provides", "provided", "providing",
     "supply", "supplies", "supplied", "supplying"),
    ("fix", "fixes", "fixed", "fixing",
     "repair", "repairs", "repaired", "repairing",
     "mend", "mends", "mended", "mending"),
    ("issue", "issues", "problem", "problems"),
    ("understand", "understands", "understood", "understanding",
     "know", "knows", "knew", "known", "knowing"),
    ("work", "works", "worked", "working",
     "function", "functions", "functioned", "functioning",
     "operate", "operates", "operated", "operating"),
    ("make", "makes", "made", "making",
     "create", "creates", "created", "creating",
     "build", "builds", "built", "building",
     "construct", "constructs", "constructed", "constructing"),
    ("start", "starts", "started", "starting",
     "begin", "begins", "began", "begun", "beginning"),
    ("stop", "stops", "stopped", "stopping",
     "end", "ends", "ended", "ending",
     "finish", "finishes", "finished", "finishing"),
    ("remove", "removes", "removed", "removing",
     "delete", "deletes", "deleted", "deleting"),
    ("change", "changes", "changed", "changing",
     "modify", "modifies", "modified", "modifying",
     "alter", "alters", "altered", "altering"),
    ("need", "needs", "needed", "needing",
     "require", "requires", "required", "requiring"),
    ("help", "helps", "helped", "helping",
     "assist", "assists", "assisted", "assisting"),
    ("show", "shows", "showed", "shown", "showing",
     "display", "displays", "displayed", "displaying"),
    ("check", "checks", "checked", "checking",
     "verify", "verifies", "verified", "verifying",
     "test", "tests", "tested", "testing"),
    ("find", "finds", "found", "finding",
     "locate", "locates", "located", "locating"),
    ("allow", "allows", "allowed", "allowing",
     "enable", "enables", "enabled", "enabling",
     "permit", "permits", "permitted", "permitting"),
    ("improve", "improves", "improved", "improving",
     "enhance", "enhances", "enhanced", "enhancing"),
    ("reduce", "reduces", "reduced", "reducing",
     "lower", "lowers", "lowered", "lowering",
     "decrease", "decreases", "decreased", "decreasing"),
    ("increase", "increases", "increased", "increasing",
     "raise", "raises", "raised", "raising"),
    ("fast", "faster", "fastest", "quick", "quicker", "quickest",
     "rapid"),
    ("slow", "slower", "slowest", "sluggish"),
    ("big", "bigger", "biggest", "large", "larger", "largest",
     "huge"),
    ("small", "smaller", "smallest", "little", "tiny"),
    ("easy", "easier", "easiest", "simple", "simpler", "simplest"),
    ("hard", "harder", "hardest", "difficult"),
    ("important", "critical"),
    ("possible", "feasible"),
    ("because", "since"),
    ("however", "though", "although"),
    ("run", "runs", "ran", "running",
     "execute", "executes", "executed", "executing"),
    ("say", "says", "said", "saying",
     "state", "states", "stated", "stating"),
    ("tell", "tells", "told", "telling",
     "inform", "informs", "informed", "informing"),
    ("think", "thinks", "thought", "thinking",
     "believe", "believes", "believed", "believing"),
    ("deploy", "deploys", "deployed", "deploying",
     "release", "releases", "released", "releasing"),
    ("error", "errors", "bug", "bugs", "fault", "faults"),
    ("buy", "buys", "bought", "buying",
     "purchase", "purchases", "purchased", "purchasing"),
    ("speak", "speaks", "spoke", "spoken", "speaking",
     "talk", "talks", "talked", "talking"),
)


def _build_synonym_index(groups: Sequence[Sequence[str]]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for group in groups:
        canonical = group[0]
        for form in group:
            existing = index.setdefault(form, canonical)
            if existing != canonical:
                raise ValueError(
                    f"evidence synonym {form!r} belongs to conflicting groups")
    return index


_SYNONYM_CANON: Dict[str, str] = _build_synonym_index(_SYNONYM_FORMS)


def _canonical(token: str) -> str:
    return _SYNONYM_CANON.get(token, token)


# A row is evidence when one of these conjunctions holds.  Each combines a
# lexical overlap signal with a strength requirement; no single number --
# and in particular no similarity score -- decides on its own.
_MIN_ENTITY_FOCUS_TERMS = 2
_MIN_DISTINCT_HITS = 2
_MIN_COVERAGE = 0.5
_MIN_BIGRAM_HITS = 2
_LONG_TERM = 6
_MIN_LCS = 0.6

# Support is decided from the best dense-ranked candidates.  A content query
# that is supported at all is supported near the top of the ranking; scanning
# a bounded window keeps the cost proportional to the question rather than to
# the size of the document.
EVIDENCE_SCAN_LIMIT = 256

EVIDENCE_STATES = (
    "supported",
    "contradicted",
    "insufficient",
    "irrelevant",
    "wrong_scope",
    "missing",
    "ambiguous",
)


def is_document_subject(subject: object) -> bool:
    """Whether a stored subject is a document/conversation search boundary."""
    value = str(subject or "").strip().lower()
    return value.startswith("conversation:") or value.startswith("document:")


def _numeric_terms(terms: Iterable[str]) -> List[str]:
    """Terms that name a literal number.

    Only a term that is itself a number qualifies; an alphanumeric token such
    as "q3" is ordinary content.
    """
    return [term for term in terms if term.isdigit()]


def evidence_terms(text: object,
                   exclude: Iterable[str] = ()) -> List[str]:
    """Ordered, deduplicated substantive terms used as evidence."""
    excluded = set(exclude)
    result: List[str] = []
    seen: Set[str] = set()
    for token in tokens(text):
        if (token in _STOP_TOKENS or token in _QUESTION_WORDS
                or token in _MARKER_TOKENS or token in _GENERIC_TERMS
                or token in excluded or len(token) < 3):
            continue
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


@functools.lru_cache(maxsize=65536)
def _fuzzy_ratio(a: str, b: str) -> float:
    """Similarity ratio, cached: the same query/corpus word pairs recur."""
    return SequenceMatcher(None, a, b).ratio()


def _fuzzy_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) < 4 or len(b) < 4:
        return False
    if a[0] != b[0] or abs(len(a) - len(b)) > 2:
        return False
    return _fuzzy_ratio(a, b) >= 0.8


def _token_matches(term: str, token: str, canonical: str) -> bool:
    """Exact/fuzzy match or a direct synonym equivalence for one token."""
    if _fuzzy_equal(term, token):
        return True
    term_canon = _SYNONYM_CANON.get(term)
    return term_canon is not None and term_canon == canonical


def _present(term: str, haystack: Sequence[str],
             canon: Optional[Sequence[str]] = None) -> bool:
    """Whether ``term`` matches any token, with the same rules as rows.

    The fuzzy comparison is inlined with its cheap rejection tests first:
    matching requires the same first character and near-identical length, so
    most tokens are rejected without constructing a ``SequenceMatcher``.  The
    result is identical to ``_token_matches`` over every token.
    """
    if canon is None:
        canon = [_canonical(token) for token in haystack]
    term_canon = _SYNONYM_CANON.get(term)
    term_len = len(term)
    for position, token in enumerate(haystack):
        if term == token:
            return True
        if term_canon is not None and term_canon == canon[position]:
            return True
        if (term_len >= 4 and len(token) >= 4
                and token[0] == term[0]
                and abs(len(token) - term_len) <= 2
                and _fuzzy_ratio(term, token) >= 0.8):
            return True
    return False


def present_terms(terms: Sequence[str], text: object) -> List[str]:
    """The subset of ``terms`` that ``text`` contains as evidence terms.

    Uses the same exact/fuzzy/synonym rules as row assessment, so a discovery
    decision and the evidence gate cannot disagree about whether a row carries
    a query term.
    """
    haystack = tokens(text)
    canon = [_canonical(token) for token in haystack]
    return [term for term in terms if _present(term, haystack, canon)]


def _bigrams(sequence: Sequence[str]) -> List[Tuple[str, str]]:
    return [(sequence[i], sequence[i + 1])
            for i in range(len(sequence) - 1)]


def _bigram_hits(grams: Sequence[Tuple[str, str]],
                 haystack: Sequence[str],
                 canon: Optional[Sequence[str]] = None) -> int:
    if canon is None:
        canon = [_canonical(token) for token in haystack]
    hits = 0
    for first, second in grams:
        if any(_token_matches(first, haystack[start], canon[start])
               and _token_matches(second, haystack[start + 1], canon[start + 1])
               for start in range(len(haystack) - 1)):
            hits += 1
    return hits


def _residual_is_specific(terms: Sequence[str], query_text: object) -> bool:
    """Whether leftover query terms assert substantive content.

    An entity subject already supplies identity, and the attribute gate has
    already decided relevance.  A single short lowercase leftover word is
    usually a misspelled attribute ("ctiy" for "city") or a query verb, and
    demanding it verbatim would abstain on supported questions.  A longer
    word, several words, or a capitalised term names content the subject's
    facts must actually address.
    """
    if not terms:
        return False
    if len(terms) >= 2:
        return True
    if len(terms[0]) >= _LONG_TERM:
        return True
    term = terms[0]
    for match in re.finditer(r"[^\W_]+", str(query_text or ""), re.UNICODE):
        surface = match.group(0)
        if surface.casefold() == term and surface[:1].isupper():
            return True
    return False


def _lcs_ratio(needles: Sequence[str], haystack: Sequence[str],
               canon: Optional[Sequence[str]] = None) -> float:
    if not needles or not haystack:
        return 0.0
    if canon is None:
        canon = [_canonical(token) for token in haystack]
    previous = [0] * (len(haystack) + 1)
    for needle in needles:
        current = [0]
        for position, token in enumerate(haystack, start=1):
            best = max(previous[position], current[-1])
            if _token_matches(needle, token, canon[position - 1]):
                best = max(best, previous[position - 1] + 1)
            current.append(best)
        previous = current
    return previous[-1] / len(needles)


def _synonym_only(term: str, haystack: Sequence[str],
                  canon: Sequence[str]) -> bool:
    """Whether a term matches only through the closed synonym table."""
    if any(_fuzzy_equal(term, token) for token in haystack):
        return False
    term_canon = _SYNONYM_CANON.get(term)
    return term_canon is not None and term_canon in canon


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_DEFINITE = re.compile(r"\bthe\s+([^\W_]+)", re.UNICODE)


def _named_terms(text: object) -> Set[str]:
    """Content tokens written capitalized in the source text.

    A capitalised content word is the only name signal available without
    parsing; lowercase transcripts simply supply none, which disables the
    co-reference check instead of guessing.
    """
    named: Set[str] = set()
    for match in _WORD.finditer(str(text or "")):
        surface = match.group(0)
        if not surface[:1].isupper():
            continue
        token = surface.casefold()
        if (token in _STOP_TOKENS or token in _QUESTION_WORDS
                or token in _MARKER_TOKENS or token in _GENERIC_TERMS
                or len(token) < 3):
            continue
        named.add(token)
    return named


def _has_definite_description(text: object) -> bool:
    """Whether the question presupposes one referent ("the car")."""
    for match in _DEFINITE.finditer(str(text or "").casefold()):
        token = match.group(1)
        if (token not in _STOP_TOKENS and token not in _QUESTION_WORDS
                and token not in _MARKER_TOKENS
                and token not in _GENERIC_TERMS and len(token) >= 3):
            return True
    return False


def _referent_ambiguous(query_text: object,
                        named_sets: Sequence[Set[str]]) -> bool:
    """Whether sufficient rows name different objects for one description.

    "Who owns the car Alice drives?" with "Alice drives the Volvo" and
    "Alice drives the Ford" is two possible questions, not one answer.  The
    check needs a singular definite description in the question and at least
    two rows whose own names are disjoint once the question's names are
    removed; anything less is left to the ordinary support rule.
    """
    if len(named_sets) < 2 or not _has_definite_description(query_text):
        return False
    query_named = _named_terms(query_text)
    objects: List[Set[str]] = []
    for named in named_sets:
        remainder = named - query_named
        if remainder:
            objects.append(remainder)
    if len(objects) < 2:
        return False
    return not set.intersection(*objects)


def assess_row(content_terms: Sequence[str],
               focus_terms: Sequence[str],
               text: object,
               *,
               document_scope: bool = False,
               haystack: Optional[Sequence[str]] = None,
               canon: Optional[Sequence[str]] = None,
               required_terms: Sequence[str] = ()) -> Dict[str, Any]:
    """Whether one row is evidence for the query's substantive content.

    ``document_scope`` marks a conversation/document boundary.  Such a scope
    identifies no entity, so the query's own content has to carry the
    evidence: a row that merely contains the two focus words (plus one other
    common word) is not enough, because common words co-occur across unrelated
    transcripts.  Entity scopes may use that weaker conjunction -- the
    verified subject already supplies identity -- but a document row must
    reach coverage or phrase-level support.

    ``required_terms`` are query terms the row has to carry for the content to
    be the same fact at all (a named number); lexical overlap alone cannot
    make a row about a different value evidence for the question.
    """
    haystack = tokens(text) if haystack is None else haystack
    canon = ([_canonical(token) for token in haystack]
             if canon is None else canon)
    hits = [term for term in content_terms if _present(term, haystack, canon)]
    long_hits = sum(1 for term in hits if len(term) >= _LONG_TERM)
    coverage = (len(hits) / len(content_terms)) if content_terms else 0.0
    focus_hit = bool(focus_terms) and all(
        _present(term, haystack, canon) for term in focus_terms)
    bigram_hits = _bigram_hits(_bigrams(content_terms), haystack, canon)
    lcs = _lcs_ratio(content_terms, haystack, canon)
    value_matches = not required_terms or any(
        _present(term, haystack, canon) for term in required_terms)
    sufficient = value_matches and (
        (not content_terms and focus_hit)
        or (not document_scope
            and focus_hit and len(focus_terms) >= _MIN_ENTITY_FOCUS_TERMS
            and len(hits) >= _MIN_DISTINCT_HITS)
        or (focus_hit and coverage >= 0.6)
        or (bigram_hits >= _MIN_BIGRAM_HITS
            and len(hits) >= _MIN_DISTINCT_HITS)
        or (bigram_hits >= 1 and coverage >= 0.6)
        or (len(content_terms) <= 2 and hits
            and len(hits) == len(content_terms))
        or (long_hits >= _MIN_DISTINCT_HITS and coverage >= _MIN_COVERAGE)
        or (lcs >= _MIN_LCS and long_hits >= 1)
    )
    return {
        "sufficient": sufficient,
        "hits": len(hits),
        "coverage": round(coverage, 4),
        "focus_hit": focus_hit,
        "bigram_hits": bigram_hits,
        "long_hits": long_hits,
        "lcs_ratio": round(lcs, 4),
        "synonym_hits": sum(
            1 for term in hits if _synonym_only(term, haystack, canon)),
    }


def _empty_summary() -> Dict[str, Any]:
    return {
        "content_terms": 0,
        "focus_terms": 0,
        "scanned_rows": 0,
        "supporting_rows": 0,
        "best_hits": 0,
        "best_coverage": 0.0,
        "best_focus_hit": False,
        "best_bigram_hits": 0,
        "best_long_hits": 0,
        "best_lcs_ratio": 0.0,
        "best_synonym_hits": 0,
        "asked_attributes": [],
        "relation_represented": True,
        "relation_linked": True,
    }


def assess_evidence(
    query_text: object,
    focus: Optional[str],
    rows: Sequence[Dict[str, Any]],
    *,
    scope_kind: str,
    explicit_scope: bool,
    exclude_terms: Iterable[str] = (),
) -> Dict[str, Any]:
    """Classify whether ``rows`` are evidence for ``query_text``.

    ``scope_kind`` is ``"document"`` when the verified subject is a
    conversation/document boundary and ``"entity"`` when it is a named
    subject.  For an entity subject the query's substantive content beyond the
    entity itself is what must appear; when the query adds no such content,
    the verified subject and attribute are the evidence.  For a document
    subject every substantive term (entity included) must appear, because the
    subject identifies no entity at all.

    The returned mapping contains one of :data:`EVIDENCE_STATES`, a
    human-readable ``reason`` for non-supporting outcomes, the memory ids of
    supporting rows, and numeric diagnostics.  It never contains row text.
    """
    focus_terms = evidence_terms(focus)
    terms = evidence_terms(query_text, exclude=exclude_terms)
    # For a verified entity, a number in the question is a literal value the
    # stored fact has to carry: a different stored number is a different
    # fact, not a paraphrase.  Document scopes are left on the textual gate
    # because an utterance's number can be a changed parameter ("do that
    # again for chapter 5" over notes about chapter 3), not a claim the
    # transcript must contain.
    required_values = (
        _numeric_terms(tokens(query_text)) if scope_kind == "entity" else [])
    if scope_kind == "entity":
        residual = [term for term in terms if term not in focus_terms]
        content_terms = (
            residual if _residual_is_specific(residual, query_text) else [])
        focus_required = False
    else:
        content_terms = list(terms)
        focus_required = True
    required_focus = focus_terms if focus_required else []

    summary = _empty_summary()
    summary["content_terms"] = len(content_terms)
    summary["focus_terms"] = len(required_focus)
    # A question that names a known relation ("who owns ...") cannot be
    # answered from a row that never expresses that relation, however much of
    # the premise it repeats.  Attribute markers are excluded from the content
    # terms above, so this check is their only path into the verdict.
    asked_attributes = attribute_ids(query_text)
    relation_represented = not asked_attributes
    summary["asked_attributes"] = sorted(asked_attributes)
    supporting: List[str] = []
    supporting_named: List[Set[str]] = []
    value_rows: List[str] = []
    value_seen = False
    best = {
        "hits": 0, "coverage": 0.0, "focus_hit": False,
        "bigram_hits": 0, "long_hits": 0, "lcs_ratio": 0.0,
        "synonym_hits": 0,
    }
    for row in rows[:EVIDENCE_SCAN_LIMIT]:
        text = row.get("text") or ""
        haystack = tokens(text)
        if not relation_represented:
            relation_represented = bool(
                attribute_ids_from_tokens(haystack) & asked_attributes)
        if required_values and any(
                _present(term, haystack) for term in required_values):
            value_seen = True
            memory_id = row.get("memory_id")
            if memory_id is not None:
                value_rows.append(memory_id)
        assessment = assess_row(content_terms, required_focus, text,
                                document_scope=scope_kind == "document",
                                haystack=haystack,
                                required_terms=required_values)
        summary["scanned_rows"] += 1
        for key in ("hits", "bigram_hits", "long_hits", "synonym_hits"):
            best[key] = max(best[key], assessment[key])
        for key in ("coverage", "lcs_ratio"):
            best[key] = max(best[key], assessment[key])
        best["focus_hit"] = best["focus_hit"] or assessment["focus_hit"]
        if assessment["sufficient"]:
            supporting_named.append(_named_terms(text))
            memory_id = row.get("memory_id")
            if memory_id is not None:
                supporting.append(memory_id)
    summary["relation_represented"] = relation_represented
    summary.update({
        "best_hits": best["hits"],
        "best_coverage": round(best["coverage"], 4),
        "best_focus_hit": bool(best["focus_hit"]),
        "best_bigram_hits": best["bigram_hits"],
        "best_long_hits": best["long_hits"],
        "best_lcs_ratio": round(best["lcs_ratio"], 4),
        "best_synonym_hits": best["synonym_hits"],
    })

    if not content_terms and not required_focus:
        # Entity subject with no textual content beyond the entity: the
        # verified subject and compatible attribute are the evidence, unless
        # the question named a number, which still has to occur.
        if scope_kind == "entity" and required_values:
            sufficient = value_seen
            if sufficient:
                supporting = list(dict.fromkeys(value_rows))
        else:
            sufficient = scope_kind == "entity" and bool(rows)
        state = "supported" if sufficient else "insufficient"
        reason = None if sufficient else "evidence_insufficient"
    elif supporting:
        sufficient = True
        state = "supported"
        reason = None
    else:
        sufficient = False
        if (scope_kind == "document" and explicit_scope
                and summary["best_hits"] == 0
                and not summary["best_focus_hit"]):
            state = "wrong_scope"
            reason = "evidence_wrong_scope"
        elif scope_kind == "entity":
            state = "irrelevant"
            reason = "evidence_irrelevant"
        else:
            state = "insufficient"
            reason = "evidence_insufficient"

    if sufficient and not relation_represented:
        # The premise matches but the asked relation is nowhere in the scanned
        # material: a relational question ("who owns the car Alice drives?")
        # would otherwise settle on the intermediate fact alone.
        sufficient = False
        state = "insufficient"
        reason = "evidence_relation_not_found"
        supporting = []
    elif (sufficient and asked_attributes
          and _referent_ambiguous(query_text, supporting_named)):
        # Two sufficient rows answer a relational question about "the
        # <thing>" with different named referents; no single attribution is
        # possible.  Without an asked relation the definite phrase is
        # ordinary content ("the asian session"), and a conversation
        # naturally names several people, so the rows still answer.
        sufficient = False
        state = "insufficient"
        reason = "evidence_referent_ambiguous"
        supporting = []
    summary["supporting_rows"] = len(supporting)

    return {
        "state": state,
        "reason": reason,
        "sufficient": sufficient,
        "scope_kind": scope_kind,
        "supporting_memory_ids": supporting,
        "summary": summary,
    }


def assess_joint_evidence(
    query_text: object,
    focus: Optional[str],
    scope_rows: Sequence[Sequence[Dict[str, Any]]],
    *,
    explicit_scope: bool = True,
) -> Dict[str, Any]:
    """Evidence across an explicit list of document scopes.

    ``scope_rows`` holds one row list per requested scope, in request order.
    A single scope can already satisfy the ordinary per-row rule.  When no
    single row covers the question, the scopes are treated as one search
    surface: every substantive term must appear in a row that carries at
    least two of the query's terms (so an isolated word in an unrelated
    message cannot contribute), and at least two such rows must be
    involved, drawn from at least two of the requested scopes.  Rows keep
    their own scope provenance; nothing is merged across scopes.

    Requiring distinct contributing scopes is what makes ``cross_scope``
    truthful: a question whose evidence sits entirely inside one requested
    conversation is not cross-scope evidence, and the union path must not
    relabel it as such.  Such a question still answers through the ordinary
    per-row rule when a row is sufficient.

    The union rule never replaces the per-row rule -- it is an additional,
    stricter path required only when the evidence is genuinely distributed.
    A query whose content appears in none of the requested scopes still
    reports ``wrong_scope``.
    """
    focus_terms = evidence_terms(focus)
    terms = evidence_terms(query_text)
    content_terms = list(terms)
    required_focus = focus_terms

    summary = _empty_summary()
    summary.update({
        "content_terms": len(content_terms),
        "focus_terms": len(required_focus),
        "scopes": len(scope_rows),
        "scopes_with_rows": 0,
        "scopes_with_evidence": 0,
        "contributing_scopes": 0,
        "contributing_rows": 0,
        "union_coverage": 0.0,
    })
    # Same relation requirement as the single-scope path: a requested union
    # is only evidence for a relational question when the asked relation
    # occurs somewhere in the requested material.
    asked_attributes = attribute_ids(query_text)
    relation_represented = not asked_attributes
    summary["asked_attributes"] = sorted(asked_attributes)
    supporting: List[str] = []
    supporting_named: List[Set[str]] = []
    supporting_links: List[Tuple[int, Set[str]]] = []
    contributors: List[str] = []
    contributor_named: List[Set[str]] = []
    contributor_links: List[Tuple[int, Set[str]]] = []
    relation_rows: List[Tuple[int, Set[str]]] = []
    contributor_scopes: Set[int] = set()
    term_scopes: Dict[str, Set[int]] = {}
    best = {
        "hits": 0, "coverage": 0.0, "focus_hit": False,
        "bigram_hits": 0, "long_hits": 0, "lcs_ratio": 0.0,
        "synonym_hits": 0,
    }
    for scope_index, rows in enumerate(scope_rows):
        if rows:
            summary["scopes_with_rows"] += 1
        scope_support = False
        scope_contributes = False
        for row in rows[:EVIDENCE_SCAN_LIMIT]:
            text = row.get("text") or ""
            haystack = tokens(text)
            canon = [_canonical(token) for token in haystack]
            named = _named_terms(text)
            if asked_attributes:
                ask_hit = bool(
                    attribute_ids_from_tokens(haystack) & asked_attributes)
                if ask_hit:
                    relation_represented = True
                    relation_rows.append((scope_index, named))
            assessment = assess_row(content_terms, required_focus, text,
                                    document_scope=True, haystack=haystack,
                                    canon=canon)
            summary["scanned_rows"] += 1
            for key in ("hits", "bigram_hits", "long_hits", "synonym_hits"):
                best[key] = max(best[key], assessment[key])
            for key in ("coverage", "lcs_ratio"):
                best[key] = max(best[key], assessment[key])
            best["focus_hit"] = best["focus_hit"] or assessment["focus_hit"]
            memory_id = row.get("memory_id")
            if assessment["sufficient"]:
                scope_support = True
                supporting_named.append(named)
                supporting_links.append((scope_index, named))
                if memory_id is not None:
                    supporting.append(memory_id)
            matched = [term for term in content_terms
                       if _present(term, haystack, canon)]
            if len(matched) >= 2:
                scope_contributes = True
                contributor_scopes.add(scope_index)
                contributor_named.append(named)
                contributor_links.append((scope_index, named))
                if memory_id is not None and memory_id not in contributors:
                    contributors.append(memory_id)
                for term in matched:
                    term_scopes.setdefault(term, set()).add(scope_index)
        if scope_support:
            summary["scopes_with_evidence"] += 1
        if scope_contributes:
            summary["contributing_scopes"] += 1
    summary["relation_represented"] = relation_represented
    summary.update({
        "supporting_rows": len(supporting),
        "best_hits": best["hits"],
        "best_coverage": round(best["coverage"], 4),
        "best_focus_hit": bool(best["focus_hit"]),
        "best_bigram_hits": best["bigram_hits"],
        "best_long_hits": best["long_hits"],
        "best_lcs_ratio": round(best["lcs_ratio"], 4),
        "best_synonym_hits": best["synonym_hits"],
        "contributing_rows": len(contributors),
    })
    distinct_terms = [term for term in content_terms if term in term_scopes]
    coverage = (len(distinct_terms) / len(content_terms)
                if content_terms else 0.0)
    summary["union_coverage"] = round(coverage, 4)

    union_sufficient = bool(
        content_terms
        and coverage == 1.0
        and len(distinct_terms) >= _MIN_DISTINCT_HITS
        and len(contributors) >= 2
        and len(contributor_scopes) >= 2
    )
    supporting_ids = list(dict.fromkeys(supporting + (
        contributors if union_sufficient else [])))
    evidence_links = list(supporting_links)
    if union_sufficient:
        evidence_links.extend(contributor_links)
    relation_linked = not asked_attributes or any(
        rel_scope == ev_scope or bool(rel_named & ev_named)
        for rel_scope, rel_named in relation_rows
        for ev_scope, ev_named in evidence_links)
    summary["relation_linked"] = relation_linked
    if not content_terms and not required_focus:
        # A document scope names no entity; an empty question cannot be
        # verified against it.
        sufficient = False
        state = "insufficient"
        reason = "evidence_insufficient"
    elif supporting or union_sufficient:
        sufficient = True
        state = "supported"
        reason = None
    else:
        sufficient = False
        if (explicit_scope and summary["best_hits"] == 0
                and not summary["best_focus_hit"]):
            state = "wrong_scope"
            reason = "evidence_wrong_scope"
        else:
            state = "insufficient"
            reason = "evidence_insufficient"

    if sufficient and not relation_represented:
        # Distributed or single-scope content matches cannot answer a
        # relational question whose relation no requested scope expresses.
        sufficient = False
        state = "insufficient"
        reason = "evidence_relation_not_found"
        supporting_ids = []
    elif sufficient and not relation_linked:
        # The relation lives in another scope and shares no written name with
        # the evidence.  A repeated common noun ("the car") is not proof that
        # the two conversations mean the same object.
        sufficient = False
        state = "insufficient"
        reason = "evidence_relation_not_linked"
        supporting_ids = []
    elif sufficient:
        evidence_named = list(supporting_named)
        if union_sufficient:
            evidence_named.extend(contributor_named)
        if _referent_ambiguous(query_text, evidence_named):
            # The requested scopes answer "the <thing>" with different named
            # referents, so there is no single referent to join across.
            sufficient = False
            state = "insufficient"
            reason = "evidence_referent_ambiguous"
            supporting_ids = []

    return {
        "state": state,
        "reason": reason,
        "sufficient": sufficient,
        "scope_kind": "document",
        "cross_scope": union_sufficient and not supporting,
        "supporting_memory_ids": supporting_ids,
        "summary": summary,
    }
