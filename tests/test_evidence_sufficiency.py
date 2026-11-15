"""Evidence-sufficiency regressions for the retrieval-to-answer decision.

The 2026-11-09 baseline answered every deliberately wrong-scope document
query as settled: the eligibility gate verified the asserted conversation and
the dense ranker returned rows from it, but nothing checked that the rows
contained the question's content.  These tests lock the correction:

* scope equality is not relevance -- a document scope must contain the
  query's substantive content before its rows may be presented;
* similarity is supporting evidence, never proof -- a high-similarity
  encoder cannot make an unrelated row answer a question;
* contradictory evidence is retained and never settled;
* verified entity subjects still answer attribute questions without extra
  query content, and multi-valued facts stay retrievable;
* every non-supporting evidence state maps to a non-settled answer contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from minemos_engine.answer import answer_contract
from minemos_engine.evidence import assess_evidence
from tests._retrieval_helpers import (
    FixedEncoder,
    RuleVerifier,
    make_memory,
    unit_vector,
)

HIGH_SIM = unit_vector(20261110)


class HighSimEncoder:
    """Every statement maps to one vector so the dense gate always passes."""

    def encode(self, text: str) -> np.ndarray:
        return HIGH_SIM.copy()


def _engine(strict: bool = True, encoder=None):
    return make_memory(
        capacity=32,
        verifier=RuleVerifier(),
        encoder=encoder or FixedEncoder(),
        strict_entity_identity=strict,
    )


def _absorb(mem, subject, predicate, obj, timestamp=1.0):
    return mem.absorb((subject, predicate, obj),
                      timestamp=timestamp, confidence=1.0)


def _document_memory(encoder=None):
    mem = _engine(encoder=encoder)
    _absorb(mem, "conversation:alpha", "segment_0",
            "we adopted the lexington budget")
    _absorb(mem, "conversation:beta", "segment_0",
            "we discussed the harbor schedule")
    return mem


def _contract(outcome):
    return answer_contract(
        outcome["status"], outcome["reason"], outcome["temporal"],
        len(outcome["rows"]))


# ---------------------------------------------------------------------------
# Document scope: membership is not relevance
# ---------------------------------------------------------------------------


def test_matching_document_content_is_supported_and_settled():
    mem = _document_memory()
    outcome = mem.eligible_query(
        "What did we adopt in the lexington budget?",
        subject="conversation:alpha")

    assert outcome["status"] == "answered"
    assert outcome["evidence"]["state"] == "supported"
    assert outcome["evidence"]["scope_kind"] == "document"
    assert outcome["evidence"]["supporting_memory_ids"]
    assert [row["subject"] for row in outcome["rows"]] == [
        "conversation:alpha"]
    assert "lexington" in outcome["rows"][0]["text"]
    assert _contract(outcome)["answer_status"] == "settled"
    mem.close()


def test_wrong_document_scope_abstains_without_rows():
    mem = _document_memory()
    outcome = mem.eligible_query(
        "What did we decide about the harbor lease?",
        subject="conversation:alpha")

    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "evidence_wrong_scope"
    assert outcome["evidence"]["state"] == "wrong_scope"
    assert outcome["evidence"]["supporting_memory_ids"] == []
    assert outcome["rows"] == []
    contract = _contract(outcome)
    assert contract["answer_status"] == "abstained"
    assert contract["answer_is_settled"] is False
    assert contract["safe_to_display_as_settled"] is False
    mem.close()


def test_unsupported_topic_in_a_real_document_scope_abstains():
    mem = _document_memory()
    outcome = mem.eligible_query(
        "What is the capital of France?", subject="conversation:alpha")

    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "evidence_wrong_scope"
    assert outcome["rows"] == []
    assert _contract(outcome)["answer_is_settled"] is False
    mem.close()


def test_partial_content_match_is_insufficient_not_supported():
    mem = _document_memory()
    outcome = mem.eligible_query(
        "What did we adopt about the lexington harbor lease?",
        subject="conversation:alpha")

    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "evidence_insufficient"
    assert outcome["evidence"]["state"] == "insufficient"
    # Two query terms did occur; partial overlap is still not an answer.
    assert outcome["evidence"]["summary"]["best_hits"] == 2
    assert outcome["rows"] == []
    mem.close()


def test_explicit_scope_never_borrows_the_other_conversations_content():
    mem = _document_memory()
    outcome = mem.eligible_query(
        "What did we adopt in the lexington budget?",
        subject="conversation:beta")

    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "evidence_wrong_scope"
    assert outcome["rows"] == []

    correct = mem.eligible_query(
        "What did we discuss about the harbor schedule?",
        subject="conversation:beta")
    assert correct["status"] == "answered"
    assert "harbor" in correct["rows"][0]["text"]
    mem.close()


def test_high_similarity_cannot_make_an_unrelated_row_evidence():
    # Every statement has identical dense similarity to the query, so the
    # dense gate cannot distinguish support from scope membership.
    mem = _document_memory(encoder=HighSimEncoder())
    wrong = mem.eligible_query(
        "What did we decide about the harbor lease?",
        subject="conversation:alpha")
    assert wrong["status"] == "abstained"
    assert wrong["reason"] == "evidence_wrong_scope"

    right = mem.eligible_query(
        "What did we adopt in the lexington budget?",
        subject="conversation:alpha")
    assert right["status"] == "answered"
    mem.close()


def test_assess_evidence_ignores_similarity_entirely():
    rows = [{
        "memory_id": "m1",
        "similarity": 0.999,
        "text": "we discussed the harbor schedule",
    }]
    outcome = assess_evidence(
        "What did we decide about the lexington lease?",
        None, rows, scope_kind="document", explicit_scope=True)

    assert outcome["state"] == "wrong_scope"
    assert outcome["sufficient"] is False
    assert outcome["supporting_memory_ids"] == []


# ---------------------------------------------------------------------------
# Entity scope: verified identity answers; unrelated content does not
# ---------------------------------------------------------------------------


def test_entity_subject_without_extra_content_is_supported():
    mem = _engine()
    _absorb(mem, "Ada Lovelace", "likes", "tea")

    outcome = mem.eligible_query("What does Ada Lovelace like?")
    assert outcome["status"] == "answered"
    assert outcome["evidence"]["state"] == "supported"
    assert outcome["evidence"]["scope_kind"] == "entity"
    assert _contract(outcome)["answer_status"] == "settled"
    mem.close()


def test_entity_attribute_not_stored_abstains_instead_of_answering():
    mem = _engine()
    _absorb(mem, "Lucian Dunmore", "works at", "the observatory")

    outcome = mem.eligible_query("Where does Lucian Dunmore live?")
    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "attribute_not_stored"
    assert outcome["evidence"]["state"] == "missing"
    assert outcome["rows"] == []
    assert _contract(outcome)["answer_is_settled"] is False
    mem.close()


def test_multi_valued_facts_are_both_retained_and_settled():
    mem = _engine()
    _absorb(mem, "Ada Lovelace", "likes", "tea")
    _absorb(mem, "Ada Lovelace", "likes", "coffee")

    outcome = mem.eligible_query("What does Ada Lovelace like?")
    assert outcome["status"] == "answered"
    texts = " ".join(row["text"] for row in outcome["rows"])
    assert "tea" in texts and "coffee" in texts
    assert outcome["evidence"]["state"] == "supported"
    assert _contract(outcome)["answer_status"] == "settled"
    mem.close()


def test_unresolved_state_conflict_is_contested_and_retained():
    mem = _engine()
    _absorb(mem, "user", "goal is",
            "publish the retrieval benchmark results", 1.0)
    _absorb(mem, "user", "goal is",
            "study distributed systems research", 2.0)

    outcome = mem.eligible_query("What is user?")
    assert outcome["status"] == "answered"
    assert outcome["evidence"]["state"] == "contradicted"
    assert len(outcome["rows"]) == 2
    assert outcome["temporal"]["answer_is_settled"] is False
    contract = _contract(outcome)
    assert contract["answer_status"] == "contested"
    assert contract["answer_is_settled"] is False
    assert contract["safe_to_display_as_settled"] is False
    mem.close()


def test_typo_document_query_still_finds_supporting_content():
    mem = _document_memory()
    outcome = mem.eligible_query(
        "What did we adopt in the lexingotn budgte?",
        subject="conversation:alpha")

    assert outcome["status"] == "answered"
    assert outcome["evidence"]["state"] == "supported"
    mem.close()


# ---------------------------------------------------------------------------
# Question framing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame", [
    "remind me about ",
    "what is the situation with ",
    "can you explain ",
])
def test_natural_question_framing_does_not_dilute_single_term_evidence(frame):
    """A meta-request word must not turn a one-term question into two.

    The frozen coverage probe found windows that settle on their own but
    abstain behind a natural frame.  Framing names the act of asking, so its
    words are not content and cannot lower coverage or the hit count.
    """
    mem = _document_memory()
    outcome = mem.eligible_query(frame + "lexington",
                                 subject="conversation:alpha")

    assert outcome["status"] == "answered"
    assert outcome["evidence"]["state"] == "supported"
    assert _contract(outcome)["answer_is_settled"] is True
    mem.close()


def test_question_frame_preserves_the_document_focus_path():
    """Framing can defeat focus extraction and erase the only support path.

    Three short terms with full coverage, no adjacent bigram and no long-term
    hit are supporting only through the focus conjunction.  The framed query
    extracts "remind me" or nothing as its focus, so without a fallback it
    abstains on evidence identical to the bare query.
    """
    mem = _engine()
    _absorb(mem, "conversation:alpha", "segment_0", "the top box put")

    bare = mem.eligible_query("box top put", subject="conversation:alpha")
    assert bare["status"] == "answered"
    assert bare["evidence"]["summary"]["best_focus_hit"] is True

    for frame in ("remind me about ", "what is the situation with "):
        outcome = mem.eligible_query(frame + "box top put",
                                     subject="conversation:alpha")
        assert outcome["status"] == "answered"
        assert outcome["evidence"]["state"] == "supported"
        assert outcome["evidence"]["summary"]["best_hits"] == 3
        assert _contract(outcome)["answer_is_settled"] is True
    mem.close()


def test_framing_words_alone_are_not_evidence():
    mem = _engine()
    _absorb(mem, "conversation:alpha", "segment_0",
            "remind them about the situation")

    outcome = mem.eligible_query("remind me about the situation",
                                 subject="conversation:alpha")

    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "evidence_insufficient"
    assert outcome["evidence"]["state"] == "insufficient"
    assert outcome["evidence"]["summary"]["content_terms"] == 0
    assert _contract(outcome)["answer_is_settled"] is False
    mem.close()


# ---------------------------------------------------------------------------
# Missing, ambiguous and unverified evidence
# ---------------------------------------------------------------------------


def test_absent_subject_reports_missing_evidence():
    mem = _engine()
    _absorb(mem, "Lucian Dunmore", "lives in", "Lisbon")

    outcome = mem.eligible_query("What is Nobody Here's salary?")
    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "subject_not_found"
    assert outcome["evidence"]["state"] == "missing"
    assert outcome["evidence"]["supporting_memory_ids"] == []
    mem.close()


def test_shared_name_reports_ambiguous_evidence():
    mem = _engine()
    _absorb(mem, "Rowan Dunmore", "is allergic to", "peanuts")
    _absorb(mem, "Rowan Ainsley", "is allergic to", "shellfish")

    outcome = mem.eligible_query("Which allergen affects Rowan")
    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "subject_ambiguous"
    assert outcome["evidence"]["state"] == "ambiguous"
    assert outcome["rows"] == []
    assert _contract(outcome)["answer_status"] == "ambiguous"
    mem.close()


def test_indirect_query_reports_ambiguous_evidence_and_unverified_status():
    mem = _engine()
    _absorb(mem, "Lucian Dunmore", "lives in", "Lisbon")

    outcome = mem.eligible_query("Which city has the best weather?")
    assert outcome["status"] == "unverified"
    assert outcome["evidence"]["state"] == "ambiguous"
    assert _contract(outcome)["answer_status"] == "unverified"
    assert _contract(outcome)["answer_is_settled"] is False
    mem.close()


# ---------------------------------------------------------------------------
# Point-in-time retrieval and the caller threshold
# ---------------------------------------------------------------------------


def test_as_of_never_returns_future_evidence():
    mem = _engine()
    _absorb(mem, "Ada Lovelace", "studies", "physics", 10.0)
    _absorb(mem, "Ada Lovelace", "studies", "mathematics", 20.0)

    before = mem.eligible_query("What does Ada Lovelace study?", as_of=5.0)
    assert before["rows"] == []
    assert before["status"] in ("abstained", "answered")
    assert _contract(before)["answer_is_settled"] is False

    middle = mem.eligible_query("What does Ada Lovelace study?", as_of=15.0)
    assert [row["text"] for row in middle["rows"]] == [
        "Ada Lovelace studies physics."]
    assert middle["evidence"]["state"] == "supported"

    latest = mem.eligible_query("What does Ada Lovelace study?", as_of=25.0)
    assert {row["text"] for row in latest["rows"]} == {
        "Ada Lovelace studies physics.",
        "Ada Lovelace studies mathematics.",
    }
    mem.close()


def test_caller_threshold_failure_reports_insufficient_evidence():
    mem = _engine()
    _absorb(mem, "Lucian Dunmore", "lives in", "Lisbon")

    answered = mem.eligible_query("Where does Lucian Dunmore live?")
    threshold = max(0.0, min(1.0, answered["top_similarity"] + 0.01))
    gated = mem.eligible_query(
        "Where does Lucian Dunmore live?", min_similarity=threshold)

    assert gated["status"] == "abstained"
    assert gated["reason"] == "low_similarity"
    assert gated["evidence"]["state"] == "insufficient"
    assert gated["evidence"]["reason"] == "low_similarity"
    assert gated["rows"] == []
    assert _contract(gated)["answer_is_settled"] is False
    mem.close()


# ---------------------------------------------------------------------------
# Contract transitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,reason,answer_status",
    [
        ("abstained", "evidence_wrong_scope", "abstained"),
        ("abstained", "evidence_insufficient", "abstained"),
        ("abstained", "evidence_irrelevant", "abstained"),
        ("abstained", "subject_not_found", "abstained"),
        ("abstained", "subject_ambiguous", "ambiguous"),
        ("unverified", "subject_not_identified", "unverified"),
        ("disabled", "safety_profile_disabled", "ungated"),
    ],
)
def test_non_supporting_outcomes_are_never_settled(
        status, reason, answer_status):
    temporal = {"answer_is_settled": True, "conflict_status": "none"}
    contract = answer_contract(status, reason, temporal, 0)

    assert contract["answer_status"] == answer_status
    assert contract["answer_is_settled"] is False
    assert contract["safe_to_display_as_settled"] is False


def test_answered_without_rows_is_empty_and_not_settled():
    temporal = {"answer_is_settled": True, "conflict_status": "none"}
    contract = answer_contract("answered", None, temporal, 0)

    assert contract["answer_status"] == "empty"
    assert contract["answer_is_settled"] is False


def test_evidence_block_carries_no_row_text():
    mem = _document_memory()
    outcome = mem.eligible_query(
        "What did we decide about the harbor lease?",
        subject="conversation:alpha")

    evidence = outcome["evidence"]
    assert set(evidence) == {
        "state", "reason", "sufficient", "scope_kind",
        "supporting_memory_ids", "summary"}
    assert "rows" not in evidence
    serialized = repr(evidence)
    for row in ("lexington budget", "harbor schedule"):
        assert row not in serialized
    mem.close()
