"""Numeric value checks for verified entity facts.

``evidence_terms`` keeps only tokens of three or more characters, so a number
such as "51" never took part in the support decision.  For a verified entity
that produced a false settled answer: the query "Is Alice 51 years old?"
against the stored fact ``Alice age 42`` was reported as supported, because
the only term that distinguished the two facts had been discarded.

A verified entity scope now treats a number in the question as a literal
value: at least one returned row of that subject must carry it.  Entity
scopes are the only place this applies.  In a document/conversation scope an
utterance's number is frequently a changed parameter rather than a claim the
transcript contains ("do that again for chapter 5" over notes about chapter
3), and the frozen follow-up evaluation measured six recall losses out of
forty when the rule was applied there; that boundary is recorded in the
issue register as a limitation, not silently widened.

Textual overlap still decides everything else; no threshold was lowered.
"""

from __future__ import annotations

from minemos_engine.evidence import assess_evidence, assess_row
from tests._retrieval_helpers import RuleVerifier, make_memory


def _engine(capacity: int = 64):
    return make_memory(capacity=capacity, verifier=RuleVerifier(),
                       strict_entity_identity=True)


def _absorb(mem, subject, predicate, text, timestamp=1.0):
    return mem.absorb((subject, predicate, text),
                      timestamp=timestamp, confidence=1.0)


# ---------------------------------------------------------------------------
# Row-level mechanism
# ---------------------------------------------------------------------------


def test_a_row_with_a_different_number_is_not_evidence():
    mismatch = assess_row(["stage", "inspection"], [], "stage 42 inspection",
                          document_scope=True, required_terms=["51"])
    assert mismatch["sufficient"] is False

    match = assess_row(["stage", "inspection"], [], "stage 51 inspection",
                       document_scope=True, required_terms=["51"])
    assert match["sufficient"] is True


def test_one_of_the_question_numbers_is_enough():
    row = assess_row(["stage", "inspection"], [], "stage 51 inspection",
                     document_scope=True, required_terms=["51", "52"])
    assert row["sufficient"] is True


# ---------------------------------------------------------------------------
# Entity evidence decision
# ---------------------------------------------------------------------------


def test_entity_question_about_a_number_rejects_the_stored_other_value():
    rows = [{"memory_id": "m1", "text": "Alice age 42."}]
    wrong = assess_evidence("Is Alice 51 years old?", "alice 51 years", rows,
                            scope_kind="entity", explicit_scope=True)
    assert wrong["sufficient"] is False
    assert wrong["state"] == "insufficient"

    right = assess_evidence("Is Alice 42 years old?", "alice 42 years", rows,
                            scope_kind="entity", explicit_scope=True)
    assert right["sufficient"] is True
    assert right["supporting_memory_ids"] == ["m1"]


def test_question_without_a_number_still_answers_from_the_entity():
    rows = [{"memory_id": "m1", "text": "Alice age 42."}]
    outcome = assess_evidence("How old is Alice?", "alice", rows,
                              scope_kind="entity", explicit_scope=True)
    assert outcome["sufficient"] is True


def test_missing_number_produces_no_supporting_rows():
    rows = [{"memory_id": "m1", "text": "Alice age 42."}]
    outcome = assess_evidence("Is Alice 51?", "alice 51", rows,
                              scope_kind="entity", explicit_scope=True)
    assert outcome["sufficient"] is False
    assert outcome["supporting_memory_ids"] == []


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


def test_engine_entity_scope_rejects_a_different_stored_number():
    mem = _engine()
    try:
        _absorb(mem, "Alice", "age", "42")

        wrong = mem.eligible_query("Is Alice 51 years old?", subject="Alice")
        assert wrong["status"] == "abstained"
        assert wrong["reason"] == "evidence_insufficient"
        assert wrong["rows"] == []

        right = mem.eligible_query("Is Alice 42 years old?", subject="Alice")
        assert right["status"] == "answered"
        assert [row["text"] for row in right["rows"]] == ["Alice age 42."]
        assert right["temporal"]["answer_is_settled"] is True

        plain = mem.eligible_query("How old is Alice?", subject="Alice")
        assert plain["status"] == "answered"
        assert plain["temporal"]["answer_is_settled"] is True
    finally:
        mem.close()


def test_engine_number_inside_the_verified_subject_still_answers():
    mem = _engine()
    try:
        _absorb(mem, "Room 51", "status", "ready")

        outcome = mem.eligible_query("What is Room 51?", subject="Room 51")
        assert outcome["status"] == "answered"
        assert [row["text"] for row in outcome["rows"]] == [
            "Room 51 status ready."]
    finally:
        mem.close()
