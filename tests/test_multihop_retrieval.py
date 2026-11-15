"""Supported-shape contract for relational and contradiction-aware retrieval.

The engine answers from retrieved evidence; it does not perform relational
joins.  A question that names a relation ("who owns ...") is therefore settled
only when the scanned material actually expresses that relation:

* one stored relation, or several stored relations under one verified subject,
  answer directly;
* a relation distributed across caller-named scopes answers when both the
  premise and the asked relation are present, and those rows are returned;
* a relational question whose asked relation is absent from the material
  abstains instead of settling on the intermediate premise;
* two rows that answer "the <thing>" with different named referents make the
  question ambiguous, so it is not settled even though each row matches;
* contradiction in a stored attribute stays contested, never settled.

Unconstrained discovery is a candidate generator, not a join: it may find only
the premise scope, and the relation guard then abstains rather than presenting
the premise as the answer.
"""

from __future__ import annotations

from minemos_engine.answer import answer_contract
from tests._retrieval_helpers import FixedEncoder, RuleVerifier, make_memory


def _engine():
    return make_memory(
        capacity=64,
        verifier=RuleVerifier(),
        encoder=FixedEncoder(),
        strict_entity_identity=True,
    )


def _absorb(mem, subject, predicate, obj, timestamp=1.0):
    mem.absorb((subject, predicate, obj),
               timestamp=timestamp, confidence=1.0)


def _contract(outcome):
    return answer_contract(
        outcome["status"], outcome["reason"], outcome["temporal"],
        len(outcome["rows"]))


def _texts(outcome):
    return " ".join(row["text"] for row in outcome["rows"])


# ---------------------------------------------------------------------------
# Supported: stored relations answer directly
# ---------------------------------------------------------------------------


def test_direct_relation_under_a_verified_subject_settles():
    mem = _engine()
    _absorb(mem, "Alice", "works at", "Acme")
    _absorb(mem, "Alice", "lives in", "Lisbon")

    outcome = mem.eligible_query("Where does Alice work?", subject="Alice")
    assert outcome["status"] == "answered"
    assert outcome["evidence"]["state"] == "supported"
    assert outcome["evidence"]["summary"]["relation_represented"] is True
    assert _contract(outcome)["answer_is_settled"] is True

    # The attribute family is recognised through its other surface forms too.
    synonym = mem.eligible_query("Where does Alice reside?", subject="Alice")
    assert synonym["status"] == "answered"
    assert synonym["evidence"]["state"] == "supported"
    mem.close()


def test_two_stored_relations_under_one_entity_both_answer():
    mem = _engine()
    _absorb(mem, "Alice", "works at", "Acme")
    _absorb(mem, "Alice", "lives in", "Lisbon")

    outcome = mem.eligible_query("Where does Alice live and work?",
                                 subject="Alice")
    assert outcome["status"] == "answered"
    texts = _texts(outcome)
    assert "Acme" in texts and "Lisbon" in texts
    assert _contract(outcome)["answer_is_settled"] is True
    mem.close()


# ---------------------------------------------------------------------------
# Unsupported: the asked relation is not in the evidence
# ---------------------------------------------------------------------------


def test_entity_relation_without_the_stored_relation_abstains():
    mem = _engine()
    _absorb(mem, "Alice", "drives", "Volvo")

    outcome = mem.eligible_query("Who owns the car Alice drives?",
                                 subject="Alice")
    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "evidence_relation_not_found"
    assert outcome["evidence"]["state"] == "insufficient"
    assert outcome["evidence"]["summary"]["relation_represented"] is False
    assert outcome["rows"] == []
    assert _contract(outcome)["answer_is_settled"] is False
    mem.close()


def test_relational_question_without_a_scope_is_unverified():
    mem = _engine()
    _absorb(mem, "conversation:alpha", "segment_0", "Alice drives the Volvo")

    outcome = mem.eligible_query("Who owns the car Alice drives?")
    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "document_scope_required"
    assert outcome["rows"] == []
    mem.close()


def test_requested_scope_without_the_relation_abstains():
    mem = _engine()
    _absorb(mem, "conversation:alpha", "segment_0", "Alice drives the Volvo")
    _absorb(mem, "conversation:beta", "segment_0", "Bob owns the Volvo")

    premise_only = mem.eligible_query(
        "Who owns the Volvo that Alice drives?",
        scope_subjects=["conversation:alpha"])
    assert premise_only["status"] == "abstained"
    assert premise_only["reason"] == "evidence_relation_not_found"
    assert premise_only["rows"] == []
    mem.close()


# ---------------------------------------------------------------------------
# Supported: evidence aggregation across caller-named scopes
# ---------------------------------------------------------------------------


def test_relation_inside_one_document_scope_settles_with_both_rows():
    mem = _engine()
    _absorb(mem, "conversation:alpha", "segment_0", "Alice drives the Volvo")
    _absorb(mem, "conversation:alpha", "segment_1", "Bob owns the Volvo")

    outcome = mem.eligible_query("Who owns the car Alice drives?",
                                 subject="conversation:alpha")
    assert outcome["status"] == "answered"
    assert outcome["evidence"]["state"] == "supported"
    texts = _texts(outcome)
    assert "Alice drives" in texts and "Bob owns" in texts
    assert _contract(outcome)["answer_is_settled"] is True
    mem.close()


def test_distributed_relation_across_requested_scopes_settles():
    mem = _engine()
    _absorb(mem, "conversation:alpha", "segment_0", "Alice drives the Volvo")
    _absorb(mem, "conversation:beta", "segment_0", "Bob owns the Volvo")

    outcome = mem.eligible_query(
        "Who owns the Volvo that Alice drives?",
        scope_subjects=["conversation:alpha", "conversation:beta"])
    assert outcome["status"] == "answered"
    assert outcome["evidence"]["state"] == "supported"
    assert outcome["scopes_present"] == [
        "conversation:alpha", "conversation:beta"]
    texts = _texts(outcome)
    assert "Alice drives" in texts and "Bob owns" in texts
    assert _contract(outcome)["answer_is_settled"] is True
    mem.close()


def test_relation_row_stays_inside_the_returned_window():
    mem = _engine()
    _absorb(mem, "conversation:alpha", "segment_0", "Alice drives the Volvo")
    for index in range(1, 7):
        _absorb(mem, "conversation:alpha", f"segment_{index}",
                f"the harbour meeting covered topic {index}")
    _absorb(mem, "conversation:beta", "segment_0", "Bob owns the Volvo")

    outcome = mem.eligible_query(
        "Who owns the Volvo that Alice drives?",
        scope_subjects=["conversation:alpha", "conversation:beta"],
        top_k=2)
    assert outcome["status"] == "answered"
    texts = _texts(outcome)
    assert "Alice drives" in texts and "Bob owns" in texts
    mem.close()


# ---------------------------------------------------------------------------
# Unsupported: discovery does not join
# ---------------------------------------------------------------------------


def test_discovery_does_not_settle_on_the_premise_alone():
    mem = _engine()
    _absorb(mem, "conversation:alpha", "segment_0", "Alice drives the Volvo")
    _absorb(mem, "conversation:beta", "segment_0", "Bob owns the Volvo")

    outcome = mem.eligible_query(
        "Who owns the car Alice drives?", discover_scopes=True)
    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "evidence_relation_not_found"
    assert outcome["discovery"]["selected_scopes"] == 1
    assert outcome["rows"] == []
    mem.close()


# ---------------------------------------------------------------------------
# Unsupported: one definite description, two named referents
# ---------------------------------------------------------------------------


def test_two_named_referents_for_one_description_are_ambiguous():
    mem = _engine()
    _absorb(mem, "conversation:alpha", "segment_0", "Alice drives the car")
    _absorb(mem, "conversation:beta", "segment_0", "Bob owns the car")

    # Across scopes the common noun repeats but names nothing: an unrelated
    # conversation's "the car" is not evidence about this one's car.
    explicit = mem.eligible_query(
        "Who owns the car Alice drives?",
        scope_subjects=["conversation:alpha", "conversation:beta"])
    assert explicit["status"] == "abstained"
    assert explicit["reason"] == "evidence_relation_not_linked"
    assert explicit["rows"] == []
    assert _contract(explicit)["answer_is_settled"] is False

    # Discovery never selected the other scope, so the relation is simply
    # absent from the requested material.
    disposable = mem.eligible_query(
        "Who owns the car Alice drives?", discover_scopes=True)
    assert disposable["status"] == "abstained"
    assert disposable["reason"] == "evidence_relation_not_found"
    assert disposable["rows"] == []
    mem.close()


def test_contested_intermediate_relation_is_not_settled():
    mem = _engine()
    _absorb(mem, "conversation:alpha", "segment_0", "Alice drives the Volvo")
    _absorb(mem, "conversation:alpha", "segment_1", "Alice drives the Ford")
    _absorb(mem, "conversation:alpha", "segment_2", "Bob owns the Volvo")
    _absorb(mem, "conversation:alpha", "segment_3", "Carol owns the Ford")

    outcome = mem.eligible_query("Who owns the car Alice drives?",
                                 subject="conversation:alpha")
    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "evidence_referent_ambiguous"
    assert outcome["rows"] == []
    assert _contract(outcome)["answer_is_settled"] is False
    mem.close()


def test_definite_phrase_in_a_plain_content_question_still_answers():
    # "the migration plan" is the question's content, not a referent to be
    # resolved by a relation: two messages in one conversation may name
    # different projects and both still answer it.
    mem = _engine()
    _absorb(mem, "conversation:alpha", "segment_0",
            "the migration plan covers Azure rollout")
    _absorb(mem, "conversation:alpha", "segment_1",
            "the migration plan covers Contoso rollout")

    outcome = mem.eligible_query("What does the migration plan cover?",
                                 subject="conversation:alpha")
    assert outcome["status"] == "answered"
    assert outcome["reason"] is None
    assert outcome["evidence"]["state"] == "supported"
    assert outcome["evidence"]["summary"]["asked_attributes"] == []
    assert len(outcome["rows"]) == 2
    assert _contract(outcome)["answer_is_settled"] is True
    mem.close()


# ---------------------------------------------------------------------------
# Contradiction-aware: contested stored attributes stay contested
# ---------------------------------------------------------------------------


def test_contested_attribute_is_contested_not_settled():
    mem = _engine()
    _absorb(mem, "Alice", "works at", "Acme", 1.0)
    _absorb(mem, "Alice", "works at", "Globex", 2.0)

    outcome = mem.eligible_query("Where does Alice work?", subject="Alice")
    assert outcome["status"] == "answered"
    assert outcome["evidence"]["state"] == "contradicted"
    assert len(outcome["rows"]) == 2
    contract = _contract(outcome)
    assert contract["answer_status"] == "contested"
    assert contract["answer_is_settled"] is False
    mem.close()
