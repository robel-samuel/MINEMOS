"""Post-fix state-conflict policy: only proven equality keeps a pair settled.

The real-export audit found that two active values of one state predicate
("goal is") produced ``unresolved_state_temporal_conflict`` findings of zero
and a settled answer whenever the lexical gate could not *separate* the
values.  The conflict check reused the consolidation compatibility gate
("cannot prove different"), which is not equality.  Two active state values
the engine cannot tell apart are still unresolved, so the answer must not be
presented as settled.

These tests lock the corrected policy:

* non-provable-difference state values are reported as unresolved and the
  answer is not settled;
* lexically equivalent (typo/abbreviation) state values stay quiet;
* configured exclusive predicates keep their existing behaviour.
"""

from __future__ import annotations

from minemos_engine.black_hole_memory import (
    UNLINKED_EXCLUSIVE_CONFLICT,
    UNRESOLVED_STATE_CONFLICT,
)
from tests._retrieval_helpers import RuleVerifier, make_memory


def _engine():
    return make_memory(
        capacity=32,
        verifier=RuleVerifier(),
        strict_entity_identity=True,
    )


def _absorb(mem, subject, predicate, obj, timestamp):
    return mem.absorb((subject, predicate, obj),
                      timestamp=timestamp, confidence=1.0)


def test_unseparable_state_values_are_unresolved_not_settled():
    mem = _engine()
    _absorb(mem, "user", "goal is",
            "publish the retrieval benchmark results", 1.0)
    _absorb(mem, "user", "the goal is",
            "study distributed systems research", 2.0)

    # The lexical value gate cannot prove these values different; that is
    # exactly the case the removed compatibility fallback used to suppress.
    assert mem._values_compatible_for_merge(
        "publish the retrieval benchmark results",
        "study distributed systems research") is True
    assert mem._lexically_equivalent_value(
        "publish the retrieval benchmark results",
        "study distributed systems research") is False

    findings = mem.exclusive_temporal_conflicts()
    assert len(findings) == 1
    finding = findings[0]
    assert finding["reason"] == UNRESOLVED_STATE_CONFLICT
    assert finding["state_attribute"] == "goal"
    assert finding["confidence_scope"] == "state_predicate_form_only"
    assert finding["conflict_status"] == "unresolved"

    outcome = mem.eligible_query("What is user?")
    assert outcome["status"] == "answered"
    assert len(outcome["rows"]) == 2
    assert outcome["temporal"]["answer_is_settled"] is False


def test_unseparable_state_values_reach_eligible_query_temporal_flag():
    mem = _engine()
    _absorb(mem, "user", "goal is",
            "publish the retrieval benchmark results", 1.0)
    _absorb(mem, "user", "goal is",
            "study distributed systems research", 2.0)

    outcome = mem.eligible_query("What is user?", subject="user")
    assert outcome["temporal"]["answer_is_settled"] is False
    reasons = {
        item["reason"]
        for item in outcome["temporal"]["temporal_conflicts"]
    }
    assert UNRESOLVED_STATE_CONFLICT in reasons


def test_lexically_equivalent_state_values_are_not_reported():
    mem = _engine()
    _absorb(mem, "user", "goal is", "Nairobi", 1.0)
    _absorb(mem, "user", "the goal is", "Nairobii", 2.0)

    assert mem._lexically_equivalent_value("Nairobi", "Nairobii") is True
    findings = [
        finding for finding in mem.exclusive_temporal_conflicts()
        if finding["reason"] == UNRESOLVED_STATE_CONFLICT
    ]
    assert findings == []

    outcome = mem.eligible_query("What is user?")
    assert outcome["temporal"]["answer_is_settled"] is True


def test_exclusive_predicate_conflicts_are_unchanged():
    mem = _engine()
    _absorb(mem, "Ada", "lives in", "Addis Ababa", 1.0)
    _absorb(mem, "Ada", "lives in", "Nairobi", 2.0)

    assert mem.classify_predicate("lives in") == "exclusive"
    findings = mem.exclusive_temporal_conflicts()
    assert len(findings) == 1
    finding = findings[0]
    assert finding["reason"] == UNLINKED_EXCLUSIVE_CONFLICT
    assert finding["state_attribute"] is None
    assert finding["confidence_scope"] == "predicate_policy_only"

    outcome = mem.eligible_query("What is Ada?")
    assert outcome["temporal"]["answer_is_settled"] is False
