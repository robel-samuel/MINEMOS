"""R5.1 state-form coverage tests.

The independent R5 audit found that several common ways of stating a current
state of one subject were invisible to the state detector: verb relations
("is based in", "works in", "is married to"), marker/modifier variants
("phase is" vs "current phase is") and article variants ("goal is" vs "the
goal is").  Two incompatible active values were therefore presented as a
settled answer.

These tests lock the coverage fix.  They also guard the two failure modes a
wider lexicon can introduce: collapsing different relations into one attribute
("lives in" vs "works at") and reclassifying a predicate that already has an
explicit cardinality policy or is not a state form at all.

State conflicts keep using the engine's existing lexical value gate: two
values are reported only when the gate can separate them.  Fixtures therefore
use values the gate can prove distinct, so a test failure names a predicate
coverage problem rather than a value-comparison limitation.
"""

from __future__ import annotations

import pytest

from minemos_engine.black_hole_memory import (
    UNLINKED_EXCLUSIVE_CONFLICT,
    UNRESOLVED_STATE_CONFLICT,
)
from tests._retrieval_helpers import RuleVerifier, make_memory


def _engine(verifier=None):
    return make_memory(
        capacity=32,
        verifier=verifier or RuleVerifier(),
        strict_entity_identity=True,
    )


def _absorb(mem, subject, predicate, obj, timestamp):
    return mem.absorb((subject, predicate, obj),
                      timestamp=timestamp, confidence=1.0)


def _active_texts(mem):
    return sorted(
        mem.slot_texts.get(id(slot), slot.text or "")
        for slot in mem.memory.slots if slot.is_active
    )


def _conflicts(mem):
    return mem.exclusive_temporal_conflicts()


def _assert_contested(mem, subject, expected_key=None):
    assert len(_active_texts(mem)) == 2
    findings = _conflicts(mem)
    assert findings, "no conflict reported"
    finding = findings[0]
    if finding["reason"] == UNRESOLVED_STATE_CONFLICT:
        assert finding["state_attribute"] == expected_key
        assert finding["confidence_scope"] == "state_predicate_form_only"
    else:
        # Some relations already have an explicit cardinality policy and are
        # reported by the older exclusive-attribute detection.
        assert finding["reason"] == UNLINKED_EXCLUSIVE_CONFLICT
        assert finding["state_attribute"] is None
        assert finding["confidence_scope"] == "predicate_policy_only"
    outcome = mem.eligible_query(f"What is {subject}?")
    assert outcome["status"] == "answered"
    assert outcome["temporal"]["answer_is_settled"] is False
    assert len(outcome["rows"]) == 2


# ── verb-form state relations ─────────────────────────────────────────────


@pytest.mark.parametrize("predicate,key", [
    ("is based in", "location"),
    ("works in", "workplace"),
    ("is married to", "spouse"),
    ("lives in", "residence"),
    ("works at", "employer"),
    ("is working at", "employer"),
    ("is working in", "workplace"),
    ("is situated in", "location"),
    ("is residing in", "residence"),
])
def test_verb_state_relations_expose_incompatible_values(predicate, key):
    mem = _engine()
    _absorb(mem, "Ada", predicate, "first value", 1.0)
    _absorb(mem, "Ada", predicate, "second value", 2.0)
    _assert_contested(mem, "Ada", key)


@pytest.mark.parametrize("variant_a,variant_b,key", [
    ("based in", "is based in", "location"),
    ("works at", "works for", "employer"),
    ("lives in", "is living in", "residence"),
    ("married to", "is married to", "spouse"),
    ("works at", "is working at", "employer"),
    ("lives in", "currently lives in", "residence"),
    ("works at", "currently working at", "employer"),
])
def test_relation_surface_variants_share_one_attribute(variant_a, variant_b, key):
    # One variant may already carry an explicit policy ("works at" is
    # exclusive, "works for" is not) or the same relation may be spelled with
    # and without the copula.  Either way the pair must surface as a conflict,
    # whichever detection path reports it.
    mem = _engine()
    _absorb(mem, "Ada", variant_a, "first value", 1.0)
    _absorb(mem, "Ada", variant_b, "second value", 2.0)
    _assert_contested(mem, "Ada", key)


# ── marker, modifier and article normalization ────────────────────────────


def test_marker_keys_normalize_articles_and_neutral_modifiers():
    mem = _engine()
    assert mem._state_attribute_key("phase is") == "phase"
    assert mem._state_attribute_key("current phase is") == "phase"
    assert mem._state_attribute_key("current phase") == "phase"
    assert mem._state_attribute_key("the goal is") == "goal"
    assert mem._state_attribute_key("a goal is") == "goal"


def test_distinguishing_modifier_is_not_collapsed():
    mem = _engine()
    # "review" names a different attribute than a bare stage: the existing
    # suite asserts this and the coverage fix must not undo it.
    assert mem._state_attribute_key("review stage") == "review stage"
    # An explicit cardinality policy still wins over state detection.
    assert mem.classify_predicate("build status") == "exclusive"
    assert mem._state_attribute_key("build status") is None


def test_marker_modifier_variant_conflict_is_exposed():
    mem = _engine()
    _absorb(mem, "Saturn project", "phase is", "design", 1.0)
    _absorb(mem, "Saturn project", "current phase is", "build", 2.0)
    _assert_contested(mem, "Saturn project", "phase")


def test_article_variant_conflict_is_exposed():
    mem = _engine()
    _absorb(mem, "Saturn project", "goal is", "June", 1.0)
    _absorb(mem, "Saturn project", "the goal is", "July", 2.0)
    _assert_contested(mem, "Saturn project", "goal")


# ── non-collision guards ──────────────────────────────────────────────────


def test_different_state_relations_do_not_conflict():
    mem = _engine()
    _absorb(mem, "Ada", "lives in", "Paris", 1.0)
    _absorb(mem, "Ada", "works at", "Acme", 2.0)

    assert len(_active_texts(mem)) == 2
    assert _conflicts(mem) == []
    outcome = mem.eligible_query("What is Ada?")
    assert outcome["temporal"]["answer_is_settled"] is True


def test_non_state_predicates_remain_non_state():
    mem = _engine()
    _absorb(mem, "Ada", "uses", "tool one", 1.0)
    _absorb(mem, "Ada", "uses", "tool two", 2.0)

    assert mem._state_attribute_key("uses") is None
    assert _conflicts(mem) == []
    assert mem.eligible_query("What is Ada?")["temporal"][
        "answer_is_settled"] is True


def test_state_predicate_detection_includes_new_forms_only():
    mem = _engine()
    for predicate in ("is based in", "works in", "is married to",
                      "current phase is", "the goal is", "is working at",
                      "currently lives in", "currently working at"):
        assert mem._is_state_predicate(predicate) is True, predicate
    for predicate in ("works as", "uses", "advises", "analysis", "plays",
                      "no longer lives in", "not working at"):
        assert mem._is_state_predicate(predicate) is False, predicate
