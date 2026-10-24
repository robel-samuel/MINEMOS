"""
MINEMOS Production Readiness - Task 2: Adversarial Verification of Contradiction Safety

Adversarial test suite targeting:
- Attribute matching vulnerabilities (copula predicates, token collisions, paraphrased predicates).
- NLI filtering vulnerabilities (asymmetry, malformed predictions, missing fields).
- Temporal invariant enforcement (timestamp 0.0, out-of-order timestamps, validity intervals).
- Contradiction anchor isolation (rejected contradictions must not corrupt anchors).
- Historical retrieval and consolidation integrity under stress.
"""

import pytest
import numpy as np
from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attack Attribute Matching (Section 2)
# ─────────────────────────────────────────────────────────────────────────────

def test_adversarial_generic_copula_predicates_do_not_collide():
    """
    Generic predicates ('is', 'was', 'has', 'states') carry no domain specificity.
    Observations sharing a generic predicate but differing in attribute/domain
    must NOT be classified as the same attribute or supersede each other.
    """
    mem = BlackHoleMemory(capacity=10)
    r1 = mem.absorb(Candidate("Alice", "is", "a certified surgeon", timestamp=10.0))
    r2 = mem.absorb(Candidate("Alice", "is", "35 years old", timestamp=20.0))
    r3 = mem.absorb(Candidate("Alice", "is", "fluent in Japanese", timestamp=30.0))

    # All three are distinct attributes of Alice; all three must remain active
    assert len(mem.memory.slots) == 3
    for s in mem.memory.slots:
        assert s.is_active
        assert s.valid_until is None

    # Check distinct recall
    surgeon_rec = mem.recall("Alice", "is")
    # All three share predicate 'is' but have distinct texts
    texts = [r["text"] for r in surgeon_rec]
    assert any("certified surgeon" in t for t in texts)
    assert any("35 years old" in t for t in texts)
    assert any("fluent in Japanese" in t for t in texts)


def test_adversarial_shared_words_different_attributes():
    """
    Different attributes that share common words (e.g., 'won the ...')
    must not collide or supersede each other.
    """
    mem = BlackHoleMemory(capacity=10)
    r1 = mem.absorb(Candidate("Bob", "won the national marathon", "championship", timestamp=10.0))
    r2 = mem.absorb(Candidate("Bob", "won the annual computer science", "award", timestamp=20.0))

    assert r1["action"] == "insert"
    assert r2["action"] == "insert"

    slots = mem.memory.slots
    assert len(slots) == 2
    assert slots[0].is_active and slots[1].is_active


def test_adversarial_different_events_similar_wording():
    """
    Different events with similar wording (e.g. repeated visits in different months)
    must coexist as distinct active memories.
    """
    mem = BlackHoleMemory(capacity=10)
    r1 = mem.absorb(Candidate("Carol", "attended the research symposium in", "Boston", timestamp=10.0))
    r2 = mem.absorb(Candidate("Carol", "attended the annual workshop in", "Boston", timestamp=20.0))

    assert r1["action"] == "insert"
    assert r2["action"] == "insert"
    assert mem.memory.slots[0].is_active
    assert mem.memory.slots[1].is_active


def test_adversarial_different_entities_same_predicate():
    """
    Different entities sharing the exact same predicate and object
    must never supersede each other or corrupt each other's anchors.
    """
    mem = BlackHoleMemory(capacity=10)
    r1 = mem.absorb(Candidate("Alice", "serves as executive director of", "Apex Labs", timestamp=10.0))
    r2 = mem.absorb(Candidate("Bob", "serves as executive director of", "Apex Labs", timestamp=20.0))

    assert r1["action"] == "insert"
    assert r2["action"] == "insert"
    assert mem.memory.slots[0].is_active
    assert mem.memory.slots[1].is_active
    assert mem.memory.slots[0].debug_subject == "Alice"
    assert mem.memory.slots[1].debug_subject == "Bob"

    slot0_id = id(mem.memory.slots[0])
    assert len(mem.slot_contra_anchors.get(slot0_id, [])) == 0


def test_adversarial_materially_distinct_values():
    """
    State change differing only in value (e.g. room assignment)
    must execute a clean temporal update when temporal ordering holds.
    """
    mem = BlackHoleMemory(capacity=10)
    r1 = mem.absorb(Candidate("Frank", "is assigned to office suite", "Room 401A", timestamp=100.0))
    r2 = mem.absorb(Candidate("Frank", "is assigned to office suite", "Room 401B", timestamp=200.0))

    # Room 401B updates Frank's office suite
    slot0 = mem.memory.slots[0]
    slot1 = mem.memory.slots[1]

    # Pre-fix/current check: does it update properly?
    active_facts = mem.recall("Frank", "is assigned to office suite")
    assert len(active_facts) == 1
    assert "Room 401B" in active_facts[0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Attack Asymmetric NLI Filtering (Section 3)
# ─────────────────────────────────────────────────────────────────────────────

def test_adversarial_malformed_nli_predictions():
    """
    Verifies that malformed, incomplete, or missing NLI fields (e.g., fwd=None,
    missing confidence, missing probabilities) fail safely without unhandled
    AttributeError or KeyError exceptions, and without corrupting state.
    """
    class MalformedVerifier:
        def verify_batch(self, pairs, **kwargs):
            return [{
                "decision": "CONTRADICTION",
                "confidence": None,
                "fwd": None,  # Malformed: None instead of dict
                "rev": None,
            }]

    mem = BlackHoleMemory(capacity=10, verifier=MalformedVerifier())
    r1 = mem.absorb(Candidate("TestSubject", "relocated to", "CityAlpha", timestamp=10.0))
    # Should not raise AttributeError: 'NoneType' object has no attribute 'get'
    r2 = mem.absorb(Candidate("TestSubject", "relocated to", "CityBeta", timestamp=20.0))

    assert r1 is not None
    assert r2 is not None


def test_adversarial_rejected_contradiction_does_not_corrupt_anchors():
    """
    When an observation is flagged as contradiction against an unrelated slot
    (different attribute or different entity), it must NOT add a contradiction
    anchor to that unrelated slot.
    """
    class MockContradictionVerifier:
        def verify_batch(self, pairs, **kwargs):
            return [{
                "decision": "CONTRADICTION",
                "confidence": 0.99,
                "p_contra": 0.99,
                "p_neutral": 0.01,
                "fwd": {"p_contra": 0.99, "p_neutral": 0.01},
                "rev": {"p_contra": 0.99, "p_neutral": 0.01}
            }]

    mem = BlackHoleMemory(capacity=10, verifier=MockContradictionVerifier())
    # Slot 0: Alice lives in Paris
    mem.absorb(Candidate("Alice", "lives in", "Paris", timestamp=10.0))
    slot0_id = id(mem.memory.slots[0])

    # Observation about a completely different attribute: Alice works at Acme
    mem.absorb(Candidate("Alice", "works at", "Acme", timestamp=20.0))

    # Slot 0 must NOT have a contradiction anchor from an unrelated attribute!
    assert mem.slot_contra_counts.get(slot0_id, 0) == 0, "Slot 0 contra_count was corrupted by unrelated attribute!"
    assert len(mem.slot_contra_anchors.get(slot0_id, [])) == 0, "Slot 0 anchor list corrupted by unrelated attribute!"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Verify Temporal Invariants (Section 4)
# ─────────────────────────────────────────────────────────────────────────────

def test_adversarial_timestamp_zero_not_overridden():
    """
    A Candidate with timestamp=0.0 must be stored with valid_from=0.0,
    NOT overridden by wall-clock time (time.time()). Subsequent updates
    with timestamp=10.0 must correctly supersede it.
    """
    mem = BlackHoleMemory(capacity=10)
    r1 = mem.absorb(Candidate("Alice", "resides in the capital of", "France", timestamp=0.0))
    assert mem.memory.slots[0].valid_from == 0.0, f"Expected valid_from=0.0 but got {mem.memory.slots[0].valid_from}"

    r2 = mem.absorb(Candidate("Alice", "does not reside in the capital of", "France", timestamp=10.0))
    assert r2["action"] == "temporal_update", f"Expected temporal_update but got {r2['action']}"

    assert not mem.memory.slots[0].is_active
    assert mem.memory.slots[0].valid_until == 10.0
    assert mem.memory.slots[1].is_active
    assert mem.memory.slots[1].valid_from == 10.0


def test_adversarial_repeated_observations_preserve_validity():
    """
    Re-observing an existing active fact (consolidation) increments update_count
    and leaves valid_until as None (does not close the validity window).
    """
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Marcus", "authored an influential", "biography monograph", timestamp=10.0))
    assert mem.memory.slots[0].valid_until is None

    mem.absorb(Candidate("Marcus", "wrote an important", "biographical treatise", timestamp=20.0))
    slot = mem.memory.slots[0]
    assert slot.is_active
    assert slot.valid_until is None
    assert slot.update_count == 2


def test_adversarial_out_of_order_stream_does_not_corrupt_newer():
    """
    When observations arrive out of chronological order:
    A fact valid from t=200 cannot be terminated by a historical claim with t=100.
    """
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Zara", "resides in", "Tokyo", timestamp=200.0))
    mem.absorb(Candidate("Zara", "resides in", "Osaka", timestamp=100.0))

    # Tokyo (t=200) was newer and must remain active
    active = mem.recall("Zara", "resides in")
    assert len(active) == 1
    assert "Tokyo" in active[0]["text"]
    assert active[0]["valid_from"] == 200.0
    assert active[0]["valid_until"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Verify Retrieval and Consolidation (Section 5)
# ─────────────────────────────────────────────────────────────────────────────

def test_adversarial_eviction_distinguished_from_supersession():
    """
    Eviction (due to capacity bounds) pops slots out of MemoryState.slots.
    Supersession (due to temporal factual update) preserves slots in MemoryState.slots
    as historical entries with is_active=False and valid_until set.
    """
    capacity = 3
    mem = BlackHoleMemory(capacity=capacity, retention_policy="tiered")

    # Ingest 2 observations about EntityA (one updating the other)
    mem.absorb(Candidate("EntityA", "status is", "active", timestamp=10.0))
    mem.absorb(Candidate("EntityA", "status is", "dormant", timestamp=20.0))

    # Currently 2 slots: slot 0 (superseded, historical), slot 1 (active)
    assert len(mem.memory.slots) == 2
    assert not mem.memory.slots[0].is_active
    assert mem.memory.slots[1].is_active

    # Now fill capacity and force an eviction
    mem.absorb(Candidate("EntityB", "location is", "SiteBeta", timestamp=30.0))
    assert len(mem.memory.slots) == 3

    # Adding a 4th slot forces an eviction
    mem.absorb(Candidate("EntityC", "location is", "SiteGamma", timestamp=40.0))
    assert len(mem.memory.slots) == 3
    assert mem.total_evictions >= 1


def test_adversarial_historical_point_in_time_lookup_integrity():
    """
    Point-in-time retrieval across multiple updates:
    t=50:  Phase 1 active
    t=150: Phase 2 active
    t=250: Phase 3 active
    """
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Saturn", "project phase is", "Phase One Alpha", timestamp=50.0))
    mem.absorb(Candidate("Saturn", "project phase is", "Phase Two Beta", timestamp=150.0))
    mem.absorb(Candidate("Saturn", "project phase is", "Phase Three Gamma", timestamp=250.0))

    # as_of=100 -> Phase One
    p1 = mem.recall("Saturn", as_of=100.0)
    assert len(p1) == 1 and "Phase One" in p1[0]["text"]

    # as_of=200 -> Phase Two
    p2 = mem.recall("Saturn", as_of=200.0)
    assert len(p2) == 1 and "Phase Two" in p2[0]["text"]

    # as_of=300 -> Phase Three
    p3 = mem.recall("Saturn", as_of=300.0)
    assert len(p3) == 1 and "Phase Three" in p3[0]["text"]
