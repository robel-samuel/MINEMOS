"""
MINEMOS Production Readiness - Task 1: Contradiction-Handling Safety Tests

Tests covering the 10 required regression test categories (A through J):
A. Same subject, different predicates/events (independent events coexist).
B. Same subject and predicate, contradictory values (genuine state update).
C. Out-of-order observations with timestamps.
D. Paraphrased observations of an existing active memory (consolidation).
E. Multiple independent facts about a single entity.
F. Point-in-time retrieval before and after a genuine update.
G. False-contradiction stress test with distractor sentences.
H. Rapid consecutive updates to the same fact.
I. Long-range query survival after many unrelated observations.
J. Direct query accuracy comparing pre-fix and post-fix behavior.
"""

import pytest
from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate


# ─────────────────────────────────────────────────────────────────────────────
# Category A: Same subject, different predicates/events
# ─────────────────────────────────────────────────────────────────────────────

def test_category_a_same_subject_different_predicates():
    """
    Observation A: 'Seraphina relocated her laboratory to Svalbard.'
    Observation B: 'Seraphina recorded an observation packet.'
    Independent events involving the same entity MUST coexist as active memories.
    Observation B must NOT invalidate Observation A.
    """
    mem = BlackHoleMemory(capacity=10)
    r1 = mem.absorb(Candidate("Seraphina", "relocated her laboratory to", "Svalbard", timestamp=100.0))
    r2 = mem.absorb(Candidate("Seraphina", "recorded an observation", "packet", timestamp=200.0))

    assert r1["action"] == "insert"
    assert r2["action"] == "insert", f"Expected insert but got {r2['action']}"

    # Both slots must be active
    slots = mem.memory.slots
    assert len(slots) == 2
    assert slots[0].is_active, "Slot 0 (laboratory location) was deactivated!"
    assert slots[0].valid_until is None, "Slot 0 valid_until was terminated inappropriately!"
    assert slots[1].is_active, "Slot 1 (observation packet) is not active!"

    # Retrieval of laboratory location returns Svalbard
    lab_recall = mem.recall("Seraphina", "relocated her laboratory to")
    assert len(lab_recall) == 1
    assert "Svalbard" in lab_recall[0]["text"]

    # Retrieval of observation packet returns the packet
    obs_recall = mem.recall("Seraphina", "recorded an observation")
    assert len(obs_recall) == 1
    assert "packet" in obs_recall[0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Category B: Same subject and predicate, contradictory values (genuine update)
# ─────────────────────────────────────────────────────────────────────────────

def test_category_b_genuine_state_update():
    """
    Genuine state change regarding the same entity and attribute must update
    the target fact correctly via temporal_update.
    """
    mem = BlackHoleMemory(capacity=10)
    r1 = mem.absorb(Candidate("Alice", "resides in the capital of", "France", timestamp=100.0))
    r2 = mem.absorb(Candidate("Alice", "does not reside in the capital of", "France", timestamp=200.0))

    assert r1["action"] == "insert"
    assert r2["action"] == "temporal_update"

    slot1 = mem.memory.slots[0]
    slot2 = mem.memory.slots[1]

    # Slot 1 terminated at t=200.0
    assert not slot1.is_active
    assert slot1.valid_until == 200.0
    assert slot1.superseded_by == id(slot2)

    # Slot 2 active from t=200.0
    assert slot2.is_active
    assert slot2.valid_from == 200.0
    assert slot2.valid_until is None
    assert slot2.supersedes == id(slot1)

    # Active recall returns only the current state
    active = mem.recall("Alice")
    assert len(active) == 1
    assert "does not reside" in active[0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Category C: Out-of-order observations with timestamps
# ─────────────────────────────────────────────────────────────────────────────

def test_category_c_out_of_order_timestamps():
    """
    An observation with an earlier timestamp (historical) must not supersede
    a newer active fact.
    """
    mem = BlackHoleMemory(capacity=10)
    # Newer observation arrives first: Bob at t=200 moved to London
    r1 = mem.absorb(Candidate("Bob", "moved his residence to", "London", timestamp=200.0))
    # Historical observation arrives later: Bob at t=100 lived in Boston
    r2 = mem.absorb(Candidate("Bob", "moved his residence to", "Boston", timestamp=100.0))

    slot1 = mem.memory.slots[0]
    # The newer fact (London at t=200) must NOT be terminated by the older fact (t=100)
    assert slot1.is_active
    assert slot1.valid_until is None


# ─────────────────────────────────────────────────────────────────────────────
# Category D: Paraphrased observations of an existing active memory
# ─────────────────────────────────────────────────────────────────────────────

def test_category_d_paraphrase_consolidation():
    """
    Paraphrased observations of an active memory must consolidate rather
    than creating false contradictions or duplicate slots.
    """
    mem = BlackHoleMemory(capacity=10)
    r1 = mem.absorb(Candidate("Marcus", "authored an influential", "biography monograph", timestamp=100.0))
    r2 = mem.absorb(Candidate("Marcus", "wrote an important", "biographical treatise", timestamp=150.0))

    assert r1["action"] == "insert"
    assert r2["action"] == "consolidate"
    assert len(mem.memory.slots) == 1
    slot = mem.memory.slots[0]
    assert slot.is_active
    assert slot.update_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Category E: Multiple independent facts about a single entity
# ─────────────────────────────────────────────────────────────────────────────

def test_category_e_multiple_independent_facts_single_entity():
    """
    An entity can possess multiple independent facts/events simultaneously.
    None of them should invalidate the others.
    """
    mem = BlackHoleMemory(capacity=10)
    facts = [
        ("Elena", "completed her doctorate in", "astrophysics", 10.0),
        ("Elena", "was appointed director of the", "observatory", 20.0),
        ("Elena", "discovered a periodic", "variable star", 30.0),
        ("Elena", "received the international", "astronomy prize", 40.0),
    ]

    for subj, pred, obj, t in facts:
        res = mem.absorb(Candidate(subj, pred, obj, timestamp=t))
        assert res["action"] == "insert"

    # All 4 facts must remain active
    assert len(mem.memory.slots) == 4
    for slot in mem.memory.slots:
        assert slot.is_active
        assert slot.valid_until is None

    # Querying each fact specifically returns the respective fact
    recall_doc = mem.recall("Elena", "completed her doctorate in")
    assert len(recall_doc) == 1 and "astrophysics" in recall_doc[0]["text"]

    recall_dir = mem.recall("Elena", "was appointed director of the")
    assert len(recall_dir) == 1 and "observatory" in recall_dir[0]["text"]

    recall_star = mem.recall("Elena", "discovered a periodic")
    assert len(recall_star) == 1 and "variable star" in recall_star[0]["text"]

    recall_prize = mem.recall("Elena", "received the international")
    assert len(recall_prize) == 1 and "astronomy prize" in recall_prize[0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Category F: Point-in-time retrieval before and after genuine update
# ─────────────────────────────────────────────────────────────────────────────

def test_category_f_point_in_time_retrieval():
    """
    Point-in-time retrieval must accurately recover historical facts prior to
    a revision and the updated fact after the revision.
    """
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Victor", "owns the majority share in", "Apex Robotics", timestamp=50.0))
    mem.absorb(Candidate("Victor", "sold all equity in", "Apex Robotics", timestamp=150.0))

    # Before update (t=100.0) -> historical fact active
    hist = mem.recall("Victor", as_of=100.0)
    assert len(hist) == 1
    assert "owns the majority share" in hist[0]["text"]

    # After update (t=200.0) -> new fact active
    curr = mem.recall("Victor", as_of=200.0)
    assert len(curr) == 1
    assert "sold all equity" in curr[0]["text"]

    # Full history chronological
    full_hist = mem.get_history("Victor")
    assert len(full_hist) == 2
    assert full_hist[0]["valid_from"] == 50.0
    assert full_hist[1]["valid_from"] == 150.0


# ─────────────────────────────────────────────────────────────────────────────
# Category G: False-contradiction stress test with distractor sentences
# ─────────────────────────────────────────────────────────────────────────────

def test_category_g_distractor_stress_test():
    """
    A core target memory must not be invalidated by a stream of distractor
    observations involving the same entity but different topics.
    """
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("David", "developed a quantum encryption", "protocol", timestamp=10.0))

    distractors = [
        ("David", "attended a symposium on", "numerical analysis", 20.0),
        ("David", "submitted an expense report for", "laboratory equipment", 30.0),
        ("David", "mentored two graduate", "researchers", 40.0),
        ("David", "reviewed a manuscript on", "lattice cryptography", 50.0),
        ("David", "delivered a keynote on", "computational complexity", 60.0),
    ]

    for subj, pred, obj, t in distractors:
        res = mem.absorb(Candidate(subj, pred, obj, timestamp=t))
        assert res["action"] == "insert", f"Distractor triggered {res['action']} instead of insert!"

    # Target fact must remain active
    target_slot = mem.memory.slots[0]
    assert target_slot.is_active
    assert target_slot.valid_until is None

    # Query for the target fact must successfully retrieve it
    results = mem.query("quantum encryption protocol", top_k=3)
    assert len(results) > 0
    assert results[0]["subject"] == "David"
    assert "quantum encryption" in results[0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Category H: Rapid consecutive updates to the same fact
# ─────────────────────────────────────────────────────────────────────────────

def test_category_h_rapid_consecutive_updates():
    """
    Rapid sequential state updates to the same attribute must form a clean,
    intact supersession chain where only the latest state is active.
    """
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Clara", "occupied the office suite in", "Building A", timestamp=100.0))
    mem.absorb(Candidate("Clara", "relocated her office suite to", "Building B", timestamp=110.0))
    mem.absorb(Candidate("Clara", "relocated her office suite to", "Building C", timestamp=120.0))

    slots = mem.memory.slots
    assert len(slots) == 3

    # Building A: [100.0, 110.0), superseded by Building B
    assert not slots[0].is_active
    assert slots[0].valid_until == 110.0
    assert slots[0].superseded_by == id(slots[1])

    # Building B: [110.0, 120.0), superseded by Building C
    assert not slots[1].is_active
    assert slots[1].valid_until == 120.0
    assert slots[1].supersedes == id(slots[0])
    assert slots[1].superseded_by == id(slots[2])

    # Building C: [120.0, None), active
    assert slots[2].is_active
    assert slots[2].valid_from == 120.0
    assert slots[2].valid_until is None
    assert slots[2].supersedes == id(slots[1])

    # Only 1 active recall for Clara's office
    active = mem.recall("Clara", "relocated her office suite to")
    assert len(active) == 1
    assert "Building C" in active[0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Category I: Long-range query survival after many unrelated observations
# ─────────────────────────────────────────────────────────────────────────────

def test_category_i_long_range_query_survival():
    """
    A fact about an entity must survive in memory and remain accurately retrievable
    after many intervening observations about other entities.
    """
    mem = BlackHoleMemory(capacity=30)
    mem.absorb(Candidate("Seraphina", "relocated her laboratory to", "Svalbard", timestamp=10.0))

    for i in range(15):
        mem.absorb(Candidate(f"Entity_{i:02d}", "published findings regarding", f"experiment_{i:02d}", timestamp=20.0 + i))

    # Seraphina's fact must still be active and queryable
    results = mem.query("Where did Seraphina relocate her laboratory?", top_k=3)
    assert len(results) > 0
    assert results[0]["subject"] == "Seraphina"
    assert "Svalbard" in results[0]["text"]
    assert results[0]["is_active"]


# ─────────────────────────────────────────────────────────────────────────────
# Category J: Direct query accuracy comparing pre-fix and post-fix behavior
# ─────────────────────────────────────────────────────────────────────────────

def test_category_j_direct_query_accuracy():
    """
    Verifies that querying for Seraphina's laboratory location returns the
    actual laboratory location, NOT an unrelated observation packet.
    """
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Seraphina", "relocated her laboratory to", "Svalbard", timestamp=100.0))
    mem.absorb(Candidate("Seraphina", "recorded an observation", "packet", timestamp=200.0))

    # Query for the laboratory location
    lab_results = mem.query("Where did Seraphina relocate her laboratory?", top_k=2)
    assert len(lab_results) > 0
    assert lab_results[0]["subject"] == "Seraphina"
    assert "Svalbard" in lab_results[0]["text"]
    assert lab_results[0]["is_active"]

    # Query for observation packet
    packet_results = mem.query("What observation packet was recorded?", top_k=2)
    assert len(packet_results) > 0
    assert "packet" in packet_results[0]["text"]
    assert packet_results[0]["is_active"]
