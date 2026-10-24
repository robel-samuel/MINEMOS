"""
tests/test_phase25.py

Validation and Integration Test Suite for MINEMOS Phase 25:
Unified Black Hole Memory with Hybrid Retrieval, Calibrated Semantic Verification,
Drift-Resistant Consolidation, Temporal Fact Management, and Tiered Retention.
"""

import pytest
import numpy as np

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState, MemorySlot
from phase2.black_hole_memory import BlackHoleMemory
from benchmarks.phase18_experiment import D_LEX


# ──────────────────────────────────────────────
#  1. Basic Absorption & Lifecycle
# ──────────────────────────────────────────────

def test_black_hole_memory_initialization():
    """Verifies that BlackHoleMemory initializes cleanly with bounded capacity."""
    mem = BlackHoleMemory(capacity=50, dim=D_LEX)
    status = mem.status()
    assert status["capacity"] == 50
    assert status["total_slots"] == 0
    assert status["active_slots"] == 0


def test_single_observation_absorption():
    """Absorbing a single candidate creates an active provisional slot."""
    mem = BlackHoleMemory(capacity=10)
    cand = Candidate("Helena", "directed the regional", "orchestral ensemble", timestamp=100.0)
    res = mem.absorb(cand)

    assert res["action"] == "insert"
    assert res["update_count"] == 1
    assert res["valid_from"] == 100.0
    assert res["valid_until"] is None

    status = mem.status()
    assert status["active_slots"] == 1
    assert status["provisional_slots"] == 1
    assert status["consolidated_slots"] == 0


# ──────────────────────────────────────────────
#  2. Consolidation of Equivalent Information
# ──────────────────────────────────────────────

def test_semantic_paraphrase_consolidation():
    """
    Absorbing a valid semantic paraphrase routes to the existing slot,
    increments update_count, and consolidates representations without drift.
    """
    mem = BlackHoleMemory(capacity=10)
    # 1. Canonical observation
    cand1 = Candidate("Helena", "directed the regional", "orchestral ensemble", timestamp=100.0)
    res1 = mem.absorb(cand1)
    slot_id1 = res1["slot_id"]

    # 2. Semantic paraphrase
    cand2 = Candidate("Helena", "conducted the local", "symphony orchestra", timestamp=110.0)
    res2 = mem.absorb(cand2)

    assert res2["action"] == "consolidate"
    assert res2["slot_id"] == slot_id1
    assert res2["update_count"] == 2

    # Status reflects transition to consolidated tier
    status = mem.status()
    assert status["active_slots"] == 1
    assert status["provisional_slots"] == 0
    assert status["consolidated_slots"] == 1


# ──────────────────────────────────────────────
#  3. Changing Facts & Temporal Updates
# ──────────────────────────────────────────────

def test_temporal_fact_revision():
    """
    When an observation contradicts an active fact for the same entity,
    the old slot is closed (valid_until set), and a new active slot is created.
    """
    mem = BlackHoleMemory(capacity=10)
    t1 = 100.0
    t2 = 200.0

    # Fact 1: Alice lives in Paris
    cand1 = Candidate("Alice", "resides in the capital of", "France", timestamp=t1)
    res1 = mem.absorb(cand1)
    slot1_id = res1["slot_id"]

    # Fact 2: Alice moved away / no longer resides in Paris
    cand2 = Candidate("Alice", "does not reside in the capital of", "France", timestamp=t2)
    res2 = mem.absorb(cand2)

    assert res2["action"] == "temporal_update"
    slot2_id = res2["slot_id"]
    assert slot2_id != slot1_id

    # Inspect slot 1 (historical)
    slot1 = [s for s in mem.memory.slots if id(s) == slot1_id][0]
    assert not slot1.is_active
    assert slot1.valid_until == t2
    assert slot1.superseded_by == slot2_id

    # Inspect slot 2 (active)
    slot2 = [s for s in mem.memory.slots if id(s) == slot2_id][0]
    assert slot2.is_active
    assert slot2.valid_from == t2
    assert slot2.valid_until is None
    assert slot2.supersedes == slot1_id


def test_point_in_time_temporal_query():
    """Point-in-time recall correctly retrieves historical vs current facts."""
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Victor", "owns the majority share in", "Apex Robotics", timestamp=50.0))
    mem.absorb(Candidate("Victor", "sold all equity in", "Apex Robotics", timestamp=150.0))

    # Current recall (active only) returns the current state
    active_facts = mem.recall("Victor")
    assert len(active_facts) == 1
    assert "sold all equity" in active_facts[0]["text"]

    # Historical recall as of t=100.0 returns the earlier state
    past_facts = mem.recall("Victor", as_of=100.0)
    assert len(past_facts) == 1
    assert "owns the majority share" in past_facts[0]["text"]

    # History returns both in chronological order
    history = mem.get_history("Victor")
    assert len(history) == 2
    assert history[0]["valid_from"] == 50.0
    assert history[1]["valid_from"] == 150.0


# ──────────────────────────────────────────────
#  4. Rejection of Dangerous Merges
# ──────────────────────────────────────────────

def test_contradiction_rejection():
    """Contradictory observations are NEVER falsely merged into active factual slots."""
    mem = BlackHoleMemory(capacity=10)
    cand_true = Candidate("Nathan", "serves as the lead", "database architect", timestamp=10.0)
    res_true = mem.absorb(cand_true)

    cand_false = Candidate("Nathan", "does not serve as the lead", "database architect", timestamp=20.0)
    res_false = mem.absorb(cand_false)

    # Must NOT consolidate into slot 0
    assert res_false["action"] != "consolidate"
    assert res_false["slot_id"] != res_true["slot_id"]


# ──────────────────────────────────────────────
#  5. Tiered Retention & Stream Stability
# ──────────────────────────────────────────────

def test_tiered_retention_protects_consolidated_under_pressure():
    """
    Under capacity pressure, uncorroborated provisional singletons are evicted
    first, while consolidated memories are retained.

    Tiered eviction only checks slot.update_count, not NLI state. We set
    update_count=2 directly after the first insert to simulate prior consolidation,
    keeping this test NLI-independent and deterministic.
    """
    capacity = 4
    mem = BlackHoleMemory(capacity=capacity, retention_policy="tiered")

    # 1. Insert Marcus and mark it as consolidated (update_count = 2)
    mem.absorb(Candidate("Marcus", "authored an influential", "biography monograph", timestamp=1.0))
    marcus_slot = next(s for s in mem.memory.slots if s.debug_subject == "Marcus")
    marcus_slot.update_count = 2  # simulate a prior paraphrase merge

    # 2. Insert 3 distractor singletons (update_count = 1 each)
    mem.absorb(Candidate("DistractorA", "occurred in", "region A", timestamp=3.0))
    mem.absorb(Candidate("DistractorB", "occurred in", "region B", timestamp=4.0))
    mem.absorb(Candidate("DistractorC", "occurred in", "region C", timestamp=5.0))

    # Memory is now full (4 slots: 1 consolidated + 3 provisional)
    assert len(mem.memory.slots) == capacity

    # 3. Insert DistractorD -> forces tiered eviction (should evict a distractor)
    mem.absorb(Candidate("DistractorD", "occurred in", "region D", timestamp=6.0))

    # The consolidated Marcus slot MUST survive
    target_survived = any(s.debug_subject == "Marcus" and s.update_count >= 2 for s in mem.memory.slots)
    assert target_survived is True
    assert mem.total_evictions == 1


# ──────────────────────────────────────────────
#  6. High-Level Natural Language Query API
# ──────────────────────────────────────────────

def test_natural_language_query_retrieval():
    """Verifies that query() retrieves relevant memory slots by semantic similarity."""
    mem = BlackHoleMemory(capacity=10)
    mem.absorb("Julian engineered a redundant power distribution grid.")
    mem.absorb("Amara discovered a prehistoric fossilized specimen.")

    results = mem.query("electrical power grid engineering", top_k=1)
    assert len(results) == 1
    assert "Julian" in results[0]["text"]
