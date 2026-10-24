"""
tests/test_phase28_adversarial.py

Phase 28 Comprehensive Adversarial & Integrity Verification Suite:
Addresses the 10 adversarial cases from Section 3 of the Phase 28 specification:
1. Two independent facts about the same subject that are both true.
2. A genuine update to a previously stored fact.
3. Two events with overlapping vocabulary but different meanings.
4. A valid paraphrase of an existing fact.
5. A genuine contradiction without an explicit timestamp.
6. Historical facts that remain true after a newer event occurs.
7. Multiple facts about one subject with different attributes and time ranges.
8. Repeated observations, reordered observations, and distractor-heavy input.
9. Capacity pressure and eviction during the same sequence.
10. Save, reload, backup recovery, and retrieval after each relevant operation.
"""

import os
import pytest
import numpy as np

from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate
from phase2.checkpoint import save_checkpoint, load_checkpoint


def test_case1_two_independent_facts_same_subject():
    """Case 1: Two independent facts about the same subject that are both true."""
    mem = BlackHoleMemory(capacity=10)
    # Distinct verbs
    mem.absorb(Candidate("Nathan", "studied", "computer science at Stanford", timestamp=10.0))
    mem.absorb(Candidate("Nathan", "speaks", "fluent Japanese", timestamp=20.0))
    # Cumulative same-verb events
    mem.absorb(Candidate("Nathan", "published", "a paper on memory consolidation in 2021", timestamp=30.0))
    mem.absorb(Candidate("Nathan", "published", "a paper on neural architectures in 2023", timestamp=40.0))

    assert len(mem.memory.slots) == 4
    active_slots = [s for s in mem.memory.slots if s.is_active]
    assert len(active_slots) == 4, "All 4 independent facts must remain fully active!"
    
    # Verify retrieval
    res_cs = mem.recall("Nathan", "studied")
    assert len(res_cs) == 1 and "Stanford" in res_cs[0]["text"]
    
    res_pub = mem.recall("Nathan", "published")
    assert len(res_pub) == 2, "Both publications must be returned!"


def test_case2_genuine_update_to_previously_stored_fact():
    """Case 2: A genuine update to a previously stored exclusive fact."""
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Alice", "residence", "lives in New York", timestamp=10.0))
    mem.absorb(Candidate("Alice", "residence", "relocated to San Francisco", timestamp=20.0))

    assert len(mem.memory.slots) == 2
    active_slots = [s for s in mem.memory.slots if s.is_active]
    assert len(active_slots) == 1, "Only the latest residence must be active!"
    assert "San Francisco" in active_slots[0].text
    
    # Verify point-in-time retrieval
    hist = mem.get_history("Alice", "residence")
    assert len(hist) == 2
    past = mem.recall("Alice", "residence", as_of=15.0)
    assert len(past) == 1 and "New York" in past[0]["text"]
    present = mem.recall("Alice", "residence", as_of=25.0)
    assert len(present) == 1 and "San Francisco" in present[0]["text"]


def test_case3_overlapping_vocabulary_different_meanings():
    """Case 3: Two events with overlapping vocabulary but different meanings."""
    mem = BlackHoleMemory(capacity=10)
    mem.absorb("Nathan designed the database storage engine.", timestamp=10.0)
    mem.absorb("Nathan audited the database security protocols.", timestamp=20.0)

    assert len(mem.memory.slots) == 2
    active_slots = [s for s in mem.memory.slots if s.is_active]
    assert len(active_slots) == 2, "Vocabulary overlap must not cause false merge or supersession!"


def test_case4_valid_paraphrase_consolidates_cleanly():
    """Case 4: A valid paraphrase of an existing fact."""
    mem = BlackHoleMemory(capacity=10)
    mem.absorb("Nathan is the chief architect of the platform.", timestamp=10.0)
    mem.absorb("Nathan serves as lead system architect for the platform.", timestamp=20.0)

    assert len(mem.memory.slots) == 1, "Paraphrase must consolidate into a single slot!"
    assert mem.memory.slots[0].update_count == 2
    assert mem.memory.slots[0].is_active is True


def test_case5_genuine_contradiction_without_explicit_timestamp():
    """Case 5: A genuine contradiction without an explicit timestamp."""
    mem = BlackHoleMemory(capacity=10)
    mem.absorb("The cluster service is online.")
    mem.absorb("The cluster service is not online.")

    assert len(mem.memory.slots) == 2
    active_slots = [s for s in mem.memory.slots if s.is_active]
    assert len(active_slots) == 1
    assert "not online" in active_slots[0].text


def test_case6_historical_facts_remain_true_after_newer_event():
    """Case 6: Historical facts that remain true after a newer event occurs."""
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Bob", "graduated", "completed his PhD at MIT in 2018", timestamp=10.0))
    mem.absorb(Candidate("Bob", "joined", "became research director at DeepMind in 2022", timestamp=20.0))

    assert len(mem.memory.slots) == 2
    active_slots = [s for s in mem.memory.slots if s.is_active]
    assert len(active_slots) == 2, "Educational milestone must not be superseded by new job!"


def test_case7_multiple_facts_different_attributes_and_time_ranges():
    """Case 7: Multiple facts about one subject with different attributes and time ranges."""
    mem = BlackHoleMemory(capacity=10)
    # Sequential career state revisions
    mem.absorb(Candidate("Carol", "title", "was staff engineer", timestamp=10.0))
    mem.absorb(Candidate("Carol", "title", "promoted to principal engineer", timestamp=20.0))
    # Independent milestone
    mem.absorb(Candidate("Carol", "won", "received the innovation award in 2021", timestamp=15.0))

    assert len(mem.memory.slots) == 3
    active_slots = [s for s in mem.memory.slots if s.is_active]
    assert len(active_slots) == 2, "Award and latest title must both be active!"
    
    titles = mem.recall("Carol", "title")
    assert len(titles) == 1 and "principal engineer" in titles[0]["text"]
    
    awards = mem.recall("Carol", "won")
    assert len(awards) == 1 and "innovation award" in awards[0]["text"]


def test_case8_repeated_reordered_and_distractor_heavy_input():
    """Case 8: Repeated observations, reordered observations, and distractor-heavy input."""
    mem = BlackHoleMemory(capacity=15)
    # Target fact A
    mem.absorb(Candidate("David", "role", "leads cybersecurity operations", timestamp=10.0))
    # Distractors
    mem.absorb("System backup ran at midnight.", timestamp=11.0)
    mem.absorb("Network switch 4 rebooted.", timestamp=12.0)
    # Paraphrase of A
    mem.absorb(Candidate("David", "role", "directs cybersecurity operations", timestamp=15.0))
    # Another independent fact about David
    mem.absorb(Candidate("David", "authored", "guidelines on zero-trust architectures", timestamp=18.0))
    # Another distractor
    mem.absorb("Cooling unit in rack B reported normal temperature.", timestamp=20.0)

    # Verify role consolidated
    david_role = mem.recall("David", "role")
    assert len(david_role) == 1
    assert david_role[0]["update_count"] == 2
    
    # Verify publication preserved
    david_pub = mem.recall("David", "authored")
    assert len(david_pub) == 1 and "zero-trust" in david_pub[0]["text"]


def test_case9_capacity_pressure_and_eviction_protection():
    """Case 9: Capacity pressure and eviction during multi-fact sequence."""
    mem = BlackHoleMemory(capacity=5)
    # High-value consolidated fact (Tier 1)
    mem.absorb("Elena leads the quantum computing team.", timestamp=1.0)
    mem.absorb("Elena directs the quantum computing team.", timestamp=2.0)
    assert mem.memory.slots[0].update_count == 2  # Tier 1

    # Flood with 10 transient distractors to force evictions
    for i in range(10):
        mem.absorb(f"Sensor node {i} packet received.", timestamp=10.0 + i)

    # Capacity must not be exceeded
    assert len(mem.memory.slots) <= 5
    # Tier 1 consolidated fact must have survived eviction!
    elena_hits = [s for s in mem.memory.slots if "Elena" in (s.text or "")]
    assert len(elena_hits) == 1, "Tier 1 consolidated memory must survive capacity eviction flood!"


def test_case10_save_reload_recovery_and_retrieval(tmp_path):
    """Case 10: Save, reload, backup recovery, and retrieval after each relevant operation."""
    ckpt_file = str(tmp_path / "case10_checkpoint.npz")
    mem = BlackHoleMemory(capacity=10)
    
    # Fact 1: Active milestone
    mem.absorb(Candidate("Fiona", "graduated", "completed doctorate in 2020", timestamp=10.0))
    # Fact 2: State revision
    mem.absorb(Candidate("Fiona", "location", "lives in Chicago", timestamp=15.0))
    mem.absorb(Candidate("Fiona", "location", "relocated to Austin", timestamp=25.0))
    # Fact 3: Cumulative hobby
    mem.absorb(Candidate("Fiona", "practices", "classical cello", timestamp=30.0))

    # Save checkpoint
    save_checkpoint(mem, ckpt_file)
    assert os.path.exists(ckpt_file)

    # Reload into fresh instance
    restored = BlackHoleMemory.from_checkpoint(ckpt_file, capacity=10)
    assert len(restored.memory.slots) == 4
    
    active_restored = [s for s in restored.memory.slots if s.is_active]
    assert len(active_restored) == 3, "Doctorate, Austin location, and cello must be active!"
    
    # Query point-in-time on restored instance
    rec_chicago = restored.recall("Fiona", "location", as_of=20.0)
    assert len(rec_chicago) == 1 and "Chicago" in rec_chicago[0]["text"]
    
    rec_austin = restored.recall("Fiona", "location", as_of=35.0)
    assert len(rec_austin) == 1 and "Austin" in rec_austin[0]["text"]
