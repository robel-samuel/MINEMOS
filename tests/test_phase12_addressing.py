import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from benchmarks.phase12_dataset import generate_dataset
from benchmarks.phase12_addressing import run_system, system_A_absorb, system_B_absorb, system_C_absorb
from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState


def test_dataset_deterministic():
    d1, _ = generate_dataset()
    d2, _ = generate_dataset()
    assert len(d1) == len(d2)
    for a, b in zip(d1, d2):
        assert (a.concept_id, a.category, a.wording_role,
                a.candidate.subject, a.candidate.predicate, a.candidate.object) == \
               (b.concept_id, b.category, b.wording_role,
                b.candidate.subject, b.candidate.predicate, b.candidate.object)


def test_all_three_ordering_cases_present():
    obs, specs = generate_dataset()
    cases = set(s.ordering_case for s in specs)
    assert cases == {"canonical_first", "paraphrase_first", "mixed"}


def test_rekeying_does_not_create_new_slot():
    mem = MemoryState(capacity=10, dim=64)
    c1 = Candidate("X", "uses", "Y", timestamp=1)
    c2 = Candidate("X", "regularly programs in", "Y", timestamp=2)
    r1 = system_B_absorb(mem, c1, None, dim=64)
    n_before = len(mem.slots)
    r2 = system_B_absorb(mem, c2, r1["slot"], dim=64)
    assert len(mem.slots) == n_before
    assert r2["action"] == "update"


def test_alias_does_not_create_new_slot_or_concept():
    mem = MemoryState(capacity=10, dim=64)
    alias_store = {}
    c1 = Candidate("X", "uses", "Y", timestamp=1)
    c2 = Candidate("X", "regularly programs in", "Y", timestamp=2)
    r1 = system_C_absorb(mem, c1, None, alias_store, dim=64)
    n_before = len(mem.slots)
    r2 = system_C_absorb(mem, c2, r1["slot"], alias_store, dim=64)
    assert len(mem.slots) == n_before
    assert len(alias_store) == 1
    assert len(alias_store[id(r1["slot"])]) == 1


def test_key_frozen_for_system_A_but_not_B():
    memA = MemoryState(capacity=10, dim=64)
    c1 = Candidate("X", "uses", "Y", timestamp=1)
    c2 = Candidate("X", "regularly programs in", "Y", timestamp=2)
    r1 = system_A_absorb(memA, c1, None, dim=64)
    key_before = r1["slot"].key.copy()
    system_A_absorb(memA, c2, r1["slot"], dim=64)
    assert np.array_equal(key_before, r1["slot"].key)

    memB = MemoryState(capacity=10, dim=64)
    r1b = system_B_absorb(memB, c1, None, dim=64)
    key_before_b = r1b["slot"].key.copy()
    system_B_absorb(memB, c2, r1b["slot"], dim=64)
    assert not np.array_equal(key_before_b, r1b["slot"].key)


def test_addressing_fairness_identical_admission_across_systems():
    obs, specs = generate_dataset()
    obs = obs[:400]
    rA = run_system(obs, capacity=500, absorb_fn="A")
    rB = run_system(obs, capacity=500, absorb_fn="B")
    rC = run_system(obs, capacity=500, absorb_fn="C")
    assert len(rA["memory"].slots) == len(rB["memory"].slots) == len(rC["memory"].slots)
    assert rA["n_updates"] == rB["n_updates"] == rC["n_updates"]
    assert rA["n_inserts"] == rB["n_inserts"] == rC["n_inserts"]


def test_contradiction_never_shares_slot_with_base_concept():
    obs, specs = generate_dataset()
    for sysname in ["A", "B", "C"]:
        r = run_system(obs, capacity=500, absorb_fn=sysname)
        slot_map = r["concept_to_slot"]
        for spec in specs[:10]:
            base_slot = slot_map.get(spec.concept_id)
            contra_slot = slot_map.get(f"{spec.concept_id}_contra")
            if base_slot is not None and contra_slot is not None:
                assert base_slot is not contra_slot, f"system {sysname}: {spec.concept_id} contradiction merged"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
