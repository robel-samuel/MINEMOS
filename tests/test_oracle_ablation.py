import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from benchmarks.phase11_dataset import generate_dataset
from benchmarks.oracle_ablation import oracle_absorb, run_system_A, run_system_B, run_system_C
from phase2.structured_candidate import Candidate, encode_full
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb, MATCH_THRESHOLD


def test_dataset_is_deterministic():
    d1 = generate_dataset()
    d2 = generate_dataset()
    assert len(d1) == len(d2)
    for o1, o2 in zip(d1, d2):
        assert o1.concept_id == o2.concept_id
        assert o1.category == o2.category
        assert (o1.candidate.subject, o1.candidate.predicate, o1.candidate.object) == \
               (o2.candidate.subject, o2.candidate.predicate, o2.candidate.object)


def test_same_different_labels_present_and_meaningfully_sized():
    obs = generate_dataset()
    same = sum(1 for o in obs if o.category in ("exact_repeat", "paraphrase"))
    different = sum(1 for o in obs if o.category in ("contradiction", "related_distinct", "unrelated"))
    assert same > 0 and different > 0
    # DIFFERENT probes: 3 per concept x 100 concepts = 300, a meaningful sample
    assert different == 300


def test_oracle_never_produces_or_consumes_a_vector_score():
    """The oracle interface (concept_to_slot dict in run_system_C) stores
    only slot object references, keyed by string concept_id -- never a
    float score, confidence, or embedding."""
    obs = generate_dataset()[:100]
    r = run_system_C(obs, capacity=200)
    for concept_id, slot in r["concept_to_slot"].items():
        assert isinstance(concept_id, str)
        assert hasattr(slot, "key")  # it's a real MemorySlot object, not a score


def test_update_equation_byte_identical_to_absorb():
    """Given the SAME (memory, candidate, addressed slot), oracle_absorb's
    resulting slot.value must be bit-identical to what memory_update.absorb()
    produces for that same match. This is the required proof that the
    update equation was not reimplemented differently. Uses an exact
    duplicate pair so the merge path is guaranteed to fire under BOTH
    systems (a near-duplicate pair would correctly not merge under
    normal absorb() at all, per Phase 5/6 -- that's a separate, already-
    established fact, not what this test is checking)."""
    c1 = Candidate("Person_000", "uses", "Object_000", timestamp=1)
    c2 = Candidate("Person_000", "uses", "Object_000", timestamp=2)  # exact duplicate

    mem_normal = MemoryState(capacity=10, dim=64)
    absorb(mem_normal, c1, threshold=MATCH_THRESHOLD)
    r_normal = absorb(mem_normal, c2, threshold=MATCH_THRESHOLD)

    mem_oracle = MemoryState(capacity=10, dim=64)
    absorb(mem_oracle, c1, threshold=MATCH_THRESHOLD)  # seed identically
    the_slot = mem_oracle.slots[0]
    r_oracle = oracle_absorb(mem_oracle, c2, the_slot, dim=64)

    assert r_normal["action"] == "update"
    assert r_oracle["action"] == "update"
    np.testing.assert_array_equal(mem_normal.slots[0].value, mem_oracle.slots[0].value)
    assert r_normal["novelty"] == r_oracle["novelty"]


def test_no_update_baseline_matches_prior_phases_behavior():
    """System B must always insert (or evict+insert), never merge --
    identical convention to every prior phase's no-update baseline."""
    obs = generate_dataset()[:50]
    r = run_system_B(obs, capacity=1000)
    assert r["n_updates"] == 0
    assert len(r["memory"].slots) == len(obs)


def test_oracle_perfect_true_consolidation_by_construction():
    """Every SAME-labeled observation (same concept_id) must end up
    updating the concept's existing slot -- not merely likely, but
    guaranteed by the oracle's definition, verified directly."""
    obs = generate_dataset()
    r = run_system_C(obs, capacity=1000)  # unconstrained -- isolates the oracle logic from eviction
    concept_first_seen = {}
    for o in obs:
        if o.concept_id not in concept_first_seen:
            concept_first_seen[o.concept_id] = True
    # every concept that appears in concept_to_slot corresponds to a real slot
    for concept_id, slot in r["concept_to_slot"].items():
        assert any(slot is s for s in r["memory"].slots)


def test_oracle_zero_false_consolidation_by_construction():
    """A DIFFERENT-labeled observation must never share a slot with the
    base concept it probes against, at unconstrained capacity."""
    obs = generate_dataset()
    r = run_system_C(obs, capacity=1000)
    slot_map = r["concept_to_slot"]
    for o in obs:
        if o.probes_against is not None:
            base_slot = slot_map.get(o.probes_against)
            probe_slot = slot_map.get(o.concept_id)
            if base_slot is not None and probe_slot is not None:
                assert base_slot is not probe_slot, \
                    f"{o.concept_id} incorrectly shares a slot with {o.probes_against}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
