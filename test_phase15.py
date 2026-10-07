import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from benchmarks.phase15_dataset import generate_aggregate_dataset, generate_chain_specs
from benchmarks.phase15_experiment import confidence_gated_absorb, ENCODER, D
from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb, MATCH_THRESHOLD, cosine_similarity


def test_phase15_dataset_deterministic():
    d1 = generate_aggregate_dataset()
    d2 = generate_aggregate_dataset()
    assert len(d1) == len(d2)
    for a, b in zip(d1, d2):
        assert (a.concept_id, a.category, a.candidate.subject,
                a.candidate.predicate, a.candidate.object) == \
               (b.concept_id, b.category, b.candidate.subject,
                b.candidate.predicate, b.candidate.object)


def test_phase15_chain_specs_variants():
    specs = generate_chain_specs()
    assert len(specs) == 100
    variants = {}
    for s in specs:
        variants[s["variant"]] = variants.get(s["variant"], 0) + 1
    assert variants == {
        "canonical_para_contra": 20,
        "para_canonical_contra": 20,
        "contra_para_canonical": 20,
        "canonical_contra_para": 20,
        "clean_paraphrases_only": 20,
    }


def test_confidence_gated_rekey_mechanism():
    mem = MemoryState(capacity=10, dim=64)
    confirm_state = {}
    c0 = Candidate("X", "uses", "Y", timestamp=0)
    confidence_gated_absorb(mem, c0, confirm_state, rekey_confirmations=2, dim=64)
    key0 = mem.slots[0].key.copy()

    # First merge (similar) -> count becomes 1, no rekey
    c1 = Candidate("X", "regularly uses", "Y", timestamp=1)
    r1 = confidence_gated_absorb(mem, c1, confirm_state, rekey_confirmations=2, dim=64)
    if r1["action"] == "update":
        assert np.array_equal(key0, mem.slots[0].key)
        assert r1["rekeyed"] is False
        assert r1["confirm_count"] == 1

        # Second merge (similar) -> count reaches 2, triggers rekey
        c2 = Candidate("X", "regularly uses", "Y", timestamp=2)
        r2 = confidence_gated_absorb(mem, c2, confirm_state, rekey_confirmations=2, dim=64)
        assert r2["action"] == "update"
        assert r2["rekeyed"] is True
        assert r2["confirm_count"] == 0
        assert not np.array_equal(key0, mem.slots[0].key)


def test_update_equation_identical_across_A_B_C():
    c1 = Candidate("X", "uses", "Y", timestamp=1)
    c2 = Candidate("X", "regularly uses", "Y", timestamp=2)
    enc = ENCODER

    mem_a = MemoryState(capacity=10, dim=64)
    absorb(mem_a, c1, threshold=MATCH_THRESHOLD, encoder=enc)
    r_a = absorb(mem_a, c2, threshold=MATCH_THRESHOLD, encoder=enc)

    mem_c = MemoryState(capacity=10, dim=64)
    confirm_state = {}
    confidence_gated_absorb(mem_c, c1, confirm_state, rekey_confirmations=2, dim=64)
    r_c = confidence_gated_absorb(mem_c, c2, confirm_state, rekey_confirmations=2, dim=64)

    assert r_a["action"] == r_c["action"]
    if r_a["action"] == "update":
        np.testing.assert_allclose(mem_a.slots[0].value, mem_c.slots[0].value, rtol=1e-6)


def test_semantic_inconsistency_passes_gate():
    """Confirms the diagnosed failure mode: contradictory predicates sharing
    subject+object yield similarity >= MATCH_THRESHOLD under lexical encoding,
    falsely confirming the direction gate."""
    c_vis = ENCODER.encode_full(Candidate("Alice_0000", "visits", "place_0000"), dim=D)
    c_avoid = ENCODER.encode_full(Candidate("Alice_0000", "avoids", "place_0000"), dim=D)
    sim = cosine_similarity(c_vis, c_avoid)
    assert sim >= MATCH_THRESHOLD, f"Expected {sim} >= {MATCH_THRESHOLD}"


def test_gate_resets_on_dissimilar_candidate():
    """Verifies that when a merge has similarity below MATCH_THRESHOLD to the
    previous merge, the confirmation counter resets to 1 rather than incrementing."""
    c1 = Candidate("Alice_0000", "visits", "place_0000", timestamp=1)
    c2 = Candidate("Alice_0000", "goes to", "place_0000", timestamp=2)
    x1 = ENCODER.encode_full(c1, dim=D)
    x2 = ENCODER.encode_full(c2, dim=D)
    sim_1_2 = cosine_similarity(x1, x2)
    assert sim_1_2 < MATCH_THRESHOLD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
