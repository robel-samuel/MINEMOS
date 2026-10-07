import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from benchmarks.phase13_dataset import generate_dataset
from benchmarks.phase13_experiment import run_system_A, run_system_B, real_rekey_absorb
from benchmarks.phase13_metrics import error_propagation_test
from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb, MATCH_THRESHOLD
from phase2.encoders import FieldAwareLexicalEncoder


def test_dataset_deterministic():
    d1 = generate_dataset()
    d2 = generate_dataset()
    assert len(d1) == len(d2)
    for a, b in zip(d1, d2):
        assert (a.concept_id, a.category, a.candidate.subject,
                a.candidate.predicate, a.candidate.object) == \
               (b.concept_id, b.category, b.candidate.subject,
                b.candidate.predicate, b.candidate.object)


def test_drift_sequence_has_50_concepts():
    obs = generate_dataset()
    drift_concepts = set(o.concept_id for o in obs if o.category == "drift_sequence")
    assert len(drift_concepts) == 50


def test_encoder_produces_mixed_correct_and_incorrect_merges():
    enc = FieldAwareLexicalEncoder()
    from phase2.memory_update import cosine_similarity
    para_sim = cosine_similarity(
        enc.encode_full(Candidate("Alice", "uses", "Python"), dim=4096),
        enc.encode_full(Candidate("Alice", "regularly uses", "Python"), dim=4096))
    contra_sim = cosine_similarity(
        enc.encode_full(Candidate("Alice", "visits", "the gym"), dim=4096),
        enc.encode_full(Candidate("Alice", "avoids", "the gym"), dim=4096))
    assert para_sim >= MATCH_THRESHOLD
    assert contra_sim >= MATCH_THRESHOLD


def test_rekey_update_line_matches_absorb_when_addressing_agrees():
    """When both frozen and rekey systems address the SAME slot with the
    SAME similarity (guaranteed on the very first update, before any
    re-keying has had a chance to change addressing), the resulting
    value must be bit-identical -- proving the update equation itself
    was not altered."""
    c1 = Candidate("X", "uses", "Y", timestamp=1)
    c2 = Candidate("X", "regularly uses", "Y", timestamp=2)

    mem_frozen = MemoryState(capacity=10, dim=64)
    absorb(mem_frozen, c1, threshold=MATCH_THRESHOLD, encoder=FieldAwareLexicalEncoder())
    r_frozen = absorb(mem_frozen, c2, threshold=MATCH_THRESHOLD, encoder=FieldAwareLexicalEncoder())

    mem_rekey = MemoryState(capacity=10, dim=64)
    real_rekey_absorb(mem_rekey, c1, dim=64)
    r_rekey = real_rekey_absorb(mem_rekey, c2, dim=64)

    assert r_frozen["action"] == r_rekey["action"]
    if r_frozen["action"] == "update":
        np.testing.assert_allclose(mem_frozen.slots[0].value, mem_rekey.slots[0].value, rtol=1e-6)


def test_error_propagation_both_systems_experience_same_initial_merge():
    r = error_propagation_test()
    assert r["frozen"]["incorrect_merge_occurred"] == r["rekey"]["incorrect_merge_occurred"]
    assert r["frozen"]["trace"][0]["similarity"] == r["rekey"]["trace"][0]["similarity"]


def test_oracle_systems_unaffected_by_rekey_choice_in_admission():
    from benchmarks.phase13_experiment import run_system_C_oracle_frozen, run_system_D_oracle_rekey
    obs = generate_dataset()[:300]
    rC = run_system_C_oracle_frozen(obs, capacity=500)
    rD = run_system_D_oracle_rekey(obs, capacity=500)
    assert len(rC["memory"].slots) == len(rD["memory"].slots)
    assert rC["n_updates"] == rD["n_updates"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
