import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from benchmarks.phase16_dataset import (
    generate_pairwise_dataset, generate_chain_specs,
    generate_long_chains, generate_aggregate_dataset, PairwiseExample
)
from benchmarks.phase16_experiment import (
    NliSemanticGate, lexical_absorb, nli_gated_absorb, oracle_absorb,
    exhaustive_nli_absorb, ENCODER, D
)
from benchmarks.phase16_metrics import evaluate_pairwise, memory_consolidation_metrics
from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb, MATCH_THRESHOLD, cosine_similarity


@pytest.fixture(scope="module")
def gate():
    return NliSemanticGate()


def test_phase16_dataset_deterministic():
    d1 = generate_pairwise_dataset()
    d2 = generate_pairwise_dataset()
    assert len(d1) == len(d2) == 350
    for a, b in zip(d1, d2):
        assert (a.concept_id, a.ground_truth, a.subcategory, a.text1, a.text2) == \
               (b.concept_id, b.ground_truth, b.subcategory, b.text1, b.text2)


def test_phase16_dataset_categories_balanced():
    pairs = generate_pairwise_dataset()
    subcats = {}
    for p in pairs:
        subcats[p.subcategory] = subcats.get(p.subcategory, 0) + 1
    expected = {
        "exact_duplicate": 50,
        "lexical_paraphrase": 50,
        "predicate_contradiction": 50,
        "subject_contradiction": 50,
        "object_contradiction": 50,
        "unrelated": 50,
        "hard_negative": 50,
    }
    assert subcats == expected


def test_semantic_gate_exact_duplicates(gate):
    p1 = "Alice visits the gym."
    p2 = "Alice visits the gym."
    r = gate.predict_pair(p1, p2)
    assert r["decision"] == "SAME"
    assert r["p_entail"] > 0.90


def test_semantic_gate_paraphrases(gate):
    p1 = "Alice visits the gym."
    p2 = "Alice goes to the gym."
    r = gate.predict_pair(p1, p2)
    assert r["decision"] == "SAME"
    assert r["p_entail"] > 0.90


def test_semantic_gate_contradictions_and_hard_negatives(gate):
    pairs = [
        ("Alice visits the gym.", "Alice avoids the gym."),
        ("Bob likes coffee.", "Bob dislikes coffee."),
        ("Alice owns a car.", "Alice sold the car."),
    ]
    for p1, p2 in pairs:
        r = gate.predict_pair(p1, p2)
        assert r["decision"] in ("CONTRADICTION", "NEUTRAL")
        assert r["p_entail"] < 0.05
        assert r["decision"] != "SAME"


def test_semantic_gate_unrelated_and_different(gate):
    pairs = [
        ("Alice visits the gym.", "Bob visits the gym."),
        ("Alice visits the gym.", "Alice visits the library."),
        ("Alice visits the gym.", "The server stores backups."),
    ]
    for p1, p2 in pairs:
        r = gate.predict_pair(p1, p2)
        assert r["decision"] != "SAME"
        assert r["p_entail"] < 0.05


def test_semantic_gate_abstain_behavior(gate):
    p1 = "Alice visits the gym."
    p2 = "Alice goes to the gym."
    r = gate.predict_pair(p1, p2, abstain_min_confidence=1.01)
    assert r["decision"] == "ABSTAIN"


def test_oracle_gate_behavior():
    mem = MemoryState(capacity=5, dim=64)
    slot_concept_ids = {}
    c1 = Candidate("Alice", "visits", "gym", timestamp=1)
    c2 = Candidate("Alice", "goes to", "gym", timestamp=2)
    c_contra = Candidate("Alice", "avoids", "gym", timestamp=3)

    r1 = oracle_absorb(mem, c1, slot_concept_ids, "concept_001", "chain", dim=64)
    assert r1["action"] == "insert"

    # Legitimate paraphrase with same concept -> merges
    r2 = oracle_absorb(mem, c2, slot_concept_ids, "concept_001", "chain", dim=64)
    assert r2["action"] == "update"
    assert r2["slot_index"] == 0

    # Contradiction with same concept -> must NOT merge
    r3 = oracle_absorb(mem, c_contra, slot_concept_ids, "concept_001", "hard_negative", dim=64)
    assert r3["action"] == "insert"
    assert r3["slot_index"] == 1


def test_update_equation_consistency_across_all_systems(gate):
    """Ensures update blend line is bit-for-bit identical across all systems."""
    c1 = Candidate("X", "uses", "Y", timestamp=1)
    c2 = Candidate("X", "regularly uses", "Y", timestamp=2)

    # 1. Lexical
    mem_lex = MemoryState(capacity=10, dim=64)
    absorb(mem_lex, c1, threshold=0.0, encoder=ENCODER)
    r_lex = absorb(mem_lex, c2, threshold=0.0, encoder=ENCODER)

    # 2. NLI Gated
    mem_sem = MemoryState(capacity=10, dim=64)
    st_sem = {}
    nli_gated_absorb(mem_sem, c1, st_sem, gate, entail_threshold=0.0, dim=64)
    r_sem = nli_gated_absorb(mem_sem, c2, st_sem, gate, entail_threshold=0.0, dim=64)

    # 3. Exhaustive NLI
    mem_exh = MemoryState(capacity=10, dim=64)
    st_exh = {}
    exhaustive_nli_absorb(mem_exh, c1, st_exh, gate, entail_threshold=0.0, dim=64)
    r_exh = exhaustive_nli_absorb(mem_exh, c2, st_exh, gate, entail_threshold=0.0, dim=64)

    # 4. Oracle
    mem_orc = MemoryState(capacity=10, dim=64)
    st_orc = {}
    oracle_absorb(mem_orc, c1, st_orc, "cid", "chain", dim=64)
    r_orc = oracle_absorb(mem_orc, c2, st_orc, "cid", "chain", dim=64)

    assert r_lex["action"] == r_sem["action"] == r_exh["action"] == r_orc["action"] == "update"
    np.testing.assert_allclose(mem_lex.slots[0].value, mem_sem.slots[0].value, rtol=1e-6)
    np.testing.assert_allclose(mem_lex.slots[0].value, mem_exh.slots[0].value, rtol=1e-6)
    np.testing.assert_allclose(mem_lex.slots[0].value, mem_orc.slots[0].value, rtol=1e-6)


def test_production_default_unchanged():
    from phase2.memory_update import MATCH_THRESHOLD
    assert MATCH_THRESHOLD == 0.75


def test_no_cross_encoder_metric_contamination():
    cand = Candidate("Alice_0000", "visits", "place_0000")
    vec = ENCODER.encode_full(cand, dim=D)
    assert np.isclose(cosine_similarity(vec, vec), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
