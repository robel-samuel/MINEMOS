import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from benchmarks.phase14_dataset import generate_aggregate_dataset, generate_contamination_chains
from benchmarks.phase14_experiment import run_system, delayed_rekey_absorb
from benchmarks.phase14_metrics import error_amplification_analysis, _run_chain
from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb, MATCH_THRESHOLD
from phase2.encoders import FieldAwareLexicalEncoder


def test_dataset_deterministic():
    d1 = generate_aggregate_dataset()
    d2 = generate_aggregate_dataset()
    assert len(d1) == len(d2)
    for a, b in zip(d1, d2):
        assert (a.concept_id, a.category, a.candidate.subject,
                a.candidate.predicate, a.candidate.object) == \
               (b.concept_id, b.category, b.candidate.subject,
                b.candidate.predicate, b.candidate.object)


def test_contamination_chains_cover_all_five_variants():
    chains = generate_contamination_chains()
    variants = set(c["variant"] for c in chains)
    assert variants == {"canonical_para_para", "para_canonical_para",
                         "canonical_hardneg_para", "para_hardneg_para", "alternating"}


def test_delayed_rekey_never_fires_before_N_merges():
    mem = MemoryState(capacity=10, dim=64)
    merge_counts = {}
    c0 = Candidate("X", "uses", "Y", timestamp=0)
    delayed_rekey_absorb(mem, c0, merge_counts, N=3, dim=64)
    key0 = mem.slots[0].key.copy()
    for i in range(1, 3):
        c = Candidate("X", "regularly uses" if i % 2 else "primarily works with", "Y", timestamp=i)
        r = delayed_rekey_absorb(mem, c, merge_counts, N=3, dim=64)
        if r["action"] == "update":
            assert np.array_equal(key0, mem.slots[0].key)
            assert r["rekeyed"] is False


def test_update_equation_identical_across_A_B_C_on_first_merge():
    c1 = Candidate("X", "uses", "Y", timestamp=1)
    c2 = Candidate("X", "regularly uses", "Y", timestamp=2)
    enc = FieldAwareLexicalEncoder()

    mem_a = MemoryState(capacity=10, dim=64)
    absorb(mem_a, c1, threshold=MATCH_THRESHOLD, encoder=enc)
    r_a = absorb(mem_a, c2, threshold=MATCH_THRESHOLD, encoder=enc)

    mem_c = MemoryState(capacity=10, dim=64)
    merge_counts = {}
    delayed_rekey_absorb(mem_c, c1, merge_counts, N=2, dim=64)
    r_c = delayed_rekey_absorb(mem_c, c2, merge_counts, N=2, dim=64)

    assert r_a["action"] == r_c["action"]
    if r_a["action"] == "update":
        np.testing.assert_allclose(mem_a.slots[0].value, mem_c.slots[0].value, rtol=1e-6)


def test_error_amplification_ratio_is_bounded_by_sequence_length():
    chains = generate_contamination_chains()
    summary = error_amplification_analysis(chains, ["A", "B"])
    for sysname, data in summary.items():
        if data["error_amplification_ratio"] is not None:
            assert 0.0 <= data["error_amplification_ratio"] <= 1.0


def test_c5_matches_frozen_key_when_dataset_never_reaches_5_merges():
    chains = generate_contamination_chains()
    chain = next(c for c in chains if c["variant"] == "canonical_hardneg_para")
    r_a = _run_chain("A", chain)
    r_c5 = _run_chain("C5", chain)
    seq_len = len(chain["sequence"])
    actions_a = [t["action"] for t in r_a["trace"][:seq_len]]
    actions_c5 = [t["action"] for t in r_c5["trace"][:seq_len]]
    assert actions_a == actions_c5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
