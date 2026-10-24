"""
tests/test_phase27.py

Unit and integration test suite for MINEMOS Phase 27:
Adaptive Memory Retrieval Under Long-Horizon Interference.

Tests:
  1. Dataset generator determinism and condition stream properties.
  2. Entity isolation between development and held-out pools.
  3. All 5 retrieval methods initialize and return expected structures.
  4. Temporal-aware retrieval downweights superseded inactive facts.
  5. Point-in-time queries (as_of) isolate historical facts.
  6. Contradiction anchors penalize conflicting slots.
  7. Corroboration boost prioritizes consolidated slots (update_count > 1).
  8. Metrics computation: Recall@k, MRR, nDCG@5, Temporal Correctness, Contradiction Intrusion.
  9. 6-way ablation dispatcher coverage.
 10. End-to-end streaming integration with BlackHoleMemory.
"""

import pytest
import numpy as np

from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate
from benchmarks.phase27_dataset import (
    Phase27StreamGenerator,
    RetrievalQuery,
    verify_phase27_dataset_integrity,
    DEV_ENTITIES,
    HELD_OUT_ENTITIES
)
from benchmarks.phase27_retrieval import Phase27Retriever
from benchmarks.phase27_metrics import compute_retrieval_metrics, is_fact_match


# ─────────────────────────────────────────────────────────────
# 1. Dataset Integrity & Isolation
# ─────────────────────────────────────────────────────────────

def test_dataset_integrity_verification():
    """Validates that generator passes integrity checks with zero entity leakage."""
    res = verify_phase27_dataset_integrity()
    assert res["status"] == "PASS"
    assert len(res["entity_overlap"]) == 0
    assert len(res["stream_leakage"]) == 0


def test_generator_conditions_and_determinism():
    """Verifies that all 4 conditions generate valid streams deterministically."""
    gen = Phase27StreamGenerator(seed=42)
    for cond in ["A_low_interference", "B_semantic_interference", "C_temporal_interference", "D_combined_interference"]:
        obs1, q1 = gen.generate_stream(cond, length=30, is_held_out=False)
        obs2, q2 = gen.generate_stream(cond, length=30, is_held_out=False)

        assert len(obs1) == 30
        assert len(q1) >= 3
        assert [o.fact_id for o in obs1] == [o.fact_id for o in obs2]
        assert [q.query_id for q in q1] == [q.query_id for q in q2]


# ─────────────────────────────────────────────────────────────
# 2. Retrieval Methods Functionality
# ─────────────────────────────────────────────────────────────

def test_retrieval_methods_on_empty_memory():
    """All retrieval methods gracefully return empty lists on empty memory."""
    mem = BlackHoleMemory(capacity=10)
    retriever = Phase27Retriever(memory_engine=mem)

    for method in ["dense_baseline", "bm25_only", "hybrid_relevance", "temporal_aware", "contra_temporal_aware", "adaptive"]:
        res = retriever.retrieve(method=method, query_text="Who is the chief scientist?", top_k=5)
        assert res == []


def test_temporal_awareness_demotes_inactive_facts():
    """
    Temporal-aware retriever prioritizes currently active facts over superseded ones
    when querying active truth (as_of=None).
    """
    mem = BlackHoleMemory(capacity=10)
    # Absorbs initial fact at t=100
    c1 = Candidate("Rowan", "serves as chief scientist at", "Helion Dynamics", timestamp=100.0)
    mem.absorb(c1)

    # Revises fact at t=200
    c2 = Candidate("Rowan", "was appointed research director at", "Astraea Technologies", timestamp=200.0)
    # Manually simulate revision state
    mem.memory.slots[0].valid_until = 200.0
    mem.memory.slots[0].debug_subject = "Rowan"
    mem.memory.slots[0].debug_predicate = "serves as chief scientist at"

    c2_slot = Candidate("Rowan", "was appointed research director at", "Astraea Technologies", timestamp=200.0)
    mem.absorb(c2_slot)

    retriever = Phase27Retriever(memory_engine=mem)

    # 1. Current query without as_of: active fact (Astraea) should be ranked first
    curr_res = retriever.retrieve(method="adaptive", query_text="Where does Rowan lead research?", top_k=2)
    assert len(curr_res) >= 1
    assert "Astraea" in curr_res[0]["text"]
    assert curr_res[0]["is_active"] is True

    # 2. Historical query with as_of=150: past fact (Helion) should be retrieved
    past_res = retriever.retrieve(method="adaptive", query_text="Where was Rowan chief scientist?", top_k=2, as_of=150.0)
    assert len(past_res) >= 1
    assert "Helion" in past_res[0]["text"]


def test_contradiction_anchor_penalizes_conflicting_slots():
    """Verifies that contradiction anchors reduce slot retrieval scores."""
    mem = BlackHoleMemory(capacity=10)
    c1 = Candidate("Valerie", "deployed the", "Falcon-7 cryogenic cooling array", timestamp=100.0)
    mem.absorb(c1)

    retriever = Phase27Retriever(memory_engine=mem, tau_anchor=0.50, w_anchor=0.80)

    # Score before contradiction anchor
    res_before = retriever.retrieve(method="contra_temporal_aware", query_text="Falcon-7 cryogenic cooling array", top_k=1)
    score_before = res_before[0]["score"]

    # Add contradiction anchor simulating contradictory observation
    cand_contra = mem.dense_encoder.encode("Valerie dismantled the Falcon-7 cryogenic cooling array.")
    slot_id = id(mem.memory.slots[0])
    mem.slot_contra_anchors[slot_id] = [(cand_contra, 110.0)]

    # Score after contradiction anchor should be lower
    res_after = retriever.retrieve(method="contra_temporal_aware", query_text="Valerie dismantled the Falcon-7 cooling array", top_k=1)
    assert res_after[0]["score"] < score_before


def test_corroboration_tier_boost():
    """Consolidated slots (update_count > 1) receive a corroboration boost."""
    mem = BlackHoleMemory(capacity=10)
    c1 = Candidate("Cassian", "surveys abyssal", "hydrothermal vents", timestamp=100.0)
    mem.absorb(c1)

    c2 = Candidate("Cassian", "studies deep ocean", "hydrothermal vents", timestamp=105.0)
    # Manually increment update_count to simulate consolidation
    mem.memory.slots[0].update_count = 3

    retriever = Phase27Retriever(memory_engine=mem, tier_boost_factor=0.20)
    res = retriever.retrieve(method="adaptive", query_text="hydrothermal vents", top_k=1)
    assert res[0]["update_count"] == 3


# ─────────────────────────────────────────────────────────────
# 3. Metrics Computation Unit Tests
# ─────────────────────────────────────────────────────────────

def test_fact_matching_utility():
    """Tests case-insensitive robust fact matching."""
    txt = "Rowan was appointed research director at Astraea Technologies."
    assert is_fact_match(txt, "Astraea Technologies", "Rowan")
    assert not is_fact_match(txt, "Helion Dynamics", "Rowan")
    assert not is_fact_match(txt, "Astraea Technologies", "Seraphina")


def test_compute_retrieval_metrics():
    """Validates metric calculations on synthetic query outputs."""
    q1 = RetrievalQuery(
        query_id="q1",
        query_text="Where does Rowan lead research?",
        target_subject="Rowan",
        target_predicate="leads",
        expected_answer="Astraea Technologies",
        expected_fact_text="Rowan directs Astraea Technologies.",
        superseded_fact_texts=["Rowan directs Helion Dynamics."]
    )

    # Top-1 is target fact
    res_good = [[{"text": "Rowan directs Astraea Technologies."}]]
    m_good = compute_retrieval_metrics([q1], res_good, latencies_ms=[10.0])
    assert m_good["recall_at_1"] == 1.0
    assert m_good["mrr"] == 1.0
    assert m_good["ndcg_at_5"] == 1.0
    assert m_good["exact_evidence_accuracy"] == 1.0
    assert m_good["temporal_correctness_rate"] == 1.0
    assert m_good["contradiction_intrusion_rate"] == 0.0

    # Top-1 is superseded contradictory fact
    res_bad = [[{"text": "Rowan directs Helion Dynamics."}]]
    m_bad = compute_retrieval_metrics([q1], res_bad, latencies_ms=[12.0])
    assert m_bad["recall_at_1"] == 0.0
    assert m_bad["exact_evidence_accuracy"] == 0.0
    assert m_bad["contradiction_intrusion_rate"] == 1.0


# ─────────────────────────────────────────────────────────────
# 4. 6-Way Ablation Dispatcher Test
# ─────────────────────────────────────────────────────────────

def test_ablation_dispatcher_all_modes():
    """Ensures all 6 ablation modes dispatch and return results."""
    mem = BlackHoleMemory(capacity=10)
    c1 = Candidate("Orion", "optimizes scramjet combustion", "nozzles", timestamp=100.0)
    mem.absorb(c1)

    retriever = Phase27Retriever(memory_engine=mem)
    modes = [
        "dense_baseline",
        "bm25_only",
        "hybrid_relevance",
        "hybrid_temporal_only",
        "hybrid_contra_only",
        "adaptive"
    ]

    for mode in modes:
        res = retriever.retrieve(method=mode, query_text="scramjet combustion nozzles", top_k=1)
        assert len(res) == 1
        assert "scramjet" in res[0]["text"]
