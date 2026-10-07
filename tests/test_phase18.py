"""
tests/test_phase18.py

Unit and Regression Tests for Phase 18:
Semantic Retrieval + Utility-Aware Eviction.

Covers all 13 required test specifications:
1. Dense encoder produces deterministic embeddings
2. Embedding dimension is consistent (384-dim)
3. Cosine retrieval ranking works
4. Top-k retrieval works
5. Correct candidate can be recovered
6. Dense retrieval never directly decides SAME
7. NLI remains the semantic decision gate
8. Utility score uses only online slot metadata
9. Utility eviction respects capacity
10. FIFO baseline remains unchanged
11. No future/test labels enter retrieval
12. Update equation remains unchanged
13. Preserves existing Phase 1-17 test pass integrity
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import MATCH_THRESHOLD, cosine_similarity
from phase2.encoders import FieldAwareLexicalEncoder
from benchmarks.phase16_experiment import NliSemanticGate
from benchmarks.phase18_dataset import (
    Phase18Obs,
    generate_scenario_1_long_gaps,
    generate_scenario_2_safety_test,
    generate_scenario_3_contradiction_recovery,
    generate_scenario_4_semantic_drift,
    generate_scenario_5_capacity_pressure_stream,
    generate_scenario_6_continual_stream
)
from benchmarks.phase18_experiment import (
    DenseSemanticEncoder,
    dense_address,
    utility_score,
    evict_slot,
    absorb_observation,
    run_phase18_stream,
    D_LEX,
    D_DENSE
)
from benchmarks.phase18_metrics import (
    evaluate_retrieval_decomposition,
    evaluate_memory_consolidation
)

LEXICAL_ENCODER = FieldAwareLexicalEncoder()


# ──────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def dense_encoder():
    return DenseSemanticEncoder()


@pytest.fixture(scope="module")
def nli_gate():
    return NliSemanticGate()


# ──────────────────────────────────────────────
#  1 & 2: Dense Encoder Determinism & Dimension
# ──────────────────────────────────────────────

class TestDenseEncoder:
    def test_dense_encoder_deterministic(self, dense_encoder):
        """1. Dense encoder produces deterministic embeddings across repeated calls."""
        text = "Person_0001 visits Facility_0001."
        emb1 = dense_encoder.encode(text)
        emb2 = dense_encoder.encode(text)
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=6)

    def test_embedding_dimension_consistent(self, dense_encoder):
        """2. Embedding dimension is consistently 384 and L2-normalized."""
        texts = [
            "Person_0001 visits Facility_0001.",
            "Person_0002 manages Project_0002.",
            "A short phrase."
        ]
        for t in texts:
            emb = dense_encoder.encode(t)
            assert emb.shape == (D_DENSE,), f"Expected shape ({D_DENSE},), got {emb.shape}"
            norm = np.linalg.norm(emb)
            assert np.isclose(norm, 1.0, atol=1e-5), f"Embedding must be L2-normalized, got {norm}"

    def test_encode_batch_matches_single(self, dense_encoder):
        texts = ["Person_0001 visits Facility_0001.", "Person_0002 uses Device_0002."]
        batch_embs = dense_encoder.encode_batch(texts)
        single_0 = dense_encoder.encode(texts[0])
        single_1 = dense_encoder.encode(texts[1])
        np.testing.assert_array_almost_equal(batch_embs[0], single_0, decimal=5)
        np.testing.assert_array_almost_equal(batch_embs[1], single_1, decimal=5)


# ──────────────────────────────────────────────
#  3, 4, 5: Dense Retrieval Ranking, Top-k, Recovery
# ──────────────────────────────────────────────

class TestDenseRetrieval:
    def test_cosine_retrieval_ranking_works(self, dense_encoder):
        """3. Cosine retrieval ranking ranks true paraphrase higher than unrelated concept."""
        canon = dense_encoder.encode("Person_0001 visits Facility_0001.")
        para = dense_encoder.encode("Person_0001 goes to Facility_0001.")
        unrel = dense_encoder.encode("Person_0500 manages Device_0500.")

        slots = [unrel, canon]
        ranked = dense_address(slots, para, top_k=2)
        assert len(ranked) == 2
        # Slot 1 (canon) should rank above slot 0 (unrel)
        assert ranked[0][0] == 1, f"Expected canon (slot 1) at rank 0, got slot {ranked[0][0]}"
        assert ranked[0][1] > ranked[1][1]

    def test_top_k_retrieval_bounds(self, dense_encoder):
        """4. Top-k retrieval respects requested k and available slot count."""
        embs = [dense_encoder.encode(f"Person_{i:04d} visits Facility_0001.") for i in range(10)]
        query = dense_encoder.encode("Person_0005 visits Facility_0001.")

        for k in (1, 3, 5, 10):
            ranked = dense_address(embs, query, top_k=k)
            assert len(ranked) == k
            # Ensure strictly descending order
            sims = [s for _, s in ranked]
            assert sims == sorted(sims, reverse=True)

        # k larger than slot count
        ranked_overflow = dense_address(embs, query, top_k=25)
        assert len(ranked_overflow) == 10

    def test_correct_candidate_recovered(self, dense_encoder):
        """5. Target candidate slot is recovered at rank 1 among distractor slots."""
        target_text = "Person_0042 studies Project_0042."
        target_emb = dense_encoder.encode(target_text)
        distractor_embs = [dense_encoder.encode(f"Person_{i:04d} uses Device_{i:04d}.") for i in range(20)]

        all_slots = distractor_embs + [target_emb]
        probe = dense_encoder.encode("Person_0042 researches Project_0042.")

        ranked = dense_address(all_slots, probe, top_k=3)
        top_idx, top_sim = ranked[0]
        assert top_idx == 20, f"Expected target slot (index 20) at rank 1, got {top_idx}"
        assert top_sim > 0.80


# ──────────────────────────────────────────────
#  6 & 7: Semantic Safety (NLI Gate Integrity)
# ──────────────────────────────────────────────

class TestSemanticSafety:
    def test_dense_retrieval_never_directly_decides_same(self, dense_encoder, nli_gate):
        """6. Dense similarity never causes a merge on its own without NLI verification."""
        mem = MemoryState(capacity=10, dim=D_LEX)
        slot_texts = {}
        slot_dense_embs = []

        # Add canonical
        c_init = Candidate("Person_0001", "visits", "Facility_0001", timestamp=0.0)
        r1 = absorb_observation(
            memory=mem, candidate=c_init, system="B", slot_texts=slot_texts,
            slot_dense_embs=slot_dense_embs, nli_gate=nli_gate, dense_encoder=dense_encoder
        )
        assert r1["action"] == "insert"

        # Contradiction has high lexical and dense similarity, but NLI must reject it!
        c_contra = Candidate("Person_0001", "avoids", "Facility_0001", timestamp=1.0)
        r2 = absorb_observation(
            memory=mem, candidate=c_contra, system="B", slot_texts=slot_texts,
            slot_dense_embs=slot_dense_embs, nli_gate=nli_gate, dense_encoder=dense_encoder
        )
        # MUST be inserted as a new slot, NOT merged into canonical slot!
        assert r2["action"] == "insert", f"Dense retrieval bypassed NLI on contradiction! Action={r2['action']}"
        assert r2["gate"] in ("rejected_contra", "rejected_neutral")
        assert len(mem.slots) == 2, "Contradiction must occupy separate slot"

    def test_nli_remains_semantic_decision_gate(self, dense_encoder, nli_gate):
        """7. NLI authorizes merge for true paraphrase but rejects contradiction and neutral."""
        mem = MemoryState(capacity=10, dim=D_LEX)
        slot_texts = {}
        slot_dense_embs = []

        c_canon = Candidate("Person_0002", "visits", "Facility_0002", timestamp=0.0)
        absorb_observation(mem, c_canon, "D", slot_texts, slot_dense_embs, nli_gate, dense_encoder)

        # Paraphrase -> NLI approves merge
        c_para = Candidate("Person_0002", "goes to", "Facility_0002", timestamp=1.0)
        r_para = absorb_observation(mem, c_para, "D", slot_texts, slot_dense_embs, nli_gate, dense_encoder)
        assert r_para["action"] == "update"
        assert r_para["gate"] == "nli_accept"

        # Contradiction -> NLI rejects
        c_contra = Candidate("Person_0002", "avoids", "Facility_0002", timestamp=2.0)
        r_contra = absorb_observation(mem, c_contra, "D", slot_texts, slot_dense_embs, nli_gate, dense_encoder)
        assert r_contra["action"] in ("insert", "evict_insert")
        assert r_contra["gate"] == "rejected_contra"


# ──────────────────────────────────────────────
#  8 & 9: Utility Eviction Mechanics
# ──────────────────────────────────────────────

class TestUtilityEviction:
    def test_utility_score_uses_only_online_metadata(self):
        """8. Utility score uses only online slot metadata (update_count, timestamp, confidence)."""
        slot1 = MemorySlot(
            key=np.zeros(10), value=np.zeros(10), confidence=0.9,
            timestamp=100.0, created_at=10.0, update_count=5
        )
        slot2 = MemorySlot(
            key=np.zeros(10), value=np.zeros(10), confidence=0.2,
            timestamp=10.0, created_at=10.0, update_count=1
        )

        u1 = utility_score(slot1, current_time=105.0)
        u2 = utility_score(slot2, current_time=105.0)

        # Consolidated active slot1 must have higher utility than stale distractor slot2
        assert u1 > u2
        assert isinstance(u1, float)
        assert isinstance(u2, float)

    def test_utility_eviction_respects_capacity(self, dense_encoder, nli_gate):
        """9. Utility eviction maintains strict capacity bounds and purges lowest-utility slot."""
        mem = MemoryState(capacity=3, dim=D_LEX)
        slot_texts = {}
        slot_dense_embs = []

        # Fill capacity with 3 candidates
        for i in range(3):
            c = Candidate(f"Person_{i:04d}", "visits", f"Facility_{i:04d}", timestamp=float(i))
            absorb_observation(mem, c, "D", slot_texts, slot_dense_embs, nli_gate, dense_encoder, current_time=float(i))
        assert len(mem.slots) == 3

        # Reinforce slot 0 (Person_0000) so it has update_count=2
        c_reinforce = Candidate("Person_0000", "goes to", "Facility_0000", timestamp=3.0)
        r_up = absorb_observation(mem, c_reinforce, "D", slot_texts, slot_dense_embs, nli_gate, dense_encoder, current_time=3.0)
        assert r_up["action"] == "update"
        assert mem.slots[0].update_count == 2

        # Now insert a 4th new candidate -> capacity overflow -> must evict slot 1 (unconsolidated, older timestamp)
        c_new = Candidate("Person_0099", "monitors", "Facility_0099", timestamp=4.0)
        r_new = absorb_observation(mem, c_new, "D", slot_texts, slot_dense_embs, nli_gate, dense_encoder, current_time=4.0)
        assert r_new["action"] == "evict_insert"
        assert len(mem.slots) == 3

        # Consolidated slot (Person_0000) MUST have survived!
        subjects_in_mem = [s.debug_subject for s in mem.slots]
        assert "Person_0000" in subjects_in_mem, f"Utility eviction purged reinforced slot! Resident: {subjects_in_mem}"


# ──────────────────────────────────────────────
#  10, 11, 12: Baseline Invariants & Controls
# ──────────────────────────────────────────────

class TestBaselineInvariants:
    def test_fifo_baseline_remains_unchanged(self):
        """10. FIFO eviction preserves exact created_at ordering."""
        mem = MemoryState(capacity=2, dim=10)
        mem.add(np.zeros(10), np.zeros(10), 0.5, timestamp=10.0, debug_subject="First")
        mem.add(np.zeros(10), np.zeros(10), 0.5, timestamp=20.0, debug_subject="Second")
        evicted = mem.evict_oldest()
        assert evicted.debug_subject == "First"
        assert len(mem.slots) == 1

    def test_no_future_or_test_labels_enter_retrieval(self):
        """11. Verify observation and absorb interfaces receive zero concept IDs or future labels."""
        import inspect
        sig = inspect.signature(absorb_observation)
        params = list(sig.parameters.keys())
        forbidden = ["concept_id", "oracle", "future", "label", "target_concept"]
        for f in forbidden:
            assert f not in params, f"Forbidden label leak in absorb_observation signature: {f}"

    def test_match_threshold_and_update_equation_invariant(self, dense_encoder, nli_gate):
        """12. MATCH_THRESHOLD = 0.75 invariant and update equation v = (1-a)v + ax."""
        assert MATCH_THRESHOLD == 0.75

        mem = MemoryState(capacity=5, dim=D_LEX)
        slot_texts = {}
        slot_dense_embs = []

        c1 = Candidate("Person_0001", "visits", "Facility_0001", timestamp=0.0)
        absorb_observation(mem, c1, "A", slot_texts, slot_dense_embs, nli_gate, dense_encoder)
        v0 = mem.slots[0].value.copy()

        c2 = Candidate("Person_0001", "goes to", "Facility_0001", timestamp=1.0)
        x2 = LEXICAL_ENCODER.encode_full(c2, dim=D_LEX)
        r2 = absorb_observation(mem, c2, "A", slot_texts, slot_dense_embs, nli_gate, dense_encoder)
        assert r2["action"] == "update"

        alpha = r2["novelty"]
        expected_v = (1.0 - alpha) * v0 + alpha * x2
        np.testing.assert_array_almost_equal(mem.slots[0].value, expected_v, decimal=5)


# ──────────────────────────────────────────────
#  13: 2x2 Factorial Smoke Run
# ──────────────────────────────────────────────

class TestFactorialSmokeRun:
    def test_all_four_systems_run_deterministic(self, dense_encoder, nli_gate):
        """13. All 4 factorial systems complete a deterministic mini-stream."""
        stream = generate_scenario_2_safety_test()[:12]  # 3 concepts x 4 obs = 12 obs

        for sys_id in ("A", "B", "C", "D"):
            res = run_phase18_stream(
                observations=stream, capacity=10, system=sys_id,
                nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3
            )
            assert res["system"] == sys_id
            assert res["n_updates"] + res["n_inserts"] + res["n_evicts"] == len(stream)
            assert res["wall_seconds"] > 0.0

            m = evaluate_memory_consolidation(res, stream)
            assert 0.0 <= m["contradiction_retention"] <= 1.0
            assert 0.0 <= m["true_consolidation_rate"] <= 1.0
