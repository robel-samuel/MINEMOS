"""
tests/test_phase19.py

Unit and Regression Tests for Phase 19:
Contradiction-Aware Retrieval, Safety Shielding, and Adaptive Verification.
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
from benchmarks.phase18_experiment import (
    DenseSemanticEncoder,
    utility_score,
    D_LEX,
    D_DENSE
)
from benchmarks.phase19_experiment import (
    contradiction_aware_utility,
    evict_slot_phase19,
    adaptive_dense_address,
    absorb_observation_phase19,
    run_phase19_stream
)
from benchmarks.phase18_dataset import Phase18Obs, make_distractor

LEXICAL_ENCODER = FieldAwareLexicalEncoder()


@pytest.fixture(scope="module")
def dense_encoder():
    return DenseSemanticEncoder()


@pytest.fixture(scope="module")
def nli_gate():
    return NliSemanticGate()


class TestContradictionShielding:
    def test_contradiction_utility_values(self):
        """Verified factual slots challenged by contradictions gain shielding bonus."""
        slot_factual = MemorySlot(
            key=np.zeros(D_LEX), value=np.zeros(D_LEX),
            confidence=0.9, timestamp=10.0, created_at=0.0
        )
        slot_factual.update_count = 3

        # Base utility
        u_base = contradiction_aware_utility(slot_factual, current_time=10.0, contradiction_count=0)
        # Challenged factual utility
        u_challenged = contradiction_aware_utility(slot_factual, current_time=10.0, contradiction_count=2)
        assert u_challenged > u_base
        assert abs(u_challenged - (u_base + 0.30)) < 1e-5

    def test_fragile_slot_penalty(self):
        """Unverified slot (update_count=0) with contradictions receives penalty."""
        slot_fragile = MemorySlot(
            key=np.zeros(D_LEX), value=np.zeros(D_LEX),
            confidence=0.9, timestamp=10.0, created_at=0.0
        )
        slot_fragile.update_count = 0
        u_base = contradiction_aware_utility(slot_fragile, current_time=10.0, contradiction_count=0)
        u_penalized = contradiction_aware_utility(slot_fragile, current_time=10.0, contradiction_count=2)
        assert u_penalized < u_base


class TestAdaptiveCandidateBeam:
    def test_prune_weak_candidates(self):
        """When top-1 dense similarity is below tau_min, beam is empty."""
        slot_embs = [np.zeros(D_DENSE)]
        cand_emb = np.zeros(D_DENSE)
        # Orthogonal vectors -> sim = 0.0 < tau_min (0.40)
        cands, policy = adaptive_dense_address(slot_embs, cand_emb, tau_min=0.40)
        assert policy == "prune_weak"
        assert len(cands) == 0

    def test_dominant_candidate_path(self):
        """When top-1 is high and margin is large, beam size is 1."""
        e1 = np.ones(D_DENSE) / np.linalg.norm(np.ones(D_DENSE))
        e2 = np.zeros(D_DENSE)
        e2[0] = 1.0
        slot_embs = [e1, e2]
        # Query is identical to e1 -> sim_1 = 1.0, sim_2 is low
        cands, policy = adaptive_dense_address(
            slot_embs, e1, tau_min=0.40, tau_high=0.75, delta_margin=0.10, k_max=3
        )
        assert policy == "dominant"
        assert len(cands) == 1
        assert cands[0][0] == 0

    def test_ambiguous_beam_expansion(self):
        """When multiple candidates are closely matched, beam expands up to k_max."""
        # Create two similar vectors
        base = np.ones(D_DENSE) / np.linalg.norm(np.ones(D_DENSE))
        v1 = base.copy()
        v2 = base.copy()
        v2[1] += 0.05
        v2 /= np.linalg.norm(v2)
        slot_embs = [v1, v2]

        cands, policy = adaptive_dense_address(
            slot_embs, base, tau_min=0.40, tau_high=0.75, delta_margin=0.20, k_max=3
        )
        assert policy == "ambiguous"
        assert len(cands) == 2


class TestBatchedNliEquivalence:
    def test_predict_batch_matches_single(self, nli_gate):
        """Batched NLI inference produces identical decisions to single-pair calls."""
        pairs = [
            ("Alice visits Hospital.", "Alice goes to Hospital."),
            ("Alice visits Hospital.", "Alice avoids Hospital."),
            ("Alice visits Hospital.", "Bob manages Lab.")
        ]
        single_preds = [nli_gate.predict_pair(p[0], p[1]) for p in pairs]
        batch_preds = nli_gate.predict_batch(pairs)

        for s_pred, b_pred in zip(single_preds, batch_preds):
            assert s_pred["decision"] == b_pred["decision"]
            assert abs(s_pred["confidence"] - b_pred["confidence"]) < 1e-4


class TestProductionInvariants:
    def test_match_threshold_frozen(self):
        assert MATCH_THRESHOLD == 0.75

    def test_update_equation_invariant(self, dense_encoder, nli_gate):
        """Core update equation slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t holds."""
        for sys_id in ("D", "D1", "D2", "D3", "D4"):
            mem = MemoryState(capacity=5, dim=D_LEX)
            c1 = Candidate("Person_0001", "visits", "Hospital_0001", timestamp=1)
            c2 = Candidate("Person_0001", "goes to", "Hospital_0001", timestamp=2)

            slot_texts = {}
            slot_dense = []
            slot_contra = {}
            slot_anchors = {}

            # Absorb canonical
            absorb_observation_phase19(
                mem, c1, sys_id, slot_texts, slot_dense, slot_contra, slot_anchors,
                nli_gate, dense_encoder, top_k=3
            )
            val_before = mem.slots[0].value.copy()

            # Absorb paraphrase
            res = absorb_observation_phase19(
                mem, c2, sys_id, slot_texts, slot_dense, slot_contra, slot_anchors,
                nli_gate, dense_encoder, top_k=3
            )
            assert res["action"] == "update"
            val_after = mem.slots[0].value
            x2 = LEXICAL_ENCODER.encode_full(c2, dim=D_LEX)
            alpha_t = res["novelty"]
            expected = (1.0 - alpha_t) * val_before + alpha_t * x2
            np.testing.assert_allclose(val_after, expected, atol=1e-6)
