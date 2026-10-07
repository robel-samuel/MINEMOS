"""
tests/test_phase20.py

Unit and Regression Tests for Phase 20:
Adaptive Contradiction Anchor Management:
- Temporal anchor decay math
- Bounded anchor FIFO eviction
- Anchor weight parameterization & backward equivalence (w=0)
- Core update equation invariant
- MATCH_THRESHOLD invariant
- Scenario 7 dataset determinism
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
from benchmarks.phase20_experiment import (
    adaptive_dense_address_phase20,
    absorb_observation_phase20,
    run_phase20_stream
)
from benchmarks.phase20_dataset import (
    generate_scenario_7_semantic_drift,
    Phase18Obs
)

LEXICAL_ENCODER = FieldAwareLexicalEncoder()


@pytest.fixture(scope="module")
def dense_encoder():
    return DenseSemanticEncoder()


@pytest.fixture(scope="module")
def nli_gate():
    return NliSemanticGate()


class TestAnchorDecay:
    def test_decay_formula(self):
        """Verify that penalty scales with exp(-lambda * delta_t)."""
        slot_emb = np.ones(D_DENSE) / np.linalg.norm(np.ones(D_DENSE))
        cand_emb = slot_emb.copy()
        anchor_emb = slot_emb.copy()  # sim = 1.0 > 0.70

        slot = MemorySlot(key=np.zeros(D_LEX), value=np.zeros(D_LEX), confidence=0.9, timestamp=0.0, created_at=0.0)
        slots = [slot]

        # Anchor created at t=10.0
        slot_contra_anchors = {id(slot): [(anchor_emb, 10.0)]}

        # Query at t=10.0 (delta_t = 0.0) -> decay = 1.0, penalty = 0.20 * 1.0 * 1.0 = 0.20
        _, _, pen_t10, _ = adaptive_dense_address_phase20(
            [slot_emb], cand_emb, slot_contra_anchors=slot_contra_anchors,
            slots=slots, current_time=10.0, use_repulsion=True, w_anchor=0.20, decay_lambda=0.01
        )
        assert abs(pen_t10 - 0.20) < 1e-5

        # Query at t=110.0 (delta_t = 100.0) -> decay = exp(-0.01 * 100) = exp(-1.0) approx 0.367879
        _, _, pen_t110, _ = adaptive_dense_address_phase20(
            [slot_emb], cand_emb, slot_contra_anchors=slot_contra_anchors,
            slots=slots, current_time=110.0, use_repulsion=True, w_anchor=0.20, decay_lambda=0.01
        )
        expected_pen = 0.20 * np.exp(-1.0)
        assert abs(pen_t110 - expected_pen) < 1e-5
        assert pen_t110 < pen_t10

    def test_zero_decay_is_static(self):
        """When decay_lambda=0.0, penalty remains unchanged over time."""
        slot_emb = np.ones(D_DENSE) / np.linalg.norm(np.ones(D_DENSE))
        cand_emb = slot_emb.copy()
        slot = MemorySlot(key=np.zeros(D_LEX), value=np.zeros(D_LEX), confidence=0.9, timestamp=0.0, created_at=0.0)
        slots = [slot]
        slot_contra_anchors = {id(slot): [(slot_emb.copy(), 0.0)]}

        _, _, pen_early, _ = adaptive_dense_address_phase20(
            [slot_emb], cand_emb, slot_contra_anchors=slot_contra_anchors,
            slots=slots, current_time=10.0, use_repulsion=True, w_anchor=0.20, decay_lambda=0.0
        )
        _, _, pen_late, _ = adaptive_dense_address_phase20(
            [slot_emb], cand_emb, slot_contra_anchors=slot_contra_anchors,
            slots=slots, current_time=5000.0, use_repulsion=True, w_anchor=0.20, decay_lambda=0.0
        )
        assert abs(pen_early - 0.20) < 1e-5
        assert abs(pen_early - pen_late) < 1e-5


class TestBoundedAnchors:
    def test_fifo_eviction_unit(self):
        """When max_anchors=2, appending beyond capacity evicts oldest anchor."""
        anchors: list[tuple[np.ndarray, float]] = []
        max_anchors = 2

        for t in (10.0, 20.0, 30.0):
            if max_anchors is not None and len(anchors) >= max_anchors:
                anchors.pop(0)
            anchors.append((np.ones(D_DENSE), t))

        assert len(anchors) == 2
        assert anchors[0][1] == 20.0
        assert anchors[1][1] == 30.0

    def test_bounded_anchor_retention_in_addressing(self):
        """Verify that addressing evaluates only the retained bounded anchors."""
        slot_emb = np.ones(D_DENSE) / np.linalg.norm(np.ones(D_DENSE))
        slot = MemorySlot(key=np.zeros(D_LEX), value=np.zeros(D_LEX), confidence=0.9, timestamp=0.0, created_at=0.0)
        slots = [slot]

        # 2 retained anchors: one at t=20, one at t=30
        slot_anchors = {
            id(slot): [
                (slot_emb.copy(), 20.0),
                (slot_emb.copy(), 30.0)
            ]
        }

        # Query at t=30 with decay_lambda=0.01
        _, _, pen, active_count = adaptive_dense_address_phase20(
            [slot_emb], slot_emb.copy(), slot_contra_anchors=slot_anchors,
            slots=slots, current_time=30.0, use_repulsion=True, w_anchor=0.20, decay_lambda=0.01
        )
        assert active_count == 2
        # Max penalty should be from the newer anchor at t=30 (delta_t=0 -> pen=0.20)
        assert abs(pen - 0.20) < 1e-5


class TestAnchorWeightEquivalence:
    def test_zero_weight_reproduces_baseline(self):
        """When w_anchor=0.0, applied penalty is strictly 0.0 regardless of anchors."""
        slot_emb = np.ones(D_DENSE) / np.linalg.norm(np.ones(D_DENSE))
        cand_emb = slot_emb.copy()
        slot = MemorySlot(key=np.zeros(D_LEX), value=np.zeros(D_LEX), confidence=0.9, timestamp=0.0, created_at=0.0)
        slots = [slot]
        slot_contra_anchors = {id(slot): [(slot_emb.copy(), 0.0)]}

        cands, _, pen, _ = adaptive_dense_address_phase20(
            [slot_emb], cand_emb, slot_contra_anchors=slot_contra_anchors,
            slots=slots, current_time=10.0, use_repulsion=True, w_anchor=0.0
        )
        assert pen == 0.0
        assert abs(cands[0][1] - 1.0) < 1e-5


class TestProductionInvariants:
    def test_match_threshold_frozen(self):
        assert MATCH_THRESHOLD == 0.75

    def test_update_equation_invariant(self, dense_encoder, nli_gate):
        """slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t strictly holds."""
        for sys_id in ("D", "D4", "D5", "D6", "D7", "D8"):
            mem = MemoryState(capacity=5, dim=D_LEX)
            c1 = Candidate("Person_0001", "visits", "Hospital_0001", timestamp=1)
            c2 = Candidate("Person_0001", "goes to", "Hospital_0001", timestamp=2)

            slot_texts = {}
            slot_dense = []
            slot_contra = {}
            slot_anchors = {}

            # Absorb canonical
            absorb_observation_phase20(
                mem, c1, sys_id, slot_texts, slot_dense, slot_contra, slot_anchors,
                nli_gate, dense_encoder, top_k=3, w_anchor=0.20, decay_lambda=0.005, max_anchors=3
            )
            val_before = mem.slots[0].value.copy()

            # Absorb paraphrase
            res = absorb_observation_phase20(
                mem, c2, sys_id, slot_texts, slot_dense, slot_contra, slot_anchors,
                nli_gate, dense_encoder, top_k=3, w_anchor=0.20, decay_lambda=0.005, max_anchors=3
            )
            assert res["action"] == "update"
            val_after = mem.slots[0].value
            x2 = LEXICAL_ENCODER.encode_full(c2, dim=D_LEX)
            alpha_t = res["novelty"]
            expected = (1.0 - alpha_t) * val_before + alpha_t * x2
            np.testing.assert_allclose(val_after, expected, atol=1e-6)


class TestScenario7Dataset:
    def test_scenario_7_structure(self):
        stream = generate_scenario_7_semantic_drift(n_concepts=5, inter_gap=2, seed=123)
        assert len(stream) > 0
        categories = {obs.category for obs in stream}
        assert "canonical" in categories
        assert "contradiction" in categories
        assert "paraphrase" in categories
        assert "temporal_update" in categories
        assert "unrelated" in categories

        # Check determinism
        stream2 = generate_scenario_7_semantic_drift(n_concepts=5, inter_gap=2, seed=123)
        assert [o.text for o in stream] == [o.text for o in stream2]
