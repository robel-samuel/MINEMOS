"""
tests/test_phase21.py

Unit and Regression Tests for Phase 21:
Generalization, Adversarial Validation, and Out-of-Distribution Evaluation.
- Frozen Phase 20 parameters check
- Core update equation invariant
- MATCH_THRESHOLD invariant
- Dataset generators for all 7 Phase 21 benchmarks
- Execution sanity checks for D, D4, and D8
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
from benchmarks.phase18_experiment import DenseSemanticEncoder, D_LEX, D_DENSE
from benchmarks.phase21_experiment import (
    run_phase21_stream,
    FROZEN_W,
    FROZEN_LAMBDA,
    FROZEN_K,
    TAU_MIN,
    TAU_HIGH,
    DELTA_MARGIN,
    K_MAX,
    TAU_ANCHOR
)
from benchmarks.phase21_dataset import (
    generate_exp1_unseen_domains,
    generate_exp2_hard_contradictions,
    generate_exp3_hard_paraphrases,
    generate_exp4_distractor_overload,
    generate_exp5_long_horizon,
    generate_exp6_temporal_changes,
    generate_exp7_held_out_benchmark
)

LEXICAL_ENCODER = FieldAwareLexicalEncoder()


@pytest.fixture(scope="module")
def dense_encoder():
    return DenseSemanticEncoder()


@pytest.fixture(scope="module")
def nli_gate():
    return NliSemanticGate()


class TestPhase21FrozenInvariants:
    def test_frozen_parameters(self):
        """Verify that Phase 20 D8 parameters are strictly frozen."""
        assert FROZEN_W == 0.15
        assert FROZEN_LAMBDA == 0.005
        assert FROZEN_K == 3
        assert TAU_MIN == 0.40
        assert TAU_HIGH == 0.75
        assert DELTA_MARGIN == 0.10
        assert K_MAX == 3
        assert TAU_ANCHOR == 0.70
        assert MATCH_THRESHOLD == 0.75

    def test_core_update_equation_invariant(self, dense_encoder, nli_gate):
        """Core update equation slot.value = (1 - alpha_t)*slot.value + alpha_t*x_t strictly holds."""
        for sys_id in ("D", "D4", "D8"):
            stream = [
                Candidate("Alice Vance", "leads development of", "Quantum Compiler", timestamp=1),
                Candidate("Alice Vance", "directs the creation of", "Quantum Compiler", timestamp=2)
            ]
            obs_stream = [
                pytest.importorskip("benchmarks.phase18_dataset").Phase18Obs(c, "c1", "canonical" if idx == 0 else "paraphrase", "test", idx)
                for idx, c in enumerate(stream)
            ]
            res = run_phase21_stream(obs_stream, capacity=5, system=sys_id, nli_gate=nli_gate, dense_encoder=dense_encoder)
            mem = res["memory"]
            assert len(mem.slots) == 1
            assert res["n_updates"] == 1
            x2 = LEXICAL_ENCODER.encode_full(stream[1], dim=D_LEX)
            alpha_t = res["trace"][1]["novelty"]
            # Check equation
            expected = (1.0 - alpha_t) * LEXICAL_ENCODER.encode_full(stream[0], dim=D_LEX) + alpha_t * x2
            np.testing.assert_allclose(mem.slots[0].value, expected, atol=1e-5)


class TestPhase21DatasetGenerators:
    def test_exp1_unseen_domains(self):
        stream = generate_exp1_unseen_domains(inter_gap=2, seed=123)
        assert len(stream) > 0
        categories = {o.category for o in stream}
        assert "canonical" in categories
        assert "paraphrase" in categories
        assert "contradiction" in categories

    def test_exp2_hard_contradictions(self):
        stream = generate_exp2_hard_contradictions(inter_gap=2, seed=123)
        assert len(stream) > 0
        contra_obs = [o for o in stream if o.category == "contradiction"]
        assert len(contra_obs) == 12

    def test_exp3_hard_paraphrases(self):
        stream = generate_exp3_hard_paraphrases(inter_gap=2, seed=123)
        assert len(stream) > 0
        para_obs = [o for o in stream if o.category == "paraphrase"]
        assert len(para_obs) == 10

    def test_exp4_distractor_overload(self):
        stream = generate_exp4_distractor_overload(n_targets=3, distractors_per_target=5, seed=123)
        assert len(stream) > 0
        targets = [o for o in stream if o.category == "canonical"]
        assert len(targets) == 3

    def test_exp5_long_horizon_scaling(self):
        stream = generate_exp5_long_horizon(n_obs=500, seed=123)
        assert len(stream) == 500

    def test_exp6_temporal_changes(self):
        stream = generate_exp6_temporal_changes(n_concepts=3, inter_gap=2, seed=123)
        assert len(stream) > 0
        cats = {o.category for o in stream}
        assert "canonical" in cats
        assert "pre_drift_probe" in cats
        assert "temporal_update" in cats
        assert "post_drift_probe" in cats
        assert "contradiction" in cats

    def test_exp7_held_out_benchmark(self):
        stream = generate_exp7_held_out_benchmark(n_obs=500, seed=123)
        assert len(stream) == 500
        # Determinism check
        stream2 = generate_exp7_held_out_benchmark(n_obs=500, seed=123)
        assert [o.text for o in stream] == [o.text for o in stream2]
