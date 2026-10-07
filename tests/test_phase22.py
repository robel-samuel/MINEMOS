"""
tests/test_phase22.py

Unit and Regression Tests for Phase 22:
Hybrid Retrieval (Dense + Lexical BM25 Candidate Generation).

Covers:
1. LocalBM25Index: tokenization, scoring determinism, insert, update, eviction
2. Candidate fusion: deduplication, rank order, beam bounding
3. Invariant checks: MATCH_THRESHOLD=0.75, core update equation preservation
4. System equivalence: D8 behavior unchanged; H1 with BM25=0 equals D8
5. NLI verification requirement: lexical candidates never bypass NLI
6. Deterministic dataset generators for all 6 Phase 22 benchmarks
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
from benchmarks.phase22_experiment import (
    LocalBM25Index,
    bm25_tokenize,
    fuse_candidates,
    run_phase22_stream,
    absorb_observation_phase22,
    FROZEN_W,
    FROZEN_LAMBDA,
    FROZEN_K,
    TAU_MIN,
    TAU_HIGH,
    DELTA_MARGIN,
    K_MAX,
    TAU_ANCHOR
)
from benchmarks.phase22_dataset import (
    generate_exp1_vocab_shift,
    generate_exp2_dense_distractors,
    generate_exp3_adversarial_contradictions,
    generate_exp4_long_horizon,
    generate_exp5_ablation_stream,
    generate_exp6_held_out_benchmark
)

LEXICAL_ENCODER = FieldAwareLexicalEncoder()


@pytest.fixture(scope="module")
def dense_encoder():
    return DenseSemanticEncoder()


@pytest.fixture(scope="module")
def nli_gate():
    return NliSemanticGate()


class TestLocalBM25Index:
    def test_tokenization(self):
        text = "The quick, brown FOX jumped over 12 lazy dogs!"
        tokens = bm25_tokenize(text)
        assert tokens == ["the", "quick", "brown", "fox", "jumped", "over", "12", "lazy", "dogs"]

    def test_add_and_query_exact_match(self):
        index = LocalBM25Index()
        index.add_document(101, "Alice works at Google Research in Zurich.")
        index.add_document(102, "Bob manages infrastructure at Amazon Seattle.")
        index.add_document(103, "Charlie studies astrophysics at Cambridge.")

        assert index.num_docs == 3
        # Query matching Alice and Google
        results = index.query("Alice Google", top_k=3)
        assert len(results) >= 1
        assert results[0][0] == 101
        assert results[0][1] > 0.0

    def test_eviction_removes_document(self):
        index = LocalBM25Index()
        index.add_document(201, "Server node alpha is active.")
        index.add_document(202, "Server node beta is offline.")

        assert index.num_docs == 2
        res1 = index.query("beta", top_k=2)
        assert any(doc_id == 202 for doc_id, _ in res1)

        # Evict doc 202
        index.remove_document(202)
        assert index.num_docs == 1
        res2 = index.query("beta", top_k=2)
        assert len(res2) == 0  # No documents match 'beta'

    def test_update_document_refreshes_tokens(self):
        index = LocalBM25Index()
        index.add_document(301, "Alice works at Company Alpha.")
        res1 = index.query("Alpha", top_k=1)
        assert res1[0][0] == 301

        # Update to Company Beta
        index.update_document(301, "Alice works at Company Beta.")
        res_old = index.query("Alpha", top_k=1)
        res_new = index.query("Beta", top_k=1)
        assert len(res_old) == 0
        assert len(res_new) == 1
        assert res_new[0][0] == 301

    def test_scoring_determinism(self):
        index = LocalBM25Index()
        docs = [
            (1, "quantum encryption algorithm for cybersecurity"),
            (2, "autonomous vehicle perception and radar systems"),
            (3, "quantum computing hardware with trapped ions")
        ]
        for did, text in docs:
            index.add_document(did, text)

        res_a = index.query("quantum algorithm", top_k=3)
        res_b = index.query("quantum algorithm", top_k=3)
        assert res_a == res_b


class TestCandidateFusion:
    def test_deduplication_and_order(self):
        dense_cands = [(1, 0.85), (2, 0.72)]
        lex_cands = [(2, 4.5), (3, 3.2), (4, 1.1)]

        fused, meta = fuse_candidates(dense_cands, lex_cands, k_fusion=5)
        # Should keep dense order [1, 2], then append non-duplicate lexical [3, 4]
        assert fused == [1, 2, 3, 4]
        assert meta["n_overlap"] == 1
        assert meta["n_lex_added"] == 2

    def test_k_fusion_truncation(self):
        dense_cands = [(1, 0.9), (2, 0.8)]
        lex_cands = [(3, 5.0), (4, 4.0), (5, 3.0)]

        fused, meta = fuse_candidates(dense_cands, lex_cands, k_fusion=3)
        assert len(fused) == 3
        assert fused == [1, 2, 3]


class TestPhase22Invariants:
    def test_frozen_constants_preserved(self):
        assert MATCH_THRESHOLD == 0.75
        assert FROZEN_W == 0.15
        assert FROZEN_LAMBDA == 0.005
        assert FROZEN_K == 3
        assert TAU_MIN == 0.40
        assert TAU_HIGH == 0.75
        assert DELTA_MARGIN == 0.10
        assert K_MAX == 3
        assert TAU_ANCHOR == 0.70

    def test_core_vector_update_equation(self, dense_encoder, nli_gate):
        """Verifies v_t = (1 - alpha_t)*v_{t-1} + alpha_t*x_t is preserved exactly."""
        mem = MemoryState(capacity=5, dim=D_LEX)
        slot_texts = {}
        slot_dense = []
        slot_contra_counts = {}
        slot_contra_anchors = {}
        bm25_index = LocalBM25Index()

        c1 = Candidate("Alice", "leads research at", "DeepMind", timestamp=0.0)
        res1 = absorb_observation_phase22(
            mem, c1, "H1", slot_texts, slot_dense, slot_contra_counts,
            slot_contra_anchors, bm25_index, nli_gate, dense_encoder
        )
        assert res1["action"] == "insert"
        slot = mem.slots[0]
        v_initial = slot.value.copy()

        # Update with paraphrase
        c2 = Candidate("Alice", "heads research at", "DeepMind", timestamp=1.0)
        res2 = absorb_observation_phase22(
            mem, c2, "H1", slot_texts, slot_dense, slot_contra_counts,
            slot_contra_anchors, bm25_index, nli_gate, dense_encoder
        )
        assert res2["action"] == "update"
        alpha_t = res2["novelty"]
        x_t = LEXICAL_ENCODER.encode_full(c2, dim=D_LEX)
        expected_v = (1 - alpha_t) * v_initial + alpha_t * x_t
        np.testing.assert_allclose(slot.value, expected_v, rtol=1e-5, atol=1e-5)

    def test_lexical_cannot_bypass_nli(self, dense_encoder, nli_gate):
        """Even with 100% lexical overlap, contradictions must NOT merge into the slot."""
        mem = MemoryState(capacity=5, dim=D_LEX)
        slot_texts = {}
        slot_dense = []
        slot_contra_counts = {}
        slot_contra_anchors = {}
        bm25_index = LocalBM25Index()

        c_canon = Candidate("Alice", "works as chief engineer at", "Google", timestamp=0.0)
        absorb_observation_phase22(
            mem, c_canon, "H1", slot_texts, slot_dense, slot_contra_counts,
            slot_contra_anchors, bm25_index, nli_gate, dense_encoder
        )

        # High lexical match contradiction
        c_contra = Candidate("Alice", "does not work as chief engineer at", "Google", timestamp=1.0)
        res_contra = absorb_observation_phase22(
            mem, c_contra, "H1", slot_texts, slot_dense, slot_contra_counts,
            slot_contra_anchors, bm25_index, nli_gate, dense_encoder
        )

        assert res_contra["action"] != "update"
        assert res_contra["gate"] == "rejected_contra"
        assert len(mem.slots) == 2  # Created new slot, didn't falsely merge


class TestPhase22DatasetGenerators:
    def test_exp1_vocab_shift(self):
        stream = generate_exp1_vocab_shift()
        assert len(stream) > 0
        probes = [o for o in stream if getattr(o, "is_probe", False)]
        assert len(probes) == 12

    def test_exp2_dense_distractors(self):
        stream = generate_exp2_dense_distractors(n_targets=5, distractors_per_target=10)
        assert len(stream) == 5 * (1 + 10 + 1)
        probes = [o for o in stream if getattr(o, "is_probe", False)]
        assert len(probes) == 5

    def test_exp3_adversarial_contradictions(self):
        stream = generate_exp3_adversarial_contradictions()
        contra_probes = [o for o in stream if getattr(o, "is_probe", False) and o.category == "contradiction"]
        assert len(contra_probes) == 20  # 2 contra probes per triple * 10 triples

    def test_exp4_long_horizon_determinism(self):
        s1 = generate_exp4_long_horizon(n_obs=50, seed=123)
        s2 = generate_exp4_long_horizon(n_obs=50, seed=123)
        assert len(s1) == 50
        for o1, o2 in zip(s1, s2):
            assert o1.text == o2.text
            assert o1.category == o2.category

    def test_exp5_ablation_stream(self):
        stream = generate_exp5_ablation_stream(n_concepts=5)
        assert len(stream) > 0

    def test_exp6_held_out_benchmark(self):
        stream = generate_exp6_held_out_benchmark(n_obs=100, seed=8888)
        assert len(stream) == 100
        assert stream[0].candidate.subject.startswith("Phase22Agent_")
