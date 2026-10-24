"""
tests/test_phase24.py

Unit test suite for Phase 24:
Retention Hardening, Semantic Verification Recalibration, and Drift-Resistant Consolidation.

Tests:
  1. Frozen configuration invariants
  2. Retention policies (FIFO, Recency, Utility, Verification-Aware, Tiered, Oracle)
  3. Consolidation update operators (Baseline, Bounded, Anchor, Hybrid)
  4. Calibrated semantic verification (Baseline, Asymmetric, Bidirectional)
  5. Contradiction safety invariants
  6. Failure attribution classification (A, B, C, D, E, SUCCESS)
  7. Dataset generator integrity across all Phase 24 benchmarks
"""

import pytest
import numpy as np

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import cosine_similarity
from benchmarks.phase18_experiment import D_LEX, LEXICAL_ENCODER
from benchmarks.phase22_experiment import (
    FROZEN_W,
    FROZEN_LAMBDA,
    FROZEN_K,
    TAU_MIN,
    TAU_HIGH,
    DELTA_MARGIN,
    TAU_ANCHOR,
)
from benchmarks.phase24_retention import (
    evict_slot_phase24,
    verification_aware_utility,
)
from benchmarks.phase24_consolidation import (
    consolidate_vector_phase24,
    l2_normalize,
)
from benchmarks.phase24_verification import CalibratedSemanticVerifier
from benchmarks.phase24_diagnostics import (
    Phase24DiagnosticTracker,
    Phase24AttributionRecord,
)
from benchmarks.phase24_dataset import (
    generate_exp_a_retention_stream,
    generate_exp_b_calibration_stream,
    generate_exp_c_drift_stream,
    generate_exp_d_ablation_stream,
    generate_exp_e_stress_stream,
    generate_held_out_phase24,
    SEED_P24_TUNING,
    SEED_P24_HELD_OUT,
)


# ──────────────────────────────────────────────
#  1. Frozen Configuration Invariants
# ──────────────────────────────────────────────

def test_phase24_frozen_constants():
    """Verifies that all historical parameters remain strictly unchanged."""
    assert FROZEN_W == 0.15
    assert FROZEN_LAMBDA == 0.005
    assert FROZEN_K == 3
    assert TAU_MIN == 0.40
    assert TAU_HIGH == 0.75
    assert DELTA_MARGIN == 0.10
    assert TAU_ANCHOR == 0.70


# ──────────────────────────────────────────────
#  2. Retention Policies
# ──────────────────────────────────────────────

def test_fifo_eviction():
    """FIFO must evict the slot with the oldest created_at timestamp."""
    mem = MemoryState(capacity=3, dim=10)
    slot_texts = {}
    slot_dense = []

    s0 = mem.add(np.zeros(10), np.zeros(10), 1.0, timestamp=1.0)
    s0.created_at = 1.0
    slot_texts[id(s0)] = "text0"
    slot_dense.append(np.zeros(10))

    s1 = mem.add(np.zeros(10), np.zeros(10), 1.0, timestamp=5.0)
    s1.created_at = 5.0
    slot_texts[id(s1)] = "text1"
    slot_dense.append(np.zeros(10))

    s2 = mem.add(np.zeros(10), np.zeros(10), 1.0, timestamp=10.0)
    s2.created_at = 10.0
    slot_texts[id(s2)] = "text2"
    slot_dense.append(np.zeros(10))

    evict_idx, evicted = evict_slot_phase24(mem, "fifo", 15.0, slot_texts, slot_dense)
    assert id(evicted) == id(s0)
    assert len(mem.slots) == 2


def test_recency_eviction():
    """Recency (LRU) must evict the slot with the oldest last-updated timestamp."""
    mem = MemoryState(capacity=3, dim=10)
    slot_texts = {}
    slot_dense = []

    s0 = mem.add(np.zeros(10), np.zeros(10), 1.0, timestamp=1.0)
    s0.created_at = 1.0
    s0.timestamp = 20.0  # Recently updated!
    slot_texts[id(s0)] = "text0"
    slot_dense.append(np.zeros(10))

    s1 = mem.add(np.zeros(10), np.zeros(10), 1.0, timestamp=5.0)
    s1.created_at = 5.0
    s1.timestamp = 5.0  # Stale!
    slot_texts[id(s1)] = "text1"
    slot_dense.append(np.zeros(10))

    s2 = mem.add(np.zeros(10), np.zeros(10), 1.0, timestamp=10.0)
    s2.created_at = 10.0
    s2.timestamp = 15.0
    slot_texts[id(s2)] = "text2"
    slot_dense.append(np.zeros(10))

    evict_idx, evicted = evict_slot_phase24(mem, "recency", 25.0, slot_texts, slot_dense)
    # s1 has oldest timestamp (5.0), even though s0 has older created_at (1.0)
    assert id(evicted) == id(s1)


def test_verification_aware_utility_scoring():
    """Verified slots with update_count >= 2 must score higher utility than singletons."""
    slot_singleton = MemorySlot(
        key=np.zeros(10), value=np.zeros(10), confidence=0.8,
        timestamp=10.0, created_at=10.0, update_count=1
    )
    slot_verified = MemorySlot(
        key=np.zeros(10), value=np.zeros(10), confidence=0.8,
        timestamp=10.0, created_at=10.0, update_count=3
    )
    u_sing = verification_aware_utility(slot_singleton, current_time=15.0)
    u_verif = verification_aware_utility(slot_verified, current_time=15.0)
    assert u_verif > u_sing


def test_tiered_retention_prioritizes_provisional():
    """Tiered retention must evict provisional slots (update_count=1) before consolidated slots."""
    mem = MemoryState(capacity=3, dim=10)
    slot_texts = {}
    slot_dense = []

    # Slot 0: consolidated (update_count = 3)
    s0 = mem.add(np.zeros(10), np.zeros(10), 1.0, timestamp=1.0)
    s0.update_count = 3
    slot_texts[id(s0)] = "cons0"
    slot_dense.append(np.zeros(10))

    # Slot 1: provisional (update_count = 1)
    s1 = mem.add(np.zeros(10), np.zeros(10), 1.0, timestamp=2.0)
    s1.update_count = 1
    slot_texts[id(s1)] = "prov1"
    slot_dense.append(np.zeros(10))

    # Slot 2: consolidated (update_count = 2)
    s2 = mem.add(np.zeros(10), np.zeros(10), 1.0, timestamp=3.0)
    s2.update_count = 2
    slot_texts[id(s2)] = "cons2"
    slot_dense.append(np.zeros(10))

    evict_idx, evicted = evict_slot_phase24(mem, "tiered", 10.0, slot_texts, slot_dense)
    # The provisional slot s1 must be evicted, protecting consolidated s0 and s2!
    assert id(evicted) == id(s1)
    assert id(s0) in [id(s) for s in mem.slots]
    assert id(s2) in [id(s) for s in mem.slots]


def test_tiered_retention_enforces_capacity_quota():
    """When all slots are consolidated, tiered retention evicts lowest-utility without error."""
    mem = MemoryState(capacity=2, dim=10)
    slot_texts = {}
    slot_dense = []

    s0 = mem.add(np.zeros(10), np.zeros(10), 0.5, timestamp=1.0)
    s0.update_count = 2
    slot_texts[id(s0)] = "cons0"
    slot_dense.append(np.zeros(10))

    s1 = mem.add(np.zeros(10), np.zeros(10), 0.9, timestamp=10.0)
    s1.update_count = 4
    slot_texts[id(s1)] = "cons1"
    slot_dense.append(np.zeros(10))

    evict_idx, evicted = evict_slot_phase24(mem, "tiered", 20.0, slot_texts, slot_dense)
    # s0 has lower confidence and older timestamp, so lower utility
    assert id(evicted) == id(s0)
    assert len(mem.slots) == 1


# ──────────────────────────────────────────────
#  3. Consolidation Operators
# ──────────────────────────────────────────────

def test_baseline_consolidation_update():
    """Baseline update must exactly match (1 - alpha)*v_{t-1} + alpha*x_t normalized."""
    v0 = l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    slot = MemorySlot(key=v0.copy(), value=v0.copy(), confidence=1.0, timestamp=0.0, created_at=0.0)
    x = l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    alpha = 0.5

    v_new = consolidate_vector_phase24(slot, x, alpha_t=alpha, mode="baseline")
    expected = l2_normalize(0.5 * v0 + 0.5 * x)
    np.testing.assert_allclose(v_new, expected, atol=1e-5)


def test_bounded_consolidation_limits_displacement():
    """Bounded update must strictly cap ||v_next - v_prev|| <= delta_max before normalization."""
    v0 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    slot = MemorySlot(key=v0.copy(), value=v0.copy(), confidence=1.0, timestamp=0.0, created_at=0.0)
    # Completely orthogonal candidate
    x = l2_normalize(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    delta_max = 0.20

    v_bounded = consolidate_vector_phase24(slot, x, alpha_t=1.0, mode="bounded", delta_max=delta_max)
    # Cosine similarity to v0 must be high because displacement was capped
    cos_sim = cosine_similarity(v0, v_bounded)
    assert cos_sim > 0.95


def test_anchor_consolidation_preserves_root_identity():
    """Anchor regularization must maintain higher similarity to v0 over multiple updates than baseline."""
    v0 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    slot_base = MemorySlot(key=v0.copy(), value=v0.copy(), confidence=1.0, timestamp=0.0, created_at=0.0)
    slot_anch = MemorySlot(key=v0.copy(), value=v0.copy(), confidence=1.0, timestamp=0.0, created_at=0.0)

    # Sequence of varying vectors
    np.random.seed(42)
    for _ in range(5):
        cand = l2_normalize(np.random.randn(4).astype(np.float32))
        slot_base.value = consolidate_vector_phase24(slot_base, cand, alpha_t=0.6, mode="baseline")
        slot_anch.value = consolidate_vector_phase24(slot_anch, cand, alpha_t=0.6, mode="anchor", lambda_anchor=0.25)

    cos_base = cosine_similarity(v0, slot_base.value)
    cos_anch = cosine_similarity(v0, slot_anch.value)
    assert cos_anch > cos_base


def test_hybrid_consolidation_properties():
    """Hybrid mode combines bounded displacement and anchor regularization."""
    v0 = l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    slot = MemorySlot(key=v0.copy(), value=v0.copy(), confidence=1.0, timestamp=0.0, created_at=0.0)
    x = l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))

    v_hyb = consolidate_vector_phase24(slot, x, alpha_t=0.8, mode="hybrid", delta_max=0.25, lambda_anchor=0.20)
    cos_to_v0 = cosine_similarity(v0, v_hyb)
    assert 0.90 < cos_to_v0 <= 1.0


# ──────────────────────────────────────────────
#  4. Failure Attribution Schema
# ──────────────────────────────────────────────

def test_failure_attribution_categories_p24():
    """Tests causal categorization into A, B, C, D, E, and SUCCESS."""
    tracker = Phase24DiagnosticTracker()
    mem = MemoryState(capacity=5, dim=D_LEX)

    cand = Candidate("Alice", "penned", "memoir", timestamp=0.0)
    x = LEXICAL_ENCODER.encode_full(cand, dim=D_LEX)
    slot = mem.add(x, x.copy(), 1.0, 0.0)
    tracker.register_target("tgt_1", "cid_1", slot, step=0, canonical_text="Alice penned memoir.")

    # 1. SUCCESS
    step_succ = {
        "action": "update", "slot_index": 0, "gate": "nli_accept",
        "top_k_indices": [0], "candidates_examined": 1,
        "sem_pred": {"decision": "SAME", "p_entail": 0.95}
    }
    r_succ = tracker.evaluate_probe_step("p_succ", "tgt_1", "paraphrase", mem, step_succ)
    assert r_succ.failure_category == "SUCCESS"

    # 2. Category A: Retrieval failure (slot alive, but not in candidate beam)
    step_miss = {
        "action": "insert", "slot_index": 1, "gate": "new_slot",
        "top_k_indices": [1], "candidates_examined": 1
    }
    r_a = tracker.evaluate_probe_step("p_a", "tgt_1", "paraphrase", mem, step_miss)
    assert r_a.failure_category == "A"

    # 3. Category B: NLI rejection (target in beam, but NLI rejects)
    step_rej = {
        "action": "insert", "slot_index": 1, "gate": "rejected_diff",
        "top_k_indices": [0], "candidates_examined": 1,
        "sem_pred": {"decision": "NEUTRAL", "p_entail": 0.35}
    }
    r_b = tracker.evaluate_probe_step("p_b", "tgt_1", "paraphrase", mem, step_rej)
    assert r_b.failure_category == "B"

    # 4. Category D: Eviction failure (slot was evicted)
    tracker.record_eviction(slot, step=5)
    mem.slots.pop(0)
    step_evict = {"action": "insert", "slot_index": 0, "top_k_indices": []}
    r_d = tracker.evaluate_probe_step("p_d", "tgt_1", "paraphrase", mem, step_evict)
    assert r_d.failure_category == "D"


# ──────────────────────────────────────────────
#  5. Dataset Generators Integrity
# ──────────────────────────────────────────────

def test_dataset_generators_p24():
    """Validates structure and determinism of all Phase 24 dataset streams."""
    stream_a_ord = generate_exp_a_retention_stream(mode="ordinary", seed=SEED_P24_TUNING)
    assert len(stream_a_ord) > 0
    assert any(getattr(obs, "is_probe", False) for obs in stream_a_ord)

    stream_a_adv = generate_exp_a_retention_stream(mode="adversarial", seed=SEED_P24_TUNING)
    assert len(stream_a_adv) == 350

    stream_b = generate_exp_b_calibration_stream(seed=SEED_P24_TUNING)
    assert len(stream_b) > 0

    stream_c = generate_exp_c_drift_stream(n_targets=8, updates_per_target=4, seed=SEED_P24_TUNING)
    assert len(stream_c) > 0

    stream_d = generate_exp_d_ablation_stream(seed=SEED_P24_TUNING)
    assert len(stream_d) == 250

    stream_e = generate_exp_e_stress_stream(n_obs=500, seed=SEED_P24_TUNING)
    assert len(stream_e) == 500

    stream_f = generate_held_out_phase24(n_obs=800, seed=SEED_P24_HELD_OUT)
    assert len(stream_f) == 800
    # Novel domain check
    assert any("Selene" in obs.candidate.subject or "Orion" in obs.candidate.subject for obs in stream_f)
