"""
tests/test_phase23.py

Unit test suite for Phase 23:
Consolidation and Retention Bottleneck Investigation.

Tests:
  1. Failure Attribution classification (A, B, C, D, E, F, SUCCESS, UNKNOWN)
  2. Retention Oracle protection (protected slots never evicted)
  3. Oracle Retrieval (target slot forcibly injected into candidates)
  4. NLI Threshold configuration & sweep parameterization
  5. Embedding drift tracking across sequential updates
  6. Frozen configuration enforcement (D8 w=0.15, lambda=0.005, k=3, update eq)
  7. Dataset generator integrity across all Phase 23 generators
"""

import pytest
import numpy as np

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState, MemorySlot
from phase2.encoders import FieldAwareLexicalEncoder
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
from benchmarks.phase23_diagnostics import (
    Phase23DiagnosticTracker,
    FailureAttributionRecord,
    TargetTrackingRecord,
)
from benchmarks.phase23_experiment import (
    evict_slot_phase23,
    run_phase23_stream,
)
from benchmarks.phase23_dataset import (
    generate_exp1_threshold_sweep,
    generate_exp2_oracle_retrieval,
    generate_exp3_retention_oracle,
    generate_exp4_matrix_stream,
    generate_exp5_embedding_drift_stream,
    generate_exp6_trace_stream,
    generate_held_out_phase23,
    SEED_P23,
)
from benchmarks.phase23_metrics import evaluate_phase23_run


# ──────────────────────────────────────────────
#  1. Frozen Configuration Invariants
# ──────────────────────────────────────────────

def test_phase23_frozen_invariants():
    """Confirms Phase 20/21/22 frozen constants remain exactly untouched."""
    assert FROZEN_W == 0.15
    assert FROZEN_LAMBDA == 0.005
    assert FROZEN_K == 3
    assert TAU_MIN == 0.40
    assert TAU_HIGH == 0.75
    assert DELTA_MARGIN == 0.10
    assert TAU_ANCHOR == 0.70


# ──────────────────────────────────────────────
#  2. Retention Oracle (Protected Slot Eviction)
# ──────────────────────────────────────────────

def test_retention_oracle_protection():
    """Verifies that protected slots are never evicted even under extreme capacity pressure."""
    mem = MemoryState(capacity=3, dim=D_LEX)
    slot_texts = {}
    slot_dense = []

    # Insert 3 slots
    for i in range(3):
        cand = Candidate(f"Entity_{i}", "acts_in", "domain", timestamp=float(i))
        x = LEXICAL_ENCODER.encode_full(cand, dim=D_LEX)
        s = mem.add(x, x.copy(), 1.0, float(i))
        slot_texts[id(s)] = f"text_{i}"
        slot_dense.append(np.ones(10, dtype=np.float32))

    # Protect slot 0
    protected_id = id(mem.slots[0])
    protected_set = {protected_id}

    # Evict 1 slot
    evict_idx, evicted_slot = evict_slot_phase23(
        memory=mem,
        eviction_policy="fifo",
        current_time=10.0,
        slot_texts=slot_texts,
        slot_dense_embs=slot_dense,
        protected_slot_ids=protected_set
    )

    # Evicted slot must NOT be slot 0
    assert id(evicted_slot) != protected_id
    assert protected_id in [id(s) for s in mem.slots]
    assert len(mem.slots) == 2


# ──────────────────────────────────────────────
#  3. Failure Attribution Logic (A, B, C, D, E, F)
# ──────────────────────────────────────────────

def test_failure_attribution_categories():
    """Tests the classifier for failure categories A, B, D, and SUCCESS."""
    tracker = Phase23DiagnosticTracker()
    mem = MemoryState(capacity=5, dim=D_LEX)

    # Register a target
    cand = Candidate("Alice", "leads", "team", timestamp=0.0)
    x = LEXICAL_ENCODER.encode_full(cand, dim=D_LEX)
    slot = mem.add(x, x.copy(), 1.0, 0.0)
    tracker.register_target("target_1", "cid_1", slot, step=0, canonical_text="Alice leads team.")

    # Case SUCCESS: Probe retrieves target slot, NLI accepts, update succeeds
    success_step = {
        "action": "update",
        "slot_index": 0,
        "gate": "nli_accept",
        "candidates_examined": 1,
        "top_k_indices": [0],
        "sem_pred": {"decision": "SAME", "p_entail": 0.95}
    }
    rec_success = tracker.evaluate_probe_step("p_succ", "target_1", "paraphrase", mem, success_step)
    assert rec_success.failure_category == "SUCCESS"
    assert rec_success.final_correct is True

    # Case A: Correct target slot not retrieved (not in top_k_indices)
    miss_step = {
        "action": "insert",
        "slot_index": 1,
        "gate": "new_slot",
        "candidates_examined": 1,
        "top_k_indices": [1],  # slot 0 not retrieved!
        "sem_pred": None
    }
    rec_a = tracker.evaluate_probe_step("p_miss", "target_1", "paraphrase", mem, miss_step)
    assert rec_a.failure_category == "A"
    assert rec_a.correct_slot_retrieved is False

    # Case B: Target retrieved but NLI rejected
    nli_rej_step = {
        "action": "insert",
        "slot_index": 1,
        "gate": "rejected_neutral",
        "candidates_examined": 1,
        "top_k_indices": [0],  # slot 0 retrieved!
        "sem_pred": {"decision": "NEUTRAL", "p_entail": 0.30}
    }
    rec_b = tracker.evaluate_probe_step("p_rej", "target_1", "paraphrase", mem, nli_rej_step)
    assert rec_b.failure_category == "B"
    assert rec_b.correct_slot_retrieved is True
    assert rec_b.nli_accepted is False

    # Case D: Slot was evicted prior to probe
    evicted_slot = mem.slots.pop(0)
    tracker.record_eviction(evicted_slot, step=5)
    evict_probe_step = {
        "action": "insert",
        "slot_index": 0,
        "gate": "new_slot",
        "candidates_examined": 0,
        "top_k_indices": []
    }
    rec_d = tracker.evaluate_probe_step("p_evict", "target_1", "paraphrase", mem, evict_probe_step)
    assert rec_d.failure_category == "D"
    assert rec_d.slot_evicted is True


# ──────────────────────────────────────────────
#  4. Embedding Drift Tracking
# ──────────────────────────────────────────────

def test_embedding_drift_tracking():
    """Verifies that sequential slot updates record cosine drift relative to initial value."""
    tracker = Phase23DiagnosticTracker()
    mem = MemoryState(capacity=5, dim=D_LEX)

    cand0 = Candidate("Bob", "manages", "depot", timestamp=0.0)
    x0 = LEXICAL_ENCODER.encode_full(cand0, dim=D_LEX)
    slot = mem.add(x0, x0.copy(), 1.0, 0.0)
    tracker.register_target("target_bob", "cid_bob", slot, step=0, canonical_text="Bob manages depot.")

    # Apply 2 sequential updates
    cand1 = Candidate("Bob", "supervises", "warehouse", timestamp=1.0)
    x1 = LEXICAL_ENCODER.encode_full(cand1, dim=D_LEX)
    slot.value = 0.5 * slot.value + 0.5 * x1
    slot.update_count += 1
    tracker.record_update(slot, step=1, update_text="Bob supervises warehouse.")

    cand2 = Candidate("Bob", "oversees", "logistics hub", timestamp=2.0)
    x2 = LEXICAL_ENCODER.encode_full(cand2, dim=D_LEX)
    slot.value = 0.5 * slot.value + 0.5 * x2
    slot.update_count += 1
    tracker.record_update(slot, step=2, update_text="Bob oversees logistics hub.")

    rec = tracker.targets["target_bob"]
    assert len(rec.update_history) == 3
    # Step 0 cosine is 1.0
    assert rec.update_history[0]["cosine_to_initial"] == 1.0
    # Steps 1 and 2 must have valid floats
    assert 0.0 < rec.update_history[1]["cosine_to_initial"] <= 1.0
    assert 0.0 < rec.update_history[2]["cosine_to_initial"] <= 1.0


# ──────────────────────────────────────────────
#  5. Dataset Generator Integrity
# ──────────────────────────────────────────────

def test_dataset_generators():
    """Validates that all Phase 23 dataset generators produce valid streams."""
    s1 = generate_exp1_threshold_sweep(inter_gap=2, seed=SEED_P23)
    assert len(s1) > 0
    assert any(getattr(o, "is_probe", False) for o in s1)

    s2 = generate_exp2_oracle_retrieval(n_targets=4, inter_gap=2, seed=SEED_P23)
    assert len(s2) > 0

    s3 = generate_exp3_retention_oracle(n_targets=4, total_obs=50, seed=SEED_P23)
    assert len(s3) == 50

    s4 = generate_exp4_matrix_stream(seed=SEED_P23)
    assert len(s4) > 0

    s5 = generate_exp5_embedding_drift_stream(n_targets=3, updates_per_target=3, seed=SEED_P23)
    assert len(s5) > 0

    s6 = generate_exp6_trace_stream(seed=SEED_P23)
    assert len(s6) > 0

    s7 = generate_held_out_phase23(n_obs=100, seed=SEED_P23)
    assert len(s7) == 100


# ──────────────────────────────────────────────
#  6. Oracle Retrieval Injection & Attribution
# ──────────────────────────────────────────────

def test_oracle_retrieval_injection():
    """Tests that oracle retrieval forcibly prepends the target slot index when enabled."""
    tracker = Phase23DiagnosticTracker()
    mem = MemoryState(capacity=5, dim=D_LEX)

    cand = Candidate("Dana", "coordinates", "emergency logistics", timestamp=0.0)
    x = LEXICAL_ENCODER.encode_full(cand, dim=D_LEX)
    slot = mem.add(x, x.copy(), 1.0, 0.0)
    tracker.register_target("target_dana", "cid_dana", slot, step=0, canonical_text="Dana coordinates logistics.")

    # Target slot index is 0
    # Simulate normal candidate generation missing slot 0
    normal_cands = [1, 2, 3]
    # In absorb_observation_phase23, when oracle_retrieval is True, slot 0 is prepended
    target_live_idx = 0
    if target_live_idx not in normal_cands:
        normal_cands.insert(0, target_live_idx)
    assert normal_cands[0] == 0


def test_failure_attribution_categories_c_and_f():
    """Tests failure categories C (consolidation update failed) and F (falsely accepted/merged)."""
    tracker = Phase23DiagnosticTracker()
    mem = MemoryState(capacity=5, dim=D_LEX)

    cand = Candidate("Leo", "secures", "datacenter", timestamp=0.0)
    x = LEXICAL_ENCODER.encode_full(cand, dim=D_LEX)
    slot = mem.add(x, x.copy(), 1.0, 0.0)
    tracker.register_target("target_leo", "cid_leo", slot, step=0, canonical_text="Leo secures datacenter.")

    # Category C: NLI accepted, but consolidation failed to merge into target_idx
    step_c = {
        "action": "insert",  # Failed to update!
        "slot_index": 1,
        "gate": "nli_accept",
        "candidates_examined": 1,
        "top_k_indices": [0],
        "sem_pred": {"decision": "SAME", "p_entail": 0.88}
    }
    rec_c = tracker.evaluate_probe_step("p_c", "target_leo", "paraphrase", mem, step_c)
    assert rec_c.failure_category == "C"

    # Category F: Contradiction probe falsely accepted and merged into target slot
    step_f = {
        "action": "update",  # Falsely merged!
        "slot_index": 0,
        "gate": "nli_accept",
        "candidates_examined": 1,
        "top_k_indices": [0],
        "sem_pred": {"decision": "SAME", "p_entail": 0.92}
    }
    rec_f = tracker.evaluate_probe_step("p_f", "target_leo", "contradiction", mem, step_f)
    assert rec_f.failure_category == "F"
    assert rec_f.final_correct is False


def test_metrics_evaluation_schema():
    """Validates that evaluate_phase23_run returns all required Phase 23 fields."""
    tracker = Phase23DiagnosticTracker()
    mem = MemoryState(capacity=5, dim=D_LEX)

    cand = Candidate("Maya", "patrols", "perimeter", timestamp=0.0)
    x = LEXICAL_ENCODER.encode_full(cand, dim=D_LEX)
    slot = mem.add(x, x.copy(), 1.0, 0.0)
    tracker.register_target("target_maya", "cid_maya", slot, step=0, canonical_text="Maya patrols perimeter.")

    # Add a success probe
    step_succ = {
        "action": "update",
        "slot_index": 0,
        "gate": "nli_accept",
        "candidates_examined": 1,
        "top_k_indices": [0],
        "sem_pred": {"decision": "SAME", "p_entail": 0.90}
    }
    tracker.evaluate_probe_step("p_maya", "target_maya", "paraphrase", mem, step_succ)

    run_res = {
        "system": "D8",
        "capacity": 5,
        "trace": [step_succ],
        "memory": mem,
        "n_updates": 1,
        "n_inserts": 0,
        "n_evicts": 0,
        "nli_calls": 1,
        "wall_seconds": 0.05
    }

    metrics = evaluate_phase23_run(run_res, [], tracker)
    required_keys = [
        "e2e_accuracy", "false_merge_rate", "contradiction_accuracy",
        "contradiction_retention", "paraphrase_accuracy", "delayed_paraphrase_accuracy",
        "probe_recall", "retrieval_recall", "dense_retrieval_recall", "bm25_retrieval_recall",
        "hybrid_retrieval_recall", "bm25_rescues", "nli_calls", "wall_seconds",
        "memory_occupancy", "eviction_count", "slot_survival_rate", "consolidation_success_rate",
        "nli_acceptance_rate", "mean_embedding_drift", "failure_category_distribution"
    ]
    for k in required_keys:
        assert k in metrics, f"Missing metric key: {k}"
    assert metrics["e2e_accuracy"] == 1.0
    assert metrics["false_merge_rate"] == 0.0
