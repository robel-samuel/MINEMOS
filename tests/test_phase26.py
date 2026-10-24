"""
tests/test_phase26.py

Unit tests for Phase 26 Semantic Verification Calibration and Contradiction Safety:
  1. Calibration recovers asymmetric valid paraphrases.
  2. Genuine contradictions with high dense similarity are never merged.
  3. Negation parity guard strictly blocks false merges.
  4. Numerical and scalar guard strictly blocks false merges.
  5. Conflicting entity guard strictly blocks false merges.
  6. Unrelated statements are rejected from consolidation.
  7. Temporal state changes trigger appropriate supersession windows.
  8. Point-in-time recall and entity history retrieval remain exact.
  9. Conservative behavior when verification is uncertain.
 10. End-to-end BlackHoleMemory consolidation with calibrated verifier.
"""

import pytest
import numpy as np

from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate
from benchmarks.phase26_dataset import get_dataset_splits, verify_dataset_integrity
from benchmarks.phase26_verification import (
    Phase26Verifier,
    extract_numbers,
    extract_negation_markers,
    check_entity_mismatch,
    semantic_safety_guards_pass
)
from benchmarks.phase26_metrics import compute_verification_metrics


# ─────────────────────────────────────────────────────────────
# 1. Dataset Integrity & Isolation
# ─────────────────────────────────────────────────────────────

def test_dataset_split_integrity():
    """Confirms 60 pairs per split, 15 per category, and zero leakage."""
    res = verify_dataset_integrity()
    assert res["status"] == "PASS"
    assert res["counts"]["calibration"] == 60
    assert res["counts"]["validation"] == 60
    assert res["counts"]["held_out"] == 60
    assert len(res["held_out_overlap"]) == 0


# ─────────────────────────────────────────────────────────────
# 2. Semantic Safety Guards Unit Tests
# ─────────────────────────────────────────────────────────────

def test_extract_numbers():
    """Validates robust number and percentage extraction."""
    t1 = "George owns a 15% stake in the logistics firm."
    t2 = "George owns an 85% stake in the logistics firm."
    nums1 = extract_numbers(t1)
    nums2 = extract_numbers(t2)
    assert nums1 == {"15%"}
    assert nums2 == {"85%"}
    assert nums1 != nums2

    # Written numbers
    t3 = "Nadia translated three classical poetry volumes."
    assert "3" in extract_numbers(t3)


def test_extract_negation_markers():
    """Validates extraction of negation tokens."""
    t1 = "Bob holds a PhD in quantum physics."
    t2 = "Bob does not hold a PhD in quantum physics."
    t3 = "Nathan never served as lead architect."
    assert len(extract_negation_markers(t1)) == 0
    assert "not" in extract_negation_markers(t2)
    assert "never" in extract_negation_markers(t3)


def test_check_entity_mismatch():
    """Confirms capitalized proper noun mismatch detection."""
    p1 = "Alice leads engineering at Nexus Labs."
    h1 = "Alice now leads engineering at Apex Systems."
    assert check_entity_mismatch(p1, h1) is True

    # Same entities
    p2 = "Alice leads engineering at Nexus Labs."
    h2 = "Alice directs engineering at Nexus Labs."
    assert check_entity_mismatch(p2, h2) is False


def test_safety_guards_veto_negation_contradiction():
    """Negation difference MUST veto fallback regardless of high dense similarity."""
    p = "Bob holds a PhD in quantum physics."
    h = "Bob does not hold a PhD in quantum physics."
    # High contra (0.99) and negation mismatch
    ok, reason = semantic_safety_guards_pass(p, h, p_contra_max=0.99)
    assert ok is False
    assert "Negation" in reason or "contradiction too high" in reason


def test_safety_guards_veto_numerical_contradiction():
    """Numerical difference MUST veto fallback."""
    p = "George owns a 15% stake in the logistics firm."
    h = "George owns an 85% stake in the logistics firm."
    ok, reason = semantic_safety_guards_pass(p, h, p_contra_max=0.45)
    assert ok is False
    assert "Numerical mismatch" in reason


def test_safety_guards_veto_entity_contradiction():
    """Conflicting entities MUST veto fallback."""
    p = "Alice leads engineering at Nexus Labs."
    h = "Alice now leads engineering at Apex Systems."
    ok, reason = semantic_safety_guards_pass(p, h, p_contra_max=0.55)
    assert ok is False
    assert "entities" in reason


# ─────────────────────────────────────────────────────────────
# 3. Pairwise Verification Behaviors
# ─────────────────────────────────────────────────────────────

def test_calibrated_recovers_asymmetric_paraphrase():
    """
    Validates that calibration recovers the known asymmetric pair:
    'Bob holds a PhD...' vs 'Bob earned a doctorate...'
    which baseline vetoed due to max_contra.
    """
    verifier = Phase26Verifier()
    p = "Bob holds a PhD in quantum physics."
    h = "Bob earned a doctorate in quantum physics."

    # Baseline Phase 25 rejects it as CONTRADICTION
    r_base = verifier.verify_pair(p, h, mode="baseline")
    assert r_base["decision"] == "CONTRADICTION"

    # Calibrated recovers it as SAME (since min_contra < 0.40 and max_entail >= 0.55)
    r_cal = verifier.verify_pair(p, h, mode="calibrated")
    assert r_cal["decision"] == "SAME"


def test_genuine_contradiction_high_dense_sim_protected():
    """
    Even when dense similarity is very high (> 0.85),
    genuine contradictions MUST be classified as CONTRADICTION across all modes.
    """
    verifier = Phase26Verifier()
    p = "Bob holds a PhD in quantum physics."
    h = "Bob does not hold a PhD in quantum physics."

    for mode in ["baseline", "calibrated", "dense_fallback", "combined"]:
        r = verifier.verify_pair(p, h, mode=mode)
        assert r["decision"] == "CONTRADICTION", f"Mode {mode} failed to reject contradiction!"
        assert r["p_contra_max"] > 0.90


def test_unrelated_statements_rejected():
    """Unrelated statements must not be accepted as SAME."""
    verifier = Phase26Verifier()
    p = "Bob holds a PhD in quantum physics."
    h = "Bob enjoys mountain hiking in Colorado."

    for mode in ["baseline", "calibrated", "dense_fallback", "combined"]:
        r = verifier.verify_pair(p, h, mode=mode)
        assert r["decision"] != "SAME", f"Mode {mode} falsely merged unrelated statements!"


# ─────────────────────────────────────────────────────────────
# 4. End-to-End BlackHoleMemory Integration
# ─────────────────────────────────────────────────────────────

def test_black_hole_memory_with_phase26_verifier():
    """
    Verifies that BlackHoleMemory with Phase26Verifier successfully:
      1. Consolidates asymmetric paraphrases
      2. Rejects genuine contradictions
      3. Handles temporal updates with valid_until closure
      4. Preserves point-in-time recall
    """
    verifier = Phase26Verifier()
    mem = BlackHoleMemory(
        capacity=10,
        verification_mode="calibrated",
        verifier=verifier
    )

    # 1. Absorb initial fact
    r1 = mem.absorb(Candidate("Bob", "holds a PhD in", "quantum physics", timestamp=100.0))
    assert r1["action"] == "insert"
    assert r1["valid_from"] == 100.0

    # 2. Absorb asymmetric paraphrase: 'earned a doctorate'
    r2 = mem.absorb(Candidate("Bob", "earned a doctorate in", "quantum physics", timestamp=110.0))
    assert r2["action"] == "consolidate"
    assert r2["update_count"] == 2

    # 3. Absorb contradiction: 'does not hold a PhD'
    r3 = mem.absorb(Candidate("Bob", "does not hold a PhD in", "quantum physics", timestamp=200.0))
    assert r3["action"] != "consolidate"
    # Same subject contradiction triggers temporal update
    assert r3["action"] == "temporal_update"

    # Historical point-in-time recall at t=105 returns the original PhD fact
    past = mem.recall("Bob", as_of=105.0)
    assert len(past) == 1
    assert "quantum physics" in past[0]["text"]

    # History contains both entries in chronological order
    history = mem.get_history("Bob")
    assert len(history) == 2
    assert history[0]["valid_from"] == 100.0
    assert history[0]["valid_until"] == 200.0
    assert history[1]["valid_from"] == 200.0
    assert history[1]["valid_until"] is None
