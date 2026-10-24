"""
benchmarks/phase24_consolidation.py

Phase 24 Drift-Resistant Vector Consolidation:
Prevents representation degradation across successive memory updates while
preserving capacity for factual updates.

Update Equations Evaluated:
  1. baseline: Frozen update equation v_t = (1 - alpha_t) * v_{t-1} + alpha_t * x_t
  2. bounded: Limits the Euclidean displacement ||v_t - v_{t-1}|| <= delta_max
  3. anchor: Regularizes v_t toward slot.key (the original root representation v_0)
  4. hybrid: Combines bounded single-step displacement with anchor regularization
"""

from __future__ import annotations
import numpy as np

from phase2.memory_state import MemorySlot


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        return vec / norm
    return vec


def consolidate_vector_phase24(
    slot: MemorySlot,
    x_t: np.ndarray,
    alpha_t: float,
    mode: str = "baseline",
    delta_max: float = 0.25,
    lambda_anchor: float = 0.20
) -> np.ndarray:
    """
    Computes updated slot.value representation according to consolidation mode.
    
    Args:
      slot: The existing MemorySlot to update (contains slot.value and slot.key)
      x_t: Candidate observation vector
      alpha_t: Update learning rate / novelty weight
      mode: 'baseline', 'bounded', 'anchor', 'hybrid'
      delta_max: Maximum allowed displacement per step for bounded/hybrid modes
      lambda_anchor: Pull-back weight toward original key v_0 for anchor/hybrid modes
      
    Returns:
      Updated, normalized value vector v_t
    """
    v_prev = slot.value
    v_0 = slot.key

    if mode == "baseline":
        # Frozen Phase 23 update equation
        v_next = (1.0 - alpha_t) * v_prev + alpha_t * x_t
        return l2_normalize(v_next)

    elif mode == "bounded":
        # Displaces by at most delta_max
        delta_raw = alpha_t * (x_t - v_prev)
        norm_delta = float(np.linalg.norm(delta_raw))
        if norm_delta > delta_max and norm_delta > 1e-8:
            delta_bounded = delta_raw * (delta_max / norm_delta)
        else:
            delta_bounded = delta_raw
        v_next = v_prev + delta_bounded
        return l2_normalize(v_next)

    elif mode == "anchor":
        # Blends candidate update with original root key v_0
        v_raw = (1.0 - alpha_t) * v_prev + alpha_t * x_t
        v_next = (1.0 - lambda_anchor) * v_raw + lambda_anchor * v_0
        return l2_normalize(v_next)

    elif mode == "hybrid":
        # First bound single-step displacement, then apply anchor regularization
        delta_raw = alpha_t * (x_t - v_prev)
        norm_delta = float(np.linalg.norm(delta_raw))
        if norm_delta > delta_max and norm_delta > 1e-8:
            delta_bounded = delta_raw * (delta_max / norm_delta)
        else:
            delta_bounded = delta_raw
        v_step = v_prev + delta_bounded
        v_next = (1.0 - lambda_anchor) * v_step + lambda_anchor * v_0
        return l2_normalize(v_next)

    else:
        raise ValueError(f"Unknown consolidation mode: '{mode}'")
