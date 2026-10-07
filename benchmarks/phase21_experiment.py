"""
benchmarks/phase21_experiment.py

Phase 21 Experiment Architecture:
Generalization, Adversarial Validation, and Out-of-Distribution Evaluation.

Strictly preserves the frozen Phase 20 configuration:
  - Anchor weight: w_0 = 0.15
  - Temporal decay constant: lambda = 0.005
  - Bounded anchor capacity: k = 3 (deterministic FIFO eviction)
  - Beam thresholds: tau_min = 0.40, tau_high = 0.75, delta_margin = 0.10, k_max = 3
  - Anchor similarity cutoff: tau_anchor = 0.70
  - CrossEncoder NLI: cross-encoder/nli-distilroberta-base (local_files_only=True)
  - Bi-Encoder: sentence-transformers/all-MiniLM-L6-v2 (local_files_only=True)
  - Core update equation: v_t = (1 - alpha_t) * v_{t-1} + alpha_t * x_t (FROZEN)
  - MATCH_THRESHOLD: 0.75 (FROZEN)

Evaluates:
  1. System D:  Phase 18 Baseline (Dense + Utility Eviction, fixed k=3, w=0)
  2. System D4: Phase 19 Baseline (Dense + Contra-Utility + Adaptive Beam + Static Unbounded Anchors w=0.20)
  3. System D8: Frozen Phase 20 Configuration (w=0.15, lambda=0.005, k=3)
"""

from __future__ import annotations
import os, sys, time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import MATCH_THRESHOLD, cosine_similarity
from phase2.encoders import FieldAwareLexicalEncoder
from benchmarks.phase16_experiment import NliSemanticGate
from benchmarks.phase18_experiment import (
    DenseSemanticEncoder,
    D_LEX,
    D_DENSE,
    LEXICAL_ENCODER,
    DENSE_MODEL_NAME
)
from benchmarks.phase20_experiment import (
    adaptive_dense_address_phase20,
    evict_slot_phase20,
    absorb_observation_phase20,
    run_phase20_stream
)

# ──────────────────────────────────────────────
#  Frozen Phase 20 Configuration Constants
# ──────────────────────────────────────────────
FROZEN_W = 0.15
FROZEN_LAMBDA = 0.005
FROZEN_K = 3
TAU_MIN = 0.40
TAU_HIGH = 0.75
DELTA_MARGIN = 0.10
K_MAX = 3
TAU_ANCHOR = 0.70


def run_phase21_stream(
    observations,
    capacity: int,
    system: str,  # 'D', 'D4', 'D8' (frozen) or experimental ablation
    nli_gate: NliSemanticGate,
    dense_encoder: DenseSemanticEncoder,
    top_k: int = 3,
    # Overrides strictly defaulted to frozen Phase 20 parameters
    w_anchor: Optional[float] = None,
    decay_lambda: Optional[float] = None,
    max_anchors: Optional[int] = None
) -> dict:
    """
    Executes a stream under the frozen Phase 20 system configuration (or baseline).
    """
    if system == "D":
        w = 0.0
        lmb = 0.0
        k = None
    elif system == "D4":
        w = 0.20
        lmb = 0.0
        k = None
    elif system == "D8":
        w = FROZEN_W if w_anchor is None else w_anchor
        lmb = FROZEN_LAMBDA if decay_lambda is None else decay_lambda
        k = FROZEN_K if max_anchors is None else max_anchors
    else:
        # Experimental ablation
        w = FROZEN_W if w_anchor is None else w_anchor
        lmb = FROZEN_LAMBDA if decay_lambda is None else decay_lambda
        k = FROZEN_K if max_anchors is None else max_anchors

    return run_phase20_stream(
        observations=observations,
        capacity=capacity,
        system=system,
        nli_gate=nli_gate,
        dense_encoder=dense_encoder,
        top_k=top_k,
        tau_min=TAU_MIN,
        tau_high=TAU_HIGH,
        delta_margin=DELTA_MARGIN,
        w_anchor=w,
        decay_lambda=lmb,
        max_anchors=k,
        anchor_sim_threshold=TAU_ANCHOR
    )
