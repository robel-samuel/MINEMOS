"""
benchmarks/phase20_metrics.py

Phase 20 Metrics & Diagnostic Analysis Suite:
1. Standard Retrieval & Memory Decomposition:
   - Top-1 and Top-K Candidate Retrieval Recall
   - Conditional NLI Decision Accuracy
   - End-to-End Decision Accuracy
   - True Consolidation Rate
   - False Consolidation Rate
   - Contradiction Retention Rate
   - Contradiction False Merge Rate
2. Phase 20 Anchor Telemetry:
   - Total Anchors Retained across all slots
   - Average Anchor Age
   - Average Applied Repulsion Penalty
   - Steps with Triggered Repulsion
3. Scenario 7 Semantic Drift Evaluation:
   - Temporal Update Acceptance / Routing
   - Adversarial Contradiction Isolation
   - Post-Drift Paraphrase Recall
"""

from __future__ import annotations
import os, sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from phase2.memory_state import MemoryState, MemorySlot
from benchmarks.phase18_metrics import (
    evaluate_retrieval_decomposition,
    evaluate_memory_consolidation,
    _nearest_slot
)
from benchmarks.phase19_metrics import evaluate_phase19_diagnostics


def evaluate_phase20_diagnostics(
    run_result: dict,
    observations,
    memory: MemoryState
) -> dict:
    """
    Computes Phase 20 specific diagnostics including anchor telemetry,
    probe outcomes, and semantic drift metrics.
    """
    trace = run_result["trace"]
    p19_diag = evaluate_phase19_diagnostics(observations, trace, memory)

    # Telemetry extracted directly from run_result
    total_anchors = run_result.get("total_anchors_retained", 0)
    avg_anchor_age = run_result.get("avg_anchor_age", 0.0)
    avg_penalty = run_result.get("avg_applied_penalty", 0.0)
    steps_with_penalty = run_result.get("steps_with_penalty", 0)

    # Semantic drift specific probe evaluations (for Scenario 7)
    temporal_probes_total = 0
    temporal_probes_merged = 0
    temporal_probes_inserted = 0

    for obs, tr in zip(observations, trace):
        if getattr(obs, "is_probe", False) and getattr(obs, "category", "") == "temporal_update":
            temporal_probes_total += 1
            if tr.get("action") == "update":
                temporal_probes_merged += 1
            elif tr.get("action") in ("insert", "evict_insert"):
                temporal_probes_inserted += 1

    temp_update_acceptance_rate = (
        round(temporal_probes_merged / temporal_probes_total, 4)
        if temporal_probes_total else 0.0
    )

    return dict(
        # Base diagnostics
        beam_distribution=p19_diag["beam_distribution"],
        avg_candidates_examined=p19_diag["avg_candidates_examined"],
        contradiction_probes_total=p19_diag["contradiction_probes_total"],
        contradiction_probes_merged=p19_diag["contradiction_probes_merged"],
        contradiction_probes_isolated=p19_diag["contradiction_probes_isolated"],
        contradiction_false_merge_rate=p19_diag["contradiction_false_merge_rate"],
        # Anchor telemetry
        total_anchors_retained=total_anchors,
        avg_anchor_age=avg_anchor_age,
        avg_applied_penalty=avg_penalty,
        steps_with_penalty=steps_with_penalty,
        # Semantic drift metrics
        temporal_probes_total=temporal_probes_total,
        temporal_probes_merged=temporal_probes_merged,
        temporal_probes_inserted=temporal_probes_inserted,
        temporal_update_acceptance_rate=temp_update_acceptance_rate
    )
