"""
benchmarks/phase21_metrics.py

Phase 21 Metrics & Out-of-Distribution Generalization Analysis:
1. Category-Stratified Diagnostics:
   - Hard Contradiction False Merge Rate (lexical overlap stress)
   - Hard Paraphrase Recall & Consolidation (vocabulary shift stress)
   - Distractor Overload Selectivity (dense cluster interference)
   - Longitudinal Temporal Probe Accuracy (pre-drift vs post-drift)
2. Global Performance Metrics:
   - Top-1 & Top-K Retrieval Recall
   - Conditional NLI Accuracy
   - End-to-End Decision Accuracy
   - True Consolidation Rate
   - Contradiction Retention Rate
3. Resource & Compute Efficiency:
   - NLI calls per observation
   - Anchors retained
   - Wall-clock throughput
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
from benchmarks.phase20_metrics import evaluate_phase20_diagnostics


def evaluate_phase21_diagnostics(
    run_result: dict,
    observations,
    memory: MemoryState
) -> dict:
    """
    Computes Phase 21 fine-grained generalization diagnostics stratified
    by probe category and adversarial stress condition.
    """
    trace = run_result["trace"]
    p20_diag = evaluate_phase20_diagnostics(run_result, observations, memory)

    # Category-specific tracking
    category_counts = {}
    category_correct = {}
    category_merges = {}

    for obs, tr in zip(observations, trace):
        if not getattr(obs, "is_probe", False):
            continue
        cat = obs.category
        category_counts[cat] = category_counts.get(cat, 0) + 1

        action = tr.get("action", "")
        is_merge = (action == "update")
        if is_merge:
            category_merges[cat] = category_merges.get(cat, 0) + 1

        # Correctness definition:
        # Paraphrases should merge; contradictions/distractors should NOT merge
        if cat in ("paraphrase", "delayed_paraphrase", "pre_drift_probe", "post_drift_probe"):
            correct = is_merge
        elif cat in ("contradiction", "unrelated", "distractor"):
            correct = not is_merge
        elif cat == "temporal_update":
            # For temporal update, either updating or creating distinct slot is valid depending on routing
            correct = True
        else:
            correct = True

        if correct:
            category_correct[cat] = category_correct.get(cat, 0) + 1

    category_accuracies = {
        cat: round(category_correct.get(cat, 0) / count, 4)
        for cat, count in category_counts.items()
    }

    category_merge_rates = {
        cat: round(category_merges.get(cat, 0) / count, 4)
        for cat, count in category_counts.items()
    }

    return dict(
        base_diagnostics=p20_diag,
        category_counts=category_counts,
        category_accuracies=category_accuracies,
        category_merge_rates=category_merge_rates,
        hard_contra_false_merge_rate=category_merge_rates.get("contradiction", 0.0),
        hard_para_consolidation_rate=category_merge_rates.get("paraphrase", 0.0),
        pre_drift_accuracy=category_accuracies.get("pre_drift_probe", 0.0),
        post_drift_accuracy=category_accuracies.get("post_drift_probe", 0.0)
    )
