"""
benchmarks/phase19_metrics.py

Phase 19 Metrics & Diagnostic Analysis Suite:
1. Retrieval Decomposition:
   - Stage-1 Top-1 and Top-K Recall
   - Stage-2 Conditional NLI Decision Accuracy
   - End-to-End Decision Accuracy
2. Memory Quality & Safety:
   - True Consolidation Rate (paraphrases correctly merged)
   - False Consolidation Rate (contradictions/distractors incorrectly merged)
   - Contradiction Retention Rate (contradictions correctly isolated)
   - Unrelated Separation Rate (distractors kept isolated)
3. Beam & Efficiency Diagnostics:
   - Beam policy usage (prune_weak, dominant, ambiguous)
   - NLI call reductions
   - Wall time and throughput
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


def evaluate_phase19_diagnostics(
    observations,
    trace: list[dict],
    memory: MemoryState
) -> dict:
    """
    Computes Phase 19 specific diagnostics:
      - Beam policy counts (prune_weak, dominant, ambiguous, fixed)
      - Contradiction-specific probe outcomes
      - Average candidates examined per observation
    """
    beam_counts: dict[str, int] = {}
    total_candidates_examined = 0
    contra_probes_total = 0
    contra_probes_merged = 0
    contra_probes_isolated = 0

    for obs, tr in zip(observations, trace):
        pol = tr.get("beam_policy", "fixed")
        beam_counts[pol] = beam_counts.get(pol, 0) + 1
        total_candidates_examined += tr.get("candidates_examined", 1)

        if getattr(obs, "is_probe", False) and getattr(obs, "category", "") in ("contradiction", "drift_contra"):
            contra_probes_total += 1
            if tr.get("action") == "update":
                contra_probes_merged += 1
            else:
                contra_probes_isolated += 1

    avg_cands = round(total_candidates_examined / len(trace), 2) if trace else 1.0
    contra_false_merge_rate = (
        round(contra_probes_merged / contra_probes_total, 4) if contra_probes_total else 0.0
    )

    return dict(
        beam_distribution=beam_counts,
        avg_candidates_examined=avg_cands,
        contradiction_probes_total=contra_probes_total,
        contradiction_probes_merged=contra_probes_merged,
        contradiction_probes_isolated=contra_probes_isolated,
        contradiction_false_merge_rate=contra_false_merge_rate
    )
