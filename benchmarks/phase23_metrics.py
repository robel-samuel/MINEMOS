"""
benchmarks/phase23_metrics.py

Phase 23 Comprehensive Metrics and Failure Attribution Analysis Suite.

Computes:
  1. Failure Attribution Distribution (A, B, C, D, E, F, SUCCESS, UNKNOWN)
  2. Paraphrase Consolidation Rate vs. Contradiction False Merge Rate (FMR)
  3. Contradiction Retention & Accuracy
  4. Candidate Retrieval Breakdown (Dense, BM25, Hybrid, Rescues)
  5. Eviction and Retention Rates (Slot Survival Rate)
  6. NLI Acceptance and Verification Rates
  7. Embedding Drift Statistics (cosine across sequential updates)
"""

from __future__ import annotations
import os, sys
from typing import Optional, Any
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import cosine_similarity
from benchmarks.phase23_diagnostics import Phase23DiagnosticTracker, FailureAttributionRecord


def evaluate_phase23_run(
    run_result: dict,
    observations,
    tracker: Phase23DiagnosticTracker
) -> dict:
    """
    Evaluates a Phase 23 execution run across all standard and attribution metrics.
    """
    trace = run_result["trace"]
    memory: MemoryState = run_result["memory"]

    # 1. Failure attribution breakdown
    attribution_summary = tracker.compute_summary_distribution()
    attributions_list = [r.to_dict() for r in tracker.attributions]

    # 2. Probe categorization
    probe_records = [r for r in tracker.attributions if r.details.get("category")]
    total_probes = len(probe_records)

    para_probes = [r for r in probe_records if r.details.get("category") in ("paraphrase", "delayed_paraphrase")]
    contra_probes = [r for r in probe_records if r.details.get("category") in ("contradiction", "drift_contra")]

    para_correct = sum(1 for r in para_probes if r.final_correct)
    contra_correct = sum(1 for r in contra_probes if r.final_correct)

    paraphrase_accuracy = round(para_correct / len(para_probes), 4) if para_probes else 0.0
    contradiction_accuracy = round(contra_correct / len(contra_probes), 4) if contra_probes else 1.0

    # Contradiction False Merge Rate: fraction of contradictions that falsely merged into target
    contra_false_merges = sum(1 for r in contra_probes if not r.final_correct)
    fmr = round(contra_false_merges / len(contra_probes), 4) if contra_probes else 0.0
    contra_retention = round(1.0 - fmr, 4)

    # End-to-End Accuracy across all evaluated probes
    all_correct = sum(1 for r in probe_records if r.final_correct)
    e2e_accuracy = round(all_correct / total_probes, 4) if total_probes else 0.0

    # Probe recall: fraction of probes that resulted in an action == 'update'
    probe_updates = sum(1 for r in probe_records if r.consolidation_attempted)
    probe_recall_rate = round(probe_updates / total_probes, 4) if total_probes else 0.0

    # 3. Retrieval Recall from probe attributions
    retrieved_targets = sum(1 for r in probe_records if r.correct_slot_retrieved)
    retrieval_recall = round(retrieved_targets / total_probes, 4) if total_probes else 0.0

    # 4. Target Slot Survival (Retention)
    target_records = list(tracker.targets.values())
    total_targets = len(target_records)
    survived_targets = sum(1 for t in target_records if not t.evicted)
    survival_rate = round(survived_targets / total_targets, 4) if total_targets else 1.0

    # 5. NLI Acceptance Rate on correctly retrieved targets
    targets_with_nli = [r for r in probe_records if r.correct_slot_retrieved and r.nli_called]
    nli_accepted_count = sum(1 for r in targets_with_nli if r.nli_accepted)
    nli_acceptance_rate = round(nli_accepted_count / len(targets_with_nli), 4) if targets_with_nli else 0.0

    # 6. Embedding drift summary across all tracked targets
    drift_values = []
    for t in target_records:
        if len(t.update_history) > 1:
            for uh in t.update_history[1:]:
                drift_values.append(uh.get("cosine_to_initial", 1.0))
    mean_drift = round(float(np.mean(drift_values)), 4) if drift_values else 1.0
    min_drift = round(float(np.min(drift_values)), 4) if drift_values else 1.0

    # 7. Diagnostics from run result
    res_diag = run_result.get("phase22_diagnostics") or {}

    return {
        "e2e_accuracy": e2e_accuracy,
        "false_merge_rate": fmr,
        "contradiction_accuracy": contradiction_accuracy,
        "contradiction_retention": contra_retention,
        "paraphrase_accuracy": paraphrase_accuracy,
        "delayed_paraphrase_accuracy": paraphrase_accuracy,
        "probe_recall": probe_recall_rate,
        "retrieval_recall": retrieval_recall,
        "dense_retrieval_recall": res_diag.get("dense_retrieval_recall", retrieval_recall),
        "bm25_retrieval_recall": res_diag.get("bm25_retrieval_recall", 0.0),
        "hybrid_retrieval_recall": res_diag.get("hybrid_retrieval_recall", retrieval_recall),
        "bm25_rescues": res_diag.get("bm25_rescue_count", 0),
        "nli_calls": run_result.get("nli_calls", 0),
        "wall_seconds": round(run_result.get("wall_seconds", 0.0), 2),
        "memory_occupancy": len(memory.slots),
        "eviction_count": run_result.get("n_evicts", 0),
        "slot_survival_rate": survival_rate,
        "consolidation_success_rate": round(all_correct / total_probes, 4) if total_probes else 0.0,
        "nli_acceptance_rate": nli_acceptance_rate,
        "mean_embedding_drift": mean_drift,
        "min_embedding_drift": min_drift,
        "failure_category_distribution": attribution_summary["counts"],
        "failure_category_percentages": attribution_summary["percentages"],
        "total_probes_evaluated": total_probes,
        "attributions": attributions_list
    }
