"""
benchmarks/phase24_metrics.py

Phase 24 Comprehensive Metrics Evaluation Suite.
"""

from __future__ import annotations
import numpy as np

from phase2.memory_state import MemoryState
from benchmarks.phase24_diagnostics import Phase24DiagnosticTracker


def evaluate_phase24_run(
    run_result: dict,
    observations,
    tracker: Phase24DiagnosticTracker
) -> dict:
    """
    Evaluates a Phase 24 run across all required scientific metrics.
    """
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

    # False Merge Rate (FMR): fraction of contradictions that erroneously merged into target slot
    contra_false_merges = sum(1 for r in contra_probes if not r.final_correct)
    fmr = round(contra_false_merges / len(contra_probes), 4) if contra_probes else 0.0
    contra_retention = round(1.0 - fmr, 4)

    # False Rejection Rate (FRR): fraction of retrieved valid paraphrases rejected by NLI
    retrieved_para_probes = [r for r in para_probes if r.correct_slot_retrieved]
    rejected_para = sum(1 for r in retrieved_para_probes if not r.nli_accepted)
    frr = round(rejected_para / len(retrieved_para_probes), 4) if retrieved_para_probes else 0.0

    # End-to-End Accuracy across all evaluated probes
    all_correct = sum(1 for r in probe_records if r.final_correct)
    e2e_accuracy = round(all_correct / total_probes, 4) if total_probes else 0.0

    # Retrieval Recall
    retrieved_targets = sum(1 for r in probe_records if r.correct_slot_retrieved)
    retrieval_recall = round(retrieved_targets / total_probes, 4) if total_probes else 0.0

    # Target Slot Survival Rate
    target_records = list(tracker.targets.values())
    total_targets = len(target_records)
    survived_targets = sum(1 for t in target_records if not t.evicted)
    survival_rate = round(survived_targets / total_targets, 4) if total_targets else 1.0

    # Representation Drift
    drift_initial_values = []
    drift_latest_values = []
    for t in target_records:
        if len(t.update_history) > 1:
            for uh in t.update_history[1:]:
                drift_initial_values.append(uh.get("cosine_to_initial", 1.0))
                drift_latest_values.append(uh.get("cosine_to_latest", 1.0))

    mean_drift_initial = round(float(np.mean(drift_initial_values)), 4) if drift_initial_values else 1.0
    min_drift_initial = round(float(np.min(drift_initial_values)), 4) if drift_initial_values else 1.0
    mean_drift_latest = round(float(np.mean(drift_latest_values)), 4) if drift_latest_values else 1.0

    return {
        "e2e_accuracy": e2e_accuracy,
        "paraphrase_accuracy": paraphrase_accuracy,
        "contradiction_accuracy": contradiction_accuracy,
        "false_merge_rate": fmr,
        "false_rejection_rate": frr,
        "contradiction_retention": contra_retention,
        "retrieval_recall": retrieval_recall,
        "slot_survival_rate": survival_rate,
        "eviction_count": run_result.get("n_evicts", 0),
        "nli_calls": run_result.get("nli_calls", 0),
        "wall_seconds": round(run_result.get("wall_seconds", 0.0), 2),
        "memory_occupancy": len(memory.slots),
        "mean_drift_to_initial": mean_drift_initial,
        "min_drift_to_initial": min_drift_initial,
        "mean_drift_to_latest": mean_drift_latest,
        "total_probes_evaluated": total_probes,
        "failure_category_distribution": attribution_summary["counts"],
        "failure_category_percentages": attribution_summary["percentages"],
        "attributions": attributions_list
    }
