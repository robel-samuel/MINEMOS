"""
benchmarks/phase26_metrics.py

Mathematical metrics for Phase 26 semantic verification and contradiction safety.

Distinguishes:
  1. Pairwise verification metrics (pure classification on NLI/guard decision)
  2. End-to-end memory metrics (consolidation, temporal supersession, recall, retention)
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np


def compute_verification_metrics(
    pairs: List[Any],
    predictions: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Computes rigorous pairwise verification metrics.
    
    Categories:
      - valid_paraphrase (target: SAME)
      - genuine_contradiction (target: CONTRADICTION)
      - unrelated (target: NEUTRAL / not SAME)
      - temporal_change (target: CONTRADICTION or TEMPORAL / not SAME)
    """
    assert len(pairs) == len(predictions), f"Mismatched counts: {len(pairs)} vs {len(predictions)}"

    n_para = 0
    para_accepted = 0
    para_false_contra = 0
    para_neutral = 0

    n_contra = 0
    contra_detected = 0
    contra_false_merge = 0

    n_unrel = 0
    unrel_rejected = 0
    unrel_false_merge = 0

    n_temp = 0
    temp_routed = 0
    temp_false_merge = 0

    total_nli_calls = 0
    total_time_ms = 0.0

    for pair, pred in zip(pairs, predictions):
        dec = pred.get("decision", "NEUTRAL")
        cat = pair.category
        total_nli_calls += pred.get("nli_invocations", 1)
        total_time_ms += pred.get("latency_ms", 0.0)

        if cat == "valid_paraphrase":
            n_para += 1
            if dec == "SAME":
                para_accepted += 1
            elif dec == "CONTRADICTION":
                para_false_contra += 1
            else:
                para_neutral += 1

        elif cat == "genuine_contradiction":
            n_contra += 1
            if dec == "CONTRADICTION":
                contra_detected += 1
            elif dec == "SAME":
                contra_false_merge += 1

        elif cat == "unrelated":
            n_unrel += 1
            if dec != "SAME":
                unrel_rejected += 1
            else:
                unrel_false_merge += 1

        elif cat == "temporal_change":
            n_temp += 1
            # In MINEMOS memory engine, temporal revisions are detected via CONTRADICTION on same entity
            if dec in ("CONTRADICTION", "TEMPORAL"):
                temp_routed += 1
            elif dec == "SAME":
                temp_false_merge += 1

    # Derived rates
    paraphrase_acceptance_rate = para_accepted / n_para if n_para > 0 else 0.0
    false_contradiction_rate = para_false_contra / n_para if n_para > 0 else 0.0
    contradiction_detection_rate = contra_detected / n_contra if n_contra > 0 else 0.0
    contradiction_false_merge_rate = contra_false_merge / n_contra if n_contra > 0 else 0.0
    unrelated_rejection_rate = unrel_rejected / n_unrel if n_unrel > 0 else 0.0
    unrelated_false_merge_rate = unrel_false_merge / n_unrel if n_unrel > 0 else 0.0
    temporal_routing_rate = temp_routed / n_temp if n_temp > 0 else 0.0
    temporal_false_merge_rate = temp_false_merge / n_temp if n_temp > 0 else 0.0

    total_false_merges = contra_false_merge + unrel_false_merge + temp_false_merge
    total_non_paraphrases = n_contra + n_unrel + n_temp
    overall_false_merge_rate = total_false_merges / total_non_paraphrases if total_non_paraphrases > 0 else 0.0

    # Macro accuracy across four categories:
    #   Correct para: SAME
    #   Correct contra: CONTRADICTION
    #   Correct unrel: not SAME
    #   Correct temp: not SAME (preferably CONTRADICTION/TEMPORAL)
    correct_total = (
        para_accepted +
        contra_detected +
        unrel_rejected +
        temp_routed
    )
    total_pairs = len(pairs)
    macro_accuracy = correct_total / total_pairs if total_pairs > 0 else 0.0

    return {
        "paraphrase_acceptance_rate": round(paraphrase_acceptance_rate, 4),
        "false_contradiction_rate": round(false_contradiction_rate, 4),
        "contradiction_detection_rate": round(contradiction_detection_rate, 4),
        "contradiction_false_merge_rate": round(contradiction_false_merge_rate, 4),
        "unrelated_rejection_rate": round(unrelated_rejection_rate, 4),
        "unrelated_false_merge_rate": round(unrelated_false_merge_rate, 4),
        "temporal_routing_rate": round(temporal_routing_rate, 4),
        "temporal_false_merge_rate": round(temporal_false_merge_rate, 4),
        "overall_false_merge_rate": round(overall_false_merge_rate, 4),
        "macro_accuracy": round(macro_accuracy, 4),
        "total_pairs": total_pairs,
        "total_nli_calls": total_nli_calls,
        "avg_nli_calls_per_pair": round(total_nli_calls / total_pairs, 2) if total_pairs > 0 else 0.0,
        "avg_latency_ms": round(total_time_ms / total_pairs, 2) if total_pairs > 0 else 0.0
    }
