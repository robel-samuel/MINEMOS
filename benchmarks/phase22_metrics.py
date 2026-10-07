"""
benchmarks/phase22_metrics.py

Phase 22 Metrics & Hybrid Retrieval Diagnostics:
1. Retrieval Recall Stratification:
   - Dense Top-k Recall: Did dense retrieval surface the canonical slot?
   - BM25 Top-k Recall: Did BM25 retrieval surface the canonical slot?
   - Hybrid Top-k Recall: Did fused candidate beam surface the canonical slot?
   - Candidate Recovery Count: Number of times BM25 rescued a candidate missed by dense.
2. Safety & Consolidation Metrics:
   - Contradiction False Merge Rate (FMR)
   - Paraphrase Consolidation Rate
   - End-to-End Decision Accuracy
3. Computational Overhead:
   - Candidate beam size before NLI
   - Time in dense embedding & retrieval vs. BM25 retrieval vs. NLI
   - NLI invocations per observation
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


def evaluate_phase22_diagnostics(
    run_result: dict,
    observations,
    memory: MemoryState
) -> dict:
    """
    Computes Phase 22 hybrid retrieval diagnostics, comparing dense vs. lexical
    vs. fused candidate generation and probe accuracy.
    """
    trace = run_result["trace"]
    p20_diag = evaluate_phase20_diagnostics(run_result, observations, memory)

    # Concept location map: maps concept_id to list of slot indices created for that concept
    concept_to_slots: dict[str, set[int]] = {}
    for obs, tr in zip(observations, trace):
        act = tr.get("action")
        s_idx = tr.get("slot_index")
        cid = getattr(obs, "concept_id", None)
        if cid and s_idx is not None and act in ("insert", "update", "evict_insert"):
            if cid not in concept_to_slots:
                concept_to_slots[cid] = set()
            concept_to_slots[cid].add(s_idx)

    # Retrieval comparison metrics across probes
    probes_total = 0
    dense_hits = 0
    bm25_hits = 0
    fused_hits = 0
    bm25_rescues = 0  # True slot missed by dense, but recovered by BM25
    dense_rescues = 0  # True slot missed by BM25, but present in dense

    category_counts: dict[str, int] = {}
    category_correct: dict[str, int] = {}
    category_merges: dict[str, int] = {}

    fusion_sizes = []
    dense_sizes = []
    lexical_sizes = []

    for obs, tr in zip(observations, trace):
        if not getattr(obs, "is_probe", False):
            continue

        probes_total += 1
        cat = obs.category
        cid = obs.concept_id
        valid_slots = concept_to_slots.get(cid, set())

        category_counts[cat] = category_counts.get(cat, 0) + 1
        action = tr.get("action", "")
        is_merge = (action == "update")
        if is_merge:
            category_merges[cat] = category_merges.get(cat, 0) + 1

        # Correctness definition
        if cat in ("paraphrase", "delayed_paraphrase", "pre_drift_probe", "post_drift_probe"):
            correct = is_merge
        elif cat in ("contradiction", "unrelated", "distractor"):
            correct = not is_merge
        else:
            correct = True

        if correct:
            category_correct[cat] = category_correct.get(cat, 0) + 1

        # Candidate source tracking
        dense_cands = [idx for idx, _ in tr.get("dense_candidates", [])]
        bm25_cands = [idx for idx, _ in tr.get("lexical_candidates", [])]
        fused_cands = tr.get("top_k_indices", [])

        dense_hit = any(idx in valid_slots for idx in dense_cands)
        bm25_hit = any(idx in valid_slots for idx in bm25_cands)
        fused_hit = any(idx in valid_slots for idx in fused_cands)

        if dense_hit:
            dense_hits += 1
        if bm25_hit:
            bm25_hits += 1
        if fused_hit:
            fused_hits += 1

        if bm25_hit and not dense_hit:
            bm25_rescues += 1
        if dense_hit and not bm25_hit:
            dense_rescues += 1

        dense_sizes.append(len(dense_cands))
        lexical_sizes.append(len(bm25_cands))
        fusion_sizes.append(len(fused_cands))

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
        probes_total=probes_total,
        dense_retrieval_recall=round(dense_hits / probes_total, 4) if probes_total > 0 else 0.0,
        bm25_retrieval_recall=round(bm25_hits / probes_total, 4) if probes_total > 0 else 0.0,
        hybrid_retrieval_recall=round(fused_hits / probes_total, 4) if probes_total > 0 else 0.0,
        bm25_rescue_count=bm25_rescues,
        dense_rescue_count=dense_rescues,
        avg_dense_candidates=round(float(np.mean(dense_sizes)), 2) if dense_sizes else 0.0,
        avg_lexical_candidates=round(float(np.mean(lexical_sizes)), 2) if lexical_sizes else 0.0,
        avg_fused_candidates=round(float(np.mean(fusion_sizes)), 2) if fusion_sizes else 0.0,
        category_counts=category_counts,
        category_accuracies=category_accuracies,
        category_merge_rates=category_merge_rates,
        contradiction_false_merge_rate=category_merge_rates.get("contradiction", 0.0),
        paraphrase_consolidation_rate=category_merge_rates.get("paraphrase", 0.0)
    )
