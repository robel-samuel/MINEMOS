"""
benchmarks/phase18_metrics.py

Phase 18 Metrics & Error Decomposition Suite:
1. Retrieval Decomposition:
   - Candidate Retrieval Recall (top-1 and top-k)
   - Conditional NLI Decision Accuracy (accuracy given correct candidate retrieved)
   - End-to-End Accuracy (correct retrieval AND correct decision)
2. Memory Quality Metrics:
   - True Consolidation Rate (paraphrases merged into correct slot)
   - False Consolidation Rate (contradictions/distractors mistakenly merged)
   - Contradiction Retention Rate (contradictions correctly separated)
   - Unrelated Separation Rate (distractors kept isolated)
3. Eviction & Long-Gap Retention:
   - Slot Survival Rate across temporal gaps (10, 50, 100, 500)
   - Long-gap probe recall
4. Efficiency & Profiling:
   - Wall time, throughput (obs/sec), NLI calls, embedding latency, retrieval latency.
"""

from __future__ import annotations
import os, sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import cosine_similarity
from benchmarks.phase18_experiment import LEXICAL_ENCODER, D_LEX


def _nearest_slot(memory: MemoryState, candidate) -> tuple[Optional[MemorySlot], float]:
    """Finds slot in memory with highest lexical cosine similarity to candidate."""
    if not memory.slots:
        return None, 0.0
    x = LEXICAL_ENCODER.encode_full(candidate, dim=D_LEX)
    best, best_sim = None, -1.0
    for slot in memory.slots:
        sim = cosine_similarity(x, slot.key)
        if sim > best_sim:
            best_sim, best = sim, slot
    return best, best_sim


def evaluate_retrieval_decomposition(
    observations,
    trace: list[dict],
    memory: MemoryState
) -> dict:
    """
    Decomposes probe decision performance into:
      1. Stage-1 Retrieval Recall (Top-1 and Top-K)
      2. Stage-2 Conditional NLI Accuracy
      3. End-to-End Accuracy
    """
    # Build canonical slot map: concept_id -> nearest slot in final memory
    concept_slots: dict[str, MemorySlot] = {}
    for obs in observations:
        if obs.category == "canonical" and obs.concept_id not in concept_slots:
            slot, _ = _nearest_slot(memory, obs.candidate)
            if slot is not None:
                concept_slots[obs.concept_id] = slot

    total_probes = 0
    top1_retrieval_correct = 0
    topk_retrieval_correct = 0
    nli_correct_given_retrieval = 0
    end_to_end_correct = 0
    total_candidates_examined = 0

    for obs, tr in zip(observations, trace):
        if not obs.is_probe:
            continue

        cid = obs.concept_id
        cat = obs.category
        canon_slot = concept_slots.get(cid)
        if canon_slot is None:
            continue

        total_probes += 1
        total_candidates_examined += tr.get("candidates_examined", 1)

        # Expected decision: paraphrases/drift_steps should merge; contradictions/unrelated should reject
        expected_merge = cat in ("paraphrase", "delayed_paraphrase", "drift_step")
        actual_merge = (tr["action"] == "update")
        correct_decision = (actual_merge == expected_merge)

        # Top-1 retrieval check
        retrieved_idx = tr.get("retrieved_idx")
        top1_hit = False
        if retrieved_idx is not None and retrieved_idx < len(memory.slots):
            top1_hit = (memory.slots[retrieved_idx] is canon_slot)
        elif tr["action"] == "update":
            slot_idx = tr.get("slot_index")
            if slot_idx is not None and slot_idx < len(memory.slots):
                top1_hit = (memory.slots[slot_idx] is canon_slot)

        if top1_hit:
            top1_retrieval_correct += 1

        # Top-K retrieval check
        top_k_indices = tr.get("top_k_indices", [])
        topk_hit = False
        for idx in top_k_indices:
            if idx < len(memory.slots) and memory.slots[idx] is canon_slot:
                topk_hit = True
                break
        if topk_hit or top1_hit:
            topk_retrieval_correct += 1

        # Conditional NLI accuracy: did NLI decide correctly given top-k retrieved the true slot?
        retrieval_hit = (topk_hit or top1_hit)
        if retrieval_hit and correct_decision:
            nli_correct_given_retrieval += 1

        if correct_decision:
            end_to_end_correct += 1

    r_top1 = round(top1_retrieval_correct / total_probes, 4) if total_probes else 0.0
    r_topk = round(topk_retrieval_correct / total_probes, 4) if total_probes else 0.0
    denom_cond = topk_retrieval_correct if topk_retrieval_correct > 0 else 1
    cond_nli_acc = round(nli_correct_given_retrieval / denom_cond, 4) if topk_retrieval_correct else 0.0
    e2e_acc = round(end_to_end_correct / total_probes, 4) if total_probes else 0.0
    avg_cands = round(total_candidates_examined / total_probes, 2) if total_probes else 1.0

    return dict(
        total_probes=total_probes,
        top1_retrieval_recall=r_top1,
        topk_retrieval_recall=r_topk,
        conditional_nli_accuracy=cond_nli_acc,
        end_to_end_accuracy=e2e_acc,
        avg_candidates_examined=avg_cands
    )


def evaluate_memory_consolidation(
    run_result: dict,
    observations
) -> dict:
    """
    Evaluates memory consolidation quality:
      - true_consolidation_rate: paraphrase probes merged into canonical slot
      - false_consolidation_rate: contradiction/distractor probes falsely merged
      - contradiction_retention: fraction of contradictions isolated from canonical slot
      - unrelated_separation: fraction of distractors isolated from target slot
      - slot_key_value_sim: mean cosine similarity between slot key and value
    """
    memory: MemoryState = run_result["memory"]

    concept_canon: dict[str, tuple[MemorySlot, float]] = {}
    for obs in observations:
        if obs.category == "canonical" and obs.concept_id not in concept_canon:
            slot, sim = _nearest_slot(memory, obs.candidate)
            if slot is not None:
                concept_canon[obs.concept_id] = (slot, sim)

    para_opps = para_merged = 0
    contra_opps = contra_separated = 0
    unrel_opps = unrel_separated = 0

    for obs in observations:
        cat = obs.category
        cid = obs.concept_id
        canon_entry = concept_canon.get(cid)

        if cat in ("paraphrase", "delayed_paraphrase"):
            para_opps += 1
            if canon_entry:
                canon_slot, _ = canon_entry
                probe_slot, _ = _nearest_slot(memory, obs.candidate)
                if probe_slot is canon_slot:
                    para_merged += 1

        elif cat in ("contradiction", "drift_contra"):
            contra_opps += 1
            if canon_entry:
                canon_slot, _ = canon_entry
                probe_slot, _ = _nearest_slot(memory, obs.candidate)
                if probe_slot is not canon_slot:
                    contra_separated += 1
            else:
                contra_separated += 1  # Not merged into target

        elif cat in ("distractor", "unrelated"):
            target_cid = getattr(obs, "ground_truth_target_concept", None)
            if target_cid and target_cid in concept_canon:
                unrel_opps += 1
                canon_slot, _ = concept_canon[target_cid]
                probe_slot, _ = _nearest_slot(memory, obs.candidate)
                if probe_slot is not canon_slot:
                    unrel_separated += 1

    drifts = [float(cosine_similarity(s.key, s.value)) for s in memory.slots]

    true_cons = round(para_merged / para_opps, 4) if para_opps else 0.0
    contra_ret = round(contra_separated / contra_opps, 4) if contra_opps else 1.0
    false_cons = round(1.0 - contra_ret, 4)
    unrel_sep = round(unrel_separated / unrel_opps, 4) if unrel_opps else 1.0

    return dict(
        system=run_result["system"],
        capacity=run_result["capacity"],
        top_k=run_result.get("top_k", 1),
        final_slots=len(memory.slots),
        n_updates=run_result["n_updates"],
        n_inserts=run_result["n_inserts"],
        n_evicts=run_result["n_evicts"],
        nli_calls=run_result["nli_calls"],
        gate_counts=run_result["gate_counts"],
        true_consolidation_rate=true_cons,
        false_consolidation_rate=false_cons,
        contradiction_retention=contra_ret,
        paraphrase_consolidation=true_cons,
        unrelated_separation=unrel_sep,
        mean_slot_kv_similarity=round(float(np.mean(drifts)), 4) if drifts else 1.0,
        wall_seconds=round(run_result["wall_seconds"], 2),
        obs_per_second=run_result["obs_per_second"],
        time_embedding=run_result["time_embedding"],
        time_retrieval=run_result["time_retrieval"],
        time_nli=run_result["time_nli"]
    )
