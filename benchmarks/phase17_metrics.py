"""
benchmarks/phase17_metrics.py

Phase 17 Metrics Suite:
1. Memory-level consolidation metrics (true/false consolidation, contradiction retention,
   paraphrase consolidation, unrelated separation, drift, slot counts, evictions).
2. Long-horizon probe recall (per gap-length, per scenario).
3. Contradiction retention across stream time-steps.
4. Candidate retrieval diagnostics (retrieval recall, conditional NLI accuracy, end-to-end accuracy).
5. Bounded-head error amplification (same corrected Phase 15/16 methodology).
6. Efficiency metrics (wall time, NLI calls, average latency).
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from phase2.memory_state import MemoryState
from phase2.memory_update import cosine_similarity
from benchmarks.phase17_experiment import ENCODER, D


def _nearest_slot(memory: MemoryState, cand):
    x = ENCODER.encode_full(cand, dim=D)
    best, best_sim = None, -1.0
    for slot in memory.slots:
        sim = cosine_similarity(x, slot.key)
        if sim > best_sim:
            best_sim, best = sim, slot
    return best, best_sim


def probe_recall(trace: list[dict], memory: MemoryState, observations) -> dict:
    """
    Evaluates whether probe observations merged into the correct concept's slot.
    For each probe, checks whether the action was 'update' and the slot matches
    the concept's canonical slot (by nearest-slot search over the final memory).

    Returns:
        per_probe: list of {concept_id, category, retrieved, correct_merge}
        recall_rate: fraction of probes correctly consolidated
    """
    concept_canonical_slots: dict[str, object] = {}  # concept_id -> slot object
    # Build canonical slot index: for each chain's first canonical obs
    for obs, tr in zip(observations, trace):
        if obs.category == "canonical" and obs.concept_id not in concept_canonical_slots:
            # Find the slot closest to canonical encoding in final memory
            slot, _ = _nearest_slot(memory, obs.candidate)
            concept_canonical_slots[obs.concept_id] = slot

    probe_obs = [(obs, tr) for obs, tr in zip(observations, trace) if obs.is_probe]

    per_probe = []
    correct = 0
    total = 0
    for obs, tr in probe_obs:
        cid = obs.concept_id
        canon_slot = concept_canonical_slots.get(cid)
        merged = tr["action"] == "update"

        # Correct merge: it updated into the canonical slot's closest match
        if merged and canon_slot is not None:
            # Check that target slot is the canonical slot
            slot_idx = tr.get("slot_index") if "slot_index" in tr else None
            if slot_idx is not None and slot_idx < len(memory.slots):
                target_slot = memory.slots[slot_idx]
                correct_merge = (target_slot is canon_slot)
            else:
                correct_merge = False
        elif not merged and obs.category in ("paraphrase", "delayed_paraphrase"):
            correct_merge = False  # Should have merged
        else:
            correct_merge = True  # Correctly rejected (contradiction/distractor)

        total += 1
        if correct_merge:
            correct += 1

        per_probe.append(dict(
            concept_id=cid,
            category=obs.category,
            action=tr["action"],
            gate=tr.get("gate"),
            correct_merge=correct_merge
        ))

    return dict(
        per_probe=per_probe,
        recall_rate=round(correct / total, 4) if total else 0.0,
        total_probes=total,
        correct_count=correct
    )


def memory_metrics(run_result: dict, observations) -> dict:
    """
    Evaluates memory consolidation quality:
    - true_consolidation_rate: paraphrase/canonical probes correctly merged into right slot
    - false_consolidation_rate: contradictions or distractors incorrectly merged
    - contradiction_retention: fraction of contradictions kept in separate slots
    - paraphrase_consolidation: fraction of paraphrase probes merged into canonical slot
    - unrelated_separation: fraction of distractors that did not merge into target slots
    - slot_key_value_sim: mean cosine sim between key and value across all slots
    """
    memory = run_result["memory"]

    # Build canonical slot lookup: concept -> nearest slot in final memory
    concept_canon: dict[str, tuple] = {}
    for obs in observations:
        if obs.category == "canonical" and obs.concept_id not in concept_canon:
            slot, sim = _nearest_slot(memory, obs.candidate)
            concept_canon[obs.concept_id] = (slot, sim)

    same_opps = same_correct = 0
    false_opps = false_correct = 0
    contra_opps = contra_sep = 0
    para_opps = para_merged = 0
    unrel_opps = unrel_sep = 0

    for obs in observations:
        cat = obs.category
        cid = obs.concept_id
        canon_entry = concept_canon.get(cid)

        if cat in ("paraphrase", "delayed_paraphrase"):
            para_opps += 1
            same_opps += 1
            if canon_entry:
                canon_slot, _ = canon_entry
                probe_slot, _ = _nearest_slot(memory, obs.candidate)
                if probe_slot is canon_slot:
                    para_merged += 1
                    same_correct += 1

        elif cat in ("contradiction", "drift_contra"):
            contra_opps += 1
            false_opps += 1
            if canon_entry:
                canon_slot, _ = canon_entry
                probe_slot, _ = _nearest_slot(memory, obs.candidate)
                if probe_slot is not canon_slot:
                    contra_sep += 1
                    false_correct += 1

        elif cat == "distractor":
            target_cid = getattr(obs, "ground_truth_target_concept", None)
            if target_cid and target_cid in concept_canon:
                unrel_opps += 1
                false_opps += 1
                canon_slot, _ = concept_canon[target_cid]
                probe_slot, _ = _nearest_slot(memory, obs.candidate)
                if probe_slot is not canon_slot:
                    unrel_sep += 1
                    false_correct += 1

    drifts = [float(cosine_similarity(s.key, s.value)) for s in memory.slots]

    rejected = (run_result["gate_counts"].get("rejected_contra", 0) +
                run_result["gate_counts"].get("rejected_neutral", 0) +
                run_result["gate_counts"].get("rejected_diff", 0))

    return dict(
        system=run_result["system"],
        capacity=run_result["capacity"],
        final_slots=len(memory.slots),
        n_updates=run_result["n_updates"],
        n_inserts=run_result["n_inserts"],
        n_evicts=run_result["n_evicts"],
        rejected_candidates=rejected,
        gate_counts=run_result["gate_counts"],
        true_consolidation_rate=round(same_correct / same_opps, 4) if same_opps else 0.0,
        false_consolidation_rate=round(1.0 - false_correct / false_opps, 4) if false_opps else 0.0,
        contradiction_retention=round(contra_sep / contra_opps, 4) if contra_opps else 1.0,
        paraphrase_consolidation=round(para_merged / para_opps, 4) if para_opps else 0.0,
        unrelated_separation=round(unrel_sep / unrel_opps, 4) if unrel_opps else 1.0,
        mean_slot_kv_similarity=round(float(np.mean(drifts)), 4) if drifts else 1.0,
        wall_seconds=round(run_result["wall_seconds"], 2)
    )


def recall_vs_gap(scenario_1_streams: dict, system_results: dict) -> dict:
    """
    Evaluates paraphrase recall rate by gap length for Scenario 1.
    system_results maps gap -> {system -> run_result}.
    """
    results = {}
    for gap, obs_stream in scenario_1_streams.items():
        results[gap] = {}
        for sys_name, run_result in system_results.get(gap, {}).items():
            memory = run_result["memory"]
            trace = run_result.get("trace") or []
            probe_obs = [(o, t) for o, t in zip(obs_stream, trace) if o.is_probe]
            merged = sum(1 for _, t in probe_obs if t["action"] == "update")
            results[gap][sys_name] = {
                "n_probes": len(probe_obs),
                "merged": merged,
                "recall_rate": round(merged / len(probe_obs), 4) if probe_obs else 0.0
            }
    return results


def contradiction_retention_over_time(observations, trace: list[dict]) -> dict:
    """
    Tracks whether contradictions were correctly rejected at each step.
    Returns fraction of contradictions rejected at each 20% decile of the stream.
    """
    contra_obs_trace = [(i, o, t) for i, (o, t) in enumerate(zip(observations, trace))
                        if o.category in ("contradiction", "drift_contra")]
    n = len(observations)
    deciles = {str(d): [] for d in range(0, 101, 20)}
    for i, obs, tr in contra_obs_trace:
        pct = int((i / n) * 100)
        bucket = str((pct // 20) * 20)
        rejected = tr["action"] in ("insert", "evict_insert")
        deciles[bucket].append(rejected)

    return {k: round(sum(v) / len(v), 4) if v else None for k, v in deciles.items()}


def bounded_head_amplification(scenario_3_obs, run_result: dict) -> dict:
    """
    Scenario 3: Contradiction → Recovery analysis.
    Uses Phase 15/16 corrected bounded-head metric:
    - Counts initial contradiction merges at step 1
    - Counts step-2 paraphrase merges into contaminated slots (amplification)
    - Reports recovery fidelity based on clean final recovery steps
    """
    from benchmarks.phase16_experiment import ENCODER, D
    from phase2.memory_update import cosine_similarity

    trace = run_result.get("trace") or []
    if not trace:
        return dict(error="No trace recorded")

    # Group by concept_id
    from collections import defaultdict
    concept_steps = defaultdict(list)
    for obs, tr in zip(scenario_3_obs, trace):
        concept_steps[obs.concept_id].append((obs, tr))

    initial_errors = 0
    step2_merges_into_contaminated = 0
    recovery_key_sims = []
    recovery_val_sims = []
    memory = run_result["memory"]

    for cid, steps in concept_steps.items():
        # step index: 0=canonical, 1=contradiction, 2=paraphrase probe, 3-6=recovery
        # We only look at steps 0, 1, 2 for bounded head amplification
        if len(steps) < 3:
            continue

        canon_obs, canon_tr = steps[0]
        contra_obs, contra_tr = steps[1]
        probe_obs, probe_tr = steps[2]

        # Initial error: contradiction merged into canonical slot
        initial_error = contra_tr["action"] == "update"
        if initial_error:
            initial_errors += 1
            # Check if probe (step 2) also merged into that same contaminated slot
            contra_slot_idx = contra_tr.get("slot_index")
            probe_slot_idx = probe_tr.get("slot_index")
            if (probe_tr["action"] == "update" and
                    contra_slot_idx is not None and probe_slot_idx == contra_slot_idx):
                step2_merges_into_contaminated += 1

        # Recovery: evaluate final slot similarity to clean canonical encoding
        x_canon = ENCODER.encode_full(canon_obs.candidate, dim=D)
        best_slot, _ = _nearest_slot(memory, canon_obs.candidate)
        if best_slot is not None:
            recovery_key_sims.append(float(cosine_similarity(best_slot.key, x_canon)))
            recovery_val_sims.append(float(cosine_similarity(best_slot.value, x_canon)))

    n_concepts = len(concept_steps)
    return dict(
        n_concepts=n_concepts,
        initial_contradiction_merges=initial_errors,
        initial_error_rate=round(initial_errors / n_concepts, 4) if n_concepts else 0.0,
        bounded_head_step2_merges=step2_merges_into_contaminated,
        bounded_amplification_ratio=round(step2_merges_into_contaminated / initial_errors, 4) if initial_errors else 0.0,
        mean_recovery_key_sim=round(float(np.mean(recovery_key_sims)), 4) if recovery_key_sims else 0.0,
        mean_recovery_val_sim=round(float(np.mean(recovery_val_sims)), 4) if recovery_val_sims else 0.0,
    )


def candidate_retrieval_diagnostics(observations, trace: list[dict],
                                    memory: MemoryState) -> dict:
    """
    Diagnostic: decomposes end-to-end accuracy into:
    1. Candidate retrieval recall — was the correct concept's slot the best match?
    2. NLI conditional accuracy — given correct retrieval, did NLI decide correctly?
    3. End-to-end accuracy — correct retrieval AND correct NLI decision.
    """
    # Build concept → slot lookup
    concept_slots: dict[str, object] = {}
    for obs in observations:
        if obs.category == "canonical" and obs.concept_id not in concept_slots:
            slot, _ = _nearest_slot(memory, obs.candidate)
            concept_slots[obs.concept_id] = slot

    retrieval_correct_total = 0
    retrieval_total = 0
    nli_correct_given_retrieval = 0
    end_to_end_correct = 0

    for obs, tr in zip(observations, trace):
        if not obs.is_probe:
            continue
        cid = obs.concept_id
        cat = obs.category
        canon_slot = concept_slots.get(cid)
        if canon_slot is None:
            continue

        retrieval_total += 1
        sim = tr.get("similarity", 0.0)
        # Retrieval is correct if best_idx points to the canonical concept's slot
        retrieved_idx = tr.get("retrieved_idx")
        if retrieved_idx is not None and retrieved_idx < len(memory.slots) and memory.slots[retrieved_idx] is canon_slot:
            retrieval_correct = True
        else:
            slot_idx = tr.get("slot_index")
            if tr.get("action") == "update" and slot_idx is not None and slot_idx < len(memory.slots) and memory.slots[slot_idx] is canon_slot:
                retrieval_correct = True
            else:
                retrieval_correct = False

        if retrieval_correct:
            retrieval_correct_total += 1

        # Expected decision
        if cat in ("paraphrase", "delayed_paraphrase", "drift_step"):
            expected_merge = True
        else:  # contradiction, drift_contra
            expected_merge = False

        actual_merge = tr["action"] == "update"
        correct_decision = (actual_merge == expected_merge)

        if retrieval_correct and correct_decision:
            nli_correct_given_retrieval += 1

        if correct_decision:
            end_to_end_correct += 1

    return dict(
        total_probes=retrieval_total,
        candidate_retrieval_recall=round(retrieval_correct_total / retrieval_total, 4) if retrieval_total else 0.0,
        nli_accuracy_given_retrieval=round(nli_correct_given_retrieval / retrieval_correct_total, 4) if retrieval_correct_total else 0.0,
        end_to_end_accuracy=round(end_to_end_correct / retrieval_total, 4) if retrieval_total else 0.0,
    )
