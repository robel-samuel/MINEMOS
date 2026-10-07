"""
run_phase23.py

Phase 23 Comprehensive Benchmark Runner:
Consolidation and Retention Bottleneck Investigation.

Runs all 6 diagnostic experiments + sealed held-out benchmark:
  1. Exp 1: NLI Threshold Sweep (0.55 - 0.85) on D8 and H1
  2. Exp 2: Oracle Retrieval (Normal D8, Normal H1, D8-O, H1-O)
  3. Exp 3: Retention Oracle (Normal D8, Normal H1, D8-R, H1-R)
  4. Exp 4: Consolidation vs Retention Matrix (D8, D8-R, D8-O, D8-OR)
  5. Exp 5: Embedding Drift Analysis
  6. Exp 6: Consolidation Traces (Step-by-step autopsy)
  7. Exp 7: Sealed Held-Out Benchmark (seed=2323)

Saves incremental checkpoints to scratch/phase23_results.json.
"""

from __future__ import annotations
import os, sys, json, time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
torch.set_num_threads(4)

from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import cosine_similarity
from benchmarks.phase16_experiment import NliSemanticGate
from benchmarks.phase18_experiment import DenseSemanticEncoder
from benchmarks.phase22_experiment import (
    FROZEN_W,
    FROZEN_LAMBDA,
    FROZEN_K,
    TAU_MIN,
    TAU_HIGH,
    DELTA_MARGIN,
    K_MAX,
    TAU_ANCHOR,
)
from benchmarks.phase23_diagnostics import Phase23DiagnosticTracker
from benchmarks.phase23_dataset import (
    generate_exp1_threshold_sweep,
    generate_exp2_oracle_retrieval,
    generate_exp3_retention_oracle,
    generate_exp4_matrix_stream,
    generate_exp5_embedding_drift_stream,
    generate_exp6_trace_stream,
    generate_held_out_phase23,
    SEED_P23,
)
from benchmarks.phase23_experiment import run_phase23_stream
from benchmarks.phase23_metrics import evaluate_phase23_run

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "scratch", "phase23_results.json")

FROZEN_CONFIG_P23 = {
    "w_anchor": FROZEN_W,
    "decay_lambda": FROZEN_LAMBDA,
    "max_anchors": FROZEN_K,
    "tau_min": TAU_MIN,
    "tau_high": TAU_HIGH,
    "delta_margin": DELTA_MARGIN,
    "k_max": K_MAX,
    "tau_anchor": TAU_ANCHOR,
    "k1_bm25": 1.5,
    "b_bm25": 0.75,
    "top_k_dense": 3,
    "top_k_bm25": 3,
    "k_fusion": 5,
    "historical_match_threshold": 0.75,
    "seed_p23": SEED_P23,
}


def serialize(obj):
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 4)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return [serialize(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize(x) for x in obj]
    return obj


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_checkpoint(data: dict):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    temp_path = CHECKPOINT_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(serialize(data), f, indent=2)
    os.replace(temp_path, CHECKPOINT_PATH)
    print(f"[Phase 23 Checkpoint] Saved to {CHECKPOINT_PATH}", file=sys.stderr)


def main():
    print("[Phase 23] Initializing models on 4 CPU threads...", file=sys.stderr)
    t_start = time.perf_counter()
    nli_gate = NliSemanticGate()
    dense_encoder = DenseSemanticEncoder()
    print(f"[Phase 23] Models loaded in {time.perf_counter() - t_start:.2f}s", file=sys.stderr)

    results = load_checkpoint()
    if "frozen_config" not in results:
        results["frozen_config"] = FROZEN_CONFIG_P23
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment 1: NLI Threshold Sweep (0.55 - 0.85)
    # ──────────────────────────────────────────────
    if "exp1_threshold_sweep" not in results:
        results["exp1_threshold_sweep"] = {}
    exp1_res = results["exp1_threshold_sweep"]

    stream_exp1 = generate_exp1_threshold_sweep(inter_gap=5, seed=SEED_P23)
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    print(f"[Phase 23] Exp 1 (Threshold Sweep): {len(stream_exp1)} obs, C=50, thresholds={thresholds}", file=sys.stderr)

    for th in thresholds:
        th_key = f"{th:.2f}"
        if th_key not in exp1_res:
            exp1_res[th_key] = {}

        for sys_id in ["D8", "H1"]:
            if sys_id in exp1_res[th_key]:
                print(f"[Phase 23] Exp 1 th={th_key} {sys_id} already checkpointed, skipping.", file=sys.stderr)
                continue
            print(f"[Phase 23] Exp 1: th={th_key}, System={sys_id}...", file=sys.stderr)
            run_res, tracker = run_phase23_stream(
                observations=stream_exp1, capacity=50, system=sys_id,
                nli_gate=nli_gate, dense_encoder=dense_encoder,
                entail_threshold=th
            )
            entry = evaluate_phase23_run(run_res, stream_exp1, tracker)
            exp1_res[th_key][sys_id] = entry
            save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment 2: Oracle Retrieval (Forcible Target Injection)
    # ──────────────────────────────────────────────
    if "exp2_oracle_retrieval" not in results:
        results["exp2_oracle_retrieval"] = {}
    exp2_res = results["exp2_oracle_retrieval"]

    stream_exp2 = generate_exp2_oracle_retrieval(n_targets=10, inter_gap=8, seed=SEED_P23)
    print(f"[Phase 23] Exp 2 (Oracle Retrieval): {len(stream_exp2)} obs, C=50", file=sys.stderr)

    exp2_configs = [
        ("D8_normal", "D8", False),
        ("H1_normal", "H1", False),
        ("D8_oracle", "D8", True),
        ("H1_oracle", "H1", True),
    ]

    for label, sys_id, oracle_ret in exp2_configs:
        if label in exp2_res:
            print(f"[Phase 23] Exp 2 {label} already checkpointed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 23] Exp 2: {label} (sys={sys_id}, oracle_ret={oracle_ret})...", file=sys.stderr)
        run_res, tracker = run_phase23_stream(
            observations=stream_exp2, capacity=50, system=sys_id,
            nli_gate=nli_gate, dense_encoder=dense_encoder,
            entail_threshold=0.75,
            oracle_retrieval=oracle_ret
        )
        entry = evaluate_phase23_run(run_res, stream_exp2, tracker)
        exp2_res[label] = entry
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment 3: Retention Oracle (Protected Slots)
    # ──────────────────────────────────────────────
    if "exp3_retention_oracle" not in results:
        results["exp3_retention_oracle"] = {}
    exp3_res = results["exp3_retention_oracle"]

    # High eviction pressure: 300 obs into C=25 forces heavy eviction
    stream_exp3 = generate_exp3_retention_oracle(n_targets=10, total_obs=300, seed=SEED_P23)
    print(f"[Phase 23] Exp 3 (Retention Oracle): {len(stream_exp3)} obs, C=25", file=sys.stderr)

    exp3_configs = [
        ("D8_normal", "D8", False),
        ("H1_normal", "H1", False),
        ("D8_protected", "D8", True),
        ("H1_protected", "H1", True),
    ]

    for label, sys_id, prot_ret in exp3_configs:
        if label in exp3_res:
            print(f"[Phase 23] Exp 3 {label} already checkpointed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 23] Exp 3: {label} (sys={sys_id}, protected_ret={prot_ret})...", file=sys.stderr)
        run_res, tracker = run_phase23_stream(
            observations=stream_exp3, capacity=25, system=sys_id,
            nli_gate=nli_gate, dense_encoder=dense_encoder,
            entail_threshold=0.75,
            protected_retention=prot_ret
        )
        entry = evaluate_phase23_run(run_res, stream_exp3, tracker)
        exp3_res[label] = entry
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment 4: Consolidation vs Retention Matrix (2x2 on D8)
    # ──────────────────────────────────────────────
    if "exp4_matrix" not in results:
        results["exp4_matrix"] = {}
    exp4_res = results["exp4_matrix"]

    stream_exp4 = generate_exp4_matrix_stream(seed=SEED_P23)
    print(f"[Phase 23] Exp 4 (Matrix 2x2): {len(stream_exp4)} obs, C=25", file=sys.stderr)

    matrix_configs = [
        ("D8", False, False),       # Normal retrieval, Normal retention
        ("D8_R", False, True),      # Normal retrieval, Protected retention
        ("D8_O", True, False),      # Oracle retrieval, Normal retention
        ("D8_OR", True, True),      # Oracle retrieval, Protected retention
    ]

    for label, oracle_ret, prot_ret in matrix_configs:
        if label in exp4_res:
            print(f"[Phase 23] Exp 4 {label} already checkpointed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 23] Exp 4: {label} (oracle={oracle_ret}, protected={prot_ret})...", file=sys.stderr)
        run_res, tracker = run_phase23_stream(
            observations=stream_exp4, capacity=25, system="D8",
            nli_gate=nli_gate, dense_encoder=dense_encoder,
            entail_threshold=0.75,
            oracle_retrieval=oracle_ret,
            protected_retention=prot_ret
        )
        entry = evaluate_phase23_run(run_res, stream_exp4, tracker)
        exp4_res[label] = entry
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment 5: Embedding Drift Analysis
    # ──────────────────────────────────────────────
    if "exp5_embedding_drift" not in results:
        results["exp5_embedding_drift"] = {}
    exp5_res = results["exp5_embedding_drift"]

    stream_exp5 = generate_exp5_embedding_drift_stream(n_targets=8, updates_per_target=4, seed=SEED_P23)
    print(f"[Phase 23] Exp 5 (Embedding Drift): {len(stream_exp5)} obs, C=50", file=sys.stderr)

    for sys_id in ["D8", "H1"]:
        if sys_id in exp5_res:
            print(f"[Phase 23] Exp 5 {sys_id} already checkpointed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 23] Exp 5: System={sys_id}...", file=sys.stderr)
        run_res, tracker = run_phase23_stream(
            observations=stream_exp5, capacity=50, system=sys_id,
            nli_gate=nli_gate, dense_encoder=dense_encoder,
            entail_threshold=0.75
        )
        entry = evaluate_phase23_run(run_res, stream_exp5, tracker)

        # Collect detailed per-target drift progressions
        drift_details = {}
        for tgt_id, rec in tracker.targets.items():
            drift_details[tgt_id] = {
                "update_count": len(rec.update_history),
                "history": rec.update_history,
                "evicted": rec.evicted
            }
        entry["drift_details"] = drift_details
        exp5_res[sys_id] = entry
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment 6: Consolidation Traces (Step-by-Step)
    # ──────────────────────────────────────────────
    if "exp6_traces" not in results:
        results["exp6_traces"] = {}
    exp6_res = results["exp6_traces"]

    stream_exp6 = generate_exp6_trace_stream(seed=SEED_P23)
    print(f"[Phase 23] Exp 6 (Traces): {len(stream_exp6)} obs, C=20", file=sys.stderr)

    for sys_id in ["D8", "H1"]:
        if sys_id in exp6_res:
            print(f"[Phase 23] Exp 6 {sys_id} already checkpointed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 23] Exp 6: System={sys_id}...", file=sys.stderr)
        run_res, tracker = run_phase23_stream(
            observations=stream_exp6, capacity=20, system=sys_id,
            nli_gate=nli_gate, dense_encoder=dense_encoder,
            entail_threshold=0.75
        )
        entry = evaluate_phase23_run(run_res, stream_exp6, tracker)
        # Store selected case traces
        case_traces = []
        for att in tracker.attributions:
            case_traces.append({
                "target_id": att.target_id,
                "probe_id": att.probe_id,
                "category": att.details.get("category"),
                "failure_category": att.failure_category,
                "slot_exists_at_probe": att.slot_exists_at_probe,
                "correct_slot_retrieved": att.correct_slot_retrieved,
                "retrieval_rank": att.retrieval_rank,
                "nli_called": att.nli_called,
                "nli_score": att.nli_score,
                "nli_accepted": att.nli_accepted,
                "final_correct": att.final_correct
            })
        entry["case_traces"] = case_traces
        exp6_res[sys_id] = entry
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment 7: Sealed Held-Out Benchmark (seed=2323)
    # ──────────────────────────────────────────────
    if "exp7_held_out" not in results:
        results["exp7_held_out"] = {}
    exp7_res = results["exp7_held_out"]

    stream_exp7 = generate_held_out_phase23(n_obs=800, seed=SEED_P23)
    print(f"[Phase 23] Exp 7 (Held-Out, seed={SEED_P23}): {len(stream_exp7)} obs, C=50", file=sys.stderr)

    for sys_id in ["D", "D4", "D8", "H1"]:
        if sys_id in exp7_res:
            print(f"[Phase 23] Exp 7 System={sys_id} already checkpointed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 23] Exp 7 (HELD-OUT): System={sys_id}...", file=sys.stderr)
        run_res, tracker = run_phase23_stream(
            observations=stream_exp7, capacity=50, system=sys_id,
            nli_gate=nli_gate, dense_encoder=dense_encoder,
            entail_threshold=0.75
        )
        entry = evaluate_phase23_run(run_res, stream_exp7, tracker)
        exp7_res[sys_id] = entry
        save_checkpoint(results)

    print("\n[Phase 23] ALL EXPERIMENTS COMPLETE.", file=sys.stderr)
    print(f"[Phase 23] Results saved to: {CHECKPOINT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
