"""
run_phase20.py

Comprehensive Phase 20 Benchmark Runner with Incremental Checkpoints.
Saves results to scratch/phase20_results.json after each experiment and sub-condition.
"""

from __future__ import annotations
import os, sys, json, time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
torch.set_num_threads(4)

from phase2.memory_state import MemoryState, MemorySlot
from benchmarks.phase16_experiment import NliSemanticGate
from benchmarks.phase18_experiment import DenseSemanticEncoder
from benchmarks.phase18_dataset import (
    generate_scenario_1_long_gaps,
    generate_scenario_5_capacity_pressure_stream,
    generate_scenario_6_continual_stream
)
from benchmarks.phase18_metrics import (
    evaluate_retrieval_decomposition,
    evaluate_memory_consolidation
)
from benchmarks.phase20_dataset import generate_scenario_7_semantic_drift
from benchmarks.phase20_experiment import run_phase20_stream
from benchmarks.phase20_metrics import evaluate_phase20_diagnostics

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "scratch", "phase20_results.json")


def serialize(obj):
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 4)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
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
    print(f"[Phase 20 Checkpoint] Saved to {CHECKPOINT_PATH}", file=sys.stderr)


def evaluate_run(res, stream):
    m = evaluate_memory_consolidation(res, stream)
    d = evaluate_retrieval_decomposition(stream, res["trace"], res["memory"])
    diag20 = evaluate_phase20_diagnostics(res, stream, res["memory"])
    return {
        "metrics": serialize(m),
        "diagnostics": serialize(d),
        "phase20_diagnostics": serialize(diag20),
        "wall_seconds": round(res["wall_seconds"], 2),
        "nli_calls": res["nli_calls"]
    }


def main():
    print(f"[Phase 20] Initializing models on {torch.get_num_threads()} CPU threads...", file=sys.stderr)
    t_start = time.perf_counter()
    nli_gate = NliSemanticGate()
    dense_encoder = DenseSemanticEncoder()
    print(f"[Phase 20] Models loaded in {time.perf_counter()-t_start:.2f}s", file=sys.stderr)

    results = load_checkpoint()

    # Shared stream for parameter sweeps: Scenario 5 (1,500 obs, C=200)
    stream_cap = generate_scenario_5_capacity_pressure_stream()

    # ──────────────────────────────────────────────
    # 1. Experiment 1: Anchor Weight Sweep
    # ──────────────────────────────────────────────
    if "exp1_weight_sweep" not in results:
        results["exp1_weight_sweep"] = {}
    exp1_res = results["exp1_weight_sweep"]

    weights = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
    for w in weights:
        w_str = str(w)
        if w_str in exp1_res:
            continue
        print(f"[Phase 20] Exp 1 (Weight Sweep): w_anchor={w}...", file=sys.stderr)
        res = run_phase20_stream(
            observations=stream_cap, capacity=200, system="D4" if w > 0 else "D",
            nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3,
            w_anchor=w, decay_lambda=0.0, max_anchors=None
        )
        exp1_res[w_str] = evaluate_run(res, stream_cap)
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    # 2. Experiment 2: Bounded Anchor Capacity Sweep
    # ──────────────────────────────────────────────
    if "exp2_capacity_sweep" not in results:
        results["exp2_capacity_sweep"] = {}
    exp2_res = results["exp2_capacity_sweep"]

    caps = [1, 3, 5, 10, None]
    for k in caps:
        k_str = "unlimited" if k is None else str(k)
        if k_str in exp2_res:
            continue
        print(f"[Phase 20] Exp 2 (Capacity Sweep): max_anchors={k_str}...", file=sys.stderr)
        res = run_phase20_stream(
            observations=stream_cap, capacity=200, system="D4",
            nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3,
            w_anchor=0.20, decay_lambda=0.0, max_anchors=k
        )
        exp2_res[k_str] = evaluate_run(res, stream_cap)
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    # 3. Experiment 3: Temporal Decay Parameter Sweep
    # ──────────────────────────────────────────────
    if "exp3_decay_sweep" not in results:
        results["exp3_decay_sweep"] = {}
    exp3_res = results["exp3_decay_sweep"]

    lambdas = [0.0, 0.0005, 0.001, 0.005, 0.01]
    for lmb in lambdas:
        lmb_str = str(lmb)
        if lmb_str in exp3_res:
            continue
        print(f"[Phase 20] Exp 3 (Decay Sweep): decay_lambda={lmb}...", file=sys.stderr)
        res = run_phase20_stream(
            observations=stream_cap, capacity=200, system="D4",
            nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3,
            w_anchor=0.20, decay_lambda=lmb, max_anchors=None
        )
        exp3_res[lmb_str] = evaluate_run(res, stream_cap)
        save_checkpoint(results)

    # Determine empirical best parameters from sweeps for D5, D6, D7, D8
    # Selection criterion: Harmonic mean of Safety (1 - false_merge_rate) and Recall (end_to_end_accuracy)
    def compute_balanced_score(entry):
        fmr = entry["phase20_diagnostics"]["contradiction_false_merge_rate"]
        e2e = entry["diagnostics"]["end_to_end_accuracy"]
        safety = max(0.0, 1.0 - fmr)
        if safety + e2e == 0:
            return 0.0
        return 2.0 * safety * e2e / (safety + e2e)

    # 1. Best weight from Exp 1 (excluding 0.0 which has no repulsion)
    w_candidates = [w for w in weights if w > 0.0 and str(w) in exp1_res]
    if w_candidates:
        best_w = max(w_candidates, key=lambda w: compute_balanced_score(exp1_res[str(w)]))
    else:
        best_w = 0.20

    # 2. Best cap from Exp 2
    k_candidates = [k for k in caps if ("unlimited" if k is None else str(k)) in exp2_res]
    if k_candidates:
        best_k = max(k_candidates, key=lambda k: compute_balanced_score(exp2_res["unlimited" if k is None else str(k)]))
    else:
        best_k = 3

    # 3. Best lambda from Exp 3
    lmb_candidates = [lmb for lmb in lambdas if str(lmb) in exp3_res]
    if lmb_candidates:
        best_lambda = max(lmb_candidates, key=lambda l: compute_balanced_score(exp3_res[str(l)]))
    else:
        best_lambda = 0.005

    print(f"[Phase 20] Empirically Selected Parameters: best_w={best_w}, best_k={best_k}, best_lambda={best_lambda}", file=sys.stderr)

    # ──────────────────────────────────────────────
    # 4. Experiment 4: 5,000-Observation Continual Stream (Critical)
    # ──────────────────────────────────────────────
    if "exp4_continual_5000" not in results:
        results["exp4_continual_5000"] = {}
    exp4_res = results["exp4_continual_5000"]

    stream_5k = generate_scenario_6_continual_stream(n_obs=5000)

    systems_5k = [
        ("D", "D", 0.0, 0.0, None),                       # Phase 18 baseline (no anchors)
        ("D4", "D4", 0.20, 0.0, None),                    # Phase 19 baseline (unbounded static)
        ("D5", "D5", 0.20, best_lambda, None),            # D4 + adaptive decay (k=unbounded)
        ("D6", "D6", 0.20, 0.0, best_k),                  # D4 + bounded anchors (k=1, lambda=0)
        ("D7", "D7", 0.20, best_lambda, best_k),          # D4 + decay + tight bound (k=1, lambda=0.01)
        ("D8", "D8", 0.15, 0.005, 3),                     # D8: Balanced multi-anchor decay (w=0.15, k=3, lambda=0.005)
    ]

    for sys_label, sys_type, w_val, lmb_val, k_val in systems_5k:
        if sys_label in exp4_res:
            print(f"[Phase 20] Exp 4: System={sys_label} already present, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 20] Exp 4 (5k Stream): System={sys_label} (w={w_val}, lambda={lmb_val}, k={k_val})...", file=sys.stderr)
        res = run_phase20_stream(
            observations=stream_5k, capacity=200, system=sys_type,
            nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3,
            w_anchor=w_val, decay_lambda=lmb_val, max_anchors=k_val
        )
        exp4_res[sys_label] = evaluate_run(res, stream_5k)
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    # ──────────────────────────────────────────────
    # 5. Experiment 6: Controlled Semantic Drift & Factual Update
    # ──────────────────────────────────────────────
    if "exp6_semantic_drift" not in results:
        results["exp6_semantic_drift"] = {}
    exp6_res = results["exp6_semantic_drift"]

    stream_drift = generate_scenario_7_semantic_drift(n_concepts=30, inter_gap=15)
    drift_systems = [
        ("D", "D", 0.0, 0.0, None),
        ("D4", "D4", 0.20, 0.0, None),
        ("D5", "D5", 0.20, best_lambda, None),
        ("D6", "D6", 0.20, 0.0, best_k),
        ("D7", "D7", 0.20, best_lambda, best_k),
        ("D8", "D8", 0.15, 0.005, 3),
    ]

    for sys_label, sys_type, w_val, lmb_val, k_val in drift_systems:
        if sys_label in exp6_res:
            continue
        print(f"[Phase 20] Exp 6 (Semantic Drift): System={sys_label}...", file=sys.stderr)
        res = run_phase20_stream(
            observations=stream_drift, capacity=100, system=sys_type,
            nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3,
            w_anchor=w_val, decay_lambda=lmb_val, max_anchors=k_val
        )
        exp6_res[sys_label] = evaluate_run(res, stream_drift)
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    # 6. Experiment 5: Long-Gap Paraphrase Recall
    # ──────────────────────────────────────────────
    if "exp5_long_gap" not in results:
        results["exp5_long_gap"] = {}
    exp5_res = results["exp5_long_gap"]

    gap_streams = generate_scenario_1_long_gaps(gaps=(10, 50, 100, 500))
    long_gap_systems = [
        ("D", "D", 0.0, 0.0, None),
        ("D4", "D4", 0.20, 0.0, None),
        ("D7", "D7", 0.20, best_lambda, best_k),
        ("D8", "D8", 0.15, 0.005, 3)
    ]

    for gap, stream in gap_streams.items():
        gap_str = str(gap)
        if gap_str not in exp5_res:
            exp5_res[gap_str] = {}
        for sys_label, sys_type, w_val, lmb_val, k_val in long_gap_systems:
            if sys_label in exp5_res[gap_str]:
                continue
            print(f"[Phase 20] Exp 5 (Long Gap): Gap={gap}, System={sys_label}...", file=sys.stderr)
            res = run_phase20_stream(
                observations=stream, capacity=50, system=sys_type,
                nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3,
                w_anchor=w_val, decay_lambda=lmb_val, max_anchors=k_val
            )
            r_dict = evaluate_run(res, stream)
            probes = [(o, t) for o, t in zip(stream, res["trace"]) if getattr(o, "is_probe", False)]
            merged = sum(1 for _, t in probes if t.get("action") == "update")
            r_dict["probe_recall_rate"] = round(merged / len(probes), 4) if probes else 0.0
            exp5_res[gap_str][sys_label] = r_dict
            save_checkpoint(results)

    # ──────────────────────────────────────────────
    # 7. Experiment 7: Component Ablation & Pareto Analysis
    # ──────────────────────────────────────────────
    if "exp7_ablation" not in results:
        print("[Phase 20] Compiling Experiment 7: Component Ablation & Pareto Summary...", file=sys.stderr)
        exp7_res = {
            "continual_5000": {
                sys_id: results["exp4_continual_5000"][sys_id]
                for sys_id in results["exp4_continual_5000"]
            },
            "semantic_drift": {
                sys_id: results["exp6_semantic_drift"][sys_id]
                for sys_id in results["exp6_semantic_drift"]
            }
        }
        results["exp7_ablation"] = exp7_res
        save_checkpoint(results)

    total_time = time.perf_counter() - t_start
    print(f"[Phase 20] All benchmark experiments completed in {total_time:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
