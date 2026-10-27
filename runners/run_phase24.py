"""
run_phase24.py

Phase 24 Comprehensive Benchmark Runner:
Retention Hardening, Semantic Verification Recalibration, and Drift-Resistant Consolidation.

Executes all 6 experiments and logs structured checkpoints into benchmarks/results/phase24_results.json:
  - Experiment A: Practical Retention Policies (FIFO, Recency, Utility, Verification-Aware, Tiered, Oracle)
  - Experiment B: Semantic Verification Recalibration (Threshold Sweep 0.55-0.75, Asymmetric, Bidirectional)
  - Experiment C: Drift-Resistant Consolidation (Baseline, Bounded, Anchor, Hybrid)
  - Experiment D: Combined 8-Cell Ablation Study
  - Experiment E: Long-Horizon & Capacity Stress Scaling
  - Experiment F: Sealed Held-Out Benchmark (seed=2424)
"""

from __future__ import annotations
import os, sys, json, time
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")

import numpy as np
import torch
torch.set_num_threads(4)

from phase2.memory_state import MemoryState, MemorySlot
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
from benchmarks.phase24_verification import CalibratedSemanticVerifier
from benchmarks.phase24_dataset import (
    generate_exp_a_retention_stream,
    generate_exp_b_calibration_stream,
    generate_exp_c_drift_stream,
    generate_exp_d_ablation_stream,
    generate_exp_e_stress_stream,
    generate_held_out_phase24,
    SEED_P24_TUNING,
    SEED_P24_HELD_OUT,
)
from benchmarks.phase24_experiment import run_phase24_stream
from benchmarks.phase24_metrics import evaluate_phase24_run

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "phase24_results.json")

FROZEN_CONFIG_P24 = {
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
    "seed_tuning": SEED_P24_TUNING,
    "seed_held_out": SEED_P24_HELD_OUT,
}


def serialize(obj: Any) -> Any:
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
    print(f"[Phase 24 Checkpoint] Saved to {CHECKPOINT_PATH}", file=sys.stderr)


def main():
    print("[Phase 24] Initializing models on 4 CPU threads...", file=sys.stderr)
    t0 = time.perf_counter()
    nli_gate = NliSemanticGate()
    calibrated_verifier = CalibratedSemanticVerifier(nli_gate)
    dense_encoder = DenseSemanticEncoder()
    print(f"[Phase 24] Models loaded in {time.perf_counter() - t0:.2f}s", file=sys.stderr)

    results = load_checkpoint()
    if "frozen_config" not in results:
        results["frozen_config"] = FROZEN_CONFIG_P24
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment A: Practical Retention Policies
    # ──────────────────────────────────────────────
    if "exp_a_retention" not in results:
        results["exp_a_retention"] = {}
    exp_a_res = results["exp_a_retention"]

    retention_policies = [
        "fifo",
        "recency",
        "utility",
        "verification_aware",
        "tiered",
        "oracle"
    ]

    for workload in ["ordinary", "adversarial"]:
        if workload not in exp_a_res:
            exp_a_res[workload] = {}

        stream_a = generate_exp_a_retention_stream(mode=workload, seed=SEED_P24_TUNING)
        capacity_a = 50 if workload == "ordinary" else 25
        print(f"\n[Phase 24] Exp A ({workload}): {len(stream_a)} obs, C={capacity_a}", file=sys.stderr)

        for pol in retention_policies:
            if pol in exp_a_res[workload]:
                print(f"[Phase 24] Exp A ({workload}) {pol} already completed, skipping.", file=sys.stderr)
                continue

            print(f"[Phase 24] Exp A ({workload}): Testing policy '{pol}'...", file=sys.stderr)
            run_res, tracker = run_phase24_stream(
                observations=stream_a,
                capacity=capacity_a,
                system="D8",
                retention_policy=pol,
                verification_mode="baseline",
                entail_threshold=0.75,
                consolidation_mode="baseline",
                calibrated_verifier=calibrated_verifier,
                dense_encoder=dense_encoder,
                protected_retention=(pol == "oracle")
            )
            entry = evaluate_phase24_run(run_res, stream_a, tracker)
            exp_a_res[workload][pol] = entry
            save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment B: Semantic Verification Recalibration
    # ──────────────────────────────────────────────
    if "exp_b_verification" not in results:
        results["exp_b_verification"] = {}
    exp_b_res = results["exp_b_verification"]

    stream_b = generate_exp_b_calibration_stream(seed=SEED_P24_TUNING)
    print(f"\n[Phase 24] Exp B (Verification): {len(stream_b)} obs, C=50", file=sys.stderr)

    threshold_sweep = [0.55, 0.60, 0.65, 0.70, 0.75]

    # B1: Standard forward NLI threshold sweep
    if "forward_threshold_sweep" not in exp_b_res:
        exp_b_res["forward_threshold_sweep"] = {}
    for th in threshold_sweep:
        th_key = f"{th:.2f}"
        if th_key in exp_b_res["forward_threshold_sweep"]:
            continue
        print(f"[Phase 24] Exp B1: Forward sweep th={th_key}...", file=sys.stderr)
        run_res, tracker = run_phase24_stream(
            observations=stream_b, capacity=50, system="D8",
            retention_policy="fifo", verification_mode="baseline",
            entail_threshold=th, consolidation_mode="baseline",
            calibrated_verifier=calibrated_verifier, dense_encoder=dense_encoder
        )
        entry = evaluate_phase24_run(run_res, stream_b, tracker)
        exp_b_res["forward_threshold_sweep"][th_key] = entry
        save_checkpoint(results)

    # B2: Asymmetric decision threshold sweep (contra=0.40, margin=0.15)
    if "asymmetric_sweep" not in exp_b_res:
        exp_b_res["asymmetric_sweep"] = {}
    for th in threshold_sweep:
        th_key = f"{th:.2f}"
        if th_key in exp_b_res["asymmetric_sweep"]:
            continue
        print(f"[Phase 24] Exp B2: Asymmetric sweep th={th_key}...", file=sys.stderr)
        run_res, tracker = run_phase24_stream(
            observations=stream_b, capacity=50, system="D8",
            retention_policy="fifo", verification_mode="asymmetric",
            entail_threshold=th, contra_threshold=0.40, consolidation_mode="baseline",
            calibrated_verifier=calibrated_verifier, dense_encoder=dense_encoder
        )
        entry = evaluate_phase24_run(run_res, stream_b, tracker)
        exp_b_res["asymmetric_sweep"][th_key] = entry
        save_checkpoint(results)

    # B3: Bidirectional symmetric equivalence sweep (with conservative contradiction veto)
    if "bidirectional_sweep" not in exp_b_res:
        exp_b_res["bidirectional_sweep"] = {}
    for th in threshold_sweep:
        th_key = f"{th:.2f}"
        if th_key in exp_b_res["bidirectional_sweep"]:
            continue
        print(f"[Phase 24] Exp B3: Bidirectional sweep th={th_key}...", file=sys.stderr)
        run_res, tracker = run_phase24_stream(
            observations=stream_b, capacity=50, system="D8",
            retention_policy="fifo", verification_mode="bidirectional",
            entail_threshold=th, contra_threshold=0.40, consolidation_mode="baseline",
            calibrated_verifier=calibrated_verifier, dense_encoder=dense_encoder
        )
        entry = evaluate_phase24_run(run_res, stream_b, tracker)
        exp_b_res["bidirectional_sweep"][th_key] = entry
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment C: Drift-Resistant Consolidation
    # ──────────────────────────────────────────────
    if "exp_c_consolidation" not in results:
        results["exp_c_consolidation"] = {}
    exp_c_res = results["exp_c_consolidation"]

    stream_c = generate_exp_c_drift_stream(n_targets=8, updates_per_target=4, seed=SEED_P24_TUNING)
    print(f"\n[Phase 24] Exp C (Consolidation Drift): {len(stream_c)} obs, C=50", file=sys.stderr)

    consolidation_modes = [
        ("baseline", 0.25, 0.20),
        ("bounded", 0.25, 0.20),
        ("anchor", 0.25, 0.20),
        ("hybrid", 0.25, 0.20),
    ]

    for mode, d_max, l_anchor in consolidation_modes:
        if mode in exp_c_res:
            print(f"[Phase 24] Exp C mode '{mode}' already completed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 24] Exp C: Testing mode '{mode}' (delta_max={d_max}, lambda={l_anchor})...", file=sys.stderr)
        run_res, tracker = run_phase24_stream(
            observations=stream_c, capacity=50, system="D8",
            retention_policy="fifo", verification_mode="baseline",
            entail_threshold=0.75, consolidation_mode=mode,
            delta_max=d_max, lambda_anchor=l_anchor,
            calibrated_verifier=calibrated_verifier, dense_encoder=dense_encoder
        )
        entry = evaluate_phase24_run(run_res, stream_c, tracker)
        # Capture per-target drift trajectory
        drift_trajectories = {}
        for tgt_id, rec in tracker.targets.items():
            drift_trajectories[tgt_id] = rec.update_history
        entry["drift_trajectories"] = drift_trajectories
        exp_c_res[mode] = entry
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment D: Combined Ablation Study (8 Cells)
    # ──────────────────────────────────────────────
    if "exp_d_ablation" not in results:
        results["exp_d_ablation"] = {}
    exp_d_res = results["exp_d_ablation"]

    stream_d = generate_exp_d_ablation_stream(seed=SEED_P24_TUNING)
    print(f"\n[Phase 24] Exp D (Combined Ablation): {len(stream_d)} obs, C=30", file=sys.stderr)

    # Select strongest candidates from Exp A, B, C:
    # A candidate: Tiered retention
    # B candidate: Calibrated Bidirectional (or Asymmetric) verification at frozen tau=0.60
    # C candidate: Hybrid drift-resistant consolidation
    SELECTED_RETENTION = "tiered"
    SELECTED_VERIFICATION = "bidirectional"
    SELECTED_TAU = 0.60
    SELECTED_CONSOLIDATION = "hybrid"

    ablation_matrix = [
        ("1_baseline_all", "fifo", "baseline", 0.75, "baseline"),
        ("2_improved_retention_only", SELECTED_RETENTION, "baseline", 0.75, "baseline"),
        ("3_improved_verification_only", "fifo", SELECTED_VERIFICATION, SELECTED_TAU, "baseline"),
        ("4_improved_consolidation_only", "fifo", "baseline", 0.75, SELECTED_CONSOLIDATION),
        ("5_retention_plus_verification", SELECTED_RETENTION, SELECTED_VERIFICATION, SELECTED_TAU, "baseline"),
        ("6_retention_plus_consolidation", SELECTED_RETENTION, "baseline", 0.75, SELECTED_CONSOLIDATION),
        ("7_verification_plus_consolidation", "fifo", SELECTED_VERIFICATION, SELECTED_TAU, SELECTED_CONSOLIDATION),
        ("8_all_combined", SELECTED_RETENTION, SELECTED_VERIFICATION, SELECTED_TAU, SELECTED_CONSOLIDATION),
    ]

    for cell_id, ret_pol, ver_mode, tau_e, cons_mode in ablation_matrix:
        if cell_id in exp_d_res:
            print(f"[Phase 24] Exp D cell '{cell_id}' already completed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 24] Exp D: Running {cell_id} (ret={ret_pol}, ver={ver_mode}, tau={tau_e}, cons={cons_mode})...", file=sys.stderr)
        run_res, tracker = run_phase24_stream(
            observations=stream_d, capacity=30, system="D8",
            retention_policy=ret_pol, verification_mode=ver_mode,
            entail_threshold=tau_e, contra_threshold=0.40,
            consolidation_mode=cons_mode, delta_max=0.25, lambda_anchor=0.20,
            calibrated_verifier=calibrated_verifier, dense_encoder=dense_encoder
        )
        entry = evaluate_phase24_run(run_res, stream_d, tracker)
        entry["config"] = {
            "retention_policy": ret_pol,
            "verification_mode": ver_mode,
            "entail_threshold": tau_e,
            "consolidation_mode": cons_mode
        }
        exp_d_res[cell_id] = entry
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment E: Long Horizon & Capacity Stress
    # ──────────────────────────────────────────────
    if "exp_e_stress" not in results:
        results["exp_e_stress"] = {}
    exp_e_res = results["exp_e_stress"]

    stream_lengths = [500, 1000, 2000]  # and test larger if feasible
    capacities = [25, 50]

    for n_len in stream_lengths:
        len_key = f"N_{n_len}"
        if len_key not in exp_e_res:
            exp_e_res[len_key] = {}

        stream_e = generate_exp_e_stress_stream(n_obs=n_len, seed=SEED_P24_TUNING)
        for cap in capacities:
            cap_key = f"C_{cap}"
            if cap_key not in exp_e_res[len_key]:
                exp_e_res[len_key][cap_key] = {}

            for sys_type, ret_p, ver_m, tau_val, cons_m in [
                ("baseline", "fifo", "baseline", 0.75, "baseline"),
                ("improved", SELECTED_RETENTION, SELECTED_VERIFICATION, SELECTED_TAU, SELECTED_CONSOLIDATION)
            ]:
                if sys_type in exp_e_res[len_key][cap_key]:
                    continue
                print(f"[Phase 24] Exp E: N={n_len}, C={cap}, System={sys_type}...", file=sys.stderr)
                t_run0 = time.perf_counter()
                run_res, tracker = run_phase24_stream(
                    observations=stream_e, capacity=cap, system="D8",
                    retention_policy=ret_p, verification_mode=ver_m,
                    entail_threshold=tau_val, contra_threshold=0.40,
                    consolidation_mode=cons_m, delta_max=0.25, lambda_anchor=0.20,
                    calibrated_verifier=calibrated_verifier, dense_encoder=dense_encoder
                )
                entry = evaluate_phase24_run(run_res, stream_e, tracker)
                entry["measured_duration"] = round(time.perf_counter() - t_run0, 2)
                exp_e_res[len_key][cap_key][sys_type] = entry
                save_checkpoint(results)

    # ──────────────────────────────────────────────
    #  Experiment F: Sealed Held-Out Benchmark (seed=2424)
    # ──────────────────────────────────────────────
    if "exp_f_held_out" not in results:
        results["exp_f_held_out"] = {}
    exp_f_res = results["exp_f_held_out"]

    stream_f = generate_held_out_phase24(n_obs=800, seed=SEED_P24_HELD_OUT)
    print(f"\n[Phase 24] Exp F (Held-Out, seed={SEED_P24_HELD_OUT}): {len(stream_f)} obs, C=50", file=sys.stderr)

    held_out_systems = [
        ("D8_baseline", "D8", "fifo", "baseline", 0.75, "baseline"),
        ("H1_baseline", "H1", "fifo", "baseline", 0.75, "baseline"),
        ("D8_improved", "D8", SELECTED_RETENTION, SELECTED_VERIFICATION, SELECTED_TAU, SELECTED_CONSOLIDATION),
        ("H1_improved", "H1", SELECTED_RETENTION, SELECTED_VERIFICATION, SELECTED_TAU, SELECTED_CONSOLIDATION),
    ]

    for label, sys_id, ret_p, ver_m, tau_val, cons_m in held_out_systems:
        if label in exp_f_res:
            print(f"[Phase 24] Exp F {label} already completed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 24] Exp F (HELD-OUT): Evaluating {label}...", file=sys.stderr)
        run_res, tracker = run_phase24_stream(
            observations=stream_f, capacity=50, system=sys_id,
            retention_policy=ret_p, verification_mode=ver_m,
            entail_threshold=tau_val, contra_threshold=0.40,
            consolidation_mode=cons_m, delta_max=0.25, lambda_anchor=0.20,
            calibrated_verifier=calibrated_verifier, dense_encoder=dense_encoder
        )
        entry = evaluate_phase24_run(run_res, stream_f, tracker)
        entry["config"] = {
            "system": sys_id,
            "retention_policy": ret_p,
            "verification_mode": ver_m,
            "entail_threshold": tau_val,
            "consolidation_mode": cons_m
        }
        exp_f_res[label] = entry
        save_checkpoint(results)

    print("\n[Phase 24] ALL EXPERIMENTS COMPLETE.", file=sys.stderr)
    print(f"[Phase 24] Results checkpointed to: {CHECKPOINT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
