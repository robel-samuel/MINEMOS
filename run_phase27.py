"""
run_phase27.py

Top-level experiment driver for MINEMOS Phase 27:
Adaptive Memory Retrieval Under Long-Horizon Interference.

Executes:
  1. Dataset integrity and isolation verification
  2. Experiment A: Retrieval Methods Across Interference Conditions
  3. Experiment B: Long-Horizon Interference Scaling (N=25, 50, 100)
  4. Experiment C: 6-Way Controlled Ablation Study
  5. Experiment D: Frozen Held-Out Benchmark Generalization
  6. Experiment E: Computational Efficiency & Latency Profiling
  7. Saves comprehensive results to scratch/phase27_results.json
"""

import os
import json
import time

from benchmarks.phase27_dataset import verify_phase27_dataset_integrity
from benchmarks.phase27_experiment import (
    run_experiment_a_conditions,
    run_experiment_b_horizon_scaling,
    run_experiment_c_ablations,
    run_experiment_d_held_out,
    run_experiment_e_complexity
)


def main():
    print("=====================================================================")
    print("  MINEMOS Phase 27: Adaptive Memory Retrieval Under Long-Horizon Interference")
    print("=====================================================================")

    # 1. Dataset verification
    print("\n[Step 1/6] Verifying dataset integrity and split isolation...")
    integrity = verify_phase27_dataset_integrity()
    print(f"  Dataset Integrity: {integrity['status']} (Dev: {integrity['dev_stream_length']} obs, Held: {integrity['held_stream_length']} obs)")
    assert integrity["status"] == "PASS", f"Dataset integrity check failed: {integrity}"

    # 2. Experiment A: Methods across conditions
    print("\n[Step 2/6] Executing Experiment A: Methods across 4 Interference Conditions...")
    exp_a = run_experiment_a_conditions()

    # 3. Experiment B: Long-horizon scaling
    print("\n[Step 3/6] Executing Experiment B: Long-Horizon Interference Scaling...")
    exp_b = run_experiment_b_horizon_scaling()

    # 4. Experiment C: 6-way ablations
    print("\n[Step 4/6] Executing Experiment C: 6-Way Controlled Ablation Study...")
    exp_c = run_experiment_c_ablations()

    # 5. Experiment D: Held-out generalization
    print("\n[Step 5/6] Executing Experiment D: Frozen Held-Out Generalization Benchmark...")
    exp_d = run_experiment_d_held_out()

    # 6. Experiment E: Complexity profiling
    print("\n[Step 6/6] Executing Experiment E: Computational Efficiency & Latency Profiling...")
    exp_e = run_experiment_e_complexity()

    # Consolidate results
    results = {
        "phase": 27,
        "timestamp": time.time(),
        "integrity": integrity,
        "experiment_a_conditions": exp_a,
        "experiment_b_horizon_scaling": exp_b,
        "experiment_c_ablations": exp_c,
        "experiment_d_held_out": exp_d,
        "experiment_e_complexity": exp_e
    }

    out_path = os.path.join("scratch", "phase27_results.json")
    os.makedirs("scratch", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nPhase 27 experiments complete. Results saved to: {out_path}")


if __name__ == "__main__":
    main()
