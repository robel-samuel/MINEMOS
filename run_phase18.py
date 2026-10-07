"""
run_phase18.py

Phase 18 Full Benchmark Suite:
Semantic Dense Retrieval + Utility-Aware Eviction (2x2 Factorial).

Experiments:
1. Top-k Dense Retrieval Ablation (k = 1, 3, 5, 10) on unconstrained capacity
2. Safety Test: SAME vs CONTRADICTION vs UNRELATED (Systems A, B, C, D)
3. 2x2 Factorial across Capacities C in {50, 100, 200, 1500}
4. Long-Gap Paraphrase Recall under G in {10, 50, 100, 500}
5. 5,000-Observation Continual Stream (Systems A, B, C, D)

Saves incremental checkpoints to scratch/phase18_results.json.
"""

from __future__ import annotations
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.phase18_dataset import (
    generate_scenario_1_long_gaps,
    generate_scenario_2_safety_test,
    generate_scenario_3_contradiction_recovery,
    generate_scenario_4_semantic_drift,
    generate_scenario_5_capacity_pressure_stream,
    generate_scenario_6_continual_stream
)
from benchmarks.phase18_experiment import (
    DenseSemanticEncoder,
    run_phase18_stream
)
from benchmarks.phase18_metrics import (
    evaluate_retrieval_decomposition,
    evaluate_memory_consolidation
)
from benchmarks.phase16_experiment import NliSemanticGate

SYSTEMS = ["A", "B", "C", "D"]
OUT_PATH = os.path.join("scratch", "phase18_results.json")


def serialize(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    return obj


def save_checkpoint(results: dict):
    os.makedirs("scratch", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Phase 18 Checkpoint] Saved to {OUT_PATH}", file=sys.stderr)


def main():
    print("[Phase 18] Initializing models locally...", file=sys.stderr)
    t_start = time.perf_counter()
    nli_gate = NliSemanticGate()
    dense_encoder = DenseSemanticEncoder()
    print(f"[Phase 18] Models loaded in {time.perf_counter()-t_start:.2f}s", file=sys.stderr)

    results = {}
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, "r") as f:
                results = json.load(f)
            print(f"[Phase 18] Loaded existing checkpoint keys: {list(results.keys())}", file=sys.stderr)
        except Exception:
            results = {}

    # ──────────────────────────────────────────────
    # 1. Top-k Retrieval Ablation (k = 1, 3, 5, 10)
    # ──────────────────────────────────────────────
    if "top_k_ablation" not in results:
        print("[Phase 18] Running Experiment 1: Top-k Retrieval Ablation...", file=sys.stderr)
        t_exp = time.perf_counter()
        stream_cap = generate_scenario_5_capacity_pressure_stream()  # 1500 obs
        k_results = {}
        for k in (1, 3, 5, 10):
            print(f"  Testing Dense Retrieval with k={k}...", file=sys.stderr)
            res = run_phase18_stream(
                observations=stream_cap, capacity=1500, system="B",
                nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=k
            )
            m = evaluate_memory_consolidation(res, stream_cap)
            diag = evaluate_retrieval_decomposition(stream_cap, res["trace"], res["memory"])
            k_results[str(k)] = {
                "metrics": serialize(m),
                "diagnostics": serialize(diag),
                "wall_seconds": round(res["wall_seconds"], 2)
            }
        results["top_k_ablation"] = k_results
        save_checkpoint(results)
        print(f"[Phase 18] Top-k ablation done in {time.perf_counter()-t_exp:.2f}s", file=sys.stderr)
    else:
        print("[Phase 18] Top-k ablation already present, skipping.", file=sys.stderr)

    # ──────────────────────────────────────────────
    # 2. Safety Test: SAME vs CONTRADICTION vs UNRELATED
    # ──────────────────────────────────────────────
    if "safety_test" not in results:
        print("[Phase 18] Running Experiment 2: Safety Test (SAME vs CONTRADICTION vs UNRELATED)...", file=sys.stderr)
        t_exp = time.perf_counter()
        stream_safety = generate_scenario_2_safety_test()
        safety_results = {}
        for sys_id in SYSTEMS:
            print(f"  Testing Safety on System {sys_id}...", file=sys.stderr)
            res = run_phase18_stream(
                observations=stream_safety, capacity=100, system=sys_id,
                nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3
            )
            m = evaluate_memory_consolidation(res, stream_safety)
            diag = evaluate_retrieval_decomposition(stream_safety, res["trace"], res["memory"])
            safety_results[sys_id] = {
                "metrics": serialize(m),
                "diagnostics": serialize(diag),
                "wall_seconds": round(res["wall_seconds"], 2)
            }
        results["safety_test"] = safety_results
        save_checkpoint(results)
        print(f"[Phase 18] Safety test done in {time.perf_counter()-t_exp:.2f}s", file=sys.stderr)
    else:
        print("[Phase 18] Safety test already present, skipping.", file=sys.stderr)

    # ──────────────────────────────────────────────
    # 3. 2x2 Factorial across Capacities C in {50, 100, 200, 1500}
    # ──────────────────────────────────────────────
    if "capacity_factorial" not in results:
        print("[Phase 18] Running Experiment 3: 2x2 Factorial Capacity Ablation...", file=sys.stderr)
        t_exp = time.perf_counter()
        stream_cap = generate_scenario_5_capacity_pressure_stream()
        cap_results = {}
        for cap in (1500, 200, 100, 50):
            cap_results[str(cap)] = {}
            for sys_id in SYSTEMS:
                print(f"  Capacity={cap}, System={sys_id}...", file=sys.stderr)
                res = run_phase18_stream(
                    observations=stream_cap, capacity=cap, system=sys_id,
                    nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3
                )
                m = evaluate_memory_consolidation(res, stream_cap)
                diag = evaluate_retrieval_decomposition(stream_cap, res["trace"], res["memory"])
                cap_results[str(cap)][sys_id] = {
                    "metrics": serialize(m),
                    "diagnostics": serialize(diag),
                    "wall_seconds": round(res["wall_seconds"], 2)
                }
        results["capacity_factorial"] = cap_results
        save_checkpoint(results)
        print(f"[Phase 18] Capacity factorial done in {time.perf_counter()-t_exp:.2f}s", file=sys.stderr)
    else:
        print("[Phase 18] Capacity factorial already present, skipping.", file=sys.stderr)

    # ──────────────────────────────────────────────
    # 4. Long-Gap Paraphrase Recall (G in {10, 50, 100, 500})
    # ──────────────────────────────────────────────
    if "long_gap_experiment" not in results:
        print("[Phase 18] Running Experiment 4: Long-Gap Paraphrase Recall...", file=sys.stderr)
        t_exp = time.perf_counter()
        gap_streams = generate_scenario_1_long_gaps(gaps=(10, 50, 100, 500))
        gap_results = {}
        for gap, stream in gap_streams.items():
            gap_results[str(gap)] = {}
            for sys_id in SYSTEMS:
                print(f"  Gap={gap}, System={sys_id}, Len={len(stream)}...", file=sys.stderr)
                res = run_phase18_stream(
                    observations=stream, capacity=50, system=sys_id,
                    nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3
                )
                m = evaluate_memory_consolidation(res, stream)
                diag = evaluate_retrieval_decomposition(stream, res["trace"], res["memory"])

                probes = [(o, t) for o, t in zip(stream, res["trace"]) if o.is_probe]
                merged = sum(1 for _, t in probes if t["action"] == "update")
                recall_rate = round(merged / len(probes), 4) if probes else 0.0

                gap_results[str(gap)][sys_id] = {
                    "metrics": serialize(m),
                    "diagnostics": serialize(diag),
                    "probe_recall_rate": recall_rate,
                    "wall_seconds": round(res["wall_seconds"], 2)
                }
        results["long_gap_experiment"] = gap_results
        save_checkpoint(results)
        print(f"[Phase 18] Long-gap experiment done in {time.perf_counter()-t_exp:.2f}s", file=sys.stderr)
    else:
        print("[Phase 18] Long-gap experiment already present, skipping.", file=sys.stderr)

    # ──────────────────────────────────────────────
    # 5. 5,000-Observation Continual Stream (C = 200)
    # ──────────────────────────────────────────────
    if "continual_5000" not in results:
        print("[Phase 18] Running Experiment 5: 5,000-Observation Continual Stream...", file=sys.stderr)
        t_exp = time.perf_counter()
        stream_5k = generate_scenario_6_continual_stream(n_obs=5000)
        continual_results = {}
        for sys_id in SYSTEMS:
            print(f"  Continual 5k, System={sys_id}...", file=sys.stderr)
            res = run_phase18_stream(
                observations=stream_5k, capacity=200, system=sys_id,
                nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3
            )
            m = evaluate_memory_consolidation(res, stream_5k)
            diag = evaluate_retrieval_decomposition(stream_5k, res["trace"], res["memory"])
            continual_results[sys_id] = {
                "metrics": serialize(m),
                "diagnostics": serialize(diag),
                "wall_seconds": round(res["wall_seconds"], 2)
            }
        results["continual_5000"] = continual_results
        save_checkpoint(results)
        print(f"[Phase 18] Continual 5k stream done in {time.perf_counter()-t_exp:.2f}s", file=sys.stderr)
    else:
        print("[Phase 18] Continual 5k stream already present, skipping.", file=sys.stderr)

    total_time = time.perf_counter() - t_start
    print(f"[Phase 18] All benchmark experiments completed in {total_time:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
