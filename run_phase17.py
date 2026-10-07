"""
run_phase17.py

Runs the Phase 17 benchmark suite for Systems A (NLI), B (Oracle), C (Lexical).
Outputs structured JSON results incrementally after each scenario.

Usage:
    python run_phase17.py
"""

from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.phase17_dataset import (
    generate_scenario_1_long_gaps,
    generate_scenario_2_repeated_contradictions,
    generate_scenario_3_contradiction_recovery,
    generate_scenario_4_semantic_drift,
    generate_scenario_5_distractor_stress,
    generate_scenario_6_capacity_pressure_stream,
    generate_scenario_7_continual_stream,
)
from benchmarks.phase17_experiment import run_stream
from benchmarks.phase17_metrics import (
    memory_metrics, contradiction_retention_over_time,
    bounded_head_amplification, candidate_retrieval_diagnostics,
)
from benchmarks.phase16_experiment import NliSemanticGate

SYSTEMS = ["A_nli", "B_oracle", "C_lexical"]
GAPS = (10, 50, 100, 500)
OUT_PATH = os.path.join("scratch", "phase17_results.json")


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
    print(f"[Phase 17 Checkpoint] Saved to {OUT_PATH}", file=sys.stderr)


def main():
    print("[Phase 17] Loading NLI gate...", file=sys.stderr)
    t_start = time.perf_counter()
    gate = NliSemanticGate()
    print(f"[Phase 17] NLI gate ready in {time.perf_counter()-t_start:.2f}s", file=sys.stderr)

    results = {}
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, "r") as f:
                results = json.load(f)
            print(f"[Phase 17] Loaded existing results keys: {list(results.keys())}", file=sys.stderr)
        except Exception:
            results = {}

    # ─── Scenario 1: Long-Gap Paraphrase Recall ───
    if "scenario_1" not in results:
        t_sc = time.perf_counter()
        print("[Phase 17] Scenario 1...", file=sys.stderr)
        s1_streams = generate_scenario_1_long_gaps(gaps=GAPS)
        scenario_1_results = {}
        rvg = {s: {} for s in SYSTEMS}

        for gap, stream in s1_streams.items():
            scenario_1_results[str(gap)] = {}
            for sys_name in SYSTEMS:
                print(f"  gap={gap} sys={sys_name} len={len(stream)}", file=sys.stderr)
                g = gate if sys_name == "A_nli" else None
                res = run_stream(stream, capacity=50, system=sys_name,
                                 gate=g, record_diagnostics=True)
                m = memory_metrics(res, stream)
                diag = candidate_retrieval_diagnostics(stream, res["trace"], res["memory"]) if res["trace"] else {}

                probes = [(o, t) for o, t in zip(stream, res["trace"]) if o.is_probe]
                merged = sum(1 for _, t in probes if t["action"] == "update")
                recall_rate = round(merged / len(probes), 4) if probes else 0.0
                rvg[sys_name][str(gap)] = recall_rate

                scenario_1_results[str(gap)][sys_name] = {
                    "metrics": serialize(m),
                    "diagnostics": serialize(diag),
                    "wall_seconds": round(res["wall_seconds"], 2),
                    "probe_recall_rate": recall_rate,
                }

        scenario_1_results["recall_vs_gap"] = rvg
        results["scenario_1"] = scenario_1_results
        save_checkpoint(results)
        print(f"[Phase 17] Scenario 1 done in {time.perf_counter()-t_sc:.2f}s", file=sys.stderr)
    else:
        print("[Phase 17] Scenario 1 already present, skipping.", file=sys.stderr)

    # ─── Scenario 2: Repeated Contradictions ───
    if "scenario_2" not in results:
        t_sc = time.perf_counter()
        print("[Phase 17] Scenario 2...", file=sys.stderr)
        s2 = generate_scenario_2_repeated_contradictions()
        scenario_2_results = {}
        for sys_name in SYSTEMS:
            print(f"  Scenario 2 sys={sys_name} len={len(s2)}", file=sys.stderr)
            g = gate if sys_name == "A_nli" else None
            res = run_stream(s2, capacity=100, system=sys_name, gate=g, record_diagnostics=True)
            m = memory_metrics(res, s2)
            crt = contradiction_retention_over_time(s2, res["trace"])
            diag = candidate_retrieval_diagnostics(s2, res["trace"], res["memory"])
            scenario_2_results[sys_name] = {
                "metrics": serialize(m),
                "contradiction_retention_over_time": serialize(crt),
                "diagnostics": serialize(diag),
                "wall_seconds": round(res["wall_seconds"], 2),
            }
        results["scenario_2"] = scenario_2_results
        save_checkpoint(results)
        print(f"[Phase 17] Scenario 2 done in {time.perf_counter()-t_sc:.2f}s", file=sys.stderr)
    else:
        print("[Phase 17] Scenario 2 already present, skipping.", file=sys.stderr)

    # ─── Scenario 3: Contradiction → Recovery ───
    if "scenario_3" not in results:
        t_sc = time.perf_counter()
        print("[Phase 17] Scenario 3...", file=sys.stderr)
        s3 = generate_scenario_3_contradiction_recovery()
        scenario_3_results = {}
        for sys_name in SYSTEMS:
            print(f"  Scenario 3 sys={sys_name} len={len(s3)}", file=sys.stderr)
            g = gate if sys_name == "A_nli" else None
            res = run_stream(s3, capacity=100, system=sys_name, gate=g, record_diagnostics=True)
            m = memory_metrics(res, s3)
            bha = bounded_head_amplification(s3, res)
            diag = candidate_retrieval_diagnostics(s3, res["trace"], res["memory"])
            scenario_3_results[sys_name] = {
                "metrics": serialize(m),
                "bounded_head_amplification": serialize(bha),
                "diagnostics": serialize(diag),
                "wall_seconds": round(res["wall_seconds"], 2),
            }
        results["scenario_3"] = scenario_3_results
        save_checkpoint(results)
        print(f"[Phase 17] Scenario 3 done in {time.perf_counter()-t_sc:.2f}s", file=sys.stderr)
    else:
        print("[Phase 17] Scenario 3 already present, skipping.", file=sys.stderr)

    # ─── Scenario 4: Semantic Drift ───
    if "scenario_4" not in results:
        t_sc = time.perf_counter()
        print("[Phase 17] Scenario 4...", file=sys.stderr)
        s4 = generate_scenario_4_semantic_drift()
        scenario_4_results = {}
        for sys_name in SYSTEMS:
            print(f"  Scenario 4 sys={sys_name} len={len(s4)}", file=sys.stderr)
            g = gate if sys_name == "A_nli" else None
            res = run_stream(s4, capacity=100, system=sys_name, gate=g, record_diagnostics=True)
            m = memory_metrics(res, s4)
            diag = candidate_retrieval_diagnostics(s4, res["trace"], res["memory"])
            scenario_4_results[sys_name] = {
                "metrics": serialize(m),
                "diagnostics": serialize(diag),
                "wall_seconds": round(res["wall_seconds"], 2),
            }
        results["scenario_4"] = scenario_4_results
        save_checkpoint(results)
        print(f"[Phase 17] Scenario 4 done in {time.perf_counter()-t_sc:.2f}s", file=sys.stderr)
    else:
        print("[Phase 17] Scenario 4 already present, skipping.", file=sys.stderr)

    # ─── Scenario 5: Distractor Stress ───
    if "scenario_5" not in results:
        t_sc = time.perf_counter()
        print("[Phase 17] Scenario 5...", file=sys.stderr)
        s5_streams = generate_scenario_5_distractor_stress(distractor_counts=(50, 100, 200, 500))
        scenario_5_results = {}
        for n_dist, stream in s5_streams.items():
            scenario_5_results[str(n_dist)] = {}
            for sys_name in SYSTEMS:
                print(f"  Scenario 5 n_dist={n_dist} sys={sys_name} len={len(stream)}", file=sys.stderr)
                g = gate if sys_name == "A_nli" else None
                res = run_stream(stream, capacity=100, system=sys_name, gate=g, record_diagnostics=True)
                m = memory_metrics(res, stream)
                diag = candidate_retrieval_diagnostics(stream, res["trace"], res["memory"])
                scenario_5_results[str(n_dist)][sys_name] = {
                    "metrics": serialize(m),
                    "diagnostics": serialize(diag),
                    "wall_seconds": round(res["wall_seconds"], 2),
                }
        results["scenario_5"] = scenario_5_results
        save_checkpoint(results)
        print(f"[Phase 17] Scenario 5 done in {time.perf_counter()-t_sc:.2f}s", file=sys.stderr)
    else:
        print("[Phase 17] Scenario 5 already present, skipping.", file=sys.stderr)

    # ─── Scenario 6: Capacity Pressure ───
    if "scenario_6" not in results:
        t_sc = time.perf_counter()
        print("[Phase 17] Scenario 6...", file=sys.stderr)
        s6 = generate_scenario_6_capacity_pressure_stream()
        scenario_6_results = {}
        for cap in [1500, 100, 50]:
            scenario_6_results[str(cap)] = {}
            for sys_name in SYSTEMS:
                print(f"  Scenario 6 cap={cap} sys={sys_name} len={len(s6)}", file=sys.stderr)
                g = gate if sys_name == "A_nli" else None
                res = run_stream(s6, capacity=cap, system=sys_name, gate=g, record_diagnostics=True)
                m = memory_metrics(res, s6)
                diag = candidate_retrieval_diagnostics(s6, res["trace"], res["memory"])
                scenario_6_results[str(cap)][sys_name] = {
                    "metrics": serialize(m),
                    "diagnostics": serialize(diag),
                    "wall_seconds": round(res["wall_seconds"], 2),
                }
        results["scenario_6"] = scenario_6_results
        save_checkpoint(results)
        print(f"[Phase 17] Scenario 6 done in {time.perf_counter()-t_sc:.2f}s", file=sys.stderr)
    else:
        print("[Phase 17] Scenario 6 already present, skipping.", file=sys.stderr)

    # ─── Scenario 7: Very Long Continual ───
    if "scenario_7" not in results:
        t_sc = time.perf_counter()
        print("[Phase 17] Scenario 7 (5000 obs)...", file=sys.stderr)
        s7 = generate_scenario_7_continual_stream(n_obs=5000)
        scenario_7_results = {}
        for sys_name in SYSTEMS:
            g = gate if sys_name == "A_nli" else None
            print(f"  Scenario 7 sys={sys_name} len={len(s7)}", file=sys.stderr)
            res = run_stream(s7, capacity=200, system=sys_name, gate=g, record_diagnostics=True)
            m = memory_metrics(res, s7)
            diag = candidate_retrieval_diagnostics(s7, res["trace"], res["memory"])
            scenario_7_results[sys_name] = {
                "metrics": serialize(m),
                "diagnostics": serialize(diag),
                "wall_seconds": round(res["wall_seconds"], 2),
            }
        results["scenario_7"] = scenario_7_results
        save_checkpoint(results)
        print(f"[Phase 17] Scenario 7 done in {time.perf_counter()-t_sc:.2f}s", file=sys.stderr)
    else:
        print("[Phase 17] Scenario 7 already present, skipping.", file=sys.stderr)

    total_time = time.perf_counter() - t_start
    print(f"[Phase 17] All benchmarks completed in {total_time:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
