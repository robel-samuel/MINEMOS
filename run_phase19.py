"""
run_phase19.py

Comprehensive Phase 19 Benchmark Runner with Incremental Checkpoints.
Saves results to scratch/phase19_results.json after each experiment.
"""

from __future__ import annotations
import os, sys, json, time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
# Set PyTorch to utilize all 4 CPU cores
torch.set_num_threads(4)

from phase2.memory_state import MemoryState, MemorySlot
from benchmarks.phase16_experiment import NliSemanticGate
from benchmarks.phase18_experiment import DenseSemanticEncoder
from benchmarks.phase18_dataset import (
    generate_scenario_1_long_gaps,
    generate_scenario_2_safety_test,
    generate_scenario_5_capacity_pressure_stream,
    generate_scenario_6_continual_stream
)
from benchmarks.phase18_metrics import (
    evaluate_retrieval_decomposition,
    evaluate_memory_consolidation
)
from benchmarks.phase19_experiment import (
    run_phase19_stream
)
from benchmarks.phase19_metrics import evaluate_phase19_diagnostics

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "scratch", "phase19_results.json")


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
    print(f"[Phase 19 Checkpoint] Saved to {CHECKPOINT_PATH}", file=sys.stderr)


def evaluate_run(res, stream):
    m = evaluate_memory_consolidation(res, stream)
    d = evaluate_retrieval_decomposition(stream, res["trace"], res["memory"])
    diag19 = evaluate_phase19_diagnostics(stream, res["trace"], res["memory"])
    return {
        "metrics": serialize(m),
        "diagnostics": serialize(d),
        "phase19_diagnostics": serialize(diag19),
        "wall_seconds": round(res["wall_seconds"], 2),
        "nli_calls": sum(1 for tr in res["trace"] if tr.get("candidates_examined", 0) > 0)
    }


def main():
    print(f"[Phase 19] Initializing models on {torch.get_num_threads()} CPU threads...", file=sys.stderr)
    t_start = time.perf_counter()
    nli_gate = NliSemanticGate()
    dense_encoder = DenseSemanticEncoder()
    print(f"[Phase 19] Models loaded in {time.perf_counter()-t_start:.2f}s", file=sys.stderr)

    results = load_checkpoint()

    # ──────────────────────────────────────────────
    # 1. Experiment 1: Contradiction-Aware Utility (D vs D1)
    # ──────────────────────────────────────────────
    if "exp1_contra_utility" not in results:
        print("[Phase 19] Running Experiment 1: Contradiction-Aware Utility (D vs D1)...", file=sys.stderr)
        t_exp = time.perf_counter()
        stream_cap = generate_scenario_5_capacity_pressure_stream()  # 1500 obs
        exp1_res = {}
        for cap in (200, 100):
            exp1_res[str(cap)] = {}
            for sys_id in ("D", "D1"):
                print(f"  Exp 1: Capacity={cap}, System={sys_id}...", file=sys.stderr)
                res = run_phase19_stream(
                    observations=stream_cap, capacity=cap, system=sys_id,
                    nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3
                )
                exp1_res[str(cap)][sys_id] = evaluate_run(res, stream_cap)
        results["exp1_contra_utility"] = exp1_res
        save_checkpoint(results)
        print(f"[Phase 19] Exp 1 done in {time.perf_counter()-t_exp:.2f}s", file=sys.stderr)
    else:
        print("[Phase 19] Exp 1 already present, skipping.", file=sys.stderr)

    # ──────────────────────────────────────────────
    # 2. Experiment 2: Adaptive Candidate Beam Exploration
    # ──────────────────────────────────────────────
    if "exp2_adaptive_beam" not in results:
        print("[Phase 19] Running Experiment 2: Adaptive Candidate Beam Exploration...", file=sys.stderr)
        t_exp = time.perf_counter()
        stream_cap = generate_scenario_5_capacity_pressure_stream()
        exp2_res = {}

        policies = [
            ("fixed_k3", 0.0, 0.75, 0.0),
            ("tau_min_030", 0.30, 0.75, 0.10),
            ("tau_min_040_std", 0.40, 0.75, 0.10),
            ("tau_min_050", 0.50, 0.75, 0.10),
            ("margin_005", 0.40, 0.75, 0.05),
            ("margin_015", 0.40, 0.75, 0.15),
        ]

        for pol_name, t_min, t_high, d_margin in policies:
            print(f"  Exp 2: Policy={pol_name} (t_min={t_min}, d_margin={d_margin})...", file=sys.stderr)
            sys_name = "D" if pol_name == "fixed_k3" else "D2"
            res = run_phase19_stream(
                observations=stream_cap, capacity=200, system=sys_name,
                nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3,
                tau_min=t_min, tau_high=t_high, delta_margin=d_margin
            )
            exp2_res[pol_name] = evaluate_run(res, stream_cap)

        results["exp2_adaptive_beam"] = exp2_res
        save_checkpoint(results)
        print(f"[Phase 19] Exp 2 done in {time.perf_counter()-t_exp:.2f}s", file=sys.stderr)
    else:
        print("[Phase 19] Exp 2 already present, skipping.", file=sys.stderr)

    # ──────────────────────────────────────────────
    # 3. Experiment 3: Safety Stress Test
    # ──────────────────────────────────────────────
    if "exp3_safety_stress" not in results:
        print("[Phase 19] Running Experiment 3: Safety Stress Test...", file=sys.stderr)
        t_exp = time.perf_counter()
        stream_safety = generate_scenario_2_safety_test()
        exp3_res = {}
        for sys_id in ("D", "D1", "D2", "D3", "D4"):
            print(f"  Exp 3: System={sys_id}...", file=sys.stderr)
            res = run_phase19_stream(
                observations=stream_safety, capacity=100, system=sys_id,
                nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3,
                tau_min=0.40, tau_high=0.75, delta_margin=0.10
            )
            exp3_res[sys_id] = evaluate_run(res, stream_safety)
        results["exp3_safety_stress"] = exp3_res
        save_checkpoint(results)
        print(f"[Phase 19] Exp 3 done in {time.perf_counter()-t_exp:.2f}s", file=sys.stderr)
    else:
        print("[Phase 19] Exp 3 already present, skipping.", file=sys.stderr)

    # ──────────────────────────────────────────────
    # 4. Experiment 4: 5,000-Observation Continual Stream (C=200)
    # ──────────────────────────────────────────────
    if "exp4_continual_5000" not in results:
        print("[Phase 19] Running Experiment 4: 5,000-Observation Continual Stream (C=200)...", file=sys.stderr)
        t_exp = time.perf_counter()
        stream_5k = generate_scenario_6_continual_stream(n_obs=5000)
        exp4_res = {}
        for sys_id in ("D", "D1", "D2", "D3", "D4"):
            print(f"  Exp 4: 5k Stream, System={sys_id}...", file=sys.stderr)
            res = run_phase19_stream(
                observations=stream_5k, capacity=200, system=sys_id,
                nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3,
                tau_min=0.40, tau_high=0.75, delta_margin=0.10
            )
            exp4_res[sys_id] = evaluate_run(res, stream_5k)
        results["exp4_continual_5000"] = exp4_res
        save_checkpoint(results)
        print(f"[Phase 19] Exp 4 done in {time.perf_counter()-t_exp:.2f}s", file=sys.stderr)
    else:
        print("[Phase 19] Exp 4 already present, skipping.", file=sys.stderr)

    # ──────────────────────────────────────────────
    # 5. Experiment 5: Capacity Sweep (C in {1500, 200, 100, 50})
    # ──────────────────────────────────────────────
    if "exp5_capacity_sweep" not in results:
        print("[Phase 19] Running Experiment 5: Capacity Sweep (D vs D3)...", file=sys.stderr)
        t_exp = time.perf_counter()
        stream_cap = generate_scenario_5_capacity_pressure_stream()
        exp5_res = {}
        for cap in (1500, 200, 100, 50):
            exp5_res[str(cap)] = {}
            for sys_id in ("D", "D3"):
                print(f"  Exp 5: Capacity={cap}, System={sys_id}...", file=sys.stderr)
                res = run_phase19_stream(
                    observations=stream_cap, capacity=cap, system=sys_id,
                    nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3,
                    tau_min=0.40, tau_high=0.75, delta_margin=0.10
                )
                exp5_res[str(cap)][sys_id] = evaluate_run(res, stream_cap)
        results["exp5_capacity_sweep"] = exp5_res
        save_checkpoint(results)
        print(f"[Phase 19] Exp 5 done in {time.perf_counter()-t_exp:.2f}s", file=sys.stderr)
    else:
        print("[Phase 19] Exp 5 already present, skipping.", file=sys.stderr)

    # ──────────────────────────────────────────────
    # 6. Experiment 6: Long-Gap Paraphrase Recall (G in {10, 50, 100, 500})
    # ──────────────────────────────────────────────
    if "exp6_long_gap" not in results:
        results["exp6_long_gap"] = {}
    
    exp6_res = results["exp6_long_gap"]
    gap_streams = generate_scenario_1_long_gaps(gaps=(10, 50, 100, 500))
    exp6_needed = False
    for gap in (10, 50, 100, 500):
        if str(gap) not in exp6_res or "D3" not in exp6_res[str(gap)]:
            exp6_needed = True
            break

    if exp6_needed:
        print("[Phase 19] Running Experiment 6: Long-Gap Paraphrase Recall (D vs D3)...", file=sys.stderr)
        t_exp = time.perf_counter()
        for gap, stream in gap_streams.items():
            gap_str = str(gap)
            if gap_str not in exp6_res:
                exp6_res[gap_str] = {}
            for sys_id in ("D", "D3"):
                if sys_id in exp6_res[gap_str]:
                    print(f"  Exp 6: Gap={gap}, System={sys_id} already present, skipping.", file=sys.stderr)
                    continue
                print(f"  Exp 6: Gap={gap}, System={sys_id}, Len={len(stream)}...", file=sys.stderr)
                res = run_phase19_stream(
                    observations=stream, capacity=50, system=sys_id,
                    nli_gate=nli_gate, dense_encoder=dense_encoder, top_k=3,
                    tau_min=0.40, tau_high=0.75, delta_margin=0.10
                )
                r_dict = evaluate_run(res, stream)
                probes = [(o, t) for o, t in zip(stream, res["trace"]) if getattr(o, "is_probe", False)]
                merged = sum(1 for _, t in probes if t.get("action") == "update")
                r_dict["probe_recall_rate"] = round(merged / len(probes), 4) if probes else 0.0
                exp6_res[gap_str][sys_id] = r_dict
                save_checkpoint(results)
        print(f"[Phase 19] Exp 6 done in {time.perf_counter()-t_exp:.2f}s", file=sys.stderr)
    else:
        print("[Phase 19] Exp 6 already complete, skipping.", file=sys.stderr)

    # ──────────────────────────────────────────────
    # 7. Experiment 7: Complete Component Ablation Summary
    # ──────────────────────────────────────────────
    if "exp7_ablation" not in results:
        print("[Phase 19] Compiling Experiment 7: Component Ablation...", file=sys.stderr)
        # Pull from exp3 (Safety) and exp4 (5k stream) to compile full ablation table
        exp7_res = {
            "safety_stress": {
                sys_id: results["exp3_safety_stress"][sys_id]
                for sys_id in ("D", "D1", "D2", "D3", "D4")
            },
            "continual_5000": {
                sys_id: results["exp4_continual_5000"][sys_id]
                for sys_id in ("D", "D1", "D2", "D3", "D4")
            }
        }
        results["exp7_ablation"] = exp7_res
        save_checkpoint(results)
        print("[Phase 19] Exp 7 compiled and saved.", file=sys.stderr)
    else:
        print("[Phase 19] Exp 7 already present, skipping.", file=sys.stderr)

    total_time = time.perf_counter() - t_start
    print(f"[Phase 19] All Phase 19 benchmark experiments completed in {total_time:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
