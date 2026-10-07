"""
run_phase22.py

Comprehensive Phase 22 Benchmark Runner with Incremental Checkpoints:
Hybrid Retrieval (Dense + Lexical BM25 Candidate Generation).

Evaluates:
  - System D: Phase 18 Baseline (Dense only, utility eviction, w=0)
  - System D4: Phase 19 Baseline (Dense only, static anchors w=0.20)
  - System D8: Frozen Phase 20/21 Baseline (Dense only, w=0.15, lambda=0.005, k=3)
  - System H1: Hybrid Retrieval System (Dense + BM25 + D8 anchors, k_fusion=5)
  - Ablations (Exp 5): H_dense_only, H_bm25_only, H_dense_bm25_no_anchor, H1

Saves results incrementally to scratch/phase22_results.json.
Experiment 6 (Held-Out) uses seed=8888 and is evaluated strictly once without tuning.
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
from benchmarks.phase18_metrics import (
    evaluate_retrieval_decomposition,
    evaluate_memory_consolidation,
)
from benchmarks.phase22_dataset import (
    generate_exp1_vocab_shift,
    generate_exp2_dense_distractors,
    generate_exp3_adversarial_contradictions,
    generate_exp4_long_horizon,
    generate_exp5_ablation_stream,
    generate_exp6_held_out_benchmark,
    SEED_P22,
)
from benchmarks.phase22_experiment import (
    run_phase22_stream,
    FROZEN_W,
    FROZEN_LAMBDA,
    FROZEN_K,
    TAU_MIN,
    TAU_HIGH,
    DELTA_MARGIN,
    K_MAX,
    TAU_ANCHOR,
)
from benchmarks.phase22_metrics import evaluate_phase22_diagnostics

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "scratch", "phase22_results.json")

FROZEN_CONFIG = {
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
    "MATCH_THRESHOLD": 0.75,
    "seed_p22": SEED_P22,
}

MAIN_SYSTEMS = ["D", "D4", "D8", "H1"]


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
    print(f"[Phase 22 Checkpoint] Saved to {CHECKPOINT_PATH}", file=sys.stderr)


def evaluate_run(res: dict, stream: list) -> dict:
    m = evaluate_memory_consolidation(res, stream)
    d = evaluate_retrieval_decomposition(stream, res["trace"], res["memory"])
    diag22 = evaluate_phase22_diagnostics(res, stream, res["memory"])
    return {
        "metrics": serialize(m),
        "diagnostics": serialize(d),
        "phase22_diagnostics": serialize(diag22),
        "wall_seconds": round(res["wall_seconds"], 2),
        "nli_calls": res["nli_calls"],
        "memory_size": len(res["memory"].slots),
        "anchor_count": int(res.get("total_anchors_retained", 0)),
        "avg_anchor_age": res.get("avg_anchor_age", 0.0),
        "avg_applied_penalty": res.get("avg_applied_penalty", 0.0),
        "time_retrieval_dense": res.get("time_retrieval_dense", 0.0),
        "time_retrieval_bm25": res.get("time_retrieval_bm25", 0.0),
    }


def probe_recall(stream, trace) -> float:
    probes = [(o, t) for o, t in zip(stream, trace) if getattr(o, "is_probe", False)]
    if not probes:
        return 0.0
    merged = sum(1 for _, t in probes if t.get("action") == "update")
    return round(merged / len(probes), 4)


def main():
    print("[Phase 22] Initializing models on 4 CPU threads...", file=sys.stderr)
    t_start = time.perf_counter()
    nli_gate = NliSemanticGate()
    dense_encoder = DenseSemanticEncoder()
    print(f"[Phase 22] Models loaded in {time.perf_counter() - t_start:.2f}s", file=sys.stderr)

    results = load_checkpoint()

    if "frozen_config" not in results:
        results["frozen_config"] = FROZEN_CONFIG
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    # Experiment 1: Vocabulary Shift (Hard Paraphrases)
    # ──────────────────────────────────────────────
    if "exp1_vocab_shift" not in results:
        results["exp1_vocab_shift"] = {}
    exp1_res = results["exp1_vocab_shift"]

    stream_exp1 = generate_exp1_vocab_shift(inter_gap=10, seed=SEED_P22)
    print(f"[Phase 22] Exp 1 (Vocab Shift): {len(stream_exp1)} obs, C=50", file=sys.stderr)

    for sys_id in MAIN_SYSTEMS:
        if sys_id in exp1_res:
            print(f"[Phase 22] Exp 1 System={sys_id} already checkpointed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 22] Exp 1: System={sys_id}...", file=sys.stderr)
        res = run_phase22_stream(
            observations=stream_exp1, capacity=50, system=sys_id,
            nli_gate=nli_gate, dense_encoder=dense_encoder,
            top_k_dense=3, top_k_bm25=3, k_fusion=5
        )
        entry = evaluate_run(res, stream_exp1)
        entry["probe_recall"] = probe_recall(stream_exp1, res["trace"])
        exp1_res[sys_id] = entry
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    # Experiment 2: Dense Semantic Distractors
    # ──────────────────────────────────────────────
    if "exp2_dense_distractors" not in results:
        results["exp2_dense_distractors"] = {}
    exp2_res = results["exp2_dense_distractors"]

    stream_exp2 = generate_exp2_dense_distractors(n_targets=10, distractors_per_target=15, seed=SEED_P22)
    print(f"[Phase 22] Exp 2 (Dense Distractors): {len(stream_exp2)} obs, C=50", file=sys.stderr)

    for sys_id in MAIN_SYSTEMS:
        if sys_id in exp2_res:
            print(f"[Phase 22] Exp 2 System={sys_id} already checkpointed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 22] Exp 2: System={sys_id}...", file=sys.stderr)
        res = run_phase22_stream(
            observations=stream_exp2, capacity=50, system=sys_id,
            nli_gate=nli_gate, dense_encoder=dense_encoder,
            top_k_dense=3, top_k_bm25=3, k_fusion=5
        )
        entry = evaluate_run(res, stream_exp2)
        entry["probe_recall"] = probe_recall(stream_exp2, res["trace"])
        exp2_res[sys_id] = entry
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    # Experiment 3: Adversarial Contradictions (High Lexical Overlap)
    # ──────────────────────────────────────────────
    if "exp3_adversarial_contradictions" not in results:
        results["exp3_adversarial_contradictions"] = {}
    exp3_res = results["exp3_adversarial_contradictions"]

    stream_exp3 = generate_exp3_adversarial_contradictions(inter_gap=10, seed=SEED_P22)
    print(f"[Phase 22] Exp 3 (Adversarial Contradictions): {len(stream_exp3)} obs, C=50", file=sys.stderr)

    for sys_id in MAIN_SYSTEMS:
        if sys_id in exp3_res:
            print(f"[Phase 22] Exp 3 System={sys_id} already checkpointed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 22] Exp 3: System={sys_id}...", file=sys.stderr)
        res = run_phase22_stream(
            observations=stream_exp3, capacity=50, system=sys_id,
            nli_gate=nli_gate, dense_encoder=dense_encoder,
            top_k_dense=3, top_k_bm25=3, k_fusion=5
        )
        entry = evaluate_run(res, stream_exp3)
        entry["probe_recall"] = probe_recall(stream_exp3, res["trace"])
        exp3_res[sys_id] = entry
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    # Experiment 4: Long-Horizon Streams (D8 vs H1 at 5k and 10k)
    # ──────────────────────────────────────────────
    if "exp4_long_horizon" not in results:
        results["exp4_long_horizon"] = {}
    exp4_res = results["exp4_long_horizon"]

    for n_obs in [5000, 10000]:
        n_str = str(n_obs)
        if n_str not in exp4_res:
            exp4_res[n_str] = {}

        for sys_id in ["D8", "H1"]:
            if sys_id in exp4_res[n_str]:
                print(f"[Phase 22] Exp 4 n={n_obs} System={sys_id} already checkpointed, skipping.", file=sys.stderr)
                continue
            print(f"[Phase 22] Exp 4 (Long-Horizon): n={n_obs}, System={sys_id}...", file=sys.stderr)
            stream_n = generate_exp4_long_horizon(n_obs=n_obs, seed=SEED_P22)
            res = run_phase22_stream(
                observations=stream_n, capacity=200, system=sys_id,
                nli_gate=nli_gate, dense_encoder=dense_encoder,
                top_k_dense=3, top_k_bm25=3, k_fusion=5
            )
            entry = evaluate_run(res, stream_n)
            entry["probe_recall"] = probe_recall(stream_n, res["trace"])
            exp4_res[n_str][sys_id] = entry
            save_checkpoint(results)

    # ──────────────────────────────────────────────
    # Experiment 5: Component Ablations
    # A: Dense only (= D8)
    # B: BM25 only (H_bm25_only)
    # C: Dense + BM25 without anchors (H_dense_bm25_no_anchor)
    # D: Dense + BM25 + D8 anchors (H1)
    # ──────────────────────────────────────────────
    if "exp5_ablation" not in results:
        results["exp5_ablation"] = {}
    exp5_res = results["exp5_ablation"]

    stream_exp5 = generate_exp5_ablation_stream(n_concepts=15, inter_gap=10, seed=SEED_P22)
    print(f"[Phase 22] Exp 5 (Ablation Stream): {len(stream_exp5)} obs, C=50", file=sys.stderr)

    ABLATION_SYSTEMS = [
        ("A_dense_only", "D8"),
        ("B_bm25_only", "H_bm25_only"),
        ("C_dense_bm25_no_anchor", "H_dense_bm25_no_anchor"),
        ("D_full_hybrid", "H1")
    ]

    for abl_label, sys_type in ABLATION_SYSTEMS:
        if abl_label in exp5_res:
            print(f"[Phase 22] Exp 5 {abl_label} already checkpointed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 22] Exp 5: Variant={abl_label} ({sys_type})...", file=sys.stderr)
        res = run_phase22_stream(
            observations=stream_exp5, capacity=50, system=sys_type,
            nli_gate=nli_gate, dense_encoder=dense_encoder,
            top_k_dense=3, top_k_bm25=3, k_fusion=5
        )
        entry = evaluate_run(res, stream_exp5)
        entry["probe_recall"] = probe_recall(stream_exp5, res["trace"])
        exp5_res[abl_label] = entry
        save_checkpoint(results)

    # ──────────────────────────────────────────────
    # Experiment 6: Sealed Held-Out Evaluation (Seed=8888)
    # ──────────────────────────────────────────────
    if "exp6_held_out" not in results:
        results["exp6_held_out"] = {
            "_metadata": {
                "seed": 8888,
                "n_obs": 2500,
                "capacity": 100,
                "warning": "SEALED — evaluated strictly once without parameter tuning."
            }
        }
    exp6_res = results["exp6_held_out"]

    stream_exp6 = generate_exp6_held_out_benchmark(n_obs=2500, seed=8888)
    print(f"[Phase 22] Exp 6 (HELD-OUT, seed=8888): {len(stream_exp6)} obs, C=100", file=sys.stderr)

    for sys_id in MAIN_SYSTEMS:
        if sys_id in exp6_res:
            print(f"[Phase 22] Exp 6 System={sys_id} already checkpointed, skipping.", file=sys.stderr)
            continue
        print(f"[Phase 22] Exp 6 (HELD-OUT): System={sys_id}...", file=sys.stderr)
        res = run_phase22_stream(
            observations=stream_exp6, capacity=100, system=sys_id,
            nli_gate=nli_gate, dense_encoder=dense_encoder,
            top_k_dense=3, top_k_bm25=3, k_fusion=5
        )
        entry = evaluate_run(res, stream_exp6)
        entry["probe_recall"] = probe_recall(stream_exp6, res["trace"])
        exp6_res[sys_id] = entry
        save_checkpoint(results)

    total_time = time.perf_counter() - t_start
    print(f"\n[Phase 22] ALL EXPERIMENTS COMPLETE in {total_time:.2f}s", file=sys.stderr)
    print(f"[Phase 22] Results saved to: {CHECKPOINT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
