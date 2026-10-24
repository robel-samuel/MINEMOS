"""
benchmarks/phase27_experiment.py

Experimental orchestration for MINEMOS Phase 27:
Adaptive Memory Retrieval Under Long-Horizon Interference.

Executes:
  - Experiment A: 5 Retrieval Methods across 4 Interference Conditions (A, B, C, D) over multiple seeds
  - Experiment B: Long-Horizon Scaling across stream lengths N in {25, 50, 100}
  - Experiment C: 6-Way Component Ablation Study
  - Experiment D: Frozen Held-Out Benchmark Generalization
  - Experiment E: Computational Efficiency and Latency Profiling
"""

from __future__ import annotations
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from phase2.black_hole_memory import BlackHoleMemory
from benchmarks.phase26_verification import Phase26Verifier
from benchmarks.phase27_dataset import (
    Phase27StreamGenerator,
    StreamObservation,
    RetrievalQuery,
    verify_phase27_dataset_integrity
)
from benchmarks.phase27_retrieval import Phase27Retriever
from benchmarks.phase27_metrics import compute_retrieval_metrics


SEEDS = [42, 123, 999]


def setup_memory_stream(
    observations: List[StreamObservation],
    capacity: int = 150
) -> Tuple[BlackHoleMemory, Phase27Retriever]:
    """
    Initializes a BlackHoleMemory engine with calibrated Phase 26 verifier,
    absorbs the stream of observations, and returns the engine + retriever.
    """
    # Use calibrated verifier parameters established in Phase 26
    verifier = Phase26Verifier(cal_tau_entail=0.50, cal_asym_margin=0.10, fallback_tau_dense=0.88)
    mem = BlackHoleMemory(
        capacity=capacity,
        retention_policy="tiered",
        verification_mode="calibrated",
        entail_threshold=0.50,
        contra_threshold=0.40,
        consolidation_mode="hybrid",
        verifier=verifier
    )

    for obs in observations:
        mem.absorb(obs.candidate, timestamp=obs.candidate.timestamp)

    retriever = Phase27Retriever(memory_engine=mem)
    return mem, retriever


def evaluate_retrieval_method(
    retriever: Phase27Retriever,
    queries: List[RetrievalQuery],
    method: str
) -> Dict[str, float]:
    """
    Runs a specified retrieval method across a list of queries and computes metrics.
    """
    query_results = []
    latencies_ms = []

    for q in queries:
        t0 = time.perf_counter()
        res = retriever.retrieve(
            method=method,
            query_text=q.query_text,
            top_k=5,
            as_of=q.as_of,
            query_time=q.query_time
        )
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        query_results.append(res)

    return compute_retrieval_metrics(queries=queries, query_results=query_results, latencies_ms=latencies_ms)


# ─────────────────────────────────────────────────────────────
# Experiment A: Method Comparison Across Interference Conditions
# ─────────────────────────────────────────────────────────────

def run_experiment_a_conditions(
    seeds: List[int] = SEEDS,
    length: int = 50
) -> Dict[str, Any]:
    """
    Compares 5 retrieval methods across 4 interference conditions (A, B, C, D)
    averaged over multiple seeds.
    """
    print("\n=======================================================")
    print("  EXPERIMENT A: Retrieval Methods Across Interference Conditions")
    print("=======================================================")

    conditions = ["A_low_interference", "B_semantic_interference", "C_temporal_interference", "D_combined_interference"]
    methods = ["dense_baseline", "bm25_only", "hybrid_relevance", "temporal_aware", "adaptive"]

    results_by_cond: Dict[str, Dict[str, Any]] = {}

    for cond in conditions:
        print(f"\n--- Testing Condition: {cond} ---")
        results_by_cond[cond] = {}

        for method in methods:
            seed_metrics = []
            for seed in seeds:
                gen = Phase27StreamGenerator(seed=seed)
                obs, queries = gen.generate_stream(condition=cond, length=length, is_held_out=False)
                _, retriever = setup_memory_stream(obs, capacity=100)
                m = evaluate_retrieval_method(retriever, queries, method)
                seed_metrics.append(m)

            # Aggregate over seeds
            agg: Dict[str, Any] = {}
            metric_keys = ["recall_at_1", "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5",
                           "exact_evidence_accuracy", "temporal_correctness_rate",
                           "contradiction_intrusion_rate", "avg_latency_ms"]
            for k in metric_keys:
                vals = [sm[k] for sm in seed_metrics]
                agg[k] = round(float(np.mean(vals)), 4)
                agg[f"{k}_std"] = round(float(np.std(vals)), 4)

            results_by_cond[cond][method] = agg
            print(f"  [{method.upper():<20}] R@1: {agg['recall_at_1']:.3f} | MRR: {agg['mrr']:.3f} | "
                  f"ExactAcc: {agg['exact_evidence_accuracy']:.3f} | TempCorr: {agg['temporal_correctness_rate']:.3f} | "
                  f"ContraIntrusion: {agg['contradiction_intrusion_rate']:.3f} | Latency: {agg['avg_latency_ms']:.1f}ms")

    return results_by_cond


# ─────────────────────────────────────────────────────────────
# Experiment B: Long-Horizon Interference Scaling
# ─────────────────────────────────────────────────────────────

def run_experiment_b_horizon_scaling(
    lengths: List[int] = [25, 50, 100],
    seeds: List[int] = SEEDS
) -> Dict[str, Any]:
    """
    Evaluates retrieval degradation as interference horizon increases.
    """
    print("\n=======================================================")
    print("  EXPERIMENT B: Long-Horizon Interference Scaling (N=25, 50, 100)")
    print("=======================================================")

    scaling_results: Dict[str, Any] = {}

    for length in lengths:
        print(f"\n--- Stream Horizon N = {length} observations ---")
        scaling_results[str(length)] = {}

        for method in ["dense_baseline", "adaptive"]:
            seed_metrics = []
            for seed in seeds:
                gen = Phase27StreamGenerator(seed=seed)
                obs, queries = gen.generate_stream(condition="D_combined_interference", length=length, is_held_out=False)
                _, retriever = setup_memory_stream(obs, capacity=120)
                m = evaluate_retrieval_method(retriever, queries, method)
                seed_metrics.append(m)

            agg = {}
            for k in ["recall_at_1", "mrr", "ndcg_at_5", "exact_evidence_accuracy", "temporal_correctness_rate", "contradiction_intrusion_rate", "avg_latency_ms"]:
                vals = [sm[k] for sm in seed_metrics]
                agg[k] = round(float(np.mean(vals)), 4)
                agg[f"{k}_std"] = round(float(np.std(vals)), 4)

            scaling_results[str(length)][method] = agg
            print(f"  [N={length} | {method.upper():<15}] R@1: {agg['recall_at_1']:.3f} | MRR: {agg['mrr']:.3f} | "
                  f"TempCorr: {agg['temporal_correctness_rate']:.3f} | ContraIntrusion: {agg['contradiction_intrusion_rate']:.3f}")

    return scaling_results


# ─────────────────────────────────────────────────────────────
# Experiment C: 6-Way Controlled Ablation Study
# ─────────────────────────────────────────────────────────────

def run_experiment_c_ablations(
    seeds: List[int] = SEEDS,
    length: int = 50
) -> Dict[str, Any]:
    """
    Evaluates 6 controlled retrieval configurations on Condition D:
      1. Dense Only (Baseline)
      2. BM25 Only
      3. Hybrid (Dense + BM25)
      4. Hybrid + Temporal Validity Weighting
      5. Hybrid + Contradiction Anchor Repulsion
      6. Full Adaptive (Hybrid + Temporal + Contradiction + Corroboration Boost)
    """
    print("\n=======================================================")
    print("  EXPERIMENT C: 6-Way Component Ablation Study")
    print("=======================================================")

    ablation_methods = [
        ("dense_baseline", "1. Dense Only (Baseline)"),
        ("bm25_only", "2. BM25 Only"),
        ("hybrid_relevance", "3. Hybrid (Dense + BM25)"),
        ("hybrid_temporal_only", "4. Hybrid + Temporal Validity"),
        ("hybrid_contra_only", "5. Hybrid + Contradiction Anchor"),
        ("adaptive", "6. Full Adaptive Multi-Factor")
    ]

    ablation_results: Dict[str, Any] = {}

    for method_key, method_name in ablation_methods:
        seed_metrics = []
        for seed in seeds:
            gen = Phase27StreamGenerator(seed=seed)
            obs, queries = gen.generate_stream(condition="D_combined_interference", length=length, is_held_out=False)
            _, retriever = setup_memory_stream(obs, capacity=100)
            m = evaluate_retrieval_method(retriever, queries, method_key)
            seed_metrics.append(m)

        agg = {}
        for k in ["recall_at_1", "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5",
                  "exact_evidence_accuracy", "temporal_correctness_rate",
                  "contradiction_intrusion_rate", "avg_latency_ms"]:
            vals = [sm[k] for sm in seed_metrics]
            agg[k] = round(float(np.mean(vals)), 4)
            agg[f"{k}_std"] = round(float(np.std(vals)), 4)

        ablation_results[method_key] = {
            "name": method_name,
            "metrics": agg
        }
        print(f"  [{method_name:<38}] R@1: {agg['recall_at_1']:.3f} | MRR: {agg['mrr']:.3f} | "
              f"ExactAcc: {agg['exact_evidence_accuracy']:.3f} | TempCorr: {agg['temporal_correctness_rate']:.3f} | "
              f"ContraIntrusion: {agg['contradiction_intrusion_rate']:.3f}")

    return ablation_results


# ─────────────────────────────────────────────────────────────
# Experiment D: Frozen Held-Out Generalization Evaluation
# ─────────────────────────────────────────────────────────────

def run_experiment_d_held_out(
    seeds: List[int] = SEEDS,
    length: int = 50
) -> Dict[str, Any]:
    """
    Evaluates retrieval methods on untouched held-out entities and relations.
    """
    print("\n=======================================================")
    print("  EXPERIMENT D: Frozen Held-Out Benchmark Generalization")
    print("=======================================================")

    held_out_results: Dict[str, Any] = {}
    methods = ["dense_baseline", "bm25_only", "hybrid_relevance", "temporal_aware", "adaptive"]

    for method in methods:
        seed_metrics = []
        for seed in seeds:
            gen = Phase27StreamGenerator(seed=seed)
            obs, queries = gen.generate_stream(condition="D_combined_interference", length=length, is_held_out=True)
            _, retriever = setup_memory_stream(obs, capacity=100)
            m = evaluate_retrieval_method(retriever, queries, method)
            seed_metrics.append(m)

        agg = {}
        for k in ["recall_at_1", "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5",
                  "exact_evidence_accuracy", "temporal_correctness_rate",
                  "contradiction_intrusion_rate", "avg_latency_ms"]:
            vals = [sm[k] for sm in seed_metrics]
            agg[k] = round(float(np.mean(vals)), 4)
            agg[f"{k}_std"] = round(float(np.std(vals)), 4)

        held_out_results[method] = agg
        print(f"  [HELD-OUT | {method.upper():<18}] R@1: {agg['recall_at_1']:.3f} | MRR: {agg['mrr']:.3f} | "
              f"ExactAcc: {agg['exact_evidence_accuracy']:.3f} | TempCorr: {agg['temporal_correctness_rate']:.3f} | "
              f"ContraIntrusion: {agg['contradiction_intrusion_rate']:.3f}")

    return held_out_results


# ─────────────────────────────────────────────────────────────
# Experiment E: Computational Complexity & Latency Profiling
# ─────────────────────────────────────────────────────────────

def run_experiment_e_complexity(
    lengths: List[int] = [25, 50, 100, 150]
) -> Dict[str, Any]:
    """
    Measures latency scaling and memory footprints.
    """
    print("\n=======================================================")
    print("  EXPERIMENT E: Computational Efficiency & Latency Profiling")
    print("=======================================================")

    gen = Phase27StreamGenerator(seed=42)
    complexity_records = []

    for length in lengths:
        obs, queries = gen.generate_stream(condition="D_combined_interference", length=length, is_held_out=False)
        mem, retriever = setup_memory_stream(obs, capacity=max(length + 10, 160))
        st = mem.status()

        # Measure latency over 20 repeated runs
        baseline_latencies = []
        adaptive_latencies = []

        for _ in range(5):
            for q in queries:
                t0 = time.perf_counter()
                retriever.retrieve_dense_baseline(q.query_text)
                baseline_latencies.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                retriever.retrieve_adaptive(q.query_text)
                adaptive_latencies.append((time.perf_counter() - t0) * 1000.0)

        record = {
            "stream_length": length,
            "total_slots": st["total_slots"],
            "footprint_bytes": st["footprint_bytes"],
            "baseline_mean_ms": round(float(np.mean(baseline_latencies)), 2),
            "baseline_p95_ms": round(float(np.percentile(baseline_latencies, 95)), 2),
            "adaptive_mean_ms": round(float(np.mean(adaptive_latencies)), 2),
            "adaptive_p95_ms": round(float(np.percentile(adaptive_latencies, 95)), 2)
        }
        complexity_records.append(record)
        print(f"  N={length:<3} | Slots={st['total_slots']} | Footprint={st['footprint_bytes']:,} B | "
              f"Base: {record['baseline_mean_ms']}ms (p95: {record['baseline_p95_ms']}ms) | "
              f"Adaptive: {record['adaptive_mean_ms']}ms (p95: {record['adaptive_p95_ms']}ms)")

    return {"scaling_profile": complexity_records}
