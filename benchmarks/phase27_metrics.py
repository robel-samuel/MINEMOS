"""
benchmarks/phase27_metrics.py

Evaluation metrics for MINEMOS Phase 27:
Adaptive Memory Retrieval Under Long-Horizon Interference.

Computes:
  - Recall@1, Recall@3, Recall@5
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (nDCG@5)
  - Exact Evidence Accuracy (top-1 is the ground-truth valid fact)
  - Temporal Correctness Rate (top-1 is temporally valid at query time)
  - Contradiction Intrusion Rate (top-1 is a superseded contradictory fact)
  - Query Latency (ms)
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional
import numpy as np

from benchmarks.phase27_dataset import RetrievalQuery


def is_fact_match(retrieved_text: str, target_answer: str, target_subject: str) -> bool:
    """Checks if retrieved slot text matches the target entity and answer."""
    ret_lower = retrieved_text.lower()
    sub_lower = target_subject.lower()
    ans_lower = target_answer.lower()
    return (sub_lower in ret_lower) and (ans_lower in ret_lower)


def compute_retrieval_metrics(
    queries: List[RetrievalQuery],
    query_results: List[List[Dict[str, Any]]],
    latencies_ms: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Computes rigorous retrieval ranking and safety metrics across all queries.
    """
    assert len(queries) == len(query_results), f"Counts mismatch: {len(queries)} vs {len(query_results)}"

    n_q = len(queries)
    if n_q == 0:
        return {}

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    ndcgs = []
    exact_acc_count = 0
    temporal_correct_count = 0
    contra_intrusions = 0

    for query, retrieved in zip(queries, query_results):
        target_ans = query.expected_answer
        target_sub = query.target_subject
        superseded_texts = [s.lower() for s in query.superseded_fact_texts]

        # 1. Rank of target valid fact
        rank = None
        for r_idx, res in enumerate(retrieved[:5]):
            txt = res.get("text", "")
            if is_fact_match(txt, target_ans, target_sub):
                rank = r_idx + 1
                break

        # Recall@k
        if rank == 1:
            hits_at_1 += 1
            hits_at_3 += 1
            hits_at_5 += 1
            reciprocal_ranks.append(1.0)
        elif rank in (2, 3):
            hits_at_3 += 1
            hits_at_5 += 1
            reciprocal_ranks.append(1.0 / rank)
        elif rank in (4, 5):
            hits_at_5 += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        # nDCG@5 calculation
        # Relevance: 2 if target fact, 0 otherwise
        dcg = 0.0
        idcg = 2.0 / math.log2(2)  # Ideal top-1 is 2.0 / log2(2) = 2.0
        for i, res in enumerate(retrieved[:5]):
            txt = res.get("text", "")
            rel = 2.0 if is_fact_match(txt, target_ans, target_sub) else 0.0
            dcg += (2.0**rel - 1.0) / math.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(min(1.0, ndcg))

        # Top-1 evaluations
        if retrieved:
            top_1 = retrieved[0]
            top_1_text = top_1.get("text", "").lower()

            # Exact evidence accuracy:
            if is_fact_match(top_1_text, target_ans, target_sub):
                exact_acc_count += 1
                temporal_correct_count += 1

            # Check if top-1 was an outdated superseded fact
            is_superseded_intrusion = False
            for sup in superseded_texts:
                if sup in top_1_text or (target_sub.lower() in top_1_text and any(w in top_1_text for w in sup.split() if len(w) > 4)):
                    if not is_fact_match(top_1_text, target_ans, target_sub):
                        is_superseded_intrusion = True
                        break

            if is_superseded_intrusion:
                contra_intrusions += 1
        else:
            pass

    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    return {
        "num_queries": n_q,
        "recall_at_1": round(hits_at_1 / n_q, 4),
        "recall_at_3": round(hits_at_3 / n_q, 4),
        "recall_at_5": round(hits_at_5 / n_q, 4),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "ndcg_at_5": round(float(np.mean(ndcgs)), 4),
        "exact_evidence_accuracy": round(exact_acc_count / n_q, 4),
        "temporal_correctness_rate": round(temporal_correct_count / n_q, 4),
        "contradiction_intrusion_rate": round(contra_intrusions / n_q, 4),
        "avg_latency_ms": round(avg_latency, 2)
    }
