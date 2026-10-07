"""
Section 46/47 scaling benchmark.

Generates synthetic (subject, predicate, object) facts at increasing
volumes, absorbs them into MemoryStore, discards the original generator
state, then measures:

  - recall accuracy       (direct lookup by subject/predicate)
  - memory footprint      (checkpoint file size, fact count)
  - compression ratio     (raw input size / memory size)
  - absorb latency        (avg per-fact)
  - recall latency        (avg per-query)
  - false memory rate     (queries for subjects that were never absorbed)

Run with:  python benchmarks/scale_benchmark.py
"""

from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.memory import MemoryStore


SUBJECTS_POOL_MULTIPLIER = 1  # each subject gets exactly one fact by default

LANGUAGES = ["Python", "Rust", "Go", "TypeScript", "Java", "C++", "Ruby", "Kotlin"]
CITIES = ["Paris", "Tokyo", "Berlin", "Lagos", "Nairobi", "Toronto", "Cairo", "Oslo"]
CARS = ["a Tesla", "a Toyota", "a Ford", "a Honda", "a BMW", "a Kia"]

PREDICATE_POOLS = {
    "uses": LANGUAGES,
    "lives_in": CITIES,
    "owns": CARS,
}


def generate_facts(n: int, seed: int = 42):
    """
    Returns a list of (subject, predicate, object) triples for n distinct
    synthetic people. Each person gets exactly one randomly chosen fact,
    so recall is a clean single-answer lookup.
    """
    rng = random.Random(seed)
    facts = []
    predicates = list(PREDICATE_POOLS.keys())
    for i in range(n):
        subject = f"person_{i}"
        predicate = rng.choice(predicates)
        obj = rng.choice(PREDICATE_POOLS[predicate])
        facts.append((subject, predicate, obj))
    return facts


def run_benchmark(n: int, out_dir: str) -> dict:
    facts = generate_facts(n)

    # raw input size, measured as if these were stored as plain sentences
    raw_text = "\n".join(f"{s} {p} {o}." for s, p, o in facts)
    raw_size_bytes = len(raw_text.encode("utf-8"))

    store = MemoryStore()

    t0 = time.perf_counter()
    for subject, predicate, obj in facts:
        store.absorb(subject, predicate, obj)
    absorb_elapsed = time.perf_counter() - t0
    avg_absorb_latency_ms = (absorb_elapsed / n) * 1000

    # ground truth kept only for scoring -- NOT accessible to recall()
    ground_truth = {(s, p): o for s, p, o in facts}

    # destroy the original facts list conceptually: only query via store
    del facts

    correct = 0
    query_times = []
    for (subject, predicate), expected_obj in ground_truth.items():
        t1 = time.perf_counter()
        result = store.recall(subject, predicate)
        query_times.append(time.perf_counter() - t1)
        if result and result[0].object == expected_obj:
            correct += 1

    accuracy = correct / len(ground_truth)
    avg_recall_latency_ms = (sum(query_times) / len(query_times)) * 1000

    # false memory rate: query subjects that were never absorbed
    rng = random.Random(999)
    false_queries = [f"ghost_person_{i}" for i in range(min(200, n))]
    false_hits = 0
    for subj in false_queries:
        result = store.recall(subj, rng.choice(list(PREDICATE_POOLS.keys())))
        if result:
            false_hits += 1
    false_memory_rate = false_hits / len(false_queries)

    ckpt_path = os.path.join(out_dir, f"checkpoint_{n}.json")
    store.checkpoint(ckpt_path)
    memory_size_bytes = os.path.getsize(ckpt_path)

    compression_ratio = raw_size_bytes / memory_size_bytes if memory_size_bytes else float("inf")

    return {
        "n_facts": n,
        "raw_size_bytes": raw_size_bytes,
        "memory_size_bytes": memory_size_bytes,
        "compression_ratio": round(compression_ratio, 3),
        "recall_accuracy": round(accuracy, 4),
        "avg_absorb_latency_ms": round(avg_absorb_latency_ms, 4),
        "avg_recall_latency_ms": round(avg_recall_latency_ms, 4),
        "false_memory_rate": round(false_memory_rate, 4),
        "active_facts": store.stats()["active_facts"],
    }


def main():
    scales = [100, 1_000, 10_000]
    out_dir = os.path.join(os.path.dirname(__file__), "_checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for n in scales:
        print(f"Running scale n={n} ...")
        r = run_benchmark(n, out_dir)
        results.append(r)

    header = ["n_facts", "raw_size_bytes", "memory_size_bytes", "compression_ratio",
              "recall_accuracy", "avg_absorb_latency_ms", "avg_recall_latency_ms",
              "false_memory_rate"]
    col_w = 18
    print("\n" + "".join(h.ljust(col_w) for h in header))
    for r in results:
        print("".join(str(r[h]).ljust(col_w) for h in header))

    # Key research question flag (Section 47):
    # does memory growth stay sub-linear relative to raw input growth?
    print("\nMemory growth vs input growth (ratio of consecutive scale steps):")
    for i in range(1, len(results)):
        prev, cur = results[i - 1], results[i]
        input_growth = cur["raw_size_bytes"] / prev["raw_size_bytes"]
        memory_growth = cur["memory_size_bytes"] / prev["memory_size_bytes"]
        sub_linear = memory_growth < input_growth
        print(f"  {prev['n_facts']} -> {cur['n_facts']}: "
              f"input x{input_growth:.2f}, memory x{memory_growth:.2f}, "
              f"{'sub-linear' if sub_linear else 'NOT sub-linear'}")

    return results


if __name__ == "__main__":
    main()
