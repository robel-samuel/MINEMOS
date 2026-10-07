"""
benchmarks/memory_scaling.py

TEST A — distinct facts at increasing scale and capacity.
ABLATION — Baseline A (no update mechanism, i.e. always insert, never
blend/reinforce -- equivalent to disabling the merge path entirely) vs
Baseline B (the full similarity + novelty + update mechanism).

Parameters are fixed here, before any run, per the experiment's
pre-registration requirement:
  DIM = 64 (structured_candidate.py)
  MATCH_THRESHOLD = 0.75 (memory_update.py)
  eviction policy = FIFO on oldest created_at
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate, encode_full
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb, MATCH_THRESHOLD
from phase2.recall import recall, decode_object
from phase2.checkpoint import save_checkpoint


def generate_distinct_facts(n: int):
    """Person_i -> likes -> Object_i, i.e. maximally distinct (subject,
    object) pairs sharing only the predicate 'likes'."""
    return [Candidate(f"person_{i:06d}", "likes", f"object_{i:06d}", timestamp=i)
            for i in range(n)]


def run_test_a(n_facts: int, capacity: int, out_dir: str, ablation: str = "full",
               dim: int = None, compute_bands: bool = False) -> dict:
    """
    ablation="full"    -> normal absorb (similarity+novelty+update)
    ablation="no_update" -> Baseline A: every candidate always creates/
        overwrites a slot without any similarity-based merge check at
        all -- i.e. the mechanism this experiment is trying to validate
        is switched off, to see whether it does anything measurable.
    dim=None -> use structured_candidate.DIM (64, the historical default).
    compute_bands=True -> also compute early/middle/recent/random exact
        recall (Phase 2A required metric), at extra cost -- off by
        default so the full 6-config x 7-D scaling sweep stays fast.
    """
    from phase2.structured_candidate import DIM as _DEFAULT_DIM
    dim = dim or _DEFAULT_DIM
    facts = generate_distinct_facts(n_facts)
    vocabulary = {f"object_{i:06d}": encode_full(Candidate("_", "_", f"object_{i:06d}"), dim=dim)
                  for i in range(n_facts)}

    mem = MemoryState(capacity=capacity, dim=dim)
    t0 = time.perf_counter()
    for c in facts:
        if ablation == "no_update":
            # Baseline A: bypass addressing/merge entirely.
            x_t = encode_full(c, dim=dim)
            if mem.is_full():
                mem.evict_oldest()
            mem.add(x_t, x_t.copy(), c.confidence, c.timestamp,
                     debug_subject=c.subject, debug_predicate=c.predicate)
        else:
            absorb(mem, c)
    absorb_elapsed = time.perf_counter() - t0
    avg_absorb_latency_ms = (absorb_elapsed / n_facts) * 1000

    # remove original facts list from further use -- query only via memory
    del facts

    correct_exact = 0
    correct_fuzzy = 0
    query_times = []
    n_queries = min(n_facts, 500)  # cap queries for speed at large n
    for i in range(n_queries):
        subject = f"person_{i:06d}"
        expected_obj = f"object_{i:06d}"
        t1 = time.perf_counter()
        r = recall(mem, subject, "likes")
        query_times.append(time.perf_counter() - t1)
        if not r["found"]:
            continue
        decoded = decode_object(r["slot"].value, vocabulary)
        if decoded == expected_obj:
            correct_exact += 1
            correct_fuzzy += 1
        elif decoded is not None:
            correct_fuzzy += 1  # found *something* in the vocabulary, just wrong one

    recall_accuracy = correct_fuzzy / n_queries
    exact_recall_accuracy = correct_exact / n_queries
    avg_recall_latency_ms = (sum(query_times) / len(query_times)) * 1000

    # false-memory: query subjects that were NEVER absorbed
    false_hits = 0
    n_false_queries = min(100, n_facts)
    for i in range(n_false_queries):
        r = recall(mem, f"ghost_person_{i:06d}", "likes")
        # "found" will always be True if memory is non-empty (argmax always
        # returns something) -- the honest false-memory test is whether the
        # returned similarity is spuriously high despite being a query for
        # something never stored.
        if r["found"] and r["similarity"] >= MATCH_THRESHOLD:
            false_hits += 1
    false_memory_rate = false_hits / n_false_queries

    ckpt_path = os.path.join(out_dir, f"scaling_{ablation}_{n_facts}_{capacity}_{dim}.npz")
    save_checkpoint(mem, ckpt_path)
    checkpoint_bytes = os.path.getsize(ckpt_path)
    fp = mem.footprint_bytes()

    raw_input_bytes = sum(len(f"{c.subject} {c.predicate} {c.object}".encode())
                           for c in generate_distinct_facts(n_facts))

    result = {
        "ablation": ablation,
        "dim": dim,
        "n_facts": n_facts,
        "capacity": capacity,
        "n_slots_used": fp["n_slots"],
        "slot_utilization": fp["n_slots"] / capacity,
        "raw_input_bytes": raw_input_bytes,
        "payload_bytes": fp["payload_bytes"],
        "total_footprint_bytes": fp["total_bytes"],
        "checkpoint_bytes": checkpoint_bytes,
        "bytes_per_retained_fact": round(fp["total_bytes"] / fp["n_slots"], 2) if fp["n_slots"] else None,
        "recall_accuracy": round(recall_accuracy, 4),
        "exact_recall_accuracy": round(exact_recall_accuracy, 4),
        "false_memory_rate": round(false_memory_rate, 4),
        "avg_absorb_latency_ms": round(avg_absorb_latency_ms, 5),
        "avg_recall_latency_ms": round(avg_recall_latency_ms, 5),
        "n_queries": n_queries,
    }

    if compute_bands:
        def band_exact_recall(indices):
            correct = 0
            for i in indices:
                r = recall(mem, f"person_{i:06d}", "likes")
                if r["found"]:
                    decoded = decode_object(r["slot"].value, vocabulary)
                    if decoded == f"object_{i:06d}":
                        correct += 1
            return round(correct / len(indices), 4) if indices else None

        import random as _random
        rng = _random.Random(3)
        band_size = min(50, n_facts)
        early = list(range(0, band_size))
        mid_start = max(0, n_facts // 2 - band_size // 2)
        middle = list(range(mid_start, min(n_facts, mid_start + band_size)))
        recent = list(range(max(0, n_facts - band_size), n_facts))
        random_sample = rng.sample(range(n_facts), band_size)

        result["early_recall"] = band_exact_recall(early)
        result["middle_recall"] = band_exact_recall(middle)
        result["recent_recall"] = band_exact_recall(recent)
        result["random_recall"] = band_exact_recall(random_sample)

    return result


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "_phase2_checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 100)
    print(f"TEST A: distinct facts, scaling (MATCH_THRESHOLD={MATCH_THRESHOLD})")
    print("=" * 100)
    configs = [
        (100, 10), (100, 100),
        (1000, 100), (1000, 1000),
        (10000, 1000), (10000, 10000),
    ]
    results = []
    for n_facts, capacity in configs:
        r = run_test_a(n_facts, capacity, out_dir, ablation="full")
        results.append(r)
        print(f"n={n_facts:6d} cap={capacity:6d}  slots_used={r['n_slots_used']:6d} "
              f"util={r['slot_utilization']:.2f}  exact_recall={r['exact_recall_accuracy']:.3f} "
              f"bytes/fact={r['bytes_per_retained_fact']} "
              f"false_mem={r['false_memory_rate']:.3f}")

    print()
    print("=" * 100)
    print("ABLATION: Baseline A (no update mechanism) vs Baseline B (full mechanism)")
    print("=" * 100)
    ablation_results = []
    for n_facts, capacity in [(1000, 1000), (1000, 100)]:
        r_full = run_test_a(n_facts, capacity, out_dir, ablation="full")
        r_none = run_test_a(n_facts, capacity, out_dir, ablation="no_update")
        ablation_results.append((r_full, r_none))
        print(f"n={n_facts} cap={capacity}:")
        print(f"  full mechanism:  recall={r_full['recall_accuracy']:.3f} slots_used={r_full['n_slots_used']}")
        print(f"  no update (A):   recall={r_none['recall_accuracy']:.3f} slots_used={r_none['n_slots_used']}")

    return results, ablation_results


if __name__ == "__main__":
    main()
