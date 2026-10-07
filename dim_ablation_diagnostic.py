"""
benchmarks/dim_ablation_diagnostic.py

Section 6 of the Phase 2A spec: before running the memory benchmark,
directly measure collision behavior at each D. Same deterministic
hashing algorithm (SHA-1 based feature hashing) at every D -- only the
number of buckets changes.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate, encode_full, _tokenize, _hash_token
from phase2.memory_update import cosine_similarity

DIMS = [64, 128, 256, 512, 1024, 2048, 4096]


def collision_stats(n_facts: int, dim: int) -> dict:
    seen_buckets = {}
    n_tokens = 0
    for i in range(n_facts):
        tokens = _tokenize(f"person_{i:06d}") + _tokenize(f"object_{i:06d}") + _tokenize("likes")
        for tok in tokens:
            idx, sign = _hash_token(tok, dim)
            seen_buckets.setdefault(idx, set()).add(tok)
            n_tokens += 1

    occupied = len(seen_buckets)
    collided = sum(1 for toks in seen_buckets.values() if len(toks) > 1)
    max_bucket = max((len(toks) for toks in seen_buckets.values()), default=0)
    avg_bucket = n_tokens / max(occupied, 1)

    return {
        "dim": dim,
        "n_tokens": n_tokens,
        "occupied_buckets": occupied,
        "buckets_with_collision": collided,
        "collision_rate": round(collided / occupied, 4) if occupied else None,
        "avg_tokens_per_occupied_bucket": round(avg_bucket, 3),
        "max_tokens_in_one_bucket": max_bucket,
    }


def unrelated_fact_similarity(dim: int) -> dict:
    facts = {
        "person1_uses_python": Candidate("person_1", "uses", "Python"),
        "person2_uses_rust": Candidate("person_2", "uses", "Rust"),
        "person3_lives_paris": Candidate("person_3", "lives_in", "Paris"),
        "person4_owns_tesla": Candidate("person_4", "owns", "Tesla"),
    }
    vecs = {k: encode_full(v, dim=dim) for k, v in facts.items()}
    keys = list(vecs.keys())
    pairs = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sim = cosine_similarity(vecs[keys[i]], vecs[keys[j]])
            pairs[f"{keys[i]} vs {keys[j]}"] = round(sim, 4)
    avg_sim = sum(pairs.values()) / len(pairs)
    return {"dim": dim, "pairwise_similarities": pairs, "avg_unrelated_similarity": round(avg_sim, 4)}


def main():
    print("=" * 100)
    print("COLLISION DIAGNOSTIC (n_facts=1000, same as prior Phase 2 diagnostic)")
    print("=" * 100)
    coll_results = []
    for d in DIMS:
        stats = collision_stats(1000, d)
        coll_results.append(stats)
        print(f"D={d:5d}  occupied={stats['occupied_buckets']:5d}/{d:5d}  "
              f"collision_rate={stats['collision_rate']}  "
              f"avg_tokens/bucket={stats['avg_tokens_per_occupied_bucket']:.2f}  "
              f"max_in_bucket={stats['max_tokens_in_one_bucket']}")

    print()
    print("=" * 100)
    print("UNRELATED-FACT PAIRWISE SIMILARITY (should approach 0 as D increases)")
    print("=" * 100)
    sim_results = []
    for d in DIMS:
        r = unrelated_fact_similarity(d)
        sim_results.append(r)
        print(f"D={d:5d}  avg_unrelated_similarity={r['avg_unrelated_similarity']}")
        for pair, sim in r["pairwise_similarities"].items():
            print(f"    {pair}: {sim}")

    return coll_results, sim_results


if __name__ == "__main__":
    main()
