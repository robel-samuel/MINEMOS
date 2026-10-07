"""
benchmarks/consolidation_ablation.py — Phase 5.

Tests whether the Black Hole update mechanism consolidates repeated/
near-duplicate observations better than a no-update baseline. Uses the
current atomic-tokenization encoder unchanged (Phase 2B), fixed
MATCH_THRESHOLD=0.75, fixed D, FIFO eviction -- only the dataset is new.

Ground truth (concept_id) is tracked ONLY by this benchmark script, for
scoring. The memory system never receives it -- it only ever sees
(subject, predicate, object, confidence, timestamp) Candidates.
"""

from __future__ import annotations

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate, encode_full
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb, cosine_similarity, MATCH_THRESHOLD
from phase2.recall import recall, decode_object

D = 4096  # fixed, matches the best-performing tested dimensionality from Phase 2A/2B


# ---------------------------------------------------------------------------
# Dataset construction (each observation tagged with a hidden concept_id)
# ---------------------------------------------------------------------------

class Obs:
    def __init__(self, candidate: Candidate, concept_id: str):
        self.candidate = candidate
        self.concept_id = concept_id  # hidden from the memory system


def category_1_exact_repetition(repeat_count: int) -> list[Obs]:
    return [Obs(Candidate("user", "uses", "Python", timestamp=t), "concept_python")
            for t in range(repeat_count)]


def category_2_near_duplicates() -> list[Obs]:
    variants = [
        Candidate("user", "uses", "Python", timestamp=1),
        Candidate("user", "uses", "python", timestamp=2),          # case variant (object)
        Candidate("user", "currently uses", "Python", timestamp=3),  # predicate variant
        Candidate("user", "has been using", "Python", timestamp=4),  # predicate variant
        Candidate("user", "primarily uses", "Python", timestamp=5),  # predicate variant
    ]
    return [Obs(c, "concept_python") for c in variants]


def category_3_distinct_facts() -> list[Obs]:
    langs = ["Python", "Rust", "Go", "JavaScript"]
    return [Obs(Candidate("user", "uses", lang, timestamp=t), f"concept_{lang.lower()}")
            for t, lang in enumerate(langs)]


def category_4_near_duplicate_distractors() -> list[Obs]:
    variants = [
        (Candidate("user", "uses", "Python", timestamp=1), "concept_user_uses_python"),
        (Candidate("user", "tested", "Python", timestamp=2), "concept_user_tested_python"),
        (Candidate("user", "teaches", "Python", timestamp=3), "concept_user_teaches_python"),
        (Candidate("user", "dislikes", "Python", timestamp=4), "concept_user_dislikes_python"),
        (Candidate("company", "uses", "Python", timestamp=5), "concept_company_uses_python"),
    ]
    return [Obs(c, cid) for c, cid in variants]


def category_5_mixed_stream(seed: int = 42) -> list[Obs]:
    rng = random.Random(seed)
    pool = []
    pool += category_1_exact_repetition(20)  # 20x python repeats
    pool += category_2_near_duplicates()
    pool += category_3_distinct_facts()
    pool += category_4_near_duplicate_distractors()
    rng.shuffle(pool)
    for t, obs in enumerate(pool):
        obs.candidate.timestamp = t
    return pool


# ---------------------------------------------------------------------------
# Running a stream through full mechanism vs no-update
# ---------------------------------------------------------------------------

def run_stream(observations: list[Obs], capacity: int, mechanism: str) -> dict:
    mem = MemoryState(capacity=capacity, dim=D)
    n_updates, n_inserts, n_evict_inserts, n_rejects = 0, 0, 0, 0

    for obs in observations:
        c = obs.candidate
        if mechanism == "no_update":
            x_t = encode_full(c, dim=D)
            if mem.is_full():
                mem.evict_oldest()
                n_evict_inserts += 1
            else:
                n_inserts += 1
            mem.add(x_t, x_t.copy(), c.confidence, c.timestamp,
                     debug_subject=c.subject, debug_predicate=c.predicate)
            # tag which concept each slot ACTUALLY came from, for scoring
            mem.slots[-1].debug_predicate = c.predicate  # keep raw predicate for query tests
        else:
            result = absorb(mem, c)
            if result["action"] == "update":
                n_updates += 1
            elif result["action"] == "insert":
                n_inserts += 1
            elif result["action"] == "evict_insert":
                n_evict_inserts += 1
            else:
                n_rejects += 1

    return dict(memory=mem, n_updates=n_updates, n_inserts=n_inserts,
                n_evict_inserts=n_evict_inserts, n_rejects=n_rejects,
                n_slots_used=len(mem.slots))


def false_consolidation_rate(memory: MemoryState, observations: list[Obs]) -> float:
    """
    For every slot, look at which concept_ids the observations that ended
    up addressing it belong to (post-hoc, using the same addressing logic
    recall would use). If a slot's set of matched concept_ids has more
    than one DISTINCT concept, that slot represents false consolidation.
    """
    from phase2.structured_candidate import encode_full as _ef
    slot_concepts = {i: set() for i in range(len(memory.slots))}
    for obs in observations:
        x_t = _ef(obs.candidate, dim=D)
        sims = [cosine_similarity(x_t, s.key) for s in memory.slots]
        if not sims:
            continue
        best = sims.index(max(sims))
        slot_concepts[best].add(obs.concept_id)

    contaminated = sum(1 for concepts in slot_concepts.values() if len(concepts) > 1)
    used = sum(1 for concepts in slot_concepts.values() if len(concepts) > 0)
    return contaminated / used if used else 0.0


def run_category(name: str, observations: list[Obs], capacity: int) -> dict:
    full = run_stream(observations, capacity, "full")
    none = run_stream(observations, capacity, "no_update")

    return {
        "category": name,
        "n_observations": len(observations),
        "n_distinct_concepts": len(set(o.concept_id for o in observations)),
        "capacity": capacity,
        "full_slots_used": full["n_slots_used"],
        "full_updates": full["n_updates"],
        "full_inserts": full["n_inserts"] + full["n_evict_inserts"],
        "full_false_consolidation": round(false_consolidation_rate(full["memory"], observations), 4),
        "no_update_slots_used": none["n_slots_used"],
        "no_update_false_consolidation": round(false_consolidation_rate(none["memory"], observations), 4),
        "slot_reduction": none["n_slots_used"] - full["n_slots_used"],
    }


# ---------------------------------------------------------------------------
# Representation drift
# ---------------------------------------------------------------------------

def representation_drift_trace(repeat_count: int = 100) -> list[dict]:
    mem = MemoryState(capacity=10, dim=D)
    canonical = encode_full(Candidate("user", "uses", "Python"), dim=D)
    trace = []
    checkpoints = {1, 5, 10, 100}
    for t in range(1, repeat_count + 1):
        result = absorb(mem, Candidate("user", "uses", "Python", timestamp=t))
        if t in checkpoints:
            key = mem.slots[result["slot_index"]].key
            sim_to_canonical = cosine_similarity(key, canonical)
            trace.append(dict(update_number=t, action=result["action"],
                               novelty=result["novelty"], similarity=result["similarity"],
                               sim_to_canonical=round(sim_to_canonical, 6)))
    return trace


# ---------------------------------------------------------------------------
# Update trace (explicit step-by-step, per spec)
# ---------------------------------------------------------------------------

def print_update_trace(observations: list[Obs], capacity: int = 10):
    mem = MemoryState(capacity=capacity, dim=D)
    for i, obs in enumerate(observations):
        c = obs.candidate
        old_keys = [s.key.copy() for s in mem.slots]
        result = absorb(mem, c)
        print(f"  Step {i+1}: input=({c.subject!r}, {c.predicate!r}, {c.object!r})  "
              f"concept={obs.concept_id}")
        print(f"    similarity={result['similarity']:.4f}  novelty={result['novelty']:.4f}  "
              f"action={result['action']}  slot_count={len(mem.slots)}")


def main():
    print("=" * 100)
    print(f"MATHEMATICAL CHECK: with atomic 3-token encoding, similarity in {{0, 1/3, 2/3, 1}}.")
    print(f"MATCH_THRESHOLD={MATCH_THRESHOLD} sits between 2/3=0.667 and 1.0 -- only EXACT")
    print("full-field matches (post-lowercase) can ever clear the merge threshold.")
    print("=" * 100)

    print()
    print("=" * 100)
    print("CATEGORY 1 — exact repetition, unconstrained capacity")
    print("=" * 100)
    for n in [1, 10, 100, 1000]:
        obs = category_1_exact_repetition(n)
        r = run_category(f"exact_repeat_{n}", obs, capacity=2000)
        print(f"  n={n:5d}: full_slots={r['full_slots_used']} (updates={r['full_updates']}) "
              f"vs no_update_slots={r['no_update_slots_used']}  "
              f"full_false_consolidation={r['full_false_consolidation']}")

    print()
    print("=" * 100)
    print("CATEGORY 2 — near-duplicates (predicate paraphrase + case variant)")
    print("=" * 100)
    obs2 = category_2_near_duplicates()
    r2 = run_category("near_duplicates", obs2, capacity=10)
    print(f"  {r2}")
    print("  Cross-phrasing recall test: fact ONLY ever stated as 'has been using',")
    print("  queried as 'uses' -- does it come back?")
    mem_test = MemoryState(capacity=10, dim=D)
    absorb(mem_test, Candidate("user", "has been using", "Python", timestamp=1))
    r = recall(mem_test, "user", "uses")
    vocab = {"python": encode_full(Candidate("_", "_", "python"), dim=D)}
    decoded = decode_object(r["slot"].value, vocab) if r["found"] else None
    print(f"    query similarity to the only stored slot: {r['similarity']:.4f}  decoded={decoded!r}")

    print()
    print("=" * 100)
    print("CATEGORY 3 — distinct facts sharing subject+predicate")
    print("=" * 100)
    obs3 = category_3_distinct_facts()
    r3 = run_category("distinct_facts", obs3, capacity=10)
    print(f"  {r3}")

    print()
    print("=" * 100)
    print("CATEGORY 4 — near-duplicate distractors")
    print("=" * 100)
    obs4 = category_4_near_duplicate_distractors()
    r4 = run_category("distractors", obs4, capacity=10)
    print(f"  {r4}")

    print()
    print("=" * 100)
    print("CATEGORY 5 — mixed stream, capacity sweep")
    print("=" * 100)
    obs5 = category_5_mixed_stream()
    for cap in [10, 25, 50, 100, 1000]:
        r5 = run_category(f"mixed_cap_{cap}", obs5, capacity=cap)
        print(f"  cap={cap:5d}: full_slots={r5['full_slots_used']:3d} vs "
              f"no_update_slots={r5['no_update_slots_used']:3d}  "
              f"n_distinct_concepts={r5['n_distinct_concepts']}  "
              f"full_false_consolidation={r5['full_false_consolidation']}  "
              f"no_update_false_consolidation={r5['no_update_false_consolidation']}")

    print()
    print("=" * 100)
    print("REPRESENTATION DRIFT — exact repeat, 100 updates")
    print("=" * 100)
    drift = representation_drift_trace(100)
    for d in drift:
        print(f"  {d}")

    print()
    print("=" * 100)
    print("UPDATE TRACE — Category 2 (near-duplicates), step by step")
    print("=" * 100)
    print_update_trace(category_2_near_duplicates(), capacity=10)


if __name__ == "__main__":
    main()
