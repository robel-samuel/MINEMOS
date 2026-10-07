"""
benchmarks/memory_recall.py — Tests B through F.

Same fixed parameters as memory_scaling.py (DIM=64, MATCH_THRESHOLD=0.75,
FIFO eviction). Not re-tuned after seeing Test A's results.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate, encode_full
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb, MATCH_THRESHOLD
from phase2.recall import recall, decode_object


# -- TEST B: repetition --------------------------------------------------

def test_b_repetition():
    mem = MemoryState(capacity=100)
    for t in range(500):
        absorb(mem, Candidate("user", "uses", "Python", timestamp=t))
    fp = mem.footprint_bytes()
    return {
        "n_absorbs": 500,
        "n_slots_used": fp["n_slots"],
        "update_count_on_slot": mem.slots[0].update_count if mem.slots else None,
        "conclusion": ("PASS: repetition did not consume additional slots"
                        if fp["n_slots"] == 1 else "FAIL: repeated identical fact created multiple slots"),
    }


# -- TEST C: paraphrase / same fact, different surface form --------------

def test_c_paraphrase():
    """Structured candidates only -- no NLP normalization exists in this
    architecture. Tests whether near-identical object STRINGS (not
    linguistic paraphrases) still land in the same slot, and honestly
    reports that true predicate-level paraphrasing (e.g. 'likes' vs
    'prefers' meaning the same relation) is NOT something this layer can
    recognize, since predicates here are compared as raw tokens."""
    results = {}

    mem1 = MemoryState(capacity=10)
    absorb(mem1, Candidate("user", "uses", "Python", timestamp=1))
    absorb(mem1, Candidate("user", "uses", "python", timestamp=2))  # case difference only
    results["case_variant_same_slot"] = len(mem1.slots) == 1

    mem2 = MemoryState(capacity=10)
    absorb(mem2, Candidate("user", "likes", "Python", timestamp=1))
    absorb(mem2, Candidate("user", "prefers", "Python", timestamp=2))  # different predicate, same meaning
    sim = None
    if len(mem2.slots) == 2:
        from phase2.memory_update import cosine_similarity
        sim = cosine_similarity(mem2.slots[0].key, mem2.slots[1].key)
    results["synonym_predicate_merged"] = len(mem2.slots) == 1
    results["synonym_predicate_similarity"] = sim
    results["note"] = ("This architecture has no predicate-synonym normalization -- "
                        "'likes' and 'prefers' are unrelated tokens to the hasher, so "
                        "they are NOT expected to merge, and did not. True paraphrase "
                        "consolidation would require the (frozen) extraction layer to "
                        "canonicalize predicates before candidates reach this memory.")
    return results


# -- TEST D: distractors --------------------------------------------------

def test_d_distractors(n_important: int = 20, n_distractors: int = 2000, capacity: int = 100):
    mem = MemoryState(capacity=capacity)
    important_facts = [Candidate(f"important_person_{i}", "likes", f"important_object_{i}", timestamp=i)
                        for i in range(n_important)]
    distractor_facts = [Candidate(f"distractor_{i}", "mentions", f"noise_{i}", timestamp=n_important + i)
                         for i in range(n_distractors)]

    # important facts absorbed first, then buried under distractors
    for c in important_facts:
        absorb(mem, c)
    for c in distractor_facts:
        absorb(mem, c)

    vocabulary = {f"important_object_{i}": encode_full(Candidate("_", "_", f"important_object_{i}"))
                  for i in range(n_important)}

    survived_exact = 0
    for i in range(n_important):
        r = recall(mem, f"important_person_{i}", "likes")
        if r["found"]:
            decoded = decode_object(r["slot"].value, vocabulary)
            if decoded == f"important_object_{i}":
                survived_exact += 1

    return {
        "n_important": n_important,
        "n_distractors": n_distractors,
        "capacity": capacity,
        "important_facts_exact_recall": round(survived_exact / n_important, 4),
        "n_slots_used": mem.footprint_bytes()["n_slots"],
    }


# -- TEST E: contradiction (measure baseline failure mode, don't fix it) --

def test_e_contradiction():
    mem = MemoryState(capacity=10)
    r1 = absorb(mem, Candidate("user", "uses", "Python", timestamp=1))
    r2 = absorb(mem, Candidate("user", "uses", "Rust", timestamp=2))

    vocabulary = {"Python": encode_full(Candidate("_", "_", "Python")),
                  "Rust": encode_full(Candidate("_", "_", "Rust"))}
    r = recall(mem, "user", "uses")
    decoded = decode_object(r["slot"].value, vocabulary) if r["found"] else None

    from phase2.memory_update import cosine_similarity
    sim_to_python = cosine_similarity(r["slot"].value, vocabulary["Python"]) if r["found"] else None
    sim_to_rust = cosine_similarity(r["slot"].value, vocabulary["Rust"]) if r["found"] else None

    return {
        "n_slots_after_contradiction": len(mem.slots),
        "second_absorb_action": r2["action"],  # "update" (blended) or "insert" (separate slot)
        "second_absorb_similarity_to_first": r2["similarity"],
        "decoded_current_value": decoded,
        "similarity_of_stored_value_to_Python": sim_to_python,
        "similarity_of_stored_value_to_Rust": sim_to_rust,
        "conclusion": (
            "Baseline blended both facts into one ambiguous vector (predicted failure mode)"
            if r2["action"] == "update" else
            "Baseline kept them as separate slots -- recall ambiguity is the failure mode instead"
        ),
    }


# -- TEST F: long-horizon recall by age -----------------------------------

def test_f_long_horizon(n_facts: int = 2000, capacity: int = 500):
    mem = MemoryState(capacity=capacity)
    facts = [Candidate(f"person_{i:06d}", "likes", f"object_{i:06d}", timestamp=i)
             for i in range(n_facts)]
    for c in facts:
        absorb(mem, c)
    del facts

    vocabulary = {f"object_{i:06d}": encode_full(Candidate("_", "_", f"object_{i:06d}"))
                  for i in range(n_facts)}

    def band_accuracy(indices):
        correct = 0
        for i in indices:
            r = recall(mem, f"person_{i:06d}", "likes")
            if r["found"]:
                decoded = decode_object(r["slot"].value, vocabulary)
                if decoded == f"object_{i:06d}":
                    correct += 1
        return round(correct / len(indices), 4)

    early = list(range(0, 50))
    middle = list(range(n_facts // 2 - 25, n_facts // 2 + 25))
    recent = list(range(n_facts - 50, n_facts))
    import random
    rng = random.Random(3)
    random_sample = rng.sample(range(n_facts), 50)

    return {
        "n_facts": n_facts,
        "capacity": capacity,
        "n_slots_used": mem.footprint_bytes()["n_slots"],
        "early_facts_exact_recall": band_accuracy(early),
        "middle_facts_exact_recall": band_accuracy(middle),
        "recent_facts_exact_recall": band_accuracy(recent),
        "random_facts_exact_recall": band_accuracy(random_sample),
    }


def main():
    print("=" * 100)
    print(f"TEST B: repetition (MATCH_THRESHOLD={MATCH_THRESHOLD})")
    print("=" * 100)
    print(test_b_repetition())

    print()
    print("=" * 100)
    print("TEST C: paraphrase / same-fact surface variation")
    print("=" * 100)
    print(test_c_paraphrase())

    print()
    print("=" * 100)
    print("TEST D: distractors")
    print("=" * 100)
    print(test_d_distractors())

    print()
    print("=" * 100)
    print("TEST E: contradiction (baseline failure mode, not fixed)")
    print("=" * 100)
    print(test_e_contradiction())

    print()
    print("=" * 100)
    print("TEST F: long-horizon recall by age")
    print("=" * 100)
    print(test_f_long_horizon())


if __name__ == "__main__":
    main()
