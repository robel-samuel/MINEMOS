"""
benchmarks/threshold_sweep.py — Phase 6.

Sweeps ONLY MATCH_THRESHOLD across {0.50..0.90}. Encoder, tokenizer,
update equation, slot structure, eviction all unchanged from Phase 2B/5.

Dataset: independently constructed pairs (not reused from Phase 5),
in three categories:
  A. exact duplicates       -- MUST consolidate at any reasonable threshold
  B. genuine paraphrases    -- SHOULD consolidate if representation allows
  C. hard negatives         -- MUST NOT consolidate
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate, encode_full
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb, cosine_similarity
from phase2.checkpoint import save_checkpoint

D = 4096

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


class Pair:
    def __init__(self, category: str, c1: Candidate, c2: Candidate, should_merge: bool, label: str):
        self.category = category
        self.c1 = c1
        self.c2 = c2
        self.should_merge = should_merge
        self.label = label


# -- A. exact duplicates (6 pairs) -----------------------------------------
A_PAIRS = [
    Pair("A", Candidate("user", "uses", "Python"), Candidate("user", "uses", "Python"), True, "python_exact"),
    Pair("A", Candidate("user", "works at", "Acme"), Candidate("user", "works at", "Acme"), True, "acme_exact"),
    Pair("A", Candidate("user", "likes", "coffee"), Candidate("user", "likes", "coffee"), True, "coffee_exact"),
    Pair("A", Candidate("user", "lives in", "Berlin"), Candidate("user", "lives in", "Berlin"), True, "berlin_exact"),
    Pair("A", Candidate("user", "manages", "team_alpha"), Candidate("user", "manages", "team_alpha"), True, "team_exact"),
    Pair("A", Candidate("user", "owns", "a bicycle"), Candidate("user", "owns", "a bicycle"), True, "bike_exact"),
]

# -- B. genuine paraphrases (10 pairs, new wording not reused from Phase 5) --
B_PAIRS = [
    Pair("B", Candidate("user", "uses", "Python"), Candidate("user", "regularly uses", "Python"), True, "python_regularly"),
    Pair("B", Candidate("user", "uses", "Python"), Candidate("user", "relies on", "Python"), True, "python_relies"),
    Pair("B", Candidate("user", "works at", "Acme"), Candidate("user", "is employed by", "Acme"), True, "acme_employed"),
    Pair("B", Candidate("user", "works at", "Acme"), Candidate("user", "works for", "Acme"), True, "acme_works_for"),
    Pair("B", Candidate("user", "likes", "coffee"), Candidate("user", "enjoys", "coffee"), True, "coffee_enjoys"),
    Pair("B", Candidate("user", "likes", "coffee"), Candidate("user", "is fond of", "coffee"), True, "coffee_fond"),
    Pair("B", Candidate("user", "lives in", "Berlin"), Candidate("user", "resides in", "Berlin"), True, "berlin_resides"),
    Pair("B", Candidate("user", "manages", "team_alpha"), Candidate("user", "leads", "team_alpha"), True, "team_leads"),
    Pair("B", Candidate("user", "owns", "a bicycle"), Candidate("user", "has", "a bicycle"), True, "bike_has"),
    Pair("B", Candidate("user", "uses", "Python"), Candidate("user", "codes primarily in", "Python"), True, "python_codes_in"),
]

# -- C. hard negatives (18 pairs) ------------------------------------------
C_PAIRS = [
    Pair("C", Candidate("user", "uses", "Python"), Candidate("user", "dislikes", "Python"), False, "python_dislikes"),
    Pair("C", Candidate("user", "uses", "Python"), Candidate("user", "teaches", "Python"), False, "python_teaches"),
    Pair("C", Candidate("user", "uses", "Python"), Candidate("company", "uses", "Python"), False, "python_company"),
    Pair("C", Candidate("user", "uses", "Python"), Candidate("user", "uses", "Rust"), False, "python_vs_rust"),
    Pair("C", Candidate("user", "works at", "Acme"), Candidate("user", "left", "Acme"), False, "acme_left"),
    Pair("C", Candidate("user", "works at", "Acme"), Candidate("user", "interviewed at", "Acme"), False, "acme_interviewed"),
    Pair("C", Candidate("user", "likes", "coffee"), Candidate("user", "hates", "coffee"), False, "coffee_hates"),
    Pair("C", Candidate("user", "likes", "coffee"), Candidate("user", "likes", "tea"), False, "coffee_vs_tea"),
    Pair("C", Candidate("user", "likes", "coffee"), Candidate("colleague", "likes", "coffee"), False, "coffee_colleague"),
    Pair("C", Candidate("user", "lives in", "Berlin"), Candidate("user", "visited", "Berlin"), False, "berlin_visited"),
    Pair("C", Candidate("user", "lives in", "Berlin"), Candidate("user", "lives in", "Paris"), False, "berlin_vs_paris"),
    Pair("C", Candidate("user", "manages", "team_alpha"), Candidate("user", "manages", "team_beta"), False, "team_alpha_vs_beta"),
    Pair("C", Candidate("user", "manages", "team_alpha"), Candidate("colleague", "manages", "team_alpha"), False, "team_colleague"),
    Pair("C", Candidate("user", "owns", "a bicycle"), Candidate("user", "sold", "a bicycle"), False, "bike_sold"),
    Pair("C", Candidate("user", "owns", "a bicycle"), Candidate("user", "owns", "a car"), False, "bike_vs_car"),
    # two-field-different negatives, for completeness
    Pair("C", Candidate("user", "uses", "Python"), Candidate("company", "teaches", "Rust"), False, "fully_unrelated_1"),
    Pair("C", Candidate("user", "likes", "coffee"), Candidate("colleague", "hates", "tea"), False, "fully_unrelated_2"),
    Pair("C", Candidate("user", "manages", "team_alpha"), Candidate("company", "sold", "a car"), False, "fully_unrelated_3"),
]

ALL_PAIRS = A_PAIRS + B_PAIRS + C_PAIRS
assert len(ALL_PAIRS) >= 30, f"only {len(ALL_PAIRS)} pairs"


def merged_status(pair: Pair, threshold: float) -> tuple[bool, dict]:
    """Fresh 2-slot memory per pair -- isolates each comparison cleanly."""
    m = MemoryState(capacity=10, dim=D)
    absorb(m, pair.c1, threshold=threshold)
    r2 = absorb(m, pair.c2, threshold=threshold)
    return r2["action"] == "update", r2


def run_threshold(threshold: float, out_dir: str) -> dict:
    true_consolidations = missed_consolidations = 0
    false_consolidations = correct_separations = 0
    absorb_times = []
    drift_examples = []

    combined_mem = MemoryState(capacity=200, dim=D)

    for pair in ALL_PAIRS:
        t0 = time.perf_counter()
        absorb(combined_mem, pair.c1, threshold=threshold)
        r2 = absorb(combined_mem, pair.c2, threshold=threshold)
        absorb_times.append(time.perf_counter() - t0)

        merged = r2["action"] == "update"
        if pair.should_merge and merged:
            true_consolidations += 1
        elif pair.should_merge and not merged:
            missed_consolidations += 1
        elif not pair.should_merge and merged:
            false_consolidations += 1
        else:
            correct_separations += 1

        if merged:
            slot = combined_mem.slots[r2["slot_index"]]
            canonical = encode_full(pair.c1, dim=D)
            # NOTE: drift must be measured on slot.value, not slot.key --
            # k_i is frozen at slot creation by design (Phase 2 spec); it
            # is v_i that the update equation actually blends. An earlier
            # version of this script measured .key here, which trivially
            # always reads 1.0 (unchanged) regardless of how much real
            # blending happened -- caught by cross-checking against a
            # hand-derived expected value before trusting the number.
            drift_examples.append((pair.label, round(cosine_similarity(slot.value, canonical), 6)))

    a_merged = sum(1 for p in A_PAIRS if merged_status(p, threshold)[0])
    b_merged = sum(1 for p in B_PAIRS if merged_status(p, threshold)[0])
    c_merged = sum(1 for p in C_PAIRS if merged_status(p, threshold)[0])

    ckpt_path = os.path.join(out_dir, f"sweep_{threshold}.npz")
    save_checkpoint(combined_mem, ckpt_path)
    state_bytes = os.path.getsize(ckpt_path)

    return dict(
        threshold=threshold,
        exact_duplicate_consolidation_rate=round(a_merged / len(A_PAIRS), 4),
        paraphrase_consolidation_rate=round(b_merged / len(B_PAIRS), 4),
        false_consolidation_rate=round(c_merged / len(C_PAIRS), 4),
        true_consolidations=true_consolidations,
        missed_consolidations=missed_consolidations,
        false_consolidations=false_consolidations,
        correct_separations=correct_separations,
        slots_used=len(combined_mem.slots),
        state_bytes=state_bytes,
        avg_absorb_latency_ms=round(sum(absorb_times) / len(absorb_times) * 1000, 5),
        representation_drift_examples=drift_examples[:5],
    )


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "_threshold_sweep_checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Dataset: {len(A_PAIRS)} exact-duplicate pairs, {len(B_PAIRS)} paraphrase pairs, "
          f"{len(C_PAIRS)} hard-negative pairs = {len(ALL_PAIRS)} total pairs")
    print()
    print(f"{'threshold':>10} {'exact_dup':>10} {'paraphrase':>11} {'false_cons':>11} "
          f"{'slots':>7} {'bytes':>8} {'latency_ms':>11}")

    results = []
    for t in THRESHOLDS:
        r = run_threshold(t, out_dir)
        results.append(r)
        print(f"{r['threshold']:>10.2f} {r['exact_duplicate_consolidation_rate']:>10.4f} "
              f"{r['paraphrase_consolidation_rate']:>11.4f} {r['false_consolidation_rate']:>11.4f} "
              f"{r['slots_used']:>7} {r['state_bytes']:>8} {r['avg_absorb_latency_ms']:>11.5f}")

    print()
    print("Searching for a threshold region where paraphrase_consolidation is HIGH")
    print("and false_consolidation is LOW simultaneously...")
    found_region = False
    for r in results:
        if r["paraphrase_consolidation_rate"] >= 0.5 and r["false_consolidation_rate"] <= 0.1:
            print(f"  FOUND at threshold={r['threshold']}: paraphrase={r['paraphrase_consolidation_rate']}, "
                  f"false_cons={r['false_consolidation_rate']}")
            found_region = True
    if not found_region:
        print("  NO threshold in the swept range satisfies both conditions simultaneously.")

    return results


if __name__ == "__main__":
    main()
