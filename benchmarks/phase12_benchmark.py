"""
benchmarks/phase12_benchmark.py
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate
from benchmarks.phase12_dataset import generate_dataset, N_CONCEPTS
from benchmarks.phase12_addressing import run_system, oracle_recall_scored, D


def concept_retention_ledger(run_result, base_concept_ids) -> float:
    slot_map = run_result["concept_to_slot"]
    memory = run_result["memory"]
    resident = sum(1 for cid in base_concept_ids
                    if slot_map.get(cid) is not None and any(slot_map[cid] is s for s in memory.slots))
    return round(resident / len(base_concept_ids), 4) if base_concept_ids else None


def addressable_recall(run_result, specs, query_by: str) -> dict:
    memory = run_result["memory"]
    alias_store = run_result["alias_store"]
    slot_map = run_result["concept_to_slot"]

    by_case = {"canonical_first": [0, 0], "paraphrase_first": [0, 0], "mixed": [0, 0]}

    for spec in specs:
        true_slot = slot_map.get(spec.concept_id)
        if true_slot is None or not any(true_slot is s for s in memory.slots):
            continue

        if query_by == "canonical":
            predicate = spec.predicate
        elif query_by == "para_0":
            predicate = spec.paraphrase_predicates[0]
        elif query_by == "para_1":
            predicate = spec.paraphrase_predicates[1]
        elif query_by == "held_out":
            predicate = spec.held_out_predicate
        else:
            raise ValueError(query_by)

        found_slot, sim = oracle_recall_scored(memory, spec.subject, predicate, dim=D, alias_store=alias_store)
        correct = found_slot is true_slot
        by_case[spec.ordering_case][1] += 1
        if correct:
            by_case[spec.ordering_case][0] += 1

    result = {case: round(c / t, 4) if t else None for case, (c, t) in by_case.items()}
    total_c = sum(c for c, t in by_case.values())
    total_t = sum(t for c, t in by_case.values())
    result["overall"] = round(total_c / total_t, 4) if total_t else None
    return result


def contradiction_separation(run_result, specs) -> float:
    memory = run_result["memory"]
    slot_map = run_result["concept_to_slot"]
    correct = 0
    total = 0
    for spec in specs:
        contra_id = f"{spec.concept_id}_contra"
        base_slot = slot_map.get(spec.concept_id)
        contra_slot = slot_map.get(contra_id)
        if base_slot is None or contra_slot is None:
            continue
        if not (any(base_slot is s for s in memory.slots) and any(contra_slot is s for s in memory.slots)):
            continue
        total += 1
        if base_slot is not contra_slot:
            correct += 1
    return round(correct / total, 4) if total else None


def false_memory_rate(run_result) -> float:
    memory = run_result["memory"]
    alias_store = run_result["alias_store"]
    hits = 0
    n = 100
    for i in range(n):
        slot, sim = oracle_recall_scored(memory, f"GhostAlice_{i:04d}", "uses", dim=D, alias_store=alias_store)
        if slot is not None and sim > 0.5:
            hits += 1
    return round(hits / n, 4)


def footprint(run_result) -> dict:
    memory = run_result["memory"]
    fp = memory.footprint_bytes()
    alias_store = run_result["alias_store"]
    alias_bytes = 0
    n_aliases = 0
    if alias_store:
        for aliases in alias_store.values():
            for a in aliases:
                alias_bytes += a.nbytes
                n_aliases += 1
    return dict(base_total_bytes=fp["total_bytes"], n_aliases=n_aliases,
                alias_bytes=alias_bytes, total_with_aliases=fp["total_bytes"] + alias_bytes,
                bytes_per_concept=round((fp["total_bytes"] + alias_bytes) / max(fp["n_slots"], 1), 2))


def run_full_comparison(capacity: int = 500):
    observations, specs = generate_dataset()
    base_concept_ids = [s.concept_id for s in specs]

    results = {}
    for sysname in ["A", "B", "C"]:
        r = run_system(observations, capacity=capacity, absorb_fn=sysname)
        metrics = dict(
            slots=len(r["memory"].slots), updates=r["n_updates"], inserts=r["n_inserts"], evicts=r["n_evicts"],
            concept_retention=concept_retention_ledger(r, base_concept_ids),
            addressable_recall_canonical_query=addressable_recall(r, specs, "canonical"),
            addressable_recall_para0_query=addressable_recall(r, specs, "para_0"),
            addressable_recall_para1_query=addressable_recall(r, specs, "para_1"),
            addressable_recall_held_out_query=addressable_recall(r, specs, "held_out"),
            contradiction_separation=contradiction_separation(r, specs),
            false_memory_rate=false_memory_rate(r),
            footprint=footprint(r),
            wall_seconds=round(r["wall_seconds"], 4),
        )
        cr = metrics["concept_retention"]
        ar = metrics["addressable_recall_canonical_query"]["overall"]
        metrics["addressing_gap_canonical"] = round(cr - ar, 4) if cr is not None and ar is not None else None
        results[sysname] = metrics

    return results, observations, specs


if __name__ == "__main__":
    results, obs, specs = run_full_comparison()
    for sysname, m in results.items():
        print(f"\n=== System {sysname} ===")
        for k, v in m.items():
            print(f"  {k}: {v}")
