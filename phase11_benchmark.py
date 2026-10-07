"""
benchmarks/phase11_benchmark.py

Runs Systems A/B/C across 4 capacity conditions and computes all
required metrics. Run twice; logical results (everything except wall
timing) must be identical, per the spec's sanity requirement.
"""

from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import encode_full
from phase2.memory_update import cosine_similarity
from phase2.recall import recall
from phase2.checkpoint import save_checkpoint
from benchmarks.phase11_dataset import generate_dataset, N_CONCEPTS
from benchmarks.oracle_ablation import run_system_A, run_system_B, run_system_C, D


def _decode_via_recall(memory, subject, predicate, expected_object):
    r = recall(memory, subject, predicate)
    if not r["found"]:
        return False
    from phase2.structured_candidate import Candidate
    canonical = encode_full(Candidate(subject, predicate, expected_object), dim=memory.dim)
    sim = cosine_similarity(r["slot"].key, canonical)
    return sim > 0.9


def oracle_ledger_concept_retention(system_c_result, base_concepts: dict) -> float:
    """
    Bypasses recall() entirely -- checks the oracle's own ledger (which
    concept_ids currently map to a real, still-resident slot) directly.
    This answers "was the concept's information ever correctly
    consolidated and does that slot still exist," independent of
    whether a cosine-similarity RECALL query using a particular
    phrasing could find it. See compute_metrics' docstring for why this
    is reported separately from concept_retention.
    """
    slot_map = system_c_result["concept_to_slot"]
    memory = system_c_result["memory"]
    resident = 0
    for concept_id in base_concepts:
        slot = slot_map.get(concept_id)
        if slot is not None and any(slot is s for s in memory.slots):
            resident += 1
    return round(resident / len(base_concepts), 4) if base_concepts else None


def compute_metrics(run_result: dict, observations, capacity: int) -> dict:
    """
    IMPORTANT DISTINCTION, discovered mid-experiment and disclosed here
    rather than silently corrected: `concept_retention` below is a
    RECALL-based metric (can a cosine-similarity query using the
    concept's "canonical" (exact_repeat) phrasing find its slot?). This
    is NOT the same question as "was the concept correctly
    consolidated" -- because MemorySlot keys are frozen at first
    creation (by design, since Phase 2), a concept whose first-arriving
    observation happened to be a PARAPHRASE (not the canonical
    phrasing) gets a slot keyed to that paraphrase's wording. A later
    query using the canonical wording can then fail to address that
    slot via cosine similarity, even though the slot's VALUE was
    perfectly, correctly consolidated. This was caught by inspecting a
    concrete failing case (concept_000) before trusting the aggregate
    number -- see phase11_report.md's Bugs/Findings section. For System
    C specifically, `oracle_ledger_concept_retention` (computed
    separately, in run_full_sweep) reports the ledger-verified truth,
    which is NOT subject to this addressing-order sensitivity.
    """
    memory = run_result["memory"]

    base_concepts = {}
    for o in observations:
        if o.category == "exact_repeat" and o.concept_id not in base_concepts:
            base_concepts[o.concept_id] = (o.candidate.subject, o.candidate.predicate, o.candidate.object)
    for o in observations:
        if o.category == "paraphrase" and o.concept_id not in base_concepts:
            base_concepts[o.concept_id] = (o.candidate.subject, o.candidate.predicate, o.candidate.object)

    retained = 0
    for concept_id, (s, p, o) in base_concepts.items():
        if _decode_via_recall(memory, s, p, o):
            retained += 1
    concept_retention = retained / len(base_concepts) if base_concepts else None

    contra_obs = [o for o in observations if o.category == "contradiction"]
    contra_retained = 0
    for o in contra_obs:
        if _decode_via_recall(memory, o.candidate.subject, o.candidate.predicate, o.candidate.object):
            contra_retained += 1
    contradiction_retention = contra_retained / len(contra_obs) if contra_obs else None

    all_concepts = {}
    for o in observations:
        if o.concept_id not in all_concepts:
            all_concepts[o.concept_id] = (o.candidate.subject, o.candidate.predicate, o.candidate.object)
    exact_correct = sum(1 for cid, (s, p, obj) in all_concepts.items()
                         if _decode_via_recall(memory, s, p, obj))
    exact_recall = exact_correct / len(all_concepts)

    false_hits = 0
    n_false_queries = 100
    for i in range(n_false_queries):
        r = recall(memory, f"GhostPerson_{i:04d}", "uses")
        if r["found"] and r["similarity"] > 0.5:
            false_hits += 1
    false_memory_rate = false_hits / n_false_queries

    return dict(
        capacity=capacity,
        n_slots=len(memory.slots),
        n_updates=run_result["n_updates"],
        n_inserts=run_result["n_inserts"],
        n_evicts=run_result["n_evicts"],
        n_rejects=run_result["n_rejects"],
        concept_retention=round(concept_retention, 4) if concept_retention is not None else None,
        contradiction_retention=round(contradiction_retention, 4) if contradiction_retention is not None else None,
        exact_recall=round(exact_recall, 4),
        false_memory_rate=round(false_memory_rate, 4),
        wall_seconds=round(run_result["wall_seconds"], 4),
    )


def true_false_consolidation(system_c_result, observations) -> dict:
    slot_map = system_c_result["concept_to_slot"]
    same_opportunities = 0
    same_correct = 0
    diff_opportunities = 0
    diff_correct = 0
    for o in observations:
        if o.category in ("exact_repeat", "paraphrase"):
            same_opportunities += 1
            slot = slot_map.get(o.concept_id)
            if slot is not None and any(slot is s for s in system_c_result["memory"].slots):
                same_correct += 1
        elif o.probes_against is not None:
            diff_opportunities += 1
            base_slot = slot_map.get(o.probes_against)
            probe_slot = slot_map.get(o.concept_id)
            if base_slot is None or probe_slot is None or base_slot is not probe_slot:
                diff_correct += 1
    return dict(
        true_consolidation_rate=round(same_correct / same_opportunities, 4) if same_opportunities else None,
        false_consolidation_rate=round(1 - diff_correct / diff_opportunities, 4) if diff_opportunities else None,
    )


def representation_drift_sample(observations, capacity: int = 1000) -> dict:
    from phase2.memory_state import MemoryState
    from benchmarks.oracle_ablation import oracle_absorb
    heavy_concept_obs = [o for o in observations if o.category == "exact_repeat"
                          and o.concept_id == "concept_099"]
    if not heavy_concept_obs:
        return {}
    mem = MemoryState(capacity=capacity, dim=D)
    canonical = encode_full(heavy_concept_obs[0].candidate, dim=D)
    trace = []
    checkpoints = {1, 5, 20, len(heavy_concept_obs)}
    slot = None
    for i, o in enumerate(heavy_concept_obs, start=1):
        r = oracle_absorb(mem, o.candidate, slot, dim=D)
        slot = r["slot"]
        if i in checkpoints:
            drift = cosine_similarity(slot.value, canonical)
            trace.append(dict(update_number=i, drift_to_canonical=round(float(drift), 6)))
    return dict(concept="concept_099", n_repeats=len(heavy_concept_obs), trace=trace)


def run_full_sweep():
    observations = generate_dataset()
    capacities = [
        ("unconstrained", 500),
        ("50pct", 200),
        ("25pct", 100),
        ("10pct", 40),
    ]

    results = {}
    for label, cap in capacities:
        results[label] = {}
        for name, fn in [("A_similarity", run_system_A), ("B_no_update", run_system_B), ("C_oracle", run_system_C)]:
            r = fn(observations, capacity=cap)
            metrics = compute_metrics(r, observations, cap)
            if name == "C_oracle":
                metrics.update(true_false_consolidation(r, observations))
                base_concepts = {}
                for o in observations:
                    if o.category == "exact_repeat" and o.concept_id not in base_concepts:
                        base_concepts[o.concept_id] = True
                for o in observations:
                    if o.category == "paraphrase" and o.concept_id not in base_concepts:
                        base_concepts[o.concept_id] = True
                metrics["oracle_ledger_concept_retention"] = oracle_ledger_concept_retention(r, base_concepts)
            results[label][name] = metrics

    return results, observations


if __name__ == "__main__":
    results, observations = run_full_sweep()
    for cap_label, systems in results.items():
        print(f"\n{'='*100}\nCAPACITY: {cap_label}\n{'='*100}")
        for sys_name, m in systems.items():
            print(f"  {sys_name}: {m}")

    print(f"\n{'='*100}\nREPRESENTATION DRIFT (heavy-repeat concept, oracle path)\n{'='*100}")
    drift = representation_drift_sample(observations)
    print(drift)
