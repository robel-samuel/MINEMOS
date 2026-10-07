"""
benchmarks/phase13_metrics.py
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate, encode_full, encode_query
from phase2.memory_update import cosine_similarity
from benchmarks.phase13_experiment import D, ENCODER


def _recall_scored(memory, subject, predicate, dim=D):
    q = encode_query(subject, predicate, dim=dim)
    if not memory.slots:
        return None, 0.0
    best_slot, best_sim = None, -1.0
    for slot in memory.slots:
        sim = cosine_similarity(q, slot.key)
        if sim > best_sim:
            best_sim, best_slot = sim, slot
    return best_slot, best_sim


def build_ground_truth(observations):
    truth = {}
    for o in observations:
        if o.category in ("exact_repeat", "paraphrase") and o.concept_id not in truth:
            truth[o.concept_id] = (o.candidate.subject, o.candidate.predicate, o.candidate.object)
    return truth


def consolidation_quality(run_result, observations) -> dict:
    memory = run_result["memory"]

    def nearest_slot(candidate):
        x = ENCODER.encode_full(candidate, dim=D)
        best_slot, best_sim = None, -1.0
        for slot in memory.slots:
            sim = cosine_similarity(x, slot.key)
            if sim > best_sim:
                best_sim, best_slot = sim, slot
        return best_slot

    base_repr = {}
    for o in observations:
        if o.category == "exact_repeat" and o.concept_id not in base_repr:
            base_repr[o.concept_id] = o.candidate

    true_opportunities = true_correct = 0
    false_opportunities = false_correct = 0
    for o in observations:
        if o.category == "paraphrase":
            true_opportunities += 1
            base_slot = nearest_slot(base_repr[o.concept_id]) if o.concept_id in base_repr else None
            para_slot = nearest_slot(o.candidate)
            if base_slot is not None and base_slot is para_slot:
                true_correct += 1
        elif o.probes_against is not None and o.category in ("hard_negative", "contradiction", "unrelated"):
            if o.probes_against not in base_repr:
                continue
            false_opportunities += 1
            base_slot = nearest_slot(base_repr[o.probes_against])
            probe_slot = nearest_slot(o.candidate)
            if base_slot is not None and base_slot is not probe_slot:
                false_correct += 1

    return dict(
        true_consolidation_rate=round(true_correct / true_opportunities, 4) if true_opportunities else None,
        false_consolidation_rate=round(1 - false_correct / false_opportunities, 4) if false_opportunities else None,
        n_slots=len(memory.slots),
        n_updates=run_result["n_updates"],
        n_inserts=run_result["n_inserts"],
        n_evicts=run_result["n_evicts"],
    )


def recall_metrics(run_result, observations) -> dict:
    memory = run_result["memory"]
    truth = build_ground_truth(observations)

    exact_correct = exact_total = 0
    for concept_id, (s, p, o) in truth.items():
        exact_total += 1
        slot, sim = _recall_scored(memory, s, p)
        if slot is not None:
            canonical = encode_full(Candidate(s, p, o), dim=D)
            if cosine_similarity(slot.key, canonical) > 0.9:
                exact_correct += 1

    false_hits = 0
    for i in range(100):
        slot, sim = _recall_scored(memory, f"GhostPerson_{i:04d}", "uses")
        if slot is not None and sim > 0.5:
            false_hits += 1

    return dict(
        exact_recall=round(exact_correct / exact_total, 4) if exact_total else None,
        false_memory_rate=round(false_hits / 100, 4),
    )


def error_propagation_test(dim: int = D):
    from phase2.memory_state import MemoryState
    from phase2.memory_update import absorb, MATCH_THRESHOLD
    from benchmarks.phase13_experiment import real_rekey_absorb

    subject, obj = "Person_ERR", "Object_ERR"
    fact_a = Candidate(subject, "uses", obj, timestamp=0)
    fact_b = Candidate(subject, "dislikes", obj, timestamp=1)
    later_paraphrases = [
        Candidate(subject, "regularly uses", obj, timestamp=2),
        Candidate(subject, "primarily works with", obj, timestamp=3),
        Candidate(subject, "relies on", obj, timestamp=4),
        Candidate(subject, "uses", obj, timestamp=5),
    ]

    results = {}
    for label, use_rekey in [("frozen", False), ("rekey", True)]:
        mem = MemoryState(capacity=20, dim=dim)
        trace = []
        if use_rekey:
            r1 = real_rekey_absorb(mem, fact_a, dim=dim)
            r2 = real_rekey_absorb(mem, fact_b, dim=dim)
        else:
            r1 = absorb(mem, fact_a, threshold=MATCH_THRESHOLD, encoder=ENCODER)
            r2 = absorb(mem, fact_b, threshold=MATCH_THRESHOLD, encoder=ENCODER)
        trace.append(dict(step="fact_B", action=r2["action"], similarity=round(r2["similarity"], 4)))
        incorrect_merge_occurred = (r2["action"] == "update")

        for i, para in enumerate(later_paraphrases):
            if use_rekey:
                r = real_rekey_absorb(mem, para, dim=dim)
            else:
                r = absorb(mem, para, threshold=MATCH_THRESHOLD, encoder=ENCODER)
            trace.append(dict(step=f"paraphrase_{i}", action=r["action"], similarity=round(r["similarity"], 4)))

        slot, sim = _recall_scored(mem, subject, "uses", dim=dim)
        canonical = encode_full(Candidate(subject, "uses", obj), dim=dim)
        final_similarity_to_clean_canonical = round(cosine_similarity(slot.value, canonical), 4) if slot else None

        results[label] = dict(
            incorrect_merge_occurred=incorrect_merge_occurred,
            trace=trace,
            final_slots=len(mem.slots),
            final_similarity_to_clean_canonical=final_similarity_to_clean_canonical,
        )
    return results
