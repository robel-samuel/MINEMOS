"""
benchmarks/phase15_metrics.py
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate, encode_full
from phase2.memory_state import MemoryState
from phase2.memory_update import cosine_similarity, MATCH_THRESHOLD, absorb
from benchmarks.phase15_experiment import ENCODER, D, confidence_gated_absorb
from benchmarks.phase13_experiment import real_rekey_absorb


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
        if o.category == "chain" and o.concept_id not in base_repr:
            base_repr[o.concept_id] = o.candidate

    true_opportunities = true_correct = 0
    false_opportunities = false_correct = 0
    for o in observations:
        if o.category == "chain":
            true_opportunities += 1
            base_slot = nearest_slot(base_repr[o.concept_id])
            this_slot = nearest_slot(o.candidate)
            if base_slot is this_slot:
                true_correct += 1
        elif o.probes_against is not None:
            if o.probes_against not in base_repr:
                continue
            false_opportunities += 1
            base_slot = nearest_slot(base_repr[o.probes_against])
            probe_slot = nearest_slot(o.candidate)
            if base_slot is not probe_slot:
                false_correct += 1

    return dict(
        true_consolidation_rate=round(true_correct / true_opportunities, 4) if true_opportunities else None,
        false_consolidation_rate=round(1 - false_correct / false_opportunities, 4) if false_opportunities else None,
        n_slots=len(memory.slots), n_updates=run_result["n_updates"],
        n_inserts=run_result["n_inserts"], n_evicts=run_result["n_evicts"],
        n_rekeys=run_result.get("n_rekeys", 0),
    )


def _run_chain_isolated(system: str, spec: dict, capacity: int = 20, dim: int = D):
    mem = MemoryState(capacity=capacity, dim=dim)
    confirm_state = {}
    trace = []

    def step(candidate):
        if system == "A":
            r = absorb(mem, candidate, threshold=MATCH_THRESHOLD, encoder=ENCODER)
        elif system == "B":
            r = real_rekey_absorb(mem, candidate, dim=dim)
            r["rekeyed"] = (r["action"] == "update")
        else:
            n_confirm = int(system[1:])
            r = confidence_gated_absorb(mem, candidate, confirm_state, n_confirm, dim=dim)
        slot_obj = mem.slots[r["slot_index"]] if r.get("slot_index") is not None else None
        trace.append(dict(action=r["action"], similarity=round(r["similarity"], 4),
                           rekeyed=r.get("rekeyed", False), slot=slot_obj))
        return r

    for c in spec["chain"]:
        step(c)

    canonical = encode_full(Candidate(spec["subject"], spec["predicate"], spec["object"]), dim=dim)
    q_vec = ENCODER.encode_full(Candidate(spec["subject"], spec["predicate"], spec["object"]), dim=dim)
    best_slot, best_sim = None, -1.0
    for slot in mem.slots:
        sim = cosine_similarity(q_vec, slot.key)
        if sim > best_sim:
            best_sim, best_slot = sim, slot

    key_sim = round(cosine_similarity(best_slot.key, canonical), 4) if best_slot else None
    value_sim = round(cosine_similarity(best_slot.value, canonical), 4) if best_slot else None

    return dict(trace=trace, n_slots_final=len(mem.slots),
                key_sim_to_clean_final=key_sim, value_sim_to_clean_final=value_sim)


def error_amplification_by_variant(specs: list, systems: list) -> dict:
    contra_position = {
        "canonical_para_contra": 2, "para_canonical_contra": 2,
        "contra_para_canonical": 0, "canonical_contra_para": 1,
    }
    results = {sysname: dict(chains_with_initial_error=0, total_additional_wrong_merges=0,
                              key_sims=[], value_sims=[])
               for sysname in systems}

    for spec in specs:
        if spec["variant"] not in contra_position:
            continue
        pos = contra_position[spec["variant"]]
        for sysname in systems:
            r = _run_chain_isolated(sysname, spec)
            contra_step = r["trace"][pos]
            initial_error = (contra_step["action"] == "update")
            if initial_error:
                results[sysname]["chains_with_initial_error"] += 1
                contaminated_slot = contra_step["slot"]
                additional = sum(
                    1 for step in r["trace"][pos + 1:]
                    if step["action"] == "update" and step["slot"] is contaminated_slot
                )
                results[sysname]["total_additional_wrong_merges"] += additional
            results[sysname]["key_sims"].append(r["key_sim_to_clean_final"])
            results[sysname]["value_sims"].append(r["value_sim_to_clean_final"])

    summary = {}
    for sysname, data in results.items():
        n = data["chains_with_initial_error"]
        ratio = round(data["total_additional_wrong_merges"] / n, 4) if n else None
        vk = [s for s in data["key_sims"] if s is not None]
        vv = [s for s in data["value_sims"] if s is not None]
        summary[sysname] = dict(
            chains_with_initial_error=n,
            error_amplification_ratio=ratio,
            mean_key_sim_to_clean=round(sum(vk) / len(vk), 4) if vk else None,
            mean_value_sim_to_clean=round(sum(vv) / len(vv), 4) if vv else None,
        )
    return summary
