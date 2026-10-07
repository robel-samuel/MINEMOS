"""
benchmarks/phase14_metrics.py
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate, encode_full
from phase2.memory_state import MemoryState
from phase2.memory_update import cosine_similarity, MATCH_THRESHOLD, absorb
from benchmarks.phase14_experiment import run_system, delayed_rekey_absorb, ENCODER, D
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
        elif o.probes_against is not None:
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
        n_slots=len(memory.slots), n_updates=run_result["n_updates"],
        n_inserts=run_result["n_inserts"], n_evicts=run_result["n_evicts"],
        n_rekeys=run_result.get("n_rekeys", 0),
    )


def _run_chain(system: str, chain: dict, capacity: int = 20, dim: int = D):
    mem = MemoryState(capacity=capacity, dim=dim)
    merge_counts = {}
    trace = []

    def step(candidate):
        if system == "A":
            r = absorb(mem, candidate, threshold=MATCH_THRESHOLD, encoder=ENCODER)
        elif system == "B":
            r = real_rekey_absorb(mem, candidate, dim=dim)
            r["rekeyed"] = (r["action"] == "update")
        else:
            N = int(system[1:])
            r = delayed_rekey_absorb(mem, candidate, merge_counts, N, dim=dim)
        slot_obj = mem.slots[r["slot_index"]] if r.get("slot_index") is not None else None
        trace.append(dict(action=r["action"], similarity=round(r["similarity"], 4),
                           rekeyed=r.get("rekeyed", False), slot=slot_obj))
        return r

    for c in chain["sequence"]:
        step(c)
    n_slots_after_chain = len(mem.slots)

    for c in chain["recovery"]:
        step(c)

    canonical = encode_full(Candidate(chain["subject"], chain["predicate"], chain["object"]), dim=dim)
    q_vec = ENCODER.encode_full(Candidate(chain["subject"], chain["predicate"], chain["object"]), dim=dim)
    best_slot, best_sim = None, -1.0
    for slot in mem.slots:
        sim = cosine_similarity(q_vec, slot.key)
        if sim > best_sim:
            best_sim, best_slot = sim, slot

    key_sim_to_clean = round(cosine_similarity(best_slot.key, canonical), 4) if best_slot else None
    value_sim_to_clean = round(cosine_similarity(best_slot.value, canonical), 4) if best_slot else None

    return dict(
        trace=trace,
        n_slots_after_chain=n_slots_after_chain,
        n_slots_final=len(mem.slots),
        key_sim_to_clean_final=key_sim_to_clean,
        value_sim_to_clean_final=value_sim_to_clean,
    )


def error_amplification_analysis(chains: list, systems: list) -> dict:
    """
    For each hardneg-containing chain, find the first incorrect merge
    (the hard-negative step, at trace index 1). Track the SLOT OBJECT it
    merged into. "Additional wrong merges" = subsequent steps within the
    ORIGINAL SEQUENCE (not the recovery tail -- recovery is a separate,
    distinct question, see recovery_key_sim/recovery_value_sim) that
    merge into that SAME contaminated slot object.

    A bug was caught and fixed here before trusting any result: an
    earlier version counted ANY subsequent "update" action anywhere in
    the trace -- including the recovery tail, and including correct
    merges into a different, legitimate slot -- producing an
    error-amplification ratio above 3.0 for a sequence where at most 1
    additional merge was structurally possible. Caught because the
    ratio was numerically impossible given the sequence length, not
    from a stack trace.
    """
    results = {sysname: dict(chains_with_initial_error=0, total_additional_wrong_merges=0,
                              recovery_key_sims=[], recovery_value_sims=[])
               for sysname in systems}

    for chain in chains:
        if "hardneg" not in chain["variant"]:
            continue
        for sysname in systems:
            r = _run_chain(sysname, chain)
            seq_len = len(chain["sequence"])
            hardneg_step = r["trace"][1]
            initial_error = (hardneg_step["action"] == "update")
            if initial_error:
                results[sysname]["chains_with_initial_error"] += 1
                contaminated_slot = hardneg_step["slot"]
                # only steps WITHIN the original sequence (not recovery),
                # and only merges into the SAME contaminated slot object
                additional = sum(
                    1 for step in r["trace"][2:seq_len]
                    if step["action"] == "update" and step["slot"] is contaminated_slot
                )
                results[sysname]["total_additional_wrong_merges"] += additional
            results[sysname]["recovery_key_sims"].append(r["key_sim_to_clean_final"])
            results[sysname]["recovery_value_sims"].append(r["value_sim_to_clean_final"])

    summary = {}
    for sysname, data in results.items():
        n = data["chains_with_initial_error"]
        ratio = round(data["total_additional_wrong_merges"] / n, 4) if n else None
        valid_key_sims = [s for s in data["recovery_key_sims"] if s is not None]
        valid_value_sims = [s for s in data["recovery_value_sims"] if s is not None]
        summary[sysname] = dict(
            chains_with_initial_error=n,
            error_amplification_ratio=ratio,
            mean_recovery_key_sim=round(sum(valid_key_sims) / len(valid_key_sims), 4) if valid_key_sims else None,
            mean_recovery_value_sim=round(sum(valid_value_sims) / len(valid_value_sims), 4) if valid_value_sims else None,
        )
    return summary
