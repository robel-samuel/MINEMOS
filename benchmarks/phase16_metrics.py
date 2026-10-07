"""
benchmarks/phase16_metrics.py

Phase 16 Metrics Suite:
1. Pairwise Metrics:
   - Accuracy, precision, recall, F1, confusion matrix
   - Contradiction detection rate, entailment detection rate, neutral detection rate
   - False entailment rate
   - Explicit metric: False Consolidation of Contradictions
   - Raw probability tracking
   - Breakdown across all 7 subcategories
2. Memory Consolidation Metrics:
   - True consolidation rate, false consolidation rate
   - Contradiction retention, paraphrase consolidation, unrelated separation
   - Slot counts, merge counts, rejected candidate counts, drift
3. Contamination Chains Experiment:
   - Initial contradiction merge rate
   - Bounded head error amplification (Step 3 paraphrase merges)
   - Recovery key & value similarity to clean canonical representation
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState
from phase2.memory_update import cosine_similarity, MATCH_THRESHOLD
from benchmarks.phase16_experiment import (
    NliSemanticGate, ENCODER, D,
    lexical_absorb, nli_gated_absorb, oracle_absorb, exhaustive_nli_absorb
)


def evaluate_pairwise(pairs, semantic_gate: NliSemanticGate = None,
                      lexical_threshold: float = 0.75,
                      entail_threshold: float = 0.50,
                      contra_threshold: float = 0.50,
                      abstain_min_confidence: float = 0.35) -> dict:
    """Evaluates Lexical Baseline vs NLI Gate on the 350-pair dataset."""
    results = {}

    # Binary Ground Truth: SAME (1) vs NOT_SAME (0)
    y_true_binary = [1 if p.ground_truth == "SAME" else 0 for p in pairs]

    # --- Lexical Evaluation ---
    y_pred_lex = []
    scores_lex = []
    for p in pairs:
        x1 = ENCODER.encode_full(p.cand1, dim=D)
        x2 = ENCODER.encode_full(p.cand2, dim=D)
        sim = float(cosine_similarity(x1, x2))
        scores_lex.append(sim)
        y_pred_lex.append(1 if sim >= lexical_threshold else 0)

    acc_lex = accuracy_score(y_true_binary, y_pred_lex)
    prec_lex, rec_lex, f1_lex, _ = precision_recall_fscore_support(y_true_binary, y_pred_lex, average="binary")
    auc_lex = roc_auc_score(y_true_binary, scores_lex) if len(set(y_true_binary)) > 1 else None
    cm_lex = confusion_matrix(y_true_binary, y_pred_lex).tolist()

    # Contradiction-specific evaluation for lexical
    contra_indices = [i for i, p in enumerate(pairs) if p.ground_truth == "CONTRADICTION"]
    contra_merges_lex = sum(y_pred_lex[i] for i in contra_indices)
    false_consolidation_contra_lex = contra_merges_lex / len(contra_indices) if contra_indices else 0.0

    results["lexical"] = {
        "accuracy": round(acc_lex, 4),
        "precision": round(prec_lex, 4),
        "recall": round(rec_lex, 4),
        "f1": round(f1_lex, 4),
        "roc_auc": round(auc_lex, 4) if auc_lex else None,
        "confusion_matrix": cm_lex,
        "false_consolidation_of_contradictions": round(false_consolidation_contra_lex, 4),
        "contradiction_detection_rate": round(1.0 - false_consolidation_contra_lex, 4),
        "entailment_detection_rate": round(rec_lex, 4),
        "false_entailment_rate": round(sum(y_pred_lex[i] for i, y in enumerate(y_true_binary) if y == 0) / (len(y_true_binary) - sum(y_true_binary)), 4),
    }

    # Breakdown by subcategory for Lexical
    subcat_lex = {}
    for p, score, pred in zip(pairs, scores_lex, y_pred_lex):
        subcat = p.subcategory
        subcat_lex.setdefault(subcat, []).append((score, pred))
    results["lexical_breakdown"] = {
        k: {
            "mean_score": round(float(np.mean([s for s, _ in v])), 4),
            "merged_rate": round(float(np.mean([pr for _, pr in v])), 4),
            "count": len(v)
        } for k, v in subcat_lex.items()
    }

    # --- NLI Semantic Evaluation ---
    if semantic_gate is not None:
        batch_pairs = [(p.text1, p.text2) for p in pairs]
        preds = semantic_gate.predict_batch(
            batch_pairs,
            entail_threshold=entail_threshold,
            contra_threshold=contra_threshold,
            abstain_min_confidence=abstain_min_confidence
        )

        y_pred_sem = [1 if pr["decision"] == "SAME" else 0 for pr in preds]
        scores_sem = [pr["p_entail"] for pr in preds]

        acc_sem = accuracy_score(y_true_binary, y_pred_sem)
        prec_sem, rec_sem, f1_sem, _ = precision_recall_fscore_support(y_true_binary, y_pred_sem, average="binary")
        auc_sem = roc_auc_score(y_true_binary, scores_sem) if len(set(y_true_binary)) > 1 else None
        cm_sem = confusion_matrix(y_true_binary, y_pred_sem).tolist()

        contra_merges_sem = sum(y_pred_sem[i] for i in contra_indices)
        false_consolidation_contra_sem = contra_merges_sem / len(contra_indices) if contra_indices else 0.0

        entail_detection_sem = rec_sem
        contra_detection_sem = sum(1 for i in contra_indices if preds[i]["decision"] == "CONTRADICTION") / len(contra_indices)
        neutral_indices = [i for i, p in enumerate(pairs) if p.ground_truth in ("DIFFERENT", "UNRELATED")]
        neutral_detection_sem = sum(1 for i in neutral_indices if preds[i]["decision"] in ("NEUTRAL", "CONTRADICTION")) / len(neutral_indices)
        false_entail_rate_sem = sum(y_pred_sem[i] for i, y in enumerate(y_true_binary) if y == 0) / (len(y_true_binary) - sum(y_true_binary))

        results["nli_gate"] = {
            "accuracy": round(acc_sem, 4),
            "precision": round(prec_sem, 4),
            "recall": round(rec_sem, 4),
            "f1": round(f1_sem, 4),
            "roc_auc": round(auc_sem, 4) if auc_sem else None,
            "confusion_matrix": cm_sem,
            "false_consolidation_of_contradictions": round(false_consolidation_contra_sem, 4),
            "contradiction_detection_rate": round(contra_detection_sem, 4),
            "entailment_detection_rate": round(entail_detection_sem, 4),
            "neutral_detection_rate": round(neutral_detection_sem, 4),
            "false_entailment_rate": round(false_entail_rate_sem, 4),
            "abstain_count": sum(1 for pr in preds if pr["decision"] == "ABSTAIN"),
        }

        # Breakdown by subcategory for NLI
        subcat_sem = {}
        for p, pr in zip(pairs, preds):
            subcat = p.subcategory
            subcat_sem.setdefault(subcat, []).append(pr)
        results["nli_breakdown"] = {
            k: {
                "mean_p_entail": round(float(np.mean([x["p_entail"] for x in v])), 4),
                "mean_p_contra": round(float(np.mean([x["p_contra"] for x in v])), 4),
                "mean_p_neutral": round(float(np.mean([x["p_neutral"] for x in v])), 4),
                "merged_rate": round(float(np.mean([1 if x["decision"] == "SAME" else 0 for x in v])), 4),
                "abstain_count": sum(1 for x in v if x["decision"] == "ABSTAIN"),
                "count": len(v)
            } for k, v in subcat_sem.items()
        }

        # Record raw probabilities
        results["raw_nli_records"] = [
            {
                "text1": p.text1,
                "text2": p.text2,
                "ground_truth": p.ground_truth,
                "subcategory": p.subcategory,
                "p_entail": round(pr["p_entail"], 5),
                "p_contra": round(pr["p_contra"], 5),
                "p_neutral": round(pr["p_neutral"], 5),
                "decision": pr["decision"]
            }
            for p, pr in zip(pairs, preds)
        ]

    return results


def memory_consolidation_metrics(run_result: dict, observations: list) -> dict:
    """Evaluates memory state consolidation quality on the aggregate stream."""
    memory = run_result["memory"]

    def nearest_slot(cand):
        x = ENCODER.encode_full(cand, dim=D)
        best_slot, best_sim = None, -1.0
        for slot in memory.slots:
            sim = cosine_similarity(x, slot.key)
            if sim > best_sim:
                best_sim, best_slot = sim, slot
        return best_slot

    base_cands = {}
    for o in observations:
        if o.category == "chain" and o.concept_id not in base_cands:
            base_cands[o.concept_id] = o.candidate

    true_opps = true_correct = 0
    false_opps = false_correct = 0
    contra_opps = contra_separated = 0
    unrel_opps = unrel_separated = 0
    para_opps = para_consolidated = 0

    for o in observations:
        if o.category == "chain":
            true_opps += 1
            b_slot = nearest_slot(base_cands[o.concept_id])
            this_slot = nearest_slot(o.candidate)
            if b_slot is this_slot:
                true_correct += 1
                para_consolidated += 1
            para_opps += 1
        elif o.category == "hard_negative":
            if o.probes_against not in base_cands:
                continue
            contra_opps += 1
            false_opps += 1
            b_slot = nearest_slot(base_cands[o.probes_against])
            this_slot = nearest_slot(o.candidate)
            if b_slot is not this_slot:
                false_correct += 1
                contra_separated += 1
        elif o.category == "unrelated":
            if o.probes_against not in base_cands:
                continue
            unrel_opps += 1
            false_opps += 1
            b_slot = nearest_slot(base_cands[o.probes_against])
            this_slot = nearest_slot(o.candidate)
            if b_slot is not this_slot:
                false_correct += 1
                unrel_separated += 1

    drifts = [float(cosine_similarity(slot.key, slot.value)) for slot in memory.slots]
    mean_drift = round(float(np.mean(drifts)), 4) if drifts else 1.0

    rejected_count = run_result.get("gate_counts", {}).get("rejected_contra", 0) + \
                     run_result.get("gate_counts", {}).get("rejected_diff", 0)

    return dict(
        system=run_result["system"],
        final_slots=len(memory.slots),
        n_updates=run_result["n_updates"],
        n_inserts=run_result["n_inserts"],
        n_evicts=run_result["n_evicts"],
        rejected_candidates=rejected_count,
        gate_counts=run_result.get("gate_counts", {}),
        true_consolidation_rate=round(true_correct / true_opps, 4) if true_opps else 0.0,
        false_consolidation_rate=round(1.0 - false_correct / false_opps, 4) if false_opps else 0.0,
        contradiction_retention=round(contra_separated / contra_opps, 4) if contra_opps else 1.0,
        paraphrase_consolidation=round(para_consolidated / para_opps, 4) if para_opps else 0.0,
        unrelated_separation=round(unrel_separated / unrel_opps, 4) if unrel_opps else 1.0,
        mean_slot_key_value_similarity=mean_drift,
        wall_seconds=round(run_result["wall_seconds"], 2)
    )


def run_contamination_chains_experiment(specs: list, systems: list,
                                        semantic_gate: NliSemanticGate = None) -> dict:
    """Evaluates contamination chains using the corrected Phase 15 bounded amplification metric."""
    contra_position = {
        "canonical_para_contra": 2, "para_canonical_contra": 2,
        "contra_para_canonical": 0, "canonical_contra_para": 1,
    }

    results = {
        s: dict(initial_errors=0, step3_merges=0, key_sims=[], val_sims=[])
        for s in systems
    }

    for spec in specs:
        v = spec["variant"]
        if v not in contra_position:
            continue
        pos = contra_position[v]
        canonical_cand = Candidate(spec["subject"], spec["predicate"], spec["object"])
        canonical_vec = ENCODER.encode_full(canonical_cand, dim=D)

        for sysname in systems:
            mem = MemoryState(capacity=20, dim=D)
            slot_texts = {}
            slot_concept_ids = {}
            trace = []

            for c in spec["chain"]:
                if sysname == "A_lexical":
                    r = lexical_absorb(mem, c, threshold=MATCH_THRESHOLD, encoder=ENCODER, dim=D)
                elif sysname == "B_nli_gate":
                    r = nli_gated_absorb(mem, c, slot_texts, semantic_gate, dim=D)
                elif sysname == "C_oracle":
                    cid = spec["concept_id"]
                    cat = "contradiction" if c.predicate == spec["chain"][pos].predicate and pos in (0, 1, 2) else "chain"
                    r = oracle_absorb(mem, c, slot_concept_ids, cid, cat, dim=D)
                elif sysname == "D_exhaustive_nli":
                    r = exhaustive_nli_absorb(mem, c, slot_texts, semantic_gate, dim=D)
                slot_obj = mem.slots[r["slot_index"]] if r.get("slot_index") is not None else None
                trace.append(dict(action=r["action"], slot=slot_obj, gate=r.get("gate")))

            contra_step = trace[pos]
            initial_error = (contra_step["action"] == "update")
            if initial_error:
                results[sysname]["initial_errors"] += 1
                contaminated_slot = contra_step["slot"]
                step3 = trace[3]
                if step3["action"] == "update" and step3["slot"] is contaminated_slot:
                    results[sysname]["step3_merges"] += 1

            best_slot = max(mem.slots, key=lambda sl: cosine_similarity(canonical_vec, sl.key))
            k_sim = float(cosine_similarity(best_slot.key, canonical_vec))
            v_sim = float(cosine_similarity(best_slot.value, canonical_vec))
            results[sysname]["key_sims"].append(k_sim)
            results[sysname]["val_sims"].append(v_sim)

    summary = {}
    for sysname, d in results.items():
        init_e = d["initial_errors"]
        summary[sysname] = {
            "initial_contradiction_merges": init_e,
            "initial_error_rate": round(init_e / 80, 4), # 80 chains contain contradictions
            "bounded_head_step3_merges": d["step3_merges"],
            "bounded_amplification_ratio": round(d["step3_merges"] / init_e, 4) if init_e else 0.0,
            "mean_recovery_key_sim": round(float(np.mean(d["key_sims"])), 4),
            "mean_recovery_val_sim": round(float(np.mean(d["val_sims"])), 4),
        }
    return summary
