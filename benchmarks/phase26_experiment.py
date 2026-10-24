"""
benchmarks/phase26_experiment.py

Phase 26 Controlled Experiment Runner.
Executes Experiments A, B, C, D (End-to-End streaming), and E (Held-Out Evaluation).
Saves comprehensive machine-readable results to scratch/phase26_results.json.
"""

from __future__ import annotations
import json
import time
from typing import Dict, Any, List

from benchmarks.phase26_dataset import get_dataset_splits
from benchmarks.phase26_metrics import compute_verification_metrics
from benchmarks.phase26_verification import Phase26Verifier
from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate


def run_experiment_a_grid(verifier: Phase26Verifier, cal_pairs: List[Any]) -> Dict[str, Any]:
    """
    Experiment A: NLI Threshold Grid Sweep on Calibration Set.
    Sweeps:
      - tau_entail in [0.50, 0.55, 0.60, 0.65]
      - asym_margin in [0.10, 0.15, 0.20]
    Under both baseline and calibrated contradiction aggregation.
    """
    print("\n--- Running Experiment A: Threshold Grid Sweep (Calibration Split) ---")
    raw_pairs = [(p.premise, p.hypothesis) for p in cal_pairs]

    tau_entails = [0.50, 0.55, 0.60, 0.65]
    asym_margins = [0.10, 0.15, 0.20]

    grid_results = []
    best_config = None
    best_score = -1.0

    # 1. Baseline Phase 25 performance on calibration
    base_preds = verifier.verify_batch(raw_pairs, mode="baseline")
    base_metrics = compute_verification_metrics(cal_pairs, base_preds)
    print(f"  [Baseline P25] ParaAcc={base_metrics['paraphrase_acceptance_rate']*100:.1f}%, "
          f"ContraDet={base_metrics['contradiction_detection_rate']*100:.1f}%, "
          f"FalseMerge={base_metrics['overall_false_merge_rate']*100:.1f}%, "
          f"MacroAcc={base_metrics['macro_accuracy']*100:.1f}%")

    # 2. Sweep calibrated mode
    for t_e in tau_entails:
        for m_a in asym_margins:
            preds = verifier.verify_batch(raw_pairs, mode="calibrated", tau_entail=t_e, asym_margin=m_a)
            m = compute_verification_metrics(cal_pairs, preds)

            record = {
                "tau_entail": t_e,
                "asym_margin": m_a,
                "metrics": m
            }
            grid_results.append(record)

            # Objective: Maximize macro accuracy subject to:
            # 1. Zero false merge on genuine contradictions
            # 2. Zero false merge on unrelated statements
            # 3. Contradiction detection >= 0.90
            is_safe = (
                m["contradiction_false_merge_rate"] == 0.0 and
                m["unrelated_false_merge_rate"] == 0.0 and
                m["contradiction_detection_rate"] >= 0.90
            )
            score = (m["macro_accuracy"] - m["overall_false_merge_rate"]) if is_safe else -1.0

            if score > best_score:
                best_score = score
                best_config = {"tau_entail": t_e, "asym_margin": m_a, "metrics": m}

    if best_config is None:
        # Fallback to safest configuration with lowest contradiction false merge
        safest = min(grid_results, key=lambda r: (r["metrics"]["contradiction_false_merge_rate"], -r["metrics"]["macro_accuracy"]))
        best_config = {"tau_entail": safest["tau_entail"], "asym_margin": safest["asym_margin"], "metrics": safest["metrics"]}

    print(f"  [Best Calibrated] tau_entail={best_config['tau_entail']}, asym_margin={best_config['asym_margin']} -> "
          f"ParaAcc={best_config['metrics']['paraphrase_acceptance_rate']*100:.1f}%, "
          f"ContraDet={best_config['metrics']['contradiction_detection_rate']*100:.1f}%, "
          f"FalseMerge={best_config['metrics']['overall_false_merge_rate']*100:.1f}%, "
          f"MacroAcc={best_config['metrics']['macro_accuracy']*100:.1f}%")


    return {
        "baseline_p25_metrics": base_metrics,
        "best_calibrated_config": best_config,
        "grid_results": grid_results
    }


def run_experiment_b_fallback(verifier: Phase26Verifier, cal_pairs: List[Any]) -> Dict[str, Any]:
    """
    Experiment B: Dense Similarity Fallback Sweep on Calibration Set.
    Sweeps tau_dense in [0.86, 0.88, 0.90, 0.92, 0.94].
    """
    print("\n--- Running Experiment B: Dense Fallback Sweep (Calibration Split) ---")
    raw_pairs = [(p.premise, p.hypothesis) for p in cal_pairs]

    tau_denses = [0.86, 0.88, 0.90, 0.92, 0.94]
    sweep_results = []
    best_config = None
    best_score = -1.0

    for t_d in tau_denses:
        preds = verifier.verify_batch(raw_pairs, mode="dense_fallback", tau_dense=t_d)
        m = compute_verification_metrics(cal_pairs, preds)
        rec = {"tau_dense": t_d, "metrics": m}
        sweep_results.append(rec)

        print(f"  [Fallback tau_dense={t_d}] ParaAcc={m['paraphrase_acceptance_rate']*100:.1f}%, "
              f"ContraDet={m['contradiction_detection_rate']*100:.1f}%, "
              f"FalseMerge={m['overall_false_merge_rate']*100:.1f}%, "
              f"MacroAcc={m['macro_accuracy']*100:.1f}%")

        is_safe = (m["contradiction_false_merge_rate"] == 0.0 and m["unrelated_false_merge_rate"] == 0.0)
        score = (m["macro_accuracy"] - m["overall_false_merge_rate"]) if is_safe else -1.0
        if score > best_score:
            best_score = score
            best_config = rec

    if best_config is None:
        best_config = min(sweep_results, key=lambda r: (r["metrics"]["contradiction_false_merge_rate"], -r["metrics"]["macro_accuracy"]))


    return {
        "best_fallback_config": best_config,
        "sweep_results": sweep_results
    }


def run_experiment_c_validation(
    verifier: Phase26Verifier,
    val_pairs: List[Any],
    best_cal: Dict[str, Any],
    best_fb: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Experiment C: Combined Verification Evaluation on Validation Set.
    Compares:
      1. baseline
      2. calibrated (Exp A frozen)
      3. dense_fallback (Exp B frozen)
      4. combined (Exp A + B frozen)
    """
    print("\n--- Running Experiment C: Candidate Comparison (Validation Split) ---")
    raw_pairs = [(p.premise, p.hypothesis) for p in val_pairs]

    t_e = best_cal["tau_entail"]
    m_a = best_cal["asym_margin"]
    t_d = best_fb["tau_dense"]

    # 1. Baseline
    p_base = verifier.verify_batch(raw_pairs, mode="baseline")
    m_base = compute_verification_metrics(val_pairs, p_base)

    # 2. Calibrated
    p_cal = verifier.verify_batch(raw_pairs, mode="calibrated", tau_entail=t_e, asym_margin=m_a)
    m_cal = compute_verification_metrics(val_pairs, p_cal)

    # 3. Dense Fallback
    p_fb = verifier.verify_batch(raw_pairs, mode="dense_fallback", tau_dense=t_d)
    m_fb = compute_verification_metrics(val_pairs, p_fb)

    # 4. Combined
    p_comb = verifier.verify_batch(raw_pairs, mode="combined", tau_entail=t_e, asym_margin=m_a, tau_dense=t_d)
    m_comb = compute_verification_metrics(val_pairs, p_comb)

    for label, m in [("Baseline P25", m_base), ("Calibrated (A)", m_cal), ("Fallback (B)", m_fb), ("Combined (C)", m_comb)]:
        print(f"  [{label}] ParaAcc={m['paraphrase_acceptance_rate']*100:.1f}%, "
              f"ContraDet={m['contradiction_detection_rate']*100:.1f}%, "
              f"FalseMerge={m['overall_false_merge_rate']*100:.1f}%, "
              f"MacroAcc={m['macro_accuracy']*100:.1f}%")

    return {
        "validation_baseline": m_base,
        "validation_calibrated": m_cal,
        "validation_fallback": m_fb,
        "validation_combined": m_comb
    }


def run_experiment_d_e2e_stream(
    cal_config: Dict[str, Any],
    fb_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Experiment D: Realistic End-to-End Streaming Memory Evaluation.
    Evaluates BlackHoleMemory under 4 verification configurations:
      1. baseline: Phase 25 baseline
      2. calibrated: Exp A thresholds
      3. combined: Exp C combined
    """
    print("\n--- Running Experiment D: End-to-End Streaming Memory Workload ---")

    # Define a realistic continuous stream with 24 observations
    observations = [
        # Initial facts (t=100..105)
        (Candidate("Helena", "directed the regional", "orchestral ensemble", timestamp=100.0), "init"),
        (Candidate("Bob", "holds a PhD in", "quantum physics", timestamp=101.0), "init"),
        (Candidate("Alice", "leads the engineering division at", "Nexus Labs", timestamp=102.0), "init"),
        (Candidate("Marcus", "authored an influential", "biography monograph", timestamp=103.0), "init"),
        (Candidate("Victor", "owns the majority share in", "Apex Robotics", timestamp=104.0), "init"),
        (Candidate("Clara", "founded an artificial intelligence", "enterprise", timestamp=105.0), "init"),

        # Paraphrase observations (should consolidate with existing slots)
        (Candidate("Helena", "conducted the local", "symphony orchestra", timestamp=110.0), "paraphrase"),
        (Candidate("Bob", "earned a doctorate in", "quantum physics", timestamp=111.0), "paraphrase"),
        (Candidate("Alice", "directs engineering at", "Nexus Labs", timestamp=112.0), "paraphrase"),
        (Candidate("Clara", "established an AI", "startup company", timestamp=113.0), "paraphrase"),

        # Genuine Contradictions / Negations (should NEVER falsely merge into active slots)
        (Candidate("Nathan", "serves as the lead", "database architect", timestamp=120.0), "init"),
        (Candidate("Nathan", "does not serve as the lead", "database architect", timestamp=121.0), "contra_probe"),

        # Temporal state revisions (should close old slot, open new slot)
        (Candidate("Alice", "now leads engineering at", "Apex Systems", timestamp=200.0), "temporal_revision"),
        (Candidate("Victor", "sold all equity in", "Apex Robotics", timestamp=201.0), "temporal_revision"),

        # Distractor flood (capacity pressure)
        (Candidate("DistractorA", "occurred in", "sector Alpha", timestamp=300.0), "distractor"),
        (Candidate("DistractorB", "occurred in", "sector Beta", timestamp=301.0), "distractor"),
        (Candidate("DistractorC", "occurred in", "sector Gamma", timestamp=302.0), "distractor"),
        (Candidate("DistractorD", "occurred in", "sector Delta", timestamp=303.0), "distractor"),
        (Candidate("DistractorE", "occurred in", "sector Epsilon", timestamp=304.0), "distractor"),
        (Candidate("DistractorF", "occurred in", "sector Zeta", timestamp=305.0), "distractor"),
        (Candidate("DistractorG", "occurred in", "sector Eta", timestamp=306.0), "distractor"),
        (Candidate("DistractorH", "occurred in", "sector Theta", timestamp=307.0), "distractor"),
        (Candidate("DistractorI", "occurred in", "sector Iota", timestamp=308.0), "distractor"),
        (Candidate("DistractorJ", "occurred in", "sector Kappa", timestamp=309.0), "distractor"),
    ]

    configs = {
        "baseline": {"verification_mode": "baseline", "entail_threshold": 0.60, "contra_threshold": 0.40},
        "calibrated": {"verification_mode": "calibrated", "entail_threshold": cal_config["tau_entail"], "contra_threshold": 0.40},
        "combined": {"verification_mode": "combined", "entail_threshold": cal_config["tau_entail"], "contra_threshold": 0.40}
    }

    results = {}

    for name, cfg in configs.items():
        v = Phase26Verifier(
            cal_tau_entail=cal_config["tau_entail"],
            cal_asym_margin=cal_config["asym_margin"],
            fallback_tau_dense=fb_config["tau_dense"]
        )
        mem = BlackHoleMemory(
            capacity=12,  # Constrained capacity to force tiered retention
            retention_policy="tiered",
            verification_mode=cfg["verification_mode"],
            entail_threshold=cfg["entail_threshold"],
            contra_threshold=cfg["contra_threshold"],
            consolidation_mode="hybrid",
            verifier=v
        )


        consolidations = 0
        false_merges = 0
        temporal_updates = 0
        inserts = 0
        actions = []

        for cand, obs_type in observations:
            res = mem.absorb(cand)
            act = res["action"]
            actions.append((obs_type, act))

            if act == "consolidate":
                consolidations += 1
                if obs_type in ("contra_probe", "temporal_revision", "distractor"):
                    false_merges += 1
            elif act == "temporal_update":
                temporal_updates += 1
            elif act in ("insert", "evict_insert"):
                inserts += 1

        # Post-stream queries
        # 1. Active recall on Alice should return Apex Systems (current)
        alice_now = mem.recall("Alice")
        alice_current_correct = (len(alice_now) == 1 and "Apex Systems" in alice_now[0]["text"])

        # 2. Point-in-time recall on Alice at t=150 should return Nexus Labs (past)
        alice_past = mem.recall("Alice", as_of=150.0)
        alice_past_correct = (len(alice_past) == 1 and "Nexus Labs" in alice_past[0]["text"])

        # 3. Bob PhD fact survived?
        bob_facts = mem.recall("Bob")
        bob_survived = len(bob_facts) >= 1

        # 4. History length for Alice
        alice_hist = mem.get_history("Alice")

        status = mem.status()

        res_record = {
            "name": name,
            "total_absorbed": len(observations),
            "consolidations": consolidations,
            "false_merges": false_merges,
            "temporal_updates": temporal_updates,
            "inserts": inserts,
            "alice_current_correct": alice_current_correct,
            "alice_past_correct": alice_past_correct,
            "alice_history_length": len(alice_hist),
            "bob_survived": bob_survived,
            "status": status
        }
        results[name] = res_record

        print(f"  [{name.upper()}] Consolidations={consolidations}, FalseMerges={false_merges}, "
              f"TemporalUpdates={temporal_updates}, AliceCur={alice_current_correct}, "
              f"AlicePast={alice_past_correct}, ActiveSlots={status['active_slots']}")

    return results


def run_experiment_e_held_out(
    verifier: Phase26Verifier,
    held_out_pairs: List[Any],
    best_cal: Dict[str, Any],
    best_fb: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Experiment E: Frozen Held-Out Test Set Evaluation.
    Evaluates all 4 configurations on untouched, novel-domain pairs.
    """
    print("\n--- Running Experiment E: Final Frozen Held-Out Evaluation ---")
    raw_pairs = [(p.premise, p.hypothesis) for p in held_out_pairs]

    t_e = best_cal["tau_entail"]
    m_a = best_cal["asym_margin"]
    t_d = best_fb["tau_dense"]

    # 1. Baseline
    p_base = verifier.verify_batch(raw_pairs, mode="baseline")
    m_base = compute_verification_metrics(held_out_pairs, p_base)

    # 2. Calibrated
    p_cal = verifier.verify_batch(raw_pairs, mode="calibrated", tau_entail=t_e, asym_margin=m_a)
    m_cal = compute_verification_metrics(held_out_pairs, p_cal)

    # 3. Dense Fallback
    p_fb = verifier.verify_batch(raw_pairs, mode="dense_fallback", tau_dense=t_d)
    m_fb = compute_verification_metrics(held_out_pairs, p_fb)

    # 4. Combined
    p_comb = verifier.verify_batch(raw_pairs, mode="combined", tau_entail=t_e, asym_margin=m_a, tau_dense=t_d)
    m_comb = compute_verification_metrics(held_out_pairs, p_comb)

    for label, m in [("Baseline P25", m_base), ("Calibrated (A)", m_cal), ("Fallback (B)", m_fb), ("Combined (C)", m_comb)]:
        print(f"  [HELD-OUT {label}] ParaAcc={m['paraphrase_acceptance_rate']*100:.1f}%, "
              f"ContraDet={m['contradiction_detection_rate']*100:.1f}%, "
              f"FalseMerge={m['overall_false_merge_rate']*100:.1f}%, "
              f"MacroAcc={m['macro_accuracy']*100:.1f}%")

    return {
        "held_out_baseline": m_base,
        "held_out_calibrated": m_cal,
        "held_out_fallback": m_fb,
        "held_out_combined": m_comb
    }


def run_full_phase26():
    print("=================================================================")
    print("  MINEMOS Phase 26: Controlled Research Experiments")
    print("=================================================================")

    splits = get_dataset_splits()
    cal_pairs = splits["calibration"]
    val_pairs = splits["validation"]
    held_out_pairs = splits["held_out"]

    verifier = Phase26Verifier()

    # 1. Experiment A: Calibration
    res_a = run_experiment_a_grid(verifier, cal_pairs)
    best_cal = res_a["best_calibrated_config"]

    # 2. Experiment B: Fallback
    res_b = run_experiment_b_fallback(verifier, cal_pairs)
    best_fb = res_b["best_fallback_config"]

    # 3. Experiment C: Validation Comparison
    res_c = run_experiment_c_validation(verifier, val_pairs, best_cal, best_fb)

    # 4. Experiment D: End-to-End Streaming Memory
    res_d = run_experiment_d_e2e_stream(best_cal, best_fb)

    # 5. Experiment E: Final Held-Out Evaluation
    res_e = run_experiment_e_held_out(verifier, held_out_pairs, best_cal, best_fb)

    full_results = {
        "phase": 26,
        "timestamp": time.time(),
        "experiment_a_calibration": res_a,
        "experiment_b_fallback": res_b,
        "experiment_c_validation": res_c,
        "experiment_d_e2e_streaming": res_d,
        "experiment_e_held_out": res_e
    }

    with open("scratch/phase26_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print("\n[SUCCESS] Full Phase 26 experimental results saved -> scratch/phase26_results.json")

    return full_results


if __name__ == "__main__":
    run_full_phase26()
