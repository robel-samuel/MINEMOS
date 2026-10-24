"""
benchmarks/phase24_dataset.py

Phase 24 Dataset Generators for Retention Hardening, Semantic Verification,
Drift-Resistant Consolidation, Long-Horizon Stress, and Held-Out Benchmark.

Random seeds:
  - Tuning & Diagnostic Seed: SEED_P24_TUNING = 2401
  - Sealed Held-Out Seed: SEED_P24_HELD_OUT = 2424
  (Historical Phase 23 seed 2323 remains strictly preserved in historical results).
"""

from __future__ import annotations
import os, sys, random
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate
from benchmarks.phase18_dataset import Phase18Obs, make_distractor
from benchmarks.phase23_dataset import (
    PARAPHRASE_PAIRS_P23,
    CONTRADICTION_PAIRS_P23,
)

PROGRESSIONS_P24 = [
    # (subject, [step0_p, step0_o], [step1_p, step1_o], [step2_p, step2_o], [step3_p, step3_o], probe)
    ("Arthur",
     [("assembled the modular", "computing workstation"),
      ("constructed the prefabricated", "desktop terminal"),
      ("built the customized", "processing rig"),
      ("configured the high-performance", "computational unit")],
     ("deployed the assembled", "workstation hardware")),

    ("Beatrice",
     [("analyzed the atmospheric", "methane concentrations"),
      ("evaluated the ambient", "gas emission levels"),
      ("monitored the airborne", "methane readings"),
      ("recorded the environmental", "trace emissions")],
     ("measured the atmospheric", "methane levels")),

    ("Clara",
     [("acquired the commercial", "maritime vessel"),
      ("purchased the cargo", "shipping freighter"),
      ("obtained the international", "transport boat"),
      ("secured the merchant", "fleet tanker")],
     ("bought the cargo", "maritime carrier")),

    ("David",
     [("authored the cryptographic", "security specification"),
      ("drafted the encryption", "protocol manual"),
      ("composed the defensive", "key exchange standards"),
      ("penned the authentication", "cipher documentation")],
     ("wrote the cryptographic", "security guidelines")),

    ("Eve",
     [("synthesized the inorganic", "polymer electrolyte"),
      ("manufactured the solid-state", "battery compound"),
      ("produced the advanced", "chemical storage matrix"),
      ("formulated the ionic", "electrolyte substance")],
     ("created the solid", "polymer electrolyte")),

    ("Felix",
     [("renovated the dilapidated", "railway terminal"),
      ("refurbished the historic", "train station depot"),
      ("restored the aging", "locomotive platform"),
      ("modernized the regional", "rail transit hub")],
     ("upgraded the central", "train terminal station")),

    ("Greta",
     [("discovered the celestial", "gamma-ray transient"),
      ("observed the stellar", "high-energy radiation burst"),
      ("detected the astronomical", "gamma pulsation event"),
      ("identified the cosmic", "emission flash")],
     ("found the celestial", "gamma-ray emission")),

    ("Hassan",
     [("audited the corporate", "financial statements"),
      ("inspected the fiscal", "accounting balance sheets"),
      ("reviewed the commercial", "revenue ledgers"),
      ("examined the enterprise", "expenditure books")],
     ("checked the company", "financial records")),
]


SEED_P24_TUNING = 2401
SEED_P24_HELD_OUT = 2424


# ──────────────────────────────────────────────
#  Experiment A: Practical Retention Workloads
# ──────────────────────────────────────────────

def generate_exp_a_retention_stream(
    mode: str = "ordinary",  # 'ordinary' or 'adversarial'
    seed: int = SEED_P24_TUNING
) -> list[Phase18Obs]:
    """
    Workloads for Experiment A (Retention Policies):
      - 'ordinary': 10 targets, moderate distractor gap (5 per target), total ~120 obs.
      - 'adversarial': 10 targets inserted early, 350 total obs, severe distractor flood
        into limited capacity (forcing >300 evictions), plus 10 adversarial contradiction pairs.
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    if mode == "ordinary":
        # Standard moderate workload
        for idx, (s, cp, co, pp, po) in enumerate(PARAPHRASE_PAIRS_P23):
            cid = f"ret_ord_{idx:03d}"
            stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid, "canonical", "exp_a_ord", t, is_probe=False))
            t += 1
            for _ in range(5):
                stream.append(make_distractor(t, timestamp=t))
                t += 1
            stream.append(Phase18Obs(Candidate(s, pp, po, timestamp=t), cid, "paraphrase", "exp_a_ord", t, is_probe=True))
            t += 1

        # Also add contradiction probes to verify safety
        for idx, (s, cp, co, xp, xo) in enumerate(CONTRADICTION_PAIRS_P23[:5]):
            cid = f"ret_ord_contra_{idx:03d}"
            stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid, "canonical", "exp_a_ord", t, is_probe=False))
            t += 1
            for _ in range(3):
                stream.append(make_distractor(t, timestamp=t))
                t += 1
            stream.append(Phase18Obs(Candidate(s, xp, xo, timestamp=t), cid, "contradiction", "exp_a_ord", t, is_probe=True))
            t += 1

    elif mode == "adversarial":
        # Severe capacity pressure workload (350 observations)
        targets = PARAPHRASE_PAIRS_P23[:10]
        contra_targets = CONTRADICTION_PAIRS_P23[:10]

        # 1. Insert canonical targets early
        for idx, (s, cp, co, _, _) in enumerate(targets):
            cid = f"ret_adv_{idx:03d}"
            stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid, "canonical", "exp_a_adv", t, is_probe=False))
            t += 1

        for idx, (s, cp, co, _, _) in enumerate(contra_targets):
            cid = f"ret_adv_c_{idx:03d}"
            stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid, "canonical", "exp_a_adv", t, is_probe=False))
            t += 1

        # Heavy distractor flood interleaved with delayed probes
        distractors_per_round = 12
        for idx, (s, _, _, pp, po) in enumerate(targets):
            cid = f"ret_adv_{idx:03d}"
            for _ in range(distractors_per_round):
                stream.append(make_distractor(t, timestamp=t))
                t += 1
            stream.append(Phase18Obs(Candidate(s, pp, po, timestamp=t), cid, "paraphrase", "exp_a_adv", t, is_probe=True))
            t += 1

        for idx, (s, _, _, xp, xo) in enumerate(contra_targets):
            cid = f"ret_adv_c_{idx:03d}"
            for _ in range(distractors_per_round):
                stream.append(make_distractor(t, timestamp=t))
                t += 1
            stream.append(Phase18Obs(Candidate(s, xp, xo, timestamp=t), cid, "contradiction", "exp_a_adv", t, is_probe=True))
            t += 1

        while len(stream) < 350:
            stream.append(make_distractor(t, timestamp=t))
            t += 1
    else:
        raise ValueError(f"Unknown retention workload mode: {mode}")

    return stream


# ──────────────────────────────────────────────
#  Experiment B: Semantic Verification Calibration Stream
# ──────────────────────────────────────────────

def generate_exp_b_calibration_stream(seed: int = SEED_P24_TUNING) -> list[Phase18Obs]:
    """Generates balanced evaluation stream for threshold sweep and calibration."""
    rng = random.Random(seed)
    stream = []
    t = 0

    # 10 paraphrase pairs
    for idx, (s, cp, co, pp, po) in enumerate(PARAPHRASE_PAIRS_P23):
        cid = f"calib_para_{idx:03d}"
        stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid, "canonical", "exp_b_calib", t, is_probe=False))
        t += 1
        for _ in range(4):
            stream.append(make_distractor(t, timestamp=t))
            t += 1
        stream.append(Phase18Obs(Candidate(s, pp, po, timestamp=t), cid, "paraphrase", "exp_b_calib", t, is_probe=True))
        t += 1

    # 10 contradiction pairs
    for idx, (s, cp, co, xp, xo) in enumerate(CONTRADICTION_PAIRS_P23):
        cid = f"calib_contra_{idx:03d}"
        stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid, "canonical", "exp_b_calib", t, is_probe=False))
        t += 1
        for _ in range(4):
            stream.append(make_distractor(t, timestamp=t))
            t += 1
        stream.append(Phase18Obs(Candidate(s, xp, xo, timestamp=t), cid, "contradiction", "exp_b_calib", t, is_probe=True))
        t += 1

    return stream


# ──────────────────────────────────────────────
#  Experiment C: Drift-Resistant Consolidation Stream
# ──────────────────────────────────────────────

def generate_exp_c_drift_stream(
    n_targets: int = 8,
    updates_per_target: int = 4,
    seed: int = SEED_P24_TUNING
) -> list[Phase18Obs]:
    """
    Sequential multi-update stream:
    Each target undergoes updates_per_target successive consolidations
    separated by distractors, followed by a final evaluation probe.
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    for idx, (subj, steps, (probe_p, probe_o)) in enumerate(PROGRESSIONS_P24[:n_targets]):
        cid = f"p24_drift_{idx:03d}"

        # Canonical Step 0
        stream.append(Phase18Obs(Candidate(subj, steps[0][0], steps[0][1], timestamp=t), cid, "canonical", "exp_c_drift", t, is_probe=False))
        t += 1

        # Multi-update sequential consolidations
        for u_idx, (up, uo) in enumerate(steps[1:updates_per_target]):
            for _ in range(3):
                stream.append(make_distractor(t, timestamp=t))
                t += 1
            stream.append(Phase18Obs(Candidate(subj, up, uo, timestamp=t), cid, "paraphrase", "exp_c_drift", t, is_probe=False))
            t += 1

        # Distractor buffer before evaluation probe
        for _ in range(4):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # Final paraphrase probe
        stream.append(Phase18Obs(Candidate(subj, probe_p, probe_o, timestamp=t), cid, "paraphrase", "exp_c_drift", t, is_probe=True))
        t += 1

    return stream


# ──────────────────────────────────────────────
#  Experiment D: Combined Ablation Stream
# ──────────────────────────────────────────────

def generate_exp_d_ablation_stream(seed: int = SEED_P24_TUNING) -> list[Phase18Obs]:
    """Unified 250-observation stream balancing retention, verification, and drift challenges."""
    rng = random.Random(seed)
    stream = []
    t = 0

    # 8 targets with 2 updates each + 8 contradiction challenges + distractors
    for idx, (subj, steps, (probe_p, probe_o)) in enumerate(PROGRESSIONS_P24[:8]):
        cid = f"abl_tgt_{idx:03d}"
        # Step 0
        stream.append(Phase18Obs(Candidate(subj, steps[0][0], steps[0][1], timestamp=t), cid, "canonical", "exp_d_abl", t, is_probe=False))
        t += 1
        for _ in range(3):
            stream.append(make_distractor(t, timestamp=t))
            t += 1
        # Step 1 update
        stream.append(Phase18Obs(Candidate(subj, steps[1][0], steps[1][1], timestamp=t), cid, "paraphrase", "exp_d_abl", t, is_probe=False))
        t += 1
        for _ in range(5):
            stream.append(make_distractor(t, timestamp=t))
            t += 1
        # Evaluation probe
        stream.append(Phase18Obs(Candidate(subj, probe_p, probe_o, timestamp=t), cid, "paraphrase", "exp_d_abl", t, is_probe=True))
        t += 1

    # Add 8 contradiction pairs to ensure contradiction safety
    for idx, (s, cp, co, xp, xo) in enumerate(CONTRADICTION_PAIRS_P23[:8]):
        cid = f"abl_contra_{idx:03d}"
        stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid, "canonical", "exp_d_abl", t, is_probe=False))
        t += 1
        for _ in range(4):
            stream.append(make_distractor(t, timestamp=t))
            t += 1
        stream.append(Phase18Obs(Candidate(s, xp, xo, timestamp=t), cid, "contradiction", "exp_d_abl", t, is_probe=True))
        t += 1

    # Fill to 250 obs with distractors
    while len(stream) < 250:
        stream.append(make_distractor(t, timestamp=t))
        t += 1

    return stream


# ──────────────────────────────────────────────
#  Experiment E: Long Horizon and Capacity Stress
# ──────────────────────────────────────────────

def generate_exp_e_stress_stream(n_obs: int = 1000, seed: int = SEED_P24_TUNING) -> list[Phase18Obs]:
    """
    Stress stream scaling to long horizons (e.g. 500, 1000, 2500, 5000, 10000).
    Maintains constant ratio of targets (canonical + update + probe + contra) surrounded
    by distractor floods.
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    topics = PROGRESSIONS_P24
    n_topics = len(topics)

    # Insert target cycles
    target_interval = max(20, n_obs // (n_topics * 3))

    for idx in range(n_topics):
        subj, steps, (probe_p, probe_o) = topics[idx]
        cid = f"stress_tgt_{idx:03d}"

        # 1. Canonical insert
        stream.append(Phase18Obs(Candidate(subj, steps[0][0], steps[0][1], timestamp=t), cid, "canonical", "exp_e_stress", t, is_probe=False))
        t += 1

        for _ in range(target_interval // 2):
            if len(stream) < n_obs:
                stream.append(make_distractor(t, timestamp=t))
                t += 1

        # 2. Consolidation update
        if len(stream) < n_obs:
            stream.append(Phase18Obs(Candidate(subj, steps[1][0], steps[1][1], timestamp=t), cid, "paraphrase", "exp_e_stress", t, is_probe=False))
            t += 1

        for _ in range(target_interval):
            if len(stream) < n_obs:
                stream.append(make_distractor(t, timestamp=t))
                t += 1

        # 3. Final probe
        if len(stream) < n_obs:
            stream.append(Phase18Obs(Candidate(subj, probe_p, probe_o, timestamp=t), cid, "paraphrase", "exp_e_stress", t, is_probe=True))
            t += 1

    # Fill remainder to n_obs
    while len(stream) < n_obs:
        stream.append(make_distractor(t, timestamp=t))
        t += 1

    return stream[:n_obs]


# ──────────────────────────────────────────────
#  Experiment F: Sealed Held-Out Benchmark (seed=2424)
# ──────────────────────────────────────────────

HELD_OUT_DOMAINS_P24 = [
    # (Subj, Canon_P, Canon_O, Para_P, Para_O, Contra_P, Contra_O)
    ("Selene", "spectroscopically identified atmospheric", "phosphine on exoplanet K2-18b",
     "detected chemical signatures of", "phosphine gas in alien atmosphere",
     "disproved the presence of", "phosphine gas in alien atmosphere"),

    ("Orion", "engineered the neuromorphic", "memristor crossbar array",
     "built the bio-inspired", "analog synaptic computing hardware",
     "scrapped all research on", "analog synaptic computing hardware"),

    ("Thalassa", "drilled into the ultra-deep", "Mariana abyssal peridotite mantle",
     "extracted rock cores from the", "hadopelagic oceanic crust",
     "halted deep drilling in the", "hadopelagic oceanic crust"),

    ("Vesper", "synthesized the artificial", "carbon-fixing enzyme cascade",
     "created the synthetic biological", "co2 assimilation pathway",
     "reported failure in creating the", "co2 assimilation pathway"),

    ("Ignis", "mapped the subterranean", "supercritical geothermal reservoir",
     "imaged the ultra-hot magma", "thermal energy formation",
     "denied the existence of the", "thermal energy formation"),

    ("Zephyr", "stabilized the levitating", "aerogel acoustic metamaterial",
     "demonstrated acoustic levitation of", "ultralight insulating solids",
     "failed to achieve levitation of", "ultralight insulating solids"),

    ("Solas", "cryogenically preserved the", "quantum entangled photon source",
     "maintained optical coherence in", "supercooled entangled light emitters",
     "observed decoherence destroying", "supercooled entangled light emitters"),

    ("Danube", "reconstructed the paleo-drainage", "fluvial river basin network",
     "modeled the prehistoric", "continental river migration path",
     "rejected hydrological models of", "continental river migration path"),
]


def generate_held_out_phase24(n_obs: int = 800, seed: int = SEED_P24_HELD_OUT) -> list[Phase18Obs]:
    """
    Exp F: Sealed Held-Out Benchmark for Phase 24 (seed=2424).
    Completely novel domains never seen during prompt engineering or parameter tuning.
    Includes unseen vocabulary, hard paraphrases, contradictions, and temporal fact shifts.
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    topics = HELD_OUT_DOMAINS_P24
    obs_per_topic = 30
    max_topics = max(1, min(len(topics), n_obs // obs_per_topic))
    topics_subset = topics[:max_topics]
    gap = max(4, (n_obs - (len(topics_subset) * 4)) // (len(topics_subset) * 3))

    for idx, (s, cp, co, pp, po, xp, xo) in enumerate(topics_subset):
        cid_p = f"p24_ho_para_{idx:03d}"
        cid_c = f"p24_ho_contra_{idx:03d}"

        # 1. Paraphrase target: Canonical insert
        stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid_p, "canonical", "p24_ho", t, is_probe=False))
        t += 1

        for _ in range(gap):
            if len(stream) < n_obs:
                stream.append(make_distractor(t, timestamp=t))
                t += 1

        # 2. Paraphrase target: Consolidating update
        if len(stream) < n_obs:
            stream.append(Phase18Obs(Candidate(s, pp, po, timestamp=t), cid_p, "paraphrase", "p24_ho", t, is_probe=False))
            t += 1

        for _ in range(gap):
            if len(stream) < n_obs:
                stream.append(make_distractor(t, timestamp=t))
                t += 1

        # 3. Paraphrase target: Final probe
        if len(stream) < n_obs:
            stream.append(Phase18Obs(Candidate(s, pp, po, timestamp=t), cid_p, "paraphrase", "p24_ho", t, is_probe=True))
            t += 1

        # 4. Contradiction target: Canonical insert
        if len(stream) < n_obs:
            stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid_c, "canonical", "p24_ho", t, is_probe=False))
            t += 1

        for _ in range(gap):
            if len(stream) < n_obs:
                stream.append(make_distractor(t, timestamp=t))
                t += 1

        # 5. Contradiction target: Probe
        if len(stream) < n_obs:
            stream.append(Phase18Obs(Candidate(s, xp, xo, timestamp=t), cid_c, "contradiction", "p24_ho", t, is_probe=True))
            t += 1

    while len(stream) < n_obs:
        stream.append(make_distractor(t, timestamp=t))
        t += 1

    return stream[:n_obs]
