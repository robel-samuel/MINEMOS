"""
benchmarks/phase23_dataset.py

Phase 23 Dataset Generators for Consolidation & Retention Bottleneck Investigation.

Generators:
  1. generate_exp1_threshold_sweep: Paraphrase + contradiction stream for NLI sweep.
  2. generate_exp2_oracle_retrieval: Stream for testing Oracle Retrieval condition.
  3. generate_exp3_retention_oracle: High-eviction stream for Retention Oracle.
  4. generate_exp4_matrix_stream: Unified stream for Consolidation vs Retention 2x2.
  5. generate_exp5_embedding_drift_stream: Sequential multi-update stream for drift.
  6. generate_exp6_trace_stream: Curated stream for detailed step-by-step traces.
  7. generate_held_out_phase23: Sealed held-out benchmark (seed=2323).
"""

from __future__ import annotations
import os, sys, random
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate
from benchmarks.phase18_dataset import Phase18Obs, make_distractor

SEED_P23 = 2323


# ──────────────────────────────────────────────
#  Curated Semantic Pairs for Sweep & Diagnostics
# ──────────────────────────────────────────────

PARAPHRASE_PAIRS_P23 = [
    # (subject, canon_p, canon_o, para_p, para_o)
    ("Helena", "directed the regional", "orchestral ensemble", "conducted the local", "symphony orchestra"),
    ("Marcus", "authored an influential", "biography monograph", "penned a notable", "historical memoir"),
    ("Elena", "navigated the remote", "mountainous terrain", "traversed the secluded", "alpine wilderness"),
    ("Julian", "engineered a redundant", "power distribution grid", "designed a fault-tolerant", "electrical supply network"),
    ("Amara", "discovered a prehistoric", "fossilized specimen", "unearthed an ancient", "paleontological artifact"),
    ("Klaus", "inspected the primary", "containment vessel", "examined the central", "storage chamber"),
    ("Soraya", "negotiated the multilateral", "trade accord", "brokered the international", "commercial treaty"),
    ("Dmitri", "calibrated the precision", "spectrometer instrument", "adjusted the sensitive", "optical sensor device"),
    ("Lila", "founded the nonprofit", "wildlife sanctuary", "established the charitable", "nature preserve"),
    ("Chen", "optimized the distributed", "consensus protocol", "improved the decentralized", "agreement algorithm"),
]

CONTRADICTION_PAIRS_P23 = [
    # (subject, canon_p, canon_o, contra_p, contra_o)
    ("Nathan", "serves as the lead", "database architect", "does not serve as the lead", "database architect"),
    ("Beatrice", "resides in the northern", "coastal district", "does not reside in the northern", "coastal district"),
    ("Victor", "owns the majority share in", "Apex Robotics", "sold all equity in", "Apex Robotics"),
    ("Nadia", "authorized the emergency", "system shutdown", "strictly forbade the emergency", "system shutdown"),
    ("Gideon", "published the classified", "financial ledger", "withheld the confidential", "financial ledger"),
    ("Miriam", "validated the cryptographic", "access key", "revoked the cryptographic", "access key"),
    ("Tariq", "promoted the senior", "security officer", "demoted the senior", "security officer"),
    ("Valerie", "purchased the historical", "downtown property", "divested the historical", "downtown property"),
    ("Liam", "repaired the damaged", "propulsion turbine", "dismantled the broken", "propulsion turbine"),
    ("Zara", "upgraded the operational", "telecom relay", "decommissioned the operational", "telecom relay"),
]


def generate_exp1_threshold_sweep(inter_gap: int = 5, seed: int = SEED_P23) -> list[Phase18Obs]:
    """Exp 1: Stream testing NLI threshold sweep (0.55 - 0.85) on paraphrase vs contradiction."""
    rng = random.Random(seed)
    stream = []
    t = 0

    # 1. Paraphrase pairs
    for idx, (s, cp, co, pp, po) in enumerate(PARAPHRASE_PAIRS_P23):
        cid = f"p23_para_{idx:03d}"
        stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid, "canonical", "exp1_sweep", t, is_probe=False))
        t += 1
        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1
        stream.append(Phase18Obs(Candidate(s, pp, po, timestamp=t), cid, "paraphrase", "exp1_sweep", t, is_probe=True))
        t += 1

    # 2. Contradiction pairs
    for idx, (s, cp, co, xp, xo) in enumerate(CONTRADICTION_PAIRS_P23):
        cid = f"p23_contra_{idx:03d}"
        stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid, "canonical", "exp1_sweep", t, is_probe=False))
        t += 1
        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1
        stream.append(Phase18Obs(Candidate(s, xp, xo, timestamp=t), cid, "contradiction", "exp1_sweep", t, is_probe=True))
        t += 1

    return stream


def generate_exp2_oracle_retrieval(n_targets: int = 12, inter_gap: int = 8, seed: int = SEED_P23) -> list[Phase18Obs]:
    """Exp 2: Stream for testing Oracle Retrieval (forcible target injection)."""
    rng = random.Random(seed)
    stream = []
    t = 0

    pairs = PARAPHRASE_PAIRS_P23[:n_targets]
    for idx, (s, cp, co, pp, po) in enumerate(pairs):
        cid = f"oracle_ret_{idx:03d}"
        stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid, "canonical", "exp2_oracle", t, is_probe=False))
        t += 1
        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1
        stream.append(Phase18Obs(Candidate(s, pp, po, timestamp=t), cid, "paraphrase", "exp2_oracle", t, is_probe=True))
        t += 1

    return stream


def generate_exp3_retention_oracle(n_targets: int = 10, total_obs: int = 400, seed: int = SEED_P23) -> list[Phase18Obs]:
    """
    Exp 3: Stream with high eviction pressure where targets are inserted early
    and probed late after hundreds of distractors.
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    targets = PARAPHRASE_PAIRS_P23[:n_targets]
    # Insert all canonical targets early
    for idx, (s, cp, co, _, _) in enumerate(targets):
        cid = f"retention_tgt_{idx:03d}"
        stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid, "canonical", "exp3_retention", t, is_probe=False))
        t += 1

    # High distractor flood
    remaining = total_obs - len(targets) - len(targets)
    distractors_per_round = remaining // len(targets)

    for idx, (s, _, _, pp, po) in enumerate(targets):
        cid = f"retention_tgt_{idx:03d}"
        for _ in range(distractors_per_round):
            stream.append(make_distractor(t, timestamp=t))
            t += 1
        stream.append(Phase18Obs(Candidate(s, pp, po, timestamp=t), cid, "paraphrase", "exp3_retention", t, is_probe=True))
        t += 1

    while len(stream) < total_obs:
        stream.append(make_distractor(t, timestamp=t))
        t += 1

    return stream


def generate_exp4_matrix_stream(seed: int = SEED_P23) -> list[Phase18Obs]:
    """Exp 4: 2x2 matrix evaluation stream (Ret/Retention oracles)."""
    return generate_exp3_retention_oracle(n_targets=10, total_obs=350, seed=seed)


def generate_exp5_embedding_drift_stream(n_targets: int = 8, updates_per_target: int = 4, seed: int = SEED_P23) -> list[Phase18Obs]:
    """
    Exp 5: Sequential multi-update stream to track embedding drift across successive consolidations.
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    # Multi-step paraphrase progressions
    PROGRESSIONS = [
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

    for idx, (subj, steps, (probe_p, probe_o)) in enumerate(PROGRESSIONS[:n_targets]):
        cid = f"drift_tgt_{idx:03d}"

        # 1. Step 0: Canonical
        stream.append(Phase18Obs(Candidate(subj, steps[0][0], steps[0][1], timestamp=t), cid, "canonical", "exp5_drift", t, is_probe=False))
        t += 1

        # Sequential updates with small distractor gaps
        for u_idx, (up, uo) in enumerate(steps[1:updates_per_target]):
            for _ in range(3):
                stream.append(make_distractor(t, timestamp=t))
                t += 1
            stream.append(Phase18Obs(Candidate(subj, up, uo, timestamp=t), cid, "paraphrase", "exp5_drift", t, is_probe=False))
            t += 1

        # Intervening gap before final probe
        for _ in range(5):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # Final probe
        stream.append(Phase18Obs(Candidate(subj, probe_p, probe_o, timestamp=t), cid, "paraphrase", "exp5_drift", t, is_probe=True))
        t += 1

    return stream


def generate_exp6_trace_stream(seed: int = SEED_P23) -> list[Phase18Obs]:
    """
    Exp 6: Curated compact stream for step-by-step trace generation:
      - 1 successful paraphrase
      - 1 hard paraphrase
      - 1 contradiction
      - 1 long-gap retention case
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    # Target 1: Paraphrase
    cid1 = "trace_para_001"
    stream.append(Phase18Obs(Candidate("Alice", "developed the automated", "quality assurance pipeline", timestamp=t), cid1, "canonical", "exp6_trace", t, is_probe=False))
    t += 1

    # Target 2: Contradiction
    cid2 = "trace_contra_002"
    stream.append(Phase18Obs(Candidate("Bob", "supervises the overseas", "manufacturing facility", timestamp=t), cid2, "canonical", "exp6_trace", t, is_probe=False))
    t += 1

    # Distractors
    for _ in range(4):
        stream.append(make_distractor(t, timestamp=t))
        t += 1

    # Probe 1: Paraphrase probe
    stream.append(Phase18Obs(Candidate("Alice", "created the automated", "testing workflow pipeline", timestamp=t), cid1, "paraphrase", "exp6_trace", t, is_probe=True))
    t += 1

    # Probe 2: Contradiction probe
    stream.append(Phase18Obs(Candidate("Bob", "does not supervise the overseas", "manufacturing facility", timestamp=t), cid2, "contradiction", "exp6_trace", t, is_probe=True))
    t += 1

    # Target 3: Long-horizon target
    cid3 = "trace_long_003"
    stream.append(Phase18Obs(Candidate("Charlie", "discovered the rare", "mineral outcrop", timestamp=t), cid3, "canonical", "exp6_trace", t, is_probe=False))
    t += 1

    # Long distractor gap
    for _ in range(30):
        stream.append(make_distractor(t, timestamp=t))
        t += 1

    # Probe 3: Long-horizon probe
    stream.append(Phase18Obs(Candidate("Charlie", "located the unusual", "geological rock formation", timestamp=t), cid3, "paraphrase", "exp6_trace", t, is_probe=True))
    t += 1

    return stream


def generate_held_out_phase23(n_obs: int = 1200, seed: int = SEED_P23) -> list[Phase18Obs]:
    """
    Exp 7: Sealed Held-Out Benchmark for Phase 23 (seed=2323).
    Features completely novel domains:
      - Marine Biology
      - Quantum Computing
      - Space Exploration
      - Architecture & Urban Planning
      - Renewable Energy
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    HELD_OUT_TOPICS = [
        # (Subj, Canon_P, Canon_O, Para_P, Para_O, Contra_P, Contra_O)
        ("Aurelia", "catalogued the bioluminescent", "deep-sea cnidarian",
         "documented the glowing", "abyssal jellyfish species",
         "denied the existence of the", "abyssal jellyfish species"),
        ("Tetsuo", "fabricated the superconducting", "flux qubit circuit",
         "constructed the cryogenic", "quantum computing processor",
         "dismantled the cryogenic", "quantum computing processor"),
        ("Cassian", "charted the uncharted", "martian subterranean lava tube",
         "mapped the unexplored", "volcanic caves on Mars",
         "refuted all claims of", "volcanic caves on Mars"),
        ("Zainab", "designed the bioclimatic", "passive ventilation tower",
         "architected the sustainable", "natural airflow building",
         "condemned the architectural", "natural airflow building"),
        ("Leopold", "patented the perovskite", "photovoltaic tandem cell",
         "registered the invention for", "high-efficiency solar layers",
         "abandoned all patent rights to", "high-efficiency solar layers"),
        ("Freja", "sequenced the extremophile", "hydrothermal archaea genome",
         "decoded the genetic structure of", "deep ocean bacteria",
         "contaminated the sample of", "deep ocean bacteria"),
        ("Rami", "stabilized the magnetic", "tokamak plasma column",
         "confined the high-temperature", "fusion reactor core",
         "lost magnetic control of the", "fusion reactor core"),
        ("Kavita", "modeled the catastrophic", "glacier calving collapse",
         "simulated the massive", "polar ice sheet detachment",
         "disproved the theory of", "polar ice sheet detachment"),
    ]

    # Determine how many topics to include based on n_obs
    # Each topic needs: 1 canon + 1 probe (para) + 1 canon + 1 probe (contra) + distractors
    obs_per_topic = 28  # 4 target obs + 24 distractors
    max_topics = max(1, min(len(HELD_OUT_TOPICS), n_obs // obs_per_topic))
    if n_obs < obs_per_topic:
        gap = 1
        topics_subset = HELD_OUT_TOPICS[:1]
    else:
        gap = 12
        topics_subset = HELD_OUT_TOPICS[:max_topics]

    for idx, (s, cp, co, pp, po, xp, xo) in enumerate(topics_subset):
        cid_p = f"p23_ho_p_{idx:03d}"
        cid_c = f"p23_ho_c_{idx:03d}"

        # 1. Canonical insert
        stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid_p, "canonical", "p23_ho", t, is_probe=False))
        t += 1

        # Intervening distractors
        for _ in range(gap):
            if len(stream) < n_obs:
                stream.append(make_distractor(t, timestamp=t))
                t += 1

        # 2. Paraphrase probe
        if len(stream) < n_obs:
            stream.append(Phase18Obs(Candidate(s, pp, po, timestamp=t), cid_p, "paraphrase", "p23_ho", t, is_probe=True))
            t += 1

        # Canonical contradiction
        if len(stream) < n_obs:
            stream.append(Phase18Obs(Candidate(s, cp, co, timestamp=t), cid_c, "canonical", "p23_ho", t, is_probe=False))
            t += 1

        for _ in range(gap):
            if len(stream) < n_obs:
                stream.append(make_distractor(t, timestamp=t))
                t += 1

        # Contradiction probe
        if len(stream) < n_obs:
            stream.append(Phase18Obs(Candidate(s, xp, xo, timestamp=t), cid_c, "contradiction", "p23_ho", t, is_probe=True))
            t += 1

    # Fill remainder to n_obs with background distractors
    while len(stream) < n_obs:
        stream.append(make_distractor(t, timestamp=t))
        t += 1

    return stream[:n_obs]
