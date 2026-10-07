"""
benchmarks/phase21_dataset.py

Deterministic Long-Horizon Dataset Generator for Phase 21:
Generalization, Adversarial Validation, and Out-of-Distribution Evaluation.

Contains 7 dedicated benchmark suites:
1. Exp 1: Unseen Semantic Domains (8 real-world domains)
2. Exp 2: Hard Contradictions (High lexical overlap, minimal semantic flips)
3. Exp 3: Hard Paraphrases (Extreme surface-form vocabulary shifts)
4. Exp 4: Semantic Distractor Overload (Dense semantic neighborhoods)
5. Exp 5: Long-Horizon Scaling Streams (5k, 10k, 25k, 50k)
6. Exp 6: Temporal Fact Changes (Longitudinal pre/post drift probing)
7. Exp 7: Completely Held-Out Generalization Benchmark
"""

from __future__ import annotations
import os, sys, random
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate
from benchmarks.phase18_dataset import Phase18Obs, make_distractor

SEED_P21 = 2121


# ──────────────────────────────────────────────
#  Experiment 1: Unseen Semantic Domains
# ──────────────────────────────────────────────

UNSEEN_DOMAIN_FACTS = [
    # 1. Employment
    {
        "cid": "unseen_emp_01",
        "canon": ("Dr. Sarah Chen", "is chief of cardiology at", "St. Jude Hospital"),
        "para": ("Dr. Sarah Chen", "serves as head cardiologist at", "St. Jude Hospital"),
        "contra": ("Dr. Sarah Chen", "was dismissed from cardiology at", "St. Jude Hospital"),
        "unrel": ("Dr. Sarah Chen", "serves as head cardiologist at", "Metro General")
    },
    # 2. Locations
    {
        "cid": "unseen_loc_01",
        "canon": ("The server cluster", "resides in", "Frankfurt DataCenter 4"),
        "para": ("The server cluster", "is physically located at", "Frankfurt DataCenter 4"),
        "contra": ("The server cluster", "was removed from", "Frankfurt DataCenter 4"),
        "unrel": ("The server cluster", "is physically located at", "Dublin Facility B")
    },
    # 3. Preferences
    {
        "cid": "unseen_pref_01",
        "canon": ("Marcus Vance", "adheres to", "a strict vegan diet"),
        "para": ("Marcus Vance", "strictly follows", "a plant-based vegan diet"),
        "contra": ("Marcus Vance", "rejects and opposes", "a vegan diet"),
        "unrel": ("Marcus Vance", "strictly follows", "a ketogenic meat diet")
    },
    # 4. Technical System States
    {
        "cid": "unseen_sys_01",
        "canon": ("Replica node 03", "operates in", "read-only maintenance mode"),
        "para": ("Replica node 03", "is configured in", "read-only maintenance status"),
        "contra": ("Replica node 03", "handles normal read-write traffic without", "maintenance mode"),
        "unrel": ("Replica node 03", "is configured in", "active master failover mode")
    },
    # 5. Ownership
    {
        "cid": "unseen_own_01",
        "canon": ("The quantum encryption patent", "belongs to", "Apex Innovations"),
        "para": ("The quantum encryption patent", "is owned by", "Apex Innovations"),
        "contra": ("The quantum encryption patent", "was surrendered by", "Apex Innovations"),
        "unrel": ("The quantum encryption patent", "is owned by", "Cobalt Dynamics")
    },
    # 6. Relationships
    {
        "cid": "unseen_rel_01",
        "canon": ("General Harrison", "reports directly to", "Secretary Vance"),
        "para": ("General Harrison", "answers directly to", "Secretary Vance"),
        "contra": ("General Harrison", "does not take orders from", "Secretary Vance"),
        "unrel": ("General Harrison", "answers directly to", "Director Thorne")
    },
    # 7. Events
    {
        "cid": "unseen_ev_01",
        "canon": ("The cybersecurity summit", "convenes in", "Geneva Conference Center"),
        "para": ("The cybersecurity summit", "takes place at", "Geneva Conference Center"),
        "contra": ("The cybersecurity summit", "was canceled at", "Geneva Conference Center"),
        "unrel": ("The cybersecurity summit", "takes place at", "Vienna Hall")
    },
    # 8. Multi-attribute facts
    {
        "cid": "unseen_multi_01",
        "canon": ("Flight AF-348", "carries 280 passengers bound for", "Tokyo Narita"),
        "para": ("Flight AF-348", "transports 280 travelers en route to", "Tokyo Narita"),
        "contra": ("Flight AF-348", "diverted away with zero passengers bound for", "Tokyo Narita"),
        "unrel": ("Flight AF-348", "transports 280 travelers en route to", "Seoul Incheon")
    }
]

def generate_exp1_unseen_domains(inter_gap: int = 15, seed: int = SEED_P21) -> list[Phase18Obs]:
    """Exp 1: Evaluates generalization to 8 unseen real-world semantic domains."""
    rng = random.Random(seed)
    stream = []
    t = 0
    for item in UNSEEN_DOMAIN_FACTS:
        cid = item["cid"]
        # 1. Canonical
        s, p, o = item["canon"]
        stream.append(Phase18Obs(Candidate(s, p, o, timestamp=t), cid, "canonical", "exp1_unseen", t, is_probe=False))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 2. Paraphrase probe
        s, p, o = item["para"]
        stream.append(Phase18Obs(Candidate(s, p, o, timestamp=t), cid, "paraphrase", "exp1_unseen", t, is_probe=True))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 3. Contradiction probe
        s, p, o = item["contra"]
        stream.append(Phase18Obs(Candidate(s, p, o, timestamp=t), cid, "contradiction", "exp1_unseen", t, is_probe=True))
        t += 1

    return stream


# ──────────────────────────────────────────────
#  Experiment 2: Hard Contradictions (High Lexical Overlap / Minimal Flips)
# ──────────────────────────────────────────────

HARD_CONTRADICTION_PAIRS = [
    ("Alice works at", "Google", "Alice does not work at", "Google"),
    ("Alice owns the", "red car", "Alice does not own the", "red car"),
    ("The server is", "online", "The server is", "offline"),
    ("Alice lives in", "Addis Ababa", "Alice does not live in", "Addis Ababa"),
    ("Database cluster primary is", "encrypted", "Database cluster primary is", "unencrypted"),
    ("Production pipeline alpha is", "enabled", "Production pipeline alpha is", "disabled"),
    ("Captain Miller authorized the", "cargo departure", "Captain Miller revoked the", "cargo departure"),
    ("The patient exhibits", "cardiac arrhythmia", "The patient is completely free of", "cardiac arrhythmia"),
    ("Node 12 communicates via", "encrypted TLS", "Node 12 communicates via", "plaintext HTTP"),
    ("Elena speaks fluent", "Mandarin Chinese", "Elena cannot understand or speak", "Mandarin Chinese"),
    ("The microservice cache is", "warm and hydrated", "The microservice cache is", "empty and cold"),
    ("Agent 07 confirmed the", "rendezvous coordinate", "Agent 07 denied the", "rendezvous coordinate")
]

def generate_exp2_hard_contradictions(inter_gap: int = 10, seed: int = SEED_P21) -> list[Phase18Obs]:
    """Exp 2: Evaluates adversarial contradiction resistance under high lexical overlap."""
    rng = random.Random(seed)
    stream = []
    t = 0
    for idx, (p_canon, o_canon, p_contra, o_contra) in enumerate(HARD_CONTRADICTION_PAIRS):
        cid = f"hard_contra_{idx:03d}"
        subj = f"Entity_{idx:03d}"

        # 1. Canonical insert
        stream.append(Phase18Obs(Candidate(subj, p_canon, o_canon, timestamp=t), cid, "canonical", "exp2_hard_contra", t, is_probe=False))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 2. Hard contradiction probe
        stream.append(Phase18Obs(Candidate(subj, p_contra, o_contra, timestamp=t), cid, "contradiction", "exp2_hard_contra", t, is_probe=True))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 3. Post-contradiction verification probe
        stream.append(Phase18Obs(Candidate(subj, p_canon, o_canon, timestamp=t), cid, "paraphrase", "exp2_hard_contra", t, is_probe=True))
        t += 1

    return stream


# ──────────────────────────────────────────────
#  Experiment 3: Hard Paraphrases (Extreme Lexical Shift)
# ──────────────────────────────────────────────

HARD_PARAPHRASE_PAIRS = [
    ("Bob purchased a", "motor vehicle", "Bob recently acquired an", "automobile"),
    ("The patient exhibits severe", "hypertension", "The individual suffers from critically elevated", "blood pressure"),
    ("The software team terminated the", "legacy database", "Engineers permanently decommissioned the", "outdated storage system"),
    ("The firm suffered severe", "financial distress", "The corporation incurred catastrophic", "monetary losses"),
    ("Dr. Aris discovered a novel", "antibacterial peptide", "The researcher isolated an unprecedented", "microbial inhibitor"),
    ("The suspect fled on foot toward the", "rail terminal", "The fugitive ran away in the direction of the", "train station"),
    ("The aircraft requested an immediate", "emergency landing", "The jet initiated an urgent diversionary", "touchdown procedure"),
    ("The architect finalized blueprints for the", "skyscraper", "The designer completed structural schematics for the", "high-rise tower"),
    ("The diplomat negotiated a bilateral", "trade pact", "The envoy hammered out a mutual", "commercial agreement"),
    ("The company dismissed twenty percent of its", "workforce", "The enterprise laid off one-fifth of its", "employees")
]

def generate_exp3_hard_paraphrases(inter_gap: int = 10, seed: int = SEED_P21) -> list[Phase18Obs]:
    """Exp 3: Evaluates paraphrase consolidation under extreme surface-form variation."""
    rng = random.Random(seed)
    stream = []
    t = 0
    for idx, (p_canon, o_canon, p_para, o_para) in enumerate(HARD_PARAPHRASE_PAIRS):
        cid = f"hard_para_{idx:03d}"
        subj = f"Subject_{idx:03d}"

        # 1. Canonical insert
        stream.append(Phase18Obs(Candidate(subj, p_canon, o_canon, timestamp=t), cid, "canonical", "exp3_hard_para", t, is_probe=False))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 2. Hard paraphrase probe
        stream.append(Phase18Obs(Candidate(subj, p_para, o_para, timestamp=t), cid, "paraphrase", "exp3_hard_para", t, is_probe=True))
        t += 1

    return stream


# ──────────────────────────────────────────────
#  Experiment 4: Semantic Distractor Overload
# ──────────────────────────────────────────────

def generate_exp4_distractor_overload(n_targets: int = 10, distractors_per_target: int = 15, seed: int = SEED_P21) -> list[Phase18Obs]:
    """
    Exp 4: Evaluates retrieval selectivity in dense semantic clusters.
    Creates 15 highly related facts per target entity sharing subject/domain semantics.
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    ATTRIBUTES = [
        ("works as principal engineer at", "Google DeepMind"),
        ("graduated with doctorate from", "Stanford University"),
        ("published foundational research on", "neural attention mechanisms"),
        ("maintains production clusters running", "Kubernetes"),
        ("resides in downtown apartment in", "Zurich Switzerland"),
        ("speaks native English and conversational", "German"),
        ("contributes open-source code to", "Apache Arrow"),
        ("leads architectural reviews for", "distributed memory systems"),
        ("drives an electric sedan manufactured by", "Tesla"),
        ("completed an alpine ascent of", "Mount Rainier"),
        ("mentors graduate researchers from", "ETH Zurich"),
        ("chairs the annual symposium on", "Information Retrieval"),
        ("registered international patents on", "vector quantization"),
        ("volunteers on weekends at the", "Community Science Center"),
        ("authors technical articles on", "high-performance computing")
    ]

    for i in range(n_targets):
        subj = f"TargetPerson_{i:03d}"
        cid_target = f"overload_target_{i:03d}"

        # Target fact (to be probed)
        target_p, target_o = ATTRIBUTES[0]
        stream.append(Phase18Obs(Candidate(subj, target_p, target_o, timestamp=t), cid_target, "canonical", "exp4_overload", t, is_probe=False))
        t += 1

        # 14 Related semantic distractors about the SAME person
        for attr_idx, (p_dist, o_dist) in enumerate(ATTRIBUTES[1:1 + distractors_per_target]):
            cid_dist = f"overload_distractor_{i:03d}_{attr_idx:02d}"
            stream.append(Phase18Obs(Candidate(subj, p_dist, o_dist, timestamp=t), cid_dist, "distractor", "exp4_overload", t, is_probe=False))
            t += 1

        # Probe the target fact with a paraphrase amid the dense neighborhood
        para_target_p = "serves as principal staff engineer at"
        stream.append(Phase18Obs(Candidate(subj, para_target_p, target_o, timestamp=t), cid_target, "paraphrase", "exp4_overload", t, is_probe=True))
        t += 1

    return stream


# ──────────────────────────────────────────────
#  Experiment 5: Long-Horizon Scaling Stream Generator
# ──────────────────────────────────────────────

def generate_exp5_long_horizon(n_obs: int = 10000, seed: int = SEED_P21) -> list[Phase18Obs]:
    """Exp 5: Scales observation streams up to 10k, 25k, or 50k observations."""
    rng = random.Random(seed)
    stream = []
    t = 0
    n_targets = max(20, n_obs // 200)

    # Base target setup
    for i in range(n_targets):
        s = f"ScalingEntity_{i:04d}"
        p = "manages operational infrastructure at"
        o = f"DataCenter_{i:04d}"
        cid = f"scale_{i:04d}"

        stream.append(Phase18Obs(Candidate(s, p, o, timestamp=t), cid, "canonical", "exp5_scale", t))
        t += 1
        stream.append(Phase18Obs(Candidate(s, "oversees server systems at", o, timestamp=t), cid, "paraphrase", "exp5_scale", t, is_probe=True))
        t += 1
        stream.append(Phase18Obs(Candidate(s, "was relieved of duties at", o, timestamp=t), cid, "contradiction", "exp5_scale", t, is_probe=True))
        t += 1

    remaining = n_obs - len(stream)
    for d_idx in range(remaining):
        if d_idx % 40 == 0:
            t_idx = (d_idx // 40) % n_targets
            s = f"ScalingEntity_{t_idx:04d}"
            o = f"DataCenter_{t_idx:04d}"
            cid = f"scale_{t_idx:04d}"
            stream.append(Phase18Obs(Candidate(s, "supervises hardware facilities at", o, timestamp=t), cid, "delayed_paraphrase", "exp5_scale", t, is_probe=True))
        else:
            stream.append(make_distractor(d_idx, timestamp=t))
        t += 1

    for idx, obs in enumerate(stream):
        obs.candidate.timestamp = idx
        obs.step_index = idx
    assert len(stream) == n_obs
    return stream


# ──────────────────────────────────────────────
#  Experiment 6: Temporal Fact Changes (Longitudinal Pre/Post Probing)
# ──────────────────────────────────────────────

def generate_exp6_temporal_changes(n_concepts: int = 20, inter_gap: int = 15, seed: int = SEED_P21) -> list[Phase18Obs]:
    """
    Exp 6: Evaluates temporal fact progression across time:
      t1: Canonical insert ("Alice works at Company A")
      t_mid: Pre-drift probe ("Alice is employed at Company A" -> should merge into slot)
      t2: Factual update ("Alice relocated and now works at Company B" -> legitimate temporal update)
      t_post: Post-drift probe ("Alice works at Company B" -> tests current validity)
      t_adv: Adversarial challenge ("Alice was terminated from Company A")
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    for i in range(n_concepts):
        s = f"Worker_{i:03d}"
        cid = f"temporal_{i:03d}"
        comp_a = f"Enterprise_Alpha_{i:03d}"
        comp_b = f"Corporation_Beta_{i:03d}"

        # 1. Canonical at t1
        stream.append(Phase18Obs(Candidate(s, "works as lead architect at", comp_a, timestamp=t), cid, "canonical", "exp6_temporal", t, is_probe=False))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 2. Pre-drift probe (t_mid)
        stream.append(Phase18Obs(Candidate(s, "is employed as chief architect at", comp_a, timestamp=t), cid, "pre_drift_probe", "exp6_temporal", t, is_probe=True))
        t += 1

        for _ in range(inter_gap * 2):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 3. Factual update (t2)
        stream.append(Phase18Obs(Candidate(s, "transitioned and now works at", comp_b, timestamp=t), cid, "temporal_update", "exp6_temporal", t, is_probe=True))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 4. Post-drift probe (t_post)
        stream.append(Phase18Obs(Candidate(s, "currently serves on staff at", comp_b, timestamp=t), cid, "post_drift_probe", "exp6_temporal", t, is_probe=True))
        t += 1

        # 5. Adversarial challenge (t_adv)
        stream.append(Phase18Obs(Candidate(s, "was fired and expelled from", comp_a, timestamp=t), cid, "contradiction", "exp6_temporal", t, is_probe=True))
        t += 1

    return stream


# ──────────────────────────────────────────────
#  Experiment 7: Held-Out Generalization Benchmark
# ──────────────────────────────────────────────

def generate_exp7_held_out_benchmark(n_obs: int = 2500, seed: int = 9999) -> list[Phase18Obs]:
    """
    Exp 7: Completely independent, un-tuned held-out evaluation stream.
    Uses completely fresh entity vocabulary, novel predicates, and balanced probes.
    """
    rng = random.Random(seed)
    stream = []
    t = 0
    n_targets = 30

    HELD_OUT_SUBJECTS = [f"HeldOutAgent_{i:04d}" for i in range(1000)]
    HELD_OUT_OBJECTS = [f"ProtectedAsset_{i:04d}" for i in range(1000)]
    HELD_OUT_PREDICATES = [
        ("safeguards the integrity of", "monitors defense protocols for", "compromised security on"),
        ("coordinates transport logistics for", "manages dispatch scheduling for", "halted all shipments of"),
        ("certifies compliance standards on", "audits regulatory adherence for", "revoked certification from"),
        ("deploys telemetry monitors to", "configures sensor telemetry on", "disabled tracking units across")
    ]

    for i in range(n_targets):
        s = HELD_OUT_SUBJECTS[i]
        o = HELD_OUT_OBJECTS[i]
        p_canon, p_para, p_contra = HELD_OUT_PREDICATES[i % len(HELD_OUT_PREDICATES)]
        cid = f"heldout_{i:04d}"

        # Canonical
        stream.append(Phase18Obs(Candidate(s, p_canon, o, timestamp=t), cid, "canonical", "exp7_heldout", t, is_probe=False))
        t += 1
        # Paraphrase
        stream.append(Phase18Obs(Candidate(s, p_para, o, timestamp=t), cid, "paraphrase", "exp7_heldout", t, is_probe=True))
        t += 1
        # Contradiction
        stream.append(Phase18Obs(Candidate(s, p_contra, o, timestamp=t), cid, "contradiction", "exp7_heldout", t, is_probe=True))
        t += 1

    remaining = n_obs - len(stream)
    for d_idx in range(remaining):
        if d_idx % 35 == 0:
            t_idx = (d_idx // 35) % n_targets
            s = HELD_OUT_SUBJECTS[t_idx]
            o = HELD_OUT_OBJECTS[t_idx]
            cid = f"heldout_{t_idx:04d}"
            _, p_para, _ = HELD_OUT_PREDICATES[t_idx % len(HELD_OUT_PREDICATES)]
            stream.append(Phase18Obs(Candidate(s, p_para, o, timestamp=t), cid, "delayed_paraphrase", "exp7_heldout", t, is_probe=True))
        else:
            # Independent distractor
            s_d = HELD_OUT_SUBJECTS[500 + (d_idx % 400)]
            o_d = HELD_OUT_OBJECTS[500 + ((d_idx + 137) % 400)]
            p_d = "inspects operational records of"
            stream.append(Phase18Obs(Candidate(s_d, p_d, o_d, timestamp=t), f"dist_{d_idx:05d}", "distractor", "exp7_heldout", t, is_probe=False))
        t += 1

    for idx, obs in enumerate(stream):
        obs.candidate.timestamp = idx
        obs.step_index = idx
    assert len(stream) == n_obs
    return stream
