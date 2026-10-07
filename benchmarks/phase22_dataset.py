"""
benchmarks/phase22_dataset.py

Deterministic Benchmark Dataset Generators for Phase 22:
Hybrid Retrieval (Dense + Lexical Candidate Generation).

Experiments:
1. Exp 1: Vocabulary Shift (Extreme lexical variation / hard paraphrases)
2. Exp 2: Dense Semantic Distractors (High distractor density per entity)
3. Exp 3: Adversarial Contradictions (High lexical overlap minimal-polarity flips)
4. Exp 4: Long-Horizon Streams (5k and 10k observations)
5. Exp 5: Ablation Streams
6. Exp 6: Completely Held-Out Benchmark (Seed=8888, novel vocabulary)
"""

from __future__ import annotations
import os, sys, random
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate
from benchmarks.phase18_dataset import Phase18Obs, make_distractor

SEED_P22 = 2222


# ──────────────────────────────────────────────
#  Experiment 1: Vocabulary Shift (Hard Paraphrases)
# ──────────────────────────────────────────────

P22_VOCAB_SHIFT_PAIRS = [
    # 1. Medical
    ("The surgeon performed a delicate", "cholecystectomy procedure",
     "The doctor completed a complex", "gallbladder surgical removal"),
    # 2. Financial
    ("The conglomerate experienced massive", "insolvency and liquidation",
     "The commercial holding company suffered catastrophic", "bankruptcy and asset dissolution"),
    # 3. Computing
    ("The engineer deprecated the outdated", "cryptographic cipher",
     "The developer phased out the obsolete", "encryption algorithm"),
    # 4. Aviation
    ("The aviator executed an unassisted", "holding pattern maneuver",
     "The pilot conducted a solo", "racetrack flight circuit"),
    # 5. Scientific
    ("The chemist synthesized a volatile", "hydrocarbon compound",
     "The researcher manufactured an unstable", "organic carbon molecule"),
    # 6. Corporate
    ("The CEO stepped down and ceded all", "executive authority",
     "The president resigned and surrendered complete", "corporate governance"),
    # 7. Real Estate
    ("The developer acquired thirty hectares of", "agricultural acreage",
     "The builder purchased seventy-four acres of", "rural farmland"),
    # 8. Cybersecurity
    ("The intrusion team exploited an unpatched", "zero-day flaw",
     "The unauthorized operators took advantage of an undocumented", "software security breach"),
    # 9. Logistics
    ("The courier expedited the priority", "consignment parcel",
     "The delivery service accelerated the urgent", "freight shipment"),
    # 10. Legal
    ("The judge granted a preliminary", "restraining injunction",
     "The magistrate issued an initial", "protective stop-order"),
    # 11. Astronomy
    ("The observatory detected an occulted", "exoplanetary body",
     "The telescope discovered a hidden", "extrasolar alien world"),
    # 12. Manufacturing
    ("The factory halted the continuous", "assembly pipeline",
     "The production facility paused the uninterrupted", "manufacturing line"),
]


def generate_exp1_vocab_shift(inter_gap: int = 10, seed: int = SEED_P22) -> list[Phase18Obs]:
    """Exp 1: Evaluates whether BM25 recovers memories across massive vocabulary divergence."""
    rng = random.Random(seed)
    stream = []
    t = 0

    for idx, (p_canon, o_canon, p_para, o_para) in enumerate(P22_VOCAB_SHIFT_PAIRS):
        cid = f"vshift_{idx:03d}"
        subj = f"Entity_{idx:03d}"

        # 1. Canonical insert
        stream.append(Phase18Obs(Candidate(subj, p_canon, o_canon, timestamp=t), cid, "canonical", "exp1_vshift", t, is_probe=False))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 2. Vocabulary-shifted paraphrase probe (shares subject, but predicate/object heavily shifted)
        stream.append(Phase18Obs(Candidate(subj, p_para, o_para, timestamp=t), cid, "paraphrase", "exp1_vshift", t, is_probe=True))
        t += 1

    return stream


# ──────────────────────────────────────────────
#  Experiment 2: Dense Semantic Distractors
# ──────────────────────────────────────────────

def generate_exp2_dense_distractors(n_targets: int = 10, distractors_per_target: int = 15, seed: int = SEED_P22) -> list[Phase18Obs]:
    """
    Exp 2: Evaluates candidate contamination and selectivity when memory contains
    dense clusters of 15 facts per subject entity.
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    ATTRIBUTES = [
        ("operates as head of research at", "OpenAI Laboratories"),
        ("graduated with highest honors from", "Cambridge University"),
        ("authored foundational papers on", "diffusion models"),
        ("deploys massive training jobs using", "Ray framework"),
        ("owns an eco-friendly residence in", "Geneva Switzerland"),
        ("speaks native French and fluent", "Japanese"),
        ("contributes open-source optimizations to", "PyTorch Core"),
        ("serves as program committee member for", "NeurIPS Conference"),
        ("commutes daily on an electric motorcycle from", "Zero Motors"),
        ("climbed the north face peak of", "Mount Eiger"),
        ("advises artificial intelligence doctoral fellows at", "Oxford University"),
        ("co-founded an international initiative on", "AI Safety and Alignment"),
        ("holds key international patents in", "sparse attention hardware"),
        ("serves on the advisory council for the", "Global Science Foundation"),
        ("frequently writes guest editorials for the", "ACM Computing Review"),
    ]

    for i in range(n_targets):
        subj = f"Scholar_{i:03d}"
        cid_target = f"dist_target_{i:03d}"

        # Target fact
        target_p, target_o = ATTRIBUTES[0]
        stream.append(Phase18Obs(Candidate(subj, target_p, target_o, timestamp=t), cid_target, "canonical", "exp2_dense_dist", t, is_probe=False))
        t += 1

        # 14 Related distractors sharing the EXACT SAME subject name
        for attr_idx, (p_dist, o_dist) in enumerate(ATTRIBUTES[1:1 + distractors_per_target]):
            cid_dist = f"dist_attr_{i:03d}_{attr_idx:02d}"
            stream.append(Phase18Obs(Candidate(subj, p_dist, o_dist, timestamp=t), cid_dist, "distractor", "exp2_dense_dist", t, is_probe=False))
            t += 1

        # Probe the target fact with a lexical paraphrase
        para_p = "serves as research director at"
        stream.append(Phase18Obs(Candidate(subj, para_p, target_o, timestamp=t), cid_target, "paraphrase", "exp2_dense_dist", t, is_probe=True))
        t += 1

    return stream


# ──────────────────────────────────────────────
#  Experiment 3: Adversarial Contradictions (High Lexical Overlap)
# ──────────────────────────────────────────────

P22_ADVERSARIAL_TRIPLES = [
    ("works at", "Google Research",
     "does not work at", "Google Research",
     "works at", "Anthropic AI"),
    ("owns a residential property in", "Zurich",
     "does not own a residential property in", "Zurich",
     "owns a residential property in", "Geneva"),
    ("is certified as chief pilot on", "Boeing 787",
     "is not certified as chief pilot on", "Boeing 787",
     "is certified as chief pilot on", "Airbus A350"),
    ("holds exclusive distribution rights in", "North America",
     "lost all exclusive distribution rights in", "North America",
     "holds exclusive distribution rights in", "Western Europe"),
    ("authorized cryptographic security clearance for", "Project Prometheus",
     "revoked cryptographic security clearance for", "Project Prometheus",
     "authorized cryptographic security clearance for", "Project Daedalus"),
    ("maintains the primary database replica in", "Frankfurt",
     "terminated the primary database replica in", "Frankfurt",
     "maintains the primary database replica in", "Stockholm"),
    ("serves as attending physician at", "Johns Hopkins Hospital",
     "was dismissed from practice at", "Johns Hopkins Hospital",
     "serves as attending physician at", "Cleveland Clinic"),
    ("signed the multilateral extradition treaty in", "The Hague",
     "refused to sign the multilateral extradition treaty in", "The Hague",
     "signed the multilateral extradition treaty in", "Brussels"),
    ("manages operational server clusters for", "Cloud Platform Alpha",
     "no longer manages operational server clusters for", "Cloud Platform Alpha",
     "manages operational server clusters for", "Cloud Platform Omega"),
    ("holds the world speed record in", "marathon sailing",
     "was disqualified from the world speed record in", "marathon sailing",
     "holds the world speed record in", "open water rowing"),
]


def generate_exp3_adversarial_contradictions(inter_gap: int = 10, seed: int = SEED_P22) -> list[Phase18Obs]:
    """
    Exp 3: Evaluates whether BM25 high lexical match induces false merges
    on explicit negation and high-overlap entity switches.
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    for idx, (p_can, o_can, p_neg, o_neg, p_alt, o_alt) in enumerate(P22_ADVERSARIAL_TRIPLES):
        subj = f"Adversary_{idx:03d}"
        cid = f"adv_contra_{idx:03d}"

        # 1. Canonical insert
        stream.append(Phase18Obs(Candidate(subj, p_can, o_can, timestamp=t), cid, "canonical", "exp3_adversarial", t, is_probe=False))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 2. Direct Negation Contradiction Probe (Massive lexical overlap)
        stream.append(Phase18Obs(Candidate(subj, p_neg, o_neg, timestamp=t), cid, "contradiction", "exp3_adversarial", t, is_probe=True))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 3. Object-Swap Contradiction Probe (Same subject/predicate, different object)
        stream.append(Phase18Obs(Candidate(subj, p_alt, o_alt, timestamp=t), cid, "contradiction", "exp3_adversarial", t, is_probe=True))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 4. Canonical Paraphrase Probe (Tests recall preservation)
        stream.append(Phase18Obs(Candidate(subj, p_can, o_can, timestamp=t), cid, "paraphrase", "exp3_adversarial", t, is_probe=True))
        t += 1

    return stream


# ──────────────────────────────────────────────
#  Experiment 4: Long-Horizon Streams (5k and 10k)
# ──────────────────────────────────────────────

def generate_exp4_long_horizon(n_obs: int = 5000, seed: int = SEED_P22) -> list[Phase18Obs]:
    """Exp 4: Long-horizon continual stream to measure computational scaling and recall."""
    rng = random.Random(seed)
    stream = []
    t = 0
    if n_obs >= 4000:
        n_targets = max(20, n_obs // 200)
    else:
        n_targets = max(1, min(20, n_obs // 4))

    for i in range(n_targets):
        s = f"HorizonEntity_{i:04d}"
        p = "manages operational network infrastructure at"
        o = f"DataCenter_{i:04d}"
        cid = f"scale_{i:04d}"

        stream.append(Phase18Obs(Candidate(s, p, o, timestamp=t), cid, "canonical", "exp4_scale", t))
        t += 1
        stream.append(Phase18Obs(Candidate(s, "oversees high-speed routing systems at", o, timestamp=t), cid, "paraphrase", "exp4_scale", t, is_probe=True))
        t += 1
        stream.append(Phase18Obs(Candidate(s, "was terminated from duties at", o, timestamp=t), cid, "contradiction", "exp4_scale", t, is_probe=True))
        t += 1

    remaining = n_obs - len(stream)
    for d_idx in range(remaining):
        if d_idx % 40 == 0:
            t_idx = (d_idx // 40) % n_targets
            s = f"HorizonEntity_{t_idx:04d}"
            o = f"DataCenter_{t_idx:04d}"
            cid = f"scale_{t_idx:04d}"
            stream.append(Phase18Obs(Candidate(s, "supervises hardware facilities at", o, timestamp=t), cid, "delayed_paraphrase", "exp4_scale", t, is_probe=True))
        else:
            stream.append(make_distractor(d_idx, timestamp=t))
        t += 1

    for idx, obs in enumerate(stream):
        obs.candidate.timestamp = idx
        obs.step_index = idx
    assert len(stream) == n_obs
    return stream


# ──────────────────────────────────────────────
#  Experiment 5: Ablation Stream
# ──────────────────────────────────────────────

def generate_exp5_ablation_stream(n_concepts: int = 15, inter_gap: int = 10, seed: int = SEED_P22) -> list[Phase18Obs]:
    """
    Exp 5: Balanced benchmark containing both lexical shifts and adversarial overlaps
    to isolate the exact contributions of Dense, BM25, and D8 Anchors.
    """
    rng = random.Random(seed)
    stream = []
    t = 0

    for i in range(n_concepts):
        cid = f"ablation_{i:03d}"
        s = f"AblationSubject_{i:03d}"

        # 1. Canonical
        stream.append(Phase18Obs(Candidate(s, "directs enterprise cybersecurity at", f"Corporation_{i:03d}", timestamp=t), cid, "canonical", "exp5_ablation", t, is_probe=False))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 2. Hard paraphrase (vocabulary shift)
        stream.append(Phase18Obs(Candidate(s, "leads digital defense protocols across", f"Corporation_{i:03d}", timestamp=t), cid, "paraphrase", "exp5_ablation", t, is_probe=True))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 3. Adversarial contradiction (high lexical overlap)
        stream.append(Phase18Obs(Candidate(s, "does not direct enterprise cybersecurity at", f"Corporation_{i:03d}", timestamp=t), cid, "contradiction", "exp5_ablation", t, is_probe=True))
        t += 1

    return stream


# ──────────────────────────────────────────────
#  Experiment 6: Sealed Held-Out Evaluation Stream
# ──────────────────────────────────────────────

def generate_exp6_held_out_benchmark(n_obs: int = 2500, seed: int = 8888) -> list[Phase18Obs]:
    """
    Exp 6: Sealed Held-Out Benchmark (Seed=8888).
    Strictly un-tuned: Novel subjects, novel predicates, novel vocabulary.
    Evaluated exactly once without parameter tuning.
    """
    rng = random.Random(seed)
    stream = []
    t = 0
    n_targets = 30

    HELD_OUT_SUBJECTS = [f"Phase22Agent_{i:04d}" for i in range(1000)]
    HELD_OUT_OBJECTS = [f"CriticalAsset_{i:04d}" for i in range(1000)]
    HELD_OUT_PREDICATES = [
        # (canonical, lexical_shifted_paraphrase, adversarial_contradiction)
        ("secures authentication credentials for", "safeguards identity verification tokens on", "leaked all authentication credentials for"),
        ("calibrates orbital trajectory telemetry on", "fine-tunes satellite navigation parameters for", "disabled orbital trajectory telemetry on"),
        ("governs ethical compliance reviews across", "administers regulatory oversight inspections for", "bypassed all ethical compliance reviews across"),
        ("manages distributed key escrow at", "supervises decentralized cryptographic storage in", "compromised distributed key escrow at"),
        ("authors technical specifications for", "drafts engineering schematics regarding", "rejected technical specifications for"),
    ]

    for i in range(n_targets):
        s = HELD_OUT_SUBJECTS[i]
        o = HELD_OUT_OBJECTS[i]
        p_canon, p_para, p_contra = HELD_OUT_PREDICATES[i % len(HELD_OUT_PREDICATES)]
        cid = f"p22_heldout_{i:04d}"

        # Canonical
        stream.append(Phase18Obs(Candidate(s, p_canon, o, timestamp=t), cid, "canonical", "exp6_heldout", t, is_probe=False))
        t += 1
        # Paraphrase
        stream.append(Phase18Obs(Candidate(s, p_para, o, timestamp=t), cid, "paraphrase", "exp6_heldout", t, is_probe=True))
        t += 1
        # Contradiction
        stream.append(Phase18Obs(Candidate(s, p_contra, o, timestamp=t), cid, "contradiction", "exp6_heldout", t, is_probe=True))
        t += 1

    remaining = n_obs - len(stream)
    for d_idx in range(remaining):
        if d_idx % 35 == 0:
            t_idx = (d_idx // 35) % n_targets
            s = HELD_OUT_SUBJECTS[t_idx]
            o = HELD_OUT_OBJECTS[t_idx]
            cid = f"p22_heldout_{t_idx:04d}"
            _, p_para, _ = HELD_OUT_PREDICATES[t_idx % len(HELD_OUT_PREDICATES)]
            stream.append(Phase18Obs(Candidate(s, p_para, o, timestamp=t), cid, "delayed_paraphrase", "exp6_heldout", t, is_probe=True))
        else:
            s_d = HELD_OUT_SUBJECTS[500 + (d_idx % 400)]
            o_d = HELD_OUT_OBJECTS[500 + ((d_idx + 137) % 400)]
            p_d = "validates operational log integrity of"
            stream.append(Phase18Obs(Candidate(s_d, p_d, o_d, timestamp=t), f"p22_dist_{d_idx:05d}", "distractor", "exp6_heldout", t, is_probe=False))
        t += 1

    for idx, obs in enumerate(stream):
        obs.candidate.timestamp = idx
        obs.step_index = idx
    assert len(stream) == n_obs
    return stream
