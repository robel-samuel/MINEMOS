"""
benchmarks/phase18_dataset.py

Deterministic Long-Horizon Dataset Generator for Phase 18:
Semantic Dense Retrieval + Utility-Aware Eviction.

Evaluates 2x2 Factorial Design across:
1. Scenario 1 — Long-Gap Paraphrase Recall (gaps: 10, 50, 100, 500)
2. Scenario 2 — Critical Safety Test: SAME vs CONTRADICTION vs UNRELATED
3. Scenario 3 — Contradiction Recovery / Bounded-Head Amplification (540 obs)
4. Scenario 4 — Semantic Drift Progression (420 obs)
5. Scenario 5 — Capacity Pressure Stream (1,500 obs for C=50, 100, 200, 1500)
6. Scenario 6 — Long Continual Stream (5,000 obs)
"""

from __future__ import annotations
import os, sys, random
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate

SEED = 1818

# Rich entity pools
SUBJECTS = [f"Person_{i:04d}" for i in range(2500)]
FACILITIES = [f"Facility_{i:04d}" for i in range(2500)]
PROJECTS = [f"Project_{i:04d}" for i in range(2500)]
DEVICES = [f"Device_{i:04d}" for i in range(2500)]

BASE_PREDICATES = ["visits", "uses", "manages", "studies", "monitors"]

PARAPHRASE_MAP = {
    "visits": [
        "goes to", "frequents", "stops by", "attends", "travels to",
        "drops in at", "regularly visits", "shows up at", "is present at"
    ],
    "uses": [
        "regularly uses", "primarily works with", "relies on", "employs",
        "operates", "utilizes", "runs experiments with", "makes use of"
    ],
    "manages": [
        "oversees", "runs", "supervises", "leads", "directs",
        "is in charge of", "administers", "heads"
    ],
    "studies": [
        "researches", "investigates", "explores", "examines", "analyzes",
        "inspects", "conducts research on", "evaluates"
    ],
    "monitors": [
        "watches", "tracks", "keeps an eye on", "observes", "surveys",
        "checks on", "regularly audits", "supervises closely"
    ],
}

CONTRADICTION_MAP = {
    "visits": "avoids",
    "uses": "dislikes",
    "manages": "was removed from",
    "studies": "dropped",
    "monitors": "neglects",
}

DRIFT_SEQUENCES = {
    "visits": [
        "goes to",
        "trains at",
        "works out at",
        "exercises regularly at",
        "avoids"  # Final contradiction transition
    ],
    "uses": [
        "regularly uses",
        "works with daily",
        "specializes in operating",
        "maintains and operates",
        "dislikes using"  # Final contradiction transition
    ],
    "manages": [
        "oversees",
        "supervises day-to-day operations of",
        "advises on management of",
        "consults on operations at",
        "was removed from"  # Final contradiction transition
    ],
    "studies": [
        "researches",
        "gathers field data on",
        "assists research on",
        "reads published literature about",
        "dropped"  # Final contradiction transition
    ],
    "monitors": [
        "watches",
        "audits periodically",
        "reviews quarterly logs of",
        "receives occasional reports on",
        "neglects"  # Final contradiction transition
    ]
}


@dataclass
class Phase18Obs:
    candidate: Candidate
    concept_id: str
    category: str  # 'canonical', 'paraphrase', 'contradiction', 'distractor', 'unrelated', 'drift_step', 'drift_contra', 'delayed_paraphrase'
    scenario: str
    step_index: int = 0
    ground_truth_target_concept: Optional[str] = None
    is_probe: bool = False

    @property
    def text(self) -> str:
        return f"{self.candidate.subject} {self.candidate.predicate} {self.candidate.object}."


def make_distractor(idx: int, timestamp: int = 0) -> Phase18Obs:
    s = SUBJECTS[1200 + (idx % 1200)]
    p = BASE_PREDICATES[idx % len(BASE_PREDICATES)]
    o = FACILITIES[1200 + ((idx + 350) % 1200)]
    cid = f"distractor_{idx:05d}"
    c = Candidate(s, p, o, timestamp=timestamp)
    return Phase18Obs(c, cid, "distractor", scenario="distractor", is_probe=False)


def generate_scenario_1_long_gaps(gaps=(10, 50, 100, 500), seed=SEED) -> dict[int, list[Phase18Obs]]:
    """
    Scenario 1: Evaluates paraphrase recall across controlled gap scales.
    10 target concepts, each tested with 2 paraphrase probes separated by `gap` distractors.
    """
    rng = random.Random(seed)
    scenario_streams = {}
    for gap in gaps:
        stream = []
        t = 0
        for i in range(10):  # 10 target concepts
            s = SUBJECTS[i]
            p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
            o = FACILITIES[i]
            cid = f"target_{i:04d}"

            # 1. Canonical insert
            c_init = Candidate(s, p, o, timestamp=t)
            stream.append(Phase18Obs(c_init, cid, "canonical", "scenario_1", t, is_probe=False))
            t += 1

            # 2. Interleaved distractors
            for _ in range(gap):
                stream.append(make_distractor(t, timestamp=t))
                t += 1

            # 3. Paraphrase probe
            para_p = PARAPHRASE_MAP[p][0]
            c_para = Candidate(s, para_p, o, timestamp=t)
            stream.append(Phase18Obs(c_para, cid, "paraphrase", "scenario_1", t, is_probe=True))
            t += 1

            # 4. Another gap
            for _ in range(gap):
                stream.append(make_distractor(t, timestamp=t))
                t += 1

            # 5. Second paraphrase probe
            para_p2 = PARAPHRASE_MAP[p][1]
            c_para2 = Candidate(s, para_p2, o, timestamp=t)
            stream.append(Phase18Obs(c_para2, cid, "paraphrase", "scenario_1", t, is_probe=True))
            t += 1

        scenario_streams[gap] = stream
    return scenario_streams


def generate_scenario_2_safety_test(seed=SEED) -> list[Phase18Obs]:
    """
    Scenario 2: Critical safety test.
    Specifically tests:
      SAME / PARAPHRASE vs CONTRADICTION vs UNRELATED
    Verifies dense retrieval does NOT bypass NLI gate on contradictions.
    20 target concepts x (1 canonical + 1 paraphrase + 1 contradiction + 1 unrelated) = 80 obs.
    """
    rng = random.Random(seed)
    stream = []
    t = 0
    for i in range(20):
        s = SUBJECTS[i]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o = FACILITIES[i]
        cid = f"safety_{i:04d}"

        # 1. Canonical
        stream.append(Phase18Obs(Candidate(s, p, o, timestamp=t), cid, "canonical", "scenario_2", t, is_probe=False))
        t += 1

        # 2. Paraphrase (should merge)
        para_p = PARAPHRASE_MAP[p][0]
        stream.append(Phase18Obs(Candidate(s, para_p, o, timestamp=t), cid, "paraphrase", "scenario_2", t, is_probe=True))
        t += 1

        # 3. Contradiction (MUST NOT merge, must be rejected by NLI)
        contra_p = CONTRADICTION_MAP[p]
        stream.append(Phase18Obs(Candidate(s, contra_p, o, timestamp=t), cid, "contradiction", "scenario_2", t, is_probe=True))
        t += 1

        # 4. Unrelated (different subject/predicate/object)
        unrel_s = SUBJECTS[2000 + i]
        unrel_o = FACILITIES[2000 + i]
        stream.append(Phase18Obs(Candidate(unrel_s, p, unrel_o, timestamp=t), f"unrelated_{i:04d}", "unrelated", "scenario_2", t, is_probe=True))
        t += 1

    return stream


def generate_scenario_3_contradiction_recovery(seed=SEED) -> list[Phase18Obs]:
    """
    Scenario 3: Contradiction Recovery & Bounded-Head Amplification Analysis.
    20 concepts: Canonical -> Contradiction -> Paraphrase Probe -> 4 Recovery Steps.
    Total: 540 observations with interleaved distractors.
    """
    rng = random.Random(seed)
    stream = []
    t = 0
    for i in range(20):
        s = SUBJECTS[i]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o = FACILITIES[i]
        cid = f"recov_{i:04d}"
        contra_p = CONTRADICTION_MAP[p]
        para_p = PARAPHRASE_MAP[p][0]

        # Step 0: Canonical
        stream.append(Phase18Obs(Candidate(s, p, o, timestamp=t), cid, "canonical", "scenario_3", 0, is_probe=False))
        t += 1

        for _ in range(3):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # Step 1: Contradiction (adversarial probe)
        stream.append(Phase18Obs(Candidate(s, contra_p, o, timestamp=t), cid, "contradiction", "scenario_3", 1, is_probe=True))
        t += 1

        for _ in range(3):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # Step 2: Paraphrase probe (test if error amplified into contaminated slot)
        stream.append(Phase18Obs(Candidate(s, para_p, o, timestamp=t), cid, "paraphrase", "scenario_3", 2, is_probe=True))
        t += 1

        # Steps 3-6: Four clean recovery reinforcements
        for rec_idx in range(4):
            for _ in range(3):
                stream.append(make_distractor(t, timestamp=t))
                t += 1
            rec_p = PARAPHRASE_MAP[p][(rec_idx + 1) % len(PARAPHRASE_MAP[p])]
            stream.append(Phase18Obs(Candidate(s, rec_p, o, timestamp=t), cid, "recovery_canonical", "scenario_3", 3 + rec_idx, is_probe=False))
            t += 1

    return stream


def generate_scenario_4_semantic_drift(seed=SEED) -> list[Phase18Obs]:
    """
    Scenario 4: 5-step semantic drift progression ending in contradiction.
    20 concepts x (1 canonical + 4 drift steps + 1 drift contra) = 420 obs with distractors.
    """
    rng = random.Random(seed)
    stream = []
    t = 0
    for i in range(20):
        s = SUBJECTS[i]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o = FACILITIES[i]
        cid = f"drift_{i:04d}"
        drift_seq = DRIFT_SEQUENCES[p]

        # 1. Canonical
        stream.append(Phase18Obs(Candidate(s, p, o, timestamp=t), cid, "canonical", "scenario_4", 0, is_probe=False))
        t += 1

        # Intermediate drift steps
        for step_idx, drift_p in enumerate(drift_seq[:-1]):
            for _ in range(3):
                stream.append(make_distractor(t, timestamp=t))
                t += 1
            stream.append(Phase18Obs(Candidate(s, drift_p, o, timestamp=t), cid, "drift_step", "scenario_4", step_idx + 1, is_probe=True))
            t += 1

        # Final drift contradiction
        for _ in range(3):
            stream.append(make_distractor(t, timestamp=t))
            t += 1
        stream.append(Phase18Obs(Candidate(s, drift_seq[-1], o, timestamp=t), cid, "drift_contra", "scenario_4", 5, is_probe=True))
        t += 1

    return stream


def generate_scenario_5_capacity_pressure_stream(seed=SEED) -> list[Phase18Obs]:
    """
    Scenario 5: 1,500 observation benchmark stream for testing capacity pressure
    under C = 1500, C = 200, C = 100, C = 50.
    100 target concepts with interleaved probes and distractors.
    """
    rng = random.Random(seed)
    stream = []
    t = 0
    for i in range(100):
        s = SUBJECTS[i]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o = FACILITIES[i]
        cid = f"cap_{i:04d}"
        contra_p = CONTRADICTION_MAP[p]
        para_p = PARAPHRASE_MAP[p][0]

        stream.append(Phase18Obs(Candidate(s, p, o, timestamp=t), cid, "canonical", "scenario_5", t, is_probe=False))
        t += 1
        stream.append(Phase18Obs(Candidate(s, para_p, o, timestamp=t), cid, "paraphrase", "scenario_5", t, is_probe=True))
        t += 1
        stream.append(Phase18Obs(Candidate(s, contra_p, o, timestamp=t), cid, "contradiction", "scenario_5", t, is_probe=True))
        t += 1

    # Fill remaining observations with structured distractors and delayed paraphrase probes
    remaining = 1500 - len(stream)
    for d_idx in range(remaining):
        if d_idx % 20 == 0:
            target_idx = (d_idx // 20) % 100
            s = SUBJECTS[target_idx]
            p = BASE_PREDICATES[target_idx % len(BASE_PREDICATES)]
            o = FACILITIES[target_idx]
            cid = f"cap_{target_idx:04d}"
            para_p2 = PARAPHRASE_MAP[p][1]
            stream.append(Phase18Obs(Candidate(s, para_p2, o, timestamp=t), cid, "paraphrase", "scenario_5", t, is_probe=True))
        else:
            stream.append(make_distractor(d_idx, timestamp=t))
        t += 1

    return stream


def generate_scenario_6_continual_stream(n_obs: int = 5000, seed=SEED) -> list[Phase18Obs]:
    """
    Scenario 6: 5,000-observation continual stream.
    40 target concepts with recurring probes and background noise distractors.
    """
    rng = random.Random(seed)
    stream = []
    t = 0
    n_targets = 40

    for i in range(n_targets):
        s = SUBJECTS[i]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o = FACILITIES[i]
        cid = f"cont_{i:04d}"
        contra_p = CONTRADICTION_MAP[p]
        paras = PARAPHRASE_MAP[p]

        stream.append(Phase18Obs(Candidate(s, p, o, timestamp=t), cid, "canonical", "scenario_6", t))
        t += 1
        stream.append(Phase18Obs(Candidate(s, paras[0], o, timestamp=t), cid, "paraphrase", "scenario_6", t, is_probe=True))
        t += 1
        stream.append(Phase18Obs(Candidate(s, contra_p, o, timestamp=t), cid, "contradiction", "scenario_6", t, is_probe=True))
        t += 1
        stream.append(Phase18Obs(Candidate(s, paras[1 % len(paras)], o, timestamp=t), cid, "paraphrase", "scenario_6", t, is_probe=True))
        t += 1

    remaining = n_obs - len(stream)
    for d_idx in range(remaining):
        if d_idx % 50 == 0:
            t_idx = (d_idx // 50) % n_targets
            s = SUBJECTS[t_idx]
            p = BASE_PREDICATES[t_idx % len(BASE_PREDICATES)]
            o = FACILITIES[t_idx]
            cid = f"cont_{t_idx:04d}"
            para_p = PARAPHRASE_MAP[p][2 % len(PARAPHRASE_MAP[p])]
            stream.append(Phase18Obs(Candidate(s, para_p, o, timestamp=t), cid, "delayed_paraphrase", "scenario_6", t, is_probe=True))
        else:
            stream.append(make_distractor(d_idx, timestamp=t))
        t += 1

    for idx, obs in enumerate(stream):
        obs.candidate.timestamp = idx
    assert len(stream) == n_obs
    return stream
