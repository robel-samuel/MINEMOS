"""
benchmarks/phase17_dataset.py

Deterministic Long-Horizon Dataset Generator for Phase 17:
Contains 7 controlled scenarios:
1. Scenario 1 — Long-Gap Paraphrase Recall (gaps: 10, 50, 100, 500, 1000)
2. Scenario 2 — Repeated Contradiction (alternating canonical vs contradiction with distractors)
3. Scenario 3 — Contradiction -> Recovery (canonical -> contra -> distractor -> canonical -> recovery)
4. Scenario 4 — Semantic Drift (5-step progressive drift ending in contradiction shift)
5. Scenario 5 — Unrelated Distractor Stress (variable distractor counts: 50 to 500 concepts)
6. Scenario 6 — Capacity Pressure Benchmark Stream (1,500 observations tested at C=1500, C=100, C=50)
7. Scenario 7 — Very Long Continual Stream (5,000 observations)
"""

from __future__ import annotations
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate

SEED = 1717

# Rich entity pools to support up to 2,000 distinct entities
SUBJECTS = [f"Person_{i:04d}" for i in range(2000)]
FACILITIES = [f"Facility_{i:04d}" for i in range(2000)]
PROJECTS = [f"Project_{i:04d}" for i in range(2000)]
DEVICES = [f"Device_{i:04d}" for i in range(2000)]

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

# 5-step semantic drift progression
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
        "leads the team at",
        "holds executive responsibility for",
        "was removed from managing"  # Final contradiction transition
    ]
}


class Phase17Obs:
    def __init__(self, candidate: Candidate, concept_id: str,
                 category: str, scenario: str, step_index: int = 0,
                 ground_truth_target_concept: str = None,
                 is_probe: bool = False):
        self.candidate = candidate
        self.concept_id = concept_id
        self.category = category  # "canonical", "paraphrase", "contradiction", "distractor", "drift"
        self.scenario = scenario  # scenario identifier
        self.step_index = step_index
        self.ground_truth_target_concept = ground_truth_target_concept or concept_id
        self.is_probe = is_probe

    @property
    def text(self) -> str:
        return f"{self.candidate.subject} {self.candidate.predicate} {self.candidate.object}."


def make_distractor(idx: int, timestamp: int = 0) -> Phase17Obs:
    s = SUBJECTS[1000 + (idx % 1000)]
    p = BASE_PREDICATES[idx % len(BASE_PREDICATES)]
    o = FACILITIES[1000 + ((idx + 300) % 1000)]
    cid = f"distractor_{idx:05d}"
    c = Candidate(s, p, o, timestamp=timestamp)
    return Phase17Obs(c, cid, "distractor", scenario="distractor", is_probe=False)


def generate_scenario_1_long_gaps(gaps=(10, 50, 100, 500, 1000), seed=SEED) -> dict:
    """Generates streams evaluating paraphrase recall across controlled gap scales."""
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
            stream.append(Phase17Obs(c_init, cid, "canonical", "scenario_1", t, is_probe=False))
            t += 1

            # 2. Interleaved distractors
            for d in range(gap):
                stream.append(make_distractor(t, timestamp=t))
                t += 1

            # 3. Paraphrase probe
            para_p = PARAPHRASE_MAP[p][0]
            c_para = Candidate(s, para_p, o, timestamp=t)
            stream.append(Phase17Obs(c_para, cid, "paraphrase", "scenario_1", t, is_probe=True))
            t += 1

            # 4. Another gap
            for d in range(gap):
                stream.append(make_distractor(t, timestamp=t))
                t += 1

            # 5. Second paraphrase probe
            para_p2 = PARAPHRASE_MAP[p][1]
            c_para2 = Candidate(s, para_p2, o, timestamp=t)
            stream.append(Phase17Obs(c_para2, cid, "paraphrase", "scenario_1", t, is_probe=True))
            t += 1

        scenario_streams[gap] = stream
    return scenario_streams


def generate_scenario_2_repeated_contradictions(seed=SEED) -> list[Phase17Obs]:
    """Generates streams with repeated alternating contradictions and interleaved distractors."""
    rng = random.Random(seed)
    stream = []
    t = 0
    for i in range(20):  # 20 target concepts
        s = SUBJECTS[i]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o = FACILITIES[i]
        cid = f"rep_contra_{i:04d}"
        contra_p = CONTRADICTION_MAP[p]

        # Sequence: canonical -> distractor(5) -> contra -> distractor(5) -> canonical -> distractor(5) -> contra
        seq = [
            ("canonical", p),
            ("contradiction", contra_p),
            ("canonical", p),
            ("contradiction", contra_p)
        ]
        for step_idx, (cat, pred) in enumerate(seq):
            c = Candidate(s, pred, o, timestamp=t)
            stream.append(Phase17Obs(c, cid, cat, "scenario_2", step_idx, is_probe=True))
            t += 1
            for _ in range(5):
                stream.append(make_distractor(t, timestamp=t))
                t += 1
    return stream


def generate_scenario_3_contradiction_recovery(seed=SEED) -> list[Phase17Obs]:
    """Generates canonical -> contra -> distractors -> canonical -> recovery."""
    rng = random.Random(seed)
    stream = []
    t = 0
    for i in range(20):
        s = SUBJECTS[i]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o = FACILITIES[i]
        cid = f"recovery_{i:04d}"
        contra_p = CONTRADICTION_MAP[p]

        # 1. Canonical
        stream.append(Phase17Obs(Candidate(s, p, o, timestamp=t), cid, "canonical", "scenario_3", 0))
        t += 1
        # 2. Contradiction
        stream.append(Phase17Obs(Candidate(s, contra_p, o, timestamp=t), cid, "contradiction", "scenario_3", 1, is_probe=True))
        t += 1
        # 3. Distractors
        for _ in range(20):
            stream.append(make_distractor(t, timestamp=t))
            t += 1
        # 4. Paraphrase probe
        para_p = PARAPHRASE_MAP[p][0]
        stream.append(Phase17Obs(Candidate(s, para_p, o, timestamp=t), cid, "paraphrase", "scenario_3", 2, is_probe=True))
        t += 1
        # 5. Clean canonical recovery repeats (4 steps)
        for rep in range(4):
            stream.append(Phase17Obs(Candidate(s, p, o, timestamp=t), cid, "recovery_canonical", "scenario_3", 3 + rep))
            t += 1
    return stream


def generate_scenario_4_semantic_drift(seed=SEED) -> list[Phase17Obs]:
    """Generates 5-step progressive drift ending in contradiction shift."""
    rng = random.Random(seed)
    stream = []
    t = 0
    for i in range(20):
        s = SUBJECTS[i]
        p_base = list(DRIFT_SEQUENCES.keys())[i % len(DRIFT_SEQUENCES)]
        o = FACILITIES[i]
        cid = f"drift_{i:04d}"
        seq = DRIFT_SEQUENCES[p_base]

        # Step 0: canonical base
        stream.append(Phase17Obs(Candidate(s, p_base, o, timestamp=t), cid, "canonical", "scenario_4", 0))
        t += 1
        for step_idx, step_pred in enumerate(seq, 1):
            is_contra = (step_idx == len(seq))
            cat = "drift_contra" if is_contra else "drift_step"
            c = Candidate(s, step_pred, o, timestamp=t)
            stream.append(Phase17Obs(c, cid, cat, "scenario_4", step_idx, is_probe=True))
            t += 1
            for _ in range(3):
                stream.append(make_distractor(t, timestamp=t))
                t += 1
    return stream


def generate_scenario_5_distractor_stress(distractor_counts=(50, 100, 200, 500), seed=SEED) -> dict:
    """Evaluates recall as the number of unique background distractor concepts scales."""
    rng = random.Random(seed)
    streams = {}
    for n_dist in distractor_counts:
        stream = []
        t = 0
        # 10 target concepts
        for i in range(10):
            s = SUBJECTS[i]
            p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
            o = FACILITIES[i]
            cid = f"stress_target_{i:04d}"
            stream.append(Phase17Obs(Candidate(s, p, o, timestamp=t), cid, "canonical", "scenario_5", 0))
            t += 1

        # Distractor pool of size n_dist
        distractor_pool = [make_distractor(d, timestamp=0) for d in range(n_dist)]
        for d_obs in distractor_pool:
            d_obs.candidate.timestamp = t
            stream.append(d_obs)
            t += 1

        # Probes for the 10 target concepts
        for i in range(10):
            s = SUBJECTS[i]
            p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
            o = FACILITIES[i]
            cid = f"stress_target_{i:04d}"
            para_p = PARAPHRASE_MAP[p][0]
            stream.append(Phase17Obs(Candidate(s, para_p, o, timestamp=t), cid, "paraphrase", "scenario_5", 1, is_probe=True))
            t += 1

        streams[n_dist] = stream
    return streams


def generate_scenario_6_capacity_pressure_stream(seed=SEED) -> list[Phase17Obs]:
    """Generates a 1,500-observation stream tested at C=1500, C=100, and C=50."""
    rng = random.Random(seed)
    stream = []
    t = 0
    # 100 concepts: canonical, paraphrase, contradiction, paraphrase, plus distractors
    for i in range(100):
        s = SUBJECTS[i]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o = FACILITIES[i]
        cid = f"cap_concept_{i:04d}"
        contra_p = CONTRADICTION_MAP[p]
        paras = PARAPHRASE_MAP[p]

        stream.append(Phase17Obs(Candidate(s, p, o, timestamp=t), cid, "canonical", "scenario_6"))
        t += 1
        stream.append(Phase17Obs(Candidate(s, paras[0], o, timestamp=t), cid, "paraphrase", "scenario_6", is_probe=True))
        t += 1
        stream.append(Phase17Obs(Candidate(s, contra_p, o, timestamp=t), cid, "contradiction", "scenario_6", is_probe=True))
        t += 1
        stream.append(Phase17Obs(Candidate(s, paras[1], o, timestamp=t), cid, "paraphrase", "scenario_6", is_probe=True))
        t += 1
        for _ in range(11):  # 11 distractors per concept -> total 15 obs per concept x 100 = 1,500
            stream.append(make_distractor(t, timestamp=t))
            t += 1

    rng.shuffle(stream)
    for idx, obs in enumerate(stream):
        obs.candidate.timestamp = idx
    assert len(stream) == 1500
    return stream


def generate_scenario_7_continual_stream(n_obs=5000, seed=SEED) -> list[Phase17Obs]:
    """Generates a long continual stream of 5,000 observations."""
    rng = random.Random(seed)
    stream = []
    t = 0
    n_targets = 200

    # 1. Interleave target chains and long distractor stretches
    for i in range(n_targets):
        s = SUBJECTS[i]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o = FACILITIES[i]
        cid = f"cont_{i:04d}"
        contra_p = CONTRADICTION_MAP[p]
        paras = PARAPHRASE_MAP[p]

        stream.append(Phase17Obs(Candidate(s, p, o, timestamp=t), cid, "canonical", "scenario_7"))
        t += 1
        stream.append(Phase17Obs(Candidate(s, paras[0], o, timestamp=t), cid, "paraphrase", "scenario_7", is_probe=True))
        t += 1
        stream.append(Phase17Obs(Candidate(s, contra_p, o, timestamp=t), cid, "contradiction", "scenario_7", is_probe=True))
        t += 1
        stream.append(Phase17Obs(Candidate(s, paras[1 % len(paras)], o, timestamp=t), cid, "paraphrase", "scenario_7", is_probe=True))
        t += 1

    # Fill remaining observations with structured distractors and delayed probes
    remaining = n_obs - len(stream)
    for d_idx in range(remaining):
        if d_idx % 50 == 0:
            # Long-delayed paraphrase probe
            t_idx = (d_idx // 50) % n_targets
            s = SUBJECTS[t_idx]
            p = BASE_PREDICATES[t_idx % len(BASE_PREDICATES)]
            o = FACILITIES[t_idx]
            cid = f"cont_{t_idx:04d}"
            para_p = PARAPHRASE_MAP[p][2 % len(PARAPHRASE_MAP[p])]
            stream.append(Phase17Obs(Candidate(s, para_p, o, timestamp=t), cid, "delayed_paraphrase", "scenario_7", is_probe=True))
        else:
            stream.append(make_distractor(d_idx, timestamp=t))
        t += 1

    for idx, obs in enumerate(stream):
        obs.candidate.timestamp = idx
    assert len(stream) == n_obs
    return stream


if __name__ == "__main__":
    s1 = generate_scenario_1_long_gaps()
    print("Scenario 1 stream lengths:", {g: len(st) for g, st in s1.items()})
    s2 = generate_scenario_2_repeated_contradictions()
    print("Scenario 2 length:", len(s2))
    s3 = generate_scenario_3_contradiction_recovery()
    print("Scenario 3 length:", len(s3))
    s4 = generate_scenario_4_semantic_drift()
    print("Scenario 4 length:", len(s4))
    s5 = generate_scenario_5_distractor_stress()
    print("Scenario 5 stream lengths:", {n: len(st) for n, st in s5.items()})
    s6 = generate_scenario_6_capacity_pressure_stream()
    print("Scenario 6 length:", len(s6))
    s7 = generate_scenario_7_continual_stream(5000)
    print("Scenario 7 length:", len(s7))
