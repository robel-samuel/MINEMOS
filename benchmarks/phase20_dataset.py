"""
benchmarks/phase20_dataset.py

Deterministic Long-Horizon Dataset Generator for Phase 20:
Adaptive Contradiction Anchor Management & Controlled Semantic Drift.

Extends Phase 18/19 datasets with Scenario 7:
Controlled Semantic Drift & Factual Update Benchmark.
Distinguishes between:
  1. Canonical Fact
  2. Malicious / Direct Contradiction
  3. Paraphrase
  4. Legitimate Temporal Update (Factual Drift)
  5. Unrelated Fact
"""

from __future__ import annotations
import os, sys, random
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate
from benchmarks.phase18_dataset import (
    Phase18Obs,
    SUBJECTS,
    FACILITIES,
    PROJECTS,
    DEVICES,
    BASE_PREDICATES,
    PARAPHRASE_MAP,
    CONTRADICTION_MAP,
    SEED,
    make_distractor,
    generate_scenario_1_long_gaps,
    generate_scenario_2_safety_test,
    generate_scenario_5_capacity_pressure_stream,
    generate_scenario_6_continual_stream
)

TEMPORAL_UPDATE_PREDICATES = {
    "visits": "now visits",
    "uses": "currently uses",
    "manages": "now oversees",
    "studies": "currently researches",
    "monitors": "now tracks",
}


def generate_scenario_7_semantic_drift(
    n_concepts: int = 30,
    inter_gap: int = 15,
    seed: int = SEED
) -> list[Phase18Obs]:
    """
    Scenario 7: Controlled Semantic Drift & Factual Update Benchmark.
    
    For each target concept:
      Step 0: Canonical fact: Person_i [predicate] Facility_i
      Interleaved distractors (inter_gap)
      Step 1: Malicious Contradiction: Person_i [contra_predicate] Facility_i
              (Adversarial attack: must NOT merge into canonical slot)
      Interleaved distractors (inter_gap)
      Step 2: Paraphrase probe: Person_i [paraphrase_predicate] Facility_i
              (Legitimate query: SHOULD retrieve & merge into canonical slot)
      Interleaved distractors (inter_gap * 4)  <-- Temporal delay
      Step 3: Legitimate Temporal Update: Person_i [temporal_predicate] Facility_{i+1000}
              (Real-world factual change: new facility assigned to person)
      Interleaved distractors (inter_gap)
      Step 4: Unrelated fact: Person_{i+2000} [predicate] Facility_{i+2000}
              (True negative: completely distinct entity)
    """
    rng = random.Random(seed)
    stream: list[Phase18Obs] = []
    t = 0

    for i in range(n_concepts):
        s = SUBJECTS[i]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o = FACILITIES[i]
        cid = f"drift_{i:04d}"

        # 1. Canonical insert
        c_init = Candidate(s, p, o, timestamp=t)
        stream.append(Phase18Obs(c_init, cid, "canonical", "scenario_7", t, is_probe=False))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 2. Malicious contradiction probe (adversarial)
        contra_p = CONTRADICTION_MAP[p]
        c_contra = Candidate(s, contra_p, o, timestamp=t)
        stream.append(Phase18Obs(c_contra, cid, "contradiction", "scenario_7", t, is_probe=True))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 3. Paraphrase probe
        para_p = PARAPHRASE_MAP[p][0]
        c_para = Candidate(s, para_p, o, timestamp=t)
        stream.append(Phase18Obs(c_para, cid, "paraphrase", "scenario_7", t, is_probe=True))
        t += 1

        # Longer gap simulating temporal passage before real-world update
        for _ in range(inter_gap * 4):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 4. Legitimate Temporal Update (Factual Drift)
        temp_p = TEMPORAL_UPDATE_PREDICATES[p]
        new_o = FACILITIES[(i + 1000) % len(FACILITIES)]
        c_temp = Candidate(s, temp_p, new_o, timestamp=t)
        stream.append(Phase18Obs(c_temp, cid, "temporal_update", "scenario_7", t, is_probe=True))
        t += 1

        for _ in range(inter_gap):
            stream.append(make_distractor(t, timestamp=t))
            t += 1

        # 5. Unrelated entity probe
        unrel_s = SUBJECTS[(i + 2000) % len(SUBJECTS)]
        unrel_o = FACILITIES[(i + 2000) % len(FACILITIES)]
        c_unrel = Candidate(unrel_s, p, unrel_o, timestamp=t)
        stream.append(Phase18Obs(c_unrel, f"unrel_{i:04d}", "unrelated", "scenario_7", t, is_probe=True))
        t += 1

    return stream
