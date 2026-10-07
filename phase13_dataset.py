"""
benchmarks/phase13_dataset.py

500 concepts (documented scope choice: task specifies 500-1000; 500
chosen for tractable runtime, consistent with prior phases' disclosed
scope reductions -- e.g. Phase 2A's compute-time-bounded sweep).

Per concept: exact repeats (A), genuine paraphrases (B), a hard negative
(C), a contradiction (D, own concept), an unrelated fact (E, own
concept). A subset of 50 concepts additionally receive a 5-step
multi-step wording-drift sequence (F), testing whether re-keying remains
useful as wording moves progressively further from the original.

Ground truth (concept_id, category, drift_step) is carried only by this
generator, used for scoring -- the memory system sees only Candidates.
"""

from __future__ import annotations
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase2.structured_candidate import Candidate

SEED = 9001
N_CONCEPTS = 500
N_DRIFT_CONCEPTS = 50

SUBJECTS = [f"Person_{i:04d}" for i in range(N_CONCEPTS)]
BASE_PREDICATES = ["uses", "teaches", "tests", "owns", "manages", "studies", "practices", "monitors"]
PARAPHRASE_POOL = {
    "uses": ["regularly uses", "primarily works with", "relies on"],
    "teaches": ["instructs others in", "mentors people on", "trains students in"],
    "tests": ["evaluates", "experiments with", "trials"],
    "owns": ["possesses", "has", "keeps"],
    "manages": ["oversees", "runs", "supervises"],
    "studies": ["researches", "investigates", "explores"],
    "practices": ["rehearses", "trains in", "hones"],
    "monitors": ["watches", "tracks", "keeps an eye on"],
}
HARD_NEGATIVE_POOL = {
    "uses": "dislikes", "teaches": "criticizes", "tests": "avoids",
    "owns": "borrowed", "manages": "visited", "studies": "mentions",
    "practices": "watched", "monitors": "discussed",
}
ANTONYM_POOL = {
    "uses": "avoids", "teaches": "refuses to teach", "tests": "ignores",
    "owns": "sold", "manages": "was removed from", "studies": "dropped",
    "practices": "gave up", "monitors": "neglects",
}
OBJECTS = [f"Object_{i:04d}" for i in range(N_CONCEPTS)]

DRIFT_SEQUENCE_TEMPLATES = [
    "uses",
    "regularly uses",
    "primarily works with",
    "frequently programs in",
    "writes software using",
]


class Obs:
    def __init__(self, candidate: Candidate, concept_id: str, category: str,
                 drift_step: int = None, probes_against: str = None):
        self.candidate = candidate
        self.concept_id = concept_id
        self.category = category
        self.drift_step = drift_step
        self.probes_against = probes_against


def generate_dataset(seed: int = SEED):
    rng = random.Random(seed)
    observations = []

    # select drift concepts explicitly from the "uses"-predicate subset,
    # rather than intersecting two independent moduli (an earlier version
    # of this function did that by mistake and produced only 13 drift
    # concepts instead of the intended 50 -- caught by inspecting the
    # generator's own summary output before running any downstream code)
    uses_predicate_concepts = [i for i in range(N_CONCEPTS) if BASE_PREDICATES[i % len(BASE_PREDICATES)] == "uses"]
    drift_concept_indices = set(uses_predicate_concepts[:N_DRIFT_CONCEPTS])

    for i in range(N_CONCEPTS):
        subject = SUBJECTS[i]
        predicate = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        obj = OBJECTS[i]
        concept_id = f"concept_{i:04d}"

        for _ in range(2):
            observations.append(Obs(Candidate(subject, predicate, obj), concept_id, "exact_repeat"))

        for p in rng.sample(PARAPHRASE_POOL[predicate], k=2):
            observations.append(Obs(Candidate(subject, p, obj), concept_id, "paraphrase"))

        hn_id = f"{concept_id}_hardneg"
        observations.append(Obs(Candidate(subject, HARD_NEGATIVE_POOL[predicate], obj), hn_id,
                                 "hard_negative", probes_against=concept_id))

        contra_id = f"{concept_id}_contra"
        observations.append(Obs(Candidate(subject, ANTONYM_POOL[predicate], obj), contra_id,
                                 "contradiction", probes_against=concept_id))

        j = (i + N_CONCEPTS // 2) % N_CONCEPTS
        unrel_id = f"{concept_id}_unrelated"
        observations.append(Obs(
            Candidate(SUBJECTS[j], BASE_PREDICATES[(i + 3) % len(BASE_PREDICATES)], OBJECTS[j]),
            unrel_id, "unrelated", probes_against=concept_id))

        if i in drift_concept_indices:
            drift_id = f"{concept_id}_drift"
            for step, wording in enumerate(DRIFT_SEQUENCE_TEMPLATES):
                observations.append(Obs(Candidate(subject, wording, obj), drift_id,
                                         "drift_sequence", drift_step=step))

    rng.shuffle(observations)
    for t, obs in enumerate(observations):
        obs.candidate.timestamp = t
    return observations


if __name__ == "__main__":
    obs = generate_dataset()
    from collections import Counter
    print("total observations:", len(obs))
    print("by category:", Counter(o.category for o in obs))
    print("distinct concept_ids:", len(set(o.concept_id for o in obs)))
