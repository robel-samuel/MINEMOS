"""
benchmarks/phase14_dataset.py

Two datasets, kept separate on purpose (matching Phase 13's pattern of
an aggregate sweep dataset plus an isolated hand-built test):

  generate_aggregate_dataset(): categories A-D (correct repeats,
    paraphrases, hard negatives, unrelated facts) across N_CONCEPTS
    concepts, for the capacity sweep / consolidation metrics.

  generate_contamination_chains(): Category E. For each of 25 dedicated
    concepts, all 5 required ordering variants (canonical->para->para,
    para->canonical->para, canonical->hardneg->para,
    para->hardneg->para, alternating), each followed by a 3-observation
    "recovery" tail of clean canonical repeats (Section 9's recovery
    test). The encoder decides merges naturally -- no fact is ever
    manually forced into a slot.
"""

from __future__ import annotations
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase2.structured_candidate import Candidate

SEED = 1414
N_CONCEPTS = 200
N_CHAIN_CONCEPTS = 25

SUBJECTS = [f"Person_{i:04d}" for i in range(max(N_CONCEPTS, N_CHAIN_CONCEPTS))]
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
OBJECTS = [f"Object_{i:04d}" for i in range(max(N_CONCEPTS, N_CHAIN_CONCEPTS))]


class Obs:
    def __init__(self, candidate: Candidate, concept_id: str, category: str,
                 probes_against: str = None):
        self.candidate = candidate
        self.concept_id = concept_id
        self.category = category
        self.probes_against = probes_against


def generate_aggregate_dataset(seed: int = SEED):
    rng = random.Random(seed)
    observations = []
    for i in range(N_CONCEPTS):
        subject, predicate, obj = SUBJECTS[i], BASE_PREDICATES[i % len(BASE_PREDICATES)], OBJECTS[i]
        concept_id = f"concept_{i:04d}"

        for _ in range(2):
            observations.append(Obs(Candidate(subject, predicate, obj), concept_id, "exact_repeat"))
        for p in rng.sample(PARAPHRASE_POOL[predicate], k=2):
            observations.append(Obs(Candidate(subject, p, obj), concept_id, "paraphrase"))
        hn_id = f"{concept_id}_hardneg"
        observations.append(Obs(Candidate(subject, HARD_NEGATIVE_POOL[predicate], obj), hn_id,
                                 "hard_negative", probes_against=concept_id))
        j = (i + N_CONCEPTS // 2) % N_CONCEPTS
        unrel_id = f"{concept_id}_unrelated"
        observations.append(Obs(
            Candidate(SUBJECTS[j], BASE_PREDICATES[(i + 3) % len(BASE_PREDICATES)], OBJECTS[j]),
            unrel_id, "unrelated", probes_against=concept_id))

    rng.shuffle(observations)
    for t, obs in enumerate(observations):
        obs.candidate.timestamp = t
    return observations


ORDERING_VARIANTS = [
    "canonical_para_para",
    "para_canonical_para",
    "canonical_hardneg_para",
    "para_hardneg_para",
    "alternating",
]


def _chain_sequence(subject, predicate, obj, variant, rng):
    canonical = Candidate(subject, predicate, obj)
    paras = [Candidate(subject, p, obj) for p in PARAPHRASE_POOL[predicate]]
    hardneg = Candidate(subject, HARD_NEGATIVE_POOL[predicate], obj)

    if variant == "canonical_para_para":
        seq = [canonical, paras[0], paras[1]]
    elif variant == "para_canonical_para":
        seq = [paras[0], canonical, paras[1]]
    elif variant == "canonical_hardneg_para":
        seq = [canonical, hardneg, paras[0]]
    elif variant == "para_hardneg_para":
        seq = [paras[0], hardneg, paras[1]]
    elif variant == "alternating":
        seq = [canonical, paras[0], canonical, paras[1], canonical]
    else:
        raise ValueError(variant)

    recovery = [Candidate(subject, predicate, obj) for _ in range(3)]
    return seq, recovery


def generate_contamination_chains(seed: int = SEED + 1):
    rng = random.Random(seed)
    chains = []
    for i in range(N_CHAIN_CONCEPTS):
        subject = f"Chain_{i:03d}"
        predicate = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        obj = f"ChainObject_{i:03d}"
        for variant in ORDERING_VARIANTS:
            concept_id = f"chain_{i:03d}_{variant}"
            seq, recovery = _chain_sequence(subject, predicate, obj, variant, rng)
            for t, c in enumerate(seq):
                c.timestamp = t
            for t, c in enumerate(recovery):
                c.timestamp = len(seq) + t
            chains.append(dict(concept_id=concept_id, subject=subject, predicate=predicate,
                                object=obj, variant=variant, sequence=seq, recovery=recovery))
    return chains


if __name__ == "__main__":
    obs = generate_aggregate_dataset()
    from collections import Counter
    print("aggregate dataset:", len(obs), "observations,", Counter(o.category for o in obs))
    chains = generate_contamination_chains()
    print("contamination chains:", len(chains), "(", N_CHAIN_CONCEPTS, "concepts x", len(ORDERING_VARIANTS), "variants)")
