"""
benchmarks/phase15_dataset.py

New, independently constructed dataset (not reused from Phase 13/14).
100 base concepts, split across 5 controlled ordering variants (20 each):
  1. canonical -> paraphrase -> contradiction
  2. paraphrase -> canonical -> contradiction
  3. contradiction -> paraphrase -> canonical
  4. canonical -> contradiction -> paraphrase
  5. clean paraphrases only (no contamination point)

Each chain is 8 observations long, giving delayed/gated confirmation
mechanisms room to operate: the required ordering triple, then
additional paraphrases, then a clean-canonical recovery tail.

Also includes standalone hard-negative and unrelated probes per concept,
used for the aggregate true/false consolidation metrics, following the
pattern established in Phases 13-14.
"""

from __future__ import annotations
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase2.structured_candidate import Candidate

SEED = 1515
N_CONCEPTS = 100

SUBJECTS = [f"Alice_{i:04d}" for i in range(N_CONCEPTS)]
BASE_PREDICATES = ["visits", "uses", "manages", "studies", "monitors"]
PARAPHRASE_POOL = {
    "visits": ["goes to", "frequents", "stops by"],
    "uses": ["regularly uses", "primarily works with", "relies on"],
    "manages": ["oversees", "runs", "supervises"],
    "studies": ["researches", "investigates", "explores"],
    "monitors": ["watches", "tracks", "keeps an eye on"],
}
CONTRADICTION_POOL = {
    "visits": "avoids", "uses": "dislikes", "manages": "was removed from",
    "studies": "dropped", "monitors": "neglects",
}
HARD_NEGATIVE_POOL = {
    "visits": "mentioned", "uses": "tested", "manages": "criticized",
    "studies": "discussed", "monitors": "reviewed",
}
OBJECTS = [f"place_{i:04d}" for i in range(N_CONCEPTS)]

ORDERING_VARIANTS = [
    "canonical_para_contra",
    "para_canonical_contra",
    "contra_para_canonical",
    "canonical_contra_para",
    "clean_paraphrases_only",
]


class Obs:
    def __init__(self, candidate: Candidate, concept_id: str, category: str,
                 variant: str = None, probes_against: str = None):
        self.candidate = candidate
        self.concept_id = concept_id
        self.category = category
        self.variant = variant
        self.probes_against = probes_against


def _build_chain(i: int, rng: random.Random):
    subject = SUBJECTS[i]
    predicate = BASE_PREDICATES[i % len(BASE_PREDICATES)]
    obj = OBJECTS[i]
    concept_id = f"concept_{i:04d}"
    variant = ORDERING_VARIANTS[i % len(ORDERING_VARIANTS)]

    canonical = Candidate(subject, predicate, obj)
    paras = [Candidate(subject, p, obj) for p in PARAPHRASE_POOL[predicate]]
    contra = Candidate(subject, CONTRADICTION_POOL[predicate], obj)

    if variant == "canonical_para_contra":
        head = [canonical, paras[0], contra]
    elif variant == "para_canonical_contra":
        head = [paras[0], canonical, contra]
    elif variant == "contra_para_canonical":
        head = [contra, paras[0], canonical]
    elif variant == "canonical_contra_para":
        head = [canonical, contra, paras[0]]
    else:
        head = [canonical, paras[0], paras[1]]

    tail = [paras[2 % len(paras)], Candidate(subject, predicate, obj),
            Candidate(subject, predicate, obj), Candidate(subject, predicate, obj),
            Candidate(subject, predicate, obj)]
    full_chain = head + tail
    for t, c in enumerate(full_chain):
        c.timestamp = t

    chain_obs = [Obs(c, concept_id, "chain", variant=variant) for c in full_chain]

    hn_id = f"{concept_id}_hardneg"
    hardneg_obs = Obs(Candidate(subject, HARD_NEGATIVE_POOL[predicate], obj), hn_id,
                       "hard_negative", probes_against=concept_id)
    j = (i + N_CONCEPTS // 2) % N_CONCEPTS
    unrel_id = f"{concept_id}_unrelated"
    unrel_obs = Obs(Candidate(SUBJECTS[j], BASE_PREDICATES[(i + 2) % len(BASE_PREDICATES)], OBJECTS[j]),
                     unrel_id, "unrelated", probes_against=concept_id)

    return chain_obs, hardneg_obs, unrel_obs, dict(concept_id=concept_id, subject=subject,
                                                     predicate=predicate, object=obj, variant=variant,
                                                     chain=full_chain)


def generate_aggregate_dataset(seed: int = SEED):
    rng = random.Random(seed)
    observations = []
    for i in range(N_CONCEPTS):
        chain_obs, hardneg_obs, unrel_obs, _ = _build_chain(i, rng)
        observations.extend(chain_obs)
        observations.append(hardneg_obs)
        observations.append(unrel_obs)
    rng.shuffle(observations)
    for t, obs in enumerate(observations):
        obs.candidate.timestamp = t
    return observations


def generate_chain_specs(seed: int = SEED):
    rng = random.Random(seed)
    specs = []
    for i in range(N_CONCEPTS):
        _, _, _, spec = _build_chain(i, rng)
        specs.append(spec)
    return specs


if __name__ == "__main__":
    obs = generate_aggregate_dataset()
    from collections import Counter
    print("aggregate dataset:", len(obs), Counter(o.category for o in obs))
    specs = generate_chain_specs()
    print("chain specs:", len(specs), Counter(s["variant"] for s in specs))
