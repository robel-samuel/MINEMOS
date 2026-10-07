"""
benchmarks/phase12_dataset.py

100 concepts, each assigned an ordering case (canonical_first,
paraphrase_first, mixed) and a repeat-count bucket (2/5/20/100).
Within a concept, observation order is fixed by its case. Across
concepts, observations are interleaved (round-robin over per-concept
queues, deterministic RNG) so absorption order isn't trivially
"concept 0 fully processed, then concept 1" -- while each concept's
internal ordering (the actual variable under test) is preserved exactly.
"""

from __future__ import annotations
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase2.structured_candidate import Candidate

SEED = 4242
N_CONCEPTS = 100

SUBJECTS = [f"Alice_{i:03d}" for i in range(N_CONCEPTS)]
BASE_PREDICATES = ["uses", "teaches", "tests", "owns", "manages", "studies", "practices", "monitors"]
PARAPHRASE_POOL = {
    "uses": ["regularly programs in", "works with", "relies on"],
    "teaches": ["instructs others in", "mentors people on", "trains students in"],
    "tests": ["evaluates", "experiments with", "trials"],
    "owns": ["possesses", "has", "keeps"],
    "manages": ["oversees", "runs", "supervises"],
    "studies": ["researches", "investigates", "explores"],
    "practices": ["rehearses", "trains in", "hones"],
    "monitors": ["watches", "tracks", "keeps an eye on"],
}
ANTONYM_POOL = {
    "uses": "avoids", "teaches": "refuses to teach", "tests": "ignores",
    "owns": "sold", "manages": "was removed from", "studies": "dropped",
    "practices": "gave up", "monitors": "neglects",
}
OBJECTS = [f"Object_{i:03d}" for i in range(N_CONCEPTS)]

CASES = ["canonical_first", "paraphrase_first", "mixed"]


class Obs:
    def __init__(self, candidate: Candidate, concept_id: str, category: str,
                 ordering_case: str, wording_role: str, probes_against: str = None):
        self.candidate = candidate
        self.concept_id = concept_id
        self.category = category
        self.ordering_case = ordering_case
        self.wording_role = wording_role
        self.probes_against = probes_against


def _repeat_bucket(i: int) -> int:
    if i < 40: return 2
    elif i < 70: return 5
    elif i < 90: return 20
    else: return 100


class ConceptSpec:
    def __init__(self, i: int, rng: random.Random):
        self.i = i
        self.concept_id = f"concept_{i:03d}"
        self.subject = SUBJECTS[i]
        self.predicate = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        self.object = OBJECTS[i]
        self.ordering_case = CASES[i % len(CASES)]
        self.repeat_count = _repeat_bucket(i)
        self.paraphrase_predicates = rng.sample(PARAPHRASE_POOL[self.predicate], k=2)
        used = set(self.paraphrase_predicates)
        self.held_out_predicate = next(p for p in PARAPHRASE_POOL[self.predicate] if p not in used)


def _build_concept_queue(spec: ConceptSpec, rng: random.Random) -> list:
    canonical_obs = [Obs(Candidate(spec.subject, spec.predicate, spec.object, timestamp=0),
                          spec.concept_id, "canonical", spec.ordering_case, "canonical")
                      for _ in range(spec.repeat_count)]
    para_obs = [Obs(Candidate(spec.subject, p, spec.object, timestamp=0),
                     spec.concept_id, "paraphrase", spec.ordering_case, f"para_{idx}")
                for idx, p in enumerate(spec.paraphrase_predicates)]

    if spec.ordering_case == "canonical_first":
        queue = canonical_obs + para_obs
    elif spec.ordering_case == "paraphrase_first":
        queue = [para_obs[0]] + canonical_obs + [para_obs[1]]
    else:
        queue = canonical_obs + para_obs
        rng.shuffle(queue)

    contra_id = f"{spec.concept_id}_contra"
    queue.append(Obs(Candidate(spec.subject, ANTONYM_POOL[spec.predicate], spec.object, timestamp=0),
                      contra_id, "contradiction", spec.ordering_case, "contradiction",
                      probes_against=spec.concept_id))
    j = (spec.i + N_CONCEPTS // 2) % N_CONCEPTS
    unrel_id = f"{spec.concept_id}_unrelated"
    queue.append(Obs(Candidate(SUBJECTS[j], BASE_PREDICATES[(spec.i + 3) % len(BASE_PREDICATES)], OBJECTS[j], timestamp=0),
                      unrel_id, "unrelated", spec.ordering_case, "unrelated",
                      probes_against=spec.concept_id))
    return queue


def generate_dataset(seed: int = SEED):
    rng = random.Random(seed)
    specs = [ConceptSpec(i, rng) for i in range(N_CONCEPTS)]
    queues = [_build_concept_queue(spec, rng) for spec in specs]

    interleave_rng = random.Random(seed + 1)
    observations = []
    remaining = list(range(len(queues)))
    while remaining:
        idx = interleave_rng.choice(remaining)
        observations.append(queues[idx].pop(0))
        if not queues[idx]:
            remaining.remove(idx)

    for t, obs in enumerate(observations):
        obs.candidate.timestamp = t

    return observations, specs


if __name__ == "__main__":
    obs, specs = generate_dataset()
    from collections import Counter
    print("total observations:", len(obs))
    print("by category:", Counter(o.category for o in obs))
    print("by ordering_case (canonical/paraphrase only):",
          Counter(o.ordering_case for o in obs if o.category in ("canonical", "paraphrase")))
