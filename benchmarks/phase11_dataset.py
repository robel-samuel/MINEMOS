"""
benchmarks/phase11_dataset.py

100 base concepts, each a (subject, predicate, object) fact. For each
concept, generates:
  - N exact repeats (N determined by a fixed frequency bucket)
  - 2 genuine paraphrases (SAME concept_id)
  - 1 contradiction (own new concept_id, explicitly tagged as a
    hard-negative probe against the base concept)
  - 1 related-but-distinct fact (own new concept_id, same subject)
  - 1 unrelated fact (own new concept_id, no relation)

Ground truth (concept_id, category, probes_against) is carried ONLY by
this generator, for scoring. The memory system never sees it except
through the oracle interface, which answers SAME/DIFFERENT only.
"""

from __future__ import annotations
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase2.structured_candidate import Candidate

SEED = 1234
N_CONCEPTS = 100

SUBJECTS = [f"Person_{i:03d}" for i in range(N_CONCEPTS)]
BASE_PREDICATES = ["uses", "teaches", "tests", "owns", "manages", "studies", "practices", "monitors"]
PARAPHRASE_POOL = {
    "uses": ["regularly uses", "works with", "relies on"],
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
RELATED_PREDICATE = {
    "uses": "tests", "teaches": "studies", "tests": "teaches", "owns": "manages",
    "manages": "monitors", "studies": "uses", "practices": "teaches", "monitors": "owns",
}
OBJECTS = [f"Object_{i:03d}" for i in range(N_CONCEPTS)]


class Observation:
    def __init__(self, candidate: Candidate, concept_id: str, category: str,
                 probes_against: str = None, timestamp: float = 0.0):
        self.candidate = candidate
        self.concept_id = concept_id       # ground truth: same underlying fact
        self.category = category           # exact_repeat | paraphrase | contradiction | related_distinct | unrelated
        self.probes_against = probes_against  # concept_id this DIFFERENT observation should stay separate from
        self.timestamp = timestamp
        candidate.timestamp = timestamp


def _repeat_bucket(i: int) -> int:
    """Controlled repetition frequency: 40 concepts x1, 30 x5, 20 x20, 10 x100."""
    if i < 40:
        return 1
    elif i < 70:
        return 5
    elif i < 90:
        return 20
    else:
        return 100


def generate_dataset(seed: int = SEED) -> list:
    rng = random.Random(seed)
    observations = []
    t = 0

    for i in range(N_CONCEPTS):
        subject = SUBJECTS[i]
        predicate = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        obj = OBJECTS[i]
        concept_id = f"concept_{i:03d}"

        # 1. exact repeats
        n_repeats = _repeat_bucket(i)
        for _ in range(n_repeats):
            observations.append(Observation(
                Candidate(subject, predicate, obj), concept_id, "exact_repeat", timestamp=t))
            t += 1

        # 2. genuine paraphrases (2 per concept) -- SAME
        para_preds = rng.sample(PARAPHRASE_POOL[predicate], k=2)
        for p in para_preds:
            observations.append(Observation(
                Candidate(subject, p, obj), concept_id, "paraphrase", timestamp=t))
            t += 1

        # 3. contradiction -- own concept, DIFFERENT, probes against base concept
        contra_pred = ANTONYM_POOL[predicate]
        contra_concept_id = f"{concept_id}_contra"
        observations.append(Observation(
            Candidate(subject, contra_pred, obj), contra_concept_id, "contradiction",
            probes_against=concept_id, timestamp=t))
        t += 1

        # 4. related-but-distinct -- same subject, different predicate, DIFFERENT
        related_pred = RELATED_PREDICATE[predicate]
        related_concept_id = f"{concept_id}_related"
        observations.append(Observation(
            Candidate(subject, related_pred, obj), related_concept_id, "related_distinct",
            probes_against=concept_id, timestamp=t))
        t += 1

        # 5. unrelated -- disjoint subject/predicate/object, DIFFERENT
        j = (i + N_CONCEPTS // 2) % N_CONCEPTS  # deterministic "far" partner
        unrel_concept_id = f"{concept_id}_unrelated"
        observations.append(Observation(
            Candidate(SUBJECTS[j], BASE_PREDICATES[(i + 3) % len(BASE_PREDICATES)], OBJECTS[j]),
            unrel_concept_id, "unrelated", probes_against=concept_id, timestamp=t))
        t += 1

    rng.shuffle(observations)
    # re-timestamp after shuffle so absorption order matches presentation order
    for idx, obs in enumerate(observations):
        obs.timestamp = idx
        obs.candidate.timestamp = idx
    return observations


if __name__ == "__main__":
    obs = generate_dataset()
    from collections import Counter
    print("total observations:", len(obs))
    print("by category:", Counter(o.category for o in obs))
    print("distinct concept_ids:", len(set(o.concept_id for o in obs)))
