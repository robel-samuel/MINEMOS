"""
benchmarks/phase16_dataset.py

Deterministic dataset generator for Phase 16:
1. Pairwise Evaluation Set: 350 balanced pairs across 7 categories (50 each):
   - Exact duplicates (SAME)
   - Lexical paraphrases (SAME)
   - Predicate contradictions (CONTRADICTION)
   - Subject contradictions (DIFFERENT)
   - Object contradictions (DIFFERENT)
   - Unrelated facts (UNRELATED)
   - Hard negatives (CONTRADICTION)
2. Long Paraphrase Chains (8 observations of progressive lexical drift)
3. Contradiction Chains (8 observations with controlled placement of contradictory hard negatives)
4. Contamination Chains (100 concepts x 5 ordering variants with 4-step recovery tail)
5. Aggregate Dataset (1,000 observations across 100 concepts for capacity sweeps)
"""

from __future__ import annotations
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate

SEED = 1616
N_CONCEPTS = 100

SUBJECTS = [f"Person_{i:04d}" for i in range(N_CONCEPTS)]
OBJECTS = [f"Facility_{i:04d}" for i in range(N_CONCEPTS)]

BASE_PREDICATES = ["visits", "uses", "manages", "studies", "monitors"]

PARAPHRASE_MAP = {
    "visits": ["goes to", "frequents", "stops by", "attends", "travels to", "drops in at"],
    "uses": ["regularly uses", "primarily works with", "relies on", "employs", "operates", "utilizes"],
    "manages": ["oversees", "runs", "supervises", "leads", "directs", "is in charge of"],
    "studies": ["researches", "investigates", "explores", "examines", "analyzes", "inspects"],
    "monitors": ["watches", "tracks", "keeps an eye on", "observes", "surveys", "checks on"],
}

CONTRADICTION_MAP = {
    "visits": "avoids",
    "uses": "dislikes",
    "manages": "was removed from",
    "studies": "dropped",
    "monitors": "neglects",
}

HARD_NEGATIVE_MAP = {
    "visits": "avoids",
    "uses": "dislikes",
    "manages": "criticized",
    "studies": "quit",
    "monitors": "ignores",
}

ORDERING_VARIANTS = [
    "canonical_para_contra",
    "para_canonical_contra",
    "contra_para_canonical",
    "canonical_contra_para",
    "clean_paraphrases_only",
]


class PairwiseExample:
    def __init__(self, cand1: Candidate, cand2: Candidate, ground_truth: str,
                 subcategory: str, concept_id: str):
        self.cand1 = cand1
        self.cand2 = cand2
        self.ground_truth = ground_truth   # "SAME", "CONTRADICTION", "DIFFERENT", "UNRELATED"
        self.subcategory = subcategory     # e.g. "exact_duplicate", "lexical_paraphrase", etc.
        self.concept_id = concept_id

    @property
    def text1(self) -> str:
        return f"{self.cand1.subject} {self.cand1.predicate} {self.cand1.object}."

    @property
    def text2(self) -> str:
        return f"{self.cand2.subject} {self.cand2.predicate} {self.cand2.object}."


class Obs:
    def __init__(self, candidate: Candidate, concept_id: str, category: str,
                 variant: str = None, probes_against: str = None):
        self.candidate = candidate
        self.concept_id = concept_id
        self.category = category           # "chain", "hard_negative", "unrelated"
        self.variant = variant
        self.probes_against = probes_against


def generate_pairwise_dataset(seed: int = SEED) -> list[PairwiseExample]:
    """Generates 350 balanced pairs across 7 categories (50 each)."""
    pairs = []
    sample_indices = list(range(50))

    # 1. Exact duplicates (SAME)
    for i in sample_indices:
        s, p, o = SUBJECTS[i], BASE_PREDICATES[i % len(BASE_PREDICATES)], OBJECTS[i]
        c1 = Candidate(s, p, o)
        c2 = Candidate(s, p, o)
        pairs.append(PairwiseExample(c1, c2, "SAME", "exact_duplicate", f"concept_{i:04d}"))

    # 2. Lexical paraphrases (SAME)
    for i in sample_indices:
        s, p, o = SUBJECTS[i], BASE_PREDICATES[i % len(BASE_PREDICATES)], OBJECTS[i]
        para_p = PARAPHRASE_MAP[p][i % len(PARAPHRASE_MAP[p])]
        c1 = Candidate(s, p, o)
        c2 = Candidate(s, para_p, o)
        pairs.append(PairwiseExample(c1, c2, "SAME", "lexical_paraphrase", f"concept_{i:04d}"))

    # 3. Predicate contradictions (CONTRADICTION)
    for i in sample_indices:
        s, p, o = SUBJECTS[i], BASE_PREDICATES[i % len(BASE_PREDICATES)], OBJECTS[i]
        contra_p = CONTRADICTION_MAP[p]
        c1 = Candidate(s, p, o)
        c2 = Candidate(s, contra_p, o)
        pairs.append(PairwiseExample(c1, c2, "CONTRADICTION", "predicate_contradiction", f"concept_{i:04d}"))

    # 4. Subject contradictions (DIFFERENT)
    for i in sample_indices:
        s1 = SUBJECTS[i]
        s2 = SUBJECTS[(i + 50) % N_CONCEPTS]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o = OBJECTS[i]
        c1 = Candidate(s1, p, o)
        c2 = Candidate(s2, p, o)
        pairs.append(PairwiseExample(c1, c2, "DIFFERENT", "subject_contradiction", f"concept_{i:04d}"))

    # 5. Object contradictions (DIFFERENT)
    for i in sample_indices:
        s = SUBJECTS[i]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o1 = OBJECTS[i]
        o2 = OBJECTS[(i + 50) % N_CONCEPTS]
        c1 = Candidate(s, p, o1)
        c2 = Candidate(s, p, o2)
        pairs.append(PairwiseExample(c1, c2, "DIFFERENT", "object_contradiction", f"concept_{i:04d}"))

    # 6. Unrelated facts (UNRELATED)
    for i in sample_indices:
        s1, p1, o1 = SUBJECTS[i], BASE_PREDICATES[i % len(BASE_PREDICATES)], OBJECTS[i]
        j = (i + 25) % N_CONCEPTS
        s2, p2, o2 = SUBJECTS[j], BASE_PREDICATES[(i + 2) % len(BASE_PREDICATES)], OBJECTS[j]
        c1 = Candidate(s1, p1, o1)
        c2 = Candidate(s2, p2, o2)
        pairs.append(PairwiseExample(c1, c2, "UNRELATED", "unrelated", f"concept_{i:04d}"))

    # 7. Hard negatives (CONTRADICTION)
    for i in sample_indices:
        s, p, o = SUBJECTS[i], BASE_PREDICATES[i % len(BASE_PREDICATES)], OBJECTS[i]
        hn_p = HARD_NEGATIVE_MAP[p]
        c1 = Candidate(s, p, o)
        c2 = Candidate(s, hn_p, o)
        pairs.append(PairwiseExample(c1, c2, "CONTRADICTION", "hard_negative", f"concept_{i:04d}"))

    assert len(pairs) == 350
    return pairs


def _build_chain(i: int, rng: random.Random):
    subject = SUBJECTS[i]
    predicate = BASE_PREDICATES[i % len(BASE_PREDICATES)]
    obj = OBJECTS[i]
    concept_id = f"concept_{i:04d}"
    variant = ORDERING_VARIANTS[i % len(ORDERING_VARIANTS)]

    canonical = Candidate(subject, predicate, obj)
    paras = [Candidate(subject, p, obj) for p in PARAPHRASE_MAP[predicate]]
    contra = Candidate(subject, CONTRADICTION_MAP[predicate], obj)

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

    # Step 3 is a paraphrase, followed by a 4-step canonical recovery tail
    tail = [
        paras[2 % len(paras)],
        Candidate(subject, predicate, obj),
        Candidate(subject, predicate, obj),
        Candidate(subject, predicate, obj),
        Candidate(subject, predicate, obj)
    ]
    full_chain = head + tail
    for t, c in enumerate(full_chain):
        c.timestamp = t

    chain_obs = [Obs(c, concept_id, "chain", variant=variant) for c in full_chain]

    hn_id = f"{concept_id}_hardneg"
    hardneg_obs = Obs(Candidate(subject, HARD_NEGATIVE_MAP[predicate], obj), hn_id,
                       "hard_negative", probes_against=concept_id)

    j = (i + N_CONCEPTS // 2) % N_CONCEPTS
    unrel_id = f"{concept_id}_unrelated"
    unrel_obs = Obs(Candidate(SUBJECTS[j], BASE_PREDICATES[(i + 2) % len(BASE_PREDICATES)], OBJECTS[j]),
                     unrel_id, "unrelated", probes_against=concept_id)

    spec = dict(
        concept_id=concept_id, subject=subject, predicate=predicate,
        object=obj, variant=variant, chain=full_chain, head_len=len(head)
    )
    return chain_obs, hardneg_obs, unrel_obs, spec


def generate_chain_specs(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    specs = []
    for i in range(N_CONCEPTS):
        _, _, _, spec = _build_chain(i, rng)
        specs.append(spec)
    return specs


def generate_long_chains(seed: int = SEED) -> list[dict]:
    """Generates extended 8-step chains for long-range paraphrase drift."""
    rng = random.Random(seed)
    long_chains = []
    for i in range(20):
        s = SUBJECTS[i]
        p = BASE_PREDICATES[i % len(BASE_PREDICATES)]
        o = OBJECTS[i]
        cid = f"long_concept_{i:04d}"
        pool = PARAPHRASE_MAP[p]
        steps = [Candidate(s, p, o, timestamp=0)]
        for t, para_p in enumerate(pool[:6], 1):
            steps.append(Candidate(s, para_p, o, timestamp=t))
        steps.append(Candidate(s, p, o, timestamp=len(steps)))
        long_chains.append(dict(concept_id=cid, subject=s, predicate=p, object=o, chain=steps))
    return long_chains


def generate_aggregate_dataset(seed: int = SEED) -> list[Obs]:
    """Generates 1,000 observations (800 chain obs + 100 hard negatives + 100 unrelated)."""
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


if __name__ == "__main__":
    p_data = generate_pairwise_dataset()
    print(f"Pairwise dataset: {len(p_data)} pairs")
    specs = generate_chain_specs()
    print(f"Chain specs: {len(specs)} concepts")
    agg = generate_aggregate_dataset()
    print(f"Aggregate dataset: {len(agg)} observations")
