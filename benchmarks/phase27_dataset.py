"""
benchmarks/phase27_dataset.py

Dataset generator and benchmark suites for MINEMOS Phase 27:
Adaptive Memory Retrieval Under Long-Horizon Interference.

Supports four interference conditions:
  - Condition A: Low interference (clean baseline, few neutral distractors)
  - Condition B: Semantic interference (high density of semantically similar distractors)
  - Condition C: Temporal interference (heavy factual supersessions and outdated facts)
  - Condition D: Combined interference (semantic distractors + temporal revisions + 
                 contradictory probes + paraphrases over long horizon)

Includes:
  - Development stream generator
  - Held-out stream generator (strictly disjoint entity and relation pool)
  - Integrity and leakage verification
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import random

from phase2.structured_candidate import Candidate


@dataclass
class StreamObservation:
    candidate: Candidate
    obs_type: str  # 'target_init', 'target_paraphrase', 'temporal_revision', 'contra_probe', 'semantic_distractor', 'lexical_distractor', 'neutral_distractor'
    fact_id: str
    target_query_ids: List[str] = field(default_factory=list)
    supersedes_fact_id: Optional[str] = None
    is_valid_at_end: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalQuery:
    query_id: str
    query_text: str
    target_subject: str
    target_predicate: str
    expected_answer: str
    expected_fact_text: str
    as_of: Optional[float] = None
    query_time: float = 1000.0
    condition: str = "A_low_interference"
    superseded_fact_texts: List[str] = field(default_factory=list)
    distractor_fact_texts: List[str] = field(default_factory=list)
    is_held_out: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Entity and Domain Pools
# ─────────────────────────────────────────────────────────────

DEV_ENTITIES = [
    ("Rowan", "astrophysicist", "observes stellar pulsars"),
    ("Seraphina", "microbiologist", "isolates thermophilic enzymes"),
    ("Thaddeus", "archaeologist", "excavates bronze age tumuli"),
    ("Valerie", "cryptographer", "designs lattice-based ciphers"),
    ("Zephyr", "quantum physicist", "measures topological invariants"),
    ("Cassian", "marine biologist", "surveys abyssal hydrothermal vents"),
    ("Orion", "aeronautical engineer", "optimizes scramjet combustion nozzles"),
    ("Lyra", "paleontologist", "classifies feathered theropod fossils"),
    ("Eldridge", "chemical synthesizer", "refines chiral catalysts"),
    ("Morrigan", "roboticist", "implements swarm navigation algorithms"),
]

HELD_OUT_ENTITIES = [
    ("Persephone", "geophysicist", "measures mantle shear wave velocities"),
    ("Barnaby", "volcanologist", "samples subglacial basaltic tephra"),
    ("Corinna", "neurobiologist", "maps optogenetic synaptic circuits"),
    ("Leopold", "urban architect", "designs passive solar skyscrapers"),
    ("Astrid", "astronomer", "tracks interstellar hyperbolic comets"),
    ("Theron", "materials scientist", "synthesizes metallic glass alloys"),
    ("Ignatius", "glaciologist", "measures Antarctic ice sheet calving"),
    ("Solene", "linguist", "deciphers proto-Elamite clay tokens"),
]

# Distractor templates for semantic interference (Condition B & D)
SEMANTIC_DISTRACTOR_TEMPLATES = [
    ("{sub} analyzed high-resolution spectra of {obj}.", "analyzed spectra", "spectral analysis"),
    ("{sub} published preliminary findings regarding {obj}.", "published findings", "academic publication"),
    ("{sub} reviewed archival field notes concerning {obj}.", "reviewed notes", "archival review"),
    ("{sub} calibrated laser spectrometers to detect {obj}.", "calibrated instruments", "instrument calibration"),
    ("{sub} coordinated an international symposium on {obj}.", "coordinated symposium", "conference coordination"),
    ("{sub} secured capital equipment grants to investigate {obj}.", "secured grant", "grant funding"),
    ("{sub} drafted an exhaustive monograph addressing {obj}.", "drafted monograph", "monograph drafting"),
    ("{sub} criticized prevailing theories related to {obj}.", "criticized theories", "scholarly critique"),
]

LEXICAL_DISTRACTOR_TEMPLATES = [
    ("The expedition near {obj} was organized by researchers.", "organized expedition", "expedition"),
    ("Historical accounts of {obj} were cataloged in the library.", "cataloged accounts", "library catalog"),
    ("A workshop on {obj} took place in the department.", "held workshop", "department workshop"),
    ("New sensors for measuring {obj} arrived at the laboratory.", "received sensors", "sensors"),
]


# ─────────────────────────────────────────────────────────────
# Stream Generator
# ─────────────────────────────────────────────────────────────

class Phase27StreamGenerator:
    """
    Generates reproducible benchmark observation streams and queries
    for testing retrieval under long-horizon interference.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_benchmark_suite(
        self,
        stream_lengths: List[int] = [25, 50, 100],
        is_held_out: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generates full test suites across all 4 conditions (A, B, C, D)
        for specified stream lengths.
        """
        conditions = ["A_low_interference", "B_semantic_interference", "C_temporal_interference", "D_combined_interference"]
        suite = {}
        for cond in conditions:
            suite[cond] = {}
            for length in stream_lengths:
                obs, queries = self.generate_stream(condition=cond, length=length, is_held_out=is_held_out)
                suite[cond][str(length)] = {
                    "observations": obs,
                    "queries": queries,
                    "condition": cond,
                    "length": length,
                    "is_held_out": is_held_out,
                    "seed": self.seed
                }
        return suite

    def generate_stream(
        self,
        condition: str,
        length: int = 50,
        is_held_out: bool = False
    ) -> Tuple[List[StreamObservation], List[RetrievalQuery]]:
        """
        Builds an observation stream and corresponding retrieval queries.
        """
        entities = HELD_OUT_ENTITIES if is_held_out else DEV_ENTITIES
        self.rng.seed(self.seed + hash(condition) % 10000 + length)

        observations: List[StreamObservation] = []
        queries: List[RetrievalQuery] = []

        # 1. Target facts (always present in all conditions)
        # We select 4 primary target entities
        target_indices = [0, 1, 2, 3] if len(entities) >= 4 else list(range(len(entities)))
        selected_targets = [entities[i] for i in target_indices]

        # Scenario setup for target entities:
        # e.g., Target 0: Affiliation / Position
        # Target 1: Key Research Discovery / Project
        # Target 2: Award / Grant / Metric
        # Target 3: Location / Facility
        t_base = 100.0

        # --- Target 0: Rowan / Persephone (Affiliation) ---
        sub0, field0, _ = selected_targets[0]
        f0_v1_text = f"{sub0} serves as chief scientist at Helion Dynamics."
        f0_v2_text = f"{sub0} was appointed research director at Astraea Technologies."
        f0_para_text = f"{sub0} directs scientific research at Astraea Technologies."

        # --- Target 1: Seraphina / Barnaby (Location/Facility) ---
        sub1, field1, _ = selected_targets[1]
        f1_v1_text = f"{sub1} conducts primary research at the Atacama High Observatory."
        f1_v2_text = f"{sub1} relocated laboratory operations to the Svalbard Polar Station."

        # --- Target 2: Thaddeus / Corinna (Metric/Scalar) ---
        sub2, field2, _ = selected_targets[2]
        f2_v1_text = f"{sub2} secured a research grant of 15 million euros."
        f2_v2_text = f"{sub2} secured an expanded research endowment of 45 million euros."

        # --- Target 3: Valerie / Leopold (Algorithm/System) ---
        sub3, field3, _ = selected_targets[3]
        f3_v1_text = f"{sub3} deployed the Falcon-7 cryogenic cooling array."
        f3_contra_text = f"{sub3} dismantled the Falcon-7 cryogenic cooling array."

        # Base insertions
        obs_f0_v1 = StreamObservation(
            candidate=Candidate(sub0, "serves as chief scientist at", "Helion Dynamics", timestamp=t_base + 1.0),
            obs_type="target_init",
            fact_id=f"{sub0}_affil_v1",
            target_query_ids=[f"q_{sub0}_affil_hist", f"q_{sub0}_affil_curr"]
        )
        observations.append(obs_f0_v1)

        obs_f1_v1 = StreamObservation(
            candidate=Candidate(sub1, "conducts primary research at", "the Atacama High Observatory", timestamp=t_base + 2.0),
            obs_type="target_init",
            fact_id=f"{sub1}_loc_v1",
            target_query_ids=[f"q_{sub1}_loc_curr"]
        )
        observations.append(obs_f1_v1)

        obs_f2_v1 = StreamObservation(
            candidate=Candidate(sub2, "secured a research grant of", "15 million euros", timestamp=t_base + 3.0),
            obs_type="target_init",
            fact_id=f"{sub2}_grant_v1",
            target_query_ids=[f"q_{sub2}_grant_curr"]
        )
        observations.append(obs_f2_v1)

        obs_f3_v1 = StreamObservation(
            candidate=Candidate(sub3, "deployed the", "Falcon-7 cryogenic cooling array", timestamp=t_base + 4.0),
            obs_type="target_init",
            fact_id=f"{sub3}_sys_v1",
            target_query_ids=[f"q_{sub3}_sys_curr"]
        )
        observations.append(obs_f3_v1)

        # Distractor entities pool (strictly non-target to prevent false intra-entity supersession)
        distractor_entities = entities[4:]

        # Distractor & Interference Construction according to condition
        current_t = t_base + 10.0

        if condition == "A_low_interference":
            # Condition A: Clean stream. Few neutral distractors (20% ratio).
            # A couple of target paraphrases, no hard contradictions or heavy supersessions.
            # Add paraphrase for Target 0
            obs_f0_para = StreamObservation(
                candidate=Candidate(sub0, "directs research operations at", "Helion Dynamics", timestamp=current_t),
                obs_type="target_paraphrase",
                fact_id=f"{sub0}_affil_para",
                target_query_ids=[f"q_{sub0}_affil_curr"]
            )
            observations.append(obs_f0_para)
            current_t += 5.0

            # Fill remaining with neutral distractors
            n_distractors = max(2, length - len(observations))
            for i in range(n_distractors):
                ent = distractor_entities[i % len(distractor_entities)]
                cand = Candidate(ent[0], "presented findings at", f"Symposium Session {i + 1}", timestamp=current_t)
                observations.append(StreamObservation(
                    candidate=cand,
                    obs_type="neutral_distractor",
                    fact_id=f"dist_neutral_{i}",
                    is_valid_at_end=False
                ))
                current_t += 2.0

            # Queries for Condition A
            queries.append(RetrievalQuery(
                query_id=f"q_{sub0}_affil_curr",
                query_text=f"Where does {sub0} serve as chief scientist or direct research?",
                target_subject=sub0,
                target_predicate="serves as chief scientist at",
                expected_answer="Helion Dynamics",
                expected_fact_text=f0_v1_text,
                condition=condition,
                is_held_out=is_held_out
            ))
            queries.append(RetrievalQuery(
                query_id=f"q_{sub1}_loc_curr",
                query_text=f"At which research facility does {sub1} conduct primary research?",
                target_subject=sub1,
                target_predicate="conducts primary research at",
                expected_answer="the Atacama High Observatory",
                expected_fact_text=f1_v1_text,
                condition=condition,
                is_held_out=is_held_out
            ))
            queries.append(RetrievalQuery(
                query_id=f"q_{sub2}_grant_curr",
                query_text=f"What grant funding amount did {sub2} secure for research?",
                target_subject=sub2,
                target_predicate="secured a research grant of",
                expected_answer="15 million euros",
                expected_fact_text=f2_v1_text,
                condition=condition,
                is_held_out=is_held_out
            ))
            queries.append(RetrievalQuery(
                query_id=f"q_{sub3}_sys_curr",
                query_text=f"What cooling system or hardware array was deployed by {sub3}?",
                target_subject=sub3,
                target_predicate="deployed the",
                expected_answer="Falcon-7 cryogenic cooling array",
                expected_fact_text=f3_v1_text,
                condition=condition,
                is_held_out=is_held_out
            ))

        elif condition == "B_semantic_interference":
            # Condition B: Dense semantic distractors.
            # Many statements with similar topics (cryogenic cooling, grants, observatories, research director)
            # but belonging to OTHER entities.
            n_semantic = max(6, length - len(observations))
            semantic_objects = [
                "Helion Dynamics aerospace division",
                "the Mauna Kea Submillimeter Observatory",
                "a research grant of 18 million euros",
                "Falcon-9 superconductivity matrix",
                "Astraea Technologies advanced propulsion lab",
                "the European Southern Observatory cluster",
                "an equipment subsidy of 12 million euros",
                "Falcon-5 thermal dissipation module",
            ]
            for i in range(n_semantic):
                ent = distractor_entities[i % len(distractor_entities)]
                tmpl = SEMANTIC_DISTRACTOR_TEMPLATES[i % len(SEMANTIC_DISTRACTOR_TEMPLATES)]
                obj = semantic_objects[i % len(semantic_objects)]
                text = tmpl[0].format(sub=ent[0], obj=obj)
                cand = Candidate(ent[0], tmpl[1], obj, timestamp=current_t)
                observations.append(StreamObservation(
                    candidate=cand,
                    obs_type="semantic_distractor",
                    fact_id=f"dist_sem_{i}",
                    is_valid_at_end=False
                ))
                current_t += 2.0

            # Queries for Condition B
            queries.append(RetrievalQuery(
                query_id=f"q_{sub0}_affil_curr",
                query_text=f"Which organization does {sub0} lead as chief scientist?",
                target_subject=sub0,
                target_predicate="serves as chief scientist at",
                expected_answer="Helion Dynamics",
                expected_fact_text=f0_v1_text,
                condition=condition,
                distractor_fact_texts=[f"{distractor_entities[0][0]} analyzed spectra of Helion Dynamics aerospace division."],
                is_held_out=is_held_out
            ))
            queries.append(RetrievalQuery(
                query_id=f"q_{sub1}_loc_curr",
                query_text=f"At which research observatory does {sub1} conduct primary research?",
                target_subject=sub1,
                target_predicate="conducts primary research at",
                expected_answer="the Atacama High Observatory",
                expected_fact_text=f1_v1_text,
                condition=condition,
                distractor_fact_texts=["the Mauna Kea Submillimeter Observatory", "the European Southern Observatory cluster"],
                is_held_out=is_held_out
            ))
            queries.append(RetrievalQuery(
                query_id=f"q_{sub2}_grant_curr",
                query_text=f"How much grant money did {sub2} secure for laboratory research?",
                target_subject=sub2,
                target_predicate="secured a research grant of",
                expected_answer="15 million euros",
                expected_fact_text=f2_v1_text,
                condition=condition,
                distractor_fact_texts=["a research grant of 18 million euros", "an equipment subsidy of 12 million euros"],
                is_held_out=is_held_out
            ))
            queries.append(RetrievalQuery(
                query_id=f"q_{sub3}_sys_curr",
                query_text=f"Which specific cryogenic cooling system did {sub3} deploy?",
                target_subject=sub3,
                target_predicate="deployed the",
                expected_answer="Falcon-7 cryogenic cooling array",
                expected_fact_text=f3_v1_text,
                condition=condition,
                distractor_fact_texts=["Falcon-9 superconductivity matrix", "Falcon-5 thermal dissipation module"],
                is_held_out=is_held_out
            ))

        elif condition == "C_temporal_interference":
            # Condition C: Temporal supersessions.
            # Target 0: Helion Dynamics -> Astraea Technologies (t=300) -> Paraphrase (t=350)
            # Target 1: Atacama -> Svalbard (t=320)
            # Target 2: 15M -> 45M (t=340)
            t_rev = current_t + 50.0

            # Revise Target 0
            obs_f0_v2 = StreamObservation(
                candidate=Candidate(sub0, "was appointed research director at", "Astraea Technologies", timestamp=t_rev),
                obs_type="temporal_revision",
                fact_id=f"{sub0}_affil_v2",
                target_query_ids=[f"q_{sub0}_affil_curr"],
                supersedes_fact_id=f"{sub0}_affil_v1",
                is_valid_at_end=True
            )
            obs_f0_v1.is_valid_at_end = False
            observations.append(obs_f0_v2)

            # Revise Target 1
            obs_f1_v2 = StreamObservation(
                candidate=Candidate(sub1, "relocated laboratory operations to", "the Svalbard Polar Station", timestamp=t_rev + 10.0),
                obs_type="temporal_revision",
                fact_id=f"{sub1}_loc_v2",
                target_query_ids=[f"q_{sub1}_loc_curr"],
                supersedes_fact_id=f"{sub1}_loc_v1",
                is_valid_at_end=True
            )
            obs_f1_v1.is_valid_at_end = False
            observations.append(obs_f1_v2)

            # Revise Target 2
            obs_f2_v2 = StreamObservation(
                candidate=Candidate(sub2, "secured an expanded research endowment of", "45 million euros", timestamp=t_rev + 20.0),
                obs_type="temporal_revision",
                fact_id=f"{sub2}_grant_v2",
                target_query_ids=[f"q_{sub2}_grant_curr"],
                supersedes_fact_id=f"{sub2}_grant_v1",
                is_valid_at_end=True
            )
            obs_f2_v1.is_valid_at_end = False
            observations.append(obs_f2_v2)

            current_t = t_rev + 30.0

            # Add subsequent paraphrase for revised Target 0
            obs_f0_para = StreamObservation(
                candidate=Candidate(sub0, "directs scientific research at", "Astraea Technologies", timestamp=current_t),
                obs_type="target_paraphrase",
                fact_id=f"{sub0}_affil_v2_para",
                target_query_ids=[f"q_{sub0}_affil_curr"],
                is_valid_at_end=True
            )
            observations.append(obs_f0_para)
            current_t += 5.0

            # Fill remainder with neutral fillers
            while len(observations) < length:
                idx = len(observations)
                ent = distractor_entities[idx % len(distractor_entities)]
                observations.append(StreamObservation(
                    candidate=Candidate(ent[0], "archived records in", f"Database Shard {idx}", timestamp=current_t),
                    obs_type="neutral_distractor",
                    fact_id=f"fill_{idx}",
                    is_valid_at_end=False
                ))
                current_t += 2.0

            # Queries for Condition C: test both CURRENT active state and HISTORICAL point-in-time
            queries.append(RetrievalQuery(
                query_id=f"q_{sub0}_affil_curr",
                query_text=f"What is {sub0}'s current institution or scientific leadership role?",
                target_subject=sub0,
                target_predicate="was appointed research director at",
                expected_answer="Astraea Technologies",
                expected_fact_text=f0_v2_text,
                as_of=None,  # Current / latest
                superseded_fact_texts=[f0_v1_text],
                condition=condition,
                is_held_out=is_held_out
            ))
            queries.append(RetrievalQuery(
                query_id=f"q_{sub0}_affil_hist",
                query_text=f"Where was {sub0} chief scientist prior to the leadership transition?",
                target_subject=sub0,
                target_predicate="serves as chief scientist at",
                expected_answer="Helion Dynamics",
                expected_fact_text=f0_v1_text,
                as_of=t_base + 50.0,  # Historical point in time
                superseded_fact_texts=[f0_v2_text],
                condition=condition,
                is_held_out=is_held_out
            ))
            queries.append(RetrievalQuery(
                query_id=f"q_{sub1}_loc_curr",
                query_text=f"Where does {sub1} currently conduct research operations?",
                target_subject=sub1,
                target_predicate="relocated laboratory operations to",
                expected_answer="the Svalbard Polar Station",
                expected_fact_text=f1_v2_text,
                as_of=None,
                superseded_fact_texts=[f1_v1_text],
                condition=condition,
                is_held_out=is_held_out
            ))
            queries.append(RetrievalQuery(
                query_id=f"q_{sub2}_grant_curr",
                query_text=f"What is the total expanded endowment amount secured by {sub2}?",
                target_subject=sub2,
                target_predicate="secured an expanded research endowment of",
                expected_answer="45 million euros",
                expected_fact_text=f2_v2_text,
                as_of=None,
                superseded_fact_texts=[f2_v1_text],
                condition=condition,
                is_held_out=is_held_out
            ))

        elif condition == "D_combined_interference":
            # Condition D: Combined interference.
            # Long-horizon stream with:
            #   - Target facts (t=100)
            #   - Semantically similar distractors (t=120..150)
            #   - Direct contradictory probes (t=160)
            #   - Temporal state revisions (t=200)
            #   - Post-revision paraphrases (t=220)
            #   - Distractor flood causing capacity eviction pressure (t=250..)
            t_inter = current_t + 20.0

            # 1. Add semantic distractors
            for i in range(4):
                ent = distractor_entities[i % len(distractor_entities)]
                cand = Candidate(ent[0], "consulted for", f"Helion Dynamics branch {i + 1}", timestamp=t_inter)
                observations.append(StreamObservation(
                    candidate=cand,
                    obs_type="semantic_distractor",
                    fact_id=f"dist_sem_{i}",
                    is_valid_at_end=False
                ))
                t_inter += 2.0

            # 2. Add contradictory probe for Target 3 (Valerie/Leopold)
            obs_f3_contra = StreamObservation(
                candidate=Candidate(sub3, "dismantled the", "Falcon-7 cryogenic cooling array", timestamp=t_inter + 5.0),
                obs_type="contra_probe",
                fact_id=f"{sub3}_sys_contra",
                supersedes_fact_id=f"{sub3}_sys_v1",
                is_valid_at_end=False
            )
            observations.append(obs_f3_contra)
            t_inter += 10.0

            # 3. Add temporal revisions for Target 0 and Target 1
            obs_f0_v2 = StreamObservation(
                candidate=Candidate(sub0, "was appointed research director at", "Astraea Technologies", timestamp=t_inter),
                obs_type="temporal_revision",
                fact_id=f"{sub0}_affil_v2",
                target_query_ids=[f"q_{sub0}_affil_curr"],
                supersedes_fact_id=f"{sub0}_affil_v1",
                is_valid_at_end=True
            )
            obs_f0_v1.is_valid_at_end = False
            observations.append(obs_f0_v2)

            obs_f1_v2 = StreamObservation(
                candidate=Candidate(sub1, "relocated laboratory operations to", "the Svalbard Polar Station", timestamp=t_inter + 5.0),
                obs_type="temporal_revision",
                fact_id=f"{sub1}_loc_v2",
                target_query_ids=[f"q_{sub1}_loc_curr"],
                supersedes_fact_id=f"{sub1}_loc_v1",
                is_valid_at_end=True
            )
            obs_f1_v1.is_valid_at_end = False
            observations.append(obs_f1_v2)
            t_inter += 15.0

            # 4. Add paraphrase for Target 0
            obs_f0_para = StreamObservation(
                candidate=Candidate(sub0, "directs scientific research at", "Astraea Technologies", timestamp=t_inter),
                obs_type="target_paraphrase",
                fact_id=f"{sub0}_affil_v2_para",
                target_query_ids=[f"q_{sub0}_affil_curr"],
                is_valid_at_end=True
            )
            observations.append(obs_f0_para)
            t_inter += 5.0

            # 5. Fill remaining length with long-horizon distractor sequence
            fill_count = 0
            while len(observations) < length:
                fill_count += 1
                ent = distractor_entities[fill_count % len(distractor_entities)]
                observations.append(StreamObservation(
                    candidate=Candidate(ent[0], "logged observation packet", f"Telemetry block {fill_count}", timestamp=t_inter),
                    obs_type="neutral_distractor",
                    fact_id=f"stream_fill_{fill_count}",
                    is_valid_at_end=False
                ))
                t_inter += 3.0

            # Queries for Condition D: Full multi-faceted battery
            queries.append(RetrievalQuery(
                query_id=f"q_{sub0}_affil_curr",
                query_text=f"What is {sub0}'s current executive research position and company?",
                target_subject=sub0,
                target_predicate="was appointed research director at",
                expected_answer="Astraea Technologies",
                expected_fact_text=f0_v2_text,
                as_of=None,
                superseded_fact_texts=[f0_v1_text],
                distractor_fact_texts=[f"{entities[4][0]} consulted for Helion Dynamics branch 1"],
                condition=condition,
                is_held_out=is_held_out
            ))
            queries.append(RetrievalQuery(
                query_id=f"q_{sub0}_affil_hist",
                query_text=f"Where was {sub0} employed as chief scientist before moving to Astraea Technologies?",
                target_subject=sub0,
                target_predicate="serves as chief scientist at",
                expected_answer="Helion Dynamics",
                expected_fact_text=f0_v1_text,
                as_of=t_base + 10.0,
                superseded_fact_texts=[f0_v2_text],
                condition=condition,
                is_held_out=is_held_out
            ))
            queries.append(RetrievalQuery(
                query_id=f"q_{sub1}_loc_curr",
                query_text=f"To which research outpost did {sub1} relocate primary operations?",
                target_subject=sub1,
                target_predicate="relocated laboratory operations to",
                expected_answer="the Svalbard Polar Station",
                expected_fact_text=f1_v2_text,
                as_of=None,
                superseded_fact_texts=[f1_v1_text],
                condition=condition,
                is_held_out=is_held_out
            ))
            queries.append(RetrievalQuery(
                query_id=f"q_{sub2}_grant_curr",
                query_text=f"What research grant amount was originally allocated to {sub2}?",
                target_subject=sub2,
                target_predicate="secured a research grant of",
                expected_answer="15 million euros",
                expected_fact_text=f2_v1_text,
                as_of=None,
                condition=condition,
                is_held_out=is_held_out
            ))

        return observations, queries


def verify_phase27_dataset_integrity() -> Dict[str, Any]:
    """
    Validates dataset generators for:
      - Determinism across same seed
      - Entity disjointness between DEV and HELD_OUT
      - Correct observation and query structures
    """
    gen1 = Phase27StreamGenerator(seed=42)
    dev_obs, dev_q = gen1.generate_stream("D_combined_interference", length=50, is_held_out=False)

    gen2 = Phase27StreamGenerator(seed=42)
    dev_obs2, dev_q2 = gen2.generate_stream("D_combined_interference", length=50, is_held_out=False)

    # 1. Determinism check
    assert len(dev_obs) == len(dev_obs2), "Non-deterministic observation count"
    assert [o.candidate.subject for o in dev_obs] == [o.candidate.subject for o in dev_obs2], "Non-deterministic stream"

    # 2. Held out disjointness check
    gen_held = Phase27StreamGenerator(seed=42)
    held_obs, held_q = gen_held.generate_stream("D_combined_interference", length=50, is_held_out=True)

    dev_subs = set(e[0] for e in DEV_ENTITIES)
    held_subs = set(e[0] for e in HELD_OUT_ENTITIES)
    overlap = dev_subs.intersection(held_subs)
    assert len(overlap) == 0, f"Entity leakage between dev and held-out: {overlap}"

    held_stream_subs = set(o.candidate.subject for o in held_obs)
    leakage = dev_subs.intersection(held_stream_subs)
    assert len(leakage) == 0, f"Dev entities leaked into held-out stream: {leakage}"

    return {
        "status": "PASS",
        "dev_stream_length": len(dev_obs),
        "held_stream_length": len(held_obs),
        "dev_queries": len(dev_q),
        "held_queries": len(held_q),
        "entity_overlap": list(overlap),
        "stream_leakage": list(leakage)
    }
