"""
benchmarks/phase8_dataset.py

New pairs, not reused from Phase 6 or Phase 7. Categories A-J per the
Phase 8 spec, each with an explicit ground-truth equivalence label.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase2.structured_candidate import Candidate


class LabeledPair:
    def __init__(self, category: str, c1: Candidate, c2: Candidate, equivalent: bool, label: str):
        self.category = category
        self.c1 = c1
        self.c2 = c2
        self.equivalent = equivalent
        self.label = label


PAIRS: list[LabeledPair] = []

# A. Exact duplicates (10)
_exact_facts = [
    ("user", "visits", "the gym"), ("user", "collects", "vinyl records"),
    ("user", "attends", "yoga class"), ("user", "writes", "short stories"),
    ("user", "volunteers at", "the shelter"), ("user", "repairs", "old clocks"),
    ("user", "grows", "tomatoes"), ("user", "practices", "the violin"),
    ("user", "manages", "the warehouse"), ("user", "monitors", "the pipeline"),
]
for s, p, o in _exact_facts:
    PAIRS.append(LabeledPair("exact_duplicate", Candidate(s, p, o), Candidate(s, p, o), True, f"exact_{o.replace(' ','_')}"))

# B. Genuine paraphrases (12)
PAIRS += [
    LabeledPair("paraphrase", Candidate("user", "visits", "the gym"), Candidate("user", "goes to", "the gym"), True, "para_gym"),
    LabeledPair("paraphrase", Candidate("user", "collects", "vinyl records"), Candidate("user", "has a collection of", "vinyl records"), True, "para_vinyl"),
    LabeledPair("paraphrase", Candidate("user", "attends", "yoga class"), Candidate("user", "participates in", "yoga class"), True, "para_yoga"),
    LabeledPair("paraphrase", Candidate("user", "writes", "short stories"), Candidate("user", "authors", "short stories"), True, "para_writes"),
    LabeledPair("paraphrase", Candidate("user", "volunteers at", "the shelter"), Candidate("user", "donates time to", "the shelter"), True, "para_shelter"),
    LabeledPair("paraphrase", Candidate("user", "repairs", "old clocks"), Candidate("user", "fixes", "old clocks"), True, "para_clocks"),
    LabeledPair("paraphrase", Candidate("user", "grows", "tomatoes"), Candidate("user", "cultivates", "tomatoes"), True, "para_tomatoes"),
    LabeledPair("paraphrase", Candidate("user", "practices", "the violin"), Candidate("user", "rehearses", "the violin"), True, "para_violin"),
    LabeledPair("paraphrase", Candidate("user", "manages", "the warehouse"), Candidate("user", "runs", "the warehouse"), True, "para_warehouse"),
    LabeledPair("paraphrase", Candidate("user", "monitors", "the pipeline"), Candidate("user", "keeps an eye on", "the pipeline"), True, "para_pipeline"),
    LabeledPair("paraphrase", Candidate("user", "uses", "Python"), Candidate("user", "does not avoid", "Python"), True, "para_python_double_neg"),
    LabeledPair("paraphrase", Candidate("user", "prefers", "Python"), Candidate("user", "leans toward", "Python"), True, "para_python_leans"),
]

# C. Genuine contradictions (12)
PAIRS += [
    LabeledPair("contradiction", Candidate("user", "visits", "the gym"), Candidate("user", "avoids", "the gym"), False, "contra_gym"),
    LabeledPair("contradiction", Candidate("user", "collects", "vinyl records"), Candidate("user", "got rid of", "vinyl records"), False, "contra_vinyl"),
    LabeledPair("contradiction", Candidate("user", "attends", "yoga class"), Candidate("user", "skips", "yoga class"), False, "contra_yoga"),
    LabeledPair("contradiction", Candidate("user", "writes", "short stories"), Candidate("user", "gave up", "short stories"), False, "contra_writes"),
    LabeledPair("contradiction", Candidate("user", "volunteers at", "the shelter"), Candidate("user", "stopped volunteering at", "the shelter"), False, "contra_shelter"),
    LabeledPair("contradiction", Candidate("user", "repairs", "old clocks"), Candidate("user", "breaks", "old clocks"), False, "contra_clocks"),
    LabeledPair("contradiction", Candidate("user", "grows", "tomatoes"), Candidate("user", "killed", "tomatoes"), False, "contra_tomatoes"),
    LabeledPair("contradiction", Candidate("user", "practices", "the violin"), Candidate("user", "neglects", "the violin"), False, "contra_violin"),
    LabeledPair("contradiction", Candidate("user", "manages", "the warehouse"), Candidate("user", "was fired from", "the warehouse"), False, "contra_warehouse"),
    LabeledPair("contradiction", Candidate("user", "monitors", "the pipeline"), Candidate("user", "ignores", "the pipeline"), False, "contra_pipeline"),
    LabeledPair("contradiction", Candidate("user", "uses", "Python"), Candidate("user", "does not use", "Python"), False, "contra_python_negation"),
    LabeledPair("contradiction", Candidate("user", "prefers", "Python"), Candidate("user", "dislikes", "Python"), False, "contra_python_dislikes"),
]

# D. Same subject, different fact (8)
PAIRS += [
    LabeledPair("same_subj_diff_fact", Candidate("user", "visits", "the gym"), Candidate("user", "collects", "vinyl records"), False, "ssdf_1"),
    LabeledPair("same_subj_diff_fact", Candidate("user", "attends", "yoga class"), Candidate("user", "writes", "short stories"), False, "ssdf_2"),
    LabeledPair("same_subj_diff_fact", Candidate("user", "volunteers at", "the shelter"), Candidate("user", "repairs", "old clocks"), False, "ssdf_3"),
    LabeledPair("same_subj_diff_fact", Candidate("user", "grows", "tomatoes"), Candidate("user", "practices", "the violin"), False, "ssdf_4"),
    LabeledPair("same_subj_diff_fact", Candidate("user", "manages", "the warehouse"), Candidate("user", "monitors", "the pipeline"), False, "ssdf_5"),
    LabeledPair("same_subj_diff_fact", Candidate("user", "visits", "the gym"), Candidate("user", "writes", "short stories"), False, "ssdf_6"),
    LabeledPair("same_subj_diff_fact", Candidate("user", "collects", "vinyl records"), Candidate("user", "manages", "the warehouse"), False, "ssdf_7"),
    LabeledPair("same_subj_diff_fact", Candidate("user", "attends", "yoga class"), Candidate("user", "monitors", "the pipeline"), False, "ssdf_8"),
]

# E. Same predicate, different object (8)
PAIRS += [
    LabeledPair("same_pred_diff_obj", Candidate("user", "visits", "the gym"), Candidate("user", "visits", "the museum"), False, "spdo_1"),
    LabeledPair("same_pred_diff_obj", Candidate("user", "collects", "vinyl records"), Candidate("user", "collects", "stamps"), False, "spdo_2"),
    LabeledPair("same_pred_diff_obj", Candidate("user", "attends", "yoga class"), Candidate("user", "attends", "art class"), False, "spdo_3"),
    LabeledPair("same_pred_diff_obj", Candidate("user", "writes", "short stories"), Candidate("user", "writes", "technical manuals"), False, "spdo_4"),
    LabeledPair("same_pred_diff_obj", Candidate("user", "repairs", "old clocks"), Candidate("user", "repairs", "bicycles"), False, "spdo_5"),
    LabeledPair("same_pred_diff_obj", Candidate("user", "grows", "tomatoes"), Candidate("user", "grows", "peppers"), False, "spdo_6"),
    LabeledPair("same_pred_diff_obj", Candidate("user", "practices", "the violin"), Candidate("user", "practices", "the piano"), False, "spdo_7"),
    LabeledPair("same_pred_diff_obj", Candidate("user", "manages", "the warehouse"), Candidate("user", "manages", "the storefront"), False, "spdo_8"),
]

# F. Same object, different subject (8)
PAIRS += [
    LabeledPair("same_obj_diff_subj", Candidate("user", "visits", "the gym"), Candidate("colleague", "visits", "the gym"), False, "sods_1"),
    LabeledPair("same_obj_diff_subj", Candidate("user", "collects", "vinyl records"), Candidate("friend", "collects", "vinyl records"), False, "sods_2"),
    LabeledPair("same_obj_diff_subj", Candidate("user", "attends", "yoga class"), Candidate("sibling", "attends", "yoga class"), False, "sods_3"),
    LabeledPair("same_obj_diff_subj", Candidate("user", "writes", "short stories"), Candidate("author", "writes", "short stories"), False, "sods_4"),
    LabeledPair("same_obj_diff_subj", Candidate("user", "repairs", "old clocks"), Candidate("technician", "repairs", "old clocks"), False, "sods_5"),
    LabeledPair("same_obj_diff_subj", Candidate("user", "grows", "tomatoes"), Candidate("neighbor", "grows", "tomatoes"), False, "sods_6"),
    LabeledPair("same_obj_diff_subj", Candidate("user", "manages", "the warehouse"), Candidate("supervisor", "manages", "the warehouse"), False, "sods_7"),
    LabeledPair("same_obj_diff_subj", Candidate("user", "uses", "Python"), Candidate("company", "uses", "Python"), False, "sods_8"),
]

# G. High lexical-overlap negatives (10)
PAIRS += [
    LabeledPair("high_overlap_negative", Candidate("user", "uses", "Python"), Candidate("user", "is not using", "Python"), False, "hol_1"),
    LabeledPair("high_overlap_negative", Candidate("user", "visits", "the gym"), Candidate("user", "used to visit", "the gym"), False, "hol_2"),
    LabeledPair("high_overlap_negative", Candidate("user", "collects", "vinyl records"), Candidate("user", "considered collecting", "vinyl records"), False, "hol_3"),
    LabeledPair("high_overlap_negative", Candidate("user", "attends", "yoga class"), Candidate("user", "has stopped attending", "yoga class"), False, "hol_4"),
    LabeledPair("high_overlap_negative", Candidate("user", "writes", "short stories"), Candidate("user", "wishes they wrote", "short stories"), False, "hol_5"),
    LabeledPair("high_overlap_negative", Candidate("user", "grows", "tomatoes"), Candidate("user", "failed to grow", "tomatoes"), False, "hol_6"),
    LabeledPair("high_overlap_negative", Candidate("user", "practices", "the violin"), Candidate("user", "never practices", "the violin"), False, "hol_7"),
    LabeledPair("high_overlap_negative", Candidate("user", "manages", "the warehouse"), Candidate("user", "no longer manages", "the warehouse"), False, "hol_8"),
    LabeledPair("high_overlap_negative", Candidate("user", "monitors", "the pipeline"), Candidate("user", "rarely monitors", "the pipeline"), False, "hol_9"),
    LabeledPair("high_overlap_negative", Candidate("user", "repairs", "old clocks"), Candidate("user", "cannot repair", "old clocks"), False, "hol_10"),
]

# H. Low lexical-overlap paraphrases (10)
PAIRS += [
    LabeledPair("low_overlap_paraphrase", Candidate("user", "visits", "the gym"), Candidate("user", "regularly exercises at", "the fitness center"), True, "lop_1"),
    LabeledPair("low_overlap_paraphrase", Candidate("user", "collects", "vinyl records"), Candidate("user", "amasses", "record albums"), True, "lop_2"),
    LabeledPair("low_overlap_paraphrase", Candidate("user", "writes", "short stories"), Candidate("user", "composes", "brief works of fiction"), True, "lop_3"),
    LabeledPair("low_overlap_paraphrase", Candidate("user", "volunteers at", "the shelter"), Candidate("user", "offers unpaid help to", "the rescue center"), True, "lop_4"),
    LabeledPair("low_overlap_paraphrase", Candidate("user", "repairs", "old clocks"), Candidate("user", "restores", "antique timepieces"), True, "lop_5"),
    LabeledPair("low_overlap_paraphrase", Candidate("user", "grows", "tomatoes"), Candidate("user", "cultivates", "garden vegetables"), True, "lop_6"),
    LabeledPair("low_overlap_paraphrase", Candidate("user", "manages", "the warehouse"), Candidate("user", "is in charge of", "the storage facility"), True, "lop_7"),
    LabeledPair("low_overlap_paraphrase", Candidate("user", "uses", "Python"), Candidate("user", "codes with", "a popular scripting language"), True, "lop_8"),
    LabeledPair("low_overlap_paraphrase", Candidate("user", "practices", "the violin"), Candidate("user", "rehearses", "a stringed instrument"), True, "lop_9"),
    LabeledPair("low_overlap_paraphrase", Candidate("user", "attends", "yoga class"), Candidate("user", "takes part in", "a mindfulness session"), True, "lop_10"),
]

# I. Temporal differences (10) -- ground truth: NOT equivalent (current vs past state differ)
PAIRS += [
    LabeledPair("temporal", Candidate("user", "used to visit", "the gym"), Candidate("user", "currently visits", "the gym"), False, "temp_1"),
    LabeledPair("temporal", Candidate("user", "previously collected", "vinyl records"), Candidate("user", "now collects", "vinyl records"), False, "temp_2"),
    LabeledPair("temporal", Candidate("user", "once wrote", "short stories"), Candidate("user", "still writes", "short stories"), False, "temp_3"),
    LabeledPair("temporal", Candidate("user", "formerly managed", "the warehouse"), Candidate("user", "currently manages", "the warehouse"), False, "temp_4"),
    LabeledPair("temporal", Candidate("user", "used to grow", "tomatoes"), Candidate("user", "presently grows", "tomatoes"), False, "temp_5"),
    LabeledPair("temporal", Candidate("user", "previously used", "Python"), Candidate("user", "currently uses", "Python"), False, "temp_6"),
    LabeledPair("temporal", Candidate("user", "once practiced", "the violin"), Candidate("user", "still practices", "the violin"), False, "temp_7"),
    LabeledPair("temporal", Candidate("user", "formerly volunteered at", "the shelter"), Candidate("user", "currently volunteers at", "the shelter"), False, "temp_8"),
    LabeledPair("temporal", Candidate("user", "used to repair", "old clocks"), Candidate("user", "still repairs", "old clocks"), False, "temp_9"),
    LabeledPair("temporal", Candidate("user", "previously monitored", "the pipeline"), Candidate("user", "currently monitors", "the pipeline"), False, "temp_10"),
]

# J. Subject/entity changes (8) -- same as F conceptually but distinct pairs, per spec's separate listing
PAIRS += [
    LabeledPair("entity_change", Candidate("user", "uses", "Python"), Candidate("company", "uses", "Python"), False, "ec_1"),
    LabeledPair("entity_change", Candidate("user", "visits", "the gym"), Candidate("team", "visits", "the gym"), False, "ec_2"),
    LabeledPair("entity_change", Candidate("user", "manages", "the warehouse"), Candidate("department", "manages", "the warehouse"), False, "ec_3"),
    LabeledPair("entity_change", Candidate("user", "monitors", "the pipeline"), Candidate("system", "monitors", "the pipeline"), False, "ec_4"),
    LabeledPair("entity_change", Candidate("user", "grows", "tomatoes"), Candidate("farm", "grows", "tomatoes"), False, "ec_5"),
    LabeledPair("entity_change", Candidate("user", "repairs", "old clocks"), Candidate("workshop", "repairs", "old clocks"), False, "ec_6"),
    LabeledPair("entity_change", Candidate("user", "writes", "short stories"), Candidate("editor", "writes", "short stories"), False, "ec_7"),
    LabeledPair("entity_change", Candidate("user", "volunteers at", "the shelter"), Candidate("club", "volunteers at", "the shelter"), False, "ec_8"),
]

# Unrelated facts (for M2 margin, matching prior phases' convention) (10)
PAIRS += [
    LabeledPair("unrelated", Candidate("user", "visits", "the gym"), Candidate("server", "processes", "requests"), False, "unrel_1"),
    LabeledPair("unrelated", Candidate("user", "collects", "vinyl records"), Candidate("factory", "produces", "steel"), False, "unrel_2"),
    LabeledPair("unrelated", Candidate("user", "attends", "yoga class"), Candidate("database", "stores", "records"), False, "unrel_3"),
    LabeledPair("unrelated", Candidate("user", "writes", "short stories"), Candidate("truck", "delivers", "packages"), False, "unrel_4"),
    LabeledPair("unrelated", Candidate("user", "volunteers at", "the shelter"), Candidate("printer", "outputs", "documents"), False, "unrel_5"),
    LabeledPair("unrelated", Candidate("user", "repairs", "old clocks"), Candidate("bank", "processes", "transactions"), False, "unrel_6"),
    LabeledPair("unrelated", Candidate("user", "grows", "tomatoes"), Candidate("satellite", "orbits", "earth"), False, "unrel_7"),
    LabeledPair("unrelated", Candidate("user", "practices", "the violin"), Candidate("bridge", "spans", "the river"), False, "unrel_8"),
    LabeledPair("unrelated", Candidate("user", "manages", "the warehouse"), Candidate("volcano", "erupted", "yesterday"), False, "unrel_9"),
    LabeledPair("unrelated", Candidate("user", "monitors", "the pipeline"), Candidate("committee", "reviewed", "the budget"), False, "unrel_10"),
]

assert len(PAIRS) >= 100, f"only {len(PAIRS)} pairs"
