"""
benchmarks/phase7_dataset.py

New pairs, not reused from Phase 5 or Phase 6. Every pair has an explicit
ground-truth label. Categories per the Phase 7 spec.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase2.structured_candidate import Candidate


class LabeledPair:
    def __init__(self, category: str, c1: Candidate, c2: Candidate,
                 equivalent: bool, label: str):
        self.category = category
        self.c1 = c1
        self.c2 = c2
        self.equivalent = equivalent  # ground truth: same underlying fact?
        self.label = label


PAIRS: list[LabeledPair] = []

# 1. Genuine paraphrases (8)
PAIRS += [
    LabeledPair("paraphrase", Candidate("user", "runs", "marathons"), Candidate("user", "competes in", "marathons"), True, "para_marathon"),
    LabeledPair("paraphrase", Candidate("user", "speaks", "French"), Candidate("user", "is fluent in", "French"), True, "para_french"),
    LabeledPair("paraphrase", Candidate("user", "drives", "a Honda"), Candidate("user", "owns", "a Honda"), True, "para_honda"),
    LabeledPair("paraphrase", Candidate("user", "studies", "physics"), Candidate("user", "majors in", "physics"), True, "para_physics"),
    LabeledPair("paraphrase", Candidate("user", "cooks", "Italian food"), Candidate("user", "prepares", "Italian food"), True, "para_cooking"),
    LabeledPair("paraphrase", Candidate("user", "reads", "sci-fi novels"), Candidate("user", "enjoys reading", "sci-fi novels"), True, "para_scifi"),
    LabeledPair("paraphrase", Candidate("user", "plays", "chess"), Candidate("user", "is skilled at", "chess"), True, "para_chess"),
    LabeledPair("paraphrase", Candidate("user", "supervises", "the finance team"), Candidate("user", "oversees", "the finance team"), True, "para_finance_team"),
]

# 2. Genuine contradictions (8)
PAIRS += [
    LabeledPair("contradiction", Candidate("user", "runs", "marathons"), Candidate("user", "refuses to run", "marathons"), False, "contra_marathon"),
    LabeledPair("contradiction", Candidate("user", "speaks", "French"), Candidate("user", "cannot speak", "French"), False, "contra_french"),
    LabeledPair("contradiction", Candidate("user", "drives", "a Honda"), Candidate("user", "sold", "a Honda"), False, "contra_honda"),
    LabeledPair("contradiction", Candidate("user", "studies", "physics"), Candidate("user", "dropped", "physics"), False, "contra_physics"),
    LabeledPair("contradiction", Candidate("user", "cooks", "Italian food"), Candidate("user", "dislikes", "Italian food"), False, "contra_cooking"),
    LabeledPair("contradiction", Candidate("user", "reads", "sci-fi novels"), Candidate("user", "hates", "sci-fi novels"), False, "contra_scifi"),
    LabeledPair("contradiction", Candidate("user", "plays", "chess"), Candidate("user", "quit", "chess"), False, "contra_chess"),
    LabeledPair("contradiction", Candidate("user", "supervises", "the finance team"), Candidate("user", "was removed from", "the finance team"), False, "contra_finance_team"),
]

# 3. Unrelated facts (8)
PAIRS += [
    LabeledPair("unrelated", Candidate("user", "runs", "marathons"), Candidate("company", "manufactures", "furniture"), False, "unrel_1"),
    LabeledPair("unrelated", Candidate("user", "speaks", "French"), Candidate("colleague", "repairs", "bicycles"), False, "unrel_2"),
    LabeledPair("unrelated", Candidate("user", "drives", "a Honda"), Candidate("project", "requires", "approval"), False, "unrel_3"),
    LabeledPair("unrelated", Candidate("user", "studies", "physics"), Candidate("system", "logs", "errors"), False, "unrel_4"),
    LabeledPair("unrelated", Candidate("user", "cooks", "Italian food"), Candidate("device", "measures", "temperature"), False, "unrel_5"),
    LabeledPair("unrelated", Candidate("user", "reads", "sci-fi novels"), Candidate("team", "shipped", "the release"), False, "unrel_6"),
    LabeledPair("unrelated", Candidate("user", "plays", "chess"), Candidate("warehouse", "stores", "inventory"), False, "unrel_7"),
    LabeledPair("unrelated", Candidate("user", "supervises", "the finance team"), Candidate("server", "handles", "requests"), False, "unrel_8"),
]

# 4. Same subject, different predicate (unrelated info about same entity) (6)
PAIRS += [
    LabeledPair("same_subj_diff_pred", Candidate("user", "runs", "marathons"), Candidate("user", "speaks", "French"), False, "ssdp_1"),
    LabeledPair("same_subj_diff_pred", Candidate("user", "drives", "a Honda"), Candidate("user", "studies", "physics"), False, "ssdp_2"),
    LabeledPair("same_subj_diff_pred", Candidate("user", "cooks", "Italian food"), Candidate("user", "reads", "sci-fi novels"), False, "ssdp_3"),
    LabeledPair("same_subj_diff_pred", Candidate("user", "plays", "chess"), Candidate("user", "supervises", "the finance team"), False, "ssdp_4"),
    LabeledPair("same_subj_diff_pred", Candidate("user", "runs", "marathons"), Candidate("user", "drives", "a Honda"), False, "ssdp_5"),
    LabeledPair("same_subj_diff_pred", Candidate("user", "speaks", "French"), Candidate("user", "cooks", "Italian food"), False, "ssdp_6"),
]

# 5. Same predicate, different object (6)
PAIRS += [
    LabeledPair("same_pred_diff_obj", Candidate("user", "runs", "marathons"), Candidate("user", "runs", "sprints"), False, "spdo_1"),
    LabeledPair("same_pred_diff_obj", Candidate("user", "speaks", "French"), Candidate("user", "speaks", "German"), False, "spdo_2"),
    LabeledPair("same_pred_diff_obj", Candidate("user", "drives", "a Honda"), Candidate("user", "drives", "a Toyota"), False, "spdo_3"),
    LabeledPair("same_pred_diff_obj", Candidate("user", "studies", "physics"), Candidate("user", "studies", "chemistry"), False, "spdo_4"),
    LabeledPair("same_pred_diff_obj", Candidate("user", "plays", "chess"), Candidate("user", "plays", "poker"), False, "spdo_5"),
    LabeledPair("same_pred_diff_obj", Candidate("user", "reads", "sci-fi novels"), Candidate("user", "reads", "biographies"), False, "spdo_6"),
]

# 6. Same object, different subject (6)
PAIRS += [
    LabeledPair("same_obj_diff_subj", Candidate("user", "runs", "marathons"), Candidate("colleague", "runs", "marathons"), False, "sods_1"),
    LabeledPair("same_obj_diff_subj", Candidate("user", "speaks", "French"), Candidate("company", "speaks", "French"), False, "sods_2"),
    LabeledPair("same_obj_diff_subj", Candidate("user", "drives", "a Honda"), Candidate("neighbor", "drives", "a Honda"), False, "sods_3"),
    LabeledPair("same_obj_diff_subj", Candidate("user", "studies", "physics"), Candidate("sibling", "studies", "physics"), False, "sods_4"),
    LabeledPair("same_obj_diff_subj", Candidate("user", "plays", "chess"), Candidate("friend", "plays", "chess"), False, "sods_5"),
    LabeledPair("same_obj_diff_subj", Candidate("user", "cooks", "Italian food"), Candidate("chef", "cooks", "Italian food"), False, "sods_6"),
]

# 7. Lexical-overlap distractors -- share literal words but mean something
#    different; designed to potentially fool a pure word-overlap encoder (6)
PAIRS += [
    LabeledPair("lexical_distractor", Candidate("user", "uses", "Python"), Candidate("user", "is not using", "Python"), False, "lex_1"),
    LabeledPair("lexical_distractor", Candidate("user", "runs", "marathons"), Candidate("user", "no longer runs", "marathons"), False, "lex_2"),
    LabeledPair("lexical_distractor", Candidate("user", "speaks", "French"), Candidate("user", "used to speak", "French"), False, "lex_3"),
    LabeledPair("lexical_distractor", Candidate("user", "plays", "chess"), Candidate("user", "has stopped playing", "chess"), False, "lex_4"),
    LabeledPair("lexical_distractor", Candidate("user", "studies", "physics"), Candidate("user", "considered studying", "physics"), False, "lex_5"),
    LabeledPair("lexical_distractor", Candidate("user", "cooks", "Italian food"), Candidate("user", "rarely cooks", "Italian food"), True, "lex_6_weak_paraphrase"),
]

# 8. Exact duplicates (6)
PAIRS += [
    LabeledPair("exact_duplicate", Candidate("user", "runs", "marathons"), Candidate("user", "runs", "marathons"), True, "exact_1"),
    LabeledPair("exact_duplicate", Candidate("user", "speaks", "French"), Candidate("user", "speaks", "French"), True, "exact_2"),
    LabeledPair("exact_duplicate", Candidate("user", "drives", "a Honda"), Candidate("user", "drives", "a Honda"), True, "exact_3"),
    LabeledPair("exact_duplicate", Candidate("user", "studies", "physics"), Candidate("user", "studies", "physics"), True, "exact_4"),
    LabeledPair("exact_duplicate", Candidate("user", "plays", "chess"), Candidate("user", "plays", "chess"), True, "exact_5"),
    LabeledPair("exact_duplicate", Candidate("user", "supervises", "the finance team"), Candidate("user", "supervises", "the finance team"), True, "exact_6"),
]

assert len(PAIRS) >= 50, f"only {len(PAIRS)} pairs"
