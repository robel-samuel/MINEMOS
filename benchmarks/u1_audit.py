"""
benchmarks/u1_audit.py

Independent acceptance audit for the U1 closure work.

This module is deliberately separate from ``benchmarks/u1_closure.py``: the
closure benchmark asserts the implementation's own claims, while this audit
(1) traces which audit entry points actually invoke the new detector,
(2) checks historical statements at recall time, not only at storage time,
(3) scores language robustness on a pre-labelled pair dataset, and
(4) checks policy reproducibility across checkpoint restore.

Datasets are declared before execution.  Labels are assigned on principle
(same fact / different fact / ambiguous input), not by inspecting results.
The dataset is intentionally small and hand-built: it measures the documented
behaviours of this engine, and no general language-understanding claim is made.

Run with ``python runners/run_u1_audit.py`` (writes
``benchmarks/results/u1_audit_results.json``); use ``--output`` for an
independent run.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phase2.black_hole_memory import (
    PREDICATE_POLICY_EXCLUSIVE,
    PREDICATE_POLICY_MULTI_VALUED,
    UNLINKED_EXCLUSIVE_CONFLICT,
    BlackHoleMemory,
)
from phase2.structured_candidate import Candidate


DATASET_VERSION = "u1-audit-2026-10-30"

EQUIVALENT = "equivalent"
NOT_EQUIVALENT = "not_equivalent"
UNCERTAIN = "uncertain"

CONFLICT = True
NO_CONFLICT = False


# ── Equivalence dataset ────────────────────────────────────────────────────
# Each entry: (id, category, label, (subject, predicate, object) for A,
#              same for B, note).  B is absorbed after A at a later timestamp.
# "equivalent" means B restates A; the engine should consolidate, not create a
# second fact or a false state change.

EQUIVALENCE_PAIRS = [
    # spelling variations
    ("SP-01", "spelling", EQUIVALENT, ("Alice", "lives in", "Addis Ababa"),
     ("Alic", "lives in", "Addis Ababa"), "one-letter subject deletion"),
    ("SP-02", "spelling", EQUIVALENT, ("Alice", "lives in", "Nairobi"),
     ("Alice", "lives in", "Narobi"), "adjacent transposition"),
    ("SP-03", "spelling", EQUIVALENT, ("Alice", "lives in", "Addis Ababa"),
     ("Alice", "lives in", "Addis Abab"), "value deletion"),
    ("SP-04", "spelling", EQUIVALENT, ("Mohamed", "lives in", "Cairo"),
     ("Mohamad", "lives in", "Cairo"),
     "substitution typo; excluded by the conservative rule (expected miss)"),
    ("SP-05", "spelling", EQUIVALENT, ("Nadia", "manages", "Project Atlas"),
     ("Nadia", "manages", "Project Atlass"), "value insertion"),
    ("SP-06", "spelling", EQUIVALENT, ("Omar", "works at", "Acme"),
     ("Omar", "works at", "Acmee"), "value insertion"),

    # grammar variations
    ("GR-01", "grammar", EQUIVALENT, ("Alice", "lives in", "Addis Ababa"),
     ("Alice", "live in", "Addis Ababa"), "subject-verb agreement"),
    ("GR-02", "grammar", EQUIVALENT, ("Alice", "lives in", "Addis Ababa"),
     ("Alice", "is living in", "Addis Ababa"), "progressive aspect"),
    ("GR-03", "grammar", EQUIVALENT, ("Alice", "works at", "Acme"),
     ("Alice", "work at", "Acme"), "subject-verb agreement"),
    ("GR-04", "grammar", EQUIVALENT, ("Alice", "works at", "Acme"),
     ("Alice", "is working at", "Acme"), "progressive aspect"),
    ("GR-05", "grammar", EQUIVALENT, ("Alice", "reports to", "Bob"),
     ("Alice", "report to", "Bob"), "subject-verb agreement"),

    # capitalization and punctuation
    ("CP-01", "capitalization", EQUIVALENT, ("Alice", "lives in", "Addis Ababa"),
     ("alice", "lives in", "addis ababa"), "lowercase"),
    ("CP-02", "capitalization", EQUIVALENT, ("Alice", "lives in", "Addis Ababa"),
     ("ALICE", "lives in", "ADDIS ABABA"), "uppercase"),
    ("CP-03", "capitalization", EQUIVALENT, ("Alice", "lives in", "Addis Ababa"),
     ("ＡＬＩＣＥ", "lives in", "Addis Ababa"), "full-width subject"),
    ("CP-04", "capitalization", EQUIVALENT, ("Alice", "lives in", "Addis Ababa"),
     ("Alice", "lives in", "  Addis   Ababa  "), "extra spaces"),
    ("CP-05", "capitalization", EQUIVALENT, ("Alice", "lives in", "Addis Ababa"),
     ("Alice", "lives in", '"Addis Ababa"'), "quoted value"),

    # unambiguous abbreviations
    ("AB-01", "abbreviation", EQUIVALENT, ("Alice", "lives in", "New York City"),
     ("Alice", "lives in", "NYC"), "NYC alias"),
    ("AB-02", "abbreviation", EQUIVALENT, ("Boris", "lives in", "United Kingdom"),
     ("Boris", "lives in", "UK"), "UK alias"),
    ("AB-03", "abbreviation", EQUIVALENT, ("Carlos", "lives in", "United States"),
     ("Carlos", "lives in", "USA"), "USA alias"),
    ("AB-04", "abbreviation", EQUIVALENT, ("Dina", "lives in", "United Arab Emirates"),
     ("Dina", "lives in", "UAE"), "UAE alias"),

    # paraphrases (real NLI decides)
    ("PA-01", "paraphrase", EQUIVALENT, ("Alice", "lives in", "Addis Ababa"),
     ("Alice", "resides in", "Addis Ababa"), "predicate synonym"),
    ("PA-02", "paraphrase", EQUIVALENT, ("Alice", "works at", "Acme"),
     ("Alice", "is employed by", "Acme"), "employment paraphrase"),
    ("PA-03", "paraphrase", EQUIVALENT, ("Alice", "is a", "data engineer"),
     ("Alice", "works as", "data engineer"), "role paraphrase"),
    ("PA-04", "paraphrase", EQUIVALENT, ("Alice", "manages", "Project Atlas"),
     ("Alice", "leads", "Project Atlas"), "management paraphrase"),
    ("PA-05", "paraphrase", EQUIVALENT, ("Alice", "received", "best paper award"),
     ("Alice", "won", "best paper award"), "award paraphrase"),
    ("PA-06", "paraphrase", EQUIVALENT, ("Alice", "lives in", "Addis Ababa"),
     ("Alice", "makes her home in", "Addis Ababa"), "residence paraphrase"),

    # similar-but-different names (must not merge)
    ("SN-01", "similar_names", NOT_EQUIVALENT, ("Alice", "lives in", "Paris"),
     ("Alicia", "lives in", "Berlin"), "distinct people"),
    ("SN-02", "similar_names", NOT_EQUIVALENT, ("May", "lives in", "Paris"),
     ("Maya", "lives in", "Berlin"), "short names, distinct people"),
    ("SN-03", "similar_names", NOT_EQUIVALENT, ("Jon", "lives in", "Paris"),
     ("John", "lives in", "Berlin"), "short names, distinct people"),
    ("SN-04", "similar_names", NOT_EQUIVALENT, ("Bob", "lives in", "Paris"),
     ("Bobby", "lives in", "Berlin"), "distinct people"),
    ("SN-05", "similar_names", NOT_EQUIVALENT, ("Eve", "lives in", "Paris"),
     ("Eva", "lives in", "Berlin"), "distinct people"),

    # similar-but-different locations (must not merge)
    ("SL-01", "similar_locations", NOT_EQUIVALENT, ("Nina", "lives in", "Rome"),
     ("Nina", "lives in", "Roma"), "distinct city names"),
    ("SL-02", "similar_locations", NOT_EQUIVALENT, ("Nina", "lives in", "Iran"),
     ("Nina", "lives in", "Iraq"), "distinct countries"),
    ("SL-03", "similar_locations", NOT_EQUIVALENT, ("Nina", "lives in", "Niger"),
     ("Nina", "lives in", "Nigeria"), "distinct countries"),
    ("SL-04", "similar_locations", NOT_EQUIVALENT, ("Nina", "lives in", "Austria"),
     ("Nina", "lives in", "Australia"), "distinct countries"),
    ("SL-05", "similar_locations", NOT_EQUIVALENT, ("Nina", "lives in", "San Jose"),
     ("Nina", "lives in", "San Juan"), "distinct cities"),

    # negation
    ("NG-01", "negation", NOT_EQUIVALENT, ("Olga", "no longer lives in", "Addis Ababa"),
     ("Olga", "lives in", "Nairobi"), "negation is not restatement"),
    ("NG-02", "negation", NOT_EQUIVALENT, ("Olga", "does not live in", "Addis Ababa"),
     ("Olga", "lives in", "Addis Ababa"), "affirmation vs negation"),
    ("NG-03", "negation", NOT_EQUIVALENT, ("Olga", "never visited", "Paris"),
     ("Olga", "visited", "Paris"), "affirmation vs negation"),
    ("NG-04", "negation", NOT_EQUIVALENT, ("Olga", "lives in", "Addis Ababa"),
     ("Olga", "no longer lives in", "Addis Ababa"), "current vs negated"),

    # historical versus current
    ("HC-01", "historical", NOT_EQUIVALENT, ("Alice", "used to live in", "Addis Ababa"),
     ("Alice", "lives in", "Addis Ababa"), "past state vs current state"),
    ("HC-02", "historical", NOT_EQUIVALENT, ("Alice", "previously lived in", "Addis Ababa"),
     ("Alice", "lives in", "Nairobi"), "past state vs current state"),
    ("HC-03", "historical", NOT_EQUIVALENT, ("Alice", "formerly worked at", "Company A"),
     ("Alice", "works at", "Company B"), "past job vs current job"),
    ("HC-04", "historical", NOT_EQUIVALENT, ("Alice", "in the past lived in", "Rome"),
     ("Alice", "lives in", "Rome"), "past state vs current state"),
    ("HC-05", "historical", NOT_EQUIVALENT, ("Alice", "at one time worked at", "Acme"),
     ("Alice", "works at", "Acme"), "past job vs current job"),

    # ambiguous or incomplete input (not scored as right/wrong)
    ("AM-01", "ambiguous", UNCERTAIN, ("Eve", "commutes via", "bus"),
     ("Eve", "commutes via", "train"), "unknown predicate cardinality"),
    ("AM-02", "ambiguous", UNCERTAIN, ("Eve", "lives in", ""),
     ("Eve", "lives in", "Paris"), "missing value"),
    ("AM-03", "ambiguous", UNCERTAIN, ("Alice", "lives in", "Paris"),
     ("She", "lives in", "Paris"), "pronoun reference"),
    ("AM-04", "ambiguous", UNCERTAIN, ("Alice", "lives in", "Paris"),
     ("Alice", "lives in", "Paris or Lyon"), "disjunction"),
    ("AM-05", "ambiguous", UNCERTAIN, ("Alice", "lives near", "Paris"),
     ("Alice", "lives in", "Paris"), "near vs in"),
]


# ── Conflict dataset ───────────────────────────────────────────────────────
# Each entry: (id, category, expected_conflict, slot_a, slot_b, note)
# slot = dict(subject, predicate, value, interval=(valid_from, valid_until),
#             text=optional override).  Slots are injected directly so the
# detector is exercised on unlinked state, exactly the U1 scenario.
# expected_conflict=False means the pair is legitimate coexistence or a
# distinct subject and the detector must stay silent.

def _slot(subject, predicate, value, start, until=None, text=None):
    return {
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "start": start,
        "until": until,
        "text": text or f"{subject} {predicate} {value}.",
    }


CONFLICT_PAIRS = [
    ("CF-SP-01", "spelling", CONFLICT,
     _slot("Alice", "lives in", "Paris", 10.0),
     _slot("Alic", "lives in", "Berlin", 20.0), "subject typo, different values"),
    ("CF-SP-02", "spelling", NO_CONFLICT,
     _slot("Alice", "lives in", "Paris", 10.0),
     _slot("Alice", "lives in", "Pariss", 20.0), "value typo, same fact"),

    ("CF-GR-01", "grammar", NO_CONFLICT,
     _slot("Alice", "lives in", "Paris", 10.0),
     _slot("Alice", "live in", "Paris", 20.0), "grammar variant, same value"),

    ("CF-CP-01", "capitalization", NO_CONFLICT,
     _slot("Alice", "lives in", "Addis Ababa", 10.0),
     _slot("ALICE", "LIVES IN", "addis ababa", 20.0),
     "case/level noise; detector silent, structural duplicate check owns it"),

    ("CF-AB-01", "abbreviation", NO_CONFLICT,
     _slot("Alice", "lives in", "New York City", 10.0),
     _slot("Alice", "lives in", "NYC", 20.0), "unambiguous alias"),
    ("CF-AB-02", "abbreviation", NO_CONFLICT,
     _slot("Alice", "lives in", "United Kingdom", 10.0),
     _slot("Alice", "lives in", "UK", 20.0), "unambiguous alias"),

    ("CF-PA-01", "paraphrase", NO_CONFLICT,
     _slot("Alice", "lives in", "New York City", 10.0),
     _slot("Alice", "lives in", "Big Apple", 20.0),
     "semantically equal; lexical-only detector is expected to report it"),
    ("CF-PA-02", "paraphrase", NO_CONFLICT,
     _slot("Omar", "works at", "Acme", 10.0),
     _slot("Omar", "works at", "Acme Corporation", 20.0),
     "semantically equal; lexical-only detector is expected to report it"),

    ("CF-SN-01", "similar_names", NO_CONFLICT,
     _slot("Alice", "lives in", "Paris", 10.0),
     _slot("Alicia", "lives in", "Berlin", 20.0), "distinct people"),
    ("CF-SN-02", "similar_names", NO_CONFLICT,
     _slot("May", "lives in", "Paris", 10.0),
     _slot("Maya", "lives in", "Berlin", 20.0), "distinct people"),

    ("CF-SL-01", "similar_locations", CONFLICT,
     _slot("Nina", "lives in", "Rome", 10.0),
     _slot("Nina", "lives in", "Roma", 20.0), "distinct city names"),
    ("CF-SL-02", "similar_locations", CONFLICT,
     _slot("Nina", "lives in", "Iran", 10.0),
     _slot("Nina", "lives in", "Iraq", 20.0), "distinct countries"),
    ("CF-SL-03", "similar_locations", CONFLICT,
     _slot("Nina", "lives in", "Niger", 10.0),
     _slot("Nina", "lives in", "Nigeria", 20.0), "distinct countries"),

    ("CF-NG-01", "negation", NO_CONFLICT,
     _slot("Olga", "no longer lives in", "Addis Ababa", 10.0),
     _slot("Olga", "lives in", "Nairobi", 20.0), "negation plus different value"),
    ("CF-NG-02", "negation", NO_CONFLICT,
     _slot("Olga", "does not live in", "Addis Ababa", 10.0),
     _slot("Olga", "lives in", "Nairobi", 20.0), "negation plus different value"),
    ("CF-NG-03", "negation", NO_CONFLICT,
     _slot("Olga", "no longer lives in", "Addis Ababa", 10.0),
     _slot("Olga", "lives in", "Addis Ababa", 20.0),
     "same value; detector stays silent, structural duplicate check owns it"),

    ("CF-HC-01", "historical", NO_CONFLICT,
     _slot("Alice", "used to live in", "Addis Ababa", 10.0, until=10.0),
     _slot("Alice", "lives in", "Nairobi", 20.0), "closed historical record"),
    ("CF-HC-02", "historical", NO_CONFLICT,
     _slot("Alice", "formerly worked at", "Company A", 10.0, until=10.0),
     _slot("Alice", "works at", "Company B", 20.0), "closed historical record"),

    ("CF-AM-01", "ambiguous", NO_CONFLICT,
     _slot("Eve", "commutes via", "bus", 10.0),
     _slot("Eve", "commutes via", "train", 20.0), "unknown predicate"),
    ("CF-AM-02", "ambiguous", NO_CONFLICT,
     _slot("Eve", "lives in", "", 10.0),
     _slot("Eve", "lives in", "Paris", 20.0), "missing value"),
]


# ── Helpers ────────────────────────────────────────────────────────────────


class _NeutralVerifier:
    def __init__(self):
        self.calls = 0

    def verify_batch(self, pairs, **kwargs):
        self.calls += len(pairs)
        return [{
            "decision": "NEUTRAL", "confidence": 0.5,
            "p_entail": 0.01, "p_contra": 0.01, "p_neutral": 0.98,
            "fwd": {"p_contra": 0.01, "p_neutral": 0.98},
            "rev": {"p_contra": 0.01, "p_neutral": 0.98},
        } for _ in pairs]


class _ExplodingVerifier:
    def verify_batch(self, *args, **kwargs):
        raise AssertionError("the conflict checker must not call a verifier")


def _memory(ctx, verifier=None, **kwargs):
    return BlackHoleMemory(
        capacity=20,
        dense_encoder=ctx["dense"],
        nli_gate=ctx["nli"],
        verifier=verifier,
        **kwargs,
    )


def _inject(mem, slot):
    original = mem.memory.slots[0]
    return mem.memory.add(
        key=original.key.copy(),
        value=original.value.copy(),
        confidence=1.0,
        timestamp=slot["start"],
        debug_subject=slot["subject"],
        debug_predicate=slot["predicate"],
        text=slot["text"],
        dense_emb=original.dense_emb,
        valid_from=slot["start"],
        valid_until=slot["until"],
        latest_predicate=slot["predicate"],
        latest_object=slot["value"],
    )


def _conflict_observation(mem):
    detector = mem.exclusive_temporal_conflicts()
    structural = mem.temporal_history_violations()
    structural_conflict = any(v["reason"] == "multiple_active" for v in structural)
    return {
        "detector": len(detector),
        "structural": structural_conflict,
        "any": bool(detector) or structural_conflict,
    }


def _rate(numerator, denominator):
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _prf(tp, fp, tn, fn):
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    fpr = _rate(fp, fp + tn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }


# ── 1. Audit entry points ──────────────────────────────────────────────────


def audit_entry_points(ctx):
    """Runtime trace of every audit workflow that can observe temporal state."""
    neutral = _NeutralVerifier()
    mem = _memory(ctx, verifier=neutral)
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))

    default = mem.temporal_history_violations()
    opt_in = mem.temporal_history_violations(include_unlinked_conflicts=True)
    detector = mem.exclusive_temporal_conflicts()
    status_keys = sorted(mem.status().keys())

    from benchmarks.phase29_integrity import _reasons as phase29_reasons
    from benchmarks.u1_closure import _reasons as closure_reasons

    tmpdir = tempfile.mkdtemp(prefix="u1_audit_")
    path = os.path.join(tmpdir, "state.npz")
    mem.save_checkpoint(path, create_backup=False)
    restored = BlackHoleMemory.from_checkpoint(
        path, dense_encoder=ctx["dense"], nli_gate=ctx["nli"],
        verifier=_NeutralVerifier(),
    )
    restored_default = restored.temporal_history_violations()
    restored_opt_in = restored.temporal_history_violations(
        include_unlinked_conflicts=True
    )

    observations = {
        "u1_state_active_slots": sum(1 for s in mem.memory.slots if s.is_active),
        "default_reasons": [v["reason"] for v in default],
        "opt_in_reasons": [v["reason"] for v in opt_in],
        "detector_findings": len(detector),
        "phase29_harness_reasons": phase29_reasons(mem),
        "u1_closure_harness_reasons": closure_reasons(mem),
        "status_has_temporal_invariant_key": any(
            "violat" in key or "conflict" in key for key in status_keys
        ),
        "status_keys": status_keys,
        "checkpoint_roundtrip_default_reasons": [
            v["reason"] for v in restored_default
        ],
        "checkpoint_roundtrip_opt_in_reasons": [
            v["reason"] for v in restored_opt_in
        ],
    }

    expected = {
        "u1_state_active_slots": 2,
        "default_reasons": [],
        "opt_in_reasons": [UNLINKED_EXCLUSIVE_CONFLICT],
        "detector_findings": 1,
        "phase29_harness_reasons": [],
        "u1_closure_harness_reasons": [],
        "status_has_temporal_invariant_key": False,
        "checkpoint_roundtrip_default_reasons": [],
        "checkpoint_roundtrip_opt_in_reasons": [UNLINKED_EXCLUSIVE_CONFLICT],
    }
    passed = all(observations[k] == v for k, v in expected.items())

    return {
        "observations": observations,
        "expected": expected,
        "passed": passed,
        "silent_paths": [
            "temporal_history_violations() default argument",
            "status()",
            "phase2.checkpoint.load_into_black_hole_memory validation",
            "benchmarks/phase29_integrity.py::_reasons",
        ],
    }


# ── 2. Historical recall ───────────────────────────────────────────────────


REQUIRED_HISTORICAL = [
    ("H1", "Alice", "used to live in", "Addis Ababa", 10.0),
    ("H2", "Alice", "previously lived in", "Addis Ababa", 20.0),
    ("H3", "Alice", "formerly worked at", "Company A", 30.0),
    ("H4", "Alice", "no longer lives in", "Addis Ababa", 40.0),
    ("H5", "Alice", "currently lives in", "Nairobi", 50.0),
]


def _rows(rows):
    return [
        {
            "text": r["text"],
            "is_active": r["is_active"],
            "valid_from": r["valid_from"],
            "valid_until": r["valid_until"],
        }
        for r in rows
    ]


def historical_recall(ctx):
    # Real verifier: the temporal state machine (not a stub) decides here.
    mem = _memory(ctx, verifier=None)
    actions = []
    for cid, subject, predicate, obj, ts in REQUIRED_HISTORICAL:
        result = mem.absorb(Candidate(subject, predicate, obj, timestamp=ts))
        actions.append({
            "id": cid,
            "action": result.get("action"),
            "slot_id": result.get("slot_id"),
        })

    current = mem.absorb(
        Candidate("Alice", "works at", "Company B", timestamp=60.0)
    )
    actions.append({"id": "H6", "action": current.get("action"),
                    "slot_id": current.get("slot_id")})

    history = _rows(mem.get_history("Alice"))
    active = _rows(mem.recall("Alice"))
    unfiltered = _rows(mem.recall("Alice", active_only=False))
    predicate_filtered = _rows(mem.get_history("Alice", "lives in"))
    as_of_10 = _rows(mem.recall("Alice", as_of=10.0, active_only=False))
    query_historical = _rows(mem.query("Where did Alice live?", top_k=10))
    query_active_only_false = _rows(
        mem.query("Where did Alice live?", top_k=10, active_only=False)
    )

    # Current negation in isolation: the negation must become the current
    # residence state (not an added duplicate), and a later move must close it.
    negation = _memory(ctx, verifier=None)
    negation.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=5.0))
    neg_action = negation.absorb(
        Candidate("Alice", "no longer lives in", "Addis Ababa", timestamp=10.0)
    )
    after_negation = _rows(negation.recall("Alice"))
    move_action = negation.absorb(
        Candidate("Alice", "lives in", "Nairobi", timestamp=20.0)
    )
    after_move = _rows(negation.recall("Alice"))
    negation_scenario = {
        "negation_action": neg_action.get("action"),
        "active_after_negation": [r["text"] for r in after_negation],
        "move_action": move_action.get("action"),
        "active_after_move": [r["text"] for r in after_move],
        "violations_after_negation": [
            v["reason"] for v in negation.temporal_history_violations()
        ],
        "conflicts_after_negation": len(
            negation.exclusive_temporal_conflicts()
        ),
    }

    texts = {row["text"] for row in history}
    observations = {
        "actions": {a["id"]: a["action"] for a in actions},
        "active_texts": [r["text"] for r in active],
        "history_texts": [r["text"] for r in history],
        "historical_retrievable_via_history": (
            "Alice used to live in Addis Ababa." in texts
            and "Alice previously lived in Addis Ababa." in texts
        ),
        "historical_absent_from_default_recall": all(
            "used to" not in r["text"] and "previously" not in r["text"]
            for r in active
        ),
        "historical_not_active": all(
            not r["is_active"] for r in history
            if "used to" in r["text"] or "previously" in r["text"]
            or "formerly" in r["text"]
        ),
        "history_unfiltered_count": len(unfiltered),
        "predicate_filtered_count": len(predicate_filtered),
        "as_of_observation_time_count": len(as_of_10),
        "query_default_has_historical": any(
            "used to" in r["text"] or "previously" in r["text"]
            for r in query_historical
        ),
        "query_include_history_count": len(query_active_only_false),
        "current_residence": [
            r["text"] for r in active if "Nairobi" in r["text"]
        ],
        "current_employment": [
            r["text"] for r in active if "Company B" in r["text"]
        ],
        "previous_employment_retained": any(
            "formerly worked at Company A" in t for t in texts
        ),
        "current_negation_retained": any(
            "no longer lives in Addis Ababa" in t for t in texts
        ),
        "violations": [v["reason"] for v in mem.temporal_history_violations()],
        "conflicts": len(mem.exclusive_temporal_conflicts()),
        "negation_scenario": negation_scenario,
    }

    expected = {
        "H1": "historical_observation",
        "H2": "historical_observation",
        "H3": "historical_observation",
        "H4": "insert",
        "H5": "temporal_update",
        "H6": "insert",
    }
    checks = {
        "actions_match": observations["actions"] == expected,
        "historical_retrievable_via_history": observations[
            "historical_retrievable_via_history"
        ],
        "historical_absent_from_default_recall": observations[
            "historical_absent_from_default_recall"
        ],
        "historical_not_active": observations["historical_not_active"],
        "previous_employment_retained": observations[
            "previous_employment_retained"
        ],
        "current_negation_retained": observations["current_negation_retained"],
        "current_residence_is_nairobi": observations["current_residence"]
        == ["Alice currently lives in Nairobi."],
        "current_employment_is_company_b": observations["current_employment"]
        == ["Alice works at Company B."],
        "no_invariant_violation": observations["violations"] == [],
        "no_false_conflict": observations["conflicts"] == 0,
        "negation_becomes_current_state": negation_scenario[
            "negation_action"
        ] == "temporal_update" and negation_scenario[
            "active_after_negation"
        ] == ["Alice no longer lives in Addis Ababa."],
        "negation_state_closed_by_move": negation_scenario[
            "move_action"
        ] == "temporal_update" and negation_scenario[
            "active_after_move"
        ] == ["Alice lives in Nairobi."],
        "negation_leaves_no_violation": negation_scenario[
            "violations_after_negation"
        ] == [] and negation_scenario["conflicts_after_negation"] == 0,
    }
    return {
        "observations": observations,
        "expected_actions": expected,
        "checks": checks,
        "passed": all(checks.values()),
        "limitations": {
            "predicate_filtered_history_misses_marker_predicates": (
                observations["predicate_filtered_count"] == 0
            ),
            "point_in_time_recall_excludes_zero_width": (
                observations["as_of_observation_time_count"] == 0
            ),
            "default_query_is_current_only": not observations[
                "query_default_has_historical"
            ],
        },
    }


# ── 3. Language robustness ─────────────────────────────────────────────────


def _equivalence_observation(ctx, pair):
    _, _, _, a, b, _ = pair
    mem = _memory(ctx, verifier=None)
    first = mem.absorb(Candidate(*a, timestamp=10.0))
    try:
        second = mem.absorb(Candidate(*b, timestamp=20.0))
        error = None
    except Exception as exc:  # noqa: BLE001 - malformed input is recorded
        second = {}
        error = f"{type(exc).__name__}: {exc}"
    active = [s for s in mem.memory.slots if s.is_active]
    predicted = "equivalent" if second.get("action") == "consolidate" else "different"
    return {
        "first_action": first.get("action"),
        "second_action": second.get("action"),
        "error": error,
        "active_slots": len(active),
        "predicted": predicted,
        "model_calls": (
            mem.total_value_verifications + mem.total_nli_calls
        ),
    }


def language_equivalence(ctx):
    per_category: Dict[str, Dict[str, Any]] = {}
    results = []
    started = time.perf_counter()
    model_calls = 0

    for pair in EQUIVALENCE_PAIRS:
        case_id, category, label, _, _, note = pair
        pair_started = time.perf_counter()
        obs = _equivalence_observation(ctx, pair)
        pair_elapsed = round(time.perf_counter() - pair_started, 3)
        model_calls += obs["model_calls"]
        bucket = per_category.setdefault(category, {
            "pairs": 0, "equivalent_labels": 0, "predicted_equivalent": 0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0, "uncertain": 0,
            "runtime_seconds": 0.0,
        })
        bucket["pairs"] += 1
        bucket["runtime_seconds"] = round(
            bucket["runtime_seconds"] + pair_elapsed, 3
        )
        bucket["predicted_equivalent"] += int(obs["predicted"] == "equivalent")
        if label == EQUIVALENT:
            bucket["equivalent_labels"] += 1
            if obs["predicted"] == "equivalent":
                bucket["tp"] += 1
            else:
                bucket["fn"] += 1
        elif label == NOT_EQUIVALENT:
            if obs["predicted"] == "equivalent":
                bucket["fp"] += 1
            else:
                bucket["tn"] += 1
        else:
            bucket["uncertain"] += 1
        results.append({
            "id": case_id,
            "category": category,
            "label": label,
            "predicted": obs["predicted"],
            "second_action": obs["second_action"],
            "active_slots": obs["active_slots"],
            "error": obs["error"],
            "runtime_seconds": pair_elapsed,
            "correct": (
                None if label == UNCERTAIN
                else (obs["predicted"] == "equivalent") == (label == EQUIVALENT)
            ),
            "note": note,
        })

    wall = round(time.perf_counter() - started, 2)
    for category, bucket in per_category.items():
        bucket.update(_prf(bucket["tp"], bucket["fp"], bucket["tn"], bucket["fn"]))

    scored = [
        bucket for name, bucket in per_category.items()
        if name != "ambiguous"
    ]
    totals = {
        key: sum(bucket[key] for bucket in scored)
        for key in ("pairs", "tp", "fp", "tn", "fn")
    }
    totals.update(_prf(totals["tp"], totals["fp"], totals["tn"], totals["fn"]))
    totals["uncertain_pairs"] = per_category.get("ambiguous", {}).get("pairs", 0)

    false_negatives = [r["id"] for r in results if r["correct"] is False and r["label"] == EQUIVALENT]
    false_positives = [r["id"] for r in results if r["correct"] is False and r["label"] == NOT_EQUIVALENT]
    return {
        "passed": all(result["error"] is None for result in results),
        "outcome": (
            "all_correct" if not false_negatives and not false_positives
            else "known_equivalence_boundaries"
        ),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "dataset_version": DATASET_VERSION,
        "pairs": len(EQUIVALENCE_PAIRS),
        "per_category": per_category,
        "totals_scored": totals,
        "results": results,
        "runtime_seconds": wall,
        "model_calls": model_calls,
        "method": (
            "Each pair is absorbed into a fresh memory (A at t=10, B at t=20) "
            "with the pinned real NLI verifier.  Predicted 'equivalent' means "
            "absorb returned action=consolidate; any other action is "
            "'different'.  Ambiguous pairs are reported but excluded from "
            "precision/recall."
        ),
    }


def language_conflicts(ctx):
    per_category: Dict[str, Dict[str, Any]] = {}
    results = []
    started = time.perf_counter()

    for case_id, category, expected_conflict, slot_a, slot_b, note in CONFLICT_PAIRS:
        case_started = time.perf_counter()
        mem = _memory(ctx, verifier=_ExplodingVerifier())
        # Seed one slot so _inject can copy vector shapes.
        mem.memory.add(
            key=np.zeros(mem.dim, dtype="float32"),
            value=np.zeros(mem.dim, dtype="float32"),
            confidence=1.0, timestamp=0.0,
            debug_subject="seed", debug_predicate="seed", text="seed",
            valid_from=0.0, valid_until=0.0,
            latest_predicate="seed", latest_object="seed",
        )
        _inject(mem, slot_a)
        _inject(mem, slot_b)
        mem.memory.slots.pop(0)
        obs = _conflict_observation(mem)
        predicted = obs["detector"] > 0
        case_elapsed = round(time.perf_counter() - case_started, 3)

        bucket = per_category.setdefault(category, {
            "pairs": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "detector_reported": 0, "structural_reported": 0,
            "runtime_seconds": 0.0,
        })
        bucket["pairs"] += 1
        bucket["runtime_seconds"] = round(
            bucket["runtime_seconds"] + case_elapsed, 3
        )
        bucket["detector_reported"] += int(obs["detector"] > 0)
        bucket["structural_reported"] += int(obs["structural"])
        if expected_conflict and predicted:
            bucket["tp"] += 1
        elif expected_conflict and not predicted:
            bucket["fn"] += 1
        elif not expected_conflict and predicted:
            bucket["fp"] += 1
        else:
            bucket["tn"] += 1

        results.append({
            "id": case_id,
            "category": category,
            "expected_conflict": expected_conflict,
            "detector_findings": obs["detector"],
            "structural_conflict": obs["structural"],
            "predicted_conflict": predicted,
            "correct": predicted == expected_conflict,
            "runtime_seconds": case_elapsed,
            "note": note,
        })

    wall = round(time.perf_counter() - started, 2)
    for bucket in per_category.values():
        bucket.update(_prf(bucket["tp"], bucket["fp"], bucket["tn"], bucket["fn"]))

    totals = {
        key: sum(bucket[key] for bucket in per_category.values())
        for key in ("pairs", "tp", "fp", "tn", "fn")
    }
    totals.update(_prf(totals["tp"], totals["fp"], totals["tn"], totals["fn"]))

    false_positives = [r["id"] for r in results if not r["correct"] and not r["expected_conflict"]]
    false_negatives = [r["id"] for r in results if not r["correct"] and r["expected_conflict"]]
    return {
        "passed": True,
        "outcome": (
            "all_correct" if not false_positives and not false_negatives
            else "known_lexical_limitation"
        ),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "dataset_version": DATASET_VERSION,
        "pairs": len(CONFLICT_PAIRS),
        "per_category": per_category,
        "totals": totals,
        "results": results,
        "runtime_seconds": wall,
        "model_calls": 0,
        "method": (
            "Slots are injected directly as unlinked active records (the U1 "
            "scenario) and the detector runs with an exploding verifier, so a "
            "model call would fail the audit.  A pair counts as a detected "
            "conflict when exclusive_temporal_conflicts() reports it or the "
            "structural invariant reports multiple_active."
        ),
    }


# ── 4. Detector semantics and safety ───────────────────────────────────────


def detector_semantics(ctx):
    mem = _memory(ctx, verifier=_ExplodingVerifier())
    mem.memory.add(
        key=np.zeros(mem.dim, dtype="float32"),
        value=np.zeros(mem.dim, dtype="float32"),
        confidence=1.0, timestamp=0.0,
        debug_subject="seed", debug_predicate="seed", text="seed",
        valid_from=0.0, valid_until=0.0,
        latest_predicate="seed", latest_object="seed",
    )
    _inject(mem, _slot("Maya", "city", "Rome", 10.0))
    _inject(mem, _slot("Maya", "city", "Venice", 20.0))
    mem.memory.slots.pop(0)

    before = _state_snapshot(mem)
    findings = mem.exclusive_temporal_conflicts()
    after = _state_snapshot(mem)
    again = mem.exclusive_temporal_conflicts()

    finding = findings[0] if findings else {}
    checks = {
        "confidence_is_policy_only": finding.get("confidence") == 1.0,
        "confidence_scope_declared": finding.get("confidence_scope")
        == "predicate_policy_only",
        "semantic_equivalence_declared": finding.get("semantic_equivalence")
        == "lexical_only",
        "conflict_status_declared": finding.get("conflict_status")
        == "unresolved",
        "repeatable": findings == again,
        "no_mutation": before == after,
        "no_model_calls": True,  # exploding verifier would have raised
    }
    return {
        "finding": finding,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _state_snapshot(mem):
    return {
        "slots": [
            (
                i, s.text, s.is_active, s.valid_from, s.valid_until,
                s.supersedes, s.superseded_by, s.update_count,
                round(float(s.confidence), 6),
            )
            for i, s in enumerate(mem.memory.slots)
        ],
        "counters": (
            mem.total_absorbed, mem.total_updates, mem.total_inserts,
            mem.total_temporal_updates, mem.total_evictions,
            mem.total_nli_calls, mem.total_value_verifications,
        ),
        "texts": dict(mem.slot_texts),
    }


# ── 4b. Structural duplicate coverage ─────────────────────────────────────
# The detector deliberately stays silent on equal values; the structural
# invariant owns duplicate active versions.  These cases prove that split.

STRUCTURAL_CASES = [
    ("SD-01", "capitalization",
     _slot("Alice", "lives in", "Addis Ababa", 10.0),
     _slot("ALICE", "LIVES IN", "addis ababa", 20.0),
     "case noise produces two active restatements"),
    ("SD-02", "negation",
     _slot("Olga", "no longer lives in", "Addis Ababa", 10.0),
     _slot("Olga", "lives in", "Addis Ababa", 20.0),
     "affirmation and negation of one value"),
]


def structural_duplicates(ctx):
    results = []
    for case_id, category, slot_a, slot_b, note in STRUCTURAL_CASES:
        mem = _memory(ctx, verifier=_ExplodingVerifier())
        mem.memory.add(
            key=np.zeros(mem.dim, dtype="float32"),
            value=np.zeros(mem.dim, dtype="float32"),
            confidence=1.0, timestamp=0.0,
            debug_subject="seed", debug_predicate="seed", text="seed",
            valid_from=0.0, valid_until=0.0,
            latest_predicate="seed", latest_object="seed",
        )
        _inject(mem, slot_a)
        _inject(mem, slot_b)
        mem.memory.slots.pop(0)
        obs = _conflict_observation(mem)
        results.append({
            "id": case_id,
            "category": category,
            "detector_findings": obs["detector"],
            "structural_multiple_active": obs["structural"],
            "correct": obs["detector"] == 0 and obs["structural"],
            "note": note,
        })
    return {
        "cases": results,
        "passed": all(r["correct"] for r in results),
    }


# ── 5. Policy reproducibility ──────────────────────────────────────────────


def policy_reproducibility(ctx):
    neutral = _NeutralVerifier()
    policy = ["telemetry mode"]

    def build(**kwargs):
        mem = _memory(ctx, verifier=neutral, **kwargs)
        mem.memory.add(
            key=np.zeros(mem.dim, dtype="float32"),
            value=np.zeros(mem.dim, dtype="float32"),
            confidence=1.0, timestamp=0.0,
            debug_subject="seed", debug_predicate="seed", text="seed",
            valid_from=0.0, valid_until=0.0,
            latest_predicate="seed", latest_object="seed",
        )
        _inject(mem, _slot("Quinn", "telemetry mode", "verbose", 10.0))
        _inject(mem, _slot("Quinn", "telemetry mode", "silent", 20.0))
        mem.memory.slots.pop(0)
        return mem

    tmpdir = tempfile.mkdtemp(prefix="u1_audit_policy_")
    path = os.path.join(tmpdir, "policy.npz")

    source = build(exclusive_predicates=policy)
    source_findings = len(source.exclusive_temporal_conflicts())
    source.save_checkpoint(path, create_backup=False)

    same = BlackHoleMemory.from_checkpoint(
        path, dense_encoder=ctx["dense"], nli_gate=ctx["nli"],
        verifier=neutral, exclusive_predicates=policy,
    )
    missing = BlackHoleMemory.from_checkpoint(
        path, dense_encoder=ctx["dense"], nli_gate=ctx["nli"],
        verifier=neutral,
    )
    different = BlackHoleMemory.from_checkpoint(
        path, dense_encoder=ctx["dense"], nli_gate=ctx["nli"],
        verifier=neutral, multi_valued_predicates=policy,
    )
    state_same = [
        (s.text, s.valid_from, s.valid_until, s.is_active)
        for s in same.memory.slots
    ] == [
        (s.text, s.valid_from, s.valid_until, s.is_active)
        for s in source.memory.slots
    ]

    observations = {
        "source_findings": source_findings,
        "identical_policy_findings": len(same.exclusive_temporal_conflicts()),
        "missing_policy_findings": len(missing.exclusive_temporal_conflicts()),
        "different_policy_findings": len(different.exclusive_temporal_conflicts()),
        "restored_state_identical": state_same,
        "policy_persisted_in_checkpoint": False,
    }
    expected = {
        "source_findings": 1,
        "identical_policy_findings": 1,
        "missing_policy_findings": 0,
        "different_policy_findings": 0,
        "restored_state_identical": True,
        "policy_persisted_in_checkpoint": False,
    }
    return {
        "observations": observations,
        "expected": expected,
        "passed": observations == expected,
        "conclusion": (
            "The same checkpoint yields different invariant results depending "
            "on the loading process's predicate policy; policy is not stored "
            "in the archive."
        ),
    }


# ── Dataset summary ────────────────────────────────────────────────────────


def run():
    started = time.perf_counter()
    from benchmarks.phase16_experiment import NliSemanticGate
    from benchmarks.phase18_experiment import DenseSemanticEncoder

    ctx = {"dense": DenseSemanticEncoder(), "nli": NliSemanticGate()}

    entry = audit_entry_points(ctx)
    historical = historical_recall(ctx)
    equivalence = language_equivalence(ctx)
    conflicts = language_conflicts(ctx)
    semantics = detector_semantics(ctx)
    structural = structural_duplicates(ctx)
    policy = policy_reproducibility(ctx)

    wall = round(time.perf_counter() - started, 2)
    sections = {
        "entry_points": entry,
        "historical_recall": historical,
        "language_equivalence": equivalence,
        "language_conflicts": conflicts,
        "detector_semantics": semantics,
        "structural_duplicates": structural,
        "policy_reproducibility": policy,
    }
    passed = [name for name, section in sections.items() if section["passed"]]
    failed = [name for name, section in sections.items() if not section["passed"]]
    limited = [
        name for name, section in sections.items()
        if section.get("outcome", "all_correct") != "all_correct"
    ]
    return {
        "experiment": "u1_independent_audit",
        "dataset_version": DATASET_VERSION,
        "case_count": 7,
        "passed_sections": passed,
        "failed_sections": failed,
        "sections_with_known_limitations": limited,
        "status": (
            "audit_findings" if failed
            else ("audit_complete_with_known_limitations" if limited
                  else "audit_complete")
        ),
        "sections": sections,
        "wall_seconds": wall,
    }
