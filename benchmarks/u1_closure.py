"""
benchmarks/u1_closure.py

U1 closure evaluation: unlinked exclusive temporal conflict detection and
noisy-language robustness.

This is a *new* experiment with its own result file.  It does not modify or
replace the Phase 29 benchmark or its canonical results (commit 05fc8e4).

Cases are grouped as:

  U1-*  conflict detection on the predicate-cardinality policy
  NL-*  equivalent restatements under noisy surface forms, and the
        historical/negation boundaries

Stub verifiers pin the state machine; two real-NLI cases keep the pinned
models' grammar/paraphrase behavior visible.  Results are written by
runners/run_u1_closure.py.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from phase2.black_hole_memory import (
    PREDICATE_POLICY_EXCLUSIVE,
    PREDICATE_POLICY_MULTI_VALUED,
    UNLINKED_EXCLUSIVE_CONFLICT,
    BlackHoleMemory,
)
from phase2.structured_candidate import Candidate


BASELINE_COMMIT = "05fc8e4d9494795e02dacaed602daeda8e882cae"

STATUS_ALL_EXPECTED = "all_expected_reproduced"
STATUS_FAILED = "unexpected_failures"
VERDICT_EXPECTED = "expected_reproduced"
VERDICT_FAILURE = "unexpected_failure"


class RuleVerifier:
    """First matching (slot substring, candidate substring) rule wins."""

    def __init__(self, rules, default="NEUTRAL"):
        self._rules = list(rules)
        self._default = default

    def _decide(self, slot_text, cand_text):
        for slot_marker, cand_marker, decision in self._rules:
            if slot_marker in slot_text and cand_marker in cand_text:
                return decision
        return self._default

    def verify_batch(self, pairs, **kwargs):
        results = []
        for slot_text, cand_text in pairs:
            decision = self._decide(slot_text, cand_text)
            contra = decision == "CONTRADICTION"
            same = decision == "SAME"
            results.append({
                "decision": decision,
                "confidence": 0.99,
                "p_entail": 0.99 if same else 0.01,
                "p_contra": 0.99 if contra else 0.01,
                "p_neutral": 0.98 if decision == "NEUTRAL" else 0.01,
                "fwd": {
                    "p_contra": 0.99 if contra else 0.01,
                    "p_neutral": 0.98 if decision == "NEUTRAL" else 0.01,
                },
                "rev": {
                    "p_contra": 0.99 if contra else 0.01,
                    "p_neutral": 0.98 if decision == "NEUTRAL" else 0.01,
                },
            })
        return results


def _memory(ctx, verifier, **kwargs):
    return BlackHoleMemory(
        capacity=10,
        dense_encoder=ctx["dense"],
        nli_gate=ctx["nli"],
        verifier=verifier,
        **kwargs,
    )


def _active(mem):
    return [s for s in mem.memory.slots if s.is_active]


def _reasons(mem):
    return [v["reason"] for v in mem.temporal_history_violations()]


def _conflict_reasons(mem):
    return [f["reason"] for f in mem.exclusive_temporal_conflicts()]


def _case(case_id, cls, description, expected, observed):
    return {
        "id": case_id,
        "class": cls,
        "description": description,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
        "provenance": "u1_closure_plan",
    }


# ── U1: unlinked exclusive conflict detection ──────────────────────────────


def case_u1_01(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))

    findings = mem.exclusive_temporal_conflicts()
    obs = {
        "active": len(_active(mem)),
        "conflicts": len(findings),
        "reason": findings[0]["reason"] if findings else None,
        "policy": findings[0]["predicate_policy"] if findings else None,
        "values": findings[0]["values"] if findings else None,
        "overlap": findings[0]["overlap"] if findings else None,
        "structural_violations": _reasons(mem),
    }
    return _case(
        "U1-01", "detection", "unlinked active exclusive conflict",
        {
            "active": 2,
            "conflicts": 1,
            "reason": UNLINKED_EXCLUSIVE_CONFLICT,
            "policy": PREDICATE_POLICY_EXCLUSIVE,
            "values": ["Rome", "Venice"],
            "overlap": True,
            "structural_violations": [],
        },
        obs,
    )


def case_u1_02(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    obs = {
        "default": _reasons(mem),
        "integrated": [
            v["reason"]
            for v in mem.temporal_history_violations(include_unlinked_conflicts=True)
        ],
    }
    return _case(
        "U1-02", "detection", "opt-in integration with structural invariant",
        {"default": [], "integrated": [UNLINKED_EXCLUSIVE_CONFLICT]},
        obs,
    )


def case_u1_03(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Rina", "received", "best paper award 2021", timestamp=10.0))
    mem.absorb(Candidate("Rina", "received", "best paper award 2023", timestamp=20.0))
    obs = {
        "active": len(_active(mem)),
        "conflicts": _conflict_reasons(mem),
        "policy": mem.classify_predicate("received"),
    }
    return _case(
        "U1-03", "silence", "multi-valued predicate",
        {"active": 2, "conflicts": [], "policy": PREDICATE_POLICY_MULTI_VALUED},
        obs,
    )


def case_u1_04(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Nadia", "commutes via", "bicycle", timestamp=10.0))
    mem.absorb(Candidate("Nadia", "commutes via", "train", timestamp=20.0))
    obs = {
        "active": len(_active(mem)),
        "conflicts": _conflict_reasons(mem),
        "policy": mem.classify_predicate("commutes via"),
    }
    return _case(
        "U1-04", "silence", "unknown predicate",
        {"active": 2, "conflicts": [], "policy": "unknown"},
        obs,
    )


def case_u1_05(ctx):
    mem = _memory(ctx, RuleVerifier([("Rome", "Venice", "CONTRADICTION")]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    obs = {
        "active": len(_active(mem)),
        "conflicts": _conflict_reasons(mem),
        "violations": _reasons(mem),
    }
    return _case(
        "U1-05", "silence", "linked version chain",
        {"active": 1, "conflicts": [], "violations": []},
        obs,
    )


def case_u1_06(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Alice", "lives in", "Paris", timestamp=10.0))
    mem.absorb(Candidate("Alice", "previously lived in", "Rome", timestamp=20.0))
    obs = {
        "active": len(_active(mem)),
        "conflicts": _conflict_reasons(mem),
        "closed": sum(1 for s in mem.memory.slots if not s.is_active),
    }
    return _case(
        "U1-06", "silence", "closed historical version",
        {"active": 1, "conflicts": [], "closed": 1},
        obs,
    )


def case_u1_07(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("May", "city", "Venice", timestamp=20.0))
    obs = {"active": len(_active(mem)), "conflicts": _conflict_reasons(mem)}
    return _case(
        "U1-07", "silence", "distinct short names",
        {"active": 2, "conflicts": []},
        obs,
    )


def case_u1_08(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    mem.absorb(Candidate("Maya", "city", "Oslo", timestamp=30.0))
    obs = {
        "pairs": [f["slot_indices"] for f in mem.exclusive_temporal_conflicts()],
    }
    return _case(
        "U1-08", "detection", "three values produce pairwise findings",
        {"pairs": [[0, 1], [0, 2], [1, 2]]},
        obs,
    )


def case_u1_09(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))

    def snapshot():
        return [
            (s.text, s.valid_from, s.valid_until, s.supersedes,
             s.superseded_by, s.update_count, id(s))
            for s in mem.memory.slots
        ]

    before = snapshot()
    status_before = mem.status()
    first = mem.exclusive_temporal_conflicts()
    second = mem.exclusive_temporal_conflicts()
    obs = {
        "deterministic": first == second,
        "unchanged": snapshot() == before,
        "status_unchanged": mem.status() == status_before,
    }
    return _case(
        "U1-09", "safety", "read-only and repeatable",
        {"deterministic": True, "unchanged": True, "status_unchanged": True},
        obs,
    )


def case_u1_10(ctx):
    mem = _memory(
        ctx, RuleVerifier([]), exclusive_predicates={"commutes via"}
    )
    mem.absorb(Candidate("Nadia", "commutes via", "bicycle", timestamp=10.0))
    mem.absorb(Candidate("Nadia", "commutes via", "train", timestamp=20.0))
    obs = {
        "policy": mem.classify_predicate("commutes via"),
        "conflicts": _conflict_reasons(mem),
    }
    return _case(
        "U1-10", "policy", "explicit exclusive override",
        {"policy": PREDICATE_POLICY_EXCLUSIVE,
         "conflicts": [UNLINKED_EXCLUSIVE_CONFLICT]},
        obs,
    )


def case_u1_11(ctx):
    mem = _memory(
        ctx, RuleVerifier([]), multi_valued_predicates={"status"}
    )
    mem.absorb(Candidate("Project X", "status", "on hold", timestamp=10.0))
    mem.absorb(Candidate("Project X", "status", "awaiting approval", timestamp=20.0))
    obs = {
        "policy": mem.classify_predicate("status"),
        "conflicts": _conflict_reasons(mem),
    }
    return _case(
        "U1-11", "policy", "explicit multi-valued override",
        {"policy": PREDICATE_POLICY_MULTI_VALUED, "conflicts": []},
        obs,
    )


def case_u1_12(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Maya", "role", "analyst", timestamp=10.0))
    # Identical restatements consolidate, so inject the imported/legacy
    # duplicate that the structural invariant must still detect.  This keeps
    # the intent of Phase 29 C3 (whose fixture relied on the old duplicate
    # insertion) in the new experiment.
    original = mem.memory.slots[0]
    mem.memory.add(
        key=original.key.copy(), value=original.value.copy(),
        confidence=original.confidence, timestamp=20.0,
        debug_subject="Maya", debug_predicate="role",
        text="Maya role analyst.", dense_emb=original.dense_emb,
        valid_from=20.0, valid_until=None,
        latest_predicate="role", latest_object="analyst",
    )
    obs = {
        "active": len(_active(mem)),
        "violations": sorted(set(_reasons(mem))),
        "conflicts": _conflict_reasons(mem),
    }
    return _case(
        "U1-12", "corruption", "injected duplicate active value",
        {"active": 2, "violations": ["multiple_active", "overlap"],
         "conflicts": []},
        obs,
    )


# ── NL: noisy-language robustness ──────────────────────────────────────────


def case_nl_01(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = mem.absorb(Candidate("Alic", "lives in", "Addis Ababa", timestamp=20.0))
    obs = {
        "action": result["action"],
        "active": len(_active(mem)),
        "stored": result.get("stored_text"),
        "surface_variant": result.get("surface_variant"),
    }
    return _case(
        "NL-01", "noise", "one-letter subject typo",
        {"action": "consolidate", "active": 1,
         "stored": "Alice lives in Addis Ababa.", "surface_variant": True},
        obs,
    )


def case_nl_02(ctx):
    mem = _memory(ctx, RuleVerifier([], default="CONTRADICTION"))
    mem.absorb(Candidate("Alice", "lives in", "Nairobi", timestamp=10.0))
    result = mem.absorb(Candidate("Alice", "lives in", "Narobi", timestamp=20.0))
    obs = {
        "action": result["action"],
        "active": len(_active(mem)),
        "stored": result.get("stored_text"),
        "temporal_updates": mem.status()["total_temporal_updates"],
    }
    return _case(
        "NL-02", "noise", "one-letter value typo under CONTRADICTION",
        {"action": "consolidate", "active": 1,
         "stored": "Alice lives in Nairobi.", "temporal_updates": 0},
        obs,
    )


def case_nl_03(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Alice", "lives in", "New York City", timestamp=10.0))
    result = mem.absorb(Candidate("Alice", "lives in", "NYC", timestamp=20.0))
    obs = {
        "action": result["action"],
        "active": len(_active(mem)),
        "stored": result.get("stored_text"),
    }
    return _case(
        "NL-03", "abbreviation", "unambiguous abbreviation",
        {"action": "consolidate", "active": 1,
         "stored": "Alice lives in New York City."},
        obs,
    )


def case_nl_04(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = mem.absorb(Candidate("Alice", "lives in", "AA", timestamp=20.0))
    obs = {
        "action": result["action"],
        "active": len(_active(mem)),
        "conflicts": _conflict_reasons(mem),
    }
    return _case(
        "NL-04", "abbreviation", "ambiguous abbreviation is surfaced, not merged",
        {"action": "insert", "active": 2,
         "conflicts": [UNLINKED_EXCLUSIVE_CONFLICT]},
        obs,
    )


def case_nl_05(ctx):
    mem = _memory(ctx, RuleVerifier([], default="CONTRADICTION"))
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = mem.absorb(Candidate(
        "  ＡＬＩＣＥ ", "LIVES IN", "  addis   ababa!!! ", timestamp=20.0,
    ))
    obs = {"action": result["action"], "active": len(_active(mem))}
    return _case(
        "NL-05", "noise", "case, width, whitespace, punctuation",
        {"action": "consolidate", "active": 1},
        obs,
    )


def case_nl_06(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = mem.absorb(
        Candidate("Alice", "used to live in", "Addis Ababa", timestamp=20.0)
    )
    historical = next(s for s in mem.memory.slots if not s.is_active)
    obs = {
        "action": result["action"],
        "active": len(_active(mem)),
        "current_text": _active(mem)[0].text,
        "historical_interval": [historical.valid_from, historical.valid_until],
        "violations": _reasons(mem),
    }
    return _case(
        "NL-06", "temporal", "historical statement recorded closed",
        {"action": "historical_observation", "active": 1,
         "current_text": "Alice lives in Addis Ababa.",
         "historical_interval": [20.0, 20.0], "violations": []},
        obs,
    )


def case_nl_07(ctx):
    mem = _memory(ctx, RuleVerifier([], default="CONTRADICTION"))
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = mem.absorb(
        Candidate("Alice", "no longer lives in", "Addis Ababa", timestamp=20.0)
    )
    obs = {
        "action": result["action"],
        "active": len(_active(mem)),
        "active_text": _active(mem)[0].text,
    }
    return _case(
        "NL-07", "temporal", "negated restatement is not consolidated",
        {"action": "temporal_update", "active": 1,
         "active_text": "Alice no longer lives in Addis Ababa."},
        obs,
    )


def case_nl_08(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    obs = {
        "numeric_equivalent": mem._lexically_equivalent_value(
            "best paper award 2021", "best paper award 2023"
        ),
        "typo_equivalent": mem._lexically_equivalent_value("Nairobi", "Narobi"),
        "substitution_equivalent": mem._lexically_equivalent_value("Iran", "Iraq"),
        "short_name_equivalent": mem._subjects_are_equivalent("May", "Maya"),
    }
    return _case(
        "NL-08", "equivalence", "typo boundary",
        {"numeric_equivalent": False, "typo_equivalent": True,
         "substitution_equivalent": False, "short_name_equivalent": False},
        obs,
    )


def case_nl_09(ctx):
    mem = _memory(ctx, verifier=None)
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    grammar = mem.absorb(
        Candidate("Alice", "is living in", "Addis Ababa", timestamp=20.0)
    )
    paraphrase = mem.absorb(
        Candidate("Alice", "based in", "Addis Ababa", timestamp=30.0)
    )
    obs = {
        "grammar": grammar["action"],
        "paraphrase": paraphrase["action"],
        "active": len(_active(mem)),
    }
    return _case(
        "NL-09", "grammar", "real NLI grammar and paraphrase",
        {"grammar": "consolidate", "paraphrase": "consolidate", "active": 1},
        obs,
    )


def case_nl_10(ctx):
    mem = _memory(ctx, verifier=None)
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = mem.absorb(
        Candidate("Alice", "used to live in", "Addis Ababa", timestamp=20.0)
    )
    obs = {
        "action": result["action"],
        "active": len(_active(mem)),
        "current_text": _active(mem)[0].text,
    }
    return _case(
        "NL-10", "temporal", "real NLI does not swallow a historical marker",
        {"action": "historical_observation", "active": 1,
         "current_text": "Alice lives in Addis Ababa."},
        obs,
    )


CASES = [
    case_u1_01, case_u1_02, case_u1_03, case_u1_04, case_u1_05,
    case_u1_06, case_u1_07, case_u1_08, case_u1_09, case_u1_10,
    case_u1_11, case_u1_12, case_nl_01, case_nl_02, case_nl_03,
    case_nl_04, case_nl_05, case_nl_06, case_nl_07, case_nl_08,
    case_nl_09, case_nl_10,
]


def run() -> Dict[str, Any]:
    from benchmarks.phase18_experiment import DenseSemanticEncoder
    from benchmarks.phase16_experiment import NliSemanticGate

    ctx = {"dense": DenseSemanticEncoder(), "nli": NliSemanticGate()}
    started = time.time()
    results: List[Dict[str, Any]] = [case(ctx) for case in CASES]
    wall = time.time() - started

    failed = [r["id"] for r in results if not r["passed"]]
    status = STATUS_FAILED if failed else STATUS_ALL_EXPECTED
    return {
        "experiment": "u1_closure",
        "baseline_commit": BASELINE_COMMIT,
        "status": status,
        "case_count": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": len(failed),
        "failed_ids": failed,
        "verdicts": {
            r["id"]: VERDICT_EXPECTED if r["passed"] else VERDICT_FAILURE
            for r in results
        },
        "results": results,
        "wall_seconds": round(wall, 2),
    }
