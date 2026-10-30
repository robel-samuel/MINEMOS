"""
benchmarks/phase29_integrity.py

Phase 29 evaluation: temporal invariant coverage and verifier telemetry.

The case matrix is defined in this module and documented in
reports/PHASE_29_PLAN.md.  Twenty cases come from the plan; L7, L8 and H4 were
added before execution and NC04 is implied by the plan without a case ID (see
the report's provenance table).  Stub verifiers pin the state machine; one
real-NLI case keeps the pinned model's value-boundary behavior visible.
Results are written by runners/run_phase29.py.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate


VERDICT_EXPECTED = "expected_reproduced"
VERDICT_FAILURE = "unexpected_failure"
VERDICT_GAP_CONFIRMED = "known_limitation_confirmed"
VERDICT_GAP_RESOLVED = "known_limitation_resolved"

STATUS_ALL_EXPECTED = "all_expected_reproduced"
STATUS_FAILED = "unexpected_failures"
STATUS_GAP_OPEN = "known_limitations_open"
STATUS_GAP_CHANGED = "known_limitation_resolved"

VERDICTS = (
    VERDICT_EXPECTED,
    VERDICT_FAILURE,
    VERDICT_GAP_CONFIRMED,
    VERDICT_GAP_RESOLVED,
)


def classify_result(case: Dict[str, Any]) -> str:
    """Four-way outcome; a known gap is never conflated with a clean pass."""
    if case["known_gap"]:
        return VERDICT_GAP_CONFIRMED if case["passed"] else VERDICT_GAP_RESOLVED
    return VERDICT_EXPECTED if case["passed"] else VERDICT_FAILURE


def summarise(results) -> Dict[str, Any]:
    verdicts = {r["id"]: classify_result(r) for r in results}
    counts = {name: 0 for name in VERDICTS}
    for verdict in verdicts.values():
        counts[verdict] += 1
    if counts[VERDICT_FAILURE]:
        status = STATUS_FAILED
    elif counts[VERDICT_GAP_CONFIRMED]:
        status = STATUS_GAP_OPEN
    elif counts[VERDICT_GAP_RESOLVED]:
        status = STATUS_GAP_CHANGED
    else:
        status = STATUS_ALL_EXPECTED
    return {"verdicts": verdicts, "counts": counts, "status": status}


class AlwaysContradiction:
    def verify_batch(self, pairs, **kwargs):
        return [
            {
                "decision": "CONTRADICTION",
                "confidence": 0.99,
                "p_entail": 0.01,
                "p_contra": 0.99,
                "p_neutral": 0.01,
                "fwd": {"p_contra": 0.99, "p_neutral": 0.01},
                "rev": {"p_contra": 0.99, "p_neutral": 0.01},
            }
            for _ in pairs
        ]


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


def _memory(ctx, verifier):
    return BlackHoleMemory(
        capacity=10,
        dense_encoder=ctx["dense"],
        nli_gate=ctx["nli"],
        verifier=verifier,
    )


def _active(mem):
    return [s for s in mem.memory.slots if s.is_active]


def _reasons(mem):
    return [v["reason"] for v in mem.temporal_history_violations()]


def _intervals(mem, subject, predicate):
    return [
        (row["valid_from"], row["valid_until"])
        for row in mem.get_history(subject, predicate)
    ]


def _case(case_id, cls, description, expected, observed, known_gap=False,
          match="exact", provenance="plan_registered"):
    if match == "subset":
        passed = True
        for key, value in expected.items():
            got = observed.get(key)
            if isinstance(value, list):
                if not set(value).issubset(set(got or [])):
                    passed = False
            elif got != value:
                passed = False
    else:
        passed = expected == observed
    return {
        "id": case_id,
        "class": cls,
        "description": description,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "known_gap": known_gap,
        "provenance": provenance,
    }


# ── Legitimate coexistence ──────────────────────────────────────────────────


def case_l1(ctx):
    mem = _memory(ctx, AlwaysContradiction())
    mem.absorb(Candidate("Maya", "member of", "red team", timestamp=10.0))
    mem.absorb(Candidate("Maya", "member of", "blue team", timestamp=20.0))
    obs = {"active": len(_active(mem)), "violations": _reasons(mem)}
    return _case("L1", "legitimate", "two team memberships",
                 {"active": 2, "violations": []}, obs)


def case_l2(ctx):
    mem = _memory(ctx, AlwaysContradiction())
    mem.absorb(Candidate("Owen", "manages", "Project Atlas", timestamp=10.0))
    mem.absorb(Candidate("Owen", "manages", "Project Borealis", timestamp=20.0))
    obs = {"active": len(_active(mem)), "violations": _reasons(mem)}
    return _case("L2", "legitimate", "two managed projects",
                 {"active": 2, "violations": []}, obs)


def case_l3(ctx):
    mem = _memory(ctx, AlwaysContradiction())
    mem.absorb(Candidate("Rina", "received", "best paper award 2021", timestamp=10.0))
    mem.absorb(Candidate("Rina", "received", "best paper award 2023", timestamp=20.0))
    obs = {"active": len(_active(mem)), "violations": _reasons(mem)}
    return _case("L3", "legitimate", "two cumulative awards",
                 {"active": 2, "violations": []}, obs)


def case_l4(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Project X", "status", "on hold", timestamp=10.0))
    mem.absorb(Candidate("Project X", "status", "awaiting approval", timestamp=20.0))
    obs = {"active": len(_active(mem)), "violations": _reasons(mem)}
    return _case("L4", "legitimate", "conservative same-attribute retention",
                 {"active": 2, "violations": []}, obs)


def case_l5(ctx):
    mem = _memory(ctx, AlwaysContradiction())
    mem.absorb(Candidate("Noah", "city", "Rome", timestamp=100.0))
    mem.absorb(Candidate("Noah", "employer", "Acme", timestamp=100.0))
    obs = {"active": len(_active(mem)), "violations": _reasons(mem)}
    return _case("L5", "legitimate", "independent predicates",
                 {"active": 2, "violations": []}, obs)


def case_l6(ctx):
    mem = _memory(ctx, AlwaysContradiction())
    mem.absorb(Candidate("Maya", "role", "analyst", timestamp=10.0))
    mem.absorb(Candidate("Maya", "role", "manager", timestamp=20.0))
    obs = {"active": len(_active(mem)), "violations": _reasons(mem)}
    return _case("L6", "legitimate", "exclusive role chain",
                 {"active": 1, "violations": []}, obs)


def case_l7(ctx):
    mem = _memory(ctx, RuleVerifier([("", "serves as", "SAME")]))
    mem.absorb(Candidate("Leo", "works as", "senior data engineer", timestamp=10.0))
    mem.absorb(Candidate("Leo", "serves as", "senior data engineer", timestamp=20.0))
    obs = {
        "active": len(_active(mem)),
        "violations": _reasons(mem),
        "intervals": _intervals(mem, "Leo", "works as"),
    }
    return _case("L7", "legitimate", "paraphrase consolidation",
                 {"active": 1, "violations": [], "intervals": [(10.0, None)]}, obs,
                 provenance="added_before_execution")


def case_l8(ctx):
    mem = _memory(ctx, RuleVerifier([("", "Maya role analyst", "SAME")]))
    mem.absorb(Candidate("Maya", "role", "analyst", timestamp=10.0))
    mem.absorb(Candidate("Maya", "role", "analyst", timestamp=20.0))
    obs = {"active": len(_active(mem)), "violations": _reasons(mem)}
    return _case("L8", "legitimate", "identical value reassertion",
                 {"active": 1, "violations": []}, obs,
                 provenance="added_before_execution")


# ── Structural corruption ──────────────────────────────────────────────────


def _chain(ctx):
    mem = _memory(ctx, AlwaysContradiction())
    mem.absorb(Candidate("Noah", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Noah", "city", "Venice", timestamp=20.0))
    return mem


def case_c1(ctx):
    mem = _chain(ctx)
    mem.memory.slots[0].valid_until = 25.0
    obs = {"violations": _reasons(mem)}
    return _case("C1", "corruption", "linked overlap",
                 {"violations": ["overlap"]}, obs, match="subset")


def case_c2(ctx):
    mem = _chain(ctx)
    mem.memory.slots[0].valid_until = None
    obs = {"violations": sorted(set(_reasons(mem)))}
    return _case("C2", "corruption", "linked two-active chain",
                 {"violations": ["multiple_active", "overlap"]}, obs,
                 match="subset")


def case_c3(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Maya", "role", "analyst", timestamp=10.0))
    mem.absorb(Candidate("Maya", "role", "analyst", timestamp=20.0))
    obs = {"active": len(_active(mem)), "violations": sorted(set(_reasons(mem)))}
    return _case("C3", "corruption", "duplicate active exclusive value",
                 {"active": 2, "violations": ["multiple_active"]}, obs,
                 match="subset")


def case_c4(ctx):
    mem = _chain(ctx)
    mem.memory.slots[0].valid_until = 5.0
    obs = {"violations": _reasons(mem)}
    return _case("C4", "corruption", "inverted validity interval",
                 {"violations": ["invalid_interval"]}, obs, match="subset")


def case_c5(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Lisbon", timestamp=10.0))
    mem.absorb(Candidate("May", "city", "Porto", timestamp=20.0))
    a, b = mem.memory.slots
    a.supersedes = id(b)
    b.superseded_by = id(a)
    obs = {"violations": _reasons(mem)}
    return _case("C5", "corruption", "cross-subject link",
                 {"violations": ["inconsistent_link"]}, obs, match="subset")


def case_c6(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Lisbon", timestamp=10.0))
    mem.absorb(Candidate("Maya", "employer", "Acme", timestamp=20.0))
    a, b = mem.memory.slots
    a.supersedes = id(b)
    b.superseded_by = id(a)
    obs = {"violations": _reasons(mem)}
    return _case("C6", "corruption", "cross-attribute link",
                 {"violations": ["inconsistent_link"]}, obs, match="subset")


# ── Historical SAME recurrence ─────────────────────────────────────────────


def case_h1(ctx):
    mem = _memory(ctx, RuleVerifier([
        ("Rome", "Venice", "CONTRADICTION"),
        ("Rome", "Rome", "SAME"),
        ("Venice", "Rome", "NEUTRAL"),
    ]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    action = mem.absorb(Candidate("Maya", "city", "Rome", timestamp=30.0))["action"]
    obs = {
        "action": action,
        "active": len(_active(mem)),
        "intervals": _intervals(mem, "Maya", "city"),
        "violations": _reasons(mem),
    }
    return _case("H1", "historical", "closed recurrence, active head contradicted",
                 {"action": "temporal_update", "active": 1,
                  "intervals": [(10.0, 20.0), (20.0, 30.0), (30.0, None)],
                  "violations": []}, obs)


def case_h2(ctx):
    mem = _memory(ctx, RuleVerifier([
        ("Rome", "Venice", "CONTRADICTION"),
        ("Rome", "Rome", "SAME"),
    ]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    action = mem.absorb(Candidate("Maya", "city", "Rome", timestamp=30.0))["action"]
    obs = {
        "action": action,
        "active": len(_active(mem)),
        "intervals": _intervals(mem, "Maya", "city"),
        "violations": _reasons(mem),
    }
    return _case("H2", "historical", "closed recurrence, active head neutral",
                 {"action": "temporal_update", "active": 1,
                  "intervals": [(10.0, 20.0), (20.0, 30.0), (30.0, None)],
                  "violations": []}, obs)


def case_h3(ctx):
    mem = _memory(ctx, RuleVerifier([("red team", "red team", "SAME")]))
    mem.absorb(Candidate("Maya", "member of", "red team", timestamp=10.0))
    mem.absorb(Candidate("Maya", "member of", "blue team", timestamp=20.0))
    mem.memory.slots[0].valid_until = 15.0
    action = mem.absorb(
        Candidate("Maya", "member of", "red team", timestamp=30.0)
    )["action"]
    obs = {"action": action, "active": len(_active(mem)), "violations": _reasons(mem)}
    return _case("H3", "historical", "multi-valued closed recurrence",
                 {"action": "insert", "active": 2, "violations": []}, obs)


def case_h4(ctx):
    mem = _memory(ctx, RuleVerifier([
        ("Rome", "Venice", "CONTRADICTION"),
        ("Rome", "Rome", "SAME"),
    ]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    mem.memory.slots[1].latest_object = "Rome"
    action = mem.absorb(Candidate("Maya", "city", "Rome", timestamp=30.0))["action"]
    obs = {
        "action": action,
        "active": len(_active(mem)),
        "intervals": _intervals(mem, "Maya", "city"),
        "violations": _reasons(mem),
    }
    return _case("H4", "historical", "closed recurrence, active value unchanged",
                 {"action": "consolidate", "active": 1,
                  "intervals": [(10.0, 20.0), (20.0, None)], "violations": []}, obs,
                 provenance="added_before_execution")


# ── Verifier boundary ──────────────────────────────────────────────────────


def case_v1(ctx):
    mem = _memory(ctx, RuleVerifier([
        ("Alice city New York.", "Alice city Boston.", "SAME"),
    ]))
    mem.absorb(Candidate("Alice", "city", "New York", timestamp=10.0))
    action = mem.absorb(Candidate("Alice", "city", "Boston", timestamp=20.0))["action"]
    obs = {"action": action, "active": len(_active(mem)),
           "value_checks": mem.status()["total_value_verifications"]}
    return _case("V1", "verifier", "changed value, whole-statement SAME",
                 {"action": "temporal_update", "active": 1, "value_checks": 1}, obs)


def case_v2(ctx):
    mem = _memory(ctx, RuleVerifier([
        ("leads cybersecurity", "directs cybersecurity", "SAME"),
    ]))
    mem.absorb(Candidate("David", "role", "leads cybersecurity operations",
                         timestamp=10.0))
    action = mem.absorb(Candidate("David", "role", "directs cybersecurity operations",
                                  timestamp=15.0))["action"]
    obs = {"action": action, "active": len(_active(mem))}
    return _case("V2", "verifier", "paraphrased value, SAME",
                 {"action": "consolidate", "active": 1}, obs)


def case_v3(ctx):
    mem = _memory(ctx, RuleVerifier([
        ("Zephyr telemetry mode diagnostic.", "Zephyr telemetry mode operational.",
         "SAME"),
    ]))
    mem.absorb(Candidate("Zephyr", "telemetry mode", "diagnostic", timestamp=10.0))
    action = mem.absorb(
        Candidate("Zephyr", "telemetry mode", "operational", timestamp=20.0)
    )["action"]
    obs = {"action": action, "active": len(_active(mem))}
    return _case("V3", "verifier", "changed value, non-exclusive attribute",
                 {"action": "insert", "active": 2}, obs)


def case_nc04(ctx):
    mem = BlackHoleMemory(
        capacity=10,
        dense_encoder=ctx["dense"],
        nli_gate=ctx["nli"],
    )
    mem.absorb(Candidate("Zephyr", "telemetry mode", "diagnostic", timestamp=10.0))
    action = mem.absorb(
        Candidate("Zephyr", "telemetry mode", "operational", timestamp=20.0)
    )["action"]
    obs = {
        "action": action,
        "active": len(_active(mem)),
        "value_checks": mem.status()["total_value_verifications"],
    }
    return _case("NC04", "real_nli", "pinned NLI SAME over changed value",
                 {"action": "insert", "active": 2, "value_checks": 1}, obs,
                 provenance="plan_implied")


# ── Known gap and telemetry ────────────────────────────────────────────────


def case_u1(ctx):
    mem = _memory(ctx, RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    obs = {"active": len(_active(mem)), "violations": _reasons(mem)}
    return _case("U1", "known_gap", "unlinked active exclusive conflict",
                 {"active": 2, "violations": []}, obs, known_gap=True)


def case_t1(ctx):
    mem = _memory(ctx, RuleVerifier([
        ("Alice city New York.", "Alice city Boston.", "SAME"),
    ]))
    mem.absorb(Candidate("Alice", "city", "New York", timestamp=10.0))
    mem.absorb(Candidate("Alice", "city", "Boston", timestamp=20.0))
    status = mem.status()
    obs = {
        "value_verifications": status["total_value_verifications"],
        "nli_calls": status["total_nli_calls"],
    }
    return _case("T1", "telemetry", "value-only verifier calls counted",
                 {"value_verifications": 1, "nli_calls": 1}, obs)


CASES = [
    case_l1, case_l2, case_l3, case_l4, case_l5, case_l6, case_l7, case_l8,
    case_c1, case_c2, case_c3, case_c4, case_c5, case_c6,
    case_h1, case_h2, case_h3, case_h4,
    case_v1, case_v2, case_v3, case_nc04,
    case_u1, case_t1,
]


def run() -> Dict[str, Any]:
    from benchmarks.phase18_experiment import DenseSemanticEncoder
    from benchmarks.phase16_experiment import NliSemanticGate

    started = time.time()
    ctx = {"dense": DenseSemanticEncoder(), "nli": NliSemanticGate()}

    results = [case(ctx) for case in CASES]
    legitimate = [r for r in results if r["class"] == "legitimate"]
    corruption = [r for r in results if r["class"] == "corruption"]
    historical = [r for r in results if r["class"] == "historical"]
    known_gaps = [r for r in results if r["known_gap"]]

    false_positives = sum(
        1 for r in legitimate if r["observed"]["violations"]
    )
    detected = sum(1 for r in corruption if r["passed"])
    historical_ok = sum(1 for r in historical if r["passed"])
    failures = [r["id"] for r in results if not r["passed"] and not r["known_gap"]]
    known_gaps_confirmed = sum(1 for r in results if r["known_gap"] and r["passed"])
    passed = sum(1 for r in results if r["passed"] and not r["known_gap"])
    summary = summarise(results)
    provenance_counts: Dict[str, int] = {}
    for result in results:
        key = result["provenance"]
        provenance_counts[key] = provenance_counts.get(key, 0) + 1
    annotated = [
        {**result, "verdict": summary["verdicts"][result["id"]]}
        for result in results
    ]

    return {
        "phase": 29,
        "timestamp": time.time(),
        "baseline_commit": "3afc2e480ba09d23c696d123fcc5c59fb5eb39a8",
        "status": summary["status"],
        "verdict_counts": summary["counts"],
        "provenance_counts": provenance_counts,
        "case_count": len(results),
        "passed": passed,
        "known_gaps_confirmed": known_gaps_confirmed,
        "failed": len(failures),
        "failed_ids": failures,
        "known_gap_ids": [r["id"] for r in known_gaps],
        "metrics": {
            "legitimate_cases": len(legitimate),
            "legitimate_false_positive_count": false_positives,
            "legitimate_false_positive_rate": round(
                false_positives / len(legitimate), 4
            ) if legitimate else 0.0,
            "corruption_cases": len(corruption),
            "corruption_detected": detected,
            "corruption_detection_rate": round(
                detected / len(corruption), 4
            ) if corruption else 0.0,
            "historical_cases": len(historical),
            "historical_overlap_free": historical_ok,
            "historical_pass_rate": round(
                historical_ok / len(historical), 4
            ) if historical else 0.0,
        },
        "wall_seconds": round(time.time() - started, 2),
        "results": annotated,
    }
