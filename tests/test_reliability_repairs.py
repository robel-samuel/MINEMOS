"""Regression tests for the three reliability defects found by the
independent validation of 2026-10-29 (commit 5964677):

  R1  canonical-subject recall/history lost the active fact after a
      case/whitespace variant of the same entity.
  R2  temporal_history_violations() flagged legitimate multi-valued or
      conservatively retained facts as exclusive overlaps.
  R3  a SAME (consolidation) decision with an identical predicate and a
      changed value erased the prior temporal state.

The tests use deterministic stub verifiers to pin the matcher/state policy;
one real-NLI test keeps the pinned model's NC04 behavior covered.
"""

from pathlib import Path

import pytest

from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate


class _AlwaysContradiction:
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


class _MapVerifier:
    """Maps a candidate-text substring to a decision; default NEUTRAL."""

    def __init__(self, rules):
        self._rules = list(rules)

    def verify_batch(self, pairs, **kwargs):
        results = []
        for _slot_text, cand_text in pairs:
            decision = "NEUTRAL"
            for marker, mapped in self._rules:
                if marker in cand_text:
                    decision = mapped
                    break
            results.append({
                "decision": decision,
                "confidence": 0.99,
                "p_entail": 0.99 if decision == "SAME" else 0.01,
                "p_contra": 0.99 if decision == "CONTRADICTION" else 0.01,
                "p_neutral": 0.01 if decision != "NEUTRAL" else 0.98,
                "fwd": {
                    "p_contra": 0.99 if decision == "CONTRADICTION" else 0.01,
                    "p_neutral": 0.01 if decision == "CONTRADICTION" else 0.98,
                },
                "rev": {
                    "p_contra": 0.99 if decision == "CONTRADICTION" else 0.01,
                    "p_neutral": 0.01 if decision == "CONTRADICTION" else 0.98,
                },
            })
        return results


@pytest.fixture(scope="module")
def models():
    from benchmarks.phase18_experiment import DenseSemanticEncoder
    from benchmarks.phase16_experiment import NliSemanticGate

    return DenseSemanticEncoder(), NliSemanticGate()


def _memory(models, verifier=None, capacity=10):
    dense, nli = models
    return BlackHoleMemory(
        capacity=capacity,
        dense_encoder=dense,
        nli_gate=nli,
        verifier=verifier,
    )


def _active_texts(mem):
    return sorted(s.text for s in mem.memory.slots if s.is_active)


def _intervals(mem, subject, predicate):
    return [
        (row["valid_from"], row["valid_until"])
        for row in mem.get_history(subject, predicate)
    ]


# ── R1: canonical-subject identity in every public lookup ──────────────────


def test_recall_and_history_match_subject_case_and_trailing_space(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Maya", "city", "Lisbon", timestamp=10.0))
    result = mem.absorb(Candidate("MAYA ", "city", "Porto", timestamp=20.0))

    assert result["action"] == "temporal_update"
    assert _intervals(mem, "Maya", "city") == [(10.0, 20.0), (20.0, None)]

    canonical = mem.recall("Maya", "city")
    assert len(canonical) == 1
    assert "Porto" in canonical[0]["text"]

    # The variant spelling must resolve to the same canonical facts.
    variant = mem.recall("MAYA ", "city")
    assert len(variant) == 1
    assert "Porto" in variant[0]["text"]
    assert len(mem.get_history("maya", "city")) == 2


def test_repeated_updates_with_equivalent_subject_spellings_keep_history(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Nadia", "city", "Oslo", timestamp=10.0))
    mem.absorb(Candidate("nadia ", "city", "Bergen", timestamp=20.0))
    result = mem.absorb(Candidate(" NADIA", "city", "Rome", timestamp=30.0))

    assert result["action"] == "temporal_update"
    assert _intervals(mem, "Nadia", "city") == [
        (10.0, 20.0),
        (20.0, 30.0),
        (30.0, None),
    ]
    assert [r["text"] for r in mem.recall("Nadia", "city")] == [" NADIA city Rome."]


def test_free_text_and_structured_ingestion_agree_on_subject_identity(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb("Maya city Lisbon.", timestamp=10.0)
    mem.absorb(Candidate("MAYA ", "city", "Porto", timestamp=20.0))

    assert _intervals(mem, "Maya", "city") == [(10.0, 20.0), (20.0, None)]
    active = mem.recall("maya", "city")
    assert len(active) == 1
    assert "Porto" in active[0]["text"]


def test_normalization_does_not_merge_distinct_subjects(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Maya", "city", "Lisbon", timestamp=10.0))
    mem.absorb(Candidate("May", "city", "Porto", timestamp=20.0))

    assert len(_active_texts(mem)) == 2
    assert len(mem.recall("Maya", "city")) == 1
    assert len(mem.recall("May", "city")) == 1


# ── R2: invariant respects multi-valued / non-equivalent semantics ─────────


def test_team_memberships_are_not_flagged_as_violations(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Maya", "member of", "red team", timestamp=10.0))
    mem.absorb(Candidate("Maya", "member of", "blue team", timestamp=20.0))

    assert len(_active_texts(mem)) == 2
    assert mem.temporal_history_violations() == []


def test_multiple_managed_projects_are_not_flagged_as_violations(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Owen", "manages", "Project Atlas", timestamp=10.0))
    mem.absorb(Candidate("Owen", "manages", "Project Borealis", timestamp=20.0))

    assert len(_active_texts(mem)) == 2
    assert mem.temporal_history_violations() == []


def test_cumulative_facts_are_not_flagged_as_violations(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Rina", "received", "best paper award 2021", timestamp=10.0))
    mem.absorb(Candidate("Rina", "received", "best paper award 2023", timestamp=20.0))

    assert len(_active_texts(mem)) == 2
    assert mem.temporal_history_violations() == []


def test_conservatively_retained_same_predicate_statements_are_not_violations(models):
    mem = _memory(models, verifier=_MapVerifier([]))
    mem.absorb(Candidate("Project X", "status", "on hold", timestamp=10.0))
    mem.absorb(Candidate("Project X", "status", "awaiting approval", timestamp=20.0))

    assert len(_active_texts(mem)) == 2
    assert mem.temporal_history_violations() == []


def test_independent_predicates_may_overlap_without_violations(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Noah", "city", "Rome", timestamp=100.0))
    mem.absorb(Candidate("Noah", "employer", "Acme", timestamp=100.0))

    assert len(_active_texts(mem)) == 2
    assert mem.temporal_history_violations() == []


def test_exclusive_role_chain_remains_violation_free(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Maya", "role", "analyst", timestamp=10.0))
    mem.absorb(Candidate("Maya", "role", "manager", timestamp=20.0))

    assert len(_active_texts(mem)) == 1
    assert mem.temporal_history_violations() == []


def test_genuine_overlap_inside_a_version_chain_is_detected(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Noah", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Noah", "city", "Venice", timestamp=20.0))

    older = next(s for s in mem.memory.slots if s.valid_from == 10.0)
    older.valid_until = 25.0  # corrupt: overlaps the successor at t=20

    reasons = [v["reason"] for v in mem.temporal_history_violations()]
    assert "overlap" in reasons


def test_multiple_active_versions_in_one_chain_are_detected(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Noah", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Noah", "city", "Venice", timestamp=20.0))

    older = next(s for s in mem.memory.slots if s.valid_from == 10.0)
    older.valid_until = None  # corrupt: two active versions of one chain

    reasons = [v["reason"] for v in mem.temporal_history_violations()]
    assert "multiple_active" in reasons


def test_duplicate_active_versions_of_one_exclusive_value_are_detected(models):
    mem = _memory(models, verifier=_MapVerifier([]))  # NEUTRAL: forces duplicate insertion
    mem.absorb(Candidate("Maya", "role", "analyst", timestamp=10.0))
    mem.absorb(Candidate("Maya", "role", "analyst", timestamp=20.0))

    assert len(_active_texts(mem)) == 2
    reasons = [v["reason"] for v in mem.temporal_history_violations()]
    assert "multiple_active" in reasons


# ── R3: SAME must not erase a changed value ────────────────────────────────


def test_same_decision_with_identical_predicate_and_changed_value_preserves_history(models):
    """NC04 policy: an NLI SAME over a different value must not consolidate."""
    mem = _memory(models, verifier=_MapVerifier([("Zephyr telemetry mode operational", "SAME")]))
    mem.absorb(Candidate("Zephyr", "telemetry mode", "diagnostic", timestamp=10.0))
    result = mem.absorb(
        Candidate("Zephyr", "telemetry mode", "operational", timestamp=20.0)
    )

    assert result["action"] == "insert"
    assert len(_active_texts(mem)) == 2
    assert any("diagnostic" in t for t in _active_texts(mem))
    assert any("operational" in t for t in _active_texts(mem))
    assert _intervals(mem, "Zephyr", "telemetry mode") == [
        (10.0, None),
        (20.0, None),
    ]
    assert mem.temporal_history_violations() == []


def test_same_decision_on_exclusive_attribute_becomes_temporal_update(models):
    mem = _memory(models, verifier=_MapVerifier([("Alice city Boston", "SAME")]))
    mem.absorb(Candidate("Alice", "city", "New York", timestamp=10.0))
    result = mem.absorb(Candidate("Alice", "city", "Boston", timestamp=20.0))

    assert result["action"] == "temporal_update"
    assert len(_active_texts(mem)) == 1
    assert "Boston" in _active_texts(mem)[0]
    assert _intervals(mem, "Alice", "city") == [(10.0, 20.0), (20.0, None)]
    assert mem.temporal_history_violations() == []


def test_true_paraphrase_still_consolidates(models):
    mem = _memory(models, verifier=_MapVerifier([("serves as", "SAME")]))
    mem.absorb(Candidate("Leo", "works as", "senior data engineer", timestamp=10.0))
    result = mem.absorb(
        Candidate("Leo", "serves as", "senior data engineer", timestamp=20.0)
    )

    assert result["action"] == "consolidate"
    assert result["update_count"] == 2
    assert len(_active_texts(mem)) == 1
    assert _intervals(mem, "Leo", "works as") == [(10.0, None)]


def test_cross_wording_paraphrase_with_different_object_still_consolidates(models):
    """Existing paraphrase policy: different predicate wording is trusted."""
    mem = _memory(models, verifier=_MapVerifier([("wrote an important", "SAME")]))
    mem.absorb(
        Candidate("Marcus", "authored an influential", "biography monograph", timestamp=100.0)
    )
    result = mem.absorb(
        Candidate("Marcus", "wrote an important", "biographical treatise", timestamp=150.0)
    )

    assert result["action"] == "consolidate"
    assert len(_active_texts(mem)) == 1
    assert _intervals(mem, "Marcus", "authored an influential") == [(100.0, None)]


def test_same_decision_with_paraphrased_value_still_consolidates(models):
    """A paraphrased value of one attribute is not a value change.

    'leads' -> 'directs ... operations' keeps the same role; the value-only
    check must recognize the paraphrase so the R3 gate does not turn ordinary
    rewording into a temporal update.  Real NLI decides both statement and
    value equivalence here.
    """
    mem = _memory(models)
    mem.absorb(Candidate("David", "role", "leads cybersecurity operations", timestamp=10.0))
    result = mem.absorb(
        Candidate("David", "role", "directs cybersecurity operations", timestamp=15.0)
    )

    assert result["action"] == "consolidate"
    assert result["update_count"] == 2
    assert len(_active_texts(mem)) == 1
    assert _intervals(mem, "David", "role") == [(10.0, None)]


def test_same_decision_on_multivalued_predicate_does_not_supersede(models):
    mem = _memory(models, verifier=_MapVerifier([("Maya member of blue team", "SAME")]))
    mem.absorb(Candidate("Maya", "member of", "red team", timestamp=10.0))
    result = mem.absorb(Candidate("Maya", "member of", "blue team", timestamp=20.0))

    assert result["action"] == "insert"
    assert len(_active_texts(mem)) == 2
    assert mem.temporal_history_violations() == []


def test_pinned_nli_nc04_does_not_consolidate_away_prior_state(models):
    dense, nli = models
    mem = _memory(models)
    mem.absorb(Candidate("Zephyr", "telemetry mode", "diagnostic", timestamp=10.0))
    result = mem.absorb(
        Candidate("Zephyr", "telemetry mode", "operational", timestamp=20.0)
    )

    decision = nli.predict_pair(
        "Zephyr telemetry mode diagnostic.",
        "Zephyr telemetry mode operational.",
    )["decision"]
    assert decision == "SAME", (
        "pinned model no longer labels this pair SAME; "
        "the deterministic SAME test still covers the policy"
    )
    assert result["action"] != "consolidate"
    assert len(_active_texts(mem)) == 2
    assert any("diagnostic" in t for t in _active_texts(mem))
    assert _intervals(mem, "Zephyr", "telemetry mode") == [
        (10.0, None),
        (20.0, None),
    ]
