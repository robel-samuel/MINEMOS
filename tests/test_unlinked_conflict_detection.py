"""Adversarial detector and predicate-policy tests.

These tests pin the behaviour of the unlinked-conflict detector:

  * the detector audits unlinked active exclusive pairs only;
  * supersession links, multi-valued/unknown predicates, negations with a
    different value, distinct entities, and closed history stay silent;
  * the detector never mutates state and never calls a model;
  * findings declare the scope of ``confidence``;
  * a bare-string predicate override is one predicate, not characters;
  * predicate policy is not persisted, so restore must use the same policy.

Known limitations not fixed here (substitution typos, lexical-only paraphrase
comparison, zero-width point-in-time recall) are documented rather than frozen
as passing tests.
"""

import numpy as np
import pytest

from minemos_engine.black_hole_memory import (
    PREDICATE_POLICY_EXCLUSIVE,
    UNLINKED_EXCLUSIVE_CONFLICT,
    BlackHoleMemory,
)
from minemos_engine.structured_candidate import Candidate


class _Neutral:
    def verify_batch(self, pairs, **kwargs):
        return [{
            "decision": "NEUTRAL", "confidence": 0.5,
            "p_entail": 0.01, "p_contra": 0.01, "p_neutral": 0.98,
            "fwd": {"p_contra": 0.01, "p_neutral": 0.98},
            "rev": {"p_contra": 0.01, "p_neutral": 0.98},
        } for _ in pairs]


class _Exploding:
    def verify_batch(self, *args, **kwargs):
        raise AssertionError("the conflict checker must not call a verifier")


@pytest.fixture(scope="module")
def models():
    from minemos_engine.intelligence.dense import DenseSemanticEncoder
    from minemos_engine.intelligence.nli import NliSemanticGate

    return DenseSemanticEncoder(), NliSemanticGate()


def _memory(models, verifier=None, **kwargs):
    dense, nli = models
    return BlackHoleMemory(
        capacity=20, dense_encoder=dense, nli_gate=nli,
        verifier=verifier, **kwargs,
    )


def _slot(subject, predicate, value, start, until=None, text=None):
    return {
        "subject": subject, "predicate": predicate, "value": value,
        "start": start, "until": until,
        "text": text or f"{subject} {predicate} {value}.",
    }


def _inject(mem, slot):
    seed = mem.memory.slots[0]
    created = mem.memory.add(
        key=seed.key.copy(), value=seed.value.copy(), confidence=1.0,
        timestamp=slot["start"], debug_subject=slot["subject"],
        debug_predicate=slot["predicate"], text=slot["text"],
        dense_emb=seed.dense_emb, valid_from=slot["start"],
        valid_until=slot["until"], latest_predicate=slot["predicate"],
        latest_object=slot["value"],
    )
    # Out-of-band fixture insertion: notify the engine's retrieval invalidation
    # path explicitly; direct MemoryState edits are not observable otherwise.
    mem._invalidate_retrieval_cache()
    return created


def _pair(models, slot_a, slot_b, verifier=None):
    mem = _memory(models, verifier=verifier or _Exploding())
    mem.memory.add(
        key=np.zeros(mem.dim, dtype="float32"),
        value=np.zeros(mem.dim, dtype="float32"),
        confidence=1.0, timestamp=0.0, debug_subject="seed",
        debug_predicate="seed", text="seed", valid_from=0.0,
        valid_until=0.0, latest_predicate="seed", latest_object="seed",
    )
    _inject(mem, slot_a)
    _inject(mem, slot_b)
    mem.memory.slots.pop(0)
    mem._invalidate_retrieval_cache()
    return mem


def _findings(mem):
    return mem.exclusive_temporal_conflicts()


def _reasons(mem):
    return [v["reason"] for v in mem.temporal_history_violations()]


def _snapshot(mem):
    return {
        "slots": [
            (
                i, s.text, s.is_active, s.valid_from, s.valid_until,
                s.supersedes, s.superseded_by, s.update_count,
                round(float(s.confidence), 6),
            )
            for i, s in enumerate(mem.memory.slots)
        ],
        "texts": dict(mem.slot_texts),
        "counters": (
            mem.total_absorbed, mem.total_updates, mem.total_inserts,
            mem.total_temporal_updates, mem.total_evictions,
            mem.total_nli_calls, mem.total_value_verifications,
        ),
    }


# ── Detection and cardinality ─────────────────────────────────────────────


def test_two_active_values_are_reported(models):
    mem = _pair(
        models,
        _slot("Maya", "lives in", "Rome", 10.0),
        _slot("Maya", "lives in", "Venice", 20.0),
    )
    findings = _findings(mem)
    assert len(findings) == 1
    assert findings[0]["reason"] == UNLINKED_EXCLUSIVE_CONFLICT
    assert findings[0]["values"] == ["Rome", "Venice"]


def test_three_active_values_are_pairwise_and_ordered(models):
    mem = _memory(models, verifier=_Exploding())
    mem.memory.add(
        key=np.zeros(mem.dim, dtype="float32"),
        value=np.zeros(mem.dim, dtype="float32"),
        confidence=1.0, timestamp=0.0, debug_subject="seed",
        debug_predicate="seed", text="seed", valid_from=0.0,
        valid_until=0.0, latest_predicate="seed", latest_object="seed",
    )
    for start, value in ((10.0, "Rome"), (20.0, "Venice"), (30.0, "Oslo")):
        _inject(mem, _slot("Maya", "lives in", value, start))
    mem.memory.slots.pop(0)
    mem._invalidate_retrieval_cache()

    assert [f["slot_indices"] for f in _findings(mem)] == [
        [0, 1], [0, 2], [1, 2],
    ]


def test_predicate_synonyms_in_one_exclusive_group_are_compared(models):
    mem = _pair(
        models,
        _slot("Cara", "lives in", "Rome", 10.0),
        _slot("Cara", "resides in", "Milan", 20.0),
    )
    assert len(_findings(mem)) == 1


def test_different_exclusive_groups_are_not_compared(models):
    mem = _pair(
        models,
        _slot("Cara", "lives in", "Rome", 10.0),
        _slot("Cara", "works at", "Acme", 20.0),
    )
    assert _findings(mem) == []


def test_multi_valued_and_unknown_predicates_stay_silent(models):
    multi = _pair(
        models,
        _slot("Dan", "likes", "tea", 10.0),
        _slot("Dan", "likes", "coffee", 20.0),
    )
    unknown = _pair(
        models,
        _slot("Eve", "commutes via", "bus", 10.0),
        _slot("Eve", "commutes via", "train", 20.0),
    )
    assert _findings(multi) == []
    assert _findings(unknown) == []


# ── Intervals, links, and scope ───────────────────────────────────────────


def test_detector_audits_active_pairs_only(models):
    """A closed unlinked version is outside the U1 scope (documented)."""
    mem = _pair(
        models,
        _slot("Finn", "lives in", "Paris", 10.0, until=20.0),
        _slot("Finn", "lives in", "Berlin", 20.0),
    )
    assert _findings(mem) == []
    assert _reasons(mem) == []


def test_malformed_interval_is_structural_and_not_a_u1_finding(models):
    mem = _pair(
        models,
        _slot("Ivan", "lives in", "Paris", 10.0, until=5.0),
        _slot("Ivan", "lives in", "Berlin", 20.0),
    )
    assert _findings(mem) == []
    assert "invalid_interval" in _reasons(mem)


def test_reciprocal_supersession_link_is_left_to_the_structural_invariant(models):
    mem = _pair(
        models,
        _slot("Jo", "lives in", "Paris", 10.0, until=20.0),
        _slot("Jo", "lives in", "Berlin", 20.0),
    )
    a, b = mem.memory.slots
    a.superseded_by, b.supersedes = id(b), id(a)
    mem._invalidate_retrieval_cache()
    assert _findings(mem) == []
    assert _reasons(mem) == []


def test_one_directional_link_is_reported_as_non_reciprocal(models):
    mem = _pair(
        models,
        _slot("Kai", "lives in", "Paris", 10.0, until=20.0),
        _slot("Kai", "lives in", "Berlin", 15.0),
    )
    a, b = mem.memory.slots
    a.superseded_by = id(b)  # B does not point back
    mem._invalidate_retrieval_cache()
    assert _findings(mem) == []
    assert "non_reciprocal_link" in _reasons(mem)


def test_half_open_boundary_is_disjoint_in_the_structural_invariant(models):
    mem = _pair(
        models,
        _slot("Lee", "lives in", "Paris", 10.0, until=20.0),
        _slot("Lee", "lives in", "Berlin", 20.0, until=30.0),
    )
    a, b = mem.memory.slots
    a.superseded_by, b.supersedes = id(b), id(a)
    mem._invalidate_retrieval_cache()
    assert _reasons(mem) == []


# ── Spelling boundaries ───────────────────────────────────────────────────


def test_distinct_subjects_with_near_identical_spelling_are_not_merged(models):
    for other in ("Alicia", "May", "Maya", "Jon"):
        mem = _pair(
            models,
            _slot("Alice", "lives in", "Paris", 10.0),
            _slot(other, "lives in", "Berlin", 20.0),
        )
        assert _findings(mem) == [], other


def test_subject_typo_still_counts_as_one_subject(models):
    mem = _pair(
        models,
        _slot("Alice", "lives in", "Paris", 10.0),
        _slot("Alic", "lives in", "Berlin", 20.0),
    )
    assert len(_findings(mem)) == 1


def test_value_typo_is_same_fact_but_substitution_is_a_conflict(models):
    typo = _pair(
        models,
        _slot("Nina", "lives in", "Paris", 10.0),
        _slot("Nina", "lives in", "Pariss", 20.0),
    )
    substitution = _pair(
        models,
        _slot("Nina", "lives in", "Rome", 10.0),
        _slot("Nina", "lives in", "Roma", 20.0),
    )
    assert _findings(typo) == []
    assert len(_findings(substitution)) == 1


# ── Negation ──────────────────────────────────────────────────────────────


def test_negation_with_a_different_value_is_not_a_conflict(models):
    mem = _pair(
        models,
        _slot("Olga", "no longer lives in", "Addis Ababa", 10.0),
        _slot("Olga", "lives in", "Nairobi", 20.0),
    )
    assert _findings(mem) == []


def test_negation_and_affirmation_of_one_value_stay_structural(models):
    mem = _pair(
        models,
        _slot("Olga", "no longer lives in", "Addis Ababa", 10.0),
        _slot("Olga", "lives in", "Addis Ababa", 20.0),
    )
    assert _findings(mem) == []
    assert "multiple_active" in _reasons(mem)


# ── Read-only, repeatability, and semantics ───────────────────────────────


def test_findings_declare_policy_scope(models):
    mem = _pair(
        models,
        _slot("Maya", "lives in", "Rome", 10.0),
        _slot("Maya", "lives in", "Venice", 20.0),
    )
    finding = _findings(mem)[0]
    assert finding["predicate_policy"] == PREDICATE_POLICY_EXCLUSIVE
    assert finding["confidence"] == 1.0
    assert finding["confidence_scope"] == "predicate_policy_only"
    assert finding["semantic_equivalence"] == "lexical_only"
    assert finding["conflict_status"] == "unresolved"
    assert finding["detection_source"] == "active_unlinked_slots"


def test_detector_is_repeatable_and_never_mutates_state(models):
    mem = _pair(
        models,
        _slot("Maya", "lives in", "Rome", 10.0),
        _slot("Maya", "lives in", "Venice", 20.0),
    )
    before = _snapshot(mem)
    first = _findings(mem)
    second = _findings(mem)
    assert first == second
    assert _snapshot(mem) == before


def test_detector_never_calls_a_model(models):
    mem = _pair(
        models,
        _slot("Maya", "lives in", "Rome", 10.0),
        _slot("Maya", "lives in", "Venice", 20.0),
    )  # exploding verifier
    before = _snapshot(mem)
    assert _findings(mem)
    assert _snapshot(mem)["counters"] == before["counters"]


# ── Policy configuration and restore ──────────────────────────────────────


def test_bare_string_policy_override_is_one_predicate(models):
    mem = _memory(
        models, verifier=_Neutral(), exclusive_predicates="telemetry mode"
    )
    assert mem._exclusive_predicates == {"telemetry mode"}
    assert mem.classify_predicate("telemetry mode") == PREDICATE_POLICY_EXCLUSIVE


def test_policy_is_process_configuration_not_checkpoint_state(models, tmp_path):
    policy = ["telemetry mode"]
    source = _memory(models, verifier=_Neutral(), exclusive_predicates=policy)
    seed = source.memory.add(
        key=np.zeros(source.dim, dtype="float32"),
        value=np.zeros(source.dim, dtype="float32"),
        confidence=1.0, timestamp=0.0, debug_subject="seed",
        debug_predicate="seed", text="seed", valid_from=0.0,
        valid_until=0.0, latest_predicate="seed", latest_object="seed",
    )
    _inject(source, _slot("Quinn", "telemetry mode", "verbose", 10.0))
    _inject(source, _slot("Quinn", "telemetry mode", "silent", 20.0))
    source.memory.slots.pop(0)
    source._invalidate_retrieval_cache()
    assert len(source.exclusive_temporal_conflicts()) == 1

    path = str(tmp_path / "policy.npz")
    source.save_checkpoint(path, create_backup=False)

    dense, nli = models
    same = BlackHoleMemory.from_checkpoint(
        path, dense_encoder=dense, nli_gate=nli, verifier=_Neutral(),
        exclusive_predicates=policy,
    )
    missing = BlackHoleMemory.from_checkpoint(
        path, dense_encoder=dense, nli_gate=nli, verifier=_Neutral(),
    )
    different = BlackHoleMemory.from_checkpoint(
        path, dense_encoder=dense, nli_gate=nli, verifier=_Neutral(),
        multi_valued_predicates=policy,
    )

    assert len(same.exclusive_temporal_conflicts()) == 1
    assert missing.exclusive_temporal_conflicts() == []
    assert different.exclusive_temporal_conflicts() == []


# ── Historical recall ─────────────────────────────────────────────────────


def test_historical_fact_is_retrievable_and_never_becomes_current(models):
    mem = _memory(models, verifier=_Neutral())
    historical = mem.absorb(
        Candidate("Alice", "used to live in", "Addis Ababa", timestamp=10.0)
    )
    current = mem.absorb(
        Candidate("Alice", "lives in", "Nairobi", timestamp=20.0)
    )

    assert historical["action"] == "historical_observation"
    assert current["action"] == "insert"

    history_texts = [row["text"] for row in mem.get_history("Alice")]
    assert "Alice used to live in Addis Ababa." in history_texts
    assert mem.recall("Alice")[0]["text"] == "Alice lives in Nairobi."
    assert all("used to" not in row["text"] for row in mem.recall("Alice"))
    assert all(
        not row["is_active"]
        for row in mem.get_history("Alice")
        if "used to" in row["text"]
    )
    # Known limitation, pinned deliberately: zero-width observations are not
    # selected by point-in-time recall.
    assert mem.recall("Alice", as_of=10.0, active_only=False) == []
    # Known limitation: predicate-filtered history matches only the current
    # base form, not the marker predicate of the historical record.
    filtered = mem.get_history("Alice", "lives in")
    assert [row["text"] for row in filtered] == ["Alice lives in Nairobi."]


# ── Crafted/mutated interval rejection ────────────────────────────────────


def test_crafted_none_valid_from_fails_closed(models):
    mem = _pair(
        models,
        _slot("Maya", "lives in", "Rome", 10.0),
        _slot("Maya", "lives in", "Venice", 20.0),
    )
    mem.memory.slots[0].valid_from = None
    mem._invalidate_retrieval_cache()
    with pytest.raises(ValueError, match="valid_from"):
        mem.temporal_history_violations()
    with pytest.raises(ValueError, match="valid_from"):
        mem.exclusive_temporal_conflicts()


def test_crafted_nan_valid_until_fails_closed(models):
    mem = _pair(
        models,
        _slot("Maya", "lives in", "Rome", 10.0),
        _slot("Maya", "lives in", "Venice", 20.0),
    )
    mem.memory.slots[1].valid_until = float("nan")
    mem._invalidate_retrieval_cache()
    with pytest.raises(ValueError, match="valid_until"):
        mem.temporal_history_violations()
    with pytest.raises(ValueError, match="valid_until"):
        mem.exclusive_temporal_conflicts()
