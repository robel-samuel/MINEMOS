"""Unlinked exclusive conflict detection and noisy-language handling.

Two active facts of one exclusive attribute can conflict without a
supersession link, and the structural invariant alone does not report it.
This suite pins the detector:

  * ``exclusive_temporal_conflicts()`` reports unlinked, overlapping active
    conflicts of exclusive predicates, without mutating state or calling a
    model;
  * multi-valued and unknown predicates stay silent;
  * equivalent restatements under case/space/unicode noise, unambiguous
    abbreviations, and one-edit spelling mistakes consolidate instead of
    creating duplicates or false state changes;
  * historical and negated statements are never swallowed by a current fact.

The detector is a distinct diagnostic from ``temporal_history_violations()``;
the latter can include its findings on request.
"""

import pytest

from minemos_engine.black_hole_memory import (
    PREDICATE_POLICY_EXCLUSIVE,
    PREDICATE_POLICY_MULTI_VALUED,
    PREDICATE_POLICY_UNKNOWN,
    UNLINKED_EXCLUSIVE_CONFLICT,
    BlackHoleMemory,
)
from minemos_engine.structured_candidate import Candidate


class _RuleVerifier:
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


class _ExplodingVerifier:
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
        capacity=10,
        dense_encoder=dense,
        nli_gate=nli,
        verifier=verifier,
        **kwargs,
    )


def _active(mem):
    return [s for s in mem.memory.slots if s.is_active]


def _conflicts(mem):
    return mem.exclusive_temporal_conflicts()


def _add_duplicate_slot(mem, subject, predicate, value, valid_from):
    """Inject a second active slot without going through absorb."""
    original = mem.memory.slots[0]
    slot = mem.memory.add(
        key=original.key.copy(), value=original.value.copy(),
        confidence=original.confidence, timestamp=valid_from,
        debug_subject=subject, debug_predicate=predicate,
        text=f"{subject} {predicate} {value}.",
        dense_emb=original.dense_emb,
        valid_from=valid_from, valid_until=None,
        latest_predicate=predicate, latest_object=value,
    )
    mem.slot_texts[id(slot)] = slot.text
    mem._invalidate_retrieval_cache()
    return slot


# ── U1 detection ───────────────────────────────────────────────────────────


def test_unlinked_exclusive_conflict_is_detected(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))

    findings = _conflicts(mem)
    assert len(findings) == 1
    finding = findings[0]
    assert finding["reason"] == UNLINKED_EXCLUSIVE_CONFLICT
    assert finding["slot_indices"] == [0, 1]
    assert finding["slot_ids"] == [id(mem.memory.slots[0]), id(mem.memory.slots[1])]
    assert finding["subject"] == "Maya"
    assert finding["predicates"] == ["city", "city"]
    assert finding["values"] == ["Rome", "Venice"]
    assert finding["valid_from"] == [10.0, 20.0]
    assert finding["valid_until"] == [None, None]
    assert finding["overlap"] is True
    assert finding["predicate_policy"] == PREDICATE_POLICY_EXCLUSIVE
    assert finding["detection_source"] == "active_unlinked_slots"
    assert finding["confidence"] == 1.0


def test_structural_invariant_can_include_unlinked_conflicts(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))

    assert mem.temporal_history_violations() == []
    integrated = mem.temporal_history_violations(include_unlinked_conflicts=True)
    assert [v["reason"] for v in integrated] == [UNLINKED_EXCLUSIVE_CONFLICT]


def test_three_values_produce_deterministic_pairwise_findings(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    mem.absorb(Candidate("Maya", "city", "Oslo", timestamp=30.0))

    findings = _conflicts(mem)
    assert [f["slot_indices"] for f in findings] == [[0, 1], [0, 2], [1, 2]]
    assert [f["values"] for f in findings] == [
        ["Rome", "Venice"],
        ["Rome", "Oslo"],
        ["Venice", "Oslo"],
    ]
    assert findings == _conflicts(mem)


def test_detector_is_read_only_and_repeatable(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))

    def snapshot():
        return [
            (
                s.text, s.valid_from, s.valid_until, s.supersedes,
                s.superseded_by, s.update_count, s.confidence, id(s),
            )
            for s in mem.memory.slots
        ]

    before = snapshot()
    status_before = mem.status()
    first = mem.exclusive_temporal_conflicts()
    second = mem.exclusive_temporal_conflicts()

    assert first == second
    assert snapshot() == before
    assert mem.status() == status_before


def test_detector_makes_no_verifier_calls_and_changes_no_counters(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))

    mem.verifier = _ExplodingVerifier()
    status_before = mem.status()
    assert len(mem.exclusive_temporal_conflicts()) == 1
    assert mem.status() == status_before


# ── Policy: what is and is not exclusive ───────────────────────────────────


def test_predicate_policy_is_explicit_not_inferred(models):
    mem = _memory(models, verifier=_RuleVerifier([]))

    assert mem.classify_predicate("city") == PREDICATE_POLICY_EXCLUSIVE
    assert mem.classify_predicate("lives_in") == PREDICATE_POLICY_EXCLUSIVE
    assert mem.classify_predicate("status") == PREDICATE_POLICY_EXCLUSIVE
    assert mem.classify_predicate("received") == PREDICATE_POLICY_MULTI_VALUED
    assert mem.classify_predicate("member of") == PREDICATE_POLICY_UNKNOWN
    assert mem.classify_predicate("manages") == PREDICATE_POLICY_UNKNOWN
    assert mem.classify_predicate("commutes via") == PREDICATE_POLICY_UNKNOWN
    assert mem.classify_predicate("capacity") == PREDICATE_POLICY_UNKNOWN


def test_multi_valued_predicates_are_not_conflicts(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Rina", "received", "best paper award 2021", timestamp=10.0))
    mem.absorb(Candidate("Rina", "received", "best paper award 2023", timestamp=20.0))

    assert len(_active(mem)) == 2
    assert _conflicts(mem) == []


def test_unknown_predicates_are_not_conflicts(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Nadia", "commutes via", "bicycle", timestamp=10.0))
    mem.absorb(Candidate("Nadia", "commutes via", "train", timestamp=20.0))

    assert len(_active(mem)) == 2
    assert _conflicts(mem) == []


def test_substring_marker_matches_do_not_create_exclusivity(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Project X", "capacity", "40 units", timestamp=10.0))
    mem.absorb(Candidate("Project X", "capacity", "80 units", timestamp=20.0))

    assert _conflicts(mem) == []


def test_explicit_exclusive_override_is_honoured(models):
    mem = _memory(
        models, verifier=_RuleVerifier([]),
        exclusive_predicates={"commutes via"},
    )
    mem.absorb(Candidate("Nadia", "commutes via", "bicycle", timestamp=10.0))
    mem.absorb(Candidate("Nadia", "commutes via", "train", timestamp=20.0))

    assert mem.classify_predicate("commutes via") == PREDICATE_POLICY_EXCLUSIVE
    assert [f["reason"] for f in _conflicts(mem)] == [UNLINKED_EXCLUSIVE_CONFLICT]


def test_explicit_multi_valued_override_suppresses_conflicts(models):
    mem = _memory(
        models, verifier=_RuleVerifier([]),
        multi_valued_predicates={"status"},
    )
    mem.absorb(Candidate("Project X", "status", "on hold", timestamp=10.0))
    mem.absorb(Candidate("Project X", "status", "awaiting approval", timestamp=20.0))

    assert mem.classify_predicate("status") == PREDICATE_POLICY_MULTI_VALUED
    assert _conflicts(mem) == []


def test_linked_versions_are_not_unlinked_conflicts(models):
    mem = _memory(models, verifier=_RuleVerifier([
        ("Rome", "Venice", "CONTRADICTION"),
    ]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))

    assert len(_active(mem)) == 1
    assert _conflicts(mem) == []
    assert mem.temporal_history_violations() == []


def test_closed_historical_versions_are_not_active_conflicts(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    mem.memory.slots[0].valid_until = 20.0
    mem._invalidate_retrieval_cache()

    assert _conflicts(mem) == []


def test_distinct_entities_are_not_conflicts(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("May", "city", "Venice", timestamp=20.0))
    mem.absorb(Candidate("Alicia", "city", "Oslo", timestamp=30.0))

    assert len(_active(mem)) == 3
    assert _conflicts(mem) == []


def test_typo_subject_is_treated_as_the_same_entity(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Alice", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Alic", "city", "Venice", timestamp=20.0))

    findings = _conflicts(mem)
    assert len(findings) == 1
    assert findings[0]["subject"] == "Alice"
    assert findings[0]["values"] == ["Rome", "Venice"]


def test_injected_duplicate_with_typo_value_is_not_a_conflict(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Alice", "city", "Nairobi", timestamp=10.0))
    _add_duplicate_slot(mem, "Alice", "city", "Narobi", valid_from=20.0)

    assert len(_active(mem)) == 2
    assert _conflicts(mem) == []


# ── Noisy-language equivalence in absorb ───────────────────────────────────


def test_typo_subject_same_fact_consolidates(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = mem.absorb(Candidate("Alic", "lives in", "Addis Ababa", timestamp=20.0))

    assert result["action"] == "consolidate"
    assert result["surface_variant"] is True
    assert len(_active(mem)) == 1
    assert _active(mem)[0].text == "Alice lives in Addis Ababa."


def test_typo_value_does_not_create_a_false_state_change(models):
    mem = _memory(models, verifier=_RuleVerifier([], default="CONTRADICTION"))
    mem.absorb(Candidate("Alice", "lives in", "Nairobi", timestamp=10.0))
    result = mem.absorb(Candidate("Alice", "lives in", "Narobi", timestamp=20.0))

    assert result["action"] == "consolidate"
    assert result["surface_variant"] is True
    slot = _active(mem)[0]
    assert slot.text == "Alice lives in Nairobi."
    assert slot.valid_from == 10.0
    assert slot.valid_until is None
    assert slot.supersedes is None and slot.superseded_by is None
    assert mem.status()["total_temporal_updates"] == 0


def test_unambiguous_abbreviation_consolidates(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Alice", "lives in", "New York City", timestamp=10.0))
    result = mem.absorb(Candidate("Alice", "lives in", "NYC", timestamp=20.0))

    assert result["action"] == "consolidate"
    assert len(_active(mem)) == 1
    assert _active(mem)[0].text == "Alice lives in New York City."


def test_ambiguous_abbreviation_is_not_silently_merged(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = mem.absorb(Candidate("Alice", "lives in", "AA", timestamp=20.0))

    assert result["action"] != "consolidate"
    assert len(_active(mem)) == 2
    assert [f["reason"] for f in _conflicts(mem)] == [UNLINKED_EXCLUSIVE_CONFLICT]


def test_case_width_whitespace_and_punctuation_noise_consolidate(models):
    mem = _memory(models, verifier=_RuleVerifier([], default="CONTRADICTION"))
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = mem.absorb(Candidate(
        "  ＡＬＩＣＥ ", "LIVES IN", "  addis   ababa!!! ",
        timestamp=20.0,
    ))

    assert result["action"] == "consolidate"
    assert len(_active(mem)) == 1


def test_numeric_values_are_never_typo_equivalent(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Rina", "received", "best paper award 2021", timestamp=10.0))
    mem.absorb(Candidate("Rina", "received", "best paper award 2023", timestamp=20.0))

    assert not mem._lexically_equivalent_value(
        "best paper award 2021", "best paper award 2023"
    )
    assert len(_active(mem)) == 2


def test_one_edit_equivalence_is_narrow(models):
    mem = _memory(models, verifier=_RuleVerifier([]))

    assert mem._lexically_equivalent_value("Nairobi", "Narobi")
    assert mem._lexically_equivalent_value("Addis Ababa", "Adis Ababa")
    # Substitutions are the ambiguous case and stay distinct.
    assert not mem._lexically_equivalent_value("Iran", "Iraq")
    assert not mem._lexically_equivalent_value("Rome", "Roma")
    # Short names are not typo-merged.
    assert not mem._subjects_are_equivalent("May", "Maya")
    assert not mem._subjects_are_equivalent("Alice", "Alicia")


# ── Historical and negated statements ──────────────────────────────────────


def test_historical_statement_is_closed_and_does_not_change_current(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = mem.absorb(
        Candidate("Alice", "used to live in", "Addis Ababa", timestamp=20.0)
    )

    assert result["action"] == "historical_observation"
    assert result["valid_from"] == 20.0
    assert result["valid_until"] == 20.0

    active = _active(mem)
    assert len(active) == 1
    assert active[0].text == "Alice lives in Addis Ababa."
    assert active[0].valid_from == 10.0
    assert active[0].valid_until is None

    historical = next(s for s in mem.memory.slots if not s.is_active)
    assert historical.valid_from == historical.valid_until == 20.0
    assert historical.supersedes is None and historical.superseded_by is None

    assert mem.temporal_history_violations() == []
    assert _conflicts(mem) == []
    assert [(r["valid_from"], r["valid_until"])
            for r in mem.get_history("Alice")] == [
        (10.0, None), (20.0, 20.0),
    ]


def test_historical_statement_with_other_value_preserves_current(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Alice", "lives in", "Paris", timestamp=10.0))
    result = mem.absorb(
        Candidate("Alice", "previously lived in", "Rome", timestamp=20.0)
    )

    assert result["action"] == "historical_observation"
    assert len(_active(mem)) == 1
    assert "Paris" in _active(mem)[0].text
    assert _conflicts(mem) == []


def test_negated_restatement_is_not_consolidated(models):
    mem = _memory(models, verifier=_RuleVerifier([], default="CONTRADICTION"))
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = mem.absorb(
        Candidate("Alice", "no longer lives in", "Addis Ababa", timestamp=20.0)
    )

    assert result["action"] == "temporal_update"
    assert len(_active(mem)) == 1
    assert "no longer" in _active(mem)[0].text


# ── Pinned real-NLI behaviour ──────────────────────────────────────────────


def test_grammar_and_paraphrase_variants_consolidate_with_pinned_nli(models):
    mem = _memory(models, verifier=None)
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))

    grammar = mem.absorb(
        Candidate("Alice", "is living in", "Addis Ababa", timestamp=20.0)
    )
    paraphrase = mem.absorb(
        Candidate("Alice", "based in", "Addis Ababa", timestamp=30.0)
    )

    assert grammar["action"] == "consolidate"
    assert paraphrase["action"] == "consolidate"
    assert len(_active(mem)) == 1


def test_historical_marker_blocks_consolidation_with_pinned_nli(models):
    mem = _memory(models, verifier=None)
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = mem.absorb(
        Candidate("Alice", "used to live in", "Addis Ababa", timestamp=20.0)
    )

    assert result["action"] == "historical_observation"
    assert len(_active(mem)) == 1
    assert _active(mem)[0].text == "Alice lives in Addis Ababa."
