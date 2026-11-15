"""Remediation 3 regression tests: interval-aware routing and cardinality.

Each test pins one of the four integrity defects found by the independent
audit of Remediation 2:

  F1  same-family cross-predicate merge destroys one attribute
  F2  backdated exclusive observation becomes a second open-ended value
  F3  a transition marker bypasses predicate cardinality
  F4  unclassified cross-predicate facts are merged on mutual SAME alone

The verifier is scripted and statement-shaped so that a value-only comparison
("Paris" vs "Nairobi") is never accidentally SAME.
"""

from __future__ import annotations

import numpy as np
import pytest

from minemos_engine.black_hole_memory import UNLINKED_EXCLUSIVE_CONFLICT
from minemos_engine.structured_candidate import Candidate
from tests._retrieval_helpers import FixedEncoder, make_memory


class ScriptedVerifier:
    """First matching (slot text, candidate text, decision) rule wins."""

    def __init__(self, rules=(), default="NEUTRAL"):
        self.rules = [
            (str(slot).lower(), str(cand).lower(), decision)
            for slot, cand, decision in rules
        ]
        self.default = default

    def _decide(self, left: str, right: str) -> str:
        left_l = (left or "").lower()
        right_l = (right or "").lower()
        for slot_marker, cand_marker, decision in self.rules:
            if slot_marker in left_l and cand_marker in right_l:
                return decision
        return self.default

    @staticmethod
    def _direction(decision, same_entail, contra):
        return {
            "decision": decision,
            "confidence": 0.99,
            "p_entail": same_entail,
            "p_contra": 0.99 if contra else 0.01,
            "p_neutral": 0.98 if decision == "NEUTRAL" else 0.01,
        }

    def verify_batch(self, pairs, **kwargs):
        out = []
        for left, right in pairs:
            decision = self._decide(left, right)
            same = decision == "SAME"
            contra = decision == "CONTRADICTION"
            out.append({
                "decision": decision,
                "confidence": 0.99,
                "p_entail_fwd": 0.95 if same else 0.01,
                "p_entail_rev": 0.93 if same else 0.01,
                "fwd": self._direction(decision, 0.95 if same else 0.01, contra),
                "rev": self._direction(decision, 0.93 if same else 0.01, contra),
            })
        return out


def _mem(verifier, encoder=None, **kwargs):
    return make_memory(capacity=20, verifier=verifier, encoder=encoder,
                       **kwargs)


def _slot(mem, memory_id):
    return next(s for s in mem.memory.slots if s.memory_id == memory_id)


def _memory_id(mem, result):
    return mem.memory.slots[result["slot_index"]].memory_id


def _text(mem, slot):
    return mem.slot_texts.get(id(slot), slot.text or "")


def _active_rows(mem, predicate):
    return mem.recall("Ada", predicate=predicate, active_only=True)


def _rows(mem, predicate, **kwargs):
    return mem.recall("Ada", predicate=predicate, **kwargs)


# ── F1: employer and work location are different attributes ────────────────


@pytest.mark.parametrize("order", ["works_at_first", "works_in_first"])
def test_f1_employer_and_work_location_are_both_retrievable(order):
    verifier = ScriptedVerifier([
        ("ada works at acme.", "ada works in london.", "SAME"),
        ("ada works in london.", "ada works at acme.", "SAME"),
    ])
    mem = _mem(verifier)
    if order == "works_at_first":
        first = mem.absorb(Candidate("Ada", "works at", "Acme", timestamp=100.0))
        second = mem.absorb(Candidate("Ada", "works in", "London", timestamp=150.0))
    else:
        first = mem.absorb(Candidate("Ada", "works in", "London", timestamp=100.0))
        second = mem.absorb(Candidate("Ada", "works at", "Acme", timestamp=150.0))

    assert second["action"] == "insert"
    assert {_text(mem, _slot(mem, _memory_id(mem, first))),
            _text(mem, _slot(mem, _memory_id(mem, second)))} == {
        "Ada works at Acme.", "Ada works in London."}
    assert len(_active_rows(mem, "works at")) == 1
    assert len(_active_rows(mem, "works in")) == 1
    assert len(mem.memory.slots) == 2


# ── F2: interval-aware backdated exclusive routing ─────────────────────────


def _three_city_timeline(verifier):
    mem = _mem(verifier)
    paris = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    rome = mem.absorb(Candidate("Ada", "lives in", "Rome", timestamp=300.0))
    assert rome["action"] == "temporal_update"
    nairobi = mem.absorb(
        Candidate("Ada", "lives in", "Nairobi", timestamp=200.0))
    return mem, paris, rome, nairobi


def _assert_three_city_timeline(mem, paris, rome, nairobi):
    paris_slot = _slot(mem, _memory_id(mem, paris))
    rome_slot = _slot(mem, _memory_id(mem, rome))
    nairobi_slot = _slot(mem, _memory_id(mem, nairobi))

    assert (paris_slot.valid_from, paris_slot.valid_until) == (100.0, 200.0)
    assert (nairobi_slot.valid_from, nairobi_slot.valid_until) == (200.0, 300.0)
    assert (rome_slot.valid_from, rome_slot.valid_until) == (300.0, None)
    assert [s.memory_id for s in mem.memory.slots if s.is_active] == [
        rome_slot.memory_id]

    assert paris_slot.superseded_by == nairobi["slot_id"]
    assert nairobi_slot.supersedes == paris["slot_id"]
    assert nairobi_slot.superseded_by == rome["slot_id"]
    assert rome_slot.supersedes == nairobi["slot_id"]

    assert [r["text"] for r in _rows(mem, "lives in", as_of=150.0)] == [
        "Ada lives in Paris."]
    assert [r["text"] for r in _rows(mem, "lives in", as_of=250.0)] == [
        "Ada lives in Nairobi."]
    assert [r["text"] for r in _rows(mem, "lives in", as_of=350.0)] == [
        "Ada lives in Rome."]

    assert mem.temporal_history_violations() == []
    assert mem.temporal_history_violations(
        include_unlinked_conflicts=True) == []
    assert mem.exclusive_temporal_conflicts() == []


def test_f2_same_against_closed_predecessor_is_interval_aware():
    verifier = ScriptedVerifier([
        ("ada lives in paris.", "ada lives in rome.", "CONTRADICTION"),
        ("ada lives in paris.", "ada lives in nairobi.", "SAME"),
    ])
    mem, paris, rome, nairobi = _three_city_timeline(verifier)

    assert nairobi["action"] == "historical_update"
    _assert_three_city_timeline(mem, paris, rome, nairobi)


def test_f2_neutral_against_closed_predecessor_is_interval_aware():
    verifier = ScriptedVerifier([
        ("ada lives in paris.", "ada lives in rome.", "CONTRADICTION"),
    ])
    mem, paris, rome, nairobi = _three_city_timeline(verifier)

    assert nairobi["action"] == "historical_update"
    _assert_three_city_timeline(mem, paris, rome, nairobi)


def test_f2_backdated_observation_before_earliest_interval():
    verifier = ScriptedVerifier()
    mem = _mem(verifier)
    paris = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=200.0))
    nairobi = mem.absorb(
        Candidate("Ada", "lives in", "Nairobi", timestamp=100.0))

    assert nairobi["action"] == "historical_update"
    nairobi_slot = _slot(mem, _memory_id(mem, nairobi))
    paris_slot = _slot(mem, _memory_id(mem, paris))
    assert (nairobi_slot.valid_from, nairobi_slot.valid_until) == (100.0, 200.0)
    assert (paris_slot.valid_from, paris_slot.valid_until) == (200.0, None)
    assert [s.memory_id for s in mem.memory.slots if s.is_active] == [
        paris_slot.memory_id]
    assert mem.exclusive_temporal_conflicts() == []


def test_f2_neutral_observation_inside_closed_interval_splits_it():
    verifier = ScriptedVerifier([
        ("ada lives in paris.", "ada lives in rome.", "CONTRADICTION"),
    ])
    mem = _mem(verifier)
    paris = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    rome = mem.absorb(Candidate("Ada", "lives in", "Rome", timestamp=200.0))
    nairobi = mem.absorb(
        Candidate("Ada", "lives in", "Nairobi", timestamp=150.0))

    assert nairobi["action"] == "historical_update"
    paris_slot = _slot(mem, _memory_id(mem, paris))
    nairobi_slot = _slot(mem, _memory_id(mem, nairobi))
    rome_slot = _slot(mem, _memory_id(mem, rome))
    assert (paris_slot.valid_from, paris_slot.valid_until) == (100.0, 150.0)
    assert (nairobi_slot.valid_from, nairobi_slot.valid_until) == (150.0, 200.0)
    assert (rome_slot.valid_from, rome_slot.valid_until) == (200.0, None)
    assert [r["text"] for r in _rows(mem, "lives in", as_of=120.0)] == [
        "Ada lives in Paris."]
    assert [r["text"] for r in _rows(mem, "lives in", as_of=150.0)] == [
        "Ada lives in Nairobi."]
    assert [r["text"] for r in _rows(mem, "lives in", as_of=210.0)] == [
        "Ada lives in Rome."]
    assert len([s for s in mem.memory.slots if s.is_active]) == 1
    assert mem.exclusive_temporal_conflicts() == []


def test_f2_neutral_observation_between_intervals_is_bounded():
    verifier = ScriptedVerifier([
        ("ada lives in paris.", "ada lives in rome.", "CONTRADICTION"),
    ])
    mem = _mem(verifier)
    paris = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    rome = mem.absorb(Candidate("Ada", "lives in", "Rome", timestamp=300.0))
    nairobi = mem.absorb(
        Candidate("Ada", "lives in", "Nairobi", timestamp=250.0))

    assert nairobi["action"] == "historical_update"
    nairobi_slot = _slot(mem, _memory_id(mem, nairobi))
    paris_slot = _slot(mem, _memory_id(mem, paris))
    rome_slot = _slot(mem, _memory_id(mem, rome))
    assert (nairobi_slot.valid_from, nairobi_slot.valid_until) == (250.0, 300.0)
    assert (paris_slot.valid_from, paris_slot.valid_until) == (100.0, 250.0)
    assert (rome_slot.valid_from, rome_slot.valid_until) == (300.0, None)
    assert [r["text"] for r in _rows(mem, "lives in", as_of=200.0)] == [
        "Ada lives in Paris."]
    assert [r["text"] for r in _rows(mem, "lives in", as_of=250.0)] == [
        "Ada lives in Nairobi."]
    assert len([s for s in mem.memory.slots if s.is_active]) == 1
    assert mem.exclusive_temporal_conflicts() == []


def test_f2_backdated_equivalent_value_consolidates_into_containing_interval():
    verifier = ScriptedVerifier([
        ("ada lives in paris.", "ada lives in rome.", "CONTRADICTION"),
        ("ada lives in paris.", "ada lives in paris.", "SAME"),
    ])
    mem = _mem(verifier)
    paris = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    mem.absorb(Candidate("Ada", "lives in", "Rome", timestamp=300.0))
    result = mem.absorb(
        Candidate("Ada", "lives in", "Paris", timestamp=200.0))

    assert result["action"] == "consolidate"
    paris_slot = _slot(mem, _memory_id(mem, paris))
    assert (paris_slot.valid_from, paris_slot.valid_until) == (100.0, 300.0)
    assert _text(mem, paris_slot) == "Ada lives in Paris."
    assert len([s for s in mem.memory.slots if s.is_active]) == 1
    assert mem.temporal_history_violations(
        include_unlinked_conflicts=True) == []


def test_f2_checkpoint_roundtrip_preserves_segmented_history(tmp_path):
    verifier = ScriptedVerifier([
        ("ada lives in paris.", "ada lives in rome.", "CONTRADICTION"),
        ("ada lives in paris.", "ada lives in nairobi.", "SAME"),
    ])
    mem, paris, rome, nairobi = _three_city_timeline(verifier)
    ids = {_text(mem, s): s.memory_id for s in mem.memory.slots}
    path = str(tmp_path / "r3_timeline.npz")
    mem.save_checkpoint(path, allowed_base_dir=str(tmp_path))

    restored = _mem(verifier)
    restored.load_checkpoint(path, allowed_base_dir=str(tmp_path))

    assert {_text(restored, s): s.memory_id for s in restored.memory.slots} == ids
    assert [
        (s.valid_from, s.valid_until)
        for s in sorted(restored.memory.slots, key=lambda s: s.valid_from)
    ] == [(100.0, 200.0), (200.0, 300.0), (300.0, None)]
    assert len([s for s in restored.memory.slots if s.is_active]) == 1
    assert restored.temporal_history_violations() == []
    assert restored.temporal_history_violations(
        include_unlinked_conflicts=True) == []
    assert [r["text"] for r in _rows(restored, "lives in", as_of=250.0)] == [
        "Ada lives in Nairobi."]


# ── F3: transition markers do not bypass cardinality ──────────────────────


def _high_similarity_encoder(texts):
    shared = np.ones(384, dtype=np.float32) / np.sqrt(384)
    return FixedEncoder({text: shared for text in texts})


def test_f3_multivalued_transition_marker_does_not_supersede():
    tea_text = "Ada likes tea."
    coffee_text = "Ada likes coffee now."
    verifier = ScriptedVerifier([(tea_text, coffee_text, "SAME")])
    mem = _mem(verifier, encoder=_high_similarity_encoder(
        [tea_text, coffee_text]))

    tea = mem.absorb(Candidate("Ada", "likes", "tea", timestamp=100.0))
    coffee = mem.absorb(
        Candidate("Ada", "likes", "coffee now", timestamp=150.0))

    assert coffee["action"] == "insert"
    assert {r["text"] for r in _active_rows(mem, "likes")} == {
        "Ada likes tea.", "Ada likes coffee now."}
    assert _slot(mem, _memory_id(mem, tea)).is_active
    assert _slot(mem, _memory_id(mem, coffee)).is_active


def test_f3_unknown_cardinality_transition_marker_does_not_supersede():
    tool_one = "Ada uses tool 1."
    tool_two = "Ada uses tool 2 now."
    verifier = ScriptedVerifier([(tool_one, tool_two, "SAME")])
    mem = _mem(verifier, encoder=_high_similarity_encoder(
        [tool_one, tool_two]))

    mem.absorb(Candidate("Ada", "uses", "tool 1", timestamp=100.0))
    result = mem.absorb(
        Candidate("Ada", "uses", "tool 2 now", timestamp=150.0))

    assert result["action"] == "insert"
    assert {r["text"] for r in _active_rows(mem, "uses")} == {
        "Ada uses tool 1.", "Ada uses tool 2 now."}
    assert len([s for s in mem.memory.slots if s.is_active]) == 2


def test_f3_exclusive_transition_marker_still_supersedes():
    verifier = ScriptedVerifier([
        ("ada lives in paris.", "ada lives in rome now.", "CONTRADICTION"),
    ])
    mem = _mem(verifier, encoder=_high_similarity_encoder([
        "Ada lives in Paris.", "Ada lives in Rome now."]))

    paris = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    result = mem.absorb(
        Candidate("Ada", "lives in", "Rome now", timestamp=150.0))

    assert result["action"] == "temporal_update"
    paris_slot = _slot(mem, _memory_id(mem, paris))
    assert (paris_slot.valid_from, paris_slot.valid_until) == (100.0, 150.0)
    assert [r["text"] for r in _active_rows(mem, "lives in")] == [
        "Ada lives in Rome now."]


def test_f3_transition_marker_corrects_the_same_object():
    owns = "Ada owns the majority share in Acme."
    sold = "Ada sold all equity in Acme."
    verifier = ScriptedVerifier([(owns, sold, "CONTRADICTION")])
    mem = _mem(verifier, encoder=_high_similarity_encoder([owns, sold]))

    mem.absorb(Candidate("Ada", "owns the majority share in", "Acme",
                         timestamp=100.0))
    result = mem.absorb(Candidate("Ada", "sold all equity in", "Acme",
                                  timestamp=150.0))

    assert result["action"] == "temporal_update"
    assert [s for s in mem.memory.slots if s.is_active][0].text == sold


# ── F4: unclassified cross-predicate facts are preserved ──────────────────


@pytest.mark.parametrize("order", ["plays_first", "listens_first"])
def test_f4_unclassified_cross_predicate_facts_are_preserved(order):
    verifier = ScriptedVerifier([
        ("ada plays guitar.", "ada listens to jazz.", "SAME"),
        ("ada listens to jazz.", "ada plays guitar.", "SAME"),
    ])
    mem = _mem(verifier)
    if order == "plays_first":
        first = mem.absorb(Candidate("Ada", "plays", "guitar", timestamp=100.0))
        second = mem.absorb(
            Candidate("Ada", "listens to", "jazz", timestamp=150.0))
    else:
        first = mem.absorb(
            Candidate("Ada", "listens to", "jazz", timestamp=100.0))
        second = mem.absorb(Candidate("Ada", "plays", "guitar", timestamp=150.0))

    assert second["action"] == "insert"
    assert len(_active_rows(mem, "plays")) == 1
    assert len(_active_rows(mem, "listens to")) == 1
    assert {_text(mem, _slot(mem, _memory_id(mem, first))),
            _text(mem, _slot(mem, _memory_id(mem, second)))} == {
        "Ada plays guitar.", "Ada listens to jazz."}


def test_f4_unclassified_paraphrase_with_equivalent_object_consolidates():
    verifier = ScriptedVerifier([
        ("ada plays guitar.", "ada performs guitar.", "SAME"),
    ])
    mem = _mem(verifier)
    mem.absorb(Candidate("Ada", "plays", "guitar", timestamp=100.0))
    result = mem.absorb(
        Candidate("Ada", "performs", "guitar", timestamp=150.0))

    assert result["action"] == "consolidate"
    assert len(mem.memory.slots) == 1


# ── Invariant: unlinked conflicts are visible to acceptance ────────────────


def test_unlinked_exclusive_conflict_is_reported_by_both_checks():
    mem = _mem(ScriptedVerifier())
    seed = mem.memory.add(
        key=np.zeros(mem.dim, dtype="float32"),
        value=np.zeros(mem.dim, dtype="float32"),
        confidence=1.0, timestamp=0.0, debug_subject="seed",
        debug_predicate="seed", text="seed", valid_from=0.0,
        valid_until=0.0, latest_predicate="seed", latest_object="seed",
    )
    for start, value in ((10.0, "Rome"), (20.0, "Venice")):
        mem.memory.add(
            key=seed.key.copy(), value=seed.value.copy(), confidence=1.0,
            timestamp=start, debug_subject="Maya", debug_predicate="lives in",
            text=f"Maya lives in {value}.", dense_emb=seed.dense_emb,
            valid_from=start, valid_until=None, latest_predicate="lives in",
            latest_object=value,
        )
    mem.memory.slots.pop(0)
    mem._invalidate_retrieval_cache()

    conflicts = mem.exclusive_temporal_conflicts()
    assert len(conflicts) == 1
    assert conflicts[0]["reason"] == UNLINKED_EXCLUSIVE_CONFLICT
    assert conflicts[0]["values"] == ["Rome", "Venice"]

    default = mem.temporal_history_violations()
    assert default == []
    full = mem.temporal_history_violations(include_unlinked_conflicts=True)
    assert [v["reason"] for v in full] == [UNLINKED_EXCLUSIVE_CONFLICT]
