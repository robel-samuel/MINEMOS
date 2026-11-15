"""Consolidation integrity and temporal correctness.

Each ``test_original_case_*`` reproduces the mechanism of one of eleven
missing-fact cases with deterministic, non-private text. The private
conversation replay that exercises the original observations lives outside the
repository; these tests pin the desired behaviour permanently.

The verifier below is deliberately scripted.  It returns the same directional
entailment shapes recorded in the failure analysis (including one-way SAME
verdicts) so the tests exercise the engine's decision policy rather than the
current opinion of the real NLI model.
"""

from __future__ import annotations

from minemos_engine.black_hole_memory import (
    PREDICATE_POLICY_EXCLUSIVE,
    PREDICATE_POLICY_MULTI_VALUED,
    PREDICATE_POLICY_UNKNOWN,
)
from minemos_engine.structured_candidate import Candidate
from tests._retrieval_helpers import (
    FixedEncoder,
    StubNLI,
    make_memory,
    rows_equal,
)


class DirectionalVerifier:
    """Recorded-shape verifier: first matching rule wins.

    A rule is ``(slot_marker, candidate_marker, decision, fwd_entail,
    rev_entail)``.  Both texts are lowercased before matching.  Unmatched
    pairs are NEUTRAL in both directions.
    """

    def __init__(self, rules=(), default=("NEUTRAL", 0.01, 0.01)):
        self.rules = [tuple(rule) for rule in rules]
        self.default = default

    def _score(self, left: str, right: str):
        left_l = (left or "").lower()
        right_l = (right or "").lower()
        for slot_marker, cand_marker, decision, fwd, rev in self.rules:
            if slot_marker in left_l and cand_marker in right_l:
                return decision, fwd, rev
        return self.default

    @staticmethod
    def _direction(decision, score):
        contra = 0.99 if decision == "CONTRADICTION" else 0.01
        neutral = 0.98 if decision == "NEUTRAL" else 0.01
        return {
            "decision": decision,
            "confidence": float(score),
            "p_entail": float(score),
            "p_contra": contra,
            "p_neutral": neutral,
        }

    def verify_batch(self, pairs, **kwargs):
        out = []
        for left, right in pairs:
            decision, fwd, rev = self._score(left, right)
            out.append({
                "decision": decision,
                "confidence": max(float(fwd), float(rev)),
                "p_entail_fwd": float(fwd),
                "p_entail_rev": float(rev),
                "fwd": self._direction(decision, fwd),
                "rev": self._direction(decision, rev),
            })
        return out


def _mem(verifier, **kwargs):
    return make_memory(capacity=20, verifier=verifier, **kwargs)


def _slot(mem, memory_id):
    return next(s for s in mem.memory.slots if s.memory_id == memory_id)


def _memory_id(mem, result):
    return mem.memory.slots[result["slot_index"]].memory_id


def _text(mem, slot):
    return mem.slot_texts.get(id(slot), slot.text or "")


def _rows(mem, predicate, **kwargs):
    return mem.recall("Ada", predicate=predicate, **kwargs)


def _active_rows(mem, predicate, **kwargs):
    return mem.recall("Ada", predicate=predicate, active_only=True, **kwargs)


# ── Original eleven cases ──────────────────────────────────────────────────


def test_original_case_00_backdated_observation_keeps_closed_slot_intact():
    """fact 0: a backdated observation must not be merged into closed history."""
    verifier = DirectionalVerifier([
        ("paris", "nairobi", "SAME", 0.98, 0.95),
        ("rome", "nairobi", "CONTRADICTION", 0.95, 0.93),
        ("paris", "rome", "CONTRADICTION", 0.94, 0.93),
    ])
    mem = _mem(verifier)
    paris = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    rome = mem.absorb(Candidate("Ada", "lives in", "Rome", timestamp=300.0))
    assert rome["action"] == "temporal_update"

    paris_id = _memory_id(mem, paris)
    result = mem.absorb(Candidate("Ada", "lives in", "Nairobi", timestamp=50.0))
    assert result["action"] == "historical_update"

    paris_slot = _slot(mem, paris_id)
    assert _text(mem, paris_slot) == "Ada lives in Paris."
    assert paris_slot.valid_from == 100.0
    assert paris_slot.valid_until == 300.0

    nairobi = _slot(mem, _memory_id(mem, result))
    assert nairobi.memory_id != paris_id
    assert _text(mem, nairobi) == "Ada lives in Nairobi."
    assert nairobi.is_active is False
    assert (nairobi.valid_from, nairobi.valid_until) == (50.0, 100.0)

    assert [r["text"] for r in _rows(mem, "lives in", as_of=75.0)] == [
        "Ada lives in Nairobi."]
    assert [r["text"] for r in _rows(mem, "lives in", as_of=150.0)] == [
        "Ada lives in Paris."]
    assert [r["text"] for r in _active_rows(mem, "lives in")] == [
        "Ada lives in Rome."]


def test_original_case_04_backdated_distinct_value_does_not_rewrite_interval():
    """fact 4: equivalence against a successor is not equivalence with history."""
    verifier = DirectionalVerifier([
        ("lima", "oslo", "CONTRADICTION", 0.93, 0.92),
        ("lima", "visa", "CONTRADICTION", 0.90, 0.88),
        ("oslo", "visa", "SAME", 0.80, 0.34),
    ])
    mem = _mem(verifier)
    lima = mem.absorb(Candidate("Ada", "lives in", "Lima", timestamp=100.0))
    oslo = mem.absorb(Candidate("Ada", "lives in", "Oslo", timestamp=300.0))
    assert oslo["action"] == "temporal_update"

    lima_id = _memory_id(mem, lima)
    result = mem.absorb(Candidate("Ada", "lives in", "worker visa", timestamp=200.0))
    assert result["action"] == "historical_update"

    lima_slot = _slot(mem, lima_id)
    assert _text(mem, lima_slot) == "Ada lives in Lima."
    assert (lima_slot.valid_from, lima_slot.valid_until) == (100.0, 200.0)

    visa_slot = _slot(mem, _memory_id(mem, result))
    assert visa_slot.memory_id != lima_id
    assert _text(mem, visa_slot) == "Ada lives in worker visa."
    assert (visa_slot.valid_from, visa_slot.valid_until) == (200.0, 300.0)

    assert [r["text"] for r in _rows(mem, "lives in", as_of=150.0)] == [
        "Ada lives in Lima."]
    assert [r["text"] for r in _rows(mem, "lives in", as_of=250.0)] == [
        "Ada lives in worker visa."]
    assert [r["text"] for r in _active_rows(mem, "lives in")] == [
        "Ada lives in Oslo."]


def test_original_case_05_superseded_history_is_never_rewritten():
    """fact 5: a normal supersession is not permission to overwrite closed text."""
    verifier = DirectionalVerifier([
        ("lisbon", "oslo", "CONTRADICTION", 0.94, 0.93),
        ("lisbon", "quito", "SAME", 0.96, 0.91),
        ("oslo", "quito", "CONTRADICTION", 0.95, 0.94),
    ])
    mem = _mem(verifier)
    lisbon = mem.absorb(Candidate("Ada", "lives in", "Lisbon", timestamp=100.0))
    oslo = mem.absorb(Candidate("Ada", "lives in", "Oslo", timestamp=300.0))
    assert oslo["action"] == "temporal_update"

    lisbon_id = _memory_id(mem, lisbon)
    result = mem.absorb(Candidate("Ada", "lives in", "Quito", timestamp=50.0))
    assert result["action"] == "historical_update"

    lisbon_slot = _slot(mem, lisbon_id)
    assert _text(mem, lisbon_slot) == "Ada lives in Lisbon."
    assert (lisbon_slot.valid_from, lisbon_slot.valid_until) == (100.0, 300.0)

    quito_slot = _slot(mem, _memory_id(mem, result))
    assert _text(mem, quito_slot) == "Ada lives in Quito."
    assert (quito_slot.valid_from, quito_slot.valid_until) == (50.0, 100.0)
    assert lisbon_slot.superseded_by == oslo["slot_id"]


def test_original_case_17_cross_predicate_relation_is_not_erased():
    """fact 17: a one-way SAME over ``has`` must not replace a ``uses`` relation."""
    verifier = DirectionalVerifier([
        ("compiler", "laptop", "SAME", 0.97, 0.17),
    ])
    mem = _mem(verifier)
    compiler = mem.absorb(Candidate("Ada", "uses", "compiler", timestamp=100.0))
    laptop = mem.absorb(Candidate("Ada", "has", "laptop", timestamp=150.0))

    compiler_slot = _slot(mem, _memory_id(mem, compiler))
    laptop_slot = _slot(mem, _memory_id(mem, laptop))
    assert compiler_slot.memory_id != laptop_slot.memory_id
    assert _text(mem, compiler_slot) == "Ada uses compiler."
    assert _text(mem, laptop_slot) == "Ada has laptop."
    assert compiler_slot.latest_predicate == "uses"
    assert laptop_slot.latest_predicate == "has"
    assert compiler_slot.is_active and laptop_slot.is_active

    uses_rows = _active_rows(mem, "uses")
    has_rows = _active_rows(mem, "has")
    assert [r["predicate"] for r in uses_rows] == ["uses"]
    assert [r["predicate"] for r in has_rows] == ["has"]
    assert "laptop" not in uses_rows[0]["text"]
    assert "compiler" not in has_rows[0]["text"]


def test_original_case_21_equal_timestamp_unrelated_values_are_independent():
    """fact 21: an unrelated value at the same timestamp is not a state change."""
    verifier = DirectionalVerifier([
        ("tool 1", "tool 2", "SAME", 0.99, 0.002),
    ])
    mem = _mem(verifier)
    first = mem.absorb(Candidate("Ada", "uses", "tool 1", timestamp=100.0))
    second = mem.absorb(Candidate("Ada", "uses", "tool 2", timestamp=100.0))

    assert first["action"] == "insert"
    assert second["action"] == "insert"
    first_slot = _slot(mem, _memory_id(mem, first))
    second_slot = _slot(mem, _memory_id(mem, second))
    assert first_slot.memory_id != second_slot.memory_id
    assert first_slot.is_active and second_slot.is_active
    assert first_slot.valid_from == second_slot.valid_from == 100.0
    assert first_slot.valid_until is None and second_slot.valid_until is None
    assert {_text(mem, first_slot), _text(mem, second_slot)} == {
        "Ada uses tool 1.", "Ada uses tool 2."}


def test_original_case_24_one_way_entailment_cannot_replace_a_value():
    """fact 24: forward-only entailment on unequal values is not equivalence."""
    verifier = DirectionalVerifier([
        ("compiler", "editor", "SAME", 0.9875, 0.002),
    ])
    mem = _mem(verifier)
    compiler = mem.absorb(Candidate("Ada", "uses", "compiler", timestamp=100.0))
    editor = mem.absorb(Candidate("Ada", "uses", "editor", timestamp=150.0))

    compiler_slot = _slot(mem, _memory_id(mem, compiler))
    editor_slot = _slot(mem, _memory_id(mem, editor))
    assert compiler_slot.memory_id != editor_slot.memory_id
    assert _text(mem, compiler_slot) == "Ada uses compiler."
    assert _text(mem, editor_slot) == "Ada uses editor."
    assert compiler_slot.is_active and editor_slot.is_active


def test_original_case_29_preference_is_not_rewritten_by_a_role():
    """fact 29: ``likes`` and ``works as`` are different relations."""
    verifier = DirectionalVerifier([
        ("jazz", "teacher", "SAME", 0.96, 0.90),
    ])
    mem = _mem(verifier)
    jazz = mem.absorb(Candidate("Ada", "likes", "jazz", timestamp=100.0))
    teacher = mem.absorb(Candidate("Ada", "works as", "teacher", timestamp=150.0))

    jazz_slot = _slot(mem, _memory_id(mem, jazz))
    teacher_slot = _slot(mem, _memory_id(mem, teacher))
    assert jazz_slot.memory_id != teacher_slot.memory_id
    assert _text(mem, jazz_slot) == "Ada likes jazz."
    assert _text(mem, teacher_slot) == "Ada works as teacher."
    assert _active_rows(mem, "likes")[0]["text"] == "Ada likes jazz."
    works_rows = _active_rows(mem, "works as")
    assert [r["predicate"] for r in works_rows] == ["works as"]
    assert works_rows[0]["original_predicate"] == "works as"


def test_original_case_30_uses_relation_survives_a_has_observation():
    """fact 30: a ``has`` observation must not erase a ``uses`` value."""
    verifier = DirectionalVerifier([
        ("brush", "canvas", "SAME", 0.962, 0.006),
    ])
    mem = _mem(verifier)
    brush = mem.absorb(Candidate("Ada", "uses", "brush", timestamp=100.0))
    canvas = mem.absorb(Candidate("Ada", "has", "canvas", timestamp=150.0))

    brush_slot = _slot(mem, _memory_id(mem, brush))
    canvas_slot = _slot(mem, _memory_id(mem, canvas))
    assert brush_slot.memory_id != canvas_slot.memory_id
    assert _text(mem, brush_slot) == "Ada uses brush."
    assert _text(mem, canvas_slot) == "Ada has canvas."
    assert brush_slot.latest_predicate == "uses"
    assert canvas_slot.latest_predicate == "has"


def test_original_case_37_unknown_numeric_predicate_is_not_exclusive():
    """fact 37: different numbers under an UNKNOWN predicate are independent."""
    verifier = DirectionalVerifier([
        ("tool 2", "tool 42", "CONTRADICTION", 0.94, 0.93),
    ])
    mem = _mem(verifier)
    assert mem.classify_predicate("uses") == PREDICATE_POLICY_UNKNOWN

    two = mem.absorb(Candidate("Ada", "uses", "tool 2", timestamp=100.0))
    forty_two = mem.absorb(Candidate("Ada", "uses", "tool 42", timestamp=200.0))

    two_slot = _slot(mem, _memory_id(mem, two))
    other_slot = _slot(mem, _memory_id(mem, forty_two))
    assert two_slot.memory_id != other_slot.memory_id
    assert two_slot.is_active and other_slot.is_active
    assert two_slot.valid_until is None
    assert _text(mem, two_slot) == "Ada uses tool 2."
    assert _text(mem, other_slot) == "Ada uses tool 42."


def test_original_case_38_public_predicate_reflects_represented_relation():
    """fact 38: recall must expose the relation the slot represents."""
    verifier = DirectionalVerifier([
        ("paris", "resides", "SAME", 0.99, 0.98),
    ])
    mem = _mem(verifier)
    mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    result = mem.absorb(Candidate("Ada", "resides in", "Paris", timestamp=150.0))
    assert result["action"] == "consolidate"

    rows = _active_rows(mem, "resides in")
    assert rows == _active_rows(mem, "lives in")
    assert len(rows) == 1
    assert rows[0]["predicate"] == "resides in"
    assert rows[0]["original_predicate"] == "lives in"
    slot = _slot(mem, rows[0]["memory_id"])
    assert slot.latest_predicate == "resides in"
    assert slot.debug_predicate == "lives in"


def test_original_case_44_backdated_independent_value_keeps_true_validity():
    """fact 44: one-way entailment must not absorb a backdated distinct value."""
    verifier = DirectionalVerifier([
        ("team alpha", "team beta", "SAME", 0.660, 0.015),
    ])
    mem = _mem(verifier)
    alpha = mem.absorb(Candidate("Ada", "uses", "team alpha", timestamp=300.0))
    beta = mem.absorb(Candidate("Ada", "uses", "team beta", timestamp=100.0))

    alpha_slot = _slot(mem, _memory_id(mem, alpha))
    beta_slot = _slot(mem, _memory_id(mem, beta))
    assert alpha_slot.memory_id != beta_slot.memory_id
    assert _text(mem, alpha_slot) == "Ada uses team alpha."
    assert _text(mem, beta_slot) == "Ada uses team beta."
    assert alpha_slot.valid_from == 300.0
    assert beta_slot.valid_from == 100.0
    assert alpha_slot.is_active and beta_slot.is_active


# ── Richer and shorter representations ─────────────────────────────────────


def test_richer_value_is_not_replaced_by_a_shorter_restatement():
    verifier = DirectionalVerifier([
        ("live improvisation", "jazz", "SAME", 0.96, 0.94),
    ])
    mem = _mem(verifier)
    rich = mem.absorb(Candidate(
        "Ada", "likes", "jazz with live improvisation", timestamp=100.0))
    short = mem.absorb(Candidate("Ada", "likes", "jazz", timestamp=150.0))

    assert short["action"] == "consolidate"
    rich_slot = _slot(mem, _memory_id(mem, rich))
    assert _text(mem, rich_slot) == "Ada likes jazz with live improvisation."
    assert rich_slot.latest_object == "jazz with live improvisation"
    assert rich_slot.update_count == 2
    assert len(mem.memory.slots) == 1


def test_richer_restatement_replaces_a_shorter_value():
    verifier = DirectionalVerifier([
        ("jazz", "live improvisation", "SAME", 0.96, 0.94),
    ])
    mem = _mem(verifier)
    short = mem.absorb(Candidate("Ada", "likes", "jazz", timestamp=100.0))
    rich = mem.absorb(Candidate(
        "Ada", "likes", "jazz with live improvisation", timestamp=150.0))

    assert rich["action"] == "consolidate"
    slot = _slot(mem, _memory_id(mem, short))
    assert _text(mem, slot) == "Ada likes jazz with live improvisation."
    assert slot.latest_object == "jazz with live improvisation"


# ── Predicate identity and cardinality ─────────────────────────────────────


def test_same_subject_different_predicates_coexist():
    verifier = DirectionalVerifier([
        ("paris", "acme", "SAME", 0.95, 0.90),
    ])
    mem = _mem(verifier)
    mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    mem.absorb(Candidate("Ada", "works at", "Acme", timestamp=150.0))

    assert len(_active_rows(mem, "lives in")) == 1
    assert len(_active_rows(mem, "works at")) == 1
    assert _active_rows(mem, "lives in")[0]["text"] == "Ada lives in Paris."
    assert _active_rows(mem, "works at")[0]["text"] == "Ada works at Acme."


def test_same_predicate_independent_values_remain_retrievable():
    verifier = DirectionalVerifier([
        ("brush", "palette", "CONTRADICTION", 0.95, 0.94),
    ])
    mem = _mem(verifier)
    mem.absorb(Candidate("Ada", "uses", "brush", timestamp=100.0))
    mem.absorb(Candidate("Ada", "uses", "palette", timestamp=150.0))

    rows = _active_rows(mem, "uses")
    assert {r["text"] for r in rows} == {
        "Ada uses brush.", "Ada uses palette."}
    assert len({r["memory_id"] for r in rows}) == 2


def test_cumulative_predicate_keeps_every_value():
    verifier = DirectionalVerifier([
        ("tea", "coffee", "CONTRADICTION", 0.92, 0.90),
        ("tea", "milk", "SAME", 0.95, 0.93),
    ])
    mem = _mem(verifier)
    assert mem.classify_predicate("likes") == PREDICATE_POLICY_MULTI_VALUED

    tea = mem.absorb(Candidate("Ada", "likes", "tea", timestamp=100.0))
    coffee = mem.absorb(Candidate("Ada", "likes", "coffee", timestamp=200.0))
    consolidated = mem.absorb(Candidate(
        "Ada", "likes", "tea with milk", timestamp=300.0))

    assert consolidated["action"] == "consolidate"
    assert consolidated["slot_index"] == tea["slot_index"]
    rows = _active_rows(mem, "likes")
    texts = {r["text"] for r in rows}
    assert texts == {"Ada likes tea with milk.", "Ada likes coffee."}
    assert len({r["memory_id"] for r in rows}) == 2
    assert _memory_id(mem, coffee) in {r["memory_id"] for r in rows}


def test_configured_exclusive_predicate_supersedes():
    verifier = DirectionalVerifier([
        ("french", "german", "CONTRADICTION", 0.94, 0.93),
    ])
    mem = _mem(verifier, exclusive_predicates=["speaks"])
    assert mem.classify_predicate("speaks") == PREDICATE_POLICY_EXCLUSIVE

    french = mem.absorb(Candidate("Ada", "speaks", "French", timestamp=100.0))
    german = mem.absorb(Candidate("Ada", "speaks", "German", timestamp=200.0))
    assert german["action"] == "temporal_update"

    french_slot = _slot(mem, _memory_id(mem, french))
    german_slot = _slot(mem, _memory_id(mem, german))
    assert french_slot.memory_id != german_slot.memory_id
    assert french_slot.is_active is False
    assert french_slot.valid_until == 200.0
    assert german_slot.is_active is True
    assert german_slot.supersedes == french["slot_id"]
    assert _text(mem, french_slot) == "Ada speaks French."
    assert _text(mem, german_slot) == "Ada speaks German."


def test_unknown_cardinality_is_not_silently_exclusive():
    verifier = DirectionalVerifier([
        ("alpha", "beta", "CONTRADICTION", 0.94, 0.93),
    ])
    mem = _mem(verifier)
    assert mem.classify_predicate("toggles") == PREDICATE_POLICY_UNKNOWN

    mem.absorb(Candidate("Ada", "toggles", "alpha", timestamp=100.0))
    mem.absorb(Candidate("Ada", "toggles", "beta", timestamp=200.0))

    rows = _active_rows(mem, "toggles")
    assert {r["text"] for r in rows} == {
        "Ada toggles alpha.", "Ada toggles beta."}
    assert all(r["valid_until"] is None for r in rows)


def test_numbered_uses_values_are_independent_tools():
    verifier = DirectionalVerifier([
        ("tool 1", "tool 2", "CONTRADICTION", 0.94, 0.93),
    ])
    mem = _mem(verifier)
    mem.absorb(Candidate("Ada", "uses", "tool 1", timestamp=100.0))
    mem.absorb(Candidate("Ada", "uses", "tool 2", timestamp=200.0))

    rows = _active_rows(mem, "uses")
    assert {r["text"] for r in rows} == {
        "Ada uses tool 1.", "Ada uses tool 2."}
    assert len(rows) == 2


# ── Equal timestamps ───────────────────────────────────────────────────────


def test_equal_time_unrelated_facts_cannot_replace_each_other():
    verifier = DirectionalVerifier([
        ("paris", "acme", "SAME", 0.95, 0.90),
    ])
    mem = _mem(verifier)
    one = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    two = mem.absorb(Candidate("Ada", "works at", "Acme", timestamp=100.0))

    one_slot = _slot(mem, _memory_id(mem, one))
    two_slot = _slot(mem, _memory_id(mem, two))
    assert one_slot.memory_id != two_slot.memory_id
    assert _text(mem, one_slot) == "Ada lives in Paris."
    assert _text(mem, two_slot) == "Ada works at Acme."
    assert one_slot.is_active and two_slot.is_active


def test_equal_time_same_chain_replaces_in_place_without_shifting_time():
    verifier = DirectionalVerifier([
        ("paris", "rome", "CONTRADICTION", 0.95, 0.94),
    ])
    mem = _mem(verifier)
    paris = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    rome = mem.absorb(Candidate("Ada", "lives in", "Rome", timestamp=100.0))

    assert rome["action"] == "temporal_update"
    assert rome.get("replaced_in_place") is True
    slot = _slot(mem, _memory_id(mem, paris))
    assert slot.memory_id == _memory_id(mem, paris)
    assert _text(mem, slot) == "Ada lives in Rome."
    assert slot.valid_from == 100.0
    assert slot.valid_until is None
    assert slot.update_count == 2
    assert len(mem.memory.slots) == 1


# ── Temporal validity ──────────────────────────────────────────────────────


def test_backdated_equivalent_observation_extends_valid_from():
    verifier = DirectionalVerifier([
        ("paris", "paris", "SAME", 0.99, 0.99),
    ])
    mem = _mem(verifier)
    mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=300.0))
    result = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))

    assert result["action"] == "consolidate"
    assert result["valid_from_extended"] is True
    slot = mem.memory.slots[0]
    assert slot.valid_from == 100.0
    assert slot.valid_until is None
    assert slot.update_count == 2

    assert [r["text"] for r in _rows(mem, "lives in", as_of=150.0)] == [
        "Ada lives in Paris."]
    assert _rows(mem, "lives in", as_of=50.0) == []


def test_as_of_respects_correction_boundaries():
    verifier = DirectionalVerifier([
        ("paris", "rome", "CONTRADICTION", 0.95, 0.94),
    ])
    mem = _mem(verifier)
    paris = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    rome = mem.absorb(Candidate("Ada", "lives in", "Rome", timestamp=300.0))

    paris_id = _memory_id(mem, paris)
    rome_id = _memory_id(mem, rome)

    before = _rows(mem, "lives in", as_of=200.0)
    after = _rows(mem, "lives in", as_of=350.0)
    assert len(before) == 1 and len(after) == 1
    assert before[0]["memory_id"] == paris_id
    assert before[0]["is_active"] is False
    assert (before[0]["valid_from"], before[0]["valid_until"]) == (100.0, 300.0)
    assert after[0]["memory_id"] == rome_id
    assert after[0]["is_active"] is True
    assert after[0]["valid_from"] == 300.0

    history = _rows(mem, "lives in", active_only=False)
    assert [(r["memory_id"], r["text"]) for r in history] == [
        (paris_id, "Ada lives in Paris."),
        (rome_id, "Ada lives in Rome."),
    ]


# ── Negation and contradiction ─────────────────────────────────────────────


def test_negation_of_a_different_object_stays_independent():
    verifier = DirectionalVerifier([
        ("north", "south", "CONTRADICTION", 0.94, 0.93),
    ])
    mem = _mem(verifier)
    mem.absorb(Candidate("Ada", "does not approve", "north", timestamp=100.0))
    mem.absorb(Candidate("Ada", "approves", "south", timestamp=150.0))

    rows = mem.recall("Ada", active_only=True)
    assert len(rows) == 2
    assert {r["text"] for r in rows} == {
        "Ada does not approve north.", "Ada approves south."}
    assert all(r["is_active"] for r in rows)


def test_explicit_correction_preserves_prior_history():
    verifier = DirectionalVerifier([
        ("paris", "rome", "CONTRADICTION", 0.95, 0.94),
        ("rome", "paris", "CONTRADICTION", 0.95, 0.94),
        ("paris", "paris", "SAME", 0.99, 0.98),
    ])
    mem = _mem(verifier)
    mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    mem.absorb(Candidate("Ada", "lives in", "Rome", timestamp=200.0))
    mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=300.0))

    rows = _rows(mem, "lives in", active_only=False)
    assert [(r["text"], r["valid_from"], r["valid_until"], r["is_active"])
            for r in rows] == [
        ("Ada lives in Paris.", 100.0, 200.0, False),
        ("Ada lives in Rome.", 200.0, 300.0, False),
        ("Ada lives in Paris.", 300.0, None, True),
    ]
    assert len({r["memory_id"] for r in rows}) == 3


# ── Checkpoint and privacy integration ─────────────────────────────────────


def _checkpoint_snapshot(mem):
    return [
        (
            slot.memory_id,
            slot.debug_subject,
            slot.latest_predicate,
            slot.debug_predicate,
            _text(mem, slot),
            slot.valid_from,
            slot.valid_until,
            slot.is_active,
            slot.redacted,
        )
        for slot in mem.memory.slots
    ]


def _reload(mem, path):
    from minemos_engine.black_hole_memory import BlackHoleMemory
    return BlackHoleMemory.from_checkpoint(
        str(path),
        dim=4096,
        dense_encoder=FixedEncoder(),
        nli_gate=StubNLI(),
        verifier=DirectionalVerifier(),
    )


def _without_slot_id(rows):
    return [{k: v for k, v in row.items() if k != "slot_id"} for row in rows]


def test_checkpoint_preserves_corrected_active_and_historical_state(tmp_path):
    verifier = DirectionalVerifier([
        ("lima", "oslo", "CONTRADICTION", 0.93, 0.92),
        ("lima", "visa", "CONTRADICTION", 0.90, 0.88),
        ("oslo", "visa", "SAME", 0.80, 0.34),
        ("jazz", "live improvisation", "SAME", 0.96, 0.94),
    ])
    mem = _mem(verifier)
    mem.absorb(Candidate("Ada", "lives in", "Lima", timestamp=100.0))
    mem.absorb(Candidate("Ada", "lives in", "Oslo", timestamp=300.0))
    mem.absorb(Candidate("Ada", "lives in", "worker visa", timestamp=200.0))
    mem.absorb(Candidate("Ada", "likes", "jazz", timestamp=400.0))
    mem.absorb(Candidate(
        "Ada", "likes", "jazz with live improvisation", timestamp=450.0))

    before = _checkpoint_snapshot(mem)
    before_lives = _rows(mem, "lives in", active_only=False)
    before_as_of = _rows(mem, "lives in", as_of=250.0)
    assert len(before_lives) == 3

    path = tmp_path / "corrected.npz"
    mem.save_checkpoint(str(path))
    restored = _reload(mem, path)

    assert _checkpoint_snapshot(restored) == before
    assert rows_equal(
        _without_slot_id(_rows(restored, "lives in", active_only=False)),
        _without_slot_id(before_lives))
    assert rows_equal(
        _without_slot_id(_rows(restored, "lives in", as_of=250.0)),
        _without_slot_id(before_as_of))
    assert _active_rows(restored, "lives in")[0]["text"] == "Ada lives in Oslo."
    assert _active_rows(restored, "likes")[0]["text"] == (
        "Ada likes jazz with live improvisation.")
    assert restored.temporal_history_violations() == []


def test_reload_preserves_temporal_links_and_rich_text(tmp_path):
    verifier = DirectionalVerifier([
        ("paris", "rome", "CONTRADICTION", 0.95, 0.94),
        ("jazz", "live improvisation", "SAME", 0.96, 0.94),
    ])
    mem = _mem(verifier)
    paris = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    rome = mem.absorb(Candidate("Ada", "lives in", "Rome", timestamp=200.0))
    mem.absorb(Candidate("Ada", "likes", "jazz", timestamp=300.0))
    mem.absorb(Candidate(
        "Ada", "likes", "jazz with live improvisation", timestamp=350.0))

    path = tmp_path / "links.npz"
    mem.save_checkpoint(str(path))
    restored = _reload(mem, path)

    paris_id = _memory_id(mem, paris)
    rome_id = _memory_id(mem, rome)
    paris_slot = _slot(restored, paris_id)
    rome_slot = _slot(restored, rome_id)
    assert paris_slot.superseded_by == id(rome_slot)
    assert rome_slot.supersedes == id(paris_slot)
    assert paris_slot.is_active is False
    assert rome_slot.is_active is True
    assert _text(restored, paris_slot) == "Ada lives in Paris."


def test_privacy_tombstone_survives_consolidation_and_checkpoint(tmp_path):
    verifier = DirectionalVerifier([
        ("paris", "paris", "SAME", 0.99, 0.99),
    ])
    mem = _mem(verifier)
    secret = mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=100.0))
    secret_id = _memory_id(mem, secret)

    assert mem.redact_memory(secret_id) is True
    assert _slot(mem, secret_id).redacted is True
    assert mem.recall("Ada", predicate="lives in") == []

    # An equivalent observation after a tombstone may create a new fact, but
    # it must never restore content into the tombstoned slot.
    mem.absorb(Candidate("Ada", "lives in", "Paris", timestamp=200.0))
    tombstone = _slot(mem, secret_id)
    assert tombstone.redacted is True
    assert (tombstone.text or "") == ""
    assert tombstone.latest_object is None
    assert tombstone.debug_subject in (None, "")

    path = tmp_path / "privacy.npz"
    mem.save_checkpoint(str(path))
    restored = _reload(mem, path)

    restored_tombstone = _slot(restored, secret_id)
    assert restored_tombstone.redacted is True
    assert (restored_tombstone.text or "") == ""
    assert restored_tombstone.latest_object is None
    active = restored.recall("Ada", predicate="lives in")
    assert all(r["memory_id"] != secret_id for r in active)
    assert all("Paris" not in (r["text"] or "") or r["memory_id"] != secret_id
               for r in active)
