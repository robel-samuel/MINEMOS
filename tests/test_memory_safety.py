"""Regression tests for the 2026-11-05 memory-safety remediation.

Covers every false-merge scenario and every unsupported-query case identified
by the natural-language robustness audit, plus the behaviours that must be
preserved: single-token typo correction, abbreviations, paraphrase
consolidation, corrections and historical retrieval.

The deterministic verifier below deliberately answers SAME for every pair so
the tests exercise the engine's own identity safeguards rather than the
real NLI model's current opinion.  The audit re-run measures the real models.
"""

from __future__ import annotations

import json

import pytest

from tests.support.nl_audit.dataset import load_jsonl
from tests.support.nl_audit.harness import CASES_PATH, LIFECYCLE_PATH
from tests.support.nl_audit.stubs import (
    EqualityVerifier,
    HashEncoder,
    StubGate,
)
from minemos_engine.black_hole_memory import BlackHoleMemory
from minemos_engine.structured_candidate import Candidate

CASES = load_jsonl(CASES_PATH)
SCENARIOS = load_jsonl(LIFECYCLE_PATH)
UNSUPPORTED = [case for case in CASES if case["kind"] == "unsupported"]
SUPPORTED = [case for case in CASES if case["kind"] != "unsupported"]
FALSE_MERGE_SCENARIOS = [
    scenario for scenario in SCENARIOS
    if scenario["category"] in ("similar_wording_distinct", "near_miss_entity")
]


class _SameVerifier:
    """Adversarial stand-in for the NLI gate: every pair looks equivalent."""

    def verify_batch(self, pairs, **kwargs):
        return [
            {
                "decision": "SAME",
                "confidence": 0.99,
                "p_entail": 0.99,
                "p_contra": 0.01,
                "p_neutral": 0.01,
                "nli_invocations": 1,
            }
            for _ in pairs
        ]


def _memory(strict: bool = False, verifier=None) -> BlackHoleMemory:
    return BlackHoleMemory(
        capacity=64,
        dim=384,
        dense_encoder=HashEncoder(),
        nli_gate=StubGate(),
        verifier=verifier or _SameVerifier(),
        strict_entity_identity=strict,
    )


def _absorb(mem: BlackHoleMemory, subject: str, predicate: str,
            obj: str, timestamp: float):
    return mem.absorb(
        Candidate(subject, predicate, obj, timestamp=timestamp),
        timestamp=timestamp,
    )


def _active(mem: BlackHoleMemory):
    return [slot for slot in mem.memory.slots if slot.is_active]


def _scenario_values(scenario, noisy: bool):
    values = []
    for observation in scenario["observations"]:
        if noisy:
            values.append((
                observation.get("noisy_subject", observation["subject"]),
                observation.get("noisy_predicate", observation["predicate"]),
                observation.get("noisy_object", observation["object"]),
                float(observation["timestamp"]),
            ))
        else:
            values.append((
                observation["subject"],
                observation["predicate"],
                observation["object"],
                float(observation["timestamp"]),
            ))
    return values


# ---------------------------------------------------------------------------
# A. Consolidation safety — every false merge from the audit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario", FALSE_MERGE_SCENARIOS,
    ids=[scenario["scenario_id"] for scenario in FALSE_MERGE_SCENARIOS])
@pytest.mark.parametrize("noisy", [False, True], ids=["clean", "noisy"])
def test_strict_identity_keeps_distinct_facts_separate(scenario, noisy):
    mem = _memory(strict=True)
    results = [
        _absorb(mem, *values)
        for values in _scenario_values(scenario, noisy)
    ]

    active = _active(mem)
    assert len(active) == scenario["expected_active_count"] == 2
    assert [result["action"] for result in results] == ["insert", "insert"]
    texts = " ".join(
        mem.slot_texts.get(id(slot), slot.text) for slot in active)
    for fact in scenario["distinct_facts"]:
        assert fact["subject"].lower() in texts.lower()
        assert fact["object"].lower() in texts.lower()


# Similar-wording scenarios whose two values are surface variants of one
# object (a plural or a typo inflection).  The frozen lifecycle dataset labels
# them "similar_wording_distinct" for strict mode, but in the default profile
# a verifier SAME may still consolidate them.
SURFACE_VARIANT_SCENARIOS = frozenset(
    {"L073", "L074", "L075", "L078", "L080"})


@pytest.mark.parametrize(
    "scenario", FALSE_MERGE_SCENARIOS,
    ids=[scenario["scenario_id"] for scenario in FALSE_MERGE_SCENARIOS])
def test_default_mode_never_merges_incompatible_values_on_verifier_same(
        scenario):
    """R5 F4: a verifier SAME does not erase distinct represented values.

    The pre-R5 default profile merged every pair this adversarial verifier
    called equivalent, including "the Roman Empire" / "the Roman Republic".
    The lexical compatibility gate now keeps genuinely different objects
    apart while still allowing surface variants (plurals, typos) to
    consolidate.
    """
    values = _scenario_values(scenario, noisy=False)
    mem = _memory(strict=False)
    for value in values:
        _absorb(mem, *value)

    merged = len(_active(mem)) == 1
    if scenario["category"] == "similar_wording_distinct":
        assert merged == (
            scenario["scenario_id"] in SURFACE_VARIANT_SCENARIOS)
        if not merged:
            texts = {
                mem.slot_texts.get(id(slot), slot.text)
                for slot in _active(mem)
            }
            assert len(texts) == 2
    else:
        # Substitution near-misses were already kept apart by the narrow
        # one-edit rule; insertions/deletions/transpositions were not.
        assert merged == (scenario["subcategory"] != "substitution")


def test_strict_identity_telemetry_records_refusals():
    mem = _memory(strict=True)
    _absorb(mem, "Darius Vance", "loves", "the Roman Republic", 1.0)
    _absorb(mem, "Darius Vance", "loves", "the Roman Empire", 2.0)
    status = mem.status()
    assert status["strict_entity_identity"] is True
    assert status["total_suppressed_value_merges"] >= 1


# ---------------------------------------------------------------------------
# Preservation: typo correction, abbreviations, paraphrase, temporal history
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subject_a,subject_b,value_a,value_b",
    [
        ("Alice", "Alic", "Nairobi", "Narobi"),
        ("Addis Ababa", "Addis Ababa", "Nairobi", "Narobi"),
        ("New York City", "NYC", "New York City", "NYC"),
    ],
)
def test_single_token_typos_and_abbreviations_still_consolidate(
        subject_a, subject_b, value_a, value_b):
    mem = _memory(strict=True)
    _absorb(mem, subject_a, "lives in", value_a, 1.0)
    result = _absorb(mem, subject_b, "lives in", value_b, 2.0)

    assert result["action"] == "consolidate"
    assert len(_active(mem)) == 1
    assert result["surface_variant"] is True


def test_paraphrase_consolidates_in_strict_mode():
    mem = _memory(strict=True)
    _absorb(mem, "Eldridge Ormsby", "lives in", "Recife", 1.0)
    result = _absorb(mem, "Eldridge Ormsby", "resides in", "Recife", 2.0)

    assert result["action"] == "consolidate"
    assert len(_active(mem)) == 1


def test_exclusive_correction_remains_a_temporal_update():
    mem = _memory(strict=True)
    first = _absorb(mem, "Darius Vance", "lives in", "Oslo", 1.0)
    result = _absorb(mem, "Darius Vance", "lives in", "Bergen", 2.0)

    assert result["action"] == "temporal_update"
    active = _active(mem)
    assert len(active) == 1
    assert "Bergen" in mem.slot_texts[id(active[0])]
    old = next(
        slot for slot in mem.memory.slots if id(slot) == first["slot_id"])
    assert old.is_active is False
    assert old.valid_until == 2.0
    assert old.superseded_by == id(active[0])
    assert active[0].supersedes == id(old)


def test_historical_statement_stays_closed_in_strict_mode():
    mem = _memory(strict=True)
    old = _absorb(
        mem, "Eldridge Ormsby", "used to live in", "Lima", 1.0)
    current = _absorb(
        mem, "Eldridge Ormsby", "lives in", "Recife", 2.0)

    old_slot = next(
        slot for slot in mem.memory.slots if id(slot) == old["slot_id"])
    current_slot = next(
        slot for slot in mem.memory.slots
        if id(slot) == current["slot_id"])
    assert old_slot.is_active is False
    assert current_slot.is_active is True
    assert len(_active(mem)) == 1


# ---------------------------------------------------------------------------
# B. Retrieval abstinence — every unsupported-query audit case
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unsupported_memory():
    mem = BlackHoleMemory(
        capacity=64,
        dim=384,
        dense_encoder=HashEncoder(),
        nli_gate=StubGate(),
        verifier=EqualityVerifier(),
    )
    for case in UNSUPPORTED:
        observation = case["observations"][0]
        _absorb(
            mem, observation["subject"], observation["predicate"],
            observation["object"], float(observation["timestamp"]))
    yield mem
    mem.close()


@pytest.mark.parametrize(
    "case", UNSUPPORTED, ids=[case["case_id"] for case in UNSUPPORTED])
@pytest.mark.parametrize("query_key", ["clean_query", "noisy_query"])
def test_unsupported_queries_abstain(case, query_key, unsupported_memory):
    outcome = unsupported_memory.eligible_query(case[query_key])

    assert outcome["status"] == "abstained"
    assert outcome["rows"] == []
    expected_reason = (
        "attribute_not_stored"
        if case["subcategory"] == "known_subject"
        else "subject_not_found")
    assert outcome["reason"] == expected_reason
    assert outcome["focus"]


def test_shared_first_name_is_ambiguous_not_guessed():
    mem = _memory(strict=True)
    _absorb(mem, "Rowan Dunmore", "is allergic to", "peanuts", 1.0)
    _absorb(mem, "Rowan Ainsley", "is allergic to", "shellfish", 1.0)

    ambiguous = mem.eligible_query("Which allergen affects Rowan")
    assert ambiguous["status"] == "abstained"
    assert ambiguous["reason"] == "subject_ambiguous"
    assert ambiguous["rows"] == []

    full = mem.eligible_query("Which allergen affects Rowan Ainsley")
    assert full["status"] == "answered"
    assert full["rows"][0]["subject"] == "Rowan Ainsley"
    assert "shellfish" in full["rows"][0]["text"]
    mem.close()


def test_absent_memory_and_low_confidence_are_different_reasons():
    mem = _memory(strict=True)
    _absorb(mem, "Lucian Dunmore", "lives in", "Lisbon", 1.0)

    absent = mem.eligible_query("What is Nobody Here's salary?")
    assert absent["status"] == "abstained"
    assert absent["reason"] == "subject_not_found"

    answered = mem.eligible_query("Where does Lucian Dunmore live?")
    assert answered["status"] == "answered"
    assert answered["rows"]
    assert answered["threshold"] is None

    gated = mem.eligible_query(
        "Where does Lucian Dunmore live?",
        min_similarity=min(1.0, answered["top_similarity"] + 0.01))
    assert gated["status"] == "abstained"
    assert gated["reason"] == "low_similarity"
    assert gated["rows"] == []
    assert gated["top_similarity"] == answered["top_similarity"]


def test_indirect_query_is_unverified_not_absent():
    mem = _memory(strict=True)
    _absorb(mem, "Lucian Dunmore", "lives in", "Lisbon", 1.0)

    outcome = mem.eligible_query("Which city has the best weather?")
    assert outcome["status"] == "unverified"
    assert outcome["reason"] == "subject_not_identified"
    assert outcome["focus"] is None

    # A query that does name an entity is an absence, not an ambiguity: the
    # gate still refuses to substitute an unrelated stored person.
    named = mem.eligible_query("What is the capital of France?")
    assert named["status"] == "abstained"
    assert named["reason"] == "subject_not_found"
    assert named["rows"] == []

    # A pronoun-only follow-up identifies no entity either.
    pronoun = mem.eligible_query("What is his favorite color?")
    assert pronoun["status"] == "unverified"
    assert pronoun["focus"] is None


def test_polite_framing_does_not_hide_the_subject():
    mem = _memory(strict=True)
    _absorb(mem, "Thaddeus Kingsley", "favorite film is", "Arrival", 1.0)
    _absorb(mem, "Keira Northcott", "favorite film is", "Heat", 1.0)

    outcome = mem.eligible_query(
        "Can you tell me which film does Thaddeus Kingsley like best?")
    assert outcome["status"] == "answered"
    assert outcome["focus"] == "thaddeus kingsley"
    assert outcome["rows"][0]["subject"] == "Thaddeus Kingsley"

    unknown = mem.eligible_query(
        "Could you tell me which film does Nobody Here like best?")
    assert unknown["status"] == "abstained"
    assert unknown["reason"] == "subject_not_found"
    mem.close()


def test_named_context_followup_resolves_the_named_entity():
    mem = _memory(strict=True)
    _absorb(mem, "Vera Ravenscroft", "studies", "materials fatigue", 1.0)
    _absorb(mem, "Vera Ravenscroft", "favorite color is", "amber", 2.0)
    _absorb(mem, "Eldridge Zeller", "studies", "fluid turbulence", 1.0)

    outcome = mem.eligible_query(
        "Earlier we discussed Vera Ravenscroft, who studies materials "
        "fatigue. What is her favorite color?")
    assert outcome["status"] == "answered"
    assert outcome["focus"] == "vera ravenscroft"
    assert outcome["rows"][0]["subject"] == "Vera Ravenscroft"
    assert any("amber" in row["text"] for row in outcome["rows"])
    mem.close()


def test_repeated_possessive_keeps_the_full_name():
    mem = _memory(strict=True)
    _absorb(mem, "Clara Everhart", "favorite color is", "teal", 1.0)
    _absorb(mem, "Rowan Everhart", "favorite color is", "rust", 1.0)

    # A surname-only focus would be ambiguous between the two Everharts; the
    # full possessive subject must be kept.
    outcome = mem.eligible_query(
        "What is Clara Everhart's Everhart's favorite color?")
    assert outcome["status"] == "answered"
    assert outcome["focus"] == "clara everhart"
    assert outcome["rows"][0]["subject"] == "Clara Everhart"
    assert "teal" in outcome["rows"][0]["text"]
    mem.close()


@pytest.fixture(scope="module")
def supported_memory():
    mem = BlackHoleMemory(
        capacity=2 * len(SUPPORTED) + 64,
        dim=384,
        dense_encoder=HashEncoder(),
        nli_gate=StubGate(),
        verifier=EqualityVerifier(),
        strict_entity_identity=True,
    )
    for case in SUPPORTED:
        for observation in case["observations"]:
            _absorb(
                mem, observation["subject"], observation["predicate"],
                observation["object"], float(observation["timestamp"]))
    yield mem
    mem.close()


def test_supported_clean_queries_still_answer_with_the_right_entity(
        supported_memory):
    for case in SUPPORTED:
        outcome = supported_memory.eligible_query(case["clean_query"])
        assert outcome["status"] == "answered", case["case_id"]
        assert outcome["rows"], case["case_id"]
        assert outcome["rows"][0]["subject"] == case["expected"]["subject"], (
            case["case_id"], outcome["rows"][0]["text"])
        # The gate filters by entity; dense ranking within one entity's
        # facts is measured by the audit, not asserted with a stub encoder.
        assert any(case["expected"]["object"] in row["text"]
                   for row in outcome["rows"]), (
            case["case_id"], [row["text"] for row in outcome["rows"]])


def test_supported_noisy_queries_never_answer_with_another_entity(
        supported_memory):
    for case in SUPPORTED:
        outcome = supported_memory.eligible_query(case["noisy_query"])
        if outcome["status"] != "answered":
            assert outcome["status"] in ("abstained", "unverified")
            continue
        assert outcome["rows"]
        assert outcome["rows"][0]["subject"] == case["expected"]["subject"], (
            case["case_id"], case["noisy_query"], outcome["rows"][0]["text"])


# ---------------------------------------------------------------------------
# Harness wiring
# ---------------------------------------------------------------------------


def test_audit_safety_profile_is_opt_in():
    from tests.support.nl_audit.harness import (
        ELIGIBILITY_METHOD,
        run_audit,
    )

    result = run_audit(
        limit=1,
        lifecycle_limit=2,
        stub=True,
        skip_qdrant=True,
        include_hybrid=False,
        raw_limit=0,
        run_raw_probe=False,
        safety=True,
    )
    summary = result["summary"]
    assert summary["config"]["safety_profile"]["enabled"] is True
    assert summary["config"]["safety_profile"]["min_similarity"] is None
    assert ELIGIBILITY_METHOD in summary["config"]["methods"]
    assert ELIGIBILITY_METHOD in summary["retrieval"]["summary"]["by_method"]
    assert summary["lifecycle"]["strict_entity_identity"] is True
    lifecycle_records = [
        record for record in result["records"]
        if record["record_type"] == "lifecycle"]
    assert lifecycle_records
    assert all(record["strict_entity_identity"] for record in lifecycle_records)

    # An abstention is not a retrieval hit: only answered outcomes may be
    # scored, otherwise the gate's recall would count rows it refused to use.
    for record in result["records"]:
        if (record["record_type"] == "retrieval"
                and record["method"] == ELIGIBILITY_METHOD
                and record["eligibility_status"] != "answered"):
            assert record["rank"] == 0
            assert record["recall@1"] == 0.0
            assert record["top5"] == []
