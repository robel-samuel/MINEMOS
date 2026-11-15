"""Fast regression tests for identity-preserving duplicate classification.

Every scenario states what the engine must decide: consolidate a genuine
duplicate, revise a temporal state, or preserve two memories and report an
unresolved near-duplicate.  The deterministic stubs isolate engine decision
logic from model semantics.
"""

from __future__ import annotations

import pytest

from tests.support.nl_audit.duplicates import (
    generate_duplicate_scenarios,
    run_duplicate_benchmark,
)
from tests.support.nl_audit.harness import build_components
from minemos_engine.black_hole_memory import BlackHoleMemory
from minemos_engine.structured_candidate import Candidate

STUB_GROUPS = ("genuine_duplicate", "ambiguous_restatement", "distinct_fact")
# D007-D011 document known limitations that the equality stub cannot reach
# (structure reordering, added detail, articles, informal verbs, noisy
# repetition).  The benchmark reports them; the fast contract tests below do
# not assert on model semantics.
MODEL_DEPENDENT = {"D007", "D008", "D009", "D010", "D011"}
STUB_SCENARIOS = [
    scenario for scenario in generate_duplicate_scenarios()
    if scenario["group"] in STUB_GROUPS
    and scenario["scenario_id"] not in MODEL_DEPENDENT
]

_COMPONENTS = {}


def _components():
    if not _COMPONENTS:
        _COMPONENTS.update(build_components(stub=True))
    return _COMPONENTS


def _memory(verifier=None, strict: bool = True) -> BlackHoleMemory:
    components = _components()
    return BlackHoleMemory(
        capacity=20,
        dim=4096,
        dense_encoder=components["encoder"],
        nli_gate=components["gate"],
        verifier=verifier or components["verifier"],
        retrieval_backend="canonical",
        strict_entity_identity=strict,
    )


def _absorb(mem: BlackHoleMemory, subject: str, predicate: str,
            obj: str, timestamp: float):
    return mem.absorb(
        Candidate(subject, predicate, obj, timestamp=timestamp),
        timestamp=timestamp,
    )


def _active(mem: BlackHoleMemory):
    return [slot for slot in mem.memory.slots
            if slot.is_active and not slot.redacted]


class _ContradictionOn:
    """Forces CONTRADICTION for candidates containing a marker."""

    def __init__(self, marker: str):
        self.marker = marker

    def verify_batch(self, pairs, **kwargs):
        results = []
        for _slot_text, cand_text in pairs:
            contradiction = self.marker in cand_text
            decision = "CONTRADICTION" if contradiction else "NEUTRAL"
            results.append({
                "decision": decision,
                "confidence": 0.99,
                "p_entail": 0.01,
                "p_contra": 0.99 if contradiction else 0.01,
                "p_neutral": 0.01 if contradiction else 0.98,
                "fwd": {
                    "p_contra": 0.99 if contradiction else 0.01,
                    "p_neutral": 0.01 if contradiction else 0.98,
                },
                "rev": {
                    "p_contra": 0.99 if contradiction else 0.01,
                    "p_neutral": 0.01 if contradiction else 0.98,
                },
            })
        return results


@pytest.fixture(scope="module")
def stub_results():
    return run_duplicate_benchmark(components=_components())


@pytest.mark.parametrize(
    "scenario", STUB_SCENARIOS, ids=[s["scenario_id"] for s in STUB_SCENARIOS])
def test_declared_expectation_holds_under_stub(stub_results, scenario):
    record = next(
        item for item in stub_results["records"]
        if item["scenario_id"] == scenario["scenario_id"])
    assert record["pass"], record


def test_exact_duplicate_consolidates_and_reports_class():
    mem = _memory()
    first = _absorb(mem, "Mira Solano", "lives in", "Lisbon", 100.0)
    second = _absorb(mem, "  mira   solano ", "Lives In", "lisbon.", 110.0)

    assert first["action"] == "insert"
    assert second["action"] == "consolidate"
    assert second["consolidation_class"] == "duplicate"
    assert second["update_count"] == 2
    assert len(_active(mem)) == 1
    assert mem.status()["total_unresolved_near_duplicates"] == 0
    mem.close()


@pytest.mark.parametrize(
    "subject_a, predicate_a, object_a, subject_b, predicate_b, object_b, field",
    [
        ("Mira Solano", "lives in", "Lisbon",
         "Mira Solnao", "lives in", "Lisbon", "subject"),
        ("Mira Solano", "lives in", "Lisbon",
         "Mira Solano", "lives in", "Lisbom", "object"),
        ("Mira Solano", "drinks", "cranberry juice",
         "Mira Solano", "dranks", "cranberry juice", "predicate"),
        ("Mira Solano", "lives in", "Lisbon",
         "Mira Solano", "lived in", "Lisbon", "predicate"),
    ],
)
def test_ambiguous_restatement_is_preserved_and_flagged(
        subject_a, predicate_a, object_a,
        subject_b, predicate_b, object_b, field):
    mem = _memory()
    first = _absorb(mem, subject_a, predicate_a, object_a, 100.0)
    second = _absorb(mem, subject_b, predicate_b, object_b, 110.0)

    assert second["action"] == "insert"
    assert second["consolidation_class"] == "unresolved_near_duplicate"
    assert second["near_duplicate_field"] == field
    assert second["near_duplicate_of"] == first["memory_id"]
    assert len(_active(mem)) == 2
    assert len(mem.memory.slots) == 2
    assert mem.status()["total_unresolved_near_duplicates"] == 1
    assert first["slot_id"] != second["slot_id"]
    for slot in _active(mem):
        assert slot.update_count == 1
    mem.close()


def test_near_miss_entities_are_never_merged():
    mem = _memory()
    results = [
        _absorb(mem, "Mira Ashford", "lives in", "Toronto", 100.0),
        _absorb(mem, "Mira Ahsford", "lives in", "Toronto", 110.0),
        _absorb(mem, "Rowan Bellamy", "works at", "Foxglove Publishing", 120.0),
        _absorb(mem, "Rowan Bellany", "works at", "Foxglove Publishing", 130.0),
    ]

    assert [r["action"] for r in results] == ["insert"] * 4
    assert len(_active(mem)) == 4
    assert all(
        r["consolidation_class"] == "unresolved_near_duplicate"
        for r in results[1::2])
    assert mem.status()["total_unresolved_near_duplicates"] == 2
    mem.close()


def test_distinct_relations_and_scopes_are_not_flagged():
    mem = _memory()
    results = [
        _absorb(mem, "Priya Thorne", "plays", "the mandolin", 100.0),
        _absorb(mem, "Priya Thorne", "owns", "the mandolin", 110.0),
        _absorb(mem, "conversation:alpha", "knows", "Mira Solano", 120.0),
        _absorb(mem, "conversation:beta", "knows", "Mira Solano", 130.0),
        _absorb(mem, "document:alpha", "mentions", "Mira Solano", 140.0),
        _absorb(mem, "document:beta", "mentions", "Mira Solano", 150.0),
    ]

    assert all(r["action"] == "insert" for r in results)
    assert all(
        r["consolidation_class"] == "distinct_fact" for r in results)
    assert len(_active(mem)) == 6
    assert mem.status()["total_unresolved_near_duplicates"] == 0
    mem.close()


def test_temporal_revision_reports_temporal_class():
    mem = _memory(verifier=_ContradictionOn("Lisbon"))
    first = _absorb(mem, "Mira Solano", "lives in", "Toronto", 100.0)
    second = _absorb(mem, "Mira Solano", "lives in", "Lisbon", 200.0)

    assert second["action"] == "temporal_update"
    assert second["consolidation_class"] == "temporal_update"
    assert second["superseded_slot_id"] == first["slot_id"]
    active = _active(mem)
    assert len(active) == 1
    assert len(mem.memory.slots) == 2
    closed = [s for s in mem.memory.slots if not s.is_active]
    assert len(closed) == 1
    assert closed[0].valid_until == 200.0
    assert mem.status()["total_unresolved_near_duplicates"] == 0
    mem.close()


def test_historical_statement_reports_historical_class():
    mem = _memory()
    _absorb(mem, "Mira Solano", "lives in", "Lisbon", 100.0)
    old = _absorb(mem, "Mira Solano", "used to live in", "Toronto", 200.0)

    assert old["action"] == "historical_observation"
    assert old["consolidation_class"] == "historical_observation"
    assert len(_active(mem)) == 1
    assert len(mem.memory.slots) == 2
    mem.close()


def test_checkpoint_roundtrip_preserves_unresolved_memories(tmp_path):
    path = str(tmp_path / "ambiguous.npz")
    source = _memory()
    _absorb(source, "Mira Solano", "lives in", "Lisbon", 100.0)
    flagged = _absorb(source, "Mira Solnao", "lives in", "Lisbon", 110.0)
    source.save_checkpoint(path, create_backup=False)

    restored = _memory()
    restored.load_checkpoint(path, allow_backup_recovery=False)
    active = _active(restored)
    assert len(active) == 2
    assert all(slot.update_count == 1 for slot in active)
    ids = {slot.memory_id for slot in active}
    assert flagged["near_duplicate_of"] in ids

    repeat = _absorb(restored, "Mira Solano", "lives in", "Lisbon", 120.0)
    assert repeat["action"] == "consolidate"
    assert repeat["consolidation_class"] == "duplicate"
    assert len(_active(restored)) == 2
    source.close()
    restored.close()
