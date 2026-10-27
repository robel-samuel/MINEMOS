"""
Regression tests for Phase 28 false-contradiction handling.

The Phase 28 cumulative-predicate whitelist covers the benchmark's
"published" case, but the discriminator still treats any unlisted
non-generic predicate as an exclusive state. NLI then misclassifies
independent facts sharing the same verb as contradictions, deactivating
the earlier fact.
"""

import pytest

from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate


@pytest.mark.parametrize(
    ("predicate", "first_object", "second_object"),
    [
        ("leads", "the storage infrastructure team", "the security engineering team"),
        ("manages", "project Orion", "project Vega"),
        ("reviewed", "paper on distributed consensus", "paper on memory safety"),
    ],
)
def test_independent_facts_with_unlisted_multivalued_predicate_are_retained(
    predicate, first_object, second_object
):
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Nathan", predicate, first_object), timestamp=10.0)
    result = mem.absorb(Candidate("Nathan", predicate, second_object), timestamp=20.0)

    assert result["action"] != "temporal_update", (
        "Independent facts sharing a multi-valued predicate must not be superseded"
    )
    active = [s for s in mem.memory.slots if s.is_active]
    assert len(active) == 2
    assert len(mem.recall("Nathan", predicate)) == 2


def test_exclusive_predicate_still_supersedes():
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Alice", "works at", "Google"), timestamp=10.0)
    result = mem.absorb(Candidate("Alice", "works at", "Meta"), timestamp=20.0)

    assert result["action"] == "temporal_update"
    active = [s for s in mem.memory.slots if s.is_active]
    assert len(active) == 1
    assert "Meta" in active[0].text
