"""
Reproduces the design doc's first success tests:

  Section 45 — FIRST PROTOTYPE
    Absorb a handful of synthetic facts, destroy the original input,
    confirm recall still works.

  Section 46 — FIRST SUCCESS TEST (scaled down)
    Also checks reinforcement (repeated facts raise confidence) and
    contradiction handling (changed facts create a new active fact while
    preserving history).

Run with:  python -m pytest tests/test_basic.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.memory import MemoryStore
from core.encoder import RuleBasedExtractor


def build_store_from_sentences(sentences):
    store = MemoryStore()
    extractor = RuleBasedExtractor()
    for s in sentences:
        triple = extractor.extract(s)
        assert triple is not None, f"Extractor failed on: {s!r}"
        store.absorb(triple.subject, triple.predicate, triple.object)
    return store


def test_section_45_basic_recall():
    sentences = [
        "Person 1 likes apples.",
        "Person 2 lives in Paris.",
        "Person 3 uses Python.",
        "Person 4 owns a Tesla.",
    ]
    store = build_store_from_sentences(sentences)
    # "Destroy the original input" -> sentences list goes out of scope;
    # everything below only touches the memory state.

    result = store.recall("person_3", "uses")
    assert len(result) == 1
    assert result[0].object == "Python"

    result = store.recall("person_2", "lives_in")
    assert result[0].object == "Paris"


def test_reinforcement_raises_confidence():
    store = MemoryStore()
    store.absorb("user", "uses", "Python")
    first_conf = store.recall("user", "uses")[0].confidence

    for _ in range(4):
        store.absorb("user", "uses", "Python")

    final = store.recall("user", "uses")[0]
    assert final.confidence > first_conf
    assert final.source_count == 5


def test_contradiction_creates_new_fact_preserves_history():
    store = MemoryStore()
    store.absorb("user", "prefers", "Python")
    store.absorb("user", "prefers", "Python")  # reinforce once

    store.absorb("user", "prefers", "Rust")  # contradiction

    active = store.recall("user", "prefers")
    assert len(active) == 1
    assert active[0].object == "Rust"

    full_history = store.recall("user", "prefers", include_history=True)
    assert len(full_history) == 2
    objects = {f.object for f in full_history}
    assert objects == {"Python", "Rust"}

    # old fact should be closed (has a valid_until), new one open
    old = [f for f in full_history if f.object == "Python"][0]
    assert old.valid_until is not None
    new = [f for f in full_history if f.object == "Rust"][0]
    assert new.valid_until is None


def test_checkpoint_restore_roundtrip(tmp_path):
    store = MemoryStore()
    store.absorb("person_3", "uses", "Python")
    store.absorb("person_4", "owns", "a Tesla")

    ckpt_path = tmp_path / "checkpoint.json"
    store.checkpoint(str(ckpt_path))

    restored = MemoryStore.restore(str(ckpt_path))
    result = restored.recall("person_3", "uses")
    assert result[0].object == "Python"
    assert restored.stats()["active_facts"] == 2


def test_consolidate_promotes_stable_facts():
    store = MemoryStore()
    for _ in range(4):
        store.absorb("user", "prefers", "Python")

    promoted = store.consolidate(min_source_count=3)
    assert len(promoted) == 1
    fact = store.recall("user", "prefers")[0]
    assert fact.confidence >= 0.9


def test_predicate_interning_shares_ids_across_subjects():
    """Multiple subjects using the same predicate should share one
    predicate-table entry rather than each getting their own copy."""
    store = MemoryStore()
    store.absorb("person_1", "uses", "Python")
    store.absorb("person_2", "uses", "Rust")
    store.absorb("person_3", "uses", "Go")

    assert len(store._id_to_pred) == 1
    assert store._pred_to_id["uses"] == 0

    assert store.recall("person_1", "uses")[0].object == "Python"
    assert store.recall("person_2", "uses")[0].object == "Rust"
    assert store.recall("person_3", "uses")[0].object == "Go"


def test_predicate_interning_multiple_distinct_predicates():
    store = MemoryStore()
    store.absorb("person_1", "uses", "Python")
    store.absorb("person_1", "lives_in", "Paris")
    store.absorb("person_2", "owns", "a Tesla")

    assert len(store._id_to_pred) == 3
    assert store.recall("person_1", "lives_in")[0].object == "Paris"
    assert store.recall("person_2", "owns")[0].object == "a Tesla"


def test_recall_unknown_predicate_returns_empty_not_error():
    store = MemoryStore()
    store.absorb("person_1", "uses", "Python")
    # "owns" was never absorbed by anyone -- must not raise, must return []
    assert store.recall("person_1", "owns") == []
    assert store.recall("nonexistent_person", "uses") == []


def test_predicate_interned_checkpoint_restore_roundtrip(tmp_path):
    store = MemoryStore()
    store.absorb("person_1", "uses", "Python")
    store.absorb("person_2", "uses", "Rust")
    store.absorb("person_1", "lives_in", "Paris")

    ckpt_path = tmp_path / "interned_checkpoint.json"
    store.checkpoint(str(ckpt_path))

    restored = MemoryStore.restore(str(ckpt_path))

    # predicate table itself must round-trip correctly
    assert restored._id_to_pred == store._id_to_pred
    assert restored._pred_to_id == store._pred_to_id

    # existing .object / .confidence access must keep working after reload
    assert restored.recall("person_1", "uses")[0].object == "Python"
    assert restored.recall("person_2", "uses")[0].object == "Rust"
    assert restored.recall("person_1", "lives_in")[0].object == "Paris"
    assert restored.recall("person_1", "uses")[0].confidence == \
        store.recall("person_1", "uses")[0].confidence


def test_no_accuracy_or_false_memory_regression_after_interning():
    """Same shape as the scale benchmark's correctness check, run small
    and inline so it's part of the fast test suite."""
    store = MemoryStore()
    ground_truth = {}
    for i in range(50):
        predicate = ["uses", "lives_in", "owns"][i % 3]
        obj = f"value_{i}"
        store.absorb(f"person_{i}", predicate, obj)
        ground_truth[(f"person_{i}", predicate)] = obj

    correct = 0
    for (subject, predicate), expected in ground_truth.items():
        result = store.recall(subject, predicate)
        if result and result[0].object == expected:
            correct += 1
    assert correct == len(ground_truth)  # accuracy == 1.0, unchanged

    false_hits = 0
    for i in range(20):
        result = store.recall(f"ghost_{i}", "uses")
        if result:
            false_hits += 1
    assert false_hits == 0  # false-memory rate == 0.0, unchanged


if __name__ == "__main__":
    # allow running without pytest too
    test_section_45_basic_recall()
    test_reinforcement_raises_confidence()
    test_contradiction_creates_new_fact_preserves_history()
    test_consolidate_promotes_stable_facts()
    print("All tests passed (manual run, checkpoint/restore test needs pytest tmp_path).")
