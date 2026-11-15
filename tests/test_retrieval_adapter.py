"""Adapter, configuration and checkpoint-boundary tests for optional retrieval.

These tests pin the integration contract:

* canonical retrieval is the default and constructs no adapter;
* optimized/incremental retrieval requires explicit selection;
* switching back to canonical is a working rollback path;
* a backend failure falls back to canonical only when configured to;
* checkpoint files carry no retrieval configuration or state version, and a
  restored optimized engine rebuilds its snapshot from the restored slots.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from minemos_engine.checkpoint import (load_into_black_hole_memory, save_checkpoint)
from minemos_engine.retrieval import (CANONICAL, INCREMENTAL, OPTIMIZED, QDRANT,
                              BACKENDS, RetrievalConfig, normalize_backend)
from minemos_engine.structured_candidate import Candidate
from tests._retrieval_helpers import (QUERY, add_facts, make_memory,
                                      rows_equal)

FACTS = [
    ("Alice", "lives in", "Paris", 10.0),
    ("Bob", "lives in", "Oslo", 10.0),
    ("Carol", "works at", "Acme", 12.0),
    ("Dan", "likes", "Chess", 15.0),
    ("Alice", "lives in", "Rome", 20.0),
]


def _strip_ids(rows):
    # slot_id is process-local; memory_id is a random uuid4 per engine.  Tests
    # that compare independently constructed engines must drop both.  Tests
    # that check checkpoint identity assert memory_id separately.
    return [{k: v for k, v in r.items()
             if k not in ("slot_id", "memory_id")} for r in rows]


def _assert_matches_canonical(mem, backend, **query):
    selected = mem.query(QUERY, **query)
    assert mem.retrieval_backend == backend
    mem.set_retrieval_backend(CANONICAL)
    try:
        canonical = mem.query(QUERY, **query)
    finally:
        mem.set_retrieval_backend(backend)
    assert rows_equal(selected, canonical)


def _warm(backend, **kwargs):
    mem = make_memory(capacity=50, backend=backend, **kwargs)
    add_facts(mem, FACTS)
    return mem


class _Boom:
    """Injected backend that always raises, with a real manager for stats."""

    strategy = "boom"

    def __init__(self, manager):
        self.manager = manager

    def query(self, *args, **kwargs):
        raise RuntimeError("injected backend failure")


# ── default configuration ──────────────────────────────────────────────────


def test_default_backend_is_canonical_and_builds_no_adapter():
    mem = make_memory(capacity=20)
    add_facts(mem, FACTS)
    assert mem.retrieval_backend == CANONICAL
    assert mem._retrieval_adapter_obj is None
    assert len(mem.query(QUERY, top_k=5)) == 5
    assert mem._retrieval_adapter_obj is None
    stats = mem.retrieval_stats()
    assert stats["backend"] == CANONICAL
    assert stats["active"] is False
    assert stats["manager"] is None


@pytest.mark.parametrize("backend,strategy", [
    (OPTIMIZED, "snapshot_full"),
    (INCREMENTAL, "snapshot_incremental"),
])
def test_optimized_requires_explicit_selection(backend, strategy):
    mem = _warm(backend)
    assert mem.retrieval_backend == backend
    results = mem.query(QUERY, top_k=5)
    assert len(results) == 5
    stats = mem.retrieval_stats()
    assert stats["backend"] == backend
    assert stats["active"] is True
    assert stats["strategy"] == strategy
    assert stats["published_version"] == mem._state_version
    _assert_matches_canonical(mem, backend, top_k=5)
    _assert_matches_canonical(mem, backend, top_k=-1)
    _assert_matches_canonical(mem, backend, top_k=100)


def test_invalid_backend_values_are_rejected():
    with pytest.raises(ValueError):
        make_memory(capacity=5, backend="vector-magic")
    mem = make_memory(capacity=5)
    with pytest.raises(ValueError):
        mem.set_retrieval_backend("nope")
    with pytest.raises(TypeError):
        normalize_backend(3)
    assert mem.retrieval_backend == CANONICAL
    assert normalize_backend(None) == CANONICAL
    assert normalize_backend(" OPTIMIZED ") == OPTIMIZED
    assert set(BACKENDS) == {CANONICAL, OPTIMIZED, INCREMENTAL, QDRANT}


def test_runtime_switch_back_to_canonical_is_reversible():
    mem = _warm(OPTIMIZED)
    optimized = mem.query(QUERY, top_k=5)
    assert mem.retrieval_stats()["active"] is True

    assert mem.set_retrieval_backend(CANONICAL) == CANONICAL
    assert mem.retrieval_stats()["active"] is False
    canonical = mem.query(QUERY, top_k=5)
    assert rows_equal(optimized, canonical)

    assert mem.set_retrieval_backend(OPTIMIZED) == OPTIMIZED
    assert rows_equal(mem.query(QUERY, top_k=5), canonical)


def test_config_from_value_variants():
    assert RetrievalConfig.from_value(None).backend == CANONICAL
    assert RetrievalConfig.from_value("incremental").backend == INCREMENTAL
    assert RetrievalConfig.from_value(
        {"backend": "optimized", "max_optimistic_attempts": 5}
    ).max_optimistic_attempts == 5
    explicit = RetrievalConfig(backend=OPTIMIZED)
    assert RetrievalConfig.from_value(explicit) is explicit
    with pytest.raises(TypeError):
        RetrievalConfig.from_value(42)
    with pytest.raises(ValueError):
        RetrievalConfig(backend="unknown")
    with pytest.raises(ValueError):
        RetrievalConfig(max_optimistic_attempts=0)
    with pytest.raises(ValueError):
        RetrievalConfig(contended_fallback="sleep")
    assert RetrievalConfig().contended_fallback == "canonical"
    assert RetrievalConfig(
        contended_fallback="locked_build").contended_fallback == "locked_build"


# ── failure routing ────────────────────────────────────────────────────────


def test_backend_failure_falls_back_to_canonical():
    mem = _warm(OPTIMIZED)
    mem.query(QUERY, top_k=5)
    adapter = mem._get_retrieval_adapter()
    good = adapter.retriever
    assert good is not None

    adapter._retriever = _Boom(good.manager)
    results = mem.query(QUERY, top_k=5)
    assert rows_equal(results, _canonical(mem))
    stats = mem.retrieval_stats()
    assert stats["errors"] == 1
    assert "injected backend failure" in stats["last_error"]


def test_backend_failure_propagates_when_fallback_disabled():
    mem = make_memory(
        capacity=50, backend=OPTIMIZED,
        retrieval_config=RetrievalConfig(
            backend=OPTIMIZED, fallback_to_canonical=False))
    add_facts(mem, FACTS)
    adapter = mem._get_retrieval_adapter()
    adapter._retriever = _Boom(adapter.retriever.manager if adapter.retriever
                               else None)
    with pytest.raises(RuntimeError):
        mem.query(QUERY, top_k=5)


def _canonical(mem):
    backend = mem.retrieval_backend
    mem.set_retrieval_backend(CANONICAL)
    try:
        return mem.query(QUERY, top_k=5)
    finally:
        mem.set_retrieval_backend(backend)


# ── contention hand-off ────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", [OPTIMIZED, INCREMENTAL])
def test_exhausted_retries_hand_off_to_canonical_without_an_error(backend):
    mem = _warm(backend, retrieval_config=RetrievalConfig(
        backend=backend, max_optimistic_attempts=1,
        contended_fallback="canonical"))
    mem.query(QUERY, top_k=5)
    manager = mem._get_retrieval_adapter().retriever.manager
    mutated = []

    def hook(attempt, built):
        if not mutated:
            mutated.append(built.version)
            mem.absorb(Candidate("Zed", "likes", "Go", timestamp=30.0))

    manager.hooks["before_publish"] = hook
    manager.invalidate()
    results = mem.query(QUERY, top_k=100)

    assert mutated, "contention was never forced"
    stats = mem.retrieval_stats()
    assert stats["errors"] == 0
    assert stats["manager"]["fallbacks"] == 1
    assert stats["manager"]["contended_fallbacks"] == 1
    assert stats["manager"]["locked_builds"] == 0
    mem.set_retrieval_backend(CANONICAL)
    assert rows_equal(results, mem.query(QUERY, top_k=100))


# ── checkpoint boundary ────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", [OPTIMIZED, INCREMENTAL])
def test_checkpoint_does_not_serialize_retrieval_state(tmp_path, backend):
    mem = _warm(backend)
    mem.query(QUERY, top_k=5)
    path = os.path.join(str(tmp_path), "state.npz")
    save_checkpoint(mem, path)

    with np.load(path, allow_pickle=False) as data:
        keys = list(data.keys())
    assert not [k for k in keys if "retrieval" in k.lower()]
    assert "state_version" not in keys
    assert "_state_version" not in keys
    assert "backend" not in keys

    expected = _canonical(mem)

    restored = make_memory(capacity=50)
    load_into_black_hole_memory(restored, path)
    assert restored.retrieval_backend == CANONICAL
    assert restored._state_version == 1
    assert ([slot.memory_id for slot in restored.memory.slots]
            == [slot.memory_id for slot in mem.memory.slots])
    assert rows_equal(_strip_ids(restored.query(QUERY, top_k=5)),
                      _strip_ids(expected))


@pytest.mark.parametrize("backend", [OPTIMIZED, INCREMENTAL])
def test_restored_optimized_engine_rebuilds_and_matches_canonical(
        tmp_path, backend):
    source = _warm(backend)
    source.query(QUERY, top_k=5)
    path = os.path.join(str(tmp_path), "state.npz")
    save_checkpoint(source, path)

    restored = make_memory(capacity=50, backend=backend)
    load_into_black_hole_memory(restored, path)
    assert restored.retrieval_backend == backend
    for top_k in (5, 100, -1):
        _assert_matches_canonical(restored, backend, top_k=top_k)
    restored.query(QUERY, top_k=5)
    stats = restored.retrieval_stats()
    assert stats["published_version"] == restored._state_version
    assert stats["slots_in_snapshot"] == len(restored.memory.slots)


# ── lifecycle integration ──────────────────────────────────────────────────


@pytest.mark.parametrize("backend", [OPTIMIZED, INCREMENTAL])
def test_mutations_and_removals_advance_the_snapshot(backend):
    mem = _warm(backend)
    mem.query(QUERY, top_k=5)
    version = mem._state_version

    add_facts(mem, [("Eve", "likes", "Go", 30.0)])
    assert mem._state_version > version
    _assert_matches_canonical(mem, backend, top_k=100)

    removed = mem.memory.slots[-1]
    assert mem._remove_slot_identity(removed) is True
    _assert_matches_canonical(mem, backend, top_k=100)
    assert all(r["slot_id"] != id(removed) for r in mem.query(QUERY, top_k=100))


@pytest.mark.parametrize("backend", [OPTIMIZED, INCREMENTAL])
def test_force_full_rebuild_drops_the_published_snapshot(backend):
    mem = _warm(backend)
    mem.query(QUERY, top_k=5)
    adapter = mem._get_retrieval_adapter()
    before = adapter.stats()["manager"]["builds"]
    adapter.force_full_rebuild()
    assert adapter.retriever.manager.published is None

    reference = make_memory(capacity=50)
    add_facts(reference, FACTS)
    assert rows_equal(_strip_ids(mem.query(QUERY, top_k=5)),
                      _strip_ids(reference.query(QUERY, top_k=5)))
    assert adapter.stats()["manager"]["builds"] == before + 1
