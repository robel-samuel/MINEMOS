"""Deterministic concurrency and snapshot-lifecycle tests.

Every ordering is forced with a lifecycle hook plus ``threading.Event``; no
test sleeps to order events.  The suite covers late mutations, stale builds,
bounded retries with both contention modes, shared snapshots, in-flight
staleness, eviction, checkpoint restore, immutability and writer/reader
convergence.
"""

from __future__ import annotations

import os
import threading

import pytest

from minemos_engine.checkpoint import (load_into_black_hole_memory, save_checkpoint)
from minemos_engine.retrieval import (IncrementalSnapshotRetriever, SnapshotContended,
                              SnapshotRetriever)
from minemos_engine.structured_candidate import Candidate
from tests._retrieval_helpers import (QUERY, add_facts, make_memory,
                                      rows_equal)

BASE_FACTS = [
    ("Alice", "lives in", "Paris", 10.0),
    ("Bob", "lives in", "Oslo", 10.0),
    ("Carol", "works at", "Acme", 12.0),
    ("Dan", "likes", "Chess", 15.0),
    ("Alice", "lives in", "Rome", 20.0),
]
NEW_FACT = ("Zed", "likes", "Go", 30.0)


def _warm(cls=SnapshotRetriever, facts=BASE_FACTS, capacity=100, **kwargs):
    mem = make_memory(capacity=capacity)
    add_facts(mem, facts)
    ret = cls(mem, **kwargs)
    ret.query(QUERY, top_k=5)
    assert ret.manager.published is not None
    return ret, mem


def _canonical_now(mem, **query):
    backend = mem.retrieval_backend
    mem.set_retrieval_backend("canonical")
    try:
        return mem.query(QUERY, **query)
    finally:
        mem.set_retrieval_backend(backend)


def _join(threads, timeout=30):
    for t in threads:
        t.join(timeout=timeout)
    stuck = [t.name for t in threads if t.is_alive()]
    assert not stuck, f"threads did not finish: {stuck}"


def _stall_hook(ret, pause, released, entered, stage="after_capture"):
    """Block the marked thread once at ``stage`` (inside its own query)."""
    local = threading.local()

    def attach():
        local.marker = True

    def hook(attempt, payload):
        if getattr(local, "marker", False) and not pause["done"]:
            pause["done"] = True
            entered.set()
            assert released.wait(10), "test did not release the stalled query"

    ret.manager.hooks[stage] = hook
    return attach


# ── late mutations and stale builds ────────────────────────────────────────


def test_mutation_after_capture_discards_and_rebuilds():
    ret, mem = _warm()
    mutated = []

    def hook(attempt, version):
        if not mutated:
            mutated.append(version)
            mem.absorb(Candidate(*NEW_FACT[:3], timestamp=NEW_FACT[3]))

    ret.manager.hooks["after_capture"] = hook
    ret.manager.invalidate()
    results = ret.query(QUERY, top_k=100)
    assert mutated, "hook never fired"
    assert rows_equal(results, _canonical_now(mem, top_k=100))
    assert ret.manager.stats["discarded"] >= 1
    assert ret.manager.published.version == mem._state_version


def test_stale_build_cannot_replace_newer_snapshot():
    ret, mem = _warm()
    ret.manager.invalidate()
    pause = {"done": False}
    released = threading.Event()
    entered = threading.Event()
    attach = _stall_hook(ret, pause, released, entered)
    results = {}
    errors = []

    def first_query():
        attach()
        try:
            results["a"] = ret.query(QUERY, top_k=100)
        except BaseException as exc:            # pragma: no cover
            errors.append(exc)

    thread = threading.Thread(target=first_query, name="stalled-reader")
    thread.start()
    assert entered.wait(10), "stalled reader never captured"

    mem.absorb(Candidate(*NEW_FACT[:3], timestamp=NEW_FACT[3]))
    second = ret.query(QUERY, top_k=100)
    newer_version = ret.manager.published.version
    assert newer_version == mem._state_version

    released.set()
    _join([thread])
    assert not errors

    # The stalled build captured an older version and must not replace the
    # snapshot published by the second query.
    assert ret.manager.published.version == newer_version
    assert rows_equal(results["a"], _canonical_now(mem, top_k=100))
    assert rows_equal(second, results["a"])


# ── bounded retries and locked fallback ────────────────────────────────────


@pytest.mark.parametrize("cls", [SnapshotRetriever,
                                 IncrementalSnapshotRetriever])
def test_retry_exhaustion_uses_one_locked_fallback(cls):
    ret, mem = _warm(cls=cls, max_optimistic_attempts=1,
                     contended_fallback="locked_build")
    mutated = []

    def hook(attempt, built):
        if not mutated:
            mutated.append(built.version)
            mem.absorb(Candidate(*NEW_FACT[:3], timestamp=NEW_FACT[3]))

    ret.manager.hooks["before_publish"] = hook
    ret.manager.invalidate()
    results = ret.query(QUERY, top_k=100)
    stats = ret.manager.stats
    assert mutated
    assert stats["discarded"] == 1
    assert stats["fallbacks"] == 1
    assert stats["locked_builds"] == 1
    assert rows_equal(results, _canonical_now(mem, top_k=100))
    assert ret.manager.published.version == mem._state_version


@pytest.mark.parametrize("cls", [SnapshotRetriever,
                                 IncrementalSnapshotRetriever])
def test_retry_exhaustion_can_hand_off_without_locking(cls):
    ret, mem = _warm(cls=cls, max_optimistic_attempts=1)
    mutated = []

    def hook(attempt, built):
        if not mutated:
            mutated.append(built.version)
            mem.absorb(Candidate(*NEW_FACT[:3], timestamp=NEW_FACT[3]))

    ret.manager.hooks["before_publish"] = hook
    ret.manager.invalidate()
    with pytest.raises(SnapshotContended):
        ret.query(QUERY, top_k=100)

    stats = ret.manager.stats
    assert mutated
    assert stats["fallbacks"] == 1
    assert stats["contended_fallbacks"] == 1
    assert stats["locked_builds"] == 0

    # The hand-off is not sticky: the next query rebuilds and stays exact.
    assert rows_equal(ret.query(QUERY, top_k=100),
                      _canonical_now(mem, top_k=100))
    assert ret.manager.published.version == mem._state_version


# ── shared immutable snapshots ─────────────────────────────────────────────


def test_concurrent_readers_share_one_snapshot():
    ret, mem = _warm()
    canonical = _canonical_now(mem, top_k=50)
    before = ret.manager.stats["publications"]
    used = {}
    guard = threading.Lock()

    def hook(attempt, snapshot):
        with guard:
            used.setdefault(threading.get_ident(), []).append(snapshot)

    ret.manager.hooks["after_selection"] = hook
    results = {}
    errors = []
    barrier = threading.Barrier(8)

    def reader(i):
        try:
            barrier.wait(10)
            for _ in range(10):
                results[i] = ret.query(QUERY, top_k=50)
        except BaseException as exc:            # pragma: no cover
            errors.append((i, exc))

    threads = [threading.Thread(target=reader, args=(i,), name=f"r{i}")
               for i in range(8)]
    for t in threads:
        t.start()
    _join(threads)

    assert not errors
    assert ret.manager.stats["publications"] == before
    snapshots = {id(s) for records in used.values() for s in records}
    assert len(snapshots) == 1
    for rows in results.values():
        assert rows_equal(rows, canonical)


def test_snapshot_is_immutable_after_publication():
    ret, _mem = _warm()
    snapshot = ret.manager.published
    with pytest.raises(AttributeError):
        snapshot.version = 99
    with pytest.raises(AttributeError):
        del snapshot._fields
    assert snapshot.index.matrix.flags.writeable is False
    assert snapshot.index.valid_from.flags.writeable is False
    assert snapshot.index.active.flags.writeable is False
    assert snapshot.index.slots == ()
    assert snapshot.index.mem.memory.slots == ()


# ── lifecycle events observed by an in-flight reader ───────────────────────


def test_in_flight_results_are_stale_but_coherent():
    ret, mem = _warm()
    before = ret.query(QUERY, top_k=100)
    pause = {"done": False}
    released = threading.Event()
    entered = threading.Event()
    attach = _stall_hook(ret, pause, released, entered,
                         stage="after_selection")
    captured = {}
    errors = []

    def reader():
        attach()
        try:
            captured["rows"] = ret.query(QUERY, top_k=100)
        except BaseException as exc:            # pragma: no cover
            errors.append(exc)

    thread = threading.Thread(target=reader, name="in-flight")
    thread.start()
    assert entered.wait(10)

    removed = mem.memory.slots[0]
    assert mem._remove_slot_identity(removed) is True
    released.set()
    _join([thread])

    assert not errors
    # The reader renders the state it selected from, even though that state is
    # now stale relative to live memory.
    assert rows_equal(captured["rows"], before)
    assert any(r["slot_id"] == id(removed) for r in captured["rows"])
    fresh = ret.query(QUERY, top_k=100)
    assert rows_equal(fresh, _canonical_now(mem, top_k=100))
    assert all(r["slot_id"] != id(removed) for r in fresh)


def test_eviction_during_construction_is_discarded():
    ret, mem = _warm(capacity=5)
    evicted = []

    def hook(attempt, version):
        if not evicted:
            evicted.append(mem._evict(0.0))

    ret.manager.hooks["after_capture"] = hook
    ret.manager.invalidate()
    results = ret.query(QUERY, top_k=100)
    assert evicted
    assert ret.manager.stats["discarded"] >= 1
    assert len(mem.memory.slots) == 4
    assert len(results) == 4
    assert rows_equal(results, _canonical_now(mem, top_k=100))
    assert ret.manager.published.n == 4


def test_checkpoint_restore_during_query_activity(tmp_path):
    ret, mem = _warm()
    path = os.path.join(str(tmp_path), "state.npz")
    save_checkpoint(mem, path)

    mem.absorb(Candidate(*NEW_FACT[:3], timestamp=NEW_FACT[3]))
    assert any(getattr(s, "debug_subject", None) == "Zed"
               for s in mem.memory.slots)

    pause = {"done": False}
    released = threading.Event()
    entered = threading.Event()
    attach = _stall_hook(ret, pause, released, entered)
    captured = {}
    errors = []

    def reader():
        attach()
        try:
            captured["rows"] = ret.query(QUERY, top_k=100)
        except BaseException as exc:            # pragma: no cover
            errors.append(exc)

    thread = threading.Thread(target=reader, name="restore-reader")
    thread.start()
    assert entered.wait(10)

    load_into_black_hole_memory(mem, path)
    restored_version = mem._state_version
    assert not any(getattr(s, "debug_subject", None) == "Zed"
                   for s in mem.memory.slots)
    released.set()
    _join([thread])

    assert not errors
    assert ret.manager.stats["discarded"] >= 1
    assert ret.manager.published.version == restored_version
    assert rows_equal(captured["rows"], _canonical_now(mem, top_k=100))
    assert all(r["subject"] != "Zed" for r in captured["rows"])


# ── incremental parity and writer/reader convergence ───────────────────────


def test_incremental_reuses_rows_and_matches_canonical_after_mutations():
    ret, mem = _warm(cls=IncrementalSnapshotRetriever)
    for i in range(6):
        mem.absorb(Candidate(f"S{i}", "likes", f"O{i}", timestamp=40.0 + i))
        assert rows_equal(ret.query(QUERY, top_k=100),
                          _canonical_now(mem, top_k=100))
    stats = ret.manager.stats
    assert stats["incremental_publications"] >= 1
    assert stats["rows_reused"] > 0
    assert stats["full_builds"] >= 1


def test_concurrent_writers_and_readers_converge():
    ret, mem = _warm(capacity=500)
    errors = []
    last = {}
    barrier = threading.Barrier(5)
    writer_done = threading.Event()
    writer_facts = [(f"W{i}", "likes", f"V{i}", 50.0 + i) for i in range(25)]

    def writer():
        try:
            barrier.wait(10)
            for fact in writer_facts:
                mem.absorb(Candidate(*fact[:3], timestamp=fact[3]))
        except BaseException as exc:            # pragma: no cover
            errors.append(exc)
        finally:
            writer_done.set()

    def reader(i):
        try:
            barrier.wait(10)
            for _ in range(25):
                try:
                    rows = ret.query(QUERY, top_k=20)
                except SnapshotContended:
                    # Same hand-off the adapter performs for a contended query.
                    rows = mem.query(QUERY, top_k=20)
                assert len({r["slot_index"] for r in rows}) == len(rows)
                assert len({r["slot_id"] for r in rows}) == len(rows)
                last[i] = rows
            # The final snapshot is taken after every write so the row-count
            # assertion below does not depend on the writer's progress.
            assert writer_done.wait(30), "writer did not finish"
            last[i] = ret.query(QUERY, top_k=20)
        except BaseException as exc:            # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, name="writer")]
    threads += [threading.Thread(target=reader, args=(i,), name=f"r{i}")
                for i in range(4)]
    for t in threads:
        t.start()
    _join(threads)

    assert not errors
    assert ret.manager.stats["build_errors"] == 0
    final = ret.query(QUERY, top_k=50)
    assert rows_equal(final, _canonical_now(mem, top_k=50))
    assert ret.manager.published.version == mem._state_version
    assert all(r["slot_id"] in {id(s) for s in mem.memory.slots} for r in final)

    # Every reader row referred to a slot identity that existed in its own
    # snapshot; ids of removed slots never reappear.
    live_ids = {id(s) for s in mem.memory.slots}
    for rows in last.values():
        assert len(rows) == 20
        assert all(isinstance(r["slot_id"], int) for r in rows)
        assert all(r["slot_id"] in live_ids for r in rows)
