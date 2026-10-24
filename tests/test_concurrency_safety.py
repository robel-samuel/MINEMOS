"""
MINEMOS Production Readiness - Task 3: Concurrency and Thread Safety Test Suite

Deterministic and stress tests for:
- Concurrent conflicting updates (split-brain prevention).
- Concurrent duplicate observations (consolidation integrity).
- Capacity bound enforcement under concurrent insertion pressure.
- Reader isolation (no uncommitted/vanished state observed).
- Concurrent anchor updates and vector retrieval without dictionary iteration errors.
- Exception safety during mutations (no corrupted state left behind).
- Concurrent checkpointing during active writes.
- Independent attributes of a single entity updated concurrently.
"""

import threading
import time
import os
import tempfile
import pytest
import numpy as np

from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate
from phase2.checkpoint import save_checkpoint, load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# 1. Concurrent Conflicting Updates (No Split-Brain)
# ─────────────────────────────────────────────────────────────────────────────

def test_concurrent_conflicting_updates_prevent_split_brain():
    """
    When multiple threads simultaneously absorb contradictory updates for the
    same entity and attribute, the engine must serialize them:
    - Exactly ONE state must remain active.
    - No duplicate active states for that attribute.
    - All prior states must have their validity closed.
    - Supersession links must form a consistent chain.
    """
    mem = BlackHoleMemory(capacity=20)
    mem.absorb(Candidate("Alice", "resides in the capital of", "France", timestamp=100.0))

    cities = ["Germany", "England", "Japan", "Italy"]
    barrier = threading.Barrier(len(cities))
    errors = []

    def worker(city, t):
        try:
            barrier.wait()
            mem.absorb(Candidate("Alice", "resides in the capital of", city, timestamp=t))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(c, 200.0 + i * 10)) for i, c in enumerate(cities)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent updates: {errors}"

    active_facts = mem.recall("Alice", "resides in the capital of")
    assert len(active_facts) == 1, f"Expected exactly 1 active residence, got {len(active_facts)}"

    # All slots in the supersession history must have valid_until set except the active one
    history = mem.get_history("Alice", "resides in the capital of")
    assert len(history) == len(cities) + 1
    active_count = sum(1 for s in history if s["is_active"])
    assert active_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Concurrent Duplicate Observations (Consolidation Integrity)
# ─────────────────────────────────────────────────────────────────────────────

def test_concurrent_duplicate_absorb_consolidates_cleanly():
    """
    Concurrent duplicate observations must consolidate into a single slot
    rather than creating duplicate slots.
    """
    mem = BlackHoleMemory(capacity=20)
    num_threads = 4
    barrier = threading.Barrier(num_threads)
    errors = []

    def worker():
        try:
            barrier.wait()
            mem.absorb(Candidate("Marcus", "wrote an important", "biographical treatise", timestamp=10.0))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors: {errors}"
    assert len(mem.memory.slots) == 1, f"Expected 1 consolidated slot, got {len(mem.memory.slots)}"
    slot = mem.memory.slots[0]
    assert slot.update_count == num_threads
    assert slot.is_active


# ─────────────────────────────────────────────────────────────────────────────
# 3. Capacity Bound Enforcement Under Concurrent Insertion Pressure
# ─────────────────────────────────────────────────────────────────────────────

def test_concurrent_inserts_respect_capacity_under_pressure():
    """
    Under heavy concurrent writes with capacity nearly full:
    - Capacity limits must NEVER be exceeded.
    - No CapacityExceeded exceptions raised to callers.
    - Eviction and insertion must form an atomic unit.
    """
    capacity = 5
    mem = BlackHoleMemory(capacity=capacity, retention_policy="tiered")
    num_writers = 15
    barrier = threading.Barrier(num_writers)
    errors = []

    def worker(idx):
        try:
            barrier.wait()
            mem.absorb(Candidate(f"Entity_{idx:02d}", "status is", f"value_{idx:02d}", timestamp=10.0 + idx))
        except Exception as e:
            errors.append((idx, type(e).__name__, str(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent inserts caused errors: {errors}"
    assert len(mem.memory.slots) <= capacity, f"Slots ({len(mem.memory.slots)}) exceeded capacity ({capacity})!"
    assert mem.total_evictions >= (num_writers - capacity)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reader Isolation (No Vanished Intermediate State)
# ─────────────────────────────────────────────────────────────────────────────

def test_read_consistency_during_concurrent_updates():
    """
    Readers querying memory while writers are actively superseding slots must
    NEVER observe an empty list (the fact temporarily disappearing) or torn state.
    """
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Seraphina", "current location is", "Outpost Alpha", timestamp=1.0))

    stop_event = threading.Event()
    read_empty_count = 0
    read_total = 0
    read_lock = threading.Lock()

    def writer():
        locs = ["Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota"]
        for i, loc in enumerate(locs):
            if stop_event.is_set():
                break
            mem.absorb(Candidate("Seraphina", "current location is", f"Outpost {loc}", timestamp=10.0 + i))
            time.sleep(0.005)
        stop_event.set()

    def reader():
        nonlocal read_empty_count, read_total
        while not stop_event.is_set():
            res = mem.recall("Seraphina", "current location is")
            with read_lock:
                read_total += 1
                if len(res) == 0:
                    read_empty_count += 1
            time.sleep(0.001)

    t_writer = threading.Thread(target=writer)
    t_readers = [threading.Thread(target=reader) for _ in range(3)]

    for tr in t_readers:
        tr.start()
    t_writer.start()

    t_writer.join()
    stop_event.set()
    for tr in t_readers:
        tr.join()

    assert read_total > 0, "No reads were executed!"
    assert read_empty_count == 0, f"Readers observed empty state {read_empty_count} times out of {read_total} reads!"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Concurrent Anchor Updates and Vector Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def test_concurrent_anchor_updates_and_retrieval():
    """
    Simultaneous queries and absorbs that update contradiction anchors must
    never raise 'dictionary changed size during iteration' or IndexError.
    """
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("SystemCluster", "operational state is", "fully online", timestamp=10.0))

    stop_event = threading.Event()
    errors = []

    def writer():
        for i in range(10):
            if stop_event.is_set():
                break
            try:
                mem.absorb(Candidate("SystemCluster", "operational state is", f"state_{i}", timestamp=20.0 + i))
            except Exception as e:
                errors.append(e)
            time.sleep(0.005)
        stop_event.set()

    def reader():
        while not stop_event.is_set():
            try:
                mem.query("What is the operational state of SystemCluster?", top_k=3)
            except Exception as e:
                errors.append(e)
            time.sleep(0.002)

    t_w = threading.Thread(target=writer)
    t_r = [threading.Thread(target=reader) for _ in range(2)]

    for t in t_r:
        t.start()
    t_w.start()

    t_w.join()
    stop_event.set()
    for t in t_r:
        t.join()

    assert not errors, f"Encountered errors during concurrent query/anchor updates: {errors}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Exception Safety (No Partially Corrupted State on Error)
# ─────────────────────────────────────────────────────────────────────────────

def test_atomic_rollback_on_failed_mutation():
    """
    If a failure occurs during slot allocation or index update:
    - The existing active slot must NOT have its validity closed.
    - Memory state remains intact and retrievable.
    """
    mem = BlackHoleMemory(capacity=5)
    mem.absorb(Candidate("Victor", "holds equity position in", "Apex Corp", timestamp=10.0))
    slot0 = mem.memory.slots[0]

    # Temporarily monkeypatch memory.add to simulate an unexpected error during insertion
    original_add = mem.memory.add

    def failing_add(*args, **kwargs):
        raise RuntimeError("Simulated OOM or backend storage failure during slot add")

    mem.memory.add = failing_add

    try:
        mem.absorb(Candidate("Victor", "holds equity position in", "Apex Corp Liquidated", timestamp=20.0))
    except RuntimeError:
        pass
    finally:
        mem.memory.add = original_add

    # Slot 0 must remain ACTIVE, because the replacing mutation failed!
    assert slot0.is_active, "Slot 0 was deactivated even though the replacement addition failed!"
    assert slot0.valid_until is None


# ─────────────────────────────────────────────────────────────────────────────
# 7. Concurrent Checkpointing and Writes
# ─────────────────────────────────────────────────────────────────────────────

def test_concurrent_checkpoint_and_writes():
    """
    Concurrent save_checkpoint() operations while memory is actively mutating
    must not cause IndexError or corrupted array snapshots.
    """
    mem = BlackHoleMemory(capacity=15)
    stop_event = threading.Event()
    errors = []

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "concurrent_test.npz")

        def writer():
            for i in range(15):
                if stop_event.is_set():
                    break
                try:
                    mem.absorb(Candidate(f"Item_{i:02d}", "property is", f"val_{i:02d}", timestamp=10.0 + i))
                except Exception as e:
                    errors.append(e)
                time.sleep(0.003)
            stop_event.set()

        def checkpointer():
            while not stop_event.is_set():
                try:
                    save_checkpoint(mem.memory, ckpt_path)
                    # Verify checkpoint can be loaded and is valid
                    loaded = load_checkpoint(ckpt_path)
                    assert len(loaded.slots) <= mem.capacity
                except Exception as e:
                    errors.append(e)
                time.sleep(0.005)

        t_w = threading.Thread(target=writer)
        t_c = threading.Thread(target=checkpointer)

        t_w.start()
        t_c.start()

        t_w.join()
        stop_event.set()
        t_c.join()

    assert not errors, f"Errors during concurrent checkpointing: {errors}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Independent Attributes Coexist Under Concurrent Writes
# ─────────────────────────────────────────────────────────────────────────────

def test_independent_attributes_coexist_under_concurrent_writes():
    """
    Concurrent writes updating multiple independent attributes of a single entity
    must all be preserved without cross-attribute collision or eviction.
    """
    mem = BlackHoleMemory(capacity=10)
    attributes = [
        ("lives in the city of", "Stockholm"),
        ("appointed director of", "National Observatory"),
        ("speaks fluent", "Swedish"),
        ("researches astrophysics at", "Uppsala University"),
    ]

    barrier = threading.Barrier(len(attributes))
    errors = []

    def worker(pred, obj, t):
        try:
            barrier.wait()
            mem.absorb(Candidate("Elena", pred, obj, timestamp=t))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(p, o, 10.0 + i * 5)) for i, (p, o) in enumerate(attributes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors: {errors}"
    assert len(mem.memory.slots) == len(attributes)

    for p, o in attributes:
        res = mem.recall("Elena", p)
        assert len(res) == 1, f"Expected 1 active slot for {p}, got {len(res)}"
        assert o in res[0]["text"]
        assert res[0]["is_active"]
