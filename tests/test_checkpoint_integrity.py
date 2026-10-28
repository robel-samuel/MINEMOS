"""
MINEMOS Production Readiness - Task 4: Checkpoint Integrity, Crash Safety, and Recovery Audit Test Suite.

Comprehensive deterministic and fault-injection tests for:
1. Atomic save & process interruption safety (last good checkpoint preserved).
2. Truncated, malformed, and empty checkpoint rejection.
3. Missing fields and invalid scalar metadata detection.
4. Dimensional and shape inconsistency validation.
5. Numeric anomaly (NaN, Inf, confidence bounds) detection.
6. SHA-256 checksum corruption detection.
7. Automatic recovery from backup (.bak) on corrupted primary.
8. Full temporal validity windows, activity states, and metadata roundtrip.
9. Backward compatibility with v1 legacy checkpoints.
10. Isolation: failed checkpoint loading preserves existing in-memory state.
11. BlackHoleMemory full persistence, auxiliary index rebuild, and query roundtrip.
12. High-concurrency checkpointing during active memory mutations.
"""

import os
import sys
import time
import shutil
import tempfile
import threading
import numpy as np
import pytest

from phase2.memory_state import MemoryState, MemorySlot
from phase2.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_into_black_hole_memory,
    CheckpointError,
    CheckpointCorruptedError,
    CheckpointValidationError,
    CheckpointNotFoundError,
    SCHEMA_VERSION
)
from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate


# ─────────────────────────────────────────────────────────────────────────────
# 1. Atomic Save & Process Interruption Safety
# ─────────────────────────────────────────────────────────────────────────────

def test_atomic_save_protects_existing_checkpoint_on_write_failure(tmp_path):
    """
    Simulate a crash / disk full failure midway through checkpoint save.
    The original valid checkpoint (and backup) must remain intact and loadable.
    """
    ckpt_path = str(tmp_path / "protected_checkpoint.npz")
    
    # Create initial valid checkpoint
    mem = MemoryState(capacity=10, dim=64)
    mem.add(key=np.ones(64, dtype=np.float32), value=np.ones(64, dtype=np.float32), confidence=0.9, timestamp=10.0)
    save_checkpoint(mem, ckpt_path)
    initial_size = os.path.getsize(ckpt_path)
    assert initial_size > 0

    # Simulate write failure by injecting an error into np.savez
    real_savez = np.savez
    def failing_savez(*args, **kwargs):
        # Create a partial file to simulate partial write before crash
        target = args[0]
        with open(target, "wb") as f:
            f.write(b"partial corrupted garbage")
        raise OSError("Simulated disk error or process termination midway through save")

    np.savez = failing_savez
    try:
        mem2 = MemoryState(capacity=10, dim=64)
        mem2.add(key=np.zeros(64, dtype=np.float32), value=np.zeros(64, dtype=np.float32), confidence=0.5, timestamp=20.0)
        with pytest.raises(CheckpointError):
            save_checkpoint(mem2, ckpt_path)
    finally:
        np.savez = real_savez

    # Verify original checkpoint was NOT corrupted or truncated
    assert os.path.exists(ckpt_path)
    assert os.path.getsize(ckpt_path) == initial_size

    # Verify checkpoint still loads perfectly
    restored = load_checkpoint(ckpt_path)
    assert len(restored.slots) == 1
    assert restored.slots[0].timestamp == 10.0
    assert np.allclose(restored.slots[0].key, np.ones(64, dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Truncated, Malformed, and Nonexistent Checkpoint Handling
# ─────────────────────────────────────────────────────────────────────────────

def test_nonexistent_checkpoint_raises_not_found(tmp_path):
    nonexistent = str(tmp_path / "does_not_exist.npz")
    with pytest.raises(CheckpointNotFoundError):
        load_checkpoint(nonexistent)


def test_truncated_and_zero_byte_files_rejected(tmp_path):
    # Empty file
    empty_path = str(tmp_path / "empty.npz")
    with open(empty_path, "wb") as f:
        pass
    with pytest.raises(CheckpointCorruptedError):
        load_checkpoint(empty_path)

    # Truncated zip header
    truncated_path = str(tmp_path / "truncated.npz")
    with open(truncated_path, "wb") as f:
        f.write(b"PK\x03\x04truncated_header_data")
    with pytest.raises(CheckpointCorruptedError):
        load_checkpoint(truncated_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Missing Fields and Scalar Metadata Invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_required_fields_rejected(tmp_path):
    path = str(tmp_path / "missing_keys.npz")
    # Save archive missing 'values'
    np.savez(
        path,
        keys=np.zeros((2, 64), dtype=np.float32),
        confidences=np.array([1.0, 1.0], dtype=np.float32),
        timestamps=np.array([1.0, 2.0], dtype=np.float64),
        created_ats=np.array([1.0, 2.0], dtype=np.float64),
        update_counts=np.array([1, 1], dtype=np.int32),
        capacity=10,
        dim=64
    )
    with pytest.raises(CheckpointValidationError) as exc:
        load_checkpoint(path)
    assert "Missing required field 'values'" in str(exc.value)


def test_invalid_capacity_and_dimension_rejected(tmp_path):
    # Capacity zero or negative
    path1 = str(tmp_path / "invalid_cap.npz")
    np.savez(
        path1,
        keys=np.zeros((1, 64), dtype=np.float32),
        values=np.zeros((1, 64), dtype=np.float32),
        confidences=np.array([1.0], dtype=np.float32),
        timestamps=np.array([1.0], dtype=np.float64),
        created_ats=np.array([1.0], dtype=np.float64),
        update_counts=np.array([1], dtype=np.int32),
        capacity=0,
        dim=64
    )
    with pytest.raises(CheckpointValidationError) as exc:
        load_checkpoint(path1)
    assert "Must be > 0" in str(exc.value)

    # Slot count exceeds capacity
    path2 = str(tmp_path / "exceeds_cap.npz")
    np.savez(
        path2,
        keys=np.zeros((5, 64), dtype=np.float32),
        values=np.zeros((5, 64), dtype=np.float32),
        confidences=np.ones(5, dtype=np.float32),
        timestamps=np.ones(5, dtype=np.float64),
        created_ats=np.ones(5, dtype=np.float64),
        update_counts=np.ones(5, dtype=np.int32),
        capacity=3,  # 5 slots > capacity 3
        dim=64
    )
    with pytest.raises(CheckpointValidationError) as exc:
        load_checkpoint(path2)
    assert "exceeds stated capacity" in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dimensional and Shape Inconsistency Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_dimension_mismatch_rejected(tmp_path):
    path = str(tmp_path / "dim_mismatch.npz")
    # Metadata states dim=64, but keys array has dim=128
    np.savez(
        path,
        keys=np.zeros((2, 128), dtype=np.float32),
        values=np.zeros((2, 128), dtype=np.float32),
        confidences=np.ones(2, dtype=np.float32),
        timestamps=np.ones(2, dtype=np.float64),
        created_ats=np.ones(2, dtype=np.float64),
        update_counts=np.ones(2, dtype=np.int32),
        capacity=10,
        dim=64
    )
    with pytest.raises(CheckpointValidationError) as exc:
        load_checkpoint(path)
    assert "shape" in str(exc.value) and "64" in str(exc.value)


def test_ragged_array_lengths_rejected(tmp_path):
    path = str(tmp_path / "length_mismatch.npz")
    # Keys has 3 rows, but values has only 2 rows
    np.savez(
        path,
        keys=np.zeros((3, 64), dtype=np.float32),
        values=np.zeros((2, 64), dtype=np.float32),
        confidences=np.ones(3, dtype=np.float32),
        timestamps=np.ones(3, dtype=np.float64),
        created_ats=np.ones(3, dtype=np.float64),
        update_counts=np.ones(3, dtype=np.int32),
        capacity=10,
        dim=64
    )
    with pytest.raises(CheckpointValidationError) as exc:
        load_checkpoint(path)
    assert "Values array shape" in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Numeric Anomaly (NaN, Inf, Bounds) Detection
# ─────────────────────────────────────────────────────────────────────────────

def test_nan_and_inf_vectors_rejected(tmp_path):
    path = str(tmp_path / "nan_vector.npz")
    bad_keys = np.ones((2, 64), dtype=np.float32)
    bad_keys[0, 5] = np.nan
    np.savez(
        path,
        keys=bad_keys,
        values=np.ones((2, 64), dtype=np.float32),
        confidences=np.ones(2, dtype=np.float32),
        timestamps=np.ones(2, dtype=np.float64),
        created_ats=np.ones(2, dtype=np.float64),
        update_counts=np.ones(2, dtype=np.int32),
        capacity=10,
        dim=64
    )
    with pytest.raises(CheckpointValidationError) as exc:
        load_checkpoint(path)
    assert "NaN or Inf" in str(exc.value)


def test_out_of_bound_confidence_rejected(tmp_path):
    path = str(tmp_path / "bad_confidence.npz")
    np.savez(
        path,
        keys=np.ones((1, 64), dtype=np.float32),
        values=np.ones((1, 64), dtype=np.float32),
        confidences=np.array([1.5], dtype=np.float32),  # > 1.0 invalid
        timestamps=np.array([10.0], dtype=np.float64),
        created_ats=np.array([10.0], dtype=np.float64),
        update_counts=np.array([1], dtype=np.int32),
        capacity=10,
        dim=64
    )
    with pytest.raises(CheckpointValidationError) as exc:
        load_checkpoint(path)
    assert "Confidence values out of bounds" in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SHA-256 Checksum Tampering & Corruption Detection
# ─────────────────────────────────────────────────────────────────────────────

def test_checksum_detects_silent_file_tampering(tmp_path):
    ckpt_path = str(tmp_path / "tamper_test.npz")
    mem = MemoryState(capacity=10, dim=64)
    mem.add(key=np.ones(64, dtype=np.float32), value=np.ones(64, dtype=np.float32), confidence=0.8, timestamp=10.0)
    save_checkpoint(mem, ckpt_path)

    # Unpack archive, modify one value in 'confidences', re-pack with original checksum
    with np.load(ckpt_path) as data:
        archive_dict = {k: data[k] for k in data.files}
    
    # Tamper with confidence value
    archive_dict["confidences"] = np.array([0.2], dtype=np.float32)
    tampered_path = str(tmp_path / "tampered.npz")
    np.savez(tampered_path, **archive_dict)

    with pytest.raises(CheckpointCorruptedError) as exc:
        load_checkpoint(tampered_path, allow_backup_recovery=False)
    assert "Integrity checksum mismatch" in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Automatic Recovery from Backup (.bak)
# ─────────────────────────────────────────────────────────────────────────────

def test_automatic_recovery_from_backup_on_primary_corruption(tmp_path):
    ckpt_path = str(tmp_path / "backup_test.npz")
    
    # Save first valid version
    mem1 = MemoryState(capacity=10, dim=64)
    mem1.add(key=np.ones(64, dtype=np.float32) * 1.0, value=np.ones(64, dtype=np.float32), confidence=0.8, timestamp=10.0)
    save_checkpoint(mem1, ckpt_path)

    # Save second valid version (mem1 is copied to .bak)
    mem2 = MemoryState(capacity=10, dim=64)
    mem2.add(key=np.ones(64, dtype=np.float32) * 2.0, value=np.ones(64, dtype=np.float32), confidence=0.9, timestamp=20.0)
    save_checkpoint(mem2, ckpt_path)

    assert os.path.exists(ckpt_path)
    assert os.path.exists(ckpt_path + ".bak")

    # Corrupt the primary checkpoint by overwriting with garbage
    with open(ckpt_path, "wb") as f:
        f.write(b"garbage_data_overwriting_primary_checkpoint")

    # Loading should automatically fall back to .bak and succeed!
    recovered = load_checkpoint(ckpt_path, allow_backup_recovery=True)
    assert len(recovered.slots) == 1
    # Recovered state is the previous valid backup
    assert recovered.slots[0].timestamp in (10.0, 20.0)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Full Temporal Validity Windows & Metadata Roundtrip
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_validity_and_metadata_preserved_across_roundtrip(tmp_path):
    ckpt_path = str(tmp_path / "temporal_metadata.npz")
    mem = MemoryState(capacity=10, dim=64)

    # Slot 1: Superseded historical slot
    s1 = mem.add(
        key=np.ones(64, dtype=np.float32) * 1,
        value=np.ones(64, dtype=np.float32) * 1,
        confidence=0.95,
        timestamp=100.0,
        debug_subject="Seraphina",
        debug_predicate="laboratory located in",
        text="Seraphina laboratory located in Svalbard.",
        dense_emb=np.ones(384, dtype=np.float32) * 0.1,
        valid_from=100.0,
        valid_until=150.0  # INACTIVE / SUPERSEDED
    )
    s1.contra_count = 1

    # Slot 2: Active replacement slot
    s2 = mem.add(
        key=np.ones(64, dtype=np.float32) * 2,
        value=np.ones(64, dtype=np.float32) * 2,
        confidence=0.98,
        timestamp=150.0,
        debug_subject="Seraphina",
        debug_predicate="laboratory located in",
        text="Seraphina laboratory located in Tromso.",
        dense_emb=np.ones(384, dtype=np.float32) * 0.2,
        valid_from=150.0,
        valid_until=None  # ACTIVE
    )

    save_checkpoint(mem, ckpt_path)
    restored = load_checkpoint(ckpt_path)

    assert len(restored.slots) == 2

    # Verify slot 1 remains superseded and inactive
    r1 = restored.slots[0]
    assert not r1.is_active
    assert r1.valid_until == 150.0
    assert r1.valid_from == 100.0
    assert r1.debug_subject == "Seraphina"
    assert r1.debug_predicate == "laboratory located in"
    assert "Svalbard" in r1.text
    assert r1.contra_count == 1
    assert r1.dense_emb is not None
    assert np.allclose(r1.dense_emb, s1.dense_emb)

    # Verify slot 2 remains active
    r2 = restored.slots[1]
    assert r2.is_active
    assert r2.valid_until is None
    assert r2.valid_from == 150.0
    assert r2.debug_subject == "Seraphina"
    assert "Tromso" in r2.text
    assert r2.dense_emb is not None
    assert np.allclose(r2.dense_emb, s2.dense_emb)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Backward Compatibility with v1 Legacy Checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def test_backward_compatibility_loads_v1_archive_cleanly(tmp_path):
    path = str(tmp_path / "legacy_v1.npz")
    # Save legacy v1 format (no schema_version, no validity/text fields)
    n = 2
    dim = 64
    np.savez(
        path,
        keys=np.ones((n, dim), dtype=np.float32),
        values=np.ones((n, dim), dtype=np.float32),
        confidences=np.array([0.75, 0.85], dtype=np.float32),
        timestamps=np.array([10.0, 20.0], dtype=np.float64),
        created_ats=np.array([10.0, 20.0], dtype=np.float64),
        update_counts=np.array([1, 2], dtype=np.int32),
        capacity=10,
        dim=dim
    )

    loaded = load_checkpoint(path)
    assert len(loaded.slots) == 2
    assert loaded.slots[0].is_active
    assert loaded.slots[0].valid_until is None
    assert loaded.slots[0].valid_from == 10.0
    assert loaded.slots[1].update_count == 2
    assert loaded.slots[0].debug_subject is None


def test_legacy_v2_process_local_links_fail_clearly(tmp_path):
    """v2 archives stored raw id() values for supersession links.

    Those ids cannot be mapped onto the restored slot objects, so loading a
    chained v2 archive must fail loudly rather than restore dangling links.
    """
    path = str(tmp_path / "legacy_v2_links.npz")
    n, dim = 2, 64
    np.savez(
        path,
        keys=np.zeros((n, dim), dtype=np.float32),
        values=np.zeros((n, dim), dtype=np.float32),
        confidences=np.ones(n, dtype=np.float32),
        timestamps=np.array([10.0, 20.0], dtype=np.float64),
        created_ats=np.array([10.0, 20.0], dtype=np.float64),
        update_counts=np.ones(n, dtype=np.int32),
        schema_version=2,
        supersedes=np.array([-1, 123456789], dtype=np.int64),
        superseded_by=np.array([123456789, -1], dtype=np.int64),
        capacity=10,
        dim=dim,
    )
    with pytest.raises(CheckpointValidationError) as exc:
        load_checkpoint(path, allow_backup_recovery=False)
    assert "cannot be remapped" in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# 10. In-Memory Isolation on Failed Load
# ─────────────────────────────────────────────────────────────────────────────

def test_failed_load_does_not_mutate_active_memory(tmp_path):
    """If load_checkpoint fails, existing memory in caller remains unmodified."""
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("TestEntity", "status is", "active", timestamp=10.0))
    initial_slots_count = len(mem.memory.slots)

    corrupted_path = str(tmp_path / "corrupted_file.npz")
    with open(corrupted_path, "wb") as f:
        f.write(b"bad_corrupt_data")

    # Attempting to load into mem should raise and leave mem untouched
    with pytest.raises(CheckpointCorruptedError):
        mem.load_checkpoint(corrupted_path, allow_backup_recovery=False)

    assert len(mem.memory.slots) == initial_slots_count
    assert mem.recall("TestEntity", "status is")[0]["is_active"]


# ─────────────────────────────────────────────────────────────────────────────
# 11. BlackHoleMemory Persistence and Query Roundtrip
# ─────────────────────────────────────────────────────────────────────────────

def test_black_hole_memory_save_and_restore_roundtrip(tmp_path):
    ckpt_path = str(tmp_path / "bhm_full.npz")
    mem = BlackHoleMemory(capacity=10)

    # Ingest facts
    mem.absorb(Candidate("Alice", "resides in", "Berlin", timestamp=100.0))
    # Contradiction: Alice moves to Munich
    mem.absorb(Candidate("Alice", "resides in", "Munich", timestamp=200.0))
    # Independent fact
    mem.absorb(Candidate("Bob", "researches", "Quantum Gravity", timestamp=300.0))

    # Save BlackHoleMemory directly
    mem.save_checkpoint(ckpt_path)

    # Restore into fresh instance
    restored = BlackHoleMemory.from_checkpoint(ckpt_path, capacity=10)

    # Query checks
    history = restored.get_history("Alice", "resides in")
    assert len(history) == 2
    # Berlin must be inactive/superseded
    assert not history[0]["is_active"]
    assert history[0]["valid_until"] == 200.0
    # Munich must be active
    assert history[1]["is_active"]
    assert history[1]["valid_until"] is None

    # Exact recall
    alice_active = restored.recall("Alice", "resides in")
    assert len(alice_active) == 1
    assert "Munich" in alice_active[0]["text"]

    # Dense semantic query
    results = restored.query("Where does Alice live?", top_k=2)
    assert len(results) >= 1
    assert "Munich" in results[0]["text"]

    # Status check
    st = restored.status()
    assert st["total_slots"] == 3
    assert st["active_slots"] == 2
    assert st["superseded_slots"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 12. High-Concurrency Checkpoint Saves During Memory Mutations
# ─────────────────────────────────────────────────────────────────────────────

def test_concurrent_saves_during_heavy_mutations(tmp_path):
    ckpt_path = str(tmp_path / "concurrent_stress.npz")
    mem = BlackHoleMemory(capacity=30)
    stop_event = threading.Event()
    errors = []

    def writer(tid: int):
        for step in range(20):
            if stop_event.is_set():
                break
            try:
                mem.absorb(Candidate(f"Item_{tid}_{step}", "observed at", f"val_{step}", timestamp=100.0 + step))
            except Exception as e:
                errors.append(f"Writer-{tid}: {e}")
            time.sleep(0.002)

    def saver(sid: int):
        while not stop_event.is_set():
            try:
                mem.save_checkpoint(ckpt_path)
                loaded = load_checkpoint(ckpt_path)
                assert len(loaded.slots) <= mem.capacity
            except Exception as e:
                errors.append(f"Saver-{sid}: {e}")
            time.sleep(0.005)

    writers = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
    savers = [threading.Thread(target=saver, args=(j,)) for j in range(2)]

    for t in writers + savers:
        t.start()

    for w in writers:
        w.join()

    stop_event.set()
    for s in savers:
        s.join()

    assert not errors, f"Errors during concurrent saves: {errors}"
    final_mem = load_checkpoint(ckpt_path)
    assert len(final_mem.slots) > 0
    assert len(final_mem.slots) <= mem.capacity
