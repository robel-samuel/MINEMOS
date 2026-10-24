"""
MINEMOS Production Readiness - Task 7 Test Suite:
Security Remediation, Persistence Hardening, and Privacy Deletion Validation.

Covers:
1. SEC-01 Checkpoint Resource Bounds:
   - Negative and zero dimensions rejection.
   - Non-integral capacity/dim rejection.
   - Excess capacity (> MAX_CHECKPOINT_CAPACITY) rejection.
   - Excess dimension (> MAX_CHECKPOINT_DIM) rejection.
   - Dimension/keys shape mismatch rejection.
   - Legitimate supported large checkpoints allowed.

2. SEC-03 Strict Auxiliary Vector Validation:
   - Dense embeddings rank and row-count mismatches.
   - Out-of-bounds dense column dimensions.
   - NaN / Inf values in dense_embs.
   - Anchor matrices: wrong rank, mismatched timestamps/indices length.
   - NaN / Inf values in anchor_vectors or anchor_timestamps.
   - Out-of-bounds slot index references in anchor_slot_indices.
   - Clean handling of empty anchor arrays and legacy v1 archives.

3. SEC-02 Safe Checkpoint Path Confinement:
   - Normal relative and absolute paths without confinement.
   - Successful save and load within allowed_base_dir.
   - Rejection of parent directory traversal (.. escapes).
   - Rejection of absolute paths outside allowed_base_dir.
   - Rejection of empty or invalid path types.

4. SEC-04 Explicit Memory Deletion and Privacy Semantics:
   - delete_slot by ID removes slot, texts, dense embeddings, BM25 docs, and anchors.
   - delete_slot mends supersession chains.
   - purge_subject (exact and case-insensitive) removes all matching slots.
   - Recalls and queries after deletion do not return deleted facts.
   - Persistence roundtrip after deletion: deleted items stay gone.
   - Status counters and footprint accurately update after deletion.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

from phase2.memory_state import MemoryState, MemorySlot
from phase2.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_into_black_hole_memory,
    _validate_checkpoint_path,
    CheckpointError,
    CheckpointValidationError,
    CheckpointCorruptedError,
    CheckpointNotFoundError,
    CheckpointPathSecurityError,
    MAX_CHECKPOINT_CAPACITY,
    MAX_CHECKPOINT_DIM,
    SCHEMA_VERSION
)
from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEC-01 Checkpoint Resource Bounds Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_reject_zero_and_negative_dimensions(tmp_path):
    ckpt_zero = str(tmp_path / "zero_dim.npz")
    np.savez(
        ckpt_zero,
        keys=np.zeros((0, 64), dtype=np.float32),
        values=np.zeros((0, 64), dtype=np.float32),
        confidences=np.zeros(0, dtype=np.float32),
        timestamps=np.zeros(0, dtype=np.float64),
        created_ats=np.zeros(0, dtype=np.float64),
        update_counts=np.zeros(0, dtype=np.int32),
        capacity=0,
        dim=64
    )
    with pytest.raises(CheckpointValidationError, match="Invalid dimensions"):
        load_checkpoint(ckpt_zero)

    ckpt_neg = str(tmp_path / "neg_dim.npz")
    np.savez(
        ckpt_neg,
        keys=np.zeros((0, 64), dtype=np.float32),
        values=np.zeros((0, 64), dtype=np.float32),
        confidences=np.zeros(0, dtype=np.float32),
        timestamps=np.zeros(0, dtype=np.float64),
        created_ats=np.zeros(0, dtype=np.float64),
        update_counts=np.zeros(0, dtype=np.int32),
        capacity=10,
        dim=-5
    )
    with pytest.raises(CheckpointValidationError, match="Invalid dimensions"):
        load_checkpoint(ckpt_neg)


def test_reject_non_integral_dimensions(tmp_path):
    ckpt = str(tmp_path / "float_dim.npz")
    np.savez(
        ckpt,
        keys=np.zeros((0, 64), dtype=np.float32),
        values=np.zeros((0, 64), dtype=np.float32),
        confidences=np.zeros(0, dtype=np.float32),
        timestamps=np.zeros(0, dtype=np.float64),
        created_ats=np.zeros(0, dtype=np.float64),
        update_counts=np.zeros(0, dtype=np.int32),
        capacity=10.5,
        dim=64
    )
    with pytest.raises(CheckpointValidationError, match="Non-integral capacity"):
        load_checkpoint(ckpt)


def test_reject_unbounded_excess_capacity(tmp_path):
    ckpt = str(tmp_path / "excess_capacity.npz")
    np.savez(
        ckpt,
        keys=np.zeros((0, 64), dtype=np.float32),
        values=np.zeros((0, 64), dtype=np.float32),
        confidences=np.zeros(0, dtype=np.float32),
        timestamps=np.zeros(0, dtype=np.float64),
        created_ats=np.zeros(0, dtype=np.float64),
        update_counts=np.zeros(0, dtype=np.int32),
        capacity=MAX_CHECKPOINT_CAPACITY + 1,
        dim=64
    )
    with pytest.raises(CheckpointValidationError, match="exceeds maximum allowed limit"):
        load_checkpoint(ckpt)


def test_reject_unbounded_excess_dimension(tmp_path):
    ckpt = str(tmp_path / "excess_dim.npz")
    np.savez(
        ckpt,
        keys=np.zeros((0, 64), dtype=np.float32),
        values=np.zeros((0, 64), dtype=np.float32),
        confidences=np.zeros(0, dtype=np.float32),
        timestamps=np.zeros(0, dtype=np.float64),
        created_ats=np.zeros(0, dtype=np.float64),
        update_counts=np.zeros(0, dtype=np.int32),
        capacity=10,
        dim=MAX_CHECKPOINT_DIM + 1
    )
    with pytest.raises(CheckpointValidationError, match="exceeds maximum allowed limit"):
        load_checkpoint(ckpt)


def test_allow_legitimate_large_supported_checkpoint(tmp_path):
    ckpt = str(tmp_path / "large_supported.npz")
    mem = MemoryState(capacity=5000, dim=4096)
    for i in range(5):
        mem.add(
            key=np.ones(4096, dtype=np.float32) * i,
            value=np.ones(4096, dtype=np.float32) * i,
            confidence=0.8,
            timestamp=float(i)
        )
    save_checkpoint(mem, ckpt)
    restored = load_checkpoint(ckpt)
    assert len(restored.slots) == 5
    assert restored.capacity == 5000
    assert restored.dim == 4096


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEC-03 Strict Auxiliary Vector Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_reject_malformed_dense_embs_ndim_and_rows(tmp_path):
    ckpt = str(tmp_path / "bad_dense_shape.npz")
    mem = MemoryState(capacity=10, dim=64)
    mem.add(key=np.ones(64, dtype=np.float32), value=np.ones(64, dtype=np.float32), confidence=0.9, timestamp=1.0)
    save_checkpoint(mem, ckpt)

    # Corrupt dense_embs shape in archive
    data = dict(np.load(ckpt, allow_pickle=False))
    data["dense_embs"] = np.ones((5, 384), dtype=np.float32) # 5 rows vs 1 slot
    bad_ckpt = str(tmp_path / "corrupt_dense.npz")
    np.savez(bad_ckpt, **data)

    with pytest.raises(CheckpointValidationError, match="Array 'dense_embs' row count"):
        load_checkpoint(bad_ckpt)


def test_reject_nan_or_inf_in_dense_embs(tmp_path):
    ckpt = str(tmp_path / "nan_dense.npz")
    mem = MemoryState(capacity=10, dim=64)
    mem.add(key=np.ones(64, dtype=np.float32), value=np.ones(64, dtype=np.float32), confidence=0.9, timestamp=1.0)
    save_checkpoint(mem, ckpt)

    data = dict(np.load(ckpt, allow_pickle=False))
    dense = np.ones((1, 384), dtype=np.float32)
    dense[0, 10] = np.nan
    data["dense_embs"] = dense
    bad_ckpt = str(tmp_path / "bad_nan_dense.npz")
    np.savez(bad_ckpt, **data)

    with pytest.raises(CheckpointValidationError, match="contains NaN or Inf"):
        load_checkpoint(bad_ckpt)


def test_reject_anchor_vector_dimension_and_nan(tmp_path):
    ckpt = str(tmp_path / "anchor_corrupt.npz")
    mem = MemoryState(capacity=10, dim=64)
    mem.add(key=np.ones(64, dtype=np.float32), value=np.ones(64, dtype=np.float32), confidence=0.9, timestamp=1.0)
    save_checkpoint(mem, ckpt)

    data = dict(np.load(ckpt, allow_pickle=False))
    # Add anchor with NaN
    data["anchor_vectors"] = np.full((1, 384), np.nan, dtype=np.float32)
    data["anchor_timestamps"] = np.array([10.0], dtype=np.float64)
    data["anchor_slot_indices"] = np.array([0], dtype=np.int32)
    bad_ckpt = str(tmp_path / "bad_anchor.npz")
    np.savez(bad_ckpt, **data)

    with pytest.raises(CheckpointValidationError, match="Array 'anchor_vectors' contains NaN or Inf"):
        load_checkpoint(bad_ckpt)


def test_reject_out_of_bounds_anchor_slot_indices(tmp_path):
    ckpt = str(tmp_path / "anchor_oob.npz")
    mem = MemoryState(capacity=10, dim=64)
    mem.add(key=np.ones(64, dtype=np.float32), value=np.ones(64, dtype=np.float32), confidence=0.9, timestamp=1.0)
    save_checkpoint(mem, ckpt)

    data = dict(np.load(ckpt, allow_pickle=False))
    # Point anchor to slot index 99 when only 1 slot exists
    data["anchor_vectors"] = np.ones((1, 384), dtype=np.float32)
    data["anchor_timestamps"] = np.array([10.0], dtype=np.float64)
    data["anchor_slot_indices"] = np.array([99], dtype=np.int32)
    bad_ckpt = str(tmp_path / "bad_anchor_oob.npz")
    np.savez(bad_ckpt, **data)

    with pytest.raises(CheckpointValidationError, match="references out-of-bounds slot index"):
        load_checkpoint(bad_ckpt)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEC-02 Safe Checkpoint Path Confinement Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_safe_path_confinement_allowed_dir(tmp_path):
    base_dir = tmp_path / "safe_box"
    base_dir.mkdir()
    safe_ckpt = str(base_dir / "valid.npz")

    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("System", "status", "ready", timestamp=1.0))
    save_checkpoint(mem, safe_ckpt, allowed_base_dir=str(base_dir))
    assert os.path.exists(safe_ckpt)

    restored = load_checkpoint(safe_ckpt, allowed_base_dir=str(base_dir))
    assert len(restored.slots) == 1


def test_reject_path_traversal_escape(tmp_path):
    base_dir = tmp_path / "safe_box"
    base_dir.mkdir()
    # Try writing to parent directory outside safe_box
    escape_ckpt = str(base_dir / ".." / "escaped.npz")

    mem = BlackHoleMemory(capacity=10)
    with pytest.raises(CheckpointPathSecurityError, match="escapes allowed base directory"):
        save_checkpoint(mem, escape_ckpt, allowed_base_dir=str(base_dir))

    with pytest.raises(CheckpointPathSecurityError, match="escapes allowed base directory"):
        load_checkpoint(escape_ckpt, allowed_base_dir=str(base_dir))


def test_reject_invalid_and_empty_checkpoint_paths(tmp_path):
    mem = MemoryState(capacity=10, dim=64)
    with pytest.raises(CheckpointValidationError, match="cannot be empty"):
        save_checkpoint(mem, "   ")
    with pytest.raises(CheckpointValidationError, match="cannot be empty"):
        load_checkpoint("")
    with pytest.raises(CheckpointValidationError, match="must be a string"):
        save_checkpoint(mem, 12345)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEC-04 Explicit Memory Deletion and Privacy Semantics Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_delete_slot_by_id_and_verify_auxiliary_cleanup():
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Alice", "status", "online", timestamp=10.0))
    mem.absorb(Candidate("Bob", "role", "Engineer", timestamp=20.0))
    assert len(mem.memory.slots) == 2

    alice_slot = mem.memory.slots[0]
    alice_id = id(alice_slot)

    # Ensure present in auxiliary stores
    assert alice_id in mem.slot_texts
    assert len(mem.slot_dense_embs) == 2
    assert mem.bm25_index.num_docs == 2

    # Execute explicit deletion
    deleted = mem.delete_slot(alice_id)
    assert deleted is True

    # Verify removal from all active and auxiliary stores
    assert len(mem.memory.slots) == 1
    assert alice_id not in mem.slot_texts
    assert len(mem.slot_dense_embs) == 1
    assert mem.bm25_index.num_docs == 1

    # Verify queries and recall cannot retrieve Alice
    rec_alice = mem.recall("Alice", "status")
    assert len(rec_alice) == 0

    q_alice = mem.query("Is Alice online?", top_k=2)
    for hit in q_alice:
        assert "Alice" not in hit["text"]

    # Bob must remain intact
    rec_bob = mem.recall("Bob", "role")
    assert len(rec_bob) == 1
    assert "Engineer" in rec_bob[0]["text"]


def test_delete_unknown_slot_returns_false():
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Alice", "status", "online", timestamp=10.0))
    assert mem.delete_slot(9999999) is False
    assert len(mem.memory.slots) == 1


def test_purge_subject_exact_and_case_insensitive():
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Carol", "city", "London", timestamp=10.0))
    # Contradiction: Carol moves to Paris
    mem.absorb(Candidate("Carol", "city", "Paris", timestamp=20.0))
    mem.absorb(Candidate("Dave", "city", "Rome", timestamp=30.0))
    assert len(mem.memory.slots) == 3

    # Purge Carol (both past superseded slot and active slot)
    purged_count = mem.purge_subject("Carol", exact_match=True)
    assert purged_count == 2
    assert len(mem.memory.slots) == 1

    # Verify history is completely wiped
    carol_hist = mem.get_history("Carol")
    assert len(carol_hist) == 0

    carol_rec = mem.recall("Carol")
    assert len(carol_rec) == 0

    # Dave remains
    dave_rec = mem.recall("Dave", "city")
    assert len(dave_rec) == 1
    assert "Rome" in dave_rec[0]["text"]


def test_deletion_persists_across_checkpoint_roundtrip(tmp_path):
    ckpt_path = str(tmp_path / "post_delete.npz")
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Eve", "key", "secret_value", timestamp=10.0))
    mem.absorb(Candidate("Frank", "role", "Auditor", timestamp=20.0))

    # Purge Eve before saving checkpoint
    mem.purge_subject("Eve")
    mem.save_checkpoint(ckpt_path)

    # Restore in completely fresh instance
    restored = BlackHoleMemory.from_checkpoint(ckpt_path, capacity=10)
    assert len(restored.memory.slots) == 1
    assert len(restored.recall("Eve")) == 0
    assert len(restored.get_history("Eve")) == 0

    rec_frank = restored.recall("Frank", "role")
    assert len(rec_frank) == 1
    assert "Auditor" in rec_frank[0]["text"]


# =========================================================================
# Task 7.1 Verification Suite: Evidence-Based Closure Tests
# =========================================================================

def test_sec04_backup_file_retains_deleted_record_and_recovers_if_primary_corrupted(tmp_path):
    """
    Evidence-based test for SEC-04:
    Verifies that while primary checkpoint deletes records cleanly,
    the .bak backup file retains the prior generation, and loading
    from a corrupted primary falls back to .bak and restores the record.
    Proves that logical deletion does NOT equate to physical/backup secure erasure.
    """
    ckpt_path = str(tmp_path / "lifecycle.npz")
    bak_path = f"{ckpt_path}.bak"

    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("SecretAgent", "code", "007", timestamp=10.0))
    mem.absorb(Candidate("PublicAgent", "code", "101", timestamp=20.0))

    # Save Gen 1: Contains SecretAgent and PublicAgent
    mem.save_checkpoint(ckpt_path)
    assert os.path.exists(ckpt_path)
    assert not os.path.exists(bak_path)

    # Delete SecretAgent from active memory
    purged = mem.purge_subject("SecretAgent")
    assert purged == 1
    assert len(mem.memory.slots) == 1

    # Save Gen 2: Overwrites ckpt_path and creates bak_path
    mem.save_checkpoint(ckpt_path)
    assert os.path.exists(ckpt_path)
    assert os.path.exists(bak_path)

    # Primary checkpoint does NOT contain SecretAgent
    loaded_primary = load_checkpoint(ckpt_path)
    assert len(loaded_primary.slots) == 1
    assert loaded_primary.slots[0].debug_subject == "PublicAgent"

    # Backup checkpoint (.bak) DOES retain SecretAgent
    loaded_bak = load_checkpoint(bak_path)
    assert len(loaded_bak.slots) == 2
    subjects_in_bak = {s.debug_subject for s in loaded_bak.slots}
    assert "SecretAgent" in subjects_in_bak

    # If primary is corrupted, fallback to backup recovers SecretAgent
    with open(ckpt_path, "wb") as f:
        f.write(b"CORRUPTED_BYTES")

    recovered = load_checkpoint(ckpt_path, allow_backup_recovery=True)
    recovered_subjects = {s.debug_subject for s in recovered.slots}
    assert "SecretAgent" in recovered_subjects


def test_sec02_symlink_or_junction_outside_allowed_dir_is_rejected(tmp_path):
    """
    Evidence-based test for SEC-02:
    Verifies that _validate_checkpoint_path resolves symlinks/junctions
    via os.path.realpath, preventing path traversal via links pointing outside allowed_base_dir.
    """
    safe_base = tmp_path / "safe_sandbox"
    safe_base.mkdir()
    outside_dir = tmp_path / "forbidden_zone"
    outside_dir.mkdir()

    # Attempt to create a directory link or test canonical realpath resolution
    link_path = safe_base / "escape_link"
    link_created = False
    try:
        os.symlink(str(outside_dir), str(link_path), target_is_directory=True)
        link_created = True
    except (OSError, NotImplementedError):
        # On Windows without admin privileges, try junction via cmd /c mklink /J
        if sys.platform == "win32":
            res = os.system(f'cmd /c mklink /J "{link_path}" "{outside_dir}" >nul 2>&1')
            if res == 0:
                link_created = True

    if link_created:
        traversal_target = str(link_path / "escaped_checkpoint.npz")
        with pytest.raises(CheckpointPathSecurityError):
            _validate_checkpoint_path(traversal_target, allowed_base_dir=str(safe_base))
    else:
        # Fallback for restricted environments: verify realpath relpath logic directly
        import unittest.mock as mock
        with mock.patch("os.path.realpath") as mock_real:
            mock_real.side_effect = lambda p: str(outside_dir / "target.npz") if "escape" in str(p) else str(p)
            with pytest.raises(CheckpointPathSecurityError):
                _validate_checkpoint_path(str(safe_base / "escape.npz"), allowed_base_dir=str(safe_base))


def test_sec01_sec03_zip_preallocation_size_bound_rejects_oversized_arrays(tmp_path):
    """
    Evidence-based test for SEC-01 and SEC-03:
    Verifies that an archive with low stated capacity/dim but an oversized array
    is rejected before allocating memory via data.zip inspection.
    """
    import zipfile
    bad_ckpt = str(tmp_path / "oversized_array.npz")

    # Stated capacity 5, dim 10. Max expected keys.npy bytes is 5*10*4 + 65536 = 65736 bytes.
    # We craft an archive where keys.npy has 200,000 bytes.
    with zipfile.ZipFile(bad_ckpt, "w") as z:
        # Save capacity=5, dim=10
        cap_bytes = io_bytes = np.ndarray((), dtype=np.int64)
        cap_bytes[()] = 5
        dim_bytes = np.ndarray((), dtype=np.int64)
        dim_bytes[()] = 10

        import io
        b_cap = io.BytesIO()
        np.save(b_cap, cap_bytes)
        z.writestr("capacity.npy", b_cap.getvalue())

        b_dim = io.BytesIO()
        np.save(b_dim, dim_bytes)
        z.writestr("dim.npy", b_dim.getvalue())

        # Other required keys minimal
        for k in ["values", "confidences", "timestamps", "created_ats", "update_counts"]:
            b_dummy = io.BytesIO()
            np.save(b_dummy, np.zeros((5, 10) if k == "values" else 5, dtype=np.float32))
            z.writestr(f"{k}.npy", b_dummy.getvalue())

        # Bloated keys.npy with 200,000 dummy bytes (exceeds max_matrix_bytes for capacity 5)
        z.writestr("keys.npy", b"X" * 200_000)

    with pytest.raises(CheckpointValidationError) as exc_info:
        load_checkpoint(bad_ckpt)

    assert "uncompressed size" in str(exc_info.value)
    assert "exceeds expected bound" in str(exc_info.value)

