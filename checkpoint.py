"""
Phase 2 — Persistence and Crash-Safe Checkpointing.

Saves the memory state as an atomic, validated .npz binary archive.
Supports schema versioning, complete slot validity window persistence,
integrity checksums, and crash-safe atomic rename with backup preservation.
"""

from __future__ import annotations

import os
import shutil
import hashlib
import time
import zipfile
import threading
from typing import Optional, Union, Any, Dict, List
import numpy as np

from phase2.memory_state import MemoryState, MemorySlot


SCHEMA_VERSION = 2
_FILE_LOCK = threading.Lock()

# Defense-in-depth resource bounds (SEC-01)
# Derived from architecture limits: max benchmark capacity 50,000, 
# standard memory dimension 4096 (D_LEX) / dense dimension 384 (D_DENSE).
# Capacity ceiling 500,000 slots (~350 MB RAM), dimension ceiling 16,384.
MAX_CHECKPOINT_CAPACITY = 500_000
MAX_CHECKPOINT_DIM = 16_384


class CheckpointError(Exception):
    """Base exception for all checkpoint-related errors."""
    pass


class CheckpointCorruptedError(CheckpointError):
    """Raised when a checkpoint file is truncated, malformed, or has an invalid checksum."""
    pass


class CheckpointValidationError(CheckpointError):
    """Raised when checkpoint data violates invariants, dimensions, or value ranges."""
    pass


class CheckpointNotFoundError(CheckpointError, FileNotFoundError):
    """Raised when the specified checkpoint file does not exist."""
    pass


class CheckpointPathSecurityError(CheckpointError):
    """Raised when a checkpoint path escapes the allowed base directory confinement."""
    pass


def _validate_checkpoint_path(path: str, allowed_base_dir: Optional[str] = None) -> str:
    """
    Validates and resolves a checkpoint filesystem path.
    If allowed_base_dir is specified, ensures the resolved path is strictly
    confined within the allowed base directory (preventing directory traversal).
    """
    if not isinstance(path, (str, os.PathLike)):
        raise CheckpointValidationError(f"Checkpoint path must be a string or PathLike, got {type(path).__name__}.")

    str_path = os.fspath(path)
    if not str_path.strip():
        raise CheckpointValidationError("Checkpoint path cannot be empty.")

    abs_path = os.path.abspath(str_path)

    if allowed_base_dir is not None:
        abs_base = os.path.abspath(allowed_base_dir)
        real_path = os.path.realpath(abs_path)
        real_base = os.path.realpath(abs_base)
        # Verify path starts with base directory and doesn't escape via .. or symlinks
        try:
            rel = os.path.relpath(real_path, real_base)
        except ValueError:
            # Different drives on Windows
            raise CheckpointPathSecurityError(
                f"Checkpoint path '{str_path}' is on a different drive than allowed base directory '{allowed_base_dir}'."
            )
        if rel == ".." or rel.startswith(f"..{os.sep}"):
            raise CheckpointPathSecurityError(
                f"Checkpoint path '{str_path}' escapes allowed base directory '{allowed_base_dir}'."
            )
        return real_path

    return abs_path


def _compute_checksum(
    keys: np.ndarray,
    values: np.ndarray,
    confidences: np.ndarray,
    timestamps: np.ndarray,
    created_ats: np.ndarray,
    update_counts: np.ndarray,
    valid_froms: np.ndarray,
    valid_untils: np.ndarray,
) -> str:
    """Computes SHA-256 digest of core numerical memory state arrays."""
    hasher = hashlib.sha256()
    hasher.update(keys.tobytes())
    hasher.update(values.tobytes())
    hasher.update(confidences.tobytes())
    hasher.update(timestamps.tobytes())
    hasher.update(created_ats.tobytes())
    hasher.update(update_counts.tobytes())
    hasher.update(valid_froms.tobytes())
    hasher.update(valid_untils.tobytes())
    return hasher.hexdigest()


def save_checkpoint(
    memory: Union[MemoryState, Any],
    path: str,
    create_backup: bool = True,
    allowed_base_dir: Optional[str] = None
) -> str:
    """
    Saves MemoryState (or BlackHoleMemory) to a crash-safe .npz archive.

    Guarantees:
      1. Serialization happens under lock; I/O occurs outside the memory lock.
      2. Writes first to an isolated temporary file in the destination directory.
      3. Preserves the prior valid checkpoint as '<path>.bak' before replacement.
      4. Atomically replaces destination path via os.replace.
      5. Full slot preservation (keys, values, validity windows, confidences, metadata, texts).
      6. Integrity checksum embedded for corruption detection.
      7. Optional allowed_base_dir restricts destination to a safe sandbox directory.
    """
    target_mem = getattr(memory, "memory", memory)
    if not hasattr(target_mem, "slots") or not hasattr(target_mem, "capacity"):
        raise CheckpointValidationError(
            f"Object of type {type(memory).__name__} does not implement the MemoryState protocol."
        )

    # Validate path and confinement
    abs_path = _validate_checkpoint_path(path, allowed_base_dir=allowed_base_dir)

    # Acquire lock hierarchy: outer memory wrapper first, then inner MemoryState
    outer_lock = getattr(memory, "_lock", None)
    inner_lock = getattr(target_mem, "_lock", None) if target_mem is not memory else None

    if outer_lock is not None:
        outer_lock.acquire()
    if inner_lock is not None:
        inner_lock.acquire()

    try:
        n = len(target_mem.slots)
        dim = int(target_mem.dim)
        capacity = int(target_mem.capacity)

        keys = np.zeros((n, dim), dtype=np.float32)
        values = np.zeros((n, dim), dtype=np.float32)
        confidences = np.zeros(n, dtype=np.float32)
        timestamps = np.zeros(n, dtype=np.float64)
        created_ats = np.zeros(n, dtype=np.float64)
        update_counts = np.zeros(n, dtype=np.int32)
        valid_froms = np.zeros(n, dtype=np.float64)
        valid_untils = np.full(n, np.nan, dtype=np.float64)
        contra_counts = np.zeros(n, dtype=np.int32)
        supersedes = np.full(n, -1, dtype=np.int64)
        superseded_by = np.full(n, -1, dtype=np.int64)

        subjects_list = []
        predicates_list = []
        texts_list = []

        # Check dense embeddings dimension
        dense_dim = None
        for slot in target_mem.slots:
            if slot.dense_emb is not None:
                dense_dim = slot.dense_emb.shape[0]
                break

        if dense_dim is not None:
            dense_embs = np.zeros((n, dense_dim), dtype=np.float32)
            has_dense = np.zeros(n, dtype=bool)
        else:
            dense_embs = np.zeros((0, 0), dtype=np.float32)
            has_dense = np.zeros(n, dtype=bool)

        for i, slot in enumerate(target_mem.slots):
            keys[i] = slot.key
            values[i] = slot.value
            confidences[i] = slot.confidence
            timestamps[i] = slot.timestamp
            created_ats[i] = slot.created_at
            update_counts[i] = slot.update_count

            valid_froms[i] = slot.valid_from if slot.valid_from is not None else slot.timestamp
            if slot.valid_until is not None:
                valid_untils[i] = slot.valid_until

            contra_counts[i] = slot.contra_count
            if slot.supersedes is not None:
                supersedes[i] = slot.supersedes
            if slot.superseded_by is not None:
                superseded_by[i] = slot.superseded_by

            subjects_list.append(slot.debug_subject or "")
            predicates_list.append(slot.debug_predicate or "")
            texts_list.append(slot.text or "")

            if dense_dim is not None and slot.dense_emb is not None:
                dense_embs[i] = slot.dense_emb
                has_dense[i] = True

        subjects_arr = np.array(subjects_list, dtype=str)
        predicates_arr = np.array(predicates_list, dtype=str)
        texts_arr = np.array(texts_list, dtype=str)

        # Operational telemetry counters if BlackHoleMemory
        telemetry = np.zeros(6, dtype=np.int64)
        if hasattr(memory, "total_absorbed"):
            telemetry[0] = memory.total_absorbed
            telemetry[1] = memory.total_updates
            telemetry[2] = memory.total_inserts
            telemetry[3] = memory.total_temporal_updates
            telemetry[4] = memory.total_evictions
            telemetry[5] = memory.total_nli_calls

        # Contradiction repulsion anchors if BlackHoleMemory
        anchor_vecs = []
        anchor_times = []
        anchor_indices = []
        if hasattr(memory, "slot_contra_anchors") and dense_dim is not None:
            slot_id_to_idx = {id(s): idx for idx, s in enumerate(target_mem.slots)}
            for s_id, a_list in memory.slot_contra_anchors.items():
                if s_id in slot_id_to_idx:
                    s_idx = slot_id_to_idx[s_id]
                    for a_vec, a_t in a_list:
                        anchor_vecs.append(a_vec)
                        anchor_times.append(a_t)
                        anchor_indices.append(s_idx)

        if anchor_vecs:
            anchor_vectors = np.array(anchor_vecs, dtype=np.float32)
            anchor_timestamps = np.array(anchor_times, dtype=np.float64)
            anchor_slot_indices = np.array(anchor_indices, dtype=np.int32)
        else:
            anchor_vectors = np.zeros((0, 0), dtype=np.float32)
            anchor_timestamps = np.zeros(0, dtype=np.float64)
            anchor_slot_indices = np.zeros(0, dtype=np.int32)

        checksum = _compute_checksum(
            keys, values, confidences, timestamps, created_ats,
            update_counts, valid_froms, valid_untils
        )

    finally:
        if inner_lock is not None:
            inner_lock.release()
        if outer_lock is not None:
            outer_lock.release()

    # Disk I/O executes outside the in-memory locks
    dir_name = os.path.dirname(abs_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    temp_path = f"{abs_path}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp.npz"
    backup_path = f"{abs_path}.bak"

    with _FILE_LOCK:
        try:
            # Write to temporary file
            np.savez(
                temp_path,
                keys=keys,
                values=values,
                confidences=confidences,
                timestamps=timestamps,
                created_ats=created_ats,
                update_counts=update_counts,
                valid_froms=valid_froms,
                valid_untils=valid_untils,
                contra_counts=contra_counts,
                supersedes=supersedes,
                superseded_by=superseded_by,
                subjects=subjects_arr,
                predicates=predicates_arr,
                texts=texts_arr,
                dense_embs=dense_embs,
                has_dense=has_dense,
                telemetry=telemetry,
                anchor_vectors=anchor_vectors,
                anchor_timestamps=anchor_timestamps,
                anchor_slot_indices=anchor_slot_indices,
                capacity=capacity,
                dim=dim,
                schema_version=SCHEMA_VERSION,
                checksum=np.array([checksum], dtype=str)
            )

            # Preserve previous valid checkpoint as backup before replacement
            if create_backup and os.path.exists(abs_path):
                try:
                    shutil.copy2(abs_path, backup_path)
                except Exception:
                    pass

            # Atomic rename with retry for Windows file locks
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    os.replace(temp_path, abs_path)
                    break
                except PermissionError:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.01 * (attempt + 1))

        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise CheckpointError(f"Failed to persist checkpoint to '{path}': {e}") from e

    return abs_path


def _validate_and_reconstruct(data: Any, path: str) -> MemoryState:
    """Validates archive components and constructs a clean MemoryState instance."""
    # 1. Required fields presence
    required_keys = ["keys", "values", "confidences", "timestamps", "created_ats", "update_counts", "capacity", "dim"]
    for k in required_keys:
        if k not in data:
            raise CheckpointValidationError(f"Missing required field '{k}' in checkpoint '{path}'.")

    try:
        cap_raw = data["capacity"]
        dim_raw = data["dim"]

        # Reject floating-point or non-integral scalar representations
        if hasattr(cap_raw, "dtype") and cap_raw.dtype.kind not in ("i", "u"):
            raise CheckpointValidationError(f"Non-integral capacity scalar (dtype={cap_raw.dtype}) in checkpoint '{path}'.")
        if hasattr(dim_raw, "dtype") and dim_raw.dtype.kind not in ("i", "u"):
            raise CheckpointValidationError(f"Non-integral dim scalar (dtype={dim_raw.dtype}) in checkpoint '{path}'.")

        cap_val = cap_raw.item() if hasattr(cap_raw, "item") else cap_raw
        dim_val = dim_raw.item() if hasattr(dim_raw, "item") else dim_raw

        if isinstance(cap_val, float) or isinstance(dim_val, float):
            raise CheckpointValidationError(f"Non-integral capacity/dim scalar in checkpoint '{path}'.")

        capacity = int(cap_val)
        dim = int(dim_val)
    except (ValueError, TypeError) as e:
        raise CheckpointValidationError(f"Invalid capacity/dim scalar in checkpoint '{path}': {e}")

    if capacity <= 0 or dim <= 0:
        raise CheckpointValidationError(f"Invalid dimensions (capacity={capacity}, dim={dim}) in '{path}'. Must be > 0.")

    if capacity > MAX_CHECKPOINT_CAPACITY:
        raise CheckpointValidationError(
            f"Checkpoint capacity {capacity} exceeds maximum allowed limit {MAX_CHECKPOINT_CAPACITY} in '{path}'."
        )
    if dim > MAX_CHECKPOINT_DIM:
        raise CheckpointValidationError(
            f"Checkpoint dimension {dim} exceeds maximum allowed limit {MAX_CHECKPOINT_DIM} in '{path}'."
        )

    # Defense-in-depth: Inspect uncompressed archive sizes before array allocation (SEC-01 / SEC-03)
    if hasattr(data, "zip") and data.zip is not None:
        max_matrix_bytes = (capacity * dim * 4) + 65536
        max_dense_bytes = (capacity * MAX_CHECKPOINT_DIM * 4) + 65536
        max_vector_bytes = (capacity * 8) + 65536
        max_text_bytes = min(capacity * 4096 + 65536, 256 * 1024 * 1024)

        for info in data.zip.infolist():
            fname = info.filename
            fsize = info.file_size
            if fname in ("keys.npy", "values.npy") and fsize > max_matrix_bytes:
                raise CheckpointValidationError(
                    f"Array '{fname}' uncompressed size {fsize} bytes exceeds expected bound {max_matrix_bytes} for capacity {capacity} and dim {dim} in '{path}'."
                )
            elif fname in ("dense_embs.npy", "anchor_vectors.npy") and fsize > max_dense_bytes:
                raise CheckpointValidationError(
                    f"Array '{fname}' uncompressed size {fsize} bytes exceeds expected bound {max_dense_bytes} for capacity {capacity} in '{path}'."
                )
            elif fname in ("confidences.npy", "timestamps.npy", "created_ats.npy", "update_counts.npy",
                           "valid_froms.npy", "valid_untils.npy", "contra_counts.npy",
                           "supersedes.npy", "superseded_by.npy", "has_dense.npy") and fsize > max_vector_bytes:
                raise CheckpointValidationError(
                    f"Array '{fname}' uncompressed size {fsize} bytes exceeds expected vector bound {max_vector_bytes} in '{path}'."
                )
            elif fname in ("texts.npy", "subjects.npy", "predicates.npy") and fsize > max_text_bytes:
                raise CheckpointValidationError(
                    f"Array '{fname}' uncompressed size {fsize} bytes exceeds expected text bound {max_text_bytes} in '{path}'."
                )

    keys = data["keys"]
    values = data["values"]
    confidences = data["confidences"]
    timestamps = data["timestamps"]
    created_ats = data["created_ats"]
    update_counts = data["update_counts"]

    n = keys.shape[0]
    if n > capacity:
        raise CheckpointValidationError(
            f"Slot count {n} exceeds stated capacity {capacity} in checkpoint '{path}'."
        )

    # 2. Shape and dimensional consistency
    if keys.ndim != 2 or keys.shape[1] != dim:
        raise CheckpointValidationError(
            f"Keys array shape {keys.shape} does not match expected (N, {dim}) in '{path}'."
        )
    if values.ndim != 2 or values.shape[1] != dim or values.shape[0] != n:
        raise CheckpointValidationError(
            f"Values array shape {values.shape} does not match keys count {n} and dim {dim} in '{path}'."
        )
    for name, arr in [("confidences", confidences), ("timestamps", timestamps),
                      ("created_ats", created_ats), ("update_counts", update_counts)]:
        if arr.ndim != 1 or arr.shape[0] != n:
            raise CheckpointValidationError(
                f"Array '{name}' has shape {arr.shape}, expected ({n},) in '{path}'."
            )

    # 3. Value ranges and sanity checks
    if np.isnan(keys).any() or np.isinf(keys).any():
        raise CheckpointValidationError(f"Keys array contains NaN or Inf in checkpoint '{path}'.")
    if np.isnan(values).any() or np.isinf(values).any():
        raise CheckpointValidationError(f"Values array contains NaN or Inf in checkpoint '{path}'.")
    if np.isnan(confidences).any() or np.isinf(confidences).any():
        raise CheckpointValidationError(f"Confidences array contains NaN or Inf in checkpoint '{path}'.")
    if (confidences < 0.0).any() or (confidences > 1.0).any():
        raise CheckpointValidationError(f"Confidence values out of bounds [0.0, 1.0] in '{path}'.")

    # 4. Optional / Schema v2 fields
    has_v2 = "schema_version" in data and int(data["schema_version"]) >= 2

    valid_froms = data["valid_froms"] if "valid_froms" in data else timestamps
    valid_untils = data["valid_untils"] if "valid_untils" in data else np.full(n, np.nan, dtype=np.float64)
    contra_counts = data["contra_counts"] if "contra_counts" in data else np.zeros(n, dtype=np.int32)
    supersedes = data["supersedes"] if "supersedes" in data else np.full(n, -1, dtype=np.int64)
    superseded_by = data["superseded_by"] if "superseded_by" in data else np.full(n, -1, dtype=np.int64)

    subjects = data["subjects"] if "subjects" in data else np.array([""] * n, dtype=str)
    predicates = data["predicates"] if "predicates" in data else np.array([""] * n, dtype=str)
    texts = data["texts"] if "texts" in data else np.array([""] * n, dtype=str)

    dense_embs = data["dense_embs"] if "dense_embs" in data else None
    has_dense = data["has_dense"] if "has_dense" in data else np.zeros(n, dtype=bool)

    # SEC-03: Validate dense_embs auxiliary array
    if dense_embs is not None:
        if dense_embs.ndim != 2:
            raise CheckpointValidationError(
                f"Array 'dense_embs' must be 2D (N, D_dense), got ndim={dense_embs.ndim} in '{path}'."
            )
        if dense_embs.shape[0] > 0:
            if dense_embs.shape[0] != n:
                raise CheckpointValidationError(
                    f"Array 'dense_embs' row count {dense_embs.shape[0]} does not match slot count {n} in '{path}'."
                )
            if dense_embs.shape[1] > MAX_CHECKPOINT_DIM or dense_embs.shape[1] <= 0:
                raise CheckpointValidationError(
                    f"Array 'dense_embs' column dimension {dense_embs.shape[1]} invalid in '{path}'."
                )
            if np.isnan(dense_embs).any() or np.isinf(dense_embs).any():
                raise CheckpointValidationError(
                    f"Array 'dense_embs' contains NaN or Inf values in '{path}'."
                )

    if has_dense.ndim != 1 or has_dense.shape[0] != n:
        raise CheckpointValidationError(
            f"Array 'has_dense' has shape {has_dense.shape}, expected ({n},) in '{path}'."
        )

    # 5. Checksum verification if present
    if "checksum" in data:
        expected_checksum = str(data["checksum"][0])
        computed_checksum = _compute_checksum(
            keys, values, confidences, timestamps, created_ats,
            update_counts, valid_froms, valid_untils
        )
        if expected_checksum != computed_checksum:
            raise CheckpointCorruptedError(
                f"Integrity checksum mismatch in '{path}' (expected {expected_checksum[:8]}..., computed {computed_checksum[:8]}...)."
            )

    # 6. Construct MemoryState cleanly
    memory = MemoryState(capacity=capacity, dim=dim)
    for i in range(n):
        v_until = float(valid_untils[i])
        v_until_val = None if (np.isnan(v_until) or v_until < 0.0) else v_until

        v_from = float(valid_froms[i])
        if v_until_val is not None and v_until_val < v_from:
            raise CheckpointValidationError(
                f"Slot {i} validity violation: valid_until ({v_until_val}) < valid_from ({v_from}) in '{path}'."
            )

        subj = str(subjects[i]) if (i < len(subjects) and subjects[i]) else None
        pred = str(predicates[i]) if (i < len(predicates) and predicates[i]) else None
        txt = str(texts[i]) if (i < len(texts) and texts[i]) else ""

        d_emb = None
        if dense_embs is not None and i < len(has_dense) and has_dense[i] and dense_embs.shape[0] > i:
            d_emb = dense_embs[i].copy()

        slot = MemorySlot(
            key=keys[i].copy(),
            value=values[i].copy(),
            confidence=float(confidences[i]),
            timestamp=float(timestamps[i]),
            created_at=float(created_ats[i]),
            update_count=int(update_counts[i]),
            debug_subject=subj,
            debug_predicate=pred,
            valid_from=v_from,
            valid_until=v_until_val,
            text=txt,
            dense_emb=d_emb,
            contra_count=int(contra_counts[i]),
            supersedes=int(supersedes[i]) if supersedes[i] >= 0 else None,
            superseded_by=int(superseded_by[i]) if superseded_by[i] >= 0 else None
        )
        memory.slots.append(slot)

    # Attach checkpoint metadata (telemetry, anchors) for callers like BlackHoleMemory
    memory._checkpoint_metadata = {}
    if "telemetry" in data:
        t_arr = data["telemetry"]
        if t_arr.ndim != 1 or t_arr.shape[0] < 6:
            raise CheckpointValidationError(f"Array 'telemetry' has invalid shape {t_arr.shape} in '{path}'.")
        memory._checkpoint_metadata["telemetry"] = t_arr.copy()

    # SEC-03: Validate anchor arrays strictly
    has_anchors = ("anchor_vectors" in data and "anchor_timestamps" in data and "anchor_slot_indices" in data)
    if has_anchors:
        a_vecs = data["anchor_vectors"]
        a_times = data["anchor_timestamps"]
        a_indices = data["anchor_slot_indices"]

        if a_vecs.ndim != 2:
            raise CheckpointValidationError(
                f"Array 'anchor_vectors' must be 2D, got ndim={a_vecs.ndim} in '{path}'."
            )
        num_anchors = a_vecs.shape[0]
        if a_times.ndim != 1 or a_times.shape[0] != num_anchors:
            raise CheckpointValidationError(
                f"Array 'anchor_timestamps' length {a_times.shape} does not match anchor count {num_anchors} in '{path}'."
            )
        if a_indices.ndim != 1 or a_indices.shape[0] != num_anchors:
            raise CheckpointValidationError(
                f"Array 'anchor_slot_indices' length {a_indices.shape} does not match anchor count {num_anchors} in '{path}'."
            )

        if num_anchors > 0:
            if a_vecs.shape[1] > MAX_CHECKPOINT_DIM or a_vecs.shape[1] <= 0:
                raise CheckpointValidationError(
                    f"Array 'anchor_vectors' column dimension {a_vecs.shape[1]} invalid in '{path}'."
                )
            if np.isnan(a_vecs).any() or np.isinf(a_vecs).any():
                raise CheckpointValidationError(
                    f"Array 'anchor_vectors' contains NaN or Inf in '{path}'."
                )
            if np.isnan(a_times).any() or np.isinf(a_times).any():
                raise CheckpointValidationError(
                    f"Array 'anchor_timestamps' contains NaN or Inf in '{path}'."
                )
            # Validate all anchor slot indices point to valid slots [0, n - 1]
            if (a_indices < 0).any() or (a_indices >= n).any():
                raise CheckpointValidationError(
                    f"Array 'anchor_slot_indices' references out-of-bounds slot index for slot count {n} in '{path}'."
                )

        memory._checkpoint_metadata["anchor_vectors"] = a_vecs.copy()
        memory._checkpoint_metadata["anchor_timestamps"] = a_times.copy()
        memory._checkpoint_metadata["anchor_slot_indices"] = a_indices.copy()

    return memory


def load_checkpoint(
    path: str,
    allow_backup_recovery: bool = True,
    allowed_base_dir: Optional[str] = None
) -> MemoryState:
    """
    Loads, validates, and reconstructs a MemoryState from a .npz checkpoint.

    Guarantees:
      1. Validates file existence, zip container integrity, array dimensions, and invariants.
      2. If target path is corrupted/invalid and a valid '<path>.bak' exists, automatically
         recovers from backup when allow_backup_recovery=True.
      3. Backward compatibility: loads v1 checkpoints cleanly with safe defaults.
      4. Fully restores temporal validity windows, slot activity states, and natural language metadata.
      5. Optional allowed_base_dir restricts checkpoint location to a designated directory.
    """
    abs_path = _validate_checkpoint_path(path, allowed_base_dir=allowed_base_dir)
    backup_path = f"{abs_path}.bak"

    with _FILE_LOCK:
        if not os.path.exists(abs_path):
            # If primary does not exist, check if a backup exists
            if allow_backup_recovery and os.path.exists(backup_path):
                return load_checkpoint(backup_path, allow_backup_recovery=False, allowed_base_dir=allowed_base_dir)
            raise CheckpointNotFoundError(f"Checkpoint file not found: '{path}'")

        try:
            with np.load(abs_path, allow_pickle=False) as data:
                return _validate_and_reconstruct(data, abs_path)
        except (zipfile.BadZipFile, EOFError, OSError, ValueError) as e:
            # File is truncated or malformed on disk or contains invalid format
            if allow_backup_recovery and os.path.exists(backup_path):
                try:
                    with np.load(backup_path, allow_pickle=False) as b_data:
                        return _validate_and_reconstruct(b_data, backup_path)
                except Exception:
                    pass
            raise CheckpointCorruptedError(f"Checkpoint file '{path}' is truncated or malformed: {e}") from e
        except (CheckpointValidationError, CheckpointCorruptedError) as e:
            # Invariant failure or checksum mismatch
            if allow_backup_recovery and os.path.exists(backup_path):
                try:
                    with np.load(backup_path, allow_pickle=False) as b_data:
                        return _validate_and_reconstruct(b_data, backup_path)
                except Exception:
                    pass
            raise
        except Exception as e:
            if allow_backup_recovery and os.path.exists(backup_path):
                try:
                    with np.load(backup_path, allow_pickle=False) as b_data:
                        return _validate_and_reconstruct(b_data, backup_path)
                except Exception:
                    pass
            raise CheckpointError(f"Unexpected error loading checkpoint '{path}': {e}") from e


def load_into_black_hole_memory(
    mem: Any,
    path: str,
    allow_backup_recovery: bool = True,
    allowed_base_dir: Optional[str] = None
) -> Any:
    """
    Loads checkpoint into an existing or newly initialized BlackHoleMemory instance,
    rebuilding all auxiliary stores (slot_texts, slot_dense_embs, slot_contra_counts,
    slot_contra_anchors, bm25_index).
    """
    loaded_memory = load_checkpoint(
        path,
        allow_backup_recovery=allow_backup_recovery,
        allowed_base_dir=allowed_base_dir
    )
    
    with mem._lock:
        mem.memory = loaded_memory
        mem.slot_texts.clear()
        mem.slot_dense_embs.clear()
        mem.slot_contra_counts.clear()
        mem.slot_contra_anchors.clear()

        # Re-initialize bm25
        from benchmarks.phase22_experiment import LocalBM25Index
        mem.bm25_index = LocalBM25Index()

        for slot in mem.memory.slots:
            slot_id = id(slot)
            mem.slot_texts[slot_id] = slot.text
            mem.slot_contra_counts[slot_id] = slot.contra_count

            if slot.dense_emb is not None:
                mem.slot_dense_embs.append(slot.dense_emb.copy())
            elif slot.text and mem.dense_encoder is not None:
                mem.slot_dense_embs.append(mem.dense_encoder.encode(slot.text))
            else:
                mem.slot_dense_embs.append(np.zeros(384, dtype=np.float32))

            if slot.text:
                mem.bm25_index.add_document(slot_id, slot.text)

        # Restore telemetry counters if present
        meta = getattr(loaded_memory, "_checkpoint_metadata", {})
        if "telemetry" in meta and hasattr(mem, "total_absorbed"):
            t = meta["telemetry"]
            if len(t) >= 6:
                mem.total_absorbed = int(t[0])
                mem.total_updates = int(t[1])
                mem.total_inserts = int(t[2])
                mem.total_temporal_updates = int(t[3])
                mem.total_evictions = int(t[4])
                mem.total_nli_calls = int(t[5])

        # Restore contradiction anchors if present
        if "anchor_vectors" in meta and hasattr(mem, "slot_contra_anchors"):
            a_vecs = meta.get("anchor_vectors")
            a_times = meta.get("anchor_timestamps")
            a_indices = meta.get("anchor_slot_indices")
            if a_vecs is not None and len(a_vecs) > 0 and len(a_vecs) == len(a_times) == len(a_indices):
                for idx_a in range(len(a_vecs)):
                    s_idx = int(a_indices[idx_a])
                    if 0 <= s_idx < len(mem.memory.slots):
                        target_s_id = id(mem.memory.slots[s_idx])
                        if target_s_id not in mem.slot_contra_anchors:
                            mem.slot_contra_anchors[target_s_id] = []
                        mem.slot_contra_anchors[target_s_id].append((a_vecs[idx_a].copy(), float(a_times[idx_a])))

    return mem
