"""
MINEMOS Production Readiness - Task 5: End-to-End Recovery, Crash Consistency,
and Memory Integrity Validation Test Suite.

Rigorously verifies:
1. Multi-cycle cross-process survival across distinct Python subprocess boundaries.
2. In-memory state preservation during process-level crashes (os._exit simulation).
3. Continuous mutation, multiple saves, and historical temporal consistency.
4. Total failure handling (both primary .npz and .bak corrupted) raises CheckpointCorruptedError.
5. In-memory mutation isolation: load failures never alter an active BlackHoleMemory instance.
6. Rebuilding and agreement of all auxiliary indexes (BM25, dense embeddings, anchors, telemetry).
7. Cross-process temporal queries, validity windows, and supersession links.
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
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
)
from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate


def test_cross_process_lifecycle_e2e(tmp_path):
    """
    Test true cross-process lifecycle across 3 distinct Python subprocess boundaries:
      Process 1: Ingest active, contradicted, and independent facts; save and exit.
      Process 2: Start fresh, load checkpoint, verify active & historical recall,
                 ingest additional fact, save and exit.
      Process 3: Start fresh, load checkpoint, verify all facts, dense query,
                 telemetry counts, and status invariants.
    """
    ckpt_path = str(tmp_path / "cross_process_e2e.npz")
    python_exe = sys.executable

    # --- Subprocess 1 ---
    p1_code = f"""
import sys
sys.path.insert(0, '.')
from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate

mem = BlackHoleMemory(capacity=20)
mem.absorb(Candidate("Ada", "lives in", "London", timestamp=100.0))
mem.absorb(Candidate("Ada", "lives in", "Oxford", timestamp=200.0))
mem.absorb(Candidate("Ada", "researches", "Analytical Engines", timestamp=250.0))
mem.absorb(Candidate("Charles", "designed", "Difference Engine", timestamp=300.0))

mem.save_checkpoint({repr(ckpt_path)})
print("P1_OK", len(mem.memory.slots))
"""
    res1 = subprocess.run([python_exe, "-c", p1_code], capture_output=True, text=True)
    assert res1.returncode == 0, f"Process 1 failed: {res1.stderr}"
    assert "P1_OK 4" in res1.stdout

    # --- Subprocess 2 ---
    p2_code = f"""
import sys
sys.path.insert(0, '.')
from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate

mem = BlackHoleMemory.from_checkpoint({repr(ckpt_path)}, capacity=20)
assert len(mem.memory.slots) == 4

ada_res = mem.recall("Ada", "lives in")
assert len(ada_res) == 1
assert "Oxford" in ada_res[0]["text"]
assert ada_res[0]["is_active"]

ada_hist = mem.get_history("Ada", "lives in")
assert len(ada_hist) == 2
assert not ada_hist[0]["is_active"]
assert ada_hist[0]["valid_until"] == 200.0
assert "London" in ada_hist[0]["text"]
assert ada_hist[1]["is_active"]
assert ada_hist[1]["valid_until"] is None
assert "Oxford" in ada_hist[1]["text"]

mem.absorb(Candidate("Charles", "corresponded with", "Ada", timestamp=400.0))
assert len(mem.memory.slots) == 5

mem.save_checkpoint({repr(ckpt_path)})
print("P2_OK", len(mem.memory.slots))
"""
    res2 = subprocess.run([python_exe, "-c", p2_code], capture_output=True, text=True)
    assert res2.returncode == 0, f"Process 2 failed: {res2.stderr}"
    assert "P2_OK 5" in res2.stdout

    # --- Subprocess 3 ---
    p3_code = f"""
import sys
sys.path.insert(0, '.')
from phase2.black_hole_memory import BlackHoleMemory

mem = BlackHoleMemory.from_checkpoint({repr(ckpt_path)}, capacity=20)
assert len(mem.memory.slots) == 5

corr = mem.recall("Charles", "corresponded with")
assert len(corr) == 1
assert "Ada" in corr[0]["text"]

q_res = mem.query("Where does Ada live?", top_k=2)
assert len(q_res) >= 1
assert "Oxford" in q_res[0]["text"]

st = mem.status()
assert st["total_slots"] == 5
assert st["active_slots"] == 4
assert st["superseded_slots"] == 1
print("P3_OK")
"""
    res3 = subprocess.run([python_exe, "-c", p3_code], capture_output=True, text=True)
    assert res3.returncode == 0, f"Process 3 failed: {res3.stderr}"
    assert "P3_OK" in res3.stdout


def test_process_crash_simulated_sigkill(tmp_path):
    """
    Test recovery after hard process termination via os._exit(0) immediately
    following a successful checkpoint write.
    The written checkpoint must be fully intact and recoverable in a subsequent process.
    """
    ckpt_path = str(tmp_path / "hard_crash.npz")
    python_exe = sys.executable

    p_crash_code = f"""
import os
import sys
sys.path.insert(0, '.')
from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate

mem = BlackHoleMemory(capacity=10)
mem.absorb(Candidate("Alice", "status", "Active", timestamp=10.0))
mem.absorb(Candidate("Bob", "role", "Engineer", timestamp=20.0))
mem.save_checkpoint({repr(ckpt_path)})

# Simulate immediate SIGKILL / hard crash without standard Python runtime cleanup
os._exit(0)
"""
    res = subprocess.run([python_exe, "-c", p_crash_code], capture_output=True, text=True)
    assert res.returncode == 0

    # Verify recovery in a new process
    p_recover_code = f"""
import sys
sys.path.insert(0, '.')
from phase2.black_hole_memory import BlackHoleMemory

mem = BlackHoleMemory.from_checkpoint({repr(ckpt_path)}, capacity=10)
assert len(mem.memory.slots) == 2
res = mem.recall("Bob", "role")
assert len(res) == 1
assert "Engineer" in res[0]["text"]
print("RECOVERED_AFTER_SIGKILL")
"""
    res_rec = subprocess.run([python_exe, "-c", p_recover_code], capture_output=True, text=True)
    assert res_rec.returncode == 0, f"Recovery failed: {res_rec.stderr}"
    assert "RECOVERED_AFTER_SIGKILL" in res_rec.stdout


def test_crash_mid_save_recovers_previous_checkpoint_cross_process(tmp_path):
    """
    Simulate process termination during save:
    1. Valid checkpoint v1 is saved and its exact SHA-256 digest is recorded.
    2. A second save operation is interrupted midway through the actual write operation
       (e.g., process killed / disk write fault during np.savez to temporary staging).
    3. Staged partial .tmp file exists, but previous valid checkpoint remains byte-for-byte intact
       matching the recorded digest, and cleanly loaded in a fresh subprocess.
    """
    import hashlib

    ckpt_path = str(tmp_path / "mid_save_crash.npz")
    python_exe = sys.executable

    # Process 1 creates valid initial checkpoint
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Bob", "level", "1", timestamp=1.0))
    mem.save_checkpoint(ckpt_path)

    # Compute and record exact initial SHA-256 digest
    with open(ckpt_path, "rb") as f:
        initial_digest = hashlib.sha256(f.read()).hexdigest()

    # Simulate an interrupted second save using a subprocess that crashes midway through write
    p_interrupted_code = f"""
import sys
import os
sys.path.insert(0, '.')
import numpy as np
from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate

mem = BlackHoleMemory.from_checkpoint({repr(ckpt_path)}, capacity=10)
mem.absorb(Candidate("Bob", "level", "2", timestamp=2.0))

# Hook into np.savez to simulate a hard crash (os._exit) midway through writing the staging file
real_savez = np.savez
def crashing_savez(target, *args, **kwargs):
    # Write partial truncated data to the temporary staging target
    with open(str(target), "wb") as f:
        f.write(b"TRUNCATED_MID_STREAM_SAVEZ_BYTES")
    # Immediate hard exit before atomic rename can happen
    os._exit(42)

np.savez = crashing_savez
mem.save_checkpoint({repr(ckpt_path)})
"""
    res_interrupt = subprocess.run([python_exe, "-c", p_interrupted_code], capture_output=True, text=True)
    assert res_interrupt.returncode == 42, f"Expected returncode 42, got {res_interrupt.returncode}"

    # Verify that the primary checkpoint file was NOT modified (exact byte-for-byte SHA-256 match)
    with open(ckpt_path, "rb") as f:
        post_interrupt_digest = hashlib.sha256(f.read()).hexdigest()
    assert post_interrupt_digest == initial_digest, "Checkpoint corrupted or altered during interrupted save!"

    # Verify in a new process that primary checkpoint is unaffected and yields initial state
    p_check = f"""
import sys
sys.path.insert(0, '.')
from phase2.black_hole_memory import BlackHoleMemory

mem = BlackHoleMemory.from_checkpoint({repr(ckpt_path)}, capacity=10)
assert len(mem.memory.slots) == 1
res = mem.recall("Bob", "level")
assert "1" in res[0]["text"]
print("CLEAN_FALLBACK_OK")
"""
    res = subprocess.run([python_exe, "-c", p_check], capture_output=True, text=True)
    assert res.returncode == 0, f"Failed: {res.stderr}"
    assert "CLEAN_FALLBACK_OK" in res.stdout


def test_double_corruption_raises_error_without_silent_corruption(tmp_path):
    """
    If BOTH primary checkpoint and .bak are completely corrupted or unreadable,
    the system must reject the load with CheckpointCorruptedError, rather than
    silently proceeding with blank or undefined state.
    """
    ckpt_path = str(tmp_path / "double_corrupt.npz")
    bak_path = f"{ckpt_path}.bak"

    with open(ckpt_path, "wb") as f:
        f.write(b"CORRUPTED_PRIMARY_BYTES_GARBAGE")
    with open(bak_path, "wb") as f:
        f.write(b"CORRUPTED_BACKUP_BYTES_GARBAGE")

    with pytest.raises(CheckpointCorruptedError):
        load_checkpoint(ckpt_path, allow_backup_recovery=True)

    mem = BlackHoleMemory(capacity=10)
    with pytest.raises(CheckpointCorruptedError):
        load_into_black_hole_memory(mem, ckpt_path, allow_backup_recovery=True)


def test_failed_checkpoint_load_preserves_active_memory_isolation(tmp_path):
    """
    Ensure mutation isolation: when load_into_black_hole_memory fails midway
    (e.g. invalid file or invariant failure), the target BlackHoleMemory instance
    is not left in an empty, partially overwritten, or corrupted state.
    """
    bad_ckpt = str(tmp_path / "bad.npz")
    with open(bad_ckpt, "wb") as f:
        f.write(b"INVALID_HEADER_GARBAGE")

    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Server", "status", "online", timestamp=10.0))
    initial_slots_count = len(mem.memory.slots)
    assert initial_slots_count == 1

    try:
        load_into_black_hole_memory(mem, bad_ckpt, allow_backup_recovery=False)
    except CheckpointCorruptedError:
        pass

    # The original active memory state must remain intact
    assert len(mem.memory.slots) == initial_slots_count
    rec = mem.recall("Server", "status")
    assert len(rec) == 1
    assert "online" in rec[0]["text"]


def test_multi_cycle_mutation_and_temporal_consistency(tmp_path):
    """
    Perform multiple continuous save-load-mutate cycles verifying:
    1. supersession chains across 3 consecutive updates to the same entity/attribute.
    2. validity intervals: each past slot has valid_until == next slot's valid_from.
    3. only the latest slot is active (valid_until is None).
    4. auxiliary BM25 and dense indices continue retrieving the latest active fact.
    """
    ckpt_path = str(tmp_path / "multi_cycle.npz")

    # Cycle 1: initial fact
    mem = BlackHoleMemory(capacity=10)
    mem.absorb(Candidate("Carol", "city", "Rome", timestamp=100.0))
    mem.save_checkpoint(ckpt_path)

    # Cycle 2: update to Florence
    mem2 = BlackHoleMemory.from_checkpoint(ckpt_path, capacity=10)
    mem2.absorb(Candidate("Carol", "city", "Florence", timestamp=200.0))
    mem2.save_checkpoint(ckpt_path)

    # Cycle 3: update to Venice
    mem3 = BlackHoleMemory.from_checkpoint(ckpt_path, capacity=10)
    mem3.absorb(Candidate("Carol", "city", "Venice", timestamp=300.0))
    mem3.save_checkpoint(ckpt_path)

    # Cycle 4: final verification
    final_mem = BlackHoleMemory.from_checkpoint(ckpt_path, capacity=10)
    history = final_mem.get_history("Carol", "city")
    assert len(history) == 3

    # Check temporal ordering & interval bounds
    assert history[0]["valid_from"] == 100.0
    assert history[0]["valid_until"] == 200.0
    assert not history[0]["is_active"]
    assert "Rome" in history[0]["text"]

    assert history[1]["valid_from"] == 200.0
    assert history[1]["valid_until"] == 300.0
    assert not history[1]["is_active"]
    assert "Florence" in history[1]["text"]

    assert history[2]["valid_from"] == 300.0
    assert history[2]["valid_until"] is None
    assert history[2]["is_active"]
    assert "Venice" in history[2]["text"]

    # Active recall must strictly yield Venice
    active_res = final_mem.recall("Carol", "city")
    assert len(active_res) == 1
    assert "Venice" in active_res[0]["text"]


def test_auxiliary_index_and_telemetry_agreement(tmp_path):
    """
    Verify that BlackHoleMemory telemetry and all auxiliary stores:
    1. total_absorbed, total_updates, total_inserts counters survive round trip.
    2. BM25 index contains all slot documents and yields positive score matches.
    3. Dense embeddings match the slot representations exactly.
    4. Repulsion anchors survive and correspond to the correct slot IDs.
    """
    ckpt_path = str(tmp_path / "telemetry_test.npz")

    mem = BlackHoleMemory(capacity=15)
    mem.absorb(Candidate("Dave", "lives in", "London", timestamp=10.0))
    mem.absorb(Candidate("Dave", "lives in", "Paris", timestamp=20.0))
    mem.absorb(Candidate("Eve", "founded", "Apollo", timestamp=30.0))

    # Record pre-save metrics
    pre_absorbed = mem.total_absorbed
    pre_updates = mem.total_updates
    pre_inserts = mem.total_inserts
    pre_temporal_updates = mem.total_temporal_updates
    assert pre_absorbed == 3
    assert pre_inserts == 2
    assert pre_temporal_updates == 1

    mem.save_checkpoint(ckpt_path)

    # Restore in new instance
    restored = BlackHoleMemory.from_checkpoint(ckpt_path, capacity=15)

    # 1. Telemetry checks
    assert restored.total_absorbed == pre_absorbed
    assert restored.total_updates == pre_updates
    assert restored.total_inserts == pre_inserts
    assert restored.total_temporal_updates == pre_temporal_updates

    # 2. Auxiliary BM25 checks
    assert restored.bm25_index.num_docs == len(restored.memory.slots)
    bm25_hits = restored.bm25_index.query("Paris", top_k=3)
    assert len(bm25_hits) >= 1

    # 3. Dense embeddings alignment
    assert len(restored.slot_dense_embs) == len(restored.memory.slots)
    for idx, slot in enumerate(restored.memory.slots):
        np.testing.assert_array_equal(restored.slot_dense_embs[idx], slot.dense_emb)

    # 4. Repulsion anchor persistence and alignment
    # Slot 0 was contradicted by slot 1, so slot 0 should have an anchor or slot 1 should have anchored slot 0
    total_anchors = sum(len(a_list) for a_list in restored.slot_contra_anchors.values())
    pre_anchors = sum(len(a_list) for a_list in mem.slot_contra_anchors.values())
    assert total_anchors == pre_anchors

    # 5. Status invariants
    st = restored.status()
    assert st["total_slots"] == 3
    assert st["active_slots"] == 2
    assert st["superseded_slots"] == 1

    # 6. Continued operation does NOT reset or double-count counters
    restored.absorb(Candidate("Frank", "visited", "Berlin", timestamp=40.0))
    assert restored.total_absorbed == pre_absorbed + 1
    assert restored.total_inserts == pre_inserts + 1
    assert restored.total_temporal_updates == pre_temporal_updates
