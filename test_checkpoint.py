import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb
from phase2.recall import recall
from phase2.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_roundtrip_preserves_recall(tmp_path):
    mem = MemoryState(capacity=10)
    absorb(mem, Candidate("person_1", "likes", "apples", timestamp=1))
    absorb(mem, Candidate("person_2", "likes", "oranges", timestamp=2))
    absorb(mem, Candidate("person_1", "likes", "apples", timestamp=3))  # reinforce

    r_before = recall(mem, "person_1", "likes")

    path = str(tmp_path / "checkpoint.npz")
    save_checkpoint(mem, path)
    restored = load_checkpoint(path)

    r_after = recall(restored, "person_1", "likes")

    assert r_before["found"] and r_after["found"]
    assert r_before["similarity"] == r_after["similarity"]
    assert restored.slots[r_after["slot_index"]].update_count == \
        mem.slots[r_before["slot_index"]].update_count


def test_checkpoint_survives_program_restart_simulation(tmp_path):
    """Simulates 'program restart' by building memory in one MemoryState,
    saving, then deliberately deleting the in-memory object and loading
    fresh from disk only."""
    mem = MemoryState(capacity=5)
    for i in range(5):
        absorb(mem, Candidate(f"person_{i}", "likes", f"object_{i}", timestamp=i))

    path = str(tmp_path / "restart_test.npz")
    save_checkpoint(mem, path)
    n_slots_before = len(mem.slots)
    del mem  # simulate process exit

    restored = load_checkpoint(path)
    assert len(restored.slots) == n_slots_before
    r = recall(restored, "person_3", "likes")
    assert r["found"]


def test_checkpoint_file_size_is_measurable(tmp_path):
    mem = MemoryState(capacity=100)
    for i in range(50):
        absorb(mem, Candidate(f"person_{i}", "likes", f"object_{i}", timestamp=i))
    path = str(tmp_path / "size_test.npz")
    save_checkpoint(mem, path)
    size = os.path.getsize(path)
    assert size > 0
    # sanity: should roughly track footprint_bytes (npz has some overhead,
    # so we only assert same order of magnitude, not exact equality)
    fp = mem.footprint_bytes()["total_bytes"]
    assert 0.5 * fp < size < 3 * fp


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
