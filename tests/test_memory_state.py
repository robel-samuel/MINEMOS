import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate, encode_full, encode_query
from phase2.memory_state import MemoryState, CapacityExceeded
from phase2.memory_update import absorb, address, novelty, cosine_similarity
from phase2.recall import recall


def test_empty_memory_novelty_is_1():
    mem = MemoryState(capacity=10)
    idx, sim = address(mem, encode_full(Candidate("user", "uses", "Python")))
    assert idx is None
    n = novelty(sim, memory_empty=True)
    assert n == 1.0


def test_distinct_facts_get_distinct_slots():
    mem = MemoryState(capacity=10)
    absorb(mem, Candidate("person_1", "likes", "apples", timestamp=1))
    absorb(mem, Candidate("person_2", "likes", "oranges", timestamp=2))
    assert len(mem.slots) == 2


def test_identical_repeated_fact_reinforces_not_duplicates():
    mem = MemoryState(capacity=10)
    for t in range(5):
        result = absorb(mem, Candidate("user", "uses", "Python", timestamp=t))
    assert len(mem.slots) == 1
    assert mem.slots[0].update_count == 5


def test_capacity_enforced_with_fifo_eviction():
    mem = MemoryState(capacity=3)
    for i in range(3):
        absorb(mem, Candidate(f"person_{i}", "likes", f"object_{i}", timestamp=i))
    assert len(mem.slots) == 3
    # 4th distinct fact must evict, not silently grow past capacity
    result = absorb(mem, Candidate("person_3", "likes", "object_3", timestamp=3))
    assert result["action"] == "evict_insert"
    assert len(mem.slots) == 3


def test_recall_finds_correct_slot():
    mem = MemoryState(capacity=10)
    absorb(mem, Candidate("person_1", "likes", "apples", timestamp=1))
    absorb(mem, Candidate("person_2", "likes", "oranges", timestamp=2))
    r = recall(mem, "person_1", "likes")
    assert r["found"]
    assert mem.slots[r["slot_index"]].debug_subject == "person_1"


def test_footprint_bytes_accounts_all_components():
    mem = MemoryState(capacity=10)
    absorb(mem, Candidate("user", "uses", "Python", timestamp=1))
    fp = mem.footprint_bytes()
    assert fp["n_slots"] == 1
    assert fp["total_bytes"] == (fp["key_bytes"] + fp["value_bytes"]
                                  + fp["confidence_bytes"] + fp["timestamp_bytes"]
                                  + fp["metadata_bytes"])
    assert fp["total_bytes"] > fp["payload_bytes"]  # total must include more than just values


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
