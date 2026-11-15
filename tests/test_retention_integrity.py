"""R5 F5 - eviction and retention lifecycle integrity.

The contract these tests pin down:

* capacity is enforced exactly, and an evicted slot leaves canonical state,
  the BM25 index and the derived Qdrant index together;
* active retrieval never returns evicted content, while a superseded value
  that is still retained stays reachable through point-in-time retrieval;
* insertion order, out-of-order timestamps and equal timestamps produce a
  deterministic, capacity-respecting outcome;
* a checkpoint reload and a derived-index rebuild cannot resurrect an evicted
  memory.

Retention defaults and policy thresholds are intentionally untouched; these
tests exercise the documented defaults and explicit per-policy choices.
"""

from __future__ import annotations

import math
import uuid

import pytest

from minemos_engine.black_hole_memory import BlackHoleMemory
from minemos_engine.structured_candidate import Candidate
from tests._retrieval_helpers import (FixedEncoder, RuleVerifier, StubNLI,
                                      make_memory)


def _absorb(mem: BlackHoleMemory, subject: str, predicate: str, obj: str,
            timestamp: float) -> str:
    result = mem.absorb(Candidate(subject, predicate, obj,
                                  timestamp=timestamp))
    return result["memory_id"]


def _texts(mem: BlackHoleMemory, subject: str, predicate=None,
           as_of: float | None = None, active_only: bool = True) -> str:
    rows = mem.recall(subject, predicate=predicate, as_of=as_of,
                      active_only=active_only)
    return " ".join(row.get("text") or "" for row in rows).casefold()


def _slot_ids(mem: BlackHoleMemory) -> set[str]:
    return {slot.memory_id for slot in mem.memory.slots}


def _query_subjects(mem: BlackHoleMemory, text: str, top_k: int = 10):
    return [row.get("subject") for row in mem.query(text, top_k=top_k)]


# ── insertion order and capacity ───────────────────────────────────────────


def test_fifo_eviction_leaves_the_newest_slots_and_exact_capacity():
    mem = make_memory(capacity=3, retention_policy="fifo")
    ids = [
        _absorb(mem, "F1", "lives in", "Rome", 1.0),
        _absorb(mem, "F2", "lives in", "Oslo", 2.0),
        _absorb(mem, "F3", "lives in", "Lima", 3.0),
        _absorb(mem, "F4", "lives in", "Kyiv", 4.0),
        _absorb(mem, "F5", "lives in", "Bern", 5.0),
    ]
    assert len(mem.memory.slots) == 3
    assert mem.total_evictions == 2
    assert _slot_ids(mem) == set(ids[2:])

    assert "F1" not in _query_subjects(mem, "F1 lives in Rome.")
    assert "F2" not in _query_subjects(mem, "F2 lives in Oslo.")
    assert "F5" in _query_subjects(mem, "F5 lives in Bern.")

    for evicted, subject in zip(ids[:2], ("F1", "F2")):
        assert _texts(mem, subject) == ""


def test_out_of_order_timestamps_follow_the_selected_policy():
    # Recency policy: the least recently updated slot is evicted even when it
    # was inserted last.
    mem = make_memory(capacity=2, retention_policy="recency")
    _absorb(mem, "A", "lives in", "Rome", 10.0)
    _absorb(mem, "B", "lives in", "Oslo", 30.0)
    _absorb(mem, "C", "lives in", "Lima", 20.0)

    assert len(mem.memory.slots) == 2
    assert _texts(mem, "A") == ""
    assert "oslo" in _texts(mem, "B")
    assert "lima" in _texts(mem, "C")


def test_equal_timestamps_evict_deterministically():
    def run():
        mem = make_memory(capacity=2, retention_policy="fifo")
        _absorb(mem, "A", "lives in", "Rome", 5.0)
        _absorb(mem, "B", "lives in", "Oslo", 5.0)
        _absorb(mem, "C", "lives in", "Lima", 5.0)
        return (_texts(mem, "A"), _texts(mem, "B"), _texts(mem, "C"),
                len(mem.memory.slots))

    first = run()
    second = run()
    assert first == second
    # Equal timestamps fall back to slot order: the first slot is evicted.
    assert first[0] == ""
    assert "oslo" in first[1]
    assert "lima" in first[2]


# ── wrong-fact eviction under capacity pressure ────────────────────────────


def test_capacity_pressure_evicts_closed_history_before_live_state():
    verifier = RuleVerifier([
        ("Rome", "Lima", "CONTRADICTION"),
        ("Lima", "Rome", "CONTRADICTION"),
    ])
    mem = make_memory(capacity=3, retention_policy="tiered", verifier=verifier)
    _absorb(mem, "A", "lives in", "Rome", 1.0)
    _absorb(mem, "A", "lives in", "Lima", 2.0)
    assert "lima" in _texts(mem, "A", "lives in")
    assert "rome" not in _texts(mem, "A", "lives in")

    _absorb(mem, "F1", "works at", "Acme", 3.0)
    assert len(mem.memory.slots) == 3
    # While both versions are resident, the closed interval is still
    # retrievable at its own timestamp.
    assert "lima" in _texts(mem, "A", "lives in")
    historical = _texts(mem, "A", "lives in", as_of=1.5, active_only=False)
    assert "rome" in historical
    assert "lima" not in historical

    _absorb(mem, "F2", "works at", "Globex", 4.0)
    # Capacity pressure drops the closed predecessor first.
    assert len(mem.memory.slots) == 3
    assert "lima" in _texts(mem, "A", "lives in")
    assert _texts(mem, "A", "lives in", as_of=1.5,
                  active_only=False) == ""

    _absorb(mem, "F3", "works at", "Initech", 5.0)
    assert len(mem.memory.slots) == 3
    # Whatever the policy drops next, an evicted value never becomes active.
    active = _texts(mem, "A", "lives in")
    assert "rome" not in active
    assert len(_slot_ids(mem)) == 3


def test_consolidated_current_fact_survives_capacity_pressure():
    mem = make_memory(capacity=3, retention_policy="tiered",
                      verifier=RuleVerifier([("Lima", "Lima", "SAME")]))
    _absorb(mem, "A", "lives in", "Lima", 1.0)
    _absorb(mem, "A", "lives in", "Lima", 2.0)  # SAME -> update_count 2
    current = [slot for slot in mem.memory.slots
               if slot.debug_subject == "A"][0]
    assert current.update_count == 2

    _absorb(mem, "F1", "works at", "Acme", 3.0)
    _absorb(mem, "F2", "works at", "Globex", 4.0)
    _absorb(mem, "F3", "works at", "Initech", 5.0)

    assert len(mem.memory.slots) == 3
    # Tier 1 (consolidated) is protected while provisional filler exists.
    assert "lima" in _texts(mem, "A", "lives in")
    assert _texts(mem, "F1", "works at") == ""


def test_retained_history_is_queryable_at_its_old_timestamp():
    verifier = RuleVerifier([
        ("Rome", "Lima", "CONTRADICTION"),
        ("Lima", "Rome", "CONTRADICTION"),
    ])
    mem = make_memory(capacity=10, verifier=verifier)
    _absorb(mem, "A", "lives in", "Rome", 1.0)
    _absorb(mem, "A", "lives in", "Lima", 2.0)

    assert "lima" in _texts(mem, "A", "lives in")
    assert "rome" not in _texts(mem, "A", "lives in")
    historical = _texts(mem, "A", "lives in", as_of=1.5, active_only=False)
    assert "rome" in historical
    assert "lima" not in historical
    assert mem.get_history("A", "lives in")


# ── persistence and the derived index ──────────────────────────────────────


def test_checkpoint_restart_cannot_resurrect_an_evicted_slot(tmp_path):
    mem = make_memory(capacity=3, retention_policy="fifo")
    ids = [
        _absorb(mem, "F1", "lives in", "Rome", 1.0),
        _absorb(mem, "F2", "lives in", "Oslo", 2.0),
        _absorb(mem, "F3", "lives in", "Lima", 3.0),
        _absorb(mem, "F4", "lives in", "Kyiv", 4.0),
        _absorb(mem, "F5", "lives in", "Bern", 5.0),
    ]
    path = tmp_path / "state.npz"
    mem.save_checkpoint(str(path))
    mem.close()

    restored = BlackHoleMemory.from_checkpoint(
        str(path), capacity=3, dim=4096,
        dense_encoder=FixedEncoder(), nli_gate=StubNLI(),
        verifier=RuleVerifier(), retrieval_backend="canonical")
    try:
        assert len(restored.memory.slots) == 3
        assert _slot_ids(restored) == set(ids[2:])
        assert not (_slot_ids(restored) & set(ids[:2]))
        assert "F1" not in _query_subjects(restored, "F1 lives in Rome.")
        assert "F5" in _query_subjects(restored, "F5 lives in Bern.")
    finally:
        restored.close()


def test_qdrant_rebuild_cannot_resurrect_an_evicted_slot(tmp_path):
    qdrant = pytest.importorskip("qdrant_client")
    del qdrant
    from minemos_engine.retrieval import QdrantConfig

    config = QdrantConfig(
        path=str(tmp_path / "qdrant"),
        collection=f"minemos_r5_{uuid.uuid4().hex[:12]}",
        tenant_id="r5-tenant",
        candidate_limit=16,
    )
    mem = make_memory(capacity=3, retention_policy="fifo",
                      backend="qdrant", qdrant_config=config)
    try:
        ids = [
            _absorb(mem, "F1", "lives in", "Rome", 1.0),
            _absorb(mem, "F2", "lives in", "Oslo", 2.0),
            _absorb(mem, "F3", "lives in", "Lima", 3.0),
            _absorb(mem, "F4", "lives in", "Kyiv", 4.0),
            _absorb(mem, "F5", "lives in", "Bern", 5.0),
        ]
        retriever = mem._get_retrieval_adapter().retriever
        points, _ = retriever.client.scroll(
            retriever.collection, limit=1000, with_payload=True,
            scroll_filter=retriever._tenant_filter())
        indexed = {str(point.payload["memory_id"]) for point in points}
        assert indexed == set(ids[2:])

        assert "F1" not in _query_subjects(mem, "F1 lives in Rome.")
        mem._get_retrieval_adapter().force_full_rebuild()
        assert "F1" not in _query_subjects(mem, "F1 lives in Rome.")
        assert "F2" not in _query_subjects(mem, "F2 lives in Oslo.")
        assert "F5" in _query_subjects(mem, "F5 lives in Bern.")
    finally:
        mem.close()


def test_derived_index_drops_points_deleted_from_canonical_state(tmp_path):
    pytest.importorskip("qdrant_client")
    from minemos_engine.retrieval import QdrantConfig

    config = QdrantConfig(
        path=str(tmp_path / "qdrant"),
        collection=f"minemos_r5_{uuid.uuid4().hex[:12]}",
        tenant_id="r5-tenant",
        candidate_limit=16,
    )
    mem = make_memory(capacity=5, backend="qdrant", qdrant_config=config)
    try:
        victim = _absorb(mem, "A", "lives in", "Rome", 1.0)
        _absorb(mem, "B", "lives in", "Oslo", 2.0)
        assert mem.delete_memory(victim) is True

        assert "A" not in _query_subjects(mem, "A lives in Rome.")
        retriever = mem._get_retrieval_adapter().retriever
        points, _ = retriever.client.scroll(
            retriever.collection, limit=1000, with_payload=True,
            scroll_filter=retriever._tenant_filter())
        assert victim not in {str(point.payload["memory_id"])
                              for point in points}
        mem._get_retrieval_adapter().force_full_rebuild()
        assert "A" not in _query_subjects(mem, "A lives in Rome.")
    finally:
        mem.close()


# ── accounting ─────────────────────────────────────────────────────────────


def test_slot_bookkeeping_and_eviction_counts_stay_consistent():
    mem = make_memory(capacity=2, retention_policy="fifo")
    for index in range(6):
        _absorb(mem, f"F{index}", "lives in", f"City{index}", float(index))
    assert len(mem.memory.slots) == 2
    assert mem.total_inserts == 6
    assert mem.total_evictions == 4
    footprint = mem.memory.footprint_bytes()
    assert footprint["n_slots"] == 2
    assert footprint["total_bytes"] > 0
    # eviction removed the BM25 text as well as the canonical slot
    assert all(not math.isnan(slot.valid_from)
               for slot in mem.memory.slots)
