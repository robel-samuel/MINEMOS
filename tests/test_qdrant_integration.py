"""Integration tests for the optional Qdrant retrieval backend.

Every test uses a real Qdrant client.  By default that is Qdrant's persistent
embedded mode (a real local database directory, not a mock).  Set
``MINEMOS_QDRANT_URL`` to run the same suite against a Qdrant server; each test
uses a unique collection and tenant.
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import replace

import pytest

pytest.importorskip("qdrant_client")

from phase2.retrieval import QdrantConfig
from phase2.retrieval.qdrant import _point_id
from phase2.structured_candidate import Candidate
from tests._retrieval_helpers import (QUERY, RuleVerifier, add_facts,
                                      make_memory, rows_equal)


def _config(tmp_path, tenant="tenant-a", collection=None,
            candidate_limit=128):
    url = os.environ.get("MINEMOS_QDRANT_URL")
    name = collection or f"minemos_test_{uuid.uuid4().hex[:12]}"
    if url:
        return QdrantConfig(
            url=url,
            api_key=os.environ.get("MINEMOS_QDRANT_API_KEY"),
            collection=name,
            tenant_id=tenant,
            timeout=15.0,
            candidate_limit=candidate_limit,
        )
    # One embedded database per test, shared by every tenant in that test so
    # the tenant-isolation check exercises a genuinely shared collection.
    return QdrantConfig(
        path=str(tmp_path / "qdrant"),
        collection=name,
        tenant_id=tenant,
        candidate_limit=candidate_limit,
    )


def _engine(config, capacity=50, verifier=None):
    return make_memory(
        capacity=capacity, backend="qdrant", qdrant_config=config,
        verifier=verifier or RuleVerifier([]))


def _retriever(mem):
    return mem._get_retrieval_adapter().retriever


def _points(mem):
    retriever = _retriever(mem)
    points, _ = retriever.client.scroll(
        retriever.collection, limit=1000, with_payload=True,
        scroll_filter=retriever._tenant_filter())
    return {str(p.id): dict(p.payload or {}) for p in points}


def _parity(rows, expected):
    """Compare result rows across processes; ``slot_id`` is process-local."""
    strip = lambda items: [  # noqa: E731 - local helper
        {k: v for k, v in row.items() if k != "slot_id"} for row in items]
    return rows_equal(strip(rows), strip(expected))


def _canonical_rows(mem, **query):
    mem.set_retrieval_backend("canonical")
    try:
        return mem.query(QUERY, **query)
    finally:
        mem.set_retrieval_backend("qdrant")
        # Switching backends drops the retriever; rebuild it from canonical.
        mem._get_retrieval_adapter().ensure_retriever()


# ── 1. Create, index, retrieve ─────────────────────────────────────────────


def test_create_index_and_retrieve(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "lives in", "Paris", 10.0),
            ("Bob", "lives in", "Oslo", 11.0),
            ("Carol", "works at", "Acme", 12.0),
        ])
        rows = mem.query(QUERY, top_k=3)
        assert len(rows) == 3
        assert rows_equal(rows, _canonical_rows(mem, top_k=3))

        stats = mem.retrieval_stats()["manager"]
        assert stats["points"] == 3
        assert stats["errors"] == 0
        payloads = _points(mem)
        assert len(payloads) == 3
        assert {p["subject"] for p in payloads.values()} == {
            "Alice", "Bob", "Carol"}
        assert all(p["tenant_id"] == cfg.tenant_id for p in payloads.values())
        assert all(len(point_id) == 36 for point_id in payloads)
    finally:
        mem.close()


# ── 2. Consolidation updates the existing vector and payload ───────────────


def test_consolidation_updates_indexed_point(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(
        cfg, verifier=RuleVerifier([("Paris", "resides", "SAME")]))
    try:
        mem.absorb(Candidate("Alice", "lives in", "Paris", timestamp=10.0))
        mem.query(QUERY, top_k=10)  # build the index before mutation
        assert mem.retrieval_stats()["manager"]["points"] == 1

        result = mem.absorb(
            Candidate("Alice", "resides in", "Paris", timestamp=20.0))
        assert result["action"] == "consolidate"
        assert mem.retrieval_stats()["manager"]["points"] == 1

        payload = next(iter(_points(mem).values()))
        assert payload["update_count"] == 2
        assert "resides" in payload["text"]
        assert payload["confidence"] > 0.5

        # Canonical filtering accepts either the original or the consolidated
        # predicate; the Qdrant pushdown must return the same candidate set.
        for predicate in ("lives in", "resides in"):
            selected = mem.query(
                QUERY, top_k=10, filters={"predicate": predicate})
            canonical = _canonical_rows(
                mem, top_k=10, filters={"predicate": predicate})
            assert rows_equal(selected, canonical), predicate
            assert len(selected) == 1
    finally:
        mem.close()


# ── 3. Deletion invalidates the vector ─────────────────────────────────────


def test_delete_slot_removes_point(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "lives in", "Paris", 10.0),
            ("Bob", "lives in", "Oslo", 11.0),
        ])
        mem.query(QUERY, top_k=10)
        victim = mem.memory.slots[0]
        victim_id = victim.memory_id
        assert _point_id(victim_id) in _points(mem)

        assert mem.delete_slot(id(victim)) is True
        assert mem.retrieval_stats()["manager"]["points"] == 1
        assert _point_id(victim_id) not in _points(mem)
        assert all(r["subject"] != "Alice" for r in mem.query(QUERY, top_k=10))
    finally:
        mem.close()


# ── 4. Metadata filtering ──────────────────────────────────────────────────


def test_metadata_filters_match_canonical(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "lives in", "Paris", 10.0),
            ("Bob", "lives in", "Oslo", 11.0),
            ("Carol", "works at", "Acme", 12.0),
        ])
        mem.query(QUERY, top_k=10)
        alice_id = next(s.memory_id for s in mem.memory.slots
                        if s.debug_subject == "Alice")

        cases = [
            {"subject": "Alice"},
            {"predicate": "works at"},
            {"min_confidence": 0.0},
            {"memory_ids": [alice_id]},
        ]
        for filters in cases:
            selected = mem.query(QUERY, top_k=10, filters=filters)
            canonical = _canonical_rows(mem, top_k=10, filters=filters)
            assert rows_equal(selected, canonical), filters
            hybrid = mem.query(
                QUERY, top_k=10, filters=filters, hybrid=True)
            assert rows_equal(hybrid, canonical), ("hybrid", filters)

        assert [r["subject"] for r in mem.query(
            QUERY, top_k=10, filters={"subject": "Alice"})] == ["Alice"]
        assert [r["subject"] for r in mem.query(
            QUERY, top_k=10, filters={"predicate": "works at"})] == ["Carol"]
        with pytest.raises(ValueError):
            mem.query(QUERY, filters={"unknown": 1})
    finally:
        mem.close()


# ── 5. Restart and reconnect ───────────────────────────────────────────────


def test_restart_reconnects_without_resurrecting_stale_vectors(tmp_path):
    """After a restart the index reconnects to its persistent store, but a new
    canonical memory is still the authority: stale tenant vectors are dropped
    rather than served as if they were live memories."""
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    add_facts(mem, [
        ("Alice", "lives in", "Paris", 10.0),
        ("Bob", "lives in", "Oslo", 11.0),
    ])
    mem.query(QUERY, top_k=5)
    assert len(_points(mem)) == 2
    mem.close()

    restarted = _engine(cfg)
    try:
        rows = restarted.query(QUERY, top_k=5)
        assert rows == []
        stats = restarted.retrieval_stats()["manager"]
        assert stats["mode"] in ("local", "server")
        assert stats["points"] == 0
        assert stats["errors"] == 0
    finally:
        restarted.close()


# ── 6. Rebuild after index loss ────────────────────────────────────────────


def test_rebuild_restores_lost_collection(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "lives in", "Paris", 10.0),
            ("Bob", "lives in", "Oslo", 11.0),
        ])
        mem.query(QUERY, top_k=5)
        retriever = _retriever(mem)
        expected = mem.query(QUERY, top_k=5)

        retriever.client.delete_collection(cfg.collection)
        result = retriever.rebuild()
        assert result["points"] == 2
        assert result["complete"] is True
        assert mem.retrieval_stats()["manager"]["points"] == 2
        assert rows_equal(mem.query(QUERY, top_k=5), expected)
    finally:
        mem.close()


# ── 7. Outage during ingestion does not lose canonical memory ──────────────


def test_qdrant_outage_during_ingestion_falls_back(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    try:
        mem.absorb(Candidate("Alice", "lives in", "Paris", timestamp=10.0))
        mem.query(QUERY, top_k=10)
        retriever = _retriever(mem)
        retriever.client.close()  # simulate the vector store going offline

        result = mem.absorb(
            Candidate("Bob", "lives in", "Oslo", timestamp=11.0))
        assert result["action"] == "insert"
        assert any(s.debug_subject == "Bob" for s in mem.memory.slots)
        assert mem.retrieval_stats()["manager"]["dirty"] is True

        rows = mem.query(QUERY, top_k=10)
        assert any(r["subject"] == "Bob" for r in rows)
        assert mem.retrieval_stats()["errors"] >= 1
    finally:
        mem.close()


# ── 8. Concurrent updates stay consistent ──────────────────────────────────


def test_concurrent_updates_keep_index_in_sync(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(cfg, capacity=40)
    try:
        mem.absorb(Candidate("Seed", "exists", "yes", timestamp=1.0))
        mem.query(QUERY, top_k=10)

        errors = []

        def worker(i):
            try:
                mem.absorb(Candidate(
                    f"Person{i}", "lives in", f"City{i}",
                    timestamp=10.0 + i))
            except Exception as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        assert mem.retrieval_stats()["manager"]["points"] == len(
            mem.memory.slots)
        assert rows_equal(mem.query(QUERY, top_k=50),
                          _canonical_rows(mem, top_k=50))
    finally:
        mem.close()


# ── 9. Duplicate indexing and retry are idempotent ─────────────────────────


def test_duplicate_sync_and_retry_are_idempotent(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "lives in", "Paris", 10.0),
            ("Bob", "lives in", "Oslo", 11.0),
        ])
        mem.query(QUERY, top_k=5)
        retriever = _retriever(mem)
        first = retriever.sync_state(force=True)
        second = retriever.sync_state(force=True)
        assert first["points"] == second["points"] == 2
        assert mem.retrieval_stats()["manager"]["points"] == 2
        retriever.rebuild()
        assert mem.retrieval_stats()["manager"]["points"] == 2
    finally:
        mem.close()


# ── 10. Index writes stay incremental ──────────────────────────────────────


def test_incremental_sync_updates_only_changed_points(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "lives in", "Paris", 10.0),
            ("Bob", "lives in", "Oslo", 11.0),
        ])
        mem.query(QUERY, top_k=5)
        retriever = _retriever(mem)
        before = retriever._upserts
        assert before == 2

        result = retriever.sync_state()
        assert result["changed"] == 0
        assert retriever._upserts == before

        mem.absorb(Candidate("Carol", "lives in", "Rome", timestamp=12.0))
        assert retriever._upserts == before + 1
    finally:
        mem.close()


# ── 11. Contradiction handling after retrieval ─────────────────────────────


def test_contradiction_supersedes_indexed_point(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(
        cfg, verifier=RuleVerifier([("Rome", "Venice", "CONTRADICTION")]))
    try:
        mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
        mem.query(QUERY, top_k=10)
        result = mem.absorb(
            Candidate("Maya", "city", "Venice", timestamp=20.0))
        assert result["action"] == "temporal_update"

        payloads = _points(mem)
        assert len(payloads) == 2
        active = [p for p in payloads.values() if p["valid_until"] is None]
        closed = [p for p in payloads.values() if p["valid_until"] is not None]
        assert len(active) == 1 and "Venice" in active[0]["text"]
        assert len(closed) == 1 and "Rome" in closed[0]["text"]

        assert [r["text"] for r in mem.query(
            QUERY, top_k=10, active_only=True)] == ["Maya city Venice."]
        historical = mem.query(QUERY, top_k=10, as_of=15.0)
        assert [r["text"] for r in historical] == ["Maya city Rome."]
    finally:
        mem.close()


# ── 12. Checkpoint restore preserves stable ids and rebuilds the index ─────


def test_checkpoint_restore_preserves_ids_and_rebuilds_index(tmp_path):
    cfg = _config(tmp_path)
    checkpoint = str(tmp_path / "memory.npz")
    mem = _engine(cfg)
    add_facts(mem, [
        ("Alice", "lives in", "Paris", 10.0),
        ("Bob", "lives in", "Oslo", 11.0),
    ])
    ids = [s.memory_id for s in mem.memory.slots]
    expected = mem.query(QUERY, top_k=5)
    mem.save_checkpoint(checkpoint)
    mem.close()

    restored = _engine(cfg)
    try:
        restored.load_checkpoint(checkpoint)
        assert [s.memory_id for s in restored.memory.slots] == ids
        assert _parity(restored.query(QUERY, top_k=5), expected)
        assert restored.retrieval_stats()["manager"]["points"] == 2
    finally:
        restored.close()


# ── 13. Tenant isolation ───────────────────────────────────────────────────


def test_tenants_are_isolated_in_a_shared_collection(tmp_path):
    collection = f"shared_{uuid.uuid4().hex[:10]}"
    cfg_a = _config(tmp_path, tenant="tenant-a", collection=collection)
    cfg_b = _config(tmp_path, tenant="tenant-b", collection=collection)
    checkpoint_a = str(tmp_path / "tenant-a.npz")

    first = _engine(cfg_a)
    add_facts(first, [("Alice", "lives in", "Paris", 10.0)])
    first.query(QUERY, top_k=5)
    first.save_checkpoint(checkpoint_a)
    first.close()

    second = _engine(cfg_b)
    try:
        add_facts(second, [("Bob", "lives in", "Oslo", 11.0)])
        rows = second.query(QUERY, top_k=5)
        assert [r["subject"] for r in rows] == ["Bob"]
        assert second.retrieval_stats()["manager"]["points"] == 1
        assert all(p["tenant_id"] == "tenant-b"
                   for p in _points(second).values())
    finally:
        second.close()

    reopened = _engine(cfg_a)
    try:
        reopened.load_checkpoint(checkpoint_a)
        rows = reopened.query(QUERY, top_k=5)
        assert [r["subject"] for r in rows] == ["Alice"]
        assert reopened.retrieval_stats()["manager"]["points"] == 1
        assert all(p["tenant_id"] == "tenant-a"
                   for p in _points(reopened).values())
    finally:
        reopened.close()


# ── 14. Eviction is mirrored in the index ──────────────────────────────────


def test_eviction_is_mirrored_in_index(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(cfg, capacity=2)
    try:
        for i, name in enumerate(("Alice", "Bob", "Carol")):
            mem.absorb(Candidate(name, "likes", "chess",
                                 timestamp=float(i)))
        rows = mem.query(QUERY, top_k=10)
        assert len(mem.memory.slots) == 2
        assert mem.retrieval_stats()["manager"]["points"] == 2
        assert rows_equal(rows, _canonical_rows(mem, top_k=10))
    finally:
        mem.close()


# ── 15. Configuration never leaks the API key ──────────────────────────────


def test_api_key_is_not_repr_or_stats(tmp_path):
    cfg = QdrantConfig(
        path=str(tmp_path / "key"), api_key="super-secret-value",
        collection="keys", tenant_id="t")
    assert "super-secret-value" not in repr(cfg)
    mem = _engine(cfg)
    try:
        add_facts(mem, [("Alice", "lives in", "Paris", 10.0)])
        assert "super-secret-value" not in repr(mem.retrieval_stats())
    finally:
        mem.close()


# ── 16. Hybrid retrieval merges Qdrant dense and BM25 candidates ──────────


def test_hybrid_merges_bm25_candidates_into_dense_beam(tmp_path):
    cfg = _config(tmp_path, candidate_limit=1)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "likes", "zebra", 1.0),
            ("Bob", "likes", "violin", 2.0),
            ("Carol", "likes", "kite", 3.0),
        ])
        mem.query("q", top_k=1)
        dense_stats = mem.retrieval_stats()["manager"]
        assert dense_stats["last_dense_candidates"] == 1
        assert dense_stats["last_fused_candidates"] == 1

        mem.query("zebra violin kite", top_k=1, hybrid=True)
        hybrid_stats = mem.retrieval_stats()["manager"]
        assert hybrid_stats["last_dense_candidates"] == 1
        assert hybrid_stats["last_fused_candidates"] == 3
    finally:
        mem.close()


def test_hybrid_rrf_fusion_reorders_the_same_candidates(tmp_path):
    cfg = _config(tmp_path, candidate_limit=1)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "likes", "zebra", 1.0),
            ("Bob", "likes", "violin", 2.0),
            ("Carol", "likes", "kite", 3.0),
        ])
        dense_hybrid = mem.query(
            "zebra violin kite", top_k=3, hybrid=True)
        rrf_hybrid = mem.query(
            "zebra violin kite", top_k=3, hybrid=True, fusion="rrf")

        # RRF changes ordering only: the candidate union, and therefore the
        # returned rows and their exact dense similarities, are unchanged.
        assert len(dense_hybrid) == 3
        assert {row["text"] for row in dense_hybrid} == {
            row["text"] for row in rrf_hybrid}
        assert {(row["text"], row["similarity"]) for row in dense_hybrid} == {
            (row["text"], row["similarity"]) for row in rrf_hybrid}

        # Fusion is deterministic and works against the same beam plumbing.
        assert rrf_hybrid == mem.query(
            "zebra violin kite", top_k=3, hybrid=True, fusion="rrf")
        narrow = mem.query(
            "zebra violin kite", top_k=1, hybrid=True, fusion="rrf")
        assert len(narrow) == 1
        assert narrow[0]["text"] in {row["text"] for row in dense_hybrid}
        stats = mem.retrieval_stats()["manager"]
        assert stats["last_dense_candidates"] == 1
        assert stats["last_fused_candidates"] == 3

        with pytest.raises(ValueError, match="fusion"):
            mem.query("q", hybrid=True, fusion="bogus")
        with pytest.raises(ValueError, match="hybrid"):
            mem.query("q", fusion="rrf")
    finally:
        mem.close()


def test_hybrid_fusion_config_supplies_the_default(tmp_path):
    cfg = replace(_config(tmp_path, candidate_limit=1), hybrid_fusion="rrf")
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "likes", "zebra", 1.0),
            ("Bob", "likes", "violin", 2.0),
            ("Carol", "likes", "kite", 3.0),
        ])
        default_rows = mem.query("zebra violin kite", top_k=3, hybrid=True)
        explicit_rows = mem.query(
            "zebra violin kite", top_k=3, hybrid=True, fusion="rrf")
        assert default_rows == explicit_rows
        stats = mem.retrieval_stats()["manager"]
        assert stats["hybrid_fusion"] == "rrf"
        assert stats["rrf_k"] == 60
    finally:
        mem.close()


def test_successful_sync_and_query_clear_stale_error(tmp_path):
    mem = _engine(_config(tmp_path))
    try:
        add_facts(mem, [
            ("Alice", "lives in", "Paris", 10.0),
            ("Bob", "lives in", "Oslo", 11.0),
        ])
        retriever = _retriever(mem)
        retriever._last_error = "injected failure"
        retriever._dirty = True
        retriever.sync_state(force=True)
        assert retriever.stats["last_error"] is None

        retriever._last_error = "injected failure"
        mem.query(QUERY, top_k=2)
        assert retriever.stats["last_error"] is None
        assert mem.retrieval_stats()["state"] == "ready"
    finally:
        mem.close()


def test_hybrid_requires_qdrant_backend(tmp_path):
    mem = _engine(_config(tmp_path))
    try:
        mem.set_retrieval_backend("canonical")
        with pytest.raises(ValueError):
            mem.query(QUERY, hybrid=True)
    finally:
        mem.close()


# ── 17. Server-backed run when explicitly configured ──────────────────────


@pytest.mark.skipif(
    not os.environ.get("MINEMOS_QDRANT_URL"),
    reason="set MINEMOS_QDRANT_URL to run the server-backed check")
def test_real_qdrant_server_roundtrip(tmp_path):
    cfg = _config(tmp_path, tenant="server-tenant")
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "lives in", "Paris", 10.0),
            ("Bob", "lives in", "Oslo", 11.0),
        ])
        rows = mem.query(QUERY, top_k=2)
        assert len(rows) == 2
        assert mem.retrieval_stats()["manager"]["mode"] == "server"
        assert mem.retrieval_stats()["manager"]["points"] == 2
    finally:
        mem.close()
