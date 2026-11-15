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

from minemos_engine.retrieval import QdrantConfig
from minemos_engine.retrieval.qdrant import _point_id
from minemos_engine.structured_candidate import Candidate
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


def test_scope_filter_matches_canonical_and_never_widens(tmp_path):
    """A joint scope filter returns exactly the canonical rows in those scopes.

    The Qdrant pushdown is an optimization of the canonical membership test:
    it must neither drop in-scope rows nor admit a row from a scope the caller
    did not request.
    """
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("conversation:alpha", "segment_0", "wing hive moss fern", 10.0),
            ("conversation:beta", "segment_0", "dune reed kelp vine", 11.0),
            ("conversation:gamma", "segment_0", "wing unrelated noise", 12.0),
        ])
        mem.query(QUERY, top_k=10)
        cases = (
            {"subjects": ["conversation:alpha", "conversation:beta"]},
            {"subjects": ["conversation:beta"]},
            {"subjects": ["conversation:alpha", "conversation:missing"]},
            {"subjects": ["conversation:alpha"]},
        )
        for filters in cases:
            selected = mem.query(QUERY, top_k=10, filters=filters)
            canonical = _canonical_rows(mem, top_k=10, filters=filters)
            assert rows_equal(selected, canonical), filters
            hybrid = mem.query(
                QUERY, top_k=10, filters=filters, hybrid=True)
            assert rows_equal(hybrid, canonical), ("hybrid", filters)

        pair = mem.query(
            QUERY, top_k=10,
            filters={"subjects": ["conversation:alpha", "conversation:beta"]})
        assert {row["subject"] for row in pair} == {
            "conversation:alpha", "conversation:beta"}
        assert mem.query(
            QUERY, top_k=10,
            filters={"subjects": ["conversation:missing"]}) == []
        with pytest.raises(ValueError):
            mem.query(QUERY, filters={"subjects": "conversation:alpha"})
        with pytest.raises(ValueError):
            mem.query(QUERY, filters={
                "subject": "conversation:alpha",
                "subjects": ["conversation:beta"]})
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


def test_rrf_fuses_the_exact_dense_ranking_not_transport_order(tmp_path):
    """RRF is defined over MINEMOS's exact ranking.

    Qdrant's response order is a candidate-selection order; in server mode it
    is approximate.  Fusion must therefore rank the dense list with the same
    exact dot product the engine serves, not whatever order the transport
    happened to return.
    """
    cfg = _config(tmp_path, candidate_limit=64)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "likes", "zebra zebra zebra", 1.0),
            ("Bob", "likes", "violin violin", 2.0),
            ("Carol", "likes", "kite kite kite kite", 3.0),
            ("Dave", "likes", "piano", 4.0),
        ])
        query = "zebra violin kite piano"
        mem.query(query, top_k=4)
        retriever = _retriever(mem)
        baseline = [
            row["memory_id"]
            for row in mem.query(
                query, top_k=4, hybrid=True, fusion="rrf")]

        class Reordered:
            def __init__(self, points):
                self.points = points

        transport = retriever.client.query_points

        def reversed_transport(*args, **kwargs):
            response = transport(*args, **kwargs)
            return Reordered(list(reversed(response.points)))

        retriever.client.query_points = reversed_transport
        try:
            scrambled = [
                row["memory_id"]
                for row in mem.query(
                    query, top_k=4, hybrid=True, fusion="rrf")]
        finally:
            retriever.client.query_points = transport

        assert scrambled == baseline
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


# ── 18. Rebuild from corrected canonical state ────────────────────────────


def test_rebuild_matches_corrected_canonical_state(tmp_path):
    from tests.test_consolidation_integrity import DirectionalVerifier

    cfg = _config(tmp_path)
    verifier = DirectionalVerifier([
        ("lima", "oslo", "CONTRADICTION", 0.93, 0.92),
        ("lima", "visa", "CONTRADICTION", 0.90, 0.88),
        ("oslo", "visa", "SAME", 0.80, 0.34),
    ])
    mem = _engine(cfg, verifier=verifier)
    try:
        mem.absorb(Candidate("Ada", "lives in", "Lima", timestamp=100.0))
        mem.absorb(Candidate("Ada", "lives in", "Oslo", timestamp=300.0))
        mem.absorb(Candidate(
            "Ada", "lives in", "worker visa", timestamp=200.0))
        mem.query(QUERY, top_k=10, active_only=False)

        expected = _canonical_rows(mem, top_k=10, active_only=False)
        assert len(expected) == 3
        canonical_texts = {row["text"] for row in expected}
        assert canonical_texts == {
            "Ada lives in Lima.",
            "Ada lives in worker visa.",
            "Ada lives in Oslo.",
        }

        retriever = _retriever(mem)
        assert len(_points(mem)) == 3
        retriever.client.delete_collection(cfg.collection)
        result = retriever.rebuild()
        assert result["points"] == 3
        assert result["complete"] is True

        rows = mem.query(QUERY, top_k=10, active_only=False)
        assert _parity(rows, expected)
        payloads = {p["text"]: p for p in _points(mem).values()}
        assert set(payloads) == canonical_texts
        assert payloads["Ada lives in Lima."]["valid_until"] == 200.0
        assert payloads["Ada lives in worker visa."]["valid_until"] == 300.0
        assert payloads["Ada lives in Oslo."]["valid_until"] is None
        assert payloads["Ada lives in Oslo."]["is_active"] is True
    finally:
        mem.close()


# ── 9. Automatic scope discovery over indexed candidates ───────────────────


def test_scope_discovery_matches_canonical_backend(tmp_path):
    """Discovery uses the selected backend; it must not change the answer."""
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    question = "wing dune hive reed moss kelp fern vine"
    try:
        add_facts(mem, [
            ("conversation:alpha", "segment_0", "wing hive moss fern", 10.0),
            ("conversation:beta", "segment_0", "dune reed kelp vine", 11.0),
            ("conversation:gamma", "segment_0", "wing unrelated noise", 12.0),
        ])
        via_qdrant = mem.eligible_query(
            question, top_k=10, discover_scopes=True)
        assert via_qdrant["status"] == "answered"
        assert set(via_qdrant["scopes"]) == {
            "conversation:alpha", "conversation:beta"}
        assert via_qdrant["evidence"]["cross_scope"] is True
        assert len(via_qdrant["rows"]) == 2

        mem.set_retrieval_backend("canonical")
        try:
            via_canonical = mem.eligible_query(
                question, top_k=10, discover_scopes=True)
        finally:
            mem.set_retrieval_backend("qdrant")
            mem._get_retrieval_adapter().ensure_retriever()

        assert via_canonical["status"] == via_qdrant["status"]
        assert via_canonical["scopes"] == via_qdrant["scopes"]
        assert via_canonical["discovery"] == via_qdrant["discovery"]
        assert [row["memory_id"] for row in via_canonical["rows"]] == [
            row["memory_id"] for row in via_qdrant["rows"]]
    finally:
        mem.close()


# ── 19. External drift, corrupt index and query-time outage ────────────────


def test_external_stale_point_and_payload_drift_are_not_authority(tmp_path):
    """The index is derived: a rogue point is deleted by sync, and even a
    drifted payload on a live point cannot change a returned answer because
    rows are hydrated from canonical state.  Rebuild restores the payload."""
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "lives in", "Paris", 10.0),
            ("Bob", "lives in", "Oslo", 11.0),
        ])
        expected = mem.query(QUERY, top_k=5)
        retriever = _retriever(mem)

        from qdrant_client import models as qm
        from minemos_engine.retrieval.qdrant import VECTOR_NAME

        rogue = str(uuid.uuid4())
        retriever.client.upsert(cfg.collection, [qm.PointStruct(
            id=rogue,
            vector={VECTOR_NAME: [0.0] * retriever.dim},
            payload={"tenant_id": cfg.tenant_id, "subject": "Mallory",
                     "text": "Mallory lives in Atlantis."},
        )], wait=True)
        synced = retriever.sync_state(force=True)
        assert synced["points"] == 2
        assert synced["stale_deleted"] == 1
        assert rogue not in _points(mem)
        assert rows_equal(mem.query(QUERY, top_k=5), expected)

        # Drift a live point's payload without changing the point count; the
        # incremental sync cannot see it, but the answer still comes from
        # canonical state.
        live_id = next(iter(_points(mem)))
        retriever.client.set_payload(
            cfg.collection, {"text": "Mallory lives in Atlantis."},
            points=[live_id])
        assert any("Atlantis" in (p.get("text") or "")
                   for p in _points(mem).values())
        rows = mem.query(QUERY, top_k=5)
        assert rows_equal(rows, expected)
        assert not any("atlantis" in (row.get("text") or "").casefold()
                       for row in rows)

        # A rebuild replaces the drifted payload from canonical state.
        rebuilt = retriever.rebuild()
        assert rebuilt["points"] == 2
        assert not any("Atlantis" in (p.get("text") or "")
                       for p in _points(mem).values())
        assert rows_equal(mem.query(QUERY, top_k=5), expected)
    finally:
        mem.close()


def test_missing_collection_falls_back_then_recovers_on_sync(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "lives in", "Paris", 10.0),
            ("Bob", "lives in", "Oslo", 11.0),
        ])
        expected = mem.query(QUERY, top_k=5)
        retriever = _retriever(mem)
        retriever.client.delete_collection(cfg.collection)

        # The search fails on the missing collection; the engine answers from
        # canonical state and reports the index error.
        rows = mem.query(QUERY, top_k=5)
        assert rows_equal(rows, expected)
        assert mem.retrieval_stats()["errors"] >= 1

        recovered = retriever.sync_state(force=True)
        assert recovered["points"] == 2
        assert recovered["complete"] is True
        assert rows_equal(mem.query(QUERY, top_k=5), expected)
    finally:
        mem.close()


def test_wrong_dimension_index_fails_closed_and_recovers(tmp_path):
    """A collection holding another encoder's vectors is refused rather than
    served; the engine answers from canonical state until the index is
    repointed at a correctly sized collection."""
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    bad_collection = f"{cfg.collection}_bad"
    try:
        add_facts(mem, [
            ("Alice", "lives in", "Paris", 10.0),
            ("Bob", "lives in", "Oslo", 11.0),
        ])
        expected = mem.query(QUERY, top_k=5)
        retriever = _retriever(mem)

        from qdrant_client import models as qm
        from minemos_engine.retrieval.qdrant import VECTOR_NAME

        retriever.client.create_collection(
            bad_collection,
            vectors_config={VECTOR_NAME: qm.VectorParams(
                size=retriever.dim + 1, distance=qm.Distance.DOT)})

        original = retriever.collection
        retriever.collection = bad_collection
        retriever._indexed_fingerprints = {}
        retriever._indexed_signatures = {}
        retriever._indexed_version = -1
        retriever._dirty = True

        rows = mem.query(QUERY, top_k=5)
        assert rows_equal(rows, expected)
        stats = mem.retrieval_stats()
        assert stats["errors"] >= 1
        assert "vectors but MINEMOS encodes" in (stats["last_error"] or "")

        retriever.collection = original
        retriever._indexed_version = -1
        retriever._dirty = True
        recovered = retriever.sync_state(force=True)
        assert recovered["points"] == 2
        assert rows_equal(mem.query(QUERY, top_k=5), expected)
    finally:
        mem.close()


def test_query_outage_falls_back_without_losing_answers(tmp_path):
    cfg = _config(tmp_path)
    mem = _engine(cfg)
    try:
        add_facts(mem, [
            ("Alice", "lives in", "Paris", 10.0),
            ("Bob", "lives in", "Oslo", 11.0),
        ])
        expected = mem.query(QUERY, top_k=5)
        _retriever(mem).client.close()

        rows = mem.query(QUERY, top_k=5)
        assert rows_equal(rows, expected)
        assert mem.retrieval_stats()["errors"] >= 1
        state = mem.retrieval_stats()["state"]
        assert state in ("degraded", "ready")
    finally:
        mem.close()


def test_lost_index_store_is_rebuilt_from_checkpoint(tmp_path):
    """Losing the whole vector store loses no memory: a fresh store is
    populated from the restored canonical checkpoint."""
    cfg = _config(tmp_path)
    checkpoint = str(tmp_path / "memory.npz")
    mem = _engine(cfg)
    add_facts(mem, [
        ("Alice", "lives in", "Paris", 10.0),
        ("Bob", "lives in", "Oslo", 11.0),
    ])
    expected = mem.query(QUERY, top_k=5)
    mem.save_checkpoint(checkpoint)
    mem.close()

    # A server cannot be repointed at a fresh directory; a new collection is
    # the equivalent store loss for the tenant.
    fresh_cfg = (
        replace(cfg, collection=f"minemos_test_{uuid.uuid4().hex[:12]}")
        if cfg.url else replace(cfg, path=str(tmp_path / "fresh_store")))
    fresh = _engine(fresh_cfg)
    try:
        fresh.load_checkpoint(checkpoint)
        assert _parity(fresh.query(QUERY, top_k=5), expected)
        assert fresh.retrieval_stats()["manager"]["points"] == 2
        assert fresh.retrieval_stats()["manager"]["errors"] == 0
    finally:
        fresh.close()
