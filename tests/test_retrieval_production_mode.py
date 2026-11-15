"""Deployment-profile tests for standard production Qdrant retrieval.

These tests never start a Qdrant server.  Production configuration is
validated eagerly, but the client is contacted lazily on the first write or
query, so selection and fail-fast behavior can be tested without a network.
The unavailable-endpoint test uses a closed loopback port to verify canonical
fallback and degradation reporting.
"""

from __future__ import annotations

import numpy as np
import pytest

from minemos_engine.black_hole_memory import BlackHoleMemory
from minemos_engine.checkpoint import load_checkpoint, load_into_black_hole_memory
from minemos_engine.retrieval import (
    CANONICAL,
    CANONICAL_ONLY,
    LEGACY,
    PRODUCTION,
    QDRANT,
    QdrantConfig,
    RetrievalConfig,
    RetrievalConfigurationError,
)
from minemos_engine.structured_candidate import Candidate
from tests._retrieval_helpers import (
    QUERY,
    FixedEncoder,
    RuleVerifier,
    StubNLI,
    add_facts,
    rows_equal,
)

FACTS = [
    ("Alice", "lives in", "Paris", 10.0),
    ("Bob", "lives in", "Oslo", 11.0),
    ("Carol", "works at", "Acme", 12.0),
]
CLOSED_PORT_URL = "http://127.0.0.1:9"


def _memory(**kwargs) -> BlackHoleMemory:
    return BlackHoleMemory(
        capacity=20,
        dim=4096,
        dense_encoder=FixedEncoder(),
        nli_gate=StubNLI(),
        verifier=RuleVerifier(),
        **kwargs,
    )


def _unreachable_config(timeout: float = 0.2) -> QdrantConfig:
    return QdrantConfig(
        url=CLOSED_PORT_URL,
        tenant_id="production-test",
        timeout=timeout,
        max_retries=0,
        retry_backoff=0.0,
    )


def _canonical_rows(facts=FACTS):
    mem = _memory()
    add_facts(mem, facts)
    return mem.query(QUERY, top_k=10)


def _strip_ids(rows):
    # Slot ids are process-local and memory ids are random per engine; callers
    # comparing independent engines assert id preservation separately.
    return [{k: v for k, v in row.items()
             if k not in ("slot_id", "memory_id")} for row in rows]


# ── default / legacy selection ─────────────────────────────────────────────


def test_legacy_default_remains_canonical(monkeypatch):
    monkeypatch.delenv("MINEMOS_RETRIEVAL_MODE", raising=False)
    mem = _memory()
    add_facts(mem, FACTS)

    assert mem.retrieval_backend == CANONICAL
    assert mem.retrieval_mode == LEGACY
    assert mem._retrieval_adapter_obj is None
    assert len(mem.query(QUERY, top_k=3)) == 3

    stats = mem.retrieval_stats()
    assert stats["backend"] == CANONICAL
    assert stats["mode"] == LEGACY
    assert stats["active"] is False
    assert stats["initialized"] is False
    assert stats["degraded"] is False
    assert stats["state"] == "canonical"


def test_explicit_canonical_backend_is_still_supported(monkeypatch):
    monkeypatch.delenv("MINEMOS_RETRIEVAL_MODE", raising=False)
    mem = _memory(retrieval_backend=CANONICAL)
    assert mem.retrieval_backend == CANONICAL
    assert mem.retrieval_mode == LEGACY


# ── production selection ───────────────────────────────────────────────────


def test_production_mode_selects_qdrant_with_server_config():
    mem = _memory(
        retrieval_mode=PRODUCTION,
        qdrant_config=QdrantConfig(
            url="http://127.0.0.1:6333", tenant_id="tenant-a"),
    )
    assert mem.retrieval_backend == QDRANT
    assert mem.retrieval_mode == PRODUCTION
    assert mem._qdrant_config.url == "http://127.0.0.1:6333"

    stats = mem.retrieval_stats()
    assert stats["backend"] == QDRANT
    assert stats["mode"] == PRODUCTION
    assert stats["active"] is True
    assert stats["initialized"] is False
    assert stats["degraded"] is False
    assert stats["state"] == "not_initialized"


def test_production_mode_from_environment(monkeypatch):
    monkeypatch.setenv("MINEMOS_RETRIEVAL_MODE", PRODUCTION)
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("QDRANT_TENANT_ID", "tenant-env")
    monkeypatch.delenv("QDRANT_PATH", raising=False)

    mem = _memory()
    assert mem.retrieval_mode == PRODUCTION
    assert mem.retrieval_backend == QDRANT
    assert mem._qdrant_config.tenant_id == "tenant-env"


def test_production_mode_accepts_matching_explicit_backend():
    mem = _memory(
        retrieval_mode=PRODUCTION,
        retrieval_backend=QDRANT,
        qdrant_config=QdrantConfig(
            url="http://127.0.0.1:6333", tenant_id="tenant-a"),
    )
    assert mem.retrieval_backend == QDRANT


def test_production_mode_rejects_non_qdrant_backend():
    for backend in ("optimized", "incremental", CANONICAL):
        with pytest.raises(RetrievalConfigurationError, match="production"):
            _memory(
                retrieval_mode=PRODUCTION,
                retrieval_backend=backend,
                qdrant_config=QdrantConfig(
                    url="http://127.0.0.1:6333", tenant_id="tenant-a"),
            )


def test_production_mode_rejects_conflicting_retrieval_config():
    with pytest.raises(RetrievalConfigurationError, match="production"):
        _memory(
            retrieval_mode=PRODUCTION,
            retrieval_config=RetrievalConfig(backend="incremental"),
            qdrant_config=QdrantConfig(
                url="http://127.0.0.1:6333", tenant_id="tenant-a"),
        )


# ── fail-fast configuration errors ─────────────────────────────────────────


def test_production_missing_configuration_is_actionable(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_PATH", raising=False)
    with pytest.raises(RetrievalConfigurationError) as excinfo:
        _memory(retrieval_mode=PRODUCTION)
    message = str(excinfo.value)
    assert "QDRANT_URL" in message
    assert "production" in message


def test_production_rejects_embedded_path(tmp_path):
    with pytest.raises(RetrievalConfigurationError, match="Embedded mode"):
        _memory(
            retrieval_mode=PRODUCTION,
            qdrant_config=QdrantConfig(
                path=str(tmp_path / "qdrant"), tenant_id="tenant-a"),
        )


def test_production_rejects_insecure_remote_without_api_key():
    with pytest.raises(RetrievalConfigurationError, match="https"):
        _memory(
            retrieval_mode=PRODUCTION,
            qdrant_config=QdrantConfig(
                url="http://qdrant.example.com:6333",
                api_key="secret",
                tenant_id="tenant-a"),
        )


def test_production_rejects_remote_without_api_key():
    with pytest.raises(RetrievalConfigurationError, match="QDRANT_API_KEY"):
        _memory(
            retrieval_mode=PRODUCTION,
            qdrant_config=QdrantConfig(
                url="https://qdrant.example.com:6333",
                tenant_id="tenant-a"),
        )


def test_production_accepts_secure_remote_with_api_key():
    mem = _memory(
        retrieval_mode=PRODUCTION,
        qdrant_config=QdrantConfig(
            url="https://qdrant.example.com:6333",
            api_key="secret",
            tenant_id="tenant-a"),
    )
    assert mem.retrieval_backend == QDRANT


def test_production_allows_explicit_insecure_internal_network():
    mem = _memory(
        retrieval_mode=PRODUCTION,
        qdrant_config=QdrantConfig(
            url="http://qdrant.internal:6333",
            api_key="secret",
            allow_insecure=True,
            tenant_id="tenant-a"),
    )
    assert mem.retrieval_backend == QDRANT


def test_production_rejects_recreate():
    with pytest.raises(RetrievalConfigurationError, match="QDRANT_RECREATE"):
        _memory(
            retrieval_mode=PRODUCTION,
            qdrant_config=QdrantConfig(
                url="http://127.0.0.1:6333",
                tenant_id="tenant-a",
                recreate=True),
        )


def test_production_rejects_malformed_url():
    with pytest.raises(RetrievalConfigurationError, match="http"):
        _memory(
            retrieval_mode=PRODUCTION,
            qdrant_config=QdrantConfig(url="localhost:6333"),
        )


# ── canonical-only mode ────────────────────────────────────────────────────


def test_canonical_only_ignores_qdrant_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MINEMOS_RETRIEVAL_MODE", CANONICAL_ONLY)
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "unused"))
    monkeypatch.delenv("QDRANT_URL", raising=False)

    mem = _memory()
    add_facts(mem, FACTS)
    assert mem.retrieval_mode == CANONICAL_ONLY
    assert mem.retrieval_backend == CANONICAL
    assert mem._qdrant_config is None
    assert len(mem.query(QUERY, top_k=3)) == 3

    with pytest.raises(RetrievalConfigurationError, match="canonical"):
        mem.set_retrieval_backend(QDRANT)


def test_canonical_only_rejects_conflicting_backend():
    with pytest.raises(RetrievalConfigurationError, match="canonical"):
        _memory(retrieval_mode=CANONICAL_ONLY, retrieval_backend=QDRANT)


def test_canonical_only_ignores_supplied_qdrant_config(tmp_path):
    mem = _memory(
        retrieval_mode=CANONICAL_ONLY,
        qdrant_config={"path": str(tmp_path / "ignored")},
    )
    assert mem.retrieval_backend == CANONICAL
    assert mem._qdrant_config is None


# ── runtime production restrictions ────────────────────────────────────────


def test_production_allows_canonical_rollback_but_not_experimental_backends():
    mem = _memory(
        retrieval_mode=PRODUCTION,
        qdrant_config=QdrantConfig(
            url="http://127.0.0.1:6333", tenant_id="tenant-a"),
    )
    assert mem.set_retrieval_backend(CANONICAL) == CANONICAL
    assert mem.retrieval_stats()["state"] == "canonical"
    assert mem.set_retrieval_backend(QDRANT) == QDRANT

    with pytest.raises(RetrievalConfigurationError, match="production"):
        mem.set_retrieval_backend("optimized")


# ── unavailable Qdrant: fallback and honest reporting ──────────────────────


def test_unavailable_qdrant_falls_back_to_canonical_and_reports_degraded(
        caplog):
    mem = _memory(
        retrieval_mode=PRODUCTION,
        qdrant_config=_unreachable_config())
    add_facts(mem, FACTS)

    with caplog.at_level("WARNING", logger="minemos_engine.black_hole_memory"):
        rows = mem.query(QUERY, top_k=10)
        assert "canonical MINEMOS retrieval" in caplog.text or \
            "canonical retrieval" in caplog.text

    assert rows_equal(_strip_ids(rows), _strip_ids(_canonical_rows()))
    stats = mem.retrieval_stats()
    assert stats["backend"] == QDRANT
    assert stats["mode"] == PRODUCTION
    assert stats["active"] is True
    assert stats["degraded"] is True
    assert stats["state"] == "degraded"
    assert stats["errors"] >= 1
    assert stats["last_error"]


def test_degraded_state_clears_after_successful_operation():
    mem = _memory(
        retrieval_mode=PRODUCTION,
        qdrant_config=QdrantConfig(
            url="http://127.0.0.1:6333", tenant_id="tenant-a"))

    class _Adapter:
        retriever = object()

        def __init__(self):
            self.payload = {}

        def stats(self):
            return self.payload

    fake = _Adapter()
    mem._retrieval_adapter_obj = fake

    # Historical error counts do not keep the backend degraded once the last
    # operation succeeded and the index is clean.
    fake.payload = {
        "errors": 4,
        "last_error": None,
        "manager": {"dirty": False, "errors": 3, "last_error": None},
    }
    stats = mem.retrieval_stats()
    assert stats["degraded"] is False
    assert stats["state"] == "ready"

    # A still-present error or a dirty index keeps the honest degraded state.
    fake.payload = {
        "errors": 4,
        "last_error": None,
        "manager": {"dirty": False, "errors": 3, "last_error": "boom"},
    }
    assert mem.retrieval_stats()["state"] == "degraded"
    fake.payload = {
        "errors": 0,
        "last_error": None,
        "manager": {"dirty": True, "errors": 0, "last_error": None},
    }
    assert mem.retrieval_stats()["state"] == "degraded"


def test_canonical_rollback_reports_healthy_canonical_state():
    mem = _memory(
        retrieval_mode=PRODUCTION,
        qdrant_config=QdrantConfig(
            url="http://127.0.0.1:6333", tenant_id="tenant-a"))

    class _Adapter:
        retriever = None

        def stats(self):
            return {
                "errors": 2,
                "last_error": "QdrantUnavailable: old failure",
                "manager": None,
            }

        def set_backend(self, backend):
            self.retriever = None

    mem._retrieval_adapter_obj = _Adapter()
    assert mem.retrieval_stats()["state"] == "degraded"

    mem.set_retrieval_backend(CANONICAL)
    stats = mem.retrieval_stats()
    assert stats["state"] == "canonical"
    assert stats["degraded"] is False


# ── checkpoint compatibility under the new profile ─────────────────────────


def test_v1_checkpoint_without_memory_ids_loads_deterministically(tmp_path):
    path = str(tmp_path / "legacy_v1.npz")
    n, dim = 2, 16
    np.savez(
        path,
        keys=np.ones((n, dim), dtype=np.float32),
        values=np.ones((n, dim), dtype=np.float32),
        confidences=np.array([0.75, 0.85], dtype=np.float32),
        timestamps=np.array([10.0, 20.0], dtype=np.float64),
        created_ats=np.array([10.0, 20.0], dtype=np.float64),
        update_counts=np.array([1, 2], dtype=np.int32),
        capacity=10,
        dim=dim,
    )

    first = load_checkpoint(path)
    second = load_checkpoint(path)
    first_ids = [slot.memory_id for slot in first.slots]
    second_ids = [slot.memory_id for slot in second.slots]

    assert len(first.slots) == 2
    assert first_ids == second_ids
    assert len(set(first_ids)) == 2
    assert first.slots[1].update_count == 2


def test_checkpoint_restore_under_production_keeps_canonical_state(tmp_path):
    source = _memory()
    add_facts(source, FACTS)
    expected = source.query(QUERY, top_k=10)
    ids = [slot.memory_id for slot in source.memory.slots]
    path = str(tmp_path / "production.npz")
    source.save_checkpoint(path)

    restored = _memory(
        retrieval_mode=PRODUCTION,
        qdrant_config=_unreachable_config())
    load_into_black_hole_memory(restored, path)

    assert [slot.memory_id for slot in restored.memory.slots] == ids
    rows = restored.query(QUERY, top_k=10)
    assert rows_equal(_strip_ids(rows), _strip_ids(expected))
    # The derived index is unavailable, but canonical state is intact.
    assert restored.retrieval_stats()["degraded"] is True


# ── opt-in fusion experiment ───────────────────────────────────────────────


def test_hybrid_fusion_defaults_to_dense_and_validates_values():
    assert QdrantConfig(
        url="http://127.0.0.1:6333").hybrid_fusion == "dense"
    assert QdrantConfig(
        url="http://127.0.0.1:6333", hybrid_fusion="RRF").hybrid_fusion == "rrf"
    with pytest.raises(ValueError, match="hybrid_fusion"):
        QdrantConfig(url="http://127.0.0.1:6333", hybrid_fusion="bogus")
    with pytest.raises(ValueError, match="rrf_k"):
        QdrantConfig(url="http://127.0.0.1:6333", rrf_k=0)


def test_rrf_order_is_deterministic_and_rank_based():
    from minemos_engine.retrieval.qdrant import _rrf_order

    # Dense: A, B, C.  Lexical: C, A, D.
    order = _rrf_order(["A", "B", "C"], ["C", "A", "D"], k=60)
    assert order == ["A", "C", "B", "D"]

    # Ties fall back to dense rank, then lexical rank, then id.
    assert _rrf_order(["A", "B"], ["B", "A"], k=60) == ["A", "B"]
    assert _rrf_order([], ["X"], k=60) == ["X"]

