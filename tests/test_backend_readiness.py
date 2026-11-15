"""Readiness reporting for optional retrieval backends.

A readiness check must stay bounded while an external index is unreachable and
must not report a required backend as available.  These tests use the embedded
Qdrant index and a fake probe client; no server is required.
"""

from __future__ import annotations

import pytest

from tests._retrieval_helpers import make_memory

pytest.importorskip("qdrant_client")

from minemos_engine.retrieval import QdrantConfig  # noqa: E402


def _config(tmp_path):
    return QdrantConfig(
        path=str(tmp_path / "qdrant"),
        collection="readiness_" + tmp_path.name,
        tenant_id="tenant-readiness",
        candidate_limit=16,
    )


def _engine(tmp_path):
    mem = make_memory(
        capacity=10, backend="qdrant", qdrant_config=_config(tmp_path))
    mem._get_retrieval_adapter().ensure_retriever()
    return mem


class _FakeProbeClient:
    def __init__(self, fail: bool = True):
        self.fail = fail
        self.calls = 0

    def get_collections(self):
        self.calls += 1
        if self.fail:
            raise ConnectionError("endpoint unreachable")
        return {"collections": []}

    def close(self):
        pass


def test_canonical_readiness_is_always_available():
    mem = make_memory()
    report = mem.retrieval_readiness()
    assert report["backend"] == "canonical"
    assert report["required"] is False
    assert report["available"] is True
    assert report["state"] == "ok"


def test_local_qdrant_readiness_is_available(tmp_path):
    mem = _engine(tmp_path)
    try:
        report = mem.retrieval_readiness()
        assert report["backend"] == "qdrant"
        assert report["required"] is True
        assert report["available"] is True
        assert report["state"] == "ok"
        assert report["initialized"] is True
    finally:
        mem.close()


def test_probe_is_cached_and_an_outage_is_reported_as_unavailable(tmp_path):
    mem = _engine(tmp_path)
    try:
        retriever = mem._get_retrieval_adapter().retriever
        fake = _FakeProbeClient(fail=True)
        retriever.mode = "server"
        retriever._probe_client = fake

        first = retriever.probe(ttl_s=60)
        second = retriever.probe(ttl_s=60)
        assert first["ok"] is False
        assert second["ok"] is False
        assert fake.calls == 1, "a cached probe must not re-contact the backend"

        report = mem.retrieval_readiness()
        assert report["available"] is False
        assert report["state"] == "unavailable"

        fake.fail = False
        recovered = retriever.probe(ttl_s=0.0)
        assert recovered["ok"] is True
        assert mem.retrieval_readiness()["available"] is True
    finally:
        mem.close()


def test_retrieval_stats_without_probe_never_contacts_the_backend(tmp_path):
    mem = _engine(tmp_path)
    try:
        retriever = mem._get_retrieval_adapter().retriever
        original = retriever.client

        class _NoCount:
            def __getattr__(self, name):
                if name == "count":
                    raise AssertionError("probe=False must not count points")
                return getattr(original, name)

        retriever.client = _NoCount()
        stats = mem.retrieval_stats(probe=False)
        assert stats["manager"]["points"] is None
        assert stats["state"] == "ready"
    finally:
        mem.close()
