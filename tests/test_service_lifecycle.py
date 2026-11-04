"""Operational lifecycle tests for the MINEMOS service.

Covers canonical checkpoint persistence, restart recovery, provenance
tracking, idempotent ingestion, stable-id deletion/redaction, source-level
actions, tenant isolation, readiness reporting, crash-safe fail-closed
startup and checkpoint path confinement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from minemos_service.engine_pool import (
    EngineInitializationError,
    EnginePool,
    EnginePoolError,
)
from minemos_service.provenance import (
    ProvenanceLedger,
    content_digest,
    fact_digest,
)
from phase2.checkpoint import CheckpointPathSecurityError, SCHEMA_VERSION
from tests._service_helpers import (
    make_config,
    make_factory,
    make_models,
    make_pool,
)


SOURCE = {"source_id": "src-1", "source_type": "conversation",
          "conversation_id": "conv-1", "segment_index": 0}


def absorb(pool: EnginePool, tenant: str = "tenant-a", *,
           subject: str = "alice", predicate: str = "city",
           value: str = "Berlin", timestamp: float = 1_700_000_000.0,
           source: dict | None = None, ingestion_id: str | None = None):
    return pool.absorb(
        tenant, subject=subject, predicate=predicate, object=value,
        timestamp=timestamp, confidence=1.0,
        source=dict(source or SOURCE), ingestion_id=ingestion_id)


def reopen(tmp_path: Path, *, tenants=("tenant-a",)) -> tuple:
    config = make_config(tmp_path)
    pool = EnginePool(config, engine_factory=make_factory(),
                      models=make_models())
    pool.initialize(list(tenants))
    return config, pool


# ── persistence and restart ────────────────────────────────────────────────


def test_ingest_persists_across_restart(tmp_path):
    config, pool = make_pool(tmp_path)
    result = absorb(pool)
    assert result["skipped"] is False
    memory_id = result["memory_id"]
    checkpoint = pool.get_runtime("tenant-a").checkpoint_path
    assert checkpoint.exists()
    with np.load(checkpoint, allow_pickle=False) as archive:
        assert int(archive["schema_version"]) == SCHEMA_VERSION
    pool.shutdown()

    _, reopened = reopen(tmp_path)
    try:
        rows = reopened.query("tenant-a", "Berlin", top_k=5)
        assert [row["memory_id"] for row in rows] == [memory_id]
        history = reopened.history("tenant-a", "alice")
        assert len(history) == 1 and history[0]["memory_id"] == memory_id
        assert reopened.verify_provenance("tenant-a")["ok"] is True
        assert reopened.get_runtime("tenant-a").ledger.status()[
            "ingested_observations"] == 1
    finally:
        reopened.shutdown()


def test_second_mutation_creates_backup(tmp_path):
    config, pool = make_pool(tmp_path)
    try:
        absorb(pool, value="Berlin", timestamp=1_700_000_000.0)
        absorb(pool, predicate="country", value="Germany",
               timestamp=1_700_000_100.0)
        info = pool.get_runtime("tenant-a")
        assert info.checkpoint_path.exists()
        assert Path(f"{info.checkpoint_path}.bak").exists()
    finally:
        pool.shutdown()


def test_reingest_after_restart_is_idempotent(tmp_path):
    config, pool = make_pool(tmp_path)
    first = absorb(pool)
    pool.shutdown()

    _, reopened = reopen(tmp_path)
    try:
        second = absorb(reopened)
        assert second["skipped"] is True
        assert second["memory_id"] == first["memory_id"]
        assert reopened.get_runtime("tenant-a").engine.total_absorbed == 1
    finally:
        reopened.shutdown()


def test_redaction_survives_restart(tmp_path):
    config, pool = make_pool(tmp_path)
    secret_value = "the-only-copy-of-this-value"
    memory_id = absorb(pool, value=secret_value)["memory_id"]
    assert pool.redact_memory("tenant-a", memory_id) is True
    pool.shutdown()

    _, reopened = reopen(tmp_path)
    try:
        assert reopened.query("tenant-a", secret_value) == []
        status = reopened.status("tenant-a")
        assert status["engine"]["redacted_slots"] == 1
        assert status["provenance"]["redacted_memories"] == 1
        ledger_text = reopened.get_runtime(
            "tenant-a").ledger.path.read_text(encoding="utf-8")
        assert secret_value not in ledger_text
    finally:
        reopened.shutdown()


def test_delete_source_and_tombstones_survive_restart(tmp_path):
    config, pool = make_pool(tmp_path)
    try:
        absorb(pool, predicate="city", value="Berlin",
               timestamp=1_700_000_000.0)
        absorb(pool, predicate="job", value="engineer",
               timestamp=1_700_000_100.0)
        result = pool.delete_source("tenant-a", "src-1")
        assert result["changed_memories"] == 2
        assert pool.query("tenant-a", "Berlin") == []
        assert pool.get_runtime(
            "tenant-a").ledger.memory_ids_for_source("src-1") == []
    finally:
        pool.shutdown()

    _, reopened = reopen(tmp_path)
    try:
        assert reopened.query("tenant-a", "Berlin") == []
        assert reopened.query("tenant-a", "engineer") == []
        assert reopened.status("tenant-a")["provenance"][
            "source_actions"] == 1
    finally:
        reopened.shutdown()


def test_redact_source_changes_all_linked_memories(tmp_path):
    config, pool = make_pool(tmp_path)
    try:
        first = absorb(pool, predicate="city", value="Berlin")["memory_id"]
        second = absorb(pool, predicate="job", value="engineer")["memory_id"]
        result = pool.redact_source("tenant-a", "src-1")
        assert result["matched_memories"] == 2
        assert result["changed_memories"] == 2
        assert pool.redact_memory("tenant-a", first) is False
        assert pool.redact_memory("tenant-a", second) is False
        assert pool.query("tenant-a", "Berlin") == []
    finally:
        pool.shutdown()


def test_fact_duplicate_is_reported_without_losing_source(tmp_path):
    config, pool = make_pool(tmp_path)
    try:
        first = absorb(pool)
        second = absorb(pool, source={"source_id": "src-2"})
        assert second["duplicate_of"] == first["memory_id"]
        verified = pool.verify_provenance("tenant-a")
        assert verified["ok"] is True
        assert verified["status"]["sources"] == 2
    finally:
        pool.shutdown()


# ── failure handling and confinement ───────────────────────────────────────


def test_backup_recovers_truncated_checkpoint(tmp_path):
    config, pool = make_pool(tmp_path)
    memory_id = absorb(pool)["memory_id"]
    checkpoint = pool.get_runtime("tenant-a").checkpoint_path
    pool.shutdown()
    assert Path(f"{checkpoint}.bak").exists()
    checkpoint.write_bytes(b"truncated")

    _, reopened = reopen(tmp_path)
    try:
        rows = reopened.query("tenant-a", "Berlin")
        assert [row["memory_id"] for row in rows] == [memory_id]
    finally:
        reopened.shutdown()


def test_unreadable_checkpoint_without_backup_fails_closed(tmp_path):
    config, pool = make_pool(tmp_path)
    absorb(pool)
    checkpoint = pool.get_runtime("tenant-a").checkpoint_path
    pool.shutdown()
    Path(f"{checkpoint}.bak").unlink()
    checkpoint.write_bytes(b"not-a-checkpoint")

    broken = EnginePool(config, engine_factory=make_factory(),
                        models=make_models())
    with pytest.raises(EngineInitializationError):
        broken.initialize(["tenant-a"])
    # An empty engine must never be substituted for unreadable canonical data.
    assert "tenant-a" not in broken._runtimes


def test_checkpoint_path_confinement(tmp_path):
    config, pool = make_pool(tmp_path)
    try:
        engine = pool.get_runtime("tenant-a").engine
        outside = tmp_path / "outside.npz"
        with pytest.raises(CheckpointPathSecurityError):
            engine.save_checkpoint(
                str(outside),
                allowed_base_dir=str(config.resolved_checkpoint_dir))
        info = pool.checkpoint("tenant-a")
        assert Path(info["checkpoint"]).is_relative_to(
            config.resolved_checkpoint_dir)
    finally:
        pool.shutdown()


def test_orphaned_provenance_is_reported_not_hidden(tmp_path):
    config, pool = make_pool(tmp_path)
    absorb(pool)
    pool.shutdown()

    ledger_path = config.resolved_provenance_dir / "tenant-a.jsonl"
    ledger = ProvenanceLedger(ledger_path)
    ledger.record_ingest({
        "ingestion_id": "orphan-1",
        "memory_id": "00000000-0000-0000-0000-000000000000",
        "source_id": "src-orphan",
        "content_sha256": content_digest("a", "b", "c"),
        "fact_sha256": fact_digest("a", "b", "c"),
    })

    reopened = EnginePool(config, engine_factory=make_factory(),
                          models=make_models())
    try:
        reopened.initialize(["tenant-a"])
        reconciliation = reopened.status("tenant-a")["reconciliation"]
        assert reconciliation["orphaned"] == 1
        assert reconciliation["newly_orphaned"] == 1
        assert reconciliation["untracked"] == 0
    finally:
        reopened.shutdown()


def test_invalid_tenant_is_rejected(tmp_path):
    config, pool = make_pool(tmp_path)
    try:
        with pytest.raises(Exception):
            pool.get_runtime("../escape")
        with pytest.raises(Exception):
            pool.get_runtime("UPPERCASE")
        pool.shutdown()
        with pytest.raises(EnginePoolError):
            pool.get_runtime("tenant-a")
    finally:
        pool.shutdown()


# ── isolation and readiness ────────────────────────────────────────────────


def test_tenant_engines_are_isolated(tmp_path):
    config, pool = make_pool(tmp_path)
    try:
        absorb(pool, tenant="tenant-a", value="Berlin")
        absorb(pool, tenant="tenant-b", value="Paris")
        rows = pool.query("tenant-a", "Berlin")
        assert len(rows) == 1 and rows[0]["subject"] == "alice"
        assert all("Paris" not in row["text"]
                   for row in pool.query("tenant-a", "Paris"))
        rows = pool.query("tenant-b", "Paris")
        assert len(rows) == 1 and rows[0]["subject"] == "alice"
        assert all("Berlin" not in row["text"]
                   for row in pool.query("tenant-b", "Berlin"))
        assert (config.resolved_checkpoint_dir / "tenant-a" /
                "state.npz").exists()
        assert (config.resolved_checkpoint_dir / "tenant-b" /
                "state.npz").exists()
    finally:
        pool.shutdown()


def test_readiness_and_shutdown_are_explicit(tmp_path):
    config, pool = make_pool(tmp_path)
    absorb(pool)
    readiness = pool.readiness()
    assert readiness["ready"] is True
    assert readiness["checkpoint_dir_writable"] is True
    assert readiness["models_loaded"] is True
    assert pool.health()["status"] == "ok"

    result = pool.shutdown()
    assert result["already_stopped"] is False
    readiness = pool.readiness()
    assert readiness["ready"] is False
    assert pool.health()["status"] == "stopped"
    assert pool.shutdown()["already_stopped"] is True


def test_status_reports_checkpoint_and_provenance(tmp_path):
    config, pool = make_pool(tmp_path)
    try:
        absorb(pool)
        status = pool.status("tenant-a")
        assert status["tenant"] == "tenant-a"
        assert status["checkpoint"]["exists"] is True
        assert status["checkpoint"]["bytes"] > 0
        assert status["provenance"]["ingested_observations"] == 1
        assert status["retrieval"]["backend"] == "canonical"
        assert status["persistence"]["pending"] is False
        assert status["persistence"]["last_error"] is None
        assert status["engine"]["live_slots"] == 1
    finally:
        pool.shutdown()


def test_checkpoint_payload_is_verified_before_response(tmp_path):
    config, pool = make_pool(tmp_path)
    try:
        absorb(pool)
        info = pool.checkpoint("tenant-a")
        assert info["bytes"] > 0
        with np.load(info["checkpoint"], allow_pickle=False) as archive:
            assert "checksum" in archive or "global_checksum" in archive
    finally:
        pool.shutdown()
