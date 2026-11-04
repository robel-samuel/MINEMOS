"""Post-integration production hardening regression tests.

Covers the engine-level gaps confirmed during the production audit:

* construction bounds for ``capacity``/``dim`` (the checkpoint layer already
  enforced these, but a live engine accepted unbounded values);
* malformed encoder output (wrong rank, wrong width, empty, NaN/Inf) is
  rejected at the observation boundary instead of corrupting ranking;
* non-finite/out-of-range confidence and timestamps are rejected;
* privacy redaction destroys slot content while preserving chain integrity,
  and the tombstone survives checkpoint round-trips without retaining text;
* the Qdrant index drops a redacted point and remains usable.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from phase2.black_hole_memory import BlackHoleMemory
from phase2.checkpoint import (
    SCHEMA_VERSION,
    _compute_checksum,
    load_checkpoint,
)
from phase2.memory_state import (
    MAX_EMBEDDING_DIM,
    MAX_MEMORY_CAPACITY,
    MemoryState,
)
from phase2.structured_candidate import Candidate
from tests._retrieval_helpers import (
    FixedEncoder,
    RuleVerifier,
    StubNLI,
    add_facts,
    make_memory,
)


class _BadLexEncoder:
    def __init__(self, vector):
        self._vector = vector

    def encode_full(self, candidate, dim=None):
        return np.asarray(self._vector, dtype=np.float32)


class _BadDenseEncoder:
    def __init__(self, vector):
        self._vector = vector

    def encode(self, text):
        return np.asarray(self._vector, dtype=np.float32)


def _engine(**kwargs):
    return BlackHoleMemory(
        capacity=kwargs.pop("capacity", 10),
        dense_encoder=kwargs.pop("dense_encoder", FixedEncoder()),
        lex_encoder=kwargs.pop("lex_encoder", None),
        nli_gate=StubNLI(),
        verifier=kwargs.pop("verifier", RuleVerifier()),
        **kwargs,
    )


# ── Construction bounds ─────────────────────────────────────────────────────


@pytest.mark.parametrize("capacity", [0, -1, -10**9, True, 10.5, "10"])
def test_memory_state_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        MemoryState(capacity=capacity, dim=64)


@pytest.mark.parametrize("dim", [0, -1, -4096, False, 64.5, "64"])
def test_memory_state_rejects_invalid_dim(dim):
    with pytest.raises(ValueError):
        MemoryState(capacity=10, dim=dim)


def test_memory_state_rejects_excessive_bounds():
    with pytest.raises(ValueError):
        MemoryState(capacity=MAX_MEMORY_CAPACITY + 1, dim=64)
    with pytest.raises(ValueError):
        MemoryState(capacity=10, dim=MAX_EMBEDDING_DIM + 1)


def test_memory_state_accepts_boundary_and_numpy_integers():
    state = MemoryState(capacity=np.int64(1), dim=np.int32(1))
    assert state.capacity == 1 and state.dim == 1


def test_engine_rejects_invalid_capacity_and_dim():
    with pytest.raises(ValueError):
        BlackHoleMemory(capacity=0, nli_gate=StubNLI(),
                        dense_encoder=FixedEncoder())
    with pytest.raises(ValueError):
        BlackHoleMemory(capacity=10, dim=0, nli_gate=StubNLI(),
                        dense_encoder=FixedEncoder())
    with pytest.raises(ValueError):
        BlackHoleMemory(capacity=MAX_MEMORY_CAPACITY + 1,
                        nli_gate=StubNLI(), dense_encoder=FixedEncoder())


# ── Encoder output validation ──────────────────────────────────────────────


@pytest.mark.parametrize("vector", [
    np.ones(3, dtype=np.float32),          # wrong width
    np.ones((1, 384), dtype=np.float32),   # wrong rank
    np.zeros(0, dtype=np.float32),         # empty
    np.full(384, np.nan, dtype=np.float32),
    np.full(384, np.inf, dtype=np.float32),
])
def test_absorb_rejects_invalid_dense_vector(vector):
    mem = _engine(dense_encoder=_BadDenseEncoder(vector))
    try:
        with pytest.raises(ValueError, match="dense embedding"):
            mem.absorb(("Alice", "city", "Paris"), timestamp=1.0)
        assert mem.memory.slots == []
        assert mem._state_version == 0
        assert mem.total_absorbed == 0
    finally:
        mem.close()


@pytest.mark.parametrize("vector", [
    np.ones(3, dtype=np.float32),
    np.zeros((1, 4096), dtype=np.float32),
    np.full(4096, np.nan, dtype=np.float32),
])
def test_absorb_rejects_invalid_lexical_vector(vector):
    mem = _engine(lex_encoder=_BadLexEncoder(vector))
    try:
        with pytest.raises(ValueError, match="lexical embedding"):
            mem.absorb(("Alice", "city", "Paris"), timestamp=1.0)
        assert mem.memory.slots == []
        assert mem._state_version == 0
    finally:
        mem.close()


def test_query_preserves_pinned_non_finite_numeric_contract():
    # Query-side NaN/width-mismatch behaviour is deliberately handled by the
    # retrieval layer (canonical must agree with optimized backends); the
    # absorb-side validation must not leak into this path.
    encoder = FixedEncoder()
    mem = _engine(dense_encoder=encoder)
    try:
        add_facts(mem, [("Alice", "city", "Paris", 1.0)])
        encoder.mapping["bad"] = np.full(384, np.nan, dtype=np.float32)
        rows = mem.query("bad")
        assert len(rows) == 1
        assert np.isnan(rows[0]["similarity"])
    finally:
        mem.close()


# ── Observation metadata validation ────────────────────────────────────────


@pytest.mark.parametrize("confidence", [np.nan, np.inf, -0.1, 1.1, None, "x"])
def test_absorb_rejects_invalid_confidence(confidence):
    mem = _engine()
    try:
        with pytest.raises(ValueError, match="confidence"):
            mem.absorb(("Alice", "city", "Paris"), timestamp=1.0,
                       confidence=confidence)
    finally:
        mem.close()


def test_absorb_rejects_invalid_candidate_confidence():
    mem = _engine()
    try:
        with pytest.raises(ValueError, match="confidence"):
            mem.absorb(Candidate("Alice", "city", "Paris", confidence=np.nan),
                       timestamp=1.0)
    finally:
        mem.close()


def test_absorb_rejects_non_finite_timestamp():
    mem = _engine()
    try:
        with pytest.raises(ValueError, match="timestamp"):
            mem.absorb(("Alice", "city", "Paris"), timestamp=float("nan"))
    finally:
        mem.close()


def test_explicit_confidence_applies_to_tuple_observations():
    mem = _engine()
    try:
        mem.absorb(("Alice", "city", "Paris"), timestamp=1.0, confidence=0.4)
        assert mem.memory.slots[0].confidence == pytest.approx(0.4)
    finally:
        mem.close()


def test_candidate_confidence_is_preserved():
    mem = _engine()
    try:
        mem.absorb(Candidate("Alice", "city", "Paris", confidence=0.7),
                   timestamp=1.0, confidence=0.9)
        assert mem.memory.slots[0].confidence == pytest.approx(0.7)
    finally:
        mem.close()


# ── Redaction semantics ────────────────────────────────────────────────────


def _redaction_engine():
    mem = make_memory(capacity=10)
    add_facts(mem, [
        ("Alice", "city", "Paris", 10.0),
        ("Bob", "city", "Oslo", 11.0),
    ])
    alice = next(s for s in mem.memory.slots if s.debug_subject == "Alice")
    return mem, alice


def test_redact_slot_destroys_content_and_excludes_reads():
    mem, alice = _redaction_engine()
    try:
        slot_id = id(alice)
        assert mem.redact_slot(slot_id) is True
        assert alice.redacted is True
        assert alice.text == ""
        assert alice.debug_subject is None
        assert alice.debug_predicate is None
        assert alice.latest_object is None
        assert float(np.linalg.norm(alice.key)) == 0.0
        assert float(np.linalg.norm(alice.value)) == 0.0
        assert float(np.linalg.norm(alice.dense_emb)) == 0.0
        assert slot_id not in mem.slot_texts
        assert slot_id not in mem.bm25_index.doc_tokens

        assert all(r["subject"] != "Alice" for r in mem.query("Alice", top_k=5))
        assert mem.recall("Alice") == []
        assert mem.get_history("Alice") == []
        status = mem.status()
        assert status["redacted_slots"] == 1
        assert status["live_slots"] == 1
        assert status["total_slots"] == 2

        assert mem.redact_slot(slot_id) is False
    finally:
        mem.close()


def test_redact_subject_matches_and_leaves_other_facts():
    mem = make_memory(capacity=10)
    try:
        add_facts(mem, [
            ("Alice", "city", "Paris", 10.0),
            ("alice", "employer", "Acme", 11.0),
            ("Bob", "city", "Oslo", 12.0),
        ])
        assert mem.redact_subject("alice", exact_match=False) == 2
        assert [s.debug_subject for s in mem.memory.slots] == [None, None, "Bob"]
        assert [r["subject"] for r in mem.query("city", top_k=5)] == ["Bob"]
    finally:
        mem.close()


def test_redacted_slot_keeps_supersession_chain_and_purges_cleanly():
    mem = make_memory(
        capacity=10,
        verifier=RuleVerifier([("Paris", "Boston", "CONTRADICTION")]))
    try:
        add_facts(mem, [
            ("Alice", "city", "Paris", 10.0),
            ("Alice", "city", "Boston", 20.0),
        ])
        old = next(s for s in mem.memory.slots if not s.is_active)
        new = next(s for s in mem.memory.slots if s.is_active)
        assert new.supersedes == id(old)
        assert old.superseded_by == id(new)

        assert mem.redact_slot(id(old)) is True
        assert new.supersedes == id(old)
        assert old.superseded_by == id(new)
        assert mem.temporal_history_violations() == []

        assert mem.purge_redacted() == 1
        assert new.supersedes is None
        assert len(mem.memory.slots) == 1
    finally:
        mem.close()


def test_redaction_is_excluded_from_conflict_detection():
    mem = make_memory(capacity=10, exclusive_predicates={"city"})
    try:
        add_facts(mem, [
            ("Alice", "city", "Paris", 10.0),
            ("Alice", "city", "Rome", 11.0),
        ])
        assert len(mem.exclusive_temporal_conflicts()) == 1
        mem.redact_subject("Alice")
        assert mem.exclusive_temporal_conflicts() == []
    finally:
        mem.close()


# ── Redaction persistence ──────────────────────────────────────────────────


def test_redacted_tombstone_survives_checkpoint_without_content(tmp_path):
    mem, alice = _redaction_engine()
    try:
        mem.redact_slot(id(alice))
        path = str(tmp_path / "state.npz")
        mem.save_checkpoint(path)
    finally:
        mem.close()

    with open(path, "rb") as handle:
        archive = handle.read()
    assert b"Alice" not in archive
    assert b"Paris" not in archive

    restored = make_memory(capacity=10)
    try:
        restored.load_checkpoint(path)
        flags = [s.redacted for s in restored.memory.slots]
        assert flags == [True, False]
        assert restored.query("Alice", top_k=5)[0]["subject"] == "Bob"
        assert restored.recall("Alice") == []
    finally:
        restored.close()


def test_loader_scrubs_content_from_forged_redacted_archive(tmp_path):
    mem = make_memory(capacity=10)
    try:
        add_facts(mem, [("Alice", "city", "Paris", 10.0)])
        path = str(tmp_path / "state.npz")
        mem.save_checkpoint(path)
    finally:
        mem.close()

    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["redacted"] = np.array([True])
    payload["checksum"] = np.array([_compute_checksum(
        payload["keys"], payload["values"], payload["confidences"],
        payload["timestamps"], payload["created_ats"],
        payload["update_counts"], payload["valid_froms"],
        payload["valid_untils"])], dtype=str)
    forged = str(tmp_path / "forged.npz")
    np.savez(forged, **payload)

    loaded = load_checkpoint(forged)
    slot = loaded.slots[0]
    assert slot.redacted is True
    assert slot.text == ""
    assert slot.debug_subject is None
    assert float(np.linalg.norm(slot.key)) == 0.0
    assert float(np.linalg.norm(slot.value)) == 0.0


def test_pre_redaction_checkpoints_still_load(tmp_path):
    mem = make_memory(capacity=10)
    try:
        add_facts(mem, [("Alice", "city", "Paris", 10.0)])
        path = str(tmp_path / "state.npz")
        mem.save_checkpoint(path)
    finally:
        mem.close()

    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload.pop("redacted")
    payload["schema_version"] = np.array(SCHEMA_VERSION - 1)
    legacy = str(tmp_path / "legacy.npz")
    np.savez(legacy, **payload)

    loaded = load_checkpoint(legacy)
    assert loaded.slots[0].redacted is False
    assert loaded.slots[0].text == "Alice city Paris."


# ── Qdrant integration ─────────────────────────────────────────────────────


def _qdrant_config(tmp_path):
    pytest.importorskip("qdrant_client")
    from phase2.retrieval import QdrantConfig
    return QdrantConfig(
        path=str(tmp_path / "qdrant"),
        collection=f"minemos_hardening_{uuid.uuid4().hex[:12]}",
        tenant_id="tenant-hardening",
    )


def test_qdrant_drops_redacted_point_and_remains_complete(tmp_path):
    cfg = _qdrant_config(tmp_path)
    mem = make_memory(capacity=20, backend="qdrant", qdrant_config=cfg)
    try:
        add_facts(mem, [
            ("Alice", "city", "Paris", 10.0),
            ("Bob", "city", "Oslo", 11.0),
        ])
        retriever = mem._get_retrieval_adapter().ensure_retriever()
        assert retriever.stats["points"] == 2

        alice = next(s for s in mem.memory.slots if s.debug_subject == "Alice")
        assert mem.redact_slot(id(alice)) is True

        stats = retriever.stats
        assert stats["points"] == 1
        assert stats["dirty"] is False
        assert stats["complete"] is True
        assert [r["subject"] for r in mem.query("Alice city", top_k=5)] == [
            "Bob"]
    finally:
        mem.close()
