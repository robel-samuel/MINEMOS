"""V9: engine temporal-ordering behaviour the semantic host relies on.

These tests pin the canonical engine's handling of event timestamps that the
opt-in semantic path supplies.  They do not change engine behaviour.  They
cover the scenarios the V9 conversation-level timestamp policy must not
violate: newer-before-older, older-before-newer, equal times, missing times
and a restart in the middle of a correction sequence.

The last test documents, rather than endorses, why the host refuses to mix
ingestion time with real event times: an ingestion-stamped fact sits at the
wall clock, so a genuinely newer event from the archived timeline arrives
"in the past" and cannot supersede it.
"""

from __future__ import annotations

import json

import pytest

from minemos_engine.structured_candidate import Candidate
from tests._retrieval_helpers import (
    FixedEncoder,
    RuleVerifier,
    make_memory,
    unit_vector,
)

WIDTH = 384
CONSTANT = unit_vector(7, width=WIDTH)
OLD, NEW, NEWER = 100.0, 200.0, 300.0
INGESTION = 1_790_000_000.0


def _engine():
    encoder = FixedEncoder(mapping={
        "Alice lives in Paris.": CONSTANT,
        "Alice lives in Rome.": CONSTANT,
        "Alice lives in Oslo.": CONSTANT,
    })
    verifier = RuleVerifier([
        ("Paris", "Rome", "CONTRADICTION"),
        ("Rome", "Paris", "CONTRADICTION"),
        ("Paris", "Oslo", "CONTRADICTION"),
        ("Oslo", "Paris", "CONTRADICTION"),
        ("Rome", "Oslo", "CONTRADICTION"),
        ("Oslo", "Rome", "CONTRADICTION"),
    ])
    return make_memory(capacity=50, verifier=verifier, encoder=encoder)


def _state(mem):
    rows = []
    for slot in mem.memory.slots:
        rows.append({
            "text": str(getattr(slot, "text", "")),
            "active": bool(getattr(slot, "is_active", False)),
            "valid_from": getattr(slot, "valid_from", None),
            "valid_until": getattr(slot, "valid_until", None),
        })
    return sorted(rows, key=lambda row: (row["text"], row["valid_from"] or 0))


def _active(mem):
    return sorted(row["text"] for row in _state(mem) if row["active"])


def _absorb(mem, value, timestamp):
    return mem.absorb(("Alice", "lives in", value), timestamp=timestamp,
                      confidence=1.0)


def test_older_then_newer_supersedes():
    mem = _engine()
    _absorb(mem, "Paris", OLD)
    result = _absorb(mem, "Rome", NEW)

    assert result["action"] == "temporal_update"
    assert _active(mem) == ["Alice lives in Rome."]
    paris = [row for row in _state(mem)
             if row["text"] == "Alice lives in Paris."][0]
    assert paris["valid_from"] == OLD
    assert paris["valid_until"] == NEW


def test_newer_then_older_keeps_the_newer_fact_current():
    mem = _engine()
    _absorb(mem, "Rome", NEW)
    result = _absorb(mem, "Paris", OLD)

    assert result["action"] in {"historical_update", "insert"}
    assert _active(mem) == ["Alice lives in Rome."]
    assert any(row["text"] == "Alice lives in Paris."
               and not row["active"] for row in _state(mem))


def test_equal_timestamps_are_deterministic_last_writer_wins():
    first = _engine()
    _absorb(first, "Paris", OLD)
    _absorb(first, "Rome", OLD)
    assert _active(first) == ["Alice lives in Rome."]

    second = _engine()
    _absorb(second, "Paris", OLD)
    _absorb(second, "Rome", OLD)
    assert _state(second) == _state(first)


def test_correction_sequence_survives_checkpoint_restart(tmp_path):
    mem = _engine()
    _absorb(mem, "Paris", OLD)
    path = str(tmp_path / "ordering.npz")
    mem.save_checkpoint(path)
    mem.close()

    from minemos_engine import BlackHoleMemory

    restored = BlackHoleMemory.from_checkpoint(
        path,
        capacity=50,
        dim=4096,
        dense_encoder=FixedEncoder(mapping={
            "Alice lives in Paris.": CONSTANT,
            "Alice lives in Rome.": CONSTANT,
        }),
        nli_gate=None,
        verifier=RuleVerifier([("Paris", "Rome", "CONTRADICTION")]),
        retrieval_backend="canonical",
    )
    try:
        result = _absorb(restored, "Rome", NEW)
        assert result["action"] == "temporal_update"
        assert _active(restored) == ["Alice lives in Rome."]
    finally:
        restored.close()


def test_replay_of_the_same_sequence_is_idempotent_for_active_state():
    def run():
        mem = _engine()
        _absorb(mem, "Paris", OLD)
        _absorb(mem, "Rome", NEW)
        _absorb(mem, "Oslo", NEWER)
        active = _active(mem)
        mem.close()
        return active

    assert run() == run() == ["Alice lives in Oslo."]


def test_ingestion_stamp_blocks_a_genuinely_newer_archived_event():
    """Documents the V8 hazard; the host policy avoids creating this state.

    Paris is absorbed with an ingestion wall-clock stamp because its source
    had no event time.  Rome is the next message and is genuinely newer on
    the archived timeline, but its explicit time is older than the wall
    clock, so the engine records it as a historical predecessor and keeps
    Paris current.
    """
    mem = _engine()
    _absorb(mem, "Paris", INGESTION)
    result = _absorb(mem, "Rome", NEW)

    assert result["action"] == "historical_update"
    assert _active(mem) == ["Alice lives in Paris."]
    mem.close()
