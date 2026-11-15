"""Long-running messy lifecycle: one archive, many sessions, every mutator.

The other lifecycle suites isolate one mechanism at a time.  This one runs a
single archive through repeated sessions of mixed traffic -- duplicate
replays, corrections, contested values, near-miss identities, conversation
growth, redactions, deletions and checkpoint restarts -- and checks after
every session that the documented invariants still hold:

* capacity is never exceeded and evicted content never returns;
* a closed interval is never active and every interval is well formed;
* redacted slots keep no content in any read path or in the checkpoint;
* duplicate ingestion does not grow the slot set;
* point-in-time reads never show later values;
* a checkpoint restore reproduces the slot state exactly.

The scenarios are deterministic: fixed scripts and a fixed seed, no wall clock.
Nothing here proposes a new retention policy; it verifies the defaults.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest

from minemos_engine.black_hole_memory import BlackHoleMemory
from tests._retrieval_helpers import FixedEncoder, RuleVerifier, StubNLI

CAPACITY = 48
SUBJECTS = ("Alice Hart", "Alice Heart", "Bob Stone", "Carol Diaz",
            "Dan Okafor")


def _new_engine(capacity=CAPACITY, verifier=None, retention_policy="fifo"):
    return BlackHoleMemory(
        capacity=capacity,
        dim=4096,
        dense_encoder=FixedEncoder(),
        nli_gate=StubNLI(),
        verifier=verifier or RuleVerifier(),
        retention_policy=retention_policy,
        strict_entity_identity=True,
    )


def _absorb(mem, subject, predicate, obj, timestamp):
    outcome = mem.absorb((subject, predicate, obj),
                         timestamp=timestamp, confidence=1.0)
    return outcome


def _rounded(value):
    return None if value is None else round(float(value), 9)


def _fingerprint(mem):
    return [
        (
            slot.memory_id,
            slot.debug_subject,
            slot.debug_predicate,
            slot.latest_predicate,
            slot.latest_object,
            slot.text,
            _rounded(slot.confidence),
            _rounded(slot.timestamp),
            _rounded(slot.created_at),
            _rounded(slot.valid_from),
            _rounded(slot.valid_until),
            int(slot.update_count),
            int(slot.contra_count),
            bool(slot.is_active),
            bool(slot.redacted),
        )
        for slot in sorted(mem.memory.slots, key=lambda s: s.memory_id)
    ]


def _restart(mem, path, tmp_path, *, capacity=CAPACITY, verifier=None,
             retention_policy="fifo"):
    """Save, close, reload into a fresh engine and prove parity."""
    before = _fingerprint(mem)
    mem.save_checkpoint(str(path), allowed_base_dir=str(tmp_path))
    mem.close()
    restored = _new_engine(capacity=capacity, verifier=verifier,
                           retention_policy=retention_policy)
    restored.load_checkpoint(str(path), allowed_base_dir=str(tmp_path))
    assert _fingerprint(restored) == before
    return restored


def _assert_structural_invariants(mem, capacity):
    slots = mem.memory.slots
    assert len(slots) <= capacity
    ids = [slot.memory_id for slot in slots]
    assert len(ids) == len(set(ids))
    status = mem.status()
    assert status["capacity"] == capacity
    assert status["total_slots"] == len(slots)
    assert status["redacted_slots"] == sum(1 for slot in slots if slot.redacted)
    for slot in slots:
        assert math.isfinite(float(slot.confidence))
        assert math.isfinite(float(slot.timestamp))
        assert math.isfinite(float(slot.valid_from))
        if slot.valid_until is not None:
            assert slot.valid_until >= slot.valid_from
            assert slot.is_active is False
        if slot.redacted:
            assert slot.text == ""
            assert slot.debug_subject is None
            assert slot.debug_predicate is None
            assert slot.latest_predicate is None
            assert slot.latest_object is None
            assert slot.dense_emb is None or not np.any(slot.dense_emb)


def _read_texts(mem, subjects, probe="archive status update"):
    parts = []
    for subject in subjects:
        for row in mem.recall(subject, active_only=False):
            parts.append(row.get("text") or "")
    top_k = max(10, min(100, len(mem.memory.slots)))
    for row in mem.query(probe, top_k=top_k):
        parts.append(row.get("text") or "")
    return "\n".join(parts).casefold()


def _ids(mem):
    return {slot.memory_id for slot in mem.memory.slots}


def _by_subject(mem, subject):
    return [slot for slot in mem.memory.slots
            if slot.debug_subject == subject]


# ---------------------------------------------------------------------------
# Eight sessions of mixed traffic with a restart after every session
# ---------------------------------------------------------------------------


def test_eight_session_messy_archive_keeps_its_contract(tmp_path):
    path = tmp_path / "archive.npz"
    verifier = RuleVerifier([
        ("value-", "corrected-", "CONTRADICTION"),
    ])
    mem = _new_engine(verifier=verifier)
    timestamp = 100.0
    redacted_contents = set()
    corrected = []  # (subject, predicate, old, new, old_timestamp)
    conversation_segments = defaultdict(int)

    try:
        for session in range(8):
            subject = SUBJECTS[session % len(SUBJECTS)]
            predicate = ("works at", "lives in", "role is", "lives in")[
                session % 4]
            value = f"value-{session}"

            before = _ids(mem)
            first = _absorb(mem, subject, predicate, value, timestamp)
            assert first["action"] == "insert"
            assert len(_ids(mem)) == len(before) + 1

            replay = _absorb(mem, subject, predicate, value, timestamp)
            assert replay["action"] == "consolidate"
            assert replay["memory_id"] == first["memory_id"]
            assert _ids(mem) == before | {first["memory_id"]}
            timestamp += 1.0

            if session % 2 == 1:
                # A correction closes the earlier interval; the old value
                # remains reachable only at its own time.
                old_mid = first["memory_id"]
                old_ts = float(first["valid_from"])
                new = _absorb(mem, subject, predicate,
                              f"corrected-{session}", timestamp)
                assert new["memory_id"] != old_mid
                corrected.append(
                    (subject, predicate, value, f"corrected-{session}",
                     old_ts))
                timestamp += 1.0

            # Near-miss entity names are distinct identities, not variants of
            # one subject.
            twin = ("Alice Heart" if subject == "Alice Hart" else "Alice Hart")
            _absorb(mem, twin, "uses", f"twin-tool-{session}", timestamp)
            timestamp += 1.0

            # Conversation growth, including a duplicated segment replay.
            conversation = "conversation:alpha" if session % 2 == 0 \
                else "conversation:beta"
            segment = conversation_segments[conversation]
            text = f"session {session} covered topic {segment} in detail"
            _absorb(mem, conversation, f"segment_{segment}", text, timestamp)
            _absorb(mem, conversation, f"segment_{segment}", text, timestamp)
            conversation_segments[conversation] = segment + 1
            timestamp += 1.0

            if session in (2, 5):
                # Privacy: the latest fact for the session subject is
                # irreversibly redacted; its text must not survive anywhere.
                victim = _absorb(mem, subject, "works at",
                                 f"private-{session}", timestamp)
                assert mem.redact_memory(victim["memory_id"]) is True
                redacted_contents.add(f"private-{session}")
                timestamp += 1.0

            mem = _restart(mem, path, tmp_path, verifier=verifier)
            _assert_structural_invariants(mem, CAPACITY)
            texts = _read_texts(mem, list(SUBJECTS))
            for content in redacted_contents:
                assert content.casefold() not in texts

        # The scripted corrections still answer at their own time and are
        # superseded at the present.
        for subject, predicate, old, new, old_ts in corrected:
            historical = " ".join(
                row["text"] for row in mem.recall(
                    subject, predicate=predicate, as_of=old_ts + 0.5,
                    active_only=False)).casefold()
            assert old.casefold() in historical
            assert new.casefold() not in historical
            active = " ".join(
                row["text"] for row in mem.recall(
                    subject, predicate=predicate)).casefold()
            assert new.casefold() in active
            assert old.casefold() not in active

        # Distinct near-miss identities never merged.
        subjects = {slot.debug_subject for slot in mem.memory.slots}
        assert "Alice Hart" in subjects and "Alice Heart" in subjects
        assert mem.total_evictions == 0
        assert mem.status()["redacted_slots"] >= 2

        # A final restart after the checks reproduces exactly.
        mem = _restart(mem, path, tmp_path, verifier=verifier)
        _assert_structural_invariants(mem, CAPACITY)
    finally:
        mem.close()


# ---------------------------------------------------------------------------
# Replay is idempotent across restarts
# ---------------------------------------------------------------------------


def _scripted_archive(mem, start=10.0):
    timestamp = start
    ids = []
    for index in range(30):
        subject = SUBJECTS[index % len(SUBJECTS)]
        predicate = ("works at", "lives in", "role is", "uses")[index % 4]
        result = _absorb(mem, subject, predicate, f"scripted-{index}",
                         timestamp)
        ids.append(result["memory_id"])
        timestamp += 1.0
    for segment in range(10):
        result = _absorb(mem, "conversation:alpha", f"segment_{segment}",
                         f"archived message number {segment}", timestamp)
        ids.append(result["memory_id"])
        timestamp += 1.0
    return ids, timestamp


def test_replay_is_idempotent_across_restarts(tmp_path):
    path = tmp_path / "replay.npz"
    mem = _new_engine()
    try:
        first_ids, next_ts = _scripted_archive(mem)
        before = _ids(mem)
        assert before == set(first_ids)
        mem = _restart(mem, path, tmp_path)

        # Replay the identical script (same timestamps) into the restored
        # archive: every observation consolidates into its own slot.
        replay_ids, _ = _scripted_archive(mem)
        assert set(replay_ids) == before
        assert _ids(mem) == before
        assert len(mem.memory.slots) == len(before)
        update_counts = {
            slot.memory_id: slot.update_count for slot in mem.memory.slots}
        assert all(count >= 2 for count in update_counts.values())

        mem = _restart(mem, path, tmp_path)
        _assert_structural_invariants(mem, CAPACITY)

        # A third replay after another restart still adds no slot.
        _scripted_archive(mem)
        assert len(mem.memory.slots) == len(before)
        assert _ids(mem) == before
        mem = _restart(mem, path, tmp_path)
        assert _ids(mem) == before
    finally:
        mem.close()


# ---------------------------------------------------------------------------
# Corrections, contested values and point-in-time answers survive restarts
# ---------------------------------------------------------------------------


def test_timeline_corrections_and_contested_values_survive_restart(tmp_path):
    path = tmp_path / "timeline.npz"
    verifier = RuleVerifier([
        ("Rome", "Lima", "CONTRADICTION"),
        ("Lima", "Rome", "CONTRADICTION"),
        ("analyst", "manager", "CONTRADICTION"),
        ("manager", "analyst", "CONTRADICTION"),
    ])
    mem = _new_engine(verifier=verifier)
    try:
        _absorb(mem, "Alice", "lives in", "Rome", 10.0)
        _absorb(mem, "Alice", "lives in", "Lima", 20.0)
        _absorb(mem, "Bob", "works at", "Acme", 30.0)
        _absorb(mem, "Bob", "works at", "Globex", 40.0)
        _absorb(mem, "Carol", "role is", "analyst", 50.0)
        _absorb(mem, "Carol", "role is", "manager", 60.0)
        _absorb(mem, "conversation:alpha", "segment_0",
                "Alice drives the Volvo", 70.0)
        _absorb(mem, "conversation:alpha", "segment_1",
                "Bob owns the Volvo", 80.0)
        # A future-dated observation must stay out of earlier reads.
        _absorb(mem, "Alice", "lives in", "Kyiv", 200.0)

        def check(engine):
            assert "rome" in " ".join(
                row["text"] for row in engine.recall(
                    "Alice", predicate="lives in", as_of=15.0,
                    active_only=False)).casefold()
            assert "lima" not in " ".join(
                row["text"] for row in engine.recall(
                    "Alice", predicate="lives in", as_of=15.0,
                    active_only=False)).casefold()
            assert "kyiv" not in " ".join(
                row["text"] for row in engine.recall(
                    "Alice", predicate="lives in", as_of=150.0,
                    active_only=False)).casefold()

            # Carol's linked correction is settled; Bob's unlinked second
            # value is contested, answered but never settled.
            carol = engine.eligible_query(
                "What is Carol's role?", subject="Carol")
            assert carol["status"] == "answered"
            assert carol["evidence"]["state"] == "supported"
            assert carol["temporal"]["answer_is_settled"] is True

            bob = engine.eligible_query(
                "Where does Bob work?", subject="Bob")
            assert bob["status"] == "answered"
            assert bob["evidence"]["state"] == "contradicted"
            assert bob["temporal"]["answer_is_settled"] is False
            texts = " ".join(row["text"] for row in bob["rows"]).casefold()
            assert "acme" in texts and "globex" in texts

            hop = engine.eligible_query(
                "Who owns the Volvo that Alice drives?",
                subject="conversation:alpha")
            assert hop["status"] == "answered"
            hop_texts = " ".join(
                row["text"] for row in hop["rows"]).casefold()
            assert "alice drives" in hop_texts
            assert "bob owns" in hop_texts

        check(mem)
        mem = _restart(mem, path, tmp_path, verifier=verifier)
        _assert_structural_invariants(mem, CAPACITY)
        check(mem)
    finally:
        mem.close()


# ---------------------------------------------------------------------------
# Capacity pressure: no resurrection, no leak, no duplicate id
# ---------------------------------------------------------------------------


def test_retention_pressure_never_resurrects_or_leaks(tmp_path):
    capacity = 12
    path = tmp_path / "pressure.npz"
    mem = _new_engine(capacity=capacity)
    written = []  # (memory_id, subject, predicate, object)
    redacted = {}
    purged = set()
    timestamp = 1_000.0
    try:
        for index in range(60):
            subject = f"Metric {index:02d}"
            value = f"reading-{index:02d}"
            result = _absorb(mem, subject, "reports", value, timestamp)
            written.append(
                (result["memory_id"], subject, "reports", value))
            timestamp += 1.0

            if index in (9, 29, 49):
                victim_id = written[-1][0]
                assert mem.redact_memory(victim_id) is True
                redacted[victim_id] = value

            if index % 10 == 9:
                mem = _restart(mem, path, tmp_path, capacity=capacity)
                _assert_structural_invariants(mem, capacity)
                assert len(mem.memory.slots) == min(capacity, index + 1)

                present = _ids(mem)
                evicted = {mid for mid, *_ in written} - present
                # Evicted content is gone from canonical state and from
                # retrieval; a reload cannot bring it back.
                assert not (evicted & present)
                texts = _read_texts(mem, [f"Metric {i:02d}"
                                          for i in range(60)])
                for mid, subject, _, value in written:
                    if mid in evicted:
                        assert value.casefold() not in texts
                for mid, value in redacted.items():
                    if mid in present:
                        assert value.casefold() not in texts

                if index == 29:
                    removed = mem.purge_redacted()
                    assert removed >= 1
                    purged |= {
                        mid for mid in redacted if mid not in _ids(mem)}
                    mem = _restart(mem, path, tmp_path, capacity=capacity)
                    assert not (purged & _ids(mem))

        # Oldest observations are the ones gone; the newest still answer.
        present = _ids(mem)
        assert written[-1][0] in present
        assert written[0][0] not in present
        assert mem.total_evictions >= 40
        active_texts = " ".join(
            row["text"] for row in mem.recall("Metric 59")
        ).casefold()
        assert "reading-59" in active_texts
        assert mem.recall("Metric 00") == []
    finally:
        mem.close()


# ---------------------------------------------------------------------------
# Redaction, purge and deletion are final across a restart
# ---------------------------------------------------------------------------


def test_deletion_and_redaction_are_final_across_a_restart(tmp_path):
    path = tmp_path / "deletion.npz"
    mem = _new_engine()
    try:
        work = _absorb(mem, "Alice", "works at", "Acme", 1.0)
        home = _absorb(mem, "Alice", "lives in", "Lisbon", 2.0)
        car = _absorb(mem, "Bob Stone", "drives", "Volvo", 3.0)
        assert mem.redact_memory(work["memory_id"]) is True
        assert mem.purge_subject("Bob Stone") == 1
        deleted_home = mem.delete_memory(home["memory_id"])
        assert deleted_home is True

        mem = _restart(mem, path, tmp_path)
        _assert_structural_invariants(mem, CAPACITY)
        texts = _read_texts(mem, ["Alice", "Bob Stone"])
        for content in ("acme", "lisbon", "volvo"):
            assert content not in texts
        assert mem.recall("Bob Stone") == []
        assert mem.recall("Alice", predicate="lives in") == []
        redacted_slots = [slot for slot in mem.memory.slots if slot.redacted]
        assert len(redacted_slots) == 1  # the work fact keeps a tombstone
        assert redacted_slots[0].memory_id == work["memory_id"]

        # Re-ingesting deleted content after recovery is allowed, but it is a
        # new memory with a new id -- the deleted id never returns.
        replacement = _absorb(mem, "Alice", "lives in", "Lisbon", 4.0)
        assert replacement["memory_id"] != home["memory_id"]
        assert home["memory_id"] not in _ids(mem)
        assert "lisbon" in " ".join(
            row["text"] for row in mem.recall(
                "Alice", predicate="lives in")).casefold()

        mem = _restart(mem, path, tmp_path)
        _assert_structural_invariants(mem, CAPACITY)
        assert home["memory_id"] not in _ids(mem)
        assert replacement["memory_id"] in _ids(mem)
    finally:
        mem.close()


# ---------------------------------------------------------------------------
# A long conversation over repeated restarts keeps point-in-time reads honest
# ---------------------------------------------------------------------------


def test_long_conversation_history_answers_point_in_time_questions(tmp_path):
    path = tmp_path / "conversation.npz"
    capacity = 80
    mem = _new_engine(capacity=capacity)
    segments = []  # (segment, text, timestamp)
    timestamp = 500.0
    try:
        for session in range(6):
            for within in range(10):
                segment = session * 10 + within
                if segment == 17:
                    text = ("the migration plan covers the harbour rollout "
                            + "and the delta checklist " * 40).strip()
                else:
                    text = (f"the migration plan covers stage {segment} "
                            f"with owner team {segment % 3}")
                _absorb(mem, "conversation:alpha", f"segment_{segment}",
                        text, timestamp)
                _absorb(mem, "conversation:alpha", f"segment_{segment}",
                        text, timestamp)  # duplicate replay
                segments.append((segment, text, timestamp))
                timestamp += 1.0
            mem = _restart(mem, path, tmp_path, capacity=capacity)
            _assert_structural_invariants(mem, capacity)

        assert len(mem.memory.slots) == len(segments)
        for segment, text, ts in segments[::9]:
            before = mem.recall(
                "conversation:alpha", predicate=f"segment_{segment}",
                as_of=ts - 0.5, active_only=False)
            assert before == []
            at_time = mem.recall(
                "conversation:alpha", predicate=f"segment_{segment}",
                as_of=ts, active_only=False)
            assert len(at_time) == 1
            assert at_time[0]["text"].casefold().find(
                text.casefold()[:40]) >= 0

        # A verified question at the end of the conversation sees all plan
        # messages and none from beyond the requested time.
        outcome = mem.eligible_query(
            "What does the migration plan cover for stage 51?",
            subject="conversation:alpha", as_of=segments[51][2])
        assert outcome["status"] == "answered"
        assert outcome["rows"]
        assert all(row["valid_from"] <= segments[51][2]
                   for row in outcome["rows"])
        assert "stage 51" in " ".join(
            row["text"] for row in outcome["rows"]).casefold()

        early = mem.eligible_query(
            "harbour rollout delta checklist",
            subject="conversation:alpha",
            as_of=segments[16][2])
        assert early["status"] == "abstained"
        assert early["rows"] == []

        after = mem.eligible_query(
            "harbour rollout delta checklist",
            subject="conversation:alpha",
            as_of=segments[17][2])
        assert after["status"] == "answered"
        assert "harbour rollout" in " ".join(
            row["text"] for row in after["rows"]).casefold()
    finally:
        mem.close()
