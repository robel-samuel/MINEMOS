"""Pre-verification gate: skipped model work must not change behaviour.

``_verification_is_inert`` exists only to avoid cross-encoder calls whose
verdict the engine's own guards would ignore.  These tests pin both halves of
that contract: inert pairs are never sent to the verifier, and every pair that
can merge, separate, or supersede still is.
"""

from __future__ import annotations

import pytest

from tests.support.nl_audit.stubs import HashEncoder, StubGate
from minemos_engine.black_hole_memory import BlackHoleMemory
from minemos_engine.structured_candidate import Candidate


class _RecordingVerifier:
    def __init__(self, decision: str = "NEUTRAL"):
        self.decision = decision
        self.calls = []

    def _prediction(self) -> dict:
        if self.decision == "SAME":
            return {
                "decision": "SAME",
                "confidence": 0.99,
                "fwd": {"p_entail": 0.99, "p_contra": 0.005,
                        "p_neutral": 0.005},
                "rev": {"p_entail": 0.99, "p_contra": 0.005,
                        "p_neutral": 0.005},
                "p_entail": 0.99,
                "p_contra": 0.005,
                "p_neutral": 0.005,
                "nli_invocations": 1,
            }
        if self.decision == "CONTRADICTION":
            return {
                "decision": "CONTRADICTION",
                "confidence": 0.95,
                "fwd": {"p_entail": 0.02, "p_contra": 0.95,
                        "p_neutral": 0.03},
                "rev": {"p_entail": 0.02, "p_contra": 0.95,
                        "p_neutral": 0.03},
                "p_entail": 0.02,
                "p_contra": 0.95,
                "p_neutral": 0.03,
                "nli_invocations": 1,
            }
        return {
            "decision": "NEUTRAL",
            "confidence": 0.9,
            "fwd": {"p_entail": 0.02, "p_contra": 0.02,
                    "p_neutral": 0.96},
            "rev": {"p_entail": 0.02, "p_contra": 0.02,
                    "p_neutral": 0.96},
            "p_entail": 0.02,
            "p_contra": 0.02,
            "p_neutral": 0.96,
            "nli_invocations": 1,
        }

    def verify_batch(self, pairs, **kwargs):
        self.calls.append(len(pairs))
        return [self._prediction() for _ in pairs]


def _memory(verifier: _RecordingVerifier) -> BlackHoleMemory:
    return BlackHoleMemory(
        capacity=64,
        dim=384,
        dense_encoder=HashEncoder(),
        nli_gate=StubGate(),
        verifier=verifier,
    )


def _absorb(mem: BlackHoleMemory, subject: str, predicate: str,
            obj: str, timestamp: float) -> dict:
    return mem.absorb(
        Candidate(subject, predicate, obj, timestamp=timestamp),
        timestamp=timestamp,
    )


def _snapshot(mem: BlackHoleMemory) -> dict:
    return {
        "slots": [
            (slot.text, slot.is_active, slot.valid_from, slot.valid_until,
             slot.update_count, slot.supersedes, slot.superseded_by)
            for slot in mem.memory.slots
        ],
        "counters": (
            mem.total_absorbed, mem.total_updates, mem.total_inserts,
            mem.total_temporal_updates, mem.total_evictions,
            mem.total_value_verifications,
        ),
    }


def test_distinct_document_predicates_skip_verification():
    verifier = _RecordingVerifier("SAME")
    mem = _memory(verifier)

    _absorb(mem, "conversation:abc", "segment_0",
            "We reviewed the roadmap today.", 1.0)
    _absorb(mem, "conversation:abc", "segment_1",
            "We reviewed the roadmap today.", 2.0)

    assert verifier.calls == []
    assert mem.total_nli_calls == 0
    assert len(mem.memory.slots) == 2
    assert all(slot.is_active for slot in mem.memory.slots)


def test_same_attribute_pairs_are_still_verified():
    verifier = _RecordingVerifier("NEUTRAL")
    mem = _memory(verifier)

    _absorb(mem, "Alice", "city", "New York", 1.0)
    _absorb(mem, "Alice", "city", "Boston", 2.0)

    # One consolidation check for the pair, then the value-equivalence check
    # the same-attribute branch runs on differing values.
    assert verifier.calls == [1, 1]
    assert mem.total_nli_calls == 1
    assert mem.total_value_verifications == 1
    assert len(mem.memory.slots) == 2


def test_relation_compatible_paraphrase_is_still_verified_and_merges():
    verifier = _RecordingVerifier("SAME")
    mem = _memory(verifier)

    _absorb(mem, "Ada", "is", "chief architect", 1.0)
    result = _absorb(mem, "Ada", "serves as", "chief architect", 2.0)

    assert verifier.calls == [1]
    assert result["action"] == "consolidate"
    assert len(mem.memory.slots) == 1


def test_antonym_contradiction_on_shared_subject_is_still_verified():
    verifier = _RecordingVerifier("CONTRADICTION")
    mem = _memory(verifier)

    _absorb(mem, "conversation:abc", "segment_0",
            "The request was approved today.", 1.0)
    result = _absorb(mem, "conversation:abc", "segment_1",
                     "The request was denied today.", 2.0)

    assert verifier.calls == [1]
    assert result["action"] == "temporal_update"
    assert mem.memory.slots[0].is_active is False
    assert mem.memory.slots[1].is_active is True
    assert mem.total_temporal_updates == 1


def test_antonym_across_subjects_remains_inert():
    verifier = _RecordingVerifier("CONTRADICTION")
    mem = _memory(verifier)

    _absorb(mem, "conversation:a", "segment_0",
            "The request was approved today.", 1.0)
    _absorb(mem, "conversation:b", "segment_1",
            "The request was denied today.", 2.0)

    assert verifier.calls == []
    assert all(slot.is_active for slot in mem.memory.slots)
    assert mem.total_temporal_updates == 0


def test_cross_conversation_same_segment_predicate_is_inert():
    verifier = _RecordingVerifier("SAME")
    mem = _memory(verifier)

    _absorb(mem, "conversation:a", "segment_0",
            "We reviewed the roadmap today.", 1.0)
    _absorb(mem, "conversation:b", "segment_0",
            "We reviewed the roadmap today.", 2.0)

    assert verifier.calls == []
    assert len(mem.memory.slots) == 2
    assert all(slot.is_active for slot in mem.memory.slots)


def test_generic_subject_merge_is_still_verified():
    verifier = _RecordingVerifier("SAME")
    mem = _memory(verifier)

    _absorb(mem, "unknown", "city", "Paris", 1.0)
    result = _absorb(mem, "Alice", "city", "Paris", 2.0)

    assert verifier.calls == [1]
    assert result["action"] == "consolidate"
    assert len(mem.memory.slots) == 1


def test_typo_subject_lexical_restatement_is_still_verified():
    verifier = _RecordingVerifier("CONTRADICTION")
    mem = _memory(verifier)

    _absorb(mem, "Alice", "city", "Paris", 1.0)
    result = _absorb(mem, "Alic", "city", "Paris", 2.0)

    assert verifier.calls == [1]
    assert result["action"] == "consolidate"
    assert len(mem.memory.slots) == 1


@pytest.mark.parametrize("decision", ["SAME", "NEUTRAL", "CONTRADICTION"])
def test_inert_stream_is_verdict_independent(decision):
    verifier = _RecordingVerifier(decision)
    mem = _memory(verifier)

    stream = [
        ("segment_0", "We reviewed the roadmap today."),
        ("segment_1", "We reviewed the roadmap today."),
        ("segment_2", "The launch date moved to Friday."),
        ("segment_3", "The launch date moved to Friday."),
    ]
    for position, (predicate, text) in enumerate(stream):
        _absorb(mem, "conversation:abc", predicate, text,
                float(position + 1))

    assert verifier.calls == []
    assert _snapshot(mem) == _snapshot(_neutral_reference())


def _neutral_reference() -> BlackHoleMemory:
    reference = _memory(_RecordingVerifier("NEUTRAL"))
    stream = [
        ("segment_0", "We reviewed the roadmap today."),
        ("segment_1", "We reviewed the roadmap today."),
        ("segment_2", "The launch date moved to Friday."),
        ("segment_3", "The launch date moved to Friday."),
    ]
    for position, (predicate, text) in enumerate(stream):
        _absorb(reference, "conversation:abc", predicate, text,
                float(position + 1))
    return reference
