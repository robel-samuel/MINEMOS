"""Phase 29 temporal-integrity tests.

Covers supersession-link validation, closed-history SAME recurrence, and the
value-only verifier telemetry added in Phase 29.  The characterization test for
unlinked legacy conflicts is deliberately expected to pass by *not* reporting
a violation; Phase 29 measures that gap rather than auto-grouping by predicate
(which would re-create the R2 false positives).
"""

import numpy as np
import pytest

from phase2.black_hole_memory import BlackHoleMemory
from phase2.checkpoint import CheckpointValidationError, load_checkpoint as load_ckpt
from phase2.structured_candidate import Candidate


class _RuleVerifier:
    """First matching (slot substring, candidate substring) rule wins."""

    def __init__(self, rules, default="NEUTRAL"):
        self._rules = list(rules)
        self._default = default

    def _decide(self, slot_text, cand_text):
        for slot_marker, cand_marker, decision in self._rules:
            if slot_marker in slot_text and cand_marker in cand_text:
                return decision
        return self._default

    def verify_batch(self, pairs, **kwargs):
        results = []
        for slot_text, cand_text in pairs:
            decision = self._decide(slot_text, cand_text)
            contra = decision == "CONTRADICTION"
            same = decision == "SAME"
            results.append({
                "decision": decision,
                "confidence": 0.99,
                "p_entail": 0.99 if same else 0.01,
                "p_contra": 0.99 if contra else 0.01,
                "p_neutral": 0.98 if decision == "NEUTRAL" else 0.01,
                "fwd": {
                    "p_contra": 0.99 if contra else 0.01,
                    "p_neutral": 0.98 if decision == "NEUTRAL" else 0.01,
                },
                "rev": {
                    "p_contra": 0.99 if contra else 0.01,
                    "p_neutral": 0.98 if decision == "NEUTRAL" else 0.01,
                },
            })
        return results


@pytest.fixture(scope="module")
def models():
    from benchmarks.phase18_experiment import DenseSemanticEncoder
    from benchmarks.phase16_experiment import NliSemanticGate

    return DenseSemanticEncoder(), NliSemanticGate()


def _memory(models, verifier=None):
    dense, nli = models
    return BlackHoleMemory(
        capacity=10,
        dense_encoder=dense,
        nli_gate=nli,
        verifier=verifier,
    )


def _reasons(mem):
    return [v["reason"] for v in mem.temporal_history_violations()]


def _active_texts(mem):
    return sorted(s.text for s in mem.memory.slots if s.is_active)


def _intervals(mem, subject, predicate):
    return [
        (row["valid_from"], row["valid_until"])
        for row in mem.get_history(subject, predicate)
    ]


# ── Link validation ────────────────────────────────────────────────────────


def test_cross_subject_link_is_reported_once(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Lisbon", timestamp=10.0))
    mem.absorb(Candidate("May", "city", "Porto", timestamp=20.0))

    a, b = mem.memory.slots
    a.supersedes = id(b)
    b.superseded_by = id(a)

    violations = mem.temporal_history_violations()
    assert [v["reason"] for v in violations] == ["inconsistent_link"]
    assert violations[0]["subject_match"] is False
    assert violations[0]["attribute_match"] is True
    assert len(_active_texts(mem)) == 2


def test_cross_attribute_link_is_reported(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Lisbon", timestamp=10.0))
    mem.absorb(Candidate("Maya", "employer", "Acme", timestamp=20.0))

    a, b = mem.memory.slots
    a.supersedes = id(b)
    b.superseded_by = id(a)

    violations = mem.temporal_history_violations()
    assert [v["reason"] for v in violations] == ["inconsistent_link"]
    assert violations[0]["subject_match"] is True
    assert violations[0]["attribute_match"] is False


def test_consistent_chain_link_still_groups(models):
    mem = _memory(models, verifier=_RuleVerifier(
        [("Rome", "Venice", "CONTRADICTION")]
    ))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))

    assert _intervals(mem, "Maya", "city") == [(10.0, 20.0), (20.0, None)]
    assert mem.temporal_history_violations() == []


# ── Closed-history SAME recurrence ─────────────────────────────────────────


def test_closed_history_recurrence_targets_active_head(models):
    mem = _memory(models, verifier=_RuleVerifier([
        ("Rome", "Venice", "CONTRADICTION"),
        ("Rome", "Rome", "SAME"),
        ("Venice", "Rome", "NEUTRAL"),
    ]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    result = mem.absorb(Candidate("Maya", "city", "Rome", timestamp=30.0))

    assert result["action"] == "temporal_update"
    assert _intervals(mem, "Maya", "city") == [
        (10.0, 20.0),
        (20.0, 30.0),
        (30.0, None),
    ]
    assert len(_active_texts(mem)) == 1
    assert mem.temporal_history_violations() == []


def test_closed_history_recurrence_consolidates_when_value_unchanged(models):
    mem = _memory(models, verifier=_RuleVerifier([
        ("Rome", "Venice", "CONTRADICTION"),
        ("Rome", "Rome", "SAME"),
        ("Venice", "Rome", "NEUTRAL"),
    ]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    # Close the historical Rome slot and make the active Venice slot carry
    # the same value as the recurrence, so the recurrence is a re-wording.
    mem.memory.slots[1].latest_object = "Rome"
    result = mem.absorb(Candidate("Maya", "city", "Rome", timestamp=30.0))

    assert result["action"] == "consolidate"
    assert len(_active_texts(mem)) == 1
    assert _intervals(mem, "Maya", "city") == [(10.0, 20.0), (20.0, None)]
    assert mem.temporal_history_violations() == []


def test_multi_valued_closed_recurrence_does_not_supersede(models):
    mem = _memory(models, verifier=_RuleVerifier([
        ("red team", "red team", "SAME"),
    ]))
    mem.absorb(Candidate("Maya", "member of", "red team", timestamp=10.0))
    mem.absorb(Candidate("Maya", "member of", "blue team", timestamp=20.0))
    mem.memory.slots[0].valid_until = 15.0

    result = mem.absorb(Candidate("Maya", "member of", "red team", timestamp=30.0))

    assert result["action"] == "insert"
    assert len(_active_texts(mem)) == 2
    assert mem.temporal_history_violations() == []


# ── Conservative retention and known gaps ─────────────────────────────────


def test_conservatively_retained_status_is_not_a_violation(models):
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Project X", "status", "on hold", timestamp=10.0))
    mem.absorb(Candidate("Project X", "status", "awaiting approval", timestamp=20.0))

    assert len(_active_texts(mem)) == 2
    assert mem.temporal_history_violations() == []


def test_unlinked_exclusive_conflict_remains_a_known_false_negative(models):
    """Characterization, not a desired end state: without provenance the
    invariant cannot distinguish an imported conflict from the conservative
    retention above, so it stays silent."""
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))

    assert len(_active_texts(mem)) == 2
    assert mem.temporal_history_violations() == []


# ── Verifier telemetry ─────────────────────────────────────────────────────


def test_value_verifier_calls_are_counted_separately(models):
    mem = _memory(models, verifier=_RuleVerifier([
        ("Alice city New York.", "Alice city Boston.", "SAME"),
    ]))
    mem.absorb(Candidate("Alice", "city", "New York", timestamp=10.0))
    result = mem.absorb(Candidate("Alice", "city", "Boston", timestamp=20.0))

    assert result["action"] == "temporal_update"
    status = mem.status()
    assert status["total_value_verifications"] == 1
    assert status["total_nli_calls"] == 1


def test_identical_values_do_not_trigger_value_verifier(models):
    mem = _memory(models, verifier=_RuleVerifier([
        ("Alice city Boston.", "Alice city Boston.", "SAME"),
    ]))
    mem.absorb(Candidate("Alice", "city", "Boston", timestamp=10.0))
    mem.absorb(Candidate("Alice", "city", "Boston", timestamp=20.0))

    status = mem.status()
    assert status["total_value_verifications"] == 0
    assert len(_active_texts(mem)) == 1


# ── Recurrence branches ────────────────────────────────────────────────────
# Expectations below were fixed before the implementation was extended to meet
# them.  Each states the required state, not the engine's current output.


_RECURRENCE_RULES = [
    ("Rome", "Venice", "CONTRADICTION"),
    ("Rome", "Rome", "SAME"),
    ("Venice", "Rome", "NEUTRAL"),
]


def test_recurrence_into_interval_gap_splices_between_versions(models):
    """A recurrence that falls between two intervals must become the interval
    [t, successor.valid_from), linked to the predecessor and the successor.

    Required: action historical_update; intervals [(10,20),(25,30),(30,None)];
    the pair (predecessor, new) and (new, successor) are reciprocal; no
    violations.
    """
    mem = _memory(models, verifier=_RuleVerifier(_RECURRENCE_RULES))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=30.0))
    # The engine produces contiguous chains; close the first interval early to
    # create the gap that a restored or repaired history can legitimately hold.
    mem.memory.slots[0].valid_until = 20.0

    result = mem.absorb(Candidate("Maya", "city", "Rome", timestamp=25.0))
    slots = mem.memory.slots

    assert result["action"] == "historical_update"
    assert _intervals(mem, "Maya", "city") == [
        (10.0, 20.0),
        (25.0, 30.0),
        (30.0, None),
    ]
    assert _active_texts(mem) == ["Maya city Venice."]
    assert slots[0].superseded_by == id(slots[2])
    assert slots[2].supersedes == id(slots[0])
    assert slots[2].superseded_by == id(slots[1])
    assert slots[1].supersedes == id(slots[2])
    assert mem.temporal_history_violations() == []


def test_recurrence_without_active_head_links_prior_version(models):
    """When every version of an attribute is closed, a recurrence is a new
    version after a gap.  It must be inserted as the only active state and
    linked to the nearest prior version so history stays traversable.

    Required: action insert; intervals [(10,20),(20,25),(30,None)]; the
    (nearest predecessor, new) pair is reciprocal; no violations.
    """
    mem = _memory(models, verifier=_RuleVerifier(_RECURRENCE_RULES))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    mem.memory.slots[1].valid_until = 25.0

    result = mem.absorb(Candidate("Maya", "city", "Rome", timestamp=30.0))
    slots = mem.memory.slots

    assert result["action"] == "insert"
    assert _intervals(mem, "Maya", "city") == [
        (10.0, 20.0),
        (20.0, 25.0),
        (30.0, None),
    ]
    assert _active_texts(mem) == ["Maya city Rome."]
    assert slots[2].supersedes == id(slots[1])
    assert slots[1].superseded_by == id(slots[2])
    assert mem.temporal_history_violations() == []


def test_recurrence_at_head_valid_from_replaces_in_place(models):
    """A recurrence arriving exactly at the active head's valid_from must use
    the equal-timestamp policy: one interval boundary is preserved, the head
    is replaced in place, and the chain stays reciprocal.

    Required: action temporal_update; intervals [(10,20),(20,None)]; one active
    value; no violations.
    """
    mem = _memory(models, verifier=_RuleVerifier(_RECURRENCE_RULES))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))

    result = mem.absorb(Candidate("Maya", "city", "Rome", timestamp=20.0))
    slots = mem.memory.slots

    assert result["action"] == "temporal_update"
    assert _intervals(mem, "Maya", "city") == [(10.0, 20.0), (20.0, None)]
    assert _active_texts(mem) == ["Maya city Rome."]
    assert slots[0].superseded_by == id(slots[1])
    assert slots[1].supersedes == id(slots[0])
    assert mem.temporal_history_violations() == []


# ── Structural link corruption ─────────────────────────────────────────────


def test_dangling_supersession_link_is_reported(models):
    """A link to an object that is not in memory cannot be grouped; it must be
    reported as dangling_link rather than silently skipped."""
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Lisbon", timestamp=10.0))
    mem.absorb(Candidate("Maya", "role", "analyst", timestamp=20.0))
    mem.memory.slots[0].superseded_by = 10 ** 12 + 7

    violations = mem.temporal_history_violations()
    assert [v["reason"] for v in violations] == ["dangling_link"]
    assert violations[0]["slot_index"] == 0


def test_self_referential_supersession_link_is_reported(models):
    """A slot that supersedes itself is structurally invalid and must be
    reported as self_link."""
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Lisbon", timestamp=10.0))
    mem.absorb(Candidate("Maya", "role", "analyst", timestamp=20.0))
    slot = mem.memory.slots[0]
    slot.supersedes = id(slot)

    violations = mem.temporal_history_violations()
    assert [v["reason"] for v in violations] == ["self_link"]
    assert violations[0]["slot_index"] == 0


def test_non_reciprocal_supersession_link_is_reported(models):
    """A one-sided link asserts a version relationship the counterpart does not
    confirm.  It must be reported as non_reciprocal_link and must not be used
    to group intervals."""
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Lisbon", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Porto", timestamp=20.0))
    mem.memory.slots[0].supersedes = id(mem.memory.slots[1])

    violations = mem.temporal_history_violations()
    assert [v["reason"] for v in violations] == ["non_reciprocal_link"]
    assert violations[0]["slot_indices"] == [0, 1]


def test_exclusive_group_membership_does_not_depend_on_value_similarity(models):
    """Every predicate in EXCLUSIVE_SYNONYM_GROUPS is recognised as the same
    attribute by construction (group membership), so an exclusive recurrence
    cannot be missed because of a distant value.  The hypothetical
    'unrecognised exclusive attribute' is therefore unreachable for configured
    groups; unconfigured predicates are deliberately non-exclusive (see the U1
    characterization test)."""
    mem = _memory(models, verifier=_RuleVerifier([]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    slot = mem.memory.slots[0]
    far = Candidate("Maya", "city", "Ouagadougou", timestamp=30.0)
    cand_dense = mem.dense_encoder.encode("Maya city Ouagadougou.")

    assert mem._is_same_attribute_or_event(slot, far, 0, cand_dense) is True

    unconfigured = Candidate("Maya", "affiliation", "Ouagadougou", timestamp=30.0)
    unconfigured_dense = mem.dense_encoder.encode("Maya affiliation Ouagadougou.")
    assert mem._is_same_attribute_or_event(
        slot, unconfigured, 0, unconfigured_dense
    ) is False


# ── Checkpoint link structures ─────────────────────────────────────────────


def test_benchmark_verdicts_distinguish_gaps_from_passes():
    """A confirmed known gap must never be reported as equivalent to a clean
    pass, and a resolved gap must prompt expectation review."""
    from benchmarks.phase29_integrity import (
        STATUS_ALL_EXPECTED,
        STATUS_FAILED,
        STATUS_GAP_CHANGED,
        STATUS_GAP_OPEN,
        VERDICT_EXPECTED,
        VERDICT_FAILURE,
        VERDICT_GAP_CONFIRMED,
        VERDICT_GAP_RESOLVED,
        classify_result,
        summarise,
    )

    def case(cid, passed, known_gap=False):
        return {"id": cid, "passed": passed, "known_gap": known_gap}

    assert classify_result(case("ok", True)) == VERDICT_EXPECTED
    assert classify_result(case("bad", False)) == VERDICT_FAILURE
    assert classify_result(case("gap", True, True)) == VERDICT_GAP_CONFIRMED
    assert classify_result(case("fixed", False, True)) == VERDICT_GAP_RESOLVED

    clean = summarise([case("a", True), case("b", True)])
    assert clean["status"] == STATUS_ALL_EXPECTED

    gap_open = summarise([case("a", True), case("gap", True, True)])
    assert gap_open["status"] == STATUS_GAP_OPEN
    assert gap_open["counts"][VERDICT_GAP_CONFIRMED] == 1

    gap_changed = summarise([case("a", True), case("gap", False, True)])
    assert gap_changed["status"] == STATUS_GAP_CHANGED

    failed = summarise([case("bad", False), case("gap", True, True)])
    assert failed["status"] == STATUS_FAILED


def _write_v3_archive(path, supersedes_idx, superseded_by_idx, n=2, dim=8):
    np.savez(
        path,
        keys=np.zeros((n, dim), dtype=np.float32),
        values=np.zeros((n, dim), dtype=np.float32),
        confidences=np.ones(n, dtype=np.float32),
        timestamps=np.array([10.0, 20.0][:n], dtype=np.float64),
        created_ats=np.array([10.0, 20.0][:n], dtype=np.float64),
        update_counts=np.ones(n, dtype=np.int32),
        schema_version=3,
        supersedes_idx=np.array(supersedes_idx, dtype=np.int64),
        superseded_by_idx=np.array(superseded_by_idx, dtype=np.int64),
        capacity=10,
        dim=dim,
    )


def test_checkpoint_roundtrip_preserves_reciprocal_links(models, tmp_path):
    """A valid version chain must survive save/load with reciprocal links and
    unchanged intervals."""
    mem = _memory(models, verifier=_RuleVerifier([
        ("Rome", "Venice", "CONTRADICTION"),
    ]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    path = str(tmp_path / "phase29_valid_links.npz")
    mem.save_checkpoint(path)

    dense, nli = models
    restored = BlackHoleMemory(
        capacity=10, dense_encoder=dense, nli_gate=nli,
        verifier=_RuleVerifier([]),
    )
    restored.load_checkpoint(path)
    slots = restored.memory.slots

    assert [(s.valid_from, s.valid_until) for s in slots] == [
        (10.0, 20.0), (20.0, None),
    ]
    assert slots[0].superseded_by == id(slots[1])
    assert slots[1].supersedes == id(slots[0])
    assert restored.temporal_history_violations() == []


@pytest.mark.parametrize("supersedes_idx,superseded_by_idx,fragment", [
    ([0, -1], [-1, -1], "links to itself"),
    ([-1, 0], [-1, -1], "does not point back"),
    ([-1, 2], [-1, -1], "out-of-bounds"),
])
def test_malformed_link_archives_are_rejected(
    tmp_path, supersedes_idx, superseded_by_idx, fragment
):
    """Self-links, non-reciprocal links, and out-of-range link indices are
    malformed checkpoint structures and must fail loudly instead of restoring
    a broken chain."""
    path = str(tmp_path / "phase29_malformed_links.npz")
    _write_v3_archive(path, supersedes_idx, superseded_by_idx)

    with pytest.raises(CheckpointValidationError) as exc:
        load_ckpt(path, allow_backup_recovery=False)
    assert fragment in str(exc.value)
