"""
Adversarial contradiction validation for Phase 28.

Targets both directions of failure:
  - false supersession: independent facts deactivated by an NLI contradiction
  - missed supersession: genuine updates left coexisting with a stale fact

The NLI-backed tests share one model instance to keep runtime bounded. The
matcher tests use a stub verifier so the attribute/marker logic is isolated
from classifier behavior.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate


class _AlwaysContradiction:
    """Forces CONTRADICTION so matcher behavior is exercised deterministically."""

    def verify_batch(self, pairs, **kwargs):
        return [
            {
                "decision": "CONTRADICTION",
                "confidence": 0.99,
                "p_entail": 0.01,
                "p_contra": 0.99,
                "p_neutral": 0.01,
                "fwd": {"p_contra": 0.99, "p_neutral": 0.01},
                "rev": {"p_contra": 0.99, "p_neutral": 0.01},
            }
            for _ in pairs
        ]


class _MapVerifier:
    """Deterministic verifier keyed by a substring of the candidate text."""

    def __init__(self, rules):
        self._rules = list(rules)

    def verify_batch(self, pairs, **kwargs):
        results = []
        for _slot_text, cand_text in pairs:
            decision = "NEUTRAL"
            for marker, mapped in self._rules:
                if marker in cand_text:
                    decision = mapped
                    break
            results.append({
                "decision": decision,
                "confidence": 0.99,
                "p_entail": 0.99 if decision == "SAME" else 0.01,
                "p_contra": 0.99 if decision == "CONTRADICTION" else 0.01,
                "p_neutral": 0.01 if decision != "NEUTRAL" else 0.98,
                "fwd": {
                    "p_contra": 0.99 if decision == "CONTRADICTION" else 0.01,
                    "p_neutral": 0.01 if decision == "CONTRADICTION" else 0.98,
                },
                "rev": {
                    "p_contra": 0.99 if decision == "CONTRADICTION" else 0.01,
                    "p_neutral": 0.01 if decision == "CONTRADICTION" else 0.98,
                },
            })
        return results


def _assert_links_consistent(mem):
    ids = {id(s) for s in mem.memory.slots}
    for slot in mem.memory.slots:
        for link in (slot.supersedes, slot.superseded_by):
            assert link is None or link in ids
    assert mem.temporal_history_violations() == []


@pytest.fixture(scope="module")
def models():
    from benchmarks.phase18_experiment import DenseSemanticEncoder
    from benchmarks.phase16_experiment import NliSemanticGate

    return DenseSemanticEncoder(), NliSemanticGate()


def _memory(models, capacity=10, verifier=None, **kwargs):
    dense, nli = models
    return BlackHoleMemory(
        capacity=capacity,
        dense_encoder=dense,
        nli_gate=nli,
        verifier=verifier,
        **kwargs,
    )


def _active(mem):
    return [s for s in mem.memory.slots if s.is_active]


# ── False supersession: matcher-level ──────────────────────────────────────


def test_substring_city_in_capacity_is_not_exclusive(models):
    """'capacity' contains 'city'; it must not match the residence group."""
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Rover", "storage capacity", "500 units"), timestamp=10.0)
    result = mem.absorb(Candidate("Rover", "cooling capacity", "2 megawatts"), timestamp=20.0)

    assert result["action"] == "insert"
    assert len(_active(mem)) == 2


def test_substring_state_in_statement_is_not_exclusive(models):
    """'statement' contains 'state'; it must not match the status group."""
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Iris", "public statement", "supports the merger"), timestamp=10.0)
    result = mem.absorb(Candidate("Iris", "operational state", "nominal"), timestamp=20.0)

    assert result["action"] == "insert"
    assert len(_active(mem)) == 2


def test_substring_now_in_known_is_not_a_transition_marker(models):
    """'known' contains 'now'; it must not be read as an explicit transition."""
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Omar", "route condition", "well known in heavy snow"), timestamp=10.0)
    result = mem.absorb(
        Candidate("Omar", "route condition", "well known in heavy snow today"),
        timestamp=20.0,
    )

    assert result["action"] == "insert"
    assert len(_active(mem)) == 2


# ── False supersession: NLI-backed ─────────────────────────────────────────


def test_unfamiliar_multivalued_predicates_coexist(models):
    """Unseen entities/predicates with the same verb must not supersede."""
    mem = _memory(models)
    mem.absorb(Candidate("Nadia", "calibrates", "the orbital gyroscope"), timestamp=10.0)
    result = mem.absorb(Candidate("Nadia", "calibrates", "the ground antenna"), timestamp=20.0)

    assert result["action"] == "insert"
    assert len(_active(mem)) == 2
    assert len(mem.recall("Nadia", "calibrates")) == 2


def test_similar_wording_with_different_objects_coexists(models):
    """Overlapping vocabulary must not be treated as one attribute."""
    mem = _memory(models)
    mem.absorb(Candidate("Rover", "reported a storage capacity", "of 500 units"), timestamp=10.0)
    result = mem.absorb(
        Candidate("Rover", "reported a cooling capacity", "of 2 megawatts"),
        timestamp=20.0,
    )

    assert result["action"] == "insert"
    assert len(_active(mem)) == 2


def test_negated_and_affirmative_independent_actions_coexist(models):
    mem = _memory(models)
    mem.absorb(Candidate("Priya", "does not manage", "the storage team"), timestamp=10.0)
    result = mem.absorb(Candidate("Priya", "manages", "the security team"), timestamp=20.0)

    assert result["action"] == "insert"
    assert len(_active(mem)) == 2


def test_negated_action_on_different_object_coexists(models):
    """Negation plus high lexical overlap must not hide distinct objects."""
    mem = _memory(models)
    mem.absorb("Priya did not approve the budget for the north district.", timestamp=10.0)
    result = mem.absorb("Priya approved the budget for the south district.", timestamp=20.0)

    assert result["action"] == "insert"
    assert len(_active(mem)) == 2


def test_negation_of_same_proposition_supersedes(models):
    mem = _memory(models)
    mem.absorb("The cluster is not online.", timestamp=10.0)
    result = mem.absorb("The cluster is online.", timestamp=20.0)

    assert result["action"] == "temporal_update"
    active = _active(mem)
    assert len(active) == 1
    assert "is online" in active[0].text
    assert "not online" not in active[0].text


def test_negation_after_consolidated_paraphrase_still_supersedes(models):
    """Consolidation rewrites slot text but not its debug predicate.

    The negation guard must still recognize the same object instead of
    silently inserting the negation as a second independent fact.
    """
    from benchmarks.phase26_verification import Phase26Verifier

    dense, nli = models
    mem = _memory(
        models,
        verifier=Phase26Verifier(nli_gate=nli, dense_encoder=dense),
        verification_mode="calibrated",
    )
    mem.absorb(Candidate("Bob", "holds a PhD in", "quantum physics", timestamp=100.0))
    consolidated = mem.absorb(
        Candidate("Bob", "earned a doctorate in", "quantum physics", timestamp=110.0)
    )
    assert consolidated["action"] == "consolidate"

    result = mem.absorb(
        Candidate("Bob", "does not hold a PhD in", "quantum physics", timestamp=200.0)
    )
    assert result["action"] == "temporal_update"
    active = _active(mem)
    assert len(active) == 1
    assert "does not" in active[0].text


# ── Missed supersession ────────────────────────────────────────────────────


def test_exclusive_city_and_employer_updates_supersede(models):
    mem = _memory(models)
    mem.absorb(Candidate("Alice", "city", "New York"), timestamp=10.0)
    assert mem.absorb(Candidate("Alice", "city", "Boston"), timestamp=20.0)["action"] == "temporal_update"
    mem.absorb(Candidate("Alice", "employer", "Acme"), timestamp=30.0)
    assert mem.absorb(Candidate("Alice", "employer", "Beta Corp"), timestamp=40.0)["action"] == "temporal_update"

    active_texts = {s.text for s in _active(mem)}
    assert len(active_texts) == 2
    assert any("Boston" in t for t in active_texts)
    assert any("Beta Corp" in t for t in active_texts)

    city = mem.recall("Alice", "city")
    employer = mem.recall("Alice", "employer")
    assert len(city) == 1 and "Boston" in city[0]["text"]
    assert len(employer) == 1 and "Beta Corp" in employer[0]["text"]


def test_subject_case_and_whitespace_variants_supersede(models):
    """Entity identity must not depend on casing or trailing whitespace."""
    mem = _memory(models)
    mem.absorb(Candidate("Nadia", "city", "Oslo"), timestamp=10.0)
    result = mem.absorb(Candidate("nadia ", "city", "Bergen"), timestamp=20.0)

    assert result["action"] == "temporal_update"
    active = _active(mem)
    assert len(active) == 1
    assert "Bergen" in active[0].text


def test_unlisted_exclusive_attribute_is_conservatively_retained(models):
    """Documented limitation: unseen single-valued attributes stay conservative.

    'commutes via' is outside EXCLUSIVE_SYNONYM_GROUPS, so the engine keeps
    both facts active rather than risk a false merge.
    """
    mem = _memory(models)
    mem.absorb(Candidate("Nadia", "commutes via", "bicycle"), timestamp=10.0)
    result = mem.absorb(Candidate("Nadia", "commutes via", "train"), timestamp=20.0)

    assert result["action"] != "temporal_update"
    assert len(_active(mem)) == 2


# ── Ordering and temporal history ──────────────────────────────────────────


def test_repeated_observations_consolidate_without_closing_window(models):
    mem = _memory(models)
    mem.absorb(Candidate("Tomas", "city", "Oslo"), timestamp=10.0)
    repeated = mem.absorb(Candidate("Tomas", "city", "Oslo"), timestamp=20.0)

    assert repeated["action"] == "consolidate"
    slot = mem.memory.slots[0]
    assert slot.update_count == 2
    assert slot.valid_from == 10.0
    assert slot.valid_until is None

    mem.absorb(Candidate("Tomas", "city", "Bergen"), timestamp=30.0)
    assert slot.valid_until == 30.0
    assert "Bergen" in _active(mem)[0].text


def test_recurring_historical_fact_becomes_active_again(models):
    """Returning to a previous employer must supersede the current one."""
    mem = _memory(models)
    mem.absorb(Candidate("Tomas", "employer", "Acme"), timestamp=10.0)
    mem.absorb(Candidate("Tomas", "employer", "Beta Corp"), timestamp=30.0)
    result = mem.absorb(Candidate("Tomas", "employer", "Acme"), timestamp=40.0)

    assert result["action"] == "temporal_update"
    active = _active(mem)
    assert len(active) == 1
    assert "Acme" in active[0].text

    history = mem.get_history("Tomas", "employer")
    assert [(h["valid_from"], h["valid_until"]) for h in history] == [
        (10.0, 30.0),
        (30.0, 40.0),
        (40.0, None),
    ]


def test_out_of_order_historical_insert_preserves_interval_chain(models):
    """A late historical fact must splice into the chain without overlapping."""
    mem = _memory(models)
    mem.absorb(Candidate("Carol", "city", "Rome"), timestamp=100.0)
    mem.absorb(Candidate("Carol", "city", "Venice"), timestamp=300.0)
    result = mem.absorb(Candidate("Carol", "city", "Florence"), timestamp=200.0)

    assert result["action"] == "historical_update"
    history = mem.get_history("Carol", "city")
    assert [(h["valid_from"], h["valid_until"]) for h in history] == [
        (100.0, 200.0),
        (200.0, 300.0),
        (300.0, None),
    ]
    assert history[0]["text"].startswith("Carol city Rome")
    assert history[1]["text"].startswith("Carol city Florence")
    assert history[2]["text"].startswith("Carol city Venice")

    assert "Rome" in mem.recall("Carol", "city", as_of=150.0)[0]["text"]
    assert "Florence" in mem.recall("Carol", "city", as_of=250.0)[0]["text"]
    assert "Venice" in mem.recall("Carol", "city", as_of=350.0)[0]["text"]


def test_two_fact_order_independence(models):
    """Arrival order must not change the final active state or intervals."""
    in_order = _memory(models)
    in_order.absorb(Candidate("Zara", "city", "Tokyo"), timestamp=100.0)
    in_order.absorb(Candidate("Zara", "city", "Osaka"), timestamp=200.0)

    reversed_order = _memory(models)
    reversed_order.absorb(Candidate("Zara", "city", "Osaka"), timestamp=200.0)
    reversed_order.absorb(Candidate("Zara", "city", "Tokyo"), timestamp=100.0)

    assert "Osaka" in _active(in_order)[0].text
    assert "Osaka" in _active(reversed_order)[0].text
    assert [(h["valid_from"], h["valid_until"]) for h in in_order.get_history("Zara", "city")] == [
        (100.0, 200.0),
        (200.0, None),
    ]
    assert [(h["valid_from"], h["valid_until"]) for h in reversed_order.get_history("Zara", "city")] == [
        (100.0, 200.0),
        (200.0, None),
    ]


# ── Retrieval and persistence ──────────────────────────────────────────────


def test_persistence_preserves_active_history_and_independent_facts(models, tmp_path):
    path = str(tmp_path / "adversarial_roundtrip.npz")
    mem = _memory(models)
    mem.absorb(Candidate("Fiona", "city", "Chicago"), timestamp=10.0)
    mem.absorb(Candidate("Fiona", "city", "Austin"), timestamp=25.0)
    mem.absorb(Candidate("Fiona", "practices", "classical cello"), timestamp=30.0)
    mem.absorb(Candidate("Fiona", "practices", "competitive chess"), timestamp=35.0)
    assert "Austin" in _active(mem)[0].text

    mem.save_checkpoint(path)
    restored = BlackHoleMemory.from_checkpoint(path, capacity=10)
    # Reuse the shared models on the restored instance so it does not reload them.
    restored.dense_encoder, restored.nli_gate = models
    restored.lex_encoder = mem.lex_encoder
    restored.verifier = mem.verifier

    assert len(restored.memory.slots) == 4
    assert len(_active(restored)) == 3
    assert len(restored.recall("Fiona", "city")) == 1
    assert "Austin" in restored.recall("Fiona", "city")[0]["text"]
    assert "Chicago" in restored.recall("Fiona", "city", as_of=20.0)[0]["text"]
    assert len(restored.recall("Fiona", "practices")) == 2

    # Superseded city must not resurface as an active query hit.
    hits = restored.query("Fiona city Chicago", top_k=4)
    assert all("Chicago" not in h["text"] for h in hits if h["is_active"])

    # Continue the chain after recovery and round-trip again.
    assert restored.absorb(Candidate("Fiona", "city", "Denver"), timestamp=40.0)["action"] == "temporal_update"
    restored.save_checkpoint(path)
    reloaded = BlackHoleMemory.from_checkpoint(path, capacity=10)
    history = reloaded.get_history("Fiona", "city")
    assert [(h["valid_from"], h["valid_until"]) for h in history] == [
        (10.0, 25.0),
        (25.0, 40.0),
        (40.0, None),
    ]
    assert "Denver" in reloaded.recall("Fiona", "city")[0]["text"]
    assert len(reloaded.recall("Fiona", "practices")) == 2


# ── Checkpoint supersession links ──────────────────────────────────────────


def _three_version_chain(models):
    mem = _memory(models, capacity=10, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Alice", "city", "Oslo"), timestamp=100.0)
    mem.absorb(Candidate("Alice", "city", "Bergen"), timestamp=200.0)
    mem.absorb(Candidate("Alice", "city", "Rome"), timestamp=300.0)
    return mem


def test_checkpoint_restores_every_supersession_link(models, tmp_path):
    path = str(tmp_path / "three_version_chain.npz")
    mem = _three_version_chain(models)
    mem.save_checkpoint(path)

    restored = _memory(models, capacity=10, verifier=_AlwaysContradiction())
    restored.load_checkpoint(path)
    slots = sorted(restored.memory.slots, key=lambda s: s.valid_from)
    assert [s.valid_from for s in slots] == [100.0, 200.0, 300.0]

    # Both directions of every adjacent relationship must point at the
    # restored counterpart object, not at a stale process-local id.
    assert slots[0].supersedes is None
    assert slots[0].superseded_by == id(slots[1])
    assert slots[1].supersedes == id(slots[0])
    assert slots[1].superseded_by == id(slots[2])
    assert slots[2].supersedes == id(slots[1])
    assert slots[2].superseded_by is None
    _assert_links_consistent(restored)

    assert [(h["valid_from"], h["valid_until"]) for h in restored.get_history("Alice", "city")] == [
        (100.0, 200.0),
        (200.0, 300.0),
        (300.0, None),
    ]

    # Continuing the chain after restoration must extend the same links.
    assert restored.absorb(Candidate("Alice", "city", "Paris"), timestamp=400.0)["action"] == "temporal_update"
    assert slots[2].valid_until == 400.0
    assert slots[2].superseded_by == id(restored.memory.slots[-1])
    assert restored.memory.slots[-1].supersedes == id(slots[2])
    _assert_links_consistent(restored)


def test_deleting_middle_restored_slot_repairs_chain(models, tmp_path):
    path = str(tmp_path / "delete_middle_chain.npz")
    mem = _three_version_chain(models)
    mem.save_checkpoint(path)

    restored = _memory(models, capacity=10, verifier=_AlwaysContradiction())
    restored.load_checkpoint(path)
    slots = sorted(restored.memory.slots, key=lambda s: s.valid_from)
    middle = slots[1]

    assert restored.delete_slot(id(middle)) is True
    remaining = sorted(restored.memory.slots, key=lambda s: s.valid_from)
    assert len(remaining) == 2
    assert remaining[0].superseded_by == id(remaining[1])
    assert remaining[1].supersedes == id(remaining[0])
    assert remaining[0].valid_until == 200.0
    assert remaining[1].valid_from == 300.0
    _assert_links_consistent(restored)

    assert "Rome" in restored.recall("Alice", "city")[0]["text"]
    assert "Oslo" in restored.recall("Alice", "city", as_of=150.0)[0]["text"]


def test_checkpoint_links_are_restorable_in_a_separate_process(models, tmp_path):
    path = str(tmp_path / "cross_process_chain.npz")
    mem = _three_version_chain(models)
    mem.save_checkpoint(path)

    script = """
import sys
from phase2.checkpoint import load_checkpoint

memory = load_checkpoint(sys.argv[1])
slots = sorted(memory.slots, key=lambda s: s.valid_from)
assert len(slots) == 3
assert slots[0].superseded_by == id(slots[1])
assert slots[1].supersedes == id(slots[0])
assert slots[1].superseded_by == id(slots[2])
assert slots[2].supersedes == id(slots[1])
assert slots[2].superseded_by is None
ids = {id(s) for s in slots}
for slot in slots:
    for link in (slot.supersedes, slot.superseded_by):
        assert link is None or link in ids
print("links-ok")
"""
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", script, path],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "links-ok" in proc.stdout


# ── Equal-timestamp policy ─────────────────────────────────────────────────


def test_equal_timestamp_contradiction_replaces_state_in_place(models):
    """Last writer at an identical valid_from wins without shifting time."""
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Tomas", "city", "Oslo"), timestamp=10.0)
    result = mem.absorb(Candidate("Tomas", "city", "Bergen"), timestamp=10.0)

    assert result["action"] == "temporal_update"
    assert result["replaced_in_place"] is True
    assert len(mem.memory.slots) == 1
    slot = mem.memory.slots[0]
    assert slot.valid_from == 10.0
    assert slot.valid_until is None
    assert slot.update_count == 2
    assert "Bergen" in slot.text
    _assert_links_consistent(mem)

    assert len(mem.get_history("Tomas", "city")) == 1
    assert "Bergen" in mem.recall("Tomas", "city", as_of=10.0)[0]["text"]


def test_multiple_observations_at_one_timestamp_leave_one_interval(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Tomas", "city", "Oslo"), timestamp=10.0)
    mem.absorb(Candidate("Tomas", "city", "Bergen"), timestamp=10.0)
    result = mem.absorb(Candidate("Tomas", "city", "Rome"), timestamp=10.0)

    assert result["action"] == "temporal_update"
    assert result["replaced_in_place"] is True
    assert len(mem.memory.slots) == 1
    assert mem.memory.slots[0].update_count == 3
    assert "Rome" in mem.memory.slots[0].text
    history = mem.get_history("Tomas", "city")
    assert [(h["valid_from"], h["valid_until"]) for h in history] == [(10.0, None)]
    _assert_links_consistent(mem)


def test_historical_insert_at_existing_valid_until_replaces_that_interval(models):
    """An insertion exactly at a closed window's end replaces the state that
    starts at that instant instead of overlapping it."""
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Zara", "city", "Rome"), timestamp=100.0)
    mem.absorb(Candidate("Zara", "city", "Venice"), timestamp=200.0)
    mem.absorb(Candidate("Zara", "city", "Milan"), timestamp=300.0)

    result = mem.absorb(Candidate("Zara", "city", "Florence"), timestamp=200.0)
    assert result["action"] == "historical_update"
    assert result["replaced_in_place"] is True
    assert len(mem.memory.slots) == 3
    _assert_links_consistent(mem)

    history = mem.get_history("Zara", "city")
    assert [(h["valid_from"], h["valid_until"]) for h in history] == [
        (100.0, 200.0),
        (200.0, 300.0),
        (300.0, None),
    ]
    assert "Florence" in history[1]["text"]
    assert "Milan" in history[2]["text"]
    assert "Venice" not in [h["text"] for h in history]
    assert "Florence" in mem.recall("Zara", "city", as_of=250.0)[0]["text"]


def test_out_of_order_ties_keep_disjoint_intervals(models):
    mem = _memory(models, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Zara", "city", "Venice"), timestamp=200.0)
    mem.absorb(Candidate("Zara", "city", "Rome"), timestamp=100.0)
    # Milan ties the active interval's valid_from: last writer replaces it.
    assert mem.absorb(Candidate("Zara", "city", "Milan"), timestamp=200.0)["action"] == "temporal_update"
    # Florence ties the historical interval's valid_from: replace in place.
    assert mem.absorb(Candidate("Zara", "city", "Florence"), timestamp=100.0)["action"] == "historical_update"

    assert [(h["valid_from"], h["valid_until"]) for h in mem.get_history("Zara", "city")] == [
        (100.0, 200.0),
        (200.0, None),
    ]
    assert "Florence" in mem.recall("Zara", "city", as_of=150.0)[0]["text"]
    assert "Milan" in mem.recall("Zara", "city", as_of=250.0)[0]["text"]
    _assert_links_consistent(mem)


# ── Eviction must not break the history chain ──────────────────────────────


def test_capacity_one_forward_update_preserves_active_state(models):
    mem = _memory(models, capacity=1, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Ines", "city", "Oslo"), timestamp=10.0)
    result = mem.absorb(Candidate("Ines", "city", "Bergen"), timestamp=20.0)

    assert result["action"] == "temporal_update"
    assert result["history_truncated"] is True
    assert len(mem.memory.slots) == 1
    active = _active(mem)
    assert len(active) == 1
    assert "Bergen" in active[0].text
    assert active[0].supersedes is None
    assert active[0].superseded_by is None
    _assert_links_consistent(mem)


def test_capacity_two_forward_update_evicts_predecessor_not_active(models):
    mem = _memory(models, capacity=2, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Ines", "city", "Oslo"), timestamp=10.0)
    mem.absorb(Candidate("Ines", "city", "Bergen"), timestamp=20.0)
    oslo = next(s for s in mem.memory.slots if s.valid_from == 10.0)
    bergen = next(s for s in mem.memory.slots if s.valid_from == 20.0)

    result = mem.absorb(Candidate("Ines", "city", "Rome"), timestamp=30.0)
    assert result["action"] == "temporal_update"
    assert result["history_truncated"] is True
    assert result["evicted_slot_id"] == id(oslo)
    assert bergen.valid_until == 30.0
    assert bergen.superseded_by == id(mem.memory.slots[-1])
    assert mem.memory.slots[-1].supersedes == id(bergen)
    assert len(_active(mem)) == 1
    assert "Rome" in _active(mem)[0].text
    _assert_links_consistent(mem)


def test_capacity_two_historical_insert_keeps_active_successor(models):
    mem = _memory(models, capacity=2, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Ines", "city", "Rome"), timestamp=100.0)
    mem.absorb(Candidate("Ines", "city", "Venice"), timestamp=300.0)
    old = mem.memory.slots[0]
    active = mem.memory.slots[1]

    result = mem.absorb(Candidate("Ines", "city", "Florence"), timestamp=200.0)
    assert result["action"] == "historical_update"
    assert result["history_truncated"] is True
    assert id(old) not in {id(s) for s in mem.memory.slots}
    assert any(s is active for s in mem.memory.slots)
    assert active.is_active
    assert active.supersedes == id(mem.memory.slots[-1])
    assert mem.memory.slots[-1].superseded_by == id(active)
    assert len(_active(mem)) == 1
    assert "Venice" in _active(mem)[0].text
    _assert_links_consistent(mem)


def test_capacity_one_historical_insert_is_rejected_without_mutation(models):
    mem = _memory(models, capacity=1, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Ines", "city", "Rome"), timestamp=100.0)
    active = mem.memory.slots[0]

    result = mem.absorb(Candidate("Ines", "city", "Florence"), timestamp=50.0)
    assert result["action"] == "historical_rejected"
    assert result["reason"] == "capacity_would_remove_active_state"
    assert len(mem.memory.slots) == 1
    assert mem.memory.slots[0] is active
    assert active.is_active
    assert "Rome" in active.text
    assert active.supersedes is None and active.superseded_by is None
    _assert_links_consistent(mem)


def test_capacity_two_historical_insert_evicts_unrelated_slot(models):
    mem = _memory(models, capacity=2, verifier=_AlwaysContradiction())
    mem.absorb(Candidate("Zara", "city", "Rome"), timestamp=300.0)
    mem.absorb(Candidate("Zara", "hobby", "chess"), timestamp=100.0)
    hobby_id = id(mem.memory.slots[1])

    result = mem.absorb(Candidate("Zara", "city", "Florence"), timestamp=200.0)
    assert result["action"] == "historical_update"
    assert hobby_id not in {id(s) for s in mem.memory.slots}
    assert len(mem.memory.slots) == 2
    assert len(_active(mem)) == 1
    assert "Rome" in _active(mem)[0].text
    _assert_links_consistent(mem)


# ── Consolidation metadata and negation matching ───────────────────────────


def test_paraphrase_consolidation_then_recurrence_does_not_reactivate_history(models):
    verifier = _MapVerifier([
        ("resides in Oslo", "SAME"),
        ("Bergen", "CONTRADICTION"),
        ("lives in Oslo", "CONTRADICTION"),
    ])
    mem = _memory(models, verifier=verifier)
    mem.absorb(Candidate("Fiona", "lives in", "Oslo"), timestamp=100.0)
    consolidated = mem.absorb(Candidate("Fiona", "resides in", "Oslo"), timestamp=110.0)
    assert consolidated["action"] == "consolidate"
    first_id = consolidated["slot_id"]

    mem.absorb(Candidate("Fiona", "lives in", "Bergen"), timestamp=200.0)
    result = mem.absorb(Candidate("Fiona", "lives in", "Oslo"), timestamp=300.0)

    assert result["action"] == "temporal_update"
    first = next(s for s in mem.memory.slots if id(s) == first_id)
    assert not first.is_active
    assert first.valid_until == 200.0
    assert result["slot_id"] != first_id
    assert len([s for s in mem.memory.slots if id(s) == first_id]) == 1

    history = mem.get_history("Fiona", "lives in")
    assert [(h["valid_from"], h["valid_until"]) for h in history] == [
        (100.0, 200.0),
        (200.0, 300.0),
        (300.0, None),
    ]
    assert "resides in Oslo" in history[0]["text"]
    _assert_links_consistent(mem)


def test_consolidated_form_preserves_history_across_checkpoint(models, tmp_path):
    verifier = _MapVerifier([
        ("resides in Oslo", "SAME"),
        ("Bergen", "CONTRADICTION"),
    ])
    mem = _memory(models, verifier=verifier)
    mem.absorb(Candidate("Fiona", "lives in", "Oslo"), timestamp=100.0)
    mem.absorb(Candidate("Fiona", "resides in", "Oslo"), timestamp=110.0)
    consolidated_text = mem.slot_texts[id(mem.memory.slots[0])]
    assert "resides in Oslo" in consolidated_text

    path = str(tmp_path / "consolidated_text.npz")
    mem.save_checkpoint(path)
    restored = _memory(models, verifier=verifier)
    restored.load_checkpoint(path)

    restored_slot = restored.memory.slots[0]
    assert restored_slot.text == consolidated_text
    assert restored_slot.latest_object == "Oslo"
    assert restored_slot.update_count == 2
    assert restored_slot.valid_from == 100.0
    assert restored_slot.is_active

    assert restored.absorb(Candidate("Fiona", "lives in", "Bergen"), timestamp=200.0)["action"] == "temporal_update"
    history = restored.get_history("Fiona", "lives in")
    assert [(h["valid_from"], h["valid_until"]) for h in history] == [
        (100.0, 200.0),
        (200.0, None),
    ]
    assert "resides in Oslo" in history[0]["text"]
    _assert_links_consistent(restored)


def test_negated_different_object_does_not_supersede_consolidated_fact(models):
    """The object must be compared against the consolidated object, not the
    whole slot text.  'doctorate' appears in the consolidated predicate but is
    not the stored object ('quantum physics'), so it must not match."""
    verifier = _MapVerifier([
        ("earned a doctorate", "SAME"),
        ("does not hold", "CONTRADICTION"),
    ])
    mem = _memory(models, verifier=verifier)
    mem.absorb(Candidate("Bob", "holds a PhD in", "quantum physics"), timestamp=100.0)
    consolidated = mem.absorb(
        Candidate("Bob", "earned a doctorate in", "quantum physics"), timestamp=110.0
    )
    assert consolidated["action"] == "consolidate"

    result = mem.absorb(
        Candidate("Bob", "does not hold a PhD in", "doctorate"), timestamp=200.0
    )
    assert result["action"] == "insert"
    assert len(mem.memory.slots) == 2
    assert len(_active(mem)) == 2
    consolidated_slot = next(s for s in mem.memory.slots if id(s) == consolidated["slot_id"])
    assert consolidated_slot.is_active
    _assert_links_consistent(mem)


def test_genuine_contradiction_after_consolidation_supersedes_with_links(models):
    verifier = _MapVerifier([
        ("earned a doctorate", "SAME"),
        ("does not hold", "CONTRADICTION"),
    ])
    mem = _memory(models, verifier=verifier)
    mem.absorb(Candidate("Bob", "holds a PhD in", "quantum physics"), timestamp=100.0)
    consolidated = mem.absorb(
        Candidate("Bob", "earned a doctorate in", "quantum physics"), timestamp=110.0
    )
    assert consolidated["action"] == "consolidate"
    first_id = consolidated["slot_id"]

    result = mem.absorb(
        Candidate("Bob", "does not hold a PhD in", "quantum physics"), timestamp=200.0
    )
    assert result["action"] == "temporal_update"
    first = next(s for s in mem.memory.slots if id(s) == first_id)
    assert first.valid_until == 200.0
    assert first.superseded_by == result["slot_id"]
    assert mem.memory.slots[-1].supersedes == first_id
    assert len(_active(mem)) == 1
    assert "does not hold" in _active(mem)[0].text
    _assert_links_consistent(mem)

    # The replacement is stored under its negated predicate, so retrieve the
    # full subject history rather than filtering on the original predicate.
    history = mem.get_history("Bob")
    assert [(h["valid_from"], h["valid_until"]) for h in history] == [
        (100.0, 200.0),
        (200.0, None),
    ]
