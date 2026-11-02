"""B1 — mutation boundary and invalidation-safety contract.

The engine owns a monotonic ``_state_version`` counter and a single
notification entry point, ``_invalidate_retrieval_cache()``.  A future
retrieval snapshot will decide "is my cached derived state current?" by
comparing versions.  These tests pin the contract that makes that check
sound:

* every engine mutation pathway bumps the counter,
* the bump happens before the first mutation, so a failure or early return
  can only over-invalidate, never leave stale state marked current,
* read-only operations do not bump,
* out-of-band test fixtures call the same notification entry point.

Scope note: the counter cannot observe a direct field write on a
``MemorySlot`` made outside the engine.  Such mutation is unsupported; the
fixtures that deliberately construct out-of-band state are checked by
``test_fixture_mutations_use_the_supported_notification``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from phase2.black_hole_memory import BlackHoleMemory
from phase2.checkpoint import load_into_black_hole_memory
from phase2.memory_state import MemorySlot, MemoryState
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


def _memory(models, capacity=10, verifier=None):
    dense, nli = models
    return BlackHoleMemory(
        capacity=capacity,
        dense_encoder=dense,
        nli_gate=nli,
        verifier=verifier or _RuleVerifier([]),
    )


def _bumps(mem, label, action):
    before = mem._state_version
    result = action()
    assert mem._state_version > before, f"{label} did not advance _state_version"
    return result


# ── 1. Every relevant pathway advances the version ─────────────────────────


def test_every_engine_mutation_pathway_advances_version(models, tmp_path):
    mem = _memory(models)
    result = _bumps(
        mem, "insert",
        lambda: mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0)),
    )
    assert result["action"] == "insert"

    mem = _memory(models, verifier=_RuleVerifier([("Rome", "Rome", "SAME")]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    result = _bumps(
        mem, "consolidate",
        lambda: mem.absorb(Candidate("Maya", "city", "Rome", timestamp=20.0)),
    )
    assert result["action"] == "consolidate"

    mem = _memory(models, verifier=_RuleVerifier([("Rome", "Venice", "CONTRADICTION")]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    result = _bumps(
        mem, "temporal_update (new slot)",
        lambda: mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0)),
    )
    assert result["action"] == "temporal_update"
    assert result["valid_until"] is None

    mem = _memory(models, verifier=_RuleVerifier([("Rome", "Venice", "CONTRADICTION")]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    result = _bumps(
        mem, "temporal_update (in place)",
        lambda: mem.absorb(Candidate("Maya", "city", "Venice", timestamp=10.0)),
    )
    assert result["action"] == "temporal_update"
    assert result["replaced_in_place"] is True

    mem = _memory(models, verifier=_RuleVerifier([
        ("Rome", "Venice", "CONTRADICTION"),
        ("Venice", "Rome", "CONTRADICTION"),
    ]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    result = _bumps(
        mem, "historical_update",
        lambda: mem.absorb(Candidate("Maya", "city", "Rome", timestamp=15.0)),
    )
    assert result["action"] == "historical_update"
    assert mem.temporal_history_violations() == []

    mem = _memory(models)
    mem.absorb(Candidate("Alice", "lives in", "Addis Ababa", timestamp=10.0))
    result = _bumps(
        mem, "historical_observation",
        lambda: mem.absorb(
            Candidate("Alice", "used to live in", "Addis Ababa", timestamp=20.0)
        ),
    )
    assert result["action"] == "historical_observation"

    mem = _memory(models, capacity=1)
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    result = _bumps(
        mem, "evict_insert",
        lambda: mem.absorb(Candidate("Bob", "hobby", "chess", timestamp=20.0)),
    )
    assert result["action"] == "evict_insert"
    assert mem.total_evictions == 1

    mem = _memory(models)
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    assert _bumps(
        mem, "delete_slot",
        lambda: mem.delete_slot(id(mem.memory.slots[0])),
    ) is True

    mem = _memory(models)
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Bob", "hobby", "chess", timestamp=20.0))
    assert _bumps(mem, "purge_subject", lambda: mem.purge_subject("Maya")) == 1

    source = _memory(models)
    source.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    path = str(tmp_path / "b1_all_paths.npz")
    source.save_checkpoint(path, create_backup=False)
    restored = _memory(models)
    _bumps(
        restored, "checkpoint restore",
        lambda: load_into_black_hole_memory(
            restored, path, allow_backup_recovery=False
        ),
    )
    assert [s.text for s in restored.memory.slots] == ["Maya city Rome."]


# ── 2. Invalidation happens before mutation begins ─────────────────────────


def test_invalidation_precedes_insert(models, monkeypatch):
    mem = _memory(models)
    observed = []
    original_add = MemoryState.add

    def recording_add(self, *args, **kwargs):
        observed.append(mem._state_version)
        return original_add(self, *args, **kwargs)

    monkeypatch.setattr(MemoryState, "add", recording_add)
    before = mem._state_version
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))

    assert observed, "insert pathway never reached MemoryState.add"
    assert observed[0] > before, "version was bumped after the insert began"


def test_invalidation_precedes_eviction(models):
    mem = _memory(models, capacity=1)
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))

    popped_at_versions = []

    class RecordingSlots(list):
        def pop(self, *args, **kwargs):
            popped_at_versions.append(mem._state_version)
            return super().pop(*args, **kwargs)

    mem.memory.slots = RecordingSlots(mem.memory.slots)
    before = mem._state_version
    result = mem.absorb(Candidate("Bob", "hobby", "chess", timestamp=20.0))

    assert result["action"] == "evict_insert"
    assert popped_at_versions, "eviction never removed a slot"
    assert popped_at_versions[0] > before


def test_invalidation_precedes_in_place_field_writes(models, monkeypatch):
    mem = _memory(models, verifier=_RuleVerifier([("Rome", "Venice", "CONTRADICTION")]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))

    writes = []
    original_setattr = MemorySlot.__setattr__

    def probe(self, name, value):
        if any(self is slot for slot in mem.memory.slots):
            writes.append((name, mem._state_version))
        original_setattr(self, name, value)

    monkeypatch.setattr(MemorySlot, "__setattr__", probe)
    before = mem._state_version
    result = mem.absorb(Candidate("Maya", "city", "Venice", timestamp=10.0))

    assert result["replaced_in_place"] is True
    assert writes, "in-place replacement performed no field write"
    assert all(version > before for _, version in writes)


def test_invalidation_precedes_removal(models):
    mem = _memory(models)
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))

    popped_at_versions = []

    class RecordingSlots(list):
        def pop(self, *args, **kwargs):
            popped_at_versions.append(mem._state_version)
            return super().pop(*args, **kwargs)

    mem.memory.slots = RecordingSlots(mem.memory.slots)
    before = mem._state_version
    assert mem.delete_slot(id(mem.memory.slots[0])) is True

    assert popped_at_versions
    assert popped_at_versions[0] > before


def test_invalidation_precedes_checkpoint_restore(models, tmp_path, monkeypatch):
    source = _memory(models)
    source.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    path = str(tmp_path / "b1_restore_order.npz")
    source.save_checkpoint(path, create_backup=False)

    target = _memory(models)
    target.absorb(Candidate("Bob", "hobby", "chess", timestamp=5.0))
    before = target._state_version

    observed = []
    original_setattr = BlackHoleMemory.__setattr__

    def probe(self, name, value):
        if name == "memory" and "_state_version" in self.__dict__:
            observed.append(self._state_version)
        original_setattr(self, name, value)

    monkeypatch.setattr(BlackHoleMemory, "__setattr__", probe)
    load_into_black_hole_memory(target, path, allow_backup_recovery=False)

    assert observed, "restore never replaced the memory state"
    assert observed[0] > before


# ── 3. Failures and early returns cannot leave the version unchanged ───────


def test_absorb_failure_after_bump(models, monkeypatch):
    mem = _memory(models)

    def failing_add(self, *args, **kwargs):
        raise RuntimeError("injected insert failure")

    monkeypatch.setattr(MemoryState, "add", failing_add)
    before = mem._state_version
    with pytest.raises(RuntimeError, match="injected insert failure"):
        mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))

    assert mem._state_version > before
    assert mem.memory.slots == []


def test_early_rejection_after_bump(models):
    mem = _memory(models, capacity=1)
    mem.absorb(Candidate("Ines", "city", "Rome", timestamp=100.0))
    active = mem.memory.slots[0]
    before = mem._state_version

    result = mem.absorb(
        Candidate("Ines", "used to live in", "Rome", timestamp=50.0)
    )

    assert result["action"] == "historical_rejected"
    assert mem._state_version > before
    assert mem.memory.slots == [active]
    assert active.is_active


def test_checkpoint_restore_failure_after_bump(models, tmp_path, monkeypatch):
    source = _memory(models)
    source.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    path = str(tmp_path / "b1_restore_failure.npz")
    source.save_checkpoint(path, create_backup=False)

    target = _memory(models)
    target.absorb(Candidate("Bob", "hobby", "chess", timestamp=5.0))
    before = target._state_version

    import benchmarks.phase22_experiment as phase22

    class BrokenBM25:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("injected rebuild failure")

    monkeypatch.setattr(phase22, "LocalBM25Index", BrokenBM25)
    with pytest.raises(RuntimeError, match="injected rebuild failure"):
        load_into_black_hole_memory(target, path, allow_backup_recovery=False)

    assert target._state_version > before
    # The restore had already replaced the slot store when it failed; derived
    # state must read as stale rather than as current.
    assert [s.text for s in target.memory.slots] == ["Maya city Rome."]


# ── 4. Insertions and removals ─────────────────────────────────────────────


def test_insert_and_remove_keep_state_and_derived_stores_aligned(models):
    mem = _memory(models)
    _bumps(
        mem, "insert",
        lambda: mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0)),
    )
    slot_id = id(mem.memory.slots[0])
    assert [r["text"] for r in mem.query("Maya city Rome", top_k=3)] == [
        "Maya city Rome."
    ]

    before = mem._state_version
    assert mem.delete_slot(slot_id) is True
    assert mem._state_version > before
    assert mem.memory.slots == []
    assert slot_id not in mem.slot_texts
    assert mem.slot_dense_embs == []
    assert mem.query("Maya city Rome", top_k=3) == []


def test_purge_subject_removes_only_the_target_entity(models):
    mem = _memory(models)
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    mem.absorb(Candidate("Bob", "hobby", "chess", timestamp=20.0))
    before = mem._state_version

    removed = mem.purge_subject("Maya")

    assert removed == 1
    assert mem._state_version > before
    assert [s.debug_subject for s in mem.memory.slots] == ["Bob"]
    assert [r["subject"] for r in mem.query("chess", top_k=3)] == ["Bob"]


# ── 5. In-place replacement and temporal metadata ──────────────────────────


def test_in_place_replacement_preserves_interval_and_updates_derived_state(models):
    mem = _memory(models, verifier=_RuleVerifier([("Rome", "Venice", "CONTRADICTION")]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    slot = mem.memory.slots[0]
    slot_id = id(slot)
    before = mem._state_version

    result = mem.absorb(Candidate("Maya", "city", "Venice", timestamp=10.0))

    assert result["replaced_in_place"] is True
    assert mem._state_version > before
    assert len(mem.memory.slots) == 1
    assert id(mem.memory.slots[0]) == slot_id
    assert (slot.valid_from, slot.valid_until) == (10.0, None)
    assert slot.latest_object == "Venice"
    assert slot.update_count == 2
    assert mem.slot_texts[slot_id] == "Maya city Venice."
    assert mem.query("Maya city Venice", top_k=1)[0]["text"] == "Maya city Venice."


def test_temporal_supersession_records_interval_and_links(models):
    mem = _memory(models, verifier=_RuleVerifier([("Rome", "Venice", "CONTRADICTION")]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    old = mem.memory.slots[0]
    before = mem._state_version

    result = mem.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))

    assert result["action"] == "temporal_update"
    assert mem._state_version > before
    new = mem.memory.slots[result["slot_index"]]
    assert old.valid_until == 20.0
    assert old.superseded_by == id(new)
    assert new.supersedes == id(old)
    assert new.valid_from == 20.0 and new.valid_until is None
    assert mem.recall("Maya", active_only=False) and mem.temporal_history_violations() == []


# ── 6. Consolidation and eviction ──────────────────────────────────────────


def test_consolidation_updates_slot_and_advances_version(models):
    mem = _memory(models, verifier=_RuleVerifier([("Rome", "Rome", "SAME")]))
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    slot = mem.memory.slots[0]
    slot_id = id(slot)
    before = mem._state_version

    result = mem.absorb(Candidate("Maya", "city", "Rome", timestamp=20.0))

    assert result["action"] == "consolidate"
    assert mem._state_version > before
    assert id(mem.memory.slots[0]) == slot_id
    assert slot.update_count == 2
    assert slot.timestamp == 20.0
    assert mem.slot_texts[slot_id] == "Maya city Rome."


def test_eviction_advances_version_and_keeps_auxiliary_stores_aligned(models):
    mem = _memory(models, capacity=1)
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    before = mem._state_version

    result = mem.absorb(Candidate("Bob", "hobby", "chess", timestamp=20.0))

    assert result["action"] == "evict_insert"
    assert mem._state_version > before
    assert mem.total_evictions == 1
    assert [s.text for s in mem.memory.slots] == ["Bob hobby chess."]
    assert len(mem.slot_dense_embs) == 1
    assert len(mem.slot_texts) == 1
    assert len(mem.slot_contra_counts) <= 1


# ── 7. Checkpoint restore and subject purge ────────────────────────────────


def test_checkpoint_restore_advances_version_and_rebuilds_derived_state(models, tmp_path):
    source = _memory(models, verifier=_RuleVerifier([("Rome", "Venice", "CONTRADICTION")]))
    source.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    source.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    path = str(tmp_path / "b1_restore.npz")
    source.save_checkpoint(path, create_backup=False)

    restored = _memory(models)
    before = restored._state_version
    restored.load_checkpoint(path, allow_backup_recovery=False)

    assert restored._state_version > before
    assert len(restored.memory.slots) == 2
    assert len(restored.slot_dense_embs) == 2
    assert [r["text"] for r in restored.query("Maya city Venice", top_k=2)][0] == (
        "Maya city Venice."
    )
    assert restored.recall("Maya", active_only=False)


def test_purge_after_restore_advances_version_and_clears_history(models, tmp_path):
    source = _memory(models, verifier=_RuleVerifier([("Rome", "Venice", "CONTRADICTION")]))
    source.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    source.absorb(Candidate("Maya", "city", "Venice", timestamp=20.0))
    path = str(tmp_path / "b1_purge.npz")
    source.save_checkpoint(path, create_backup=False)

    restored = _memory(models)
    restored.load_checkpoint(path, allow_backup_recovery=False)
    before = restored._state_version

    removed = restored.purge_subject("Maya")

    assert removed == 2
    assert restored._state_version > before
    assert restored.memory.slots == []
    assert restored.slot_texts == {}
    assert restored.query("Maya city Venice", top_k=2) == []


# ── 8. Out-of-band fixtures use the supported notification ─────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AUDITED_FIXTURE_FILES = (
    "test_phase27.py",
    "test_phase29_temporal_integrity.py",
    "test_reliability_repairs.py",
    "test_u1_audit.py",
    "test_u1_closure.py",
)
_SLOT_FIELDS = {
    "key", "value", "confidence", "timestamp", "created_at", "update_count",
    "debug_subject", "debug_predicate", "valid_from", "valid_until", "text",
    "dense_emb", "contra_count", "superseded_by", "supersedes",
    "latest_predicate", "latest_object",
}
_LIST_MUTATORS = {"pop", "append", "remove", "insert", "clear", "extend"}


def _is_slot_field_write(node):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    else:
        return False

    def target_writes_state(target):
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(target_writes_state(item) for item in target.elts)
        if isinstance(target, ast.Attribute) and target.attr in _SLOT_FIELDS:
            return True
        if isinstance(target, ast.Subscript):
            rendered = ast.unparse(target)
            return ".slot_texts[" in rendered or ".slot_contra_anchors[" in rendered
        return False

    return any(target_writes_state(target) for target in targets)


def _is_slot_list_mutation(node):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    return (
        node.func.attr in _LIST_MUTATORS
        and ast.unparse(node.func.value).endswith(".slots")
    )


def _is_engine_state_add(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add"
        and ast.unparse(node.func.value).endswith(".memory")
    )


class _ScopeScan(ast.NodeVisitor):
    """Collect mutations and notification calls, excluding nested scopes."""

    def __init__(self):
        self.mutations = []
        self.notifications = 0

    def _skip_nested(self, node):
        return None

    visit_FunctionDef = _skip_nested
    visit_AsyncFunctionDef = _skip_nested
    visit_ClassDef = _skip_nested
    visit_Lambda = _skip_nested

    def visit_Call(self, node):
        if _is_slot_list_mutation(node) or _is_engine_state_add(node):
            self.mutations.append(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "_invalidate_retrieval_cache"
        ):
            self.notifications += 1
        self.generic_visit(node)

    def _visit_assignment(self, node):
        if _is_slot_field_write(node):
            self.mutations.append(node)
        self.generic_visit(node)

    visit_Assign = _visit_assignment
    visit_AnnAssign = _visit_assignment
    visit_AugAssign = _visit_assignment


def test_fixture_mutations_use_the_supported_notification():
    problems = []
    for name in _AUDITED_FIXTURE_FILES:
        path = _REPO_ROOT / "tests" / name
        tree = ast.parse(path.read_text(encoding="utf-8"))

        scopes = [("module", tree.body)]
        scopes.extend(
            (f"{node.name}:{node.lineno}", node.body)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        for label, body in scopes:
            scan = _ScopeScan()
            for statement in body:
                scan.visit(statement)
            if scan.mutations and scan.notifications == 0:
                lines = sorted({m.lineno for m in scan.mutations})
                problems.append(f"{name}:{label} mutates at lines {lines}")

    assert problems == [], (
        "out-of-band fixture mutations must call _invalidate_retrieval_cache(): "
        + "; ".join(problems)
    )


def test_state_version_is_only_advanced_by_the_invalidation_entry_point():
    engine_path = _REPO_ROOT / "phase2" / "black_hole_memory.py"
    tree = ast.parse(engine_path.read_text(encoding="utf-8"))
    engine = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BlackHoleMemory"
    )
    mutating_methods = set()
    for node in engine.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.AugAssign)
                and isinstance(child.target, ast.Attribute)
                and child.target.attr == "_state_version"
            ):
                mutating_methods.add(node.name)
    assert mutating_methods == {"_invalidate_retrieval_cache"}

    checkpoint_path = _REPO_ROOT / "phase2" / "checkpoint.py"
    assert "_state_version" not in checkpoint_path.read_text(encoding="utf-8")


# ── 9. Unrelated operations do not invalidate ──────────────────────────────


def test_read_only_paths_and_no_op_deletes_do_not_invalidate(models, tmp_path):
    mem = _memory(models)
    mem.absorb(Candidate("Maya", "city", "Rome", timestamp=10.0))
    before = mem._state_version

    mem.query("Maya city Rome", top_k=3)
    mem.recall("Maya")
    mem.get_history("Maya")
    mem.status()
    mem.temporal_history_violations()
    mem.exclusive_temporal_conflicts()
    mem.save_checkpoint(str(tmp_path / "b1_readonly.npz"), create_backup=False)
    assert mem.delete_slot(10 ** 12 + 7) is False
    assert mem.purge_subject("Nobody") == 0

    assert mem._state_version == before


def test_version_starts_at_zero_and_invalidation_returns_new_value():
    mem = BlackHoleMemory(capacity=2)

    assert mem._state_version == 0
    assert mem._invalidate_retrieval_cache() == 1
    assert mem._invalidate_retrieval_cache() == 2
    assert mem._state_version == 2
