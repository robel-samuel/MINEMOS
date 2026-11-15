"""Near-duplicate candidate index: exact parity with the brute-force scan.

The index is an accelerator for ``_find_near_duplicate_slot``.  It must never
change the answer: for every candidate, the reported slot / field / edit kind
has to match an exhaustive scan of the same live slots.
"""

from __future__ import annotations

import numpy as np

from tests.support.nl_audit.harness import build_components
from minemos_engine.black_hole_memory import BlackHoleMemory
from minemos_engine.structured_candidate import Candidate

_COMPONENTS = {}


def _components():
    if not _COMPONENTS:
        _COMPONENTS.update(build_components(stub=True))
    return _COMPONENTS


def _memory(capacity: int = 40) -> BlackHoleMemory:
    components = _components()
    return BlackHoleMemory(
        capacity=capacity,
        dim=4096,
        dense_encoder=components["encoder"],
        nli_gate=components["gate"],
        verifier=components["verifier"],
        retrieval_backend="canonical",
        strict_entity_identity=True,
    )


def _brute_force(mem: BlackHoleMemory, candidate: Candidate):
    """The pre-optimization scan, kept here as the reference oracle."""
    for slot in mem.memory.slots:
        if not slot.is_active or slot.redacted:
            continue
        slot_subject = slot.debug_subject
        slot_predicate = slot.latest_predicate or slot.debug_predicate
        slot_object = mem._slot_value_text(slot)
        fields = (
            ("subject", slot_subject, candidate.subject),
            ("predicate", slot_predicate, candidate.predicate),
            ("object", slot_object, candidate.object),
        )
        equal = 0
        edited = None
        for name, value_a, value_b in fields:
            tokens_a = mem._comparison_tokens(value_a)
            tokens_b = mem._comparison_tokens(value_b)
            if tokens_a and tokens_a == tokens_b:
                equal += 1
                continue
            kind = mem._surface_variant_kind(value_a, value_b)
            if kind is None or edited is not None:
                edited = None
                break
            edited = (name, kind)
        if equal == 2 and edited is not None:
            return slot, edited[0], edited[1]
    return None


def _result(triple):
    if triple is None:
        return None
    slot, field, edit = triple
    return (slot.memory_id, field, edit)


FACTS = [
    ("Mira Ashford", "lives in", "Lisbon"),
    ("Mira Ashford", "works at", "Northwind Labs"),
    ("Mira Ashford", "role", "architect"),
    ("Mira Ashford", "age", "42"),
    ("Jonas Reed", "lives in", "Porto"),
    ("Jonas Reed", "works at", "Northwind Lab"),
    ("Jonas Reed", "role", "engineer"),
    ("Aiko Tanaka", "lives in", "Osaka"),
    ("Aiko Tanaka", "works at", "Foxglove Publishing"),
    ("Aiko Tanaka", "role", "editor"),
]

CANDIDATES = [
    ("Mira Ashford", "lives in", "Lisbon"),
    ("Mira Ahsford", "lives in", "Lisbon"),
    ("Mira Ashford", "lives in", "Lisbon, Portugal"),
    ("Mira Ashford", "works at", "Northwind Lab"),
    ("Jonas Reed", "works at", "Northwind Labs"),
    ("Jonas Reed", "live in", "Porto"),
    ("Aiko Tanaka", "works at", "Foxglove"),
    ("Aiko Tanaka", "worked at", "Foxglove Publishing"),
    ("Unknown", "likes", "Lisbon"),
    ("Mira Ashford", "age", "24"),
]


def test_indexed_detection_matches_brute_force():
    mem = _memory()
    for offset, (subject, predicate, obj) in enumerate(FACTS):
        mem.absorb(
            Candidate(subject, predicate, obj, timestamp=float(offset)),
            timestamp=float(offset),
        )
    mismatches = []
    for offset, (subject, predicate, obj) in enumerate(CANDIDATES):
        candidate = Candidate(subject, predicate, obj,
                              timestamp=100.0 + offset)
        indexed = _result(mem._find_near_duplicate_slot(candidate))
        brute = _result(_brute_force(mem, candidate))
        if indexed != brute:
            mismatches.append((candidate, indexed, brute))
    assert not mismatches, mismatches
    assert mem._near_dup_index is not None
    mem.close()


def test_index_matches_brute_force_after_temporal_update_and_eviction():
    mem = _memory(capacity=12)
    for offset, (subject, predicate, obj) in enumerate(FACTS):
        mem.absorb(
            Candidate(subject, predicate, obj, timestamp=float(offset)),
            timestamp=float(offset),
        )
    # A later observation of the same attribute either revises the slot
    # (invalidating the index) or is preserved as a new fact (extending it).
    # Both paths must leave detection in agreement with a full scan.
    update = Candidate("Mira Ashford", "lives in", "Madrid", timestamp=50.0)
    mem.absorb(update, timestamp=50.0)

    candidates = [
        Candidate("Mira Ashford", "lives in", "Madrid", timestamp=60.0),
        Candidate("Mira Ashford", "lives in", "Lisbon", timestamp=61.0),
        Candidate("Mira Ashford", "livs in", "Madrid", timestamp=62.0),
        Candidate("Jonas Reed", "works at", "Northwind Lab", timestamp=63.0),
    ]
    for candidate in candidates:
        assert (_result(mem._find_near_duplicate_slot(candidate))
                == _result(_brute_force(mem, candidate)))

    # Capacity pressure evicts a slot; detection must still agree.
    mem.absorb(
        Candidate("Noor Haddad", "lives in", "Amman", timestamp=70.0),
        timestamp=70.0)
    probe = Candidate("Noor Hadad", "lives in", "Amman", timestamp=80.0)
    assert (_result(mem._find_near_duplicate_slot(probe))
            == _result(_brute_force(mem, probe)))
    mem.close()


def test_redaction_removes_candidate():
    mem = _memory()
    for offset, (subject, predicate, obj) in enumerate(FACTS):
        mem.absorb(
            Candidate(subject, predicate, obj, timestamp=float(offset)),
            timestamp=float(offset),
        )
    target = next(slot for slot in mem.memory.slots
                  if slot.debug_subject == "Mira Ashford"
                  and slot.latest_object == "Lisbon")
    mem.redact_memory(target.memory_id)
    probe = Candidate("Mira Ahsford", "lives in", "Lisbon", timestamp=90.0)
    assert mem._find_near_duplicate_slot(probe) is None
    assert _brute_force(mem, probe) is None
    mem.close()


def test_large_state_uses_bounded_candidate_buckets():
    mem = _memory(capacity=300)
    dim = 4096
    for i in range(250):
        key = np.zeros(dim, dtype=np.float32)
        slot = mem.memory.add(
            key=key, value=key.copy(), confidence=0.9,
            timestamp=float(i),
            debug_subject=f"Person {i:03d}",
            debug_predicate="field",
            text=f"Person {i:03d} field value {i:03d}",
            latest_predicate="field",
            latest_object=f"value {i:03d}",
        )
        mem.slot_texts[id(slot)] = slot.text
        mem.slot_dense_embs.append(key)
        mem.bm25_index.add_document(id(slot), slot.text,
                                    stable_id=slot.memory_id)
    mem._invalidate_near_duplicate_index()

    probe = Candidate("Person 001", "field", "value 001", timestamp=500.0)
    mem._find_near_duplicate_slot(probe)
    buckets = mem._near_dup_index
    assert len(buckets[("subject", ("person", "001"))]) == 1
    assert len(buckets[("predicate", ("field",))]) == 250
    assert len(buckets[("object", ("value", "001"))]) == 1
    mem.close()
