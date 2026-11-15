"""Deterministic BM25 ordering.

Equal BM25 scores used to be ordered by ``id(slot)``, which changes between
processes and between a live engine and a restored checkpoint.  These tests
pin the stable tie-break contract: equal scores are ordered by the caller's
stable identity, independent of insertion order or process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from minemos_engine.intelligence.bm25 import LocalBM25Index
from tests.support.nl_audit.harness import build_components
from minemos_engine.black_hole_memory import BlackHoleMemory
from minemos_engine.structured_candidate import Candidate

REPO = Path(__file__).resolve().parents[1]

_COMPONENTS = {}


def _components():
    if not _COMPONENTS:
        _COMPONENTS.update(build_components(stub=True))
    return _COMPONENTS


def _memory() -> BlackHoleMemory:
    components = _components()
    return BlackHoleMemory(
        capacity=20,
        dim=4096,
        dense_encoder=components["encoder"],
        nli_gate=components["gate"],
        verifier=components["verifier"],
        retrieval_backend="canonical",
        strict_entity_identity=True,
    )


def _build(docs):
    index = LocalBM25Index()
    for slot_id, text, stable_id in docs:
        index.add_document(slot_id, text, stable_id=stable_id)
    return index


# Documents of equal length that each contain the query term once: every
# score ties exactly, so only the tie-break decides the order.
TIE_DOCS = [
    (1, "alpha beta", "m-01"),
    (2, "alpha gamma", "m-02"),
    (3, "alpha delta", "m-03"),
    (4, "alpha epsilon", "m-04"),
    (5, "alpha zeta", "m-05"),
]


def test_equal_scores_rank_by_stable_identity_not_insertion_order():
    forward = _build(TIE_DOCS).query("alpha", top_k=5)
    backward = _build(list(reversed(TIE_DOCS))).query("alpha", top_k=5)
    assert [doc_id for doc_id, _ in forward] == [1, 2, 3, 4, 5]
    assert forward == backward
    scores = {score for _, score in forward}
    assert len(scores) == 1


def test_tie_order_survives_update_round_trip():
    index = _build(TIE_DOCS)
    index.update_document(3, "alpha delta", stable_id="m-03")
    assert [doc_id for doc_id, _ in index.query("alpha", top_k=5)] == [
        1, 2, 3, 4, 5]
    index.remove_document(2)
    assert [doc_id for doc_id, _ in index.query("alpha", top_k=5)] == [
        1, 3, 4, 5]


def test_tie_order_is_identical_in_a_separate_process():
    script = textwrap.dedent(
        """
        import json
        from minemos_engine.intelligence.bm25 import LocalBM25Index

        docs = %r
        # Insert in a different order than the test process; slot ids and
        # stable ids are unchanged, so the ranking must be unchanged.
        index = LocalBM25Index()
        for slot_id, text, stable_id in docs:
            index.add_document(slot_id, text, stable_id=stable_id)
        print(json.dumps(index.query("alpha", top_k=5)))
        """
    ) % (list(reversed(TIE_DOCS)),)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO), capture_output=True, text=True, timeout=120,
        check=True,
    )
    remote = json.loads(completed.stdout)
    local = [list(row) for row in _build(TIE_DOCS).query("alpha", top_k=5)]
    assert remote == local


def test_checkpoint_restore_keeps_tie_order(tmp_path):
    mem = _memory()
    facts = [
        ("Alice", "likes", "alpha beta"),
        ("Bob", "likes", "alpha gamma"),
        ("Carol", "likes", "alpha delta"),
        ("Dave", "likes", "alpha epsilon"),
    ]
    for offset, (subject, predicate, obj) in enumerate(facts):
        mem.absorb(
            Candidate(subject, predicate, obj, timestamp=float(offset)),
            timestamp=float(offset),
        )

    def ranking(engine: BlackHoleMemory):
        mapping = {id(slot): i for i, slot in enumerate(engine.memory.slots)}
        return engine.bm25_index.query(
            "alpha", top_k=10, id_to_slot_index=mapping)

    before = ranking(mem)
    assert len(before) == len(facts)
    assert len({score for _, score in before}) == 1

    path = tmp_path / "tie_order.npz"
    mem.save_checkpoint(str(path))
    restored = BlackHoleMemory.from_checkpoint(
        str(path),
        capacity=20,
        dim=4096,
        dense_encoder=_components()["encoder"],
        nli_gate=_components()["gate"],
        verifier=_components()["verifier"],
        retrieval_backend="canonical",
        strict_entity_identity=True,
    )
    assert ranking(restored) == before
    restored.close()
    mem.close()
