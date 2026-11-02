"""Numerical and behavioural parity for the optional retrieval backends.

Every test compares the packaged optimized/incremental backends against the
canonical ``BlackHoleMemory.query()`` on the same static state.  The 650-case
audit in ``scratch/.../final_review/integration_parity.py`` is the exhaustive
version; these are the regression pins for each class of divergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from phase2.retrieval import (IncrementalSnapshotRetriever,
                              SnapshotRetriever, _slice_count)
from phase2.retrieval.snapshot import VectorizedRetriever
from tests._retrieval_helpers import (QUERY, FixedEncoder, Q_VECTOR, WIDTH,
                                      add_facts, make_memory, rows_equal,
                                      unit_vector)


def _retrievers(mem):
    return (SnapshotRetriever(mem), IncrementalSnapshotRetriever(mem))


def _assert_parity(mem, cases):
    full, incr = _retrievers(mem)
    for case in cases:
        canonical = mem.query(QUERY, **case)
        for ret in (full, incr):
            assert rows_equal(ret.query(QUERY, **case), canonical), (
                type(ret).__name__, case)


def _add_row(mem, row, *, subject=None, valid_from=0.0, valid_until=None,
             text=None):
    i = len(mem.memory.slots)
    subject = subject or f"E{i:05d}"
    text = text or f"{subject} p O{i:05d}."
    key = np.zeros(1, dtype=np.float32)
    slot = mem.memory.add(
        key=key, value=key, confidence=1.0, timestamp=float(i),
        debug_subject=subject, debug_predicate="p", text=text,
        dense_emb=row, valid_from=valid_from, valid_until=valid_until)
    mem.slot_texts[id(slot)] = text
    mem.slot_dense_embs.append(row)
    return slot


def _finish(mem):
    mem._invalidate_retrieval_cache()
    return mem


# ── top_k semantics ────────────────────────────────────────────────────────


def test_slice_count_matches_canonical_list_slice():
    for m in (0, 1, 5, 20):
        items = list(range(m))
        for k in (0, 1, 3, 5, 100, -1, -2, -5, -100, None):
            assert _slice_count(k, m) == len(items[:k])
    with pytest.raises(TypeError):
        _slice_count(2.5, 5)
    with pytest.raises(TypeError):
        _slice_count(1.0, 5)


def test_negative_none_and_large_top_k_match_canonical():
    mem = make_memory(capacity=30)
    add_facts(mem, [(f"S{i}", "likes", f"O{i}", float(i))
                    for i in range(10)])
    _assert_parity(mem, [{"top_k": k} for k in
                         (0, 1, 3, 10, 100, -1, -2, -5, -100, None)])


def test_non_integer_top_k_raises_the_canonical_type_error():
    mem = make_memory(capacity=10)
    add_facts(mem, [("S", "likes", "O", 1.0)])
    full, incr = _retrievers(mem)
    for k in (2.5, 1.0):
        with pytest.raises(TypeError):
            mem.query(QUERY, top_k=k)
        for ret in (full, incr):
            with pytest.raises(TypeError):
                ret.query(QUERY, top_k=k)


def test_top_k_none_returns_every_filtered_candidate():
    mem = make_memory(capacity=20)
    add_facts(mem, [(f"S{i}", "likes", f"O{i}", float(i))
                    for i in range(6)])
    for ret in _retrievers(mem):
        assert len(ret.query(QUERY, top_k=None)) == 6


# ── all-None embeddings ────────────────────────────────────────────────────


def _all_none_memory(query_width: int = WIDTH) -> "BlackHoleMemory":
    mem = make_memory(
        capacity=10, encoder=FixedEncoder(mapping={QUERY: unit_vector(17, query_width)}))
    for _ in range(5):
        _add_row(mem, None)
    return _finish(mem)


def test_all_none_rows_negative_top_k_matches_canonical():
    mem = _all_none_memory()
    for ret in _retrievers(mem):
        for k in (5, 100, -1, -2, -100, None):
            canonical = mem.query(QUERY, top_k=k)
            assert rows_equal(ret.query(QUERY, top_k=k), canonical)
            assert all(r["similarity"] == 0.0 for r in canonical)


def test_all_none_width_mismatch_uses_zero_similarities_not_a_failing_matmul():
    # Canonical never touches the query vector when every row is None, so a
    # 16-wide query against 384-wide slots must not raise.
    mem = _all_none_memory(query_width=16)
    canonical = mem.query(QUERY, top_k=5)
    assert len(canonical) == 5
    assert all(r["similarity"] == 0.0 for r in canonical)
    for ret in _retrievers(mem):
        assert rows_equal(ret.query(QUERY, top_k=5), canonical)
        assert rows_equal(ret.query(QUERY, top_k=-1), mem.query(QUERY, top_k=-1))


# ── non-finite inputs ──────────────────────────────────────────────────────


def test_nan_and_inf_rows_match_canonical_order_and_values():
    rows = [
        np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.array([0.8, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.full(8, np.inf, dtype=np.float32),
        np.full(8, -np.inf, dtype=np.float32),
        np.array([np.nan, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        np.array([0.7, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    ]
    q = np.zeros(8, dtype=np.float32)
    q[0] = 1.0
    mem = make_memory(capacity=20, encoder=FixedEncoder(mapping={QUERY: q}))
    for row in rows:
        _add_row(mem, row)
    _finish(mem)
    for case in ({"top_k": 10}, {"top_k": 10, "active_only": False},
                 {"top_k": -1}):
        canonical = mem.query(QUERY, **case)
        for ret in _retrievers(mem):
            assert rows_equal(ret.query(QUERY, **case), canonical)


def test_nan_and_inf_queries_match_canonical():
    mem = make_memory(capacity=20)
    for i in range(8):
        _add_row(mem, unit_vector(100 + i, 8))
    _finish(mem)
    for value in (np.nan, np.inf, -np.inf):
        q = np.zeros(8, dtype=np.float32)
        q[0] = value
        mem.dense_encoder = FixedEncoder(mapping={QUERY: q})
        canonical = mem.query(QUERY, top_k=10)
        for ret in _retrievers(mem):
            assert rows_equal(ret.query(QUERY, top_k=10), canonical)


# ── ordering, temporal filters, dtype and failures ─────────────────────────


def test_exact_ties_keep_ascending_slot_order():
    mem = make_memory(capacity=40)
    for _ in range(16):
        _add_row(mem, Q_VECTOR.copy())
    _finish(mem)
    for k in (5, 16, 100):
        canonical = mem.query(QUERY, top_k=k)
        assert [r["slot_index"] for r in canonical] == list(
            range(len(canonical)))
        for ret in _retrievers(mem):
            assert rows_equal(ret.query(QUERY, top_k=k), canonical)


def test_near_ties_rigorous_bound_nominates_every_row_needed():
    base = unit_vector(12)
    alt = unit_vector(13)
    alt = alt - float(alt @ base) * base
    alt = (alt / np.linalg.norm(alt)).astype(np.float32)
    mem = make_memory(capacity=600)
    for i in range(500):
        row = (base.astype(np.float64) + (i + 1) * 1e-7 * alt).astype(np.float32)
        _add_row(mem, row)
    _finish(mem)
    canonical = mem.query(QUERY, top_k=50)
    ret = SnapshotRetriever(mem, verify_bound=True)
    assert rows_equal(ret.query(QUERY, top_k=50), canonical)


def test_temporal_filters_match_canonical():
    mem = make_memory(capacity=20)
    specs = [(0.0, 5.0), (0.0, 10.0), (5.0, 10.0), (5.0, None),
             (10.0, 20.0), (10.0, None), (15.0, 20.0), (20.0, None),
             (20.0, 20.0), (25.0, None)]
    for i, (vf, vu) in enumerate(specs):
        _add_row(mem, unit_vector(200 + i), valid_from=vf, valid_until=vu)
    _finish(mem)
    cases = [{"top_k": 50, "as_of": a} for a in
             (None, -1.0, 0.0, 4.9, 5.0, 10.0, 12.0, 15.0, 20.0, 25.0)]
    cases += [{"top_k": 50, "active_only": False},
              {"top_k": 50, "as_of": 20.0, "active_only": True}]
    _assert_parity(mem, cases)


def test_dtype_mix_matches_canonical_and_mixed_dimensions_raise():
    mem = make_memory(capacity=10)
    for i in range(6):
        _add_row(mem, unit_vector(300 + i).astype(
            np.float64 if i % 2 else np.float32))
    _finish(mem)
    _assert_parity(mem, [{"top_k": 5}, {"top_k": 50}])

    bad = make_memory(capacity=10)
    _add_row(bad, unit_vector(400))
    _add_row(bad, unit_vector(401))
    _add_row(bad, unit_vector(402, 16))
    _finish(bad)
    with pytest.raises(ValueError):
        bad.query(QUERY, top_k=5)
    for ret in _retrievers(bad):
        with pytest.raises(ValueError):
            ret.query(QUERY, top_k=5)


def test_empty_engine_and_empty_filter_match_canonical():
    empty = make_memory(capacity=5)
    empty._invalidate_retrieval_cache()
    for k in (5, 0, -1, None, 2.5):
        # Canonical returns before it ever slices on an empty engine.
        canonical = empty.query(QUERY, top_k=k)
        assert canonical == []
        for ret in _retrievers(empty):
            assert ret.query(QUERY, top_k=k) == []

    mem = make_memory(capacity=10)
    _add_row(mem, unit_vector(500), valid_from=10.0, valid_until=20.0)
    _finish(mem)
    for ret in _retrievers(mem):
        assert ret.query(QUERY, top_k=10, as_of=0.0) == []
        with pytest.raises(TypeError):
            ret.query(QUERY, top_k=2.5, as_of=0.0)
    with pytest.raises(TypeError):
        mem.query(QUERY, top_k=2.5, as_of=0.0)


def test_larger_candidate_sets_match_canonical():
    mem = make_memory(capacity=2500)
    rng = np.random.default_rng(31)
    for _ in range(2000):
        _add_row(mem, rng.standard_normal(WIDTH).astype(np.float32))
    _finish(mem)
    for k in (1, 5, 50, 100, 2000):
        canonical = mem.query(QUERY, top_k=k)
        for ret in _retrievers(mem):
            assert rows_equal(ret.query(QUERY, top_k=k), canonical)


def test_live_retriever_reports_vectorized_and_fallback_modes():
    mem = make_memory(capacity=10)
    add_facts(mem, [("S", "likes", "O", 1.0)])
    live = VectorizedRetriever(mem)
    assert len(live.query(QUERY, top_k=5)) == 1
    assert live.last_diagnostics["mode"] == "vectorized"

    mem.dense_encoder = FixedEncoder(
        mapping={QUERY: np.full(WIDTH, np.nan, dtype=np.float32)})
    live = VectorizedRetriever(mem)
    rows = live.query(QUERY, top_k=5)
    assert live.last_diagnostics["mode"] == "fallback"
    assert live.last_diagnostics["reason"] == "non_finite_query"
    assert rows_equal(rows, mem.query(QUERY, top_k=5))
