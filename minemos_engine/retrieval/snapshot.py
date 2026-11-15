"""Vectorized, snapshot-isolated retrieval with corrected numerical selection.

This module combines the isolated research components into one implementation:

* the resident-matrix scorer from the original vectorized prototype;
* the immutable capture/build/publish contract from B5;
* the rigorous selection bound, non-finite fallback and canonical ``top_k``
  slice semantics from B4;
* an all-None fast path that never multiplies a fabricated matrix, so a query
  whose dimension differs from the surviving rows cannot raise where the
  canonical loop would return zero similarities.

Canonical ``BlackHoleMemory.query()`` remains the reference implementation.
Returned similarity values are always the canonical per-row ``np.dot`` value,
rounded to four decimals; the matrix product only nominates candidates.
"""

from __future__ import annotations

import operator
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

Hook = Callable[[int, Any], None]


class SnapshotInvariantError(RuntimeError):
    """A built snapshot failed its internal consistency checks."""


class NeedsCanonicalFallback(RuntimeError):
    """Raised internally when the rigorous band cannot be applied."""


class SnapshotContended(RuntimeError):
    """Optimistic retries were exhausted; the caller should use canonical.

    Raised only when ``contended_fallback="canonical"``.  The alternative
    ``"locked_build"`` mode keeps the original B5 behaviour: build once more
    while holding the engine lock.  A locked build can stall writers for the
    full build duration, so the packaged default avoids it.
    """

    def __init__(self, attempts: int, version: int):
        super().__init__(
            f"snapshot build discarded {attempts} times at version {version}")
        self.attempts = attempts
        self.version = version


def _slice_count(top_k: Any, m: int) -> int:
    """Number of rows ``candidates[:top_k]`` returns for ``m`` candidates."""
    if top_k is None:
        return m
    k = operator.index(top_k)      # same TypeError as a list slice for floats
    if k >= 0:
        return min(k, m)
    return max(0, m + k)


class SlotFields(NamedTuple):
    slot_id: int
    memory_id: str
    subject: Optional[str]
    predicate: Optional[str]
    original_predicate: Optional[str]
    text: str
    confidence: float
    update_count: int
    is_active: bool
    valid_from: float
    valid_until: Optional[float]


def _represented_predicate(slot) -> str:
    return (slot.latest_predicate or slot.debug_predicate or "").strip().lower()


def _fields_for(slot, texts: Dict[int, str]) -> SlotFields:
    valid_until = slot.valid_until
    return SlotFields(
        slot_id=id(slot),
        memory_id=slot.memory_id,
        subject=slot.debug_subject,
        predicate=_represented_predicate(slot),
        original_predicate=slot.debug_predicate,
        text=texts.get(id(slot), slot.text),
        confidence=round(float(slot.confidence), 4),
        update_count=slot.update_count,
        is_active=valid_until is None,
        valid_from=slot.valid_from,
        valid_until=valid_until,
    )


def _result_rows(fields: Tuple[SlotFields, ...], chosen: np.ndarray,
                 exact: np.ndarray) -> List[Dict[str, Any]]:
    out = []
    for idx, sim in zip(chosen, exact):
        f = fields[int(idx)]
        out.append({
            "slot_index": int(idx),
            "slot_id": f.slot_id,
            "memory_id": f.memory_id,
            "similarity": round(float(sim), 4),
            "subject": f.subject,
            "predicate": f.predicate,
            "original_predicate": f.original_predicate,
            "text": f.text,
            "confidence": f.confidence,
            "update_count": f.update_count,
            "is_active": f.is_active,
            "valid_from": f.valid_from,
            "valid_until": f.valid_until,
        })
    return out


class VectorizedRetriever:
    """Read-only resident-matrix view of one logical engine state."""

    def __init__(self, mem: Any, verify_bound: bool = False):
        self.verify_bound = bool(verify_bound)
        self.mem = mem
        self.dense_encoder = getattr(mem, "dense_encoder", None)
        self.n = 0
        self.matrix: Optional[np.ndarray] = None
        self.valid_from: Optional[np.ndarray] = None
        self.valid_until: Optional[np.ndarray] = None
        self.active: Optional[np.ndarray] = None
        self.slots: List[Any] = []
        self._rows: List[Optional[np.ndarray]] = []
        self._row_norm = np.empty(0, dtype=np.float64)
        self._row_u = np.empty(0, dtype=np.float64)
        self.width = 0
        self._anchors: Dict[int, List[Tuple[np.ndarray, float]]] = {}
        self.matrix_bytes = 0
        self._gamma_cache: Dict[float, np.ndarray] = {}
        self.last_diagnostics: Dict[str, Any] = {}
        self.rebuild()

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #
    def rebuild(self) -> None:
        slots = self.mem.memory.slots
        self.slots = list(slots)
        n = len(slots)
        self.n = n
        self._anchors = {}
        self._gamma_cache = {}

        if n == 0:
            self.matrix = np.empty((0, 0), dtype=np.float32)
            self.valid_from = np.empty(0, dtype=np.float64)
            self.valid_until = np.empty(0, dtype=np.float64)
            self.active = np.empty(0, dtype=bool)
            self._rows = []
            self._row_norm = np.empty(0, dtype=np.float64)
            self._row_u = np.empty(0, dtype=np.float64)
            self.width = 0
            self.matrix_bytes = 0
            return

        rows: List[Optional[np.ndarray]] = []
        dims = []
        dtypes = []
        for idx, slot in enumerate(slots):
            emb = (self.mem.slot_dense_embs[idx]
                   if idx < len(self.mem.slot_dense_embs) else slot.dense_emb)
            if emb is None:
                rows.append(None)
                continue
            if not isinstance(emb, np.ndarray):
                raise ValueError(
                    f"slot {idx} dense embedding is not a numpy array: {type(emb)!r}")
            dims.append(emb.shape)
            dtypes.append(emb.dtype)
            rows.append(emb)

        if dims:
            if len(set(dims)) != 1:
                raise ValueError(
                    "inconsistent dense embedding shapes cannot be stacked: "
                    f"{sorted(set(dims))}")
            width = int(dims[0][0])
            out_dtype = np.result_type(*dtypes)
            matrix = np.zeros((n, width), dtype=out_dtype)
            for idx, row in enumerate(rows):
                if row is not None:
                    matrix[idx] = row
            self.matrix = matrix
            self.width = width
            self.matrix_bytes = int(matrix.nbytes)
        else:
            # Every slot has no dense embedding.  Canonical scores all of them
            # as 0.0 without ever touching the query vector, so the matrix must
            # not fabricate a width: a zero-width matrix makes the product path
            # unreachable and the query dimension irrelevant.
            self.matrix = np.zeros((n, 0), dtype=np.float32)
            self.width = 0
            self.matrix_bytes = 0

        self._rows = rows
        self._row_norm = np.array(
            [0.0 if r is None else float(np.linalg.norm(r.astype(np.float64)))
             for r in rows], dtype=np.float64)
        self._row_u = np.array(
            [0.0 if r is None else float(np.finfo(r.dtype).eps) / 2.0
             for r in rows], dtype=np.float64)
        self.valid_from = np.array([float(s.valid_from) for s in slots],
                                   dtype=np.float64)
        self.valid_until = np.array(
            [np.inf if s.valid_until is None else float(s.valid_until)
             for s in slots], dtype=np.float64)
        self.active = np.array([bool(s.is_active) for s in slots], dtype=bool)

        contra = getattr(self.mem, "slot_contra_anchors", {}) or {}
        for idx, slot in enumerate(slots):
            anchors = contra.get(id(slot), [])
            if anchors:
                self._anchors[idx] = list(anchors)

    # ------------------------------------------------------------------ #
    # Masking
    # ------------------------------------------------------------------ #
    def _mask(self, as_of: Optional[float], active_only: bool) -> np.ndarray:
        if as_of is None:
            if active_only:
                return self.active
            return np.ones(self.n, dtype=bool)
        # Canonical excludes a row when valid_from > as_of or when a finite
        # valid_until <= as_of.  NaN comparisons are False there, so the
        # negated form below keeps NaN rows exactly as the canonical loop.
        excluded = (self.valid_from > as_of) | (self.valid_until <= as_of)
        return ~excluded

    # ------------------------------------------------------------------ #
    # Exact scoring and rigorous bounds
    # ------------------------------------------------------------------ #
    def _exact(self, idx: int, q_dense: np.ndarray) -> float:
        row = self._rows[idx]
        return 0.0 if row is None else float(np.dot(row, q_dense))

    def _gamma_rows(self, q_u: float) -> np.ndarray:
        cached = self._gamma_cache.get(q_u)
        if cached is not None:
            return cached
        u = np.maximum(self._row_u, q_u)
        nu = self.width * u
        if np.any(nu >= 1.0):
            raise NeedsCanonicalFallback("d*u >= 1 for a row precision")
        gamma = nu / (1.0 - nu)
        self._gamma_cache[q_u] = gamma
        return gamma

    def _err_bound(self, indices: np.ndarray, q_dense: np.ndarray) -> np.ndarray:
        """Upper bound on |matrix_product - canonical_row_dot|.

        Uses ``gamma_d = d*u/(1-d*u)`` per row (Higham Thm 3.1), inflates the
        computed float64 norms by their own rounding bound, and covers the
        float64 evaluation of the bound itself.
        """
        if indices.size == 0:
            return np.empty(0, dtype=np.float64)
        q = np.asarray(q_dense)
        q_u = float(np.finfo(q.dtype).eps) / 2.0
        q_norm = float(np.linalg.norm(q.astype(np.float64)))
        gamma = self._gamma_rows(q_u)

        u64 = float(np.finfo(np.float64).eps) / 2.0
        nu_norm = (self.width + 2) * u64
        norm_inflate = (1.0 + nu_norm / (1.0 - nu_norm)) ** 2
        bound_rounding = 1.0 / (1.0 - 5.0 * u64)
        scale = q_norm * norm_inflate * bound_rounding
        return 2.0 * gamma[indices] * self._row_norm[indices] * scale

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _rank_exact(self, indices: np.ndarray, sims: np.ndarray,
                    q_dense: np.ndarray, top_k: Any):
        """Return (chosen, exact, info) using the canonical slice semantics."""
        m = int(indices.size)
        k_eff = _slice_count(top_k, m)
        if k_eff <= 0:
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64),
                    {"n_candidates": m, "k_effective": 0, "n_nominated": 0,
                     "max_bound": 0.0})
        if not np.isfinite(sims).all():
            raise NeedsCanonicalFallback("non-finite matrix score")
        err = self._err_bound(indices, q_dense)
        if k_eff == m:
            pos = np.arange(m)
            s = np.asarray(sims, dtype=np.float64)
        else:
            if not np.isfinite(err).all():
                raise NeedsCanonicalFallback("non-finite error bound")
            s = sims.astype(np.float64)
            u64 = float(np.finfo(np.float64).eps) / 2.0
            delta = 4.0 * u64 * (max(-float(s.min()), float(s.max()))
                                 + float(err.max()))
            w = err + delta
            kth = np.partition(s - w, m - k_eff)[m - k_eff]
            pos = np.flatnonzero((s + w) >= kth)
            if pos.size < k_eff:
                raise NeedsCanonicalFallback("selection band shorter than k")
        chosen = indices[pos]
        exact = np.array([self._exact(int(i), q_dense) for i in chosen],
                         dtype=np.float64)
        if not np.isfinite(exact).all():
            raise NeedsCanonicalFallback("non-finite exact score")
        if self.verify_bound and pos.size:
            observed = np.abs(s[pos] - exact)
            err_pos = err[pos]
            if not np.all(observed <= err_pos):
                worst = float(np.max(observed - err_pos))
                raise AssertionError(
                    f"selection band violated on a returned candidate by "
                    f"{worst:g}; d={self.width} n={m} k={k_eff}")

        order = np.argsort(-exact, kind="stable")[:k_eff]
        return chosen[order], exact[order], {
            "n_candidates": m,
            "k_effective": k_eff,
            "n_nominated": int(chosen.size),
            "max_bound": float(np.max(err)) if err.size else 0.0,
        }

    def _canonical_rank(self, mask: np.ndarray, q_dense: np.ndarray,
                        top_k: Any):
        """Exact replica of the canonical loop over this snapshot state."""
        candidates = []
        for idx in np.flatnonzero(mask):
            idx = int(idx)
            row = self._rows[idx]
            sim = float(np.dot(row, q_dense)) if row is not None else 0.0
            candidates.append((sim, idx))
        candidates.sort(key=lambda x: -x[0])
        chosen = []
        exact = []
        for sim, idx in candidates[:top_k]:     # canonical slice semantics
            chosen.append(idx)
            exact.append(sim)
        return (np.asarray(chosen, dtype=np.int64),
                np.asarray(exact, dtype=np.float64))

    def resolve(self, q_dense: np.ndarray, top_k: Any,
                as_of: Optional[float], active_only: bool):
        """Return (chosen, exact, info) for one query against this state."""
        mask = self._mask(as_of, active_only)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            # Canonical still evaluates ``candidates[:top_k]`` on the empty
            # list, so a non-index top_k raises even here.
            k_eff = _slice_count(top_k, 0)
            return (np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.float64),
                    {"mode": "empty", "n_candidates": 0,
                     "k_effective": k_eff, "n_nominated": 0})

        if self.width == 0:
            # Every row is None: canonical yields 0.0 for each, independent of
            # the query vector and its dimension.
            sims = np.zeros(indices.size, dtype=np.float64)
        elif indices.size * 2 < self.n:
            sims = self.matrix[indices] @ q_dense
        else:
            sims = (self.matrix @ q_dense)[indices]

        fallback_reason = None
        if not np.isfinite(np.asarray(q_dense)).all():
            fallback_reason = "non_finite_query"
        elif not np.isfinite(sims).all():
            fallback_reason = "non_finite_matrix_score"

        if fallback_reason is None:
            try:
                chosen, exact, info = self._rank_exact(
                    indices, sims, q_dense, top_k)
            except NeedsCanonicalFallback as exc:
                fallback_reason = str(exc)
            else:
                info["mode"] = "vectorized"
                info["reason"] = None
                return chosen, exact, info

        chosen, exact = self._canonical_rank(mask, q_dense, top_k)
        return chosen, exact, {
            "mode": "fallback",
            "reason": fallback_reason,
            "n_candidates": int(indices.size),
            "k_effective": _slice_count(top_k, int(indices.size)),
            "n_nominated": None,
        }

    # ------------------------------------------------------------------ #
    # Live (non-snapshot) query API, used by numerical tests
    # ------------------------------------------------------------------ #
    def query(self, query_text: str, top_k: Any = 5,
              as_of: Optional[float] = None,
              active_only: bool = True) -> List[Dict[str, Any]]:
        if self.n == 0:
            self.last_diagnostics = {"mode": "empty", "n_candidates": 0,
                                     "k_effective": 0, "n_nominated": 0}
            return []
        q_dense = self.dense_encoder.encode(query_text)
        return self.query_with_vector(q_dense, top_k, as_of, active_only)

    def query_with_vector(self, q_dense: np.ndarray, top_k: Any = 5,
                          as_of: Optional[float] = None,
                          active_only: bool = True) -> List[Dict[str, Any]]:
        if self.n == 0:
            self.last_diagnostics = {"mode": "empty", "n_candidates": 0,
                                     "k_effective": 0, "n_nominated": 0}
            return []
        chosen, exact, info = self.resolve(q_dense, top_k, as_of, active_only)
        self.last_diagnostics = info
        return _result_rows(tuple(_fields_for(s, getattr(
            self.mem, "slot_texts", {}) or {}) for s in self.slots),
            chosen, exact)


_DEAD_VIEW = SimpleNamespace(memory=SimpleNamespace(slots=()),
                             slot_dense_embs=[], slot_contra_anchors={},
                             dense_encoder=None, slot_texts={})


class FrozenSnapshot:
    """One immutable logical state: version, index, fields and identities."""

    __slots__ = ("_version", "_index", "_fields", "_slot_ids", "_rows",
                 "_frozen")

    def __init__(self, version: int, index: VectorizedRetriever,
                 fields: Tuple[SlotFields, ...],
                 slot_ids: Tuple[int, ...]):
        object.__setattr__(self, "_version", int(version))
        object.__setattr__(self, "_fields", tuple(fields))
        object.__setattr__(self, "_slot_ids", tuple(slot_ids))
        object.__setattr__(self, "_rows", tuple(index._rows))
        self._freeze_index(index)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        raise AttributeError("FrozenSnapshot is immutable")

    def __delattr__(self, name):
        raise AttributeError("FrozenSnapshot is immutable")

    @staticmethod
    def _freeze_index(index: VectorizedRetriever) -> None:
        for arr in (index.matrix, index.valid_from, index.valid_until,
                    index.active, index._row_norm, index._row_u):
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)
        index._rows = tuple(index._rows)

        def _immutable(*_args, **_kwargs):
            raise AttributeError("FrozenSnapshot index is immutable")

        index.rebuild = _immutable
        index.slots = ()
        index.mem = _DEAD_VIEW
        index.dense_encoder = None
        index._anchors = {}

    @property
    def version(self) -> int:
        return self._version

    @property
    def n(self) -> int:
        return self._index.n

    @property
    def fields(self) -> Tuple[SlotFields, ...]:
        return self._fields

    @property
    def slot_ids(self) -> Tuple[int, ...]:
        return self._slot_ids

    @property
    def rows(self) -> Tuple[Any, ...]:
        return self._rows

    @property
    def index(self) -> VectorizedRetriever:
        return self._index

    def select(self, q_dense: np.ndarray, top_k: Any,
               as_of: Optional[float], active_only: bool):
        return self._index.resolve(q_dense, top_k, as_of, active_only)

    def render(self, chosen: np.ndarray,
               exact: np.ndarray) -> List[Dict[str, Any]]:
        return _result_rows(self._fields, chosen, exact)

    def search(self, q_dense: np.ndarray, top_k: Any,
               as_of: Optional[float], active_only: bool):
        chosen, exact, info = self.select(q_dense, top_k, as_of, active_only)
        return self.render(chosen, exact), info

    def __repr__(self) -> str:
        return f"FrozenSnapshot(version={self._version}, n={self.n})"


class CaptureView:
    """Read-only stand-in for the engine during an out-of-lock build."""

    def __init__(self, slots, dense, dense_encoder, slot_texts):
        self.memory = SimpleNamespace(slots=list(slots))
        self.slot_dense_embs = list(dense)
        self.slot_contra_anchors: Dict[int, list] = {}
        self.dense_encoder = dense_encoder
        self.slot_texts = slot_texts


class SnapshotManager:
    """Capture under the lock, build outside, publish only if the version holds."""

    CONTENTION_MODES = ("canonical", "locked_build")

    def __init__(self, mem, max_optimistic_attempts: int = 3,
                 hooks: Optional[Dict[str, Hook]] = None,
                 record_publications: bool = False,
                 verify_bound: bool = False,
                 contended_fallback: str = "canonical"):
        if int(max_optimistic_attempts) < 1:
            raise ValueError("max_optimistic_attempts must be >= 1")
        if contended_fallback not in self.CONTENTION_MODES:
            raise ValueError(
                f"contended_fallback must be one of {self.CONTENTION_MODES}, "
                f"got {contended_fallback!r}")
        self.mem = mem
        self.verify_bound = bool(verify_bound)
        self.max_optimistic_attempts = int(max_optimistic_attempts)
        self.contended_fallback = contended_fallback
        self.hooks: Dict[str, Hook] = dict(hooks or {})
        self._snapshot: Optional[FrozenSnapshot] = None
        self._build_attempt = 0
        self._locked_build = False
        self.stats: Dict[str, Any] = {
            "captures": 0, "builds": 0, "publications": 0, "adoptions": 0,
            "discarded": 0, "build_errors": 0, "fallbacks": 0,
            "contended_fallbacks": 0, "locked_builds": 0, "queries": 0,
            "capture_s": 0.0, "capture_n": 0,
            "publish_wait_s": 0.0, "publish_hold_s": 0.0, "publish_n": 0,
            "build_s": 0.0, "build_n": 0, "last_build_s": 0.0,
            "locked_build_s": 0.0, "locked_build_n": 0,
            "select_s": 0.0, "select_n": 0,
            "fallback_selections": 0, "vectorized_selections": 0,
        }
        self._publication_log: Optional[List[Dict[str, Any]]] = (
            [] if record_publications else None)

    def _fire(self, stage: str, attempt: int, payload: Any) -> None:
        hook = self.hooks.get(stage)
        if hook is not None:
            hook(attempt, payload)

    def _log(self, **entry) -> None:
        if self._publication_log is not None:
            self._publication_log.append(entry)

    @property
    def publication_log(self) -> Optional[List[Dict[str, Any]]]:
        return self._publication_log

    @property
    def published(self) -> Optional[FrozenSnapshot]:
        with self.mem._lock:
            return self._snapshot

    def invalidate(self) -> None:
        """Drop the published snapshot so the next query rebuilds from state."""
        with self.mem._lock:
            self._snapshot = None

    # -- build ---------------------------------------------------------- #
    def _build(self, capture) -> FrozenSnapshot:
        version, slots, dense = capture
        view = CaptureView(slots, dense, self.mem.dense_encoder,
                           self.mem.slot_texts)
        index = VectorizedRetriever(view, verify_bound=self.verify_bound)
        if not self._locked_build:
            self._fire("after_index_built", self._build_attempt, index)
        fields = tuple(_fields_for(s, view.slot_texts) for s in slots)
        if not self._locked_build:
            self._fire("after_fields_captured", self._build_attempt, fields)
        slot_ids = tuple(id(s) for s in slots)
        snapshot = FrozenSnapshot(version, index, fields, slot_ids)
        self._validate(snapshot)
        return snapshot

    def _build_timed(self, capture) -> FrozenSnapshot:
        t0 = time.perf_counter()
        built = self._build(capture)
        elapsed = time.perf_counter() - t0
        self.stats["build_s"] += elapsed
        self.stats["build_n"] += 1
        self.stats["last_build_s"] = elapsed
        return built

    def _validate(self, snapshot: FrozenSnapshot) -> None:
        n = snapshot.n
        if len(snapshot.fields) != n:
            raise SnapshotInvariantError(
                f"field count {len(snapshot.fields)} != index rows {n}")
        if len(snapshot.slot_ids) != n:
            raise SnapshotInvariantError(
                f"slot id count {len(snapshot.slot_ids)} != index rows {n}")
        if len(snapshot.rows) != n:
            raise SnapshotInvariantError(
                f"captured row count {len(snapshot.rows)} != index rows {n}")
        if n and snapshot.index.matrix.shape[0] != n:
            raise SnapshotInvariantError(
                f"matrix rows {snapshot.index.matrix.shape[0]} != {n}")
        if tuple(f.slot_id for f in snapshot.fields) != snapshot.slot_ids:
            raise SnapshotInvariantError(
                "captured field identities do not match the captured slots")

    # -- acquisition ---------------------------------------------------- #
    def _capture_locked(self) -> Tuple[int, tuple, tuple]:
        return (self.mem._state_version,
                tuple(self.mem.memory.slots),
                tuple(self.mem.slot_dense_embs))

    def acquire(self) -> Tuple[FrozenSnapshot, dict]:
        self.stats["queries"] += 1
        builds = 0
        attempted = 0
        discarded = 0

        for attempt in range(self.max_optimistic_attempts):
            t0 = time.perf_counter()
            with self.mem._lock:
                version = self.mem._state_version
                snap = self._snapshot
                if snap is not None and snap.version == version:
                    self._log(action="matched", version=version,
                              attempt=attempt)
                    if attempt == 0:
                        self.stats["last_read_lock_wait_s"] = (
                            time.perf_counter() - t0)
                    return snap, {
                        "attempts": builds, "build_attempts": attempted,
                        "fallback": False,
                        "snapshot_version": snap.version,
                        "matched_existing": True,
                    }
                t_capture = time.perf_counter()
                capture = (version, tuple(self.mem.memory.slots),
                           tuple(self.mem.slot_dense_embs))
                self.stats["capture_s"] += time.perf_counter() - t_capture
                self.stats["capture_n"] += 1
                self.stats["captures"] += 1
            self._fire("after_capture", attempt, version)

            self._build_attempt = attempt
            attempted += 1
            try:
                built = self._build_timed(capture)
            except Exception:
                self.stats["build_errors"] += 1
                with self.mem._lock:
                    moved = self.mem._state_version != version
                if moved and attempt + 1 < self.max_optimistic_attempts:
                    continue
                raise
            builds += 1
            self.stats["builds"] += 1
            self._fire("before_publish", attempt, built)

            t_publish = time.perf_counter()
            with self.mem._lock:
                t_hold = time.perf_counter()
                self.stats["publish_wait_s"] += t_hold - t_publish
                self.stats["publish_n"] += 1
                if self.mem._state_version != version:
                    discarded += 1
                    self.stats["discarded"] += 1
                    self.stats["publish_hold_s"] += (
                        time.perf_counter() - t_hold)
                    self._log(action="discarded", version=version,
                              current=self.mem._state_version,
                              attempt=attempt)
                    continue
                current = self._snapshot
                if current is None or current.version < version:
                    self._snapshot = built
                    self.stats["publications"] += 1
                    self._log(action="published", version=version,
                              attempt=attempt)
                else:
                    self.stats["adoptions"] += 1
                    self._log(action="adopted", version=current.version,
                              attempt=attempt)
                winner = self._snapshot
                info = {
                    "attempts": builds, "build_attempts": attempted,
                    "fallback": False,
                    "snapshot_version": winner.version,
                    "matched_existing": False,
                }
                self.stats["publish_hold_s"] += time.perf_counter() - t_hold
            self._fire("after_publish", attempt, winner)
            return winner, info

        # Bounded fallback: the lock is held for the whole build, so no writer
        # can interleave and no post-build re-check is needed.
        self._fire("fallback", self.max_optimistic_attempts,
                   {"discarded": discarded})
        self.stats["fallbacks"] += 1
        if self.contended_fallback == "canonical":
            # A locked build can block writers for the whole build.  The
            # engine's canonical path already gives exact results, so hand the
            # query back instead of holding the lock for a rebuild.
            version = self.mem._state_version
            self.stats["contended_fallbacks"] += 1
            self._log(action="contended_fallback", version=version,
                      attempt=self.max_optimistic_attempts)
            raise SnapshotContended(self.max_optimistic_attempts, version)
        with self.mem._lock:
            current = self._snapshot
            if current is None or current.version != self.mem._state_version:
                self._build_attempt = self.max_optimistic_attempts
                t_locked = time.perf_counter()
                self._locked_build = True
                attempted += 1
                try:
                    built = self._build_timed(self._capture_locked())
                finally:
                    self._locked_build = False
                self._snapshot = built
                self.stats["builds"] += 1
                self.stats["locked_builds"] += 1
                self.stats["locked_build_s"] += time.perf_counter() - t_locked
                self.stats["locked_build_n"] += 1
                builds += 1
                self._log(action="locked_build", version=built.version)
            winner = self._snapshot
            info = {
                "attempts": builds, "build_attempts": attempted,
                "fallback": True,
                "snapshot_version": winner.version,
                "matched_existing": False,
            }
        self._fire("after_publish", self.max_optimistic_attempts, winner)
        return winner, info


class SnapshotRetriever:
    """Query API over a ``SnapshotManager``; results are snapshot-owned."""

    strategy = "snapshot_full"

    def __init__(self, mem, max_optimistic_attempts: int = 3,
                 hooks: Optional[Dict[str, Hook]] = None,
                 record_publications: bool = False,
                 max_retries: Optional[int] = None,
                 verify_bound: bool = False,
                 contended_fallback: str = "canonical"):
        if max_retries is not None:
            max_optimistic_attempts = max_retries
        self.mem = mem
        self.verify_bound = bool(verify_bound)
        self.manager = SnapshotManager(
            mem, max_optimistic_attempts=max_optimistic_attempts,
            hooks=hooks, record_publications=record_publications,
            verify_bound=verify_bound,
            contended_fallback=contended_fallback)
        self.hooks: Dict[str, Hook] = self.manager.hooks
        self._last_info: Dict[str, Any] = {}

    def _fire(self, stage: str, attempt: int, payload: Any) -> None:
        hook = self.hooks.get(stage)
        if hook is not None:
            hook(attempt, payload)

    def force_full_rebuild(self) -> None:
        self.manager.invalidate()

    def query(self, query_text: str, top_k: Any = 5,
              as_of: Optional[float] = None,
              active_only: bool = True) -> List[Dict[str, Any]]:
        return self.query_with_details(query_text, top_k, as_of,
                                       active_only)[0]

    def query_with_details(self, query_text: str, top_k: Any = 5,
                           as_of: Optional[float] = None,
                           active_only: bool = True):
        t_acq = time.perf_counter()
        snapshot, info = self.manager.acquire()
        acquire_s = time.perf_counter() - t_acq
        if snapshot.n == 0:
            self._fire("after_selection", 0, snapshot)
            details = {"strategy": self.strategy, "early_empty": True,
                       "encoded": False, **info,
                       "acquire_ms": acquire_s * 1e3}
            self._last_info = details
            return [], details
        t_enc = time.perf_counter()
        q_dense = self.mem.dense_encoder.encode(query_text)
        encode_s = time.perf_counter() - t_enc
        t_sel = time.perf_counter()
        chosen, exact, sel_info = snapshot.select(
            q_dense, top_k, as_of, active_only)
        select_s = time.perf_counter() - t_sel
        self.manager.stats["select_s"] += select_s
        self.manager.stats["select_n"] += 1
        mode = sel_info.get("mode")
        if mode == "fallback":
            self.manager.stats["fallback_selections"] += 1
        elif mode == "vectorized":
            self.manager.stats["vectorized_selections"] += 1
        self._fire("after_selection", 0, snapshot)
        t_render = time.perf_counter()
        results = snapshot.render(chosen, exact)
        render_s = time.perf_counter() - t_render
        details = {"strategy": self.strategy, "early_empty": False,
                   "encoded": True, **info,
                   "selection_mode": mode,
                   "selection_reason": sel_info.get("reason"),
                   "n_candidates": sel_info.get("n_candidates"),
                   "n_nominated": sel_info.get("n_nominated"),
                   "acquire_ms": acquire_s * 1e3,
                   "encode_ms": encode_s * 1e3,
                   "select_ms": select_s * 1e3,
                   "render_ms": render_s * 1e3}
        self._last_info = details
        return results, details

    @property
    def last_info(self) -> Dict[str, Any]:
        return dict(self._last_info)
