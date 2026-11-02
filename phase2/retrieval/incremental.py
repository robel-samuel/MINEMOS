"""Incremental copy-on-write snapshot rebuilding on top of the full backend.

The capture/build/publish contract, numerical selection and result rendering
are inherited from :mod:`phase2.retrieval.snapshot`.  This module only adds an
observer-only row delta computed between the current published snapshot and a
captured engine state.  Any transition that cannot be proven safe falls back
to a full rebuild, which remains the reference oracle.

Incremental builds still allocate one new matrix per update.  That is
allocator pressure, not a leak: unchanged rows and fields are shared with the
previous snapshot, and compacting to a full rebuild happens every
``max_incremental_updates`` publications.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np

from . import snapshot as C


class IncrementalUnsafe(RuntimeError):
    """The delta cannot be proven safe; use the full-rebuild oracle."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Delta(NamedTuple):
    reused: int
    replaced: int
    added: int
    removed: int
    allocated_bytes: int
    reason: str


class BaseState(NamedTuple):
    snapshot: "IncrementalSnapshot"
    slots: tuple
    memory: Any
    slot_ids: tuple


class IncrementalSnapshot(C.FrozenSnapshot):
    """FrozenSnapshot with build provenance; still immutable after construction."""

    __slots__ = ("_mode", "_base_version", "_delta")

    def __init__(self, version: int, index, fields, slot_ids,
                 mode: str, base_version: Optional[int], delta: Delta):
        super().__init__(version, index, fields, slot_ids)
        object.__setattr__(self, "_mode", str(mode))
        object.__setattr__(self, "_base_version", base_version)
        object.__setattr__(self, "_delta", delta)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def base_version(self) -> Optional[int]:
        return self._base_version

    @property
    def delta(self) -> Delta:
        return self._delta


def _new_index(matrix: np.ndarray, valid_from: np.ndarray,
               valid_until: np.ndarray, active: np.ndarray,
               rows: List[Optional[np.ndarray]], row_norm: np.ndarray,
               row_u: np.ndarray, width: int,
               verify_bound: bool = False) -> C.VectorizedRetriever:
    """Construct a VectorizedRetriever shell without running its rebuild loop."""
    index = object.__new__(C.VectorizedRetriever)
    index.verify_bound = bool(verify_bound)
    index.mem = C._DEAD_VIEW
    index.dense_encoder = None
    index.n = int(matrix.shape[0])
    index.matrix = matrix
    index.valid_from = valid_from
    index.valid_until = valid_until
    index.active = active
    index.slots = ()
    index._rows = tuple(rows)
    index._row_norm = row_norm
    index._row_u = row_u
    index.width = int(width)
    index._anchors = {}
    index.matrix_bytes = int(matrix.nbytes)
    index._gamma_cache = {}
    index.last_diagnostics = {}
    return index


class IncrementalSnapshotManager(C.SnapshotManager):
    """Full-rebuild manager plus an incremental builder and explicit fallback."""

    def __init__(self, mem, max_optimistic_attempts: int = 3,
                 hooks: Optional[Dict[str, C.Hook]] = None,
                 record_publications: bool = False,
                 max_incremental_updates: int = 64,
                 verify_bound: bool = False,
                 contended_fallback: str = "canonical"):
        user_hooks = dict(hooks or {})
        self._user_after_publish = user_hooks.get("after_publish")
        user_hooks["after_publish"] = self._after_publish
        super().__init__(mem, max_optimistic_attempts=max_optimistic_attempts,
                         hooks=user_hooks,
                         record_publications=record_publications,
                         verify_bound=verify_bound,
                         contended_fallback=contended_fallback)
        self.max_incremental_updates = max(0, int(max_incremental_updates))
        self.stats.update({
            "incremental_builds": 0,
            "full_builds": 0,
            "incremental_build_s": 0.0,
            "full_build_s": 0.0,
            "incremental_publications": 0,
            "full_publications": 0,
            "rows_reused": 0,
            "rows_replaced": 0,
            "rows_added": 0,
            "rows_removed": 0,
            "matrix_bytes_allocated": 0,
        })
        self.stats["full_rebuild_reasons"] = {}
        self._base: Optional[BaseState] = None
        self._base_lock = threading.Lock()
        self._incremental_since_full = 0
        self._force_full = False
        self._tls = threading.local()

    # -- base tracking -------------------------------------------------- #
    def _after_publish(self, attempt: int, winner) -> None:
        built = getattr(self._tls, "built", None)
        if built is not None and built[0] is winner:
            # Publications can complete out of order (the lock is released
            # before this hook runs), so the compare-and-set is serialised.
            with self._base_lock:
                if self._base is None or winner.version > self._base.snapshot.version:
                    self._base = BaseState(winner, built[1], built[2], built[3])
            if winner.mode == "full":
                self._incremental_since_full = 0
                self.stats["full_publications"] += 1
            else:
                self._incremental_since_full += 1
                self.stats["incremental_publications"] += 1
        self._tls.built = None
        hook = self._user_after_publish
        if hook is not None:
            hook(attempt, winner)

    @property
    def base_state(self) -> Optional[BaseState]:
        return self._base

    @property
    def incremental_since_full(self) -> int:
        return self._incremental_since_full

    def force_full_rebuild(self) -> None:
        self._force_full = True

    # -- build ---------------------------------------------------------- #
    def _build(self, capture) -> IncrementalSnapshot:
        self._tls.built = None
        # A restore that swaps ``mem.memory`` also advances ``_state_version``,
        # so such a capture is discarded at publication.
        if len(capture) == 3:
            capture = (capture[0], capture[1], capture[2], self.mem.memory)
        t0 = time.perf_counter()
        try:
            snapshot = self._build_with_fallback(capture)
        except Exception:
            self._tls.built = None
            raise
        elapsed = time.perf_counter() - t0
        self.stats["full_build_s" if snapshot.mode == "full"
                   else "incremental_build_s"] += elapsed
        slots = capture[1]
        memory = capture[3]
        self._tls.built = (snapshot, tuple(slots), memory,
                           tuple(id(s) for s in slots))
        return snapshot

    def _build_with_fallback(self, capture) -> IncrementalSnapshot:
        reason = self._full_reason(capture)
        if reason is not None:
            return self._full_build(capture, reason)
        try:
            snapshot = self._incremental_build(capture)
        except IncrementalUnsafe as exc:
            return self._full_build(capture, exc.reason)
        if not self._locked_build:
            self._fire("after_index_built", self._build_attempt, snapshot.index)
            self._fire("after_fields_captured", self._build_attempt,
                       snapshot.fields)
        return snapshot

    def _full_reason(self, capture) -> Optional[str]:
        version, slots, dense, memory = capture
        if self._force_full:
            return "forced"
        base = self._base
        if base is None:
            return "no_base"
        snapshot = base.snapshot
        if base.memory is not memory:
            return "memory_replaced"
        if snapshot.version >= version:
            return "base_not_older"
        if not slots or not base.slots:
            return "empty_state"
        if (len(base.slots) != snapshot.n
                or len(base.slot_ids) != snapshot.n
                or len(snapshot.fields) != snapshot.n
                or len(snapshot.rows) != snapshot.n):
            return "inconsistent_base"
        seen = set()
        for slot in slots:
            sid = id(slot)
            if sid in seen:
                return "duplicate_slot_ids"
            seen.add(sid)
        if self.max_incremental_updates and (
                self._incremental_since_full >= self.max_incremental_updates):
            return "periodic_compaction"
        return None

    def _full_build(self, capture, reason: str) -> IncrementalSnapshot:
        built = super()._build(capture[:3])
        self._force_full = False
        snapshot = IncrementalSnapshot(
            built.version, built.index, built.fields, built.slot_ids,
            mode="full", base_version=None,
            delta=Delta(0, 0, built.n, 0,
                        int(built.index.matrix.nbytes), reason),
        )
        self._validate(snapshot)
        self.stats["full_builds"] += 1
        self.stats["matrix_bytes_allocated"] += int(built.index.matrix.nbytes)
        reasons = self.stats["full_rebuild_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        return snapshot

    def _incremental_build(self, capture) -> IncrementalSnapshot:
        version, slots, dense, memory = capture
        base = self._base
        if base is None:
            raise IncrementalUnsafe("no_base")
        snapshot = base.snapshot
        base_index = snapshot.index
        base_fields = snapshot.fields
        base_rows = snapshot.rows
        base_slots = base.slots
        base_ids = base.slot_ids
        n_old = snapshot.n
        n = len(slots)
        if n == 0 or n_old == 0:
            raise IncrementalUnsafe("empty_state")

        rows, dims, dtypes = self._resolve_rows(slots, dense)
        if not dims:
            raise IncrementalUnsafe("all_rows_none")
        if len(set(dims)) != 1:
            raise ValueError(
                "inconsistent dense embedding shapes cannot be stacked: "
                f"{sorted(set(dims))}")
        width = int(dims[0][0])
        out_dtype = np.result_type(*dtypes)
        if width != int(base_index.width):
            raise IncrementalUnsafe("width_changed")
        if out_dtype != base_index.matrix.dtype:
            raise IncrementalUnsafe("dtype_changed")

        if len(base_slots) != n_old or len(base_ids) != n_old:
            raise IncrementalUnsafe("inconsistent_base")
        old_index: Dict[int, int] = {}
        for j, slot in enumerate(base_slots):
            sid = id(slot)
            if sid in old_index or base_ids[j] != sid:
                raise IncrementalUnsafe("inconsistent_base")
            old_index[sid] = j

        texts = self.mem.slot_texts
        fields: List[Any] = [None] * n
        planned_rows: List[Optional[np.ndarray]] = [None] * n
        survivor_new: List[int] = []
        survivor_old: List[int] = []
        changed_new: List[int] = []
        reused = replaced = added = 0

        for i, slot in enumerate(slots):
            sid = id(slot)
            old = old_index.get(sid)
            new_row = rows[i]
            if old is not None and base_slots[old] is slot:
                old_row = base_rows[old]
                old_fields = base_fields[old]
                text = texts.get(sid, slot.text)
                unchanged = (
                    new_row is old_row
                    and slot.debug_subject == old_fields.subject
                    and slot.debug_predicate == old_fields.predicate
                    and text == old_fields.text
                    and round(float(slot.confidence), 4) == old_fields.confidence
                    and slot.update_count == old_fields.update_count
                    and slot.valid_from == old_fields.valid_from
                    and slot.valid_until == old_fields.valid_until
                )
                if unchanged:
                    survivor_new.append(i)
                    survivor_old.append(old)
                    fields[i] = old_fields
                    planned_rows[i] = old_row
                    reused += 1
                    continue
                fields[i] = C._fields_for(slot, texts)
                planned_rows[i] = new_row
                changed_new.append(i)
                replaced += 1
                continue
            fields[i] = C._fields_for(slot, texts)
            planned_rows[i] = new_row
            changed_new.append(i)
            added += 1

        removed = n_old - reused - replaced
        slot_ids = tuple(id(s) for s in slots)
        if (not changed_new and reused == n == n_old
                and tuple(survivor_new) == tuple(range(n))
                and tuple(survivor_old) == tuple(range(n))):
            same = IncrementalSnapshot(
                version, base_index, tuple(fields), slot_ids,
                mode="incremental", base_version=snapshot.version,
                delta=Delta(reused, 0, 0, removed, 0, "no_visible_change"),
            )
            self._validate(same)
            self.stats["incremental_builds"] += 1
            self.stats["rows_reused"] += reused
            return same

        index = self._assemble_index(
            snapshot, slots, planned_rows, fields, survivor_new,
            survivor_old, changed_new, width, out_dtype)
        snap = IncrementalSnapshot(
            version, index, tuple(fields), slot_ids,
            mode="incremental", base_version=snapshot.version,
            delta=Delta(reused, replaced, added, removed,
                        int(index.matrix.nbytes
                            + index._row_norm.nbytes + index._row_u.nbytes
                            + index.valid_from.nbytes
                            + index.valid_until.nbytes
                            + index.active.nbytes),
                        "delta"),
        )
        self._validate(snap)
        self.stats["incremental_builds"] += 1
        self.stats["rows_reused"] += reused
        self.stats["rows_replaced"] += replaced
        self.stats["rows_added"] += added
        self.stats["rows_removed"] += removed
        self.stats["matrix_bytes_allocated"] += int(index.matrix.nbytes)
        return snap

    def _resolve_rows(self, slots, dense):
        rows = []
        dims = []
        dtypes = []
        for idx, slot in enumerate(slots):
            row = (dense[idx] if idx < len(dense) else slot.dense_emb)
            if row is not None:
                if not isinstance(row, np.ndarray):
                    raise ValueError(
                        f"slot {idx} dense embedding is not a numpy array: "
                        f"{type(row)!r}")
                dims.append(row.shape)
                dtypes.append(row.dtype)
            rows.append(row)
        return rows, dims, dtypes

    def _assemble_index(self, snapshot, slots, planned_rows, fields,
                        survivor_new, survivor_old, changed_new,
                        width, out_dtype):
        n = len(slots)
        base_index = snapshot.index
        matrix = np.zeros((n, width), dtype=out_dtype)
        row_norm = np.zeros(n, dtype=np.float64)
        row_u = np.zeros(n, dtype=np.float64)
        valid_from = np.empty(n, dtype=np.float64)
        valid_until = np.empty(n, dtype=np.float64)
        active = np.empty(n, dtype=bool)

        if survivor_new:
            s_new = np.asarray(survivor_new, dtype=np.int64)
            s_old = np.asarray(survivor_old, dtype=np.int64)
            matrix[s_new] = base_index.matrix[s_old]
            row_norm[s_new] = base_index._row_norm[s_old]
            row_u[s_new] = base_index._row_u[s_old]
            valid_from[s_new] = base_index.valid_from[s_old]
            valid_until[s_new] = base_index.valid_until[s_old]
            active[s_new] = base_index.active[s_old]

        for i in changed_new:
            row = planned_rows[i]
            if row is not None:
                matrix[i] = row
                row_norm[i] = float(
                    np.linalg.norm(row.astype(np.float64)))
                row_u[i] = float(np.finfo(row.dtype).eps) / 2.0
            slot = slots[i]
            valid_from[i] = float(slot.valid_from)
            valid_until[i] = (np.inf if slot.valid_until is None
                              else float(slot.valid_until))
            active[i] = bool(slot.is_active)

        return _new_index(matrix, valid_from, valid_until, active,
                          planned_rows, row_norm, row_u, width,
                          verify_bound=self.verify_bound)


class IncrementalSnapshotRetriever(C.SnapshotRetriever):
    """Query API over the incremental manager; results stay snapshot-owned."""

    strategy = "snapshot_incremental"

    def __init__(self, mem, max_optimistic_attempts: int = 3,
                 hooks: Optional[Dict[str, C.Hook]] = None,
                 record_publications: bool = False,
                 max_incremental_updates: int = 64,
                 max_retries: Optional[int] = None,
                 verify_bound: bool = False,
                 contended_fallback: str = "canonical"):
        if max_retries is not None:
            max_optimistic_attempts = max_retries
        self.mem = mem
        self.verify_bound = bool(verify_bound)
        self.manager = IncrementalSnapshotManager(
            mem, max_optimistic_attempts=max_optimistic_attempts,
            hooks=hooks, record_publications=record_publications,
            max_incremental_updates=max_incremental_updates,
            verify_bound=verify_bound,
            contended_fallback=contended_fallback)
        self.hooks = self.manager.hooks
        self._last_info: Dict[str, Any] = {}

    def force_full_rebuild(self) -> None:
        self.manager.force_full_rebuild()
        self.manager.invalidate()

    def query_with_details(self, query_text: str, top_k: Any = 5,
                           as_of: Optional[float] = None,
                           active_only: bool = True):
        results, details = super().query_with_details(
            query_text, top_k, as_of, active_only)
        snapshot = self.manager.published
        if snapshot is not None:
            details["snapshot_mode"] = getattr(snapshot, "mode", "full")
            details["base_version"] = getattr(snapshot, "base_version", None)
            delta = getattr(snapshot, "delta", None)
            details["delta"] = delta._asdict() if delta is not None else None
        self._last_info = details
        return results, details
