"""Qdrant-backed candidate retrieval for MINEMOS.

Division of responsibility:

* MINEMOS owns slot creation, consolidation, contradiction handling, retention
  and the canonical memory state.  Qdrant stores only a dense candidate vector
  plus a small filter payload per live slot.
* Qdrant returns candidate memory ids.  Every id is resolved against the live
  engine and re-ranked with MINEMOS's own exact dot-product math, temporal
  gating and metadata filters.
* Deleting the Qdrant collection loses no memory.  ``sync_state(force=True)``
  or ``rebuild()`` reconstructs it from the canonical slots.

The index is idempotent: point ids are stable ``MemorySlot.memory_id`` values,
upserts converge to the current slot, and stale ids are deleted.  A failed
index write never rolls back an accepted memory; the index is marked dirty and
queries fall back to canonical until a later sync succeeds.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .qdrant_config import QdrantConfig

VECTOR_NAME = "dense"
_LOCAL_MODE = "local"
_SERVER_MODE = "server"


class QdrantUnavailable(RuntimeError):
    """The index could not serve a consistent answer; use canonical retrieval."""


def _point_id(memory_id: Any) -> str:
    """Qdrant accepts UUID strings; normalize any stable engine id to one."""
    text = str(memory_id)
    try:
        return str(uuid.UUID(text))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, text))


def _fingerprint(dense: np.ndarray, payload: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(dense, dtype=np.float32).tobytes())
    digest.update(repr((
        payload.get("text"), payload.get("subject_key"),
        payload.get("predicate_key"), payload.get("latest_predicate_key"),
        payload.get("confidence"),
        payload.get("update_count"), payload.get("valid_from"),
        payload.get("valid_until"), payload.get("created_at"),
        payload.get("timestamp"), payload.get("is_active"),
    )).encode("utf-8"))
    return digest.hexdigest()


class QdrantRetriever:
    """One retriever per engine/tenant, owning a Qdrant collection handle."""

    strategy = "qdrant_dense"

    @property
    def manager(self) -> "QdrantRetriever":
        """Adapter compatibility: this retriever owns its own lifecycle."""
        return self

    def force_full_rebuild(self) -> Dict[str, Any]:
        return self.rebuild()

    def __init__(self, mem: Any, config: QdrantConfig):
        self.mem = mem
        self.config = config
        self.tenant_id = config.tenant_id
        self.collection = config.collection
        self._sync_lock = threading.RLock()
        self._indexed_fingerprints: Dict[str, str] = {}
        self._indexed_signatures: Dict[str, Tuple[Any, ...]] = {}
        self._indexed_version = -1
        self._dirty = True
        self._complete = False
        self._closed = False
        self._snapshot: Optional[Any] = None

        self._errors = 0
        self._last_error: Optional[str] = None
        self._upserts = 0
        self._deletes = 0
        self._searches = 0
        self._search_ms_total = 0.0
        self._sync_ms_total = 0.0
        self._last_sync_ms = None
        self._last_search_ms = None
        self._last_dense_candidates = None
        self._last_fused_candidates = None
        self._retry_after = 0.0

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise QdrantUnavailable(
                "the qdrant backend requires the 'qdrant-client' package") from exc

        if config.url:
            self.mode = _SERVER_MODE
            self.client = QdrantClient(
                url=config.url,
                api_key=config.api_key,
                timeout=config.timeout,
                prefer_grpc=config.prefer_grpc,
                grpc_port=config.grpc_port,
            )
        else:
            self.mode = _LOCAL_MODE
            if config.path == ":memory:":
                self.client = QdrantClient(location=":memory:")
            else:
                self.client = QdrantClient(path=config.path)

        self._ensure_collection(recreate=config.recreate)
        self.sync_state(force=True, use_retry=False)

    # ------------------------------------------------------------------ #
    # Collection lifecycle
    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        return int(getattr(self.mem.dense_encoder, "dim", 384))

    def _collection_vectors(self) -> Any:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            return vectors.get(VECTOR_NAME)
        return vectors

    def _ensure_collection(self, recreate: bool = False) -> None:
        from qdrant_client import models as qm

        if recreate and self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        if not self.client.collection_exists(self.collection):
            vectors = qm.VectorParams(
                size=self.dim, distance=qm.Distance.DOT,
                on_disk=True if self.config.on_disk else None)
            self.client.create_collection(
                self.collection, vectors_config={VECTOR_NAME: vectors})
            self._ensure_payload_indexes()
            return

        vector = self._collection_vectors()
        size = getattr(vector, "size", None)
        if size is not None and int(size) != self.dim:
            raise QdrantUnavailable(
                f"collection {self.collection!r} stores {size}-d vectors but "
                f"MINEMOS encodes {self.dim}-d dense vectors; set "
                f"QdrantConfig.recreate=True or use a different collection")

    def _ensure_payload_indexes(self) -> None:
        if not self.config.ensure_payload_indexes:
            return
        # Local-mode Qdrant accepts these calls but warns they have no effect.
        # Skipping them keeps test output clean; server mode gets the indexes.
        if self.mode == _LOCAL_MODE:
            return
        from qdrant_client import models as qm

        indexes = (
            ("tenant_id", qm.PayloadSchemaType.KEYWORD),
            ("subject_key", qm.PayloadSchemaType.KEYWORD),
            ("predicate_key", qm.PayloadSchemaType.KEYWORD),
            ("memory_id", qm.PayloadSchemaType.KEYWORD),
            ("confidence", qm.PayloadSchemaType.FLOAT),
            ("update_count", qm.PayloadSchemaType.INTEGER),
            ("valid_from", qm.PayloadSchemaType.FLOAT),
            ("valid_until", qm.PayloadSchemaType.FLOAT),
        )
        for field, schema in indexes:
            try:
                self.client.create_payload_index(
                    self.collection, field_name=field,
                    field_schema=schema, wait=True)
            except Exception as exc:  # noqa: BLE001 - index is an optimization
                self._last_error = f"payload_index:{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ #
    # Tenant scoping and filters
    # ------------------------------------------------------------------ #
    def _tenant_condition(self) -> Any:
        from qdrant_client import models as qm

        if not self.tenant_id:
            raise QdrantUnavailable("tenant_id is mandatory for Qdrant")
        return qm.FieldCondition(
            key="tenant_id", match=qm.MatchValue(value=self.tenant_id))

    def _tenant_filter(self) -> Any:
        from qdrant_client import models as qm

        return qm.Filter(must=[self._tenant_condition()])

    def _query_filter(self, as_of: Optional[float], active_only: bool,
                      filters: Optional[Dict[str, Any]]) -> Any:
        from qdrant_client import models as qm

        must: List[Any] = [self._tenant_condition()]
        if as_of is not None:
            must.append(qm.FieldCondition(
                key="valid_from", range=qm.Range(lte=float(as_of))))
            must.append(qm.Filter(should=[
                qm.FieldCondition(
                    key="valid_until", range=qm.Range(gt=float(as_of))),
                qm.IsNullCondition(is_null=qm.PayloadField(key="valid_until")),
            ]))
        elif active_only:
            must.append(qm.IsNullCondition(
                is_null=qm.PayloadField(key="valid_until")))

        if filters:
            if filters.get("subject") is not None:
                must.append(qm.FieldCondition(
                    key="subject_key",
                    match=qm.MatchValue(
                        value=self.mem._normalize_subject(filters["subject"]))))
            if filters.get("predicate") is not None:
                wanted = str(filters["predicate"]).strip().lower()
                # Canonical filtering accepts either the slot's original or its
                # consolidated (latest) predicate; the pushdown must not narrow
                # the candidate set more than the engine's own rule.
                must.append(qm.Filter(should=[
                    qm.FieldCondition(
                        key="predicate_key",
                        match=qm.MatchValue(value=wanted)),
                    qm.FieldCondition(
                        key="latest_predicate_key",
                        match=qm.MatchValue(value=wanted)),
                ]))
            if filters.get("min_confidence") is not None:
                must.append(qm.FieldCondition(
                    key="confidence",
                    range=qm.Range(gte=float(filters["min_confidence"]))))
            if filters.get("min_update_count") is not None:
                must.append(qm.FieldCondition(
                    key="update_count",
                    range=qm.Range(
                        gte=int(filters["min_update_count"]))))
            if filters.get("memory_ids") is not None:
                must.append(qm.FieldCondition(
                    key="memory_id",
                    match=qm.MatchAny(any=[
                        _point_id(m) for m in filters["memory_ids"]
                    ])))
        return qm.Filter(must=must)

    def _scroll_tenant_ids(self) -> set:
        ids: set = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, scroll_filter=self._tenant_filter(),
                limit=512, offset=offset,
                with_payload=False, with_vectors=False)
            ids.update(str(point.id) for point in points)
            if offset is None:
                return ids

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #
    @staticmethod
    def _slot_signature(slot: Any, text: str) -> Tuple[Any, ...]:
        """Cheap change detector for incremental index builds.

        MINEMOS replaces a slot's dense embedding by assigning a new array
        rather than mutating it in place.  The embedding itself is stored next
        to this signature (see ``_build_records``) so it stays alive and its
        identity cannot be confused with a later array that reuses the same
        ``id()``.
        """
        return (
            text,
            slot.debug_subject,
            slot.debug_predicate,
            float(slot.confidence),
            int(slot.update_count),
            float(slot.valid_from),
            None if slot.valid_until is None else float(slot.valid_until),
            bool(slot.is_active),
            float(slot.created_at),
            float(slot.timestamp),
        )

    def _build_records(self, force: bool = False
                       ) -> Tuple[int, List[Any], Dict[str, str],
                                  Dict[str, Tuple[Any, ...]], bool]:
        """Snapshot live slots and return ``(version, changed_points,
        fingerprints, signatures, complete)``.

        Unchanged slots reuse their cached fingerprint and are not turned into
        new ``PointStruct`` objects.  Called with no network I/O while the
        engine lock is held.
        """
        from qdrant_client import models as qm

        with self.mem._lock:
            slots = list(self.mem.memory.slots)
            texts = dict(getattr(self.mem, "slot_texts", {}) or {})
            embeddings = list(getattr(self.mem, "slot_dense_embs", []) or [])
            version = int(self.mem._state_version)

        points: List[Any] = []
        fingerprints: Dict[str, str] = {}
        signatures: Dict[str, Tuple[Any, ...]] = {}
        complete = True
        seen: set = set()
        for idx, slot in enumerate(slots):
            # Match canonical retrieval's embedding source and precedence.
            dense = (embeddings[idx] if idx < len(embeddings)
                     else getattr(slot, "dense_emb", None))
            if dense is None:
                complete = False
                continue
            point_id = _point_id(slot.memory_id)
            if point_id in seen:
                raise QdrantUnavailable(
                    f"duplicate memory_id {slot.memory_id!r}; refusing to "
                    f"overwrite one live memory's vector with another's")
            seen.add(point_id)
            text = texts.get(id(slot), slot.text or "")
            signature = self._slot_signature(slot, text)
            cached = self._indexed_fingerprints.get(point_id)
            cached_signature = self._indexed_signatures.get(point_id)
            if (not force and cached is not None
                    and cached_signature is not None
                    and cached_signature[0] is dense
                    and cached_signature[1:] == signature):
                fingerprints[point_id] = cached
                signatures[point_id] = cached_signature
                continue
            payload = {
                "memory_id": str(slot.memory_id),
                "tenant_id": self.tenant_id,
                "subject": slot.debug_subject,
                "subject_key": self.mem._normalize_subject(
                    slot.debug_subject),
                "predicate": slot.debug_predicate,
                "predicate_key": str(
                    slot.debug_predicate or "").strip().lower(),
                "latest_predicate": slot.latest_predicate,
                "latest_predicate_key": str(
                    slot.latest_predicate or "").strip().lower(),
                "text": text,
                "confidence": float(slot.confidence),
                "update_count": int(slot.update_count),
                "valid_from": float(slot.valid_from),
                "valid_until": (
                    None if slot.valid_until is None
                    else float(slot.valid_until)),
                "is_active": bool(slot.is_active),
                "created_at": float(slot.created_at),
                "timestamp": float(slot.timestamp),
            }
            fingerprints[point_id] = _fingerprint(dense, payload)
            signatures[point_id] = (dense,) + signature
            points.append(qm.PointStruct(
                id=point_id,
                vector={VECTOR_NAME: np.asarray(
                    dense, dtype=np.float32).tolist()},
                payload=payload,
            ))
        return version, points, fingerprints, signatures, complete

    def _upsert_batches(self, points: Sequence[Any]) -> None:
        for start in range(0, len(points), self.config.batch_size):
            batch = list(points[start:start + self.config.batch_size])
            self.client.upsert(self.collection, batch, wait=True)
            self._upserts += len(batch)

    def _delete_ids(self, ids: Iterable[str]) -> None:
        from qdrant_client import models as qm

        values = sorted(set(ids))
        for start in range(0, len(values), self.config.batch_size):
            batch = values[start:start + self.config.batch_size]
            self.client.delete(
                self.collection,
                points_selector=qm.PointIdsList(points=batch), wait=True)
            self._deletes += len(batch)

    def sync_state(self, force: bool = False,
                   use_retry: bool = True) -> Dict[str, Any]:
        """Converge Qdrant to the current canonical slot set.

        Unchanged slots are skipped.  Stale tenant points are deleted.  The
        operation is idempotent and safe to retry after partial failure.
        """
        if self._closed:
            raise QdrantUnavailable("Qdrant retriever is closed")
        now = time.monotonic()
        if self._dirty and not force and now < self._retry_after:
            raise QdrantUnavailable(
                f"Qdrant sync cooling down after: {self._last_error}")

        with self._sync_lock:
            started = time.perf_counter()
            self._indexed_version = -1
            self._snapshot = None
            try:
                self._ensure_collection()
                version, points, fingerprints, signatures, complete = (
                    self._build_records(force=force))
                attempts = (self.config.max_retries + 1) if use_retry else 1
                last_exc: Optional[Exception] = None
                for attempt in range(attempts):
                    try:
                        tracked = set(self._indexed_fingerprints)
                        # A cheap count detects external drift (collection
                        # loss, foreign writers, restart with an empty cache);
                        # only then is a full tenant scroll required.
                        actual = int(self.client.count(
                            self.collection,
                            count_filter=self._tenant_filter(),
                            exact=True).count)
                        if force or actual != len(tracked):
                            existing = self._scroll_tenant_ids()
                        else:
                            existing = tracked
                        changed = points
                        stale = existing - set(fingerprints)
                        if changed:
                            self._upsert_batches(changed)
                        if stale:
                            self._delete_ids(stale)
                        last_exc = None
                        break
                    except Exception as exc:  # noqa: BLE001 - retried below
                        last_exc = exc
                        self._last_error = (
                            f"{type(exc).__name__}: {exc}")
                        if attempt + 1 < attempts:
                            time.sleep(
                                self.config.retry_backoff * (attempt + 1))
                if last_exc is not None:
                    raise last_exc

                self._indexed_fingerprints = fingerprints
                self._indexed_signatures = signatures
                self._indexed_version = version
                self._complete = complete
                self._dirty = not complete
                self._last_sync_ms = round(
                    (time.perf_counter() - started) * 1000.0, 3)
                self._sync_ms_total += self._last_sync_ms
                self._snapshot = SimpleNamespace(
                    version=version, n=len(fingerprints))
                return {
                    "version": version,
                    "points": len(fingerprints),
                    "changed": len(changed),
                    "stale_deleted": len(stale),
                    "complete": complete,
                    "sync_ms": self._last_sync_ms,
                }
            except Exception as exc:  # noqa: BLE001 - failure is isolated
                self._errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._dirty = True
                self._complete = False
                self._indexed_version = -1
                self._snapshot = None
                self._retry_after = now + self.config.retry_backoff
                raise QdrantUnavailable(
                    f"Qdrant sync failed: {type(exc).__name__}: {exc}"
                ) from exc

    def rebuild(self) -> Dict[str, Any]:
        """Delete this tenant's points and rebuild them from canonical slots.

        Other tenants sharing the collection are untouched.
        """
        with self._sync_lock:
            if self._closed:
                raise QdrantUnavailable("Qdrant retriever is closed")
            try:
                self._ensure_collection()
                self._delete_ids(self._scroll_tenant_ids())
            except Exception as exc:  # noqa: BLE001
                self._dirty = True
                raise QdrantUnavailable(
                    f"Qdrant rebuild could not clear tenant points: {exc}"
                ) from exc
            self._indexed_fingerprints = {}
            self._indexed_signatures = {}
            self._dirty = True
        return self.sync_state(force=True)

    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #
    def _effective_limit(self, top_k: Any, live_count: int) -> int:
        try:
            requested = int(top_k)
        except (TypeError, ValueError):
            requested = 1
        needed = requested if requested > 0 else live_count
        return min(max(needed, self.config.candidate_limit), live_count)

    def query(self, query_text: str, top_k: Any = 5,
              as_of: Optional[float] = None,
              active_only: bool = True,
              filters: Optional[Dict[str, Any]] = None,
              hybrid: bool = False) -> List[Dict[str, Any]]:
        if self._closed:
            raise QdrantUnavailable("Qdrant retriever is closed")
        with self._sync_lock:
            if (self._dirty or self._indexed_version < 0
                    or self._indexed_version != int(self.mem._state_version)):
                self.sync_state()
            if self._dirty or not self._complete:
                raise QdrantUnavailable(
                    "Qdrant index is incomplete; canonical retrieval required")
            if self._indexed_version != int(self.mem._state_version):
                raise QdrantUnavailable(
                    "memory changed during Qdrant sync; canonical retrieval "
                    "required")

            with self.mem._lock:
                if not self.mem.memory.slots:
                    return []
                slots = list(self.mem.memory.slots)
                texts = dict(getattr(self.mem, "slot_texts", {}) or {})
                embeddings = list(
                    getattr(self.mem, "slot_dense_embs", []) or [])
                version = int(self.mem._state_version)
                lexical: List[Tuple[int, float]] = []
                if hybrid:
                    id_to_slot_index = {
                        id(slot): idx for idx, slot in enumerate(slots)}
                    lexical = self.mem.bm25_index.query(
                        query_text=query_text,
                        top_k=self.mem.top_k_bm25,
                        id_to_slot_index=id_to_slot_index)

            limit = self._effective_limit(top_k, len(slots))
            if limit <= 0:
                return []
            q_dense = self.mem.dense_encoder.encode(query_text)
            started = time.perf_counter()
            try:
                response = self.client.query_points(
                    self.collection,
                    query=np.asarray(
                        q_dense, dtype=np.float32).tolist(),
                    using=VECTOR_NAME,
                    query_filter=self._query_filter(
                        as_of, active_only, filters),
                    limit=limit,
                    with_payload=False,
                    with_vectors=False,
                )
            except Exception as exc:  # noqa: BLE001
                self._errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise QdrantUnavailable(
                    f"Qdrant search failed: {type(exc).__name__}: {exc}"
                ) from exc

            self._searches += 1
            self._last_search_ms = round(
                (time.perf_counter() - started) * 1000.0, 3)
            self._search_ms_total += self._last_search_ms

            with self.mem._lock:
                if int(self.mem._state_version) != version:
                    raise QdrantUnavailable(
                        "memory changed during Qdrant search; canonical "
                        "retrieval required")
                live_by_id = {
                    _point_id(slot.memory_id): (idx, slot)
                    for idx, slot in enumerate(slots)
                }
                # Hybrid candidate merge mirrors MINEMOS's internal fusion:
                # dense ranks first, then lexical ranks not already present.
                candidate_ids = [str(point.id) for point in response.points]
                dense_candidates = len(candidate_ids)
                if hybrid:
                    seen = set(candidate_ids)
                    for idx, _score in lexical:
                        if 0 <= idx < len(slots):
                            point_id = _point_id(slots[idx].memory_id)
                            if point_id not in seen:
                                seen.add(point_id)
                                candidate_ids.append(point_id)
                self._last_dense_candidates = dense_candidates
                self._last_fused_candidates = len(candidate_ids)
                rows = []
                for candidate_id in candidate_ids:
                    found = live_by_id.get(candidate_id)
                    if found is None:
                        continue  # stale point from an earlier state
                    idx, slot = found
                    if as_of is not None:
                        if (slot.valid_from > as_of
                                or (slot.valid_until is not None
                                    and slot.valid_until <= as_of)):
                            continue
                    elif active_only and not slot.is_active:
                        continue
                    text = texts.get(id(slot), slot.text or "")
                    if filters and not self.mem._slot_matches_filters(
                            slot, filters):
                        continue
                    emb = (embeddings[idx] if idx < len(embeddings)
                           else getattr(slot, "dense_emb", None))
                    sim = float(np.dot(emb, q_dense)) if emb is not None else 0.0
                    rows.append((sim, slot, idx, text))

                rows.sort(key=lambda item: (-item[0], item[2]))
                return [
                    {
                        "slot_index": idx,
                        "slot_id": id(slot),
                        "similarity": round(sim, 4),
                        "subject": slot.debug_subject,
                        "predicate": slot.debug_predicate,
                        "text": text,
                        "confidence": round(float(slot.confidence), 4),
                        "update_count": slot.update_count,
                        "is_active": slot.is_active,
                        "valid_from": slot.valid_from,
                        "valid_until": slot.valid_until,
                    }
                    for sim, slot, idx, text in rows[:top_k]
                ]

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> Dict[str, Any]:
        try:
            points = int(self.client.count(
                self.collection, count_filter=self._tenant_filter(),
                exact=True).count)
        except Exception:  # noqa: BLE001 - stats must never raise
            points = None
        searches = self._searches
        return {
            "strategy": self.strategy,
            "mode": self.mode,
            "collection": self.collection,
            "tenant_id": self.tenant_id,
            "vector": VECTOR_NAME,
            "dim": self.dim,
            "candidate_limit": self.config.candidate_limit,
            "points": points,
            "indexed_version": self._indexed_version,
            "dirty": self._dirty,
            "complete": self._complete,
            "errors": self._errors,
            "last_error": self._last_error,
            "upserts": self._upserts,
            "deletes": self._deletes,
            "searches": searches,
            "last_search_ms": self._last_search_ms,
            "avg_search_ms": (
                round(self._search_ms_total / searches, 3)
                if searches else None),
            "last_dense_candidates": self._last_dense_candidates,
            "last_fused_candidates": self._last_fused_candidates,
            "last_sync_ms": self._last_sync_ms,
        }

    def close(self) -> None:
        with self._sync_lock:
            if not self._closed:
                self._closed = True
                try:
                    self.client.close()
                except Exception:  # noqa: BLE001 - best effort
                    pass
