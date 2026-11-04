"""Append-only provenance ledger keyed by stable memory ids.

The engine's checkpoint stores facts, not their origin.  This ledger restores
traceability without changing the checkpoint schema: one JSON line per
ingested observation, plus tombstones for redaction and deletion.  Raw text is
never written here.  Only identifiers, timestamps, the engine's stable
``memory_id`` and SHA-256 digests are retained, so deleting or redacting source
content does not leave a second copy behind in the provenance log.

The ledger is advisory for recovery.  A crash between an engine mutation and a
checkpoint save can leave an ingested record whose memory is absent from the
restored checkpoint; :meth:`ProvenanceLedger.reconcile` marks those records
``orphaned`` instead of silently trusting them.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

LEDGER_VERSION = 1
_INGEST_KEYS = {
    "v", "event", "seq", "at", "ingestion_id", "memory_id", "source_id",
    "source_type", "document_id", "conversation_id", "segment_index",
    "original_timestamp", "observed_timestamp", "confidence",
    "content_sha256", "fact_sha256", "duplicate_of", "action",
}
_FORBIDDEN_KEYS = {
    "text", "subject", "predicate", "object", "content", "document",
    "message", "body", "raw", "source_text",
}


class ProvenanceError(RuntimeError):
    """The provenance ledger is malformed or cannot be written."""


def content_digest(subject: str, predicate: str, object_text: str) -> str:
    """Digest of the complete observation content (never stored in clear)."""
    payload = "\x1f".join((subject.strip(), predicate.strip(),
                           object_text.strip()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fact_digest(subject: str, predicate: str, object_text: str) -> str:
    """Digest of the complete fact, used for duplicate observation reporting."""
    payload = "\x1f".join((subject.strip().casefold(),
                           predicate.strip().casefold(),
                           object_text.strip().casefold()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_ingestion_id(
    *,
    tenant: str,
    source: Dict[str, Any],
    subject: str,
    predicate: str,
    object_text: str,
    timestamp: Optional[float],
    confidence: float,
    explicit: Optional[str] = None,
) -> str:
    """Deterministic ingestion id for retry-safe ingestion.

    Supplying the same material twice produces the same id, so a retried
    request is skipped instead of being absorbed twice.  When the caller does
    not supply a timestamp, the service resolves one before calling this
    function; otherwise repeated requests would differ.
    """
    if explicit:
        return str(explicit)
    payload = {
        "tenant": tenant,
        "source_id": source.get("source_id"),
        "source_type": source.get("source_type"),
        "document_id": source.get("document_id"),
        "conversation_id": source.get("conversation_id"),
        "segment_index": source.get("segment_index"),
        "subject": subject,
        "predicate": predicate,
        "object_sha": hashlib.sha256(
            object_text.encode("utf-8")).hexdigest(),
        "timestamp": timestamp,
        "confidence": float(confidence),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProvenanceLedger:
    """Thread-safe, append-only JSONL ledger for one tenant."""

    def __init__(self, path: Path, fsync: bool = True):
        self.path = Path(path)
        self._fsync = bool(fsync)
        self._lock = threading.RLock()
        self._ingest_by_id: Dict[str, Dict[str, Any]] = {}
        self._memory_records: Dict[str, List[Dict[str, Any]]] = {}
        self._source_memories: Dict[str, Set[str]] = {}
        self._fact_index: Dict[str, str] = {}
        self._tombstones: Dict[str, str] = {}
        self._orphaned: Set[str] = set()
        self._source_actions: List[Dict[str, Any]] = []
        self._seq = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            try:
                os.chmod(self.path.parent, 0o700)
            except OSError:
                pass
        self._load()

    # -- loading ---------------------------------------------------------- #
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ProvenanceError(
                f"cannot read provenance ledger {self.path}: {exc}") from exc
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                # A torn final line is the expected crash artefact; anything
                # earlier means the ledger was corrupted and must not be
                # silently truncated.
                if number == len(lines):
                    break
                raise ProvenanceError(
                    f"provenance ledger {self.path} has malformed line "
                    f"{number}") from exc
            self._apply(event)

    def _apply(self, event: Dict[str, Any]) -> None:
        kind = event.get("event")
        seq = event.get("seq")
        if isinstance(seq, int):
            self._seq = max(self._seq, seq)
        if kind == "ingest":
            ingestion_id = event.get("ingestion_id")
            memory_id = event.get("memory_id")
            if not ingestion_id or not memory_id:
                return
            if ingestion_id in self._ingest_by_id:
                return
            self._ingest_by_id[ingestion_id] = event
            self._memory_records.setdefault(memory_id, []).append(event)
            source_id = event.get("source_id")
            if source_id:
                self._source_memories.setdefault(source_id, set()).add(
                    memory_id)
            fact = event.get("fact_sha256")
            if fact:
                self._fact_index.setdefault(fact, memory_id)
        elif kind in ("redact", "delete"):
            memory_id = event.get("memory_id")
            if memory_id:
                self._tombstones[memory_id] = kind
        elif kind == "orphaned":
            memory_id = event.get("memory_id")
            if memory_id:
                self._orphaned.add(memory_id)
        elif kind == "source_action":
            self._source_actions.append(event)

    # -- writing ---------------------------------------------------------- #
    def _append(self, event: Dict[str, Any]) -> Dict[str, Any]:
        self._seq += 1
        event = dict(event)
        event["v"] = LEDGER_VERSION
        event["seq"] = self._seq
        event["at"] = event.get("at") or _now()
        line = json.dumps(
            event, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"))
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                if self._fsync:
                    os.fsync(handle.fileno())
        except OSError as exc:
            raise ProvenanceError(
                f"cannot append to provenance ledger {self.path}: {exc}"
            ) from exc
        self._apply(event)
        return event

    # -- queries ---------------------------------------------------------- #
    def is_ingested(self, ingestion_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._ingest_by_id.get(ingestion_id)

    def records_for_memory(self, memory_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._memory_records.get(memory_id, ()))

    def memory_ids_for_source(self, source_id: str,
                              active_only: bool = True) -> List[str]:
        with self._lock:
            ids = set(self._source_memories.get(source_id, ()))
            if active_only:
                ids = {
                    mid for mid in ids
                    if mid not in self._tombstones and mid not in self._orphaned
                }
            return sorted(ids)

    def sources(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                source_id: {
                    "memory_ids": len(ids),
                    "active_memory_ids": len({
                        mid for mid in ids
                        if mid not in self._tombstones
                        and mid not in self._orphaned
                    }),
                }
                for source_id, ids in sorted(self._source_memories.items())
            }

    def fact_owner(self, fact_sha256: str) -> Optional[str]:
        with self._lock:
            return self._fact_index.get(fact_sha256)

    # -- mutation --------------------------------------------------------- #
    def record_ingest(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Records one ingested observation.

        Returns ``(record, already_ingested)``.  ``already_ingested`` is true
        when the ingestion id was seen before; the caller should not absorb the
        observation again.
        """
        unknown = set(record) - _INGEST_KEYS
        if unknown:
            raise ProvenanceError(
                f"provenance record has unsupported fields: {sorted(unknown)}")
        forbidden = set(record) & _FORBIDDEN_KEYS
        if forbidden:
            raise ProvenanceError(
                "provenance must not contain raw content fields: "
                f"{sorted(forbidden)}")
        for field in ("ingestion_id", "memory_id", "source_id",
                      "content_sha256", "fact_sha256"):
            if not record.get(field):
                raise ProvenanceError(
                    f"provenance record is missing {field!r}")
        with self._lock:
            existing = self._ingest_by_id.get(record["ingestion_id"])
            if existing is not None:
                return existing, True
            event = self._append({"event": "ingest", **record})
            return event, False

    def record_memory_action(self, memory_id: str, action: str,
                             source_id: Optional[str] = None) -> Dict[str, Any]:
        if action not in ("redact", "delete"):
            raise ProvenanceError(f"unknown memory action {action!r}")
        with self._lock:
            return self._append({
                "event": action,
                "memory_id": memory_id,
                "source_id": source_id,
            })

    def record_source_action(self, source_id: str, action: str,
                             memory_count: int) -> Dict[str, Any]:
        if action not in ("redact", "delete"):
            raise ProvenanceError(f"unknown source action {action!r}")
        with self._lock:
            return self._append({
                "event": "source_action",
                "source_id": source_id,
                "action": action,
                "memory_count": int(memory_count),
            })

    def record_orphans(self, memory_ids: Iterable[str]) -> int:
        """Marks ledger entries whose canonical memory is absent after restore."""
        added = 0
        with self._lock:
            for memory_id in sorted(set(memory_ids)):
                if memory_id in self._orphaned or memory_id in self._tombstones:
                    continue
                self._append({"event": "orphaned", "memory_id": memory_id})
                added += 1
        return added

    def reconcile(self, canonical_memory_ids: Iterable[str]) -> Dict[str, Any]:
        """Reconciles the ledger against the canonical checkpoint.

        Entries whose memory is absent are marked orphaned (the crash-safe
        ordering writes provenance before the checkpoint).  Canonical memories
        with no ledger entry are reported as ``untracked`` but are never given
        fabricated provenance; they are visible so the operator can investigate.
        """
        canonical = set(canonical_memory_ids)
        with self._lock:
            active = {
                mid for mid in self._memory_records
                if mid not in self._tombstones and mid not in self._orphaned
            }
            missing = active - canonical
            untracked = canonical - set(self._memory_records)
        marked = self.record_orphans(missing)
        return {
            "orphaned": len(missing),
            "newly_orphaned": marked,
            "untracked": len(untracked),
            "untracked_memory_ids": sorted(untracked)[:100],
        }

    # -- reporting -------------------------------------------------------- #
    def status(self) -> Dict[str, Any]:
        with self._lock:
            size = self.path.stat().st_size if self.path.exists() else 0
            return {
                "ledger": str(self.path),
                "events": self._seq,
                "ingested_observations": len(self._ingest_by_id),
                "tracked_memories": len(self._memory_records),
                "sources": len(self._source_memories),
                "redacted_memories": sum(
                    1 for value in self._tombstones.values()
                    if value == "redact"),
                "deleted_memories": sum(
                    1 for value in self._tombstones.values()
                    if value == "delete"),
                "orphaned_memories": len(self._orphaned),
                "source_actions": len(self._source_actions),
                "bytes": size,
            }

    def verify(self) -> Dict[str, Any]:
        problems: List[str] = []
        with self._lock:
            for ingestion_id, event in self._ingest_by_id.items():
                if event.get("event") != "ingest":
                    problems.append(
                        f"{ingestion_id}: unexpected event type")
                if set(event) & _FORBIDDEN_KEYS:
                    problems.append(
                        f"{ingestion_id}: contains raw content")
                for field in ("content_sha256", "fact_sha256"):
                    value = event.get(field)
                    if not isinstance(value, str) or len(value) != 64:
                        problems.append(
                            f"{ingestion_id}: invalid {field}")
                if not event.get("memory_id"):
                    problems.append(f"{ingestion_id}: missing memory_id")
        return {
            "ok": not problems,
            "problems": problems[:50],
            "status": self.status(),
        }


def _now() -> float:
    import time
    return time.time()
