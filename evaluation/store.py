"""Dedicated, persistent storage for real-world memory evaluation.

The evaluation environment is deliberately separate from any production or
research store: it owns its own directory, canonical checkpoint, provenance
ledger and results.  It never resets automatically.  Deleting or redacting a
source removes the normalized segment file that contains the source text;
the append-only event log records only identifiers, counts and timestamps.

Filesystem retention caveat: removal is not secure erasure.  Filesystem
snapshots, copy-on-write layers, disk images and backups can retain deleted
content.  Operators must apply their own backup retention rules; this module
cannot reach into external backups.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from minemos_service.config import ServiceConfig, validate_tenant_id
from minemos_service.engine_pool import (
    EnginePool,
    ModelSet,
    EngineFactory,
)
from minemos_service.provenance import (
    content_digest,
    fact_digest,
    make_ingestion_id,
)
from minemos_service.security import find_sensitive, redact_sensitive

from .formats import Segment, Observation, order_key

FORMAT_VERSION = 1
DEFAULT_TENANT = "evaluation"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_FORBIDDEN_INGESTION_FIELDS = {
    "text", "subject", "predicate", "object", "content", "message",
    "body", "raw",
}


class EvaluationStoreError(RuntimeError):
    """The evaluation store is missing, corrupted or unsafe to modify."""


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2,
                  sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_jsonl(path: Path, entries: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(
                entry, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            if number == len(lines):
                break
            raise EvaluationStoreError(
                f"{path} has malformed line {number}") from exc
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


class EvaluationStore:
    """One isolated evaluation directory."""

    def __init__(self, root: str | Path, tenant: str = DEFAULT_TENANT):
        self.root = Path(root).expanduser().resolve()
        self.tenant = validate_tenant_id(tenant)
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise EvaluationStoreError(
                f"{self.root} is not an initialized evaluation store; run "
                "'python -m evaluation.ingest init' first")
        try:
            self.manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationStoreError(
                f"cannot read evaluation manifest: {exc}") from exc
        if int(self.manifest.get("format_version", -1)) != FORMAT_VERSION:
            raise EvaluationStoreError(
                "unsupported evaluation format version "
                f"{self.manifest.get('format_version')!r}")
        self.checkpoint_dir = self.root / "checkpoints"
        self.provenance_dir = self.root / "provenance"
        self.sources_dir = self.root / "sources"
        self.ingestions_path = self.root / "ingestions.jsonl"
        self.source_events_path = self.root / "source_events.jsonl"
        self.queries_dir = self.root / "queries"
        self.ground_truth_dir = self.root / "ground_truth"
        self.results_dir = self.root / "results"
        for path in (self.checkpoint_dir, self.provenance_dir,
                     self.sources_dir, self.queries_dir,
                     self.ground_truth_dir, self.results_dir):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def init(cls, root: str | Path, tenant: str = DEFAULT_TENANT,
             description: str = "") -> "EvaluationStore":
        root = Path(root).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            return cls(root, tenant=tenant)
        if root.exists() and any(root.iterdir()):
            raise EvaluationStoreError(
                f"refusing to initialize {root}: directory is not empty and "
                "has no evaluation manifest")
        root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(manifest_path, {
            "format_version": FORMAT_VERSION,
            "created_at": _now_iso(),
            "tenant": validate_tenant_id(tenant),
            "description": description,
            "authoritative_state": "checkpoints/",
            "qdrant_role": "derived index only; never authoritative",
        })
        return cls(root, tenant=tenant)

    # -- source files ----------------------------------------------------- #
    def _source_path(self, source_id: str) -> Path:
        digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:32]
        return self.sources_dir / f"{digest}.jsonl"

    def existing_segment_keys(self, source_id: str) -> set[str]:
        path = self._source_path(source_id)
        recorded = {
            entry.get("segment_key")
            for entry in _read_jsonl(path)
            if entry.get("segment_key")
        }
        return recorded | self._tombstone_keys(source_id)

    def _tombstone_keys(self, source_id: str) -> set[str]:
        keys: set[str] = set()
        for event in _read_jsonl(self.source_events_path):
            if event.get("source_id") != source_id:
                continue
            if event.get("event") in ("delete", "redact", "segment_redact",
                                      "segment_delete"):
                keys.update(event.get("segment_keys") or ())
        return keys

    def source_tombstoned(self, source_id: str) -> Optional[str]:
        """Returns the destructive action applied to the whole source, if any."""
        action = None
        for event in _read_jsonl(self.source_events_path):
            if (event.get("source_id") == source_id
                    and event.get("event") in ("delete", "redact")):
                action = event["event"]
        return action

    def record_segment(self, segment: Segment,
                       ingestion_ids: Sequence[str]) -> None:
        path = self._source_path(segment.source_id)
        entry = {
            "segment_key": segment.segment_key,
            "source_id": segment.source_id,
            "source_type": segment.source_type,
            "document_id": segment.document_id,
            "conversation_id": segment.conversation_id,
            "segment_index": segment.segment_index,
            "speaker": segment.speaker,
            "title": segment.title,
            "original_timestamp": segment.original_timestamp,
            "timestamp_inferred": segment.timestamp_inferred,
            "char_start": segment.char_start,
            "char_end": segment.char_end,
            "content_sha256": segment.content_sha256,
            "ingestion_ids": list(ingestion_ids),
            "recorded_at": _now_iso(),
            "text": segment.text,
        }
        _append_jsonl(path, entry)

    def segments(self, source_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(self._source_path(source_id))

    def record_ingestion(self, record: Dict[str, Any]) -> None:
        forbidden = set(record) & _FORBIDDEN_INGESTION_FIELDS
        if forbidden:
            raise EvaluationStoreError(
                f"ingestion records must not contain content: {sorted(forbidden)}")
        _append_jsonl(self.ingestions_path, record)

    def record_source(self, source_id: str, source_type: str,
                      segment_count: int, ingestion_count: int,
                      skipped_count: int, extractor: str) -> None:
        _append_jsonl(self.source_events_path, {
            "event": "source",
            "source_id": source_id,
            "source_type": source_type,
            "segment_count": int(segment_count),
            "ingestion_count": int(ingestion_count),
            "skipped_count": int(skipped_count),
            "extractor": extractor,
            "at": _now_iso(),
        })

    def record_deletion(self, source_id: str, action: str,
                        changed_memories: int) -> None:
        path = self._source_path(source_id)
        segment_keys = [
            entry.get("segment_key") for entry in _read_jsonl(path)
            if entry.get("segment_key")
        ]
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            raise EvaluationStoreError(
                f"could not remove source text for {source_id}: {exc}"
            ) from exc
        _append_jsonl(self.source_events_path, {
            "event": action,
            "source_id": source_id,
            "changed_memories": int(changed_memories),
            "segment_keys": segment_keys,
            "at": _now_iso(),
        })

    def purge_memory_contents(self, memory_id: str, action: str) -> int:
        """Removes source-text segments whose ingestion produced ``memory_id``.

        The engine destroys the memory; this removes the evaluation store's
        normalized copy too.  Segment keys are tombstoned so a later rerun of
        the same external source does not silently re-ingest them.
        """
        if action not in ("redact", "delete"):
            raise EvaluationStoreError(f"unknown action {action!r}")
        matching_ingestions = {
            entry.get("ingestion_id")
            for entry in _read_jsonl(self.ingestions_path)
            if entry.get("memory_id") == memory_id
        }
        if not matching_ingestions:
            return 0
        removed = 0
        for source_id, state in self.sources().items():
            if not state["active"]:
                continue
            path = self._source_path(source_id)
            entries = _read_jsonl(path)
            kept = []
            tombstoned = []
            for entry in entries:
                if set(entry.get("ingestion_ids") or ()) & matching_ingestions:
                    tombstoned.append(entry.get("segment_key"))
                    removed += 1
                else:
                    kept.append(entry)
            if not tombstoned:
                continue
            _atomic_write_jsonl(path, kept)
            _append_jsonl(self.source_events_path, {
                "event": "segment_redact" if action == "redact"
                else "segment_delete",
                "source_id": source_id,
                "memory_id": memory_id,
                "segment_keys": [key for key in tombstoned if key],
                "at": _now_iso(),
            })
        return removed

    def sources(self) -> Dict[str, Dict[str, Any]]:
        state: Dict[str, Dict[str, Any]] = {}
        for event in _read_jsonl(self.source_events_path):
            source_id = event.get("source_id")
            if not source_id:
                continue
            if event.get("event") == "source":
                state[source_id] = {
                    "source_id": source_id,
                    "source_type": event.get("source_type"),
                    "segment_count": event.get("segment_count", 0),
                    "ingestion_count": event.get("ingestion_count", 0),
                    "skipped_count": event.get("skipped_count", 0),
                    "extractor": event.get("extractor"),
                    "ingested_at": event.get("at"),
                    "active": True,
                }
            elif event.get("event") in ("delete", "redact"):
                if source_id in state:
                    state[source_id]["active"] = False
                    state[source_id][f"{event['event']}_at"] = event.get("at")
        return state

    # -- verification ----------------------------------------------------- #
    def verify(self) -> Dict[str, Any]:
        problems: List[str] = []
        sources = self.sources()
        for source_id, state in sources.items():
            path = self._source_path(source_id)
            if state["active"] and not path.exists():
                problems.append(
                    f"{source_id}: active source has no segment file")
            if not state["active"] and path.exists():
                problems.append(
                    f"{source_id}: deleted source still has segment text")
        ingestion_ids: set[str] = set()
        for entry in _read_jsonl(self.ingestions_path):
            if set(entry) & _FORBIDDEN_INGESTION_FIELDS:
                problems.append("ingestion log contains raw content")
            ingestion_id = entry.get("ingestion_id")
            if ingestion_id:
                ingestion_ids.add(ingestion_id)
        for source_id, state in sources.items():
            if not state["active"]:
                continue
            for segment in self.segments(source_id):
                for ingestion_id in segment.get("ingestion_ids", ()):
                    if ingestion_id not in ingestion_ids:
                        problems.append(
                            f"{source_id}: segment references unknown "
                            f"ingestion id {ingestion_id}")
        return {
            "ok": not problems,
            "problems": problems[:50],
            "sources": len(sources),
            "active_sources": sum(
                1 for state in sources.values() if state["active"]),
            "ingestions": len(ingestion_ids),
        }

    # -- results ---------------------------------------------------------- #
    def new_run_dir(self, run_id: str) -> Path:
        if not re.match(r"^[A-Za-z0-9._:-]{1,128}$", run_id):
            raise EvaluationStoreError("run id has an invalid format")
        path = self.results_dir / run_id
        if path.exists():
            raise EvaluationStoreError(
                f"result version {run_id} already exists; results are "
                "immutable and versioned")
        path.mkdir(parents=True)
        return path


@dataclass
class EvaluationEnvironment:
    """An evaluation store plus its dedicated canonical engine."""

    store: EvaluationStore
    config: ServiceConfig
    pool: EnginePool
    tenant: str
    started_at: float

    @classmethod
    def open(cls, root: str | Path, tenant: str = DEFAULT_TENANT,
             retrieval_mode: str = "canonical",
             engine_factory: Optional[EngineFactory] = None,
             models: Optional[ModelSet] = None,
             qdrant_url: Optional[str] = None,
             qdrant_path: Optional[str] = None,
             qdrant_api_key: Optional[str] = None) -> "EvaluationEnvironment":
        store = EvaluationStore(root, tenant=tenant)
        config = ServiceConfig.load(
            env={},
            overrides={
                "data_dir": store.root,
                "checkpoint_dir": store.checkpoint_dir,
                "provenance_dir": store.provenance_dir,
                "environment": "development",
                "allow_unauthenticated_dev": True,
                "persist_after_mutation": True,
                "retrieval_mode": retrieval_mode,
                "qdrant_url": qdrant_url,
                "qdrant_path": qdrant_path,
                "qdrant_api_key": qdrant_api_key,
            },
        )
        pool = EnginePool(config, engine_factory=engine_factory,
                          models=models)
        pool.initialize([tenant], probe_retrieval=True)
        return cls(
            store=store,
            config=config,
            pool=pool,
            tenant=tenant,
            started_at=time.time(),
        )

    def close(self) -> Dict[str, Any]:
        return self.pool.shutdown()

    # -- ingestion -------------------------------------------------------- #
    def ingest_segments(self, segments: Sequence[Segment], *,
                        mode: str = "verbatim",
                        source_type: Optional[str] = None,
                        credential_policy: str = "reject",
                        dry_run: bool = False) -> Dict[str, Any]:
        if mode not in ("verbatim", "structured"):
            raise EvaluationStoreError(
                "mode must be 'verbatim' or 'structured'")
        if not segments:
            raise EvaluationStoreError("no segments to ingest")
        ordered = sorted(segments, key=order_key)
        source_ids = {segment.source_id for segment in ordered}
        if len(source_ids) != 1:
            raise EvaluationStoreError(
                "one ingestion call must contain exactly one source_id")
        source_id = next(iter(source_ids))
        source_type = source_type or ordered[0].source_type
        tombstones = self.store.source_tombstoned(source_id)
        if tombstones:
            raise EvaluationStoreError(
                f"source {source_id!r} was previously {tombstones}; use a new "
                "source_id to ingest it again instead of silently reviving "
                "deleted content")

        prepared: List[Tuple[Segment, List[Observation]]] = []
        for segment in ordered:
            for value in (segment.source_id, segment.document_id,
                          segment.conversation_id, segment.speaker,
                          segment.title):
                if value and find_sensitive(value):
                    raise EvaluationStoreError(
                        f"segment {segment.segment_key} has credentials in "
                        "its source identifiers; use opaque identifiers")
            text_kinds = find_sensitive(segment.text)
            if text_kinds:
                if credential_policy == "reject":
                    raise EvaluationStoreError(
                        f"segment {segment.segment_key} appears to contain "
                        "credentials; remove it or choose redact")
                segment = replace(segment, text=redact_sensitive(segment.text))
            observations = list(segment.observations)
            if mode == "verbatim":
                from .formats import verbatim_observations
                observations = verbatim_observations([segment])
            if not observations:
                raise EvaluationStoreError(
                    f"segment {segment.segment_key} has no observations; "
                    "supply a structured extractor or use verbatim mode")
            cleaned: List[Observation] = []
            for observation in observations:
                values = (observation.subject, observation.predicate,
                          observation.object)
                kinds = [kind for value in values
                         for kind in find_sensitive(value)]
                if kinds:
                    if credential_policy == "reject":
                        raise EvaluationStoreError(
                            f"segment {segment.segment_key} appears to "
                            "contain credentials; remove it or choose redact")
                    observation = Observation(
                        subject=redact_sensitive(observation.subject),
                        predicate=redact_sensitive(observation.predicate),
                        object=redact_sensitive(observation.object),
                        confidence=observation.confidence,
                        timestamp=observation.timestamp,
                    )
                cleaned.append(observation)
            prepared.append((segment, cleaned))

        existing = self.store.existing_segment_keys(source_id)
        run_timestamp = self.started_at
        ingested = 0
        skipped = 0
        planned = sum(len(observations) for _, observations in prepared)
        if dry_run:
            return {
                "dry_run": True,
                "source_id": source_id,
                "segments": len(prepared),
                "observations": planned,
                "already_recorded_segments": sum(
                    1 for segment, _ in prepared
                    if segment.segment_key in existing),
            }

        for segment, observations in prepared:
            segment_ingestion_ids: List[str] = []
            if segment.segment_key not in existing:
                for observation in observations:
                    timestamp = observation.timestamp
                    if timestamp is None:
                        timestamp = segment.original_timestamp
                    if timestamp is None:
                        timestamp = run_timestamp
                    source = {
                        "source_id": segment.source_id,
                        "source_type": source_type,
                        "document_id": segment.document_id,
                        "conversation_id": segment.conversation_id,
                        "segment_index": segment.segment_index,
                        "original_timestamp": segment.original_timestamp,
                    }
                    ingestion_id = make_ingestion_id(
                        tenant=self.tenant,
                        source=source,
                        subject=observation.subject,
                        predicate=observation.predicate,
                        object_text=observation.object,
                        timestamp=timestamp,
                        confidence=observation.confidence,
                    )
                    result = self.pool.absorb(
                        self.tenant,
                        subject=observation.subject,
                        predicate=observation.predicate,
                        object=observation.object,
                        timestamp=timestamp,
                        confidence=observation.confidence,
                        source=source,
                        ingestion_id=ingestion_id,
                    )
                    segment_ingestion_ids.append(ingestion_id)
                    if result.get("skipped"):
                        skipped += 1
                    else:
                        ingested += 1
                    self.store.record_ingestion({
                        "ingestion_id": ingestion_id,
                        "memory_id": result.get("memory_id"),
                        "source_id": segment.source_id,
                        "source_type": source_type,
                        "document_id": segment.document_id,
                        "conversation_id": segment.conversation_id,
                        "segment_index": segment.segment_index,
                        "segment_key": segment.segment_key,
                        "observed_timestamp": timestamp,
                        "original_timestamp": segment.original_timestamp,
                        "timestamp_inferred": segment.timestamp_inferred,
                        "content_sha256": content_digest(
                            observation.subject, observation.predicate,
                            observation.object),
                        "fact_sha256": fact_digest(
                            observation.subject, observation.predicate,
                            observation.object),
                        "duplicate_of": result.get("duplicate_of"),
                        "skipped": bool(result.get("skipped")),
                        "action": result.get("action"),
                        "extractor": mode,
                        "ingested_at": _now_iso(),
                    })
                self.store.record_segment(segment, segment_ingestion_ids)

        self.store.record_source(
            source_id=source_id,
            source_type=source_type,
            segment_count=len(prepared),
            ingestion_count=ingested,
            skipped_count=skipped,
            extractor=mode,
        )
        return {
            "dry_run": False,
            "source_id": source_id,
            "segments": len(prepared),
            "observations": planned,
            "ingested": ingested,
            "skipped": skipped,
            "new_segments": sum(
                1 for segment, _ in prepared
                if segment.segment_key not in existing),
        }

    # -- deletion --------------------------------------------------------- #
    def redact_memory(self, memory_id: str) -> Dict[str, Any]:
        changed = self.pool.redact_memory(self.tenant, memory_id)
        removed = self.store.purge_memory_contents(
            memory_id, "redact") if changed else 0
        return {
            "memory_id": memory_id,
            "changed": bool(changed),
            "source_segments_removed": removed,
        }

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        changed = self.pool.delete_memory(self.tenant, memory_id)
        removed = self.store.purge_memory_contents(
            memory_id, "delete") if changed else 0
        return {
            "memory_id": memory_id,
            "changed": bool(changed),
            "source_segments_removed": removed,
        }

    def delete_source(self, source_id: str) -> Dict[str, Any]:
        result = self.pool.delete_source(self.tenant, source_id)
        self.store.record_deletion(
            source_id, "delete", result["changed_memories"])
        return result

    def redact_source(self, source_id: str) -> Dict[str, Any]:
        result = self.pool.redact_source(self.tenant, source_id)
        self.store.record_deletion(
            source_id, "redact", result["changed_memories"])
        return result
