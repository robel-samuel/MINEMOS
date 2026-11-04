"""Per-tenant engine lifecycle, persistence and provenance coordination.

The pool owns one canonical ``BlackHoleMemory`` per tenant.  Expensive models
are constructed once and shared read-only across tenants; canonical state,
checkpoints, provenance ledgers and Qdrant collection scopes are per tenant.
Every checkpoint write goes through an explicit ``allowed_base_dir`` so a
tenant identifier can never escape the configured checkpoint directory.

Ordering rule for mutations: absorb, append provenance, then save the
checkpoint.  A crash can therefore leave an orphaned provenance entry, which
startup reconciliation reports, but cannot leave a persisted memory with no
provenance record.  ``reconcile`` never invents provenance for untracked
memories; it reports them for operator investigation.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from .config import ServiceConfig, validate_tenant_id
from .provenance import (
    ProvenanceError,
    ProvenanceLedger,
    content_digest,
    fact_digest,
    make_ingestion_id,
)


class EnginePoolError(RuntimeError):
    """Base class for service-layer engine failures."""


class EngineInitializationError(EnginePoolError):
    """A tenant engine or its checkpoint could not be initialized."""


class PersistenceError(EnginePoolError):
    """The canonical checkpoint could not be written."""


@dataclass(frozen=True)
class ModelSet:
    dense_encoder: Any = None
    nli_gate: Any = None
    lex_encoder: Any = None


def build_default_models() -> ModelSet:
    """Loads the shared encoders exactly once per process."""
    from benchmarks.phase16_experiment import NliSemanticGate
    from benchmarks.phase18_experiment import (
        DenseSemanticEncoder,
        LEXICAL_ENCODER,
    )
    return ModelSet(
        dense_encoder=DenseSemanticEncoder(),
        nli_gate=NliSemanticGate(),
        lex_encoder=LEXICAL_ENCODER,
    )


EngineFactory = Callable[[ServiceConfig, str, ModelSet], Any]


def default_engine_factory(config: ServiceConfig, tenant_id: str,
                           models: ModelSet) -> Any:
    from phase2.black_hole_memory import BlackHoleMemory
    from phase2.retrieval.qdrant_config import QdrantConfig

    qdrant_config = None
    backend = None
    if config.qdrant_url or config.qdrant_path:
        qdrant_config = QdrantConfig(
            url=config.qdrant_url,
            path=None if config.qdrant_url else config.qdrant_path,
            api_key=config.qdrant_api_key,
            collection=f"{config.qdrant_collection_prefix}_{tenant_id}",
            tenant_id=tenant_id,
            timeout=config.qdrant_timeout,
            recreate=False,
        )
        backend = "qdrant"
    return BlackHoleMemory(
        capacity=config.engine_capacity,
        dim=config.engine_dim,
        dense_encoder=models.dense_encoder,
        nli_gate=models.nli_gate,
        lex_encoder=models.lex_encoder,
        retrieval_backend=backend,
        qdrant_config=qdrant_config,
        retrieval_mode=config.retrieval_mode,
    )


@dataclass
class TenantRuntime:
    tenant_id: str
    engine: Any
    ledger: ProvenanceLedger
    checkpoint_path: Path
    lock: threading.RLock = field(default_factory=threading.RLock)
    pending_persistence: bool = False
    persistence_error: Optional[str] = None
    last_persisted_at: Optional[float] = None
    reconciliation: Dict[str, Any] = field(default_factory=dict)

    def canonical_memory_ids(self) -> Set[str]:
        with self.engine._lock:
            return {
                slot.memory_id for slot in self.engine.memory.slots
                if not slot.redacted
            }


class EnginePool:
    """Owns all tenant engines for one service process."""

    def __init__(self, config: ServiceConfig,
                 engine_factory: Optional[EngineFactory] = None,
                 models: Optional[ModelSet] = None):
        self.config = config
        self._factory = engine_factory or default_engine_factory
        self._models = models
        self._runtimes: Dict[str, TenantRuntime] = {}
        self._lock = threading.RLock()
        self._started_at = time.time()
        self._shutdown = False
        self._initialized = False
        self.config.resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.resolved_provenance_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            for path in (self.config.resolved_provenance_dir,
                         self.config.resolved_checkpoint_dir):
                try:
                    os.chmod(path, 0o700)
                except OSError:
                    pass

    # -- lifecycle -------------------------------------------------------- #
    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def models(self) -> ModelSet:
        if self._models is None:
            self._models = build_default_models()
        return self._models

    def initialize(self, tenants: Iterable[str] = (),
                   probe_retrieval: bool = True) -> Dict[str, Any]:
        if self._shutdown:
            raise EnginePoolError("engine pool has been shut down")
        self.models  # load once, before any tenant engine is constructed
        loaded = []
        for tenant in tenants:
            runtime = self.get_runtime(tenant)
            if probe_retrieval:
                self._probe_retrieval(runtime)
            loaded.append(tenant)
        self._initialized = True
        return {"tenants": loaded, "models_loaded": True}

    def get_runtime(self, tenant: str) -> TenantRuntime:
        tenant = validate_tenant_id(tenant)
        with self._lock:
            if self._shutdown:
                raise EnginePoolError("engine pool has been shut down")
            existing = self._runtimes.get(tenant)
            if existing is not None:
                return existing
            runtime = self._create_runtime(tenant)
            self._runtimes[tenant] = runtime
            return runtime

    def _create_runtime(self, tenant: str) -> TenantRuntime:
        checkpoint_dir = self.config.resolved_checkpoint_dir
        tenant_dir = checkpoint_dir / tenant
        tenant_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = tenant_dir / "state.npz"
        engine = None
        try:
            engine = self._factory(self.config, tenant, self.models)
            if checkpoint_path.exists():
                engine.load_checkpoint(
                    str(checkpoint_path),
                    allowed_base_dir=str(checkpoint_dir))
            ledger = ProvenanceLedger(
                self.config.resolved_provenance_dir / f"{tenant}.jsonl")
            runtime = TenantRuntime(
                tenant_id=tenant,
                engine=engine,
                ledger=ledger,
                checkpoint_path=checkpoint_path,
            )
            runtime.reconciliation = ledger.reconcile(
                runtime.canonical_memory_ids())
            return runtime
        except Exception as exc:
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    pass
            raise EngineInitializationError(
                f"tenant {tenant!r} could not be initialized: "
                f"{type(exc).__name__}") from exc

    def _probe_retrieval(self, runtime: TenantRuntime) -> None:
        try:
            stats = runtime.engine.retrieval_stats()
            if stats.get("backend") == "canonical":
                return
            if not stats.get("initialized"):
                adapter = runtime.engine._get_retrieval_adapter()
                adapter.ensure_retriever().sync_state()
        except Exception:
            # The engine records the failure and queries fall back to
            # canonical; readiness reports the degraded state.
            pass

    def _ensure_saved(self, runtime: TenantRuntime) -> None:
        if runtime.persistence_error or runtime.pending_persistence:
            self._save(runtime)

    def _save(self, runtime: TenantRuntime) -> str:
        runtime.pending_persistence = True
        try:
            path = runtime.engine.save_checkpoint(
                str(runtime.checkpoint_path),
                create_backup=True,
                allowed_base_dir=str(self.config.resolved_checkpoint_dir),
            )
        except Exception as exc:
            runtime.persistence_error = type(exc).__name__
            raise PersistenceError(
                f"tenant {runtime.tenant_id!r} checkpoint could not be "
                f"written ({type(exc).__name__})") from exc
        runtime.pending_persistence = False
        runtime.persistence_error = None
        runtime.last_persisted_at = time.time()
        return path

    # -- mutation --------------------------------------------------------- #
    def absorb(self, tenant: str, *, subject: str, predicate: str,
               object: str, timestamp: Optional[float],
               confidence: float, source: Dict[str, Any],
               ingestion_id: Optional[str] = None) -> Dict[str, Any]:
        runtime = self.get_runtime(tenant)
        if timestamp is None:
            timestamp = time.time()
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        content_sha = content_digest(subject, predicate, object)
        fact_sha = fact_digest(subject, predicate, object)
        ingestion_id = make_ingestion_id(
            tenant=tenant,
            source=source,
            subject=subject,
            predicate=predicate,
            object_text=object,
            timestamp=timestamp,
            confidence=confidence,
            explicit=ingestion_id,
        )
        with runtime.lock:
            self._ensure_saved(runtime)
            existing = runtime.ledger.is_ingested(ingestion_id)
            if existing is not None:
                return {
                    "action": existing.get("action"),
                    "memory_id": existing.get("memory_id"),
                    "ingestion_id": ingestion_id,
                    "duplicate_of": existing.get("duplicate_of"),
                    "skipped": True,
                }
            duplicate_of = runtime.ledger.fact_owner(fact_sha)
            result = runtime.engine.absorb(
                (subject, predicate, object),
                timestamp=timestamp,
                confidence=confidence,
            )
            memory_id = result.get("memory_id")
            if not memory_id:
                raise EnginePoolError(
                    "engine did not return a stable memory id")
            try:
                runtime.ledger.record_ingest({
                    "event": "ingest",
                    "ingestion_id": ingestion_id,
                    "memory_id": memory_id,
                    "source_id": source["source_id"],
                    "source_type": source.get("source_type", "observation"),
                    "document_id": source.get("document_id"),
                    "conversation_id": source.get("conversation_id"),
                    "segment_index": source.get("segment_index"),
                    "original_timestamp": source.get("original_timestamp"),
                    "observed_timestamp": timestamp,
                    "confidence": float(confidence),
                    "content_sha256": content_sha,
                    "fact_sha256": fact_sha,
                    "duplicate_of": duplicate_of,
                    "action": result.get("action"),
                })
            except ProvenanceError:
                # Without provenance the new memory would be untraceable.
                # Destroy its content immediately and refuse to persist it.
                try:
                    runtime.engine.redact_memory(memory_id)
                except Exception:
                    pass
                runtime.persistence_error = "provenance"
                raise
            if self.config.persist_after_mutation:
                self._save(runtime)
            out = dict(result)
            out["ingestion_id"] = ingestion_id
            out["duplicate_of"] = duplicate_of
            out["skipped"] = False
            return out

    # -- reads ------------------------------------------------------------ #
    def query(self, tenant: str, query: str, top_k: int = 5,
              as_of: Optional[float] = None) -> List[Dict[str, Any]]:
        return self.get_runtime(tenant).engine.query(
            query, top_k=top_k, as_of=as_of)

    def recall(self, tenant: str, subject: str,
               predicate: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.get_runtime(tenant).engine.recall(
            subject, predicate=predicate)

    def history(self, tenant: str, subject: str,
                predicate: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.get_runtime(tenant).engine.get_history(
            subject, predicate=predicate)

    # -- deletion / redaction --------------------------------------------- #
    def _record_memory_action(self, runtime: TenantRuntime, memory_id: str,
                              action: str,
                              source_id: Optional[str] = None) -> None:
        try:
            runtime.ledger.record_memory_action(
                memory_id, action, source_id=source_id)
        except ProvenanceError:
            runtime.persistence_error = "provenance"
            raise

    def redact_memory(self, tenant: str, memory_id: str) -> bool:
        runtime = self.get_runtime(tenant)
        with runtime.lock:
            self._ensure_saved(runtime)
            changed = runtime.engine.redact_memory(memory_id)
            if changed:
                self._record_memory_action(runtime, memory_id, "redact")
                if self.config.persist_after_mutation:
                    self._save(runtime)
            return changed

    def delete_memory(self, tenant: str, memory_id: str) -> bool:
        runtime = self.get_runtime(tenant)
        with runtime.lock:
            self._ensure_saved(runtime)
            changed = runtime.engine.delete_memory(memory_id)
            if changed:
                self._record_memory_action(runtime, memory_id, "delete")
                if self.config.persist_after_mutation:
                    self._save(runtime)
            return changed

    def redact_source(self, tenant: str, source_id: str) -> Dict[str, int]:
        return self._source_action(tenant, source_id, "redact")

    def delete_source(self, tenant: str, source_id: str) -> Dict[str, int]:
        return self._source_action(tenant, source_id, "delete")

    def _source_action(self, tenant: str, source_id: str,
                       action: str) -> Dict[str, int]:
        runtime = self.get_runtime(tenant)
        with runtime.lock:
            self._ensure_saved(runtime)
            memory_ids = runtime.ledger.memory_ids_for_source(source_id)
            changed = 0
            for memory_id in memory_ids:
                if action == "redact":
                    ok = runtime.engine.redact_memory(memory_id)
                else:
                    ok = runtime.engine.delete_memory(memory_id)
                if ok:
                    self._record_memory_action(
                        runtime, memory_id, action, source_id=source_id)
                    changed += 1
            try:
                runtime.ledger.record_source_action(source_id, action, changed)
            except ProvenanceError:
                runtime.persistence_error = "provenance"
                raise
            if changed and self.config.persist_after_mutation:
                self._save(runtime)
            return {
                "source_id": source_id,
                "matched_memories": len(memory_ids),
                "changed_memories": changed,
            }

    # -- explicit checkpoint --------------------------------------------- #
    def checkpoint(self, tenant: str) -> Dict[str, Any]:
        runtime = self.get_runtime(tenant)
        with runtime.lock:
            path = self._save(runtime)
            return {
                "tenant": tenant,
                "checkpoint": str(path),
                "bytes": Path(path).stat().st_size,
                "saved_at": runtime.last_persisted_at,
            }

    def verify_provenance(self, tenant: str) -> Dict[str, Any]:
        return self.get_runtime(tenant).ledger.verify()

    # -- status ----------------------------------------------------------- #
    def status(self, tenant: str) -> Dict[str, Any]:
        runtime = self.get_runtime(tenant)
        return {
            "tenant": tenant,
            "engine": runtime.engine.status(),
            "retrieval": runtime.engine.retrieval_stats(),
            "provenance": runtime.ledger.status(),
            "checkpoint": self._checkpoint_info(runtime),
            "persistence": {
                "pending": runtime.pending_persistence,
                "last_error": runtime.persistence_error,
                "last_saved_at": runtime.last_persisted_at,
            },
            "reconciliation": runtime.reconciliation,
        }

    def _checkpoint_info(self, runtime: TenantRuntime) -> Dict[str, Any]:
        path = runtime.checkpoint_path
        info: Dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
        }
        if path.exists():
            stat = path.stat()
            info.update({
                "bytes": stat.st_size,
                "modified_at": stat.st_mtime,
            })
        backup = Path(f"{path}.bak")
        info["backup_exists"] = backup.exists()
        if backup.exists():
            info["backup_bytes"] = backup.stat().st_size
        return info

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok" if not self._shutdown else "stopped",
            "started_at": self._started_at,
            "uptime_seconds": round(time.time() - self._started_at, 3),
            "models_loaded": self._models is not None,
            "tenant_count": len(self._runtimes),
        }

    def readiness(self) -> Dict[str, Any]:
        tenants: Dict[str, Any] = {}
        ready = not self._shutdown
        degraded = False
        with self._lock:
            runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            retrieval = runtime.engine.retrieval_stats()
            retrieval_degraded = bool(retrieval.get("degraded"))
            retrieval_pending = (
                retrieval.get("backend") not in (None, "canonical")
                and not retrieval.get("initialized"))
            if retrieval_degraded or retrieval_pending:
                degraded = True
            if runtime.persistence_error:
                ready = False
            tenants[runtime.tenant_id] = {
                "retrieval_state": retrieval.get("state"),
                "retrieval_degraded": retrieval_degraded,
                "persistence_pending": runtime.pending_persistence,
                "persistence_error": runtime.persistence_error,
            }
        writable = self._directory_writable(
            self.config.resolved_checkpoint_dir)
        if not writable:
            ready = False
        return {
            "ready": bool(ready),
            "degraded": bool(degraded),
            "checkpoint_dir_writable": writable,
            "models_loaded": self._models is not None,
            "tenants": tenants,
        }

    @staticmethod
    def _directory_writable(path: Path) -> bool:
        probe = path / f".write-probe-{os.getpid()}-{threading.get_ident()}"
        try:
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            probe.unlink()
            return True
        except OSError:
            try:
                probe.unlink()
            except OSError:
                pass
            return False

    # -- shutdown --------------------------------------------------------- #
    def shutdown(self) -> Dict[str, Any]:
        with self._lock:
            if self._shutdown:
                return {"saved": [], "errors": [], "already_stopped": True}
            runtimes = list(self._runtimes.values())
            self._shutdown = True
        saved: List[str] = []
        errors: List[Dict[str, str]] = []
        for runtime in runtimes:
            with runtime.lock:
                try:
                    path = self._save(runtime)
                    saved.append(path)
                except PersistenceError as exc:
                    errors.append({
                        "tenant": runtime.tenant_id,
                        "error": type(exc).__name__,
                    })
                finally:
                    try:
                        runtime.engine.close()
                    except Exception as exc:  # noqa: BLE001 - shutdown best effort
                        errors.append({
                            "tenant": runtime.tenant_id,
                            "error": type(exc).__name__,
                        })
        return {"saved": saved, "errors": errors, "already_stopped": False}
