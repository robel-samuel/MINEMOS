# Qdrant Integration

Status: implemented and tested. Canonical MINEMOS retrieval remains the default.

MINEMOS remains the product, the intelligence engine, and the architectural
authority. Qdrant is an optional persistent vector index inside MINEMOS. It is
not a second memory authority, does not participate in consolidation,
contradiction handling, retention, or eviction decisions, and cannot decide what
survives.

## Division of responsibility

| Concern | Owner |
|---|---|
| Slot creation, consolidation, contradiction/supersession | MINEMOS |
| Retention, tiering, eviction | MINEMOS |
| Temporal validity, `as_of` semantics | MINEMOS |
| Stable memory identity (`MemorySlot.memory_id`) | MINEMOS |
| Canonical persistence and checkpoint recovery | MINEMOS |
| Dense vector indexing and candidate search | Qdrant |
| Metadata filter pushdown | Qdrant |
| Candidate resolution, exact re-ranking, final result set | MINEMOS |
| BM25 index and hybrid candidate merge | MINEMOS |
| Tenant scoping / isolation | Both (Qdrant filter is mandatory; MINEMOS re-checks) |

Qdrant stores one point per eligible live or historical slot:

- point id: canonical `memory_id` (UUID)
- named vector: `dense` (384-d, DOT distance, same `all-MiniLM-L6-v2` encoder as
  canonical MINEMOS; no second embedding pipeline)
- payload: `memory_id`, `tenant_id`, `subject`/`subject_key`,
  `predicate`/`predicate_key`, `latest_predicate`/`latest_predicate_key`,
  `text`, `confidence`, `update_count`, `valid_from`, `valid_until`,
  `is_active`, `created_at`, `timestamp`

Every Qdrant hit is resolved back to a live canonical slot. Unknown point ids
are ignored. Temporal and metadata filters are re-applied exactly against the
engine, so Qdrant cannot widen or mis-rank the final result set.

## Enabling it

```python
from phase2.black_hole_memory import BlackHoleMemory
from phase2.retrieval import QdrantConfig

mem = BlackHoleMemory(
    capacity=1000,
    retrieval_backend="qdrant",
    qdrant_config=QdrantConfig(
        url="http://127.0.0.1:6333",   # or path="/var/lib/minemos-qdrant"
        api_key="...",                  # optional; never logged or serialized
        collection="minemos",
        tenant_id="customer-a",
    ),
)
```

Environment-driven configuration is also supported:

| Variable | Meaning |
|---|---|
| `QDRANT_URL` | Remote Qdrant endpoint (mutually exclusive with `QDRANT_PATH`) |
| `QDRANT_PATH` | Embedded persistent Qdrant directory, or `:memory:` |
| `QDRANT_API_KEY` | API key; excluded from `repr` and stats |
| `QDRANT_COLLECTION` | Collection name (default `minemos_memories`) |
| `QDRANT_TENANT_ID` | Mandatory tenant scope (default `default`) |
| `QDRANT_TIMEOUT` | Client timeout in seconds |
| `QDRANT_PREFER_GRPC`, `QDRANT_GRPC_PORT` | Remote transport options |
| `QDRANT_RECREATE` | Drop and recreate the collection at startup |
| `QDRANT_CANDIDATE_LIMIT` | Dense candidate beam size (default 128) |
| `QDRANT_BATCH_SIZE` | Upsert/delete batch size (default 256) |
| `QDRANT_MAX_RETRIES`, `QDRANT_RETRY_BACKOFF` | Retry policy for index sync |
| `QDRANT_PAYLOAD_INDEXES` | Create payload indexes in server mode (default true) |
| `QDRANT_ON_DISK` | Store vectors on disk in server mode |

`QdrantConfig` rejects configurations with both `url` and `path`, missing
tenant/collection names, and non-positive limits.

### Hybrid retrieval

```python
rows = mem.query("which researcher studies marine biology", top_k=5)
hybrid = mem.query("which researcher studies marine biology", top_k=5,
                   hybrid=True)
```

`hybrid=True` requires the Qdrant backend. The dense candidates returned by
Qdrant and the existing BM25 candidates (`top_k_bm25`) are merged with the same
ordered-deduplication rule already used by MINEMOS consolidation; MINEMOS then
applies temporal/metadata filters and exact dot-product re-ranking. Hybrid does
not introduce a second embedding pipeline and does not discard either existing
retriever.

### Metadata filtering

```python
mem.query("project status", filters={
    "subject": "alice",
    "predicate": "works_on",
    "min_confidence": 0.6,
    "min_update_count": 2,
    "memory_ids": [...],
})
```

Filters are translated into Qdrant payload conditions and re-applied exactly by
MINEMOS. Unknown filter keys raise `ValueError`.

## Lifecycle synchronization

- `absorb()` (insert, consolidate, temporal update) commits canonical state
  first, then reconciles the index.
- `delete_slot()` and `purge_subject()` reconcile after each logical deletion.
- `save_checkpoint()` persists `memory_id` (checkpoint schema v4); older v1-v3
  archives load unchanged and receive deterministic fallback ids.
- `load_checkpoint()` replaces canonical state, then forces a full index
  reconcile through `_sync_qdrant_after_restore()`.
- `sync_state()` is an idempotent convergence pass: unchanged slots are skipped,
  changed fingerprints are upserted in batches, stale tenant points are deleted.
- `rebuild()` deletes only this tenant's points and reconstructs them from
  canonical slots.

Index reconciliation compares a cheap tenant point count against tracked
fingerprints. A count mismatch triggers a full tenant scroll and stale-point
reconciliation; matching counts use incremental diffs. `force=True` always
performs the full pass.

## Failure isolation and recovery

A Qdrant failure never rolls back an accepted memory:

1. Canonical mutation commits.
2. Index sync is attempted. On failure the index is marked dirty, the error is
   recorded in `retrieval_stats()`, and retries use bounded backoff.
3. Queries detect the dirty/incomplete index and raise internally, causing
   `BlackHoleMemory` to answer from canonical retrieval.
4. The next successful sync, `rebuild()`, checkpoint restore, or retriever
   restart restores index service. If the collection is lost, canonical state is
   untouched and `rebuild()` recreates it.

`mem.close()` closes the Qdrant client. Runtime backend switches close the
previous retriever before constructing a new one.

## Multi-tenancy

`tenant_id` is mandatory. Every search, scroll, count, delete, and rebuild is
scoped by a Qdrant `tenant_id` filter. A count mismatch in one tenant triggers
reconciliation of that tenant only. Deleting or rebuilding one tenant never
touches another tenant's points in a shared collection.

## Production notes

- Use server mode (`QDRANT_URL`) for production. Embedded mode (`QDRANT_PATH`)
  is a real persistent database but uses brute-force search rather than the
  server's HNSW index.
- Payload indexes are created in server mode for tenant, key, confidence,
  update-count, and validity fields. Embedded mode skips them because Qdrant
  ignores them there.
- The index is derived. Back up canonical checkpoints, not Qdrant, as the source
  of truth. Qdrant snapshots can accelerate recovery but are not required.
- Do not expose the API key through logs or `retrieval_stats()`; neither path
  includes it.
- Historical Phase 31B/31C conclusions remain unchanged: Qdrant was rejected as
  an authoritative memory store. Here it is only a derived retrieval index.

## Tests

`tests/test_qdrant_integration.py` runs 17 real-Qdrant tests plus one
server-only roundtrip (skipped unless `MINEMOS_QDRANT_URL` is set). Coverage
includes creation and indexing, retrieval, consolidation, deletion, metadata
filters, restart behavior, rebuild after collection loss, ingestion outage
fallback, concurrent updates, incremental upsert discipline, idempotent
sync/retry, contradiction/supersession, checkpoint restore, tenant isolation,
eviction mirroring, API-key non-leakage, hybrid candidate merge, and the
Qdrant-backend requirement for hybrid.

By default the suite uses persistent embedded mode. Set `MINEMOS_QDRANT_URL`
(and optionally `MINEMOS_QDRANT_API_KEY`) to run the same tests, plus the
server-only roundtrip, against a real Qdrant server.

```bash
.venv/Scripts/python.exe -m pytest tests/test_qdrant_integration.py -q
MINEMOS_QDRANT_URL=http://127.0.0.1:6333 \
  .venv/Scripts/python.exe -m pytest tests/test_qdrant_integration.py -q
```

## Benchmark

`benchmarks/qdrant_retrieval_benchmark.py` compares MINEMOS canonical dense,
Qdrant dense, BM25, and hybrid retrieval on the same canonical memory at
increasing sizes. It reports Recall@1/3/5, MRR, nDCG@5, p50/p95 latency, CPU
time, ingestion time with live indexing, rebuild time, and RSS.

```bash
.venv/Scripts/python.exe -m benchmarks.qdrant_retrieval_benchmark
MINEMOS_QDRANT_URL=http://127.0.0.1:6333 \
  .venv/Scripts/python.exe -m benchmarks.qdrant_retrieval_benchmark
```

Results are written to `benchmarks/results/qdrant_integration_benchmark.json`
(embedded mode) or `benchmarks/results/qdrant_integration_benchmark_server.json`
(server mode). The historical Phase 27 and Phase 31 results are not modified or
reinterpreted.

Measured on 2026-11-03 for 100/300/1000 target facts (200/600/2000 indexed
points): Qdrant dense returned the same rows and metrics as canonical dense in
both modes. Server-mode p50 latency was lower than embedded mode (19.9 vs
34.2 ms at 200 points; 59.9 vs 134.5 ms at 2,000) but still above in-process
canonical retrieval (14.3 and 41.8 ms). Per-fact ingestion with live indexing was
68.7 / 66.6 / 138.9 ms in server mode, close to embedded mode, so ingestion is
dominated by per-mutation synchronization rather than transport. Hybrid metrics
equal dense because the final exact re-ranking is dense-based. These synthetic
measurements do not claim Qdrant is faster or more accurate than canonical
retrieval.
