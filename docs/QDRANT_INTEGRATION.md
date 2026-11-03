# Qdrant Integration

Status: implemented and tested.  The `production` profile uses Qdrant as the
standard retrieval backend for new deployments.  Existing installations and
development keep the canonical default unless they opt in.

MINEMOS remains the product, the intelligence engine, and the architectural
authority. Qdrant is a derived persistent vector index inside MINEMOS. It is
not a second memory authority, does not participate in consolidation,
contradiction handling, retention, or eviction decisions, and cannot decide what
survives.

## Deployment modes

`retrieval_mode` selects how the retrieval backend is chosen.  It resolves in
this order: constructor argument, `MINEMOS_RETRIEVAL_MODE`, then
`RetrievalConfig.mode`; omitted values stay in `legacy` mode.  `production` is a
retrieval deployment profile and does not change MINEMOS's status as an
experimental memory platform.

| Mode | Selection | Behaviour |
|---|---|---|
| `legacy` (default) | no selector | Canonical retrieval unless a backend is selected explicitly. Existing installations behave exactly as before. |
| `production` | `retrieval_mode="production"` or `MINEMOS_RETRIEVAL_MODE=production` | Qdrant is the standard backend. Configuration is validated strictly at construction; missing or unsafe settings raise `RetrievalConfigurationError` instead of silently serving from canonical. Runtime unavailability still falls back to canonical, reports `degraded`, and logs one warning. |
| `canonical` | `retrieval_mode="canonical"` or `MINEMOS_RETRIEVAL_MODE=canonical` | Canonical-only. Qdrant is never constructed or contacted, even when `QDRANT_*` variables are present. Intended for development, tests, recovery, and existing installations that choose not to run an index. |

```python
from phase2.black_hole_memory import BlackHoleMemory
from phase2.retrieval import QdrantConfig

# Production: Qdrant is the standard retrieval backend.
mem = BlackHoleMemory(
    capacity=1000,
    retrieval_mode="production",
    qdrant_config=QdrantConfig(
        url="https://qdrant.internal:6333",
        api_key="...",
        collection="minemos",
        tenant_id="customer-a",
    ),
)

# Canonical-only: never touches Qdrant.
mem = BlackHoleMemory(capacity=1000, retrieval_mode="canonical")
```

Equivalent environment-driven selection:

```bash
MINEMOS_RETRIEVAL_MODE=production \
QDRANT_URL=https://qdrant.internal:6333 \
QDRANT_API_KEY=... \
QDRANT_TENANT_ID=customer-a \
  python app.py
```

### Production configuration rules

`production` mode validates the resolved Qdrant configuration before the engine
starts.  It requires:

- `QDRANT_URL` (or `QdrantConfig.url`). Embedded `QDRANT_PATH` is rejected.
- An `http`/`https` URL with a hostname.
- For non-loopback hosts: HTTPS (or `QDRANT_ALLOW_INSECURE=true` for a trusted
  plain-HTTP internal network) and `QDRANT_API_KEY`.
- `QDRANT_RECREATE=false`; automatic collection recreation is rejected in
  production.

Failures raise `phase2.retrieval.RetrievalConfigurationError` with the specific
missing or unsafe setting named in the message.  A configuration error is not a
runtime outage: it fails startup rather than silently degrading to canonical.

### Runtime state and fallback

`mem.retrieval_stats()` reports the active backend and one of four states:

| State | Meaning |
|---|---|
| `canonical` | Canonical retrieval is active by configuration. |
| `not_initialized` | A non-canonical backend is selected but not constructed yet. |
| `ready` | The selected backend answered the last retrieval. |
| `degraded` | The selected backend is configured but unavailable; answers are coming from canonical fallback. |

The payload also includes `mode`, `initialized`, and `degraded`.  In
`production` mode, an index-write failure and a query-time fallback each log a
one-time warning; accepted memory mutations never roll back and canonical state
remains intact.  The state returns to `ready` after the index synchronizes.

Runtime backend switching is constrained by mode: `production` permits the
standard `qdrant` backend or an emergency `canonical` rollback;
`canonical` permits only `canonical`.

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

The examples below use explicit construction.  In `production` mode the same
settings are required; see [Deployment modes](#deployment-modes).

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
| `MINEMOS_RETRIEVAL_MODE` | `legacy`, `production`, or `canonical` |
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
| `QDRANT_ALLOW_INSECURE` | Permit insecure HTTP to a non-loopback host in production |
| `QDRANT_HYBRID_FUSION` | Default hybrid ordering: `dense` or `rrf` |
| `QDRANT_RRF_K` | Reciprocal-rank constant for `rrf` (default 60) |

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

The default ordering is **dense**: BM25 can only add candidates that dense
retrieval did not return, and the final order is exact dense similarity.  The
existing integration benchmark therefore reports hybrid metrics equal to dense
on its lexically separable corpus; this is a property of the ordering, not a
measurement of BM25's ranking value.

The audit conclusion is that this merge is not a true score fusion: BM25
contributes candidates, not ordering.  Raw BM25 scores are unbounded and not on
the same scale as exact dot-product similarity, so score normalization would be
corpus- and query-dependent.  Rank fusion is the safer first experiment because
it needs only candidate order and leaves membership and the reported exact
dense similarity untouched.

#### Opt-in rank fusion

`fusion="rrf"` (or `QDRANT_HYBRID_FUSION=rrf`) is an experimental, opt-in
alternative that orders the merged candidate union by reciprocal rank instead
of dense order.  It changes ordering only: the candidate union is identical,
`similarity` remains the exact dense dot product, and no consolidation,
retention, contradiction, temporal, or eviction logic changes.

```python
rows = mem.query("which researcher studies marine biology", top_k=5,
                 hybrid=True, fusion="rrf")
```

`rrf_k` defaults to 60 (`QDRANT_RRF_K`).  This option exists so rank fusion can
be measured against the dense and BM25 baselines; it is not enabled by default
and no claim of improved ranking quality is made without measurement.

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

- New production deployments should set `retrieval_mode="production"` and a
  server-mode `QDRANT_URL`.  Existing installations remain on the `legacy`
  canonical default until they explicitly opt in.
- Embedded mode (`QDRANT_PATH`) is supported for development and tests but is
  rejected in `production`; it uses brute-force search rather than the server's
  HNSW index.
- Payload indexes are created in server mode for tenant, key, confidence,
  update-count, and validity fields. Embedded mode skips them because Qdrant
  ignores them there.
- The index is derived. Back up canonical checkpoints, not Qdrant, as the source
  of truth. Qdrant snapshots can accelerate recovery but are not required.
- Checkpoint schema v4 persists stable `memory_id` values. v1-v3 archives load
  unchanged; deterministic ids are assigned where an archive predates them, so
  enabling the index does not require a checkpoint migration.
- Do not expose the API key through logs or `retrieval_stats()`; neither path
  includes it.
- Historical Phase 31B/31C conclusions remain unchanged: Qdrant was rejected as
  an authoritative memory store. Here it is only a derived retrieval index.

## Tests

`tests/test_retrieval_production_mode.py` covers backend selection and mode
resolution, production startup validation (missing URL, embedded path, insecure
remote, missing API key, recreate, malformed URL), canonical-only isolation,
runtime rollback restrictions, unavailable-Qdrant fallback with `degraded`
stats and warnings, checkpoint compatibility under production, and hybrid
fusion defaults/validation including RRF ordering math.

`tests/test_qdrant_integration.py` runs 20 real-Qdrant tests plus one
server-only roundtrip (skipped unless `MINEMOS_QDRANT_URL` is set). Coverage
includes creation and indexing, retrieval, consolidation, deletion, metadata
filters, restart behavior, rebuild after collection loss, ingestion outage
fallback, concurrent updates, incremental upsert discipline, idempotent
sync/retry, contradiction/supersession, checkpoint restore, tenant isolation,
eviction mirroring, API-key non-leakage, hybrid candidate merge, the
Qdrant-backend requirement for hybrid, opt-in RRF ordering, and stale-error
clearing after a successful sync/query.

By default the suite uses persistent embedded mode. Set `MINEMOS_QDRANT_URL`
(and optionally `MINEMOS_QDRANT_API_KEY`) to run the same tests, plus the
server-only roundtrip, against a real Qdrant server.

```bash
.venv/Scripts/python.exe -m pytest tests/test_retrieval_production_mode.py -q
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
.venv/Scripts/python.exe -m benchmarks.qdrant_retrieval_benchmark \
    --sizes 100 300 --include-rrf \
    --output benchmarks/results/qdrant_hybrid_fusion_benchmark.json
```

Results are written to `benchmarks/results/qdrant_integration_benchmark.json`
(embedded mode) or `benchmarks/results/qdrant_integration_benchmark_server.json`
(server mode).  The opt-in RRF run writes a separate
`benchmarks/results/qdrant_hybrid_fusion_benchmark.json` so the historical
dense/hybrid results remain intact. The historical Phase 27 and Phase 31 results
are not modified or reinterpreted.

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

The opt-in RRF run (`--include-rrf`, 100/300 target facts, stored in
`benchmarks/results/qdrant_hybrid_fusion_benchmark.json`) measured MRR 0.775
vs 0.771 at 200 points and 0.750 vs 0.714 at 600 points, with nDCG@5 0.976 vs
0.971 and 0.973 vs 0.944. Recall@1 rose from 0.475 to 0.500 at the larger size;
Recall@5 was 1.000 for every method. BM25 remained 1.000 on that synthetic
corpus. With one corpus, one seed, and 40 queries per size, this is not
evidence of a general ranking improvement; `hybrid_fusion="dense"` remains the
default.
