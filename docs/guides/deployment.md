# Deployment

MINEMOS is a research system. `retrieval_mode="production"` is the strict
retrieval deployment profile; it does not make MINEMOS a validated production
memory service. Before deploying, read the status and limitations in the
[README](../../README.md#status-and-maturity) and
[Research](../research/README.md).

## Deployment profiles

| Profile | When to use | Retrieval |
|---|---|---|
| `canonical` | Development, tests, recovery, installations without an index | Exact in-process scan; Qdrant never constructed |
| `legacy` | Existing installations that have not opted in | Canonical unless a backend is explicitly selected |
| `production` | New deployments that require Qdrant candidate search | Qdrant validated at startup; canonical fallback at runtime |

Select with `retrieval_mode=` or `MINEMOS_RETRIEVAL_MODE`. See
[Configuration](configuration.md) for resolution order.

## Topology

```text
Application process
├── MINEMOS BlackHoleMemory        canonical state and memory decisions
│   ├── SQLite-free .npz checkpoints (authoritative persistence)
│   └── BM25 index (in-process, rebuilt on load)
└── Qdrant connection (derived retrieval index)
    └── collection, tenant-scoped points
```

Qdrant should run as a server in this profile. Embedded mode is for development
and tests and is rejected by production validation.

The process owns one canonical memory. For multiple tenants, use one
`tenant_id` per Qdrant scope and one canonical checkpoint per tenant; do not
share a canonical memory across tenant scopes.

## Production checklist

1. Use Python 3.12 and the pinned requirements.
2. Cache both Hugging Face models in the runtime image or volume.
3. Set `retrieval_mode="production"` (or the environment variable).
4. Provide a server-mode `QDRANT_URL`.
5. Use HTTPS and `QDRANT_API_KEY` for any non-loopback host.
6. Keep `QDRANT_RECREATE=false`.
7. Set an explicit `QDRANT_TENANT_ID` and collection name.
8. Persist canonical checkpoints outside the container filesystem.
9. Monitor `retrieval_stats()`; alert on `state="degraded"` and on a persistent
   `dirty` index.
10. Back up canonical checkpoints, not Qdrant. Qdrant is rebuildable.

## Startup behavior

A misconfigured production deployment fails at construction with
`RetrievalConfigurationError` naming the missing or unsafe setting. This is
intentional: a production profile that cannot reach its required index should not
silently serve from canonical while reporting normal status.

A runtime outage is different. Accepted memory mutations still commit, the index
is marked dirty, queries fall back to canonical, and `retrieval_stats()` reports
`degraded`. A one-time warning is logged. Service recovers after a successful
sync, rebuild, or checkpoint restore.

## Scale and performance expectations

The repository contains controlled benchmarks, not production capacity claims.
Published measurements cover synthetic corpora up to 1,000 target facts in the
Qdrant integration benchmark and storage-only experiments at larger sizes. They
do not establish 100K-scale production quality, latency SLOs, or superiority over
canonical retrieval.

Operational implications observed in the benchmarks:

- Per-mutation index synchronization favors consistency over ingestion
  throughput. Ingestion cost grows with slot count.
- Embedded Qdrant uses brute-force search; server mode uses HNSW.
- Qdrant dense results matched canonical dense results in the measured
  synthetic benchmark because the final ranking is exact and runs in MINEMOS.
- Hybrid retrieval with dense ordering does not change metrics when the final
  re-rank is dense; BM25 expands candidates only.
- Cluster operation, replication, cross-region failover, and authentication
  policy design were not evaluated.

See [Benchmark Methodology](../research/benchmark-methodology.md) for the full
tables and caveats.

## Rollback

`mem.set_retrieval_backend("canonical")` switches the live engine to canonical
retrieval immediately. This is the emergency path in `production` mode. The next
query scans canonical state and does not consult Qdrant. Canonical status is
reported as healthy regardless of earlier index errors.

To rebuild an index after loss or suspected drift, use the retriever's
`rebuild()` through the Qdrant adapter, or restore a checkpoint and let the
restore path force a full sync. See
[Qdrant Integration](../architecture/qdrant-integration.md#failure-isolation-and-recovery).

## What is not covered

- No hosted service, container image, or orchestration manifest is provided.
- No authentication or authorization layer is included around the engine.
- No guarantee of exactly-once ingestion across processes.
- No multi-process writer coordination for a shared checkpoint.
- No validated high-availability Qdrant configuration.

Treat deployment as an engineering integration around an experimental memory
engine, not as a turnkey production package.
