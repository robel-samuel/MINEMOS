# Monitoring

MINEMOS exposes status through methods on the engine. There is no bundled
metrics exporter, dashboard, or alerting integration.

## Memory status

```python
status = mem.status()
```

Returns:

| Field | Meaning |
|---|---|
| `capacity` | Configured engine capacity |
| `total_slots` | All stored slots, active and superseded |
| `active_slots` | Slots with `valid_until is None` |
| `provisional_slots` | Active slots with `update_count <= 1` |
| `consolidated_slots` | Active slots with `update_count > 1` |
| `superseded_slots` | Inactive historical slots |
| `total_absorbed` | Observations processed |
| `total_updates` | Consolidated observations |
| `total_inserts` | Inserted slots |
| `total_temporal_updates` | Supersessions |
| `total_evictions` | Evictions |
| `total_nli_calls` | Verifier invocations |
| `total_value_verifications` | Value-equivalence verifier calls |
| `footprint_bytes` | Process memory footprint estimate for slot arrays |

## Retrieval status

```python
stats = mem.retrieval_stats()
```

Top-level fields include `backend`, `mode`, `active`, `initialized`, `degraded`,
`state`, `strategy`, `manager`, and `published_version`. When a retriever is
initialized, the payload also includes backend-specific counters such as
`errors`, `last_error`, `searches`, and index-sync details.

`state` is one of:

| State | Meaning |
|---|---|
| `canonical` | Canonical retrieval is active by configuration |
| `not_initialized` | A non-canonical backend is selected but not built yet |
| `ready` | The selected backend answered the last retrieval |
| `degraded` | The backend is configured but unavailable; queries fall back to canonical |

`degraded` is derived from the current index state (`dirty` or `last_error`), not
from cumulative historical errors. It clears after a successful sync or query.
A canonical rollback always reports `canonical`, `degraded=False`.

## What to watch

- `state == "degraded"` for longer than a transient outage.
- `manager.dirty == True` persisting across operations.
- Repeated `last_error` values naming the same connection or collection.
- `memory.status()` showing sustained eviction growth with a capacity that is
  too small for the workload.
- Ingestion latency increasing with slot count when Qdrant synchronization is
  enabled. This is expected with the consistency-first default.

## Logs

In `production` mode:

- the first index-write failure logs a warning that canonical memory is intact
  and queries fall back to canonical;
- the first query-time fallback logs a warning.

Later failures in the same process do not repeat these warnings; inspect
`retrieval_stats()` for current state. API keys are excluded from stats,
`repr`, and logs.

## Capacity and footprint

`capacity` is a hard limit on stored slots, including superseded history until
eviction. A stream with frequent supersession therefore needs headroom beyond
the number of currently active facts. Tiered retention protects consolidated
slots but caps them at 80% of capacity.

The footprint value is an estimate of stored slot arrays, not total process RSS.
Qdrant memory and model memory are outside it.
