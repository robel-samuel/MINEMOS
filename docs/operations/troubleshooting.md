# Troubleshooting

## `ModuleNotFoundError: No module named 'transformers'`

Commands are running outside the project virtual environment. Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Then run commands from the repository root so `phase2` is importable.

## Model download or loading failures

- Run the model download step in [Installation](../guides/installation.md)
  while online.
- Check `HF_HOME` if models were cached elsewhere.
- Set `TRANSFORMERS_OFFLINE=1` only after the cache is populated.
- Corporate proxies may block Hugging Face; configure the proxy before the
  download step.

## `RetrievalConfigurationError`

This is raised at construction in `retrieval_mode="production"`. Common causes:

| Message theme | Fix |
|---|---|
| Missing `QDRANT_URL` | Set a server URL or pass `QdrantConfig(url=...)` |
| Embedded path rejected | Do not use `QDRANT_PATH` in production; use a server |
| URL has no scheme or host | Use `http://host:port` or `https://host:port` |
| HTTPS required | Use HTTPS or set `QDRANT_ALLOW_INSECURE=true` on a trusted network |
| API key required | Set `QDRANT_API_KEY` for non-loopback hosts |
| Recreate rejected | Set `QDRANT_RECREATE=false` |

Configuration errors fail startup deliberately. They are not runtime outages
and do not silently downgrade to canonical.

## `ValueError: hybrid retrieval requires the qdrant backend`

`hybrid=True` is only supported with Qdrant. Either select the Qdrant backend or
omit `hybrid`.

## `ValueError: fusion requires hybrid=True`

`fusion="rrf"` is only meaningful when BM25 candidates are merged. Call
`query(..., hybrid=True, fusion="rrf")` or omit `fusion`.

## `state` reports `degraded`

The configured index is unavailable or dirty, and queries are being answered by
canonical retrieval.

1. Inspect `mem.retrieval_stats()` for `last_error` and `manager.dirty`.
2. Check that the Qdrant server is reachable from the process.
3. Verify the collection and tenant exist.
4. Rebuild or resync the index if it was lost.
5. As an immediate mitigation, `mem.set_retrieval_backend("canonical")`.

Accepted memory mutations are not rolled back because the index failed.
Canonical state remains valid.

## `state` stays `not_initialized`

A non-canonical backend is selected but no retriever has been built yet. This is
normal before the first query or write-path sync. If it persists after a query,
the retriever could not be constructed; check the Qdrant configuration.

## Checkpoint load failures

See [Recovery](recovery.md#what-can-go-wrong). Do not edit `.npz` archives by
hand. If the primary archive is corrupt, test the `.bak` archive before loading
it.

## High ingestion latency

With Qdrant enabled, every accepted mutation triggers an index reconciliation.
This is the consistency-first default. Ingestion cost grows with slot count, as
shown in the synthetic benchmark. Alternatives:

- Use the canonical backend when an index is not required.
- Ingest in a controlled maintenance window.
- Batch at the application layer and accept that the index reflects only
  committed canonical mutations.

There is no built-in bulk-ingest API.

## Memory fills quickly

Capacity counts active and superseded slots until eviction. Frequent
supersession consumes capacity. Increase `capacity` for the workload or reduce
history retention requirements. `status()` distinguishes provisional,
consolidated, and superseded slots.

## Duplicate-looking facts

Consolidation depends on the NLI verifier and entity/attribute rules. A
high-lexical-variation paraphrase may be rejected and inserted separately. This
is a measured limitation of the current verifier, not a configuration error.
Inspect `absorb()` actions and the stored text to confirm.

## Qdrant embedded database errors

Embedded mode is intended for development and tests. Use one process per
embedded path, and do not point production mode at it. For concurrent
deployments, use a Qdrant server.

## Tests fail only in some runs

A full-suite run under heavy load can be sensitive to timing. Re-run the failing
test in isolation; if it passes consistently alone, treat it as load-sensitive
rather than a deterministic defect, and report the conditions. Do not weaken
assertions to hide a reproducible failure.
