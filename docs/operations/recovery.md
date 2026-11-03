# Recovery

Canonical checkpoints are the source of truth. A Qdrant index, an optimized
snapshot, or an incremental snapshot can be rebuilt from canonical state.

## Save a checkpoint

```python
mem.save_checkpoint("state/memory.npz")
```

Options:

```python
mem.save_checkpoint(
    "state/memory.npz",
    create_backup=True,              # keep the previous archive as .bak
    allowed_base_dir="state",        # confine the path to a directory
)
```

The archive is written to a temporary file and atomically renamed. The previous
valid archive is preserved as `<path>.bak` when backup creation is enabled.
Checksums are embedded, and capacity/dimension bounds are validated.

## Restore a checkpoint

Into an existing engine:

```python
mem.load_checkpoint("state/memory.npz")
```

Or construct directly:

```python
mem = BlackHoleMemory.from_checkpoint(
    "state/memory.npz",
    capacity=100,        # use the capacity recorded for the deployment
    dim=4096,
)
```

`load_checkpoint` accepts `allow_backup_recovery=True` (default) and
`allowed_base_dir`. Loading replaces canonical slots and rebuilds the text map,
dense embeddings, BM25 index, contradiction counters, anchors, and telemetry
from the archive. If Qdrant is active, a forced full index reconcile runs after
the state is installed.

The loaded `MemoryState` carries its own capacity and dimension from the
archive. Construct the engine with matching values so `status()` reports the
same capacity as the restored state.

## What can go wrong

| Error | Cause | Action |
|---|---|---|
| `CheckpointNotFoundError` | Path does not exist | Verify the path; check for a failed save |
| `CheckpointValidationError` | Schema, dimension, capacity, or value violation | Use a valid archive; do not edit archives manually |
| `CheckpointCorruptedError` | Truncated file or checksum mismatch | Restore the `.bak` archive if it is valid |
| `CheckpointPathSecurityError` | Path escapes `allowed_base_dir` | Use a path inside the allowed directory |
| `ValueError` from load | Archive is incompatible with the engine assumptions | Check schema version and dimensions |

v1–v3 archives remain readable. They receive deterministic fallback
`memory_id` values because stable IDs were introduced in schema v4.

## Backup limitation

The `.bak` archive contains the state from before the most recent save. If a
slot was deleted and a new checkpoint saved, the `.bak` still contains that
slot. Restoring it recovers deleted information. This is documented behavior,
not secure erasure. Destroy backups explicitly if deletion must be permanent.

## Qdrant recovery

If Qdrant is lost or inconsistent:

1. Canonical memory is unaffected. Confirm `retrieval_stats()` reports
   `degraded` with a canonical fallback.
2. Restore canonical state from a checkpoint if needed.
3. Rebuild the tenant's points from canonical slots through the Qdrant
   retriever's `rebuild()`, or force a sync after restore.
4. Confirm `state` returns to `ready` and `degraded` is false.

Never restore Qdrant as an authority over canonical memory. A Qdrant snapshot
can speed recovery, but it is not required and cannot override canonical state.

## Rollback to canonical

```python
mem.set_retrieval_backend("canonical")
```

This takes effect on the next query. Canonical retrieval does not consult
Qdrant, snapshot state, or index metadata. It is the emergency rollback in
`production` mode and the only allowed backend in `canonical` mode.

## Multi-process recovery

One engine owns one canonical checkpoint. Do not run multiple writer processes
against the same archive and expect coordinated updates. Partition memory by
tenant or process, or serialize writes in the application layer.
