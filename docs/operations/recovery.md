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
    capacity=100,        # use the capacity recorded for this memory
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

v1–v4 archives remain readable. They receive deterministic fallback
`memory_id` values because stable IDs were introduced in schema v4. Schema v5
adds the privacy-redaction tombstone flag; older archives load with every slot
marked not redacted.

## Privacy deletion and redaction

Two distinct operations are available:

| Operation | Effect |
|---|---|
| `delete_slot(slot_id)` / `purge_subject(subject)` | Removes the slot(s) entirely and repairs supersession chains |
| `redact_slot(slot_id)` / `redact_subject(subject)` | Destroys the content and keeps a chain-safe tombstone |

Redaction overwrites the key/value vectors and dense embedding with zeros,
clears the text and subject/predicate labels, drops contradiction anchors and
the BM25 document, and excludes the slot from query, recall, history, invariant
reporting, and the Qdrant index. The checkpoint writes only the tombstone, so a
later save does not reintroduce the content. `purge_redacted()` permanently
removes the tombstones when they are no longer needed.

Redaction cannot rewrite archives or `.bak` files that already contain the
content. Destroy those files explicitly if erasure must cover historical
backups.

## Backup behavior

The `.bak` archive contains the state from before the most recent save. If a
slot was deleted and a new checkpoint saved, the `.bak` still contains that
slot. Restoring it recovers deleted information. This is documented behavior,
not secure erasure. Destroy backups explicitly if deletion must be permanent.

## Restore behavior

- Saving with `create_backup=True` keeps the previous archive as `<path>.bak`.
- A truncated or corrupt primary raises `CheckpointCorruptedError` unless
  `allow_backup_recovery=True` is set and a valid backup exists.
- Loading a path outside `allowed_base_dir` raises `CheckpointPathSecurityError`.
- A redacted slot stays redacted after save and reload. The archive stores an
  empty text with the tombstone flag, and no redacted value remains in it.
- A `.bak` written before a redaction still contains the redacted content.
  Restoring it brings that content back.
- `purge_redacted()` removes the tombstone; the purged memory does not return
  after the next save and load.
- With Qdrant active, loading a checkpoint triggers a forced reconcile. Point
  ids equal the canonical `memory_id` set and stored texts equal the originals.

## Qdrant recovery

If Qdrant is lost or inconsistent:

1. Canonical memory is unaffected. Confirm `retrieval_stats()` reports
   `degraded` with a canonical fallback.
2. Restore canonical state from a checkpoint if needed.
3. Call `mem.sync_retrieval_after_restore()` (or reload a checkpoint) to force
   a full reconcile of this tenant's points from canonical state.
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
