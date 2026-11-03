# Retrieval

Retrieval answers a natural-language query by returning ranked memory rows. The
canonical path is the reference implementation; optional backends accelerate or
relocate candidate search but never change the memory decisions that produced
the state.

## Query API

```python
rows = mem.query(
    "where does alice work",
    top_k=5,
    as_of=None,          # point-in-time query when set
    active_only=True,    # ignored when as_of is set
    filters=None,        # subject, predicate, min_confidence, ...
    hybrid=False,        # requires the qdrant backend
    fusion=None,         # "dense" (default) or opt-in "rrf"
)
```

Each row contains:

```text
slot_index, slot_id, similarity, subject, predicate, text,
confidence, update_count, is_active, valid_from, valid_until
```

`similarity` is the exact dot product between the query embedding and the slot's
dense embedding. Embeddings are L2-normalized, so this is cosine similarity.

For exact attribute lookup, use `recall(subject, predicate=None, as_of=None,
active_only=True)`. It returns rows in `valid_from` order without similarity
ranking. `get_history(subject, predicate=None)` is `recall` with
`active_only=False`.

## Canonical retrieval

The default path scans the eligible canonical slots, computes exact dense
similarity, sorts, and slices to `top_k`.

Eligibility:

- with `as_of`: `valid_from <= as_of < valid_until` (or an open window);
- otherwise `active_only=True` returns only active slots;
- `filters` are applied against canonical slot fields before ranking.

Canonical retrieval is deliberately simple: one exact pass over the live state.
It is the fallback for every optional backend and the default for existing
installations.

## Encoders

| Purpose | Encoder | Dimension |
|---|---|---|
| Slot key/value content and consolidation | `FieldAwareLexicalEncoder` (field-prefixed token hashing) | 4096 by default |
| Addressing, query similarity, Qdrant vectors | `all-MiniLM-L6-v2` sentence embeddings | 384 |

No second embedding pipeline is introduced by hybrid retrieval.

## Qdrant-backed retrieval

With the Qdrant backend selected, MINEMOS sends the query embedding to Qdrant
with tenant, temporal, and metadata filters applied in the vector query. Qdrant
returns candidate IDs; MINEMOS resolves each ID against canonical slots, drops
unknown or stale IDs, re-applies filters exactly, and re-ranks by exact dense
similarity.

The index is derived and rebuildable. If it is unavailable or dirty, queries
fall back to canonical retrieval and `retrieval_stats()` reports `degraded`.
See [Qdrant Integration](qdrant-integration.md) for configuration, lifecycle,
tenancy, and failure isolation.

## Hybrid candidate expansion

`hybrid=True` requires the Qdrant backend. Qdrant dense candidates are merged
with candidates from MINEMOS's in-process BM25 index. BM25 can add slots that
dense search did not return.

The default ordering is `dense`: the merged union is re-ranked by exact dense
similarity, so BM25 contributes candidate membership, not ordering. This is why
hybrid can equal dense metrics on corpora where BM25 would have ranked
differently.

`fusion="rrf"` is an opt-in experiment that orders the same candidate union by
reciprocal rank (`rrf_k=60` by default). It changes ordering only: membership
and the reported exact similarity are unchanged. It is not enabled by default
and no general ranking improvement is claimed. See
[Benchmark Methodology](../research/benchmark-methodology.md) for the measured
synthetic result.

## Metadata filters

```python
rows = mem.query("project status", filters={
    "subject": "alice",
    "predicate": "works_on",
    "min_confidence": 0.6,
    "min_update_count": 2,
    "memory_ids": ["<uuid>"],
})
```

- Qdrant receives the filters as payload conditions when it is the backend.
- MINEMOS re-applies the filters against canonical state, so Qdrant cannot widen
  or mis-rank the final result.
- Unknown filter keys raise `ValueError`.
- With the canonical backend, the same filters are applied in-process.
- Metadata filters are not supported by the optimized or incremental backends;
  those queries hand off to canonical.

## Optional in-process backends

`retrieval_backend="optimized"` and `"incremental"` use published snapshots of
canonical state:

- `optimized`: rebuilds a snapshot and serves an exact vectorized scan.
- `incremental`: applies copy-on-write array deltas to an existing snapshot.

Both preserve exact dense ranking parity with canonical. A snapshot is
stale-but-coherent: it may omit mutations that happened after capture, but every
row belongs to one engine state. Contention is an expected hand-off to
canonical, not a correctness failure. Configuration lives in
`phase2/retrieval/config.py`.

These backends are opt-in and reversible at runtime with
`mem.set_retrieval_backend("canonical")`. Canonical is always the default.

## Backend selection

```python
mem.retrieval_backend   # "canonical", "optimized", "incremental", "qdrant"
mem.retrieval_mode      # "legacy", "production", "canonical"
mem.retrieval_stats()   # backend, mode, state, degraded, manager details
mem.set_retrieval_backend("canonical")
```

Mode resolution order is constructor argument, `MINEMOS_RETRIEVAL_MODE`, then
`RetrievalConfig.mode`, defaulting to `legacy`. The `production` mode requires
Qdrant and validates its configuration at startup; the `canonical` mode never
contacts Qdrant. Details are in
[Configuration](../guides/configuration.md) and
[Deployment](../guides/deployment.md).

## What retrieval does not do

- It does not decide whether a fact should be remembered. That happens during
  `absorb()`.
- It does not use lexical scoring in the canonical path.
- It does not normalize BM25 scores against dense scores. Rank fusion was added
  as the safer experiment because those scales are not comparable.
- It does not guarantee that the top-ranked row is the correct fact. End-to-end
  correctness also depends on consolidation, verification, and retention.
