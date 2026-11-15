# Getting Started

This guide takes you from a checkout to a working memory that absorbs
observations, resolves a changed fact, answers a query, and survives a restart.

## Requirements

- Python 3.12.
- CPU-only PyTorch; no GPU is required.
- Disk space for the virtual environment and the two model downloads.
- Optional: a Qdrant server. The canonical engine does not need one.

Full setup, including the model download step, is in
[Installation](guides/installation.md).

## Create a memory

Run this from the repository root.

```python
from minemos import BlackHoleMemory, Candidate

mem = BlackHoleMemory(capacity=50)
```

`capacity` is the hard limit on stored slots, including superseded history.
`BlackHoleMemory` loads the two local models on first use, so the first call
takes a few seconds.

## Absorb observations

```python
mem.absorb(Candidate("Alice", "city", "New York"), timestamp=10.0)
result = mem.absorb(Candidate("Alice", "city", "Boston"), timestamp=20.0)

print(result["action"])       # temporal_update
print(result["memory_id"])    # stable UUID for the new active slot
```

The second observation contradicts the first. MINEMOS closes the first slot's
validity window at `20.0` and opens a new active slot, keeping both.

`absorb()` accepts three observation forms:

| Form | Example |
|---|---|
| `Candidate` | `Candidate("Alice", "city", "Boston", confidence=0.9, timestamp=20.0)` |
| Tuple | `("Alice", "city", "Boston")` |
| Simple sentence | `"Alice city Boston."` |

A string is split into three whitespace-separated parts. It is a convenience for
already-structured facts, not a natural-language extractor. If you ingest
conversation or document text, extract `(subject, predicate, object)` yourself
and pass structured observations. See
[Data Ingestion](guides/data-ingestion.md).

## Query current state

```python
for row in mem.query("what city is Alice in", top_k=3):
    print(row["text"], row["valid_from"], row["valid_until"])
# Alice city Boston. 20.0 None
```

`query()` ranks by exact dense similarity over canonical state and returns
dictionaries with the slot text, confidence, validity bounds, and `memory_id`.

For an exact attribute lookup, use `recall()`:

```python
for row in mem.recall("Alice", "city"):
    print(row["text"])
# Alice city Boston.
```

## Look back in time

```python
for row in mem.recall("Alice", "city", as_of=15.0):
    print(row["text"], row["valid_from"], row["valid_until"])
# Alice city New York. 10.0 20.0
```

Superseded facts stay queryable. `get_history("Alice", "city")` returns the whole
chain in validity order.

## Persist and restore

```python
mem.save_checkpoint("state/memory.npz", create_backup=True)

restored = BlackHoleMemory.from_checkpoint("state/memory.npz")
print(restored.status()["active_slots"])   # 1
```

`from_checkpoint()` builds a new engine and restores canonical state, the text
map, dense embeddings, the BM25 index, and supersession links. To replace the
state of an existing engine, call `mem.load_checkpoint(path)` instead.

## Choose a retrieval backend

The default is canonical retrieval with no external service:

```python
mem = BlackHoleMemory(capacity=100, retrieval_mode="canonical")
```

To use a Qdrant server as a derived retrieval index:

```python
from minemos import BlackHoleMemory, QdrantConfig

mem = BlackHoleMemory(
    capacity=1000,
    retrieval_backend="qdrant",
    qdrant_config=QdrantConfig(
        url="http://127.0.0.1:6333",
        collection="minemos_memories",
        tenant_id="customer-a",
    ),
)
```

Qdrant accelerates candidate search. It never decides what is remembered,
superseded, or evicted, and every hit is resolved back to a live canonical slot.
See [Qdrant Integration](architecture/qdrant-integration.md).

## Run the tests

```bash
python -m pytest tests/ -q
```

Cache both models first; the suite loads them. Qdrant tests use an embedded
database by default.

## Next steps

- [Python API](python-api.md) — every public engine method and its result shape.
- [Integration](integration.md) — adding memory to a chat application or agent.
- [Memory Lifecycle](architecture/memory-lifecycle.md) — what each absorb
  decision means.
- [Configuration](guides/configuration.md) — capacity, retention, retrieval
  modes, and environment variables.