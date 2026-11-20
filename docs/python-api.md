# Python API

MINEMOS is a Python library. The public surface is small: one engine class, one
observation type, and dictionaries for results. This page documents every
supported call.

```python
from minemos import BlackHoleMemory, Candidate
```

`minemos` is the supported import path for applications. The implementation
lives in `minemos_engine/`; those module paths are stable within a release but are not
the integration contract.

## `BlackHoleMemory`

```python
BlackHoleMemory(
    capacity=100,
    dim=4096,
    retention_policy="tiered",
    consolidation_mode="hybrid",
    retrieval_mode=None,
    retrieval_backend=None,
    qdrant_config=None,
    exclusive_predicates=None,
    multi_valued_predicates=None,
    strict_entity_identity=False,
)
```

| Parameter | Default | Meaning |
|---|---|---|
| `capacity` | `100` | Maximum slots, including superseded history |
| `dim` | `4096` | Lexical key/value dimension |
| `retention_policy` | `"tiered"` | `tiered`, `fifo`, `recency`, `utility`, `verification_aware` |
| `consolidation_mode` | `"hybrid"` | `hybrid`, `bounded`, `anchor`, `baseline` |
| `retrieval_mode` | `None` | `legacy`, `production`, or `canonical` |
| `retrieval_backend` | `None` | `canonical`, `optimized`, `incremental`, or `qdrant` |
| `qdrant_config` | `None` | `QdrantConfig`, dict, or environment lookup |
| `exclusive_predicates` | `None` | Predicates that may hold only one active value |
| `multi_valued_predicates` | `None` | Predicates that may hold several active values |
| `strict_entity_identity` | `False` | Treat one-edit name differences as uncertain identity |

Policy overrides apply to the process and are not written into checkpoints.
Predicate exclusivity is never inferred from the predicate name alone: an
unlisted predicate is treated as unknown, not exclusive. The full parameter and
environment reference is in [Configuration](guides/configuration.md).

One engine owns one canonical memory. Use one `BlackHoleMemory` per tenant and do
not share a checkpoint between writer processes.

## Absorbing observations

```python
result = mem.absorb(
    Candidate("Alice", "city", "Boston", confidence=0.9),
    timestamp=1767312000.0,
)
```

`absorb(observation, timestamp=None, confidence=1.0)` accepts a `Candidate`, a
`(subject, predicate, object)` tuple, or a three-part string. A `Candidate`
carries its own `confidence` and `timestamp`; the explicit arguments apply to
tuples and strings.

The result always contains `action`, `slot_index`, `slot_id`, `text`, and
`valid_from`. Depending on the decision it also contains `memory_id`,
`similarity`, `decision`, `update_count`, `valid_until`, `superseded_slot_id`,
`evicted_slot_id`, or `consolidation_class`.

| `action` | Meaning |
|---|---|
| `insert` | New independent fact added |
| `evict_insert` | New fact added after capacity eviction |
| `consolidate` | Same fact restated; the stored wording/vector was updated |
| `temporal_update` | Value changed; the old slot's window closed and a successor opened |
| `historical_update` | A past-dated observation that matches an existing slot |
| `historical_observation` | A past-dated observation recorded as a closed interval |
| `historical_rejected` | A past-dated observation that would contradict an open fact |

`result["memory_id"]` is the stable UUID of the affected slot. It survives
checkpoint reloads and is the identifier used by the optional Qdrant index.

## Querying

```python
rows = mem.query(
    "what city is Alice in",
    top_k=5,
    as_of=None,
    active_only=True,
    filters={"subject": "alice", "min_confidence": 0.6},
    hybrid=False,
    fusion=None,
)
```

Each row is a dictionary:

```text
slot_index, slot_id, memory_id, similarity, subject, predicate,
original_predicate, text, confidence, update_count, is_active,
valid_from, valid_until
```

`similarity` is the exact cosine similarity between the query embedding and the
slot's dense embedding. Filters accept `subject`, `predicate`, `min_confidence`,
`min_update_count`, and `memory_ids`. Unknown filter keys raise `ValueError`.

`hybrid=True` requires the Qdrant backend. `fusion="rrf"` is an opt-in ordering
mode that also requires `hybrid=True`. Both are documented in
[Retrieval](architecture/retrieval.md#hybrid-candidate-expansion).

## Answer-aware queries

`query()` ranks memory. `eligible_query()` additionally decides whether the
returned rows are evidence for the question, and reports when they are not.

```python
outcome = mem.eligible_query("what city is Alice in", top_k=3)
print(outcome["status"], outcome["reason"])
print(outcome["evidence"]["state"])
print(outcome["temporal"]["answer_is_settled"])
```

| `status` | Meaning |
|---|---|
| `answered` | The subject and attribute were verified and evidence was sufficient |
| `abstained` | The query could not be answered; `reason` explains why |
| `unverified` | The query named no resolvable entity; ungated rows are diagnostic |
| `disabled` | No eligibility decision was requested or possible |

Abstention reasons include `subject_not_found`, `attribute_not_stored`,
`subject_ambiguous`, `evidence_insufficient`, `document_scope_required`,
`scope_not_found`, `scope_not_discovered`, and `low_similarity`.

The outcome also contains `rows`, `focus`, `query_attributes`, `top_similarity`,
`eligible_slots`, `subject_matched_slots`, `document_scoped_slots`, `evidence`,
`temporal`, and—when scopes were used—`scopes`, `scopes_present`, and
`scopes_missing`.

Optional arguments:

| Argument | Purpose |
|---|---|
| `subject` | Explicit entity assertion; must match a stored subject |
| `scope_subjects` | Joint search across explicit document/conversation scopes |
| `discover_scopes` | Discover document scopes from the query content |
| `context_text` | Previous-turn text for a follow-up; requires an explicit scope |
| `current_turn` | Caller's own message to exclude if it was already ingested |
| `min_similarity` | Optional similarity floor; no threshold is applied by default |

### Presentation contract

`answer_contract()` converts an outcome into a presentation decision:

```python
from minemos import answer_contract

presentation = answer_contract(
    outcome["status"], outcome["reason"],
    outcome["temporal"], len(outcome["rows"]),
)
print(presentation["answer_status"])
# settled | contested | ambiguous | abstained | unverified | empty | ungated

if not presentation["safe_to_display_as_settled"]:
    ...  # present the rows as uncertain, or ask for clarification
```

`answer_is_settled` is true only when the eligibility gate answered the question
and no unresolved temporal conflict was found among the returned facts.

## Exact recall and history

```python
mem.recall("Alice", "city")                    # active values
mem.recall("Alice", "city", as_of=15.0)        # point-in-time value
mem.get_history("Alice", "city")               # whole chain, oldest first
```

`recall()` filters by normalized subject (case, width, and edge punctuation
insensitive) and optional predicate, returns rows in `valid_from` order, and
does not rank by similarity.

## Deletion and redaction

```python
mem.delete_slot(slot_id)                 # remove one slot object
mem.delete_memory(memory_id)             # remove by stable UUID
mem.purge_subject("Alice")               # remove every slot for a subject

mem.redact_slot(slot_id)                 # destroy content, keep a tombstone
mem.redact_memory(memory_id, reason="gdpr-request")
mem.redact_subject("Alice")
mem.purge_redacted()                     # drop tombstones permanently
```

Deletion removes the slot and repairs supersession links. Redaction zeroes the
key/value vectors and dense embedding, clears the text and labels, drops the
BM25 document and contradiction anchors, and excludes the slot from every read
path, checkpoint payload, and Qdrant index. The tombstone keeps the chain
intact. A later save cannot reintroduce the content.

Redaction does not rewrite backups that already contain the content. See
[Recovery](operations/recovery.md#privacy-deletion-and-redaction).

## Checkpoints

```python
mem.save_checkpoint("state/memory.npz", create_backup=True,
                    allowed_base_dir="state")
mem.load_checkpoint("state/memory.npz")

same = BlackHoleMemory.from_checkpoint("state/memory.npz", capacity=100)
```

Archives are written atomically and checksummed. `create_backup=True` keeps the
previous archive as `<path>.bak`. `allowed_base_dir` confines the path.
Loading replaces canonical state and rebuilds the text map, dense embeddings,
BM25 index, anchors, and telemetry. If Qdrant is active, a forced index
reconcile runs after the state is installed.

The archive records capacity and dimension. Construct the engine with matching
values so `status()` reports the same capacity as the restored state.

## Diagnostics

```python
mem.status()            # slot counts, telemetry, footprint
mem.retrieval_stats()   # backend, mode, state, degraded, last error
mem.retrieval_readiness()  # non-blocking readiness view
mem.close()             # release the Qdrant client, if any
```

`status()` reports `capacity`, `total_slots`, `live_slots`, `active_slots`,
`provisional_slots`, `consolidated_slots`, `superseded_slots`,
`redacted_slots`, absorb/eviction counters, and `footprint_bytes`.
`retrieval_stats()` reports `backend`, `mode`, `state`
(`canonical`, `not_initialized`, `ready`, `degraded`), `initialized`, and
`degraded`.

## `MemorySlot`

Slots are reachable through `mem.memory.slots` for inspection. Do not mutate
their fields directly: out-of-band writes bypass cache invalidation and the
index reconciliation contract.

```text
key, value, confidence, timestamp, created_at, memory_id, update_count,
debug_subject, debug_predicate, valid_from, valid_until, text, dense_emb,
contra_count, supersedes, superseded_by, latest_predicate, latest_object,
redacted
```

`key` is frozen at creation; `value` changes on consolidation. `is_active` is
true while `valid_until` is `None`. Supersession links are process-local `id()`
values; use `memory_id` for anything that must outlive the process.

## Semantic resolution (separate package)

The resolution layer lives in `minemos.resolution`, not in the engine package.
It is candidate-only: `resolve()` returns a record and never writes memory.

```python
from minemos.resolution import (
    ResolutionConfig, Resolver, ShadowRunner, SourceMessage,
)

resolver = Resolver(ResolutionConfig(mode="shadow"))  # default
message = SourceMessage(
    conversation_id="conv-1",
    message_id="m-1",
    text="My sister now works at Acme and I live in Berlin.",
    speaker="owner",
    timestamp=1730000000.0,
    order=0,
)
result = resolver.resolve(message)
record = result.record

record.facts        # FactCandidate: subject, predicate, value, polarity, ...
record.attributions # speaker, subject, scope, status, evidence
record.relationships  # duplicate | temporal_update | historical_observation |
                      # distinct_fact | unresolved, with evidence
record.normalized   # original text, normalized text, reversible edits
record.language     # primary, confidence, mixed, spans
record.abstentions  # explicit reasons the layer declined
```

```python
runner = ShadowRunner()            # shadow mode only
runner.run(message, baseline_facts=())
runner.run_conversation(messages)  # shared identity session per conversation
runner.summary()                   # counters only; no message text
runner.report.log_line()           # one content-free line for application logs
```

`Resolver.resolve_conversation()` resolves an ordered conversation and keeps a
per-conversation identity session so pronouns and repeated name variants can
resolve. Every record is checked against the contract invariants in
`minemos/resolution/contracts.py` before it is returned; a violation raises
`ResolutionContractError`. The JSON form from `record.to_dict()` can be checked
against the published schema with `minemos.resolution.validate_with_schema()`.

The engine bridge is explicit and opt-in. `minemos.resolution.absorb_record()`
only sends candidates when the resolver is in `active` mode, the subject is
resolved, the fact is positively asserted, and it is not classified
`duplicate` or `unresolved`. Even then the engine owns consolidation,
contradiction, supersession, temporal intervals, retention and eviction.

`native_observations()` converts a record into values the native contract
already accepts — `Candidate` objects for the engine, observation mappings for
the service payload — plus an explicit rejection reason for every fact that
was not converted:

```python
from minemos.resolution import ResolutionConfig, native_observations

conversion = native_observations(
    record,
    ResolutionConfig(mode="active"),
    segment_timestamp=segment.timestamp,      # original conversation time
    ingestion_timestamp=run_started_at,       # stable per ingestion run
)
for item in conversion.observations:
    engine.absorb(item.as_candidate())
conversion.rejected        # [RejectedFact(fact_id, reason), ...]
conversion.to_dict()       # content-free: ids, timestamp source, reasons
```

Timestamps follow a fixed fallback order (semantic, segment, ingestion) and
are never `0.0`; a fact with no valid timestamp is rejected as
`invalid_timestamp`. Ingestion ids are deterministic hashes of source
identity plus fact content, so replaying a message does not duplicate it.

The ingestion step of that fallback is gated by `ingestion_fallback`
(default `True` for backward compatibility). Callers that ingest archived
conversations should pass `ingestion_fallback=False`: a fact with no event
time is then rejected as `timestamp_unavailable` rather than being stamped
with ingestion time, because a wall-clock stamp can never be superseded by a
genuinely newer archived message. Decide the policy per conversation — all
timed (convert), mixed (convert timed messages only), untimed (abstain, or
opt in explicitly with `ingestion_fallback=True`).

There is no network client in this package and no default endpoint. Diagnostics
carry counters, stage names, hashes and ids — not message text or fact values.

## Semantic ingestion (opt-in)

`minemos.semantic` is the supported way to turn raw conversational messages into
canonical memory. It composes the pieces above; it adds no second memory
authority. A complete runnable example lives in
[`examples/semantic_ingestion.py`](../examples/semantic_ingestion.py).

```text
raw messages
  → segmentation + four-stage resolution   (minemos.resolution.ingest_conversation)
  → eligibility + native fields            (minemos.resolution.native_observations)
  → your sink                              (engine or service pool)
```

Every write requires an explicit `ResolutionConfig(mode="active")`. In
`shadow` (the default) or `disabled` mode `SemanticIngestor` raises
`SemanticIngestionError` instead of silently doing nothing, so a misconfigured
host cannot believe it wrote memory it did not write. Passing `sink=None` runs
the same pipeline as a dry run and writes nothing.

### Embedded engine

```python
from minemos import BlackHoleMemory, EngineSink, ResolutionConfig
from minemos import SemanticIngestor, SourceMessage

mem = BlackHoleMemory(capacity=10_000)
ingestor = SemanticIngestor(
    EngineSink(mem),
    config=ResolutionConfig(mode="active"),
    ingestion_timestamp=1730000000.0,   # stable replay stamp
)
report = ingestor.ingest_conversation([
    SourceMessage(conversation_id="conv-1", message_id="m-1",
                  text="I work at Acme Corp.", speaker="user",
                  timestamp=1730000000.0, order=0),
])

report.absorbed          # observations handed to the engine
report.rejected          # eligible-fact rejections (negation, ambiguity, ...)
report.reasons()         # {"non_asserted_polarity": 1, ...}
report.summary()         # counters and ids only; never message text
mem.recall("user")       # canonical rows written by the engine
```

`EngineSink` calls `BlackHoleMemory.absorb`, so consolidation, contradiction,
supersession, retention and eviction stay engine decisions. It has no
provenance ledger; replay deduplication there is the engine's consolidation
decision.

### Service pool (recommended for deployments)

```python
from minemos import ResolutionConfig, SemanticIngestor, ServiceSink
from minemos_service.engine_pool import EnginePool

ingestor = SemanticIngestor(
    ServiceSink(pool, "tenant-a"),        # EnginePool instance
    config=ResolutionConfig(mode="active"),
)
report = ingestor.ingest_conversation(messages)

pool.recall("tenant-a", "user")
pool.verify_provenance("tenant-a")        # {"ok": True, ...}
```

`ServiceSink` forwards to `EnginePool.absorb`, which records provenance,
deduplicates by deterministic ingestion id and persists the checkpoint. The
service, not the semantic layer, owns the canonical lifecycle.

The service HTTP boundary remains available: convert with
`native_observations(...)` and post
`observation.to_service_observation()` to `POST /v1/ingest`. That route applies
the configured credential policy to observation content and never logs request
bodies. A host that calls `ServiceSink` directly is already inside the trusted
service process and must apply its own input policy.

### Timestamps

`conversation_timestamp_mode(messages)` reports `timed`, `mixed`, `untimed` or
`empty`. The ingestor follows this policy per conversation:

- **timed** — every message is converted with its explicit event time.
- **mixed** — untimed messages cannot be placed on the event timeline, so their
  facts are rejected as `timestamp_unavailable`.
- **untimed** — facts are abstained unless the host passes
  `allow_ingestion_timestamp=True`; only then does ingestion time stand in.

### Replay and restart

Ingestion ids are deterministic, so a replay of the same messages is safe:

```python
replay = ingestor.ingest_conversation(messages)   # after a crash
replay.absorbed == 0                              # service sink skips duplicates
replay.skipped == len(report.ingestion_ids())
```

`ServiceSink` deduplicates durably through the provenance ledger across process
restarts. `EngineSink` callers can pass `known_ingestion_ids=...` to skip ids
they already applied; that set is caller-owned and grows with every id you keep,
so long-lived hosts should prune or cap it. Restart persistence is the engine's:
checkpoint, then construct a new `BlackHoleMemory` / `EnginePool` pointed at the
same directory.

### Failures and abstention

- A message whose resolution raises is counted in `failed_messages` and skipped;
  the remaining messages are still processed. An oversized message refused by
  `IngestionPolicy(overflow="reject")` raises `MessageTooLongError` before any
  write.
- A single failed sink call is counted in `failed_observations`; canonical
  memory is never corrupted by one failure.
- Negated facts, hypotheticals, unresolved subjects and duplicate facts are
  rejected with an explicit reason (see `NativeConversion.decisions()`).
- One `SemanticIngestor` owns one stateful resolver and is **not thread-safe**.
  Use one ingestor per conversation stream, or guard it with a lock; separate
  ingestors may run concurrently.

### Activation and quality

Semantic ingestion stays **opt-in** until quality and safety requirements for a
deployment are met. The extraction layer is heuristic and has not been
validated against independent human labels; treat `report.reasons()` as
diagnostics, not as an accuracy measurement. No new dependency is required —
the layer uses only the existing package.

