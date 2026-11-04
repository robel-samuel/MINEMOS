# Real-Data Ingestion

The evaluation package (`evaluation/`) ingests naturally occurring material into
an isolated MINEMOS store. It is separate from the service and from every
research benchmark. Nothing in this package is part of the canonical engine API.

## What the engine accepts

The engine stores observations, not documents. Ingestion therefore has two
explicit stages:

1. **Parse** a source file into boundary-preserving segments.
2. **Extract** observations from each segment, either verbatim (one memory per
   segment) or from an authorized upstream structured extractor.

No template heuristic is presented as general-purpose fact extraction.

## Source files and boundaries

Sources are UTF-8 JSONL files. Every segment keeps its `source_id`,
`document_id`/`conversation_id`, `segment_index`, character offsets, optional
speaker, and original timestamp. A conversation is never flattened into an
unstructured bag of facts.

| `--source-type` | File shape |
|---|---|
| `conversation` | Lines with `conversation_id` and either `messages[]` (`content`/`text`, `role`/`speaker`, `timestamp`) or a single message per line |
| `document` | Lines with `document_id`/`id` and `text`, or an explicit `segments[]` list |
| `notes`, `journal`, `technical_doc`, `research_doc`, `project_history`, `other` | Prose treated as a document; paragraphs and then sentences are split at ≤2000 characters |
| `observation` | Lines with `subject`, `predicate`, `object`, optional `confidence`/`timestamp` |

Conversation example:

```jsonl
{"conversation_id":"conv-2026-01","timestamp":1767225600,"messages":[{"role":"user","content":"I moved to Berlin in March.","timestamp":1767225600},{"role":"assistant","content":"Noted."}]}
{"conversation_id":"conv-2026-01","timestamp":1767229200,"messages":[{"role":"user","content":"I moved to Munich last week.","timestamp":1767229200}]}
```

Observation example:

```jsonl
{"subject":"user","predicate":"city","object":"Berlin","confidence":0.95,"timestamp":1767225600}
{"subject":"user","predicate":"city","object":"Munich","confidence":0.95,"timestamp":1767312000}
```

### Timestamps

- `original_timestamp` is a Unix float taken from the source when present.
- Missing timestamps are allowed and counted in dry runs; they are not invented
  from the file name or ingestion order.
- `ingestion_timestamp` is recorded separately by the store for every
  observation. Temporal evaluation can then distinguish source time from
  arrival time.

### Provenance fields

Every ingested observation records `source_id`, `source_type`, `document_id`,
`segment_index`, `segment_key` (`source_id:document_id:segment_index`),
`ingestion_timestamp`, and content/fact digests. The provenance ledger contains
digests and identifiers only — never source text.

## Structured extraction

Verbatim mode writes one segment per memory. Structured mode requires an
observation mapping aligned by `segment_key`:

```jsonl
{"segment_key":"conv-2026-01:conv-2026-01:2","subject":"user","predicate":"city","object":"Munich"}
```

```bash
python -m evaluation.ingest ingest --root evaluation-data \
  --source conversations.jsonl --source-type conversation \
  --source-id conv-2026-01 --mode structured \
  --observations extracted.jsonl
```

Structured mode is applied before the engine sees anything, so it is the
authorized point at which an upstream extractor may be introduced. The mapping
must cover the segments it claims; a structured file with no observations is
rejected.

## Ingestion commands

```bash
# Create an isolated store (refuses a non-empty directory without a manifest).
python -m evaluation.ingest init --root evaluation-data --tenant evaluation

# Validate parsing without loading models or writing state.
python -m evaluation.ingest ingest --root evaluation-data \
  --source conversations.jsonl --source-type conversation \
  --source-id conv-2026-01 --dry-run

# Ingest verbatim.
python -m evaluation.ingest ingest --root evaluation-data \
  --source conversations.jsonl --source-type conversation \
  --source-id conv-2026-01 --mode verbatim

# Inspect, delete or redact.
python -m evaluation.ingest list-sources --root evaluation-data
python -m evaluation.ingest delete-source --root evaluation-data \
  --source-id conv-2026-01
python -m evaluation.ingest redact-source --root evaluation-data \
  --source-id conv-2026-01
python -m evaluation.ingest delete-memory --root evaluation-data \
  --memory-id <uuid>
python -m evaluation.ingest redact-memory --root evaluation-data \
  --memory-id <uuid>
```

Ingestion is idempotent per source: re-running the same source skips
already-ingested segments rather than duplicating them. A source must contain
exactly one `source_id` per file.

## Credentials and sensitive content

- `--credential-policy reject` (default) refuses content or identifiers that
  contain credentials.
- `--credential-policy redact` replaces detected credentials with a placeholder
  before storage.
- Sensitive personal information is not silently filtered. Operators must
  either pre-clean sources or accept that they are stored; evaluation queries
  and reports must never reproduce it.

Do not send private sources to any external service — model download, index
service, or extractor — without explicit authorization.

## Deletion, redaction and retention

Deleting or redacting a source removes the normalized segment file that holds
its text and writes a tombstone; the source cannot be revived under the same
`source_id`. Deleting a memory removes it from canonical state and Qdrant.
Redaction zeroes content and excludes the memory from reads, checkpoints and
retrieval.

**This is not secure erasure.** Filesystem snapshots, copy-on-write layers, disk
images, and backups may retain earlier copies. The store cannot reach external
backups. Document your own backup-retention rules before ingesting sensitive
material, and treat the evaluation directory as containing sensitive data even
after deletion.

## Isolation

An evaluation store owns its own directory, manifest, canonical checkpoint,
provenance ledger, normalized sources, queries, ground truth and results. It
never points at a production checkpoint. Use a distinct `tenant` per evaluation
environment; one canonical memory serves one tenant.

See [Evaluation Protocol](../operations/evaluation-protocol.md) for the
pre-registered experiment and [Service](service.md) for the HTTP path.
