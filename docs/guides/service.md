# Service Layer

`minemos_service` is a small, stdlib-only HTTP boundary around the canonical
MINEMOS engine. It adds authentication, authorization, tenant isolation,
request validation, rate limiting, structured logging, health/readiness
reporting, per-tenant canonical checkpoints, and an append-only provenance
ledger.

It is intended for **controlled deployment**: a private network or a
reverse-proxied single-tenant/limited-tenant deployment. It is not a hosted
multi-tenant SaaS platform, it has no external identity provider, and it does
not make MINEMOS a validated production memory service.

The service owns no memory of its own. The canonical `.npz` checkpoint remains
authoritative; Qdrant, when configured, is a derived index only.

## Start

```bash
# Validate configuration only. No models or checkpoints are loaded.
python -m minemos_service --check-config

# Print the effective, secret-free configuration.
python -m minemos_service --print-config

# Load models and every configured tenant checkpoint, then exit.
python -m minemos_service --check

# Run the server.
python -m minemos_service
```

Startup is fail-closed:

- invalid or unsafe configuration exits with status 2;
- an unreadable tenant checkpoint or failed engine initialization exits with
  status 1;
- an untraceable provenance write fails the request and sets the tenant's
  persistence error, so `/readyz` reports not ready;
- SIGINT/SIGTERM stop accepting requests, save every tenant checkpoint, and
  close engines.

## Configuration

All settings come from the process environment. The service never reads a
`.env` file by itself and never invents credentials.

| Variable | Default | Meaning |
|---|---|---|
| `MINEMOS_HOST` | `127.0.0.1` | Bind address |
| `MINEMOS_PORT` | `8080` | Bind port (`0` is allowed in development only) |
| `MINEMOS_ENVIRONMENT` | `development` | `development` or `production` |
| `MINEMOS_DATA_DIR` | `var/minemos` | Base data directory |
| `MINEMOS_CHECKPOINT_DIR` | `<data>/checkpoints` | Canonical `.npz` checkpoints |
| `MINEMOS_PROVENANCE_DIR` | `<data>/provenance` | Append-only JSONL ledgers |
| `MINEMOS_API_KEYS_FILE` | unset | Restricted JSON file of API keys |
| `MINEMOS_API_KEYS` | unset | Inline JSON key list (prefer the file) |
| `MINEMOS_DEV_INSECURE_NO_AUTH` | `false` | Development-only tenant header fallback |
| `MINEMOS_TLS_CERT_FILE` | unset | PEM certificate chain |
| `MINEMOS_TLS_KEY_FILE` | unset | PEM private key |
| `MINEMOS_ALLOW_INSECURE_HTTP` | `false` | Explicit opt-out of TLS enforcement |
| `MINEMOS_MAX_BODY_BYTES` | `1048576` | Maximum request body |
| `MINEMOS_MAX_OBSERVATIONS_PER_REQUEST` | `256` | Maximum observations per ingest |
| `MINEMOS_MAX_OBJECT_CHARS` | `8192` | Maximum object length |
| `MINEMOS_MAX_QUERY_CHARS` | `4096` | Maximum query length |
| `MINEMOS_REQUEST_TIMEOUT` | `30` | Socket timeout in seconds |
| `MINEMOS_RATE_LIMIT_PER_MINUTE` | `600` | Token refill rate per key |
| `MINEMOS_RATE_LIMIT_BURST` | `100` | Token bucket capacity per key |
| `MINEMOS_ENGINE_CAPACITY` | `10000` | Slots per tenant engine |
| `MINEMOS_ENGINE_DIM` | `4096` | Lexical dimension |
| `MINEMOS_RETRIEVAL_MODE` | unset | `legacy`, `production`, or `canonical` |
| `MINEMOS_QDRANT_URL` | unset | Server-mode Qdrant HTTP(S) endpoint |
| `MINEMOS_QDRANT_API_KEY` | unset | Qdrant API key (never logged) |
| `MINEMOS_QDRANT_COLLECTION_PREFIX` | `minemos_memories` | Per-tenant collection prefix |
| `MINEMOS_QDRANT_TIMEOUT` | `5` | Qdrant request timeout |
| `MINEMOS_PERSIST_AFTER_MUTATION` | `true` | Save checkpoint after each mutation |
| `MINEMOS_CREDENTIAL_POLICY` | `reject` | `reject` or `redact` credential-like content |
| `MINEMOS_LOG_LEVEL` | `INFO` | JSON log level |

`MINEMOS_QDRANT_PATH` exists for embedded Qdrant in development and tests. It is
rejected in production.

### Key file

```json
{
  "keys": [
    {"key_id": "reader-1", "key": "<high-entropy-placeholder>", "tenant": "tenant-a", "roles": ["reader"]},
    {"key_id": "writer-1", "key": "<high-entropy-placeholder>", "tenant": "tenant-a", "roles": ["reader", "writer"]},
    {"key_id": "admin-1",  "key": "<high-entropy-placeholder>", "tenant": "tenant-a", "roles": ["admin"]}
  ]
}
```

On POSIX the file must not be accessible by group or other
(`mode & 0o077 == 0`). In production every key must be at least 32 characters.
`admin` implies reader and writer. A tenant is always derived from the
authenticated key; request content cannot select a different tenant.

### Production validation

`MINEMOS_ENVIRONMENT=production` requires all of the following:

- API keys are configured, none is shorter than 32 characters, and
  `MINEMOS_DEV_INSECURE_NO_AUTH` is false;
- `MINEMOS_DATA_DIR` is an absolute path;
- `MINEMOS_RETRIEVAL_MODE=production` and a server-mode `MINEMOS_QDRANT_URL`;
- embedded Qdrant (`MINEMOS_QDRANT_PATH`) is absent;
- a non-loopback bind uses TLS, unless `MINEMOS_ALLOW_INSECURE_HTTP=true` is set
  explicitly for a trusted private network;
- a non-loopback Qdrant URL uses HTTPS and has an API key, unless the same
  explicit insecure opt-out is set.

## HTTP API

| Method | Path | Role | Purpose |
|---|---|---|---|
| `GET` | `/healthz` | public | Process liveness |
| `GET` | `/readyz` | public | Readiness, degradation, checkpoint writability |
| `GET` | `/v1/status` | reader | Engine, retrieval, provenance, checkpoint status |
| `POST` | `/v1/ingest` | writer | Ingest one or more observations |
| `POST` | `/v1/query` | reader | Retrieve current or point-in-time facts |
| `POST` | `/v1/history` | reader | Full subject/predicate history |
| `POST` | `/v1/redact` | writer | Redact one memory by stable `memory_id` |
| `POST` | `/v1/delete` | writer | Delete one memory by stable `memory_id` |
| `POST` | `/v1/source/redact` | writer | Redact every memory linked to a source |
| `POST` | `/v1/source/delete` | writer | Delete every memory linked to a source |
| `POST` | `/v1/admin/checkpoint` | admin | Force a canonical checkpoint save |

All protected requests use `Authorization: Bearer <key>`. `GET /healthz` and
`GET /readyz` are public so a load balancer can probe them without a
credential; they expose no tenant data.

### Ingest

```bash
curl -sS http://127.0.0.1:8080/v1/ingest \
  -H "Authorization: Bearer $MINEMOS_WRITER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "source": {
      "source_id": "notes-2026-11",
      "source_type": "notes",
      "document_id": "notes-2026-11"
    },
    "observations": [
      {"subject": "alice", "predicate": "city", "object": "Berlin",
       "timestamp": 1735689600.0, "confidence": 0.9}
    ]
  }'
```

Response (200):

```json
{
  "tenant": "tenant-a",
  "results": [{"action": "insert", "memory_id": "<uuid>", "ingestion_id": "<hash>", "duplicate_of": null, "skipped": false, "content_redacted": false}],
  "ingested": 1,
  "skipped": 0,
  "failed": 0,
  "unprocessed": 0,
  "partial": false
}
```

- A deterministic ingestion id makes retries idempotent. Re-sending the same
  observation returns `skipped: true` and does not create a second memory.
- `duplicate_of` names a prior memory with the same complete fact.
- The process-local `slot_id` is never returned; use `memory_id`.
- Status codes: `200` success, `207` partial success, `422` validation or
  sensitive-content rejection, `503` persistence/provenance unavailable.

### Query and history

```bash
curl -sS http://127.0.0.1:8080/v1/query \
  -H "Authorization: Bearer $MINEMOS_READER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "where does alice live", "top_k": 5, "as_of": 1735689600.0}'
```

`as_of` is optional and performs a point-in-time query against half-open
validity intervals. `POST /v1/history` takes `{"subject": "...",
"predicate": "..."}` and returns the full chain, including superseded states.

### Redaction and deletion

```bash
curl -sS http://127.0.0.1:8080/v1/redact \
  -H "Authorization: Bearer $MINEMOS_WRITER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"memory_id": "<uuid>"}'
```

Redaction destroys key/value vectors, text, embeddings, labels, contradiction
anchors, and the BM25 document, then keeps a content-free tombstone so
supersession links stay valid. Redacted content is excluded from reads, new
checkpoints, and Qdrant.

`/v1/source/redact` and `/v1/source/delete` act on every active memory linked to
the source id in the provenance ledger. They do not remove a backup checkpoint
that predates the action. See
[Data Ingestion](data-ingestion.md#deletion-redaction-and-retention).

### Status

`/v1/status` is tenant-scoped and reports:

- `engine`: slot counts, absorbed/update/eviction counters, footprint bytes;
- `retrieval`: selected backend, mode, initialization, degraded state;
- `provenance`: ledger path, event/ingestion/source counts, redacted/deleted/
  orphaned counts;
- `checkpoint`: path, existence, size, modification time, backup existence;
- `persistence`: pending flag, last error, last save time;
- `reconciliation`: orphaned/untracked counts observed at startup.

## Validation and safety behavior

- Unknown JSON fields are rejected rather than silently ignored.
- JSON bodies must have `Content-Length` and
  `Content-Type: application/json`; chunked bodies are rejected.
- Control characters, oversized fields, non-finite timestamps, booleans used as
  numbers, and out-of-range `top_k` are rejected.
- Credential-like content is detected before storage. `reject` refuses it;
  `redact` replaces the matched value with `[REDACTED:<kind>]`. Credentials in
  source identifiers are always rejected because rewriting them would silently
  change source attribution.
- Errors are returned as `{"error": {"code": "...", "message": "..."},
  "request_id": "..."}`. Internal exception text is not returned.
- Responses set `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`,
  and `X-Request-ID`.
- Rate limiting is a per-key token bucket. A rejected request receives `429`
  and `Retry-After`.

## Logging

Logs are one JSON object per line on stderr. Request logs include request id,
method, path, status, duration, tenant, and key id — never request bodies,
query text, memory content, API keys, or Qdrant credentials. Free-form string
fields are scrubbed with the same credential patterns used for validation.

## Persistence layout

```text
<data>/
├── checkpoints/
│   └── <tenant>/
│       ├── state.npz
│       └── state.npz.bak
└── provenance/
    └── <tenant>.jsonl
```

Checkpoint paths are confined with `allowed_base_dir` to the checkpoint
directory. Loading a corrupt primary falls back to the `.bak` archive; if both
are unreadable, startup fails instead of serving an empty memory. The ledger
contains digests and stable memory ids, never raw text.

`persist_after_mutation` saves after each accepted ingest, redaction, deletion,
or source action. A crash after the ledger append but before the checkpoint save
leaves an orphaned provenance record; startup reconciliation reports it rather
than hiding it or fabricating provenance for canonical memories.

## What is verified, and what is not

Verified locally with deterministic stub encoders and the stdlib server:

- authentication, role enforcement, tenant-derived isolation;
- request validation, body limits, safe errors, security headers;
- rate-limit behavior and secret-free JSON logging;
- TLS listener using a locally generated self-signed certificate;
- checkpoint persistence, backup recovery, fail-closed corrupt checkpoints;
- provenance reconciliation, stable-id redaction/deletion, restart survival;
- health/readiness and graceful shutdown behavior.

Not verified in this environment:

- a real server-mode Qdrant deployment over TLS with an API key (no Qdrant
  server or credentials were available);
- certificate chains issued by a public/enterprise CA or client-certificate
  authentication;
- external identity providers, key rotation APIs, or OAuth;
- multi-process writers, horizontal scaling, or shared checkpoint locking;
- load balancer, container, and orchestrator integration;
- independently audited penetration testing.
