# Configuration

Configuration is process-level. Retrieval settings and predicate overrides are
not written into checkpoints; an archive stores memory facts, not deployment
policy.

## Engine parameters

`BlackHoleMemory(...)` accepts:

| Parameter | Default | Purpose |
|---|---|---|
| `capacity` | `100` | Maximum stored slots, including superseded history |
| `dim` | `4096` | Lexical key/value dimension |
| `retention_policy` | `"tiered"` | Eviction policy: `tiered`, `fifo`, `recency`, `utility`, `verification_aware`; `oracle` is diagnostic only |
| `verification_mode` | `"bidirectional"` | NLI comparison mode |
| `entail_threshold` | `0.60` | SAME threshold |
| `contra_threshold` | `0.40` | Contradiction threshold |
| `consolidation_mode` | `"hybrid"` | `hybrid`, `bounded`, `anchor`, `baseline` |
| `delta_max` | `0.25` | Maximum per-step value displacement |
| `lambda_anchor` | `0.20` | Pull toward the frozen key |
| `top_k_dense` | `8` | Dense candidate beam during absorption |
| `top_k_bm25` | `8` | BM25 candidate beam |
| `k_fusion` | `10` | Fused candidate count during absorption |
| `w_anchor`, `decay_lambda`, `max_anchors` | engine defaults | Anchor repulsion controls |
| `exclusive_predicates` | `None` | Explicit exclusive-predicate overrides |
| `multi_valued_predicates` | `None` | Explicit multi-valued-predicate overrides |
| `retrieval_backend` | `None` | `canonical`, `optimized`, `incremental`, `qdrant` |
| `retrieval_config` | `None` | `RetrievalConfig`, backend name, or dict |
| `retrieval_mode` | `None` | `legacy`, `production`, `canonical` |
| `qdrant_config` | `None` | `QdrantConfig`, dict, or environment lookup |
| `dense_encoder`, `nli_gate`, `verifier`, `lex_encoder` | `None` | Dependency injection for tests or custom models |

Defaults such as `delta_max` and `lambda_anchor` define the memory decision.
Changing them changes what the engine stores, so treat them as behavioral
settings rather than tuning knobs; existing checkpoints do not record them.

Predicate policy overrides accept an iterable of predicates. Exclusivity is
never inferred from predicate identity alone: an unlisted predicate is treated as
unknown, not exclusive.

## Retrieval modes

`retrieval_mode` selects the deployment profile:

| Mode | Behavior |
|---|---|
| `legacy` (default) | Canonical retrieval unless a backend is selected explicitly. Existing behavior is unchanged. |
| `production` | Qdrant is the required retrieval backend. Configuration is validated at construction; missing or unsafe settings raise `RetrievalConfigurationError`. Runtime outages fall back to canonical and report `degraded`. |
| `canonical` | Canonical only. Qdrant is never constructed or contacted, even if `QDRANT_*` variables are set. |

Resolution order:

1. `retrieval_mode=` constructor argument;
2. `MINEMOS_RETRIEVAL_MODE` environment variable;
3. `RetrievalConfig.mode`;
4. `legacy`.

In `production` mode the only permitted runtime backends are `qdrant` and an
emergency `canonical` rollback. In `canonical` mode only `canonical` is
permitted.

## Qdrant environment variables

| Variable | Meaning |
|---|---|
| `MINEMOS_RETRIEVAL_MODE` | `legacy`, `production`, or `canonical` |
| `QDRANT_URL` | Remote Qdrant endpoint; mutually exclusive with `QDRANT_PATH` |
| `QDRANT_PATH` | Embedded persistent database directory, or `:memory:` |
| `QDRANT_API_KEY` | API key; never included in `repr`, logs, or stats |
| `QDRANT_COLLECTION` | Collection name (default `minemos_memories`) |
| `QDRANT_TENANT_ID` | Tenant scope (default `default`) |
| `QDRANT_TIMEOUT` | Client timeout in seconds |
| `QDRANT_PREFER_GRPC`, `QDRANT_GRPC_PORT` | Remote transport options |
| `QDRANT_RECREATE` | Drop and recreate the collection at startup; rejected in production |
| `QDRANT_CANDIDATE_LIMIT` | Dense candidate limit (default `128`) |
| `QDRANT_BATCH_SIZE` | Upsert/delete batch size (default `256`) |
| `QDRANT_MAX_RETRIES`, `QDRANT_RETRY_BACKOFF` | Index-sync retry policy |
| `QDRANT_PAYLOAD_INDEXES` | Create payload indexes in server mode (default true) |
| `QDRANT_ON_DISK` | Store vectors on disk in server mode |
| `QDRANT_ALLOW_INSECURE` | Permit plain HTTP to a non-loopback host in production |
| `QDRANT_HYBRID_FUSION` | Default hybrid ordering: `dense` or `rrf` |
| `QDRANT_RRF_K` | Reciprocal-rank constant for `rrf` (default `60`) |

Do not commit real credentials. The repository's `.env.example` documents only
optional Hugging Face and CPU settings; Qdrant variables are read from the
process environment or passed explicitly through `QdrantConfig`.

## Production Qdrant rules

`retrieval_mode="production"` requires:

- `QDRANT_URL` pointing at a Qdrant server; embedded `QDRANT_PATH` is rejected;
- an `http` or `https` URL with a hostname;
- HTTPS and `QDRANT_API_KEY` for non-loopback hosts, unless
  `QDRANT_ALLOW_INSECURE=true` is explicitly set for a trusted internal network;
- `QDRANT_RECREATE=false`;
- a valid collection and tenant scope.

Loopback hosts are exempt from the TLS and API-key requirements so a local
sidecar can run without credentials. Configuration failures stop startup; they
do not silently downgrade to canonical.

## RetrievalConfig

`RetrievalConfig` controls the optional in-process snapshot backends:

| Field | Default | Meaning |
|---|---|---|
| `backend` | `"canonical"` | Selected backend |
| `mode` | `"legacy"` | Deployment profile |
| `max_optimistic_attempts` | `3` | Snapshot build retries |
| `max_incremental_updates` | `64` | Delta updates before a rebuild |
| `fallback_to_canonical` | `True` | Hand off on contention or failure |
| `verify_bound` | `False` | Verify numerical bounds |
| `contended_fallback` | `"canonical"` | `canonical` or `locked_build` |

## Environment file

Copy `.env.example` to `.env` only if you need optional settings such as
`HF_HOME`, `TRANSFORMERS_OFFLINE`, or a Qdrant endpoint. Most users need no
environment variables. `.env` is ignored by git and must not be committed.

## Next steps

- [Qdrant Integration](../architecture/qdrant-integration.md)
- [Retrieval](../architecture/retrieval.md)
- [Python API](../python-api.md)
