# Integrating MINEMOS with an AI Application

MINEMOS is an in-process memory engine. It is not a hosted service and does not
call a model provider. Your application decides when to write observations, when
to read memory, and what to do with the result. This guide shows the shape of
that integration for a chat assistant, an agent, or a tool-using application.

## The two integration points

1. **Write path.** After a user turn or a tool result, convert what should be
   remembered into `(subject, predicate, object)` observations and call
   `absorb()`.
2. **Read path.** Before the model generates an answer, query memory and put the
   selected facts into the prompt.

Everything else—consolidation, contradiction handling, temporal validity,
retention, persistence—happens inside the engine.

```text
user turn ──► extract observations ──► absorb() ──► canonical state ──► checkpoint
                                                          │
user question ──► eligible_query() ──► prompt context ────┘
```

## Choosing the subject

The subject is the identity your application can address later. Use a stable
identifier, not a pronoun:

| Application | Recommended subject |
|---|---|
| Single-user assistant | `user` |
| Multi-user chat | The account or user ID (`user:42`) |
| Agents and tools | The agent, project, or entity being described |
| Conversation archive | `conversation:<conversation-id>` (a document boundary) |

The predicate is the attribute (`city`, `works_at`, `preferred_language`,
`owns`). The object is the value. Keep predicates consistent across writers so
that corrections land on the same attribute.

A document/conversation subject is a search boundary, not a person. Facts stored
under `conversation:<id>` are retrieved as scoped evidence, not as facts about an
entity. Use structured subjects when you need entity-level corrections.

## Writing observations

```python
from minemos import BlackHoleMemory, Candidate

mem = BlackHoleMemory(capacity=10_000)

result = mem.absorb(
    Candidate("user", "preferred_language", "Python"),
    timestamp=turn.timestamp,
)
print(result["action"])   # insert, consolidate, temporal_update, ...
```

Pass the source timestamp. If you omit it, the engine uses the current time.
Source time and arrival time are different things: a backfilled conversation
should carry its original timestamps so point-in-time queries and supersession
stay correct.

Do not store entire messages as the object. That turns memory into a transcript
store and defeats correction handling. Extract the state you want to be able to
update later. If you need verbatim transcripts with provenance, store them in
your own document store and use MINEMOS for the facts.

## Reading memory

Plain retrieval:

```python
rows = mem.query("which language does the user prefer", top_k=5)
context = "\n".join(row["text"] for row in rows)
```

Answer-aware retrieval, when the assistant must not present a contested fact as
settled:

```python
from minemos import answer_contract

outcome = mem.eligible_query("what is the user's preferred language", top_k=5)
presentation = answer_contract(
    outcome["status"], outcome["reason"],
    outcome["temporal"], len(outcome["rows"]),
)

if presentation["answer_status"] == "settled":
    context = "\n".join(row["text"] for row in outcome["rows"])
elif presentation["answer_status"] == "contested":
    context = "\n".join(row["text"] for row in outcome["rows"])
    # Tell the model that the facts conflict and ask for clarification.
else:
    context = ""   # no verified evidence; answer without memory
```

`eligible_query()` distinguishes "no such fact" from "the subject is ambiguous"
from "the rows do not contain the question's content". That distinction is what
lets an assistant say "I do not know" instead of guessing from a similar row.

## Follow-ups and conversation scopes

For transcript-style memory, `eligible_query()` accepts an explicit scope:

```python
outcome = mem.eligible_query(
    "what did we decide about the launch date",
    scope_subjects=["conversation:conv-1042", "conversation:conv-1043"],
)
```

`scope_subjects` searches the named document boundaries as one surface. If any
requested scope has no visible record, the query abstains with
`scope_not_found` rather than silently narrowing the request.

For a follow-up such as "what about that?", pass the previous turn as
`context_text` together with an explicit scope:

```python
outcome = mem.eligible_query(
    "what about the launch",
    subject="conversation:conv-1042",
    context_text=previous_turn_text,
)
```

Context is used to build the evidence question. It never widens the scope and is
never stored. If the named scope has no record carrying the follow-up's content,
the query abstains rather than searching the rest of the tenant.

If your application replays a message and then immediately answers it, pass
`current_turn=message`. Rows that repeat the caller's own turn are excluded from
evidence.

## Persistence in an application

Keep one engine per tenant and persist after writes:

```python
from pathlib import Path

class TenantMemory:
    def __init__(self, tenant: str, root: Path, capacity: int = 10_000):
        self.path = root / tenant / "memory.npz"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.mem = BlackHoleMemory.from_checkpoint(
                str(self.path), capacity=capacity)
        else:
            self.mem = BlackHoleMemory(capacity=capacity)

    def observe(self, subject, predicate, obj, timestamp):
        return self.mem.absorb(Candidate(subject, predicate, obj),
                               timestamp=timestamp)

    def flush(self):
        self.mem.save_checkpoint(str(self.path), create_backup=True)
```

Operational rules:

- Save at a cadence your application can tolerate. Checkpoints are atomic but
  proportional to resident state; very large memories should be saved less often
  than every turn.
- One process writes one checkpoint. In-process calls are thread-safe; multiple
  writer processes sharing a file are not coordinated.
- A `.bak` archive can contain facts deleted after the previous save. Treat
  backups as data and destroy them explicitly if deletion must be permanent.

## Multi-user applications

Tenant isolation is a deployment decision. Keep separate engines, directories,
and optional Qdrant collections per tenant:

```python
from minemos import BlackHoleMemory, QdrantConfig

mem = BlackHoleMemory(
    capacity=10_000,
    retrieval_backend="qdrant",
    qdrant_config=QdrantConfig(
        url="http://127.0.0.1:6333",
        collection="minemos_memories",
        tenant_id=account_id,
    ),
)
```

A Qdrant collection can be shared across tenants because every operation is
scoped by `tenant_id`. Canonical memory is never shared: one canonical memory
serves one tenant.

## Agents and tool results

For an agent, treat each durable fact from a tool result as an observation:

```python
result = run_tool("crm.lookup", {"account": "acme"})
mem.absorb(Candidate("account:acme", "plan", result["plan"]))
mem.absorb(Candidate("account:acme", "renewal_date", result["renewal"]))
```

Transient tool output that is only valid for one turn should stay in the agent's
working context, not in persistent memory. Store what a later session would need.

## Optional: resolving noisy text before writing

If your write path receives raw chat text, the optional resolution layer can
normalize it and propose candidate facts with speaker attribution before you
call `absorb()`. It is a separate package (`minemos.resolution`), it never
writes memory itself, and it is not wired into the engine: you call it, inspect
the candidates, and keep the final decision. See
[Semantic Resolution Layer](architecture/semantic-resolution.md).

If you want the adapter that segments, resolves and writes in one call, use
`minemos.semantic.SemanticIngestor` with an `EngineSink` or `ServiceSink`; it is
opt-in (`ResolutionConfig(mode="active")`) and still routes every write through
`absorb()`. See
[Semantic ingestion](python-api.md#semantic-ingestion-opt-in) and
[`examples/semantic_ingestion.py`](../examples/semantic_ingestion.py).

Whatever produces the observation, the `(subject, predicate, object)` you pass
to `absorb()` remains your statement about what the text means. The engine
still owns consolidation, contradiction, temporal validity and retention.

## Serving memory to other processes

The public repository does not ship an HTTP service. If your application is not
written in Python, or the model runtime is a separate process, put a thin RPC
layer of your own in front of the engine: authenticate the caller, derive the
tenant from the authenticated identity, then call `absorb` or `eligible_query`.
Do not let callers choose a tenant ID directly. The engine's thread safety is
in-process only; the boundary between processes is yours to secure.

## Next steps

- [Data Ingestion](guides/data-ingestion.md) — turning documents and
  conversations into observations.
- [Python API](python-api.md) — complete method and result reference.
- [Memory Lifecycle](architecture/memory-lifecycle.md) — what the engine does
  with each observation.
- [Retrieval](architecture/retrieval.md) — ranking, filters, and temporal
  queries.