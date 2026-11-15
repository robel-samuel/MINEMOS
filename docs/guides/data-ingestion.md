# Data Ingestion

MINEMOS stores observations, not documents. Ingestion therefore has two explicit
stages that your application controls:

1. **Parse** a source into boundary-preserving segments.
2. **Extract** observations from each segment as `(subject, predicate, object)`.

The engine does not ship a general-purpose natural-language fact extractor. It
accepts structured observations and decides what to do with each one. Anything
that turns free text into observations belongs to your application, where you can
choose and audit the extractor.

## What the engine accepts

```python
from minemos import BlackHoleMemory, Candidate

mem = BlackHoleMemory(capacity=10_000)
mem.absorb(Candidate("user", "city", "Berlin"), timestamp=1767225600)
```

| Input | Behavior |
|---|---|
| `Candidate(subject, predicate, object, confidence=1.0, timestamp=0.0)` | Used as given; the candidate carries its own confidence and timestamp |
| `(subject, predicate, object)` tuple | Wrapped in a candidate; the `absorb` arguments supply timestamp and confidence |
| `"Subject predicate object."` string | Split on the first two spaces, remainder is the object; a string with fewer than three parts becomes `entity states <text>` |

The string form is a convenience for already-structured facts. It is not a
sentence parser and should not be used for conversations or documents.

## Choosing subjects and predicates

| Source | Subject | Predicate | Object |
|---|---|---|---|
| User profile | `user` or the account ID | `city`, `language`, `employer` | The value |
| Conversation archive | `conversation:<conversation-id>` | Segment position, e.g. `segment_12` | The message text |
| Project notes | `project:<name>` | `deadline`, `owner`, `status` | The value |
| External record | The record's stable ID | The field name | The field value |

Rules that keep corrections working:

- A subject is an identity your application will use again. Do not store
  pronouns or display names that change.
- A predicate is an attribute, not a sentence. Reuse the same predicate for the
  same attribute across writers.
- If a predicate must hold one value at a time, declare it with
  `exclusive_predicates=["city", "employer"]`. An unlisted predicate is treated
  as unknown, not exclusive.
- Store values, not whole messages, when you need entity-level corrections.
  Conversation-scoped segments are search material, not entity facts.

## Ingesting a conversation

```python
import json
from pathlib import Path

from minemos import BlackHoleMemory, Candidate

mem = BlackHoleMemory(capacity=10_000)

for line in Path("conversation.jsonl").read_text(encoding="utf-8").splitlines():
    record = json.loads(line)
    conversation_id = record["conversation_id"]
    subject = f"conversation:{conversation_id}"

    for index, message in enumerate(record["messages"]):
        mem.absorb(
            Candidate(
                subject,
                f"segment_{index}",
                message["content"],
                timestamp=message.get("timestamp", record["timestamp"]),
            ),
            timestamp=message.get("timestamp", record["timestamp"]),
        )

mem.save_checkpoint("state/conversations.npz")
```

Each message keeps its own boundary and timestamp. The conversation is never
flattened into a bag of unrelated facts.

For entity facts inside the conversation, extract them and store them under the
entity's normal subject:

```python
mem.absorb(Candidate("user", "city", "Berlin"), timestamp=turn_timestamp)
mem.absorb(Candidate("user", "city", "Munich"), timestamp=later_turn_timestamp)
```

The second observation supersedes the first; both remain queryable by time.

## Point-in-time queries over ingested content

```python
# What was true at a given moment?
mem.recall("user", "city", as_of=1767225600)

# What does an explicit conversation scope contain?
outcome = mem.eligible_query(
    "where did the user say they moved",
    scope_subjects=["conversation:conv-2026-01"],
)
```

The engine filters by the interval `[valid_from, valid_until)`. Source timestamps
are what make this meaningful, so pass them.

## Re-ingestion is not duplication

Absorbing the same fact again is a decision, not an append. A repeated or
paraphrased fact consolidates into the existing slot and raises its confidence.
A changed value supersedes the old one. Re-running an ingestion job over the same
source therefore does not create duplicate facts, but it does re-apply the
decision to the current state.

If you need to know how a re-ingested row was treated, inspect the `action` and
`consolidation_class` in the `absorb()` result:

```python
result = mem.absorb(Candidate("user", "city", "Berlin"), timestamp=t)
if result["action"] == "consolidate":
    ...  # same fact restated
elif result["action"] == "temporal_update":
    ...  # value changed
```

## Large documents

Split prose into paragraph-sized segments and keep an ordinal predicate:

```python
subject = f"document:{document_id}"
for index, paragraph in enumerate(paragraphs):
    mem.absorb(Candidate(subject, f"segment_{index}", paragraph,
                         timestamp=document_timestamp))
```

Segments keep retrieval focused. A single observation containing an entire
document dilutes similarity and cannot be corrected or partly redacted.

Use `segment_<index>` rather than a natural-language predicate so that
document-scoped records are not mistaken for entity attributes.

## Sensitive content

MINEMOS does not filter sensitive personal information. If a source contains
data you should not store, clean it before ingestion or do not ingest it.

- Do not put credentials, API keys, or tokens into an observation. They become
  ordinary memory content and are persisted in checkpoints.
- Redact before ingestion when the content is sensitive but the fact is needed.
- Redaction after ingestion destroys the content in canonical state and the
  Qdrant index, but cannot rewrite checkpoints and backups written earlier.
  See [Recovery](../operations/recovery.md#privacy-deletion-and-redaction).

## Deletion and lifecycle

```python
mem.redact_subject("user")          # destroy all content for one subject
mem.delete_memory(memory_id)        # remove one slot by stable id
mem.purge_subject("conversation:conv-2026-01")
mem.purge_redacted()                # drop tombstones when they are no longer needed
```

`redact_*` keeps a chain-safe tombstone so supersession links remain valid; the
content is excluded from queries, history, checkpoints, and the index.
`delete_*` and `purge_*` remove the records. Neither operation rewrites files
already written to disk outside the current checkpoint, so apply your own backup
retention rules before ingesting sensitive material.

## Isolation

An ingestion job should own its own engine and checkpoint. Do not point a bulk
ingestion process at a checkpoint that another process is writing. Partition by
tenant and keep one writer per canonical memory.

See [Integration](../integration.md) for the application-side pattern and
[Python API](../python-api.md) for every call used here.