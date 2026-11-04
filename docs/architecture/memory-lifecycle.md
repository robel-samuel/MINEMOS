# Memory Lifecycle

MINEMOS treats memory as a lifecycle, not a write-only store. Every observation
passes through addressing, verification, a decision, and capacity enforcement.
Historical state remains queryable after it is superseded.

## Slot representation

```text
MemorySlot
├── key             frozen lexical vector captured at slot creation
├── value           consolidated lexical vector; changes only on SAME
├── confidence      scalar in [0, 1]
├── timestamp       last update time
├── created_at      creation time
├── memory_id       stable UUID across checkpoint reloads
├── update_count    number of accepted observations
├── valid_from      start of the validity window
├── valid_until     end of the window, or None while active
├── text            current stored wording
├── dense_emb       dense embedding used for addressing and query
├── supersedes / superseded_by
└── latest_predicate / latest_object
```

The key is frozen so addressing remains stable while the value changes. Dense
embeddings are kept separately because the key/value vector is the lexical
representation used by the consolidation equation; dense similarity drives
addressing, retrieval, and the novelty score.

## Absorb pipeline

`BlackHoleMemory.absorb(observation, timestamp=None, confidence=1.0)` accepts a
`Candidate`, a `(subject, predicate, object)` tuple, or a simple
`"subject predicate object."` string.

1. **Parse and normalize.** Timestamps default to `Candidate.timestamp` or the
   current time.
2. **Encode.** The candidate is encoded lexically at the configured dimension
   and densely with `all-MiniLM-L6-v2`.
3. **Retrieve candidates.** A dense beam with adaptive anchor repulsion is
   combined with BM25 candidates. This internal candidate set is how the engine
   notices that an observation refers to something already stored.
4. **Verify.** The calibrated NLI verifier compares the stored text and the
   candidate text in both directions and returns `SAME`, `CONTRADICTION`, or
   `NEUTRAL`.
5. **Decide.** The observation is consolidated, superseded, or inserted under
   the rules below.
6. **Enforce capacity.** If a new slot is required and memory is full, tiered
   eviction frees one slot first.
7. **Reconcile the optional index.** If Qdrant is active, it is synchronized
   after the canonical mutation commits.

Engine construction enforces `1 <= capacity <= 500,000` and
`1 <= dim <= 16,384`; larger or non-positive values raise `ValueError`.
Absorb also rejects malformed encoder output (wrong rank or width, empty,
NaN/Inf) and non-finite or out-of-range timestamps/confidences before any
state is mutated, so an invalid observation cannot leave a partially updated
or unpersistable memory.

### Decision rules

| Situation | Result |
|---|---|
| Verifier says `SAME`, same entity, same attribute and value | Consolidate into the slot; increment `update_count` |
| `SAME` but the candidate belongs to a different concrete entity | Do not merge; continue evaluating other candidates |
| `SAME` over a different value of an exclusive attribute | Temporal supersession (or an independent observation if the attribute is not exclusive) |
| `SAME` against a closed historical slot | Treated as a recurrence, not merged into the closed window |
| Verifier says `CONTRADICTION`, same subject and attribute, not a directional artifact | Close the active window and insert a successor |
| `CONTRADICTION` involving independent co-facts | Kept as an independent active observation |
| Near-identical surface variant (typo, spacing, unambiguous abbreviation) | Consolidate the fact but keep the incumbent wording |
| `NEUTRAL` or no acceptable candidate | Insert a new slot |

The entity-grounding guard prevents independent facts about the same subject
from being merged or marked contradictory simply because their wording is
similar. Predicate cardinality is explicit: only configured exclusive groups are
treated as exclusive, multi-valued predicates are listed separately, and
unknown predicates are never assumed exclusive.

Historical statements such as "Alice previously lived in Oslo" are recorded as
closed observations. They do not consolidate into, or supersede, a current
state.

### Return value

`absorb()` returns a dictionary describing what happened. The `action` field is
one of:

| Action | Meaning |
|---|---|
| `consolidate` | Existing slot updated with a paraphrased restatement |
| `insert` | New active slot created |
| `evict_insert` | A slot was evicted to make room, then a new slot was created |
| `temporal_update` | Active slot closed and a successor inserted |
| `historical_update` | A predecessor was inserted into an existing chain |
| `historical_observation` | A past-state statement stored as a closed observation |
| `historical_rejected` | A past-state statement rejected under the temporal rules |

The result also includes `slot_index`, `slot_id`, `update_count`, validity
bounds, and the stored text where applicable.

## Consolidation

SAME observations update the value vector. With the default `hybrid` mode, the
implementation first bounds the single-step displacement, then regularizes the
result toward the frozen key:

```text
delta = alpha * (x_t - v_prev)
if ||delta|| > delta_max: delta = delta * (delta_max / ||delta||)
v_step = v_prev + delta
v_next = normalize((1 - lambda_anchor) * v_step + lambda_anchor * key)
```

where:

- `alpha = novelty = 1 - max dense similarity` to the candidate set;
- `delta_max = 0.25` bounds per-step displacement;
- `lambda_anchor = 0.20` pulls the value toward the frozen key.

The update is followed by `confidence = min(1, max(confidence, incoming) + 0.05)`
and `update_count += 1`. Because the key never changes, repeated paraphrases
cannot indefinitely rotate the stored representation away from its original
meaning.

Alternative consolidation modes (`baseline`, `bounded`, `anchor`) exist for
controlled comparison. The default is `hybrid`.

## Contradiction and supersession

When a genuine contradiction is established for the same entity and exclusive
attribute:

1. A successor slot is created with `valid_from = t_now`.
2. Only after the successor is successfully added does the predecessor receive
   `valid_until = t_now` and a `superseded_by` link.
3. The predecessor remains stored and is returned by point-in-time queries
   inside its old window.

Equal timestamps use a deterministic last-writer-wins replacement that preserves
the existing interval and links. Backward observations are inserted as
predecessors and linked into the same chain rather than creating overlapping
active states.

## Temporal validity

- `valid_from` and `valid_until` define a half-open interval
  `[valid_from, valid_until)`.
- `query(as_of=...)` and `recall(as_of=...)` return the state active at that
  instant.
- `get_history(subject, predicate)` returns active and superseded states,
  ordered by `valid_from`.
- `is_active` is true exactly while `valid_until is None`.
- `exclusive_temporal_conflicts()` reports active facts that cannot both hold
  for one exclusive attribute but are not linked by supersession. This detector
  is diagnostic and does not mutate memory.

## Retention and eviction

Memory capacity is a hard limit. When a new slot is needed and the state is full,
one slot is evicted according to `retention_policy`:

- `tiered` (default): slots with `update_count >= 2` are consolidated (Tier 1);
  slots with `update_count <= 1` are provisional (Tier 0). Tier 0 is evicted
  first. Tier 1 is capped at 80% of capacity, after which the lowest-utility
  Tier 1 slot is eligible.
- `fifo`: oldest `created_at`.
- `recency`: least recently updated (`timestamp`).
- `utility`: lowest contradiction-aware utility.
- `verification_aware`: lowest verification-aware utility.
- `oracle`: diagnostic only; not a deployable policy.

The live temporal successor is protected from eviction so that link chains never
point at a removed slot. If every slot is protected, the policy falls back to all
slots and the caller's capacity constraints take precedence.

## Checkpoint persistence

`save_checkpoint(path)` writes an atomic, checksummed `.npz` archive and can keep
a previous archive as a backup. `load_checkpoint(path)` validates schema,
capacity, dimensions, checksums, and path confinement before restoring state.
`BlackHoleMemory.from_checkpoint(path, **kwargs)` constructs and restores in one
step.

Schema v4 persists stable `memory_id` values and schema v5 the privacy-redaction
tombstone. v1–v4 archives load unchanged and slots receive deterministic
fallback IDs where the archive predates them. Configuration such as predicate
overrides and retrieval mode is not stored in the archive.

Backup archives retain the pre-save state, including records deleted since that
backup. This is documented behavior, not secure erasure. See
[Recovery](../operations/recovery.md) for the operational procedure.
