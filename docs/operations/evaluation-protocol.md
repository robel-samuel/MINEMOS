# 30-Day Continuous Evaluation Protocol

**Status: NOT STARTED.** The environment, ingestion pipeline and scoring
harness exist and are locally verified. No 30-day run has been performed, and
no result in this repository may be read as one. This document is the
pre-registration: it fixes the question, data rules, schedule, queries, metrics
and stopping rules *before* the experiment begins.

## Question

Does MINEMOS preserve, consolidate, correct, date, retrieve and delete
naturally occurring personal/project information over 30 days of continuous
ingestion, without losing provenance or leaking superseded facts?

## Hypotheses

| ID | Hypothesis | Falsified by |
|---|---|---|
| H1 | Long-term retention: facts ingested on day 1 are retrievable on day 30 | any day-1 fact unrecoverable without explicit deletion |
| H2 | Consolidation: repeated observations do not inflate slot count unboundedly | duplicate count grows with each repeated statement |
| H3 | Contradiction resolution: a newer exclusive fact supersedes an older one | both values returned as active, or old value ranked active |
| H4 | Temporal reasoning: `as_of` returns the value valid at that time | wrong interval returned |
| H5 | Retrieval quality degrades gracefully with volume | recall@5 falls below the pre-registered floor |
| H6 | Redaction/deletion is durable across restart | deleted content reappears after restart/restore |
| H7 | Provenance survives restart and backup recovery | a memory loses its source link or the ledger loses parity |

No hypothesis is accepted from a single example; each requires the scheduled
query set to pass across the full run.

## Environment

- Dedicated directory outside the repository working tree, e.g.
  `$MINEMOS_EVAL_ROOT` (never a production path).
- `python -m evaluation.ingest init --root "$MINEMOS_EVAL_ROOT" --tenant evaluation`
- One tenant per environment. Canonical checkpoint is authoritative; Qdrant, if
  used, is derived and rebuildable.
- Models cached before day 0 so startup is deterministic.
- The environment is never reset automatically. A restart resumes the same
  checkpoint.

## Data rules (fixed before day 0)

1. Use material the operator owns or is authorized to process.
2. No private source leaves the machine without explicit authorization.
3. Record for each source: `source_id`, `source_type`, original timestamps,
   and the authorization basis.
4. Do not pre-clean facts to make retrieval easy. Do not insert expected
   answers into sources, normalized segments, checkpoints or the ledger.
5. Ground truth is authored separately from sources, before the run, and kept
   under `ground_truth/`. It is stored as facts (`subject`, `predicate`,
   `object`) with `as_of` where temporal.
6. Credentials are rejected by default. Sensitive personal information is the
   operator's responsibility; if it must be present, document it and plan
   deletion.

## Schedule

| When | Action |
|---|---|
| Day 0 | Initialize store, fix query set, capture baseline run, record commit hash, model versions, capacity and seed |
| Days 1–30 | Ingest the day's real material once, at a roughly consistent time; record the ingestion command and manifest |
| Daily | Run the full query set; results go to an immutable `results/<run_id>/` directory |
| Daily | Restart the process at least once per week to exercise checkpoint load |
| Weekly | Ingest a deliberate contradiction and a deliberate deletion/redaction; verify supersession and tombstone behavior |
| Day 10, 20, 30 | Run the same query set against the growing store and compare to day 0 baseline |
| Day 30 | Freeze the store, run the final query set, write the report with all failures |

A day with no real material is recorded as a no-ingestion day; it is not
backfilled with synthetic content.

## Query set

One JSONL (or JSON array) file, versioned by content hash. Every run records
the file's SHA-256 in `summary.json`. Query types supported by
`evaluation.run`:

| Type | Required fields | Measures |
|---|---|---|
| `retrieval` | `query`, `expected` | natural ingestion, retrieval accuracy, volume scaling |
| `temporal` | `query`, `as_of`, `expected`, optionally `expected_inactive` | temporal reasoning, historical reconstruction |
| `contradiction` | `query`, `expected` (active), `expected_inactive` (superseded) | contradiction detection/resolution, inactive leakage |
| `history` | `subject` | supersession chain and consolidation |
| `context` | `query`, `expected` | context reconstruction, multi-memory questions |
| `duplicate` | `source_id`, `expected_count` | duplicate detection |

Example:

```jsonl
{"query_id":"q-city-now","type":"retrieval","query":"Where does the user live?","expected":[{"subject":"user","predicate":"city","object":"Munich"}]}
{"query_id":"q-city-march","type":"temporal","query":"Where did the user live in March?","as_of":1769817600,"expected":[{"subject":"user","predicate":"city","object":"Berlin"}],"expected_inactive":[{"subject":"user","predicate":"city","object":"Munich"}]}
{"query_id":"q-city-history","type":"history","subject":"user"}
{"query_id":"q-source-recall","type":"duplicate","source_id":"conv-2026-01","expected_count":1}
```

Rules:

- Expected answers are used only for scoring and are never passed to the
  engine or written to checkpoint/provenance.
- The query file is frozen for the run. Adding queries creates a new version
  and is reported as such; it does not retroactively change earlier scores.
- Queries must be answerable from the sources. If a source changes the answer,
  the query is updated only by adding a new dated query, not by editing history.

## Command

```bash
python -m evaluation.run --root "$MINEMOS_EVAL_ROOT" \
  --queries "$MINEMOS_EVAL_ROOT/queries/round-01.jsonl" \
  --tenant evaluation --retrieval-mode canonical \
  --baseline "$MINEMOS_EVAL_ROOT/results/<day-0-run>/summary.json"
```

Outputs per run (immutable directory `results/<timestamp>-<hash8>/`):

- `summary.json` — pass rate, per-type pass rate and mean metrics, latency
  p50/p95/max, resource use, engine/retrieval/provenance/checkpoint status,
  baseline deltas.
- `per_query.jsonl` — one row per query with metrics and retrieved rows.
- `failures.jsonl` — every failure with its type, reason and metrics.
- `metadata.json` — tenant, store path, effective config, Python version,
  query hash.

## Metrics

- `recall_at_k`, `mrr`, `ndcg_at_k` per retrieval-type query.
- `matched_count` / `expected_count`.
- `inactive_leaked` for temporal and contradiction queries (must be empty).
- History queries: longest common subsequence against the expected chain.
- Duplicate queries: actual vs expected memory count for the source.
- Latency p50/p95/max and process RSS before/after each run.
- Capacity: active facts, superseded slots, slot count, evictions.
- Restart/recovery: pass rate after each restart must equal the pre-restart
  run for the same query set, modulo intentional weekly changes.

Pre-registered floor: **no expected fact may be lost through normal retention
over 30 days**, and **no superseded fact may appear as active**. Retrieval
quality floors are set per query set at day 0 and copied into the final report.

## Failure recording

A failing query is never deleted, skipped or annotated away. It is recorded
with its reason (`expected_fact_not_retrieved`, `superseded_fact_leaked`, a
schema/engine exception, etc.). The final report must list every failure and
distinguish:

- engine defect,
- ingestion/extraction defect,
- query/ground-truth defect,
- environment/infrastructure defect,
- expected behavior for a low-confidence or missing timestamp.

## Capability coverage

| Capability | Evidence in the run |
|---|---|
| Natural ingestion | daily sources from conversations, notes, docs, project history |
| Long-term retention | day-1 facts queried on day 30 |
| Consolidation | repeated statements vs slot count |
| Contradiction detection/resolution | weekly deliberate contradiction + contradiction queries |
| Temporal reasoning | `as_of` queries with `expected_inactive` |
| Retrieval accuracy | recall@k/MRR/nDCG per day |
| Context reconstruction | `context` queries spanning several memories |
| Duplicate detection | `duplicate` queries after repeated ingestion |
| Redaction | weekly redaction followed by restart and re-query |
| Checkpoint recovery | weekly restart, one backup restore exercise |
| Resource consumption | RSS and latency per run |
| Retrieval quality vs volume | day 0/10/20/30 baseline comparison |

## Stopping rules

- Stop and report if canonical checkpoint integrity fails, if a deleted or
  redacted fact reappears, or if tenant isolation breaks.
- A crash, model failure, or Qdrant outage is an incident, not a reset. Recover
  from checkpoint; record the gap.
- The run concludes after 30 elapsed days. Running fewer days cannot be
  described as a 30-day result.

## Reporting

The final report states the actual start/end dates, number of ingestion days,
source count, query version, environment commit, every failure, and the
verification level (locally verified vs. independently verified). It must not
claim completion before the 30 days have elapsed.

See [Data Ingestion](../guides/data-ingestion.md) for formats and
[Recovery](recovery.md) for checkpoint/backup behavior.
