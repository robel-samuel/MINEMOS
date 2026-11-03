# Benchmark Methodology and Results

All measurements below are from controlled experiments on synthetic datasets.
They are not production guarantees. Raw results are stored under
[`benchmarks/results/`](../../benchmarks/results/) and are the authoritative
record.

## Failure attribution schema (Phase 23)

Every end-to-end failure is assigned to the first stage where it occurred:

| Code | Stage | Description |
|---|---|---|
| A | Candidate generation | Correct target not retrieved |
| B | NLI verification | Target retrieved, NLI rejected |
| C | Consolidation update | NLI accepted, update failed |
| D | Retention/eviction | Slot evicted before the probe |
| E | Final probe retrieval | Slot survived but probe missed it |
| F | Semantic adjudication | False merge or false contradiction |
| SUCCESS | — | Correct end-to-end outcome |

Phase 23 failure shares at scale:

| Mechanism | Share | Category |
|---|---:|---|
| Slot eviction | 48.2% | D |
| NLI rejection of paraphrases | 31.4% | B |
| Candidate retrieval miss | 12.1% | A/E |
| Consolidation drift/update | 5.8% | C |
| False consolidation/safety | 2.5% | F |

## Phase 28 integration benchmarks

**Benchmark 1 — retention under capacity pressure (C=25, N=320):**

- Target survival: **100.0% (20/20)**, all Tier 1 consolidated slots protected.
- Tier 0 evictions: 275, all from provisional memory.
- Phase 23 baseline: 0.0%; Phase 23 oracle: 40.0%.

**Benchmark 2 — paraphrase verification and false merge (N=60 held-out pairs):**

- Paraphrase acceptance: **86.7%**
- Contradiction detection: **100.0%**
- False merge rate: **0.00%**
- Macro accuracy: **90.0%**

**Benchmark 3 — retrieval under four interference conditions:**

- R@1 = MRR = **1.000** in all conditions (all five methods tie at this ceiling).
- Contradiction intrusion = **0.000** (cached Phase 27 results).

**Benchmark 4 — consolidation drift (10-statement chain, 9 updates):**

- Baseline EMA minimum similarity to `v0`: 0.9520
- Phase 28 anchor-bounded minimum similarity: **0.9826**
- Drift reduction: **63.65%**

**Benchmark 5 — false contradiction and temporal supersession (10 adversarial tests):**

- False contradiction rate: **0.0%** (100% in the Phase 27 baseline)
- Genuine supersession rate: **100.0%**

## Qdrant integration benchmark (2026-11-03)

This is a separate benchmark for the derived Qdrant index. It does not modify or
reinterpret Phase 27 or Phase 31 results.

Environment: Windows 11, Python 3.12.10, `all-MiniLM-L6-v2` (384-d), Qdrant
1.20.0 embedded persistent mode (brute-force search), 40 queries per size,
`top_k=5`, `candidate_limit=128`. Each target fact is paired with one lexically
overlapping distractor, so indexed points equal twice the target facts. All
methods answer the same queries against the same canonical memory; Qdrant and
hybrid results are exactly re-ranked by MINEMOS.

### Embedded mode

| Target facts | Method | R@1 | R@5 | MRR | nDCG@5 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | MINEMOS dense | 0.550 | 1.000 | 0.771 | 0.971 | 20.7 | 26.4 |
| 100 | Qdrant dense | 0.550 | 1.000 | 0.771 | 0.971 | 34.2 | 41.0 |
| 100 | BM25 | 1.000 | 1.000 | 1.000 | 1.000 | 0.8 | 1.3 |
| 100 | Hybrid | 0.550 | 1.000 | 0.771 | 0.971 | 35.0 | 38.7 |
| 300 | MINEMOS dense | 0.475 | 1.000 | 0.714 | 0.944 | 22.9 | 26.9 |
| 300 | Qdrant dense | 0.475 | 1.000 | 0.714 | 0.944 | 59.6 | 66.4 |
| 300 | BM25 | 1.000 | 1.000 | 1.000 | 1.000 | 0.9 | 1.7 |
| 300 | Hybrid | 0.475 | 1.000 | 0.714 | 0.944 | 73.3 | 91.0 |
| 1,000 | MINEMOS dense | 0.275 | 0.875 | 0.538 | 0.799 | 31.1 | 62.5 |
| 1,000 | Qdrant dense | 0.275 | 0.875 | 0.538 | 0.799 | 134.5 | 202.3 |
| 1,000 | BM25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.8 | 4.5 |
| 1,000 | Hybrid | 0.275 | 0.875 | 0.538 | 0.799 | 128.9 | 144.1 |

| Target facts | Indexed points | Ingest total | Ingest per fact | Ingest CPU | Rebuild | RSS after ingest |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 200 | 10.1 s | 50.5 ms | 13.2 s | 1.72 s | 475.3 MB |
| 300 | 600 | 38.0 s | 63.3 ms | 59.5 s | 5.21 s | 507.1 MB |
| 1,000 | 2,000 | 302.1 s | 151.1 ms | 468.6 s | 17.66 s | 598.9 MB |

All sizes reported `errors=0`, `last_error=null`, and upserts exactly equal the
initial index plus one rebuild (400 / 1,200 / 4,000), confirming unchanged slots
are not re-written.

### Server mode

The same benchmark was run against a real Qdrant 1.20.0 server at
`127.0.0.1:6333` (HNSW, HTTP transport) with the same corpus, seed, and 40
queries per size. Results are in
[`qdrant_integration_benchmark_server.json`](../../benchmarks/results/qdrant_integration_benchmark_server.json).

| Target facts | Method | R@1 | R@5 | MRR | nDCG@5 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | MINEMOS dense | 0.550 | 1.000 | 0.771 | 0.971 | 14.3 | 18.0 |
| 100 | Qdrant dense | 0.550 | 1.000 | 0.771 | 0.971 | 19.9 | 35.9 |
| 100 | BM25 | 1.000 | 1.000 | 1.000 | 1.000 | 0.4 | 0.6 |
| 100 | Hybrid | 0.550 | 1.000 | 0.771 | 0.971 | 20.0 | 36.3 |
| 300 | MINEMOS dense | 0.475 | 0.950 | 0.714 | 0.944 | 16.9 | 24.7 |
| 300 | Qdrant dense | 0.475 | 0.950 | 0.714 | 0.944 | 20.4 | 43.3 |
| 300 | BM25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.3 | 2.0 |
| 300 | Hybrid | 0.475 | 0.950 | 0.714 | 0.944 | 29.9 | 35.9 |
| 1,000 | MINEMOS dense | 0.275 | 0.875 | 0.538 | 0.799 | 41.8 | 269.4 |
| 1,000 | Qdrant dense | 0.275 | 0.875 | 0.538 | 0.799 | 59.9 | 181.3 |
| 1,000 | BM25 | 1.000 | 1.000 | 1.000 | 1.000 | 4.7 | 14.0 |
| 1,000 | Hybrid | 0.275 | 0.875 | 0.538 | 0.799 | 52.8 | 78.5 |

| Target facts | Indexed points | Ingest total | Ingest per fact | Rebuild | RSS after ingest |
|---:|---:|---:|---:|---:|---:|
| 100 | 200 | 13.7 s | 68.7 ms | 0.17 s | 475.0 MB |
| 300 | 600 | 39.9 s | 66.6 ms | 0.35 s | 503.6 MB |
| 1,000 | 2,000 | 277.8 s | 138.9 ms | 3.13 s | 580.4 MB |

Server-mode runs reported `mode=server`, `errors=0`, and `last_error=null` at
every size.

### What these measurements show

- Qdrant dense and canonical MINEMOS dense returned identical rows here. Qdrant
  changes retrieval transport, not retrieval quality.
- BM25 is strongest on this synthetic, lexically separable corpus.
- Hybrid metrics equal dense because MINEMOS's final exact re-ranking is dense;
  BM25 only expands the candidate set.
- Embedded Qdrant adds latency over in-process dense retrieval (34 vs 21 ms p50
  at 200 points; 134 vs 31 ms at 2,000 points). Embedded mode scans without HNSW,
  so this does not predict server-mode latency.
- With per-mutation synchronization, ingestion cost grows with slot count
  (50.5 to 151.1 ms per fact) while index writes stay minimal. This is a
  consistency-first default, not a throughput claim.
- RSS is process-level and cumulative across sizes.
- Server-mode Qdrant latency is below embedded mode (19.9 vs 34.2 ms p50 at 200
  points; 59.9 vs 134.5 ms at 2,000) but still above in-process canonical dense
  retrieval (14.3 and 41.8 ms). At 2,000 points canonical dense p95 (269.4 ms)
  exceeded Qdrant's (181.3 ms); this is tail variance, not a quality difference.
- Server transport did not reduce ingestion cost.
- This benchmark does not establish that Qdrant is faster or more accurate than
  canonical retrieval, and it does not revise the Phase 31B/31C verdict.

## Opt-in RRF experiment (2026-11-03)

Run with `--include-rrf` at 100 and 300 target facts. Results are stored
separately in
[`qdrant_hybrid_fusion_benchmark.json`](../../benchmarks/results/qdrant_hybrid_fusion_benchmark.json);
historical results are unchanged. `hybrid_rrf` changes ordering only. Candidate
membership and the reported exact dense similarity are identical to `hybrid`.

| Target facts | Method | R@1 | R@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| 100 | Qdrant dense | 0.550 | 1.000 | 0.771 | 0.971 |
| 100 | Hybrid (dense order) | 0.550 | 1.000 | 0.771 | 0.971 |
| 100 | Hybrid RRF | 0.550 | 1.000 | 0.775 | 0.976 |
| 100 | BM25 | 1.000 | 1.000 | 1.000 | 1.000 |
| 300 | Qdrant dense | 0.475 | 1.000 | 0.714 | 0.944 |
| 300 | Hybrid (dense order) | 0.475 | 1.000 | 0.714 | 0.944 |
| 300 | Hybrid RRF | 0.500 | 1.000 | 0.750 | 0.973 |
| 300 | BM25 | 1.000 | 1.000 | 1.000 | 1.000 |

- RRF ordering improved MRR/nDCG@5 slightly at 200 points (+0.004/+0.005) and by
  +0.036/+0.029 at 600 points, with R@1 rising from 0.475 to 0.500 at the larger
  size. Recall@5 was already 1.000 for every method.
- BM25 remained perfect on this synthetic, lexically separable corpus; RRF
  narrows but does not close the gap.
- This is one corpus, one seed, and 40 queries per size. It does not support a
  general claim that RRF improves ranking quality. The default remains exact
  dense ordering. Latency differences were not stable across sizes and are not
  interpreted.

## Reproducing the Qdrant benchmark

```bash
# Embedded mode
python -m benchmarks.qdrant_retrieval_benchmark

# Server mode
MINEMOS_QDRANT_URL=http://127.0.0.1:6333 \
  python -m benchmarks.qdrant_retrieval_benchmark

# RRF experiment, separate output
python -m benchmarks.qdrant_retrieval_benchmark \
  --sizes 100 300 --include-rrf \
  --output benchmarks/results/qdrant_hybrid_fusion_benchmark.json
```

Default sizes are `100 300 1000 2000`, 40 queries per size, seed 7, top-k 5.
The script writes raw JSON; it does not overwrite historical phase results.

## Interpreting results

- Synthetic corpora with known ground truth do not establish real-world
  accuracy.
- Retrieval metrics measure whether the correct slot is retrieved, not whether
  the memory lifecycle made the correct earlier decision.
- Benchmarks that reuse cached Phase 27 results are labeled as such.
- No result in this document supports a claim of 100K-scale production quality,
  universal superiority, or production readiness.
