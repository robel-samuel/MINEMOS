# Phase History

Each phase built on a frozen baseline from earlier phases. Numbers below are the
reported experimental results at the time of each phase; they are not re-run or
reinterpreted for this documentation. Raw reports and JSON artifacts remain the
authoritative record.

## Early phases (1–22)

| Phases | Focus and outcome |
|---|---|
| 1–6 | Structured subject/predicate/object storage, confidence, contradiction handling, temporal validity, threshold sensitivity |
| 7–8 | Encoder ablation from atomic tokens to field-aware lexical encoding and semantic embeddings |
| 9–10 | NLI integration and an audit of verification behavior |
| 11–12 | Scaling benchmarks and structured addressing |
| 13–16 | Semantic NLI gate (`distilroberta-base`) and system ablations A/B/C/D |
| 17–19 | Utility-aware eviction, dense semantic retrieval (`all-MiniLM-L6-v2`), contradiction-aware utility |
| 20–21 | Adaptive anchor management (frozen `w*`, `λ*`, `k*`) and anchor-beam stress tests |
| 22 | Hybrid BM25 + dense retrieval, reported 3.9–5.8× recall improvement |

The complete early reports are in [`reports/`](../../reports/).

## Phase 23: Failure attribution

**Report:** [`reports/PHASE_23_REPORT.md`](../../reports/PHASE_23_REPORT.md)
**Results:** [`benchmarks/results/phase23_results.json`](../../benchmarks/results/phase23_results.json)

Seven experiments used a five-category (plus success) failure schema to locate
where the lifecycle failed:

| Code | Stage |
|---|---|
| A | Candidate generation: correct target not retrieved |
| B | NLI verification: target retrieved but rejected |
| C | Consolidation: accepted but update failed |
| D | Retention/eviction: slot evicted before the probe |
| E | Final retrieval: slot survived but probe missed it |
| F | Semantic adjudication: false merge or false contradiction |

Findings:

- Retention (Category D) dominated failures at scale: **48.2%**.
- Oracle retention lifted end-to-end accuracy from **0.0% to 40.0%**.
- Hybrid BM25 improved retrieval recall **3.9–5.8×** but did not lift end-to-end
  accuracy while eviction remained the bottleneck.

## Phase 24: Retention, verification, drift

**Report:** [`reports/PHASE_24_REPORT.md`](../../reports/PHASE_24_REPORT.md)
**Results:** [`benchmarks/results/phase24_results.json`](../../benchmarks/results/phase24_results.json)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Tiered retention eliminates Category D eviction | Confirmed | 88% target survival vs 0% FIFO on the adversarial stream |
| Calibrated verification improves paraphrase acceptance | Confirmed | False merge rate 0.0%, contradiction accuracy 1.0%, paraphrase acceptance increased |
| Drift-resistant consolidation bounds representation drift | Confirmed | Anchor-bounded hybrid bounded to `delta_max = 0.25` |
| Retention and verification interact positively | Confirmed | Combined system improved over the FIFO baseline |

## Phase 25: Unified engine

**Report:** [`reports/PHASE_25_REPORT.md`](../../reports/PHASE_25_REPORT.md)
**Results:** [`benchmarks/results/phase25_e2e_results.json`](../../benchmarks/results/phase25_e2e_results.json)

Integrated the Phase 24 improvements into `phase2/black_hole_memory.py` and
validated 19 end-to-end checks covering streaming absorption, temporal
supersession, tiered eviction, drift resistance, and query accuracy.

Limitations recorded at the time included paraphrase misclassification, no
serialization layer, and no concurrency support. Persistence and concurrency
work followed in later tasks; the current state is documented in
[Architecture](../architecture/overview.md).

## Phase 26: Calibrated bidirectional verification

**Report:** [`reports/PHASE_26_REPORT.md`](../../reports/PHASE_26_REPORT.md)
**Results:** [`benchmarks/results/phase26_results.json`](../../benchmarks/results/phase26_results.json)

Addressed directional asymmetry in NLI scoring. Held-out split, N=60:

| Metric | Phase 25 baseline | Phase 26 calibrated |
|---|---:|---:|
| Paraphrase acceptance | 80.0% | **86.7%** |
| Contradiction detection | 100.0% | **100.0%** |
| False merge rate | 0.00% | **0.00%** |
| Macro accuracy | — | **90.0%** |

Streaming multi-turn end-to-end: 4/4 factual consolidations (up from 3/4), zero
false merges, correct temporal supersession.

## Phase 27: Long-horizon interference

**Report:** [`reports/PHASE_27_REPORT.md`](../../reports/PHASE_27_REPORT.md)
**Results:** [`benchmarks/results/phase27_results.json`](../../benchmarks/results/phase27_results.json)

Adaptive retrieval retained accuracy across low, semantic, temporal, and combined
interference:

| Condition | R@1 | MRR | Contradiction intrusion | Temporal correctness |
|---|---:|---:|---:|---:|
| Low | 1.000 | 1.000 | 0.000 | 1.000 |
| Semantic | 1.000 | 1.000 | 0.000 | 1.000 |
| Temporal | 1.000 | 1.000 | 0.000 | 1.000 |
| Combined | 1.000 | 1.000 | 0.000 | 1.000 |
| Held-out stream | 1.000 | 1.000 | 0.000 | 1.000 |

Reported retrieval latency was 14–20 ms per query on CPU, under 2.6 ms overhead
over the baseline. The phase also identified a defect: independent facts about
one subject could be falsely treated as contradictions. Phase 28 addressed it.

## Phase 28: Failure closure

**Report:** [`reports/PHASE_28_REPORT.md`](../../reports/PHASE_28_REPORT.md)
**Results:** [`benchmarks/results/phase28_results.json`](../../benchmarks/results/phase28_results.json)

| Failure mode | Phase 23 baseline | Phase 28 result | Status |
|---|---:|---:|---|
| Eviction (D), target survival | 0.0% | **100.0% (20/20)** | Fixed in tested scope |
| Paraphrase (B), acceptance | 68.6% | **86.7%** | Partially mitigated |
| Retrieval (A/E), MRR | ~0.879 | **1.000** | Fixed in tested scope |
| Drift (C), minimum similarity to `v0` | 0.120 | **0.9826** | Fixed in tested scope |
| False merges (F), FMR | 2.50% | **0.00%** | Fixed in tested scope |
| Phase 27 defect, false contradictions | 100.0% | **0.0%** | Fixed in tested scope |

The residual 13.3% paraphrase rejection rate is attributed to the limits of
`cross-encoder/nli-distilroberta-base` on high-lexical-variation paraphrases.
The false-contradiction row covers the benchmarked allowlisted cases; the
post-publication entity-grounding correction and the adversarial regression
tests extend coverage. The Phase 28 suite at the time contained 337 tests; the
current repository suite is larger.

## Phase 29: Temporal invariants

**Report:** [`reports/PHASE_29_REPORT.md`](../../reports/PHASE_29_REPORT.md)
**Plan:** [`reports/PHASE_29_PLAN.md`](../../reports/PHASE_29_PLAN.md)
**Results:** [`benchmarks/results/phase29_results.json`](../../benchmarks/results/phase29_results.json)

Phase 29 tested whether `temporal_history_violations()` is sound and complete
over states the engine can hold, and whether SAME-over-changed-value handling is
safe when the matched slot is historical.

Changes and findings:

- Link validation now rejects inconsistent, dangling, self, and non-reciprocal
  links rather than silently grouping them.
- A recurrence matching a closed historical slot resolves the active chain head
  instead of inserting a second unlinked active version.
- When every version is closed, the recurrence is linked to the nearest prior
  version.
- Benchmark verdicts are explicit. The runner exits 0 only for a clean pass, 2
  while a documented limitation is open, 1 for an unexpected failure, and 3
  when a limitation changes state.
- 24 executed cases: 23 passed and 1 known limitation remained open at the time.
  Of these, 20 were pre-registered; four were added before the run and disclosed
  in the report.

The report records one remaining false-negative class and corrects an earlier
provenance overstatement. It does not claim full end-to-end accuracy.

## U1: Unlinked exclusive conflicts

**Report:** [`reports/U1_AUDIT_REPORT.md`](../../reports/U1_AUDIT_REPORT.md)
**Closure report:** [`reports/U1_CLOSURE_REPORT.md`](../../reports/U1_CLOSURE_REPORT.md)
**Results:** [`benchmarks/results/u1_audit_results.json`](../../benchmarks/results/u1_audit_results.json),
[`u1_closure_results.json`](../../benchmarks/results/u1_closure_results.json)

Two active facts can disagree about an exclusive attribute without a structural
supersession link. `exclusive_temporal_conflicts()` reports such pairs, and
`temporal_history_violations(include_unlinked_conflicts=True)` can include them.
Detection is opt-in; default structural results are unchanged.

Verdict: **partially resolved**. The detector gap, tests, and reports are
published, but defaults, recall on noisy language, and policy persistence remain
open. The audit report records the remaining gaps; this documentation does not
upgrade the verdict.

## Phase 31: Storage and vector-backend evaluation

Phase 31A evaluated storage and paging behavior. Phase 31B and 31C evaluated
PostgreSQL and Qdrant as authoritative memory stores, including tenant isolation,
failure/recovery, and 1K/10K/100K storage-only experiments.

Historical verdict: **Qdrant was rejected as an authoritative memory store**;
canonical persistence was not replaced. The current Qdrant integration uses a
different role — a derived retrieval index that MINEMOS can rebuild and ignore.
That integration does not revise the Phase 31 verdict or claim Qdrant
superiority. Phase 31 artifacts are preserved outside the public report set; the
summary above is the durable conclusion.

## Current baseline

The current suite collects **542 tests** (541 passed and 1 skipped in the last
full local run before this documentation change). Current limitations and status
are maintained in the [README](../../README.md), not inferred from historical
phase outcomes.
