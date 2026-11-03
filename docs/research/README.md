# Research

MINEMOS is an experimental platform. The research program asks how a
fixed-capacity memory should consolidate, verify, retain, and retrieve factual
knowledge from a stream of observations, and where end-to-end failures actually
originate.

Historical results are evidence, not current production guarantees. They were
produced on synthetic datasets with known ground truth. Reports and benchmark
JSON files are preserved unchanged as the research record.

## How the work is organized

| Layer | Contents |
|---|---|
| [Phase history](history.md) | What each phase asked, changed, and measured |
| [Benchmark methodology](benchmark-methodology.md) | Failure attribution schema, benchmark conditions, Qdrant tables, caveats |
| [`reports/`](../../reports/) | Per-phase reports, audits, and security reviews |
| [`benchmarks/results/`](../../benchmarks/results/) | Machine-readable result artifacts |
| [`runners/`](../../runners/) | Reproducible phase runners |
| [`benchmarks/`](../../benchmarks/) | Experiment code and datasets |

## Phase map

| Phases | Focus |
|---|---|
| 1–6 | Structured fact storage, confidence, contradiction, temporal validity, thresholds |
| 7–8 | Encoder ablation: atomic, field-aware lexical, semantic |
| 9–10 | NLI integration and audit |
| 11–12 | Scaling benchmarks and structured addressing |
| 13–16 | Semantic NLI gate and system ablations |
| 17–19 | Utility-aware eviction, dense retrieval, contradiction-aware utility |
| 20–21 | Adaptive anchors and anchor-beam integration |
| 22 | Hybrid BM25 + dense retrieval |
| 23 | Causal failure attribution and retention oracle |
| 24 | Tiered retention, calibrated verification, drift-resistant consolidation |
| 25 | Unified `BlackHoleMemory` engine |
| 26 | Calibrated bidirectional verification and safety guards |
| 27 | Adaptive retrieval under long-horizon interference |
| 28 | Phase 23 failure closure and end-to-end integrity |
| 29 | Temporal invariant coverage and verifier telemetry |
| U1 | Unlinked exclusive-conflict detection |
| 31A–31C | Storage/vector-backend evaluation; Qdrant rejected as authoritative store |

## Research integrity rules

- Frozen baseline parameters are experimental controls. Changing one requires a
  new phase, a baseline comparison, and documented results.
- Positive and negative findings are both reported.
- Experimental results are not presented as production guarantees.
- Benchmark JSON and historical reports are not edited to improve outcomes.
- Claims about implemented behavior are checked against source code.

Contribution rules are in [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Scope boundaries

The current engine is validated on synthetic observation streams. It is not
validated for:

- open-domain conversational memory or multilingual data;
- 100K-scale production workloads with latency SLOs;
- high-availability or multi-region deployment;
- decoding memory back into free-form text (query returns slot text and vectors,
  not generated language).

See [Known limitations in the README](../../README.md#limitations) for the
current list.
