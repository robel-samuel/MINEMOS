"""
run_phase28.py

Phase 28: Phase 23 Failure Closure & End-to-End Memory Integrity Validation Runner.

Executes the full Phase 28 quantitative benchmark suite, which closes the five
unresolved failure modes identified in Phase 23:

  Benchmark 1 — Retention Under Capacity Pressure (C=25, N=320)
      Category D: Tiered retention eliminates premature slot eviction.

  Benchmark 2 — Paraphrase Verification & False Merge Rate
      Category B: Calibrated bidirectional NLI verification improves paraphrase
                  acceptance from 68.6% → 86.7% while maintaining 0% FMR.

  Benchmark 3 — Retrieval Under Interference & Distractor Overload
      Category A/E: Hybrid BM25 + Dense fusion achieves R@1 = MRR = 1.000
                    across all 4 interference conditions.

  Benchmark 4 — Representation Consolidation Drift
      Category C: Anchor-bounded hybrid consolidation reduces drift by 63.65%
                  relative to baseline EMA.

  Benchmark 5 — False Contradictions & Temporal Supersession
      Category F / Phase 27 Defect: Entity-grounding guard eliminates false
                                     contradictions on independent facts (100% → 0%).

  Benchmark 6 — Integrated Held-Out Generalization (Phase 27 held-out stream)
      All improvements generalize to unseen entities and relations.

Results are saved incrementally to benchmarks/results/phase28_results.json.
Expected runtime: 30–60 minutes on 4 CPU cores.
"""

from __future__ import annotations
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.phase28_closure import main as run_benchmark


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
