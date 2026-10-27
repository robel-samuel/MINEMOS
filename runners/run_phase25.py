"""
run_phase25.py

Phase 25 Implementation and Validation Runner.

Phase 25 built BlackHoleMemory — the unified memory engine combining:
  - Hybrid candidate retrieval (Dense bi-encoder + BM25, fused k=10)
  - Bidirectional NLI verification gate
  - Temporal validity windows and supersession
  - Drift-resistant bounded consolidation
  - Tiered retention policy (Tier 0 provisional / Tier 1 consolidated)

This runner re-executes the Phase 25 end-to-end validation workload and
saves machine-readable results to benchmarks/results/phase25_e2e_results.json.
"""

from __future__ import annotations
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.phase25_e2e import run as run_benchmark


def main():
    ok = run_benchmark()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
