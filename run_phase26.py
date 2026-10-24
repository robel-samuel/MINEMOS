"""
run_phase26.py

Top-level entrypoint for executing MINEMOS Phase 26 research experiments:
  - Experiment A: NLI Threshold Calibration Grid Sweep
  - Experiment B: Guarded Dense Similarity Fallback Sweep
  - Experiment C: Candidate Comparison on Validation Split
  - Experiment D: End-to-End Streaming Memory Evaluation
  - Experiment E: Final Frozen Held-Out Benchmark Evaluation
"""

import sys
import time

sys.path.insert(0, ".")

from benchmarks.phase26_experiment import run_full_phase26

if __name__ == "__main__":
    t0 = time.time()
    results = run_full_phase26()
    elapsed = time.time() - t0
    print(f"\nPhase 26 experiment execution finished in {elapsed:.1f}s.")
