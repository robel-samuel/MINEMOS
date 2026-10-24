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
saves machine-readable results to scratch/phase25_e2e_results.json.

Note: The primary Phase 25 benchmark script is scratch/run_phase25_e2e.py.
This top-level runner delegates to it for consistency with other phase runners.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def main():
    # The full Phase 25 E2E harness lives in scratch/run_phase25_e2e.py.
    # Import and invoke its main() to keep the benchmark self-contained.
    import importlib.util

    script_path = os.path.join(os.path.dirname(__file__), "scratch", "run_phase25_e2e.py")
    if not os.path.exists(script_path):
        print(
            f"[Phase 25] ERROR: benchmark script not found at {script_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("run_phase25_e2e", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "main"):
        mod.main()
    else:
        print(
            "[Phase 25] Benchmark script executed (no explicit main() entry point).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
