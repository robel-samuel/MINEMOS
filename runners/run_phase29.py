"""
run_phase29.py

Top-level driver for MINEMOS Phase 29:
Temporal Invariant Coverage and Verifier Telemetry.

Runs the case matrix in benchmarks/phase29_integrity.py and writes the result
file (canonical by default; pass --output for an audit run).  Each case records
its provenance, so additional cases are never presented as pre-registered.

Exit codes make the four benchmark outcomes explicit:
  0  all expected results reproduced; no documented known gap is open
  1  at least one unexpected failure
  2  no unexpected failures, but a documented known limitation is still open
  3  a documented known limitation changed state (resolved); expectations need
     review
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.phase29_integrity import (  # noqa: E402
    STATUS_ALL_EXPECTED,
    STATUS_FAILED,
    STATUS_GAP_CHANGED,
    STATUS_GAP_OPEN,
    run,
)

EXIT_CODES = {
    STATUS_ALL_EXPECTED: 0,
    STATUS_FAILED: 1,
    STATUS_GAP_OPEN: 2,
    STATUS_GAP_CHANGED: 3,
}

DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "benchmarks", "results", "phase29_results.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Phase 29 case matrix.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Result path (default: benchmarks/results/phase29_results.json)",
    )
    args = parser.parse_args(argv)

    report = run()
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps({
        "status": report["status"],
        "verdict_counts": report["verdict_counts"],
        "provenance_counts": report["provenance_counts"],
        "case_count": report["case_count"],
        "passed": report["passed"],
        "known_gaps_confirmed": report["known_gaps_confirmed"],
        "failed": report["failed"],
        "failed_ids": report["failed_ids"],
        "known_gap_ids": report["known_gap_ids"],
        "metrics": report["metrics"],
        "wall_seconds": report["wall_seconds"],
    }, indent=2))
    print(f"Wrote {out_path}")
    if report["status"] == STATUS_GAP_OPEN:
        print(
            "NOTE: known limitation(s) still open; this run is NOT a full pass."
        )
    return EXIT_CODES[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
