"""
run_u1_closure.py

Top-level driver for the MINEMOS U1 closure experiment.

Runs the case matrix in benchmarks/u1_closure.py and writes a separate result
file.  It never writes benchmarks/results/phase29_results.json.

Exit codes:
  0  every expected result reproduced
  1  at least one unexpected failure
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.u1_closure import (  # noqa: E402
    STATUS_ALL_EXPECTED,
    STATUS_FAILED,
    run,
)

EXIT_CODES = {
    STATUS_ALL_EXPECTED: 0,
    STATUS_FAILED: 1,
}

DEFAULT_OUTPUT = os.path.join(
    REPO_ROOT, "benchmarks", "results", "u1_closure_results.json"
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the U1 closure case matrix.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Result path (default: benchmarks/results/u1_closure_results.json)",
    )
    args = parser.parse_args(argv)

    report = run()
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps({
        "status": report["status"],
        "case_count": report["case_count"],
        "passed": report["passed"],
        "failed": report["failed"],
        "failed_ids": report["failed_ids"],
        "wall_seconds": report["wall_seconds"],
    }, indent=2))
    print(f"Wrote {out_path}")
    return EXIT_CODES[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
