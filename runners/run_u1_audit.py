"""
run_u1_audit.py

Runs the U1 independent acceptance audit and writes the result file
(default: benchmarks/results/u1_audit_results.json; pass --output for an
independent run).  Exit codes:

  0  audit completed; every section matched its expected observation
  1  audit completed but at least one section diverged from expectation
  2  audit could not run (import/runtime failure)
"""

import argparse
import json
import os
import sys
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.u1_audit import run  # noqa: E402

DEFAULT_OUTPUT = os.path.join(
    REPO_ROOT, "benchmarks", "results", "u1_audit_results.json"
)


def _summary(report):
    equivalence = report["sections"]["language_equivalence"]["totals_scored"]
    conflicts = report["sections"]["language_conflicts"]["totals"]
    return {
        "status": report["status"],
        "passed_sections": report["passed_sections"],
        "failed_sections": report["failed_sections"],
        "sections_with_known_limitations": report[
            "sections_with_known_limitations"
        ],
        "equivalence_outcome": report["sections"]["language_equivalence"][
            "outcome"
        ],
        "conflict_outcome": report["sections"]["language_conflicts"]["outcome"],
        "equivalence_pairs": report["sections"]["language_equivalence"]["pairs"],
        "equivalence_precision": equivalence["precision"],
        "equivalence_recall": equivalence["recall"],
        "equivalence_f1": equivalence["f1"],
        "equivalence_fpr": equivalence["false_positive_rate"],
        "conflict_pairs": report["sections"]["language_conflicts"]["pairs"],
        "conflict_precision": conflicts["precision"],
        "conflict_recall": conflicts["recall"],
        "conflict_f1": conflicts["f1"],
        "conflict_fpr": conflicts["false_positive_rate"],
        "wall_seconds": report["wall_seconds"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the U1 acceptance audit.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    try:
        report = run()
    except Exception:  # noqa: BLE001 - the audit must report, not crash
        traceback.print_exc()
        return 2

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(_summary(report), indent=2))
    print(f"Wrote {out_path}")
    return 0 if not report["failed_sections"] else 1


if __name__ == "__main__":
    sys.exit(main())
