"""Measures exact fact-value extraction on the frozen V4 evaluation set.

The set is synthetic and authored independently of the implementation: every
row states the fact a careful reader would record, the exact value, and the
evidence text from the original message.  Nothing here is derived from what
the resolver currently returns.

Reported independently, because they answer different questions:

* ``recall``   — of the facts a reader would record, how many were extracted
  with the right subject, predicate, polarity and value;
* ``precision`` — of the facts the resolver emitted, how many were right;
* ``value_exactness`` — strict equality of the value string for matched facts;
* ``evidence_exactness`` — the evidence text equals the source text, and the
  evidence span addresses it;
* ``no_fact_compliance`` — messages that assert nothing (hypothetical,
  interrogative, intention, opinion, unsupported) emit no fact.

An emitted fact with the wrong value is a false positive *and* leaves the
expected fact unmatched: over-capture costs both numbers.  That is the point.

Usage (from the repository root)::

    python evaluation/resolution/v4_value_extraction/measure_value_extraction.py
    python evaluation/resolution/v4_value_extraction/measure_value_extraction.py \
        --ablate-v4-trims
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pathlib
import re
import sys
from typing import Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minemos.resolution import Resolver, SourceMessage  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
CASES = HERE / "cases.jsonl"
MANIFEST = HERE / "manifest.json"
RESULTS = HERE / "results" / "v4_value_extraction.json"

_SPACE = re.compile(r"\s+")


def _canon(text: str) -> str:
    return _SPACE.sub(" ", text.casefold()).strip()


def _subject_key(subject) -> str:
    if subject.kind == "speaker":
        return "speaker"
    if subject.kind == "derived" and subject.relation:
        return f"derived:{subject.relation}"
    if subject.kind == "named":
        return f"named:{_canon(subject.label)}"
    return f"{subject.kind}:{_canon(subject.label)}"


def _expected_subject(spec: str) -> str:
    return spec


def _expected_key(item: dict) -> Tuple[str, str, str]:
    return (item["predicate"], _expected_subject(item["subject"]),
            item["polarity"])


def _emitted_key(fact) -> Tuple[str, str, str]:
    return (fact.predicate, _subject_key(fact.subject), fact.polarity)


def load_cases() -> List[dict]:
    digest = hashlib.sha256(CASES.read_bytes()).hexdigest()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if digest != manifest["sha256"]:
        raise SystemExit(
            f"frozen set changed: cases.jsonl sha256 {digest} does not match "
            f"manifest {manifest['sha256']}")
    return [json.loads(line) for line in
            CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ablate_v4_trims() -> None:
    """Disables only the V4 trailing trims, to show the set is sensitive.

    This is an ablation of the current source, not a V3 build: nothing else
    changes.
    """
    # The package re-exports the ``interpret`` *function* under the same name
    # as its module, so the module is fetched from sys.modules directly.
    interpret = importlib.import_module("minemos.resolution.interpret")

    interpret._TRAILING_LANGUAGE = re.compile(r"(?!x)x")
    interpret._strip_new_clause = lambda value: value


def evaluate(cases: List[dict]) -> dict:
    rows = []
    counters = {
        "expected": 0, "emitted": 0, "matched": 0,
        "value_exact": 0, "evidence_exact": 0, "grounded": 0,
        "no_fact_cases": 0, "no_fact_clean": 0,
    }
    categories: Dict[str, dict] = {}
    failures = []

    for case in cases:
        cid = case["case_id"]
        expected = case["expect"]
        resolver = Resolver()
        record = resolver.resolve(SourceMessage(
            conversation_id=f"v4-{cid}", message_id=f"{cid}-m0",
            text=case["text"], speaker="speaker", timestamp=1.0, order=0)).record
        emitted = list(record.facts)

        cat = categories.setdefault(case["category"], {
            "cases": 0, "expected": 0, "emitted": 0, "matched": 0,
            "no_fact_cases": 0, "no_fact_clean": 0})
        cat["cases"] += 1
        cat["expected"] += len(expected)
        cat["emitted"] += len(emitted)
        counters["expected"] += len(expected)
        counters["emitted"] += len(emitted)

        if not expected:
            counters["no_fact_cases"] += 1
            cat["no_fact_cases"] += 1
            if not emitted:
                counters["no_fact_clean"] += 1
                cat["no_fact_clean"] += 1
            else:
                failures.append({
                    "case_id": cid, "category": case["category"],
                    "kind": "unexpected_fact",
                    "reason": case.get("no_fact_reason"),
                    "got": [{"predicate": f.predicate, "value": f.value}
                            for f in emitted],
                })
            rows.append({"case_id": cid, "category": case["category"],
                         "expected": 0, "emitted": len(emitted),
                         "matched": 0})
            continue

        remaining = list(emitted)
        case_matches = []
        case_misses = []
        for item in expected:
            key = _expected_key(item)
            found = None
            for fact in remaining:
                if _emitted_key(fact) != key:
                    continue
                if _canon(fact.value) != _canon(item["value"]):
                    continue
                found = fact
                break
            if found is None:
                case_misses.append(item)
                continue
            remaining.remove(found)
            case_matches.append((item, found))

        counters["matched"] += len(case_matches)
        cat["matched"] += len(case_matches)
        for item, fact in case_matches:
            if fact.value == item["value"]:
                counters["value_exact"] += 1
            evidence_text = " ".join(e.text for e in fact.evidence)
            if _canon(evidence_text) == _canon(item["evidence"]):
                counters["evidence_exact"] += 1
            source = case["text"]
            if all(source[e.start:e.end] == e.text for e in fact.evidence):
                counters["grounded"] += 1
            else:
                failures.append({
                    "case_id": cid, "category": case["category"],
                    "kind": "ungrounded_evidence",
                    "expected": item["evidence"],
                    "got": [(e.start, e.end, e.text) for e in fact.evidence],
                })
            if fact.value != item["value"]:
                failures.append({
                    "case_id": cid, "category": case["category"],
                    "kind": "value_not_exact",
                    "predicate": fact.predicate,
                    "expected": item["value"], "got": fact.value,
                })
        for item in case_misses:
            failures.append({
                "case_id": cid, "category": case["category"],
                "kind": "missing_fact", "expected": item,
                "known_gap": case.get("known_gap"),
            })

        # Every emitted fact that no expected fact claimed is a false positive.
        for fact in remaining:
            similar = [item for item in expected
                       if item["predicate"] == fact.predicate
                       and _expected_subject(item["subject"])
                       == _subject_key(fact.subject)]
            if similar:
                kind = "wrong_value"
                expected_values = [item["value"] for item in similar]
            elif any(_canon(item["value"]) == _canon(fact.value)
                     for item in expected):
                kind = "wrong_subject_or_polarity"
                expected_values = None
            else:
                kind = "extra_fact"
                expected_values = None
            failures.append({
                "case_id": cid, "category": case["category"], "kind": kind,
                "predicate": fact.predicate, "value": fact.value,
                "subject": _subject_key(fact.subject),
                "expected_value": expected_values,
            })

        rows.append({"case_id": cid, "category": case["category"],
                     "expected": len(expected), "emitted": len(emitted),
                     "matched": len(case_matches)})

    def ratio(part: int, whole: int) -> Optional[float]:
        return round(part / whole, 4) if whole else None

    summary = {
        "cases": len(cases),
        "expected_facts": counters["expected"],
        "emitted_facts": counters["emitted"],
        "matched_facts": counters["matched"],
        "recall": ratio(counters["matched"], counters["expected"]),
        "precision": ratio(counters["matched"], counters["emitted"]),
        "value_exactness": ratio(counters["value_exact"],
                                 counters["matched"]),
        "evidence_exactness": ratio(counters["evidence_exact"],
                                    counters["matched"]),
        "evidence_grounded": ratio(counters["grounded"], counters["matched"]),
        "no_fact_compliance": ratio(counters["no_fact_clean"],
                                    counters["no_fact_cases"]),
        "no_fact_cases": counters["no_fact_cases"],
    }
    for name, cat in categories.items():
        cat["precision"] = ratio(cat["matched"], cat["emitted"])
        cat["recall"] = ratio(cat["matched"], cat["expected"])
        cat["no_fact_compliance"] = ratio(cat["no_fact_clean"],
                                          cat["no_fact_cases"])
    return {
        "dataset": "v4_value_extraction",
        "summary": summary,
        "by_category": dict(sorted(categories.items())),
        "cases": rows,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablate-v4-trims", action="store_true",
                        help="disable the V4 trailing trims (ablation)")
    parser.add_argument("--json", action="store_true",
                        help="print the full report as JSON")
    args = parser.parse_args()

    cases = load_cases()
    label = "v4"
    if args.ablate_v4_trims:
        _ablate_v4_trims()
        label = "v4-ablation-no-trailing-trims"

    report = evaluate(cases)
    report["variant"] = label
    summary = report["summary"]
    results_path = RESULTS if not args.ablate_v4_trims else RESULTS.with_name(
        "v4_value_extraction_ablation.json")
    print(f"variant {label}")
    print(f"cases {summary['cases']} expected {summary['expected_facts']} "
          f"emitted {summary['emitted_facts']} "
          f"matched {summary['matched_facts']}")
    print(f"recall {summary['recall']} precision {summary['precision']}")
    print(f"value_exactness {summary['value_exactness']} "
          f"evidence_exactness {summary['evidence_exactness']} "
          f"evidence_grounded {summary['evidence_grounded']}")
    print(f"no_fact_compliance {summary['no_fact_compliance']} "
          f"({summary['no_fact_cases']} cases)")
    for name, cat in report["by_category"].items():
        print(f"  {name:<28} cases {cat['cases']:>2} "
              f"matched {cat['matched']}/{cat['expected']} "
              f"precision {cat['precision']} recall {cat['recall']}")
    if report["failures"]:
        print(f"failures {len(report['failures'])}")
        for failure in report["failures"]:
            print("  " + json.dumps(failure, ensure_ascii=False,
                                    sort_keys=True))
    else:
        print("failures 0")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                            + "\n", encoding="utf-8")
    print(f"wrote {results_path.relative_to(ROOT)}")
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
