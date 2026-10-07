"""
Scores an extractor against benchmarks/uncertainty_eval_set.py along five
independent dimensions, per Step 3's requirement not to collapse
everything into one number.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.uncertainty_eval_set import EVAL_CASES


def _confidence_band(candidate) -> str:
    if candidate is None:
        return "missing"
    if candidate.negated:
        return "rejected"
    if candidate.confidence is not None and candidate.confidence <= 0.3:
        return "low"
    return "high"


def score_extractor(extractor, default_subject="user"):
    n = len(EVAL_CASES)
    results = {
        "subject_correct": 0, "predicate_correct": 0, "object_correct": 0,
        "confidence_band_correct": 0, "negation_correct": 0,
        "overall_correct": 0, "missing": 0,
    }
    per_case = []

    for case in EVAL_CASES:
        c = extractor.extract_sentence(case["text"])
        row = {"text": case["text"], "expected": case, "got": c}

        if c is None:
            results["missing"] += 1
            per_case.append(row)
            continue

        got_subject = c.subject or default_subject
        subj_ok = got_subject == case["subject"]
        pred_ok = c.predicate == case["predicate"]
        obj_ok = (case["object"].lower() in c.object.lower()
                  or c.object.lower() in case["object"].lower())
        band_ok = _confidence_band(c) == case["confidence_band"]
        neg_ok = c.negated == case["negated"]
        overall_ok = subj_ok and pred_ok and obj_ok and band_ok and neg_ok

        results["subject_correct"] += subj_ok
        results["predicate_correct"] += pred_ok
        results["object_correct"] += obj_ok
        results["confidence_band_correct"] += band_ok
        results["negation_correct"] += neg_ok
        results["overall_correct"] += overall_ok

        row.update(subj_ok=subj_ok, pred_ok=pred_ok, obj_ok=obj_ok,
                    band_ok=band_ok, neg_ok=neg_ok, overall_ok=overall_ok)
        per_case.append(row)

    accuracy = {
        "subject_accuracy": results["subject_correct"] / n,
        "predicate_accuracy": results["predicate_correct"] / n,
        "object_accuracy": results["object_correct"] / n,
        "confidence_band_accuracy": results["confidence_band_correct"] / n,
        "negation_accuracy": results["negation_correct"] / n,
        "overall_accuracy": results["overall_correct"] / n,
        "missing_rate": results["missing"] / n,
        "n_cases": n,
    }
    return accuracy, per_case


def print_failures(per_case, category_filter=None):
    for row in per_case:
        exp = row["expected"]
        if category_filter and exp["confidence_band"] != category_filter:
            continue
        got = row["got"]
        ok = row.get("overall_ok", False)
        if ok:
            continue
        print(f"  FAIL: {row['text']!r}")
        print(f"    expected: predicate={exp['predicate']!r} object={exp['object']!r} "
              f"band={exp['confidence_band']!r} subject={exp['subject']!r}")
        if got is None:
            print(f"    got:      None (extraction failed entirely)")
        else:
            print(f"    got:      predicate={got.predicate!r} object={got.object!r} "
                  f"confidence={got.confidence!r} subject={got.subject!r} negated={got.negated}")
