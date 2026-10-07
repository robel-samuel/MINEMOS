"""
benchmarks/phase7_similarity_analysis.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.memory_update import cosine_similarity
from phase2.encoders import AtomicEncoder, FieldAwareLexicalEncoder, TfidfLexicalEncoder
from benchmarks.phase7_dataset import PAIRS


def compute_similarities(encoder, dim: int = 4096) -> dict:
    sims_by_category = {}
    for pair in PAIRS:
        v1 = encoder.encode_full(pair.c1, dim=dim)
        v2 = encoder.encode_full(pair.c2, dim=dim)
        sim = cosine_similarity(v1, v2)
        sims_by_category.setdefault(pair.category, []).append((pair.label, sim))
    return sims_by_category


def category_means(sims_by_category: dict) -> dict:
    return {cat: round(sum(s for _, s in vals) / len(vals), 4)
            for cat, vals in sims_by_category.items()}


def paraphrase_vs_contradiction_auc(sims_by_category: dict) -> dict:
    """Simple AUC via pairwise comparison (Mann-Whitney U statistic)."""
    paraphrase_sims = [s for _, s in sims_by_category.get("paraphrase", [])]
    contradiction_sims = [s for _, s in sims_by_category.get("contradiction", [])]
    if not paraphrase_sims or not contradiction_sims:
        return {}
    wins = sum(1 for p in paraphrase_sims for c in contradiction_sims if p > c)
    ties = sum(1 for p in paraphrase_sims for c in contradiction_sims if p == c)
    total = len(paraphrase_sims) * len(contradiction_sims)
    auc = (wins + 0.5 * ties) / total
    pct_above_every_contradiction = sum(
        1 for p in paraphrase_sims if p > max(contradiction_sims)
    ) / len(paraphrase_sims)
    return dict(auc=round(auc, 4),
                pct_paraphrases_above_every_contradiction=round(pct_above_every_contradiction, 4),
                max_contradiction_sim=round(max(contradiction_sims), 4),
                min_paraphrase_sim=round(min(paraphrase_sims), 4))


def threshold_search(sims_by_category: dict) -> list:
    """For candidate thresholds, compute precision/recall treating
    'paraphrase' as positive and 'contradiction'+'unrelated'+'lexical_distractor'
    as negative."""
    positives = [s for _, s in sims_by_category.get("paraphrase", [])]
    negatives = ([s for _, s in sims_by_category.get("contradiction", [])] +
                 [s for _, s in sims_by_category.get("unrelated", [])] +
                 [s for _, s in sims_by_category.get("lexical_distractor", [])])
    candidates = sorted(set(round(s, 3) for s in positives + negatives))
    results = []
    for t in candidates:
        tp = sum(1 for s in positives if s >= t)
        fn = sum(1 for s in positives if s < t)
        fp = sum(1 for s in negatives if s >= t)
        tn = sum(1 for s in negatives if s < t)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        results.append(dict(threshold=t, tp=tp, fn=fn, fp=fp, tn=tn,
                             precision=round(precision, 4) if precision is not None else None,
                             recall=round(recall, 4) if recall is not None else None))
    return results


def run_for_encoder(name: str, encoder) -> dict:
    sims = compute_similarities(encoder)
    means = category_means(sims)
    auc_info = paraphrase_vs_contradiction_auc(sims)
    thresh = threshold_search(sims)

    m1 = means.get("paraphrase", 0) - means.get("contradiction", 0)
    m2 = means.get("paraphrase", 0) - means.get("unrelated", 0)

    return dict(encoder=name, category_means=means,
                separation_margin_M=round(m1, 4), separation_margin_M2=round(m2, 4),
                auc_info=auc_info, threshold_table=thresh)


def main():
    print(f"Dataset: {len(PAIRS)} pairs across "
          f"{len(set(p.category for p in PAIRS))} categories")
    print()

    encoders = [
        ("A_atomic", AtomicEncoder()),
        ("B_field_aware_lexical", FieldAwareLexicalEncoder()),
    ]
    corpus = [f"{p.c1.subject} {p.c1.predicate} {p.c1.object}" for p in PAIRS] + \
             [f"{p.c2.subject} {p.c2.predicate} {p.c2.object}" for p in PAIRS]
    encoders.append(("C_lexical_statistical_NOT_semantic", TfidfLexicalEncoder(corpus)))

    all_results = {}
    for name, enc in encoders:
        r = run_for_encoder(name, enc)
        all_results[name] = r
        print("=" * 100)
        print(f"ENCODER: {name}")
        print("=" * 100)
        print("Category means:", r["category_means"])
        print(f"Separation margin M (paraphrase - contradiction) = {r['separation_margin_M']}")
        print(f"Separation margin M2 (paraphrase - unrelated)    = {r['separation_margin_M2']}")
        print("Pair-level separability:", r["auc_info"])
        print()

    return all_results


if __name__ == "__main__":
    main()
