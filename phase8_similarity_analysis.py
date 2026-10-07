"""
benchmarks/phase8_similarity_analysis.py
"""

from __future__ import annotations
import os, sys, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.memory_update import cosine_similarity, MATCH_THRESHOLD
from phase2.encoders import AtomicEncoder, SemanticEmbeddingEncoder
from benchmarks.phase8_dataset import PAIRS


def compute_similarities(encoder, dim=4096) -> dict:
    sims = {}
    for p in PAIRS:
        v1 = encoder.encode_full(p.c1, dim=dim)
        v2 = encoder.encode_full(p.c2, dim=dim)
        sim = cosine_similarity(v1, v2)
        sims.setdefault(p.category, []).append((p.label, sim, p.equivalent))
    return sims


def category_stats(sims: dict) -> dict:
    out = {}
    for cat, vals in sims.items():
        xs = [s for _, s, _ in vals]
        out[cat] = dict(mean=round(statistics.mean(xs), 4),
                         median=round(statistics.median(xs), 4),
                         min=round(min(xs), 4), max=round(max(xs), 4),
                         std=round(statistics.pstdev(xs), 4), n=len(xs))
    return out


def separability(sims: dict) -> dict:
    para = [s for _, s, _ in sims.get("paraphrase", [])]
    contra = [s for _, s, _ in sims.get("contradiction", [])]
    unrel = [s for _, s, _ in sims.get("unrelated", [])]

    wins = sum(1 for p in para for c in contra if p > c)
    ties = sum(1 for p in para for c in contra if p == c)
    total = len(para) * len(contra)
    auc = (wins + 0.5 * ties) / total if total else None
    pct_above_all_contra = sum(1 for p in para if p > max(contra)) / len(para) if para and contra else None

    M1 = round(statistics.mean(para) - statistics.mean(contra), 4) if para and contra else None
    M2 = round(statistics.mean(para) - statistics.mean(unrel), 4) if para and unrel else None

    return dict(auc=round(auc, 4) if auc is not None else None,
                pct_paraphrase_above_all_contradiction=round(pct_above_all_contra, 4) if pct_above_all_contra is not None else None,
                M1_paraphrase_minus_contradiction=M1, M2_paraphrase_minus_unrelated=M2)


def classification_at_threshold(sims: dict, threshold: float, label: str) -> dict:
    """Precision/recall/F1 treating 'equivalent'=True as positive class,
    evaluated across ALL pairs (not just paraphrase/contradiction)."""
    tp = fp = fn = tn = 0
    for cat, vals in sims.items():
        for _, s, equiv in vals:
            pred_equiv = s >= threshold
            if equiv and pred_equiv: tp += 1
            elif not equiv and pred_equiv: fp += 1
            elif equiv and not pred_equiv: fn += 1
            else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None
    return dict(threshold_source=label, threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=round(precision, 4) if precision is not None else None,
                recall=round(recall, 4) if recall is not None else None,
                f1=round(f1, 4) if f1 is not None else None,
                false_positive_rate=round(fpr, 4) if fpr is not None else None,
                false_negative_rate=round(fnr, 4) if fnr is not None else None)


def posthoc_threshold_sweep(sims: dict) -> list:
    all_scores = sorted(set(round(s, 3) for vals in sims.values() for _, s, _ in vals))
    return [classification_at_threshold(sims, t, "post_hoc_secondary") for t in all_scores]


def run_for_encoder(name: str, encoder) -> dict:
    sims = compute_similarities(encoder)
    stats = category_stats(sims)
    sep = separability(sims)

    predefined = classification_at_threshold(sims, MATCH_THRESHOLD, "PREDEFINED_black_hole_threshold")
    posthoc = posthoc_threshold_sweep(sims)
    best_posthoc = max((r for r in posthoc if r["f1"] is not None), key=lambda r: r["f1"], default=None)

    return dict(encoder=name, category_stats=stats, separability=sep,
                predefined_threshold_result=predefined,
                best_posthoc_result=best_posthoc)


def main():
    print(f"Dataset: {len(PAIRS)} pairs, {len(set(p.category for p in PAIRS))} categories")
    print()
    for name, enc in [("A_atomic", AtomicEncoder()),
                       ("D_spacy_semantic_embedding", SemanticEmbeddingEncoder())]:
        r = run_for_encoder(name, enc)
        print("=" * 100)
        print(f"ENCODER: {name}")
        print("=" * 100)
        for cat in ["paraphrase", "contradiction", "unrelated", "high_overlap_negative", "low_overlap_paraphrase", "exact_duplicate"]:
            if cat in r["category_stats"]:
                print(f"  {cat}: {r['category_stats'][cat]}")
        print(f"  Separability: {r['separability']}")
        print(f"  PREDEFINED threshold ({MATCH_THRESHOLD}) result: {r['predefined_threshold_result']}")
        print(f"  Best POST-HOC threshold result: {r['best_posthoc_result']}")
        print()


if __name__ == "__main__":
    main()
