"""
benchmarks/phase26_verification.py

Phase 26 Semantic Verification and Contradiction Safety Engine.

Provides:
  1. Phase 25 Baseline Verifier (reproducible, frozen)
  2. Phase 26 Calibrated Bidirectional Verifier (Experiment A)
  3. Guarded Dense Similarity Fallback (Experiment B)
  4. Combined Verification Pipeline (Experiment C)

Strict Safety Invariants:
  - Similarity alone never establishes semantic equivalence without semantic checks.
  - Hard contradiction signals (p_contra >= tau_hard) can NEVER be overridden.
  - Negation parity mismatch immediately vetos fallback.
  - Numerical / scalar mismatch immediately vetos fallback.
  - Named entity mismatch immediately vetos fallback.
"""

from __future__ import annotations
import re
import time
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import torch

from benchmarks.phase16_experiment import NliSemanticGate
from benchmarks.phase18_experiment import DenseSemanticEncoder

# Negation lexicon for deterministic guard
NEGATION_TOKENS = {
    "not", "never", "no", "neither", "nor", "none", "n't",
    "cannot", "can't", "won't", "don't", "doesn't", "didn't",
    "failed", "refused", "denied", "without", "no longer"
}

# Temporal state transition markers
TEMPORAL_MARKERS = {
    "now", "currently", "became", "promoted", "relocated", "switched",
    "transitioned", "resigned", "graduated", "returned", "former", "formerly"
}


def extract_numbers(text: str) -> Set[str]:
    """Extracts normalized numbers, percentages, and currencies from text."""
    # Match integers, floats, percentages, e.g., '10', '10.5', '15%'
    raw = re.findall(r"\b\d+(?:\.\d+)?%?", text.lower())
    # Word numbers for small values
    word_nums = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "first": "1", "second": "2", "third": "3"
    }
    tokens = text.lower().split()
    for t in tokens:
        clean = re.sub(r"[^\w]", "", t)
        if clean in word_nums:
            raw.append(word_nums[clean])
    return set(raw)


def extract_negation_markers(text: str) -> Set[str]:
    """Extracts negation markers present in text."""
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return {t for t in tokens if t in NEGATION_TOKENS}


def check_entity_mismatch(premise: str, hypothesis: str) -> bool:
    """
    Checks if capitalized named entities differ between premise and hypothesis.
    Returns True if an entity mismatch is detected (unsafe to merge).
    """
    # Extract capitalized words (excluding sentence start)
    words_p = premise.split()
    words_h = hypothesis.split()

    caps_p = {re.sub(r"[^\w]", "", w) for w in words_p[1:] if w and w[0].isupper() and len(w) > 2}
    caps_h = {re.sub(r"[^\w]", "", w) for w in words_h[1:] if w and w[0].isupper() and len(w) > 2}

    # If both have prominent named entities, check if there is an explicit collision
    # e.g., {'Nexus', 'Labs'} vs {'Apex', 'Systems'} or {'Tokyo'} vs {'Berlin'}
    if caps_p and caps_h:
        diff_p = caps_p - caps_h
        diff_h = caps_h - caps_p
        if diff_p and diff_h:
            # Mutual disjoint proper nouns indicates entity conflict
            return True
    return False


def semantic_safety_guards_pass(
    premise: str,
    hypothesis: str,
    p_contra_max: float
) -> Tuple[bool, str]:
    """
    Evaluates deterministic safety guards before allowing any dense similarity fallback.
    
    Returns:
      (passed, reason)
    """
    # Guard 1: Hard contradiction signal from NLI
    if p_contra_max >= 0.70:
        return False, f"NLI contradiction too high (p_contra={p_contra_max:.3f} >= 0.70)"

    # Guard 2: Negation mismatch
    neg_p = extract_negation_markers(premise)
    neg_h = extract_negation_markers(hypothesis)
    if bool(neg_p) != bool(neg_h):
        return False, f"Negation mismatch: P={neg_p} vs H={neg_h}"

    # Guard 3: Numerical / scalar mismatch
    nums_p = extract_numbers(premise)
    nums_h = extract_numbers(hypothesis)
    if nums_p and nums_h and nums_p != nums_h:
        return False, f"Numerical mismatch: P={nums_p} vs H={nums_h}"

    # Guard 4: Conflicting named entities
    if check_entity_mismatch(premise, hypothesis):
        return False, "Conflicting named entities detected"

    return True, "All safety guards passed"


class Phase26Verifier:
    """
    Unified Verification System supporting all experimental modes:
      - 'baseline': Unmodified Phase 25 verification
      - 'calibrated': Experiment A (calibrated bidirectional aggregation & thresholds)
      - 'dense_fallback': Experiment B (guarded dense fallback on baseline)
      - 'combined': Experiment C (calibrated + guarded dense fallback)
    """

    def __init__(
        self,
        nli_gate: Optional[NliSemanticGate] = None,
        dense_encoder: Optional[DenseSemanticEncoder] = None,
        # Default Phase 25 baseline parameters
        baseline_tau_entail: float = 0.60,
        baseline_tau_contra: float = 0.40,
        baseline_asym_margin: float = 0.15,
        # Experiment A calibrated parameters
        cal_tau_entail: float = 0.55,
        cal_tau_bilateral_contra: float = 0.40,
        cal_tau_hard_contra: float = 0.75,
        cal_asym_margin: float = 0.15,
        # Experiment B dense fallback parameters
        fallback_tau_dense: float = 0.88,
        fallback_min_entail: float = 0.20
    ):
        self.nli_gate = nli_gate or NliSemanticGate()
        self.dense_encoder = dense_encoder or DenseSemanticEncoder()

        # Baseline
        self.baseline_tau_entail = baseline_tau_entail
        self.baseline_tau_contra = baseline_tau_contra
        self.baseline_asym_margin = baseline_asym_margin

        # Experiment A
        self.cal_tau_entail = cal_tau_entail
        self.cal_tau_bilateral_contra = cal_tau_bilateral_contra
        self.cal_tau_hard_contra = 0.70
        self.cal_asym_margin = cal_asym_margin


        # Experiment B
        self.fallback_tau_dense = fallback_tau_dense
        self.fallback_min_entail = fallback_min_entail

    def verify_pair(
        self,
        premise: str,
        hypothesis: str,
        mode: str = "baseline",
        tau_entail: Optional[float] = None,
        tau_contra: Optional[float] = None,
        asym_margin: Optional[float] = None,
        tau_dense: Optional[float] = None,
        entail_threshold: Optional[float] = None,
        contra_threshold: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluates a single pair of statements."""
        res = self.verify_batch([(premise, hypothesis)], mode=mode,
                                tau_entail=tau_entail, tau_contra=tau_contra,
                                asym_margin=asym_margin, tau_dense=tau_dense,
                                entail_threshold=entail_threshold,
                                contra_threshold=contra_threshold,
                                **kwargs)
        return res[0]

    def verify_batch(
        self,
        pairs: List[Tuple[str, str]],
        mode: str = "baseline",
        tau_entail: Optional[float] = None,
        tau_contra: Optional[float] = None,
        asym_margin: Optional[float] = None,
        tau_dense: Optional[float] = None,
        entail_threshold: Optional[float] = None,
        contra_threshold: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batched verification across candidate pairs.
        Evaluates forward and reverse NLI, computes dense similarity,
        and executes the requested decision pipeline.
        """
        if entail_threshold is not None and tau_entail is None:
            tau_entail = entail_threshold
        if contra_threshold is not None and tau_contra is None:
            tau_contra = contra_threshold

        if not pairs:
            return []

        t0 = time.perf_counter()

        # 1. Bidirectional NLI predictions
        fwd_pairs = pairs
        rev_pairs = [(h, p) for p, h in pairs]
        all_pairs = fwd_pairs + rev_pairs

        raw_all = self.nli_gate.predict_batch(all_pairs)
        n = len(pairs)
        fwd_res = raw_all[:n]
        rev_res = raw_all[n:]

        # 2. Dense similarities (computed only if mode uses fallback, or for logging)
        dense_sims = []
        if mode in ("dense_fallback", "combined"):
            for p, h in pairs:
                e_p = self.dense_encoder.encode(p)
                e_h = self.dense_encoder.encode(h)
                sim = float(np.dot(e_p, e_h))
                dense_sims.append(sim)
        else:
            dense_sims = [0.0] * n

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_pair_ms = elapsed_ms / n

        results = []
        for i, ((premise, hypothesis), fwd, rev, sim) in enumerate(zip(pairs, fwd_res, rev_res, dense_sims)):
            p_e_fwd, p_c_fwd = fwd["p_entail"], fwd["p_contra"]
            p_e_rev, p_c_rev = rev["p_entail"], rev["p_contra"]

            max_contra = max(p_c_fwd, p_c_rev)
            min_contra = min(p_c_fwd, p_c_rev)
            max_entail = max(p_e_fwd, p_e_rev)
            min_entail = min(p_e_fwd, p_e_rev)
            conf = max(fwd["confidence"], rev["confidence"])

            decision = "NEUTRAL"
            decision_path = "default_neutral"
            fallback_applied = False
            fallback_reason = ""

            # ─────────────────────────────────────────────────────────────
            # Mode 1: Baseline (Unmodified Phase 25 Logic)
            # ─────────────────────────────────────────────────────────────
            if mode in ("baseline", "bidirectional"):
                t_e = tau_entail if tau_entail is not None else self.baseline_tau_entail
                t_c = tau_contra if tau_contra is not None else self.baseline_tau_contra
                m_a = asym_margin if asym_margin is not None else self.baseline_asym_margin

                # Strict max_contra veto
                if max_contra >= t_c or (p_c_fwd >= 0.35 and p_c_rev >= 0.35):
                    decision = "CONTRADICTION"
                    decision_path = "baseline_contra_veto"
                elif max_entail >= t_e and (max_entail - max_contra >= m_a):
                    decision = "SAME"
                    decision_path = "baseline_entail_accept"
                else:
                    decision = "NEUTRAL"
                    decision_path = "baseline_neutral"

            # ─────────────────────────────────────────────────────────────
            # Mode 2: Calibrated (Experiment A)
            # ─────────────────────────────────────────────────────────────
            elif mode == "calibrated":
                t_e = tau_entail if tau_entail is not None else self.cal_tau_entail
                t_bi_c = self.cal_tau_bilateral_contra
                t_hard_c = self.cal_tau_hard_contra
                m_a = asym_margin if asym_margin is not None else self.cal_asym_margin

                # Calibrated Contradiction Veto:
                # 1. Bilateral contradiction: both directions show contradiction signal
                is_bilateral_contra = (min_contra >= t_bi_c) or (p_c_fwd >= 0.35 and p_c_rev >= 0.35)
                # 2. Hard unilateral contradiction: one direction is extreme
                is_hard_unilateral_contra = (max_contra >= t_hard_c)

                if is_bilateral_contra or is_hard_unilateral_contra:
                    decision = "CONTRADICTION"
                    decision_path = "calibrated_contra_veto"
                elif max_entail >= t_e and (max_entail - min_contra >= m_a):
                    decision = "SAME"
                    decision_path = "calibrated_entail_accept"
                else:
                    decision = "NEUTRAL"
                    decision_path = "calibrated_neutral"

            # ─────────────────────────────────────────────────────────────
            # Mode 3: Dense Fallback on Baseline (Experiment B)
            # ─────────────────────────────────────────────────────────────
            elif mode == "dense_fallback":
                t_e = tau_entail if tau_entail is not None else self.baseline_tau_entail
                t_c = tau_contra if tau_contra is not None else self.baseline_tau_contra
                m_a = asym_margin if asym_margin is not None else self.baseline_asym_margin
                t_d = tau_dense if tau_dense is not None else self.fallback_tau_dense

                # Run baseline first
                if max_contra >= t_c or (p_c_fwd >= 0.35 and p_c_rev >= 0.35):
                    raw_dec = "CONTRADICTION"
                elif max_entail >= t_e and (max_entail - max_contra >= m_a):
                    raw_dec = "SAME"
                else:
                    raw_dec = "NEUTRAL"

                # Fallback path: If rejected by baseline (as CONTRADICTION or NEUTRAL),
                # check if high dense similarity + deterministic safety guards justify SAME
                if raw_dec != "SAME" and sim >= t_d:
                    guards_ok, reason = semantic_safety_guards_pass(premise, hypothesis, max_contra)
                    if guards_ok and max_entail >= self.fallback_min_entail:
                        decision = "SAME"
                        decision_path = "guarded_dense_fallback"
                        fallback_applied = True
                        fallback_reason = f"guards_pass, sim={sim:.3f}>={t_d}"
                    else:
                        decision = raw_dec
                        decision_path = f"fallback_rejected_by_guards: {reason}"
                        fallback_reason = reason
                else:
                    decision = raw_dec
                    decision_path = f"baseline_{raw_dec.lower()}"

            # ─────────────────────────────────────────────────────────────
            # Mode 4: Combined (Experiment C: Calibrated + Guarded Dense Fallback)
            # ─────────────────────────────────────────────────────────────
            elif mode == "combined":
                t_e = tau_entail if tau_entail is not None else self.cal_tau_entail
                t_bi_c = self.cal_tau_bilateral_contra
                t_hard_c = self.cal_tau_hard_contra
                m_a = asym_margin if asym_margin is not None else self.cal_asym_margin
                t_d = tau_dense if tau_dense is not None else self.fallback_tau_dense

                is_bilateral_contra = (min_contra >= t_bi_c) or (p_c_fwd >= 0.35 and p_c_rev >= 0.35)
                is_hard_unilateral_contra = (max_contra >= t_hard_c)

                if is_bilateral_contra or is_hard_unilateral_contra:
                    raw_dec = "CONTRADICTION"
                elif max_entail >= t_e and (max_entail - min_contra >= m_a):
                    raw_dec = "SAME"
                else:
                    raw_dec = "NEUTRAL"

                if raw_dec != "SAME" and sim >= t_d:
                    guards_ok, reason = semantic_safety_guards_pass(premise, hypothesis, max_contra)
                    if guards_ok and max_entail >= self.fallback_min_entail:
                        decision = "SAME"
                        decision_path = "combined_guarded_fallback"
                        fallback_applied = True
                        fallback_reason = f"guards_pass, sim={sim:.3f}>={t_d}"
                    else:
                        decision = raw_dec
                        decision_path = f"calibrated_{raw_dec.lower()}"
                        fallback_reason = reason
                else:
                    decision = raw_dec
                    decision_path = f"calibrated_{raw_dec.lower()}"

            else:
                raise ValueError(f"Unknown verification mode: '{mode}'")

            results.append({
                "decision": decision,
                "decision_path": decision_path,
                "p_entail_max": round(max_entail, 4),
                "p_contra_max": round(max_contra, 4),
                "p_contra_min": round(min_contra, 4),
                "p_entail_fwd": round(p_e_fwd, 4),
                "p_contra_fwd": round(p_c_fwd, 4),
                "p_entail_rev": round(p_e_rev, 4),
                "p_contra_rev": round(p_c_rev, 4),
                "dense_sim": round(sim, 4),
                "fallback_applied": fallback_applied,
                "fallback_reason": fallback_reason,
                "nli_invocations": 2,
                "latency_ms": round(per_pair_ms, 2)
            })

        return results
