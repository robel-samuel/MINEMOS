"""
benchmarks/phase24_verification.py

Phase 24 Semantic Verification & Recalibration Suite:
Investigates reasons for valid paraphrase rejection and evaluates calibrated,
asymmetric, and bidirectional verification policies under conservative contradiction safety.

Verification Modes:
  1. 'baseline': Forward-only NLI with sweepable entail_threshold in [0.55, 0.60, 0.65, 0.70, 0.75].
  2. 'asymmetric': Conservative contra_threshold (0.40) + margin constraint (p_entail - p_contra >= 0.15).
  3. 'bidirectional': Tests both (premise, hypothesis) and (hypothesis, premise) with strict contradiction veto.
"""

from __future__ import annotations
from typing import Optional, Any
import torch

from benchmarks.phase16_experiment import NliSemanticGate


class CalibratedSemanticVerifier:
    """
    Wraps NliSemanticGate with calibrated thresholding, asymmetric safety margins,
    and optional bidirectional equivalence verification.
    """

    def __init__(self, nli_gate: NliSemanticGate):
        self.nli_gate = nli_gate

    def verify_pair(
        self,
        premise: str,
        hypothesis: str,
        mode: str = "baseline",
        entail_threshold: float = 0.75,
        contra_threshold: float = 0.50,
        abstain_min_confidence: float = 0.35,
        asym_margin: float = 0.15
    ) -> dict:
        """Verifies a single premise-hypothesis pair."""
        return self.verify_batch(
            [(premise, hypothesis)],
            mode=mode,
            entail_threshold=entail_threshold,
            contra_threshold=contra_threshold,
            abstain_min_confidence=abstain_min_confidence,
            asym_margin=asym_margin
        )[0]

    def verify_batch(
        self,
        pairs: list[tuple[str, str]],
        mode: str = "baseline",
        entail_threshold: float = 0.75,
        contra_threshold: float = 0.50,
        abstain_min_confidence: float = 0.35,
        asym_margin: float = 0.15
    ) -> list[dict]:
        """
        Batched semantic verification.
        
        Returns list of dicts with:
          - decision: 'SAME', 'CONTRADICTION', 'NEUTRAL', 'ABSTAIN'
          - p_entail, p_contra, p_neutral, confidence
          - nli_invocations: number of pair evaluations executed
        """
        if not pairs:
            return []

        if mode == "baseline":
            # Forward only
            raw_results = self.nli_gate.predict_batch(
                pairs,
                entail_threshold=entail_threshold,
                contra_threshold=contra_threshold,
                abstain_min_confidence=abstain_min_confidence
            )
            for r in raw_results:
                r["nli_invocations"] = 1
            return raw_results

        elif mode == "asymmetric":
            # Conservative contradiction threshold + entailment margin
            raw_results = self.nli_gate.predict_batch(
                pairs,
                entail_threshold=entail_threshold,
                contra_threshold=contra_threshold,
                abstain_min_confidence=abstain_min_confidence
            )
            calibrated = []
            for r in raw_results:
                p_e = r["p_entail"]
                p_c = r["p_contra"]
                p_n = r["p_neutral"]
                conf = r["confidence"]

                if conf < abstain_min_confidence:
                    dec = "ABSTAIN"
                elif p_c >= contra_threshold:
                    # Conservative contradiction rejection
                    dec = "CONTRADICTION"
                elif p_e >= entail_threshold and (p_e - p_c >= asym_margin):
                    # Entailment verified with safe margin over contradiction
                    dec = "SAME"
                elif p_c >= 0.30:
                    dec = "CONTRADICTION"
                else:
                    dec = "NEUTRAL"

                calibrated.append({
                    "decision": dec,
                    "p_entail": p_e,
                    "p_contra": p_c,
                    "p_neutral": p_n,
                    "confidence": conf,
                    "nli_invocations": 1
                })
            return calibrated

        elif mode == "bidirectional":
            # Evaluates forward (P, H) and reverse (H, P)
            fwd_pairs = pairs
            rev_pairs = [(h, p) for p, h in pairs]
            all_pairs = fwd_pairs + rev_pairs

            raw_all = self.nli_gate.predict_batch(
                all_pairs,
                entail_threshold=entail_threshold,
                contra_threshold=contra_threshold,
                abstain_min_confidence=abstain_min_confidence
            )
            n = len(pairs)
            fwd_res = raw_all[:n]
            rev_res = raw_all[n:]

            combined = []
            for fwd, rev in zip(fwd_res, rev_res):
                p_e_fwd, p_c_fwd, p_n_fwd = fwd["p_entail"], fwd["p_contra"], fwd["p_neutral"]
                p_e_rev, p_c_rev, p_n_rev = rev["p_entail"], rev["p_contra"], rev["p_neutral"]

                # Maximum contradiction across either direction acts as conservative veto
                max_contra = max(p_c_fwd, p_c_rev)
                max_entail = max(p_e_fwd, p_e_rev)
                avg_entail = 0.5 * (p_e_fwd + p_e_rev)
                conf = max(fwd["confidence"], rev["confidence"])

                # Strict contradiction veto: if either direction triggers contradiction
                if max_contra >= contra_threshold or (p_c_fwd >= 0.35 and p_c_rev >= 0.35):
                    dec = "CONTRADICTION"
                elif conf < abstain_min_confidence:
                    dec = "ABSTAIN"
                elif max_entail >= entail_threshold and (max_entail - max_contra >= asym_margin):
                    dec = "SAME"
                else:
                    dec = "NEUTRAL"

                combined.append({
                    "decision": dec,
                    "p_entail": round(max_entail, 4),
                    "p_contra": round(max_contra, 4),
                    "p_neutral": round(min(p_n_fwd, p_n_rev), 4),
                    "confidence": round(conf, 4),
                    "nli_invocations": 2,
                    "fwd": fwd,
                    "rev": rev
                })
            return combined

        else:
            raise ValueError(f"Unknown verification mode: '{mode}'")
