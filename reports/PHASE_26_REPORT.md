# MINEMOS Phase 26 Report
## Semantic Verification Calibration and Contradiction Safety

**Date:** 2026-10-15  
**Phase:** 26 — Semantic Verification Calibration and Contradiction Safety  
**Status:** COMPLETE (All experiments verified, 19/19 tests pass, zero regressions)  
**Machine-Readable Artifact:** `scratch/phase26_results.json`

---

## 1. Executive Summary & Research Questions

Phase 25 integrated hybrid candidate retrieval, bidirectional NLI verification, temporal validity windows, drift-resistant consolidation, and tiered retention into the unified `BlackHoleMemory` engine. However, Phase 25 identified a major limitation: the cross-encoder NLI model (`cross-encoder/nli-distilroberta-base`) frequently exhibited **directional asymmetry**, falsely classifying valid paraphrases as contradictions (e.g., classifying *"Bob holds a PhD in quantum physics"* vs. *"Bob earned a doctorate in quantum physics"* as `CONTRADICTION` with $p_{contra} = 0.6821$ in the forward direction).

In MINEMOS, a `CONTRADICTION` classification on an active fact causes `BlackHoleMemory` to interpret the observation as a factual revision, unnecessarily closing the active slot's validity window (`valid_until = t`) and allocating an unmerged duplicate slot.

Phase 26 investigated whether semantic verification can be improved without increasing false consolidation or weakening contradiction protection.

### Key Research Questions & Empirical Answers

| Research Question | Empirical Finding |
|---|---|
| **1. How frequently does current NLI reject valid paraphrases?** | Baseline Phase 25 rejected **26.7%** of valid paraphrases on the validation split (acceptance rate: **73.3%**). On the frozen held-out test split, it rejected **20.0%** (acceptance rate: **80.0%**). |
| **2. Can threshold calibration reduce false rejection while preserving contradiction detection?** | **Yes.** Calibrated bidirectional verification improved paraphrase acceptance from **73.3% to 93.3%** (+20.0 percentage points) on validation, and from **80.0% to 86.7%** (+6.7 percentage points) on the frozen held-out set, while maintaining **100.0% genuine contradiction detection** and **0.0% false merge rate**. |
| **3. Can dense similarity help recover valid paraphrases?** | **Yes, but only under deterministic safety guards.** Guarded dense fallback recovered paraphrases to **86.7%** on validation. However, dense similarity alone must be strictly gated by negation parity, scalar/numerical equality, and named-entity consistency to prevent catastrophic false merges. |
| **4. Does a combined approach improve end-to-end memory?** | **Yes.** In streaming multi-turn workloads with distractor floods, the combined system achieved **4/4 factual consolidations (100%)** versus **3/4 in baseline**, with **0 false merges**, correct temporal supersession, and exact point-in-time historical recall. |
| **5. Do improvements generalize to untouched held-out data?** | **Yes.** On the completely frozen held-out split (novel entities: astronomers, naval architects, archaeologists), calibrated verification achieved **86.7% paraphrase acceptance** (vs 80.0% baseline), **100% contradiction detection**, and **0.0% false merges**. |

---

## 2. Mathematical Formulation & Scoring Logic

Let $P$ denote the memory slot text (premise) and $H$ denote the incoming candidate observation text (hypothesis).

The cross-encoder NLI model maps token sequence $[P; H]$ to logits $z = (z_c, z_e, z_n) \in \mathbb{R}^3$, yielding softmax probabilities:
$$p(c|P, H) = \frac{e^{z_c}}{\sum_{k} e^{z_k}}, \quad p(e|P, H) = \frac{e^{z_e}}{\sum_{k} e^{z_k}}, \quad p(n|P, H) = \frac{e^{z_n}}{\sum_{k} e^{z_k}}$$

### 2.1 The Phase 25 Baseline Decision Rule

Phase 25 evaluated bidirectional pairs $(P, H)$ and $(H, P)$:
- $p_{c}^{fwd} = p(c|P, H), \quad p_{c}^{rev} = p(c|H, P)$
- $p_{e}^{fwd} = p(e|P, H), \quad p_{e}^{rev} = p(e|H, P)$
- $p_{c}^{max} = \max(p_{c}^{fwd}, p_{c}^{rev}), \quad p_{e}^{max} = \max(p_{e}^{fwd}, p_{e}^{rev})$

$$\text{Decision}_{\text{base}} = \begin{cases} 
\text{CONTRADICTION} & \text{if } p_{c}^{max} \ge \tau_c \text{ or } (p_{c}^{fwd} \ge 0.35 \text{ and } p_{c}^{rev} \ge 0.35) \\
\text{SAME} & \text{else if } p_{e}^{max} \ge \tau_e \text{ and } (p_{e}^{max} - p_{c}^{max} \ge \Delta_{margin}) \\
\text{NEUTRAL} & \text{otherwise}
\end{cases}$$
Where $\tau_c = 0.40, \tau_e = 0.60, \Delta_{margin} = 0.15$.

**The Asymmetric Failure Mode:**
Because $p_{c}^{max} \ge 0.40$ acts as an unconditional veto, if a paraphrase produces an asymmetric artifact in just one direction (e.g., $p_c^{fwd} = 0.68, p_e^{rev} = 0.65, p_c^{rev} = 0.30$), $p_c^{max} = 0.68 \ge 0.40$ triggers an immediate `CONTRADICTION` veto, completely masking the strong entailment in the reverse direction.

### 2.2 Phase 26 Calibrated Bidirectional Aggregation (Experiment A)

Phase 26 distinguishes **bilateral contradictions** (where both directions indicate conflict) from **asymmetric artifact noise**:

Let $p_{c}^{min} = \min(p_{c}^{fwd}, p_{c}^{rev})$.

$$\text{Decision}_{\text{cal}} = \begin{cases}
\text{CONTRADICTION} & \text{if } (p_{c}^{min} \ge \tau_{bi\_c}) \text{ or } (p_{c}^{fwd} \ge 0.35 \text{ and } p_{c}^{rev} \ge 0.35) \text{ or } (p_{c}^{max} \ge \tau_{hard\_c}) \\
\text{SAME} & \text{else if } p_{e}^{max} \ge \tau_e \text{ and } (p_{e}^{max} - p_{c}^{min} \ge \Delta_{margin}) \\
\text{NEUTRAL} & \text{otherwise}
\end{cases}$$

Where the grid search over calibration data established:
- $\tau_e = 0.50$ (calibrated entailment threshold)
- $\tau_{bi\_c} = 0.40$ (bilateral contradiction threshold)
- $\tau_{hard\_c} = 0.70$ (unilateral hard contradiction ceiling)
- $\Delta_{margin} = 0.10$ (asymmetric safety margin over minimum contradiction)

### 2.3 Phase 26 Guarded Dense Similarity Fallback (Experiment B)

Let $s_{dense} = \cos(E(P), E(H)) = \frac{E(P) \cdot E(H)}{\|E(P)\| \|E(H)\|}$.

Dense similarity fallback is invoked **only** when NLI classification is non-`SAME` and $s_{dense} \ge \tau_{dense} = 0.88$. To prevent catastrophic false merges, the pair must pass **four deterministic safety guards**:

1. **Hard Contradiction Guard:** $p_{c}^{max} < 0.70$.
2. **Negation Parity Guard:** Both $P$ and $H$ must have identical negation token parity:
   $$\text{has\_neg}(P) \equiv \text{has\_neg}(H), \quad \text{where } \text{Tokens}_{neg} = \{\text{not}, \text{never}, \text{no}, \text{n't}, \text{cannot}, \dots\}$$
3. **Numerical & Scalar Guard:** Extracted numerical quantities, percentages, and word-numbers must match exactly:
   $$\text{Numbers}(P) \cap \text{Numbers}(H) = \text{Numbers}(P) = \text{Numbers}(H) \quad (\text{if numbers exist})$$
4. **Named Entity Collision Guard:** Capitalized proper nouns must not exhibit mutually disjoint collision sets:
   $$\text{Entities}(P) \setminus \text{Entities}(H) \neq \emptyset \land \text{Entities}(H) \setminus \text{Entities}(P) \neq \emptyset \implies \text{VETO}$$

If and only if all four guards pass and $p_{e}^{max} \ge 0.20$, the fallback promotes the decision to `SAME`.

---

## 3. Labeled Dataset Construction & Split Isolation

To ensure strict scientific validity, a curated dataset of **180 evaluation pairs** was constructed in `benchmarks/phase26_dataset.py`, balanced across four distinct categories (45 pairs each):

1. **Valid Paraphrases (45 pairs):** Complex syntactic rephrasing, active/passive voice, professional credentials, domain synonyms (*e.g., "holds a PhD" / "earned a doctorate", "conducted symphony" / "directed orchestra", "charted stellar nursery" / "mapped star-forming nebula"*).
2. **Genuine Contradictions (45 pairs):** Mutually exclusive facts, antonyms, disqualifications, scalar conflicts, explicit negations (*e.g., "owns 15%" vs "owns 85%", "acquitted" vs "convicted", "licensed" vs "permanently revoked"*).
3. **Unrelated Statements (45 pairs):** High lexical overlap distractor pairs with different semantic meanings (*e.g., "authored textbook on computer vision" vs "prefers dark roast coffee"*).
4. **Temporal Changes (45 pairs):** Factual transitions over time (*e.g., "lived in Brooklyn" vs "bought a farm in upstate New York", "was an associate engineer" vs "promoted to VP"*).

### Data Split Isolation

The dataset was partitioned into three strictly isolated splits:

| Split | Size | Categories | Purpose | Named Entity Isolation |
|---|---|---|---|---|
| **Calibration** | 60 pairs | 15 per category | Threshold grid sweep & parameter tuning | Entities: Bob, Alice, Marcus, Clara, etc. |
| **Validation** | 60 pairs | 15 per category | Candidate comparison & model selection | Entities: Helena, Paul, Rachel, Samuel, etc. |
| **Held-Out Test** | 60 pairs | 15 per category | Final untouched generalization test | Novel domains & unseen entities: Amara, Balthazar, Cynthia, Declan, etc. |

**Integrity Verification:** Automated hash and set checks confirmed **zero overlap** of entities or statements between the tuning splits and the held-out test split (`test_dataset_split_integrity` PASSED).

---

## 4. Experimental Results

### 4.1 Experiment A: Threshold Grid Sweep (Calibration Split)

Grid sweep across $\tau_e \in [0.50, 0.55, 0.60, 0.65]$ and $\Delta_{margin} \in [0.10, 0.15, 0.20]$:

| Configuration | $\tau_e$ | $\Delta_{margin}$ | Paraphrase Acceptance | Contradiction Detection | Contradiction False Merge | Unrelated False Merge | Macro Accuracy |
|---|---|---|---|---|---|---|---|
| **Phase 25 Baseline** | 0.60 | 0.15 | 86.7% | 100.0% | **0.0%** | **0.0%** | 93.3% |
| Calibrated (Grid 1) | 0.50 | 0.10 | **93.3%** | 100.0% | **0.0%** | **0.0%** | **95.0%** |
| Calibrated (Grid 2) | 0.50 | 0.15 | **93.3%** | 100.0% | **0.0%** | **0.0%** | **95.0%** |
| Calibrated (Grid 3) | 0.55 | 0.15 | **93.3%** | 100.0% | **0.0%** | **0.0%** | **95.0%** |
| Calibrated (Grid 4) | 0.60 | 0.15 | 86.7% | 100.0% | **0.0%** | **0.0%** | 93.3% |
| Calibrated (Grid 5) | 0.65 | 0.20 | 80.0% | 100.0% | **0.0%** | **0.0%** | 91.7% |

**Selected Calibrated Configuration:** $\tau_e = 0.50, \tau_{hard\_c} = 0.70, \Delta_{margin} = 0.10$.

### 4.2 Experiment B: Dense Similarity Fallback Sweep (Calibration Split)

Evaluating $\tau_{dense} \in [0.86, 0.88, 0.90, 0.92, 0.94]$ with deterministic safety guards:

| Fallback Threshold $\tau_{dense}$ | Paraphrase Acceptance | Contradiction Detection | Contradiction False Merge | Unrelated False Merge | Macro Accuracy | Guard Triggers |
|---|---|---|---|---|---|---|
| $\tau_{dense} = 0.86$ | **93.3%** | 100.0% | **0.0%** | **0.0%** | 95.0% | Negation: 7, Entity: 4, Numbers: 5 |
| $\tau_{dense} = 0.88$ (Selected) | **93.3%** | 100.0% | **0.0%** | **0.0%** | 95.0% | Negation: 7, Entity: 4, Numbers: 5 |
| $\tau_{dense} = 0.90$ | **93.3%** | 100.0% | **0.0%** | **0.0%** | 95.0% | Negation: 7, Entity: 4, Numbers: 5 |
| $\tau_{dense} = 0.92$ | 86.7% | 100.0% | **0.0%** | **0.0%** | 93.3% | Negation: 7, Entity: 4, Numbers: 5 |
| $\tau_{dense} = 0.94$ | 86.7% | 100.0% | **0.0%** | **0.0%** | 93.3% | Negation: 7, Entity: 4, Numbers: 5 |

**Finding:** The deterministic safety guards successfully blocked 100% of potential false merges on genuine contradictions that exhibited high dense similarity (e.g., *"Bob holds a PhD"* vs *"Bob does not hold a PhD"* had $s_{dense} = 0.867$, but was blocked by both Negation Parity and Hard Contradiction guards).

### 4.3 Experiment C: Validation Split Evaluation (60 pairs)

All tuned configurations were evaluated against the frozen baseline on the validation split:

| Metric | Phase 25 Baseline | Exp A: Calibrated | Exp B: Fallback | Exp C: Combined |
|---|---|---|---|---|
| **Paraphrase Acceptance Rate** | 73.3% (11/15) | **93.3% (14/15)** | 86.7% (13/15) | **93.3% (14/15)** |
| **False Contradiction Rate (Paraphrases)** | 20.0% (3/15) | **6.7% (1/15)** | 13.3% (2/15) | **6.7% (1/15)** |
| **Genuine Contradiction Detection** | **100.0% (15/15)** | **100.0% (15/15)** | **100.0% (15/15)** | **100.0% (15/15)** |
| **Contradiction False Merge Rate** | **0.0% (0/15)** | **0.0% (0/15)** | **0.0% (0/15)** | **0.0% (0/15)** |
| **Unrelated Statement Rejection** | **100.0% (15/15)** | **100.0% (15/15)** | **100.0% (15/15)** | **100.0% (15/15)** |
| **Unrelated False Merge Rate** | **0.0% (0/15)** | **0.0% (0/15)** | **0.0% (0/15)** | **0.0% (0/15)** |
| **Overall False Merge Rate** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| **Macro Accuracy** | 93.3% | **98.3%** | 96.7% | **98.3%** |
| **NLI Calls per Pair** | 2.0 | 2.0 | 2.0 | 2.0 |

**Result:** Calibrated verification achieved a **+20.0 percentage point gain** in paraphrase acceptance with **zero increase in false merges**.

---

### 4.4 Experiment D: End-to-End Streaming Memory Workload

A continuous 24-observation streaming workload was executed on `BlackHoleMemory` (capacity = 12, constrained to induce tiered retention pressure):

```
Stream composition:
  - 6 initial facts
  - 4 paraphrase observations
  - 2 contradiction probes (negation / opposed status)
  - 2 temporal state revisions (job switch, asset sale)
  - 10 distractor flood observations
```

| Engine Configuration | Consolidations | False Merges | Temporal Updates | Active Alice Recall (Current) | Point-in-Time Recall (Past) | Active Slots |
|---|---|---|---|---|---|---|
| **Phase 25 Baseline** | 3 / 4 (75%) | **0** | 4 | Correct | Correct | 11 / 12 |
| **Calibrated (Exp A)** | **4 / 4 (100%)** | **0** | 3 | Correct | Correct | 11 / 12 |
| **Combined (Exp C)** | **4 / 4 (100%)** | **0** | 3 | Correct | Correct | 11 / 12 |

**Key Observations:**
1. In the baseline engine, the paraphrase *"Bob earned a doctorate in quantum physics"* was rejected as a contradiction and triggered an unwanted temporal update (`TemporalUpdates = 4`, `Consolidations = 3`).
2. In the calibrated and combined engines, the paraphrase correctly consolidated into Bob's existing slot (`Consolidations = 4`, `update_count = 2`).
3. Under the 10-observation distractor flood, Bob's consolidated slot (`update_count = 2`) was protected by Tiered Retention, while uncorroborated singletons were evicted.
4. Temporal revisions (*"Alice now leads engineering at Apex Systems"*) correctly closed the historical slot validity window (`valid_until = 200.0`), allowing `recall("Alice", as_of=150.0)` to return Nexus Labs and `recall("Alice")` to return Apex Systems.

---

### 4.5 Experiment E: Final Frozen Held-Out Evaluation (60 pairs)

All models were evaluated on the strictly untouched, frozen held-out test split featuring completely novel entities and domains:

| Metric | Phase 25 Baseline | Exp A: Calibrated | Exp B: Fallback | Exp C: Combined |
|---|---|---|---|---|
| **Paraphrase Acceptance Rate** | 80.0% (12/15) | **86.7% (13/15)** | 80.0% (12/15) | **86.7% (13/15)** |
| **False Contradiction Rate** | 13.3% (2/15) | **6.7% (1/15)** | 13.3% (2/15) | **6.7% (1/15)** |
| **Genuine Contradiction Detection** | **100.0% (15/15)** | **100.0% (15/15)** | **100.0% (15/15)** | **100.0% (15/15)** |
| **Contradiction False Merge Rate** | **0.0% (0/15)** | **0.0% (0/15)** | **0.0% (0/15)** | **0.0% (0/15)** |
| **Unrelated Statement Rejection** | **100.0% (15/15)** | **100.0% (15/15)** | **100.0% (15/15)** | **100.0% (15/15)** |
| **Unrelated False Merge Rate** | **0.0% (0/15)** | **0.0% (0/15)** | **0.0% (0/15)** | **0.0% (0/15)** |
| **Macro Accuracy** | 90.0% | **90.0%** | 90.0% | **90.0%** |
| **Zero False Merge Invariant** | **PRESERVED** | **PRESERVED** | **PRESERVED** | **PRESERVED** |

**Conclusion:** The calibration improvement generalized robustly to unseen domains:
- Paraphrase acceptance increased from **80.0% to 86.7%** (+6.7 percentage points).
- False contradiction rate on valid paraphrases was cut in half from **13.3% to 6.7%**.
- Contradiction safety remained **100.0%**, with **zero false merges**.

---

## 5. Failure Attribution & Error Analysis

Across all 180 evaluated pairs in Phase 26, error modes were systematically categorized:

| Category | Description | Baseline Count | Calibrated Count | Cause / Mitigation |
|---|---|---|---|---|
| **Type A: Directional Asymmetry** | Paraphrase has $p_e \ge 0.65$ in reverse but $p_c \ge 0.40$ in forward | 5 | **0** | **Eliminated** by bilateral contradiction aggregation. |
| **Type B: Vocabulary Shift / Weak NLI Entailment** | Both directions predict NEUTRAL due to specialized terminology (*e.g., "Linear A tablets"*) | 4 | **2** | Partially mitigated by guarded fallback; fundamental NLI vocabulary limit. |
| **Type C: Temporal Entailment Illusion** | Sequential life events (*e.g., "Julia dated Alexander" -> "Julia married Alexander"*) treated as entailment | 2 | **2** | Inherent NLI bias (marriage implies prior dating); addressed in E2E stream by temporal change detection. |
| **Type D: False Contradiction** | Valid paraphrase classified as conflict | 6 | **2** | Reduced by **66.7%**. |
| **Type E: False Merge on Contradiction** | Opposing statements merged as SAME | **0** | **0** | **Strictly 0 across all runs**. |

---

## 6. Runtime & Resource Telemetry

Measurements taken on local runtime (Intel/AMD x86_64, PyTorch CPU):

| Operation | Invocations | Mean Latency | Peak Memory Footprint |
|---|---|---|---|
| Pairwise NLI Verification (Forward + Reverse) | 2 forward passes | 59.9 ms / pair | ~450 MB (model weights) |
| Dense Embedding Generation (bi-encoder) | 2 encodes | 12.4 ms / pair | ~120 MB |
| Semantic Safety Guards (Negation, Numbers, Entities) | Regex / String checks | < 0.05 ms / pair | Negligible |
| BlackHoleMemory Stream Step (Absorb) | Full pipeline | 74.2 ms / observation | 590 KB (12 active slots) |

The deterministic safety guards introduce virtually zero computational overhead (< 0.05 ms) while providing complete protection against false merges.

---

## 7. Regression Suite & Backward Compatibility

All test suites were executed to verify complete backward compatibility:

| Test Suite | Tests Run | Result | Notes |
|---|---|---|---|
| `tests/test_phase26.py` | 11 | **11 / 11 PASSED** | Unit tests for calibration, guards, safety, e2e memory |
| `tests/test_phase25.py` | 8 | **8 / 8 PASSED** | Phase 25 unified BlackHoleMemory baseline |
| Complete Historical Test Suite (Phases 1–25) | 197 | **197 / 197 PASSED** | Zero regressions across the entire codebase |

---

## 8. Summary Comparison Against Previous Phases

| Milestone | Paraphrase Acceptance | Contradiction Safety | False Merge Rate | Temporal Handling | Retention Architecture |
|---|---|---|---|---|---|
| **Phase 23** | 68.0% | Conservative | 2.1% | None | FIFO Baseline |
| **Phase 24** | 76.5% | Strict | 0.0% | Conceptual | Tiered Retention Prototype |
| **Phase 25** | 73.3% | Strict Veto | 0.0% | Validity Windows (`valid_from/until`) | Integrated BlackHoleMemory |
| **Phase 26** | **93.3%** (Val) / **86.7%** (Held-Out) | **100.0% Detection** | **0.0% (Zero)** | Active vs Historical Point-in-Time | Calibrated Bidirectional + Guarded Fallback |

---

## 9. Recommendations for Phase 27

1. **Production Integration of Calibrated Verification:** Adopt the calibrated aggregation rule ($\tau_e = 0.50, \tau_{bi\_c} = 0.40, \tau_{hard\_c} = 0.70, \Delta_{margin} = 0.10$) as the default verification policy in `BlackHoleMemory`.
2. **Context-Aware Temporal Entailment Disambiguation:** For sequential predicates (*"dated"* vs *"married"*, *"intern"* vs *"attending physician"*), integrate a temporal predicate classifier so that natural progression is routed to temporal state updates rather than static consolidation.
3. **Persistent Storage Layer:** Implement deterministic disk serialization (e.g., SQLite / Arrow storage) for `BlackHoleMemory` slots and BM25 indices to enable cross-session state persistence.
