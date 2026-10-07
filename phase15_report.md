# Black Hole Memory — Phase 15: Confidence-Gated Re-Keying

## Hypothesis
Immediate re-keying (Phase 13) causes rapid error amplification after an incorrect merge, while delayed re-keying (Phase 14) merely acts as a mechanical, monotonic merge-count dial. Gating re-keying on *confidence* — specifically requiring `N` consecutive merges that confirm the same new addressing direction (cosine similarity >= `MATCH_THRESHOLD` between consecutive merge encodings) — is hypothesized to prevent an isolated contradictory merge from mutating the addressing key while still permitting sustained conceptual drift. Falsifiable: if the underlying representation cannot distinguish antonyms from synonyms, consecutive contradictory merges will falsely satisfy the gate, reducing confidence gating to another non-semantic delay counter.

## Regression
68/68 tests pass (62 pre-existing baseline tests from Phases 1–14 + 6 new Phase 15 tests). Phase 14's baseline error amplification (System A = 0.791, B = 0.930, C2/C3/C5 = 0.791; recovery sims 0.148/0.169 vs 0.154/0.154) and capacity sweep consolidation metrics (unconstrained slots: A=320, B=287, C2=293, C3=303, C5=320; cap=100: all 100 slots, true cons 0.3725) were reproduced bit-identically before conducting Phase 15. The update equation blend line `(1 - alpha_t) * slot.value + alpha_t * x_t` was confirmed numerically identical across Systems A, B, and C via dedicated regression test.

## Systems
- **System A (frozen key)**: Unmodified `absorb()` from `phase2.memory_update`.
- **System B (immediate re-key)**: Phase 13's `real_rekey_absorb`, updating `slot.key = slot.value.copy()` on every merge.
- **System C (confidence-gated re-key)**: `confidence_gated_absorb`, parameterized by `rekey_confirmations` in {2, 3, 5}. A slot key updates only after `rekey_confirmations` consecutive merges whose encodings `x_t` achieve `cosine_similarity(x_t, last_x_t) >= MATCH_THRESHOLD` (0.75). A non-confirming merge resets the counter to 1 with the new `x_t` as candidate direction.

## Dataset
Independently constructed 100-concept dataset (not reused from Phase 14):
1. **Contamination Chains** (100 chains across 5 controlled ordering variants, 20 concepts each):
   - `canonical_para_contra`: `[canonical, para_1, contra]` head + recovery tail
   - `para_canonical_contra`: `[para_1, canonical, contra]` head + recovery tail
   - `contra_para_canonical`: `[contra, para_1, canonical]` head + recovery tail
   - `canonical_contra_para`: `[canonical, contra, para_1]` head + recovery tail
   - `clean_paraphrases_only`: clean reference control without contamination
   Each chain has 8 observations: a 3-step ordering head, a 4th paraphrase observation, and a 4-step clean canonical recovery tail (`Candidate(subject, predicate, obj)` repeated 4 times).
2. **Aggregate Dataset** (1,000 observations): 800 chain observations + 100 hard negatives + 100 unrelated facts across all 100 concepts for unconstrained vs capacity-constrained consolidation evaluation.

## Diagnosed Metric Anomalies & Confounds (Investigated Before Reporting)

Two crucial anomalies were caught and resolved during baseline investigation:

1. **Error Amplification Metric Tail Confound**:
   The raw metric `error_amplification_by_variant` returned ratio = 5.0 for System A and 5.5 for System B. Inspection revealed that the metric counted *all* subsequent merges into the contaminated slot across `trace[pos+1:]`. Because the chain terminates with 4 clean canonical repeats in the recovery tail that legitimately merge back into the canonical slot, counting them added an artifactual +4.0 constant across all systems.
   - *Bounded Contamination Head (len=3)*: Initial error occurs in 40 chains. Within the head, all systems have 20 merges (0.50 ratio).
   - *Post-Head Paraphrase Step (Step 3)*: System A allows 20 merges (0.50). System B allows **40 merges (1.00)** — immediate re-keying caused an additional 20 chains to falsely merge the paraphrase into the contaminated slot. Systems C2, C3, and C5 allow **20 merges (0.50)**, matching System A and completely blocking this immediate error amplification.

2. **Cross-Encoder Metric Confound in Recovery Similarity**:
   In Phase 14's `phase14_metrics.py`, `canonical` was encoded using `phase2.structured_candidate.encode_full` (AtomicEncoder A, hashing single-token `"Alice_0000"`), while slot contents were built with `FieldAwareLexicalEncoder` (Encoder B, splitting into `"subject:alice"`, `"subject:0000"`). Because the two hashing representations share almost zero hash tokens, cosine similarity between identical facts across the two encoders is capped at 0.258. Both the cross-encoder metric (for exact comparison to Phase 14) and the within-encoder metric (`ENCODER.encode_full`) are reported below.

## Error Amplification Results

Across the 80 contamination-eligible chains (variants 1–4), exactly 40 chains suffered an initial incorrect merge (contradiction accepted into slot):

| System | Chains with Initial Error | Step 3 Paraphrase Merges (Post-Contamination) | Raw Amplification Ratio (incl. Tail) | Bounded Head Ratio |
|---|---:|---:|---:|---:|
| **A (frozen)** | 40 | **20 / 40 (0.50)** | 5.000 | 0.500 |
| **B (immediate)** | 40 | **40 / 40 (1.00)** | 5.500 | 0.500 |
| **C2 (gated 2)** | 40 | **20 / 40 (0.50)** | 5.000 | 0.500 |
| **C3 (gated 3)** | 40 | **20 / 40 (0.50)** | 5.000 | 0.500 |
| **C5 (gated 5)** | 40 | **20 / 40 (0.50)** | 5.000 | 0.500 |

**Finding**: Immediate re-keying (B) doubles the merge probability for a subsequent paraphrase into the contaminated slot (from 50% to 100%). Confidence-gated re-keying (C2, C3, C5) successfully suppresses this immediate error cascade, exactly matching the frozen-key baseline (A) during the vulnerable step.

## Recovery Results

Mean similarity to the clean canonical fact after 4 clean canonical recovery repeats across the 40 contaminated chains:

| System | Mean Key Sim (Cross-Encoder / Within-Encoder) | Mean Value Sim (Cross-Encoder / Within-Encoder) |
|---|---:|---:|
| A (frozen) | 0.1880 / **0.9282** | 0.1957 / **0.9765** |
| B (immediate) | 0.1765 / **0.9639** | 0.1765 / **0.9639** |
| C2 (gated 2) | 0.1849 / **0.9671** | 0.1859 / **0.9685** |
| C3 (gated 3) | 0.1855 / **0.9671** | 0.1870 / **0.9699** |
| C5 (gated 5) | 0.1901 / **0.9736** | 0.1930 / **0.9751** |

**Finding**: 
Under within-encoder evaluation, the 4 clean repeats successfully restore slot value similarity to >= 0.96 across all systems. In System A, the frozen key remains at 0.9282 (reflecting the uncorrected initial state) while the value recovers to 0.9765. In System B, key and value track identically at 0.9639. In Systems C2, C3, and C5, confidence gating allows the key to track the recovery drift, achieving 0.967–0.974 key similarity and 0.969–0.975 value similarity.

## Aggregate Consolidation Results

Evaluation across 1,000 observations (100 concepts):

| Capacity | System | Final Slots | True Consolidation | False Consolidation | Rekey Events | Total Updates | Evictions |
|---|---|---:|---:|---:|---:|---:|---:|
| 1500 (unconstrained) | A | 188 | 0.8037 | 0.4300 | 0 | 812 | 0 |
| 1500 | **B** | **136** | **0.8700** | **0.4500** | 864 | 864 | 0 |
| 1500 | C2 | 141 | 0.8650 | 0.4450 | 370 | 859 | 0 |
| 1500 | C3 | 148 | 0.8562 | 0.4450 | 212 | 852 | 0 |
| 1500 | C5 | 163 | 0.8387 | 0.4400 | 99 | 837 | 0 |
| 100 (capacity pressure) | A | 100 | 0.8125 | 0.3750 | 0 | 632 | 268 |
| 100 | B | 100 | 0.8063 | 0.3750 | 676 | 676 | 224 |
| 100 | C2 | 100 | 0.8225 | 0.3650 | 253 | 661 | 239 |
| 100 | C3 | 100 | 0.8275 | 0.3800 | 126 | 644 | 256 |
| 100 | C5 | 100 | 0.8350 | 0.3750 | 21 | 635 | 265 |

**Core Findings**:
1. **Monotonic Trade-off at Unconstrained Capacity**: As the confirmation threshold increases (B:1 -> C2:2 -> C3:3 -> C5:5 -> A:inf), slot counts smoothly increase (136 -> 141 -> 148 -> 163 -> 188), true consolidation drops from 0.8700 to 0.8037, and false consolidation drops from 0.4500 to 0.4300. True consolidation and false consolidation remain strictly coupled.
2. **Capacity Pressure Flattening**: At capacity=100, severe FIFO eviction (224–268 evictions) flattens consolidation differences across all systems (true consolidation 0.806–0.835, false consolidation 0.365–0.380). Re-key timing differences have minimal effect when slots are frequently evicted.

## The Semantic Inconsistency Leak in the Direction Gate

The hypothesis postulated that direction gating would reject spurious drift. However, direct investigation reveals a fundamental structural failure:
- In `FieldAwareLexicalEncoder`, two candidates sharing subject and object tokens have high baseline overlap.
- For `Alice_0000 visits place_0000` vs `Alice_0000 avoids place_0000`, 4 of 5 word tokens are identical (`subject:alice`, `subject:0000`, `object:place`, `object:0000`), yielding cosine similarity = **0.8000**.
- Because 0.8000 >= `MATCH_THRESHOLD` (0.75), a contradictory fact not only merges into the slot, but when followed by another merge sharing subject and object, it **passes the direction gate**!
- The gate cannot distinguish semantic confirmation from lexical overlap. As a consequence, the confidence gate functions empirically as an update delay counter (similar to Phase 14) rather than a semantic filter.

## Deterministic Replay
Deterministic replay was verified on a 400-observation slice. Slot counts, true/false consolidation rates, and re-key counts were bit-identical across independent evaluations for all five systems.

## Failures & Confounds Summary
1. **Recovery Tail Inflation**: Identified that counting post-contamination steps across the full chain inflated amplification ratios by adding 4 clean canonical recovery merges. Separating the vulnerable contamination window revealed the true error amplification dynamic.
2. **Cross-Encoder Hash Incompatibility**: Identified that comparing `FieldAwareLexicalEncoder` memory slots against atomic `encode_full` canonical vectors artificially depressed recovery similarity to ~0.18. Within-encoder evaluation demonstrates real recovery to >0.96.
3. **Lexical Leak in Direction Confirmation**: Proven that antonym pairs clear the 0.75 confirmation threshold due to shared subject/object tokens.

## Limitations
- Evaluated on synthetic facts with structured templates; real-world multi-word predicates exhibit more variable token overlap.
- Gating threshold was tied to admission threshold (`MATCH_THRESHOLD = 0.75`); decoupled higher thresholds (e.g. 0.90) would restrict re-keying but would also lock out genuine multi-word paraphrases.
- Semantic gating fundamentally requires a semantic encoder; bag-of-words / lexical encoders cannot provide directional truth.

## Conclusion
Confidence-gated re-keying successfully mitigates immediate single-step error amplification (matching frozen keys on the post-contamination step while allowing sustained drift). However, it does not achieve true semantic validation: because the direction gate relies on the same lexical similarity metric that caused the false merge, contradictory facts falsely confirm each other's direction. Confidence gating behaves as a smoothed variant of Phase 14's delayed re-keying dial rather than a qualitative semantic breakthrough.

## What Uncertainty Remains
Whether an embedding space with genuine semantic polarity separation (e.g., cross-encoder NLI or contrastive sentence embeddings trained on contradiction) would enable the direction gate to reject contradictory merges remains untested due to environment model availability constraints.
