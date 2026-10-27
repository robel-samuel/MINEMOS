# Black Hole Memory — Phase 6: Similarity-Threshold Sweep

## 1. Research Question
Can any single similarity threshold allow genuine paraphrases to
consolidate while keeping semantically different facts separate?

## 2. Hypothesis
Not assumed true. Phase 5 predicted mathematically (and this phase
verifies on an independent dataset) that with atomic 3-field encoding,
similarity only takes values in {0, 1/3, 2/3, 1}, so genuine paraphrases
and hard negatives that differ in exactly one field are likely
indistinguishable by any threshold.

## 3. Experimental Controls
Encoder, tokenizer, update equation, slot structure, eviction (FIFO),
capacity (200, generous — isolates threshold effect from capacity
pressure) all unchanged from Phase 2B/5. Only `MATCH_THRESHOLD` was
swept, via one small additive parameter added to `absorb()` (default
unchanged, verified via regression: 36/36 tests pass identically before
and after).

## 4. Dataset Construction
34 independently-constructed pairs, none reused from Phase 5: 6 exact-
duplicate pairs (Category A), 10 genuine-paraphrase pairs (Category B,
new wording: "regularly uses", "relies on", "is employed by", "enjoys",
"resides in", "leads", "codes primarily in", etc.), 18 hard-negative
pairs (Category C, including 15 one-field-different cases and 3
two-field-different cases for completeness).

## 5. Mathematical Prediction
Checked directly before running the full sweep: every Category B example
given in the task spec and every Category C example measured **exactly
0.6667** similarity — mathematically indistinguishable, predicting no
threshold could separate them.

## 6. Results Table

| threshold | exact_dup_rate | paraphrase_rate | false_consolidation_rate | slots_used | state_bytes | latency_ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 1.0000 | 1.0000 | 0.8333 | 9 | 297,154 | 0.170 |
| 0.55 | 1.0000 | 1.0000 | 0.8333 | 9 | 297,154 | 0.164 |
| 0.60 | 1.0000 | 1.0000 | 0.8333 | 9 | 297,154 | 0.213 |
| 0.65 | 1.0000 | 1.0000 | 0.8333 | 9 | 297,154 | 0.158 |
| **0.70** | 1.0000 | **0.0000** | **0.0000** | 34 | 1,116,954 | 0.366 |
| 0.75 | 1.0000 | 0.0000 | 0.0000 | 34 | 1,116,954 | 0.380 |
| 0.80 | 1.0000 | 0.0000 | 0.0000 | 34 | 1,116,954 | 0.360 |
| 0.85 | 1.0000 | 0.0000 | 0.0000 | 34 | 1,116,954 | 0.370 |
| 0.90 | 1.0000 | 0.0000 | 0.0000 | 34 | 1,116,954 | 0.353 |

**The entire sweep is a hard binary switch at exactly 2/3 = 0.667**,
matching the mathematical prediction precisely: below it, paraphrase and
false-consolidation move together (1.00 / 0.83); at or above it, both
drop to zero together. Exact-duplicate consolidation is 1.0 at every
threshold tested, as expected (similarity=1.0 always clears any
threshold ≤0.90).

**No threshold in the swept range satisfies "paraphrase ≥0.5 AND
false-consolidation ≤0.1" simultaneously** — confirmed by direct search,
not eyeballed.

## 7. Confusion Analysis
At threshold=0.50-0.65 (representative: 0.60): true_consolidations=16
(6 exact + 10 paraphrase), missed_consolidations=0,
false_consolidations=15 (of 18 hard negatives), correct_separations=3.
The 3 correctly-separated hard negatives are exactly the **two-field-
different** cases (`fully_unrelated_1/2/3`) — these have similarity
≤1/3, safely below even the lowest threshold tested. **Every
one-field-different hard negative (15 of 18) is indistinguishable from
genuine paraphrases at any threshold that captures paraphrases.**

At threshold=0.70-0.90: true_consolidations=6 (exact only),
missed_consolidations=10 (all paraphrases), false_consolidations=0,
correct_separations=18 (all).

## 8. Memory/Latency Results
State size scales directly with slots_used (9 slots → 297KB, 34 slots →
1.1MB at D=4096 — consistent with Phase 2A's per-slot byte accounting).
Latency stayed in the sub-millisecond range at every threshold; not a
limiting factor for this experiment.

## 9. Representation-Drift Analysis
**A measurement bug was caught and fixed mid-experiment, disclosed
rather than silently corrected**: the first drift measurement compared
`slot.key` before and after an update, always reading 1.0 — because
`k_i` is architecturally frozen at slot creation (documented since
Phase 2); only `v_i` (`slot.value`) actually changes under the update
equation. A hand-derived expected value (≈0.96) didn't match the
observed 1.0, which is what caught the bug. Corrected measurement, on
`slot.value`: at every threshold where paraphrase merging occurs
(0.50-0.65), the resulting value settles at **exactly 0.9631 cosine
similarity to the canonical representation, uniformly across all 10
paraphrase pairs**. This is real, moderate, non-catastrophic movement —
not collapse toward something unrelated, and not the "zero drift" found
in Phase 5 (which was specific to threshold=0.75, where only exact
matches, at similarity=1.0/novelty=0, ever triggered an update at all).

## 10. Failure Analysis
Root cause, unchanged from Phase 5's diagnosis and now further
confirmed on an independent, larger dataset: atomic per-field
tokenization collapses similarity into 4 discrete levels. Any pair
differing in exactly one of three fields — whether that difference is a
genuine paraphrase ("uses" → "regularly uses") or a genuine semantic
opposite ("uses" → "dislikes") — produces the identical similarity value
(0.667). No threshold operating on this single scalar can separate two
categories that occupy the exact same point in the representation's
range. This is not a tuning problem; it is a resolution/information
problem — the representation itself has discarded the information
needed to distinguish these cases before similarity is even computed.

## 11. Scientific Interpretation
Three genuinely different things, kept explicitly separate per the
task's requirement:
- **Exact deduplication**: works perfectly, at every threshold tested.
  This is real but trivial — a hash-equality check would do the same
  thing.
- **Near-duplicate consolidation**: does not work at any threshold.
  "Near-duplicate" and "hard negative" are mathematically the same
  distance from each other in this representation.
- **Genuine semantic consolidation** (recognizing that two differently-
  worded statements mean the same thing, while distinguishing statements
  that don't): not demonstrated, and this experiment gives a precise,
  structural reason why the current architecture cannot demonstrate it,
  rather than leaving it an open question.

## 12. Limitations
- The dataset's one-field-different negatives and one-field-different
  paraphrases were constructed by design to occupy the same similarity
  level, in order to test the specific mathematical prediction — this is
  intentional (that's what "hard negative" means here), not a flaw, but
  it means this experiment cannot rule out that some future encoding
  scheme with continuous, graded similarity (rather than 4 discrete
  levels) might behave differently. That would be a representation
  change, out of scope for this threshold-only experiment.
- Only 34 pairs were tested; larger paraphrase/negative sets were not
  constructed, though the discrete-similarity mathematics predicts the
  qualitative conclusion would not change with more data.
- Capacity was held generously unconstrained (200) specifically to
  isolate the threshold's effect from eviction pressure; capacity
  interactions with threshold choice were not explored here.

## 13. Conclusion
**The current Black Hole mechanism cannot perform useful semantic
consolidation with a single fixed similarity threshold on this atomic
representation — not because the threshold is wrong, but because the
representation itself makes genuine paraphrases and genuine
contradictions mathematically indistinguishable.** No threshold region
exists where paraphrase consolidation is high and false consolidation is
low; the sweep is a hard binary switch at 0.667 with nothing achievable
in between. The mechanism's only demonstrated, real capability across
Phases 5 and 6 combined is exact-duplicate deduplication. **The
representation itself — not the threshold, not the update equation —
would need to change (e.g., a continuous/graded encoding capable of
placing "uses" and "regularly uses" closer together than "uses" and
"dislikes," which atomic single-token-per-field hashing cannot do by
construction) before this mechanism could plausibly demonstrate genuine
semantic consolidation.** Per the task's instruction, no further
experiment is recommended until this conclusion has been reviewed.
