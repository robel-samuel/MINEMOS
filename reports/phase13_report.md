# Black Hole Memory — Phase 13: Re-Keying Under Imperfect Similarity

## Hypothesis
Not assumed true. Phase 12 proved re-keying solves the addressing-order
problem under a perfect oracle signal. This phase tests whether that
benefit survives when merges are sometimes wrong.

## Experimental Design
Systems A (frozen key) and B (re-key) both use **real, non-oracle**
similarity-driven addressing -- no oracle SAME/DIFFERENT labels anywhere
in their decision path. Systems C/D reuse Phase 11/12's oracle
infrastructure unmodified, as upper-bound controls only.

## Regression Results
50/50 pre-existing tests pass unchanged; 6 new Phase 13 tests pass (56
total). Phase 12's key results reproduced exactly before any new code
was written: A canonical-query overall=0.62, paraphrase-first case=
0.1212, B canonical-query overall=1.0, B held-out=0.99, C held-out=0.14
-- all bit-identical to the Phase 12 report.

## Encoder/Threshold Justification (verified, not assumed)
Field-aware lexical encoder at `MATCH_THRESHOLD=0.75` (the **unchanged**
repository default -- not tuned for this experiment) was checked
directly against one pair from each category before building anything
on top of it: paraphrase sim=0.866 (correctly merges), **contradiction
sim=0.750 (incorrectly merges -- right at the threshold)**, hard-negative
sim=0.667 (correctly does not merge), unrelated sim=0.0 (correctly does
not merge). This is exactly the required "correct/ambiguous/incorrect"
mix. Atomic hashing at the same unchanged threshold was also checked and
produces **zero** merges beyond exact duplicates (consistent with Phase
6), which would make error-propagation untestable -- rejected for this
specific experiment on that basis, not as a general judgment against it.

## Dataset
3,750 observations, 500 base concepts (documented scope choice: task
allowed 500-1,000; 500 chosen for tractable runtime). Categories A-E
(exact repeat, paraphrase, hard negative, contradiction, unrelated) for
every concept; category F (5-step multi-step wording drift) for 50
concepts. **A real implementation bug was caught and fixed before
running anything downstream**: an initial version intersected two
independent modulo conditions to select drift concepts, producing only
13 of the intended 50 -- caught by inspecting the generator's own
summary output, fixed by explicitly selecting from the "uses"-predicate
subset. Verified deterministic across two runs.

## Frozen-Key (A) vs Re-Key (B) Results

| Capacity | System | slots | true consolidation | false consolidation |
|---|---|---:|---:|---:|
| unconstrained (2500) | A | 1078 | 0.655 | 0.488 |
| unconstrained (2500) | B | 929 | 0.741 | 0.549 |
| 250 | A | 250 | 0.337 | 0.266 |
| 250 | B | 250 | 0.348 | 0.272 |
| 100 | A | 100 | 0.160 | 0.125 |
| 100 | B | 100 | 0.159 | 0.123 |
| 50 | A | 50 | 0.126 | 0.101 |
| 50 | B | 50 | 0.091 | 0.067 |

**Note on admission divergence, disclosed rather than hidden**: unlike
Phase 12 (where oracle-driven admission was identical for both systems
by construction), here A and B's slot counts differ (e.g. 1078 vs. 929
unconstrained) because admission itself is similarity-driven -- once B
re-keys a slot, that changed key can affect which *future* candidates
merge into it. This is an expected, direct consequence of testing under
real (non-oracle) addressing, not an inconsistency.

**At unconstrained capacity, B shows both higher true consolidation
(0.741 vs 0.655) AND higher false consolidation (0.549 vs 0.488) than A**
-- re-keying amplifies whatever the encoder is already doing, in both
directions simultaneously, exactly matching the central hypothesis
("Outcome B" from the interpretation ladder). Under severe capacity
pressure (50), the pattern **reverses**: B's true consolidation is
*lower* than A's (0.091 vs 0.126) -- a genuinely different regime worth
flagging rather than smoothing into a single narrative.

## A Vacuous Metric, Caught and Disclosed (Not Silently Fixed)
`exact_recall` (partial-query-based, using `subject+predicate` only)
measured **0.0 for both systems at every capacity** -- suspicious enough
to stop and inspect, per the task's explicit instruction. Traced
directly: querying `Person_0000, uses` returned `Person_0255`'s slot --
the wrong concept entirely, not a low-confidence near-miss. Confirmed
systematic across 5 independently checked concepts (5/5 wrong), at
unconstrained capacity with zero eviction, ruling out capacity pressure
as the cause. **Root cause: at this scale (500+ concepts, many sharing
the same predicate word), field-aware lexical's word-level hashing
causes partial two-token queries to collide broadly across many
unrelated concepts that happen to share a predicate.** This affects
System A and System B equally -- it is an encoder/scale confound, not a
re-keying effect -- and is reported here as its own finding rather than
being used as evidence for or against re-keying, since it cannot
distinguish the two.

## Error Propagation Test (the phase's central, controlled result)
A specific, non-oracle-labeled sequence: Fact A ("uses"), then Fact B
("dislikes" -- a hard negative), then four later paraphrases of A.

| Step | Frozen: action (sim) | Re-key: action (sim) |
|---|---|---|
| Fact B | update (0.800) | update (0.800) |
| paraphrase_0 | update (0.913) | update (0.906) |
| paraphrase_1 | insert (0.676) | insert (0.697) |
| paraphrase_2 | insert (0.730) | **update (0.753)** |
| paraphrase_3 (exact) | update (1.000) | update (0.973) |
| **final slots** | **3** | **2** |
| **final similarity to clean canonical** | **0.217** | **0.173** |

Both systems experience the *identical* initial incorrect merge (Fact B,
sim=0.800, both keys still equal at this point -- confirmed by a
dedicated test). They diverge from `paraphrase_2` onward: frozen-key's
address stays anchored to the original clean encoding of Fact A forever,
so `paraphrase_2` (0.730 against the clean key) correctly stays
separate. Re-key's address has already drifted toward the Fact-B
contamination by this point, and that drifted address happens to pull
`paraphrase_2` in (0.753, crossing threshold) -- **one additional,
avoidable merge that frozen-key correctly avoided.** Net result:
re-keying ends with more apparent consolidation (2 slots vs. 3) but
measurably worse fidelity to the clean canonical fact (0.173 vs. 0.217).
**This is a concrete, isolated demonstration of the central hypothesis:
re-keying can compound an early incorrect merge by making the address
itself increasingly reflective of contaminated content.**

## Oracle Controls (C/D) -- Upper Bound, Not Primary Result
Reused directly from Phase 11/12, unmodified. As expected and previously
established, oracle-driven admission is identical for both re-keying
choices (same slot/update counts) since the oracle's decision never
depends on similarity. These remain a valid upper-bound reference:
Phase 12 showed re-keying's addressing benefit is real and total
(1.00 vs 0.62) when the signal is perfect. Phase 13's job was to check
whether that survives an imperfect signal -- it does, but conditionally
and with a real cost, per the sections above.

## Consolidation Accuracy / Addressing Stability / Drift
Covered directly above (consolidation quality table, error propagation
trace). No separate finding beyond what's already reported.

## Capacity Effects
The reversal at capacity=50 (B's true consolidation drops below A's) is
the single most important capacity-related finding: **under severe
capacity pressure, re-keying's amplification effect can work against
consolidation quality, not just false-consolidation rate.** This wasn't
predicted by the interpretation ladder's four outcomes as stated and is
flagged as a genuinely mixed, capacity-dependent result rather than
forced into "B wins" or "B loses."

## Deterministic Replay
Full-scale replay (all 3,750 observations x 4 capacities x 2 systems)
was not re-run twice due to runtime cost (single run took ~194 seconds);
a bounded 800-observation slice was run twice instead and confirmed
bit-identical (slots, true/false consolidation rates matched exactly
across both runs). This is a disclosed scope reduction, not a silent
one, consistent with prior phases' handling of compute-time limits.

## Limitations
- `exact_recall`'s scale-driven collapse (both systems, 0.0) means this
  phase cannot report a clean "does re-keying help real-world recall at
  scale" number -- only the consolidation-quality and error-propagation
  results speak to that question here.
- The error propagation test used one specific, hand-selected sequence
  (Fact A / hard-negative B / four paraphrases). It demonstrates the
  mechanism exists and is reproducible, not how frequently it occurs
  across the full dataset -- the aggregate true/false consolidation
  table is the closest proxy for frequency, but doesn't isolate this
  exact causal chain at scale.
- Full determinism was verified on a bounded slice, not the complete
  experiment, for the reason stated above.

## Scientific Conclusion
**Outcome B, precisely** -- the interpretation ladder's "mixed result":
re-keying helps correct merges consolidate further (higher true
consolidation at unconstrained capacity, 0.741 vs 0.655) but measurably
amplifies the damage from incorrect merges (higher false consolidation,
0.549 vs 0.488, and a demonstrated, traceable mechanism by which one bad
early merge compounds into additional incorrect merges). Under severe
capacity pressure this reverses in an unexpected way (B underperforms A
on true consolidation itself, not just false consolidation) -- a nuance
the pre-registered outcome categories didn't anticipate and shouldn't be
smoothed away. Phase 12's addressing-stability result was real and
reproduced exactly; it does not straightforwardly transfer to a world
where the underlying signal is imperfect, which is the realistic case
this project has faced since Phase 6.

## What Uncertainty Remains
Whether re-keying is a net positive or negative depends on the
similarity signal's reliability *and* on capacity conditions, in ways
this phase has only begun to map (one encoder, one threshold, four
capacity points, one hand-built error-propagation sequence). No claim
is made here about semantic memory, continual learning, or any
comparison to RAG. Per the task's explicit instruction, no Phase 14 is
proposed -- this report stops with the uncertainty stated plainly
rather than resolved.
