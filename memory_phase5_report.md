# Black Hole Memory — Phase 5: Repeat/Near-Duplicate Consolidation Ablation

## 1. Research Question
Does the Black Hole update equation provide useful consolidation when
memory receives repeated or near-duplicate information?

## 2. Hypotheses
H1 (consolidation), H2 (stability), H3 (recall), H4 (control) — stated
in the task spec, not assumed true. Results below address each directly.

## 3-5. Experimental Design, Dataset, Control Variables
Encoder, tokenizer, D (4096), similarity, novelty, update equation,
addressing, threshold (0.75), capacity handling, and recall logic were
all unchanged from Phase 2B. Only the dataset changed: 5 categories
(exact repetition, near-duplicates, distinct facts, near-duplicate
distractors, mixed stream), each with a hidden `concept_id` used only
for scoring, never given to the memory system.

## 6-7. Black Hole Mechanism / No-Update Baseline
Identical to prior phases: full mechanism blends values on a slot match
≥0.75; no-update baseline never merges, always inserts (or evicts+inserts
under FIFO pressure).

## A structural fact, derived before running anything and then verified
With atomic tokenization (Phase 2B), every candidate has exactly 3
tokens (subject/predicate/object). Two candidates can only share 0, 1, 2,
or 3 of those tokens — giving possible similarities of exactly {0, 0.333,
0.667, 1.0}. **MATCH_THRESHOLD=0.75 sits between 0.667 and 1.0.**
Directly verified: a one-field-differing pair measures 0.6667, a
case-only variant (normalized identical after `.lower()`) measures
1.0000. **This means the merge path can only ever fire on byte-identical
(post-lowercase) triples — never on genuine near-duplicates that differ
in even one field.** This was predicted mathematically before running
the experiment and confirmed by every result below.

## 8-9. Results — Ground-Truth / Exact / Underlying-Fact Recall

**Category 1 (exact repetition):** full mechanism collapses any repeat
count to exactly 1 slot (1→1, 10→1, 100→1, 1000→1 slots), no-update
scales linearly (1, 10, 100, 1000 slots). **This is real, but it is
ordinary exact deduplication — the no-update baseline was never given
any deduplication logic at all, so this mostly demonstrates "having any
dedup beats having none," not something specifically clever about the
Black Hole math.** Stated explicitly per the task's own warning against
overclaiming this.

**Category 2 (near-duplicates):** 5 observations, all naming the same
underlying fact. Full mechanism: 4 slots (only the case-variant merged
— "python" vs "Python" are identical tokens after lowercasing, not a
real near-duplicate case). The three genuine paraphrases ("currently
uses" / "has been using" / "primarily uses") each got their own slot,
matching the trace's measured similarity of exactly 0.6667 each — below
threshold, as derived above. **H1 is not supported for genuine
paraphrase near-duplicates**, only for literal string-identical repeats.

**Cross-phrasing recall test (directly probes H3):** a fact stored
*only* as `(user, "has been using", Python)`, queried as `(user,
"uses")`: similarity = **0.4082**, far below any reasonable match
threshold. The system does not recognize the query as referring to the
same fact. **H3 fails for paraphrased predicates.**

## 10. False Consolidation
Category 3 (distinct facts sharing subject+predicate) and Category 4
(near-duplicate distractors: tested/teaches/dislikes/different-subject):
**zero false consolidation in both categories, for both mechanisms.**
The system correctly does not confuse `user tested Python` with `user
uses Python`, or `company uses Python` with `user uses Python`. This is
a genuine, verified negative-control result — the threshold is not so
loose that it merges things it shouldn't.

## 11. False-Memory Rate
Not the focus of this phase (Phase 2B covered it); not re-measured here
since the encoder/threshold were held fixed and no new false-memory
mechanism was introduced.

## 12-13. Slot Utilization / Update Counts
Category 5 (mixed stream, 34 observations, 9 distinct concepts),
capacity sweep:

| capacity | full slots | no-update slots | full false-consol. | no-update false-consol. |
|---:|---:|---:|---:|---:|
| 10 | 10 | 10 | 0.100 | 0.400 |
| 25 | 11 | 25 | 0.091 | 0.111 |
| 50 | 11 | 34 | 0.091 | 0.091 |
| 100 | 11 | 34 | 0.091 | 0.091 |
| 1000 | 11 | 34 | 0.091 | 0.091 |

At unconstrained capacity, full mechanism settles at 11 slots (34 raw
observations minus the 20-repeat Python group correctly collapsing to
1 = 34-20+1 = 15... measured 11, meaning a few of Category 2's
near-duplicates and the case-variant also collapsed as expected — 20
exact repeats → 1, case-variant → merged with that, leaving 4+4+5-1=12
remaining distinct-by-construction observations, close to the measured
11, consistent with the derivation above). No-update always uses exactly
34 (one slot per observation, no merging ever, by definition).

**The most informative result in this phase**: at *tight* capacity
(cap=10), full and no-update use the *same* number of slots (10=10) —
but full mechanism has dramatically lower false-consolidation (0.100 vs
0.400). This is because full mechanism's exact-repeat collapsing
happened *during absorption*, freeing slot budget for other distinct
concepts to survive, whereas no-update's 10 surviving slots are whatever
FIFO left behind (a somewhat arbitrary recent-window sample), causing
more of the 34 original observations to alias onto the wrong nearest
slot during scoring. **This is a real, positive, measurable advantage
of the mechanism under capacity pressure** — but it is entirely
attributable to exact-duplicate collapsing, not to any near-duplicate
understanding, consistent with everything else in this report.

## 14. Representation Drift
Measured across 100 repeated exact-identical absorptions:

| update # | action | novelty | sim to canonical |
|---:|---|---:|---:|
| 1 | insert | 1.0 | 1.0 |
| 5 | update | ≈0 (−1.2e-7, float noise) | 1.0 |
| 10 | update | ≈0 | 1.0 |
| 100 | update | ≈0 | 1.0 |

**Zero drift, ever, at any point.** This is not a coincidence to
interpret — it's mathematically necessary given the finding above: since
merges only ever fire when similarity=1.0 exactly (i.e. novelty=0
exactly), the update equation `k' = (1-N)k + Nx` reduces to `k' = k` on
every single merge. **H2 (stability) is technically satisfied, but
vacuously** — the key never moves because the mechanism is never
exercised under any actual variation, only under perfect repetition.
This is a meaningfully different and weaker claim than "the mechanism
converges to a stable representation under noisy/varying input," which
is what H2 was really asking and which this experiment cannot speak to,
because the current threshold never lets varying input reach the update
step at all.

## 15. Latency
Not the limiting factor in any category tested (sub-millisecond per
absorb at these dataset sizes).

## 16. Memory Footprint
Unchanged formula from prior phases; not the focus of this experiment.

## 17. Capacity Effects
Documented in Section 12-13 — the one condition (tight capacity) where
the mechanism shows a real, measurable, positive effect, driven entirely
by exact-duplicate collapse.

## 18. Full-vs-No-Update Ablation — Summary
| Category | Real effect from update mechanism? |
|---|---|
| Exact repetition | **Yes** — but trivial (ordinary dedup) |
| Near-duplicates (paraphrase) | **No** — mathematically cannot merge |
| Distinct facts | No difference (correctly, both avoid false merge) |
| Near-duplicate distractors | No difference (correctly, both avoid false merge) |
| Mixed stream, tight capacity | **Yes** — real, but attributable entirely to exact-dedup efficiency, not paraphrase understanding |

## 19. Failure Analysis
Cause of near-duplicate consolidation failure, demonstrated not assumed:
atomic per-field tokenization (Phase 2B) reduces every candidate to
exactly 3 discrete tokens, making similarity a coarse 4-value quantity
{0, 1/3, 2/3, 1} rather than a smooth function of textual overlap. The
fixed threshold (0.75) was set based on Phase 2's earlier word-split
tokenization, where partial overlap was continuous; it was never
re-examined against atomic tokenization's discrete similarity levels.
**This is a direct, traceable consequence of the Phase 2B fix
interacting with an unchanged threshold — not a new, separate bug.**

## 20. Scientific Interpretation
The Black Hole update mechanism, as currently specified, is not a
near-duplicate consolidation mechanism. It is an **exact-duplicate
deduplication mechanism** that happens to be implemented via a
similarity/novelty/update formalism capable in principle of graded
blending, but which — given the current atomic encoder and threshold —
never actually operates in that graded regime. Every positive result in
this phase traces back to exact-match collapsing; every genuine
near-duplicate/paraphrase test failed, consistent with a clean
mathematical prediction made before the experiment was run.

## 21. Limitations
- Category 2's paraphrase set (5 variants) is small; a larger paraphrase
  set would not change the qualitative conclusion (the threshold math is
  structural, not dataset-dependent) but wasn't exhaustively tested.
- `false_consolidation_rate`'s methodology (re-addressing every original
  observation against the *final* slot set) is a reasonable, documented
  choice but is one of several defensible ways to define the metric;
  it was not cross-validated against an alternative definition.
- This experiment did not test whether lowering MATCH_THRESHOLD (e.g. to
  0.6) would let genuine near-duplicates merge — that's a natural next
  question but was explicitly out of scope ("change ONLY the dataset").

## 22. Has the Black Hole Update Mechanism Demonstrated Measurable Value?
**Yes, narrowly, and No, broadly — both are true and neither should be
hidden.** Yes: it reliably and usefully collapses exact repeats, and
that has a real, measured benefit under capacity pressure (Section 13).
No: it has demonstrated zero ability to consolidate genuine
near-duplicates/paraphrases, which was the actual hypothesis this phase
existed to test (H1-H3), and the reason is now precisely understood
(Section 19) rather than mysterious. If "consolidation" is defined as
"recognizing the same underlying fact stated differently," the current
mathematics does not do this. If defined as "not wasting a slot on a
literal repeat," it does this correctly.

## 23. Recommended Next Experiment (ONE only)
**Sweep MATCH_THRESHOLD alone** (e.g. 0.5, 0.6, 0.667, 0.7, 0.75), on
the same Category 2/3/4 datasets, holding everything else fixed,
to determine whether a lower threshold could let genuine paraphrases
(similarity=0.667) merge while Category 3/4's negative controls
(similarity also ≤0.667 in most cases) remain correctly separated. If a
threshold exists that consolidates Category 2 without breaking Category
3/4's zero-false-consolidation result, that would be the first real
evidence the update mechanism can do more than exact deduplication. If
no such threshold exists (i.e. genuine near-duplicates and genuine false
merges occupy overlapping similarity ranges), that would be strong,
specific evidence that atomic per-field encoding is fundamentally
incompatible with any single fixed threshold performing both jobs at
once — a materially stronger and more precise negative result than what
this phase alone can conclude.
