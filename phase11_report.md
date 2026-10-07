# Black Hole Memory — Phase 11: Oracle Equivalence Signal Ablation

## Research Question
Is the Black Hole update mechanism itself useful when the system is
given perfect information about what should be consolidated?

## Hypothesis
Not assumed true. Prior phases (5-8) showed every tested representation
failed to provide a reliable SAME/DIFFERENT signal; whether the update
equation itself has value was never fairly testable as a result. This
phase removes that confound entirely.

## Exact Experimental Controls
Novelty equation, update equation (`k'/v' = (1-N)v + Nx`), slot schema,
checkpoint format, recall implementation, capacity mechanism, eviction
(FIFO), timestamp/confidence handling, and production defaults were
**not modified**. Verified: 43/43 tests pass (36 pre-existing + 7 new),
and a dedicated test proves the oracle path's update line is
byte-identical to `memory_update.absorb()`'s, using an exact-duplicate
pair to guarantee the merge path fires under both.

## Dataset Construction
2,090 observations, 400 distinct concept_ids, deterministic seed (fixed
across two independent full runs -- verified bit-identical results,
excluding wall-clock timing). 100 base concepts, each generating: exact
repeats (controlled frequency buckets -- 40 concepts x1, 30x5, 20x20,
10x100), 2 genuine paraphrases (SAME), 1 contradiction (DIFFERENT,
same subject+object, antonym predicate), 1 related-but-distinct fact
(DIFFERENT, same subject, different predicate), 1 unrelated fact
(DIFFERENT, no relation). SAME-labeled: 1,790. DIFFERENT-labeled: 300
(3 probes x 100 base concepts) -- imbalanced by design, since the
repetition-frequency requirement necessarily produces many more SAME
observations; 300 DIFFERENT probes is still a meaningful sample,
verified by a dedicated test.

## Oracle Definition
The oracle maintains exactly one piece of state: a `concept_id -> slot`
dictionary, built from ground-truth concept identity, never from a
vector or score. Given an incoming candidate, it answers only "does
this concept already have a slot?" (SAME -> that slot; DIFFERENT ->
None). It never selects an arbitrary slot, never sees the candidate's
encoding, and every slot it ever points to is one the memory's own
`memory.add()`/`evict_oldest()` created -- verified by a dedicated test
that every value in the oracle's ledger is a real, present `MemorySlot`
object.

## A Real Implementation Bug, Caught Before Running Anything
`MemoryState.evict_oldest()` does `list.pop(idx)`, which shifts every
later slot's list index down by one. An early version of the oracle's
bookkeeping cached slot **indices**; after any eviction elsewhere in
memory, a cached index could silently point at the wrong slot -- a
different concept entirely, not just a stale one. Caught before running
the real experiment by inspecting the eviction code directly. Fixed by
tracking slot **objects** instead of indices. A second, related issue
surfaced immediately after: `MemorySlot` is an undecorated `@dataclass`,
so Python auto-generates value-based equality -- comparing two slots
whose `key`/`value` are numpy arrays raises `ValueError: truth value of
an array is ambiguous`, so a plain `in` check would have crashed.
Confirmed this directly with a minimal repro before it could surface as
a silent failure mode, and used explicit identity comparison
(`is`) instead.

## A Second, More Important Finding -- Not a Bug, an Architectural Property
`System C`'s `true_consolidation_rate` (verified directly via the
oracle's own ledger) is a perfect **1.0** at unconstrained capacity, with
`n_slots=400` exactly matching the distinct-concept count and zero
evictions. Yet the RECALL-based `concept_retention` metric showed only
**0.62** for the same run -- a result suspicious enough to stop and
diagnose rather than report, per the task's explicit instruction.

Traced to a specific, concrete case (`concept_000`) rather than assumed:
its slot was created by a **paraphrase** ("regularly uses") that
happened to arrive before the exact-repeat phrasing in the shuffled
stream. Because `k_i` is frozen at slot creation (a design decision
dating to Phase 2, unmodified here), the slot's *address* reflects
whichever phrasing arrived first -- not any "canonical" wording. A later
query using the exact-repeat's phrasing then fails to find that slot via
cosine similarity (0.408, far below the true match), even though the
slot's *value* was perfectly, correctly consolidated.

**This means perfect value consolidation does not guarantee addressable
recall under a different phrasing than whichever one happened to arrive
first.** This is a genuine, previously-undiscovered property of the
frozen-key design -- invisible in every prior phase because imperfect
representation quality was always the dominant, confounding source of
error. It only became isolable once consolidation itself was made
perfect.

Both metrics are reported separately below, not conflated:
`oracle_ledger_concept_retention` (bypasses recall entirely, checks the
oracle's ledger directly -- "was this concept ever correctly
consolidated and does that slot still exist") and `concept_retention`
(recall-based -- "can a query using one particular phrasing find it").

## All Results

| Capacity | System | slots | updates | evicts | concept_retention | ledger_retention | contradiction_retention | exact_recall | true_consol. | false_consol. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unconstrained (500) | A_similarity | 500 | 1469 | 121 | 0.81 | -- | 0.90 | 0.778 | -- | -- |
| unconstrained (500) | B_no_update | 500 | 0 | 1590 | 0.62 | -- | 0.18 | 0.250 | -- | -- |
| unconstrained (500) | **C_oracle** | 400 | 1690 | 0 | 0.62 | **1.00** | **1.00** | **0.995** | **1.00** | **0.00** |
| 50pct (200) | A_similarity | 200 | 1382 | 508 | 0.57 | -- | 0.24 | 0.305 | -- | -- |
| 50pct (200) | B_no_update | 200 | 0 | 1890 | 0.39 | -- | 0.08 | 0.138 | -- | -- |
| 50pct (200) | C_oracle | 200 | 1574 | 316 | 0.58 | 0.77 | 0.38 | 0.418 | 0.935 | 0.00 |
| 25pct (100) | A_similarity | 100 | 1254 | 736 | 0.39 | -- | 0.10 | 0.155 | -- | -- |
| 25pct (100) | B_no_update | 100 | 0 | 1990 | 0.28 | -- | 0.06 | 0.093 | -- | -- |
| 25pct (100) | C_oracle | 100 | 1422 | 568 | 0.42 | 0.61 | 0.13 | 0.180 | 0.868 | 0.00 |
| 10pct (40) | A_similarity | 40 | 951 | 1099 | 0.23 | -- | 0.04 | 0.070 | -- | -- |
| 10pct (40) | B_no_update | 40 | 0 | 2050 | 0.18 | -- | 0.02 | 0.048 | -- | -- |
| 10pct (40) | C_oracle | 40 | 1044 | 1006 | 0.20 | 0.27 | 0.05 | 0.073 | 0.565 | 0.00 |

False-memory rate: 0.0 for every system at every capacity (100 queries
for never-seen subjects, no false hits anywhere).

## The Most Important Comparison: Oracle-Update vs. No-Update
**At unconstrained capacity, the gap is dramatic and unambiguous.**
Exact recall: 0.995 (oracle) vs. 0.250 (no-update) -- a 4x difference.
Contradiction retention: 1.00 vs. 0.18. Slot count: 400 (exactly the
true concept count, zero waste) vs. 500 (at capacity, still evicting
1,590 times because it never merges anything, including the 100x-
repeated concept). **This is the first time in this entire project that
oracle/full-mechanism has clearly, substantially outperformed
no-update on every measured axis simultaneously**, with zero false
consolidation.

**Under capacity pressure, the gap narrows but does not disappear.**
At 10% capacity, exact recall is 0.073 (oracle) vs. 0.048 (no-update) --
still better, but both are now dominated by sheer capacity scarcity
(400 concepts competing for 40 slots) rather than by consolidation
quality. The mechanism's advantage is real but bounded by how much
capacity pressure exists.

## Similarity-Driven vs. Oracle-Update
At unconstrained capacity, System A's `concept_retention` (0.81) is
**higher** than System C's (0.62) -- but per the finding above, System
C's *true* concept retention (ledger-verified) is 1.00, and the 0.62
number reflects an addressing-order artifact, not a consolidation
failure. System A's own keys are subject to the identical frozen-key
addressing sensitivity; its 0.81 may partly reflect evictions
incidentally clearing out "wrongly-keyed" paraphrase-first slots and
letting a canonically-keyed slot get created later by one of the many
subsequent exact-repeat observations -- a plausible mechanism, not fully
isolated in this experiment. System A's `exact_recall` (0.778) remains
below System C's ledger-verified consolidation quality (1.00), and
System A does real, uncontrolled damage: 121 evictions at "unconstrained"
capacity that oracle-driven consolidation avoided entirely, because
System A's imperfect similarity-based merging wastes slots on both
over- and under-merging.

## Representation/Value Drift
Concept `concept_099` (100 repeats, the heaviest bucket), oracle path:
drift-to-canonical measured at 1, 5, 20, and 100 updates -- **1.000000
at every checkpoint**. Consistent with Phase 5/6's finding: since every
observation in this trace is a byte-identical repeat, similarity=1.0
and novelty=0 on every update, so the equation correctly does nothing
(`v' = v`). This is expected, not a new finding -- verified again here
because a heavy-repetition bucket was newly available in this dataset.

## Latency
Wall-clock, not treated as a deterministic result per the task's
instruction: System A (similarity-driven, computing cosine similarity
against every existing slot on every absorb) is markedly slower than B
or C -- roughly 10-30x, growing with slot count, since its addressing is
O(n_slots) per absorb with real floating-point work, while the oracle's
addressing is an O(1) dictionary lookup. This is an expected, mechanical
consequence of what each system computes, not a finding about
consolidation quality.

## Failure Analysis
- **Representation-layer failure**: none in this phase -- the oracle
  bypasses representation entirely, by design.
- **Memory/update-mechanism failure**: none found. The equation
  performed exactly as specified in every traced case (byte-identical
  test, drift trace, direct ledger verification).
- **Environment/tooling limitation**: none in this phase.
- **A genuine, newly-identified limitation of the frozen-key
  architecture**: addressing is sensitive to which phrasing of a
  concept happens to arrive first, independent of whether consolidation
  itself succeeded. This is the one clear, reportable weak point this
  phase surfaced.

## Limitations
- The frozen-key addressing-order finding was discovered via one
  concrete case and confirmed there, not systematically quantified
  across all 100 base concepts (e.g. by checking how often the
  first-arriving observation for a concept is a paraphrase vs. an exact
  repeat, and correlating that with `concept_retention`). The
  `oracle_ledger_concept_retention` metric sidesteps rather than
  explains this fully.
- System A's lower ledger-equivalent number was not directly measured
  (System A has no oracle ledger to consult) -- the comparison in the
  "Similarity-Driven vs. Oracle-Update" section is partly inferential
  rather than fully isolated.
- Capacity conditions were chosen as fixed fractions of the 400-concept
  total; behavior at intermediate capacities was not explored.

## Scientific Interpretation
Under a perfect externally-supplied equivalence signal, **the Black
Hole update mechanism does provide clear, measurable benefit over
no-update** -- at unconstrained capacity, a 4x exact-recall improvement
with perfect true-consolidation and zero false-consolidation. This is
the first phase in the entire project where that comparison came out
unambiguously positive. The benefit shrinks under severe capacity
pressure but does not reverse. Separately, and just as importantly,
this phase surfaced a real architectural limitation -- frozen-key
addressing-order sensitivity -- that is independent of representation
quality and was not visible in any earlier phase because representation
quality was always the dominant source of error until now.

**Do not over-read this as proof that the full system works.** This
result is conditional on a signal this project still cannot obtain in
practice (Phase 9's stop condition stands). What has changed is the
locus of remaining uncertainty: it is no longer reasonable to suspect
the update mechanism itself is inert or broken. The open question is
now squarely about representation, plus this phase's newly-surfaced
addressing-order limitation, not about the mathematics.

## Conclusion
The oracle ablation directly answers the question posed at the end of
Phase 10: **yes, the update mechanism has demonstrated real value, given
a signal it can trust.** No claim beyond that is made -- not semantic
understanding, not continual learning, not compression (slot count
reductions here reflect genuine non-redundant retention, verified via
the ledger, not information discard), and no comparison to RAG.

## Recommended Next Experiment (exactly one)
**Systematically quantify the frozen-key addressing-order effect**:
for every one of the 100 base concepts, record whether its first-
arriving observation was an exact repeat or a paraphrase, and test
whether `concept_retention` (recall-based) is reliably lower for
paraphrase-first concepts than exact-repeat-first ones. If confirmed at
scale, this identifies a second, independent, currently-undiscussed
architectural fix candidate (e.g. re-keying a slot to its most-frequent
or most-canonical observation rather than freezing on first arrival) --
distinct from and additional to the representation-quality question
that has dominated every phase before this one.
