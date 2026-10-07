# Black Hole Memory — Phase 12: Consolidation-Aware Addressing

## Research Question
Does making memory addressing consolidation-aware improve retrieval of
already-consolidated concepts without changing the update mechanism or
introducing a semantic model?

## Hypothesis
Not assumed true. Phase 11 found a gap between ledger-verified concept
retention (1.00) and recall-based retention (~0.62), traced to a
paraphrase-first arrival. This phase tests two specific, minimal fixes.

## Systems A/B/C
**A (frozen-key control)**: literally Phase 11's `oracle_absorb`,
imported and called unmodified -- not reimplemented, so there is no risk
of the control subtly drifting from Phase 11's actual behavior.
**B (re-key on consolidation)**: identical update; after a SAME-update,
`slot.key` is replaced with the post-update `slot.value`.
**C (multi-key alias)**: identical update; primary key stays frozen
exactly as in A; the incoming candidate's encoding is additionally
stored as an address-only alias, keyed by slot identity (not list index
-- consistent with the eviction-safety fix from Phase 11).

## Exact Experimental Controls
Novelty equation, update equation, eviction, capacity, oracle SAME/
DIFFERENT labels, value representation -- unmodified. Verified directly,
not assumed: all three systems produce **identical** slot counts (300),
insert counts (300), and update counts (1,730) on the same stream --
confirming addressing is the only thing that differs, exactly as
required. 50/50 tests pass (43 pre-existing + 7 new).

## A Clarification Worth Stating Precisely, Not Glossed Over
Concept-level *value* trajectories are **not** numerically identical
across A/B/C over multiple updates, and this was checked directly
before being reported either as a problem or dismissed. Traced to a
2-observation case: at the very first update, A and B compute identical
similarity/novelty (both systems' keys are still equal -- insertion
never re-keys). Divergence begins only from the *second* update onward,
purely because System B's key has by then legitimately changed, which
correctly changes the novelty/alpha computed on the next update. **The
update equation itself is unchanged -- verified byte-identical in code
across all three systems.** What differs is the equation's *input* (the
current key), which is precisely and only the variable this experiment
is testing.

## Dataset Construction
100 concepts, 2,030 observations, deterministic seed (verified
bit-identical across two independent full runs). Each concept assigned
one of three ordering cases (canonical_first, paraphrase_first, mixed --
582-666 concepts each) and a repeat-count bucket (2/5/20/100). Per-
concept internal ordering is exact and controlled; observations are
interleaved *across* concepts via a deterministic round-robin so
absorption order isn't trivially sequential. Each concept also includes
a contradiction and an unrelated control, plus a `held_out_predicate` --
a paraphrase wording that never appears during absorption but names the
same concept, used to test generalization.

## Query Construction
Four query conditions per concept: canonical wording, each of the two
seen paraphrases, and the held-out (never-absorbed) wording. Success is
defined strictly against ground-truth concept identity (does the
returned *slot object* match the oracle ledger's slot for that
concept?) -- not "did something similar-looking come back."

## Results (unconstrained capacity, 500)

| System | canonical query | para_0 query | para_1 query | held-out query | contradiction sep. | false-memory |
|---|---:|---:|---:|---:|---:|---:|
| A (frozen) | 0.620 | 0.660 | 0.510 | 0.550 | 1.00 | 0.00 |
| **B (re-key)** | **1.000** | **1.000** | **1.000** | **0.990** | 1.00 | 0.00 |
| C (alias) | 1.000 | 1.000 | 1.000 | **0.140** | 1.00 | 0.00 |

Breakdown of System A's canonical-query recall by ordering case -- the
clearest confirmation of Phase 11's hypothesized mechanism:

| Ordering case | A: canonical-query recall |
|---|---:|
| canonical_first | **1.000** |
| paraphrase_first | **0.121** |
| mixed | 0.727 |

This is precisely the predicted pattern: when the canonical wording
happens to arrive first, frozen-key addressing works perfectly. When a
paraphrase arrives first, canonical-wording queries fail the large
majority of the time. `mixed` (shuffled per-concept) falls in between,
as expected.

## Addressing-Gap Analysis
`addressing_gap = concept_retention (1.00 for all three systems) -
addressable_recall_canonical`:

| System | addressing gap |
|---|---:|
| A | **0.380** |
| B | **0.000** |
| C | **0.000** |

System B eliminates the gap entirely for every wording actually seen
during absorption. System C also eliminates it for seen wordings -- but
only for those.

## The Decisive, Unexpected Finding: Generalization to Unseen Wording
System B's held-out recall (0.990) is dramatically higher than System
C's (0.140), despite both scoring identically (1.000) on every *seen*
wording. Mechanism, confirmed by construction: System B's re-keying
sets the slot's key to the running consolidated *value* -- effectively a
centroid of every wording blended so far -- so a genuinely new
paraphrase that shares the same subject/object still lands close to
that centroid. System C's aliases are literal, one-per-wording
addresses; they provide zero benefit for a wording that was never one of
them. **This is a real generalization advantage for re-keying that the
addressing-gap metric alone would not have revealed** -- worth surfacing
explicitly rather than stopping at "both eliminate the gap."

## Contradiction Separation / False-Memory
Identical across all three systems: 1.00 contradiction separation, 0.00
false-memory rate. Neither fix introduces any regression on the
negative-control side.

## Memory Footprint

| System | base bytes | n_aliases | alias bytes | total | bytes/concept |
|---|---:|---:|---:|---:|---:|
| A | 9,837,600 | 0 | 0 | 9,837,600 | 32,792 |
| B | 9,837,600 | 0 | 0 | 9,837,600 | 32,792 |
| C | 9,837,600 | 1,730 | 28,344,320 | 38,181,920 | 127,273 |

System C costs **~3.9x** the storage of A/B for strictly worse
generalization. This is an unambiguous trade-off, not a close call.

## Latency
Wall-clock (not treated as deterministic, reported separately per the
task's instruction): A=0.095s, B=0.159s, C=0.325s for the full 2,030-
observation stream. B's re-keying adds a small, expected constant cost
per update (recomputing/copying the key). C's cost is higher and grows
with alias count, since scoring recall must check every alias on every
query.

## Failure Analysis
- No representation-layer failure (oracle-driven throughout).
- No update-mechanism failure -- verified identical admission decisions
  across all three systems, and the value-divergence question was
  investigated directly rather than left as an unexplained anomaly.
- No environment/tooling limitation encountered.
- System C's failure mode (poor generalization to unseen wording) is a
  genuine, structural property of alias-based addressing, not a bug --
  aliases are exact-match bookkeeping by design; expecting them to
  generalize would be a different, unimplemented mechanism.

## Limitations
- Held-out generalization was tested with exactly one unseen wording
  per concept (the 3rd paraphrase pool entry). Whether System B's
  centroid-based generalization holds for wordings more different than
  this project's paraphrase pool was not tested.
- System B's re-keying was not tested under repeated *contradictory*
  updates reaching the same slot (this dataset's contradictions always
  create their own separate concept, by oracle design) -- an open
  question for a future phase, not this one.
- Both fixes were tested only under the oracle's perfect SAME/DIFFERENT
  signal; this says nothing about how either interacts with an
  imperfect, representation-driven signal (Phases 5-8).

## Scientific Interpretation
Per the task's explicit constraint: this result should be read only as
**addressing stability**, not semantic understanding. Re-keying (System
B) makes a consolidated memory reliably findable regardless of which
equivalent wording arrived first, and does so via a mechanically simple,
non-semantic operation -- replacing the address with the already-computed
consolidated value. Its generalization to never-seen wording is a
consequence of that value being an average over multiple wordings, not
of the system understanding meaning.

## Conclusion
**System B (re-key-on-consolidation) resolves the addressing-order
problem Phase 11 identified, completely, at zero additional storage
cost, and with a genuine (if secondary) generalization benefit that
System C's alias approach does not share.** System C achieves the same
result for seen wordings only, at ~4x the storage cost, with no
generalization benefit -- a strictly dominated option given these
results. The interpretation ladder from the task spec resolves cleanly:
"If B improves retrieval: re-keying may solve the addressing-order
problem" -- confirmed, decisively. "If C improves retrieval more than B:
multiple addresses may be preferable" -- not supported; C does not
improve *more* than B, and loses badly on the one axis (generalization)
where they differ.

## Recommended Next Experiment (exactly one)
**Test System B's re-keying under the imperfect, representation-driven
similarity signal from Phases 6-8, instead of the oracle.** This phase
deliberately isolated addressing from representation quality; the
natural next question is whether re-keying's benefits survive contact
with a real (non-oracle) similarity signal -- specifically, whether
repeatedly re-keying a slot based on imperfectly-merged values (where
some merges may have been the false-consolidation events documented in
Phases 6-8) causes the address to drift toward something *worse* than a
frozen key would have been, or whether it remains a net improvement
even under representation noise.
