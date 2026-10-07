# Black Hole Memory — Phase 2B: Atomic-Tokenization Experiment

## Note on document discrepancy
The task document's "CURRENT STATE" section describes a Phase 3
(importance-aware admission/eviction) and Phase 4 (recency tie-break) as
already completed. **Nothing in this codebase implements either.** Every
verified experiment through Phase 2A used plain FIFO eviction, and no
importance-weighting or recency-tie-break code exists anywhere in the
repository. This experiment proceeds using what is actually in the code
(FIFO), not the described Phase 3/4 behavior, and this discrepancy is
flagged rather than silently absorbed into the research narrative.

## 1. Research Question
Does removing the tokenizer-induced shared-token floor (Phase 2A,
Section 7) change the measured behavior of the Black Hole update
mechanism, specifically the full-vs-no-update ablation result?

## 2. Hypothesis
The shared "person"/predicate-word floor was contaminating false-memory
measurements and possibly the ablation; removing it should improve
false-memory rate and might reveal a real update-mechanism advantage
previously masked by this confound.

## 3. What Was Changed
`phase2/structured_candidate.py`'s `encode_full`/`encode_query`: each
field (subject, predicate, object) is now hashed as **one atomic token**
(e.g. `"subject:person_000001"`) rather than word-split
(`"person"`+`"000001"`). This is the only code change made.

## 4. What Was Deliberately NOT Changed
Update equation, cosine similarity, novelty equation, addressing logic,
capacity, admission/eviction (still FIFO), MATCH_THRESHOLD (0.75),
dataset generator/seed, D values tested, scoring method. No semantic
embeddings, learned parameters, recency, importance, or contradiction
handling were added.

## 5. Baseline Results (pre-change, reconfirmed)
36/36 tests passing. D=64, (1000,1000): exact_recall=0.240, slots=127,
false_mem=1.0. D=64, (1000,100): exact_recall=0.192, false_mem=0.99.
D=4096, (1000,1000): exact_recall=0.972, false_mem=1.0. D=4096,
(1000,100): exact_recall=0.024, false_mem=0.06. All identical to the
Phase 2A report.

## 6. Post-Tokenizer-Fix Results
36/36 tests still pass (behavioral properties preserved; exact numeric
outputs were not expected to match, since tokenization intentionally
changed).

## 7. Collision/Similarity Analysis
Unrelated-fact pairwise similarity, atomic tokenization:

| D | shared-predicate pair | fully-unrelated pair |
|---:|---:|---:|
| 64 | 0.333 | 0.000 (one pair: -0.333, residual collision) |
| 512 | 0.333 | 0.000 |
| 4096 | 0.333 | 0.000 |

The floor is now exactly attributable to genuinely shared predicates
(1 of 3 tokens shared → cosine = 1/3), not spurious subject overlap — a
correct, expected, and arguably desirable floor (predicates *should* be
comparable; that's what makes contradiction/repeat detection possible at
all). The one residual collision artifact at D=64 (a pair sharing zero
fields yet showing -0.333 similarity) had already vanished by D=512.

## 8-9. Test A: Exact Recall

| D | (100,10) | (100,100) | (1000,100) | (1000,1000) |
|---:|---:|---:|---:|---:|
| 64 | 0.040 | 0.440 | 0.000 | 0.086 |
| 512 | 0.080 | 0.940 | 0.002 | 0.454 |
| 4096 | 0.100 | 1.000 | 0.002 | 0.904 |

**Unexpected finding, reported as measured, not smoothed over: exact
recall is uniformly WORSE than the old (word-split) tokenization at
every D tested** — e.g. D=4096/(1000,1000): 0.972 (old) → 0.904 (new);
D=64/(1000,1000): 0.240 (old) → 0.086 (new). Root cause, verified
directly: atomic tokenization reduces each candidate to exactly 3
nonzero vector components (one per field) instead of ~15-20 word-level
sub-tokens. This removes the shared-substring floor (good), but also
removes the redundancy that was making the old encoding more *robust*
to any single hash collision — with only 3 tokens, one field-hash
collision now aliases a full third of a candidate's representation,
whereas before it was diluted across many overlapping word-hashes. Fewer
spurious matches, but each real collision that does occur is more
damaging. This is a genuine trade-off, not a strict improvement.

## 10. False-Memory Rate

| D | (100,100) | (1000,1000) |
|---:|---:|---:|
| 64 | 0.820 | 1.000 |
| 512 | 0.220 | 0.920 |
| 4096 | 0.020 | 0.230 |

Substantially improved at D=4096 for the smaller config (0.02, down from
effectively total failure before), and meaningfully improved at
(1000,1000) too (1.0 → 0.23), though still far from zero. Confirms part
of the hypothesis: the tokenizer floor was a real contributor to
false-memory failure, and removing it helps, especially at moderate
scale — but does not fully solve it at larger unconstrained-capacity
scale.

## 11. Capacity Effects
Capacity-constrained configs ((1000,100)) collapsed to near-zero exact
recall (0.000-0.002) at every D — worse than the old tokenization's
already-poor capacity-pressure numbers. Consistent with the
single-point-of-failure explanation: under heavy eviction pressure, the
sparser 3-token representation is even less forgiving of any addressing
error.

## 12. Latency
Comparable to Phase 2A; not the limiting factor (absorb latency stayed
under ~3ms even at D=4096, n=1000).

## 13. Memory Footprint
Unchanged formula; footprint scales with D exactly as before (536 bytes/
fact at D=64 up to ~32.8KB/fact at D=4096) — atomic tokenization doesn't
change vector size, only what's encoded into it.

## 14. Regression-Test Results
36/36 passed both before and after the change. No hidden or skipped
failures.

## 15. Failure Cases
- Exact recall regression at every tested D (Section 8-9) — a real cost
  of this fix, not just a benefit.
- False-memory rate improved substantially but did not reach zero even
  at D=4096 with unconstrained capacity (0.23).
- Capacity-constrained recall remains near-zero regardless of D or
  tokenization scheme.

## 16. Full-Mechanism vs No-Update Ablation

| D | config | full recall | no-update recall | diff |
|---:|---|---:|---:|---:|
| 64 | (1000,1000) | 0.086 | 0.078 | +0.008 |
| 64 | (1000,100) | 0.000 | 0.002 | -0.002 |
| 512 | (1000,1000) | 0.454 | 0.456 | -0.002 |
| 512 | (1000,100) | 0.002 | 0.002 | +0.000 |
| 4096 | (1000,1000) | 0.904 | 0.904 | **+0.000** |
| 4096 | (1000,100) | 0.002 | 0.002 | **+0.000** |

At D=4096, full and no-update are **identical on every measured
metric**, including slots used (1000 = 1000, i.e. essentially no merging
occurred at all) and false-memory rate. The small ±0.008/-0.002
differences at lower D are within noise, not a consistent signal in
either direction.

**Important methodological limitation, not previously stated this
precisely**: Test A's dataset is constructed so every fact has a unique
subject — there are no repeats and no genuine near-duplicates by design.
The merge/update path can only diverge from no-update behavior when a
candidate's similarity to an existing key clears 0.75, which (with the
tokenizer floor now removed) essentially only happens for exact repeats
or coincidental residual collision. **Test A structurally cannot
exercise the update mechanism's intended use case** (Test B's repetition
scenario, confirmed working in the original Phase 2 report, is a
different dataset). The near-zero ablation difference here is therefore
expected by dataset design, not a fully general verdict on whether the
update mechanism has value anywhere.

## 17. Interpretation
The tokenizer fix worked as intended for what it targeted: the
similarity floor is now attributable only to genuine field overlap, and
false-memory rate improved meaningfully. But it introduced a real,
measured cost (worse exact recall via reduced token redundancy) that
was not anticipated in the original hypothesis, and the critical
ablation — while now cleaner and more trustworthy — is answering a
narrower question than "does the update mechanism ever help": it's
answering "does it help distinguish already-known-distinct facts from
each other," which the mechanism was never designed to do.

## 18. Limitations
- The exact-recall regression (Section 8-9) is itself only partially
  explained (single-point-of-failure theory, verified structurally via
  nonzero-component count, but not verified via a controlled collision-
  rate comparison at matched conditions).
- Test A/ablation scope limitation (Section 16) means this experiment
  cannot rule out the update mechanism having value on repeat-heavy or
  near-duplicate-heavy data; it can only report that it shows no value
  on all-distinct data, which is close to the null hypothesis by
  construction.
- Only 4 of the original 6 Test A configs were re-run (matching Phase
  2A's reduced-scope precedent for compute-time reasons); n=10,000
  configs were not re-verified under atomic tokenization.

## 19. Has the Black Hole Update Equation Demonstrated Measurable Benefit?
**Still no**, and the evidence is now cleaner than before: at the
highest tested D, full mechanism and no-update are bit-identical on
every metric on the all-distinct-facts dataset. This experiment
substantially improved confidence in *why* — the ablation is no longer
confounded by tokenizer artifacts — but it also revealed that this
specific ablation was never capable of showing the mechanism working, by
the dataset's own design. The honest state: zero evidence of benefit
where tested, and the place most likely to show benefit (repeated/near-
duplicate facts) has not yet been tested with the cleaned-up encoder.

## Recommended Next Experiment (ONE only)
**Re-run the full-vs-no-update ablation on a dataset that actually
contains repeated and near-duplicate facts** (Test B/C-style data, not
Test A's all-distinct facts), using the current atomic-tokenization
encoder and the same fixed D/threshold/capacity parameters. This is the
first ablation that would give the update mechanism a fair chance to
show a difference from no-update, since Test A structurally cannot.
