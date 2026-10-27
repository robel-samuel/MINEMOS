# Black Hole Memory — Phase 7: Representation Ablation

## Research Question
Can a richer representation create a measurable separation between
semantic equivalence and contradiction while leaving the memory
mechanism completely unchanged?

## Hypothesis
Not assumed true. Phase 6 demonstrated the atomic encoder cannot
separate paraphrases from contradictions. This phase tests whether that
is fixable by representation alone, or whether it reflects something
deeper.

## Frozen Components
Similarity function, novelty equation, addressing, MATCH_THRESHOLD
(0.75), update equation, slot structure, capacity, eviction, recall,
checkpointing, evaluation methodology. Verified via regression: 36/36
tests pass identically before and after all Phase 7 changes. The only
code change at the mechanism layer was one additive `encoder` parameter
on `absorb()`, defaulting to the existing atomic encoder so every prior
test and result is reproduced exactly.

## Encoder Definitions
- **Encoder A (control)**: unmodified Phase 2B/5/6 atomic encoder,
  wrapped, not reimplemented.
- **Encoder B (field-aware lexical)**: each field word-split and
  field-prefixed (`"predicate:regularly"`, `"predicate:uses"`), so
  multi-word fields contribute partial overlap. Deterministic, no
  hand-written rules for any specific sentence.
- **Encoder C (semantic embedding): not implemented.** No pretrained
  embedding model is available in this environment — the network
  allowlist has no model-hosting domain, and no weights are
  pre-installed. Reported directly as a limitation rather than worked
  around. A TF-IDF lexical-statistical variant was tested as an
  additional comparison point, explicitly labeled **not** semantic (the
  same honesty standard applied to TF-IDF throughout this project's
  earlier extraction-layer phases).

## Dataset Construction
54 new, independently-constructed pairs (none reused from Phase 5/6)
across all 8 required categories: paraphrase (8), contradiction (8),
unrelated (8), same-subject-different-predicate (6),
same-predicate-different-object (6), same-object-different-subject (6),
lexical-overlap distractors (6), exact duplicates (6).

## Mathematical Expectation
None stated in advance beyond "richer word-level overlap should, in
principle, allow more graded similarity than atomic's 4 discrete
levels" — this phase was explicitly exploratory rather than predicted
in closed form, unlike Phase 6.

## Complete Similarity Results

| Encoder | paraphrase mean | contradiction mean | unrelated mean | Margin M | Margin M2 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| A (atomic) | 0.6667 | 0.6667 | 0.0000 | 0.0000 | 0.6667 | 0.500 |
| B (field-aware lexical) | 0.6522 | 0.6754 | 0.0000 | **-0.0232** | 0.6522 | **0.445** |
| C (TF-IDF, not semantic) | 0.4179 | 0.4318 | 0.0000 | **-0.0139** | 0.4179 | 0.484 |

**None of the three tested encoders separate paraphrases from
contradictions. Two of the three (B and C) show a *negative* margin —
contradictions score marginally *higher* than paraphrases on average.**
AUC hovers at or below chance level (0.50) for all three.

## Pair-Level Separability
`pct_paraphrases_above_every_contradiction = 0.0` for every encoder — no
paraphrase pair scored higher than the single worst-case contradiction
pair, for any encoder tested. For Encoder A this is a mathematical
certainty (all one-field-different pairs tie at exactly 0.6667). For B
and C, it's an empirical finding, not a certainty — but it holds anyway.

## Root Cause, Traced to a Specific Mechanism (not just observed)
Inspected individual pairs to understand *why* Encoder B failed, rather
than reporting the aggregate and stopping. Example: `contra_scifi`
("reads sci-fi novels" vs "hates sci-fi novels" — a genuine
contradiction, zero literal word overlap in the predicate) scored
**0.80**, higher than several genuine paraphrases. Cause: both facts
share the multi-word object "sci-fi novels" (2-3 tokens), which
dominates the total token count and **dilutes the contribution of the
one field that actually carries the meaning difference.** This is a
structural flaw, not noise: field-aware word-splitting makes similarity
sensitive to how many words each field happens to contain, regardless
of which field is doing the semantic work of agreeing or disagreeing.
A short predicate change is swamped by a long shared object (or vice
versa) essentially at random, depending on sentence construction.

## Memory-Level Results (Phase 6 dataset, frozen threshold=0.75, encoder swapped)

| Encoder | exact_dup_rate | paraphrase_rate | false_consolidation_rate | full_slots | no_update_slots |
|---|---:|---:|---:|---:|---:|
| A (atomic) | 1.0 | 0.0 | 0.0 | 34 | 68 |
| B (field-aware lexical) | 1.0 | **0.5** | **0.333** | 23 | 68 |

Encoder B does move the needle versus Encoder A at the memory level —
paraphrase consolidation rises from 0.0 to 0.5. But false consolidation
rises right along with it, from 0.0 to 0.333. **This fails the
experiment's explicit success criterion** ("high paraphrase
consolidation AND low false consolidation") — the representation
change produced a real effect, but not a useful one.

## Full-vs-No-Update Ablation
For both encoders, the full mechanism used substantially fewer slots
than no-update (A: 34 vs 68; B: 23 vs 68) — confirming the update
mechanism itself is doing real, measurable work at this threshold, for
both encoders. **This means the memory mathematics is not the
bottleneck here** — it correctly acts on whatever similarity the
encoder gives it. The bottleneck is entirely upstream, in what
similarity values the encoder produces.

## Latency / Memory Footprint
Comparable between encoders (~0.09-0.10ms/absorb); not a differentiator.
State bytes lower for B (756KB vs 1.1MB) simply because it merged more
aggressively (fewer slots), not because of any per-slot efficiency gain.

## Failure Cases
- Every genuine paraphrase in the Phase 6 dataset that shares zero
  literal words with its original phrasing (e.g. "likes"→"enjoys",
  "manages"→"leads") remains unmerged by Encoder B for the same reason
  it failed under Encoder A: no representation tested here can recognize
  synonymy that isn't also lexical overlap.
- Encoder C (TF-IDF) underperformed Encoder B on every metric, including
  AUC — consistent with TF-IDF weighting rare words rather than encoding
  meaning, exactly as documented in its own module docstring.

## Limitations
- Only one richer representation (word-level, field-prefixed hashing)
  was tested as "Encoder B" — other non-neural representations (e.g.
  character n-grams, edit-distance-based similarity) were not attempted
  and might behave differently; this experiment does not rule them out.
- No genuine pretrained semantic embedding was available to test, which
  is the most likely candidate for actually solving this problem — this
  experiment can say what atomic and lexical representations *cannot*
  do, but cannot speak to what a real embedding model could do.
- The dataset's paraphrase/contradiction pairs were constructed with
  roughly matched sentence lengths where possible, but the field-length-
  dilution effect (Root Cause section) means small, uncontrolled
  differences in field word-count between pairs materially affect
  results — a more length-balanced dataset might shift the exact
  numbers, though it's unclear it would change the qualitative
  conclusion (near-chance AUC).

## Scientific Conclusion
**Outcome C.** No tested representation — atomic, field-aware lexical,
or TF-IDF — provides useful separation between genuine paraphrases and
genuine contradictions. Two of the three richer representations
performed *worse* than the trivial atomic control on the primary
separability metric (AUC). The memory-level experiment confirms the
bottleneck is representational, not mathematical: the same update
mechanism, given the same frozen threshold, correctly acts on whatever
similarity signal it receives — it just receives a signal that cannot
distinguish "worded differently" from "means something different,"
regardless of which of the three tested encoding schemes produced that
signal. Per the task's own outcome framework, this points toward the
current hypothesis requiring a deeper representational change than
anything achievable with deterministic lexical/hash-based encoding —
not toward tuning the existing approaches further.

## Recommended Next Experiment (exactly one)
**Determine whether a genuine pretrained sentence-embedding model
(obtained through a legitimate channel outside this environment's
current constraints, e.g. run once offline and the resulting vectors
supplied as static, precomputed input) produces AUC and separation-
margin results meaningfully above what atomic/lexical/TF-IDF achieved
here (≈0.44-0.50).** This is the one representational category Phase 7
could not test due to environment constraints, and it's the most
direct way to determine whether "Outcome C" reflects a fundamental
limit of this project's memory mechanism or specifically a limit of
non-learned representations — a distinction this phase's results cannot
resolve on their own.
