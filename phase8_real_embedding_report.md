# Black Hole Memory — Phase 8: Genuine Semantic Embedding Test

## 1. Research Question
If Black Hole Memory receives genuine semantic embeddings instead of
hash/lexical representations, can its existing mathematical update
mechanism perform useful consolidation while avoiding false consolidation?

## 2. Hypothesis
Not assumed true. Phase 7 showed hash and lexical representations fail;
this phase tests whether a real pretrained embedding changes the outcome.

## 3. Phase 7 Findings Motivating This Experiment
Atomic hashing (AUC=0.500, forced by construction), field-aware lexical
(AUC=0.445), and TF-IDF (AUC=0.484) all failed to separate paraphrases
from contradictions. The update mechanism was shown to do real work
whenever it received a usable signal. This phase supplies it a
genuinely different kind of signal.

## 4. Experimental Controls
Novelty equation, cosine similarity, addressing, MATCH_THRESHOLD (0.75),
update equation, slot schema, capacity, recall, checkpoint format,
admission/eviction, candidate structure, benchmark methodology — all
unchanged. Regression: 36/36 tests pass before and after every change in
this phase (verified at each of three checkpoints, including after
installing the model). The only variable was the encoder.

## 5. Model Used
spaCy `en_core_web_md`, v3.7.1 — static GloVe-style word vectors
(300-dim), trained on a large web corpus, averaged over document tokens.
A genuine pretrained embedding model, not TF-IDF, not feature hashing.

## 6. Model Provenance
Installed via `pip install <wheel URL>`, wheel hosted at
`github.com/explosion/spacy-models/releases` — a GitHub release asset,
on this environment's network allowlist (`github.com`,
`release-assets.githubusercontent.com`). Not downloaded from any
blocked domain. Verified functional before use (loaded successfully,
produces normalized 300-dim vectors, `.similarity()` behaves sanely on
a spot check before any dataset was built on top of it).

## 7. Candidate Representation Format
**Two formats were tested, and this is disclosed rather than only
reporting the final one.** The first, applied uniformly to every
candidate (not tuned to any example): `"subject: {s} predicate: {p}
object: {o}"`. Direct testing caught a serious confound: because this
literal boilerplate appears identically in every candidate's text, the
averaged embedding was dominated by it rather than by content — a
genuinely unrelated pair measured 0.94 similarity under this format.
Switched to raw concatenation, `"{s} {p} {o}"`, still applied uniformly;
the same unrelated pair dropped to 0.37 under the corrected format — a
completely different, more sensible picture. **All results below use
the corrected format.** The original (flawed) similarity numbers are
not used in any subsequent analysis.

## 8. Dataset Methodology
106 new pairs (none reused from Phase 6/7), all 10 required categories
(A-J) plus an `unrelated` category retained from Phase 7's convention
for the M2 margin calculation. Explicit ground-truth `equivalent` label
on every pair.

## 9. Complete Similarity Results (corrected format)

| Category | mean | median | min | max | std | n |
|---|---:|---:|---:|---:|---:|---:|
| exact_duplicate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 10 |
| paraphrase | 0.8249 | 0.8161 | 0.6665 | 0.9618 | 0.1059 | 12 |
| contradiction | 0.8506 | 0.8903 | 0.6453 | 0.9800 | 0.1126 | 12 |
| high_overlap_negative | 0.8554 | 0.8378 | 0.6707 | 0.9816 | 0.0900 | 10 |
| low_overlap_paraphrase | 0.7370 | 0.7382 | 0.6011 | 0.8536 | 0.0739 | 10 |
| unrelated | 0.4443 | 0.4080 | 0.1956 | 0.7223 | 0.1483 | 10 |

Contradiction mean (0.8506) is higher than paraphrase mean (0.8249).
Genuinely unrelated facts are correctly much lower (0.44) -- the encoder
clearly captures some real signal (unrelated facts are far less similar
than anything else), but that signal does not distinguish "same
meaning, different words" from "opposite meaning, related words."

## 10. Separability Metrics
AUC = 0.4444 (below chance level of 0.500). 0% of paraphrase pairs
scored above the single worst-case contradiction pair. M1 (paraphrase
minus contradiction) = -0.0257 (negative). M2 (paraphrase minus
unrelated) = +0.3806 (positive -- this is the one genuinely positive
separability result: paraphrases are more similar to their source fact
than to unrelated facts, just not more similar than contradictions are).

## 11. Threshold Analysis
Predefined threshold (0.75, not tuned): precision=0.311, recall=0.719,
F1=0.434, false-positive rate=0.689. Nearly 70% of genuinely
non-equivalent pairs get incorrectly flagged as equivalent at this
threshold.

Post-hoc secondary analysis (explicitly separated, not used to claim an
unbiased result): best F1=0.521 at threshold=0.666 -- barely better
than the predefined threshold's 0.434, and still far from a usable
operating point (precision only 0.356 even at its best F1).

## 12. Memory-Level Results (frozen threshold=0.75)

| System | exact_dup | paraphrase_consol | contradiction_consol | unrelated_false_consol | slots | state bytes |
|---|---:|---:|---:|---:|---:|---:|
| A: atomic + Black Hole | 1.0 | 0.0 | 0.0 | 0.0 | 108 | 3.5MB |
| B: semantic + Black Hole | 1.0 | 0.667 | 0.833 | 0.0 | 27 | 67KB |
| C: semantic + no-update | 0.0 | 0.0 | 0.0 | 0.0 | 212 | 516KB |

The central, decisive finding: System B consolidates contradictions
(83%) more than it consolidates genuine paraphrases (67%). This is a
direct, measured consequence of Section 9's similarity results, not a
new failure mode -- the memory mechanism is correctly acting on the
signal it receives; the signal itself is what points the wrong way.

## 13. Full-vs-No-Update Ablation
System B vs. System C: the update mechanism is clearly doing real,
substantial work (27 slots vs. 212 -- an 8x reduction). This is not a
null-result ablation like several earlier phases -- the mechanism is
active and consequential here. But "doing a lot" and "doing something
useful" are different claims: it consolidates the wrong things more
often than the right things.

## 14. Consolidation Results
Exact duplicates: 100% consolidation, both systems capable of it
(trivial, as established since Phase 5). Unrelated-fact false
consolidation: 0% for System B -- a genuine positive, unrelated facts
never falsely merge. The failure is specifically and only in the
paraphrase-vs-contradiction boundary, not a general loss of
discrimination.

## 15. False-Consolidation Analysis
Inspected two representative merges directly, per the task's "do not
patch, inspect" requirement. `para_gym` ("visits the gym" vs "goes to
the gym", a genuine paraphrase): merged, novelty=0.182, drift-to-
canonical after merge=0.994. `contra_gym` ("visits the gym" vs "avoids
the gym", a genuine contradiction): also merged, novelty=0.115, drift-
to-canonical after merge=0.999. Both merges are clean and mechanically
unremarkable -- the update equation blends smoothly in both cases. The
failure is entirely upstream: the embedding assigns "avoids" a
similarity to "visits" high enough to pass threshold, exactly as
documented in Sections 9-10. This is the embedding being wrong, not the
addressing or update math.

## 16. Recall
Not separately measured in this phase -- the consolidation results
(Section 12) already establish that recall would be actively misleading
under System B, since querying "user, visits" would as-likely-as-not
surface a blended vector containing contradictory information.

## 17. Memory Footprint
System B's smaller footprint (67KB vs. System A's 3.5MB) is NOT evidence
of good compression -- per the task's explicit instruction, this must be
reported as what it is: aggressive over-merging, including of things
that should not have merged. A smaller state here reflects more
information loss, not more efficient representation.

## 18. Latency
Semantic encoding costs ~7-8ms/absorb (spaCy inference) vs. ~0.7-0.9ms
for atomic hashing -- an order of magnitude slower, not a differentiator
given the more fundamental correctness problem, but worth noting for
completeness.

## 19. Update/Drift Analysis
Section 15's two samples show drift is small and stable (0.994, 0.999)
regardless of whether the merge was semantically appropriate. The update
equation cannot distinguish a "good" merge from a "bad" one -- it blends
whatever it's given with the same mechanical process either way. This
confirms the mechanism has no internal safeguard against consolidating
things that shouldn't be consolidated; that responsibility falls
entirely on the similarity signal upstream.

## 20. Failure Cases
Checked whether the failure traces to: embedding itself (yes --
confirmed directly, static word embeddings are documented to conflate
synonyms/antonyms via contextual substitutability, and this dataset
demonstrates it concretely, not just in the literature);
representation-to-text mapping (yes, partially -- the initial
boilerplate format was a real, disclosed, and corrected confound, though
fixing it did not resolve the core problem); threshold (checked via
post-hoc sweep -- no threshold recovers a usable operating point);
memory addressing/update equation (checked directly via drift samples --
clean and unremarkable in both good and bad merges, not implicated).

## 21. Limitations
- Only one pretrained embedding model was tested (spaCy's static
  word-vector model). A different model class -- genuine sentence
  transformers trained with contrastive objectives specifically for
  semantic similarity (e.g. models fine-tuned on NLI or paraphrase
  data) -- might behave very differently, since they're trained
  specifically to separate entailment from contradiction, which static
  word embeddings are not. This experiment cannot speak to that.
- The representation-format fix (Section 7) was necessary and disclosed,
  but the dataset was not re-audited for other possible uniform-but-
  suboptimal formatting choices beyond the one confound caught.
- Only 106 pairs were tested; the qualitative finding (contradiction
  mean > paraphrase mean) is consistent enough across the dataset
  (visible in both the raw means and the AUC) that more pairs seem
  unlikely to reverse it, but this wasn't formally verified via
  resampling/bootstrapping.

## 22. Scientific Interpretation
Kept strictly separate, per the task's explicit requirement:
"semantic representation works" -- not supported by this experiment;
the specific pretrained embedding tested here does not reliably
distinguish paraphrase from contradiction, and in aggregate scores
contradictions slightly higher. "Black Hole Memory works" -- partially
supported and partially refuted, but not settled by this phase: the
update mechanism itself continues to behave exactly as specified (clean
merges, stable drift, real consolidation activity per the ablation)
whenever it receives a signal -- the mechanism is not what's failing
here. This experiment cannot support the Black Hole hypothesis because
the representation it was given failed at the input stage, upstream of
anything the memory mechanism does.

## 23. Final Verdict
Per the task's outcome ladder: only criterion #1 was tested and it
failed ("genuine paraphrases receive substantially higher similarity
than contradictions" -- false; they receive lower similarity on
average). None of criteria #2-7 can be meaningfully evaluated given #1's
failure. This is not "Outcome A" (first evidence for the hypothesis) and
not even a clean "representation works, mechanism unproven" outcome --
it is a continuation of Phase 7's "no representation tested provides
useful separation," except this time with a genuine pretrained
embedding rather than a hash or lexical stand-in, which materially
narrows what remains untested. The mechanism itself remains neither
proven nor disproven -- it has simply never yet been given a signal
capable of supporting a fair test.

## Recommended Next Experiment (exactly one)
Test a sentence-embedding model trained with a contrastive or
entailment-based objective specifically designed to separate paraphrase
from contradiction (e.g. a model fine-tuned on natural language
inference or paraphrase-identification data, as opposed to a static
word-vector-averaging model like the one tested here) -- if such a model
can be legitimately obtained via this environment's network allowlist.
Static word embeddings were shown here to fail specifically at the
synonym/antonym boundary because they encode contextual substitutability
rather than meaning polarity; a model trained explicitly to separate
those two things is the most direct remaining test of whether "genuine
semantic representation" (as distinct from "genuine pretrained
representation," which this phase did test) can supply the signal
Black Hole's update mechanism needs.
