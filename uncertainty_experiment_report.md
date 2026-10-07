# Uncertainty-Phrase Predicate Classification — Final Extraction Experiment

## 1. Baseline results

27/27 existing tests passed before any change. Main A/B benchmark (unchanged
throughout this experiment): rule extractor 1.0 recall / 1.148-1.153x
compression; semantic encoder 1.0 recall / 1.148-1.153x compression
(matching, since the free-text fix in the prior phase).

## 2. Test-set description

33 newly-written sentences (`benchmarks/uncertainty_eval_set.py`), none
reused from any existing test, the generator, or the exemplar bank.
Covers direct statements, uncertain statements, negation, historical
statements, strong-confidence statements, 10 distinct predicate types,
5 subject types, and unseen paraphrases. Ground truth recorded across 5
independent dimensions (subject, predicate, object, confidence band,
negation) per `benchmarks/uncertainty_scorer.py`, so failures can be
localized rather than collapsed into one score.

## 3. Baseline scores (before any fix)

| Dimension | Accuracy |
|---|---:|
| Subject | 1.000 |
| Predicate | 0.667 |
| Object | 1.000 |
| Confidence band | 0.697 |
| Negation | 0.939 |
| **Overall (all 5 must agree)** | 0.545 |
| Missing (no extraction at all) | 0.000 |

Predicate classification and confidence-band classification are the two
weak dimensions; subject and object extraction are solid.

## 4. Root cause (demonstrated, not assumed)

Printed the top-3 similarity scores and matched exemplar text for every
failing case. Pattern, shown directly by the numbers:

```
"I'm considering using Rust."          tokens: [consider, using, rust]
  0.371  framework          "I'm using Django as the framework."
  0.360  backend_language   "Most of my backend code is written in Rust."
```

```
"I'm not sure whether I'll move to Rust."   tokens: [not, sure, whether, ll, move, rust]
  0.439  dislikes_language  "I'm not a fan of TypeScript."
  0.342  backend_language   "Most of my backend code is written in Rust."
```

With only 2-5 content tokens surviving stopword removal on short hedged
sentences, **a single coincidental shared token with one exemplar
sentence dominates the cosine score** — "using" alone routes a Rust
sentence to `framework` because that predicate's only exemplar happens to
contain "using"; "not" alone routes it to `dislikes_language` because
that predicate's exemplars happen to use "not" in "I'm not a fan of...".
This is nearest-single-exemplar matching being fragile against sparse,
uneven exemplar sets — an exemplar-selection / TF-IDF-weighting
interaction, not a stemming or cue-word problem.

## 5. Fix attempted

Switched predicate matching from nearest-single-exemplar to
nearest-centroid: average each predicate's exemplar vectors into one
prototype before comparing, a standard technique (Rocchio-style
classification) that should dilute exactly the single-token-domination
failure mode observed. Implementation preserved the `Candidate`
interface, touched no memory-layer code, required no model download, and
was not tuned to any specific eval-set sentence.

## 6. Before/after metrics

| Dimension | Before | After (centroid) |
|---|---:|---:|
| Subject | 1.000 | 0.939 ↓ |
| Predicate | 0.667 | 0.667 → (same count, different failures) |
| Object | 1.000 | 0.939 ↓ |
| Confidence band | 0.697 | 0.667 ↓ |
| Negation | 0.939 | 0.879 ↓ |
| Overall | 0.545 | 0.545 → |
| Missing rate | 0.000 | 0.061 ↓ (new silent failures) |

Existing test suite: **1 of 27 tests newly failed**
(`test_1_semantic_paraphrase_consolidation_unseen_wording` — one of the
four originally-passing unseen paraphrases, "I mostly build server-side
applications with Python.", was reclassified from `backend_language` to
`deploy_platform` under centroid matching).

## 7. Regression analysis

The fix produced **no net improvement and multiple regressions**:
predicate accuracy — the dimension it targeted — was unchanged in count
(22/33 both times, a different 11 wrong each time), while four other
dimensions got measurably worse and two eval cases that previously
extracted *something* now extract nothing at all. This is not a
close call or a minor tradeoff to accept; it's a clean negative result
across the board.

**Decision: reverted.** Per the experiment's explicit stop condition
("if the change produces no meaningful improvement, revert it rather
than accumulating unnecessary complexity"), the codebase is back to the
exact pre-experiment state — confirmed by rerunning the eval set and
getting identical baseline numbers, rerunning all 27 tests (pass), and
rerunning the main conversation A/B benchmark (unchanged: 1.0 recall,
1.148-1.153x compression at all three scales).

## 8. Remaining failures (unresolved, not hidden)

Uncertainty-phrase predicate misclassification is **still present**,
unchanged from the prior report: roughly a third of predicate
classifications are wrong, concentrated in short hedged/uncertain
sentences where lexical overlap with the wrong exemplar dominates.
Confidence-band accuracy (0.697) means about 30% of the time the
low/high/rejected confidence assignment doesn't match expectation either
— partly because the uncertainty cue-phrase list (`UNCERTAINTY_PATTERNS`)
is narrow (misses "may use", "could potentially", "leaning towards...
but haven't decided", "there's a chance", "it's still up in the air") and
doesn't fire on plausible hedges outside its exact phrase list.

## 9. Interpretation

The root cause was correctly diagnosed and demonstrated with real
intermediate scores, and the fix was principled, not ad hoc — but a
correct diagnosis doesn't guarantee the first reasonable-sounding fix
actually helps net, and this is a clear example of that. The deeper issue
is that TF-IDF similarity on short sentences is inherently noisy
regardless of whether matching is done against individual exemplars or
centroids; centroids just moved which sentences fail rather than reducing
how many fail. This ceiling looks structural to the TF-IDF approach
itself, not fixable by adjusting how exemplars are aggregated.

## 10. Recommendation

**Freeze the extraction layer here, as instructed.** The extraction
layer is now sufficiently characterized: strong on subject/object
extraction and negation detection, weak (~65-70%) on predicate
classification and confidence-band assignment for short hedged
sentences, and this weakness appears to be a structural property of
lexical similarity matching rather than something one more targeted
patch will fix. Continuing to iterate here risks exactly the
accumulating-complexity-without-improvement pattern this experiment was
designed to catch. The next phase's plan — testing Black Hole's actual
memory mechanism (M_t, Update, consolidation, retention, forgetting) on
clean structured candidates that bypass extraction uncertainty entirely —
is the right move: it isolates the mechanism this whole project is
actually about from a measurement problem (extraction accuracy) that has
now been characterized rather than solved.
