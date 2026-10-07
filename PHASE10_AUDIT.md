# Phase 10 — Controlled Project Audit

## 1. Original Claims
The Black Hole Memory project set out to demonstrate:
1. An LLM memory system can continuously absorb experience and consolidate
   it into a compact persistent state smaller than the raw input.
2. Semantically equivalent statements (paraphrases) can be recognized and
   merged into one memory unit, while contradictory/unrelated statements
   remain separate.
3. A similarity/novelty/update mathematical mechanism
   (`k' = (1-N)k + Nx`) is the vehicle for that consolidation.
4. The resulting system would outperform raw storage and, eventually,
   RAG-style retrieval, on memory efficiency and recall.
5. (Per task history, not verified in this repository — see Section 8)
   that importance-aware retention and recency tie-breaking could
   improve retention under capacity pressure.

## 2. Experiments Completed

| Phase | Experiment | Result | Conclusion |
|---|---|---|---|
| Extraction (pre-Phase-2) | Rule-based + TF-IDF semantic extraction | Paraphrase generalization worked partially; uncertainty-phrase predicate classification ~65-70% | Extraction frozen as a characterized, imperfect layer; bypassed for memory experiments |
| 2 | Structured-candidate memory, D=64 hash encoding | Exact recall ~24% unconstrained; false-memory ~99-100% | **Negative** — severe hash collision |
| 2A | Dimensionality sweep, D=64→4096 | Exact recall rose to ~97% unconstrained; false-memory stayed ~99-100% regardless of D; update-vs-no-update identical at every D | **Mixed** — collision fixed, mechanism still unproven, false-memory unfixed by D alone |
| 2B | Atomic (whole-field) tokenization | Removed shared-substring floor; exact recall dropped slightly vs. word-split (0.972→0.904 at D=4096); false-memory improved (1.0→0.23) | **Mixed** — real trade-off, not a clean win |
| 5 | Repeat/near-duplicate consolidation, atomic encoding | Exact duplicates: 100% consolidate. Genuine paraphrases: 0% consolidate (mathematically forced — similarity capped at 0.667, below 0.75 threshold) | **Negative** for consolidation beyond exact dedup; mechanism confirmed inert on paraphrases |
| 6 | Threshold sweep (0.50-0.90), 34 independent pairs | Below 0.667: paraphrase and false-consolidation both ≈1.0/0.83 (indistinguishable). At/above 0.667: both 0.0 | **Negative** — no threshold separates paraphrase from contradiction under atomic encoding |
| 7 | Representation ablation: atomic vs. field-aware lexical vs. TF-IDF, 54 pairs | AUC 0.500 / 0.445 / 0.484 — all at or below chance | **Negative** — no tested representation separates paraphrase from contradiction |
| 8 | Real pretrained embedding (spaCy en_core_web_md), 106 pairs | AUC 0.444; contradiction mean similarity (0.85) > paraphrase mean (0.82); memory-level: contradiction consolidation (83%) > paraphrase consolidation (67%) | **Negative** — genuine pretrained embedding still fails; update mechanism shown active (27 vs. 212 slots) but acting on a bad signal |
| 9 | NLI/entailment-aware representation | No legitimate model obtainable (huggingface.co blocked; alternatives exhausted) | **Not tested** — environment-blocked, not a representation failure |

Across every phase where the mechanism *could* be exercised (5, 6, 7, 8),
the update equation itself behaved predictably and mechanically (clean
merges, stable/measured drift, real slot-count reduction whenever
similarity permitted it). It was never the identified point of failure.

## 3. What Has Actually Been Proven
- The update equation, as specified, correctly performs exact-duplicate
  deduplication and repetition-based reinforcement (Phase 5, Test B
  equivalent).
- Increasing encoding dimensionality reduces hash-collision noise and
  materially improves exact recall for distinct facts under
  unconstrained capacity (Phase 2A, verified with direct token/bucket
  measurements, not assumed).
- Atomic (whole-field) tokenization removes a specific, demonstrated
  cross-fact contamination bug (Phase 2A/2B — the "person" shared-
  substring floor), at a measured cost to exact recall.
- None of the four representation types actually tested (hash,
  field-aware lexical, TF-IDF, static pretrained embedding) can reliably
  separate genuine paraphrases from genuine contradictions — this is
  directly measured (AUC ≤0.50 in all four cases), not inferred.
- The memory mechanism correctly executes its own specification
  (addressing, novelty, update, capacity, persistence) — verified by
  36 passing tests plus direct inspection in every phase from 5 onward.

## 4. What Has Been Falsified
- **"Black Hole Memory achieves genuine semantic consolidation"** — not
  merely unproven but actively contradicted: Phase 8's real embedding
  produced *higher* consolidation of contradictions than paraphrases at
  the memory level (0.833 vs. 0.667), the opposite of the intended
  behavior.
- **"A single similarity threshold can separate paraphrase from
  contradiction under atomic encoding"** — mathematically disproven in
  Phase 6 (both categories occupy the identical similarity value,
  0.667) and confirmed empirically across a 34-pair independent dataset.
- **"Fewer stored slots implies better compression"** — repeatedly
  shown false when the reduction comes from merging things that
  shouldn't have merged (Phase 7's Encoder B, Phase 8's System B); this
  project's own reports explicitly reject conflating slot reduction with
  compression.

## 5. What Remains Unproven
- Whether a representation trained specifically for entailment/
  contradiction (as opposed to context-based word similarity) could
  give the update mechanism a workable signal — **untested**, not
  falsified (Phase 9).
- Whether the update mechanism provides a measurable advantage over
  simple deduplication *given a good similarity signal* — every ablation
  run so far either had no signal worth exploiting (Phases 5, 6, 7) or a
  signal that was actively backwards (Phase 8). The "does the math help"
  question has literally never been tested under favorable conditions.
- Whether the system can outperform RAG at any real task — never
  attempted; explicitly deferred in every phase's scope.
- Whether the originally-claimed importance-aware retention and
  recency tie-break mechanisms ("Phase 3"/"Phase 4" in prior task
  framing) actually behave as described — **see Section 8: this code
  does not exist in the repository**, so this claim is neither proven
  nor falsified by this project's actual work; it appears to be a
  discrepancy in how the project's history was described to this agent
  across sessions, not a completed and verified result.

## 6. Representation Results

| Representation | AUC (paraphrase vs. contradiction) | Status |
|---|---:|---|
| Hash (atomic, D=64-4096) | 0.500 (mathematically forced tie) | Tested, failed |
| Field-aware lexical | 0.445 | Tested, failed (below chance) |
| TF-IDF | 0.484 | Tested, failed (below chance) |
| Static pretrained embedding (spaCy GloVe-style) | 0.444 | Tested, failed (below chance) |
| NLI/entailment-aware | — | **Not experimentally evaluated because a legitimate pretrained NLI model could not be obtained in this environment.** |

## 7. Memory Mechanism Results
Separated explicitly, per this audit's mandate:

**Representation/signal quality**: consistently the identified point of
failure across every phase. Whatever encoder was used, similarity values
failed to reliably rank true paraphrases above true contradictions.

**Update/retention mechanism**: behaved exactly as specified in every
inspected case.
- Merges only occur when similarity clears the threshold — verified by
  direct trace in Phases 5, 6, 8.
- Drift after a merge is small and stable (e.g. 0.994-0.999 in Phase 8's
  samples) whether the merge was semantically appropriate or not — the
  equation has no built-in judgment about merge quality, by design; it
  executes whatever the representation tells it to.
- The ablation (update vs. no-update) shows real, substantial activity
  whenever the signal permits merging (Phase 8: 27 vs. 212 slots) and
  correctly shows zero difference when the dataset structurally excludes
  repeats (Phase 5's Test-A-style all-distinct data).
- **The mechanism has never been shown to fail on its own terms.** Its
  apparent failures in every phase trace to the input it was given, not
  to the equation itself.

## 8. Environment Constraints
Phase 9 established, via four documented attempts, that no legitimate
pretrained NLI/entailment model can be obtained in this environment:
huggingface.co (the near-universal distribution channel for such models)
is not on the network allowlist, and no equivalent alternative channel
(GitHub releases, bundled pip packages) yielded a genuine, verifiable
NLI model. One candidate package (`entail`) was found to depend on the
`openai` API rather than a local model and was correctly rejected per
the task's explicit prohibition on LLM-generated labels.

**This audit found and corrected a related item**: Phase 9's `entail`
install pulled in eight transitive dependencies (`openai`, `nltk`,
`rouge-score`, `httpcore2`, `httpx2`, `jiter`, `sniffio`, `truststore`)
that were not fully removed when the primary package was uninstalled.
None were referenced anywhere in the project's code (verified by direct
grep). All eight have now been uninstalled as part of this audit.
`scikit-image` remains installed and is also unreferenced by any project
code; its origin does not trace to the Phase 9 install sequence, so it
is reported as present-but-unexplained rather than misattributed.

## 9. Current Scientific Status
**Partially supported.**

Not "supported": the central claim (genuine semantic consolidation) has
been actively contradicted by the best representation tested so far
(Phase 8), not merely left unproven.

Not "falsified": the update mechanism itself — the actual novel
component this project set out to validate — has never been tested
under conditions where it had a fair chance to succeed. Every
representation tried either provided no exploitable signal (ties at a
single value) or an actively misleading one. A mechanism cannot be
fairly called falsified when it has correctly executed its specification
in every test and the tests that would isolate its own contribution
(good-signal ablations) have not yet been run under favorable
conditions.

Not "inconclusive": too much has actually been determined and ruled out
to describe this as an open question. Four specific representation
approaches have been eliminated with real, measured evidence, not
assumption.

"Partially supported" is the only status that doesn't overstate the
positive (Section 3) or understate what's been ruled out (Section 4).

## 10. Recommended Next Experimental Question
**Does the Black Hole update mechanism provide a measurable advantage
over deduplication when given a representation with GENUINE, verified
separation between paraphrase and contradiction — using ground-truth
labels as that representation, not a real encoder?**

This does not require pretending an NLI model exists. It uses the
dataset's own known ground-truth equivalence labels (already collected
across Phases 6-8) to construct an *oracle* similarity signal — e.g.
similarity = 1.0 for ground-truth-equivalent pairs, 0.0 otherwise —
feeding the exact same unmodified memory mechanism. This isolates the
one variable that has never been cleanly tested: given a signal that is
*by construction* correct, does the update equation's specific behavior
(as opposed to simple deduplication) produce any measurable benefit? If
even an oracle signal shows no advantage over no-update, that would be
strong, direct evidence about the mechanism itself, independent of any
representation's quality — resolving Section 5's largest open question
without requiring any additional model access this environment cannot
provide.

---

## Audit Verification

**Full test suite result**: 36/36 tests passed (`tests/test_basic.py`,
`tests/test_checkpoint.py`, `tests/test_memory_state.py`,
`tests/test_semantic_encoder.py`), run in full before and after the
dependency cleanup performed as part of this audit.

**No existing test files were modified.** Only unused, unreferenced
dependencies (`openai`, `nltk`, `rouge-score`, `httpcore2`, `httpx2`,
`jiter`, `sniffio`, `truststore`) were uninstalled; no project source
file in `phase2/`, `core/`, or `benchmarks/` was changed.

**No production algorithm was modified.** This audit performed
inspection, dependency cleanup, and documentation only, per its
mandate — `MemoryState`, `memory_update.py`'s equations, `encoders.py`,
and `structured_candidate.py` are byte-for-byte unchanged from their
state at the end of Phase 9.
