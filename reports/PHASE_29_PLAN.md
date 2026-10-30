# MINEMOS Phase 29 Plan — Temporal Invariant Coverage and Verifier Telemetry

**Date:** October 29, 2026
**Baseline commit:** `3afc2e480ba09d23c696d123fcc5c59fb5eb39a8`
**Status:** PRE-REGISTERED — written before implementation and execution.

---

## 1. Why this phase

Phase 28 closed the five Phase 23 failure modes, and the October 29 reliability
repair (`3afc2e4`) fixed the three defects found by independent validation (R1
canonical-subject lookup, R2 invariant false positives, R3 SAME-over-changed-value
history loss). The repair review left six residual risks:

1. The invariant may miss unlinked exclusive histories, especially state imported
   from legacy or external checkpoints.
2. Link-based grouping does not verify that linked slots share a logical subject
   and attribute.
3. SAME decisions on non-active (historical) slots need tests for unintended
   interval overlaps.
4. Value equivalence depends on the configured verifier; borderline results need
   explicit treatment.
5. `telemetry mode` is not classified exclusive; retaining both values is
   conservative but leaves its temporal semantics unresolved.
6. Additional value-equivalence verifier calls are not reflected in telemetry.

README's roadmap also names full-scale integrated end-to-end accuracy as the key
pending experiment. That experiment is **out of scope for Phase 29**: it measures
retrieval/retention at scale, whereas the evidence above concerns the diagnostic
invariant and the temporal boundary. Mixing them would make a failure impossible
to attribute.

## 2. Research question

After the R1–R3 repairs, is `temporal_history_violations()` sound and complete
over the state sources the engine can hold, and does the SAME-over-changed-value
policy avoid interval overlap when the matched slot is historical rather than
active?

Sub-questions:

- **Q1 (link integrity):** Are linked slots always the same canonical subject and
  a compatible attribute, and does the invariant report violations instead of
  silently grouping structurally inconsistent links?
- **Q2 (historical SAME):** If a SAME decision matches a closed historical slot
  and no other candidate is selected, does the recurrence supersede the active
  state, or does it insert a second active slot with an unreported overlap?
- **Q3 (conservative retention):** Does adding structural link checks preserve
  the R2 result that conservatively retained same-attribute statements are not
  reported as violations?
- **Q4 (unlinked legacy conflicts):** Can the invariant detect two active
  non-equivalent values of a known exclusive attribute when no supersession link
  exists, without flagging the conservative-retention case from R2?
- **Q5 (verifier boundary):** Which outcomes (SAME / NEUTRAL / CONTRADICTION) are
  treated as value equivalence, and is that treatment stable across the configured
  verifier?
- **Q6 (telemetry):** Are value-only verifier calls visible in `status()`?

## 3. Hypotheses

- **H1:** Link validation will report cross-subject and cross-attribute links
  without changing any interval outcome for engine-produced chains.
- **H2:** A closed-history SAME recurrence currently falls through to a plain
  insert when the active chain head is not selected; the resulting two active
  slots are invisible to the link/duplicate-only grouping.
- **H3:** Making the closed-history recurrence deterministically target the
  active chain head removes the overlap while preserving multi-valued inserts.
- **H4:** Structural link checks do not reintroduce the R2 false positives.
- **H5:** Unlinked legacy conflicts remain a known false-negative class because
  auto-grouping by predicate identity would re-create R2 false positives; the
  phase will measure, not "fix" it.
- **H6:** Value equivalence is decision-based (only SAME is equivalent), and
  value-only calls can be counted separately without changing `total_nli_calls`.

## 4. Baseline

- Engine commit `3afc2e4`.
- Prior results: 20/20 reliability tests, 143 focused tests, 357 full-suite tests,
  54/54 independent scenarios (repair run).
- Phase 28 committed benchmark JSON untouched.
- Baseline run for this phase is the engine at `3afc2e4` before any Phase 29 edit;
  the historical-recurrence case is executed first to confirm H2.

## 5. Controlled variables

- Pinned dense encoder `sentence-transformers/all-MiniLM-L6-v2` and NLI gate
  `cross-encoder/nli-distilroberta-base`, CPU, `local_files_only=True`.
- Verification mode `bidirectional`, entail 0.60, contra 0.40, thresholds frozen.
- Deterministic stub verifiers for state-machine cases; real NLI used only where
  the verifier decision itself is under test.
- Capacity, timestamps, subjects, predicates, and objects fixed in the case
  matrix before execution.
- No retention-policy, consolidation, or checkpoint-schema changes.

## 6. Acceptance criteria

- **A1:** All engine-produced legitimate-coexistence cases return no violations
  (zero false positives).
- **A2:** Each injected structural corruption is reported: linked overlap,
  multiple active, duplicate active value, invalid interval, cross-subject link,
  cross-attribute link.
- **A3:** A closed-history SAME recurrence either consolidates into the active
  head or supersedes/historically splices it; no case leaves two active slots
  with an unreported overlap.
- **A4:** Multi-valued and cumulative facts still coexist after the recurrence
  fix.
- **A5:** `status()` exposes value-only verifier calls separately from
  `total_nli_calls`.
- **A6:** The R2 conservative-retention case still returns `[]`.
- **A7:** No regression: Phase 29 tests, the 20 repair tests, and the full suite
  pass; Phase 28 benchmark JSON is unchanged.

## 7. Pre-registered case matrix

| ID | Class | Setup | Expected |
| --- | --- | --- | --- |
| L1 | legitimate | two team memberships | no violations |
| L2 | legitimate | two managed projects | no violations |
| L3 | legitimate | two cumulative awards | no violations |
| L4 | legitimate | conservative retention (`status`) | no violations |
| L5 | legitimate | independent predicates | no violations |
| L6 | legitimate | exclusive role chain | no violations |
| C1 | corruption | linked overlap | `overlap` |
| C2 | corruption | linked two-active | `multiple_active` |
| C3 | corruption | duplicate active value | `multiple_active` |
| C4 | corruption | inverted interval | `invalid_interval` |
| C5 | corruption | cross-subject link | `inconsistent_link` |
| C6 | corruption | cross-attribute link | `inconsistent_link` |
| H1 | historical SAME | closed value recurs; active head CONTRADICTION | `temporal_update`, contiguous history |
| H2 | historical SAME | closed value recurs; active head NEUTRAL | no unreported active overlap |
| H3 | historical SAME | multi-valued closed recurrence | insert, no violations |
| V1 | verifier boundary | changed value, SAME statement | no consolidation |
| V2 | verifier boundary | paraphrased value, SAME statement | consolidation |
| V3 | verifier boundary | changed value, non-exclusive attribute | independent insert |
| U1 | known gap | unlinked active exclusive conflict | documented false negative |
| T1 | telemetry | changed value triggers value check | counter increments |

## 8. Tests

New `tests/test_phase29_temporal_integrity.py` covers C5, C6, L6, H1, H2, T1
deterministically, plus a real-NLI boundary check. Existing
`tests/test_reliability_repairs.py` remains the R1–R3 regression anchor.

## 9. Planned changes

1. `temporal_history_violations()`: validate each link pair before union; report
   `inconsistent_link` with subject/attribute match flags; do not union invalid
   links.
2. `absorb()`: track a closed-history SAME match; if no other target is selected,
   resolve the active chain head and consolidate/supersede/splice it.
3. `_values_are_equivalent()`: increment `total_value_verifications`; expose it in
   `status()`.
4. New `benchmarks/phase29_integrity.py`, `runners/run_phase29.py`, and
   `benchmarks/results/phase29_results.json`.

## 10. Out of scope / carried risks

- Unlinked legacy conflicts remain a measured false-negative class (risk 1).
- Adding `telemetry mode` to the exclusive groups is not done without evidence
  that it is single-valued in the deployed domain (risk 5).
- Verifier confidence is not used as a value-equivalence threshold; only the
  decision is. Borderline behavior is measured, not hidden (risk 4).
- The full-scale Phase 23 end-to-end experiment remains pending.
- No checkpoint schema change, no benchmark baseline rewrite, no commit.
