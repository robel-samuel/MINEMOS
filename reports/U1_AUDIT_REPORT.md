# MINEMOS — U1 Closure Independent Acceptance Audit

**Date:** October 30, 2026
**Baseline commit:** `05fc8e4d9494795e02dacaed602daeda8e882cae`
**Repository state:** all changes uncommitted; nothing pushed.
**Scope:** independent verification of the U1 closure against its original
acceptance requirements. Phase 30 was not started. This audit is not a
production-readiness claim and does not redesign the engine.

**Method.** A dedicated audit harness (`benchmarks/u1_audit.py`, run by
`runners/run_u1_audit.py`) exercises the live engine with the pinned encoders
and NLI model, records raw observations, and evaluates pre-registered
expectations. The canonical result file is
`benchmarks/results/u1_audit_results.json` (`audit_complete_with_known_limitations`,
10.99 s). Additional adversarial coverage is in `tests/test_u1_audit.py`
(21 tests). Scratch probes are preserved under `scratch/u1_audit/` (git-ignored).

---

## 1. Audit entry points and integration call paths

### 1.1 Method

A memory was loaded with two active, unlinked slots asserting different values
of one exclusive predicate (Rome / Venice for `Maya.city`) — the exact U1 shape.
Each audit surface was then invoked on the same state:

| Surface | Observation |
|---|---|
| `temporal_history_violations()` (default) | `[]` |
| `temporal_history_violations(include_unlinked_conflicts=True)` | `["unlinked_exclusive_temporal_conflict"]` |
| `exclusive_temporal_conflicts()` | 1 finding |
| `status()` | no temporal-invariant key (`status_keys` listed in the result file) |
| checkpoint save → load → default invariant | `[]` |
| checkpoint save → load → opt-in invariant | `["unlinked_exclusive_temporal_conflict"]` |
| `benchmarks/phase29_integrity.py::_reasons` | `[]` |
| `benchmarks/u1_closure.py` structural path | `[]` (detector path is explicit) |

### 1.2 Call graph

```
BlackHoleMemory.absorb()
└── consolidation / temporal-update state machine
    └── no invariant check .................................. SILENT by design

BlackHoleMemory.status()
└── counters and footprint only ............................. SILENT (no temporal key)

temporal_history_violations(include_unlinked_conflicts=False)   [default]
├── structural check: linked chains, equal-value active slots,
│   interval ordering ....................................... SILENT for U1
└── include_unlinked_conflicts=True
    └── exclusive_temporal_conflicts() ...................... CAUGHT

exclusive_temporal_conflicts()  [direct public API] ......... CAUGHT

checkpoint.save_checkpoint()
└── serializes state; no invariant run
checkpoint.load_checkpoint() / load_into_black_hole_memory()
└── validates dimensions, checksum, per-slot
    valid_until >= valid_from only ........................... SILENT for U1

benchmarks/phase29_integrity.py::_reasons()
└── temporal_history_violations() default .................... SILENT

benchmarks/u1_closure.py
├── _structural_reasons() default ............................ SILENT (intended)
└── explicit detector calls where U1 is the subject .......... CAUGHT

benchmarks/u1_audit.py
└── explicit detector / opt-in calls ......................... CAUGHT
```

### 1.3 Findings

Four intended audit workflows still miss U1 silently:

1. **`temporal_history_violations()` default argument.** This is the Phase 29
   compatibility choice, but any caller that treats an empty result as "no
   temporal problem" gets a false negative for U1.
2. **`status()`.** It exposes `superseded_slots`, `total_temporal_updates`, etc.,
   but no conflict or violation count, so operators have no signal.
3. **Checkpoint load validation.** `load_checkpoint()` checks array integrity,
   checksums, and per-slot `valid_until >= valid_from`, but does not run the
   cross-slot structural invariant or the U1 detector. An archive containing an
   unlinked exclusive conflict loads cleanly.
4. **`benchmarks/phase29_integrity.py::_reasons`.** The Phase 29 harness keeps
   reproducing its historical expectations because it uses the default, which
   is correct for reproducibility but means it is not an audit for U1.

The closure harness and the new audit harness both call the detector
explicitly, so they are the only surfaces that currently catch U1.

### 1.4 Compatibility implications of changing defaults

Changing the default of `temporal_history_violations()` to include unlinked
conflicts would:

- change the output of every caller that currently relies on the Phase 29
  structural result (including the preserved `phase29_results.json` replay);
- make `[]` no longer mean "structurally consistent" for code that never opted
  in;
- alter the meaning of "violation" from a structural/link property to a
  policy-aware property.

No default was changed in this audit. The compatible route is to keep the
opt-in and make the detection explicit at each audit surface.

---

## 2. Historical retrieval verification

The five required statements plus a current-employment control were absorbed
sequentially with the real pinned NLI verifier at timestamps 10–60:

| ID | Statement | Action |
|---|---|---|
| H1 | Alice used to live in Addis Ababa. | `historical_observation` |
| H2 | Alice previously lived in Addis Ababa. | `historical_observation` |
| H3 | Alice formerly worked at Company A. | `historical_observation` |
| H4 | Alice no longer lives in Addis Ababa. | `insert` |
| H5 | Alice currently lives in Nairobi. | `temporal_update` |
| H6 | Alice works at Company B. | `insert` |

Final active state: `["Alice currently lives in Nairobi.", "Alice works at Company B."]`.
History (unfiltered): 6 rows, including both previous residences, the previous
employer, and the negation. No structural violation and no detector conflict.

### 2.1 Actual recall behavior

| Path | Result |
|---|---|
| `get_history("Alice")` / `recall(active_only=False)` | 6 rows, historical included, `is_active=False` |
| `recall("Alice")` (default) | 2 current rows; no historical |
| `query("Where did Alice live?")` (default) | no historical rows |
| `query(..., active_only=False)` | 6 rows, historical included |
| `get_history("Alice", "lives in")` | 0 rows |
| `recall("Alice", as_of=10.0, active_only=False)` | 0 rows |

The distinction between previous and current residence and between previous and
current employment is correct: the historical rows never become active, and the
current rows are exactly Nairobi and Company B. The current negation is
retained and becomes the current residence state, then is closed by the move
(H4 `insert` → active negation; H5 `temporal_update` → active Nairobi).

### 2.2 Limitations (measured, not inferred)

1. **Point-in-time recall excludes zero-width observations.** Historical rows
   are stored as `[t, t]` and the half-open `as_of` filter never matches them.
   `recall("Alice", as_of=10.0, active_only=False)` returns nothing, even though
   the historical fact was observed at exactly 10.0.
2. **Predicate-filtered history misses marker predicates.** The historical rows
   store `used to live in` / `previously lived in`, and the current row stores
   `currently lives in`; a query for the canonical attribute `lives in` returns
   zero rows for this subject.
3. **Default query is current-only.** `query()` does not surface historical
   rows without `active_only=False`.

So a historical fact does **not** become current accidentally — this part of
the requirement is met. It also is not lost: it is retrievable through the
history and unfiltered recall paths. But it is inaccessible through two
reasonable retrieval interfaces (point-in-time and predicate-filtered history),
which fails the stronger reading of the acceptance requirement.

### 2.3 Smallest proposed repair (not implemented)

- Match historical rows on a canonical attribute derived by stripping the
  historical marker from the predicate, or store `canonical_predicate` alongside
  the raw predicate. This is additive and does not change current-state recall.
- Decide explicitly whether a zero-width row is observable at its observation
  instant. If yes, special-case `valid_from == valid_until` in the `as_of`
  filter (include at exactly `t`); if no, document that point-in-time recall
  intentionally omits observations of unknown duration.

Both are retrieval-layer changes; neither requires deleting, rewriting, or
linking facts.

---

## 3. Language robustness evaluation

### 3.1 Dataset construction

`EQUIVALENCE_PAIRS` contains 50 labeled pairs across the ten requested
categories. Each pair is `(subject, predicate, object)` absorbed at `t` and then
restated at `t+1`; the prediction comes from `absorb()` with the real NLI gate
(not from the lexical detector), and is whether the second statement
consolidated as the same fact. Labels: `equivalent`, `not_equivalent`,
`uncertain`.

`CONFLICT_PAIRS` contains 20 pairs injected directly as unlinked slots, so the
detector is tested on exactly the state U1 describes. Predicted conflict means
`exclusive_temporal_conflicts()` reports the pair; structural duplicate
reporting is recorded separately and is not scored as detector output.

Scoring rules (fixed before execution):

- an `equivalent` pair is correct when the second absorb consolidates
  (one active slot, no false state change);
- a `not_equivalent` pair is correct when the two facts remain distinct;
- `uncertain` pairs are reported but excluded from precision/recall;
- the detector is scored only on its own output; structural duplicate
  detection is a separate invariant and is shown in §3.4.

### 3.2 Equivalence results

45 scored pairs + 5 uncertain. Real pinned NLI, 103 model calls, 7.33 s.

| Category | Pairs | TP | FP | TN | FN | Uncertain | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spelling | 6 | 5 | 0 | 0 | 1 | 0 | 1.000 | 0.833 | 0.909 | n/a |
| grammar | 5 | 5 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | n/a |
| capitalization | 5 | 5 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | n/a |
| abbreviation | 4 | 4 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | n/a |
| paraphrase | 6 | 6 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | n/a |
| similar names | 5 | 0 | 0 | 5 | 0 | 0 | n/a | n/a | n/a | 0.000 |
| similar locations | 5 | 0 | 0 | 5 | 0 | 0 | n/a | n/a | n/a | 0.000 |
| negation | 4 | 0 | 0 | 4 | 0 | 0 | n/a | n/a | n/a | 0.000 |
| historical | 5 | 0 | 0 | 5 | 0 | 0 | n/a | n/a | n/a | 0.000 |
| ambiguous | 5 | 0 | 0 | 0 | 0 | 5 | n/a | n/a | n/a | n/a |
| **Total** | **50** | **25** | **0** | **19** | **1** | **5** | **1.000** | **0.962** | **0.980** | **0.000** |

Confusion matrix (45 scored): TP 25, FP 0, TN 19, FN 1.

The single false negative is **SP-04** (`Mohamed` / `Mohamad`), a single-letter
substitution typo that the conservative rule deliberately excludes. The result
file records it as an expected miss, not a surprise. All five ambiguous inputs
are excluded from the metrics by their label, not scored as right or wrong. The
engine has no UNCERTAIN prediction: it answered `different` for AM-01/02/03
(kept the facts distinct) and `equivalent` for AM-04 (disjunction) and AM-05
(`lives near` vs `lives in`). Those five observations are reported in full, and
none is counted as an error.

Runtime per category: spelling 0.975 s, grammar 0.799 s, capitalization
0.856 s, abbreviation 0.516 s, paraphrase 0.800 s, similar names 0.618 s,
similar locations 0.812 s, negation 0.582 s, historical 0.694 s, ambiguous
0.672 s.

### 3.3 Detector conflict results

20 pairs, 0 model calls, 0.0 s.

| Category | Pairs | TP | FP | TN | FN | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spelling | 2 | 1 | 0 | 1 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| grammar | 1 | 0 | 0 | 1 | 0 | n/a | n/a | n/a | 0.000 |
| capitalization | 1 | 0 | 0 | 1 | 0 | n/a | n/a | n/a | 0.000 |
| abbreviation | 2 | 0 | 0 | 2 | 0 | n/a | n/a | n/a | 0.000 |
| paraphrase | 2 | 0 | 2 | 0 | 0 | 0.000 | n/a | n/a | 1.000 |
| similar names | 2 | 0 | 0 | 2 | 0 | n/a | n/a | n/a | 0.000 |
| similar locations | 3 | 3 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | n/a |
| negation | 3 | 0 | 0 | 3 | 0 | n/a | n/a | n/a | 0.000 |
| historical | 2 | 0 | 0 | 2 | 0 | n/a | n/a | n/a | 0.000 |
| ambiguous | 2 | 0 | 0 | 2 | 0 | n/a | n/a | n/a | 0.000 |
| **Total** | **20** | **4** | **2** | **14** | **0** | **0.667** | **1.000** | **0.800** | **0.125** |

Confusion matrix: TP 4, FP 2, TN 14, FN 0.

The two false positives are **CF-PA-01** (`New York City` / `Big Apple`) and
**CF-PA-02** (`Acme` / `Acme Corporation`). Both are semantically equal and
both are expected false positives for a lexical-only, model-free detector. They
are not scoring bugs and are not counted as unlabeled errors.

### 3.4 Structural duplicate check (separate invariant)

| Case | Detector findings | Structural `multiple_active` | Correct |
|---|---:|---|---|
| SD-01 case-noise duplicate | 0 | true | yes |
| SD-02 affirmation + negation of one value | 0 | true | yes |

This separation is why CF-CP-01 (case noise) and CF-NG-03 (same-value
negation) are `NO_CONFLICT` for the detector: the structural invariant owns
equal-value duplicates, and the detector owns different-value unlinked pairs.

### 3.5 Interpretation

Results are strong for controlled spelling, grammar, capitalization,
abbreviation, and paraphrase variants, and for keeping similar-but-different
names and locations apart. They do not demonstrate general language
understanding: the dataset is small and manually constructed, the paraphrase
category is six pairs, and the detector is intentionally lexical. The reported
precision/recall describe this fixed dataset, not open-domain behavior.

---

## 4. Detector output semantics

A finding currently contains:

```
confidence          1.0
confidence_scope    "predicate_policy_only"
semantic_equivalence "lexical_only"
conflict_status     "unresolved"
```

**Risk.** A consumer reading only `confidence: 1.0` could interpret it as
certainty that the two facts semantically conflict. It is not: 1.0 means the
predicate policy deterministically classifies both predicates as EXCLUSIVE.
Value equivalence is lexical, and the conflict is deliberately left unresolved
(no winner selected). Before this audit the finding had no scope fields, so
this misinterpretation was possible.

**Assessment.** The three explicit fields now separate the three concepts the
audit asked about:

| Concept | Current representation | Determinism |
|---|---|---|
| predicate policy determination | `predicate_policy` + `confidence(1.0)` + `confidence_scope` | deterministic |
| semantic equivalence verification | `semantic_equivalence="lexical_only"` | deterministic, no model |
| overall conflict status | `conflict_status="unresolved"` | no resolution attempted |

**Recommendation (not implemented).** A schema change is not required for the
separation; the additive fields already prevent silent misreading for consumers
that read them. If a breaking cleanup is desired later, deprecate `confidence`
in favor of `policy_confidence` and add a numeric `semantic_confidence: null`
for lexical checks. That would break existing key access, so it needs a major
version and approval.

**Compatibility.** All three fields are additive; the `confidence` value and
all existing keys are unchanged. No consumer is forced to change.

All seven semantics checks pass: `confidence_is_policy_only`,
`confidence_scope_declared`, `semantic_equivalence_declared`,
`conflict_status_declared`, `repeatable`, `no_mutation`, `no_model_calls`.

---

## 5. Predicate policy reproducibility

A checkpoint containing one unlinked exclusive conflict (`telemetry mode`:
`verbose` / `silent`) was saved, then restored three times with different policy
configuration:

| Configuration | Detector findings |
|---|---:|
| identical (`telemetry mode` exclusive, source policy) | 1 |
| missing (no overrides) | 0 |
| different (`telemetry mode` reclassified multi-valued) | 0 |
| restored slot state identical in all three | yes |

The archive contains the facts but not the predicate policy. The same checkpoint
therefore yields different invariant results depending on the loading process's
configuration. `policy_persisted_in_checkpoint` is false.

This is the reproducibility gap: invariant results are a function of
`(archive, process policy)`, not of the archive alone. The closure report
documented this as a design choice ("an archive stores facts, not process
configuration"), but it means an audit performed in one process cannot be
reproduced from the artifact alone.

**Proposed mechanism (not implemented; checkpoint schema untouched).** Add an
optional `policy_manifest` group to the checkpoint containing the sorted
exclusive and multi-valued predicate sets and a manifest version. On load:

- absent manifest → current behavior (silent, backward compatible);
- present manifest → use it to seed policy, and raise a dedicated
  `PolicyMismatchError` (or warn, configurable) when the loading process
  explicitly configures a conflicting policy.

This is additive, keeps v1/v2/v3 archives loadable, and makes invariant results
reproducible from the archive. It changes what a checkpoint *contains*, so it
requires the schema approval that this audit does not grant itself.

---

## 6. U1 detector correctness — new adversarial tests

`tests/test_u1_audit.py` adds 21 tests. Focused run: **21 passed in 33.80 s**.
They cover:

| Test | Property verified |
|---|---|
| `test_two_active_values_are_reported` | baseline U1 pair → 1 finding |
| `test_three_active_values_are_pairwise_and_ordered` | 3 values → `[0,1], [0,2], [1,2]` |
| `test_predicate_synonyms_in_one_exclusive_group_are_compared` | synonym group shares a policy |
| `test_different_exclusive_groups_are_not_compared` | unrelated exclusive attributes stay silent |
| `test_multi_valued_and_unknown_predicates_stay_silent` | R2 guarantee preserved |
| `test_detector_audits_active_pairs_only` | closed unlinked history is out of scope |
| `test_malformed_interval_is_structural_and_not_a_u1_finding` | `valid_until < valid_from` → structural `invalid_interval` |
| `test_reciprocal_supersession_link_is_left_to_the_structural_invariant` | linked chain is not a U1 finding |
| `test_one_directional_link_is_reported_as_non_reciprocal` | structural link asymmetry |
| `test_half_open_boundary_is_disjoint_in_the_structural_invariant` | `[10,20)` + `[20,30)` is not overlap |
| `test_distinct_subjects_with_near_identical_spelling_are_not_merged` | Alicia/May/Maya/Jon |
| `test_subject_typo_still_counts_as_one_subject` | one edit still one subject |
| `test_value_typo_is_same_fact_but_substitution_is_a_conflict` | indel vs substitution boundary |
| `test_negation_with_a_different_value_is_not_a_conflict` | negation guard |
| `test_negation_and_affirmation_of_one_value_stay_structural` | structural owns same-value duplicates |
| `test_findings_declare_policy_scope` | semantics fields |
| `test_detector_is_repeatable_and_never_mutates_state` | two runs identical; full state snapshot unchanged |
| `test_detector_never_calls_a_model` | exploding verifier + zero call counters |
| `test_bare_string_policy_override_is_one_predicate` | `"telemetry mode"` is not parsed as characters |
| `test_policy_is_process_configuration_not_checkpoint_state` | 1 / 0 / 0 findings across policy configurations |
| `test_historical_fact_is_retrievable_and_never_becomes_current` | history includes it, active recall does not, as-of excludes it |

Residual risk found while writing these tests: a slot with `valid_from=None`
(crafted or mutated state) raises `TypeError` in both the detector and the
structural invariant. The normal construction path never produces this; it is
recorded as residual, not repaired, because adding an interval fallback would
change invariant semantics without approval.

### 6.1 Defects the audit itself exposed (and the narrow fixes made)

Three changes were made to `phase2/black_hole_memory.py` during the audit. Each
was driven by a reproduced failure, not by cleanup:

1. **False conflict for negation + move.** `no longer lives in Addis Ababa`
   plus `lives in Nairobi` produced 1 finding. Fixed with a negation guard.
   Before: 1 finding (CF-NG-01/02 failed). After: 0 findings; regression tests
   added. Compatibility: the detector is new API; same-value negation remains
   visible structurally.
2. **Bare-string policy override split into characters.**
   `exclusive_predicates="telemetry mode"` configured `{t,e,l,m,o,r,y,...}`.
   Fixed with `_predicate_config_set`. Before: character set; after:
   `{"telemetry mode"}`. Compatibility: list/tuple behavior unchanged; the
   string form now matches its documented meaning.
3. **Unscoped `confidence: 1.0`.** Added `confidence_scope`,
   `semantic_equivalence`, and `conflict_status` (see §4). Compatibility:
   additive keys only.

All three are uncommitted; none changes the default arguments or return
contract of a pre-existing API. The only behavior change is inside the new
detector (the negation guard), where the prior behavior was a measured false
positive.

---

## 7. Exact validation results

| Run | Result |
|---|---|
| Audit harness (`runners/run_u1_audit.py`) | `audit_complete_with_known_limitations`, 10.99 s |
| Audit harness replay (independent run, scratch output) | all sections and metrics identical; only `slot_ids` and timings differ (15.97 s) |
| New audit tests | 21 passed in 33.80 s |
| U1 closure benchmark | 22/22, `all_expected_reproduced`, 1.28 s |
| Focused regression set (closure + Phase 29 + reliability + Phase 28 contradiction) | 103 passed in 46.51 s |
| Full suite group A (adversarial, 16–18, 28, 29, reliability, U1) | 227 passed in 158.67 s (exit 0) |
| Full suite group B (checkpoint, concurrency, 12–15, 19–27, security) | 201 passed in 637.71 s (exit 0) |
| Phase 29 canonical `phase29_results.json` SHA-256 | `93e9a75db5a68e5ef1ab156920971fc1d857982709de2f39434d5281a93ab137` (unchanged) |

The suite now collects **428 tests in `tests/`** (407 before this audit + 21 new).
Group A and group B are disjoint and together cover all 428. Every reported test
was executed; no result is inferred from an unexecuted run.

Note: the U1 closure report's §3.3 table records the pre-audit totals (407
tests, 206 + 201). It is left unchanged because it documents the closure run at
the time it was made; this report supersedes it for the current tree.

---

## 8. Remaining limitations

1. **Default audit paths still miss U1.** `temporal_history_violations()` and
   `status()` remain silent unless a caller opts in. This is deliberate
   compatibility, but it means U1 detection is opt-in, not ambient.
2. **Checkpoint validation does not audit temporality.** Loading an archive with
   an unlinked exclusive conflict succeeds.
3. **Policy is not persisted.** Identical archives produce 1 / 0 / 0 findings
   under different process configurations.
4. **Detector value equivalence is lexical.** `New York City` / `Big Apple` and
   `Acme` / `Acme Corporation` are reported as conflicts (2 measured false
   positives). A model-free checker cannot resolve them.
5. **Unknown predicates are never detected.** `commutes via`-style attributes
   need explicit configuration.
6. **Substitution typos are not merged.** `Mohamed` / `Mohamad` remains two
   subjects (1 measured false negative in equivalence).
7. **Historical facts are not reachable by point-in-time or canonical-predicate
   history.** They are reachable by subject history and unfiltered recall.
8. **Zero-width history has no inferred validity window.** The engine does not
   invent start/end times, so `as_of` cannot place the fact in an interval.
9. **Detector scope is active pairs.** An unlinked active/closed overlap is not
   reported by the detector (it skips inactive slots) nor by the structural
   invariant (it only groups linked or equal-value slots). This is a residual
   gap distinct from U1.
10. **Crafted `valid_from=None` raises `TypeError`.** Only reachable through
    mutated or hand-built state.

---

## 9. Acceptance verdict

| Original requirement | Verdict | Basis |
|---|---|---|
| A checker exists for unlinked exclusive active conflicts | **Met** | `exclusive_temporal_conflicts()`, 1 finding on the U1 shape |
| Intended audit workflows invoke it | **Not met** | 4 silent surfaces (§1.3); detection is opt-in |
| Historical facts do not become current | **Met** | H1–H3 stay closed and inactive; H4/H5 are current state; no invariant violation |
| Historical facts remain accessible | **Partially met** | subject history and unfiltered recall yes; as-of and predicate-filtered history no |
| Detector output semantics are unambiguous | **Partially met** | scope fields added; bare `confidence` remains a legacy misread risk |
| Predicate policy results are reproducible from an archive | **Not met** | 1 / 0 / 0 findings for one checkpoint (§5) |
| Detector is correct on adversarial temporal state | **Met for tested scope** | 21 tests; residual active/closed gap and `None` interval |
| Detector is read-only, repeatable, model-free | **Met** | snapshot unchanged, identical repeats, exploding verifier |
| Language handling is robust | **Partially met** | 25/26 equivalent, 19/19 distinct; 2 paraphrase FPs and 1 conservative FN |
| No facts deleted, rewritten, linked, merged, or selected by the audit | **Met** | detector is read-only; no production API default changed |

**Classification: PARTIALLY RESOLVED.**

The specific U1 detection defect is closed: a dedicated, read-only, model-free
checker reports the unlinked exclusive pair that the structural invariant
cannot see, and its behavior is covered by 21 adversarial tests. But the broader
acceptance criteria are not all met: U1 is still invisible on the default audit
paths, historical facts are unreachable through point-in-time and
canonical-predicate history, and predicate policy is not reproducible from a
checkpoint. Those are substantive gaps, not test-count gaps, so this audit does
not declare U1 CLOSED.

---

## 10. Repository state

- Uncommitted: `phase2/black_hole_memory.py`, `tests/test_reliability_repairs.py`
  (fixture from the closure work), plus the new audit/closure harnesses, runners,
  result files, tests, and reports.
- No commit, push, history rewrite, schema change, dependency change, or Phase 30
  work was performed.
- `benchmarks/results/phase29_results.json` was not modified.
